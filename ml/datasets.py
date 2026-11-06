"""Phrase extraction, synthetic fallback, song-level splits and torch datasets.

A :class:`Phrase` is a short monophonic melody in beats.  Phrases come from
melody tracks of MIDI files (:func:`phrases_from_midi`, used with POP909),
from guitar riff tracks (:func:`phrases_from_guitar_midi` for Lakh-style MIDI,
:func:`phrases_from_dadagp` for DadaGP token files) or from the MIDI-free
generator :func:`synthetic_phrases` so that tests and smoke runs work offline.  :class:`PhraseDataset` turns phrases into noisy sketches on the
fly with a :class:`ml.sketch_simulator.SketchSimulator` and returns the
tensors consumed by :mod:`ml.model`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from core.representation import INTERVAL_MAX, Sketch, interval_to_class

if TYPE_CHECKING:  # avoid a circular import at runtime
    from ml.sketch_simulator import SketchSimulator

# Gesture kinds the model sees as a multi-hot vector per note.  Order matters
# (it is baked into checkpoints), so append rather than reorder.
GESTURE_KINDS: tuple[str, ...] = (
    "ascending_run", "descending_run", "repeated_note", "repeated_figure",
    "pedal_tone", "arpeggio", "big_leap",
)
N_CONTINUOUS_FEATURES = 6  # ioi, log ioi, y, dy, confidence, missing-flag
FEATURE_VERSION = 1
PAD_TARGET = -100  # ignore_index for cross-entropy


# --------------------------------------------------------------------------- #
# Phrase
# --------------------------------------------------------------------------- #


# Genre conditioning. The data source is the first (coarse) genre label; the
# model gets a learned embedding per genre and is trained with the label
# dropped to "unknown" part of the time, so inference works with or without
# a genre choice.
GENRES: tuple[str, ...] = ("unknown", "pop", "rock", "metal")
SOURCE_GENRE: dict[str, str] = {
    "pop909": "pop",
    "lakh_guitar": "rock", "guitar_midi": "rock",
    "dadagp": "metal",
}


# Rhythm classes: the gap between consecutive notes in beats. Sixteenth-note
# multiples plus the common triplet values; the model predicts one class per
# gap next to the interval class, so timing is denoised with the pitch.
RHYTHM_CLASSES: tuple[float, ...] = (
    1 / 6, 1 / 3, 2 / 3,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0,
)
_RHYTHM_LOG = np.log(np.array(RHYTHM_CLASSES))


def rhythm_class(beats: float) -> int:
    """Nearest rhythm class (in log space, so 1/3 vs 0.25 is judged by ratio)."""
    return int(np.argmin(np.abs(_RHYTHM_LOG - np.log(max(float(beats), 1e-3)))))


def class_to_beats(c: int) -> float:
    return float(RHYTHM_CLASSES[int(c)])


def genre_of_source(source: str) -> str:
    return SOURCE_GENRE.get(source, "unknown")


def genre_id(name: str | None) -> int:
    """Index into GENRES; unknown names map to 0 ("unknown")."""
    return GENRES.index(name) if name in GENRES else 0


@dataclass
class Phrase:
    """A short monophonic melody; onsets/durations in beats, first onset at 0."""

    song_id: str  # the song id in the song database (as a string)
    pitches: list[int]
    onsets_beats: list[float]
    durations_beats: list[float]
    source: str = "unknown"
    rendition_id: int | None = None  # which file of the song the phrase was cut from

    def __len__(self) -> int:
        return len(self.pitches)

    @property
    def intervals(self) -> list[int]:
        return [b - a for a, b in zip(self.pitches, self.pitches[1:])]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Phrase":
        return cls(song_id=str(d["song_id"]), pitches=[int(p) for p in d["pitches"]],
                   onsets_beats=[float(x) for x in d["onsets_beats"]],
                   durations_beats=[float(x) for x in d["durations_beats"]],
                   source=str(d.get("source", "unknown")),
                   rendition_id=None if d.get("rendition_id") is None else int(d["rendition_id"]))


def save_phrases(phrases: Iterable[Phrase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in phrases:
            f.write(json.dumps(p.to_dict()) + "\n")


def load_phrases(path: str | Path) -> list[Phrase]:
    out: list[Phrase] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Phrase.from_dict(json.loads(line)))
    return out


# --------------------------------------------------------------------------- #
# MIDI extraction
# --------------------------------------------------------------------------- #


def _pick_melody_instrument(pm: Any):
    """POP909 names its melody track "MELODY"; otherwise use a heuristic."""
    cands = [ins for ins in pm.instruments if not ins.is_drum and len(ins.notes) > 0]
    if not cands:
        return None
    for ins in cands:
        if ins.name.strip().upper() == "MELODY":
            return ins
    # "monophonic-ish": fraction of notes that do not overlap the previous one
    def score(ins) -> tuple[float, float]:
        notes = sorted(ins.notes, key=lambda n: (n.start, -n.pitch))
        overlaps = sum(1 for a, b in zip(notes, notes[1:]) if b.start < a.end - 1e-3)
        mono = 1.0 - overlaps / max(1, len(notes) - 1)
        mean_pitch = float(np.mean([n.pitch for n in notes]))
        return (1.0 if mono > 0.7 else 0.0, mean_pitch)

    return max(cands, key=score)


def _monophonic_notes(notes: list[Any], onset_tol: float = 0.02) -> list[Any]:
    """Keep the highest note at each onset and trim notes that overlap the next."""
    notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
    kept: list[Any] = []
    for n in notes:
        if kept and abs(n.start - kept[-1].start) <= onset_tol:
            continue  # same onset: the first (highest) one wins
        kept.append(n)
    return kept


def split_segment(starts: list[float], durs: list[float], min_notes: int,
                  max_notes: int) -> list[list[int]]:
    """Cut a run of note indices into phrases of at most ``max_notes`` notes.

    Pop melodies rarely rest for long, so a rest threshold alone leaves runs
    of 50+ notes.  Instead of chopping those every ``max_notes`` notes (which
    cuts through the middle of musical phrases) we recursively split the run
    at its biggest internal gap: the longest rest before a note, with the
    inter-onset interval as a tie-breaker.  Both halves must keep at least
    ``min_notes`` notes where possible.  Returns index lists in time order,
    some of which may still be shorter than ``min_notes`` (the caller drops
    those).
    """
    n = len(starts)
    if n <= max_notes:
        return [list(range(n))]
    lo, hi = min_notes, n - min_notes  # candidate cut positions (cut before index j)
    if lo >= hi:
        lo, hi = 1, n - 1
    best, best_key = lo, (-1.0, -1.0)
    for j in range(lo, hi + 1):
        rest = starts[j] - (starts[j - 1] + durs[j - 1])
        ioi = starts[j] - starts[j - 1]
        key = (round(rest, 4), round(ioi, 4))
        if key > best_key:
            best, best_key = j, key
    left = split_segment(starts[:best], durs[:best], min_notes, max_notes)
    right = split_segment(starts[best:], durs[best:], min_notes, max_notes)
    return left + [[i + best for i in seg] for seg in right]


def phrases_from_midi(path: str | Path, min_notes: int = 5, max_notes: int = 16,
                      rest_split_beats: float = 2.0, song_id: str | None = None,
                      source: str = "midi") -> list[Phrase]:
    """Extract monophonic phrases (in beats) from one MIDI file.

    The melody track is chosen by name ("MELODY") or by a monophony/high-pitch
    heuristic.  Times are converted to beats through the file's tempo map, so
    songs with tempo changes (common in POP909) still get a steady beat grid.
    The track is first cut at rests longer than ``rest_split_beats``; any run
    still longer than ``max_notes`` is cut at its largest internal gaps by
    :func:`split_segment`.  Phrases shorter than ``min_notes`` are dropped.
    """
    import pretty_midi  # heavy import, keep it local

    path = Path(path)
    pm = pretty_midi.PrettyMIDI(str(path))
    ins = _pick_melody_instrument(pm)
    if ins is None:
        return []
    notes = _monophonic_notes(ins.notes)
    if not notes:
        return []
    sid = song_id or path.stem
    res = float(pm.resolution)

    def to_beats(t: float) -> float:
        return float(pm.time_to_tick(t)) / res

    starts = [to_beats(n.start) for n in notes]
    ends = [to_beats(n.end) for n in notes]
    # trim overlaps so durations never run past the next onset
    durs = [max(0.125, min(e - s, (starts[i + 1] - s) if i + 1 < len(starts) else e - s))
            for i, (s, e) in enumerate(zip(starts, ends))]

    # 1) hard cuts at long rests
    runs: list[list[int]] = [[0]]
    for i in range(1, len(notes)):
        prev = runs[-1][-1]
        rest = starts[i] - (starts[prev] + durs[prev])
        if rest > rest_split_beats:
            runs.append([])
        runs[-1].append(i)

    # 2) cut over-long runs at their biggest gaps, drop the too-short ones
    phrases: list[Phrase] = []
    for run in runs:
        for seg in split_segment([starts[i] for i in run], [durs[i] for i in run], min_notes, max_notes):
            idx = [run[k] for k in seg]
            if len(idx) < min_notes:
                continue
            t0 = starts[idx[0]]
            phrases.append(Phrase(sid, [int(notes[i].pitch) for i in idx],
                                  [round(starts[i] - t0, 4) for i in idx],
                                  [round(durs[i], 4) for i in idx], source))
    return phrases


# --------------------------------------------------------------------------- #
# Guitar riffs: guitar tracks in general MIDI files and DadaGP token files
# --------------------------------------------------------------------------- #
#
# A riff line is the *lowest* note of each onset: the root of a power chord is
# what you hum, and a lead line is monophonic anyway so the rule costs nothing
# there.  Onsets with more than MAX_RIFF_CHORD notes are strummed chords, not
# riff notes; they end the current phrase instead of being dropped silently
# (dropping them would fake a rest in the middle of a riff).

GUITAR_PROGRAMS = range(24, 32)  # General MIDI 25-32: nylon ... overdrive, distortion, harmonics
GUITAR_NAME_KEYS = ("guitar", "gtr")
MAX_RIFF_CHORD = 3  # a power chord with its octave

# (start_beat, end_beat, pitches sounding from that onset)
Onset = tuple[float, float, list[int]]


def riff_line_phrases(onsets: list[Onset], song_id: str, source: str, min_notes: int = 5,
                      max_notes: int = 16, rest_split_beats: float = 2.0,
                      max_chord: int = MAX_RIFF_CHORD) -> list[Phrase]:
    """Phrases from one guitar track given as onsets ``(start, end, pitches)`` in beats.

    Keeps the lowest pitch per onset, cuts at strummed chords (more than
    ``max_chord`` notes), at rests longer than ``rest_split_beats`` and, for
    over-long runs, at their largest gaps (:func:`split_segment`).  Phrases
    shorter than ``min_notes`` or with an interval beyond the model's range
    (±24 semitones, e.g. an octave-displaced MIDI transcription) are dropped.
    """
    runs: list[list[tuple[float, float, int]]] = [[]]  # (start, duration, pitch)
    for start, end, pitches in sorted(onsets, key=lambda o: o[0]):
        if not pitches or len(pitches) > max_chord:
            if runs[-1]:
                runs.append([])
            continue
        if runs[-1]:
            prev_start, prev_dur, _ = runs[-1][-1]
            if start - (prev_start + prev_dur) > rest_split_beats:
                runs.append([])
        runs[-1].append((start, end - start, min(pitches)))

    phrases: list[Phrase] = []
    for run in runs:
        if len(run) < min_notes:
            continue
        starts = [s for s, _, _ in run]
        # trim durations so no note runs past the next onset
        durs = [max(0.125, min(d, (starts[i + 1] - s) if i + 1 < len(run) else d))
                for i, (s, d, _) in enumerate(run)]
        for seg in split_segment(starts, durs, min_notes, max_notes):
            if len(seg) < min_notes:
                continue
            pitches = [run[i][2] for i in seg]
            if any(abs(b - a) > INTERVAL_MAX for a, b in zip(pitches, pitches[1:])):
                continue
            t0 = starts[seg[0]]
            phrases.append(Phrase(song_id, pitches, [round(starts[i] - t0, 4) for i in seg],
                                  [round(durs[i], 4) for i in seg], source))
    return phrases


def unique_phrases(phrases: list[Phrase]) -> list[Phrase]:
    """Drop exact repeats (same pitches and onsets): riffs repeat and doubled
    guitar tracks copy each other, and copies would only inflate the metrics."""
    seen: set[tuple] = set()
    out: list[Phrase] = []
    for p in phrases:
        key = (p.song_id, tuple(p.pitches), tuple(p.onsets_beats))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _is_guitar_track(ins: Any) -> bool:
    name = ins.name.lower()
    if ins.is_drum or not ins.notes or "bass" in name:
        return False
    return ins.program in GUITAR_PROGRAMS or any(k in name for k in GUITAR_NAME_KEYS)


def _onsets_from_notes(notes: list[Any], to_beats: Any, tol_beats: float = 0.05) -> list[Onset]:
    """Group notes that start (almost) together; the onset ends with its lowest note."""
    out: list[list[Any]] = []  # [start, end, pitches]
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        start, end, pitch = to_beats(n.start), to_beats(n.end), int(n.pitch)
        if out and start - out[-1][0] <= tol_beats:
            if pitch < min(out[-1][2]):
                out[-1][1] = end
            out[-1][2].append(pitch)
        else:
            out.append([start, end, [pitch]])
    return [(s, e, p) for s, e, p in out]


def phrases_from_guitar_midi(path: str | Path, min_notes: int = 5, max_notes: int = 16,
                             rest_split_beats: float = 2.0, song_id: str | None = None,
                             source: str = "guitar_midi") -> list[Phrase]:
    """Riff phrases from every guitar track of a general MIDI file (e.g. Lakh).

    A track counts as guitar when its program is one of the General MIDI
    guitars (25-32) or its name mentions "guitar"/"gtr"; bass tracks are
    excluded by name.  See :func:`riff_line_phrases` for the riff-line rule.
    """
    import pretty_midi  # heavy import, keep it local

    path = Path(path)
    pm = pretty_midi.PrettyMIDI(str(path))
    sid = song_id or path.stem
    res = float(pm.resolution)

    def to_beats(t: float) -> float:
        return float(pm.time_to_tick(t)) / res

    phrases: list[Phrase] = []
    for ins in pm.instruments:
        if _is_guitar_track(ins):
            phrases.extend(riff_line_phrases(_onsets_from_notes(ins.notes, to_beats), sid, source,
                                             min_notes, max_notes, rest_split_beats))
    return unique_phrases(phrases)


# DadaGP (https://github.com/dada-bots/dadaGP) writes one token per line:
# "distorted0:note:s6:f3" = instrument track, string 6, fret 3; "wait:480" =
# advance 480 ticks (960 per beat); "nfx:tie"/"nfx:dead" qualify the note
# token just before them; "downtune:2" = every guitar tuned 2 semitones down.
DADAGP_TICKS_PER_BEAT = 960
DADAGP_GUITARS = ("distorted", "clean")  # bass, leads, pads and drums are not riff tracks
DADAGP_STRING_PITCH = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40, 7: 35}  # standard tuning, 7-string adds a low B


def dadagp_track_events(tokens: Iterable[str]) -> dict[str, list[tuple[float, list[int] | None]]]:
    """Per guitar track, ``(beat, pitches)`` for each note event and ``(beat, None)``
    for each rest.  Tied and dead notes are removed from their event, so an
    event with an empty pitch list is a sustain (tie) or a muted chug."""
    downtune, tick = 0, 0
    tracks: dict[str, list[tuple[float, list[int] | None]]] = {}
    last_note: tuple[list[int], int] | None = None  # (pitches of the current event, pitch)
    for tok in tokens:
        parts = tok.strip().split(":")
        head = parts[0]
        if head == "wait" and len(parts) == 2:
            tick += int(parts[1])
            last_note = None
        elif head == "downtune" and len(parts) == 2:
            downtune = int(parts[1])
        elif head == "nfx" and len(parts) >= 2:
            if last_note is not None and parts[1] in ("tie", "dead") and last_note[1] in last_note[0]:
                last_note[0].remove(last_note[1])
        elif head.rstrip("0123456789") in DADAGP_GUITARS and len(parts) >= 2:
            events = tracks.setdefault(head, [])
            beat = tick / DADAGP_TICKS_PER_BEAT
            if parts[1] == "rest":
                events.append((beat, None))
            elif parts[1] == "note" and len(parts) == 4 and parts[2][:1] == "s" and parts[3][:1] == "f":
                string = int(parts[2][1:])
                if string not in DADAGP_STRING_PITCH:
                    continue
                pitch = DADAGP_STRING_PITCH[string] + int(parts[3][1:]) - downtune
                if not events or events[-1][0] != beat or events[-1][1] is None:
                    events.append((beat, []))
                events[-1][1].append(pitch)  # type: ignore[union-attr]
                last_note = (events[-1][1], pitch)  # type: ignore[arg-type]
    return tracks


def _onsets_from_dadagp_events(events: list[tuple[float, list[int] | None]]) -> list[Onset]:
    """Each sounding event lasts until the next rest or sounding event (ties
    and dead notes in between just extend it); the last one gets one beat."""
    onsets: list[Onset] = []
    cuts = [(beat, pitches) for beat, pitches in events if pitches is None or pitches]
    for i, (beat, pitches) in enumerate(cuts):
        if pitches is None:
            continue
        end = cuts[i + 1][0] if i + 1 < len(cuts) else beat + 1.0
        onsets.append((beat, max(end, beat + 0.125), list(pitches)))
    return onsets


def phrases_from_dadagp(path: str | Path, min_notes: int = 5, max_notes: int = 16,
                        rest_split_beats: float = 2.0, song_id: str | None = None,
                        source: str = "dadagp") -> list[Phrase]:
    """Riff phrases from one DadaGP token file (every distorted/clean guitar track).

    Repeats are not unrolled, so a riff that the tab repeats four times
    appears once.  See :func:`riff_line_phrases` for the riff-line rule.
    """
    path = Path(path)
    sid = song_id or path.name.split(".")[0]
    with path.open(encoding="utf-8", errors="replace") as f:
        tracks = dadagp_track_events(f)
    phrases: list[Phrase] = []
    for events in tracks.values():
        phrases.extend(riff_line_phrases(_onsets_from_dadagp_events(events), sid, source,
                                         min_notes, max_notes, rest_split_beats))
    return unique_phrases(phrases)


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #

SYNTH_SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}
PITCH_LO, PITCH_HI = 40, 79


def _scale_pitches(root: int, scale: str) -> np.ndarray:
    pcs = SYNTH_SCALES[scale]
    return np.array(sorted({12 * o + root + d for o in range(2, 7) for d in pcs
                            if PITCH_LO <= 12 * o + root + d <= PITCH_HI}))


def _random_degrees(rng: np.random.Generator, n: int, n_deg: int, start: int) -> list[int]:
    """Mostly stepwise walk over scale-degree indices with occasional leaps."""
    deg = [start]
    for _ in range(n - 1):
        r = rng.random()
        if r < 0.55:
            step = int(rng.choice([-1, 1]))
        elif r < 0.72:
            step = 0
        elif r < 0.92:
            step = int(rng.choice([-3, -2, 2, 3]))
        else:
            step = int(rng.choice([-5, -4, 4, 5]))
        nxt = deg[-1] + step
        if nxt < 0 or nxt >= n_deg:
            nxt = deg[-1] - step
        deg.append(int(np.clip(nxt, 0, n_deg - 1)))
    return deg


def synthetic_phrases(n: int, seed: int = 0, min_notes: int = 5, max_notes: int = 16,
                      phrases_per_song: int = 4) -> list[Phrase]:
    """Generate ``n`` plausible short melodies without any MIDI data.

    Each "song" shares a key/scale/tempo grid across ``phrases_per_song``
    phrases so that :func:`split_by_song` is meaningful.
    """
    rng = np.random.default_rng(seed)
    out: list[Phrase] = []
    scale_names = list(SYNTH_SCALES)
    song = -1
    for i in range(n):
        if i % phrases_per_song == 0:
            song += 1
            root = int(rng.integers(0, 12))
            scale = scale_names[int(rng.integers(0, len(scale_names)))]
            grid = float(rng.choice([0.5, 1.0, 0.5]))
            pitches_in_scale = _scale_pitches(root, scale)
            ioi_choices = np.array([grid, grid, grid, 2 * grid, 2 * grid, 3 * grid, 4 * grid])
        n_deg = len(pitches_in_scale)
        length = int(rng.integers(min_notes, max_notes + 1))
        start = int(rng.integers(n_deg // 4, 3 * n_deg // 4))
        r = rng.random()
        if r < 0.3 and length >= 6:
            # repeated figure: motif of 3-5 notes repeated, maybe transposed by a degree
            m = int(rng.integers(3, min(5, length // 2) + 1))
            motif = _random_degrees(rng, m, n_deg, start)
            degs: list[int] = []
            shift = 0
            while len(degs) < length:
                degs.extend(int(np.clip(d + shift, 0, n_deg - 1)) for d in motif)
                if rng.random() < 0.4:
                    shift += int(rng.choice([-1, 1]))
            degs = degs[:length]
        elif r < 0.45:
            # repeated note / pedal: alternate a pedal with a wandering line
            pedal = start
            line = _random_degrees(rng, length, n_deg, min(n_deg - 1, start + 3))
            degs = [pedal if k % 2 == 0 else line[k] for k in range(length)]
        else:
            degs = _random_degrees(rng, length, n_deg, start)
        pitches = [int(pitches_in_scale[d]) for d in degs]
        iois = rng.choice(ioi_choices, size=length - 1)
        if rng.random() < 0.3 and length >= 4:
            # a run of even subdivisions
            iois[: length // 2] = grid
        onsets = np.concatenate([[0.0], np.cumsum(iois)])
        durs = list(iois * rng.choice([1.0, 1.0, 0.5])) + [float(grid)]
        out.append(Phrase(f"synth{seed}_{song:04d}", pitches, [float(o) for o in onsets],
                          [float(max(0.125, d)) for d in durs], "synthetic"))
    return out


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def split_by_song(phrases: list[Phrase], val_frac: float = 0.1, test_frac: float = 0.1,
                  seed: int = 0) -> dict[str, list[Phrase]]:
    """Split phrases into train/val/test with no song shared between splits."""
    songs = sorted({p.song_id for p in phrases})
    rng = np.random.default_rng(seed)
    rng.shuffle(songs)
    n = len(songs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_ids = set(songs[:n_test])
    val_ids = set(songs[n_test:n_test + n_val])
    out: dict[str, list[Phrase]] = {"train": [], "val": [], "test": []}
    for p in phrases:
        key = "test" if p.song_id in test_ids else ("val" if p.song_id in val_ids else "train")
        out[key].append(p)
    return out


def split_ids(splits: dict[str, list[Phrase]]) -> dict[str, list[str]]:
    return {k: sorted({p.song_id for p in v}) for k, v in splits.items()}


def apply_split_ids(phrases: list[Phrase], ids: dict[str, list[str]]) -> dict[str, list[Phrase]]:
    lookup = {sid: k for k, v in ids.items() for sid in v}
    out: dict[str, list[Phrase]] = {k: [] for k in ids}
    for p in phrases:
        k = lookup.get(p.song_id)
        if k is not None:
            out[k].append(p)
    return out


# --------------------------------------------------------------------------- #
# Features and torch dataset
# --------------------------------------------------------------------------- #


def sketch_to_features(sketch: Sketch) -> dict[str, np.ndarray]:
    """Per-note model inputs from a sketch (tempo- and register-invariant).

    Returns ``cont`` (n, 6): normalised IOI (in units of the median IOI, i.e.
    roughly beats), log IOI, centred height / 12, height difference / 12,
    confidence, missing flag; and ``gest`` (n, K) gesture multi-hot.
    """
    sketch = sketch.sorted()
    n = len(sketch)
    t = np.asarray(sketch.onsets, dtype=float)
    iois = np.diff(t) if n > 1 else np.zeros(0)
    pos = iois[iois > 1e-4]
    med = float(np.median(pos)) if len(pos) else 1.0
    ioi = np.concatenate([[0.0], iois / med]) if n > 0 else np.zeros(0)
    log_ioi = np.where(ioi > 0, np.log(np.maximum(ioi, 1e-3)), 0.0)

    ys = sketch.heights
    missing = np.array([y is None for y in ys], dtype=float)
    y = np.array([0.0 if v is None else float(v) for v in ys], dtype=float)
    known = missing < 0.5
    if known.any():
        y = np.where(known, y - y[known].mean(), 0.0)
    dy = np.zeros(n)
    for i in range(1, n):
        if known[i] and known[i - 1]:
            dy[i] = y[i] - y[i - 1]
    conf = np.array([o.confidence for o in sketch.observations], dtype=float)
    cont = np.stack([ioi, log_ioi, y / 12.0, dy / 12.0, conf, missing], axis=1).astype(np.float32)

    gest = np.zeros((n, len(GESTURE_KINDS)), dtype=np.float32)
    for g in sketch.gestures:
        if g.kind in GESTURE_KINDS:
            k = GESTURE_KINDS.index(g.kind)
            lo, hi = max(0, g.start), min(n - 1, g.end)
            if lo <= hi:
                gest[lo:hi + 1, k] = 1.0
    return {"cont": cont, "gest": gest}


def example_to_tensors(sketch: Sketch, intervals: list[int] | None,
                       iois_beats: list[float] | None = None) -> dict[str, torch.Tensor]:
    """Model inputs for one sketch; ``intervals`` and ``iois_beats`` (true gaps in
    beats, one per interval) add the training targets when given."""
    f = sketch_to_features(sketch)
    n = len(f["cont"])
    item = {
        "cont": torch.from_numpy(f["cont"]),
        "gest": torch.from_numpy(f["gest"]),
        "mask": torch.ones(n, dtype=torch.bool),
    }
    item["genre"] = torch.tensor(genre_id(sketch.meta.get("genre")), dtype=torch.long)
    if intervals is not None:
        item["target"] = torch.tensor([interval_to_class(d) for d in intervals], dtype=torch.long)
    if iois_beats is not None:
        item["rhythm"] = torch.tensor([rhythm_class(b) for b in iois_beats], dtype=torch.long)
    return item


class PhraseDataset(Dataset):
    """Phrases -> simulated sketches -> tensors, resampled every epoch.

    Noise is drawn from an RNG seeded by ``(seed, epoch, index)`` so runs are
    reproducible and a validation set is fixed when ``epoch`` is kept at 0.
    """

    def __init__(self, phrases: list[Phrase], simulator: "SketchSimulator", seed: int = 0,
                 epoch: int = 0, genre_dropout: float = 0.0) -> None:
        self.phrases = phrases
        self.simulator = simulator
        self.seed = seed
        self.epoch = epoch
        self.genre_dropout = genre_dropout  # probability of showing the model "unknown" instead of the label

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.phrases)

    def simulate(self, idx: int):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        return self.simulator.simulate(self.phrases[idx], rng=rng)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        res = self.simulate(idx)
        genre = genre_of_source(self.phrases[idx].source)
        if self.genre_dropout > 0:
            rng = np.random.default_rng([self.seed, self.epoch, idx, 7])
            if rng.random() < self.genre_dropout:
                genre = "unknown"
        res.sketch.meta["genre"] = genre
        iois = [b - a for a, b in zip(res.onsets_beats, res.onsets_beats[1:])]
        return example_to_tensors(res.sketch, res.intervals, iois)


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of dataset items to the batch max length."""
    B = len(items)
    n_max = max(int(it["cont"].shape[0]) for it in items)
    cont = torch.zeros(B, n_max, items[0]["cont"].shape[1])
    gest = torch.zeros(B, n_max, items[0]["gest"].shape[1])
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    has_target = "target" in items[0]
    has_rhythm = "rhythm" in items[0]
    target = torch.full((B, max(n_max - 1, 1)), PAD_TARGET, dtype=torch.long)
    rhythm = torch.full((B, max(n_max - 1, 1)), PAD_TARGET, dtype=torch.long)
    for b, it in enumerate(items):
        n = it["cont"].shape[0]
        cont[b, :n] = it["cont"]
        gest[b, :n] = it["gest"]
        mask[b, :n] = True
        if has_target:
            target[b, :n - 1] = it["target"]
        if has_rhythm:
            rhythm[b, :n - 1] = it["rhythm"]
    batch = {"cont": cont, "gest": gest, "mask": mask,
             "genre": torch.stack([it.get("genre", torch.tensor(0)) for it in items])}
    if has_target:
        batch["target"] = target
    if has_rhythm:
        batch["rhythm"] = rhythm
    return batch
