"""Phrase extraction, synthetic fallback, song-level splits and torch datasets.

A :class:`Phrase` is a short monophonic melody in beats.  Phrases come either
from MIDI files (:func:`phrases_from_midi`, used with POP909) or from the
MIDI-free generator :func:`synthetic_phrases` so that tests and smoke runs
work offline.  :class:`PhraseDataset` turns phrases into noisy sketches on the
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

from core.representation import Sketch, interval_to_class

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


@dataclass
class Phrase:
    """A short monophonic melody; onsets/durations in beats, first onset at 0."""

    song_id: str
    pitches: list[int]
    onsets_beats: list[float]
    durations_beats: list[float]
    source: str = "unknown"

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
                   source=str(d.get("source", "unknown")))


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


def phrases_from_midi(path: str | Path, min_notes: int = 5, max_notes: int = 16,
                      rest_split_beats: float = 2.0, song_id: str | None = None,
                      source: str = "midi") -> list[Phrase]:
    """Extract monophonic phrases (in beats) from one MIDI file.

    The melody track is chosen by name ("MELODY") or by a monophony/high-pitch
    heuristic.  Phrases are split at rests longer than ``rest_split_beats`` or
    when they reach ``max_notes``; phrases shorter than ``min_notes`` are
    dropped.
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

    phrases: list[Phrase] = []
    cur: list[int] = []

    def flush() -> None:
        if len(cur) >= min_notes:
            idx = list(cur)
            t0 = starts[idx[0]]
            phrases.append(Phrase(sid, [int(notes[i].pitch) for i in idx],
                                  [round(starts[i] - t0, 4) for i in idx],
                                  [round(durs[i], 4) for i in idx], source))
        cur.clear()

    for i in range(len(notes)):
        if cur:
            prev = cur[-1]
            rest = starts[i] - (starts[prev] + durs[prev])
            if rest > rest_split_beats or len(cur) >= max_notes:
                flush()
        cur.append(i)
    flush()
    return phrases


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


def example_to_tensors(sketch: Sketch, intervals: list[int] | None) -> dict[str, torch.Tensor]:
    f = sketch_to_features(sketch)
    n = len(f["cont"])
    item = {
        "cont": torch.from_numpy(f["cont"]),
        "gest": torch.from_numpy(f["gest"]),
        "mask": torch.ones(n, dtype=torch.bool),
    }
    if intervals is not None:
        item["target"] = torch.tensor([interval_to_class(d) for d in intervals], dtype=torch.long)
    return item


class PhraseDataset(Dataset):
    """Phrases -> simulated sketches -> tensors, resampled every epoch.

    Noise is drawn from an RNG seeded by ``(seed, epoch, index)`` so runs are
    reproducible and a validation set is fixed when ``epoch`` is kept at 0.
    """

    def __init__(self, phrases: list[Phrase], simulator: "SketchSimulator", seed: int = 0,
                 epoch: int = 0) -> None:
        self.phrases = phrases
        self.simulator = simulator
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.phrases)

    def simulate(self, idx: int):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        return self.simulator.simulate(self.phrases[idx], rng=rng)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        res = self.simulate(idx)
        return example_to_tensors(res.sketch, res.intervals)


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of dataset items to the batch max length."""
    B = len(items)
    n_max = max(int(it["cont"].shape[0]) for it in items)
    cont = torch.zeros(B, n_max, items[0]["cont"].shape[1])
    gest = torch.zeros(B, n_max, items[0]["gest"].shape[1])
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    has_target = "target" in items[0]
    target = torch.full((B, max(n_max - 1, 1)), PAD_TARGET, dtype=torch.long)
    for b, it in enumerate(items):
        n = it["cont"].shape[0]
        cont[b, :n] = it["cont"]
        gest[b, :n] = it["gest"]
        mask[b, :n] = True
        if has_target:
            target[b, :n - 1] = it["target"]
    batch = {"cont": cont, "gest": gest, "mask": mask}
    if has_target:
        batch["target"] = target
    return batch
