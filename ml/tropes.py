"""Mine a catalogue of musical tropes from symbolic phrases, per genre.

"Trope" here means a recurring pattern we can count in real music: interval
n-grams, rhythm n-grams, named figures (chug runs, gallops, pedal tones,
tritone moves ...), scale preferences and contour habits. The catalogue is a
plain JSON file; :mod:`core.motifs` reads it to label candidate riffs and to
make better suggestions, with no ML involved.

Counts are *document frequencies*: a pattern is counted once per song, so a
song that repeats one riff fifty times does not dominate. Run::

    uv run python -m ml.tropes --data data/processed/mixed.jsonl --out data/tropes.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.inference import SCALES
from ml.datasets import Phrase, genre_of_source, load_phrases, rhythm_class, RHYTHM_CLASSES

INTERVAL_CLIP = 12  # n-grams use intervals clipped to +-12 so octave-and-more leaps pool


# --------------------------------------------------------------------------- #
# named trope detectors: phrase -> list of (name, start, end)
# --------------------------------------------------------------------------- #


def _iois(p: Phrase) -> list[float]:
    return [b - a for a, b in zip(p.onsets_beats, p.onsets_beats[1:])]


def _runs(seq: list[int], pred, min_len: int) -> list[tuple[int, int]]:
    """Maximal runs of consecutive *interval* indices satisfying ``pred``; returns note spans."""
    out, i = [], 0
    while i < len(seq):
        if pred(seq[i]):
            j = i
            while j + 1 < len(seq) and pred(seq[j + 1]):
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j + 1))  # notes i .. j+1
            i = j + 1
        else:
            i += 1
    return out


def detect_tropes(p: Phrase) -> list[tuple[str, int, int]]:
    iv = [b - a for a, b in zip(p.pitches, p.pitches[1:])]
    gaps = _iois(p)
    found: list[tuple[str, int, int]] = []
    n = len(p.pitches)
    if n < 3:
        return found
    # chug run: 3+ repeated notes in a row
    for s, e in _runs(iv, lambda d: d == 0, 3):
        found.append(("chug_run", s, e))
    # ascending / descending run: 3+ steps in one direction
    for s, e in _runs(iv, lambda d: 0 < d <= 2, 3):
        found.append(("ascending_run", s, e))
    for s, e in _runs(iv, lambda d: -2 <= d < 0, 3):
        found.append(("descending_run", s, e))
    # pedal tone: notes alternate with a fixed pitch for 5+ notes
    for start in range(n - 4):
        for end in range(min(n, start + 16), start + 4, -1):
            seg = p.pitches[start:end]
            ev, od = seg[0::2], seg[1::2]
            if len(set(ev)) == 1 and len(set(od)) > 1 and all(x != ev[0] for x in od):
                found.append(("pedal_tone", start, end - 1))
                break
    # tritone move and octave leap
    for k, d in enumerate(iv):
        if abs(d) == 6:
            found.append(("tritone_move", k, k + 1))
        if abs(d) == 12:
            found.append(("octave_leap", k, k + 1))
    # gallop: eighth + two sixteenths (long-short-short at sixteenth speed), at least twice
    if len(gaps) >= 6:
        hits = [k for k in range(len(gaps) - 2)
                if 0 < gaps[k + 1] <= 0.26 and abs(gaps[k + 1] - gaps[k + 2]) < 0.05
                and 1.8 <= gaps[k] / gaps[k + 1] <= 2.2]
        if len(hits) >= 2:
            for k in hits:
                found.append(("gallop", k, k + 3))
    # repeated figure: a figure of `period` notes repeats at least twice (two figures of p
    # notes give 2p-1 intervals), and it is not just a chug of identical notes
    m = len(iv)
    if any(d != 0 for d in iv):
        for period in range(2, (m + 1) // 2 + 1):
            if m >= 2 * period - 1 and all(iv[i] == iv[i + period] for i in range(m - period)):
                found.append(("repeated_figure", 0, n - 1))
                break
    # pentatonic lick: 6+ notes using 4+ pitch classes that all fit one pentatonic scale,
    # with the minor-third leaps that give the pentatonic its sound
    pcs = {x % 12 for x in p.pitches}
    if n >= 6 and len(pcs) >= 4 and sum(abs(d) == 3 for d in iv) >= 2:
        for root in range(12):
            if pcs <= {(root + d) % 12 for d in SCALES["minor_pentatonic"]}:
                found.append(("pentatonic_lick", 0, n - 1))
                break
    # ends on its lowest note after a descent (a common riff ending)
    if p.pitches[-1] == min(p.pitches) and p.pitches[-2] > p.pitches[-1]:
        found.append(("ends_on_lowest_note", n - 2, n - 1))
    return found


# --------------------------------------------------------------------------- #
# scale fit and contour
# --------------------------------------------------------------------------- #


SCALE_FAMILIES: dict[str, tuple[str, ...]] = {
    # relative major/minor share pitch classes, so only the family is identifiable from pitches alone
    "pentatonic": ("minor_pentatonic",),
    "diatonic": ("minor",),  # covers major and all its modes
    "blues": ("blues",),
    "harmonic_minor": ("harmonic_minor",),
}


def scale_family(pitches: list[int]) -> str:
    """Smallest scale family that contains every pitch class, else 'chromatic'."""
    pcs = {x % 12 for x in pitches}
    for family, scales in SCALE_FAMILIES.items():
        for scale in scales:
            for root in range(12):
                if pcs <= {(root + d) % 12 for d in SCALES[scale]}:
                    return family
    return "chromatic"


def best_scale(pitches: list[int]) -> tuple[str, float]:
    """Kept for callers that want a named scale: (family, fraction of notes in it)."""
    return scale_family(pitches), 1.0


def contour_word(intervals: Iterable[int]) -> str:
    """Compressed up/down/same word, e.g. 'u d s u'. Runs are collapsed."""
    out: list[str] = []
    for d in intervals:
        c = "u" if d > 0 else ("d" if d < 0 else "s")
        if not out or out[-1] != c:
            out.append(c)
    return " ".join(out)


# --------------------------------------------------------------------------- #
# mining
# --------------------------------------------------------------------------- #


@dataclass
class GenreStats:
    songs: set = field(default_factory=set)
    phrases: int = 0
    interval_uni: Counter = field(default_factory=Counter)
    interval_ngrams: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    interval_ngram_songs: dict[int, dict] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    rhythm_ngrams: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    rhythm_ngram_songs: dict[int, dict] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    figure_ngram_songs: dict[int, dict] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    tropes: Counter = field(default_factory=Counter)
    trope_songs: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    trope_examples: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    scales: Counter = field(default_factory=Counter)
    contours: Counter = field(default_factory=Counter)
    range_semitones: list[int] = field(default_factory=list)
    length_notes: list[int] = field(default_factory=list)


def _clip(d: int) -> int:
    return max(-INTERVAL_CLIP, min(INTERVAL_CLIP, d))


def mine(phrases: list[Phrase], ngram_max: int = 5, examples_per_trope: int = 5) -> dict[str, GenreStats]:
    stats: dict[str, GenreStats] = defaultdict(GenreStats)
    for p in phrases:
        if len(p.pitches) < 3:
            continue
        g = genre_of_source(p.source)
        for key in (g, "all"):
            st = stats[key]
            st.songs.add(p.song_id)
            st.phrases += 1
            iv = [_clip(b - a) for a, b in zip(p.pitches, p.pitches[1:])]
            gaps = [rhythm_class(x) for x in _iois(p)]
            st.interval_uni.update(iv)
            for k in range(2, ngram_max + 1):
                for i in range(len(iv) - k + 1):
                    gram = tuple(iv[i:i + k])
                    st.interval_ngrams[k][gram] += 1
                    st.interval_ngram_songs[k][gram].add(p.song_id)
                    rg = tuple(gaps[i:i + k])
                    st.rhythm_ngrams[k][rg] += 1
                    st.rhythm_ngram_songs[k][rg].add(p.song_id)
                    st.figure_ngram_songs[k][(gram, rg)].add(p.song_id)
            for name, s, e in detect_tropes(p):
                st.tropes[name] += 1
                st.trope_songs[name].add(p.song_id)
                if len(st.trope_examples[name]) < examples_per_trope and p.song_id not in {x["song"] for x in st.trope_examples[name]}:
                    st.trope_examples[name].append({"song": p.song_id, "pitches": p.pitches[s:e + 1],
                                                    "onsets_beats": [round(x - p.onsets_beats[s], 3) for x in p.onsets_beats[s:e + 1]]})
            st.scales[scale_family(p.pitches)] += 1
            st.contours[contour_word(iv)] += 1
            st.range_semitones.append(max(p.pitches) - min(p.pitches))
            st.length_notes.append(len(p.pitches))
    return stats


def to_catalogue(stats: dict[str, GenreStats], top: int = 40, min_songs: int = 3) -> dict[str, Any]:
    out: dict[str, Any] = {"version": 1, "interval_clip": INTERVAL_CLIP,
                           "rhythm_classes": list(RHYTHM_CLASSES), "genres": {}}
    for g, st in stats.items():
        n_songs = max(len(st.songs), 1)
        total_iv = sum(st.interval_uni.values()) or 1
        genre: dict[str, Any] = {
            "songs": len(st.songs), "phrases": st.phrases,
            "interval_unigram": {str(k): round(v / total_iv, 4) for k, v in sorted(st.interval_uni.items())},
            "interval_ngrams": [], "rhythm_ngrams": [], "figures": [],
            "tropes": {}, "scales": {}, "contours": [],
            "range_semitones_median": float(np.median(st.range_semitones)) if st.range_semitones else 0.0,
            "length_notes_median": float(np.median(st.length_notes)) if st.length_notes else 0.0,
        }
        for k in sorted(st.interval_ngram_songs):
            rows = [(gram, len(songs)) for gram, songs in st.interval_ngram_songs[k].items() if len(songs) >= min_songs]
            rows.sort(key=lambda r: -r[1])
            genre["interval_ngrams"] += [{"pattern": list(gram), "songs": c, "song_frac": round(c / n_songs, 4)}
                                         for gram, c in rows[:top]]
            rows = [(gram, len(songs)) for gram, songs in st.rhythm_ngram_songs[k].items() if len(songs) >= min_songs]
            rows.sort(key=lambda r: -r[1])
            genre["rhythm_ngrams"] += [{"pattern": [round(RHYTHM_CLASSES[c], 4) for c in gram], "songs": c2,
                                        "song_frac": round(c2 / n_songs, 4)} for gram, c2 in rows[:top]]
            rows = [(key, len(songs)) for key, songs in st.figure_ngram_songs[k].items() if len(songs) >= min_songs]
            rows.sort(key=lambda r: -r[1])
            genre["figures"] += [{"intervals": list(gram), "gaps": [round(RHYTHM_CLASSES[c], 4) for c in rg],
                                  "songs": c2, "song_frac": round(c2 / n_songs, 4)} for (gram, rg), c2 in rows[:top // 2]]
        for name, cnt in st.tropes.most_common():
            genre["tropes"][name] = {"phrases": cnt, "phrase_frac": round(cnt / max(st.phrases, 1), 4),
                                     "songs": len(st.trope_songs[name]),
                                     "song_frac": round(len(st.trope_songs[name]) / n_songs, 4),
                                     "examples": st.trope_examples[name]}
        tot = sum(st.scales.values()) or 1
        genre["scales"] = {k: round(v / tot, 4) for k, v in st.scales.most_common()}
        tot = sum(st.contours.values()) or 1
        genre["contours"] = [{"word": w, "frac": round(c / tot, 4)} for w, c in st.contours.most_common(top)]
        out["genres"][g] = genre
    return out


def summary(cat: dict[str, Any]) -> str:
    lines = []
    for g, genre in cat["genres"].items():
        lines.append(f"\n== {g}: {genre['songs']} songs, {genre['phrases']} phrases, "
                     f"median range {genre['range_semitones_median']:.0f} st, median length {genre['length_notes_median']:.0f} notes")
        lines.append("  tropes (share of songs): " + ", ".join(f"{n} {t['song_frac']:.0%}" for n, t in genre["tropes"].items()))
        lines.append("  scale family (smallest fitting): " + ", ".join(f"{k} {v:.0%}" for k, v in list(genre["scales"].items())[:5]))
        top_iv = sorted(genre["interval_unigram"].items(), key=lambda kv: -kv[1])[:6]
        lines.append("  intervals: " + ", ".join(f"{k}: {v:.0%}" for k, v in top_iv))
        tri = [x for x in genre["interval_ngrams"] if len(x["pattern"]) == 3][:5]
        lines.append("  top interval trigrams: " + "; ".join(f"{x['pattern']} ({x['song_frac']:.0%} of songs)" for x in tri))
        rh = [x for x in genre["rhythm_ngrams"] if len(x["pattern"]) == 3][:3]
        lines.append("  top rhythm trigrams (beats): " + "; ".join(f"{x['pattern']} ({x['song_frac']:.0%})" for x in rh))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/mixed.jsonl")
    ap.add_argument("--out", default="data/tropes.json")
    ap.add_argument("--top", type=int, default=40, help="patterns kept per n-gram length and genre")
    ap.add_argument("--min-songs", type=int, default=3)
    ap.add_argument("--ngram-max", type=int, default=5)
    args = ap.parse_args(argv)
    phrases = load_phrases(args.data)
    cat = to_catalogue(mine(phrases, args.ngram_max), args.top, args.min_songs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cat, indent=0))
    print(f"{len(phrases)} phrases -> {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")
    print(summary(cat))


if __name__ == "__main__":
    main()
