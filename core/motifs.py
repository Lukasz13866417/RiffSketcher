"""Hierarchical gesture / trope library and rule-based gesture suggestions.

A *gesture* is a musical annotation the user attaches to a note or a region.
Some are articulation (palm mute, bend), some describe pitch shape
(ascending run, repeated figure) and some are high-level structure (buildup).
Inference consumes gestures as soft or hard constraints, see
:mod:`core.constraints`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .representation import GestureTag, Sketch

# scope: "note" = single note, "region" = a span of notes
GESTURE_LIBRARY: dict[str, dict[str, Any]] = {
    "articulation": {
        "label": "Articulation",
        "items": {
            "palm_mute": {"label": "Palm mute", "scope": "region", "desc": "Muted, chunky attack"},
            "accent": {"label": "Accent", "scope": "note", "desc": "Hit harder"},
            "bend": {"label": "Bend", "scope": "note", "desc": "Pitch bends up after the attack",
                     "params": {"semitones": 2}},
            "slide": {"label": "Slide", "scope": "note", "desc": "Slides into the next note"},
            "vibrato": {"label": "Vibrato", "scope": "note", "desc": "Wobble on a held note"},
            "staccato": {"label": "Staccato", "scope": "region", "desc": "Short, detached notes"},
        },
    },
    "contour": {
        "label": "Pitch shape",
        "items": {
            "ascending_run": {"label": "Ascending run", "scope": "region", "desc": "Each note higher than the last"},
            "descending_run": {"label": "Descending run", "scope": "region", "desc": "Each note lower than the last"},
            "repeated_note": {"label": "Repeated note", "scope": "region", "desc": "Same pitch throughout"},
            "tremolo": {"label": "Tremolo / chug run", "scope": "note",
                        "desc": "One tap stands for a fast run of this note; hold the tap as long as the run"},
            "repeated_figure": {"label": "Repeated figure", "scope": "region",
                                "desc": "A short pattern that repeats", "params": {"period": 4}},
            "pedal_tone": {"label": "Pedal tone", "scope": "region",
                           "desc": "Alternates with a fixed low note", "params": {"period": 2}},
            "arpeggio": {"label": "Arpeggio", "scope": "region", "desc": "Chord tones in sequence"},
            "big_leap": {"label": "Big leap", "scope": "note", "desc": "Jumps far (octave-ish) to the next note"},
        },
    },
    "structure": {
        "label": "Structure",
        "items": {
            "buildup": {"label": "Buildup", "scope": "region", "desc": "Rising tension, gets louder / higher"},
            "tapping_run": {"label": "Tapping / fast run", "scope": "region",
                            "desc": "Fast, even notes; taps you missed can be filled in"},
            "call": {"label": "Call", "scope": "region", "desc": "First half of a call-and-response"},
            "response": {"label": "Response", "scope": "region", "desc": "Answering phrase"},
            "ending": {"label": "Ending", "scope": "region", "desc": "Resolves / lands on the home note"},
        },
    },
}


def flat_library() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cat_key, cat in GESTURE_LIBRARY.items():
        for key, item in cat["items"].items():
            out[key] = {**item, "category": cat_key}
    return out


def is_known(kind: str) -> bool:
    return kind in flat_library()


# --------------------------------------------------------------------------- #
# Rule-based suggestions
# --------------------------------------------------------------------------- #


def _filled_heights(sketch: Sketch) -> np.ndarray:
    ys = [o.y for o in sketch.observations]
    if not ys:
        return np.zeros(0)
    idx = [i for i, y in enumerate(ys) if y is not None]
    if not idx:
        return np.zeros(len(ys))
    known = np.array([ys[i] for i in idx], dtype=float)
    return np.interp(np.arange(len(ys)), np.array(idx, dtype=float), known)


def suggest_gestures(sketch: Sketch, min_run: int = 3, same_tol: float = 0.6) -> list[GestureTag]:
    """Propose gestures from the raw sketch using simple rules.

    These are *suggestions* with modest confidence; the user accepts or
    rejects them in the UI.
    """
    sketch = sketch.sorted()
    y = _filled_heights(sketch)
    n = len(y)
    out: list[GestureTag] = []
    if n < 2:
        return out
    d = np.diff(y)
    sign = np.where(d > same_tol, 1, np.where(d < -same_tol, -1, 0))

    # monotonic runs and repeated notes
    i = 0
    while i < n - 1:
        j = i
        while j < n - 1 and sign[j] == sign[i]:
            j += 1
        length = j - i + 1  # notes in the run
        if length >= min_run:
            kind = {1: "ascending_run", -1: "descending_run", 0: "repeated_note"}[int(sign[i])]
            conf = min(0.9, 0.5 + 0.1 * (length - min_run))
            out.append(GestureTag(kind, i, j, conf))
        i = j if j > i else i + 1

    # big leaps
    for k, dk in enumerate(d):
        if abs(dk) >= 10:
            out.append(GestureTag("big_leap", k, k, 0.6))

    # repeated figure via autocorrelation of the height sequence
    if n >= 6:
        yc = y - y.mean()
        best_p, best_r = None, 0.0
        for p in range(2, n // 2 + 1):
            a, b = yc[:-p], yc[p:]
            if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best_r:
                best_p, best_r = p, r
        if best_p is not None and best_r > 0.85:
            out.append(GestureTag("repeated_figure", 0, n - 1, min(0.9, best_r), {"period": best_p}))

    # pedal tone: every other note is the same low value
    if n >= 4:
        evens, odds = y[0::2], y[1::2]
        for name, pedal, other in (("even", evens, odds), ("odd", odds, evens)):
            if np.std(pedal) < same_tol and np.mean(other) - np.mean(pedal) > 2.0:
                out.append(GestureTag("pedal_tone", 0, n - 1, 0.6,
                                      {"period": 2, "offset": 0 if name == "even" else 1}))
                break

    # buildup: slow upward trend in heights with shortening IOIs
    if n >= 6:
        t = np.array(sketch.onsets)
        iois = np.diff(t)
        trend = np.polyfit(np.arange(n), y, 1)[0]
        speeding = np.polyfit(np.arange(len(iois)), iois, 1)[0] < 0
        if trend > 0.3 and speeding:
            out.append(GestureTag("buildup", 0, n - 1, 0.5))

    return out


# --------------------------------------------------------------------------- #
# Trope catalogue (mined from real music by ml/tropes.py)
# --------------------------------------------------------------------------- #

from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
import json  # noqa: E402

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "data" / "tropes.json"

TROPE_LABELS: dict[str, str] = {
    "chug_run": "chug run (repeated note)",
    "ascending_run": "ascending run",
    "descending_run": "descending run",
    "pedal_tone": "pedal tone",
    "tritone_move": "tritone move",
    "octave_leap": "octave leap",
    "gallop": "gallop rhythm",
    "repeated_figure": "repeated figure",
    "pentatonic_lick": "pentatonic lick",
    "ends_on_lowest_note": "ends on its lowest note",
}


@lru_cache(maxsize=1)
def load_catalogue(path: str | None = None) -> dict[str, Any]:
    """The mined catalogue, or an empty one when the file is missing."""
    p = Path(path) if path else CATALOGUE_PATH
    if not p.exists():
        return {"genres": {}}
    return json.loads(p.read_text())


def catalogue_genre(genre: str | None, catalogue: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cat = catalogue if catalogue is not None else load_catalogue()
    genres = cat.get("genres", {})
    if genre in genres:
        return genres[genre]
    return genres.get("all")


def _detect_named(pitches: list[int], gaps: list[float]) -> list[tuple[str, int, int]]:
    """Run the catalogue's named-trope detectors on a concrete riff."""
    from ml.datasets import Phrase  # light dataclass, no torch import
    from ml.tropes import detect_tropes

    onsets = [0.0]
    for g in gaps:
        onsets.append(onsets[-1] + g)
    return detect_tropes(Phrase("riff", list(pitches), onsets, [1.0] * len(pitches), "riff"))


def label_tropes(pitches: list[int], gaps_beats: list[float], genre: str | None = None) -> dict[str, Any]:
    """Name the tropes a candidate riff uses and say how common they are in the genre.

    Returns ``{"tropes": [{kind, label, start, end, song_frac}], "familiarity": f,
    "familiar_ngrams": k, "ngrams": m}`` where familiarity is the share of the
    riff's interval trigrams that are frequent in the genre's catalogue.
    """
    g = catalogue_genre(genre)
    found = _detect_named(pitches, gaps_beats) if len(pitches) >= 3 else []
    seen: set[str] = set()
    tropes = []
    for kind, s, e in found:
        if kind in seen:
            continue
        seen.add(kind)
        share = (g or {}).get("tropes", {}).get(kind, {}).get("song_frac")
        tropes.append({"kind": kind, "label": TROPE_LABELS.get(kind, kind), "start": s, "end": e,
                       "song_frac": share})
    familiar, total = 0, 0
    if g:
        clip = load_catalogue().get("interval_clip", 12)
        common = {tuple(x["pattern"]) for x in g.get("interval_ngrams", []) if len(x["pattern"]) == 3}
        iv = [max(-clip, min(clip, b - a)) for a, b in zip(pitches, pitches[1:])]
        for i in range(len(iv) - 2):
            total += 1
            familiar += tuple(iv[i:i + 3]) in common
    return {"tropes": tropes, "familiarity": (familiar / total) if total else None,
            "familiar_ngrams": familiar, "ngrams": total}


def catalogue_summary(genre: str | None = None) -> dict[str, Any]:
    """Compact view of one genre's catalogue for the API / UI."""
    g = catalogue_genre(genre)
    if not g:
        return {"available": False}
    return {
        "available": True, "songs": g["songs"], "phrases": g["phrases"],
        "tropes": [{"kind": k, "label": TROPE_LABELS.get(k, k), "song_frac": v["song_frac"],
                    "example": (v["examples"][0] if v.get("examples") else None)}
                   for k, v in sorted(g["tropes"].items(), key=lambda kv: -kv[1]["song_frac"])],
        "scales": g["scales"],
        "interval_unigram": g["interval_unigram"],
        "top_interval_trigrams": [x for x in g["interval_ngrams"] if len(x["pattern"]) == 3][:10],
        "top_rhythm_trigrams": [x for x in g["rhythm_ngrams"] if len(x["pattern"]) == 3][:5],
        "contours": g["contours"][:8],
    }
