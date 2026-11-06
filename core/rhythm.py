"""Tempo estimation, grid quantisation and multi-pass merging for tapped onsets.

The user taps in real time, so we get onset times in seconds with jitter.
We need beats: a tempo plus onsets snapped to a subdivision grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RhythmResult:
    tempo: float  # BPM
    onsets_beats: list[float]  # quantised onsets, in beats, first note at 0
    durations_beats: list[float]  # gap to next note (last = median gap)
    grid: float  # grid unit in beats (0.25 = sixteenth note)
    error: float  # mean absolute quantisation error in seconds


def tempo_candidates(lo: float = 60.0, hi: float = 200.0, step: float = 1.0) -> np.ndarray:
    return np.arange(lo, hi + 1e-9, step)


def _quantisation_cost(iois: np.ndarray, tempo: float, grid: float) -> tuple[float, np.ndarray]:
    """Cost of snapping IOIs to multiples of ``grid`` beats at ``tempo``.

    Returns (cost, quantised IOIs in grid units).  Cost mixes the relative
    error and a mild penalty on complex multiples so that we prefer, say, a
    quarter-note pulse over a tempo twice as fast with every note on an eighth.
    """
    unit_s = 60.0 / tempo * grid
    units = iois / unit_s
    q = np.maximum(1, np.round(units))
    err = np.abs(units - q) / np.maximum(units, 1e-6)
    # complexity: prefer q values that are 1, 2, 4, 8 (powers of two in grid units)
    complexity = np.array([0.0 if (int(v) & (int(v) - 1)) == 0 else 0.15 for v in q])
    # mild preference against very fast grids (tiny unit => everything is "many units")
    speed_pen = 0.02 * np.mean(np.log1p(q))
    return float(np.mean(err) + np.mean(complexity) + speed_pen), q


def estimate_tempo(onsets_s: list[float], tempo_hint: float | None = None,
                   grid: float = 0.25) -> tuple[float, float]:
    """Return (tempo, cost).  ``tempo_hint`` narrows the search to +-10%."""
    onsets = np.asarray(sorted(onsets_s), dtype=float)
    if len(onsets) < 2:
        return (tempo_hint or 120.0), 0.0
    iois = np.diff(onsets)
    iois = iois[iois > 1e-3]
    if len(iois) == 0:
        return (tempo_hint or 120.0), 0.0
    if tempo_hint:
        cands = tempo_candidates(tempo_hint * 0.9, tempo_hint * 1.1, 0.5)
    else:
        cands = tempo_candidates()
    best_t, best_c = float(cands[0]), math.inf
    for t in cands:
        c, _ = _quantisation_cost(iois, float(t), grid)
        # gentle prior towards typical riff tempos (90-150)
        c += 0.0005 * abs(float(t) - 120.0) / 10.0
        if c < best_c:
            best_t, best_c = float(t), c
    return best_t, best_c


def quantise(onsets_s: list[float], tempo: float | None = None, tempo_hint: float | None = None,
             grid: float = 0.25) -> RhythmResult:
    """Snap onsets to a grid. Tempo is estimated unless given explicitly."""
    onsets = np.asarray(sorted(onsets_s), dtype=float)
    if len(onsets) == 0:
        return RhythmResult(tempo or tempo_hint or 120.0, [], [], grid, 0.0)
    if tempo is None:
        tempo, _ = estimate_tempo(list(onsets), tempo_hint, grid)
    unit_s = 60.0 / tempo * grid
    rel = onsets - onsets[0]
    # quantise cumulatively from IOIs so that drift is not amplified
    q_beats = [0.0]
    for ioi in np.diff(rel):
        units = max(1, int(round(ioi / unit_s)))
        q_beats.append(q_beats[-1] + units * grid)
    q = np.asarray(q_beats)
    err = float(np.mean(np.abs(q * 60.0 / tempo - rel))) if len(q) > 1 else 0.0
    gaps = list(np.diff(q))
    last = float(np.median(gaps)) if gaps else 1.0
    durations = [float(g) for g in gaps] + [last]
    return RhythmResult(float(tempo), [float(b) for b in q], durations, grid, err)


def merge_passes(passes: list[list[float]], tolerance_s: float = 0.08) -> list[float]:
    """Merge several recording passes of the same riff into one onset list.

    Each pass is aligned to the first by its first onset, then onsets closer
    than ``tolerance_s`` across passes are averaged.  Onsets present in only
    some passes are kept (useful when fast sections get dropped taps).
    """
    passes = [sorted(p) for p in passes if p]
    if not passes:
        return []
    ref = passes[0]
    clusters: list[list[float]] = [[t - ref[0]] for t in ref]
    for p in passes[1:]:
        off = p[0]
        for t in p:
            t = t - off
            # find nearest cluster centre
            best, best_d = None, tolerance_s
            for c in clusters:
                d = abs(float(np.mean(c)) - t)
                if d < best_d:
                    best, best_d = c, d
            if best is None:
                clusters.append([t])
            else:
                best.append(t)
    merged = sorted(float(np.mean(c)) for c in clusters)
    return merged


def beats_to_seconds(beats: float, tempo: float) -> float:
    return beats * 60.0 / tempo


def fill_missed_taps(sketch, kinds: tuple[str, ...] = ("tapping_run",), tolerance: float = 0.3):
    """Insert notes the user probably missed inside fast-run regions.

    For each gesture region of a kind in ``kinds`` the base spacing is the
    smallest inter-onset gap in the region. Any gap that is close to an
    integer multiple ``k >= 2`` of it (within ``tolerance`` of the multiple) is
    split into ``k`` even gaps by inserting ``k - 1`` observations with unknown
    height and low confidence, so the pitch model fills them in. Returns a new
    sorted sketch and the number of inserted notes; gesture and constraint
    indices are remapped.
    """
    from .representation import Observation, Sketch, GestureTag, Constraint

    sketch = sketch.sorted()
    obs = sketch.observations
    n = len(obs)
    # (position after original index i, time) for every insertion
    inserts: list[tuple[int, float]] = []
    for g in sketch.gestures:
        if g.kind not in kinds:
            continue
        idx = [i for i in g.indices() if 0 <= i < n]
        if len(idx) < 3:
            continue
        gaps = [obs[i + 1].t - obs[i].t for i in idx[:-1]]
        base = min(x for x in gaps if x > 1e-3) if any(x > 1e-3 for x in gaps) else None
        if base is None:
            continue
        for i, gap in zip(idx[:-1], gaps):
            k = int(round(gap / base))
            if k >= 2 and abs(gap / base - k) <= tolerance:
                for j in range(1, k):
                    inserts.append((i, obs[i].t + gap * j / k))
    if not inserts:
        return sketch, 0
    # build the new observation list and the old->new index map
    new_obs: list[Observation] = []
    remap: dict[int, int] = {}
    by_after: dict[int, list[float]] = {}
    for i, t in inserts:
        by_after.setdefault(i, []).append(t)
    for i, o in enumerate(obs):
        remap[i] = len(new_obs)
        new_obs.append(o)
        for t in sorted(by_after.get(i, [])):
            new_obs.append(Observation(t, None, 0.1))
    gestures = [GestureTag(g.kind, remap[g.start], remap[g.end], g.confidence, dict(g.params))
                for g in sketch.gestures]
    cons = [Constraint(c.kind, remap[c.i], None if c.j is None else remap[c.j], c.value)
            for c in sketch.constraints]
    out = Sketch(new_obs, gestures, cons, sketch.tempo_hint, dict(sketch.meta))
    out.meta["inserted"] = len(inserts)
    return out, len(inserts)
