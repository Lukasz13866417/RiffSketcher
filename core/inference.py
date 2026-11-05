"""Deterministic reconstruction: sketch -> ranked candidate riffs.

This is phase 3 of the plan ("first playback").  It has no learned
components; it is a Viterbi search over scale tones under a simple cost:

* emission:   how far a pitch is from the height the user drew
* scale:      penalty for notes outside the candidate key/scale
* transition: mild preference for small intervals, hard constraints from
              gestures and explicit user assertions
* key prior:  common scales first, chromatic as a last resort

Every candidate key/scale/height-scaling triple yields one best path; the
paths are de-duplicated and ranked.  The same interface
(:class:`Reconstructor`) will later be implemented by the trained model, so
the API and UI don't care which one is behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .constraints import ConstraintSet, apply_articulation, build_constraints
from .representation import (N_INTERVAL_CLASSES, NOTE_NAMES, Note, Riff, Sketch,
                             interval_to_class)
from .rhythm import quantise

SCALES: dict[str, list[int]] = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "chromatic": list(range(12)),
}

SCALE_PRIOR: dict[str, float] = {
    "minor": 0.0, "major": 0.0, "minor_pentatonic": 0.2, "major_pentatonic": 0.4,
    "blues": 0.6, "dorian": 0.5, "phrygian": 0.6, "mixolydian": 0.5,
    "harmonic_minor": 0.7, "chromatic": 1.5,
}

CHORD_TONES = {0, 2, 4}  # scale-degree indices of a triad (root, third, fifth)


@dataclass
class InferenceConfig:
    center_pitch: int = 52  # E3, comfortable guitar riff register
    pitch_lo: int = 36  # C2
    pitch_hi: int = 84  # C6
    height_scalings: tuple[float, ...] = (1.0, 0.8, 1.25)
    window: float = 4.0  # +- semitones around the drawn height to consider
    sigma: float = 0.8  # emission std (semitones) at confidence 1
    out_of_scale: float = 2.5
    leap_per_semitone: float = 0.0
    big_leap_penalty: float = 0.3
    tonic_end_bonus: float = 0.3
    tonic_start_bonus: float = 0.3
    n_candidates: int = 5
    scales: tuple[str, ...] = tuple(SCALES)
    roots: tuple[int, ...] = tuple(range(12))


class Reconstructor(Protocol):
    def reconstruct(self, sketch: Sketch, n: int = 5) -> list[Riff]: ...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def filled_heights(sketch: Sketch) -> np.ndarray:
    ys = [o.y for o in sketch.observations]
    n = len(ys)
    if n == 0:
        return np.zeros(0)
    idx = [i for i, y in enumerate(ys) if y is not None]
    if not idx:
        return np.zeros(n)
    known = np.array([ys[i] for i in idx], dtype=float)
    return np.interp(np.arange(n), np.array(idx, dtype=float), known)


def key_name(root: int, scale: str) -> str:
    return f"{NOTE_NAMES[root]} {scale.replace('_', ' ')}"


def _targets(sketch: Sketch, cs: ConstraintSet, cfg: InferenceConfig, scaling: float) -> np.ndarray:
    y = filled_heights(sketch) * scaling
    if len(y) == 0:
        return y
    if cs.fixed_pitch:
        # anchor so that fixed pitches are matched on average
        shift = float(np.mean([p - y[i] for i, p in cs.fixed_pitch.items() if i < len(y)]))
    else:
        shift = cfg.center_pitch - float(np.mean(y))
    return y + shift


def _viterbi(targets: np.ndarray, conf: np.ndarray, cs: ConstraintSet, cfg: InferenceConfig,
             root: int, scale: str, extra_fixed: dict[int, int]) -> tuple[list[int], float]:
    n = len(targets)
    scale_pcs = {(root + d) % 12 for d in SCALES[scale]}
    chord_pcs = {(root + SCALES[scale][d]) % 12 for d in CHORD_TONES if d < len(SCALES[scale])}
    fixed = {**cs.fixed_pitch, **extra_fixed}

    states: list[np.ndarray] = []
    emis: list[np.ndarray] = []
    for i in range(n):
        if i in fixed:
            s = np.array([fixed[i]])
        else:
            w = cfg.window + 3.0 * (1.0 - conf[i])
            lo = max(cfg.pitch_lo, int(math.floor(targets[i] - w)))
            hi = min(cfg.pitch_hi, int(math.ceil(targets[i] + w)))
            if lo > hi:  # drawn height is outside the playable range: use the nearest edge
                lo = hi = cfg.pitch_lo if targets[i] < cfg.pitch_lo else cfg.pitch_hi
            s = np.arange(lo, hi + 1)
        sigma = cfg.sigma + 2.0 * (1.0 - conf[i])
        e = ((s - targets[i]) / sigma) ** 2
        if scale != "chromatic":
            e = e + np.where(np.isin(s % 12, list(scale_pcs)), 0.0, cfg.out_of_scale)
        if i in cs.arpeggio:
            e = e - np.where(np.isin(s % 12, list(chord_pcs)), 0.5, 0.0)
        if i == n - 1 and (i in cs.ending or True):
            e = e - np.where(s % 12 == root, cfg.tonic_end_bonus * (2.0 if i in cs.ending else 1.0), 0.0)
        if i == 0:
            e = e - np.where(s % 12 == root, cfg.tonic_start_bonus, 0.0)
        states.append(s)
        emis.append(e)

    INF = 1e9
    cost = emis[0].copy()
    back: list[np.ndarray] = []
    for i in range(1, n):
        prev, cur = states[i - 1], states[i]
        d = cur[None, :] - prev[:, None]  # interval matrix prev x cur
        tr = cfg.leap_per_semitone * np.abs(d).astype(float)
        if (i - 1) in cs.big_leap:
            tr = tr + np.where(np.abs(d) >= 7, -1.0, cfg.big_leap_penalty)
        else:
            tr = tr + np.where(np.abs(d) > 7, cfg.big_leap_penalty, 0.0)
        # hard constraints on the step i-1 -> i
        if (i - 1) in cs.direction:
            tr = np.where(np.sign(d) == cs.direction[i - 1], tr, INF)
        if (i - 1) in cs.interval:
            tr = np.where(d == cs.interval[i - 1], tr, INF)
        total = cost[:, None] + tr + emis[i][None, :]
        back.append(np.argmin(total, axis=0))
        cost = np.min(total, axis=0)
    # backtrack
    j = int(np.argmin(cost))
    best = float(cost[j])
    path = [int(states[-1][j])]
    for i in range(n - 1, 0, -1):
        j = int(back[i - 1][j])
        path.append(int(states[i - 1][j]))
    path.reverse()
    return path, best


def _solve_with_same(targets, conf, cs, cfg, root, scale) -> tuple[list[int], float]:
    """Run Viterbi, then enforce non-adjacent 'same pitch' pairs by fixing and rerunning."""
    extra: dict[int, int] = {}
    path, cost = _viterbi(targets, conf, cs, cfg, root, scale, extra)
    for _ in range(4):
        changed = False
        for i, j in cs.same:
            if i < len(path) and j < len(path) and path[i] != path[j]:
                a, b = sorted((i, j))
                extra[b] = path[a] if a not in extra else extra[a]
                changed = True
        if not changed:
            break
        path, cost = _viterbi(targets, conf, cs, cfg, root, scale, extra)
    return path, cost


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


@dataclass
class DeterministicReconstructor:
    cfg: InferenceConfig = field(default_factory=InferenceConfig)

    def reconstruct(self, sketch: Sketch, n: int | None = None) -> list[Riff]:
        n = n or self.cfg.n_candidates
        sketch = sketch.sorted()
        if len(sketch) == 0:
            return []
        cs = build_constraints(sketch)
        rhythm = quantise(sketch.onsets, tempo_hint=sketch.tempo_hint)
        conf = np.array([o.confidence for o in sketch.observations], dtype=float)

        results: dict[tuple[int, ...], tuple[float, str]] = {}
        for scaling in self.cfg.height_scalings:
            targets = _targets(sketch, cs, self.cfg, scaling)
            scale_pen = 0.0 if scaling == 1.0 else 1.0
            for scale in self.cfg.scales:
                for root in self.cfg.roots:
                    path, cost = _solve_with_same(targets, conf, cs, self.cfg, root, scale)
                    if cost >= 1e8:
                        continue  # infeasible under hard constraints
                    cost += SCALE_PRIOR[scale] + scale_pen
                    key = tuple(path)
                    if key not in results or cost < results[key][0]:
                        results[key] = (cost, key_name(root, scale))

        ranked = sorted(results.items(), key=lambda kv: kv[1][0])[:n]
        riffs: list[Riff] = []
        for pitches, (cost, kname) in ranked:
            notes = [Note(p, o, d) for p, o, d in zip(pitches, rhythm.onsets_beats, rhythm.durations_beats)]
            riff = Riff(notes=notes, tempo=rhythm.tempo, key=kname, score=-cost,
                        meta={"method": "deterministic", "grid": rhythm.grid})
            riffs.append(apply_articulation(riff, cs))
        return riffs

    def interval_distribution(self, sketch: Sketch, n: int = 20) -> np.ndarray:
        """Soft (n-1, 49) distribution over interval classes, from candidate scores.

        Lets the deterministic engine be evaluated with the same metrics as a
        learned model (see ``ml/evaluate.py``).
        """
        riffs = self.reconstruct(sketch, n=n)
        m = len(sketch) - 1
        dist = np.full((max(m, 0), N_INTERVAL_CLASSES), 1e-6)
        if not riffs or m <= 0:
            return dist / dist.sum(axis=1, keepdims=True)
        scores = np.array([r.score for r in riffs])
        w = np.exp(scores - scores.max())
        w /= w.sum()
        for r, wi in zip(riffs, w):
            for k, d in enumerate(r.intervals):
                dist[k, interval_to_class(d)] += wi
        return dist / dist.sum(axis=1, keepdims=True)


def reconstruct(sketch: Sketch, n: int = 5, cfg: InferenceConfig | None = None) -> list[Riff]:
    return DeterministicReconstructor(cfg or InferenceConfig()).reconstruct(sketch, n)
