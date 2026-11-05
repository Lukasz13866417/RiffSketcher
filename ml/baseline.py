"""Non-neural reference predictors with the same interface as the model.

Every predictor exposes ``predict(sketch) -> (n-1, 49)`` row-normalised
distributions over interval classes and ``candidates(sketch, k)`` returning
up to ``k`` full interval sequences, best first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from core.inference import SCALES, DeterministicReconstructor, InferenceConfig, filled_heights
from core.representation import (INTERVAL_MAX, INTERVAL_MIN, N_INTERVAL_CLASSES, Observation,
                                 Sketch, class_to_interval)

CLASS_INTERVALS = np.arange(INTERVAL_MIN, INTERVAL_MAX + 1, dtype=float)


class IntervalPredictor(Protocol):
    name: str

    def predict(self, sketch: Sketch) -> np.ndarray: ...

    def candidates(self, sketch: Sketch, k: int = 5) -> list[list[int]]: ...


def normalise_rows(dist: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dist = np.maximum(np.asarray(dist, dtype=float), 0.0) + eps
    return dist / dist.sum(axis=1, keepdims=True)


def topk_sequences(dist: np.ndarray, k: int = 5) -> list[list[int]]:
    """Top-k joint sequences under per-step independent distributions (beam)."""
    m = dist.shape[0]
    if m == 0:
        return [[]]
    logp = np.log(normalise_rows(dist))
    beams: list[tuple[float, list[int]]] = [(0.0, [])]
    for step in range(m):
        order = np.argsort(-logp[step])[:k]
        nxt = [(s + float(logp[step, c]), seq + [class_to_interval(int(c))]) for s, seq in beams for c in order]
        nxt.sort(key=lambda x: -x[0])
        beams = nxt[:k]
    return [seq for _, seq in beams]


def _gaussian_rows(centres: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    z = (CLASS_INTERVALS[None, :] - centres[:, None]) / sigmas[:, None]
    return normalise_rows(np.exp(-0.5 * z ** 2))


@dataclass
class ContourBaseline:
    """Round drawn height differences to the nearest semitone.

    Optionally snaps the rounded heights to the best-fitting of a few
    scales (by in-scale fraction) and mixes that peaked guess with the raw
    rounded guess.  ``sigma`` widens with low confidence.
    """

    sigma: float = 0.8
    scale_snap: bool = True
    snap_weight: float = 0.5
    scales: tuple[str, ...] = ("major", "minor", "minor_pentatonic", "dorian", "mixolydian", "blues")
    name: str = field(default="contour", init=False)

    def _snap_to_scale(self, p: np.ndarray) -> np.ndarray:
        best, best_frac = None, -1.0
        for scale in self.scales:
            pcs = SCALES[scale]
            for root in range(12):
                in_scale = {(root + d) % 12 for d in pcs}
                frac = float(np.mean([(int(x) % 12) in in_scale for x in p]))
                if frac > best_frac:
                    best, best_frac = (root, scale), frac
        assert best is not None
        root, scale = best
        allowed = {(root + d) % 12 for d in SCALES[scale]}
        out = p.copy()
        for i, x in enumerate(p):
            x = int(x)
            if x % 12 in allowed:
                continue
            for delta in (1, -1, 2, -2):
                if (x + delta) % 12 in allowed:
                    out[i] = x + delta
                    break
        return out

    def predict(self, sketch: Sketch) -> np.ndarray:
        sketch = sketch.sorted()
        n = len(sketch)
        if n < 2:
            return np.zeros((0, N_INTERVAL_CLASSES))
        y = filled_heights(sketch)
        conf = np.array([o.confidence for o in sketch.observations], dtype=float)
        known = np.array([o.y is not None for o in sketch.observations])
        pair_conf = np.minimum(conf[:-1], conf[1:]) * (known[:-1] & known[1:])
        sig = self.sigma + 2.0 * (1.0 - pair_conf)
        raw = np.round(y)
        dist = _gaussian_rows(np.diff(raw), sig)
        if self.scale_snap and n >= 3:
            snapped = self._snap_to_scale(raw)
            dist = (1.0 - self.snap_weight) * dist + self.snap_weight * _gaussian_rows(np.diff(snapped), sig)
        return normalise_rows(dist)

    def candidates(self, sketch: Sketch, k: int = 5) -> list[list[int]]:
        return topk_sequences(self.predict(sketch), k)


@dataclass
class DeterministicBaseline:
    """Wraps :class:`core.inference.DeterministicReconstructor`."""

    n_candidates: int = 20
    cfg: InferenceConfig = field(default_factory=InferenceConfig)
    name: str = field(default="deterministic", init=False)

    def __post_init__(self) -> None:
        self.engine = DeterministicReconstructor(self.cfg)

    def _safe_sketch(self, sketch: Sketch) -> Sketch:
        """Clip extreme heights: the engine's pitch window is [pitch_lo, pitch_hi]
        around ``center_pitch`` and it fails when a target falls outside it."""
        ys = [o.y for o in sketch.observations if o.y is not None]
        if not ys:
            return sketch
        mean = float(np.mean(ys))
        lo = mean - (self.cfg.center_pitch - self.cfg.pitch_lo) + 2.0
        hi = mean + (self.cfg.pitch_hi - self.cfg.center_pitch) - 2.0
        if min(ys) >= lo and max(ys) <= hi:
            return sketch
        obs = [Observation(o.t, None if o.y is None else float(np.clip(o.y, lo, hi)), o.confidence)
               for o in sketch.observations]
        return Sketch(obs, list(sketch.gestures), list(sketch.constraints), sketch.tempo_hint,
                      dict(sketch.meta))

    def predict(self, sketch: Sketch) -> np.ndarray:
        m = max(len(sketch) - 1, 0)
        try:
            dist = self.engine.interval_distribution(self._safe_sketch(sketch), n=self.n_candidates)
        except ValueError:  # empty candidate set inside the engine
            dist = np.ones((m, N_INTERVAL_CLASSES))
        return normalise_rows(dist)

    def candidates(self, sketch: Sketch, k: int = 5) -> list[list[int]]:
        try:
            riffs = self.engine.reconstruct(self._safe_sketch(sketch), n=k)
        except ValueError:
            riffs = []
        return [r.intervals for r in riffs] or [[0] * (len(sketch) - 1)]
