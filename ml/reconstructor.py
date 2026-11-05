"""Model-backed implementation of :class:`core.inference.Reconstructor`.

Wraps a trained :class:`ml.model.SketchTransformer` so the API and UI can use
it exactly like :class:`core.inference.DeterministicReconstructor`:

* candidate interval sequences come from beam search plus a few samples,
* they are anchored to absolute pitches (fixed-pitch facts win, otherwise the
  riff is centred on the guitar register),
* candidates that break the user's hard constraints are dropped, and if too
  few survive the deterministic engine fills the gap, so the user always gets
  something that respects what they asserted,
* each riff gets a best-fit key label and the articulation flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.constraints import apply_articulation, build_constraints, violations
from core.inference import (SCALES, DeterministicReconstructor, InferenceConfig, filled_heights,
                            key_name)
from core.representation import Note, Riff, Sketch, class_to_interval
from core.rhythm import quantise

DEFAULT_CHECKPOINT = os.environ.get("RIFF_MODEL", "runs/pop909/best.pt")


def guess_key(pitches: list[int]) -> str:
    """Best-fitting key label: most in-scale notes, ties broken by tonic hits."""
    if not pitches:
        return "unknown"
    pcs = [p % 12 for p in pitches]
    best, best_score = "C major", -1.0
    for scale, degrees in SCALES.items():
        if scale == "chromatic":
            continue
        for root in range(12):
            in_scale = sum((pc - root) % 12 in degrees for pc in pcs)
            tonic = (pcs[0] == root) * 0.5 + (pcs[-1] == root) * 0.7
            score = in_scale + tonic - 0.05 * len(degrees)  # prefer smaller scales on ties
            if score > best_score:
                best, best_score = key_name(root, scale), score
    return best


@dataclass
class TransformerReconstructor:
    checkpoint: str = DEFAULT_CHECKPOINT
    model: Any = None  # a SketchTransformer; loaded lazily from ``checkpoint`` if None
    device: str | None = None
    center_pitch: int = 52
    n_samples: int = 6
    fallback: DeterministicReconstructor = field(default_factory=lambda: DeterministicReconstructor(InferenceConfig()))
    meta: dict[str, Any] = field(default_factory=dict)

    # -- loading ------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        import torch

        from .model import load_checkpoint

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_checkpoint(self.checkpoint, dev)
        self.device = dev
        self.meta = {k: ckpt[k] for k in ("config", "epoch", "sim") if k in ckpt}

    @property
    def available(self) -> bool:
        try:
            self._ensure_model()
            return True
        except Exception:  # missing torch, missing checkpoint, bad file
            return False

    # -- generation ---------------------------------------------------------- #
    def _interval_candidates(self, sketch: Sketch, n: int) -> list[tuple[list[int], float]]:
        import torch

        m = len(sketch) - 1
        if m <= 0:
            return [([], 0.0)]
        batch = self.model._batch_from_sketch(sketch)
        out: dict[tuple[int, ...], float] = {}
        with torch.no_grad():
            for mode, k in (("beam", max(2 * n, 8)), ("sample", self.n_samples)):
                cls, lp, _ = self.model.generate(batch, k, 1.0, mode=mode)
                for j in range(cls.shape[1]):
                    seq = tuple(class_to_interval(int(c)) for c in cls[0, j, :m])
                    score = float(lp[0, j])
                    if score > -1e8 and (seq not in out or score > out[seq]):
                        out[seq] = score
        return sorted(((list(s), sc) for s, sc in out.items()), key=lambda x: -x[1])

    def _anchor(self, intervals: list[int], sketch: Sketch, fixed: dict[int, int]) -> list[int]:
        rel = np.concatenate([[0], np.cumsum(intervals)]).astype(int)
        if fixed:
            shift = int(round(np.mean([p - rel[i] for i, p in fixed.items() if i < len(rel)])))
        else:
            shift = int(round(self.center_pitch - rel.mean()))
        return [int(np.clip(p + shift, 24, 100)) for p in rel]

    def reconstruct(self, sketch: Sketch, n: int = 5) -> list[Riff]:
        sketch = sketch.sorted()
        if len(sketch) == 0:
            return []
        self._ensure_model()
        cs = build_constraints(sketch)
        rhythm = quantise(sketch.onsets, tempo_hint=sketch.tempo_hint)

        riffs: list[Riff] = []
        seen: set[tuple[int, ...]] = set()
        for intervals, score in self._interval_candidates(sketch, n):
            pitches = self._anchor(intervals, sketch, cs.fixed_pitch)
            notes = [Note(p, o, d) for p, o, d in zip(pitches, rhythm.onsets_beats, rhythm.durations_beats)]
            riff = Riff(notes, tempo=rhythm.tempo, key=guess_key(pitches), score=score,
                        meta={"method": "transformer", "grid": rhythm.grid})
            if violations(riff, cs) or tuple(pitches) in seen:
                continue
            seen.add(tuple(pitches))
            riffs.append(apply_articulation(riff, cs))
            if len(riffs) >= n:
                break

        if len(riffs) < n:  # the model could not satisfy the facts: let the search engine fill in
            for r in self.fallback.reconstruct(sketch, n):
                if tuple(r.pitches) in seen:
                    continue
                r.meta["method"] = "deterministic (fallback)"
                r.score = min(r.score, -50.0)  # rank below model candidates
                seen.add(tuple(r.pitches))
                riffs.append(r)
                if len(riffs) >= n:
                    break
        return riffs


def load_default(checkpoint: str = DEFAULT_CHECKPOINT) -> TransformerReconstructor | None:
    """Return a ready reconstructor, or None when torch or the checkpoint is missing."""
    rec = TransformerReconstructor(checkpoint=checkpoint)
    return rec if os.path.exists(checkpoint) and rec.available else None
