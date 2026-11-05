"""Model-backed implementation of :class:`core.inference.Reconstructor`.

Wraps a trained :class:`ml.model.SketchTransformer` so the API and UI can use
it exactly like :class:`core.inference.DeterministicReconstructor`:

* candidate interval sequences come from beam search plus a few samples for
  diversity; an optional contour-fit term (``guide_weight``) can be added to
  the beam score but is off by default because it measured worse,
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
from core.representation import Note, Riff, Sketch, class_to_interval, interval_to_class
from core.rhythm import quantise

# runs/mixed/best.pt is trained on POP909 melodies plus Lakh guitar riffs; it beats
# the POP909-only model on both sources (see ml/README.md, "Results: mixed data").
DEFAULT_CHECKPOINT = os.environ.get("RIFF_MODEL", "runs/genre/best.pt")


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
    # Weight of an explicit contour-fit term in the beam search. Default 0: on the
    # POP909 test split any positive weight lowered accuracy (see ml/README.md),
    # the model already reads the contour better than a Gaussian fit does.
    guide_weight: float = 0.0
    guide_sigma: float = 0.8  # semitones of tolerance at confidence 1 (grows as confidence drops)
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
        self.meta = {k: ckpt[k] for k in ("config", "epoch", "sim", "sim_level", "genres_seen") if k in ckpt}
        self.meta.setdefault("sim", self.meta.get("sim_level"))

    @property
    def genres(self) -> list[str]:
        """Genre names the loaded model was trained on ([] for unconditioned checkpoints).

        A genre embedding that never saw data is random noise, so only genres
        recorded in the checkpoint (``genres_seen``) are offered; a checkpoint
        without that record falls back to all names the config allows.
        """
        from .datasets import GENRES

        self._ensure_model()
        if not self.model.cfg.n_genres:
            return []
        seen = self.meta.get("genres_seen")
        return list(seen) if seen else list(GENRES[: self.model.cfg.n_genres])

    @property
    def available(self) -> bool:
        try:
            self._ensure_model()
            return True
        except Exception:  # missing torch, missing checkpoint, bad file
            return False

    # -- generation ---------------------------------------------------------- #
    def _guide_terms(self, sketch: Sketch) -> tuple[np.ndarray, np.ndarray]:
        """Target relative pitch per note (drawn height minus the first) and its sigma."""
        y = filled_heights(sketch)
        conf = np.array([o.confidence for o in sketch.observations], dtype=float)
        missing = np.array([o.y is None for o in sketch.observations])
        sigma = self.guide_sigma + 2.0 * (1.0 - conf)
        sigma[missing] = 1e3  # no information: effectively no guidance
        return y - y[0], sigma

    def _guided_beam(self, sketch: Sketch, beam: int) -> list[tuple[list[int], float, float]]:
        """Beam search over interval classes scored by model log-prob + contour fit.

        Returns ``(intervals, model_logp, total_score)`` per surviving beam.
        """
        import torch
        import torch.nn.functional as F

        from .model import BOS

        target, sigma = self._guide_terms(sketch)
        m = len(sketch) - 1
        batch = self.model._batch_from_sketch(sketch)
        with torch.no_grad():
            memory, mask = self.model.encode(batch)
            # (total, model_logp, intervals, relative pitch so far)
            beams: list[tuple[float, float, list[int], int]] = [(0.0, 0.0, [], 0)]
            for k in range(1, m + 1):
                tokens = torch.tensor([[BOS] + [interval_to_class(d) for d in seq] for _, _, seq, _ in beams],
                                      dtype=torch.long, device=memory.device)
                logits = self.model.decode(tokens, memory.expand(len(beams), -1, -1),
                                           mask.expand(len(beams), -1))[:, -1]
                lp = F.log_softmax(logits, dim=-1).cpu().numpy()
                cand: list[tuple[float, float, list[int], int]] = []
                for (tot, mlp, seq, rel), row in zip(beams, lp):
                    top = np.argsort(row)[::-1][: max(beam, 8)]
                    for c in top:
                        d = class_to_interval(int(c))
                        new_rel = rel + d
                        guide = -self.guide_weight * 0.5 * ((new_rel - target[k]) / sigma[k]) ** 2
                        cand.append((tot + float(row[c]) + guide, mlp + float(row[c]), seq + [d], new_rel))
                cand.sort(key=lambda x: -x[0])
                beams = cand[:beam]
        return [(seq, mlp, tot) for tot, mlp, seq, _ in beams]

    def _interval_candidates(self, sketch: Sketch, n: int) -> list[tuple[list[int], float]]:
        """Candidate interval sequences with their model log-probability, best first.

        Ranking uses the guided score (model + contour); the returned score is
        the model's own log-probability so the UI shows something interpretable.
        """
        import torch

        m = len(sketch) - 1
        if m <= 0:
            return [([], 0.0)]
        target, sigma = self._guide_terms(sketch)
        out: dict[tuple[int, ...], tuple[float, float]] = {}  # seq -> (total, model_logp)
        for seq, mlp, tot in self._guided_beam(sketch, max(2 * n, 8)):
            out[tuple(seq)] = (tot, mlp)
        batch = self.model._batch_from_sketch(sketch)
        with torch.no_grad():
            cls, lp, _ = self.model.generate(batch, self.n_samples, 1.0, mode="sample")
        for j in range(cls.shape[1]):
            seq = tuple(class_to_interval(int(c)) for c in cls[0, j, :m])
            mlp = float(lp[0, j])
            rel = np.concatenate([[0], np.cumsum(seq)])
            tot = mlp - self.guide_weight * 0.5 * float(np.sum(((rel[1:] - target[1:]) / sigma[1:]) ** 2))
            if mlp > -1e8 and (seq not in out or tot > out[seq][0]):
                out[seq] = (tot, mlp)
        ranked = sorted(out.items(), key=lambda kv: -kv[1][0])
        return [(list(seq), mlp) for seq, (_, mlp) in ranked]

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
        if sketch.meta.get("genre") not in self.genres:
            sketch.meta["genre"] = "unknown"  # unknown or untrained label: let the model guess
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
