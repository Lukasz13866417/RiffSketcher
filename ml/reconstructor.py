"""Model-backed implementation of :class:`core.inference.Reconstructor`.

Wraps a trained :class:`ml.model.SketchTransformer` so the API and UI can use
it exactly like :class:`core.inference.DeterministicReconstructor`:

* candidate interval sequences come from a *constrained* beam search plus
  masked samples for diversity: at every step the interval classes that would
  break a user fact (direction, exact interval, "same as note i", or a pinned
  pitch once the sequence is anchored) are masked out and the model's
  probabilities renormalise over the rest, so facts cannot be violated;
  an optional contour-fit term (``guide_weight``) is off by default,
* pins are enforced relative to the earliest pinned note (no drawn heights
  involved), and the riff is shifted onto that pin afterwards; without pins it
  is centred on the guitar register,
* only when the facts contradict each other does a candidate survive with
  violations; those are dropped and the deterministic engine fills the gap,
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
DEFAULT_CHECKPOINT = os.environ.get("RIFF_MODEL", "runs/genre_facts/best.pt")
# A second checkpoint used only for timing: its rhythm head beats the rule-based quantiser by ten
# points, but sharing an encoder with the rhythm loss costs two points of interval accuracy, so
# pitches come from DEFAULT_CHECKPOINT and gaps from this one. Unset or missing -> quantiser.
DEFAULT_RHYTHM_CHECKPOINT = os.environ.get("RIFF_RHYTHM_MODEL", "runs/genre_facts_rh/best.pt")


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
    rhythm_checkpoint: str | None = DEFAULT_RHYTHM_CHECKPOINT  # timing model; None = use ``model`` or the quantiser
    rhythm_model: Any = None
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
        if self.rhythm_model is None and self.rhythm_checkpoint and os.path.exists(self.rhythm_checkpoint):
            rm, _ = load_checkpoint(self.rhythm_checkpoint, dev)
            self.rhythm_model = rm if rm.has_rhythm else None
        self.meta["timing"] = ("model" if (self.rhythm_model is not None or self.model.has_rhythm) else "quantiser")

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

    # -- constrained decoding ----------------------------------------------- #

    @staticmethod
    def _propagate(cs) -> None:
        """Derive the facts the user's facts imply, in place.

        "Same" pairs form equivalence classes; a pin on any member pins them
        all, and a pair whose both members are pinned is already satisfied.
        """
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in cs.same:
            parent[find(i)] = find(j)
        classes: dict[int, list[int]] = {}
        for x in list(parent):
            classes.setdefault(find(x), []).append(x)
        for members in classes.values():
            pinned = [cs.fixed_pitch[m] for m in members if m in cs.fixed_pitch]
            if pinned:
                for m in members:
                    cs.fixed_pitch.setdefault(m, pinned[0])

    @staticmethod
    def _anchor_pin(cs) -> tuple[int, int] | None:
        """(index, pitch) of the earliest pinned note, or None.

        Pins are enforced *relative to this note* during decoding, which needs
        no drawn heights: a pin on note j > i means rel[j] - rel[i] == pin_j -
        pin_i exactly. (An earlier version derived note 0's absolute pitch from
        the drawn offset to the first pin; drawn heights are noisy and
        compressed, so that forced wrong intervals and lowered accuracy.)
        """
        if not cs.fixed_pitch:
            return None
        i = min(cs.fixed_pitch)
        return i, cs.fixed_pitch[i]

    @staticmethod
    def _allowed(cs, k: int, rel: list[int], anchor: tuple[int, int] | None) -> np.ndarray | None:
        """Boolean mask over interval classes for the step into note ``k``, given the
        relative pitches ``rel`` of notes 0..k-1 (``rel[0] == 0``) and the anchor pin
        ``(i0, pitch)`` (or None). ``None`` means every class is allowed."""
        from core.representation import INTERVAL_MIN, N_INTERVAL_CLASSES

        exact: int | None = None
        sign: int | None = cs.direction.get(k - 1)
        if (k - 1) in cs.interval:
            exact = cs.interval[k - 1]
        for i, j in cs.same:
            a, b = sorted((i, j))
            if b == k and a < k:
                exact = rel[a] - rel[k - 1] if exact is None else exact
        if anchor is not None and k in cs.fixed_pitch and k > anchor[0]:
            need = (cs.fixed_pitch[k] - anchor[1]) + rel[anchor[0]] - rel[k - 1]
            exact = need if exact is None else exact
        # reachability: a later pin or "same" target must stay within 24 semitones per remaining step,
        # otherwise the beam can wander somewhere the fact becomes unsatisfiable
        from core.representation import INTERVAL_MAX
        reach: list[tuple[int, int]] = []  # (target relative pitch, steps left after this one)
        if anchor is not None and k > anchor[0]:  # the anchor note's relative pitch is known only after it
            for j, pj in cs.fixed_pitch.items():
                if j > k:
                    reach.append(((pj - anchor[1]) + rel[anchor[0]], j - k))
        for i, j in cs.same:
            a, b = sorted((i, j))
            if b > k and a < k:
                reach.append((rel[a], b - k))
        if exact is None and sign is None and not reach:
            return None
        classes = np.arange(N_INTERVAL_CLASSES) + INTERVAL_MIN
        ok = np.ones(N_INTERVAL_CLASSES, dtype=bool)
        if exact is not None:
            ok &= classes == exact
        if sign is not None:
            ok &= np.sign(classes) == sign
        for target, steps in reach:
            ok &= np.abs(target - (rel[k - 1] + classes)) <= INTERVAL_MAX * steps
        return ok

    def _predict_gaps(self, batch, memory, m: int) -> list[int]:
        """Gap classes from the timing model if there is one, else from this
        model's own rhythm head, else [] (the quantiser is used)."""
        import torch

        if self.rhythm_model is not None:
            with torch.no_grad():
                rmem, _ = self.rhythm_model.encode(batch)
                rcls = self.rhythm_model._rhythm_classes(rmem, 1, m)
        else:
            rcls = self.model._rhythm_classes(memory, 1, m)
        return rcls[0, 0].tolist() if rcls is not None else []

    def _decode(self, sketch: Sketch, cs, anchor: tuple[int, int] | None, beam: int, n_samples: int,
                rng: np.random.Generator) -> list[tuple[list[int], float, float, list[int]]]:
        """Constrained beam search plus masked ancestral samples.

        Returns ``(intervals, model_logp, total_score, rhythm_classes)`` per
        candidate, best first. Model log-probs are of the *masked, renormalised*
        distributions, i.e. the probability under the facts.
        """
        import torch

        from .model import BOS

        target, sigma = self._guide_terms(sketch)
        m = len(sketch) - 1
        batch = self.model._batch_from_sketch(sketch)
        out: list[tuple[list[int], float, float, list[int]]] = []
        with torch.no_grad():
            memory, mask = self.model.encode(batch)
            rs = self._predict_gaps(batch, memory, m)

            def step_rows(seqs: list[list[int]]) -> np.ndarray:
                tokens = torch.tensor([[BOS] + [interval_to_class(d) for d in seq] for seq in seqs],
                                      dtype=torch.long, device=memory.device)
                lp, _ = self.model.step_logprobs(tokens, memory.expand(len(seqs), -1, -1),
                                                 mask.expand(len(seqs), -1))
                return lp.cpu().numpy()

            from core.representation import INTERVAL_MIN

            def masked(row: np.ndarray, k: int, rel: list[int]) -> np.ndarray:
                ok = self._allowed(cs, k, rel, anchor)
                if ok is None:
                    ok = np.ones(len(row), dtype=bool)
                if k < m:  # one-step lookahead: drop choices that leave the next step unsatisfiable
                    look = ok.copy()
                    for c in np.flatnonzero(ok):
                        nxt = self._allowed(cs, k + 1, rel + [rel[-1] + c + INTERVAL_MIN], anchor)
                        if nxt is not None and not nxt.any():
                            look[c] = False
                    if look.any():
                        ok = look
                if ok.all():
                    return row
                if not ok.any():  # contradictory facts: leave the step unmasked
                    return row
                r = np.where(ok, row, -np.inf)
                return r - np.log(np.exp(r[ok]).sum())  # renormalise over the allowed classes

            # beam: (total, model_logp, intervals, relative pitches so far)
            beams: list[tuple[float, float, list[int], list[int]]] = [(0.0, 0.0, [], [0])]
            for k in range(1, m + 1):
                rows = step_rows([b[2] for b in beams])
                cand: list[tuple[float, float, list[int], list[int]]] = []
                for (tot, mlp, seq, rel), row in zip(beams, rows):
                    row = masked(row, k, rel)
                    top = [c for c in np.argsort(row)[::-1][: max(beam, 8)] if np.isfinite(row[c])]
                    for c in top:
                        d = class_to_interval(int(c))
                        new_rel = rel[-1] + d
                        guide = -self.guide_weight * 0.5 * ((new_rel - target[k]) / sigma[k]) ** 2
                        cand.append((tot + float(row[c]) + guide, mlp + float(row[c]), seq + [d], rel + [new_rel]))
                cand.sort(key=lambda x: -x[0])
                beams = cand[:beam]
            out.extend((seq, mlp, tot, rs) for tot, mlp, seq, _ in beams)

            # masked ancestral samples for diversity
            samples: list[tuple[float, list[int], list[int]]] = [(0.0, [], [0]) for _ in range(n_samples)]
            for k in range(1, m + 1):
                rows = step_rows([s[1] for s in samples])
                nxt = []
                for (mlp, seq, rel), row in zip(samples, rows):
                    row = masked(row, k, rel)
                    p = np.exp(row - row.max()); p[~np.isfinite(row)] = 0.0; p /= p.sum()
                    c = int(rng.choice(len(p), p=p))
                    d = class_to_interval(c)
                    nxt.append((mlp + float(row[c]), seq + [d], rel + [rel[-1] + d]))
                samples = nxt
            for mlp, seq, rel in samples:
                tot = mlp - self.guide_weight * 0.5 * float(np.sum(((np.array(rel[1:]) - target[1:]) / sigma[1:]) ** 2))
                out.append((seq, mlp, tot, rs))
        return out

    def _interval_candidates(self, sketch: Sketch, n: int, cs=None,
                             anchor: tuple[int, int] | None = None) -> list[tuple[list[int], float]]:
        """Candidate interval sequences with their model log-probability, best first.

        Ranking uses the total score (model + optional contour term); the
        returned score is the model's log-probability under the facts.
        """
        m = len(sketch) - 1
        if m <= 0:
            return [([], 0.0)]
        if cs is None:
            cs = build_constraints(sketch)
        rng = np.random.default_rng(0)
        out: dict[tuple[int, ...], tuple[float, float, list[int]]] = {}  # seq -> (total, model_logp, rhythm)
        for seq, mlp, tot, rs in self._decode(sketch, cs, anchor, max(2 * n, 8), self.n_samples, rng):
            key = tuple(seq)
            if key not in out or tot > out[key][0]:
                out[key] = (tot, mlp, rs)
        ranked = sorted(out.items(), key=lambda kv: -kv[1][0])
        self._last_rhythms = {seq: rs for seq, (_, _, rs) in ranked}
        return [(list(seq), mlp) for seq, (_, mlp, _) in ranked]

    def _anchor(self, intervals: list[int], sketch: Sketch, fixed: dict[int, int],
                anchor: tuple[int, int] | None = None) -> list[int]:
        rel = np.concatenate([[0], np.cumsum(intervals)]).astype(int)
        if anchor is not None and anchor[0] < len(rel):
            shift = anchor[1] - int(rel[anchor[0]])  # exact: later pins were masked relative to this note
        elif fixed:
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
        self._propagate(cs)
        anchor = self._anchor_pin(cs)
        rhythm = quantise(sketch.onsets, tempo_hint=sketch.tempo_hint)

        riffs: list[Riff] = []
        seen: set[tuple[int, ...]] = set()
        for intervals, score in self._interval_candidates(sketch, n, cs, anchor):
            pitches = self._anchor(intervals, sketch, cs.fixed_pitch, anchor)
            rs = getattr(self, "_last_rhythms", {}).get(tuple(intervals), [])
            timing_model = self.rhythm_model if self.rhythm_model is not None else self.model
            if timing_model.has_rhythm and len(rs) == len(intervals) and intervals:
                onsets, durations, tempo = timing_model.rhythm_from_classes(rs, sketch.onsets)
                meta = {"method": "transformer", "rhythm": "model"}
            else:
                onsets, durations, tempo = rhythm.onsets_beats, rhythm.durations_beats, rhythm.tempo
                meta = {"method": "transformer", "rhythm": "quantised", "grid": rhythm.grid}
            notes = [Note(p, o, d) for p, o, d in zip(pitches, onsets, durations)]
            riff = Riff(notes, tempo=tempo, key=guess_key(pitches), score=score, meta=meta)
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
