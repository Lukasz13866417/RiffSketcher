"""Metrics and a CLI comparing the baselines with a trained SketchTransformer.

Metrics (over a list of ``(sketch, true_intervals)`` examples)
    interval_acc   argmax of the per-step distribution equals the true interval
    nll            mean per-note negative log-likelihood of the true interval
    ece            expected calibration error of the argmax confidence (10 bins)
    topk_recon     fraction of sketches whose exact interval sequence is among
                   the k candidate sequences (beam / top-k), k=5 by default
    seq_acc        exact-sequence accuracy of the best candidate (k=1)

CLI
    uv run python -m ml.evaluate --checkpoint runs/first/best.pt --data data/processed/phrases.jsonl --split test
    uv run python -m ml.evaluate --checkpoint runs/smoke/best.pt --synthetic 200
    uv run python -m ml.evaluate --checkpoint runs/pop909/best.pt runs/mixed/best.pt --data data/processed/mixed.jsonl

With several checkpoints the models are named transformer1, transformer2, ...
in the order given.  When the data mixes sources (``Phrase.source``) a table
per source follows the overall one.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from core.representation import Sketch, interval_to_class
from ml.baseline import ContourBaseline, DeterministicBaseline, IntervalPredictor
from ml.datasets import Phrase, apply_split_ids, genre_of_source, load_phrases, synthetic_phrases
from ml.sketch_simulator import SketchSimulator

Example = tuple[Sketch, list[int]]


def make_examples(phrases: list[Phrase], simulator: SketchSimulator, seed: int = 0) -> list[Example]:
    out: list[Example] = []
    for i, p in enumerate(phrases):
        r = simulator.simulate(p, rng=np.random.default_rng([seed, i]))
        out.append((r.sketch, r.intervals))
    return out


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(ece)


@dataclass
class ExampleScore:
    """What one sketch contributes to the metrics (kept per example so the
    same predictions can be aggregated per data source)."""

    conf: np.ndarray  # argmax confidence per note
    correct: np.ndarray  # argmax == true interval, per note
    nll: np.ndarray  # -log p(true interval), per note
    hit_k: bool  # exact sequence among the k candidates
    hit_1: bool  # exact sequence is the best candidate


def score_examples(predictor: IntervalPredictor, examples: list[Example], k: int = 5,
                   progress: Callable[[int], None] | None = None) -> list[ExampleScore]:
    out: list[ExampleScore] = []
    for i, (sketch, intervals) in enumerate(examples):
        if len(intervals) == 0:
            continue
        dist = np.asarray(predictor.predict(sketch), dtype=float)
        assert dist.shape == (len(intervals), dist.shape[1]), "predictor returned a wrong shape"
        cls = np.array([interval_to_class(d) for d in intervals])
        p_true = dist[np.arange(len(cls)), cls]
        cands = predictor.candidates(sketch, k)
        truth = list(intervals)
        out.append(ExampleScore(dist.max(axis=1), (dist.argmax(axis=1) == cls).astype(float),
                                -np.log(np.maximum(p_true, 1e-12)),
                                any(list(c) == truth for c in cands[:k]),
                                bool(cands) and list(cands[0]) == truth))
        if progress:
            progress(i)
    return out


def aggregate(scores: list[ExampleScore], k: int = 5) -> dict[str, float]:
    conf = np.concatenate([s.conf for s in scores]) if scores else np.zeros(0)
    correct = np.concatenate([s.correct for s in scores]) if scores else np.zeros(0)
    nll = np.concatenate([s.nll for s in scores]) if scores else np.zeros(0)
    n = max(len(scores), 1)
    return {
        "interval_acc": float(correct.mean()) if len(correct) else 0.0,
        "nll": float(nll.mean()) if len(nll) else 0.0,
        "ece": expected_calibration_error(conf, correct),
        f"top{k}_recon": sum(s.hit_k for s in scores) / n,
        "seq_acc": sum(s.hit_1 for s in scores) / n,
        "n": float(len(scores)),
    }


def evaluate_predictor(predictor: IntervalPredictor, examples: list[Example], k: int = 5,
                       progress: Callable[[int], None] | None = None) -> dict[str, float]:
    t0 = time.time()
    metrics = aggregate(score_examples(predictor, examples, k, progress), k)
    metrics["seconds"] = time.time() - t0
    return metrics


def format_table(results: dict[str, dict[str, float]]) -> str:
    keys = [k for k in next(iter(results.values())) if k not in ("n",)]
    head = f"{'predictor':<16}" + "".join(f"{k:>12}" for k in keys)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lines.append(f"{name:<16}" + "".join(f"{r[k]:>12.3f}" for k in keys))
    n = int(next(iter(results.values()))["n"])
    lines.append(f"(n = {n} sketches; higher is better except nll and ece)")
    return "\n".join(lines)


def load_examples_from_args(args: argparse.Namespace) -> tuple[list[Example], list[str]]:
    """Simulated sketches for the chosen split plus each one's phrase source."""
    if args.synthetic:
        phrases = synthetic_phrases(args.synthetic, seed=args.seed + 1000)
    else:
        phrases = load_phrases(args.data)
        splits_path = args.splits or Path(args.data).with_name("splits.json")
        if Path(splits_path).exists():
            ids = json.loads(Path(splits_path).read_text())
            phrases = apply_split_ids(phrases, ids)[args.split]
        if args.limit:
            phrases = phrases[:args.limit]
    if getattr(args, "source", None):
        phrases = [p for p in phrases if p.source == args.source]
    sim = SketchSimulator(args.seed, args.sim)
    examples = make_examples(phrases, sim, seed=args.seed)
    # genre label shown to conditioned models: the phrase's own (via its source), none, or a fixed one
    mode = getattr(args, "genre", "auto")
    for (sketch, _), p in zip(examples, phrases):
        sketch.meta["genre"] = genre_of_source(p.source) if mode == "auto" else mode
    return examples, [p.source for p in phrases]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, nargs="*", default=None,
                    help="one or more best.pt files to compare")
    ap.add_argument("--data", type=str, default="data/processed/phrases.jsonl")
    ap.add_argument("--splits", type=str, default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--synthetic", type=int, default=0, help="evaluate on N synthetic phrases")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sim", default="medium", help="noise preset: easy/medium/hard")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-deterministic", action="store_true", help="skip the (slower) deterministic engine")
    ap.add_argument("--source", default=None, help="only phrases from this source (e.g. pop909, lakh_guitar)")
    ap.add_argument("--genre", default="auto",
                    help="genre label given to conditioned models: auto (from the source), unknown, or a name")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    examples, sources = load_examples_from_args(args)
    print(f"{len(examples)} sketches, sim={args.sim}, k={args.k}")
    predictors: list[IntervalPredictor] = [ContourBaseline()]
    if not args.no_deterministic:
        predictors.append(DeterministicBaseline())
    for i, path in enumerate(args.checkpoint or []):
        from ml.model import count_parameters, load_checkpoint
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, ckpt = load_checkpoint(path, device)
        model.name = Path(path).parent.name if len(args.checkpoint or []) > 1 else "transformer"
        print(f"loaded {path} as {model.name}: {count_parameters(model) / 1e6:.2f}M params, "
              f"epoch {ckpt.get('epoch')}, val_acc {ckpt.get('val_acc', float('nan')):.3f}")
        predictors.append(model)

    scores: dict[str, list[ExampleScore]] = {}
    for pred in predictors:
        t0 = time.time()
        scores[pred.name] = score_examples(pred, examples, k=args.k)
        print(f"  {pred.name}: done in {time.time() - t0:.1f}s")
    print()
    print(format_table({name: aggregate(sc, args.k) for name, sc in scores.items()}))
    for source in sorted(set(sources)) if len(set(sources)) > 1 else []:
        sel = [i for i, s in enumerate(sources) if s == source]
        print(f"\nsource = {source}")
        print(format_table({name: aggregate([sc[i] for i in sel], args.k) for name, sc in scores.items()}))


if __name__ == "__main__":
    main()
