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
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from core.representation import Sketch, interval_to_class
from ml.baseline import ContourBaseline, DeterministicBaseline, IntervalPredictor
from ml.datasets import Phrase, apply_split_ids, load_phrases, synthetic_phrases
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


def evaluate_predictor(predictor: IntervalPredictor, examples: list[Example], k: int = 5,
                       progress: Callable[[int], None] | None = None) -> dict[str, float]:
    confs: list[float] = []
    corrects: list[float] = []
    nlls: list[float] = []
    hits_k, hits_1, n_seq = 0, 0, 0
    t0 = time.time()
    for i, (sketch, intervals) in enumerate(examples):
        if len(intervals) == 0:
            continue
        dist = np.asarray(predictor.predict(sketch), dtype=float)
        assert dist.shape == (len(intervals), dist.shape[1]), "predictor returned a wrong shape"
        cls = np.array([interval_to_class(d) for d in intervals])
        p_true = dist[np.arange(len(cls)), cls]
        nlls.extend((-np.log(np.maximum(p_true, 1e-12))).tolist())
        am = dist.argmax(axis=1)
        confs.extend(dist.max(axis=1).tolist())
        corrects.extend((am == cls).astype(float).tolist())
        cands = predictor.candidates(sketch, k)
        truth = list(intervals)
        hits_k += int(any(list(c) == truth for c in cands[:k]))
        hits_1 += int(bool(cands) and list(cands[0]) == truth)
        n_seq += 1
        if progress:
            progress(i)
    conf_a, corr_a = np.array(confs), np.array(corrects)
    return {
        "interval_acc": float(corr_a.mean()) if len(corr_a) else 0.0,
        "nll": float(np.mean(nlls)) if nlls else 0.0,
        "ece": expected_calibration_error(conf_a, corr_a),
        f"top{k}_recon": hits_k / max(n_seq, 1),
        "seq_acc": hits_1 / max(n_seq, 1),
        "n": float(n_seq),
        "seconds": time.time() - t0,
    }


def format_table(results: dict[str, dict[str, float]]) -> str:
    keys = [k for k in next(iter(results.values())) if k not in ("n",)]
    head = f"{'predictor':<16}" + "".join(f"{k:>12}" for k in keys)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lines.append(f"{name:<16}" + "".join(f"{r[k]:>12.3f}" for k in keys))
    n = int(next(iter(results.values()))["n"])
    lines.append(f"(n = {n} sketches; higher is better except nll and ece)")
    return "\n".join(lines)


def load_examples_from_args(args: argparse.Namespace) -> list[Example]:
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
    sim = SketchSimulator(args.seed, args.sim)
    return make_examples(phrases, sim, seed=args.seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--data", type=str, default="data/processed/phrases.jsonl")
    ap.add_argument("--splits", type=str, default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--synthetic", type=int, default=0, help="evaluate on N synthetic phrases")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sim", default="medium", help="noise preset: easy/medium/hard")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-deterministic", action="store_true", help="skip the (slower) deterministic engine")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    examples = load_examples_from_args(args)
    print(f"{len(examples)} sketches, sim={args.sim}, k={args.k}")
    predictors: list[IntervalPredictor] = [ContourBaseline()]
    if not args.no_deterministic:
        predictors.append(DeterministicBaseline())
    if args.checkpoint:
        from ml.model import count_parameters, load_checkpoint
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, ckpt = load_checkpoint(args.checkpoint, device)
        model.name = "transformer"
        print(f"loaded {args.checkpoint}: {count_parameters(model) / 1e6:.2f}M params, "
              f"epoch {ckpt.get('epoch')}, val_acc {ckpt.get('val_acc', float('nan')):.3f}")
        predictors.append(model)

    results: dict[str, dict[str, float]] = {}
    for pred in predictors:
        results[pred.name] = evaluate_predictor(pred, examples, k=args.k)
        print(f"  {pred.name}: done in {results[pred.name]['seconds']:.1f}s")
    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
