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

from core.representation import Sketch, interval_to_class, class_to_interval
from ml.baseline import ContourBaseline, DeterministicBaseline, IntervalPredictor
from ml.datasets import Phrase, apply_split_ids, load_phrases, phrase_genre, synthetic_phrases
from ml.sketch_simulator import SketchSimulator

Example = tuple[Sketch, list[int]]


def make_examples(phrases: list[Phrase], simulator: SketchSimulator, seed: int = 0) -> list[Example]:
    return [ex for ex, *_ in make_examples_with_rhythm(phrases, simulator, seed)]


def make_examples_with_rhythm(phrases: list[Phrase], simulator: SketchSimulator,
                              seed: int = 0) -> list[tuple[Example, list[float], list[float]]]:
    """Examples plus the true gaps between consecutive notes and the true note lengths, in beats."""
    out: list[tuple[Example, list[float], list[float]]] = []
    for i, p in enumerate(phrases):
        r = simulator.simulate(p, rng=np.random.default_rng([seed, i]))
        iois = [b - a for a, b in zip(r.onsets_beats, r.onsets_beats[1:])]
        out.append(((r.sketch, r.intervals), iois, list(r.durations_beats)))
    return out


def duration_accuracy(examples_with_rhythm, model=None) -> dict[str, float]:
    """Fraction of notes whose predicted duration class (share of the gap) is right.

    ``legato`` is the old rule: every note lasts until the next onset ("full").
    ``hold`` reads the class straight off the tap hold time where recorded (and
    "full" elsewhere); ``model`` is the duration head of a checkpoint that has one.
    """
    from ml.datasets import DURATION_CLASSES, duration_class, duration_ratios
    from ml.datasets import sketch_to_features
    full = [i for i, (name, _) in enumerate(DURATION_CLASSES) if name == "full"][0]
    legato_hit, hold_hit, m_hit, n = 0, 0, 0, 0
    for (sketch, _), iois, durs in examples_with_rhythm:
        if not iois or len(durs) != len(iois) + 1:
            continue
        truth = [duration_class(r) for r in duration_ratios(durs, iois)]
        legato_hit += sum(int(c == full) for c in truth)
        f = sketch_to_features(sketch)["cont"]
        hold_guess = [duration_class(float(row[14])) if row[15] > 0.5 else full for row in f]
        hold_hit += sum(int(a == b) for a, b in zip(hold_guess, truth))
        if model is not None and getattr(model, "has_duration", False):
            model.generate(model._batch_from_sketch(sketch), 1, mode="greedy")
            pred = model._last_duration[0, :len(truth)].tolist()
            m_hit += sum(int(a == b) for a, b in zip(pred, truth))
        n += len(truth)
    out = {"legato": legato_hit / max(n, 1), "hold": hold_hit / max(n, 1)}
    if model is not None and getattr(model, "has_duration", False):
        out["model"] = m_hit / max(n, 1)
    return out


def expansion_accuracy(phrases: list[Phrase], model, seed: int = 0, limit: int | None = 500) -> dict[str, float]:
    """Free decoding of an expansion model on *reduced* sketches (tremolo taps, dropped
    notes, no facts): how often the output has the right length, the right interval
    sequence, and the right number of notes per tremolo tap."""
    from dataclasses import replace
    from ml.sketch_simulator import get_config
    cfg = replace(get_config("medium"), expand=True, double_prob=0.0, facts_prob=0.0)
    sim = SketchSimulator(seed, cfg)
    n = len_ok = seq_ok = 0
    trem = trem_ok = trem_1 = 0
    per_tap_err = 0.0
    taps_total = 0
    for i, p in enumerate(phrases[:limit] if limit else phrases):
        r = sim.simulate(p, rng=np.random.default_rng([seed, 7, i]))
        if not r.align or len(r.out_pitches) < 2:
            continue
        model.generate(model._batch_from_sketch(r.sketch), 1, mode="greedy", force_advance=False)
        al = [a for a in model._last_align[0, 0].tolist() if a >= 0]
        m = len(al) - 1
        pred = [class_to_interval(int(c)) for c in model._last_classes[0, 0, :m].tolist()] if hasattr(model, "_last_classes") else None
        true_iv = [b - a for a, b in zip(r.out_pitches, r.out_pitches[1:])]
        n += 1
        if len(al) == len(r.out_pitches):
            len_ok += 1
            if pred is not None and pred == true_iv:
                seq_ok += 1
        true_counts = {}
        for a in r.align:
            true_counts[a] = true_counts.get(a, 0) + 1
        pred_counts = {}
        for a in al:
            pred_counts[a] = pred_counts.get(a, 0) + 1
        for k in true_counts:
            taps_total += 1
            per_tap_err += abs(true_counts[k] - pred_counts.get(k, 0))
        for g in r.sketch.gestures:
            if g.kind == "tremolo":
                trem += 1
                diff = abs(true_counts.get(g.start, 0) - pred_counts.get(g.start, 0))
                trem_ok += int(diff == 0)
                trem_1 += int(diff <= 1)
    return {"n": n, "length": len_ok / max(n, 1), "sequence": seq_ok / max(n, 1),
            "tremolo_count": trem_ok / max(trem, 1), "tremolo_within_1": trem_1 / max(trem, 1), "tremolo_taps": trem,
            "notes_per_tap_error": per_tap_err / max(taps_total, 1)}


def rhythm_accuracy(examples_with_rhythm: list[tuple[Example, list[float]]], model=None) -> dict[str, float]:
    """Fraction of gaps whose predicted rhythm class matches the true one.

    ``quantise`` is the rule-based grid snap (``core.rhythm.quantise``); ``model``
    is the greedy gap prediction of a checkpoint with a rhythm head.
    """
    from core.rhythm import quantise
    from ml.datasets import rhythm_class
    q_hit, m_hit, n = 0, 0, 0
    for (sketch, _), iois, _durs in examples_with_rhythm:
        if not iois:
            continue
        truth = [rhythm_class(b) for b in iois]
        q = quantise(sketch.onsets, tempo_hint=sketch.tempo_hint)
        q_gaps = [rhythm_class(b - a) for a, b in zip(q.onsets_beats, q.onsets_beats[1:])]
        q_hit += sum(int(a == b) for a, b in zip(q_gaps, truth))
        if model is not None and getattr(model, "has_rhythm", False):
            model.generate(model._batch_from_sketch(sketch), 1, mode="greedy")
            pred = model._last_rhythm[0, 0, :len(truth)].tolist()
            m_hit += sum(int(a == b) for a, b in zip(pred, truth))
        n += len(truth)
    out = {"quantise": q_hit / max(n, 1)}
    if model is not None and getattr(model, "has_rhythm", False):
        out["model"] = m_hit / max(n, 1)
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
    from dataclasses import replace
    from ml.sketch_simulator import get_config
    cfg = get_config(args.sim)
    if getattr(args, "no_facts", False):
        cfg = replace(cfg, facts_prob=0.0)
    sim = SketchSimulator(args.seed, cfg)
    args._phrases = phrases
    with_rhythm = make_examples_with_rhythm(phrases, sim, seed=args.seed)
    examples = [ex for ex, *_ in with_rhythm]
    args._with_rhythm = with_rhythm
    # genre label shown to conditioned models: the phrase's own (via its source), none, or a fixed one
    mode = getattr(args, "genre", "auto")
    for (sketch, _), p in zip(examples, phrases):
        sketch.meta["genre"] = phrase_genre(p) if mode == "auto" else mode
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
    ap.add_argument("--no-rhythm", action="store_true", help="skip the rhythm accuracy comparison")
    ap.add_argument("--no-facts", action="store_true", help="simulate sketches without user facts")
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
    # rhythm: rule-based grid snap vs any checkpoint with a rhythm head
    rh_models = [p for p in predictors if getattr(p, "has_rhythm", False)]
    if not args.no_rhythm:
        print("\nrhythm (gap class) accuracy:")
        base = rhythm_accuracy(args._with_rhythm, None)
        print(f"  quantise (rule-based)   {base['quantise']:.3f}")
        for p in rh_models:
            print(f"  {p.name:22s}  {rhythm_accuracy(args._with_rhythm, p)['model']:.3f}")
        for p in predictors:
            if getattr(getattr(p, "cfg", None), "expand", False):
                e = expansion_accuracy(args._phrases, p, seed=args.seed, limit=min(500, args.limit or 500))
                print(f"\nexpansion ({p.name}; {e['n']} reduced sketches, free decoding, no facts):")
                print(f"  right length {e['length']:.3f}   right interval sequence {e['sequence']:.3f}")
                print(f"  tremolo taps ({e['tremolo_taps']}): right note count {e['tremolo_count']:.3f}, within one {e['tremolo_within_1']:.3f}")
                print(f"  mean |notes per tap - true| {e['notes_per_tap_error']:.3f}")
        print("\nduration (length class) accuracy:")
        base = duration_accuracy(args._with_rhythm, None)
        print(f"  legato (every note full)   {base['legato']:.3f}")
        print(f"  tap hold time alone        {base['hold']:.3f}")
        for p in predictors:
            if getattr(p, "has_duration", False):
                print(f"  {p.name:22s}  {duration_accuracy(args._with_rhythm, p)['model']:.3f}")
    for source in sorted(set(sources)) if len(set(sources)) > 1 else []:
        sel = [i for i, s in enumerate(sources) if s == source]
        print(f"\nsource = {source}")
        print(format_table({name: aggregate([sc[i] for i in sel], args.k) for name, sc in scores.items()}))


if __name__ == "__main__":
    main()
