"""Probe: does a checkpoint reproduce a chug-heavy metal riff drawn exactly?

The riff is the one from the "Decoding experiments" section of ml/README.md:
16 notes, mostly repeated E2, with moves of 1, 3 and 6 semitones, eighth
notes at 100 bpm.  The sketch is "drawn exactly": heights are the true
semitone offsets from the first note, every note at confidence 0.8, no
gesture tags and no tempo hint, i.e. what a careful user could draw.

    uv run python -m experiments.chug_probe runs/pop909/best.pt runs/mixed/best.pt

For each checkpoint it prints the top-1 interval accuracy (argmax of the
per-step distribution), the best candidate's accuracy and whether the exact
riff is among the top 5 candidates, plus the top candidate itself.
"""

from __future__ import annotations

import sys

import numpy as np

from core.representation import Observation, Sketch, interval_to_class
from ml.model import load_checkpoint

CHUG_RIFF = [40, 40, 40, 41, 40, 40, 43, 40, 40, 40, 46, 40, 40, 43, 41, 40]  # E2 chugs, F2, G2, Bb2
BPM = 100.0


def chug_sketch(pitches: list[int] = CHUG_RIFF, confidence: float = 0.8) -> Sketch:
    step = 60.0 / BPM / 2  # eighth notes
    obs = [Observation(i * step, float(p - pitches[0]), confidence) for i, p in enumerate(pitches)]
    return Sketch(obs)


def probe(checkpoint: str, pitches: list[int] = CHUG_RIFF, k: int = 5) -> dict:
    model, ckpt = load_checkpoint(checkpoint, "cpu")
    sketch = chug_sketch(pitches)
    truth = [b - a for a, b in zip(pitches, pitches[1:])]
    cls = np.array([interval_to_class(d) for d in truth])
    dist = np.asarray(model.predict(sketch))
    cands = [list(c) for c in model.candidates(sketch, k)]
    best = cands[0] if cands else []
    return {
        "checkpoint": checkpoint,
        "epoch": ckpt.get("epoch"),
        "top1_interval_acc": float((dist.argmax(axis=1) == cls).mean()),
        "best_candidate_acc": float(np.mean([a == b for a, b in zip(best, truth)])) if best else 0.0,
        "exact_in_top5": any(c == truth for c in cands),
        "best_candidate": best,
        "truth": truth,
    }


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    print(f"riff: {CHUG_RIFF}\ntrue intervals: {[b - a for a, b in zip(CHUG_RIFF, CHUG_RIFF[1:])]}")
    for path in argv:
        r = probe(path)
        print(f"\n{path} (epoch {r['epoch']})")
        print(f"  top-1 interval acc {r['top1_interval_acc']:.2f}   best-candidate acc "
              f"{r['best_candidate_acc']:.2f}   exact riff in top-5: {r['exact_in_top5']}")
        print(f"  best candidate intervals: {r['best_candidate']}")


if __name__ == "__main__":
    main(sys.argv[1:])
