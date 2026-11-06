"""CLI: synthetic phrases for smoke tests and offline development.

Real music goes through the song database (``ml.db``): ``ingest`` the raw
folders, ``split`` by song and ``export`` the training JSONL. This script
only generates the MIDI-free synthetic phrases that the tests and
``ml.train --smoke`` use::

    uv run python -m ml.prepare_dataset --synthetic 2000 --out data/processed/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.datasets import save_phrases, split_by_song, split_ids, synthetic_phrases


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=int, required=True, help="number of synthetic phrases")
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic.jsonl"))
    ap.add_argument("--min-notes", type=int, default=5)
    ap.add_argument("--max-notes", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    phrases = synthetic_phrases(args.synthetic, seed=args.seed, min_notes=args.min_notes, max_notes=args.max_notes)
    splits = split_by_song(phrases, args.val_frac, args.test_frac, args.seed)
    save_phrases(phrases, args.out)
    args.out.with_name("splits.json").write_text(json.dumps(split_ids(splits), indent=1))
    print(f"wrote {len(phrases)} synthetic phrases to {args.out}; "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
