"""CLI: extract phrases from a MIDI directory (or synthesise them) and split by song.

Examples
--------
    uv run python -m ml.prepare_dataset --midi-dir data/raw/POP909 --out data/processed/phrases.jsonl
    uv run python -m ml.prepare_dataset --synthetic 2000 --out data/processed/synthetic.jsonl

Writes the phrases as JSONL and a ``splits.json`` (song-level train/val/test
ids) next to the output file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ml.datasets import (Phrase, phrases_from_midi, save_phrases, split_by_song, split_ids,
                         synthetic_phrases)


def collect_midi_phrases(midi_dir: Path, min_notes: int, max_notes: int,
                         limit: int | None = None) -> list[Phrase]:
    files = sorted(p for p in midi_dir.rglob("*") if p.suffix.lower() in (".mid", ".midi"))
    # POP909 ships "<id>/<id>.mid" plus versions in "<id>/versions/"; skip the latter
    files = [f for f in files if "versions" not in f.parts]
    if limit:
        files = files[:limit]
    out: list[Phrase] = []
    n_fail = 0
    try:
        from tqdm import tqdm
        it = tqdm(files, desc="midi")
    except ImportError:  # pragma: no cover
        it = files
    for f in it:
        song_id = f.parent.name if f.parent != midi_dir else f.stem
        try:
            out.extend(phrases_from_midi(f, min_notes=min_notes, max_notes=max_notes,
                                         song_id=song_id, source=str(f.relative_to(midi_dir))))
        except Exception as e:  # corrupt files happen; keep going
            n_fail += 1
            print(f"  skip {f}: {e}")
    if n_fail:
        print(f"{n_fail} file(s) failed to parse")
    return out


def summarise(phrases: list[Phrase], splits: dict[str, list[Phrase]]) -> str:
    lengths = np.array([len(p) for p in phrases])
    intervals = Counter(abs(d) for p in phrases for d in p.intervals)
    total = sum(intervals.values()) or 1
    lines = [
        f"phrases: {len(phrases)}  songs: {len({p.song_id for p in phrases})}",
        f"notes/phrase: mean {lengths.mean():.1f}  min {lengths.min()}  max {lengths.max()}",
        "interval |d| histogram: " + ", ".join(f"{d}:{intervals[d] / total:.2f}" for d in range(0, 8))
        + f", >=8:{sum(v for d, v in intervals.items() if d >= 8) / total:.2f}",
        "splits: " + ", ".join(f"{k}={len(v)} phrases/{len({p.song_id for p in v})} songs"
                               for k, v in splits.items()),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--midi-dir", type=Path, help="directory searched recursively for .mid files")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic phrases instead")
    ap.add_argument("--out", type=Path, default=Path("data/processed/phrases.jsonl"))
    ap.add_argument("--min-notes", type=int, default=5)
    ap.add_argument("--max-notes", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="max MIDI files (debug)")
    args = ap.parse_args(argv)

    if args.midi_dir is not None:
        if not args.midi_dir.exists():
            raise SystemExit(f"{args.midi_dir} does not exist; see ml/README.md for the download step")
        phrases = collect_midi_phrases(args.midi_dir, args.min_notes, args.max_notes, args.limit)
    elif args.synthetic > 0:
        phrases = synthetic_phrases(args.synthetic, seed=args.seed, min_notes=args.min_notes,
                                    max_notes=args.max_notes)
    else:
        raise SystemExit("give --midi-dir or --synthetic N")
    if not phrases:
        raise SystemExit("no phrases extracted")

    splits = split_by_song(phrases, args.val_frac, args.test_frac, args.seed)
    save_phrases(phrases, args.out)
    splits_path = args.out.with_name("splits.json")
    splits_path.write_text(json.dumps(split_ids(splits), indent=1))
    print(summarise(phrases, splits))
    print(f"wrote {args.out} and {splits_path}")


if __name__ == "__main__":
    main()
