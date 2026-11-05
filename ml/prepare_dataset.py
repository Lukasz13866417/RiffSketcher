"""CLI: extract phrases from one or more sources, tag them and split by song.

Examples
--------
    uv run python -m ml.prepare_dataset --midi-dir data/raw/pop909/POP909 --out data/processed/pop909.jsonl
    uv run python -m ml.prepare_dataset --midi-dir data/raw/pop909/POP909 \\
        --guitar-midi-dir data/raw/lakh_clean/clean_midi --max-per-song 40 \\
        --out data/processed/mixed.jsonl
    uv run python -m ml.prepare_dataset --synthetic 2000 --out data/processed/synthetic.jsonl

Sources (any combination; each phrase's ``source`` field says where it came from)
    --midi-dir         melody track per file (POP909 layout: ``<id>/<id>.mid``), tag ``pop909``
    --guitar-midi-dir  guitar tracks of general MIDI files (Lakh ``clean_midi``: ``Artist/Title.mid``,
                       ``Title.1.mid`` versions share a song id), tag ``lakh_guitar``
    --dadagp-dir       DadaGP token files (``*.txt``), tag ``dadagp``

Writes the phrases as JSONL and a ``splits.json`` (song-level train/val/test
ids) next to the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ml.datasets import (Phrase, phrases_from_dadagp, phrases_from_guitar_midi, phrases_from_midi,
                         save_phrases, split_by_song, split_ids, synthetic_phrases)

LOADERS = {"melody": phrases_from_midi, "guitar": phrases_from_guitar_midi, "dadagp": phrases_from_dadagp}
# (loader, path, song_id, source, min_notes, max_notes)
Job = tuple[str, Path, str, str, int, int]


def midi_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mid", ".midi"))


def pop909_jobs(root: Path, min_notes: int, max_notes: int) -> list[Job]:
    # POP909 ships "<id>/<id>.mid" plus versions in "<id>/versions/"; skip the latter
    files = [f for f in midi_files(root) if "versions" not in f.parts]
    return [("melody", f, f.parent.name if f.parent != root else f.stem, "pop909", min_notes, max_notes)
            for f in files]


def lakh_song_id(path: Path, root: Path) -> str:
    """``Artist/Title`` with Lakh's ``.1``/``.2`` version suffix removed, so every
    transcription of a song lands in the same split."""
    title = re.sub(r"\.\d+$", "", path.stem)
    artist = path.parent.name if path.parent != root else ""
    return f"{artist}/{title}" if artist else title


def guitar_jobs(root: Path, min_notes: int, max_notes: int) -> list[Job]:
    return [("guitar", f, lakh_song_id(f, root), "lakh_guitar", min_notes, max_notes) for f in midi_files(root)]


def dadagp_jobs(root: Path, min_notes: int, max_notes: int) -> list[Job]:
    files = sorted(p for p in root.rglob("*.txt"))
    return [("dadagp", f, f.name.split(".")[0], "dadagp", min_notes, max_notes) for f in files]


def run_job(job: Job) -> list[Phrase]:
    """Extract one file; corrupt files happen, so failures return no phrases."""
    loader, path, song_id, source, min_notes, max_notes = job
    warnings.simplefilter("ignore")  # pretty_midi warns about odd tempo/key events
    try:
        return LOADERS[loader](path, min_notes=min_notes, max_notes=max_notes, song_id=song_id, source=source)
    except Exception as e:
        print(f"  skip {path}: {e}")
        return []


def collect_phrases(jobs: list[Job], workers: int = 1) -> list[Phrase]:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **kw):
            return it
    out: list[Phrase] = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for phrases in tqdm(pool.map(run_job, jobs, chunksize=8), total=len(jobs), desc="files"):
                out.extend(phrases)
    else:
        for job in tqdm(jobs, desc="files"):
            out.extend(run_job(job))
    return out


def cap_per_song(phrases: list[Phrase], max_per_song: int | None, seed: int = 0) -> list[Phrase]:
    """Keep at most ``max_per_song`` phrases per song (a random subset, in
    time order): Lakh songs with several guitar tracks would otherwise
    outweigh POP909 many times over."""
    if not max_per_song:
        return phrases
    by_song: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(phrases):
        by_song[p.song_id].append(i)
    rng = np.random.default_rng(seed)
    keep: set[int] = set()
    for idx in by_song.values():
        keep.update(idx if len(idx) <= max_per_song else rng.choice(idx, max_per_song, replace=False).tolist())
    return [p for i, p in enumerate(phrases) if i in keep]


def split_keeping(phrases: list[Phrase], base_ids: dict[str, list[str]] | None, val_frac: float,
                  test_frac: float, seed: int) -> dict[str, list[Phrase]]:
    """Song-level split that keeps the assignments of an earlier ``splits.json``.

    Songs listed in ``base_ids`` stay in their split and only new songs are
    drawn at random, so a model trained on the earlier file is never
    evaluated on songs it saw in training.
    """
    fixed = {sid: k for k, v in (base_ids or {}).items() for sid in v}
    splits = split_by_song([p for p in phrases if p.song_id not in fixed], val_frac, test_frac, seed)
    for p in phrases:
        if p.song_id in fixed:
            splits[fixed[p.song_id]].append(p)
    return splits


def summarise(phrases: list[Phrase], splits: dict[str, list[Phrase]]) -> str:
    lines = [f"phrases: {len(phrases)}  songs: {len({p.song_id for p in phrases})}"]
    for source in sorted({p.source for p in phrases}):
        sub = [p for p in phrases if p.source == source]
        lengths = np.array([len(p) for p in sub])
        intervals = Counter(abs(d) for p in sub for d in p.intervals)
        total = sum(intervals.values()) or 1
        lines.append(f"[{source}] phrases: {len(sub)}  songs: {len({p.song_id for p in sub})}  "
                     f"notes/phrase: mean {lengths.mean():.1f}  min {lengths.min()}  max {lengths.max()}")
        lines.append(f"[{source}] interval |d| histogram: "
                     + ", ".join(f"{d}:{intervals[d] / total:.2f}" for d in range(0, 8))
                     + f", >=8:{sum(v for d, v in intervals.items() if d >= 8) / total:.2f}")
    lines.append("splits: " + ", ".join(f"{k}={len(v)} phrases/{len({p.song_id for p in v})} songs"
                                        for k, v in splits.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--midi-dir", type=Path, help="melody-track MIDI files (POP909 layout), searched recursively")
    ap.add_argument("--guitar-midi-dir", type=Path, help="general MIDI files; guitar tracks only (Lakh)")
    ap.add_argument("--dadagp-dir", type=Path, help="DadaGP token files (*.txt), searched recursively")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic phrases instead")
    ap.add_argument("--out", type=Path, default=Path("data/processed/phrases.jsonl"))
    ap.add_argument("--min-notes", type=int, default=5)
    ap.add_argument("--max-notes", type=int, default=16)
    ap.add_argument("--max-per-song", type=int, default=None, help="cap phrases per song (guitar sources)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-splits", type=Path, default=None,
                    help="an earlier splits.json whose song assignments are kept")
    ap.add_argument("--limit", type=int, default=None, help="max files per source (debug)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)

    jobs: list[Job] = []
    for root, make_jobs in ((args.midi_dir, pop909_jobs), (args.guitar_midi_dir, guitar_jobs),
                            (args.dadagp_dir, dadagp_jobs)):
        if root is None:
            continue
        if not root.exists():
            raise SystemExit(f"{root} does not exist; see ml/README.md for the download step")
        new = make_jobs(root, args.min_notes, args.max_notes)
        jobs.extend(new[:args.limit] if args.limit else new)

    if jobs:
        phrases = collect_phrases(jobs, args.workers)
        guitar = cap_per_song([p for p in phrases if p.source != "pop909"], args.max_per_song, args.seed)
        phrases = [p for p in phrases if p.source == "pop909"] + guitar
    elif args.synthetic > 0:
        phrases = synthetic_phrases(args.synthetic, seed=args.seed, min_notes=args.min_notes,
                                    max_notes=args.max_notes)
    else:
        raise SystemExit("give a source directory or --synthetic N")
    if not phrases:
        raise SystemExit("no phrases extracted")

    base_ids = json.loads(args.base_splits.read_text()) if args.base_splits else None
    splits = split_keeping(phrases, base_ids, args.val_frac, args.test_frac, args.seed)
    save_phrases(phrases, args.out)
    splits_path = args.out.with_name("splits.json")
    splits_path.write_text(json.dumps(split_ids(splits), indent=1))
    print(summarise(phrases, splits))
    print(f"wrote {args.out} and {splits_path}")


if __name__ == "__main__":
    main()
