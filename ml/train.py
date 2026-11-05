"""CLI: train the SketchTransformer on phrases with on-the-fly sketch simulation.

    uv run python -m ml.train --data data/processed/phrases.jsonl --epochs 20 --out runs/first
    uv run python -m ml.train --smoke      # a few steps on synthetic phrases, CPU, < 1 min

Cross-entropy over interval classes (padding masked), AdamW, warmup + cosine
LR, best checkpoint (by validation interval accuracy) saved as ``best.pt``
with the model config inside it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.representation import N_INTERVAL_CLASSES
from ml.datasets import (GENRES, PAD_TARGET, PhraseDataset, apply_split_ids, collate, genre_of_source, load_phrases,
                         split_by_song, split_ids, synthetic_phrases)
from ml.model import ModelConfig, SketchTransformer, count_parameters, save_checkpoint
from ml.sketch_simulator import SketchSimulator


def lr_at(step: int, total: int, warmup: int, base: float, floor: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model: SketchTransformer, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Teacher-forced validation loss and interval accuracy."""
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        target = batch["target"]
        valid = target != PAD_TARGET
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, N_INTERVAL_CLASSES), target.reshape(-1),
                                                 ignore_index=PAD_TARGET, reduction="sum")
        tot_loss += float(loss)
        tot_correct += int(((logits.argmax(-1) == target) & valid).sum())
        tot_n += int(valid.sum())
    return tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1)


def train(train_phrases, val_phrases, model_cfg: ModelConfig, sim_level: str, out: Path,
          epochs: int = 20, batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
          max_steps_per_epoch: int | None = None, seed: int = 0, device: str | None = None,
          genre_dropout: float = 0.0, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sim = SketchSimulator(seed, sim_level)
    train_ds = PhraseDataset(train_phrases, sim, seed=seed, genre_dropout=genre_dropout)
    val_ds = PhraseDataset(val_phrases, sim, seed=seed + 1, epoch=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # genres with training data; the UI only offers these, an untrained genre embedding is noise
    genres_seen = ["unknown"] + sorted({genre_of_source(p.source) for p in train_phrases} - {"unknown"}) \
        if model_cfg.n_genres else []
    model = SketchTransformer(model_cfg).to(device_t)
    log(f"device {device_t}; model {count_parameters(model) / 1e6:.2f}M params; "
        f"{len(train_ds)} train / {len(val_ds)} val phrases; sim={sim_level}; "
        f"genres={'on (dropout %.2f)' % genre_dropout if model_cfg.n_genres else 'off'}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    steps_per_epoch = len(train_loader) if max_steps_per_epoch is None else min(len(train_loader), max_steps_per_epoch)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, min(200, total_steps // 10))

    out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_acc, step = -1.0, 0
    for epoch in range(epochs):
        model.train()
        train_ds.set_epoch(epoch + 1)
        t0 = time.time()
        losses: list[float] = []
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup, lr)
            batch = {k: v.to(device_t) for k, v in batch.items()}
            loss = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        val_loss, val_acc = evaluate(model, val_loader, device_t)
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "val_loss": val_loss, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"],
               "seconds": time.time() - t0}
        history.append(rec)
        log(f"epoch {rec['epoch']:3d}  train {rec['train_loss']:.4f}  val {val_loss:.4f}  "
            f"val_acc {val_acc:.3f}  lr {rec['lr']:.2e}  ({rec['seconds']:.1f}s)")
        extra = {"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc, "sim_level": sim_level,
                 "history": history, "genres_seen": genres_seen}
        save_checkpoint(model, str(out / "last.pt"), **extra)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, str(out / "best.pt"), **extra)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    log(f"best val_acc {best_acc:.3f}; checkpoints in {out}")
    return {"history": history, "best_val_acc": best_acc, "model": model}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default="data/processed/phrases.jsonl")
    ap.add_argument("--splits", type=str, default=None, help="splits.json (default: next to --data)")
    ap.add_argument("--synthetic", type=int, default=0, help="train on N synthetic phrases instead of --data")
    ap.add_argument("--smoke", action="store_true", help="tiny model, few steps on synthetic data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--sim", default="medium", help="noise preset: easy/medium/hard")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/first"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-genre", action="store_true", help="train without the genre embedding")
    ap.add_argument("--genre-dropout", type=float, default=0.2,
                    help="fraction of training sketches shown with genre 'unknown'")
    ap.add_argument("--source", default=None,
                    help="train and validate only on phrases from this source (per-genre model)")
    args = ap.parse_args(argv)
    n_genres = 0 if args.no_genre else len(GENRES)

    if args.smoke:
        phrases = synthetic_phrases(256, seed=args.seed)
        splits = split_by_song(phrases, 0.2, 0.0, args.seed)
        cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, n_genres=n_genres)
        epochs = args.epochs if args.epochs != 20 else 3
        out = args.out if args.out != Path("runs/first") else Path("runs/smoke")
        train(splits["train"], splits["val"], cfg, args.sim, out, epochs=epochs, batch_size=32,
              lr=1e-3, max_steps_per_epoch=args.max_steps_per_epoch or 10, seed=args.seed,
              device=args.device or "cpu", genre_dropout=args.genre_dropout)
        return

    if args.synthetic:
        phrases = synthetic_phrases(args.synthetic, seed=args.seed)
        splits = split_by_song(phrases, 0.1, 0.1, args.seed)
    else:
        phrases = load_phrases(args.data)
        splits_path = Path(args.splits) if args.splits else Path(args.data).with_name("splits.json")
        if splits_path.exists():
            splits = apply_split_ids(phrases, json.loads(splits_path.read_text()))
        else:
            print(f"no {splits_path}; splitting by song now")
            splits = split_by_song(phrases, 0.1, 0.1, args.seed)
    if args.source:
        splits = {k: [p for p in v if p.source == args.source] for k, v in splits.items()}
    print({k: len(v) for k, v in splits.items()})
    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, d_ff=args.d_ff,
                      dropout=args.dropout, n_genres=n_genres)
    train(splits["train"], splits["val"], cfg, args.sim, args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
          max_steps_per_epoch=args.max_steps_per_epoch, seed=args.seed, device=args.device,
          genre_dropout=args.genre_dropout)


if __name__ == "__main__":
    main()
