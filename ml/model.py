"""SketchTransformer: conditional Transformer from sketch tokens to intervals.

Encoder
    One token per tapped note built from the continuous features (IOI,
    height, confidence, missing flag) and the gesture multi-hot vector, with
    learned positional embeddings and, when ``n_genres > 0``, a learned genre
    embedding added to every token (``sketch.meta["genre"]``; see
    ``ml.datasets.GENRES``).
Decoder
    Autoregressive over the 49 interval classes.  Step ``k`` predicts the
    interval between notes ``k`` and ``k+1``; its input is the previous
    interval token plus a projection of the encoder states of notes ``k`` and
    ``k+1`` so the decoder has direct local evidence as well as cross
    attention over the whole sketch.
Inference
    ``generate`` (greedy / sampling / beam), ``predict`` (marginal per-step
    distribution, same interface as :mod:`ml.baseline`) and
    ``riffs_from_sketch`` which returns :class:`core.representation.Riff`
    candidates timed with :func:`core.rhythm.quantise`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.representation import N_INTERVAL_CLASSES, Note, Riff, Sketch, class_to_interval
from core.rhythm import quantise
from ml.datasets import (GESTURE_KINDS, N_CONTINUOUS_FEATURES, PAD_TARGET, RHYTHM_CLASSES, collate,
                         example_to_tensors)

BOS = N_INTERVAL_CLASSES  # extra decoder-input token


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 64
    n_cont: int = N_CONTINUOUS_FEATURES
    n_gest: int = len(GESTURE_KINDS)
    n_genres: int = 0  # 0 = no genre conditioning (checkpoints trained before it existed)
    n_rhythm: int = 0  # 0 = no rhythm head; else len(RHYTHM_CLASSES): one gap class per interval,
    #                    predicted from the encoder states of the two notes (not autoregressive)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class SketchTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.in_proj = nn.Sequential(nn.Linear(c.n_cont + c.n_gest, c.d_model), nn.GELU(),
                                     nn.Linear(c.d_model, c.d_model))
        self.enc_pos = nn.Embedding(c.max_len, c.d_model)
        self.genre_emb = nn.Embedding(c.n_genres, c.d_model) if c.n_genres > 0 else None
        enc_layer = nn.TransformerEncoderLayer(c.d_model, c.n_heads, c.d_ff, c.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, c.n_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(c.d_model)

        self.tok_emb = nn.Embedding(N_INTERVAL_CLASSES + 1, c.d_model)
        # rhythm head: gap k from the encoder states of notes k and k+1. Non-autoregressive on
        # purpose: a decoder-side version reached 0.85 teacher-forced but 0.60 in free decoding
        # because wrong gaps compounded, and it cost interval accuracy by sharing the decoder.
        self.out_rhythm = nn.Sequential(nn.Linear(2 * c.d_model, c.d_model), nn.GELU(),
                                        nn.Linear(c.d_model, c.n_rhythm)) if c.n_rhythm > 0 else None
        self.dec_pos = nn.Embedding(c.max_len, c.d_model)
        self.local_proj = nn.Linear(2 * c.d_model, c.d_model)
        dec_layer = nn.TransformerDecoderLayer(c.d_model, c.n_heads, c.d_ff, c.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, c.n_layers)
        self.dec_norm = nn.LayerNorm(c.d_model)
        self.out = nn.Linear(c.d_model, N_INTERVAL_CLASSES)

    # -- pieces ------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory (B, n, d), mask (B, n) True = real note)."""
        cont, gest, mask = batch["cont"], batch["gest"], batch["mask"]
        cont = cont[..., :self.cfg.n_cont]  # older checkpoints know a prefix of the feature columns
        B, n, _ = cont.shape
        x = self.in_proj(torch.cat([cont, gest], dim=-1))
        pos = torch.arange(n, device=cont.device).clamp(max=self.cfg.max_len - 1)
        x = x + self.enc_pos(pos)[None]
        if self.genre_emb is not None:
            genre = batch.get("genre")
            if genre is None:
                genre = torch.zeros(B, dtype=torch.long, device=cont.device)
            x = x + self.genre_emb(genre.clamp(max=self.cfg.n_genres - 1))[:, None, :]
        mem = self.encoder(x, src_key_padding_mask=~mask)
        return self.enc_norm(mem), mask

    @property
    def has_rhythm(self) -> bool:
        return self.out_rhythm is not None

    def rhythm_logits(self, memory: torch.Tensor) -> torch.Tensor | None:
        """Gap-class logits (B, n-1, n_rhythm) from consecutive encoder states, or None."""
        if self.out_rhythm is None:
            return None
        pair = torch.cat([memory[:, :-1], memory[:, 1:]], dim=-1)
        if pair.shape[1] == 0:  # single-note sketch
            pair = torch.cat([memory, memory], dim=-1)
        return self.out_rhythm(pair)

    def decode_hidden(self, tokens: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                      rtokens: torch.Tensor | None = None) -> torch.Tensor:
        """Decoder hidden states (B, L, d) for interval inputs ``tokens`` (BOS +
        previous classes). ``rtokens`` is accepted and ignored for API stability."""
        B, L = tokens.shape
        n = memory.shape[1]
        pos = torch.arange(L, device=tokens.device)
        x = self.tok_emb(tokens) + self.dec_pos(pos.clamp(max=self.cfg.max_len - 1))[None]
        # local evidence: encoder states of note k and k+1 for step k
        idx_a = pos.clamp(max=n - 1)
        idx_b = (pos + 1).clamp(max=n - 1)
        local = torch.cat([memory[:, idx_a], memory[:, idx_b]], dim=-1)
        x = x + self.local_proj(local)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
        # step k is valid iff note k+1 exists
        step_valid = mask[:, 1:1 + L] if n - 1 >= L else F.pad(mask[:, 1:], (0, L - (n - 1)), value=False)
        tgt_pad = ~step_valid
        tgt_pad[:, 0] = False  # never mask everything (nan guard)
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_pad,
                         memory_key_padding_mask=~mask)
        return self.dec_norm(h)

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
               rtokens: torch.Tensor | None = None) -> torch.Tensor:
        """Interval logits (B, L, 49)."""
        return self.out(self.decode_hidden(tokens, memory, mask, rtokens))

    def decode_joint(self, tokens: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                     rtokens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(interval logits (B, L, 49), rhythm logits (B, n-1, n_rhythm) or None).

        The rhythm logits come from the encoder and do not depend on ``tokens``.
        """
        return self.out(self.decode_hidden(tokens, memory, mask)), self.rhythm_logits(memory)

    def _prev_tokens(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Teacher-forcing decoder inputs: BOS followed by the true previous classes."""
        target = batch["target"]
        B = target.shape[0]
        prev = torch.cat([torch.full((B, 1), BOS, dtype=torch.long, device=target.device),
                          target[:, :-1].clamp(min=0)], dim=1)
        return prev, None

    # -- training ---------------------------------------------------------- #

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Teacher-forced interval logits (B, n-1, 49) from ``batch['target']``."""
        return self.forward_joint(batch)[0]

    def forward_joint(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Teacher-forced (interval logits, rhythm logits or None)."""
        memory, mask = self.encode(batch)
        prev, rprev = self._prev_tokens(batch)
        return self.decode_joint(prev, memory, mask, rprev)

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Interval cross-entropy plus, with a rhythm head and targets, rhythm cross-entropy."""
        iv, rh = self.forward_joint(batch)
        loss = F.cross_entropy(iv.reshape(-1, N_INTERVAL_CLASSES), batch["target"].reshape(-1),
                               ignore_index=PAD_TARGET)
        if rh is not None and "rhythm" in batch:
            L = batch["rhythm"].shape[1]
            loss = loss + F.cross_entropy(rh[:, :L].reshape(-1, self.cfg.n_rhythm),
                                          batch["rhythm"][:, :rh.shape[1]].reshape(-1),
                                          ignore_index=PAD_TARGET)
        return loss

    def step_logprobs(self, tokens: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                      rtokens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Log-probabilities for the next interval step (B, 49); the second item is None
        (gaps are predicted by the encoder-side head, see ``rhythm_logits``)."""
        iv = self.decode(tokens, memory, mask)
        return F.log_softmax(iv[:, -1], dim=-1), None

    # -- inference --------------------------------------------------------- #

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], n_samples: int = 1, temperature: float = 1.0,
                 mode: str = "sample") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode interval classes for every sketch in the batch.

        Returns ``(classes (B, S, L), logp (B, S), step_probs (B, S, L, 49))``
        where ``L = n_max - 1`` and ``S = n_samples``.  ``mode`` is ``"greedy"``,
        ``"sample"`` (temperature-scaled) or ``"beam"`` (``n_samples`` beams).
        Padded steps hold class 0 / uniform probabilities.
        """
        self.eval()
        memory, mask = self.encode(batch)
        if mode == "beam":
            return self._beam(memory, mask, n_samples)
        B, n, d = memory.shape
        L = max(n - 1, 1)
        S = n_samples if mode == "sample" else 1
        mem = memory.repeat_interleave(S, dim=0)
        msk = mask.repeat_interleave(S, dim=0)
        tokens = torch.full((B * S, 1), BOS, dtype=torch.long, device=memory.device)
        probs_all = []
        logp = torch.zeros(B * S, device=memory.device)
        for step in range(L):
            logits = self.decode(tokens, mem, msk)[:, -1] / max(temperature, 1e-4)
            p = F.softmax(logits, dim=-1)
            if mode == "greedy":
                nxt = p.argmax(-1)
            else:
                nxt = torch.multinomial(p, 1).squeeze(-1)
            logp = logp + torch.log(p.gather(1, nxt[:, None]).squeeze(1) + 1e-12)
            probs_all.append(p)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        self._last_rhythm = self._rhythm_classes(memory, S, L)
        classes = tokens[:, 1:].view(B, S, L)
        step_probs = torch.stack(probs_all, dim=1).view(B, S, L, N_INTERVAL_CLASSES)
        if mode == "greedy" and n_samples > 1:
            classes = classes.expand(B, n_samples, L)
            step_probs = step_probs.expand(B, n_samples, L, N_INTERVAL_CLASSES)
            logp = logp.view(B, 1).expand(B, n_samples)
        else:
            logp = logp.view(B, S)
        return classes, logp, step_probs

    def _rhythm_classes(self, memory: torch.Tensor, S: int, L: int) -> torch.Tensor | None:
        """Greedy gap classes (B, S, L) from the encoder head, or None without one."""
        rl = self.rhythm_logits(memory)
        if rl is None:
            return None
        cls = rl.argmax(-1)
        if cls.shape[1] < L:
            cls = F.pad(cls, (0, L - cls.shape[1]))
        return cls[:, :L].unsqueeze(1).expand(-1, S, -1).contiguous()

    def _beam(self, memory: torch.Tensor, mask: torch.Tensor, beam: int):
        """Per-example beam search (sequences are short, so a plain loop is fine)."""
        B, n_max, _ = memory.shape
        L = max(n_max - 1, 1)
        self._last_rhythm = self._rhythm_classes(memory, beam, L)
        out_cls = torch.zeros(B, beam, L, dtype=torch.long, device=memory.device)
        out_lp = torch.full((B, beam), -1e9, device=memory.device)
        out_pr = torch.full((B, beam, L, N_INTERVAL_CLASSES), 1.0 / N_INTERVAL_CLASSES, device=memory.device)
        for b in range(B):
            n_b = int(mask[b].sum())
            steps = max(n_b - 1, 0)
            mem_b, msk_b = memory[b:b + 1, :n_b], mask[b:b + 1, :n_b]
            beams: list[tuple[float, list[int], list[torch.Tensor]]] = [(0.0, [], [])]
            for _ in range(steps):
                tokens = torch.tensor([[BOS] + seq for _, seq, _ in beams], dtype=torch.long,
                                      device=memory.device)
                lp, _ = self.step_logprobs(tokens, mem_b.expand(len(beams), -1, -1),
                                           msk_b.expand(len(beams), -1))
                cand: list[tuple[float, list[int], list[torch.Tensor]]] = []
                for (s, seq, pr), row in zip(beams, lp):
                    top = torch.topk(row, min(beam, N_INTERVAL_CLASSES))
                    for v, c in zip(top.values.tolist(), top.indices.tolist()):
                        cand.append((s + v, seq + [c], pr + [row.exp()]))
                cand.sort(key=lambda x: -x[0])
                beams = cand[:beam]
            for j, (s, seq, pr) in enumerate(beams[:beam]):
                if seq:
                    out_cls[b, j, :len(seq)] = torch.tensor(seq, device=memory.device)
                    out_pr[b, j, :len(seq)] = torch.stack(pr)
                out_lp[b, j] = s
        return out_cls, out_lp, out_pr

    def _batch_from_sketch(self, sketch: Sketch) -> dict[str, torch.Tensor]:
        item = example_to_tensors(sketch, None)
        return {k: v.to(self.device) for k, v in collate([item]).items()}

    @torch.no_grad()
    def predict(self, sketch: Sketch, n_samples: int = 8, temperature: float = 1.0) -> np.ndarray:
        """Marginal (n-1, 49) interval distribution: average of the per-step
        softmax along the greedy path and ``n_samples - 1`` sampled paths."""
        n = len(sketch)
        if n < 2:
            return np.zeros((0, N_INTERVAL_CLASSES))
        batch = self._batch_from_sketch(sketch)
        _, _, p_greedy = self.generate(batch, 1, mode="greedy")
        probs = [p_greedy[0, 0]]
        if n_samples > 1:
            _, _, p_s = self.generate(batch, n_samples - 1, temperature, mode="sample")
            probs.extend(p_s[0])
        dist = torch.stack(probs).mean(0)[:n - 1].cpu().numpy().astype(float)
        return dist / dist.sum(axis=1, keepdims=True)

    @torch.no_grad()
    def candidates(self, sketch: Sketch, k: int = 5) -> list[list[int]]:
        """Top-``k`` interval sequences by beam search, best first."""
        n = len(sketch)
        if n < 2:
            return [[]]
        cls, lp, _ = self.generate(self._batch_from_sketch(sketch), k, mode="beam")
        out: list[list[int]] = []
        for j in range(k):
            if lp[0, j] > -1e8:
                out.append([class_to_interval(int(c)) for c in cls[0, j, :n - 1]])
        return out

    @torch.no_grad()
    def rhythm_from_classes(self, classes, onsets_s: list[float]) -> tuple[list[float], list[float], float]:
        """Turn predicted gap classes into (onsets_beats, durations_beats, tempo).

        The tempo is the least-squares fit of predicted beats to tapped seconds.
        """
        gaps = [RHYTHM_CLASSES[int(c)] for c in classes]
        onsets = [0.0]
        for g in gaps:
            onsets.append(onsets[-1] + g)
        span_s = (onsets_s[-1] - onsets_s[0]) if len(onsets_s) > 1 else 0.0
        tempo = 60.0 * onsets[-1] / span_s if span_s > 1e-3 and onsets[-1] > 0 else 120.0
        tempo = float(min(240.0, max(40.0, tempo)))
        durations = gaps + [float(np.median(gaps)) if gaps else 1.0]
        return onsets, durations, tempo

    def riffs_from_sketch(self, sketch: Sketch, n: int = 5, start_pitch: int = 52,
                          mode: str = "beam", temperature: float = 1.0) -> list[Riff]:
        """Candidate riffs: beam/sampled intervals realised from ``start_pitch``
        with onsets quantised by :func:`core.rhythm.quantise`."""
        sketch = sketch.sorted()
        if len(sketch) == 0:
            return []
        rhythm = quantise(sketch.onsets, tempo_hint=sketch.tempo_hint)
        m = len(sketch) - 1
        batch = self._batch_from_sketch(sketch)
        if m == 0:
            seqs, lps = [[]], [0.0]
        else:
            cls, lp, _ = self.generate(batch, n, temperature, mode=mode)
            seqs = [[class_to_interval(int(c)) for c in cls[0, j, :m]] for j in range(cls.shape[1])]
            lps = lp[0].tolist()
        riffs: list[Riff] = []
        seen: set[tuple[int, ...]] = set()
        for seq, score in sorted(zip(seqs, lps), key=lambda x: -x[1]):
            key = tuple(seq)
            if key in seen or score < -1e8:
                continue
            seen.add(key)
            pitches = [start_pitch] + list(start_pitch + np.cumsum(seq)) if seq else [start_pitch]
            pitches = [int(np.clip(p, 24, 100)) for p in pitches]
            notes = [Note(p, o, d) for p, o, d in zip(pitches, rhythm.onsets_beats, rhythm.durations_beats)]
            riffs.append(Riff(notes, tempo=rhythm.tempo, score=float(score),
                              meta={"method": "transformer", "mode": mode, "grid": rhythm.grid}))
            if len(riffs) >= n:
                break
        return riffs


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #


def save_checkpoint(model: SketchTransformer, path: str, **extra: Any) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model.cfg.to_dict(), **extra}, path)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[SketchTransformer, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SketchTransformer(ModelConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
