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
from ml.datasets import (GESTURE_KINDS, N_CONTINUOUS_FEATURES, PAD_TARGET, RHYTHM_CLASSES, class_to_ratio,
                         collate, example_to_tensors)

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
    n_branch: int = 0  # encoder layers owned by the timing branch, on top of the shared trunk
    branch_detach: bool = False  # True: the timing branch reads the trunk but never reshapes it
    n_duration: int = 0  # 0 = no duration head; else len(DURATION_CLASSES): one length class per note,
    #                      read from the same timing branch as the gap head
    n_expert_layers: int = 0  # encoder layers (the last ones) whose attention is shared but whose MLP
    #                            is per task: a pitch stream feeds the decoder, a timing stream feeds the
    #                            gap and duration heads. An alternative to n_branch (whole private layers).
    expand: bool = False  # expansion decoder (implies joint): a fourth field per step says whether the
    #                       note advances to the next tap (1) or is an extra note of the current tap (0);
    #                       output length is free, decoding ends on an advance past the last tap
    joint: bool = False  # joint-token decoder: every decoder step emits interval, gap and duration,
    #                      conditioned on the previous step's three tokens (needs n_rhythm and n_duration);
    #                      the encoder-side timing branch is not built

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class TaskExpertLayer(nn.Module):
    """A pre-norm transformer layer with one shared attention and one MLP per task.

    Attention mixes information across notes, which is the same job for every
    task, so it runs once on the mean of the task streams; each stream then
    adds the attention output and its own MLP, so the tasks stop competing for
    MLP weights while still shaping the same attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_tasks: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_tasks)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                                nn.Linear(d_ff, d_model)) for _ in range(n_tasks)])
        self.drop = nn.Dropout(dropout)

    def forward(self, streams: list[torch.Tensor], key_padding_mask: torch.Tensor | None) -> list[torch.Tensor]:
        shared = torch.stack(streams).mean(0)
        h = self.norm1(shared)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        a = self.drop(a)
        out = []
        for x, n2, f in zip(streams, self.norm2, self.ffn):
            x = x + a
            out.append(x + self.drop(f(n2(x))))
        return out


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
        n_shared = c.n_layers - (c.n_expert_layers if c.n_rhythm > 0 and not c.joint else 0)
        assert n_shared >= 0, "n_expert_layers cannot exceed n_layers"
        self.encoder = nn.TransformerEncoder(enc_layer, n_shared, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(c.d_model)
        # per-task MLP layers: shared attention, a pitch stream and a timing stream
        self.expert_layers = nn.ModuleList([TaskExpertLayer(c.d_model, c.n_heads, c.d_ff, c.dropout, 2)
                                            for _ in range(c.n_layers - n_shared)])
        self._timing_memory: torch.Tensor | None = None

        self.tok_emb = nn.Embedding(N_INTERVAL_CLASSES + 1, c.d_model)
        # rhythm head: gap k from the encoder states of notes k and k+1. Non-autoregressive on
        # purpose: a decoder-side version reached 0.85 teacher-forced but 0.60 in free decoding
        # because wrong gaps compounded, and it cost interval accuracy by sharing the decoder.
        self.out_rhythm = nn.Sequential(nn.Linear(2 * c.d_model, c.d_model), nn.GELU(),
                                        nn.Linear(c.d_model, c.n_rhythm)) if c.n_rhythm > 0 else None
        # One model, two branches: the trunk (input projection, genre embedding, ``encoder``) is
        # shared; the pitch branch is the decoder below; the rhythm branch is ``n_branch`` encoder
        # layers of its own, so the gap loss has capacity that is not the trunk's. A head read
        # straight off the trunk (n_branch=0, the first version) cost two points of interval
        # accuracy because the gap loss reshaped the shared states.
        if c.joint:  # timing comes out of the decoder, see decode_joint_hidden
            assert c.n_rhythm > 0 and c.n_duration > 0, "a joint decoder needs gap and duration classes"
            self.out_rhythm = None
        self.rhythm_branch = nn.TransformerEncoder(enc_layer, c.n_branch, enable_nested_tensor=False) \
            if c.n_rhythm > 0 and c.n_branch > 0 and not c.joint else None
        self.branch_norm = nn.LayerNorm(c.d_model) if (self.rhythm_branch is not None or self.expert_layers) else None
        # duration head: how long note k sounds as a share of its gap, from the timing state of note k
        self.out_duration = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(),
                                          nn.Linear(c.d_model, c.n_duration)) \
            if c.n_rhythm > 0 and c.n_duration > 0 and not c.joint else None
        # joint-token decoder: previous gap and duration classes are decoder inputs next to the
        # previous interval; within a step the gap is predicted given the interval and the
        # duration given both (factorised heads on one decoder state); note 0's duration, which
        # no step covers, comes from its encoder state
        if c.expand:
            assert c.joint, "expand needs the joint decoder"
            self.adv_emb = nn.Embedding(3, c.d_model)  # 0 stay, 1 next, 2 BOS
            self.out_adv = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, 2))
            self.out_iv_x = nn.Sequential(nn.Linear(2 * c.d_model, c.d_model), nn.GELU(),
                                          nn.Linear(c.d_model, N_INTERVAL_CLASSES))  # interval given advance
        if c.joint:
            self.gap_emb = nn.Embedding(c.n_rhythm + 1, c.d_model)  # last index = BOS
            self.dur_emb = nn.Embedding(c.n_duration + 1, c.d_model)
            self.out_gap_j = nn.Sequential(nn.Linear(2 * c.d_model, c.d_model), nn.GELU(),
                                           nn.Linear(c.d_model, c.n_rhythm))
            self.out_dur_j = nn.Sequential(nn.Linear(3 * c.d_model, c.d_model), nn.GELU(),
                                           nn.Linear(c.d_model, c.n_duration))
            self.out_dur0 = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(),
                                          nn.Linear(c.d_model, c.n_duration))
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
        gest = gest[..., :self.cfg.n_gest]  # older checkpoints know a prefix of the gesture kinds
        x = self.in_proj(torch.cat([cont, gest], dim=-1))
        pos = torch.arange(n, device=cont.device).clamp(max=self.cfg.max_len - 1)
        x = x + self.enc_pos(pos)[None]
        if self.genre_emb is not None:
            genre = batch.get("genre")
            if genre is None:
                genre = torch.zeros(B, dtype=torch.long, device=cont.device)
            x = x + self.genre_emb(genre.clamp(max=self.cfg.n_genres - 1))[:, None, :]
        mem = self.encoder(x, src_key_padding_mask=~mask)
        if self.expert_layers:
            streams = [mem, mem]
            for layer in self.expert_layers:
                streams = layer(streams, ~mask)
            mem = streams[0]
            self._timing_memory = self.branch_norm(streams[1])  # read by _timing_states for this batch
        return self.enc_norm(mem), mask

    @property
    def has_rhythm(self) -> bool:
        return self.out_rhythm is not None or self.cfg.joint

    @property
    def has_duration(self) -> bool:
        return self.out_duration is not None or self.cfg.joint

    @property
    def bos_gap(self) -> int:
        return self.cfg.n_rhythm

    @property
    def bos_dur(self) -> int:
        return self.cfg.n_duration

    def _timing_states(self, memory: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Per-note states of the timing branch (the trunk itself without branch layers; the
        timing stream of the last ``encode`` call with per-task MLP layers)."""
        if self.expert_layers and self._timing_memory is not None and self._timing_memory.shape[:2] == memory.shape[:2]:
            return self._timing_memory
        h = memory.detach() if self.cfg.branch_detach else memory
        if self.rhythm_branch is not None:
            h = self.branch_norm(self.rhythm_branch(h, src_key_padding_mask=None if mask is None else ~mask))
        return h

    def timing_logits(self, memory: torch.Tensor, mask: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(gap logits (B, n-1, n_rhythm) or None, duration logits (B, n, n_duration) or None)."""
        if self.out_rhythm is None:
            return None, None
        h = self._timing_states(memory, mask)
        return self._gap_logits(h), (self.out_duration(h) if self.out_duration is not None else None)

    def rhythm_logits(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Gap-class logits (B, n-1, n_rhythm) from consecutive timing states, or None without a head."""
        return self.timing_logits(memory, mask)[0]

    def duration_logits(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | None:
        return self.timing_logits(memory, mask)[1]

    def _gap_logits(self, h: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)
        if pair.shape[1] == 0:  # single-note sketch
            pair = torch.cat([h, h], dim=-1)
        return self.out_rhythm(pair)

    def decode_hidden(self, tokens: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                      rtokens: torch.Tensor | None = None, dtokens: torch.Tensor | None = None,
                      atokens: torch.Tensor | None = None, align: torch.Tensor | None = None) -> torch.Tensor:
        """Decoder hidden states (B, L, d) for interval inputs ``tokens`` (BOS +
        previous classes). In joint mode ``rtokens`` (BOS + previous gap classes) and
        ``dtokens`` (BOS + previous notes' duration classes) are added; otherwise
        they are ignored. In expansion mode ``atokens`` (BOS + previous advance
        tokens) are added too and ``align`` (B, L), the tap each step's current note
        belongs to (PAD_TARGET past the end), selects the local evidence."""
        B, L = tokens.shape
        n = memory.shape[1]
        pos = torch.arange(L, device=tokens.device)
        x = self.tok_emb(tokens) + self.dec_pos(pos.clamp(max=self.cfg.max_len - 1))[None]
        if self.cfg.joint:
            if rtokens is None:
                rtokens = torch.full_like(tokens, self.bos_gap)
            if dtokens is None:
                dtokens = torch.full_like(tokens, self.bos_dur)
            x = x + self.gap_emb(rtokens) + self.dur_emb(dtokens)
        if self.cfg.expand:
            if atokens is None:
                atokens = torch.full_like(tokens, 2)
            x = x + self.adv_emb(atokens)
        if self.cfg.expand and align is not None:
            # local evidence: the current tap and the next one, per step and per example
            valid = align != PAD_TARGET
            a = align.clamp(min=0, max=n - 1)
            b = (a + 1).clamp(max=n - 1)
            local = torch.cat([torch.gather(memory, 1, a[..., None].expand(-1, -1, memory.shape[-1])),
                               torch.gather(memory, 1, b[..., None].expand(-1, -1, memory.shape[-1]))], dim=-1)
            x = x + self.local_proj(local)
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
            tgt_pad = ~valid
            tgt_pad[:, 0] = False
            h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_pad,
                             memory_key_padding_mask=~mask)
            return self.dec_norm(h)
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
        return self.out(self.decode_hidden(tokens, memory, mask)), self.rhythm_logits(memory, mask)

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

    def forward_joint(self, batch: dict[str, torch.Tensor], ss_prob: float = 0.0
                      ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Teacher-forced (interval logits, gap logits or None, duration logits or None).

        In joint mode the duration logits cover every note (note 0 from its encoder
        state), and ``ss_prob`` is scheduled sampling: that share of the fed-back
        previous tokens is replaced by the model's own greedy prediction, so free
        decoding is not the first time it sees its own mistakes.
        """
        memory, mask = self.encode(batch)
        prev, _ = self._prev_tokens(batch)
        if not self.cfg.joint:
            iv = self.out(self.decode_hidden(prev, memory, mask))
            rh, du = self.timing_logits(memory, mask)
            return iv, rh, du
        gprev, dprev = self._prev_timing_tokens(batch)
        aprev = None
        if self.cfg.expand:
            adv = batch["advance"]
            aprev = torch.cat([torch.full((adv.shape[0], 1), 2, dtype=torch.long, device=adv.device),
                               adv[:, :-1].clamp(min=0)], 1)
        if ss_prob > 0:
            with torch.no_grad():
                iv0, rh0, du0 = self._joint_logits(batch, memory, mask, prev, gprev, dprev, aprev)
                flip = torch.rand_like(prev, dtype=torch.float) < ss_prob
                flip[:, 0] = False  # BOS stays
                prev = torch.where(flip, torch.cat([prev[:, :1], iv0.argmax(-1)[:, :-1]], 1), prev)
                gprev = torch.where(flip, torch.cat([gprev[:, :1], rh0.argmax(-1)[:, :-1]], 1), gprev)
                dprev = torch.where(flip, torch.cat([dprev[:, :1], du0.argmax(-1)[:, 1:prev.shape[1]]], 1), dprev)
                if aprev is not None and self._last_adv_logits is not None:
                    aprev = torch.where(flip, torch.cat([aprev[:, :1], self._last_adv_logits.argmax(-1)[:, :-1]], 1), aprev)
        return self._joint_logits(batch, memory, mask, prev, gprev, dprev, aprev)

    def _prev_timing_tokens(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing timing inputs: BOS + previous gap classes, BOS + the previous note's
        duration class (step k emits note k+1's duration, so its input is note k's)."""
        target = batch["target"]
        B, L = target.shape
        g = torch.full((B, 1), self.bos_gap, dtype=torch.long, device=target.device)
        d = torch.full((B, 1), self.bos_dur, dtype=torch.long, device=target.device)
        gprev = torch.cat([g, batch["rhythm"][:, :-1].clamp(min=0)], 1)
        dur = batch["duration"].clamp(min=0)
        dprev = torch.cat([d, dur[:, 1:L]], 1) if L > 1 else d
        return gprev, dprev

    _last_adv_logits: torch.Tensor | None = None

    def _joint_logits(self, batch, memory, mask, prev, gprev, dprev, aprev=None):
        """Joint-mode logits with the within-step heads conditioned on the true tokens.
        In expansion mode the advance logits of the pass are kept in ``_last_adv_logits``."""
        if self.cfg.expand:
            h = self.decode_hidden(prev, memory, mask, gprev, dprev, aprev, batch["align"])
            self._last_adv_logits = self.out_adv(h)
            iv = self.out_iv_x(torch.cat([h, self.adv_emb(batch["advance"].clamp(min=0))], -1))
        else:
            h = self.decode_hidden(prev, memory, mask, gprev, dprev)
            iv = self.out(h)
        true_iv = batch["target"].clamp(min=0)
        true_gap = batch["rhythm"].clamp(min=0)
        rh = self.out_gap_j(torch.cat([h, self.tok_emb(true_iv)], -1))
        du = self.out_dur_j(torch.cat([h, self.tok_emb(true_iv), self.gap_emb(true_gap)], -1))
        du0 = self.out_dur0(memory[:, :1])
        return iv, rh, torch.cat([du0, du], 1)

    def loss(self, batch: dict[str, torch.Tensor], rhythm_weight: float = 1.0,
             duration_weight: float = 1.0, ss_prob: float = 0.0) -> torch.Tensor:
        """Interval cross-entropy plus, with the timing heads and their targets, ``rhythm_weight``
        times the gap cross-entropy and ``duration_weight`` times the duration cross-entropy."""
        iv, rh, du = self.forward_joint(batch, ss_prob)
        loss = F.cross_entropy(iv.reshape(-1, N_INTERVAL_CLASSES), batch["target"].reshape(-1),
                               ignore_index=PAD_TARGET)
        if rh is not None and "rhythm" in batch:
            L = batch["rhythm"].shape[1]
            loss = loss + rhythm_weight * F.cross_entropy(rh[:, :L].reshape(-1, self.cfg.n_rhythm),
                                                          batch["rhythm"][:, :rh.shape[1]].reshape(-1),
                                                          ignore_index=PAD_TARGET)
        if du is not None and "duration" in batch:
            n = min(du.shape[1], batch["duration"].shape[1])
            loss = loss + duration_weight * F.cross_entropy(du[:, :n].reshape(-1, self.cfg.n_duration),
                                                            batch["duration"][:, :n].reshape(-1),
                                                            ignore_index=PAD_TARGET)
        if self.cfg.expand and "advance" in batch and self._last_adv_logits is not None:
            loss = loss + F.cross_entropy(self._last_adv_logits.reshape(-1, 2), batch["advance"].reshape(-1),
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
                 mode: str = "sample", force_advance: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode interval classes for every sketch in the batch.

        Returns ``(classes (B, S, L), logp (B, S), step_probs (B, S, L, 49))``
        where ``L = n_max - 1`` and ``S = n_samples``.  ``mode`` is ``"greedy"``,
        ``"sample"`` (temperature-scaled) or ``"beam"`` (``n_samples`` beams).
        Padded steps hold class 0 / uniform probabilities.
        """
        self.eval()
        memory, mask = self.encode(batch)
        if self.cfg.expand and not force_advance:
            return self._generate_expand(memory, mask, n_samples, temperature, mode)
        if self.cfg.joint:
            return self._generate_joint(memory, mask, n_samples, temperature, mode)
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
        self._last_rhythm = self._rhythm_classes(memory, S, L, mask)
        self._last_duration = self._duration_classes(memory, mask)
        classes = tokens[:, 1:].view(B, S, L)
        step_probs = torch.stack(probs_all, dim=1).view(B, S, L, N_INTERVAL_CLASSES)
        if mode == "greedy" and n_samples > 1:
            classes = classes.expand(B, n_samples, L)
            step_probs = step_probs.expand(B, n_samples, L, N_INTERVAL_CLASSES)
            logp = logp.view(B, 1).expand(B, n_samples)
        else:
            logp = logp.view(B, S)
        return classes, logp, step_probs

    # -- joint-token decoding ------------------------------------------- #

    def joint_step(self, tokens: torch.Tensor, gtokens: torch.Tensor, dtokens: torch.Tensor,
                   memory: torch.Tensor, mask: torch.Tensor, atokens: torch.Tensor | None = None,
                   align: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One joint decoder step for a batch of prefixes: (interval log-probs (B, 49), state (B, d)).
        In expansion mode the interval log-probs are those for *advancing*; use
        :meth:`expand_step` for both choices."""
        h = self.decode_hidden(tokens, memory, mask, gtokens, dtokens, atokens, align)[:, -1]
        if self.cfg.expand:
            one = torch.ones(h.shape[0], dtype=torch.long, device=h.device)
            return F.log_softmax(self.out_iv_x(torch.cat([h, self.adv_emb(one)], -1)), -1), h
        return F.log_softmax(self.out(h), -1), h

    def expand_step(self, tokens, gtokens, dtokens, atokens, align, memory, mask
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One expansion step: (advance log-probs (B, 2), interval log-probs if staying (B, 49),
        interval log-probs if advancing (B, 49), state (B, d))."""
        h = self.decode_hidden(tokens, memory, mask, gtokens, dtokens, atokens, align)[:, -1]
        adv = F.log_softmax(self.out_adv(h), -1)
        zero = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        stay = F.log_softmax(self.out_iv_x(torch.cat([h, self.adv_emb(zero)], -1)), -1)
        nxt = F.log_softmax(self.out_iv_x(torch.cat([h, self.adv_emb(zero + 1)], -1)), -1)
        return adv, stay, nxt, h

    def joint_timing(self, h: torch.Tensor, interval_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy gap and duration classes given the decoder state and the chosen interval."""
        g = self.out_gap_j(torch.cat([h, self.tok_emb(interval_cls)], -1)).argmax(-1)
        d = self.out_dur_j(torch.cat([h, self.tok_emb(interval_cls), self.gap_emb(g)], -1)).argmax(-1)
        return g, d

    def duration0(self, memory: torch.Tensor) -> torch.Tensor:
        """Duration class of note 0 (B,) in joint mode."""
        return self.out_dur0(memory[:, 0]).argmax(-1)

    def _generate_joint(self, memory, mask, n_samples, temperature, mode):
        """Joint decoding; sets ``_last_rhythm`` (B, S, L) and ``_last_duration`` (B, n) (first
        sample) plus ``_last_duration_all`` (B, S, n)."""
        B, n, _ = memory.shape
        L = max(n - 1, 1)
        dur0 = self.duration0(memory)
        if mode == "beam":
            classes, logp, probs, gaps, durs = self._beam_joint(memory, mask, n_samples)
        else:
            S = n_samples if mode == "sample" else 1
            mem = memory.repeat_interleave(S, 0)
            msk = mask.repeat_interleave(S, 0)
            dev = memory.device
            tokens = torch.full((B * S, 1), BOS, dtype=torch.long, device=dev)
            gtok = torch.full((B * S, 1), self.bos_gap, dtype=torch.long, device=dev)
            dtok = torch.full((B * S, 1), self.bos_dur, dtype=torch.long, device=dev)
            logp = torch.zeros(B * S, device=dev)
            probs_all = []
            for step in range(L):
                at, al = self._forced_tokens(tokens.shape[0], step, n, mem.device) if self.cfg.expand else (None, None)
                lp, h = self.joint_step(tokens, gtok, dtok, mem, msk, at, al)
                p = F.softmax(lp / max(temperature, 1e-4), -1)
                nxt = p.argmax(-1) if mode == "greedy" else torch.multinomial(p, 1).squeeze(-1)
                logp = logp + torch.log(p.gather(1, nxt[:, None]).squeeze(1) + 1e-12)
                probs_all.append(p)
                g, d = self.joint_timing(h, nxt)
                tokens = torch.cat([tokens, nxt[:, None]], 1)
                gtok = torch.cat([gtok, g[:, None]], 1)
                dtok = torch.cat([dtok, d[:, None]], 1)
            classes = tokens[:, 1:].view(B, S, L)
            gaps = gtok[:, 1:].view(B, S, L)
            durs = torch.cat([dur0.repeat_interleave(S, 0)[:, None], dtok[:, 1:]], 1).view(B, S, n)
            probs = torch.stack(probs_all, 1).view(B, S, L, N_INTERVAL_CLASSES)
            if mode == "greedy" and n_samples > 1:
                classes = classes.expand(B, n_samples, L)
                probs = probs.expand(B, n_samples, L, N_INTERVAL_CLASSES)
                logp = logp.view(B, 1).expand(B, n_samples)
                gaps = gaps.expand(B, n_samples, L)
                durs = durs.expand(B, n_samples, n)
            else:
                logp = logp.view(B, S)
        self._last_rhythm = gaps.contiguous()
        self._last_duration_all = durs.contiguous()
        self._last_duration = durs[:, 0].contiguous()
        return classes, logp, probs

    @staticmethod
    def _forced_tokens(B: int, step: int, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Expansion inputs when every step advances (one output note per tap): advance tokens
        BOS, 1, 1, ... and alignment 0, 1, 2, ... for steps 0..step."""
        at = torch.full((B, step + 1), 1, dtype=torch.long, device=device)
        at[:, 0] = 2
        al = torch.arange(step + 1, device=device).clamp(max=n - 1)[None].expand(B, -1).contiguous()
        return at, al

    def _generate_expand(self, memory, mask, n_samples, temperature, mode):
        """Free-length decoding, one sketch at a time: notes are emitted until the decoder
        advances past the last tap (or a cap). Returns tensors padded to the longest
        output, with ``_last_rhythm`` (B, S, L), ``_last_duration_all`` (B, S, L+1) and
        ``_last_align`` (B, S, L+1); padded steps hold -1 in ``_last_align``."""
        B, n, _ = memory.shape
        dev = memory.device
        S = n_samples
        results: list[list[tuple[list[int], list[int], list[int], list[int], float, list[torch.Tensor]]]] = []
        for b in range(B):
            n_b = int(mask[b].sum())
            mem_b, msk_b = memory[b:b + 1, :n_b], mask[b:b + 1, :n_b]
            if mode == "beam":
                hyps = self._beam_expand(mem_b, msk_b, n_b, S)
            else:
                hyps = [self._sample_expand(mem_b, msk_b, n_b, mode == "greedy", temperature) for _ in range(S if mode == "sample" else 1)]
                if mode == "greedy":
                    hyps = hyps * S
            results.append(hyps)
        L = max(max(len(h[0]) for h in hyps) for hyps in results) or 1
        cls = torch.zeros(B, S, L, dtype=torch.long, device=dev)
        gaps = torch.zeros(B, S, L, dtype=torch.long, device=dev)
        durs = torch.zeros(B, S, L + 1, dtype=torch.long, device=dev)
        aligns = torch.full((B, S, L + 1), -1, dtype=torch.long, device=dev)
        lp = torch.full((B, S), -1e9, device=dev)
        pr = torch.full((B, S, L, N_INTERVAL_CLASSES), 1.0 / N_INTERVAL_CLASSES, device=dev)
        dur0 = self.duration0(memory)
        for b, hyps in enumerate(results):
            for s, (seq, gs, ds, al, score, probs) in enumerate(hyps[:S]):
                m = len(seq)
                if m:
                    cls[b, s, :m] = torch.tensor(seq, device=dev)
                    gaps[b, s, :m] = torch.tensor(gs, device=dev)
                    durs[b, s, 1:m + 1] = torch.tensor(ds, device=dev)
                    pr[b, s, :m] = torch.stack(probs)
                durs[b, s, 0] = dur0[b]
                aligns[b, s, :m + 1] = torch.tensor(al, device=dev)
                lp[b, s] = score
        self._last_rhythm = gaps
        self._last_duration_all = durs
        self._last_duration = durs[:, 0]
        self._last_align = aligns
        self._last_classes = cls
        return cls, lp, pr

    def _expand_inputs(self, hyps, dev):
        tok = torch.tensor([[BOS] + h[0] for h in hyps], dtype=torch.long, device=dev)
        gt = torch.tensor([[self.bos_gap] + h[1] for h in hyps], dtype=torch.long, device=dev)
        dt = torch.tensor([[self.bos_dur] + h[2] for h in hyps], dtype=torch.long, device=dev)
        at = torch.tensor([[2] + h[3] for h in hyps], dtype=torch.long, device=dev)
        al = torch.tensor([h[4] for h in hyps], dtype=torch.long, device=dev)
        return tok, gt, dt, at, al

    def _sample_expand(self, mem, msk, n, greedy: bool, temperature: float, max_extra: int = 48):
        """One free-length hypothesis: (intervals, gaps, durations, aligns, logp, probs)."""
        dev = mem.device
        seq, gs, ds, advs, al, probs = [], [], [], [], [0], []
        score = 0.0
        while True:
            tok, gt, dt, at, ali = self._expand_inputs([(seq, gs, ds, advs, al)], dev)
            adv, stay, nxt, h = self.expand_step(tok, gt, dt, at, ali, mem, msk)
            a_now = al[-1]
            if len(seq) >= n - 1 + max_extra:
                adv_choice = 1
            else:
                pa = F.softmax(adv[0] / max(temperature, 1e-4), -1)
                adv_choice = int(pa.argmax()) if greedy else int(torch.multinomial(pa, 1))
            score += float(adv[0, adv_choice])
            if adv_choice == 1 and a_now >= n - 1:
                break  # end of sequence
            row = nxt[0] if adv_choice == 1 else stay[0]
            p = F.softmax(row / max(temperature, 1e-4), -1)
            c = int(p.argmax()) if greedy else int(torch.multinomial(p, 1))
            score += float(row[c])
            g, d = self.joint_timing(h, torch.tensor([c], device=dev))
            seq.append(c); gs.append(int(g)); ds.append(int(d)); advs.append(adv_choice)
            al.append(a_now + adv_choice); probs.append(p)
        return seq, gs, ds, al, score, probs

    def _beam_expand(self, mem, msk, n, beam: int, max_extra: int = 48):
        """Beam search over (advance, interval) choices with free length; finished hypotheses
        are ranked by mean log-prob per step."""
        dev = mem.device
        live = [([], [], [], [], [0], 0.0, [])]  # seq, gaps, durs, advs, aligns, score, probs
        done = []
        while live and len(done) < beam * 4:
            live_src = live
            tok, gt, dt, at, ali = self._expand_inputs([h[:5] for h in live], dev)
            adv, stay, nxt, h = self.expand_step(tok, gt, dt, at, ali, mem.expand(len(live), -1, -1),
                                                 msk.expand(len(live), -1))
            cand = []
            for i, (seq, gs, ds, advs, al, score, probs) in enumerate(live):
                a_now = al[-1]
                over = len(seq) >= n - 1 + max_extra
                for choice in (1, 0):
                    if choice == 0 and over:
                        continue
                    s_adv = score + float(adv[i, choice])
                    if choice == 1 and a_now >= n - 1:
                        done.append((seq, gs, ds, al, s_adv / max(len(seq) + 1, 1), probs))
                        continue
                    row = nxt[i] if choice == 1 else stay[i]
                    top = torch.topk(row, min(beam, N_INTERVAL_CLASSES))
                    for v, c in zip(top.values.tolist(), top.indices.tolist()):
                        cand.append((s_adv + v, i, choice, c, row.exp()))
            cand.sort(key=lambda x: -x[0])
            cand = cand[:beam]
            if not cand:
                break
            hc = h[[c[1] for c in cand]]
            ic = torch.tensor([c[3] for c in cand], dtype=torch.long, device=dev)
            g, d = self.joint_timing(hc, ic)
            live = []
            for (s, i, choice, c, pr), gi, di in zip(cand, g.tolist(), d.tolist()):
                seq, gs, ds, advs, al, _, probs = live_src[i]
                live.append((seq + [c], gs + [gi], ds + [di], advs + [choice], al + [al[-1] + choice], s, probs + [pr]))
        done.sort(key=lambda x: -x[4])
        return done[:beam] if done else [([], [], [], [0], 0.0, [])]

    def _beam_joint(self, memory, mask, beam):
        B, n_max, _ = memory.shape
        L = max(n_max - 1, 1)
        dev = memory.device
        dur0 = self.duration0(memory)
        out_cls = torch.zeros(B, beam, L, dtype=torch.long, device=dev)
        out_gap = torch.zeros(B, beam, L, dtype=torch.long, device=dev)
        out_dur = torch.zeros(B, beam, n_max, dtype=torch.long, device=dev)
        out_lp = torch.full((B, beam), -1e9, device=dev)
        out_pr = torch.full((B, beam, L, N_INTERVAL_CLASSES), 1.0 / N_INTERVAL_CLASSES, device=dev)
        for b in range(B):
            n_b = int(mask[b].sum())
            steps = max(n_b - 1, 0)
            mem_b, msk_b = memory[b:b + 1, :n_b], mask[b:b + 1, :n_b]
            # (score, intervals, gaps, durations, probs)
            beams: list[tuple[float, list[int], list[int], list[int], list[torch.Tensor]]] = [(0.0, [], [], [], [])]
            for step in range(steps):
                tok = torch.tensor([[BOS] + s[1] for s in beams], dtype=torch.long, device=dev)
                gt = torch.tensor([[self.bos_gap] + s[2] for s in beams], dtype=torch.long, device=dev)
                dt = torch.tensor([[self.bos_dur] + s[3] for s in beams], dtype=torch.long, device=dev)
                at, al = self._forced_tokens(len(beams), step, n_b, dev) if self.cfg.expand else (None, None)
                lp, h = self.joint_step(tok, gt, dt, mem_b.expand(len(beams), -1, -1), msk_b.expand(len(beams), -1), at, al)
                cand = []
                for bi, ((s, seq, gs, ds, pr), row) in enumerate(zip(beams, lp)):
                    top = torch.topk(row, min(beam, N_INTERVAL_CLASSES))
                    for v, c in zip(top.values.tolist(), top.indices.tolist()):
                        cand.append((s + v, seq + [c], gs, ds, pr + [row.exp()], bi))
                cand.sort(key=lambda x: -x[0])
                cand = cand[:beam]
                hc = h[[c[5] for c in cand]]
                ic = torch.tensor([c[1][-1] for c in cand], dtype=torch.long, device=dev)
                g, d = self.joint_timing(hc, ic)
                beams = [(s, seq, gs + [int(gi)], ds + [int(di)], pr)
                         for (s, seq, gs, ds, pr, _), gi, di in zip(cand, g.tolist(), d.tolist())]
            for j, (s, seq, gs, ds, pr) in enumerate(beams[:beam]):
                if seq:
                    out_cls[b, j, :len(seq)] = torch.tensor(seq, device=dev)
                    out_gap[b, j, :len(gs)] = torch.tensor(gs, device=dev)
                    out_dur[b, j, 1:1 + len(ds)] = torch.tensor(ds, device=dev)
                    out_pr[b, j, :len(seq)] = torch.stack(pr)
                out_dur[b, j, 0] = dur0[b]
                out_lp[b, j] = s
        return out_cls, out_lp, out_pr, out_gap, out_dur

    def _duration_classes(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Greedy duration classes (B, n) from the duration head, or None without one."""
        dl = self.duration_logits(memory, mask)
        return None if dl is None else dl.argmax(-1)

    def _rhythm_classes(self, memory: torch.Tensor, S: int, L: int,
                        mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Greedy gap classes (B, S, L) from the rhythm branch, or None without one."""
        rl = self.rhythm_logits(memory, mask)
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
        self._last_rhythm = self._rhythm_classes(memory, beam, L, mask)
        self._last_duration = self._duration_classes(memory, mask)
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
    def rhythm_from_classes(self, classes, onsets_s: list[float],
                            duration_classes=None) -> tuple[list[float], list[float], float]:
        """Turn predicted gap classes (and duration classes, if given) into
        (onsets_beats, durations_beats, tempo).

        Without duration classes every note lasts until the next onset (the last
        one a median gap); with them each lasts its class's share of that gap. The
        tempo is the fit of predicted beats to tapped seconds.
        """
        gaps = [RHYTHM_CLASSES[int(c)] for c in classes]
        onsets = [0.0]
        for g in gaps:
            onsets.append(onsets[-1] + g)
        span_s = (onsets_s[-1] - onsets_s[0]) if len(onsets_s) > 1 else 0.0
        tempo = 60.0 * onsets[-1] / span_s if span_s > 1e-3 and onsets[-1] > 0 else 120.0
        tempo = float(min(240.0, max(40.0, tempo)))
        durations = gaps + [float(np.median(gaps)) if gaps else 1.0]
        if duration_classes is not None and len(duration_classes) >= len(durations):
            durations = [round(class_to_ratio(int(c)) * g, 4) for c, g in zip(duration_classes, durations)]
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
