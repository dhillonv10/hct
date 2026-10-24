"""LncFM: a small hierarchical nucleotide-level encoder for (l)ncRNA transcripts.

    nt embedding -> 2 conv blocks (nt resolution) -> stride-4 downsample
                 -> N pre-LN transformer layers with RoPE (1/4 resolution)
                 -> upsample + skip -> conv block -> heads

Why this shape (see PLAN.md): lncRNAs are short (median ~0.8 kb) but repeat-rich and
low-complexity, so we keep single-nucleotide input (no tokenizer that destroys repeat/motif
boundaries) yet pay attention cost only at 1/4 length. Pure PyTorch, no custom kernels, so it
runs unchanged on CPU, Apple MPS and CUDA.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, MASK, NT0, UNK = 0, 1, 2, 6      # A,C,G,U = 2..5 ; N/other = 6
VOCAB = 7

_LUT = np.full(256, UNK, dtype=np.uint8)
for _i, _c in enumerate("ACGU"):
    _LUT[ord(_c)] = NT0 + _i
_LUT[ord("T")] = NT0 + 3


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


@dataclass
class LncFMConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    d_nt: int = 64
    stride: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    max_len: int = 1024       # nucleotides per window (multiple of stride)

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "tiny":  dict(d_model=128, n_layers=4,  n_heads=4, d_nt=32),   # ~1M   CPU pilots
    "small": dict(d_model=256, n_layers=8,  n_heads=4, d_nt=64),   # ~7M   M4 first real run
    "base":  dict(d_model=512, n_layers=12, n_heads=8, d_nt=96),   # ~40M  M4 stretch / Colab
}


class ConvBlock(nn.Module):
    """Residual conv block. Normalisation is per-position over channels (LayerNorm), NOT over the
    length axis: a GroupNorm/BatchNorm over (C, L) would mix in padded positions and make an
    embedding depend on what else is in the batch (verified failure, see selfcheck.py)."""

    def __init__(self, c, k=9):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.conv = nn.Conv1d(c, c, k, padding=k // 2)
        self.pw = nn.Conv1d(c, c, 1)

    def forward(self, x, keep):             # x: (B,C,L)  keep: (B,1,L) float
        h = self.norm(x.transpose(1, 2)).transpose(1, 2) * keep
        h = self.pw(F.gelu(self.conv(h)))
        return (x + h) * keep


def _rope(x, cos, sin):                      # x: (B,H,T,Dh)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.p = p

    def forward(self, x, cos, sin, key_ok):  # key_ok: (B,1,1,T) bool
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok, dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class Layer(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h, p)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, cos, sin, key_ok):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, key_ok))
        return x + self.drop(self.ff(self.n2(x)))


class LncFM(nn.Module):
    def __init__(self, cfg: LncFMConfig):
        super().__init__()
        self.cfg = cfg
        d, dn, s = cfg.d_model, cfg.d_nt, cfg.stride
        self.emb = nn.Embedding(VOCAB, dn, padding_idx=PAD)
        self.conv_in = nn.ModuleList([ConvBlock(dn), ConvBlock(dn)])
        self.down = nn.Conv1d(dn, d, s, stride=s)
        self.layers = nn.ModuleList([Layer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.up = nn.ConvTranspose1d(d, dn, s, stride=s)
        self.conv_out = ConvBlock(dn)
        self.lm_head = nn.Linear(dn, VOCAB)
        self.junction_head = nn.Linear(dn, 1)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))   # contrastive projection
        dh = d // cfg.n_heads
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        pos = torch.arange(cfg.max_len // s + 8).float()
        ang = torch.outer(pos, inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, return_hidden: bool = False):
        """ids: (B,L) long, L % stride == 0. Returns dict with nt_logits, junction_logits, pooled, [hiddens]."""
        B, L = ids.shape
        s = self.cfg.stride
        keep = (ids != PAD).float().unsqueeze(1)                        # (B,1,L)
        x = self.emb(ids).transpose(1, 2) * keep                         # (B,dn,L)
        for blk in self.conv_in:
            x = blk(x, keep)
        skip = x
        h = self.down(x).transpose(1, 2)                                 # (B,T,d)
        T = h.shape[1]
        tok_ok = (F.max_pool1d(keep, s) > 0).squeeze(1)                  # (B,T) bool
        key_ok = tok_ok[:, None, None, :]
        cos, sin = self.cos[:T], self.sin[:T]
        hiddens = []
        for lyr in self.layers:
            h = lyr(h, cos, sin, key_ok)
            if return_hidden:
                hiddens.append(h)
        h = self.norm(h)
        w = tok_ok.unsqueeze(-1).float()
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        u = self.up(h.transpose(1, 2)) + skip                            # (B,dn,L)
        u = self.conv_out(u, keep)
        u = u.transpose(1, 2)                                            # (B,L,dn)
        out = {"nt_logits": self.lm_head(u), "junction_logits": self.junction_head(u).squeeze(-1),
               "pooled": pooled, "tok_ok": tok_ok}
        if return_hidden:
            out["hiddens"] = hiddens
        return out

    def project(self, pooled):
        return F.normalize(self.proj(pooled), dim=-1)


def build(preset: str = "small", **over) -> LncFM:
    kw = dict(PRESETS[preset]); kw.update(over)
    return LncFM(LncFMConfig(**kw))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
