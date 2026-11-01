"""Compact students: one architecture family across the whole size ladder, so "retention vs size" means one thing.

A dilated residual CNN over single-nucleotide tokens. The shape follows the supervised baseline prior art uses for these tasks
(BIO-Distiller: a tuned three-layer CNN), widened and deepened to make a ladder; Optimus 5-Prime, the task's original model, is
also a small CNN, so this is the architecture the task is known to suit.

Every forward returns the scalar prediction, the pooled penultimate vector (the feature distillation matches on this) and the
per-token map, so one model serves every arm.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .teacher import ALPHABET_SIZE, PAD

PRESETS = {                                                                     # name: (channels, blocks, kernel)
    "nano":   (32, 3, 5),
    "tiny":   (64, 3, 5),
    "small":  (128, 4, 5),
    "medium": (256, 4, 5),
    "large":  (384, 6, 5),
}


class Res(nn.Module):
    def __init__(self, c, k, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding="same", dilation=dilation), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, k, padding="same", dilation=dilation), nn.BatchNorm1d(c))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Student(nn.Module):
    def __init__(self, preset="tiny", dropout=0.2):
        super().__init__()
        c, n, k = PRESETS[preset]
        self.preset, self.dim = preset, c
        self.embed = nn.Embedding(ALPHABET_SIZE, c, padding_idx=PAD)
        self.blocks = nn.ModuleList([Res(c, k, 2 ** min(i, 3)) for i in range(n)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(c, 1)

    def forward(self, tokens):
        pad = tokens.eq(PAD)
        x = self.embed(tokens).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        x = x.transpose(1, 2).masked_fill(pad.unsqueeze(-1), 0.0)
        feat = x.sum(-2) / (~pad).sum(-1).clamp(min=1)[:, None]
        return self.out(self.dropout(feat)).squeeze(-1), feat, x

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build(preset: str, dropout: float = 0.2) -> Student:
    return Student(preset, dropout)
