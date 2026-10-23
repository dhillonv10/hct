"""CpG tokenisation from genomic context.

A CpG token is built from *where the site is and what it sits in* (chromosome, multi-scale genomic
position, CpG-island relation, gene-region flags, regulatory class, enhancer / DHS, probe design)
plus an optional learned identity vector that is randomly hidden during training. Because the
context path alone carries information, a CpG that was never seen in training (e.g. an EPIC-only
probe) can still be given a meaningful token instead of an untrained id embedding.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..data.annotation import CHROMS, GENE_GROUPS, ISLAND, REG_GROUPS
from ..data.store import CpGTable


def genomic_position_features(pos: np.ndarray, n_scales: int, lo_bp: float = 1e3, hi_bp: float = 1e8) -> np.ndarray:
    """sin/cos of 2*pi*pos/period for log-spaced periods; nearby CpGs share coarse-scale phase."""
    periods = np.logspace(math.log10(lo_bp), math.log10(hi_bp), n_scales)
    ang = 2.0 * np.pi * pos.astype(np.float64)[:, None] / periods[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


class CpGContextEncoder(nn.Module):
    """Maps the (static) CpG table to token embeddings ``e`` of shape ``[N, d_model]``."""

    def __init__(self, cpgs: CpGTable, d_model: int, n_pos_scales: int = 14, use_genomic_context: bool = True,
                 use_id_embedding: bool = True, id_dropout: float = 0.25):
        super().__init__()
        N = len(cpgs.ids)
        self.n, self.d = N, d_model
        self.use_ctx, self.use_id, self.id_dropout = use_genomic_context, use_id_embedding, id_dropout
        self.register_buffer("chrom", torch.as_tensor(cpgs.chrom, dtype=torch.long), persistent=True)
        self.register_buffer("island", torch.as_tensor(cpgs.island, dtype=torch.long), persistent=True)
        self.register_buffer("reg", torch.as_tensor(cpgs.reg, dtype=torch.long), persistent=True)
        self.register_buffer("binary", torch.as_tensor(cpgs.binary, dtype=torch.float32), persistent=True)
        self.register_buffer("posfeat", torch.as_tensor(genomic_position_features(cpgs.pos, n_pos_scales)), persistent=True)
        if use_genomic_context:
            self.chrom_emb = nn.Embedding(len(CHROMS), d_model)
            self.island_emb = nn.Embedding(len(ISLAND), d_model)
            self.reg_emb = nn.Embedding(len(REG_GROUPS), d_model)
            self.bin_proj = nn.Linear(len(GENE_GROUPS) + 3, d_model)
            self.pos_proj = nn.Linear(2 * n_pos_scales, d_model)
        if use_id_embedding or not use_genomic_context:
            self.id_emb = nn.Embedding(N, d_model)
            nn.init.normal_(self.id_emb.weight, std=0.1)
        self.mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self) -> torch.Tensor:
        e = torch.zeros(self.n, self.d, device=self.chrom.device)
        if self.use_ctx:
            e = (
                self.chrom_emb(self.chrom) + self.island_emb(self.island) + self.reg_emb(self.reg)
                + self.bin_proj(self.binary) + self.pos_proj(self.posfeat)
            )
        if hasattr(self, "id_emb"):
            ide = self.id_emb.weight
            if self.training and self.id_dropout > 0 and self.use_ctx:
                keep = (torch.rand(self.n, 1, device=ide.device) > self.id_dropout).float()
                ide = ide * keep
            e = e + ide
        return e + self.mlp(e)
