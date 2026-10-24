"""EpiSet: a set-transformer over (CpG, beta) tokens with an optional mixture-of-experts FFN.

One model serves every input format of the LongevityBench DNAm tasks, because every format is a
*set of CpGs with values*: the 100 lowest + 100 highest methylated sites, a fixed 353-site panel, a
clock's own sites, or the sites that differ most between two donors. The format is told to the model
through a role embedding, not through a different architecture.

Token  = id(cpg) + role + dir(cpg) * z(beta) + phi(beta)
         ^ identity   ^ format  ^ CpG-specific slope   ^ nonlinear value response
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..age import dage_dt, t_to_age

ROLES = {"low": 0, "high": 1, "fixed": 2, "clock": 3, "pair_a": 4, "pair_b": 5}
N_DECADES = 11  # 0-9 ... 90-99, 100+


@dataclass
class SetConfig:
    n_tokens_vocab: int = 40000      # CpG vocabulary size (a pad/unknown id is appended)
    d_model: int = 96
    d_emb: int = 64
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    n_fourier: int = 8
    n_clocks: int = 12
    moe_experts: int = 0             # 0 = dense FFN; >0 = sparse mixture of experts
    moe_top_k: int = 2
    moe_balance: float = 0.01
    moe_level: str = "token"          # "token": router per token; "sample": one routing decision per sample from the pooled set
    linear_route: bool = False       # add an explicit per-CpG linear term sum_i w_i z_i (extrapolates; cannot memorise cohorts)


class MoEFFN(nn.Module):
    """Token-level sparse MoE: a linear router picks ``top_k`` experts per token; outputs are mixed by
    the renormalised router weights. Experts are computed densely (E is small), gates are sparse.
    ``aux`` is the Switch-style load-balancing loss (E * sum_e fraction_e * mean_prob_e)."""

    def __init__(self, d: int, ff: int, n_experts: int, top_k: int, dropout: float, sample_level: bool = False):
        super().__init__()
        self.E, self.k, self.sample_level = n_experts, top_k, sample_level
        self.sparse_compute = True
        self.router = nn.Linear(d, n_experts)
        self.w1 = nn.Parameter(torch.randn(n_experts, d, ff) * d**-0.5)
        self.b1 = nn.Parameter(torch.zeros(n_experts, ff))
        self.w2 = nn.Parameter(torch.randn(n_experts, ff, d) * ff**-0.5)
        self.b2 = nn.Parameter(torch.zeros(n_experts, d))
        self.drop = nn.Dropout(dropout)
        self.aux = torch.zeros(())
        self.last_load: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        live = torch.ones(x.shape[:2], device=x.device) if pad is None else (~pad).float()
        if self.sample_level:       # one decision per sample from the permutation-invariant mean of its live tokens
            pooled = (x * live.unsqueeze(-1)).sum(1) / live.sum(1, keepdim=True).clamp_min(1.0)
            probs = torch.softmax(self.router(pooled), -1).unsqueeze(1).expand(-1, x.shape[1], -1)      # [B,T,E], constant over T
        else:
            probs = torch.softmax(self.router(x), -1)                               # [B,T,E]
        topv, topi = probs.topk(self.k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, topi, topv / topv.sum(-1, keepdim=True))
        if self.sample_level and self.sparse_compute:      # whole samples are routed: run each expert only on its samples
            g = gates[:, 0, :]
            out = torch.zeros_like(x)
            for e in range(self.E):
                idx = (g[:, e] > 0).nonzero(as_tuple=True)[0]
                if idx.numel():
                    y = self.drop(torch.nn.functional.gelu(x[idx] @ self.w1[e] + self.b1[e])) @ self.w2[e] + self.b2[e]
                    out = out.index_add(0, idx, g[idx, e].view(-1, 1, 1) * y)
        else:
            h = torch.nn.functional.gelu(torch.einsum("btd,edf->btef", x, self.w1) + self.b1)
            y = torch.einsum("btef,efd->bted", self.drop(h), self.w2) + self.b2
            out = (gates.unsqueeze(-1) * y).sum(2)
        n = live.sum().clamp_min(1.0)
        frac = ((gates > 0).float() * live.unsqueeze(-1)).sum((0, 1)) / (n * self.k)
        mean_p = (probs * live.unsqueeze(-1)).sum((0, 1)) / n
        self.aux = self.E * (frac * mean_p).sum()
        self.last_load = frac.detach()
        return out


class Block(nn.Module):
    def __init__(self, cfg: SetConfig):
        super().__init__()
        d = cfg.d_model
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        if cfg.moe_experts > 0:
            self.ffn = MoEFFN(d, cfg.ff_mult * d, cfg.moe_experts, cfg.moe_top_k, cfg.dropout, cfg.moe_level == "sample")
        else:
            self.ffn = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ff_mult * d, d))
        self.moe = cfg.moe_experts > 0
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pad):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0])
        h = self.n2(x)
        return x + self.drop(self.ffn(h, pad) if self.moe else self.ffn(h))


class EpiSet(nn.Module):
    def __init__(self, cfg: SetConfig, cpg_mean: np.ndarray | None = None, cpg_std: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        V, d, de = cfg.n_tokens_vocab, cfg.d_model, cfg.d_emb
        self.pad_id = V
        self.id_emb = nn.Embedding(V + 1, de, padding_idx=V)
        self.dir_emb = nn.Embedding(V + 1, de, padding_idx=V)
        nn.init.normal_(self.id_emb.weight, std=0.1); nn.init.normal_(self.dir_emb.weight, std=0.1)
        with torch.no_grad():
            self.id_emb.weight[V].zero_(); self.dir_emb.weight[V].zero_()
        self.id_proj, self.dir_proj = nn.Linear(de, d), nn.Linear(de, d)
        self.role_emb = nn.Embedding(len(ROLES), d)
        self.clock_emb = nn.Embedding(cfg.n_clocks, d)
        self.sex_emb = nn.Embedding(3, d)                      # 0 unknown, 1 male, 2 female
        self.register_buffer("freqs", 2.0 ** torch.arange(cfg.n_fourier) * math.pi, persistent=True)
        self.val_mlp = nn.Sequential(nn.Linear(2 * cfg.n_fourier + 1, d), nn.GELU(), nn.Linear(d, d))
        m = np.full(V + 1, 0.5, np.float32) if cpg_mean is None else np.append(cpg_mean, 0.5).astype(np.float32)
        s = np.full(V + 1, 0.25, np.float32) if cpg_std is None else np.append(np.maximum(cpg_std, 0.02), 0.25).astype(np.float32)
        self.register_buffer("cpg_mean", torch.as_tensor(m)); self.register_buffer("cpg_std", torch.as_tensor(s))
        if cfg.linear_route:
            self.lin_w = nn.Embedding(V + 1, 1, padding_idx=V); nn.init.zeros_(self.lin_w.weight)
            self.lin_gain = nn.Parameter(torch.ones(len(ROLES)))      # per-view gain on the shared linear weights
            self.lin_bias = nn.Embedding(len(ROLES), 1); nn.init.zeros_(self.lin_bias.weight)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.age_head = nn.Linear(d, 2)                        # mu_t, raw sigma
        self.decade_head = nn.Linear(d, N_DECADES)
        self.register_buffer("sigma_scale", torch.ones(1))

    # ---------------------------------------------------------------------------------------
    def embed(self, cpg, beta, role):
        pad = cpg < 0
        ids = torch.where(pad, torch.full_like(cpg, self.pad_id), cpg)
        beta = torch.nan_to_num(beta, nan=0.0)
        z = ((beta - self.cpg_mean[ids]) / self.cpg_std[ids]).clamp(-8, 8)
        ang = beta.unsqueeze(-1) * self.freqs
        phi = self.val_mlp(torch.cat([torch.sin(ang), torch.cos(ang), beta.unsqueeze(-1)], -1))
        x = self.id_proj(self.id_emb(ids)) + self.dir_proj(self.dir_emb(ids)) * z.unsqueeze(-1) + phi + self.role_emb(role)
        return x * (~pad).unsqueeze(-1), pad, z * (~pad)

    def linear_term(self, cpg, role, pad, z):
        """sum_i w[cpg_i] * z_i * gain[view] + bias[view] -- a ridge-like linear model in the standardised betas."""
        ids = torch.where(pad, torch.full_like(cpg, self.pad_id), cpg)
        w = self.lin_w(ids).squeeze(-1) * z * (~pad)
        return (w * self.lin_gain[role]).sum(1) + self.lin_bias(role[:, 0]).squeeze(-1)

    def forward(self, cpg, beta, role, sex, clock=None):
        """cpg [B,T] long (<0 = padding), beta [B,T], role [B,T] long, sex [B] long, clock [B] long or None."""
        x, pad, z = self.embed(cpg, beta, role)
        B = x.shape[0]
        head = self.cls.expand(B, -1) + self.sex_emb(sex)
        if clock is not None:
            head = head + self.clock_emb(clock)
        x = torch.cat([head.unsqueeze(1), x], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for blk in self.blocks:
            x = blk(x, pad)
        h = self.norm(x[:, 0])
        mu, raw = self.age_head(h).unbind(-1)
        if self.cfg.linear_route:
            mu = mu + self.linear_term(cpg, role, pad[:, 1:], z)
        sigma = nn.functional.softplus(raw - 1.0) + 1e-2
        return mu, sigma, self.decade_head(h)

    def moe_aux(self) -> torch.Tensor:
        aux = [b.ffn.aux for b in self.blocks if b.moe]
        return torch.stack(aux).mean() if aux else torch.zeros(())

    # ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def predict_all(self, cpg, beta, role, sex, clock=None, batch_size: int = 128):
        """(mu_t, sigma_t scaled, decade logits) as numpy, in batches."""
        self.eval()
        mus, sigs, decs = [], [], []
        for i in range(0, len(cpg), batch_size):
            sl = slice(i, i + batch_size)
            mu, sig, dec = self(cpg[sl], beta[sl], role[sl], sex[sl], None if clock is None else clock[sl])
            mus.append(mu); sigs.append(sig * self.sigma_scale); decs.append(dec)
        return torch.cat(mus).double().numpy(), torch.cat(sigs).double().numpy(), torch.cat(decs).numpy()

    @torch.no_grad()
    def predict_age(self, cpg, beta, role, sex, clock=None, batch_size: int = 128):
        self.eval()
        out_mu, out_sig = [], []
        for i in range(0, len(cpg), batch_size):
            sl = slice(i, i + batch_size)
            mu, sig, _ = self(cpg[sl], beta[sl], role[sl], sex[sl], None if clock is None else clock[sl])
            out_mu.append(mu); out_sig.append(sig * self.sigma_scale)
        mu, sig = torch.cat(out_mu).double().numpy(), torch.cat(out_sig).double().numpy()
        return t_to_age(mu), sig * dage_dt(mu), mu
