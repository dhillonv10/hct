"""EpiTrajectory-L: a hierarchical, genome-block transformer over sparse visible CpG tokens, with a latent-ODE
dynamics head and a masked-CpG reconstruction decoder.

Adapted from the EpiTrajectory-XL idea (local attention inside genome blocks -> pooled block tokens -> global attention ->
latent state -> {age, future methylome, direction of time}) to what runs on a CPU:
  * the encoder sees only the *visible, observed* CpGs of a sample (any subset: 200 extreme sites, a 353-site panel,
    a clock's sites, or a dense profile), grouped by genomic block, so one network serves every input format;
  * masked CpGs are not encoded (MAE-style); a light decoder predicts them from the block context;
  * the latent state h is moved forward in time by a neural ODE (RK4) and read out as per-CpG slopes added to the earlier
    profile, so the forecast is "persistence + learned drift" by construction.
Interface-compatible with EpiSet (forward -> mu, sigma, decade logits) so the same trainer and evaluators apply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from .episet import Block, EpiSet, SetConfig


@dataclass
class HierConfig(SetConfig):
    local_layers: int = 2
    global_layers: int = 3
    block_size: int = 128
    k_cap: int = 128                 # max tokens per block per sample (sparse formats use far fewer)
    ode_steps: int = 4
    max_slope: float = 0.01          # cap on the per-year change of a CpG in the forecast head


def block_assignment(cpg_ids, annotation, block_size: int = 128) -> np.ndarray:
    """Genome-ordered block id for each CpG (consecutive ``block_size`` CpGs of the model vocabulary share a block)."""
    ann = annotation.reindex(cpg_ids)
    key = ann.chrom.to_numpy(float) * 1e10 + ann.pos.to_numpy(float)
    key = np.where(np.isfinite(key), key, np.nanmax(key) + 1)
    rank = np.empty(len(key), np.int64); rank[np.argsort(key, kind="stable")] = np.arange(len(key))
    return rank // block_size


class LatentVectorField(nn.Module):
    """dh/dt = f(h, t); zero-initialised last layer so the untrained flow is the identity."""

    def __init__(self, d: int, n_freq: int = 8):
        super().__init__()
        self.register_buffer("freq", torch.logspace(-2, 1, n_freq))
        self.net = nn.Sequential(nn.LayerNorm(d + 2 * n_freq), nn.Linear(d + 2 * n_freq, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, t):
        ph = t.unsqueeze(-1) * self.freq
        return self.net(torch.cat([h, torch.sin(ph), torch.cos(ph)], -1))


class EpiHier(EpiSet):
    def __init__(self, cfg: HierConfig, block_of: np.ndarray, cpg_mean=None, cpg_std=None):
        super().__init__(replace(cfg, n_layers=0), cpg_mean, cpg_std)     # token embeddings, heads, buffers; no flat blocks
        self.cfg = cfg
        d = cfg.d_model
        self.nb = int(block_of.max()) + 1
        self.register_buffer("block_of", torch.as_tensor(np.append(block_of, self.nb), dtype=torch.long))      # pad id -> nb
        self.local = nn.ModuleList([Block(cfg) for _ in range(cfg.local_layers)])
        self.glob = nn.ModuleList([Block(cfg) for _ in range(cfg.global_layers)])
        self.pool_score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.block_emb = nn.Embedding(self.nb, d)
        self.recon = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.vfield = LatentVectorField(d)
        self.future_ctx = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        nn.init.zeros_(self.future_ctx[-1].weight); nn.init.zeros_(self.future_ctx[-1].bias)
        self.future_bias = nn.Parameter(torch.zeros(cfg.n_tokens_vocab))
        self.dir_head = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))

    # ------------------------------------------------------------------------------------------ encoder
    def encode(self, cpg, beta, role, sex, clock=None):
        """-> (h [B,d] latent state, block states [B,nb,d], non-empty-block mask [B,nb])."""
        x, pad, _ = self.embed(cpg, beta, role)
        B, T, d = x.shape
        dev = x.device
        ids = torch.where(pad, torch.full_like(cpg, self.pad_id), cpg)
        blk = self.block_of[ids]
        ar = torch.arange(T, device=dev)
        order = (blk * T + ar).argsort(1)
        bs = blk.gather(1, order)
        arr = ar.expand(B, T)
        new = torch.ones_like(bs, dtype=torch.bool); new[:, 1:] = bs[:, 1:] != bs[:, :-1]
        slot = arr - torch.where(new, arr, torch.zeros_like(arr)).cummax(1).values        # position inside its block
        valid = bs < self.nb
        K = max(1, min(self.cfg.k_cap, int(slot[valid].max()) + 1)) if bool(valid.any()) else 1
        keep = valid & (slot < K)
        xs = x.gather(1, order.unsqueeze(-1).expand(-1, -1, d))
        flat = (torch.arange(B, device=dev).unsqueeze(1) * self.nb + bs.clamp(max=self.nb - 1)) * K + slot.clamp(max=K - 1)
        grid = x.new_zeros(B * self.nb * K, d).index_copy(0, flat[keep], xs[keep]).view(B, self.nb, K, d)
        gpad = torch.ones(B * self.nb * K, dtype=torch.bool, device=dev); gpad[flat[keep]] = False
        gpad = gpad.view(B, self.nb, K)
        sel = ~gpad.all(-1)                                                     # blocks that contain at least one visible token
        seq, spad = grid[sel], gpad[sel]
        for layer in self.local:
            seq = layer(seq, spad)
        w = self.pool_score(seq).squeeze(-1).masked_fill(spad, -1e4).softmax(-1)
        pooled = (seq * w.unsqueeze(-1)).sum(1) + self.block_emb(torch.arange(self.nb, device=dev).expand(B, -1)[sel])
        blocks = x.new_zeros(B, self.nb, d); blocks[sel] = pooled
        head = self.cls.expand(B, -1) + self.sex_emb(sex)
        if clock is not None:
            head = head + self.clock_emb(clock)
        g = torch.cat([head.unsqueeze(1), blocks], 1)
        gmask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ~sel], 1)
        for layer in self.glob:
            g = layer(g, gmask)
        return self.norm(g[:, 0]), g[:, 1:], sel

    def forward(self, cpg, beta, role, sex, clock=None):
        h, _, _ = self.encode(cpg, beta, role, sex, clock)
        mu, raw = self.age_head(h).unbind(-1)
        return mu, nn.functional.softplus(raw - 1.0) + 1e-2, self.decade_head(h)

    # ------------------------------------------------------------------------------------------ masked reconstruction
    def reconstruct(self, h, gblocks, tgt):
        """beta of masked CpGs ``tgt`` [B,Q] (vocabulary ids) from their identity, block context and the sample state."""
        ids = tgt.clamp(min=0)
        e = self.id_proj(self.id_emb(ids))
        gb = gblocks.gather(1, self.block_of[ids].clamp(max=self.nb - 1).unsqueeze(-1).expand(-1, -1, gblocks.shape[-1]))
        return torch.sigmoid(self.recon(torch.cat([e, gb, h.unsqueeze(1).expand(-1, tgt.shape[1], -1)], -1)).squeeze(-1))

    # ------------------------------------------------------------------------------------------ dynamics
    def integrate(self, h, dt):
        """Differentiable RK4 of dh/dt = f(h, t) over ``dt`` years (time rescaled by 1/10)."""
        s = (dt / 10.0) / self.cfg.ode_steps
        t = torch.zeros_like(dt); state = h
        for _ in range(self.cfg.ode_steps):
            k1 = self.vfield(state, t)
            k2 = self.vfield(state + 0.5 * s[:, None] * k1, t + 0.5 * s)
            k3 = self.vfield(state + 0.5 * s[:, None] * k2, t + 0.5 * s)
            k4 = self.vfield(state + s[:, None] * k3, t + s)
            state = state + (s[:, None] / 6) * (k1 + 2 * k2 + 2 * k3 + k4); t = t + s
        return state

    def forecast(self, h, dt, x0, probe):
        """beta of CpGs ``probe`` [Q] after ``dt`` years: x0 + dt * slope(h(dt))."""
        hf = self.integrate(h, dt)
        e = self.id_proj(self.id_emb(probe))
        slope = self.cfg.max_slope * torch.tanh(self.future_ctx(hf) @ e.T / math.sqrt(e.shape[-1]) + self.future_bias[probe])
        return hf, torch.clamp(x0[:, probe] + dt[:, None] * slope, 0.0, 1.0)

    def direction_logit(self, h1, h2):
        return self.dir_head(torch.cat([h1, h2, h2 - h1, (h2 - h1).abs()], -1)).squeeze(-1)


# ---------------------------------------------------------------------------------------------- batches
def invert_vocab(u2m: np.ndarray, n_vocab: int) -> np.ndarray:
    """vocabulary id -> universe column (-1 if absent)."""
    v2u = np.full(n_vocab, -1, np.int64)
    cols = np.flatnonzero(u2m >= 0)
    v2u[u2m[cols]] = cols
    return v2u


def profile_matrix(prof: np.ndarray, rows: np.ndarray, v2u: np.ndarray) -> np.ndarray:
    """float32 [len(rows), V] beta over the model vocabulary with NaN where missing."""
    cols = np.where(v2u >= 0, v2u, 0)
    x = np.asarray(prof[rows])[:, cols].astype(np.float32)
    x[(x == 255) | (v2u < 0)[None, :]] = np.nan
    return x / 100.0


def mask_batch(x: np.ndarray, rng: np.random.RandomState, vis_frac: float, n_target: int = 256, max_vis: int = 1500):
    """Random visible subset (``vis_frac`` of the observed CpGs) and ``n_target`` masked observed CpGs to reconstruct.
    Returns (vis_cpg [B,Tv] with -1 pad, vis_beta, tgt_cpg [B,Q] with -1 pad, tgt_beta)."""
    obs = np.isfinite(x)
    u = rng.rand(*x.shape)
    vis = obs & (u < vis_frac)
    # cap the visible count so tensors stay small
    tv = int(min(max_vis, max(1, vis.sum(1).max())))
    key = np.where(vis, u, 2.0)
    vi = np.argsort(key, axis=1)[:, :tv]
    vis_ok = np.take_along_axis(vis, vi, 1)
    tkey = np.where(obs & ~vis, rng.rand(*x.shape), 2.0)
    ti = np.argsort(tkey, axis=1)[:, :n_target]
    tgt_ok = np.take_along_axis(obs & ~vis, ti, 1)
    return (np.where(vis_ok, vi, -1), np.where(vis_ok, np.take_along_axis(x, vi, 1), 0.0).astype(np.float32),
            np.where(tgt_ok, ti, -1), np.where(tgt_ok, np.take_along_axis(x, ti, 1), 0.0).astype(np.float32))
