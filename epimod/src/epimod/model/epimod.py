"""EpiMod: genome-aware, module-decomposed transformer for DNA-methylation age."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..age import dage_dt, t_to_age
from ..config import ModelConfig
from ..data.store import CpGTable
from .modules import ModulePooling
from .tokens import CpGContextEncoder


@dataclass
class EpiModOutput:
    mu_t: torch.Tensor                 # [B]   age in transformed space
    sigma_t: torch.Tensor              # [B]   predictive sd in transformed space
    contrib: torch.Tensor              # [B,K] per-module additive contributions (sum + bias == mu_t)
    sex_logit: torch.Tensor            # [B]
    z_hat: torch.Tensor | None         # [B,N] reconstruction of z-scored beta (SSL decoder)
    h: torch.Tensor                    # [B,K,d] final module tokens
    assign: torch.Tensor | None = None  # [K,N] CpG->module assignment used in this forward pass


class EpiMod(nn.Module):
    def __init__(self, cpgs: CpGTable, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig(n_cpgs=len(cpgs.ids))
        cpgs = cpgs.head(cfg.n_cpgs)
        self.cfg, self.cpg_ids = cfg, list(cpgs.ids)
        N, d, K = len(cpgs.ids), cfg.d_model, cfg.n_modules
        self.register_buffer("mean", torch.as_tensor(cpgs.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(cpgs.std, dtype=torch.float32).clamp_min(1e-2))
        self.register_buffer("sigma_scale", torch.ones(1))   # post-hoc calibration of the predictive sd
        self.ctx = CpGContextEncoder(cpgs, d, cfg.n_pos_scales, cfg.use_genomic_context, cfg.use_id_embedding, cfg.id_dropout)
        self.pool = ModulePooling(d, K, N, cfg.d_key, cfg.n_basis, cfg.d_nl, cfg.linear_skip)
        self.mod_emb = nn.Parameter(torch.randn(K, d) * 0.02)
        if cfg.n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d, cfg.n_heads, cfg.ff_mult * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        else:
            self.blocks = nn.Identity()
        self.out_norm = nn.LayerNorm(d)
        self.read_w = nn.Parameter(torch.randn(K, d) * (d**-0.5) / K**0.5)
        self.read_b = nn.Parameter(torch.zeros(K))
        self.bias0 = nn.Parameter(torch.zeros(1))
        self.sigma_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.sex_head = nn.Linear(d, 1)
        if cfg.ssl_decoder:
            self.ssl_r = nn.Linear(d, d)

    # ------------------------------------------------------------------ re-targeting to other CpG sets
    _N_DEPENDENT = ("mean", "std", "ctx.chrom", "ctx.island", "ctx.reg", "ctx.binary", "ctx.posfeat", "pool.s_free")

    def rebind(self, cpgs: CpGTable) -> "EpiMod":
        """Same learned weights on a different CpG table (other array / probe subset / unseen probes).

        Only possible when tokens are built from genomic context alone (``use_id_embedding=False``);
        a learned per-CpG id embedding ties the weights to the training panel."""
        if self.cfg.use_id_embedding:
            raise ValueError("rebind needs a model trained with use_id_embedding=False")
        from dataclasses import replace

        new = EpiMod(cpgs, replace(self.cfg, n_cpgs=len(cpgs.ids)))
        sd = {k: v for k, v in self.state_dict().items() if k not in self._N_DEPENDENT}
        missing, unexpected = new.load_state_dict(sd, strict=False)
        assert not unexpected and set(missing) <= set(self._N_DEPENDENT), (missing, unexpected)
        return new.to(next(self.parameters()).device)

    @torch.no_grad()
    def warm_start_linear(self, w: np.ndarray, intercept: float, read_shrink: float = 0.05) -> None:
        """Make the model start as the linear clock ``intercept + w . z``.

        ``s_free`` is set so the summed per-module linear terms equal ``w . z`` under the *current*
        module assignment; the nonlinear readout starts small so training refines, not replaces, it."""
        if not self.cfg.linear_skip:
            raise ValueError("model has no linear route")
        N = len(self.cpg_ids)
        colsum = self.pool.assignment(self.ctx().detach()).sum(0)                 # sum_k A_ki
        s = torch.as_tensor(w, dtype=torch.float32) / (N**0.5 * colsum)
        self.pool.s_free.copy_(s - self.pool.s_ctx(self.ctx().detach()).squeeze(-1))
        self.bias0.fill_(float(intercept))
        self.read_w.mul_(read_shrink); self.read_b.zero_()

    # ------------------------------------------------------------------ helpers
    def standardise(self, beta: torch.Tensor):
        """beta [B,N] with NaN for missing -> (z, beta0, miss)."""
        miss = ~torch.isfinite(beta)
        b0 = torch.where(miss, torch.zeros_like(beta), beta)
        z = torch.where(miss, torch.zeros_like(beta), (beta - self.mean) / self.std).clamp(-6.0, 6.0)
        return z, b0, miss

    def assignment(self) -> torch.Tensor:
        return self.pool.assignment(self.ctx())

    # ------------------------------------------------------------------ forward
    def forward(self, beta: torch.Tensor, want_recon: bool = False) -> EpiModOutput:
        z, b0, miss = self.standardise(beta)
        e = self.ctx()
        A = self.pool.assignment(e)
        m, skip = self.pool(e, A, z, b0, miss)
        h = self.out_norm(self.blocks(m + self.mod_emb))
        contrib = (h * self.read_w).sum(-1) + self.read_b            # [B,K]
        if skip is not None:
            contrib = contrib + skip                                # linear clock-like term of each module
        mu = self.bias0 + contrib.sum(-1)
        if self.cfg.distributional:
            sigma = nn.functional.softplus(self.sigma_head(h.mean(1)).squeeze(-1) - 1.0) + 1e-2  # floor ~0.2 y
        else:
            sigma = torch.full_like(mu, 0.1)
        sex = self.sex_head(h.mean(1)).squeeze(-1)
        z_hat = None
        if want_recon and self.cfg.ssl_decoder:
            r = self.ssl_r(e)                                                # [N,d]
            B, K, d = h.shape
            s = (h.reshape(B * K, d) @ r.t()).view(B, K, -1)                 # [B,K,N]
            z_hat = (s * A.unsqueeze(0)).sum(1) * (A.shape[1] ** 0.5)
        return EpiModOutput(mu, sigma, contrib, sex, z_hat, h, A)

    # ------------------------------------------------------------------ inference API
    @torch.no_grad()
    def predict(self, beta: np.ndarray | torch.Tensor, batch_size: int = 128, z_interval: float = 1.645) -> dict[str, np.ndarray]:
        """Age in years with a predictive interval, per-module contributions and sex probability."""
        self.eval()
        x = torch.as_tensor(np.asarray(beta, dtype=np.float32))
        mus, sigs, cons, sexes = [], [], [], []
        for i in range(0, len(x), batch_size):
            o = self.forward(x[i : i + batch_size])
            mus.append(o.mu_t); sigs.append(o.sigma_t); cons.append(o.contrib); sexes.append(torch.sigmoid(o.sex_logit))
        mu = torch.cat(mus).double().numpy()
        sig = (torch.cat(sigs) * self.sigma_scale).double().numpy()
        return {
            "age": t_to_age(mu),
            "age_lo": t_to_age(mu - z_interval * sig),
            "age_hi": t_to_age(mu + z_interval * sig),
            "sigma_years": sig * dage_dt(mu),
            "mu_t": mu,
            "module_contrib_t": torch.cat(cons).numpy(),
            "p_female": torch.cat(sexes).numpy(),
        }
