"""Latent epigenetic-module pooling.

K learned *modules* each own a soft assignment over the CpG panel. The assignment is computed from
CpG embeddings only (not from the sample's methylation values), so every module is a fixed,
inspectable CpG set -- one can read off which sites, genes, islands or regulatory classes a module
covers. A module's activation for a sample is the assignment-weighted sum of per-CpG value
vectors, which come from two paths:

* linear:    ``z_i * u_i``       (z-scored beta times a CpG-specific direction): a clock-like term;
* nonlinear: ``sum_j rbf_j(beta_i) * W_ij``  (a smooth per-CpG response curve over beta in [0, 1]).

Both paths are contracted with the assignment via single GEMMs, so cost is O(B * N * K * d) with no
``[B, N, d]`` token tensor ever formed: tens of thousands of CpGs are tractable on a CPU.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ModulePooling(nn.Module):
    def __init__(self, d_model: int, n_modules: int, n_cpgs: int, d_key: int = 32, n_basis: int = 6, d_nl: int = 16,
                 linear_skip: bool = True):
        super().__init__()
        self.K, self.d, self.J, self.C = n_modules, d_model, n_basis, d_nl
        self.q = nn.Parameter(torch.randn(n_modules, d_key))
        self.k_proj = nn.Linear(d_model, d_key)
        self.log_scale = nn.Parameter(torch.tensor(-0.5 * math.log(d_key)))
        self.u_proj = nn.Linear(d_model, d_model)
        # direct clock-like route: one scalar weight per CpG, pooled through the same module assignment.
        # s = s_ctx(e) (transferable across CpG sets) + s_free (per-CpG; warm-started from a ridge prior)
        self.skip = linear_skip
        if linear_skip:
            self.s_ctx = nn.Linear(d_model, 1)
            nn.init.zeros_(self.s_ctx.weight); nn.init.zeros_(self.s_ctx.bias)
            self.s_free = nn.Parameter(torch.zeros(n_cpgs))
        self.w_nl = nn.Linear(d_model, n_basis * d_nl)
        self.nl_out = nn.Linear(d_nl, d_model)
        self.register_buffer("centers", torch.linspace(0.0, 1.0, n_basis), persistent=True)
        self.width = 1.0 / max(n_basis - 1, 1)
        self.norm = nn.LayerNorm(d_model)

    def assignment(self, e: torch.Tensor) -> torch.Tensor:
        """Soft CpG->module assignment ``A`` of shape ``[K, N]`` (each row sums to one)."""
        logits = (self.q @ self.k_proj(e).t()) * self.log_scale.exp()
        return torch.softmax(logits, dim=-1)

    def forward(self, e: torch.Tensor, A: torch.Tensor, z: torch.Tensor, beta: torch.Tensor, miss: torch.Tensor):
        """``z`` z-scored values (0 where missing), ``beta`` raw beta (0 where missing), ``miss`` bool.

        Returns ``(m, skip)``: module activations ``[B,K,d]`` and the per-module linear term ``[B,K]``
        (``None`` when the linear skip is disabled)."""
        B, N = z.shape
        sqrt_n = math.sqrt(N)
        skip = None
        if self.skip:
            s = self.s_ctx(e).squeeze(-1) + self.s_free
            skip = (z @ (A * s.unsqueeze(0)).t()) * sqrt_n                                # [B,K]
        # linear path: [B,N] @ [N,K*d]
        U = self.u_proj(e)                                              # [N,d]
        W = (A.t().unsqueeze(-1) * U.unsqueeze(1)).reshape(N, self.K * self.d)
        m_lin = (z @ W).view(B, self.K, self.d) * sqrt_n
        # nonlinear path: per-CpG RBF response, pooled through the same assignment
        Wnl = self.w_nl(e).view(N, self.J, self.C)                      # [N,J,C]
        keep = (~miss).unsqueeze(-1).float()
        T = z.new_zeros(B, N, self.C)
        for j in range(self.J):
            phi = torch.exp(-(((beta - self.centers[j]) / self.width) ** 2))   # [B,N]
            T = T + (phi.unsqueeze(-1) * keep) * Wnl[:, j, :].unsqueeze(0)
        m_nl = (A @ T.permute(1, 0, 2).reshape(N, B * self.C)).view(self.K, B, self.C).permute(1, 0, 2) * sqrt_n
        return self.norm(m_lin + self.nl_out(m_nl)), skip
