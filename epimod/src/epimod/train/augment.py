"""Augmentations of *measured* methylation profiles.

Nothing here fabricates a subject: every operation perturbs a real profile the way a different
lab / normalisation / array would, or hides part of it so the model must infer it.
"""
from __future__ import annotations

import torch

from ..config import TrainConfig


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


def augment_batch(beta: torch.Tensor, cfg: TrainConfig, gen: torch.Generator, template_pool: torch.Tensor | None = None):
    """Return (augmented beta with NaN = hidden/missing, ssl_mask).

    ``ssl_mask`` marks CpGs that were measured but hidden by the masking step; the SSL loss
    reconstructs exactly those from the clean profile."""
    B, N = beta.shape
    x = beta.clone()
    observed = torch.isfinite(x)
    # 1. monotone distortion in logit space (dynamic-range / normalisation differences between pipelines)
    sel = (torch.rand(B, 1, generator=gen) < cfg.distort_prob) & True
    a = torch.exp((torch.rand(B, 1, generator=gen) - 0.5) * 0.3)          # slope in [0.86, 1.16]
    c = torch.randn(B, 1, generator=gen) * 0.15
    x = torch.where(sel & observed, torch.sigmoid(_logit(x) * a + c), x)
    # 2. measurement noise
    sd = torch.rand(B, 1, generator=gen) * cfg.noise_max
    x = torch.where(observed, (x + torch.randn(B, N, generator=gen) * sd).clamp(0.0, 1.0), x)
    # 3. borrow the real missingness pattern of another sample (e.g. a different array's probe set)
    if template_pool is not None and cfg.template_miss_prob > 0:
        use = torch.rand(B, generator=gen) < cfg.template_miss_prob
        if use.any():
            j = torch.randint(0, template_pool.shape[0], (int(use.sum()),), generator=gen)
            x[use] = torch.where(torch.isfinite(template_pool[j]), x[use], torch.full_like(x[use], float("nan")))
    # 4. masking for the self-supervised objective
    frac = torch.rand(B, 1, generator=gen) * cfg.mask_max
    hide = (torch.rand(B, N, generator=gen) < frac) & torch.isfinite(x)
    x = torch.where(hide, torch.full_like(x, float("nan")), x)
    return x, hide
