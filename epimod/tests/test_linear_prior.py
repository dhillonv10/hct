from dataclasses import replace

import numpy as np
import torch

from epimod.config import ModelConfig
from epimod.model import EpiMod
from epimod.model.linear_prior import fit_dual_ridge

CFG = ModelConfig(n_cpgs=300, d_model=32, n_modules=6, n_layers=1, n_heads=2, d_nl=8, dropout=0.0)


def _toy(n=120, p=300, seed=0):
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, p).astype(np.float32)
    w = np.zeros(p, np.float32); w[:10] = 0.3
    y = Z @ w + 0.5 + rng.randn(n) * 0.05
    groups = np.repeat(np.arange(6), n // 6)
    return Z, y.astype(np.float32), groups, w


def test_dual_ridge_recovers_linear_signal_and_picks_a_lambda():
    Z, y, groups, w = _toy()
    coef, b, lam, cv = fit_dual_ridge(Z, y, groups, lambdas=(1, 10, 100))
    assert np.corrcoef(Z @ coef + b, y)[0, 1] > 0.95
    assert lam in cv and min(cv.values()) == cv[lam]


def test_warm_start_makes_model_equal_the_ridge_clock_at_init(table):
    Z, y, groups, _ = _toy()
    coef, b, _, _ = fit_dual_ridge(Z, y, groups, lambdas=(10,))
    m = EpiMod(table, CFG).eval()
    m.warm_start_linear(coef, b)
    with torch.no_grad():                       # silence the nonlinear readout to isolate the linear route
        m.read_w.zero_(); m.read_b.zero_()
        x = torch.from_numpy(Z[:16] * m.std.numpy() + m.mean.numpy())    # beta whose z-score is Z (no clipping hit)
        mu = m(x).mu_t.numpy()
    np.testing.assert_allclose(mu, Z[:16] @ coef + b, rtol=1e-3, atol=1e-3)


def test_skip_off_has_no_linear_parameters(table):
    m = EpiMod(table, replace(CFG, linear_skip=False))
    assert not any(n.startswith("pool.s_") for n, _ in m.named_parameters())
