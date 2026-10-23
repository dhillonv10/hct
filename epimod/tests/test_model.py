from dataclasses import replace

import numpy as np
import pytest
import torch

from epimod.config import ModelConfig
from epimod.data.store import CpGTable
from epimod.model import EpiMod

from conftest import make_table

SMALL = ModelConfig(n_cpgs=300, d_model=32, n_modules=6, n_layers=2, n_heads=2, d_nl=8)


def test_output_shapes_and_exact_additive_decomposition(table, batch):
    m = EpiMod(table, SMALL).eval()
    o = m(torch.from_numpy(batch), want_recon=True)
    assert o.mu_t.shape == (8,) and o.sigma_t.shape == (8,) and o.contrib.shape == (8, 6)
    assert o.z_hat.shape == batch.shape and o.h.shape == (8, 6, 32)
    # age is an exact sum of per-module contributions (+ one global bias)
    assert torch.allclose(o.mu_t, m.bias0 + o.contrib.sum(-1), atol=1e-6)
    assert (o.sigma_t > 0).all()


def test_module_assignment_is_a_distribution_over_cpgs(table):
    m = EpiMod(table, SMALL)
    A = m.assignment()
    assert A.shape == (6, 300)
    assert torch.allclose(A.sum(-1), torch.ones(6), atol=1e-5) and (A >= 0).all()


def test_handles_missing_values_and_all_missing(table):
    m = EpiMod(table, SMALL).eval()
    x = torch.full((2, 300), float("nan"))
    o = m(x)
    assert torch.isfinite(o.mu_t).all() and torch.isfinite(o.contrib).all()


def test_invariant_to_cpg_order_when_table_is_permuted_with_it(table, batch):
    """Pooling is a sum over CpGs, so reordering CpGs together with their annotations changes nothing."""
    torch.manual_seed(0)
    cfg = replace(SMALL, use_id_embedding=False, dropout=0.0)
    m = EpiMod(table, cfg).eval()
    perm = np.random.RandomState(3).permutation(300)
    m2 = EpiMod(table.subset(perm), cfg).eval()
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if k not in EpiMod._N_DEPENDENT}, strict=False)
    a = m(torch.from_numpy(batch)).mu_t
    b = m2(torch.from_numpy(batch[:, perm])).mu_t
    assert torch.allclose(a, b, atol=1e-4)


def test_gradients_reach_every_learnable_component(table, batch):
    m = EpiMod(table, SMALL).train()
    o = m(torch.from_numpy(batch), want_recon=True)
    (o.mu_t.sum() + o.sigma_t.sum() + o.sex_logit.sum() + o.z_hat.pow(2).mean()).backward()
    dead = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not dead, dead


@pytest.mark.parametrize(
    "over",
    [dict(n_layers=0), dict(use_genomic_context=False), dict(use_id_embedding=False), dict(distributional=False),
     dict(ssl_decoder=False), dict(n_modules=1)],
)
def test_ablation_configs_run(table, batch, over):
    m = EpiMod(table, replace(SMALL, **over)).eval()
    o = m(torch.from_numpy(batch))
    assert torch.isfinite(o.mu_t).all()


def test_rebind_moves_weights_to_a_new_cpg_table(table, batch):
    cfg = replace(SMALL, use_id_embedding=False, dropout=0.0)
    m = EpiMod(table, cfg).eval()
    other = make_table(120, seed=9)                      # a different, smaller probe set
    m2 = m.rebind(other).eval()
    assert m2.cfg.n_cpgs == 120
    o = m2(torch.from_numpy(np.nan_to_num(batch[:, :120], nan=0.5)))
    assert torch.isfinite(o.mu_t).all()
    # learned weights are shared, N-dependent buffers are not
    assert torch.equal(m.pool.q, m2.pool.q) and m2.mean.shape[0] == 120


def test_rebind_refused_when_ids_are_learned(table):
    with pytest.raises(ValueError):
        EpiMod(table, SMALL).rebind(table)


def test_predict_api_returns_years_and_ordered_interval(table, batch):
    m = EpiMod(table, SMALL)
    p = m.predict(batch)
    assert p["age"].shape == (8,) and np.all(p["age_lo"] <= p["age_hi"])
    assert p["module_contrib_t"].shape == (8, 6) and np.all(p["sigma_years"] > 0)
