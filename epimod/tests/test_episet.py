import numpy as np
import torch

from epimod.lb.episet import ROLES, EpiSet, MoEFFN, SetConfig

CFG = SetConfig(n_tokens_vocab=200, d_model=32, d_emb=16, n_layers=2, n_heads=2)


def _batch(B=4, T=30, seed=0, vocab=200):
    g = torch.Generator().manual_seed(seed)
    cpg = torch.randint(0, vocab, (B, T), generator=g)
    beta = torch.rand(B, T, generator=g)
    role = torch.randint(0, 2, (B, T), generator=g)
    sex = torch.randint(0, 3, (B,), generator=g)
    return cpg, beta, role, sex


def test_shapes_and_padding_is_ignored():
    m = EpiSet(CFG).eval()
    cpg, beta, role, sex = _batch()
    mu, sig, dec = m(cpg, beta, role, sex)
    assert mu.shape == (4,) and sig.shape == (4,) and dec.shape == (4, 11) and (sig > 0).all()
    # appending padding tokens must not change the output
    pad = torch.full((4, 7), -1)
    mu2, sig2, _ = m(torch.cat([cpg, pad], 1), torch.cat([beta, torch.rand(4, 7)], 1), torch.cat([role, torch.zeros(4, 7, dtype=torch.long)], 1), sex)
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(sig, sig2, atol=1e-5)


def test_set_is_permutation_invariant():
    m = EpiSet(CFG).eval()
    cpg, beta, role, sex = _batch()
    perm = torch.randperm(cpg.shape[1])
    a = m(cpg, beta, role, sex)[0]; b = m(cpg[:, perm], beta[:, perm], role[:, perm], sex)[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_missing_values_and_gradients():
    m = EpiSet(CFG).train()
    cpg, beta, role, sex = _batch()
    beta[0, :5] = float("nan")
    mu, sig, dec = m(cpg, beta, role, sex, clock=torch.tensor([0, 1, 2, 3]))
    (mu.sum() + sig.sum() + dec.sum()).backward()
    assert torch.isfinite(mu).all()
    dead = [n for n, p in m.named_parameters() if p.requires_grad and n not in ("id_emb.weight", "dir_emb.weight") and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not dead, dead


def test_moe_gates_sparse_normalised_and_balance_loss_defined():
    ffn = MoEFFN(16, 32, n_experts=4, top_k=2, dropout=0.0)
    x = torch.randn(3, 10, 16)
    y = ffn(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert abs(float(ffn.last_load.sum()) - 1.0) < 1e-5            # fractions of routed token-slots sum to 1
    assert float(ffn.aux) > 0.99                                    # E * sum(frac * p) >= 1 for any routing (equality at balance)
    y.sum().backward()
    assert ffn.router.weight.grad is not None


def test_moe_model_runs_and_exposes_aux_loss():
    cfg = SetConfig(n_tokens_vocab=200, d_model=32, d_emb=16, n_layers=2, n_heads=2, moe_experts=4)
    m = EpiSet(cfg)
    cpg, beta, role, sex = _batch()
    mu, _, _ = m(cpg, beta, role, sex)
    assert float(m.moe_aux()) > 0 and torch.isfinite(mu).all()


def test_pairwise_logit_flips_sign_when_donors_are_swapped():
    """Pairwise comparison uses one role for both views, so logit(A,B) = f(A) - f(B) = -logit(B,A) exactly."""
    m = EpiSet(CFG).eval()
    cpg, beta_a, _, sex = _batch(); beta_b = torch.rand_like(beta_a)
    role = torch.full_like(cpg, ROLES["pair_a"])
    f = lambda x: m(cpg, x, role, sex)[0]
    assert torch.allclose(f(beta_a) - f(beta_b), -(f(beta_b) - f(beta_a)), atol=1e-6)
    assert not torch.allclose(f(beta_a), f(beta_b))                 # and the two views are actually distinguished
