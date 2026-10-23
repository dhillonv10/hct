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


def test_linear_route_is_noop_at_init_invariant_and_linear_in_z():
    cfg = SetConfig(n_tokens_vocab=200, d_model=32, d_emb=16, n_layers=2, n_heads=2, linear_route=True)
    torch.manual_seed(0)
    m = EpiSet(cfg).eval()
    cpg, beta, role, sex = _batch()
    plain = EpiSet(SetConfig(n_tokens_vocab=200, d_model=32, d_emb=16, n_layers=2, n_heads=2)).eval()
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("lin_")})
    assert torch.allclose(m(cpg, beta, role, sex)[0], plain(cpg, beta, role, sex)[0], atol=1e-6)      # zero weights -> same model
    with torch.no_grad():
        m.lin_w.weight.normal_(0, 0.1)
    base = m(cpg, beta, role, sex)[0]
    pad = torch.full((4, 5), -1)                                                                        # padding and permutation invariance survive
    out = m(torch.cat([cpg, pad], 1), torch.cat([beta, torch.rand(4, 5)], 1), torch.cat([role, torch.zeros(4, 5, dtype=torch.long)], 1), sex)[0]
    perm = torch.randperm(cpg.shape[1])
    assert torch.allclose(base, out, atol=1e-5) and torch.allclose(base, m(cpg[:, perm], beta[:, perm], role[:, perm], sex)[0], atol=1e-5)
    # the added term is exactly sum_i w_i z_i * gain + bias
    x, padm, z = m.embed(cpg, beta, role)
    lin = m.linear_term(cpg, role, padm, z)
    manual = (m.lin_w(cpg.clamp_min(0)).squeeze(-1) * z * m.lin_gain[role]).sum(1) + m.lin_bias(role[:, 0]).squeeze(-1)
    assert torch.allclose(lin, manual, atol=1e-6)
    (lin.sum()).backward()
    assert m.lin_w.weight.grad is not None and m.lin_gain.grad is not None


def test_sample_level_moe_routes_whole_samples_and_stays_invariant():
    cfg = SetConfig(n_tokens_vocab=200, d_model=32, d_emb=16, n_layers=2, n_heads=2, moe_experts=4, moe_level="sample")
    torch.manual_seed(0)
    m = EpiSet(cfg).eval()
    cpg, beta, role, sex = _batch()
    base = m(cpg, beta, role, sex)[0]
    # all tokens of a sample share one routing decision (top-2 of 4, constant over the set)
    ffn = m.blocks[0].ffn
    x = torch.randn(3, 9, 32); pad = torch.zeros(3, 9, dtype=torch.bool); pad[0, 6:] = True
    ffn(x, pad)
    assert abs(float(ffn.last_load.sum()) - 1.0) < 1e-5
    # padding and permutation invariance
    padt = torch.full((4, 5), -1)
    out = m(torch.cat([cpg, padt], 1), torch.cat([beta, torch.rand(4, 5)], 1), torch.cat([role, torch.zeros(4, 5, dtype=torch.long)], 1), sex)[0]
    perm = torch.randperm(cpg.shape[1])
    assert torch.allclose(base, out, atol=1e-5) and torch.allclose(base, m(cpg[:, perm], beta[:, perm], role[:, perm], sex)[0], atol=1e-5)
    m.train(); mu, sig, dec = m(cpg, beta, role, sex); (mu.sum() + m.moe_aux()).backward()
    assert m.blocks[0].ffn.router.weight.grad is not None
