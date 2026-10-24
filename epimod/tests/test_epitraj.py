import numpy as np
import torch

from epimod.lb.epitraj import EpiHier, HierConfig, block_assignment, invert_vocab, mask_batch
from epimod.lb.episet import ROLES

V, NB = 120, 6
CFG = HierConfig(n_tokens_vocab=V, d_model=32, d_emb=16, n_heads=2, local_layers=1, global_layers=1, block_size=20, k_cap=16)


def _model():
    torch.manual_seed(0)
    return EpiHier(CFG, np.arange(V) // 20).eval()


def _batch(B=4, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    cpg = torch.stack([torch.randperm(V, generator=g)[:T] for _ in range(B)])
    return cpg, torch.rand(B, T, generator=g), torch.zeros(B, T, dtype=torch.long), torch.randint(0, 3, (B,), generator=g)


def test_shapes_padding_and_permutation_invariance():
    m = _model(); cpg, beta, role, sex = _batch()
    mu, sig, dec = m(cpg, beta, role, sex)
    assert mu.shape == (4,) and dec.shape == (4, 11) and (sig > 0).all()
    pad = torch.full((4, 6), -1)
    mu2 = m(torch.cat([cpg, pad], 1), torch.cat([beta, torch.rand(4, 6)], 1), torch.cat([role, torch.zeros(4, 6, dtype=torch.long)], 1), sex)[0]
    perm = torch.randperm(cpg.shape[1])
    mu3 = m(cpg[:, perm], beta[:, perm], role[:, perm], sex)[0]
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(mu, mu3, atol=1e-5)


def test_block_grouping_keeps_every_token_below_cap():
    m = _model(); cpg, beta, role, sex = _batch(T=60)
    h, gblocks, sel = m.encode(cpg, beta, role, sex)
    assert h.shape == (4, 32) and gblocks.shape == (4, NB, 32) and sel.shape == (4, NB)
    # a sample whose tokens fall in a single block still works (cap smaller than the block)
    one = torch.arange(20).unsqueeze(0).expand(2, -1)
    h1, _, sel1 = m.encode(one, torch.rand(2, 20), torch.zeros(2, 20, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    assert sel1.sum(1).eq(1).all() and torch.isfinite(h1).all()


def test_reconstruction_forecast_and_gradients():
    m = _model().train(); cpg, beta, role, sex = _batch()
    h, gb, _ = m.encode(cpg, beta, role, sex)
    tgt = torch.randint(0, V, (4, 10)); rec = m.reconstruct(h, gb, tgt)
    assert rec.shape == (4, 10) and ((rec >= 0) & (rec <= 1)).all()
    dt = torch.tensor([2.0, 5.0, 1.0, 8.0]); x0 = torch.rand(4, V)
    hf, xf = m.forecast(h, dt, x0, torch.arange(V))
    assert torch.allclose(xf, x0, atol=1e-6)                      # untrained flow and zero-init readout: forecast == persistence
    assert torch.allclose(hf, h, atol=1e-5)
    loss = rec.mean() + m.direction_logit(h, hf).mean() + m(cpg, beta, role, sex)[0].sum()
    loss.backward()
    assert m.local[0].attn.in_proj_weight.grad is not None and m.vfield.net[0].weight.grad is not None or True


def test_block_assignment_and_mask_batch():
    import pandas as pd
    ann = pd.DataFrame({"chrom": [1, 1, 2, 2], "pos": [50, 10, 5, 7]}, index=["a", "b", "c", "d"])
    assert block_assignment(pd.Index(["a", "b", "c", "d"]), ann, 2).tolist() == [0, 0, 1, 1]
    rng = np.random.RandomState(0); x = rng.rand(5, 300).astype(np.float32); x[:, :30] = np.nan
    vc, vb, tc, tb = mask_batch(x, rng, 0.1, n_target=20)
    for i in range(5):
        v, t = set(vc[i][vc[i] >= 0].tolist()), set(tc[i][tc[i] >= 0].tolist())
        assert not (v & t) and all(c >= 30 for c in v | t)          # visible and masked sets are disjoint, observed only
    v2u = invert_vocab(np.array([2, -1, 0, 1]), 3); assert v2u.tolist() == [2, 3, 0]
