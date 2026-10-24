"""Correctness checks. Run these on every new machine (especially before trusting MPS/CUDA):

    python -m lncfm.selfcheck            # CPU checks + device parity if MPS/CUDA is present

1. padding invariance  - an embedding must not depend on the other sequences in the batch
2. span masking        - ~15% of positions masked, labels only on masked positions
3. junction labels     - label index == last nucleotide of each non-final exon
4. overfit             - loss falls on a fixed tiny batch (the optimisation path works)
5. device parity       - same weights, same input: CPU vs MPS/CUDA logits agree
"""
from __future__ import annotations

import numpy as np
import torch

from . import pretrain
from .model import PAD, build, encode, pick_device


def _batch(seqs):
    L = -(-max(len(s) for s in seqs) // 4) * 4
    X = np.zeros((len(seqs), L), np.int64)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = encode(s)
    return torch.from_numpy(X)


def check_padding_invariance(model, tol=1e-4):
    rng = np.random.default_rng(0)
    a = "".join(rng.choice(list("ACGU"), 203))        # deliberately not a multiple of 4
    b = "".join(rng.choice(list("ACGU"), 1000))
    model.eval()
    with torch.no_grad():
        z1 = model(_batch([a]))["pooled"][0]
        z2 = model(_batch([a, b]))["pooled"][0]
    d = float((z1 - z2).abs().max())
    assert d < tol, f"embedding depends on batch padding (max|diff|={d:.3g})"
    return d


def main():
    torch.manual_seed(0)
    m = build("tiny")
    print(f"[1] padding invariance, max|diff| = {check_padding_invariance(m):.2e}  (must be < 1e-4)")
    rng = np.random.default_rng(0)
    mk = pretrain.span_mask(2000, rng)
    print(f"[2] span mask fraction = {mk.mean():.3f} (target 0.15)")
    assert 0.12 < mk.mean() < 0.25
    seq = "ACGU" * 105 + "AC"                          # 422 nt, exons 91 + 218 + 109 = 418 (+4 spare)
    j = np.cumsum((91, 218, 109))[:-1] - 1
    x, y, jl = pretrain.make_view(seq, j.astype(np.int32), rng, 512, False)
    assert set(np.where(jl == 1)[0]) == {90, 308}, np.where(jl == 1)[0]
    ids = encode(seq).astype(np.int64)
    masked = y != -100
    assert (y[masked] == ids[masked]).all(), "labels must equal the original nucleotides"
    assert ((x != ids) <= masked).all(), "inputs may only differ from the original at masked positions"
    print("[3] junction labels at", sorted(np.where(jl == 1)[0].tolist()), "(expected [90, 308]); mask labels consistent")
    X, Y, J = pretrain.collate([pretrain.make_view("ACGU" * 100, j.astype(np.int32), rng, 512, False) for _ in range(8)])
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    m.train()
    first = last = None
    for i in range(40):
        loss, _ = pretrain.compute_loss(m, X, Y, J, 0.3, 0.3)
        opt.zero_grad(); loss.backward(); opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    print(f"[4] overfit loss {first:.3f} -> {last:.3f}")
    assert last < first
    # LoRA wrapper: identical to the base layer at init, and only the adapters receive gradients
    import torch.nn as nn
    from .finetune import LoRALinear
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
    xx = torch.randn(5, 16)
    assert torch.allclose(lora(xx), base(xx), atol=1e-6), "LoRA must start exactly at the pretrained layer"
    for p_ in base.parameters():
        p_.requires_grad = False
    lora(xx).sum().backward()
    assert lora.A.grad is not None and lora.B.grad is not None and base.weight.grad is None
    print("[4b] LoRA wrapper: exact at init; gradients reach only A and B")
    dev = pick_device("auto")
    if dev.type != "cpu":
        m.eval()
        with torch.no_grad():
            ref = m(X)["nt_logits"]
            out = m.to(dev)(X.to(dev))["nt_logits"].cpu()
        d = float((ref - out).abs().max())
        print(f"[5] {dev.type} vs cpu logits max|diff| = {d:.2e}")
        assert d < 2e-3, "device results differ from CPU"
        m.cpu()
        d2 = check_padding_invariance(m.to(dev), tol=1e-3)
        print(f"    padding invariance on {dev.type}: {d2:.2e}")
    else:
        print("[5] no accelerator present: device parity not checked")
    print("selfcheck OK")


if __name__ == "__main__":
    main()
