"""One-hot CNN baseline (the standard 'small neural net trained from scratch' comparator in RNA benchmarks).

Same protocol as every other provider: for each dev fold train on the other dev folds (20% of them held out for
epoch selection), predict the fold; optionally fit on all dev and predict the locked test set. Sequences are
clipped/padded to `max_len` nucleotides; global max+mean pooling gives a length-agnostic summary.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import evalkit
from .kmer import encode_ints


def _onehot(seqs, max_len):
    X = np.zeros((len(seqs), 4, max_len), np.float32)
    M = np.zeros((len(seqs), max_len), np.float32)
    for i, s in enumerate(seqs):
        a = encode_ints(s[:max_len])
        ok = a >= 0
        X[i, a[ok], np.where(ok)[0]] = 1.0
        M[i, :len(a)] = 1.0
    return torch.from_numpy(X), torch.from_numpy(M)


class CNN(nn.Module):
    def __init__(self, n_out, c=64):
        super().__init__()
        self.c1 = nn.Conv1d(4, c, 9, padding=4)
        self.c2 = nn.Conv1d(c, 2 * c, 9, padding=8, dilation=2)
        self.c3 = nn.Conv1d(2 * c, 2 * c, 9, padding=16, dilation=4)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(4 * c, n_out)

    def forward(self, x, m):
        h = F.gelu(self.c1(x)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        mm = m.unsqueeze(1)
        mx = (h - 1e4 * (1 - mm)).max(-1).values
        mean = (h * mm).sum(-1) / mm.sum(-1).clamp(min=1)
        return self.out(self.drop(torch.cat([mx, mean], 1)))


def _fit(Xo, Mo, y, kind, tr, te_list, epochs, seed, device, lr=2e-3, bs=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_out = y.shape[1] if kind == "multi" else 1
    ys = y.astype(np.float32)
    if kind == "reg":
        ys = (ys - ys[tr].mean()) / (ys[tr].std() + 1e-6)
    perm = rng.permutation(tr); nva = max(int(0.2 * len(perm)), 16); va, tr = perm[:nva], perm[nva:]
    net = CNN(n_out).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)

    def pred(idx):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                b = idx[i:i + 128]
                out.append(net(Xo[b].to(device), Mo[b].to(device)).float().cpu().numpy())
        o = np.concatenate(out)
        return o if kind == "multi" else o.ravel()

    best, state = -9, None
    for ep in range(epochs):
        net.train()
        for i in range(0, len(tr), bs):
            b = rng.permutation(tr)[i:i + bs] if False else tr[rng.permutation(len(tr))[i:i + bs]]
            out = net(Xo[b].to(device), Mo[b].to(device))
            yb = torch.from_numpy(ys[b]).to(device)
            loss = F.mse_loss(out.squeeze(-1), yb) if kind == "reg" else F.binary_cross_entropy_with_logits(out.squeeze(-1) if kind == "clf" else out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = pred(va)
        sc = evalkit._spearman(ys[va], pv) if kind == "reg" else (evalkit._auc_fast(ys[va], pv) if kind == "clf"
             else float(np.nanmean([evalkit._auc_fast(ys[va][:, j], pv[:, j]) for j in range(pv.shape[1])])))
        if sc > best:
            best, state = sc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return [pred(t) for t in te_list]


def cnn_predict(task, final=False, epochs=12, max_len=1000, seed=0, device="cpu"):
    """Returns dict(dev=oof array, test=array or None) aligned with task.df rows."""
    Xo, Mo = _onehot(list(task.df.seq), max_len)
    fold, y, kind = task.df.fold.values, task.y, task.kind
    shape = y.shape if kind == "multi" else (len(y),)
    oof = np.full(shape, np.nan)
    for f in sorted(set(fold[fold >= 0])):
        te = np.where(fold == f)[0]; tr = np.where((fold >= 0) & (fold != f))[0]
        oof[te] = _fit(Xo, Mo, y, kind, tr, [te], epochs, seed + f, device)[0]
    test = None
    if final:
        test = np.full(shape, np.nan)
        te = np.where(fold == -1)[0]; tr = np.where(fold >= 0)[0]
        test[te] = _fit(Xo, Mo, y, kind, tr, [te], epochs, seed + 99, device)[0]
    return {"dev": oof, "test": test}
