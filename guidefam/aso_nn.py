"""Tier-1 model for ASO efficacy: a ~1e5-parameter two-branch CNN + dense head.

  ASO branch     : 28 positions x [nucleotide, sugar (DNA/MOE/cEt/OMe/F), backbone (PO/PS)] embeddings -> Conv1d -> flatten
  context branch : 88-nt window of the target RNA starting 30 nt 5' of the site (nucleotide embedding + unpaired probability) -> 2 x Conv1d -> pooled
  dense branch   : the label-free handcrafted table (thermodynamics, accessibility summaries, gapmer geometry, dose, cell line, ...), standardised
Everything is fitted on training rows only (standardisation included); early stopping uses held-out GENE GROUPS of the training set.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import aso as A

P_ASO, W_CTX, PRE = 28, 88, 30
SUG = {"<pad>": 0, "DNA": 1, "MOE": 2, "CET": 3, "OME": 4, "F": 5}
BB = {"<pad>": 0, "PO": 1, "PS": 2}
NUC = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}


def arrays(d: pd.DataFrame, X: pd.DataFrame, acc: np.ndarray, idx: pd.Series):
    """Integer / float arrays for the net. `X` is `aso.features(d, acc, idx)`; rows without a located site get an empty context window."""
    n = len(d)
    nt, sg, bb = (np.zeros((n, P_ASO), np.int64) for _ in range(3))
    for i, (s, su, ba) in enumerate(zip(d.aso.values, d.sugar_mods.values, d.backbone_mods.values)):
        nt[i, :min(len(s), P_ASO)] = [NUC.get(c, 0) for c in s[:P_ASO]]
        t = A._tokens(su)[:P_ASO]
        sg[i, :len(t)] = [SUG.get(x.upper(), 0) for x in t]
        t = A._tokens(ba)[:P_ASO]
        bb[i, :len(t)] = [BB.get(x, 0) for x in t]
    cx = np.zeros((n, W_CTX), np.int64)
    cu = np.zeros((n, W_CTX), np.float32)
    ci = d.rna_context.map(idx).values
    for i, (c, s) in enumerate(zip(d.rna_context.values, d.aso.values)):
        if not isinstance(c, str) or np.isnan(ci[i]):
            continue
        p = c.find(A.rc_dna(s).replace("T", "U"))
        if p < 0:
            continue
        lo, hi = p - PRE, p - PRE + W_CTX
        a, b = max(lo, 0), min(hi, len(c))
        cx[i, a - lo:b - lo] = [NUC.get(ch, 0) for ch in c[a:b]]
        u = acc[int(ci[i])][a:b]
        cu[i, a - lo:b - lo] = np.nan_to_num(u, nan=0.5)
    dense = X[[c for c in X.columns if not re.match(r"^p\d+[ACGT]$", c)]].copy()
    for c in ("cell_line", "transfection"):                       # categorical codes -> one-hot (-1 = other / unknown)
        dense = pd.concat([dense.drop(columns=c), pd.get_dummies(dense[c], prefix=c).astype(np.float32)], axis=1)
    return nt, sg, bb, cx, cu, dense


class Net(nn.Module):
    def __init__(self, n_dense: int, ch: int = 32, drop: float = 0.2):
        super().__init__()
        self.e_nt, self.e_sg, self.e_bb = nn.Embedding(5, 8, padding_idx=0), nn.Embedding(6, 4, padding_idx=0), nn.Embedding(3, 4, padding_idx=0)
        self.a_conv = nn.Sequential(nn.Conv1d(16, ch, 5, padding=2), nn.GELU(), nn.Conv1d(ch, 16, 3, padding=1), nn.GELU())
        self.a_fc = nn.Sequential(nn.Flatten(), nn.Linear(16 * P_ASO, 64), nn.GELU(), nn.Dropout(drop))
        self.e_cx = nn.Embedding(5, 8, padding_idx=0)
        self.c_conv = nn.Sequential(nn.Conv1d(9, ch, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
                                    nn.Conv1d(ch, ch, 7, padding=6, dilation=2), nn.GELU(), nn.MaxPool1d(2))
        self.c_fc = nn.Sequential(nn.Flatten(), nn.Linear(ch * (W_CTX // 4), 64), nn.GELU(), nn.Dropout(drop))
        self.d_fc = nn.Sequential(nn.Linear(n_dense, 64), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Dropout(drop), nn.Linear(96, 1))

    def forward(self, nt, sg, bb, cx, cu, dn):
        a = torch.cat([self.e_nt(nt), self.e_sg(sg), self.e_bb(bb)], -1).transpose(1, 2)
        c = torch.cat([self.e_cx(cx), cu.unsqueeze(-1)], -1).transpose(1, 2)
        z = torch.cat([self.a_fc(self.a_conv(a)), self.c_fc(self.c_conv(c)), self.d_fc(dn)], 1)
        return self.head(z).squeeze(-1)


class ASONet:
    """fit(rows, y, groups) / predict(rows) on index arrays into the prepared tensors."""

    def __init__(self, T, max_epochs=20, patience=4, bs=512, lr=2e-3, seed=0, threads=4):
        self.T, self.max_epochs, self.patience, self.bs, self.lr, self.seed = T, max_epochs, patience, bs, lr, seed
        torch.set_num_threads(threads)

    def _batch(self, ix):
        nt, sg, bb, cx, cu, dn = self.T
        return nt[ix], sg[ix], bb[ix], cx[ix], cu[ix], dn[ix]

    def fit(self, tr, y, groups, val_frac=0.1):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        ug = np.unique(groups[tr])
        vg = set(rng.choice(ug, max(1, int(len(ug) * val_frac)), replace=False))
        vm = np.array([g in vg for g in groups[tr]])
        t_idx, v_idx = tr[~vm], tr[vm]
        dn = self.T[5]
        self.mu = dn[t_idx].mean(0, keepdim=True)
        self.sd = dn[t_idx].std(0, keepdim=True).clamp_min(1e-2)
        self.net = Net(dn.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.max_epochs * math.ceil(len(t_idx) / self.bs))
        yt = torch.as_tensor(y, dtype=torch.float32)
        best, state, wait = 1e9, None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            perm = rng.permutation(t_idx)
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad()
                loss = nn.functional.smooth_l1_loss(self.net(*self._in(b)), yt[b], beta=0.2)
                loss.backward()
                opt.step()
                sched.step()
            vl = float(nn.functional.smooth_l1_loss(torch.as_tensor(self._pred(v_idx)), yt[v_idx], beta=0.2))
            if vl < best:
                best, wait, state = vl, 0, {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.epochs_ = ep + 1
        self.n_params = sum(p.numel() for p in self.net.parameters())
        return self

    def _in(self, ix):
        nt, sg, bb, cx, cu, dn = self._batch(ix)
        return nt, sg, bb, cx, cu, ((dn - self.mu) / self.sd).clamp(-5.0, 5.0)         # clip: a level unseen in the training fold must not explode

    def _pred(self, ix):
        self.net.eval()
        with torch.no_grad():
            return np.concatenate([self.net(*self._in(ix[i:i + 2048])).numpy() for i in range(0, len(ix), 2048)])

    def predict(self, ix):
        return self._pred(np.asarray(ix))


def tensors(nt, sg, bb, cx, cu, dense: pd.DataFrame):
    dn = dense.astype(np.float32).copy()
    miss = dn.isna().any()
    for c in dn.columns[miss.values]:
        dn[c + "_na"] = dn[c].isna().astype(np.float32)
    med = dn.median()
    dn = dn.fillna(med).fillna(0.0)
    return (torch.as_tensor(nt), torch.as_tensor(sg), torch.as_tensor(bb), torch.as_tensor(cx), torch.as_tensor(cu), torch.as_tensor(dn.values.astype(np.float32)))
