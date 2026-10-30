"""MiniPairFormer: a ~25k-parameter transformer for guide / target-site activity whose attention is biased by local RNA structure.

Tokens: the 23-nt guide (complement of the target, as in TIGER's tables) and the target window (6 nt 5' flank | 23-nt target | 6 nt 3' flank).
Attention is plain scaled dot-product plus an additive bias per head:
  target-target : a_h * log(P_ij + eps), P = base-pair probabilities of the folded local transcript window (ViennaRNA partition function) -- structure-aware attention
  guide-target  : learned bias indexed by the offset between a guide position and a target position (lets heads learn the alignment diagonal and its neighbourhood)
  guide-guide   : learned relative-position bias
Each target token also receives its unpaired probability (accessibility) as an input feature. Mean-pooled guide and target representations are concatenated with
TIGER's scalar descriptors and fed to a small MLP. `use_structure=False` removes both structure inputs (ablation).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import features as F

FLANK, GL = 6, 23
TL = FLANK + GL + FLANK            # 35 target-window tokens
L = GL + TL                        # 58 tokens in total


def pair_features(d: pd.DataFrame, transcripts: dict, half: int = 75):
    """Per guide: P (TL x TL pair probabilities) and u (TL unpaired probabilities) for the target window, from folding transcript[start-half : end+half]."""
    import RNA
    P = np.zeros((len(d), TL, TL), np.float16)
    U = np.ones((len(d), TL), np.float16)
    for r, (g, s0) in enumerate(zip(d.gene.values, d.start.values.astype(int) - 1)):
        tx = transcripts[g].replace("T", "U")
        lo, hi = max(0, s0 - FLANK - half), min(len(tx), s0 + GL + FLANK + half)
        sub = tx[lo:hi]
        fc = RNA.fold_compound(sub)
        _, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
        fc.pf()
        B = np.array(fc.bpp(), dtype=np.float32)[1:, 1:]
        B = B + B.T
        a = s0 - FLANK - lo                                     # window start inside sub
        for_idx = np.arange(a, a + TL)
        ok = (for_idx >= 0) & (for_idx < len(sub))
        ii = for_idx[ok]
        P[r][np.ix_(np.where(ok)[0], np.where(ok)[0])] = B[np.ix_(ii, ii)]
        U[r, np.where(ok)[0]] = np.clip(1.0 - B[ii].sum(axis=1), 0.0, 1.0)
    return P, U


def _tokens(d: pd.DataFrame):
    """Integer tokens (0 = unknown/pad, 1-4 = A,C,G,T): the guide (TIGER's guide_seq, complement of the target) and the 35-nt target window."""
    tok = lambda oh: np.where(oh.sum(-1) > 0, oh.argmax(-1) + 1, 0)
    return tok(F.onehot(list(d["guide_seq"]), GL)), tok(F.target_window(d, FLANK))


class _Layer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias):                              # x (B, L, d); bias (B, h, L, L)
        B, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk) + bias, dim=-1)
        x = x + self.drop(self.o((att @ v).transpose(1, 2).reshape(B, n, d)))
        return x + self.drop(self.ff(self.n2(x)))


class MiniPairFormer(nn.Module):
    def __init__(self, n_scalar: int, d: int = 32, heads: int = 4, layers: int = 2, ff: int = 64, drop: float = 0.1, use_structure: bool = True):
        super().__init__()
        self.use_structure, self.h = use_structure, heads
        self.emb = nn.Embedding(5, d, padding_idx=0)
        self.pos_g, self.pos_t = nn.Parameter(torch.zeros(GL, d)), nn.Parameter(torch.zeros(TL, d))
        self.typ = nn.Parameter(torch.zeros(2, d))
        self.unp = nn.Linear(1, d, bias=False)
        self.rel_gt = nn.Parameter(torch.zeros(heads, 2 * (TL + GL), ))       # guide-target offset bias
        self.rel_gg = nn.Parameter(torch.zeros(heads, 2 * GL))
        self.a_struct = nn.Parameter(torch.zeros(heads))
        self.layers = nn.ModuleList([_Layer(d, heads, ff, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d + n_scalar, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))
        gi, tj = torch.arange(GL)[:, None], torch.arange(TL)[None, :]
        self.register_buffer("off_gt", (tj - (gi + FLANK) + TL + GL).long())   # index of (target_j - aligned_target_of_guide_i)
        self.register_buffer("off_gg", (torch.arange(GL)[None, :] - torch.arange(GL)[:, None] + GL).long())
        nn.init.normal_(self.pos_g, std=0.02), nn.init.normal_(self.pos_t, std=0.02), nn.init.normal_(self.typ, std=0.02)

    def forward(self, g_tok, t_tok, scal, P=None, U=None):
        B = g_tok.shape[0]
        xg = self.emb(g_tok) + self.pos_g + self.typ[0]
        xt = self.emb(t_tok) + self.pos_t + self.typ[1]
        if self.use_structure and U is not None:
            xt = xt + self.unp(U.unsqueeze(-1))
        x = torch.cat([xg, xt], dim=1)                                            # (B, 58, d)
        bias = x.new_zeros(B, self.h, L, L)
        bias[:, :, :GL, GL:] = self.rel_gt[:, self.off_gt]
        bias[:, :, GL:, :GL] = self.rel_gt[:, self.off_gt].transpose(-1, -2)
        bias[:, :, :GL, :GL] = self.rel_gg[:, self.off_gg]
        if self.use_structure and P is not None:
            bias[:, :, GL:, GL:] = self.a_struct.view(1, -1, 1, 1) * torch.log(P.float().unsqueeze(1) + 1e-3)
        for lyr in self.layers:
            x = lyr(x, bias)
        x = self.norm(x)
        z = torch.cat([x[:, :GL].mean(1), x[:, GL:].mean(1), scal], dim=1)
        return self.head(z).squeeze(-1)


class PairFormerRegressor:
    """fit(train_df, y) / predict(df) like Tiger2DRegressor. The dataframe needs guide_seq, target_seq, 5p_context, 3p_context, block and (for structure) P/U lookups by row index."""

    def __init__(self, P=None, U=None, use_structure: bool = True, scalars: bool = True, max_epochs: int = 150, patience: int = 20, batch_size: int = 256,
                 lr: float = 2e-3, seed: int = 0, cols=None):
        self.P, self.U, self.use_structure, self.use_scalars = P, U, use_structure, scalars
        self.max_epochs, self.patience, self.bs, self.lr, self.seed, self.fixed_cols = max_epochs, patience, batch_size, lr, seed, cols

    def _in(self, d):
        g, t = _tokens(d)
        sc = F.scalar_frame(d).reindex(columns=self.cols).fillna(self.mu)
        sc = ((sc - self.mu) / self.sd).values.astype(np.float32) if self.use_scalars else np.zeros((len(d), 0), np.float32)
        ix = d.index.values
        P = torch.from_numpy(self.P[ix].astype(np.float32)) if (self.use_structure and self.P is not None) else None
        U = torch.from_numpy(self.U[ix].astype(np.float32)) if (self.use_structure and self.U is not None) else None
        return torch.from_numpy(g).long(), torch.from_numpy(t).long(), torch.from_numpy(sc), P, U

    def fit(self, d, y, val_frac: float = 0.1):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.cols = list(self.fixed_cols) if self.fixed_cols is not None else list(F.scalar_frame(d.iloc[:2]).columns)
        sf = F.scalar_frame(d).reindex(columns=self.cols)
        self.mu, self.sd = sf.mean(), sf.std().replace(0, 1.0)
        g, t, sc, P, U = self._in(d)
        y = torch.from_numpy(np.asarray(y, np.float32))
        ub = np.unique(d["block"].values)
        vb = set(rng.choice(ub, max(1, int(len(ub) * val_frac)), replace=False))
        vm = np.array([b in vb for b in d["block"].values])
        tr, va = np.where(~vm)[0], np.where(vm)[0]
        self.net = MiniPairFormer(sc.shape[1], use_structure=self.use_structure and P is not None)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        lc = lambda p, q: (p - q + torch.nn.functional.softplus(-2 * (p - q)) - math.log(2)).mean()
        sel = lambda ix: (g[ix], t[ix], sc[ix], None if P is None else P[ix], None if U is None else U[ix])
        best, state, wait = float("inf"), None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad()
                lc(self.net(*sel(b)), y[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lc(self.net(*sel(va)), y[va]))
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

    def predict(self, d):
        g, t, sc, P, U = self._in(d)
        self.net.eval()
        with torch.no_grad():
            return np.concatenate([self.net(g[i:i + 1024], t[i:i + 1024], sc[i:i + 1024], None if P is None else P[i:i + 1024],
                                            None if U is None else U[i:i + 1024]).numpy() for i in range(0, len(g), 1024)])
