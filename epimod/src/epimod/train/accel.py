"""Cross-fitted biological-age acceleration head.

A chronological-age clock is not obliged to read older in disease. This stage adds a tiny
zero-initialised head ``c(h) = sum_k <w_k, h_k>`` on the *frozen* module tokens ``h`` so that
``age' = f(mu_t + c)`` keeps healthy controls on the identity line yet places aging-accelerating
condition (AAC) samples above same-study controls of similar age.

Training signal comes from the benchmark's case/control labels, so it is only ever evaluated
**cross-fitted**: the head that scores a sample was trained without any sample from that sample's
study (``fold_by="study"``) or without any study of that disease class (``fold_by="class"``, the
strict test of generalising to a disease the head has never seen).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..age import age_to_t
from ..eval.computage_protocol import CLASS_OF


@torch.no_grad()
def extract_features(model: nn.Module, X: np.ndarray, batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    hs, mus, cons = [], [], []
    for i in range(0, len(X), batch_size):
        o = model(torch.from_numpy(X[i : i + batch_size].astype(np.float32)))
        hs.append(o.h); mus.append(o.mu_t); cons.append(o.contrib)
    return {"h": torch.cat(hs).numpy(), "mu_t": torch.cat(mus).numpy(), "contrib": torch.cat(cons).numpy()}


def _t_to_age_torch(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t <= 0, 21.0 * torch.exp(t.clamp(max=0.0)) - 1.0, 21.0 * t + 20.0)


@dataclass
class AccelConfig:
    steps: int = 300
    lr: float = 5e-3
    weight_decay: float = 1e-2
    tau_years: float = 3.0       # softness of the pairwise ranking loss
    w_hc: float = 1.0            # keep controls on the identity line
    age_window: float = 7.0      # case/control pairs must be this close in age
    max_pairs: int = 200_000
    seed: int = 0


class AccelHead(nn.Module):
    def __init__(self, K: int, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(K, d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h * self.w).sum((-1, -2))

    def per_module(self, h: torch.Tensor) -> torch.Tensor:
        return (h * self.w).sum(-1)


def _pairs(meta: pd.DataFrame, rows: np.ndarray, cfg: AccelConfig) -> tuple[np.ndarray, np.ndarray]:
    """(case, control) index pairs within each study, matched on age, plus per-pair weights.

    Weights are ``1 / (#pairs of that study x condition)``: every benchmark test counts once, so a
    3,000-sample cohort cannot dominate the learned direction."""
    rng = np.random.RandomState(cfg.seed)
    m = meta.iloc[rows]
    pos = {s: i for i, s in enumerate(meta.index)}
    out, wts = [], []
    for gse, g in m.groupby("DatasetID"):
        ctrl = g.index[g.Condition == "HC"]
        for cond in set(g.Condition) - {"HC"}:
            case = g.index[g.Condition == cond]
            if len(ctrl) == 0 or len(case) == 0:
                continue
            ca, ha = g.loc[case, "Age"].to_numpy(), g.loc[ctrl, "Age"].to_numpy()
            block = []
            for i, a in zip(case, ca):
                ok = np.flatnonzero(np.abs(ha - a) <= cfg.age_window)
                if len(ok):
                    for j in rng.choice(ok, size=min(3, len(ok)), replace=False):
                        block.append((pos[i], pos[ctrl[j]]))
            if block:      # a condition with no age-matched controls contributes no pairs
                out += block; wts += [1.0 / len(block)] * len(block)
    pairs = np.array(out, dtype=np.int64).reshape(-1, 2)
    w = np.array(wts, dtype=np.float32)
    if len(pairs) > cfg.max_pairs:
        keep = rng.choice(len(pairs), cfg.max_pairs, replace=False)
        pairs, w = pairs[keep], w[keep]
    return pairs, w


def fit_head(feats: dict, meta: pd.DataFrame, train_rows: np.ndarray, cfg: AccelConfig = AccelConfig()) -> AccelHead:
    torch.manual_seed(cfg.seed)
    h, mu = torch.from_numpy(feats["h"]), torch.from_numpy(feats["mu_t"])
    age = torch.as_tensor(np.array(meta.Age.to_numpy(np.float32), copy=True))
    is_hc = torch.as_tensor(np.array((meta.Condition == "HC").to_numpy(), copy=True))
    pairs_np, pw_np = _pairs(meta, train_rows, cfg)
    pairs, pw = torch.from_numpy(pairs_np), torch.from_numpy(pw_np)
    tr = torch.zeros(len(meta), dtype=torch.bool); tr[torch.as_tensor(train_rows)] = True
    hc_idx = torch.nonzero(tr & is_hc & (age >= 18) & (age <= 90)).squeeze(-1)
    head = AccelHead(h.shape[1], h.shape[2])
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.steps):
        delta = _t_to_age_torch(mu + head(h)) - age                    # age acceleration in years, all samples
        loss = torch.zeros(())
        if len(hc_idx):
            loss = loss + cfg.w_hc * nn.functional.smooth_l1_loss(delta[hc_idx], (_t_to_age_torch(mu) - age)[hc_idx])
        if len(pairs):
            diff = delta[pairs[:, 0]] - delta[pairs[:, 1]]
            loss = loss + (nn.functional.softplus(-diff / cfg.tau_years) * pw).sum() / pw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def cross_fitted_age(feats: dict, meta: pd.DataFrame, fold_by: str = "study", n_folds: int = 5,
                     cfg: AccelConfig = AccelConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Return (accelerated-age predictions in years, per-module acceleration contributions)."""
    assert meta.index.is_unique, "sample ids must be unique"
    studies = np.array(sorted(meta.DatasetID.unique()))
    if fold_by == "study":
        rng = np.random.RandomState(cfg.seed)
        order = rng.permutation(len(studies))
        fold_of_study = {studies[i]: k % n_folds for k, i in enumerate(order)}
        fold = meta.DatasetID.map(fold_of_study).to_numpy()
    elif fold_by == "class":
        # a study's folds = disease classes it contains; its HCs follow the class of its cases
        cls = meta.groupby("DatasetID").Condition.agg(lambda s: next((CLASS_OF.get(c) for c in s if c != "HC"), None))
        classes = sorted({c for c in cls.dropna()})
        fold_of_study = {g: (classes.index(c) if c in classes else 0) for g, c in cls.items()}
        fold = meta.DatasetID.map(fold_of_study).to_numpy()
        n_folds = len(classes)
    else:
        raise ValueError(fold_by)
    mu, h = feats["mu_t"], feats["h"]
    out_t = np.array(mu, copy=True)
    out_mod = np.zeros_like(feats["contrib"])
    for f in range(n_folds):
        te = np.flatnonzero(fold == f); tr = np.flatnonzero(fold != f)
        if len(te) == 0:
            continue
        head = fit_head(feats, meta, tr, cfg)
        with torch.no_grad():
            ht = torch.from_numpy(h[te])
            out_t[te] = mu[te] + head(ht).numpy()
            out_mod[te] = head.per_module(ht).numpy()
    from ..age import t_to_age

    return t_to_age(out_t), out_mod
