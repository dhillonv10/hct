"""Closed-form ridge prior used to warm-start the linear route of EpiMod.

Summing a learned weight over ~10^4 CpGs makes plain Adam over-shoot (each weight moves ~lr per
step in a coordinated direction, so the output moves ~sqrt(N) too fast). Starting that route from
the ridge solution -- fitted in dual form on the *training* studies only -- sidesteps the problem
and lets the module transformer spend its capacity on what a linear clock cannot express.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.model_selection import GroupKFold


def fit_dual_ridge(Z: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   lambdas=(1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6), n_splits: int = 4) -> tuple[np.ndarray, float, float, dict[float, float]]:
    """Return (weights ``[N]``, intercept, chosen lambda, {lambda: grouped-CV median |error|})."""
    Z = np.asarray(Z, dtype=np.float32)
    K = (Z @ Z.T).astype(np.float64)
    yc = y.astype(np.float64)

    def solve(idx_tr, lam):
        mu = yc[idx_tr].mean()
        Ktr = K[np.ix_(idx_tr, idx_tr)] + lam * np.eye(len(idx_tr))
        return linalg.solve(Ktr, yc[idx_tr] - mu, assume_a="pos"), mu

    cv = {}
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(Z, yc, groups))
    for lam in lambdas:
        errs = []
        for tr, te in folds:
            alpha, mu = solve(tr, lam)
            errs.append(np.abs(K[np.ix_(te, tr)] @ alpha + mu - yc[te]))
        cv[float(lam)] = float(np.median(np.concatenate(errs)))
    best = min(cv, key=cv.get)
    alpha, mu = solve(np.arange(len(yc)), best)
    return (Z.T @ alpha.astype(np.float32)).astype(np.float32), float(mu), best, cv
