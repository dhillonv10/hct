"""Simple, strong baselines on exactly the inputs each LongevityBench DNAm task exposes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV

from ..age import age_to_t, t_to_age
from .store import MISSING, LBStore, extremes_tokens, pair_tokens, sample_pairs
from .tasks import CLOCKS, Vocab, SEX_ID


# ------------------------------------------------------------------------------ fixed / clock sets
class DenseClock:
    """Ridge / gradient boosting on a fixed CpG list; missing values -> training mean."""

    def __init__(self, kind: str = "ridge"):
        self.kind = kind

    def _x(self, X):
        X = np.where(np.isfinite(X), X, self.mean_)
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, age: np.ndarray):
        self.mean_ = np.nanmean(X, axis=0); self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.5)
        self.std_ = np.maximum(np.nanstd(X, axis=0), 0.02)
        Z, y = self._x(X), age_to_t(age)
        if self.kind == "ridge":
            self.m_ = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Z, y)
        elif self.kind == "hgb":
            self.m_ = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0).fit(Z, y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return t_to_age(self.m_.predict(self._x(X)))


# ------------------------------------------------------------------------------ extremes -> age
class BagOfCpGs:
    """Ridge on multi-hot 'CpG is in the lowest / highest list' features (+ sex) -- the natural linear
    model for the benchmark's value-selected inputs. With ``values`` the listed beta enters as well
    (one slope per CpG and list), so the model can use how extreme a listed value is."""

    def __init__(self, alpha: float = 30.0, values: bool = False):
        self.alpha, self.values = alpha, values

    def _feat(self, cpg: np.ndarray, role: np.ndarray, sex: np.ndarray, beta: np.ndarray | None = None) -> sp.csr_matrix:
        B, T = cpg.shape
        V = self.V
        ok = cpg >= 0
        col = np.where(ok, cpg + V * role, 0)
        rows = np.repeat(np.arange(B), T).reshape(B, T)
        parts = [sp.csr_matrix((ok.ravel().astype(np.float32), (rows.ravel(), col.ravel())), shape=(B, 2 * V))]
        if self.values:
            v = np.where(ok, (beta - 0.5) * 2.0, 0.0).astype(np.float32)
            parts.append(sp.csr_matrix((v.ravel(), (rows.ravel(), col.ravel())), shape=(B, 2 * V)))
        parts.append(sp.csr_matrix((np.ones(B, np.float32), (np.arange(B), np.minimum(sex, 2))), shape=(B, 3)))
        return sp.hstack(parts).tocsr()

    def fit(self, cpg, role, sex, age, V: int, beta=None):
        self.V = V
        self.m_ = Ridge(alpha=self.alpha, solver="sparse_cg").fit(self._feat(cpg, role, sex, beta), age_to_t(age))
        return self

    def predict_t(self, cpg, role, sex, beta=None) -> np.ndarray:
        return self.m_.predict(self._feat(cpg, role, sex, beta))

    def predict(self, cpg, role, sex, beta=None) -> np.ndarray:
        return t_to_age(self.predict_t(cpg, role, sex, beta))


# ------------------------------------------------------------------------------ pairwise
class PairLogit:
    """Logistic regression on sparse (beta_A - beta_B) features over the listed CpGs; trained on both
    orientations so the decision is antisymmetric."""

    def __init__(self, C: float = 0.5):
        self.C = C

    def _feat(self, cpg: np.ndarray, ba: np.ndarray, bb: np.ndarray) -> sp.csr_matrix:
        B, T = cpg.shape
        ok = cpg >= 0
        d = np.where(ok, ba - bb, 0.0).astype(np.float32)
        rows = np.repeat(np.arange(B), T).reshape(B, T)
        return sp.csr_matrix((d.ravel(), (rows.ravel(), np.where(ok, cpg, 0).ravel())), shape=(B, self.V))

    def fit(self, cpg, ba, bb, a_older, V: int):
        self.V = V
        X = self._feat(cpg, ba, bb)
        X2 = sp.vstack([X, -X]); y2 = np.concatenate([a_older, 1 - a_older])
        self.m_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=300, solver="lbfgs").fit(X2, y2)
        return self

    def predict_logit(self, cpg, ba, bb) -> np.ndarray:
        return self._feat(cpg, ba, bb) @ self.m_.coef_.ravel()
