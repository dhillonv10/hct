"""Elastic-net baseline on exactly the same CpG panel, studies and age transform as EpiMod."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold

from ..age import t_to_age
from ..data.store import MethylData


class ElasticNetClock:
    def __init__(self, l1_ratio=(0.2, 0.5, 0.9), n_alphas=40, cv=4, seed=0):
        self.l1_ratio, self.n_alphas, self.cv, self.seed = list(l1_ratio), n_alphas, cv, seed

    def _z(self, X):
        X = X.astype(np.float32)
        Z = (X - self.mean_) / self.std_
        return np.where(np.isfinite(Z), np.clip(Z, -6, 6), 0.0)

    def fit(self, data: MethylData):
        self.mean_, self.std_ = data.cpgs.mean.astype(np.float32), np.maximum(data.cpgs.std, 1e-2).astype(np.float32)
        y, _ = data.tensors()
        groups = data.meta["DatasetID"].to_numpy()
        # study-grouped folds, so alpha is chosen for generalisation to *unseen studies*
        folds = list(GroupKFold(n_splits=self.cv).split(np.zeros(len(y)), y, groups))
        kw = dict(l1_ratio=self.l1_ratio, cv=folds, max_iter=2000, tol=1e-3, n_jobs=4, random_state=self.seed)
        try:                                   # scikit-learn >= 1.7: an int `alphas` is the grid size
            self.model_ = ElasticNetCV(alphas=self.n_alphas, **kw)
        except TypeError:                      # older releases
            self.model_ = ElasticNetCV(n_alphas=self.n_alphas, **kw)
        self.model_.fit(self._z(data.X), y)
        return self

    def predict(self, X) -> np.ndarray:
        return t_to_age(self.model_.predict(self._z(X)))

    @property
    def n_nonzero(self) -> int:
        return int((self.model_.coef_ != 0).sum())
