"""Leave-one-gene-out evaluation with per-gene target normalisation computed on the TRAINING genes only.

A model is anything with `fit(train_df, ya)` / `predict(df)`; `ya` is activity (higher = more active), normalised per gene as in TIGER
(centre = gene median, scale = q90 - q10 of the gene's own PM LFCs), so that genes with different knockdown depth are comparable.
Predictions come back on that normalised scale; within-gene metrics are unaffected by it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def activity_target(d: pd.DataFrame, mode: str = "tiger") -> np.ndarray:
    """Per-(source, gene) target, oriented so higher = more active.  Computed from the rows given, so call it on the TRAINING rows only.

      tiger  : (median - y) / (q90 - q10) within each gene -- TIGER's 'FrequentistQuantile' normalisation
      rank   : rank-Gauss of the activity within each gene (robust to the saturated tail of strong guides)
      global : centre on the gene median, divide by the source-wide (q90 - q10) of the residuals; for sources with only a handful of guides per gene
    A `source` column (hek / flow / hap1) selects the mode per source: `global` is forced on hap1, whose genes have ~8 guides each.
    """
    from scipy.stats import norm
    src = d["source"] if "source" in d else pd.Series("hek", index=d.index)
    out = pd.Series(0.0, index=d.index)
    for sname, ds in d.groupby(src):
        m = "global" if sname == "hap1" else mode
        g = ds.groupby("gene").y
        med = g.transform("median")
        if m == "tiger":
            sc = g.transform(lambda x: x.quantile(0.9) - x.quantile(0.1)).replace(0, np.nan)
            v = -(ds.y - med) / sc
        elif m == "rank":
            v = -ds.groupby("gene").y.transform(lambda x: norm.ppf((x.rank(method="average") - 0.5) / len(x)))
        elif m == "global":
            res = ds.y - med
            v = -res / (res.quantile(0.9) - res.quantile(0.1))
        else:
            raise ValueError(m)
        out[ds.index] = v.fillna(0.0)
    return out.values


class GBM:
    """LightGBM regressor on a precomputed feature table (`X` is indexed like the dataframe the model is fit on)."""

    def __init__(self, X: pd.DataFrame, cols=None, seed: int = 0, **params):
        import lightgbm as lgb
        self.X, self.cols = X, (list(cols) if cols is not None else list(X.columns))
        self.params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, random_state=seed, n_jobs=4)
        self.params.update(params)
        self.lgb = lgb

    def fit(self, d, ya):
        self.m = self.lgb.LGBMRegressor(**self.params).fit(self.X.loc[d.index, self.cols], ya)
        return self

    def predict(self, d):
        return self.m.predict(self.X.loc[d.index, self.cols])


class Ridge:
    def __init__(self, X: pd.DataFrame, cols=None, alpha: float = 30.0):
        from sklearn.linear_model import Ridge as R
        from sklearn.preprocessing import StandardScaler
        self.X, self.cols, self.R, self.S, self.alpha = X, (list(cols) if cols is not None else list(X.columns)), R, StandardScaler, alpha

    def fit(self, d, ya):
        x = self.X.loc[d.index, self.cols]
        self.mu = x.mean()
        self.sc = self.S().fit(x.fillna(self.mu))
        self.m = self.R(alpha=self.alpha).fit(self.sc.transform(x.fillna(self.mu)), ya)
        return self

    def predict(self, d):
        return self.m.predict(self.sc.transform(self.X.loc[d.index, self.cols].fillna(self.mu)))


def logo_predict(d: pd.DataFrame, make_model, genes=None, extra: pd.DataFrame | None = None, mode: str = "tiger", verbose: bool = True) -> np.ndarray:
    """Out-of-fold predictions: each gene is predicted by a model that never saw it.  `extra` rows (other screens) are always in training."""
    out = pd.Series(np.nan, index=d.index)
    for g in (genes or sorted(d.gene.unique())):
        te = d.gene == g
        train = d[~te] if extra is None else pd.concat([d[~te], extra])
        m = make_model().fit(train, activity_target(train, mode))
        out[te] = m.predict(d[te])
        if verbose:
            print(f"    held out {g:9s} n={int(te.sum()):5d}", flush=True)
    return out.values


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, make_model, mode: str = "tiger") -> np.ndarray:
    return make_model().fit(train, activity_target(train, mode)).predict(test)


class RankGBM(GBM):
    """LightGBM LambdaRank: learn to order guides WITHIN a gene. Relevance = within-gene activity quantile (0..n_levels-1); one query per gene."""

    def __init__(self, X, cols=None, seed: int = 0, n_levels: int = 5, **params):
        base = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, random_state=seed, n_jobs=1, label_gain=list(range(n_levels * 2)))
        base.update(params)
        super().__init__(X, cols=cols, seed=seed)
        self.params, self.n_levels = base, n_levels

    def fit(self, d, ya):
        ya = pd.Series(ya, index=d.index)
        rel = ya.groupby(d.gene.values).transform(lambda x: np.minimum((x.rank(pct=True) * self.n_levels).astype(int), self.n_levels - 1)).astype(int)
        o = np.argsort(d.gene.values, kind="stable")                    # rows of a query must be contiguous
        dd, rr = d.iloc[o], rel.iloc[o]
        grp = dd.groupby("gene", sort=False).size().values
        self.m = self.lgb.LGBMRanker(**self.params).fit(self.X.loc[dd.index, self.cols], rr.values, group=grp)
        return self


class TabDPT:
    """Pretrained tabular foundation model (TabDPT, Layer 6; open safetensors weights) as the learner on a precomputed feature table.
    In-context: `fit` stores the training rows, `predict` conditions on them. Missing values are mean-imputed from the training rows."""

    def __init__(self, X: pd.DataFrame, cols=None, n_ensembles: int = 8, device: str = "cpu", seed: int = 0):
        self.X, self.cols = X, (list(cols) if cols is not None else list(X.columns))
        self.n_ensembles, self.device, self.seed = n_ensembles, device, seed

    def fit(self, d, ya):
        from tabdpt import TabDPTRegressor
        x = self.X.loc[d.index, self.cols]
        self.mu = x.mean()
        self.m = TabDPTRegressor(device=self.device)
        self.m.fit(x.fillna(self.mu).values.astype(np.float32), np.asarray(ya, np.float32))
        return self

    def predict(self, d):
        x = self.X.loc[d.index, self.cols].fillna(self.mu).values.astype(np.float32)
        return np.asarray(self.m.predict(x, n_ensembles=self.n_ensembles)).ravel()
