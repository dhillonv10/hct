"""Age acceleration and deviation-from-healthy-aging as disease signals, for any methylation dataset.

Reference (fitted once, cached): ComputAge TRAIN-split healthy controls only, on the compact CpG panel
  clock       ridge on the standardised panel, Horvath-transformed age
  AA          clock age - age, residualised on age within the reference (so it is age-independent)
  deviation   Mahalanobis distance (Ledoit-Wolf) of the PCA(k) embedding from its age-conditional spline mean

Typical use (e.g. AML cohorts):  rows = samples, columns = CpG ids (cg...), plus a metadata table with age and a group label.
CpGs missing from your data are mean-imputed (reported as panel coverage); 450K / EPIC data cover the panel fully, 27K does not.
Caveats that matter for haematological malignancy: leukaemic blasts shift cell-type composition and the methylome far outside the
healthy range, so large AA / deviation can reflect composition and purity rather than ageing; compare like with like
(same tissue, platform, preprocessing) and treat the score as a disease marker, not as biological age."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from ..age import age_to_t, t_to_age
from ..lb.lite import load_lite_universe
from ..paths import subdir


class HealthyReference:
    def __init__(self, k: int = 30, alpha: float = 3000.0):
        self.k, self.alpha = k, alpha

    @classmethod
    def fit(cls, k: int = 30, alpha: float = 3000.0) -> "HealthyReference":
        from ..lb.store import lb_meta, load_lb_store
        self = cls(k, alpha)
        self.cpgs = load_lite_universe()
        st = load_lb_store(include_series=set(lb_meta(None, geo=False).DatasetID), geo=False)
        m = st.meta; ref = ((m.split == "train") & (m.Class == "HC")).to_numpy()
        X = np.asarray(st.prof, dtype=np.float32)[ref]; X = np.where(X == 255, np.nan, X / 100.0)
        age = m.Age.to_numpy(float)[ref]
        self.mu = np.where(np.isfinite(np.nanmean(X, 0)), np.nanmean(X, 0), 0.5); self.sd = np.maximum(np.nanstd(X, 0), 0.02)
        Z = (np.where(np.isfinite(X), X, self.mu) - self.mu) / self.sd
        self.clock = Ridge(alpha=alpha).fit(Z, age_to_t(age))
        self.aa_fit = np.polyfit(age, t_to_age(self.clock.predict(Z)) - age, 1)
        self.pca = PCA(k, random_state=0).fit(Z); E = self.pca.transform(Z)
        self.man = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(alpha=5.0)).fit(age[:, None], E)
        self.prec = LedoitWolf().fit(E - self.man.predict(age[:, None])).precision_
        self.n_ref = int(ref.sum())
        return self

    def save(self, path=None):
        pickle.dump(self, open(path or subdir("eval") / "healthy_reference.pkl", "wb"))

    @staticmethod
    def load(path=None) -> "HealthyReference":
        p = path or subdir("eval") / "healthy_reference.pkl"
        try:
            return pickle.load(open(p, "rb"))
        except FileNotFoundError:
            r = HealthyReference.fit(); r.save(p); return r

    def score(self, X: np.ndarray, cpg_ids, age: np.ndarray) -> pd.DataFrame:
        """X [samples, len(cpg_ids)] beta in [0,1] (NaN = missing)."""
        cols = pd.Index(self.cpgs).get_indexer(pd.Index(cpg_ids))
        have = cols >= 0
        full = np.full((len(X), len(self.cpgs)), np.nan, np.float32); full[:, cols[have]] = np.asarray(X, np.float32)[:, have]
        cover = float(np.isfinite(full).any(0).mean())
        Z = (np.where(np.isfinite(full), full, self.mu) - self.mu) / self.sd
        age = np.asarray(age, float)
        pred = t_to_age(self.clock.predict(Z)); aa = pred - age - np.polyval(self.aa_fit, age)
        R = self.pca.transform(Z) - self.man.predict(age[:, None])
        out = pd.DataFrame({"age": age, "clock_age": pred, "age_acceleration": aa, "abs_age_acceleration": np.abs(aa),
                            "deviation": np.sqrt(np.einsum("bi,ij,bj->b", R, self.prec, R))})
        out.attrs["panel_coverage"] = cover
        return out


def group_report(scores: pd.DataFrame, group: pd.Series, control, study: pd.Series | None = None) -> pd.DataFrame:
    """Per group (and study): n, mean score, Cohen's d vs controls, Mann-Whitney p, AUC (higher score = case)."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    rows = []
    keys = [(None, scores.index)] if study is None else [(s, scores.index[study.to_numpy() == s]) for s in sorted(study.unique())]
    for s, idx in keys:
        ctrl = idx[group.loc[idx].to_numpy() == control]
        for g in sorted(set(group.loc[idx]) - {control}):
            case = idx[group.loc[idx].to_numpy() == g]
            if len(case) < 3 or len(ctrl) < 3:
                continue
            for col in ("age_acceleration", "abs_age_acceleration", "deviation"):
                a, b = scores.loc[case, col].to_numpy(), scores.loc[ctrl, col].to_numpy()
                sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
                rows.append({"study": s, "group": g, "score": col, "n_case": len(a), "n_control": len(b), "mean_case": a.mean(), "mean_control": b.mean(),
                             "cohens_d": (a.mean() - b.mean()) / sp if sp > 0 else np.nan, "mannwhitney_p": stats.mannwhitneyu(a, b).pvalue,
                             "auc": roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]),
                             "age_case": scores.loc[case, "age"].mean(), "age_control": scores.loc[ctrl, "age"].mean()})
    return pd.DataFrame(rows)
