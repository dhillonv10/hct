"""Probe/evaluation utilities shared by every benchmark task.

All numbers reported by the benchmark go through these functions so that
(a) every model gets the same probe head and hyper-parameter search,
(b) every metric comes with a bootstrap 95% CI over test items.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def bootstrap_ci(metric, y, p, n_boot: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(metric(y[i], p[i]))
        except ValueError:  # single-class resample
            continue
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _spearman(y, p):
    return float(spearmanr(y, p).statistic)


def probe_classification(Xtr, ytr, Xva, yva, Xte, yte, head: str = "logreg", n_boot: int = 500):
    """Fit on train, pick regularisation on val, report on test. Returns dict."""
    sc = StandardScaler().fit(Xtr)
    Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    if head == "logreg":
        best, best_auc = None, -1
        for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr_, ytr)
            auc = roc_auc_score(yva, m.predict_proba(Xva_)[:, 1])
            if auc > best_auc:
                best, best_auc = m, auc
        model = best
    elif head == "gbm":
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                               l2_regularization=1.0, class_weight="balanced",
                                               early_stopping=True, random_state=0).fit(Xtr, ytr)
        Xte_ = Xte
    else:
        raise ValueError(head)
    p = model.predict_proba(Xte_)[:, 1]
    auc = roc_auc_score(yte, p)
    lo, hi = bootstrap_ci(roc_auc_score, np.asarray(yte), p, n_boot)
    return {"auroc": float(auc), "auroc_lo": lo, "auroc_hi": hi,
            "auprc": float(average_precision_score(yte, p)), "n_test": int(len(yte)),
            "pos_rate": float(np.mean(yte))}


def probe_regression(Xtr, ytr, Xva, yva, Xte, yte, head: str = "ridge", n_boot: int = 500):
    sc = StandardScaler().fit(Xtr)
    Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    if head == "ridge":
        best, best_r = None, -9
        for a in (1e0, 1e1, 1e2, 1e3, 1e4):
            m = Ridge(alpha=a).fit(Xtr_, ytr)
            r = _spearman(yva, m.predict(Xva_))
            if r > best_r:
                best, best_r = m, r
        model = best
    elif head == "gbm":
        model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=250,
                                              l2_regularization=1.0, early_stopping=True, random_state=0).fit(Xtr, ytr)
        Xte_ = Xte
    else:
        raise ValueError(head)
    p = model.predict(Xte_)
    lo, hi = bootstrap_ci(_spearman, np.asarray(yte), p, n_boot)
    return {"spearman": _spearman(yte, p), "spearman_lo": lo, "spearman_hi": hi, "n_test": int(len(yte))}


def fmt_auc(r: dict) -> str:
    return f"{r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]"


def fmt_rho(r: dict) -> str:
    return f"{r['spearman']:.3f} [{r['spearman_lo']:.3f},{r['spearman_hi']:.3f}]"


# ----------------------------------------------------------------------------- chromosome-grouped CV
def cv_eval(X, y, folds, kind: str, head: str = "auto", n_boot: int = 500, seed: int = 0):
    """Leave-one-chromosome-group-out CV with pooled out-of-fold predictions.

    kind: 'clf' (y in {0,1}) or 'reg'. head: 'auto' -> logreg/ridge, or 'gbm'.
    Hyper-parameters are chosen on a random 20% split of the *training folds only*.
    """
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    rng = np.random.default_rng(seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for f in np.unique(folds):
        te = folds == f
        tr_idx = np.where(~te)[0]
        rng.shuffle(tr_idx)
        nva = max(int(0.2 * len(tr_idx)), 20)
        va_idx, tr_idx = tr_idx[:nva], tr_idx[nva:]
        Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]
        if kind == "clf":
            sc = StandardScaler().fit(Xtr)
            if head in ("auto", "logreg"):
                best, best_auc = None, -1
                for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
                    a = roc_auc_score(yva, m.predict_proba(sc.transform(Xva))[:, 1])
                    if a > best_auc:
                        best, best_auc = m, a
                oof[te] = best.predict_proba(sc.transform(X[te]))[:, 1]
            else:
                m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200, l2_regularization=1.0,
                                                   class_weight="balanced", random_state=0).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
                oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            sc = StandardScaler().fit(Xtr)
            if head in ("auto", "ridge"):
                best, best_r = None, -9
                for a_ in (1e0, 1e1, 1e2, 1e3, 1e4):
                    m = Ridge(alpha=a_).fit(sc.transform(Xtr), ytr)
                    r = _spearman(yva, m.predict(sc.transform(Xva)))
                    if r > best_r:
                        best, best_r = m, r
                oof[te] = best.predict(sc.transform(X[te]))
            else:
                m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=200, l2_regularization=1.0,
                                                  random_state=0).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
                oof[te] = m.predict(X[te])
    if kind == "clf":
        auc = roc_auc_score(y, oof)
        lo, hi = bootstrap_ci(roc_auc_score, y, oof, n_boot)
        return {"auroc": float(auc), "auroc_lo": lo, "auroc_hi": hi, "auprc": float(average_precision_score(y, oof)),
                "n": int(len(y)), "pos_rate": float(np.mean(y)), "_oof": oof, "_y": y}
    rho = _spearman(y, oof)
    lo, hi = bootstrap_ci(_spearman, y, oof, n_boot)
    return {"spearman": rho, "spearman_lo": lo, "spearman_hi": hi, "n": int(len(y)), "_oof": oof, "_y": y}


def paired_diff(y, p_a, p_b, kind: str, n_boot: int = 1000, seed: int = 0):
    """Bootstrap the metric difference A-B on the SAME items. Returns (delta, lo, hi, p_one_sided_A<=B)."""
    metric = roc_auc_score if kind == "clf" else _spearman
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    d0 = metric(y, p_a) - metric(y, p_b)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            ds.append(metric(y[i], p_a[i]) - metric(y[i], p_b[i]))
        except ValueError:
            continue
    ds = np.asarray(ds)
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds <= 0).mean())
