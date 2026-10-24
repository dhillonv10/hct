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


# =============================================================================== v2 protocol
# dev CV (folds >= 0) + locked test (fold == -1), leak-pruned subsets, multi-label probes.
def _auc_fast(y, p):
    """Rank-based AUROC (ties averaged). Returns nan if one class is missing."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strat_auroc(y, p, strata):
    """Mean of within-stratum AUROCs weighted by n_pos*n_neg: removes confounding by the stratifier (e.g. expression)."""
    y, p, strata = np.asarray(y), np.asarray(p), np.asarray(strata)
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = _auc_fast(y[m], p[m])
        if a == a:
            w = float((y[m] == 1).sum() * (y[m] == 0).sum())
            num += a * w; den += w
    return num / den if den else float("nan")


def _fit_predict(Xtr, ytr, Xte, kind, rng, head="auto"):
    """Probe with hyper-parameters chosen on an inner 20% split, then refit on all of Xtr.
    kind: 'clf' | 'reg' | 'multi' (multi-output ridge on 0/1 labels, scores used for AUROC)."""
    idx = rng.permutation(len(ytr))
    nva = max(int(0.2 * len(idx)), 20)
    va, tr = idx[:nva], idx[nva:]
    sc = StandardScaler().fit(Xtr[tr])
    A, B = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
    if kind == "clf" and head in ("auto", "logreg"):
        best, ba = 1e-2, -1
        for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(A, ytr[tr])
            a = roc_auc_score(ytr[va], m.predict_proba(B)[:, 1])
            if a > ba:
                best, ba = C, a
        sc2 = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=best, max_iter=3000, class_weight="balanced").fit(sc2.transform(Xtr), ytr)
        return m.predict_proba(sc2.transform(Xte))[:, 1]
    if kind == "clf" and head == "gbm":
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200, l2_regularization=1.0,
                                           class_weight="balanced", random_state=0).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if kind == "reg" and head in ("auto", "ridge"):
        best, br = 1e2, -9
        for a_ in (1e0, 1e1, 1e2, 1e3, 1e4):
            r = _spearman(ytr[va], Ridge(alpha=a_).fit(A, ytr[tr]).predict(B))
            if r > br:
                best, br = a_, r
        sc2 = StandardScaler().fit(Xtr)
        return Ridge(alpha=best).fit(sc2.transform(Xtr), ytr).predict(sc2.transform(Xte))
    if kind == "reg" and head == "gbm":
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=200, l2_regularization=1.0,
                                             random_state=0).fit(Xtr, ytr).predict(Xte)
    if kind == "multi":
        best, bs = 1e3, -1
        for a_ in (1e1, 1e2, 1e3, 1e4, 1e5):
            P = Ridge(alpha=a_).fit(A, ytr[tr]).predict(B)
            s = np.nanmean([_auc_fast(ytr[va][:, j], P[:, j]) for j in range(P.shape[1])])
            if s > bs:
                best, bs = a_, s
        sc2 = StandardScaler().fit(Xtr)
        return Ridge(alpha=best).fit(sc2.transform(Xtr), ytr).predict(sc2.transform(Xte))
    raise ValueError((kind, head))


def dev_oof(X, y, fold, kind, head="auto", seed=0):
    """Leave-fold-out predictions for dev items (fold >= 0). Test items (fold == -1) get NaN."""
    X, y, fold = np.asarray(X), np.asarray(y), np.asarray(fold)
    rng = np.random.default_rng(seed)
    out = np.full(y.shape if kind == "multi" else len(y), np.nan)
    for f in sorted(set(fold[fold >= 0])):
        te, tr = fold == f, fold >= 0
        tr = tr & ~te
        out[te] = _fit_predict(X[tr], y[tr], X[te], kind, rng, head)
    return out


def final_pred(X, y, fold, kind, head="auto", seed=0):
    """Fit on ALL dev items, predict the locked test items (fold == -1). Dev items get NaN."""
    X, y, fold = np.asarray(X), np.asarray(y), np.asarray(fold)
    rng = np.random.default_rng(seed)
    out = np.full(y.shape if kind == "multi" else len(y), np.nan)
    te = fold == -1
    out[te] = _fit_predict(X[fold >= 0], y[fold >= 0], X[te], kind, rng, head)
    return out


def summarize_single(y, p, kind, n_boot=300):
    m = ~np.isnan(p)
    y, p = np.asarray(y)[m], np.asarray(p)[m]
    if len(y) < 20:
        return {"n": int(len(y))}
    if kind == "clf":
        lo, hi = bootstrap_ci(roc_auc_score, y, p, n_boot)
        return {"auroc": float(roc_auc_score(y, p)), "auroc_lo": lo, "auroc_hi": hi, "n": int(len(y)), "pos_rate": float(np.mean(y))}
    lo, hi = bootstrap_ci(_spearman, y, p, n_boot)
    return {"spearman": _spearman(y, p), "spearman_lo": lo, "spearman_hi": hi, "n": int(len(y))}


def summarize_multi(Y, P, strata, rbps, n_boot=300, min_pos=5, seed=0, min_rbps_ok=15):
    """Per-RBP AUROC and confounder-stratified AUROCs; macro means with bootstrap CI over RBPs.
    strata: dict name -> per-item stratum labels (e.g. expression quintile; expression x length)."""
    m = ~np.isnan(P[:, 0])
    Y, P = np.asarray(Y)[m], P[m]
    strata = {k: np.asarray(v)[m] for k, v in strata.items()}
    rows = []
    for j, name in enumerate(rbps):
        if Y[:, j].sum() >= min_pos and (1 - Y[:, j]).sum() >= min_pos:
            r = {"rbp": name, "n_pos": int(Y[:, j].sum()), "auroc": _auc_fast(Y[:, j], P[:, j])}
            for k, st in strata.items():
                r[f"auroc_strat_{k}"] = strat_auroc(Y[:, j], P[:, j], st)
            rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return {"n_rbps": 0, "underpowered": True}, t
    rng = np.random.default_rng(seed)
    out = {"n_rbps": int(len(t)), "n_genes": int(m.sum()), "underpowered": bool(len(t) < min_rbps_ok)}
    for col in [c for c in t.columns if c.startswith("auroc")]:
        v = t[col].values
        bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
        out[f"macro_{col}"], out[f"macro_{col}_lo"], out[f"macro_{col}_hi"] = float(np.nanmean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return out, t


import pandas as pd  # noqa: E402  (used by summarize_multi)
