"""Metrics for guide-efficacy models. The unit of generalization is the gene (or target region), never the individual guide.

  within-gene Spearman : mean over genes of rank correlation between measured and predicted activity -- the 'which guides do I order' question,
                         and immune to between-gene differences in knockdown depth.
  pooled Spearman      : across all guides; mixes gene-level offsets with guide-level signal (TIGER reports this after per-gene de-normalization).
  AUROC                : active vs inactive (threshold from the non-targeting distribution).
  CIs                  : block bootstrap over 50-nt target regions (overlapping guides share a site, so guides are not independent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 3 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0 else float("nan")


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(pearsonr(a[ok], b[ok])[0]) if ok.sum() > 3 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0 else float("nan")


def auroc(label, score) -> float:
    label, score = np.asarray(label), np.asarray(score, float)
    ok = ~np.isnan(score)
    return float(roc_auc_score(label[ok], score[ok])) if len(np.unique(label[ok])) == 2 else float("nan")


def within_group(df: pd.DataFrame, group: str, y: str, p: str, fn=spearman, min_n: int = 30) -> float:
    v = [fn(g[y], g[p]) for _, g in df.groupby(group) if len(g) >= min_n]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def summarize(df: pd.DataFrame, pred: str, y: str = "act", group: str = "gene", label: str | None = "active") -> dict:
    """df[y] is oriented so higher = more active; df[pred] likewise."""
    out = {"n": int(df[pred].notna().sum()),
           "rho_within": within_group(df, group, y, pred, spearman),
           "r_within": within_group(df, group, y, pred, pearson),
           "rho_pooled": spearman(df[y], df[pred])}
    if label and label in df:
        out["auroc_pooled"] = auroc(df[label], df[pred])
        out["auroc_within"] = within_group(df, group, label, pred, auroc)
    return out


def block_bootstrap(df: pd.DataFrame, stat, block: str = "block", n: int = 1000, seed: int = 0):
    """stat(df)->float or dict.  Returns (point, lo, hi) per key, resampling whole blocks."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.values for k, g in df.groupby(block)}
    keys = np.array(list(groups))
    pt = stat(df)
    scalar = not isinstance(pt, dict)
    pt = {"v": pt} if scalar else pt
    draws = {k: [] for k in pt}
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        s = stat(df.loc[idx].reset_index(drop=True))
        s = {"v": s} if scalar else s
        for k in pt:
            draws[k].append(s[k])
    res = {k: (pt[k], float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))) for k in pt}
    return res["v"] if scalar else res


def paired_bootstrap(df: pd.DataFrame, pa: str, pb: str, metric, block: str = "block", n: int = 1000, seed: int = 0):
    """metric(df, col)->float. Returns (delta = metric(pa) - metric(pb), lo, hi, share of resamples where A > B)."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.values for k, g in df.groupby(block)}
    keys = np.array(list(groups))
    d0 = metric(df, pa) - metric(df, pb)
    ds = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        s = df.loc[idx].reset_index(drop=True)
        ds.append(metric(s, pa) - metric(s, pb))
    ds = np.array(ds)
    return d0, float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5)), float(np.nanmean(ds > 0))
