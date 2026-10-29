"""siRNA development on Huesken ONLY (contiguous-chunk CV ~ held-out target genes). Mix and Takayuki are not read here.

    python -m guidefam.scripts.sirna_dev --oligo OLIGOFORMER/data
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import cv
from .. import sirna as SI


def score(y, yb, p):
    return {"PCC": float(np.corrcoef(y, p)[0, 1]), "SRCC": float(spearmanr(y, p)[0]), "AUC": float(roc_auc_score(yb, p)), "PRC": float(average_precision_score(yb, p))}


def chunk_cv(d, X, make, k=10):
    fold = np.arange(len(d)) * k // len(d)              # contiguous chunks of the gene-ordered file
    oof = np.full(len(d), np.nan)
    for f in range(k):
        te = fold == f
        m = make().fit(d[~te], d.y[~te].values)
        oof[te] = m.predict(d[te])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oligo", required=True)
    a = ap.parse_args()
    d = SI.load(f"{a.oligo}/Hu.csv")
    X = SI.tabular(d)
    yb = d["label"].gt(0.7).astype(int).values       # OligoFormer's binary label: >70 % of maximal inhibition
    rows = {}
    for nm in ("reynolds", "uitei"):
        rows[nm] = score(d.y.values, yb, X[nm].values)
    td = [c for c in X.columns if c.startswith("td")]
    seq = [c for c in X.columns if c.startswith("s") and "_" in c]
    sets = {"ridge_td": td, "ridge_seq": seq, "ridge_seq+td": seq + td, "ridge_all": list(X.columns)}
    for nm, cols in sets.items():
        oof = chunk_cv(d, X, lambda cols=cols: cv.Ridge(X, cols=cols, alpha=30.0))
        rows[nm] = score(d.y.values, yb, oof)
    for nm, cols in {"gbm_all": list(X.columns), "gbm_seq+td": seq + td}.items():
        oof = chunk_cv(d, X, lambda cols=cols: cv.GBM(X, cols=cols, n_jobs=1, n_estimators=300, num_leaves=7, min_child_samples=15))
        rows[nm] = score(d.y.values, yb, oof)
    print(f"Huesken, {len(d)} siRNAs, 10 contiguous chunks (approximately target-held-out)")
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
