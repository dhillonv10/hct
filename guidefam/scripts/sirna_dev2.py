"""siRNA development, round 2 (Huesken contiguous-chunk CV only): local accessibility, frozen RNA-FM embeddings, ensembling.

    python -m guidefam.scripts.sirna_dev2 --oligo OLIGOFORMER/data --rnafm MODEL_DIR --emb-cache CACHE_DIR
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import cv
from .. import lm
from .. import sirna as SI
from .sirna_dev import chunk_cv, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oligo", required=True)
    ap.add_argument("--rnafm", required=True)
    ap.add_argument("--emb-cache", required=True)
    a = ap.parse_args()

    d = SI.load(f"{a.oligo}/Hu.csv")
    yb = d["label"].gt(0.7).astype(int).values
    X = SI.tabular(d)
    base = list(X.columns)
    acc = SI.accessibility(d)
    fl = SI.flank_frame(d)
    X = pd.concat([X, acc, fl], axis=1)
    fl_cols = list(fl.columns)
    es = lm.embed(d.sirna.tolist(), a.rnafm, a.emb_cache)
    ew = lm.embed(d.win.str.replace("X", "").tolist(), a.rnafm, a.emb_cache, max_len=80)
    E = pd.DataFrame(np.hstack([es, ew]), index=d.index, columns=[f"es{i}" for i in range(es.shape[1])] + [f"ew{i}" for i in range(ew.shape[1])])
    X = pd.concat([X, E], axis=1)
    acc_cols, es_cols, ew_cols = list(acc.columns), [c for c in E if c.startswith("es")], [c for c in E if c.startswith("ew")]

    oofs, rows = {}, {}

    def run(name, make):
        oofs[name] = chunk_cv(d, X, make)
        rows[name] = score(d.y.values, yb, oofs[name])

    run("ridge_base", lambda: cv.Ridge(X, cols=base, alpha=30.0))
    for al in (30, 100, 300):
        run(f"ridge_base+flank(a={al})", lambda al=al: cv.Ridge(X, cols=base + fl_cols, alpha=float(al)))
    run("ridge_base+acc", lambda: cv.Ridge(X, cols=base + acc_cols, alpha=30.0))
    for al in (300, 1000, 3000):
        run(f"ridge_base+emb_sirna(a={al})", lambda al=al: cv.Ridge(X, cols=base + es_cols, alpha=float(al)))
        run(f"ridge_base+emb_both(a={al})", lambda al=al: cv.Ridge(X, cols=base + es_cols + ew_cols, alpha=float(al)))
    gp = dict(n_jobs=1, n_estimators=300, num_leaves=7, min_child_samples=15)
    run("gbm_base", lambda: cv.GBM(X, cols=base, **gp))
    run("gbm_base+acc", lambda: cv.GBM(X, cols=base + acc_cols, **gp))
    run("gbm_base+flank", lambda: cv.GBM(X, cols=base + fl_cols, **gp))
    rk = lambda v: rankdata(v) / len(v)
    oofs["ens(ridge_base+acc, gbm_base+acc)"] = (rk(oofs["ridge_base+acc"]) + rk(oofs["gbm_base+acc"])) / 2
    rows["ens(ridge_base+acc, gbm_base+acc)"] = score(d.y.values, yb, oofs["ens(ridge_base+acc, gbm_base+acc)"])
    print(f"Huesken {len(d)} siRNAs, 10 contiguous chunks")
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
