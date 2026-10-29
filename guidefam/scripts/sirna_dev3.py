"""siRNA development, round 3 (Huesken contiguous-chunk CV only): flank length and the ridge+GBM rank ensemble. Last round before the design is frozen."""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import cv
from .. import sirna as SI
from .sirna_dev import chunk_cv, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oligo", required=True)
    a = ap.parse_args()
    d = SI.load(f"{a.oligo}/Hu.csv")
    yb = d["label"].gt(0.7).astype(int).values
    X = SI.tabular(d)
    base = list(X.columns)
    rows, oofs = {}, {}
    gp = dict(n_jobs=1, n_estimators=300, num_leaves=7, min_child_samples=15)
    rk = lambda v: rankdata(v) / len(v)
    for n in (4, 8, 12, 19):
        fl = SI.flank_frame(d, n)
        XX = pd.concat([X, fl], axis=1)
        cols = base + list(fl.columns)
        oofs[f"ridge n={n}"] = chunk_cv(d, XX, lambda: cv.Ridge(XX, cols=cols, alpha=100.0))
        oofs[f"gbm n={n}"] = chunk_cv(d, XX, lambda: cv.GBM(XX, cols=cols, **gp))
        oofs[f"ens n={n}"] = (rk(oofs[f"ridge n={n}"]) + rk(oofs[f"gbm n={n}"])) / 2
        for k in (f"ridge n={n}", f"gbm n={n}", f"ens n={n}"):
            rows[k] = score(d.y.values, yb, oofs[k])
    print(f"Huesken {len(d)} siRNAs, 10 contiguous chunks")
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
