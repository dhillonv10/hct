"""Audit: how much does early-stopping on the evaluation fold (the authors' cross-validation recipe) flatter the Tiger2D score?

For each listed held-out gene we train the same network twice on the other genes: (a) early stopping on an inner split of the training genes (clean),
(b) early stopping on the held-out gene itself and restoring its best epoch (the authors' predictor_validation.py). Same seed, same data otherwise.
"""
import argparse

import numpy as np
import pandas as pd

from .. import cv
from .. import data as D
from .. import metrics as M
from ..tiger_torch import Tiger2DRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--genes", default="CCT7,SF3A2,RPA1")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    d = D.load_tiger(a.data, "off-target")
    rows = []
    for g in a.genes.split(","):
        te = d.gene == g
        tr = d[~te]
        ytr = cv.activity_target(tr)
        yte = cv.activity_target(d[te])
        for seed in range(a.seeds):
            clean = Tiger2DRegressor(seed=seed).fit(tr, ytr)
            peek = Tiger2DRegressor(seed=seed).fit(tr, ytr, peek=(d[te], yte))
            r = {"gene": g, "seed": seed, "clean": M.spearman(d[te].act, clean.predict(d[te])), "peeking": M.spearman(d[te].act, peek.predict(d[te])),
                 "epochs_clean": clean.epochs_, "epochs_peek": peek.epochs_}
            rows.append(r)
            print(r, flush=True)
    t = pd.DataFrame(rows)
    print(t.groupby("gene")[["clean", "peeking"]].mean().round(3).to_string())
    print("mean over genes: clean %.3f  peeking %.3f  (difference %+.3f)" % (t.clean.mean(), t.peeking.mean(), (t.peeking - t.clean).mean()))


if __name__ == "__main__":
    main()
