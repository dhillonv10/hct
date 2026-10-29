"""Two analyses with the FROZEN siRNA design (ridge a=100 + GBM on sequence, thermodynamic features and 8-nt flanks):

 1. How much does a shuffled 5-fold split (OligoFormer's intra-dataset protocol) flatter a model versus a split that keeps targets together?
    Same model, Huesken, three splits: shuffled 5-fold (their recipe), contiguous 5-fold, contiguous 10-fold.
 2. Leave-one-dataset-out: train on two of {Hu, Mix, Taka}, test on the third. Mix and Taka were already evaluated once (final_sirna.py); this adds
    training-set composition only, with no re-tuning, and all outcomes are reported.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from .. import cv
from .. import sirna as SI
from .final_sirna import metrics


def feats(d):
    X = SI.tabular(d)
    fl = SI.flank_frame(d, 8)
    return pd.concat([X, fl], axis=1), list(X.columns), list(fl.columns)


def fit_predict(tr, te, Xtr, Xte, cols):
    rk = lambda v: rankdata(v) / len(v)
    r = cv.Ridge(Xtr, cols=cols, alpha=100.0).fit(tr, tr.y.values)
    g = cv.GBM(Xtr, cols=cols, n_jobs=1, n_estimators=300, num_leaves=7, min_child_samples=15).fit(tr, tr.y.values)
    r.X = Xte
    g.X = Xte
    return (rk(r.predict(te)) + rk(g.predict(te))) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oligo", required=True)
    a = ap.parse_args()
    D = {n: SI.load(f"{a.oligo}/{n}.csv") for n in ("Hu", "Mix", "Taka")}

    d = D["Hu"]
    X, base, flc = feats(d)
    cols = base + flc
    yb = SI.binary(d, "intra")                       # train.py's label for Huesken (~50 % positives), so AUC/PRC are comparable with the paper
    print("=== 1. Huesken: does the split flatter the model? (same frozen model)")
    for label, k, shuf in (("shuffled 5-fold (OligoFormer recipe)", 5, True), ("contiguous 5-fold", 5, False), ("contiguous 10-fold", 10, False)):
        order = shuffle(np.arange(len(d)), random_state=42) if shuf else np.arange(len(d))
        fold = np.empty(len(d), int)
        fold[order] = np.arange(len(d)) * k // len(d)
        oof = np.full(len(d), np.nan)
        for f in range(k):
            te = fold == f
            oof[te] = fit_predict(d[~te], d[te], X, X, cols)
        m = metrics(d.y.values, yb, oof)
        print(f"  {label:38s} PCC {m['PCC']:.3f}  AUC {m['AUC']:.3f}  PRC {m['PRC']:.3f}")
    print("  (OligoFormer, published, shuffled 5-fold on Huesken: PCC 0.711  AUC 0.862  PRC 0.810)")

    print("\n=== 2. Leave-one-dataset-out (frozen design, no tuning)")
    XX = {}
    for n, dd in D.items():
        XX[n] = feats(dd)[0]
    for held in ("Mix", "Taka", "Hu"):
        trn = [n for n in D if n != held]
        tr = pd.concat([D[n].assign(_src=n) for n in trn], ignore_index=True)
        Xtr = pd.concat([XX[n] for n in trn], ignore_index=True)
        te, Xte = D[held], XX[held]
        p = fit_predict(tr, te, Xtr, Xte, cols)
        yb_te = SI.binary(te, "intra" if held == "Hu" else "inter")
        m = metrics(te.y.values, yb_te, p)
        only_hu = fit_predict(D["Hu"], te, XX["Hu"], Xte, cols) if held != "Hu" else None
        s = f"  train {'+'.join(trn):9s} -> test {held:5s}: PCC {m['PCC']:.3f}  AUC {m['AUC']:.3f}  PRC {m['PRC']:.3f}"
        if only_hu is not None:
            m0 = metrics(te.y.values, yb_te, only_hu)
            s += f"   | Hu only: PCC {m0['PCC']:.3f}  AUC {m0['AUC']:.3f}"
        print(s)


if __name__ == "__main__":
    main()
