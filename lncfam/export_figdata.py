#!/usr/bin/env python3
"""Tidy CSVs for plotting. Every figure in the write-up should be drawable from these four files alone.

  python3 export_figdata.py --work /home/user/lncfam_data/rnu42/work --out fig_data

Writes:
  variants_scored.csv  one row per SNV: position, ref, alt, category, assay function score and q-value, the
                       paper's CADD and ViennaRNA ddG, and one column per language model. The source table for
                       scatter plots, heatmaps and positional tracks.
  metrics.csv          tidy metric table: (metric, score, value, lo, hi, n) for every contrast in the report, so
                       bar charts with intervals need no recomputation.
  roc_curves.csv       (contrast, score, fpr, tpr) for every score on both labelled contrasts.
  by_position.csv      per transcript position: mean assay score, mean of each predictor, and how many of the
                       three possible substitutions were called depleted. For the positional track.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from rnu42_sge import auroc, boot_auroc, boot_spearman, spearman

DEPLETED = -0.302                      # the paper's own call threshold, q < 0.01


def roc_points(y, x):
    """FPR/TPR at every threshold, ordered by descending score."""
    y, x = np.asarray(y), np.asarray(x, float)
    o = np.argsort(-x, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    tp = np.concatenate([[0], tp]) / max(y.sum(), 1)
    fp = np.concatenate([[0], fp]) / max((1 - y).sum(), 1)
    return fp, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="fig_data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v = pd.read_csv(os.path.join(a.work, "variants.csv"))
    models = [c for c in v.columns if c.startswith("lm_")]
    scores = [("CADD_score", "CADD v1.7"), ("ddG_bind", "ViennaRNA ddG")] + \
             [(c, c.replace("lm_", "RiNALMo-")) for c in models]

    v.to_csv(os.path.join(a.out, "variants_scored.csv"), index=False)

    v = v[v.function_score_mean.notna()].copy()
    v["depleted"] = (v.function_score_mean < DEPLETED).astype(int)
    lab = v[v.y.notna()].copy()
    lab["y"] = lab.y.astype(int)

    rows, roc = [], []
    contrasts = [("ReNU vs population", lab, lab.y.values),
                 ("depleted vs not", v, v.depleted.values)]
    for cname, frame, yy in contrasts:
        for col, name in scores:
            m = frame[frame[col].notna()]
            yv = yy[frame[col].notna().values]
            au = auroc(yv, m[col].values)
            lo, hi = boot_auroc(yv, m[col].values)
            rows.append(dict(metric=cname, score=name, value=au, lo=lo, hi=hi, n=len(m),
                             n_pos=int(yv.sum())))
            fpr, tpr = roc_points(yv, m[col].values)
            roc.append(pd.DataFrame(dict(contrast=cname, score=name, fpr=fpr, tpr=tpr)))
        # the assay itself, as the ceiling
        au = auroc(yy, -frame.function_score_mean.values)
        rows.append(dict(metric=cname, score="SGE assay (ceiling)", value=au, lo=np.nan, hi=np.nan,
                         n=len(frame), n_pos=int(yy.sum())))

    for col, name in scores:
        m = v[v[col].notna()]
        r = -spearman(m[col].values, m.function_score_mean.values)
        lo, hi = boot_spearman(m[col].values, m.function_score_mean.values)
        lo, hi = sorted([-lo, -hi])
        rows.append(dict(metric="Spearman vs function score", score=name, value=r, lo=lo, hi=hi,
                         n=len(m), n_pos=np.nan))

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "metrics.csv"), index=False)
    pd.concat(roc, ignore_index=True).to_csv(os.path.join(a.out, "roc_curves.csv"), index=False)

    agg = {"function_score_mean": "mean", "CADD_score": "mean", "ddG_bind": "mean", "depleted": "sum"}
    agg.update({c: "mean" for c in models})
    bypos = v.groupby("pos").agg(agg).reset_index().rename(columns={"depleted": "n_depleted_of_3"})
    bypos["n_variants"] = v.groupby("pos").size().values
    bypos.to_csv(os.path.join(a.out, "by_position.csv"), index=False)

    print(f"wrote 4 files to {a.out}/")
    for f in sorted(os.listdir(a.out)):
        print(f"  {f:24s} {os.path.getsize(os.path.join(a.out, f)):>8,d} bytes")


if __name__ == "__main__":
    main()
