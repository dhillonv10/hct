"""Audit: how much does TIGER-style de-normalisation with the held-out gene's OWN labels flatter a pooled correlation?

TIGER normalises each gene's LFCs by that gene's median and (q90 - q10), trains in that space, then maps predictions back with the same per-gene
numbers -- including for the held-out gene, so the held-out gene's label distribution leaks into its predictions. We take one set of leave-one-gene-out
predictions (in normalised space) and map them back to LFC two ways:
  oracle : the held-out gene's own median / scale (TIGER's recipe)
  fair   : one constant median / scale for every gene (the mean over the TRAINING genes), which is all a new gene would get
and score pooled Pearson / Spearman of observed LFC vs predicted LFC.   Within-gene metrics are identical in both (a per-gene affine map).
"""
import argparse

import numpy as np
import pandas as pd

from .. import cv
from .. import data as D
from .. import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--cols", default="gbm_core,gbm_full,seq_ridge")
    ap.add_argument("--genes", default="all", help="all | dev")
    a = ap.parse_args()
    d = D.load_tiger(a.data, "off-target")
    oof = pd.read_csv(a.oof, index_col=0)
    if a.genes == "dev":
        d = d[~d.gene.isin(D.LOCKED_GENES)]
    p = D.gene_norm_params(d)                                   # per-gene median ('loc') and q90-q10 ('scale') of the PM LFCs
    print(f"{'model':12s} {'protocol':8s} {'pooled Pearson':>15s} {'pooled Spearman':>16s} {'within-gene Spearman':>22s}")
    for c in a.cols.split(","):
        if c not in oof:
            continue
        x = d.join(oof[[c]], how="inner").dropna(subset=[c])
        loc, sc = x.gene.map(p["loc"]), x.gene.map(p["scale"])
        pred_oracle = loc - sc * x[c]                           # activity p is higher = more depleted
        fair = []
        for g, xg in x.groupby("gene"):
            others = p.drop(index=g)
            fair.append(pd.Series(others["loc"].mean() - others["scale"].mean() * xg[c], index=xg.index))
        pred_fair = pd.concat(fair).reindex(x.index)             # align on the row index (groups were concatenated in gene order)
        for nm, pr in (("oracle", pred_oracle), ("fair", pred_fair)):
            print(f"{c:12s} {nm:8s} {M.pearson(x.y, pr):15.3f} {M.spearman(x.y, pr):16.3f} {M.within_group(x.assign(q=-pr), 'gene', 'act', 'q'):22.3f}")


if __name__ == "__main__":
    main()
