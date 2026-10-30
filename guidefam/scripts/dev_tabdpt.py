"""Development run: TabDPT (pretrained tabular foundation model, open weights, not trained on biology) as the learner on the Cas13d tabular features. 12 development genes only."""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import cv
from .. import data as D
from .. import features as F
from .. import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiger-oof", required=True)
    ap.add_argument("--gbm-oof", required=True)
    ap.add_argument("--n-ens", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)
    X = F.tabular(d, "full", flank=2)
    genes = [g for g in sorted(d.gene.unique()) if g not in D.LOCKED_GENES]
    path = os.path.join(a.out, "tabdpt_oof.csv")
    oof = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame(index=d.index)
    if "tabdpt" not in oof or oof.loc[d.gene.isin(genes), "tabdpt"].isna().any():
        t0 = time.time()
        p = cv.logo_predict(d, lambda: cv.TabDPT(X, n_ensembles=a.n_ens), genes=genes, verbose=True)
        oof["tabdpt"] = np.where(np.isnan(p), oof["tabdpt"] if "tabdpt" in oof else np.nan, p)
        oof.to_csv(path)
        print(f"tabdpt {time.time() - t0:.0f}s for {len(genes)} folds", flush=True)
    ev = d[d.gene.isin(genes)].join(oof)
    ev["tiger2d"] = pd.read_csv(a.tiger_oof, index_col=0)["tiger2d_port"].reindex(ev.index)
    ev["gbm"] = pd.read_csv(a.gbm_oof, index_col=0)["gbm"].reindex(ev.index)
    rk = lambda c: ev.groupby("gene")[c].rank(pct=True)
    print("\n12 development genes; within-gene Spearman")
    for c in ("gbm", "tiger2d", "konermann", "tabdpt"):
        print(f"  {c:10s} {M.within_group(ev, 'gene', 'act', c):.4f}")
    ev["ens3"] = (rk("gbm") + rk("tiger2d") + rk("konermann")) / 3
    ev["ens4"] = (rk("gbm") + rk("tiger2d") + rk("konermann") + rk("tabdpt")) / 4
    ev["gbm_tabdpt"] = (rk("gbm") + rk("tabdpt")) / 2
    ev["tiger_tabdpt"] = (rk("tiger2d") + rk("tabdpt")) / 2
    for c in ("ens3", "ens4", "gbm_tabdpt", "tiger_tabdpt"):
        print(f"  {c:12s} {M.within_group(ev, 'gene', 'act', c):.4f}")


if __name__ == "__main__":
    main()
