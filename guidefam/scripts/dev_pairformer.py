"""Development run of MiniPairFormer (12 development genes only): with / without the RNA-structure attention bias, plus value as an ensemble member.

    python -m guidefam.scripts.dev_pairformer --data DATA --pairs pairs_off-target.npz --out OUT --variants pf_struct,pf_nostruct
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import cv
from .. import data as D
from .. import metrics as M
from ..pairformer import PairFormerRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="pf_struct,pf_nostruct")
    ap.add_argument("--tiger-oof", default=None)
    ap.add_argument("--gbm-oof", default=None)
    ap.add_argument("--genes", default="dev")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)
    z = np.load(a.pairs)
    P, U = z["P"], z["U"]
    genes = [g for g in sorted(d.gene.unique()) if (g not in D.LOCKED_GENES if a.genes == "dev" else True)]
    path = os.path.join(a.out, "pf_oof.csv")
    oof = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame(index=d.index)
    npar = {}
    for v in a.variants.split(","):
        if v in oof and oof.loc[d.gene.isin(genes), v].notna().all():
            continue
        t0 = time.time()
        mk = lambda v=v: PairFormerRegressor(P=P, U=U, use_structure=(v == "pf_struct"), scalars=(v != "pf_seqonly"))
        p = cv.logo_predict(d, mk, genes=genes, verbose=False)
        oof[v] = np.where(np.isnan(p), oof[v] if v in oof else np.nan, p)
        oof.to_csv(path)
        m = mk().fit(d[d.gene != genes[0]].iloc[:2000], cv.activity_target(d[d.gene != genes[0]].iloc[:2000]))
        print(f"[{v}] {time.time() - t0:.0f}s total for {len(genes)} folds, parameters = {m.n_params}", flush=True)
    ev = d[d.gene.isin(genes)].join(oof)
    if a.tiger_oof:
        ev["tiger2d"] = pd.read_csv(a.tiger_oof, index_col=0)["tiger2d_port"].reindex(ev.index)
    if a.gbm_oof:
        ev["gbm"] = pd.read_csv(a.gbm_oof, index_col=0)["gbm"].reindex(ev.index)
    rk = lambda c: ev.groupby("gene")[c].rank(pct=True)
    print(f"\n{len(genes)} genes ({a.genes}); within-gene Spearman")
    for c in [c for c in ["gbm", "tiger2d", "konermann"] + a.variants.split(",") if c in ev]:
        print(f"  {c:12s} {M.within_group(ev, 'gene', 'act', c):.4f}")
    if all(c in ev for c in ("gbm", "tiger2d", "konermann")):
        ev["ens3"] = (rk("gbm") + rk("tiger2d") + rk("konermann")) / 3
        print(f"  {'ens3':12s} {M.within_group(ev, 'gene', 'act', 'ens3'):.4f}")
        for v in a.variants.split(","):
            if v in ev:
                ev["e4"] = (rk("gbm") + rk("tiger2d") + rk("konermann") + rk(v)) / 4
                ev["e2"] = (rk("gbm") + rk(v)) / 2
                print(f"  ens3 + {v:10s} {M.within_group(ev, 'gene', 'act', 'e4'):.4f}     gbm + {v:10s} {M.within_group(ev, 'gene', 'act', 'e2'):.4f}")


if __name__ == "__main__":
    main()
