"""ASO Atlas, DEVELOPMENT genes only, stage 2: Tier-1 net and the Tier-2 rank ensemble. Reads the GBM out-of-fold predictions written by aso_dev.

  python -m guidefam.scripts.aso_dev2 --csv data_untrusted/aso_zenodo_20794660/aso.csv.gz --cache data_untrusted/cache --out guidefam/results/aso

Same gene-group 5-fold split as aso_dev (sklearn GroupKFold is deterministic), so the members are directly comparable and can be rank-averaged.
Rank average = within-screen rank of each member's prediction, averaged (a screen is the unit that is ranked).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .. import aso as A
from .. import aso_nn as N
from .aso_access import load_cache


def screen_rank(d, col):
    return d.groupby("screen")[col].rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    d = A.load(a.csv)
    d = d[d.split == "dev"].reset_index(drop=True)
    o = pd.read_csv(os.path.join(a.out, "aso_dev_oof.csv"))
    assert len(o) == len(d) and (o.aso.values == d.aso.values).all() and (o.screen.values == d.screen.values).all(), "OOF file does not match the dev table"
    gbm = "p_base+thermo+access"
    d[gbm] = o[gbm].values
    acc, idx = load_cache(a.cache)
    X = A.features(d, acc, idx)
    T = N.tensors(*N.arrays(d, X, acc, idx))
    folds = list(GroupKFold(5).split(X, groups=d.group.values))
    groups, y = d.group.values, d.y.values
    p = np.zeros(len(d))
    for k, (tr, te) in enumerate(folds):
        t = time.time()
        ps = []
        for s in range(a.seeds):
            m = N.ASONet(T, max_epochs=a.epochs, seed=s).fit(tr, y, groups)
            ps.append(m.predict(te))
        p[te] = np.mean(ps, 0)
        print(f"fold {k}: {m.epochs_} epochs, {m.n_params} params, {time.time() - t:.0f}s", flush=True)
    d["p_net"] = p
    d["r_gbm"], d["r_net"] = screen_rank(d, gbm), screen_rank(d, "p_net")
    d["p_ens"] = (d.r_gbm + d.r_net) / 2

    print("\nDEV, gene groups held out (within-screen Spearman, mean over screens n>=20)")
    for nm, c in (("GBM (base+thermo+access)", gbm), ("Net (~1e5 params)", "p_net"), ("rank ensemble GBM+Net", "p_ens")):
        s = A.summarize(d, c)
        pt, lo, hi = A.bootstrap(d, c, n=a.boot)
        print(f"  {nm:28s} rho_screen {s['rho_screen']:.3f} [{lo:.3f},{hi:.3f}]  rho_group {s['rho_group']:.3f}", flush=True)
    for lab, c1, c2 in (("Net - GBM", "p_net", gbm), ("ensemble - GBM", "p_ens", gbm), ("ensemble - Net", "p_ens", "p_net")):
        pt, lo, hi = A.bootstrap(d, c1, c2, n=a.boot)
        print(f"  {lab:20s} {pt:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    d[["aso", "screen", "group", "y", gbm, "p_net", "p_ens"]].to_csv(os.path.join(a.out, "aso_dev_oof2.csv"), index=False)


if __name__ == "__main__":
    main()
