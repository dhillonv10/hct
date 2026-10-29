"""Cas13d on-target baseline ladder, leave-one-gene-out on the HEK293T screen.

    python -m guidefam.scripts.ladder_cas13d --data DATA --pred "TIGER/predictions (other models)" --out RESULTS [--models a,b,c]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import cv
from .. import data as D
from .. import features as F
from .. import metrics as M


def build(d, name, X_core, X_full):
    if name == "scalars_gbm":
        return lambda: cv.GBM(X_core, cols=[c for c in X_core.columns if c in F.SCALARS or c.startswith("junction")])
    if name == "seq_ridge":
        return lambda: cv.Ridge(X_core, cols=[c for c in X_core.columns if c not in F.SCALARS])
    if name == "gbm_core":
        return lambda: cv.GBM(X_core)
    if name == "gbm_full":
        return lambda: cv.GBM(X_full)
    if name == "tiger2d_port":
        from ..tiger_torch import Tiger2DRegressor
        return lambda: Tiger2DRegressor(flank=1, scalars=True)
    if name == "tiger2d_seqonly":
        from ..tiger_torch import Tiger2DRegressor
        return lambda: Tiger2DRegressor(flank=1, scalars=False)
    raise SystemExit(f"unknown model {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="scalars_gbm,seq_ridge,gbm_core,gbm_full,tiger2d_port")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)
    X_core, X_full = F.tabular(d, "core"), F.tabular(d, "full")
    oof_path = os.path.join(a.out, "oof_hek293t_logo.csv")
    oof = pd.read_csv(oof_path, index_col=0) if os.path.exists(oof_path) else pd.DataFrame(index=d.index)
    for name in a.models.split(","):
        if name in oof:
            print("have", name)
            continue
        t0 = time.time()
        print(f"[{name}]", flush=True)
        oof[name] = cv.logo_predict(d, build(d, name, X_core, X_full), verbose=(name.startswith("tiger")))
        oof.to_csv(oof_path)
        print(f"   {time.time() - t0:.0f}s", flush=True)

    ev = pd.concat([d[["gene", "block", "act", "active"]], d[["deepcas13", "konermann", "rf"]], oof], axis=1)
    rows = []
    for c in [c for c in ev.columns if c not in ("gene", "block", "act", "active")]:
        r = M.block_bootstrap(ev, lambda x, c=c: M.summarize(x, c), n=a.boot)
        rows.append({"model": c, **{k: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for k, v in r.items() if k != "n"}})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(a.out, "ladder_hek293t_logo.csv"), index=False)
    # per-gene within-gene Spearman, for the reader who wants to see the spread behind the mean
    pg = pd.DataFrame({c: {g: M.spearman(x.act, x[c]) for g, x in ev.groupby("gene")} for c in ev.columns if c not in ("gene", "block", "act", "active")})
    print(pg.round(2).to_string())


if __name__ == "__main__":
    main()
