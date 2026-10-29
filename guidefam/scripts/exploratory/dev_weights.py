import sys, time
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import cv, data as D, features as F, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = D.load_tiger(f"{S}/data/tiger", "off-target")
X = F.tabular(d, "full", flank=2)
dev = [g for g in sorted(d.gene.unique()) if g not in D.LOCKED_GENES]
rep = d[D.LFC]
d["rep_var"] = rep.var(axis=1, ddof=1).fillna(rep.var(axis=1, ddof=1).median())

class GBMV(cv.GBM):
    """GBM with optional replicate-reliability weights and/or a robust objective. Weights come from the training rows only."""
    def __init__(self, X, weight=None, objective=None, d=None, **kw):
        super().__init__(X, **kw); self.weight, self.d = weight, d
        if objective: self.params["objective"] = objective
    def fit(self, tr, ya):
        w = None
        if self.weight:
            v = self.d.loc[tr.index, "rep_var"].values; w = 1.0 / (1.0 + self.weight * v / np.median(v))
        self.m = self.lgb.LGBMRegressor(**self.params).fit(self.X.loc[tr.index, self.cols], ya, sample_weight=w); return self

path = f"{S}/results/dev/dev_oof.csv"; oof = pd.read_csv(path, index_col=0)
exps = {"w_rep2": dict(weight=2.0), "w_rep8": dict(weight=8.0), "huber": dict(objective="huber"), "w_rep4_huber": dict(weight=4.0, objective="huber")}
for nm, kw in exps.items():
    if nm in oof: continue
    t0 = time.time(); oof[nm] = cv.logo_predict(d, lambda kw=kw: GBMV(X, d=d, n_jobs=4, **kw), genes=dev, verbose=False); oof.to_csv(path); print(nm, f"{time.time()-t0:.0f}s", flush=True)
x = d[d.gene.isin(dev)].join(oof, how="inner")
ref = "base"
for c in ["base"] + list(exps):
    z = x.copy()
    s = M.within_group(z, "gene", "act", c)
    if c != ref:
        d0, lo, hi, _ = M.paired_bootstrap(z.reset_index(drop=True), c, ref, lambda q, col: M.within_group(q, "gene", "act", col), n=300)
        print(f"{c:14s} within-gene rho {s:.4f}   vs base {d0:+.4f} [{lo:+.4f},{hi:+.4f}]")
    else:
        print(f"{c:14s} within-gene rho {s:.4f}")
