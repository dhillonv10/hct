import sys, time
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import cv, data as D, features as F, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = D.load_tiger(f"{S}/data/tiger", "off-target")
X = F.tabular(d, "full", flank=2); base = list(X.columns)
T, Mo = F.thermo_frame(d), F.motif_frame(d)
X = pd.concat([X, T, Mo], axis=1)
dev = [g for g in sorted(d.gene.unique()) if g not in D.LOCKED_GENES]
print("has_TTTT fraction:", Mo.has_TTTT.mean().round(3), " mean activity |has_TTTT=1 vs 0 (z within gene):",
      d.assign(z=d.groupby("gene").act.transform(lambda s: (s - s.mean()) / s.std()), t=Mo.has_TTTT).groupby("t").z.mean().round(3).to_dict(), flush=True)
path = f"{S}/results/dev/dev_oof.csv"; oof = pd.read_csv(path, index_col=0)
exps = {"thermo": base + list(T.columns), "motif": base + list(Mo.columns), "thermo+motif": base + list(T.columns) + list(Mo.columns)}
for nm, cols in exps.items():
    if nm in oof: continue
    t0 = time.time(); oof[nm] = cv.logo_predict(d, lambda cols=cols: cv.GBM(X, cols=cols, n_jobs=4), genes=dev, verbose=False); oof.to_csv(path); print(nm, f"{time.time()-t0:.0f}s", flush=True)
x = d[d.gene.isin(dev)].join(oof, how="inner").reset_index(drop=True)
print(f"{'base':14s} within-gene rho {M.within_group(x, 'gene', 'act', 'base'):.4f}")
for c in exps:
    d0, lo, hi, _ = M.paired_bootstrap(x, c, "base", lambda q, col: M.within_group(q, "gene", "act", col), n=300)
    print(f"{c:14s} within-gene rho {M.within_group(x, 'gene', 'act', c):.4f}   vs base {d0:+.4f} [{lo:+.4f},{hi:+.4f}]")
