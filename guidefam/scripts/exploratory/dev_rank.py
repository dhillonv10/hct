import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import cv, data as D, features as F, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = D.load_tiger(f"{S}/data/tiger", "off-target")
X = F.tabular(d, "full", flank=2)
dev = [g for g in sorted(d.gene.unique()) if g not in D.LOCKED_GENES]
path = f"{S}/results/dev/dev_oof.csv"
oof = pd.read_csv(path, index_col=0)
exps = {
  "lambdarank_5": lambda: cv.RankGBM(X, n_levels=5),
  "lambdarank_10": lambda: cv.RankGBM(X, n_levels=10),
  "gbm_bag5": None,
}
for nm, mk in exps.items():
    if nm in oof or mk is None: continue
    oof[nm] = cv.logo_predict(d, mk, genes=dev, verbose=False); oof.to_csv(path)
# 5-seed bagged regression GBM
if "gbm_bag5" not in oof:
    ps = [cv.logo_predict(d, lambda s=s: cv.GBM(X, seed=s, n_jobs=1), genes=dev, verbose=False) for s in range(5)]
    oof["gbm_bag5"] = np.nanmean(ps, axis=0); oof.to_csv(path)
x = d[d.gene.isin(dev)].join(oof, how="inner")
for c in ["base", "lambdarank_5", "lambdarank_10", "gbm_bag5"]:
    print(f"{c:15s} within-gene rho {M.within_group(x, 'gene', 'act', c):.4f}  auroc_within {M.within_group(x, 'gene', 'active', c, M.auroc):.4f}")
