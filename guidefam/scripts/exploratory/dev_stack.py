import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import comparators as C, cv, data as D, features as F, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = C.add_all(D.load_tiger(f"{S}/data/tiger", "off-target"), f"{S}/ext/tiger/predictions (other models)")
X = F.tabular(d, "full", flank=2)
for c in ("konermann", "deepcas13"):
    X[f"pub_{c}"] = d.groupby("gene")[c].rank(pct=True)           # within-transcript percentile: computable for any transcript a user scores
dev = [g for g in sorted(d.gene.unique()) if g not in D.LOCKED_GENES]
path = f"{S}/results/dev/dev_oof.csv"
oof = pd.read_csv(path, index_col=0)
base_cols = [c for c in X.columns if not c.startswith("pub_")]
for nm, extra in {"stack_konermann": ["pub_konermann"], "stack_konermann+deepcas13": ["pub_konermann", "pub_deepcas13"]}.items():
    if nm not in oof:
        oof[nm] = cv.logo_predict(d, lambda extra=extra: cv.GBM(X, cols=base_cols + extra, n_jobs=1), genes=dev, verbose=False); oof.to_csv(path)
x = d[d.gene.isin(dev)].join(oof, how="inner")
x["gbm"] = x["base"]
rk = lambda c: x.groupby("gene")[c].rank(pct=True)
x["avg_gbm_konermann"] = (rk("gbm") + rk("konermann")) / 2
x["avg_stack_konermann"] = (rk("stack_konermann") + rk("konermann")) / 2
for c in ("gbm", "konermann", "avg_gbm_konermann", "stack_konermann", "stack_konermann+deepcas13", "avg_stack_konermann"):
    print(f"{c:28s} within-gene rho {M.within_group(x, 'gene', 'act', c):.4f}   auroc_within {M.within_group(x, 'gene', 'active', c, M.auroc):.4f}")
