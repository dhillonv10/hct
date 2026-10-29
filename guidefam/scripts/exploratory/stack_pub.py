import sys, itertools
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import comparators as C, data as D, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = C.add_all(D.load_tiger(f"{S}/data/tiger", "off-target"), f"{S}/ext/tiger/predictions (other models)")
oof = pd.read_csv(f"{S}/results/dev/dev_oof.csv", index_col=0)
dev = d[~d.gene.isin(D.LOCKED_GENES)].join(oof[["base"]], how="inner").rename(columns={"base": "gbm"})
rk = lambda c: dev.groupby("gene")[c].rank(pct=True)
for c in ("gbm", "konermann", "rf", "deepcas13"): dev["r_" + c] = rk(c)
def wg(col): return M.within_group(dev, "gene", "act", col)
print("12 development genes; within-gene Spearman")
for c in ("gbm", "konermann", "rf", "deepcas13"): print(f"  {c:28s} {wg(c):.4f}")
for combo in [("gbm", "konermann"), ("gbm", "rf"), ("gbm", "deepcas13"), ("gbm", "konermann", "rf"), ("gbm", "konermann", "rf", "deepcas13"), ("konermann", "rf")]:
    dev["e"] = sum(dev["r_" + c] for c in combo) / len(combo)
    print(f"  {'+'.join(combo):28s} {wg('e'):.4f}")
print("correlation of within-gene ranks (dev genes): ")
print(dev[["r_gbm", "r_konermann", "r_rf", "r_deepcas13"]].corr(method="spearman").round(2).to_string())
