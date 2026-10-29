import sys, itertools
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import comparators as C, data as D, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = C.add_all(D.load_tiger(f"{S}/data/tiger", "off-target"), f"{S}/ext/tiger/predictions (other models)")
oof = pd.read_csv(f"{S}/results/dev/dev_oof.csv", index_col=0)[["base"]].rename(columns={"base": "gbm"})
tg = pd.read_csv(f"{S}/results/tiger/oof_hek293t_logo.csv", index_col=0)[["tiger2d_port"]].rename(columns={"tiger2d_port": "tiger"})
x = d[~d.gene.isin(D.LOCKED_GENES)].join(oof, how="inner").join(tg, how="inner")
for c in ("gbm", "tiger", "konermann", "rf"): x["r_" + c] = x.groupby("gene")[c].rank(pct=True)
wg = lambda col: M.within_group(x, "gene", "act", col)
print("12 development genes (locked genes untouched); within-gene Spearman")
for c in ("gbm", "tiger", "konermann", "rf"): print(f"  {c:26s} {wg(c):.4f}")
print("rank correlations:\n", x[["r_gbm", "r_tiger", "r_konermann", "r_rf"]].corr(method="spearman").round(2).to_string())
for combo in [("gbm", "konermann"), ("gbm", "tiger"), ("tiger", "konermann"), ("gbm", "tiger", "konermann"), ("gbm", "tiger", "konermann", "rf")]:
    x["e"] = sum(x["r_" + c] for c in combo) / len(combo)
    print(f"  {'+'.join(combo):26s} {wg('e'):.4f}")
