import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
x = pd.read_csv(f"{S}/results/final_cas13d/ext_preds_flow-cytometry.csv")
pos = x.guide_id.str.extract(r":(\d+)-")[0].astype(float)
x["block"] = x.gene + ":" + (pos // 50).astype(int).astype(str)
wg = lambda z, c: M.within_group(z, "gene", "act", c)
print("flow-cytometry (CD46/CD55/TFRC), models trained on HEK293T only; n =", len(x))
for a, b in [("gbm+tiger2d", "tiger2d_port"), ("gbm+tiger2d", "gbm"), ("tiger2d_port", "gbm"), ("ens3", "tiger2d_port"), ("ens3", "gbm+tiger2d")]:
    z = x[x[a].notna() & x[b].notna()].reset_index(drop=True)
    d0, lo, hi, sh = M.paired_bootstrap(z, a, b, wg, n=1000)
    print(f"  {a:14s} - {b:14s} {d0:+.3f} [{lo:+.3f},{hi:+.3f}]")
per = pd.DataFrame({c: {g: M.spearman(xg.act, xg[c]) for g, xg in x.groupby("gene")} for c in ["tiger2d_port", "gbm", "gbm+tiger2d", "konermann", "ens3"]})
print(per.round(3).to_string())
