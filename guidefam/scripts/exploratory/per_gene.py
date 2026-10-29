import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import comparators as C, data as D, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = C.add_all(D.load_tiger(f"{S}/data/tiger", "off-target"), f"{S}/ext/tiger/predictions (other models)")
o = pd.read_csv(f"{S}/results/final_cas13d/hek_oof.csv", index_col=0)
x = d.join(o[["gbm", "tiger2d_port"]])
r = lambda c: x.groupby("gene")[c].rank(pct=True)
x["ens3"] = (r("gbm") + r("tiger2d_port") + r("konermann")) / 3
x["gbm+tiger"] = (r("gbm") + r("tiger2d_port")) / 2
t = pd.DataFrame({c: {g: M.spearman(xg.act, xg[c]) for g, xg in x.groupby("gene")} for c in ["tiger2d_port", "konermann", "gbm", "gbm+tiger", "ens3"]})
t["locked"] = [g in D.LOCKED_GENES for g in t.index]
t["ens3-tiger"] = t.ens3 - t.tiger2d_port
t["ens3-best_single"] = t.ens3 - t[["tiger2d_port", "konermann", "gbm"]].max(axis=1)
print(t.round(3).to_string())
print("\ngenes where ens3 > tiger2d:", int((t["ens3-tiger"] > 0).sum()), "/ 16   (locked:", int((t.loc[t.locked, "ens3-tiger"] > 0).sum()), "/ 4)")
print("genes where ens3 > the best single model:", int((t["ens3-best_single"] > 0).sum()), "/ 16")
print("mean per-gene delta vs tiger: %.3f   sd across genes: %.3f   (t-test p over genes)" % (t["ens3-tiger"].mean(), t["ens3-tiger"].std()))
from scipy.stats import ttest_1samp, wilcoxon
print("   one-sample t p = %.4f ; Wilcoxon signed-rank p = %.4f" % (ttest_1samp(t["ens3-tiger"], 0).pvalue, wilcoxon(t["ens3-tiger"]).pvalue))
