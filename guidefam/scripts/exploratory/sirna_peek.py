import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import sirna as SI
O = sys.argv[1]
for n in ["Hu", "Mix", "Taka"]:
    d = SI.load(f"{O}/{n}.csv")
    d["grp"] = SI.target_groups(d.win)
    sz = d.grp.value_counts()
    print(f"\n== {n}: {len(d)} siRNAs, {d.grp.nunique()} target groups; group size min/median/max = {sz.min()}/{int(sz.median())}/{sz.max()}")
    print("label quantiles:", d.y.quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict(), " frac>0.7:", round((d.y > 0.7).mean(), 3))
    print("td columns:", sum(c.startswith('td') for c in d.columns), " dup siRNAs:", d.sirna.duplicated().sum())
    print("group sizes:", sz.sort_index().tolist()[:40])
