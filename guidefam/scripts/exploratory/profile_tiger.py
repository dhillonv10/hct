import sys
import numpy as np
import pandas as pd

D = sys.argv[1]
for name in ["off-target", "hap-titration", "flow-cytometry"]:
    d = pd.read_csv(f"{D}/{name}.csv.gz")
    print(f"\n===== {name}  {d.shape}")
    print("guide_type:", d.guide_type.value_counts().to_dict())
    print("genes:", d.gene.nunique(), d.gene.value_counts().head(20).to_dict())
    print("guide_fold:", d.guide_fold.value_counts().sort_index().to_dict())
    print("target_fold:", d.target_fold.value_counts().sort_index().to_dict())
    pm = d[d.guide_type == "PM"]
    print("PM n =", len(pm), " guide_seq len:", pm.guide_seq.str.len().value_counts().to_dict(), " target_seq len:", pm.target_seq.str.len().value_counts().head(3).to_dict())
    L = pm[["lfc_r1", "lfc_r2", "lfc_r3"]]
    print("lfc replicate availability:", L.notna().sum().to_dict())
    c = L.corr()
    print("replicate Pearson:\n", c.round(3).to_string())
    m = L.mean(axis=1)
    print("mean-lfc quantiles:", m.quantile([.01, .1, .5, .9, .99]).round(3).to_dict())
    # ceiling: corr(single replicate, mean of the others)
    for i, r in enumerate(["lfc_r1", "lfc_r2", "lfc_r3"]):
        others = [x for x in ["lfc_r1", "lfc_r2", "lfc_r3"] if x != r]
        o = pm[others].mean(axis=1)
        ok = pm[r].notna() & o.notna()
        print(f"  {r} vs mean(others): r={np.corrcoef(pm.loc[ok, r], o[ok])[0, 1]:.3f}")
    print("PM per gene (fold):", pm.groupby("gene").agg(n=("guide_id", "size"), tf=("target_fold", "first")).T.to_string()[:900])
    if name == "off-target":
        print(d[d.guide_type != "PM"].head(3).T.to_string()[:2500])
