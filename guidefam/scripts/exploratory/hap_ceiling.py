import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import data as D, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
for name, mn in (("hap-titration", 5), ("off-target", 30), ("flow-cytometry", 30)):
    d = D.load_tiger(f"{S}/data/tiger", name)
    rho = []
    for r in D.LFC:
        others = [c for c in D.LFC if c != r]
        t = pd.DataFrame({"gene": d.gene, "a": -d[others].mean(axis=1), "p": -d[r]}).dropna()
        rho.append(M.within_group(t, "gene", "a", "p", min_n=mn))
    rho1 = float(np.nanmean(rho))
    k = d[D.LFC].notna().sum(axis=1).mean()
    rel = k * rho1 / (1 + (k - 1) * rho1)
    print(f"{name:15s} within-gene Spearman, one replicate vs mean of the others: {rho1:.3f}  (avg replicates/guide {k:.1f}) -> reliability of the mean ~ {rel:.2f}, max attainable corr ~ {np.sqrt(rel):.2f}")
    print(f"{'':15s} spread of mean LFC within a gene: median std {d.groupby('gene').y.std().median():.2f}; fraction of guides below -2: {(d.y < -2).mean():.2f}")
