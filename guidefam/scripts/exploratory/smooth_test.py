import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import data as D, metrics as M
d = D.load_tiger(sys.argv[1], "off-target")
oof = pd.read_csv(sys.argv[2], index_col=0)
dev = d[~d.gene.isin(D.LOCKED_GENES)].copy()
dev["p"] = oof["base"].reindex(dev.index)
def smooth(x, k, lam):
    x = x.sort_values("start"); s = x.start.values; p = x.p.values; out = np.empty(len(x))
    for i in range(len(x)):
        lo, hi = np.searchsorted(s, s[i] - k), np.searchsorted(s, s[i] + k, side="right")
        nb = np.r_[p[lo:i], p[i + 1:hi]]
        out[i] = p[i] if len(nb) == 0 else (1 - lam) * p[i] + lam * nb.mean()
    return pd.Series(out, index=x.index)
print("dev genes only (12); base within-gene rho =", round(M.within_group(dev, "gene", "act", "p"), 4))
for k in (1, 2, 3, 5, 8, 12):
    row = []
    for lam in (0.25, 0.5, 0.75, 1.0):
        dev["ps"] = pd.concat([smooth(x, k, lam) for _, x in dev.groupby("gene")])
        row.append(f"lam={lam}: {M.within_group(dev, 'gene', 'act', 'ps'):.4f}")
    print(f"k={k:>2}  " + "   ".join(row))
