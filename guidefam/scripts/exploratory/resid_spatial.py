import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import data as D, cv
from guidefam.metrics import spearman
d = D.load_tiger(sys.argv[1], "off-target")
oof = pd.read_csv(sys.argv[2], index_col=0)
dev = d[~d.gene.isin(D.LOCKED_GENES)].copy()
dev["p"] = oof["base"].reindex(dev.index)
dev["a"] = cv.activity_target(dev)           # per-gene normalised activity (dev genes only; analysis, not model selection)
dev["r"] = dev["a"] - dev.groupby("gene").p.transform(lambda x: x)  # placeholder, replaced below
# residual of activity after a per-gene linear fit on the prediction (removes calibration offset, keeps what the model misses)
res = []
for g, x in dev.groupby("gene"):
    b = np.polyfit(x.p, x.a, 1); res.append(pd.Series(x.a - np.polyval(b, x.p), index=x.index))
dev["r"] = pd.concat(res)
print("lag(nt)  corr(activity)  corr(residual)  n_pairs")
for lo, hi in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 100000)]:
    ca, cr, n = [], [], 0
    for g, x in dev.groupby("gene"):
        x = x.sort_values("start")
        s, a, r = x.start.values, x.a.values, x.r.values
        for i in range(0, len(x), 1):
            j0 = np.searchsorted(s, s[i] + lo); j1 = np.searchsorted(s, s[i] + hi, side="right")
            if j1 > j0:
                k = np.arange(j0, min(j1, j0 + 3))          # a few partners per guide keeps this fast
                ca += list(zip([a[i]] * len(k), a[k])); cr += list(zip([r[i]] * len(k), r[k])); n += len(k)
    ca, cr = np.array(ca), np.array(cr)
    print(f"{lo:>4}-{hi:<6} {np.corrcoef(ca.T)[0,1]:>10.3f} {np.corrcoef(cr.T)[0,1]:>14.3f} {n:>10d}")
