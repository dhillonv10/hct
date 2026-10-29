import sys, time
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from guidefam import cv, sirna as SI
from guidefam.scripts.sirna_dev import chunk_cv, score
O = sys.argv[1]
d = SI.load(f"{O}/Hu.csv")
yb = SI.binary(d, "intra")
X = SI.tabular(d); base = list(X.columns); fl = SI.flank_frame(d, 8); X = pd.concat([X, fl], axis=1); cols = base + list(fl.columns)
rk = lambda v: rankdata(v) / len(v)
oofs, rows = {}, {}
for nm, mk in {"ridge": lambda: cv.Ridge(X, cols=cols, alpha=100.0),
               "gbm": lambda: cv.GBM(X, cols=cols, n_jobs=1, n_estimators=300, num_leaves=7, min_child_samples=15),
               "tabdpt": lambda: cv.TabDPT(X, cols=cols, n_ensembles=4),
               "tabdpt_noflank": lambda: cv.TabDPT(X, cols=base, n_ensembles=4)}.items():
    t0 = time.time(); oofs[nm] = chunk_cv(d, X, mk); rows[nm] = score(d.y.values, yb, oofs[nm]); print(nm, f"{time.time()-t0:.0f}s", flush=True)
oofs["ens(ridge,gbm)"] = (rk(oofs["ridge"]) + rk(oofs["gbm"])) / 2
oofs["ens(ridge,gbm,tabdpt)"] = (rk(oofs["ridge"]) + rk(oofs["gbm"]) + rk(oofs["tabdpt"])) / 3
oofs["ens(gbm,tabdpt)"] = (rk(oofs["gbm"]) + rk(oofs["tabdpt"])) / 2
for k in ("ens(ridge,gbm)", "ens(ridge,gbm,tabdpt)", "ens(gbm,tabdpt)"): rows[k] = score(d.y.values, yb, oofs[k])
print("Huesken contiguous 10-chunk CV; AUC/PRC use the file's binary label (~50% positive)")
print(pd.DataFrame(rows).T.round(3).to_string())
