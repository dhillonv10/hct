import sys, time
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import cv, data as D, features as F, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = D.load_tiger(f"{S}/data/tiger", "off-target"); X = F.tabular(d, "full", flank=2)
g = "CCT7"; te = d.gene == g
for ne in (4, 8):
    t0 = time.time()
    m = cv.TabDPT(X, n_ensembles=ne).fit(d[~te], cv.activity_target(d[~te]))
    p = m.predict(d[te])
    print(f"TabDPT n_ensembles={ne}: held-out {g} within-gene rho {M.spearman(d[te].act, p):.3f}   {time.time()-t0:.0f}s   (GBM on this gene: 0.677, Tiger2D: 0.666)", flush=True)
