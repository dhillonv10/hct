"""Routed ridge experts for LB-0195 (per clock): latent group from the clock's own CpG profile (GMM on PCA scores),
one ridge expert per group, mean-imputation for sites the prompt does not show.  (K, alpha) per clock are chosen on
held-out validation series; LB-0195 is scored once."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture

from epimod.age import age_to_t, t_to_age
from epimod.lb.store import lb_meta, load_lb_store, lb_series
from epimod.lb.tasks import CLOCKS, load_parsed, mae
from epimod.lb.train import task_sets
from epimod.train.loop import split_studies

parsed = load_parsed(); excl = lb_series(parsed)
meta = lb_meta(excl); tr_m, va_m = split_studies(meta, 0.15, 0)
tr = load_lb_store(exclude_series=excl, include_series=set(meta[tr_m].DatasetID))
va = load_lb_store(exclude_series=excl, include_series=set(meta[va_m].DatasetID))
uni = tr.universe; sets = task_sets(parsed, uni)
age_tr, age_va = tr.meta.Age.to_numpy(float), va.meta.Age.to_numpy(float); ytr = age_to_t(age_tr)
c = parsed["LB-0195"]; clk = np.array(c.clock); age_c = np.array(c.age)
pred_r, pred_g, res = np.zeros(len(age_c)), np.zeros(len(age_c)), {}


class Routed:
    def __init__(self, Z, y, K, alpha):
        self.K = K
        if K == 1:
            self.m = [Ridge(alpha=alpha).fit(Z, y)]; return
        self.pca = PCA(min(12, Z.shape[1] - 1), random_state=0).fit(Z)
        self.g = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=2).fit(self.pca.transform(Z))
        R = self.g.predict_proba(self.pca.transform(Z))
        self.m = [Ridge(alpha=alpha).fit(Z, y, sample_weight=R[:, k] + 1e-6) for k in range(K)]

    def predict(self, Z):
        if self.K == 1:
            return t_to_age(self.m[0].predict(Z))
        r = self.g.predict_proba(self.pca.transform(Z))
        return t_to_age(sum(r[:, k] * self.m[k].predict(Z) for k in range(self.K)))


for name in CLOCKS:
    if name not in sets.clock_u:
        continue
    cids = list(uni[sets.clock_u[name]])
    X, Xv = tr.columns(cids), va.columns(cids)
    mean = np.nanmean(X, 0); mean = np.where(np.isfinite(mean), mean, 0.5); std = np.maximum(np.nanstd(X, 0), 0.02)
    z = lambda M: (np.where(np.isfinite(M), M, mean) - mean) / std
    Z, Zv = z(X), z(Xv)
    grid, best = [], None
    for K in (1, 4, 6, 8):
        for alpha in ((30.0, 300.0, 3000.0) if len(cids) < 500 else (300.0, 3000.0, 30000.0)):
            v = mae(Routed(Z, ytr, K, alpha).predict(Zv), age_va); grid.append((K, alpha, v))
            if best is None or v < best[0]:
                best = (v, K, alpha)
    g1 = min((g for g in grid if g[0] == 1), key=lambda g: g[2])
    mr, mg = Routed(Z, ytr, best[1], best[2]), Routed(Z, ytr, 1, g1[1])
    pos = {cid: j for j, cid in enumerate(cids)}; rows = np.flatnonzero(clk == name)
    Xi = np.full((len(rows), len(cids)), np.nan, np.float32)
    for r, i in enumerate(rows):
        for cid, b in zip(c.ids[i], c.beta[i]):
            if cid in pos:
                Xi[r, pos[cid]] = b
    pred_r[rows], pred_g[rows] = mr.predict(z(Xi)), mg.predict(z(Xi))
    res[name] = {"K": best[1], "alpha": best[2], "val_mae_routed": best[0], "val_mae_global": g1[2],
                 "LB_routed": mae(pred_r[rows], age_c[rows]), "LB_global": mae(pred_g[rows], age_c[rows])}
    print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()}), flush=True)
res["LB-0195/routed"], res["LB-0195/global"] = mae(pred_r, age_c), mae(pred_g, age_c)
print(f"LB-0195 routed ridge experts: MAE {res['LB-0195/routed']:.2f} | global ridge {res['LB-0195/global']:.2f}   (paper: EN 8.93, best LLM 13.17)", flush=True)
out = sys.argv[1] if len(sys.argv) > 1 else "routed_clock.npz"
np.savez(out, **{"LB-0195/pred": pred_r, "LB-0195/ridge": pred_g}); json.dump(res, open(out.replace(".npz", ".json"), "w"), indent=1)
