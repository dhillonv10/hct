"""Sample-level routed experts on the fixed 353-CpG panel (LB-0190): a latent group is inferred from the profile itself
(Gaussian mixture on PCA scores of the panel, no tissue / platform labels), and each group has its own ridge expert.
Prediction = sum_k r_k(x) * expert_k(x).  K and alpha are chosen on held-out validation SERIES; LB-0190 is scored once."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture

from epimod.age import age_to_t, t_to_age
from epimod.lb.store import lb_meta, load_lb_store, lb_series
from epimod.lb.tasks import load_parsed, mae
from epimod.train.loop import split_studies

parsed = load_parsed(); excl = lb_series(parsed)
meta = lb_meta(excl); tr_m, va_m = split_studies(meta, 0.15, 0)
tr = load_lb_store(exclude_series=excl, include_series=set(meta[tr_m].DatasetID))
va = load_lb_store(exclude_series=excl, include_series=set(meta[va_m].DatasetID))
t = parsed["LB-0190"]; ids = list(t.ids[0]); Xlb = np.array(t.beta, np.float32); age_lb = np.array(t.age)
Xtr, Xva = tr.columns(ids), va.columns(ids)
age_tr, age_va = tr.meta.Age.to_numpy(float), va.meta.Age.to_numpy(float)
mean = np.nanmean(Xtr, 0); mean = np.where(np.isfinite(mean), mean, 0.5); std = np.maximum(np.nanstd(Xtr, 0), 0.02)
z = lambda X: (np.where(np.isfinite(X), X, mean) - mean) / std
Ztr, Zva, Zlb = z(Xtr), z(Xva), z(Xlb)
ytr = age_to_t(age_tr)
pca = PCA(12, random_state=0).fit(Ztr)


def fit_predict(K, alpha, targets):
    if K == 1:
        m = Ridge(alpha=alpha).fit(Ztr, ytr)
        return [t_to_age(m.predict(Z)) for Z in targets]
    g = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=2).fit(pca.transform(Ztr))
    R = g.predict_proba(pca.transform(Ztr))
    experts = [Ridge(alpha=alpha).fit(Ztr, ytr, sample_weight=R[:, k] + 1e-6) for k in range(K)]
    out = []
    for Z in targets:
        r = g.predict_proba(pca.transform(Z)); mu = sum(r[:, k] * experts[k].predict(Z) for k in range(K))
        out.append(t_to_age(mu))
    return out


res, best = [], None
for K in (1, 2, 3, 4, 6, 8, 12):
    for alpha in (30.0, 300.0, 3000.0):
        pv, = fit_predict(K, alpha, [Zva]); v = mae(pv, age_va)
        res.append({"K": K, "alpha": alpha, "val_mae": v}); print(f"K={K:2d} alpha={alpha:7.0f}: val MAE {v:.2f}", flush=True)
        if best is None or v < best[0]:
            best = (v, K, alpha)
v, K, alpha = best
base_alpha = min((r for r in res if r["K"] == 1), key=lambda r: r["val_mae"])["alpha"]
pl_routed, = fit_predict(K, alpha, [Zlb]); pl_global, = fit_predict(1, base_alpha, [Zlb])
out = {"grid": res, "selected": {"K": K, "alpha": alpha, "val_mae": v}, "global_alpha": base_alpha,
       "LB-0190_routed": mae(pl_routed, age_lb), "LB-0190_global_ridge": mae(pl_global, age_lb)}
print(json.dumps({k: v for k, v in out.items() if k != "grid"}), flush=True)
np.savez(sys.argv[1] if len(sys.argv) > 1 else "routed.npz", **{"LB-0190/pred": pl_routed, "LB-0190/ridge": pl_global})
json.dump(out, open((sys.argv[1] if len(sys.argv) > 1 else "routed.npz").replace(".npz", ".json"), "w"), indent=1)
