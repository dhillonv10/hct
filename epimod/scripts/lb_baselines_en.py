"""Elastic-net baseline for LB-0195 (per-clock, on that clock's CpGs; missing values mean-imputed), the model class the
paper uses as its simple baseline. alpha / l1_ratio are chosen by cross-validation *grouped by study* inside the training data."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold

from epimod.age import age_to_t, t_to_age
from epimod.lb.store import lb_meta, load_lb_store, lb_series
from epimod.lb.tasks import CLOCKS, load_parsed, mae
from epimod.lb.train import task_sets
from epimod.train.loop import split_studies

parsed = load_parsed(); excl = lb_series(parsed)
meta_all = lb_meta(excl); tr_m, va_m = split_studies(meta_all, 0.15, 0)
tr = load_lb_store(exclude_series=excl, include_series=set(meta_all[tr_m].DatasetID))
va = load_lb_store(exclude_series=excl, include_series=set(meta_all[va_m].DatasetID))
uni = tr.universe; sets = task_sets(parsed, uni)
age_tr, age_va = tr.meta.Age.to_numpy(float), va.meta.Age.to_numpy(float)
groups = tr.meta.DatasetID.to_numpy()
c = parsed["LB-0195"]; clk = np.array(c.clock); age_c = np.array(c.age); pred = np.zeros(len(age_c)); res = {}
for name in CLOCKS:
    if name not in sets.clock_u:
        continue
    cols = sets.clock_u[name]; cids = list(uni[cols])
    X, Xv = tr.columns(cids), va.columns(cids)
    mean = np.where(np.isfinite(np.nanmean(X, 0)), np.nanmean(X, 0), 0.5); std = np.maximum(np.nanstd(X, 0), 0.02)
    z = lambda M: (np.where(np.isfinite(M), M, mean) - mean) / std
    cv = list(GroupKFold(3).split(X, groups=groups))
    m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=30, cv=cv, max_iter=3000, n_jobs=1, random_state=0).fit(z(X), age_to_t(age_tr))
    pos = {cid: j for j, cid in enumerate(cids)}
    for i in np.flatnonzero(clk == name):
        x = np.full(len(cids), np.nan, np.float32)
        for cid, b in zip(c.ids[i], c.beta[i]):
            if cid in pos:
                x[pos[cid]] = b
        pred[i] = t_to_age(m.predict(z(x[None, :])))[0]
    sel = clk == name
    res[name] = {"alpha": float(m.alpha_), "l1_ratio": float(m.l1_ratio_), "nonzero": int((m.coef_ != 0).sum()), "n_cpg": len(cids),
                 "val_mae": mae(t_to_age(m.predict(z(Xv))), age_va), "LB_mae": mae(pred[sel], age_c[sel])}
    print(name, json.dumps(res[name]), flush=True)
res["LB-0195/en_all"] = mae(pred, age_c)
print(f"LB-0195 per-clock elastic net: MAE {res['LB-0195/en_all']:.2f}   (paper: elastic net 8.93, best LLM 13.17)", flush=True)
out = sys.argv[1] if len(sys.argv) > 1 else "lb_baselines_en.json"
json.dump(res, open(out, "w"), indent=1); np.savez(out.replace(".json", "_preds.npz"), **{"LB-0195/en": pred})
