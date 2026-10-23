"""Extremes baselines (LB-0010 age MAE, LB-0002 decade balanced accuracy): ridge on multi-hot / multi-hot x value features.

Same training data, pools and validation series as lb_run.py. Hyper-parameters (alpha, values on/off) are chosen on the
held-out validation *series*; the LongevityBench items are then scored once, with the chosen setting only."""
import json, sys
import numpy as np

from epimod.age import age_to_t
from epimod.lb.baselines import BagOfCpGs
from epimod.lb.store import lb_meta, load_lb_store, lb_series
from epimod.lb.tasks import (balanced_accuracy, build_vocab, decade_label_to_id, decade_posterior, eval_single, extremes_batch,
                             extremes_cache, load_parsed, make_pools, mae)
from epimod.train.loop import split_studies

parsed = load_parsed(); excl = lb_series(parsed)
meta_all = lb_meta(excl); tr_m, va_m = split_studies(meta_all, 0.15, 0)
tr = load_lb_store(exclude_series=excl, include_series=set(meta_all[tr_m].DatasetID))
va = load_lb_store(exclude_series=excl, include_series=set(meta_all[va_m].DatasetID))
uni = tr.universe
odd = np.arange(500) % 2 == 1
pools = make_pools(parsed, uni, ext_mask=odd, pair_mask=odd); vocab = build_vocab(tr, parsed, pools)
rng = np.random.RandomState(0)


def feats(store, cache, aug):
    b = extremes_batch(store, vocab, cache, np.arange(store.n), 0.0, rng, 0.04 if aug else 0.0)
    return b["cpg"].numpy(), b["role"].numpy(), b["sex"].numpy(), b["beta"].numpy()


Ftr, Fva = feats(tr, extremes_cache(tr, pools), True), feats(va, extremes_cache(va, pools), False)
age_tr, age_va = tr.meta.Age.to_numpy(float), va.meta.Age.to_numpy(float)
V = len(vocab)
grid, best = [], None
for values in (False, True):
    for alpha in (3.0, 10.0, 30.0, 100.0, 300.0):
        m = BagOfCpGs(alpha, values).fit(Ftr[0], Ftr[1], Ftr[2], age_tr, V, Ftr[3])
        v = mae(m.predict(*Fva[:3], Fva[3]), age_va)
        grid.append({"values": values, "alpha": alpha, "val_mae": v}); print(f"val MAE  values={values!s:5} alpha={alpha:6.1f}: {v:.2f}", flush=True)
        if best is None or v < best[0]:
            best = (v, m, values, alpha)
v, m, values, alpha = best
sigma = float(np.std(age_to_t(age_va) - m.predict_t(*Fva[:3], Fva[3])))
print(f"selected on validation series: values={values} alpha={alpha} (val MAE {v:.2f}, sigma_t {sigma:.3f})", flush=True)

out = {"selected": {"values": values, "alpha": alpha, "val_mae": v}, "grid": grid}
for lb in ("LB-0010", "LB-0002"):
    t = parsed[lb]; b = eval_single(t, vocab, uni); age = np.array(t.age)
    mu = m.predict_t(b["cpg"].numpy(), b["role"].numpy(), b["sex"].numpy(), b["beta"].numpy())
    o = np.arange(len(age)) % 2 == 1; splits = {"all": np.ones(len(age), bool), "even": ~o, "odd": o}
    if lb == "LB-0010":
        from epimod.age import t_to_age
        pred = t_to_age(mu); out[lb] = {k: mae(pred[s], age[s]) for k, s in splits.items()}
    else:
        true = np.array([decade_label_to_id(l) for l in t.label]); post = decade_posterior(mu, np.full(len(mu), sigma)).argmax(1)
        out[lb] = {k: balanced_accuracy(post[s], true[s]) for k, s in splits.items()}
    print(lb, json.dumps(out[lb]), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "lb_baselines_ext.json", "w"), indent=1)
