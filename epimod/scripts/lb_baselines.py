"""Simple baselines on exactly the inputs of each LongevityBench DNAm task (same training data / pools as lb_run.py)."""
import json, sys, time
import numpy as np, pandas as pd

from epimod.lb.baselines import DenseClock, PairLogit
from epimod.lb.store import load_lb_store, lb_series, sample_pairs
from epimod.lb.tasks import CLOCKS, build_vocab, eval_pair, load_parsed, make_pools, mae, pair_batch
from epimod.lb.train import task_sets
from epimod.train.loop import split_studies

parsed = load_parsed(); store = load_lb_store(exclude_series=lb_series(parsed)); uni = store.universe
tr_m, va_m = split_studies(store.meta, 0.15, 0)
tr, va = store.subset(tr_m), store.subset(va_m)
odd = np.arange(500) % 2 == 1
pools = make_pools(parsed, uni, ext_mask=odd, pair_mask=odd); vocab = build_vocab(store, parsed, pools); sets = task_sets(parsed, uni)
age_tr, age_va = tr.meta.Age.to_numpy(float), va.meta.Age.to_numpy(float)
res = {}

# ---- LB-0190: fixed 353 CpGs
ids353 = list(parsed["LB-0190"].ids[0])
Xtr, Xva = tr.columns(ids353), va.columns(ids353)
t = parsed["LB-0190"]; Xlb = np.array(t.beta); assert list(t.ids[0]) == ids353; age_lb = np.array(t.age)
for kind in ("ridge", "hgb"):
    m = DenseClock(kind).fit(Xtr, age_tr)
    res[f"LB-0190/{kind}"] = {"my_val_mae": mae(m.predict(Xva), age_va), "LB_mae": mae(m.predict(Xlb), age_lb)}
    print(f"LB-0190 {kind}: my-val MAE {res[f'LB-0190/{kind}']['my_val_mae']:.2f} | LB eval MAE {res[f'LB-0190/{kind}']['LB_mae']:.2f}   (paper: ridge 4.30, best LLM 5.82, Horvath-as-is 9.5)", flush=True)

# ---- LB-0195: per-clock ridge, mean-imputing CpGs that the prompt does not show
c = parsed["LB-0195"]; clk = np.array(c.clock); age_c = np.array(c.age); pred = np.zeros(len(age_c))
for name in CLOCKS:
    if name not in sets.clock_u: continue
    cols = sets.clock_u[name]; cids = list(uni[cols])
    m = DenseClock("ridge").fit(tr.columns(cids), age_tr)
    pos = {cid: j for j, cid in enumerate(cids)}
    for i in np.flatnonzero(clk == name):
        x = np.full(len(cids), np.nan, np.float32)
        for cid, b in zip(c.ids[i], c.beta[i]):
            if cid in pos: x[pos[cid]] = b
        pred[i] = m.predict(x[None, :])[0]
    sel = clk == name; res[f"LB-0195/{name}"] = mae(pred[sel], age_c[sel])
res["LB-0195/ridge_all"] = mae(pred, age_c)
print(f"LB-0195 per-clock ridge: MAE {res['LB-0195/ridge_all']:.2f}   (paper: elastic net 8.93, best LLM 13.17)", flush=True)
print("   per clock:", {k.split('/')[1]: round(v, 2) for k, v in res.items() if k.startswith('LB-0195/') and k != 'LB-0195/ridge_all'}, flush=True)

# ---- LB-0006: pair logistic regression on listed differences
rng = np.random.RandomState(0)
ia, ib = sample_pairs(age_tr, 60000, rng, 10.0, tr.meta.DatasetID.to_numpy())
cpg, ba, bb = [], [], []
for k in range(0, len(ia), 500):
    va_, vb_ = pair_batch(tr, vocab, ia[k:k+500], ib[k:k+500], 160, pools.pair)
    cpg.append(va_["cpg"].numpy()); ba.append(va_["beta"].numpy()); bb.append(vb_["beta"].numpy())
cpg, ba, bb = np.concatenate(cpg), np.concatenate(ba), np.concatenate(bb)
y = (age_tr[ia] > age_tr[ib]).astype(np.float32)
t6 = parsed["LB-0006"]; a_, b_ = eval_pair(t6, vocab, uni); truth = np.array(t6.label) == "A"
for C in (0.05, 0.5, 5.0):
    m = PairLogit(C).fit(cpg, ba, bb, y, len(vocab))
    logit = m.predict_logit(a_["cpg"].numpy(), a_["beta"].numpy(), b_["beta"].numpy())
    ok = (logit > 0) == truth
    res[f"LB-0006/logit_C{C}"] = {"all": float(ok.mean()), "even": float(ok[~odd].mean()), "odd": float(ok[odd].mean())}
    print(f"LB-0006 pair-logit C={C}: concordance all {ok.mean():.3f} | even {ok[~odd].mean():.3f} | odd {ok[odd].mean():.3f}   (paper: naive Bayes 0.61, best LLM 0.868)", flush=True)
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "lb_baselines.json", "w"), indent=1)
