"""Label-free comparison of LB-0010 prompt inputs with generated validation inputs (value at each list rank, identity overlap)."""
import numpy as np
from epimod.lb.store import lb_meta, load_lb_store, lb_series
from epimod.lb.tasks import build_vocab, extremes_cache, extremes_batch, load_parsed, make_pools
from epimod.train.loop import split_studies

parsed = load_parsed(); excl = lb_series(parsed)
meta_all = lb_meta(excl); tr_m, va_m = split_studies(meta_all, 0.15, 0)
va = load_lb_store(exclude_series=excl, include_series=set(meta_all[va_m].DatasetID))
uni = va.universe; odd = np.arange(500) % 2 == 1
pools = make_pools(parsed, uni, ext_mask=odd, pair_mask=odd)
cache = extremes_cache(va, pools)
t = parsed["LB-0010"]
ranks = [0, 4, 9, 24, 49, 74, 99]
def show(name, low, high):
    print(f"{name:14s} low@rank {ranks}: " + " ".join(f"{np.median(low[:, r]):.2f}" for r in ranks) + " | high@rank: " + " ".join(f"{np.median(high[:, r]):.2f}" for r in ranks))
lb_low = np.array([np.sort(np.array(b)[~np.array(h, bool)])[:100] for b, h in zip(t.beta, t.is_high)])
lb_high = np.array([np.sort(np.array(b)[np.array(h, bool)])[::-1][:100] for b, h in zip(t.beta, t.is_high)])
show("LB-0010 items", lb_low, lb_high)
show("generated val", np.round(cache["low_val"], 2), np.round(cache["high_val"], 2))
print("items with per-list length != 100:", sum(1 for h in t.is_high if (np.array(h) == 0).sum() != 100 or (np.array(h) == 1).sum() != 100))
# identity overlap: fraction of list slots whose CpG is among the 300 most frequent identities in generated val lists
from collections import Counter
cnt = Counter(cache["low_col"].ravel().tolist()); top = {c for c, _ in cnt.most_common(300)}
gen = np.mean([c in top for c in cache["low_col"].ravel()])
lbm = []
for ids, h in zip(t.ids, t.is_high):
    u = uni.get_indexer(list(np.array(ids)[~np.array(h, bool)])); lbm.extend(int(x) in top for x in u)
print(f"low-list slots in generated top-300 identities: generated {gen:.3f} | LB items {np.mean(lbm):.3f}")

# ---- is the gap just between-series variation? per-series label-free stats on generated lists (all training+val series)
print("\nper-series (generated lists): series | n | low@rank0 | low@rank99 | high@rank0 | high@rank99 | low-list top-300 coverage")
ser = va.meta.DatasetID.to_numpy()
for s in sorted(set(ser)):
    r = np.flatnonzero(ser == s)
    lv, hv = np.round(cache["low_val"][r], 2), np.round(cache["high_val"][r], 2)
    cov = np.mean([c in top for c in cache["low_col"][r].ravel()])
    print(f"{s:10s} {len(r):4d} | {np.median(lv[:,0]):.2f} {np.median(lv[:,99]):.2f} | {np.median(hv[:,0]):.2f} {np.median(hv[:,99]):.2f} | {cov:.3f}")
lbser = np.array([g if isinstance(g, str) else g[0] for g in t.gse])
print("\nLB-0010 items by series: series | n | low@rank0 | low@rank99 | high@rank0 | high@rank99 | coverage")
for s in [x for x, _ in __import__("collections").Counter(lbser).most_common(12)]:
    r = np.flatnonzero(lbser == s)
    cov = np.mean([int(u) in top for i in r for u in uni.get_indexer(list(np.array(t.ids[i])[~np.array(t.is_high[i], bool)]))])
    print(f"{s:10s} {len(r):4d} | {np.median(lb_low[r,0]):.2f} {np.median(lb_low[r,99]):.2f} | {np.median(lb_high[r,0]):.2f} {np.median(lb_high[r,99]):.2f} | {cov:.3f}")
