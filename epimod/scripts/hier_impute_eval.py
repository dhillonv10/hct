"""Methylome completion: given a fixed subset of CpGs, reconstruct the others on HELD-OUT cohorts.

Scenarios (visible sets): the CpGs a 27K array measures ("27K->450K"), the 353-CpG Horvath panel (LB-0190), a random 10% set.
Baselines: training mean per CpG; multi-output ridge from the visible CpGs (linear completion).  Metrics on validation series:
mean absolute error over target CpGs and per-CpG R^2 against the mean predictor (cluster = validation series).
    python scripts/hier_impute_eval.py runs/lb/hier_a/model.pt out.json"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge

from epimod.data.annotation import load_probe_annotation
from epimod.lb.episet import ROLES
from epimod.lb.epitraj import EpiHier, HierConfig, block_assignment, invert_vocab, profile_matrix
from epimod.lb.lite import load_lite_universe
from epimod.lb.store import lb_meta, lb_series, load_lb_store
from epimod.lb.tasks import Vocab, load_parsed
from epimod.paths import subdir
from epimod.train.loop import split_studies

torch.set_num_threads(4)
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
uni = load_lite_universe(); ids = pd.Index(ck["vocab_ids"]); u2m = np.full(len(uni), -1, np.int64); pos = uni.get_indexer(ids); u2m[pos[pos >= 0]] = np.flatnonzero(pos >= 0)
hc = HierConfig(**ck["cfg"]); model = EpiHier(hc, block_assignment(ids, load_probe_annotation(), hc.block_size)); model.load_state_dict(ck["state"]); model.eval()
v2u = invert_vocab(u2m, len(ids)); V = len(ids)

parsed = load_parsed(); excl = lb_series(parsed); meta_all = lb_meta(excl); tr_m, va_m = split_studies(meta_all, 0.15, 0)
tr = load_lb_store(exclude_series=excl, include_series=set(meta_all[tr_m].DatasetID)); va = load_lb_store(exclude_series=excl, include_series=set(meta_all[va_m].DatasetID))
rng = np.random.RandomState(0)
rtr = rng.choice(tr.n, 6000, replace=False); rva = rng.choice(va.n, min(1500, va.n), replace=False)
Xtr = profile_matrix(tr.prof, rtr, v2u); Xva = profile_matrix(va.prof, rva, v2u); ser = va.meta.DatasetID.to_numpy()[rva]
mu = np.nanmean(Xtr, 0); mu = np.where(np.isfinite(mu), mu, 0.5); sd = np.maximum(np.nanstd(Xtr, 0), 0.02)

# ---- visible sets
gpl = subdir("annot") / "GPL13534_full.txt"
skip = next(i for i, l in enumerate(open(gpl)) if l.startswith("ID\tName\tAddressA_ID"))
m27 = pd.read_csv(gpl, sep="\t", skiprows=skip, usecols=["ID", "Methyl27_Loci"], dtype=str, low_memory=False).drop_duplicates("ID").set_index("ID").Methyl27_Loci.eq("TRUE")
vis_sets = {"27K array CpGs": np.flatnonzero(m27.reindex(ids).fillna(False).to_numpy()),
            "Horvath 353 panel": np.flatnonzero(ids.isin(list(parsed["LB-0190"].ids[0]))),
            "random 10%": np.sort(rng.choice(V, V // 10, replace=False))}
res = {}
for name, vis in vis_sets.items():
    tgt = np.setdiff1d(np.arange(V), vis)
    Ztr = (np.where(np.isfinite(Xtr[:, vis]), Xtr[:, vis], mu[vis]) - mu[vis]) / sd[vis]; Zva = (np.where(np.isfinite(Xva[:, vis]), Xva[:, vis], mu[vis]) - mu[vis]) / sd[vis]
    Ytr = np.where(np.isfinite(Xtr[:, tgt]), Xtr[:, tgt], mu[tgt])
    ridge = Ridge(alpha=300.0).fit(Ztr, Ytr); P_lin = ridge.predict(Zva)
    # model: encode the visible tokens, reconstruct all targets
    P_mod = np.zeros((len(rva), len(tgt)), np.float32)
    with torch.no_grad():
        for s in range(0, len(rva), 16):
            sl = slice(s, s + 16); xb = Xva[sl][:, vis]; ok = np.isfinite(xb)
            cpg = torch.as_tensor(np.where(ok, vis[None, :], -1)); beta = torch.as_tensor(np.where(ok, xb, 0.0).astype(np.float32))
            h, gb, _ = model.encode(cpg, beta, torch.full(cpg.shape, ROLES["fixed"]), torch.zeros(len(cpg), dtype=torch.long))
            for q in range(0, len(tgt), 2048):
                P_mod[sl, q:q + 2048] = model.reconstruct(h, gb, torch.as_tensor(tgt[q:q + 2048]).unsqueeze(0).expand(len(cpg), -1)).numpy()
    Y = Xva[:, tgt]; ok = np.isfinite(Y)
    def score(P):
        mae = np.nanmean(np.where(ok, np.abs(P - Y), np.nan), axis=1)
        sse = np.nansum(np.where(ok, (P - Y) ** 2, np.nan), 0); sst = np.nansum(np.where(ok, (Y - np.nanmean(Y, 0)) ** 2, np.nan), 0)
        return float(np.nanmean(mae)), float(1 - sse.sum() / max(sst.sum(), 1e-9))
    base = np.broadcast_to(mu[tgt], Y.shape)
    res[name] = {"n_visible": int(len(vis)), "n_target": int(len(tgt)), "mean": score(base), "ridge": score(P_lin), "EpiTrajectory-L": score(P_mod)}
    print(name, {k: (round(v[0], 4), round(v[1], 3)) if isinstance(v, tuple) else v for k, v in res[name].items()}, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
