#!/usr/bin/env python3
"""EpiClock: three tests of whether the transformer family can do something a linear clock cannot.  No retraining: it reads the finished models.

Put this file next to epiclock.py and run it from the folder you ran epiclock.py in:

    python epiclock_tests.py                     # all three tests, arms tiny-pre, base-pre, large-pre
    python epiclock_tests.py --tests 1           # one test
    python epiclock_tests.py --arms tiny-pre,small-pre,base-pre,large-pre,xl-pre

Everything is written to runs/epiclock_clean/tests/ (TESTS_REPORT.md is the summary).  Each test is cached; an interrupted run continues, --redo starts over.

THE DECISION RULES (fixed here, before any result was seen; the verdicts printed at the end are mechanical)
  Test 1  missing CpGs.  Hide most CpGs (random 20 / 10 / 5 % kept, plus the real 3,709-CpG MethylGPT array subset) and score the 2,830 healthy test controls.
          A: PASS if, in at least 2 of the 3 random low-availability conditions, the largest arm beats the better of {ridge, elastic net} fitted once on all CpGs
             (missing = training mean) by at least 0.3 y in median absolute error, with a study-level 95% interval for the paired difference that excludes zero.
          B: PASS if in every condition it is within 0.3 y of an elastic net REFIT on exactly the CpGs that are present (the strongest baseline: it needs a new model per pattern).
  Test 2  out-of-distribution detection.  Deviation score = masked-CpG reconstruction error.  Cancer / cell lines / non-cancer disease (never trained on) versus held-out healthy samples,
          all restricted to the same 3,709 CpGs.  Baselines (given the benefit of picking the best on the test data): distance from the healthy mean, PCA residual, PCA Mahalanobis.
          PASS if the largest arm's AUC beats the best baseline by at least 0.03 with a study-level interval above zero, BOTH for cancer (all tissues) vs healthy test controls
          AND for blood-type cancer vs healthy blood.
  Test 3  imputation of unmeasured CpGs.  Hide 25% of the visible CpGs of each healthy test sample; predict them from the rest (the same visible CpGs for every method).
          Baselines: training mean, PCA (probabilistic, k = 10 / 30 / 100) and k-nearest-neighbour (k = 10 / 30), best of each picked on the test cells.
          PASS if the largest arm's mean absolute error is at least 3% lower than the best baseline, with a study-level interval for the difference below zero.
"""
import argparse
import collections
import io
import os
import pickle
import re
import sys
import tarfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epiclock as E

log = E.log
BLOOD = re.compile(r"blood|leuk|lymph|pbmc|mononuclear|marrow|monocyt|neutrophil|granulocyt|\bt[- ]?cell|\bb[- ]?cell|nk cell", re.I)
SIZE_ORDER = list(E.PRESETS)


# ======================================================================================================== helpers
class SB:
    """Study-level bootstrap: whole studies are resampled (the honest unit; samples within a study share batch effects).  Draws are shared, so comparisons are paired."""

    def __init__(self, groups, B, seed=0):
        rng = np.random.default_rng(seed); _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        by = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
        self.draws = [np.concatenate([by[k] for k in rng.integers(0, len(by), len(by))]) for _ in range(B)]

    def stat(self, x, f=np.median):
        return np.array([f(x[i]) for i in self.draws])


def ci(v, d=2, signed=False):
    lo, hi = np.percentile(v, [2.5, 97.5]); f = f"{{:+.{d}f}}" if signed else f"{{:.{d}f}}"
    return "[" + f.format(lo) + ", " + f.format(hi) + "]"


def pkl_load(p):
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def pkl_save(obj, p):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, p)


def observed_fraction(X, chunk=4000):
    return np.concatenate([(X[s:s + chunk] != 255).mean(1) for s in range(0, len(X), chunk)])


def build_env(args):
    smoke = os.environ.get("EPICLOCK_SMOKE") == "1"
    a = types.SimpleNamespace(data=os.environ.get("EPICLOCK_DATA"), out=args.out, device=args.device, threads=0, memory_fraction=E.CONFIG["memory_fraction"], seed=0,
                              eval_every=E.CONFIG["eval_every"], patience=E.CONFIG["patience"], ssl_steps=10 ** 9, ft_steps=10 ** 9, val_draws=E.CONFIG["val_draws"],
                              test_draws=args.draws, boot=args.boot, no_selftest=True, smoke=smoke, arms="")
    orig = E.check_corpus_stamp

    def soft_stamp(e):                              # read-only analysis: an unstamped older folder is accepted (with a note); a folder stamped with ANOTHER corpus is refused
        try:
            orig(e)
        except SystemExit as ex:
            if "different training corpus" in str(ex.code):
                raise
            log("[note] the output folder has no corpus stamp (it was made by an earlier version of epiclock.py); reading it anyway - nothing in it is modified.")
    E.check_corpus_stamp = soft_stamp
    e = E.make_env(a); e.tdir = Path(args.out) / "tests"; e.tdir.mkdir(parents=True, exist_ok=True); E.LOGF = e.tdir / "tests.log"; e.smoke = smoke
    return e


def find_models(e, names):
    out = []
    for n in names:
        ar = E.parse_arm(n)
        if ar["kind"] == "dis":
            log(f"[skip] {n}: the NOT-CLEAN model saw non-cancer-disease profiles; these tests are for the clean family"); continue
        mp = E.arm_dir(e, ar) / "model.pt"
        if not mp.exists():
            log(f"[skip] {n}: {mp} not found"); continue
        out.append((n, ar, mp))
    if not out:
        sys.exit("none of the requested arms has a trained model in the output folder.")
    out.sort(key=lambda t: SIZE_ORDER.index(t[1]["size"]))
    return out


# ======================================================================================================== linear baselines (same recipe as epiclock.py, optionally on a subset of CpGs)
def enet_fit(e, cols, n_alphas):
    from sklearn.linear_model import enet_path
    c, sp = e.c, e.sp; tr, va = sp["train"], sp["val"]
    Xtr, Xva = E.standardize(c.X[tr][:, cols], e.mean[cols], e.std[cols]), E.standardize(c.X[va][:, cols], e.mean[cols], e.std[cols])
    ytr, ava = E.age_to_t(e.age[tr]), e.age[va]
    alphas, coefs, _ = enet_path(Xtr, ytr - ytr.mean(), l1_ratio=0.5, n_alphas=n_alphas, eps=1e-3)
    ev = [np.median(np.abs(E.t_to_age(Xva @ coefs[:, k] + ytr.mean()) - ava)) for k in range(coefs.shape[1])]; k = int(np.argmin(ev))
    return coefs[:, k].astype(np.float32), float(ytr.mean())


def ridge_fit_cols(e, cols):
    return E.ridge_fit(types.SimpleNamespace(X=e.c.X[:, cols], meta=e.c.meta), e.sp, e.mean[cols], e.std[cols], lambda *a: None)


def lin_predict(Xrows, cols, mean, std, w, b):
    return E.t_to_age(E.standardize(Xrows[:, cols], mean[cols], std[cols]) @ w + b)


# ======================================================================================================== TEST 1: missing CpGs
def t1(e, models, args):
    f = e.tdir / "t1.pkl"
    if f.exists() and not args.redo:
        return pkl_load(f)
    hc = e.sp["test"]; X = e.c.X[hc]; age = e.age[hc]; grp = e.c.meta.DatasetID.to_numpy()[hc]; P = X.shape[1]; allc = np.arange(P); na = 5 if e.smoke else 20
    conds = {"all CpGs (100%)": (np.ones(P, bool), 1.0)}
    for fr in args.keep:
        S = np.zeros(P, bool); S[np.random.default_rng(1000 + int(fr * 1000)).choice(P, int(round(fr * P)), replace=False)] = True; conds[f"random {fr:.0%} of CpGs kept"] = (S, fr)
    frac = observed_fraction(e.c.X); ext = np.flatnonzero((frac > 0.3) & (frac < 0.5))
    if len(ext) >= 50:
        S = (e.c.X[ext] != 255).mean(0) > 0.9; conds[f"MethylGPT array subset ({int(S.sum())} CpGs, {S.mean():.0%})"] = (S, float(S.mean()))
    log(f"[test 1] {len(X)} healthy test controls of {len(set(grp))} studies; conditions: {list(conds)}")
    rg = E.get_ridge(e); en_w, en_b = enet_fit(e, allc, na)
    full_r = lin_predict(X, allc, e.mean, e.std, rg["w"], rg["b"]); full_e = lin_predict(X, allc, e.mean, e.std, en_w, en_b)
    log(f"[test 1] sanity: ridge {np.median(np.abs(full_r - age)):.3f} / elastic net {np.median(np.abs(full_e - age)):.3f} on all CpGs (the main report had 3.561 / 3.273)")
    FIX_R, FIX_E, REF_R, REF_E = "ridge, fitted once on all CpGs (missing = mean)", "elastic net, fitted once on all CpGs (missing = mean)", "ridge REFIT on the present CpGs", "elastic net REFIT on the present CpGs"
    pred = {c: {} for c in conds}
    for c, (S, fr) in conds.items():
        Xm = X.copy(); Xm[:, ~S] = 255; cols = np.flatnonzero(S)
        pred[c][FIX_R] = E.t_to_age(E.standardize(Xm, e.mean, e.std) @ rg["w"] + rg["b"]); pred[c][FIX_E] = E.t_to_age(E.standardize(Xm, e.mean, e.std) @ en_w + en_b)
        if S.all():
            pred[c][REF_R], pred[c][REF_E] = pred[c][FIX_R], pred[c][FIX_E]
        else:
            r2 = ridge_fit_cols(e, cols); w2, b2 = enet_fit(e, cols, na)
            pred[c][REF_R] = lin_predict(Xm, cols, e.mean, e.std, r2["w"], r2["b"]); pred[c][REF_E] = lin_predict(Xm, cols, e.mean, e.std, w2, b2)
        log(f"[test 1] {c}: baselines done (ridge once {np.median(np.abs(pred[c][FIX_R] - age)):.2f}, elastic net once {np.median(np.abs(pred[c][FIX_E] - age)):.2f}, elastic net refit {np.median(np.abs(pred[c][REF_E] - age)):.2f})")
    cache_f = e.tdir / "t1_cache.pkl"; cache = pkl_load(cache_f) or {}
    for arm, ar, mp in models:
        todo = [c for c in conds if f"{arm}|{c}" not in cache]
        if todo:
            model, ck = E.load_model(mp, e.dev)
            for c in todo:
                Xm = X.copy(); Xm[:, ~conds[c][0]] = 255
                cache[f"{arm}|{c}"] = E.t_to_age(E.predict_t(model, Xm, ck["K"], args.draws, np.random.default_rng(17))); pkl_save(cache, cache_f)
                log(f"[test 1] {arm} | {c}: MedAE {np.median(np.abs(cache[f'{arm}|{c}'] - age)):.2f}")
            del model; E.free_mem(e.dev)
        for c in conds:
            pred[c][arm] = cache[f"{arm}|{c}"]
    res = dict(pred=pred, age=age, grp=grp, conds={c: v[1] for c, v in conds.items()}, arms=[m[0] for m in models], fixed=(FIX_R, FIX_E), refit=(REF_R, REF_E))
    pkl_save(res, f); return res


def t1_report(r, args):
    pred, age, grp = r["pred"], r["age"], r["grp"]; sb = SB(grp, args.boot); err = {c: {m: np.abs(p - age) for m, p in d.items()} for c, d in pred.items()}
    methods = list(r["fixed"]) + list(r["refit"]) + r["arms"]
    tab = pd.DataFrame({m: {c: np.median(err[c][m]) for c in pred} for m in methods}).round(2)
    ciw = pd.DataFrame({m: {c: ci(sb.stat(err[c][m])) for c in pred} for m in methods})
    big = r["arms"][-1]; low = [c for c in pred if c.startswith("random") and r["conds"][c] <= 0.2 + 1e-9]; lines, passA = [], 0
    for c in low:
        base = min(r["fixed"], key=lambda m: np.median(err[c][m])); d = sb.stat(err[c][big]) - sb.stat(err[c][base]); lo, hi = np.percentile(d, [2.5, 97.5]); pt = np.median(err[c][big]) - np.median(err[c][base])
        ok = pt <= -0.3 and hi < 0; passA += ok; lines.append(f"  - {c}: {big} minus {base.split(',')[0]} (the better fixed baseline) = {pt:+.2f} y, study-level 95% CI [{lo:+.2f}, {hi:+.2f}] -> {'beats it' if ok else 'does not clearly beat it'}")
    gaps = {c: np.median(err[c][big]) - np.median(err[c][r["refit"][1]]) for c in pred}; passB = all(g <= 0.3 for g in gaps.values()); A = passA >= 2
    txt = ["## Test 1 - missing CpGs: median absolute error (years) on the 2,830 healthy test controls", "```", tab.to_string(), "```",
           "Study-level 95% intervals:", "```", ciw.to_string(), "```",
           f"Claim A (more robust than a model fitted once on all CpGs; judged on {big}, conditions with <= 20% of CpGs kept): {'PASS' if A else 'FAIL'} ({passA} of {len(low)} conditions)"] + lines + \
          [f"Claim B (within 0.3 y of an elastic net refit on exactly the CpGs present, in every condition): {'PASS' if passB else 'FAIL'}; gaps to the refit: " + ", ".join(f"{c.split(' (')[0]} {g:+.2f}" for c, g in gaps.items()), "",
           "How to read it: a fixed model that tolerates missing CpGs is useful because one model serves every array; it cannot beat a model refit for the exact pattern, so Claim B is the practical one. "
           "Note the linear part of EpiClock reads the whole standardized row (missing = mean) and was fine-tuned without random CpG dropout, so a result close to the fixed ridge is the expected failure mode."]
    return "\n".join(txt), dict(A=A, B=passB)


# ======================================================================================================== TEST 2: out-of-distribution detection
def mg_shards(src):
    if src.is_dir():
        for f in sorted(src.rglob("*.parquet")):
            yield f.name, f
    else:
        with tarfile.open(src, "r|gz") as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(".parquet"):
                    yield m.name, io.BytesIO(tar.extractfile(m).read())


def stream_positives(e, H, args):
    f = e.tdir / "ood_profiles.npz"
    if f.exists() and not args.redo:
        z = np.load(f, allow_pickle=True); return z["X"], z["ids"].astype(str), z["cls"].astype(str), z["pidx"]
    import pyarrow.parquet as pq
    classes = ("CANCER", "cell culture / line", "non-cancer disease"); Hs = H.drop_duplicates("sample id").set_index("sample id"); known = set(e.c.meta.index.astype(str))
    cand = Hs[Hs.tier.isin(classes) & ~Hs.index.isin(known)]; want = cand.tier.to_dict(); quota = {c: args.max_per_class for c in classes}; got = collections.Counter()
    d = e.data / "external"; src = os.environ.get("EPICLOCK_MG_SOURCE"); src = Path(src) if src else Path(E.download(E.MG_URL, d / "processed_type3_parquet_shuffled.tar.gz", "MethylGPT pre-training set"))
    probes = pd.read_csv(E.download(E.MG_PROBES_URL, d / "probe_ids_type3.csv", "MethylGPT CpG list"), index_col=0).iloc[:, 0].astype(str)
    pos = pd.Index(probes).get_indexer(e.c.cpgs); pidx = np.flatnonzero(pos >= 0); pcols = pos[pidx]; Xs, ids_all, cls_all, seen = [], [], [], set(); n_seen = 0; t0 = time.time()
    log(f"[test 2] streaming the MethylGPT set for held-out cancer / cell-culture / non-cancer-disease profiles (up to {args.max_per_class} per class; {len(want)} candidate ids)")
    try:
        for name, fobj in mg_shards(src):
            try:
                for batch in pq.ParquetFile(fobj).iter_batches(batch_size=500, columns=["id", "data"]):
                    ids = np.array(batch.column("id").to_pylist(), dtype=object).astype(str); n = len(ids); n_seen += n
                    ok = np.array([(i in want) and got[want[i]] < quota[want[i]] and i not in seen for i in ids], dtype=bool)
                    if not ok.any():
                        continue
                    v = batch.column("data").flatten().to_numpy(zero_copy_only=False).reshape(n, -1)[:, pcols][ok]
                    X = np.where(np.isfinite(v) & (v >= 0) & (v <= 1), np.rint(v * 100), 255).astype(np.uint8); good = (X != 255).mean(1) >= 0.5
                    for i, x in zip(ids[ok][good], X[good]):
                        c = want[i]
                        if got[c] < quota[c]:
                            got[c] += 1; seen.add(i); Xs.append(x); ids_all.append(i); cls_all.append(c)
            except Exception as ex:
                log(f"[test 2] {name[:40]}: {type(ex).__name__}: {str(ex)[:80]} - skipped the rest of this shard")
            log(f"[test 2] {n_seen} profiles read, kept {dict(got)} ({(time.time() - t0) / 60:.0f} min)")
            if all(got[c] >= quota[c] for c in classes):
                break
    except (tarfile.ReadError, EOFError, OSError) as ex:
        log(f"[test 2] archive ended early ({type(ex).__name__}); using what was read")
    if not Xs:
        sys.exit("[test 2] no cancer / culture / disease profiles were found in the MethylGPT set.")
    X, ids, cls = np.stack(Xs), np.array(ids_all), np.array(cls_all); np.savez_compressed(f, X=X, ids=ids, cls=cls, pidx=pidx); return X, ids, cls, pidx


def auc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg])); n1, n0 = len(pos), len(neg); return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def t2(e, models, args):
    f = e.tdir / "t2.pkl"
    if f.exists() and not args.redo:
        return pkl_load(f)
    H = E.ewas_table(e); Xp, ids_p, cls_p, pidx = stream_positives(e, H, args); P = len(e.c.cpgs); Hs = H.drop_duplicates("sample id").set_index("sample id")
    Xpos = np.full((len(Xp), P), 255, np.uint8); Xpos[:, pidx] = Xp
    hc, va = e.sp["test"], e.sp["val"]; meta = e.c.meta
    def masked(rows):
        Z = np.full_like(rows, 255); Z[:, pidx] = rows[:, pidx]; return Z
    Xhc, Xva = masked(e.c.X[hc]), masked(e.c.X[va]); rows = np.concatenate([Xpos, Xhc, Xva]); n1, n2 = len(Xpos), len(Xpos) + len(Xhc)
    seg = dict(pos=slice(0, n1), hc=slice(n1, n2), val=slice(n2, len(rows)))
    tissue_p = Hs.reindex(ids_p)["tissue"].astype(str).str.lower().to_numpy(); proj_p = Hs.reindex(ids_p)["project id"].astype(str).to_numpy()
    tissue_hc = meta.Tissue.astype(str).str.lower().to_numpy()[hc]; grp_hc = meta.DatasetID.astype(str).to_numpy()[hc]; grp_va = meta.DatasetID.astype(str).to_numpy()[va]
    blood_p = np.array([bool(BLOOD.search(t)) for t in tissue_p]); blood_hc = np.array([bool(BLOOD.search(t)) for t in tissue_hc])
    log(f"[test 2] positives: {dict(collections.Counter(cls_p))}; negatives: {len(Xhc)} healthy test controls ({int(blood_hc.sum())} blood), {len(Xva)} held-out clean validation samples; all restricted to {len(pidx)} CpGs")
    scores = {}                                              # higher = more unlike healthy
    for arm, ar, mp in models:
        model, ck = E.load_model(mp, e.dev); t0 = time.time()
        scores[arm] = E.recon_error(model, rows, ck["K"], 4, np.random.default_rng(23)); del model; E.free_mem(e.dev); log(f"[test 2] {arm}: deviation scores for {len(rows)} profiles ({time.time() - t0:.0f} s)")
    # ---- baselines on the same restricted rows
    from sklearn.utils.extmath import randomized_svd
    mu_, sd_ = e.mean[pidx], e.std[pidx]
    def zs(R):
        z = np.clip((R[:, pidx].astype(np.float32) / 100.0 - mu_) / sd_, -6, 6); o = R[:, pidx] != 255; z[~o] = 0.0; return z, o
    pool = e.sp["ssl"]; sub = np.random.default_rng(3).choice(pool, min(20000, len(pool)), replace=False); Ztr, Otr = zs(e.c.X[sub]); keep = Otr.mean(1) >= 0.5; Ztr = Ztr[keep]
    cen = Ztr.mean(0); U, S, Vt = randomized_svd(Ztr - cen, n_components=200, n_iter=4, random_state=0); lam = S ** 2 / (len(Ztr) - 1)
    Z, O = zs(rows); Zc = Z - cen; sc = Zc @ Vt.T; base = {}
    base["distance from the healthy mean (mean |z|)"] = (np.abs(Z) * O).sum(1) / O.sum(1)
    for k in (50, 200):
        res = Zc - sc[:, :k] @ Vt[:k]; base[f"PCA-{k} residual"] = ((res ** 2) * O).sum(1) / O.sum(1)
    base["PCA-50 Mahalanobis"] = ((sc[:, :50] ** 2) / lam[:50]).sum(1)
    # ---- AUCs with study-level bootstrap (positives and negatives resampled by study, independently)
    cancer = cls_p == "CANCER"
    pos_sets = {"cancer (all tissues)": np.flatnonzero(cancer), "cancer (blood-type tissue)": np.flatnonzero(cancer & blood_p),
                "cell culture / line": np.flatnonzero(cls_p == "cell culture / line"), "non-cancer disease": np.flatnonzero(cls_p == "non-cancer disease")}
    neg_sets = {"healthy test controls (all)": (np.flatnonzero(np.ones(len(Xhc), bool)) + n1, grp_hc), "healthy test controls (blood)": (np.flatnonzero(blood_hc) + n1, grp_hc[blood_hc]),
                "held-out clean validation samples": (np.arange(len(Xva)) + n2, grp_va)}
    allsc = {**{f"EpiClock {a}": s for a, s in scores.items()}, **base}; out = []; B = args.boot
    for pn, pidx_ in pos_sets.items():
        if len(pidx_) < 20:
            continue
        sbp = SB(proj_p[pidx_], B, 1)
        for nn, (nidx, ngrp) in neg_sets.items():
            if len(nidx) < 20:
                continue
            sbn = SB(ngrp, B, 2); row = dict(positives=pn, n_pos=len(pidx_), negatives=nn, n_neg=len(nidx)); draws = {}
            for m, s in allsc.items():
                sp_, sn_ = s[seg["pos"]][pidx_], s[nidx]; row[m] = auc(sp_, sn_); draws[m] = np.array([auc(sp_[dp], sn_[dn]) for dp, dn in zip(sbp.draws, sbn.draws)])
            row["_draws"] = draws; out.append(row)
        log(f"[test 2] AUCs done for {pn}")
    comp = {pn: dict(n=int(len(ix)), top_tissues=dict(collections.Counter(tissue_p[ix]).most_common(4))) for pn, ix in pos_sets.items() if len(ix)}
    comp["healthy test controls"] = dict(n=int(len(hc)), top_tissues=dict(collections.Counter(tissue_hc).most_common(4)))
    res = dict(rows=out, models=[f"EpiClock {a}" for a in scores], baselines=list(base), tissues=comp, arms=[m[0] for m in models]); pkl_save(res, f); return res


def t2_report(r, args):
    big = f"EpiClock {r['arms'][-1]}"; tab, lines, ok_rows = [], [], {}
    for row in r["rows"]:
        d = row["_draws"]; bb = max(r["baselines"], key=lambda m: row[m]); diff = d[big] - d[bb]; lo, hi = np.percentile(diff, [2.5, 97.5]); pt = row[big] - row[bb]
        rec = dict(positives=row["positives"], negatives=row["negatives"], n=f"{row['n_pos']} / {row['n_neg']}")
        for m in r["models"] + r["baselines"]:
            rec[m] = f"{row[m]:.3f} {ci(d[m], 3)}"
        rec["model - best baseline"] = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}] (best baseline: {bb})"; tab.append(rec); ok_rows[(row["positives"], row["negatives"])] = (pt >= 0.03 and lo > 0)
    k1, k2 = ("cancer (all tissues)", "healthy test controls (all)"), ("cancer (blood-type tissue)", "healthy test controls (blood)")
    verdict = bool(ok_rows.get(k1, False) and ok_rows.get(k2, False)); df = pd.DataFrame(tab)
    txt = ["## Test 2 - out-of-distribution detection: AUC of the score for 'not a healthy sample' (0.5 = no signal), study-level 95% intervals", "```", df.to_string(index=False), "```",
           "Tissue make-up (a score that only detects 'not blood' would look good here):", "```", "\n".join(f"{k}: n={v['n']}, top tissues {v['top_tissues']}" for k, v in r["tissues"].items()), "```",
           f"Verdict ({big} versus the best baseline, margin >= 0.03 with the interval above zero, for BOTH cancer (all tissues) vs healthy test controls AND blood-type cancer vs healthy blood): {'PASS' if verdict else 'FAIL'}",
           f"  cancer (all tissues) vs healthy test controls: {'yes' if ok_rows.get(k1) else 'no'};  blood-type cancer vs healthy blood: {'yes' if ok_rows.get(k2) else ('no' if k2 in ok_rows else 'not enough samples')}", "",
           "Caveats: the positives come from the MethylGPT pipeline and most negatives from the GEO bundle, so a source/platform difference is mixed into every AUC; the PCA baselines carry the same confound, which is why the comparison is against them. "
           "The baselines are picked on the test data, which favours them."]
    return "\n".join(txt), dict(passed=verdict)


# ======================================================================================================== TEST 3: imputation of unmeasured CpGs
def ppca_predict(V, lam, sigma2, mu, idx_v, val_v, ok_v, idx_t, k, chunk=100):
    Vk = V[:k]; prior = np.diag((sigma2 / lam[:k]).astype(np.float64)); out = np.zeros(idx_t.shape, np.float32)
    for s in range(0, len(idx_v), chunk):
        sl = slice(s, s + chunk); okf = ok_v[sl].astype(np.float32); Vv = Vk[:, idx_v[sl]] * okf[None]; resid = (val_v[sl] - mu[idx_v[sl]]) * okf
        A = np.einsum("kcm,jcm->ckj", Vv, Vv).astype(np.float64) + prior[None]; rhs = np.einsum("kcm,cm->ck", Vv, resid).astype(np.float64)
        sc = np.linalg.solve(A, rhs[..., None])[..., 0].astype(np.float32); out[sl] = mu[idx_t[sl]] + np.einsum("ck,kct->ct", sc, Vk[:, idx_t[sl]])
    return np.clip(out, 0, 1)


def knn_predict(T, Xv, M, idx_t, k_list, chunk=250):
    outs = {k: np.zeros(idx_t.shape, np.float32) for k in k_list}; T2 = T ** 2; kmax = max(k_list)
    for s in range(0, len(Xv), chunk):
        m = M[s:s + chunk]; xm = Xv[s:s + chunk] * m; D = (xm ** 2).sum(1, keepdim=True) - 2 * xm @ T.T + m @ T2.T
        nn_ = D.topk(kmax, largest=False).indices; it = torch.as_tensor(idx_t[s:s + chunk]); acc = torch.zeros(it.shape)
        for j in range(kmax):
            acc += torch.gather(T[nn_[:, j]], 1, it)
            if j + 1 in k_list:
                outs[j + 1][s:s + chunk] = (acc / (j + 1)).numpy()
    return outs


def t3(e, models, args):
    f = e.tdir / "t3.pkl"
    if f.exists() and not args.redo:
        return pkl_load(f)
    from sklearn.utils.extmath import randomized_svd
    hc = e.sp["test"]; X = e.c.X[hc]; grp = e.c.meta.DatasetID.to_numpy()[hc]; P = X.shape[1]; pool = e.sp["ssl"]; fr = observed_fraction(e.c.X[pool])
    full = pool[fr >= 0.95]; sel = np.random.default_rng(5).choice(full, min(15000, len(full)), replace=False)
    Xp = e.c.X[sel].astype(np.float32) / 100.0; Xp = np.where(e.c.X[sel] == 255, e.mean[None, :], Xp).astype(np.float32); mu = Xp.mean(0); Xc = Xp - mu
    _, S, Vt = randomized_svd(Xc, n_components=100, n_iter=4, random_state=0); lam = (S ** 2 / (len(Xc) - 1)).astype(np.float64); tot = float(Xc.var(0, ddof=1).sum())
    sig = {k: max((tot - lam[:k].sum()) / (P - k), 1e-8) for k in (10, 30, 100)}; knn_T = torch.as_tensor(Xp[:8000]); log(f"[test 3] reference: {len(Xp)} fully measured pre-training profiles; testing on {len(X)} healthy test controls")
    res = dict(arms=[], rows=[])
    for arm, ar, mp in models:
        model, ck = E.load_model(mp, e.dev); K = ck["K"]; nt = int(K * 0.25)
        idx, ib, valid, is_mask, beta = E.ssl_batch(X, K, np.random.default_rng(31)); n = len(X); pred = np.zeros((n, nt), np.float32)
        with torch.no_grad():
            for s in range(0, n, 64):
                pred[s:s + 64] = model.reconstruct(idx[s:s + 64], ib[s:s + 64], valid[s:s + 64], is_mask[s:s + 64]).cpu().numpy()[:, :nt]
        del model; E.free_mem(e.dev)
        idx, valid, beta = idx.numpy(), valid.numpy(), beta.numpy(); it, tv, truth = idx[:, :nt], valid[:, :nt], beta[:, :nt]; iv, vv, bv = idx[:, nt:], valid[:, nt:], beta[:, nt:]
        Mv = np.zeros((n, P), np.float32); rr = np.repeat(np.arange(n), iv.shape[1]).reshape(iv.shape); Mv[rr[vv], iv[vv]] = 1.0; Xv = np.zeros((n, P), np.float32); Xv[rr[vv], iv[vv]] = bv[vv]
        preds = {f"EpiClock {arm}": pred, "training mean": np.tile(e.mean, (n, 1))[np.arange(n)[:, None], it]}
        for k in (10, 30, 100):
            preds[f"PCA k={k}"] = ppca_predict(Vt, lam, sig[k], mu, iv, bv, vv, it, k)
        for k, p in knn_predict(knn_T, torch.as_tensor(Xv), torch.as_tensor(Mv), it, (10, 30)).items():
            preds[f"KNN k={k}"] = p
        vf = tv.astype(np.float32); per = {m: (np.abs(p - truth) * vf).sum(1) / np.maximum(vf.sum(1), 1) for m, p in preds.items()}
        log(f"[test 3] {arm}: hidden {int(vf.sum())} CpGs ({nt} per sample, {iv.shape[1]} visible): " + ", ".join(f"{m} {v.mean():.4f}" for m, v in per.items()))
        res["arms"].append(arm); res["rows"].append(dict(arm=arm, K=K, per_sample=per))
    res["grp"] = grp; pkl_save(res, f); return res


def t3_report(r, args):
    sb = SB(r["grp"], args.boot); rows, lines, verd = [], [], None
    for row in r["rows"]:
        per = row["per_sample"]; me = f"EpiClock {row['arm']}"; pca = min((m for m in per if m.startswith("PCA")), key=lambda m: per[m].mean()); knn = min((m for m in per if m.startswith("KNN")), key=lambda m: per[m].mean())
        best = pca if per[pca].mean() <= per[knn].mean() else knn; d = sb.stat(per[me], np.mean) - sb.stat(per[best], np.mean); lo, hi = np.percentile(d, [2.5, 97.5]); rel = (per[me].mean() - per[best].mean()) / per[best].mean()
        rows.append({"arm": row["arm"], "tokens": row["K"], "training mean": round(per["training mean"].mean(), 4), pca: round(per[pca].mean(), 4), knn: round(per[knn].mean(), 4), "EpiClock": round(per[me].mean(), 4),
                     "EpiClock vs best baseline": f"{per[me].mean() - per[best].mean():+.4f} ({rel:+.1%}) {ci(d, 4, True)}", "error cut vs training mean": f"{1 - per[me].mean() / per['training mean'].mean():.0%}"})
        verd = (rel <= -0.03 and hi < 0, best, rel, lo, hi, row["arm"])
    txt = ["## Test 3 - imputation: mean absolute error (beta units) on CpGs hidden from healthy test samples (same visible CpGs for every method)", "```", pd.DataFrame(rows).to_string(index=False), "```",
           f"Verdict ({verd[5]} versus the best baseline, >= 3% lower with a study-level interval below zero): {'PASS' if verd[0] else 'FAIL'}  (best baseline {verd[1]}; relative difference {verd[2]:+.1%}, 95% CI of the absolute difference [{verd[3]:+.4f}, {verd[4]:+.4f}])", "",
           "Caveats: the transformer sees only its K tokens by design; the baselines are given exactly the same visible CpGs, and the best k of each is picked on the test cells, which favours them. "
           "Imputation helps people with a smaller array or a damaged sample, but only if it is clearly better than a PCA fit; that is what this checks."]
    return "\n".join(txt), dict(passed=bool(verd[0]))


# ======================================================================================================== main
def main():
    ap = argparse.ArgumentParser(description="Three tests of what the EpiClock transformer can do that a linear clock cannot (no retraining).")
    ap.add_argument("--tests", default="1,2,3"); ap.add_argument("--arms", default=None, help="comma list, default tiny-pre,base-pre,large-pre"); ap.add_argument("--out", default=os.environ.get("EPICLOCK_OUT", "runs/epiclock_clean"))
    ap.add_argument("--device", default=os.environ.get("EPICLOCK_DEVICE", "auto")); ap.add_argument("--draws", type=int, default=8, help="CpG subsets averaged per age prediction (the main report used 16)")
    ap.add_argument("--boot", type=int, default=300); ap.add_argument("--keep", type=float, nargs="+", default=[0.2, 0.1, 0.05], help="fractions of CpGs kept in test 1")
    ap.add_argument("--max-per-class", type=int, default=3000, help="test 2: profiles per class"); ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(); e = build_env(args)
    arms = (args.arms or "tiny-pre,base-pre,large-pre").split(","); models = find_models(e, arms); log(f"models: {[m[0] for m in models]}")
    todo = {x.strip() for x in args.tests.split(",")}; parts, verdict = [], {}
    for k, fn, rp in (("1", t1, t1_report), ("2", t2, t2_report), ("3", t3, t3_report)):
        if k in todo:
            log(f"##### test {k}"); r = fn(e, models, args); txt, v = rp(r, args); parts.append(txt); verdict[k] = v; print("\n" + txt + "\n", flush=True)
    summary = ["# EpiClock differentiator tests", "", "Models: " + ", ".join(m[0] for m in models) + f"; test set: {len(e.sp['test'])} healthy controls of held-out ComputAge benchmark studies.", "",
               "## Verdicts (rules fixed before the results; see the header of epiclock_tests.py)", ""]
    names = {"1": "Test 1 (missing CpGs)", "2": "Test 2 (out-of-distribution detection)", "3": "Test 3 (imputation)"}
    for k, v in verdict.items():
        summary.append(f"- {names[k]}: " + (f"claim A {'PASS' if v['A'] else 'FAIL'}, claim B {'PASS' if v['B'] else 'FAIL'}" if k == "1" else 'PASS' if v["passed"] else "FAIL"))
    (e.tdir / "TESTS_REPORT.md").write_text("\n".join(summary + [""] + parts) + "\n"); log(f"wrote {e.tdir / 'TESTS_REPORT.md'}")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
