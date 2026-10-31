#!/usr/bin/env python3
"""fold_compare.py -- the pre-registered 5-fold comparison in CONFIRMATORY_PROTOCOL.md (do not change the model, weights or metric after seeing results).

  python fold_compare.py --atlas /path/to/aso-atlas-2 --v1 aso.csv.gz --cache cache --out fold_work
Steps (each cached under --out): features for the 180,026-row OligoAI-ready table -> our LightGBM per fold -> merge with the authors' out-of-fold OligoAI predictions -> report.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from guidefam import aso as A                                                  # noqa: E402

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=40, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
              reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=0)           # identical to compare/aso_compare.py
KEY = ["custom_id", "helm_annotation", "cell_line", "target_RNA", "aso_sequence_5_to_3", "dosage", "inhibition_percent"]
pat = lambda c: c.split("/")[-1].split("_table_")[0]


def load_table(a):
    R = os.path.join(a.atlas, "data/oligostack/processed/oligoai_train.csv.gz")
    d = pd.read_csv(R, low_memory=False)
    folds = pd.read_csv(os.path.join(a.atlas, "aso-atlas-2-release/folds/oligoai_folds.csv.gz"))
    d["fold"] = d.helm_annotation.map(dict(zip(folds.helm_annotation, folds.fold)))
    assert d.fold.notna().all(), "some rows have no fold"
    d["fold"] = d.fold.astype(int)
    d["patent"] = d.custom_id.map(pat)
    d["aso"] = d.aso_sequence_5_to_3.str.upper().str.replace("U", "T")
    d["cells_per_well"] = np.nan                                               # not in this table; the feature is treated as missing for every row
    d["y"] = d.inhibition_percent / 100.0
    d["screen"] = d.custom_id + "|" + d.cell_line.fillna("NA") + "|" + d.dosage.astype(str)
    return d.reset_index(drop=True)


def accessibility(a, d):
    acc_p, ctx_p = os.path.join(a.out, "access.npy"), os.path.join(a.out, "access_ctx.txt")
    if os.path.exists(acc_p):
        return np.load(acc_p), pd.Series(np.arange(sum(1 for _ in open(ctx_p))), index=[l.rstrip("\n") for l in open(ctx_p)])
    old = np.load(os.path.join(a.cache, "aso_access.npy"))
    oldc = [l.rstrip("\n") for l in open(os.path.join(a.cache, "aso_access_ctx.txt"))]
    have = set(oldc)
    new = sorted(set(d.rna_context.dropna()) - have)
    print(f"accessibility: {len(have)} cached contexts, {len(new)} new to compute", flush=True)
    add = A.accessibility(new, os.cpu_count() or 2) if new else np.zeros((0, old.shape[1]), np.float32)
    acc, ctx = np.vstack([old, add]), oldc + new
    np.save(acc_p, acc)
    open(ctx_p, "w").write("\n".join(ctx) + "\n")
    return acc, pd.Series(np.arange(len(ctx)), index=ctx)


def rank_gauss(screen, y):
    r = y.groupby(screen).rank(method="average")
    n = y.groupby(screen).transform("size")
    return np.asarray(norm.ppf((r - 0.5) / n))


def fit_folds(a, d, X):
    import lightgbm as lgb
    out = os.path.join(a.out, "ours_oof.npy")
    if os.path.exists(out):
        return np.load(out)
    Xc = X.copy()
    for c in ("transfection", "cell_line"):
        Xc[c] = Xc[c].where(Xc[c] >= 0).astype("category")
    p = np.full(len(d), np.nan)
    for k in range(5):
        tr, te = (d.fold != k).values, (d.fold == k).values
        t = d[tr]
        m = lgb.LGBMRegressor(**PARAMS).fit(Xc[tr], rank_gauss(t.screen, t.y))
        p[te] = m.predict(Xc[te])
        print(f"  fold {k}: trained on {int(tr.sum())}, predicted {int(te.sum())}", flush=True)
    np.save(out, p)
    return p


def attach_oligoai(a, d):
    parts = []
    for k in range(5):
        q = pd.read_parquet(os.path.join(a.atlas, f"data/results/oligoai_fold{k}_predictions.parquet"))
        q["fold_pred"] = k
        parts.append(q)
    q = pd.concat(parts, ignore_index=True)
    assert len(q) == len(d), (len(q), len(d))
    for df in (d, q):
        df["occ"] = df.groupby(KEY, dropna=False).cumcount()
    m = d.merge(q[KEY + ["occ", "prediction", "fold_pred"]], on=KEY + ["occ"], how="left")
    assert len(m) == len(d) and m.prediction.notna().all(), f"{int(m.prediction.isna().sum())} rows did not match an OligoAI prediction"
    assert (m.fold == m.fold_pred).all(), "fold labels of table and predictions disagree"
    return m.rename(columns={"prediction": "oligoai"})


def screen_table(df, ours, unit, col_screen="screen", min_n=20):
    rows = []
    for sid, g in df.groupby(col_screen):
        if len(g) < min_n:
            continue
        y = rankdata(g.y.values)
        if y.std() == 0:
            continue
        ra, rb = rankdata(g[ours].values) / len(g), rankdata(g.oligoai.values) / len(g)
        mix = 0.5 * ra + 0.5 * rb
        c = lambda v: np.corrcoef(y, rankdata(v))[0, 1] if np.std(v) > 0 else np.nan
        top = lambda v: g.y.values[np.argmax(v)] / g.y.mean() if g.y.mean() > 0 else np.nan
        rows.append(dict(screen=sid, unit=g[unit].iloc[0], fold=int(g.fold.mode().iloc[0]), ours=c(ra), oligoai=c(rb), mix=c(mix),
                         top_ours=top(g[ours].values), top_oligoai=top(g.oligoai.values), top_mix=top(mix)))
    return pd.DataFrame(rows).dropna(subset=["ours", "oligoai", "mix"])


def boot(t, cols, n, seed=0):
    rng = np.random.default_rng(seed)
    units = t.unit.unique()
    idx = {u: np.flatnonzero(t.unit.values == u) for u in units}
    v = {c: t[c].values for c in cols}
    res = {c: np.empty(n) for c in cols}
    for i in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(units, len(units))])
        for c in cols:
            res[c][i] = v[c][ii].mean()
    return res


def report(name, t, n, say):
    say(f"\n=== {name}: {len(t)} screens, {t.unit.nunique()} patents")
    b = boot(t, ["ours", "oligoai", "mix"], n)
    for c, lab in (("ours", "ours (compact LightGBM)"), ("oligoai", "OligoAI (authors' out-of-fold)"), ("mix", "50/50 rank average")):
        lo, hi = np.percentile(b[c], [2.5, 97.5])
        say(f"  {lab:34s} within-screen Spearman {t[c].mean():.3f} [{lo:.3f},{hi:.3f}]")
    for x, y_, lab in (("mix", "oligoai", "MIX - OligoAI   (PRIMARY)" if name.startswith("ALL") else "MIX - OligoAI"), ("ours", "oligoai", "ours - OligoAI"), ("mix", "ours", "MIX - ours")):
        dd = b[x] - b[y_]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        say(f"  {lab:34s} {t[x].mean() - t[y_].mean():+.3f} [{lo:+.3f},{hi:+.3f}]" + (f"   P(MIX better) = {np.mean(dd > 0):.3f}" if x == "mix" and y_ == "oligoai" else ""))
    say(f"  median top-1 / mean target: ours {t.top_ours.median():.2f} | OligoAI {t.top_oligoai.median():.2f} | MIX {t.top_mix.median():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--v1", required=True, help="the original Zenodo table aso.csv.gz (only used to define strata)")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="fold_work")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_table(a)
    print(f"{len(d)} rows, {d.patent.nunique()} patents, {d.screen.nunique()} screens; rows per fold {d.fold.value_counts().sort_index().tolist()}", flush=True)
    acc, idx = accessibility(a, d)
    fp = os.path.join(a.out, "features.pkl")
    if os.path.exists(fp):
        X = pd.read_pickle(fp)
    else:
        X = A.features(d, acc, idx)
        X.to_pickle(fp)
    print(f"features {X.shape}; target site found for {X.site_found.mean():.1%} of rows", flush=True)
    d["ours"] = fit_folds(a, d, X)
    d = attach_oligoai(a, d)
    v1 = pd.read_csv(a.v1, low_memory=False)
    v1_pat, v1_seq = set(v1.custom_id.map(pat)), set(v1.aso_sequence_5_to_3.str.upper())
    d["new_patent"], d["new_seq"] = ~d.patent.isin(v1_pat), ~d.aso.isin(v1_seq)
    rep = []

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        rep.append(s)
    say(f"Rows {len(d)} | patents {d.patent.nunique()} | new-patent rows {int(d.new_patent.sum())} ({d[d.new_patent].patent.nunique()} patents) | new-sequence rows {int(d.new_seq.sum())}")
    t = screen_table(d, "ours", "patent")
    report("ALL ROWS, screens = (table, cell line, dose)", t, a.boot, say)
    say("\n  per fold (MIX - OligoAI): " + ", ".join(f"fold {k}: {g.mix.mean() - g.oligoai.mean():+.3f} (n={len(g)})" for k, g in t.groupby("fold")) +
        f"  -> MIX better in {int(sum((g.mix.mean() > g.oligoai.mean()) for _, g in t.groupby('fold')))} of 5 folds")
    report("STRATUM A: rows from patents absent from the Aug-2025 table", screen_table(d[d.new_patent], "ours", "patent"), a.boot, say)
    report("STRATUM B: rows whose ASO sequence is absent from the Aug-2025 table", screen_table(d[d.new_seq], "ours", "patent"), a.boot, say)
    report("SECONDARY: per-table Spearman (OligoAI's own definition)", screen_table(d, "ours", "patent", col_screen="custom_id"), a.boot, say)
    open(os.path.join(a.out, "fold_compare_report.txt"), "w").write("\n".join(rep) + "\n")
    d[["custom_id", "patent", "screen", "fold", "y", "ours", "oligoai", "new_patent", "new_seq"]].to_csv(os.path.join(a.out, "fold_compare_rows.csv.gz"), index=False)


if __name__ == "__main__":
    main()
