#!/usr/bin/env python3
"""aso_compare.py -- compact ASO model vs the released OligoAI (RiNALMo-giga fine-tune, ~650M parameters), two settings, fixed before looking.

  S1  OligoAI's own protocol: OligoAI's released split by PATENT (seed 42). Ours is trained on the same training patents and both are scored on the
      same held-out test patents. Fair, but the split is not gene-independent (44% of test rows share a gene group with training; both models see that).
  S2  Gene-held-out: ours is trained on the 246 DEV gene groups only and scored on the 81 LOCKED groups (never used for any design decision).
      OligoAI was trained on 80% of patents of the SAME table, so it has seen most of these genes: S2 favours OligoAI. If ours wins here, the win is conservative.

Our model (fixed before the locked groups were scored): LightGBM, sequence + gapmer chemistry + experiment + thermodynamics + target-site accessibility,
within-screen rank-Gauss target (best dev score 0.399). Metric: mean within-screen Spearman over screens with >= 20 ASOs (OligoAI's own headline metric),
95% CI by bootstrap over patents (S1) or gene groups (S2).

  python aso_compare.py prepare  --csv aso.csv.gz --out work          # writes work/ids_to_score.csv  (rows OligoAI must score)
  python oligoai_score.py --csv aso.csv.gz --ckpt OligoAI_11_09_25.ckpt --ids work/ids_to_score.csv --out work/oligoai_pred.csv --check --bf16
  python aso_compare.py evaluate --csv aso.csv.gz --out work --pred work/oligoai_pred.csv [--cache cache]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from guidefam import aso as A                                                  # noqa: E402

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=40, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
              reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=0)


def oligoai_split(raw, seed=42, tr=0.8, va=0.1):
    """Identical to rinalmo/data/downstream/aso/dataset.py: np.random.seed(42) + permutation of the unique patents in file order."""
    pid = raw.custom_id.map(lambda c: c.split("/")[-1].split("_table_")[0])
    np.random.seed(seed)
    perm = np.random.permutation(pid.unique())
    n = len(perm)
    a, b = int(tr * n), int(va * n)
    part = {p: "train" for p in perm[:a]} | {p: "val" for p in perm[a:a + b]} | {p: "test" for p in perm[a + b:]}
    return pid.map(part).values, pid.values


def load_all(a):
    d = A.load(a.csv)
    raw = pd.read_csv(a.csv, low_memory=False)
    assert raw.inhibition_percent.notna().all(), "row positions would shift if rows were dropped"
    split, pid = oligoai_split(raw)
    d["oa_split"], d["patent"] = split[d.raw_id.values], pid[d.raw_id.values]
    return d, raw


def prepare(a):
    d, raw = load_all(a)
    os.makedirs(a.out, exist_ok=True)
    te = d[d.oa_split == "test"]
    lk = d[d.split == "locked"]
    ids = pd.DataFrame({"raw_id": np.union1d(te.raw_id.values, lk.raw_id.values)})
    ids.to_csv(os.path.join(a.out, "ids_to_score.csv"), index=False)
    print(f"S1 test patents: {te.patent.nunique()} patents, {te.screen.nunique()} screens, {len(te)} rows")
    print(f"S2 locked groups: {lk.group.nunique()} groups, {lk.screen.nunique()} screens, {len(lk)} rows")
    print(f"OligoAI must score {len(ids)} rows -> {os.path.join(a.out, 'ids_to_score.csv')}")


def rank_gauss_by_screen(screen, y):
    from scipy.stats import norm
    r = y.groupby(screen).rank(method="average")
    n = y.groupby(screen).transform("size")
    return np.asarray(norm.ppf((r - 0.5) / n))


def fit_predict(X, d, tr_mask, te_mask):
    import lightgbm as lgb
    Xc = X.copy()
    for c in ("transfection", "cell_line"):
        Xc[c] = Xc[c].where(Xc[c] >= 0).astype("category")
    tr = d[tr_mask]
    m = lgb.LGBMRegressor(**PARAMS).fit(Xc[tr_mask], rank_gauss_by_screen(tr.screen, tr.y))
    return m.predict(Xc[te_mask])


def top1_ratio(d, col):
    r = []
    for _, g in d.groupby("screen"):
        if len(g) >= 20 and g.y.mean() > 0:
            r.append(g.y.values[np.argmax(g[col].values)] / g.y.mean())
    return float(np.median(r)), len(r)


def report(d, name, cols, boot, say):
    """d has columns y, screen, group (the bootstrap unit) and one prediction column per model."""
    say(f"\n=== {name}: {len(d)} rows, {d.screen.nunique()} screens, {d.group.nunique()} bootstrap units")
    for lab, c in cols.items():
        s = A.summarize(d, c)
        pt, lo, hi = A.bootstrap(d, c, n=boot)
        t1, nt = top1_ratio(d, c)
        say(f"  {lab:34s} within-screen Spearman {pt:.3f} [{lo:.3f},{hi:.3f}] ({s['n_screens']} screens); median top-1 / mean target {t1:.2f}")
    labs = list(cols)
    if len(labs) >= 2:
        pt, lo, hi = A.bootstrap(d, cols[labs[0]], cols[labs[1]], n=boot)
        say(f"  {labs[0]} minus {labs[1]}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
    return


def evaluate(a):
    d, raw = load_all(a)
    pred = pd.read_csv(a.pred)
    cache = a.cache
    acc_p, ctx_p = os.path.join(cache, "aso_access.npy"), os.path.join(cache, "aso_access_ctx.txt")
    if os.path.exists(acc_p):
        acc = np.load(acc_p)
        ctx = [l.rstrip("\n") for l in open(ctx_p)]
        idx = pd.Series(np.arange(len(ctx)), index=ctx)
    else:
        print("accessibility cache not found: computing (slow, ~20-40 min)...")
        ctx = sorted(d.rna_context.dropna().unique())
        acc = A.accessibility(ctx, os.cpu_count() or 2)
        os.makedirs(cache, exist_ok=True)
        np.save(acc_p, acc)
        open(ctx_p, "w").write("\n".join(ctx) + "\n")
        idx = pd.Series(np.arange(len(ctx)), index=ctx)
    X = A.features(d, acc, idx)
    d = d.merge(pred, on="raw_id", how="left")
    if a.dry_run:                                                              # plumbing test only: labels shuffled within each screen, so no number means anything
        d["y"] = d.groupby("screen").y.transform(lambda s: s.sample(frac=1, random_state=0).values)
        print("DRY RUN: labels shuffled within screens; do not read any number below")
    rep = []

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        rep.append(s)
    # ---- S1: OligoAI's protocol
    te = (d.oa_split == "test").values
    d["ours"] = np.nan
    d.loc[te, "ours"] = fit_predict(X, d, (d.oa_split == "train").values, te)
    s1 = d[te & d.oligoai_pred.notna()].copy()
    miss = int((te & d.oligoai_pred.isna()).sum())
    if miss:
        say(f"WARNING: {miss} S1 test rows have no OligoAI prediction (dropped from both models)")
    s1["group"] = s1.patent
    report(s1, "S1  OligoAI's protocol (patent split, seed 42); both trained on the same training patents",
           {"ours (compact LightGBM)": "ours", "OligoAI (RiNALMo-giga, ~650M)": "oligoai_pred"}, a.boot, say)
    # ---- S2: gene-held-out
    lk = (d.split == "locked").values
    d["ours2"] = np.nan
    d.loc[lk, "ours2"] = fit_predict(X, d, (d.split == "dev").values, lk)
    s2 = d[lk & d.oligoai_pred.notna()].copy()
    report(s2, "S2  gene-held-out (our 81 LOCKED gene groups; OligoAI has seen most of these genes -> favours OligoAI)",
           {"ours (compact LightGBM)": "ours2", "OligoAI (RiNALMo-giga, ~650M)": "oligoai_pred"}, a.boot, say)
    # ---- OligoAI on rows from its own training patents (sanity: should be high) and on genes it never saw
    seen = set(d[d.oa_split == "train"].group)
    s2n = s2[~s2.group.isin(seen)]
    if s2n.screen.nunique() >= 30:
        report(s2n, "S2b locked groups that do NOT occur in OligoAI's training patents (the cleanest gene-held-out subset for OligoAI)",
               {"ours (compact LightGBM)": "ours2", "OligoAI (RiNALMo-giga, ~650M)": "oligoai_pred"}, a.boot, say)
    else:
        say(f"\nS2b: only {s2n.screen.nunique()} locked screens have genes absent from OligoAI's training patents; subset too small to report.")
    say("\nNotes: S2 and S2b depend on OligoAI's checkpoint being trained on the 80% patent split of this table (reproduced from its released code, not verified against the paper).")
    say("OligoAI's cEt wings are encoded as padding in the released pipeline (vocabulary key 'cET' vs data token 'CET'); reproduced here to match its predictions.")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "aso_compare_report.txt"), "w").write("\n".join(rep) + "\n")
    d[["raw_id", "screen", "group", "patent", "oa_split", "split", "y", "ours", "ours2", "oligoai_pred"]].to_csv(os.path.join(a.out, "aso_compare_rows.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prepare", "evaluate"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="work")
    ap.add_argument("--pred", default="")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="shuffle labels within screens (tests the plumbing without looking at the locked groups)")
    a = ap.parse_args()
    (prepare if a.stage == "prepare" else evaluate)(a)


if __name__ == "__main__":
    main()
