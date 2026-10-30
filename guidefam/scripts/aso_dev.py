"""ASO Atlas, DEVELOPMENT genes only: T0 baselines and the split-leakage test. Locked genes are dropped at load and never fitted, scored or printed.

  python -m guidefam.scripts.aso_dev --csv data_untrusted/aso_zenodo_20794660/aso.csv.gz --cache data_untrusted/cache --out guidefam/results/aso

Reports within-screen Spearman (mean over screens with n >= 20; 'group' column = mean over gene groups of their screens' mean, so a gene with 82 screens counts once).
  1. no-fit single features (what a rule of thumb gets)
  2. LightGBM, gene-held-out 5-fold, feature ablation: base (sequence + chemistry + experiment) / + thermodynamics / + target accessibility
  3. the same model under four split types: gene groups held out (the honest one), patents held out (OligoAI's released rule), screens held out, random rows held out
  4. target = raw inhibition vs within-screen rank-Gauss (computed on training rows only)
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold, KFold

from .. import aso as A
from .aso_access import load_cache

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=40, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
              reg_lambda=5.0, n_jobs=4, verbose=-1, random_state=0)
THERMO = ["dg_duplex", "dg_self"]
ACCESS = ["site_found", "u_site", "u_site5", "u_site3", "u_min", "u_fl5", "u_fl3", "u_ctx", "mfe_ctx"]


def rank_gauss_by_screen(screen: pd.Series, y: pd.Series) -> np.ndarray:
    r = y.groupby(screen).rank(method="average")
    n = y.groupby(screen).transform("size")
    return np.asarray(norm.ppf((r - 0.5) / n))


def oof(X, y, folds, target="raw", screen=None):
    p = np.full(len(X), np.nan)
    cats = [c for c in ("transfection", "cell_line") if c in X]
    Xc = X.copy()
    for c in cats:
        Xc[c] = Xc[c].where(Xc[c] >= 0).astype("category")
    for tr, te in folds:
        yt = y.iloc[tr]
        t = yt.values if target == "raw" else rank_gauss_by_screen(screen.iloc[tr], yt)
        m = lgb.LGBMRegressor(**PARAMS).fit(Xc.iloc[tr], t)
        p[te] = m.predict(Xc.iloc[te])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = A.load(a.csv)
    d = d[d.split == "dev"].reset_index(drop=True)
    print(f"DEV only: {len(d)} rows, {d.group.nunique()} gene groups, {d.screen.nunique()} screens", flush=True)
    acc, idx = load_cache(a.cache)
    X = A.features(d, acc, idx)
    base = [c for c in X.columns if c not in THERMO + ACCESS]
    sets = {"base": base, "base+thermo": base + THERMO, "base+thermo+access": base + THERMO + ACCESS}
    print(f"features: base {len(base)}, +thermo {len(THERMO)}, +access {len(ACCESS)}; site found in context for {X.site_found.mean():.1%} of rows", flush=True)

    rows = []

    def record(name, col, extra=""):
        s = A.summarize(d, col)
        pt, lo, hi = A.bootstrap(d, col, n=a.boot)
        rows.append(dict(model=name, split=extra, **s, ci_lo=lo, ci_hi=hi))
        print(f"  {name:42s} {extra:8s} rho_screen {s['rho_screen']:.3f} [{lo:.3f},{hi:.3f}]  rho_group {s['rho_group']:.3f}  ({s['n_screens']} screens)", flush=True)

    print("\n1. single features, no fitting (sign chosen on the full dev set, so slightly optimistic)")
    for c in ["gc", "dg_duplex", "dg_self", "u_site", "u_min", "frac_PS", "log_dose", "len"]:
        s = A.summarize(d.assign(f=X[c]), "f")
        s2 = A.summarize(d.assign(f=-X[c]), "f")
        best = max((s["rho_screen"], 1), (s2["rho_screen"], -1))
        print(f"  {c:12s} rho_screen {best[0]:+.3f} (sign {best[1]:+d})", flush=True)
        rows.append(dict(model=f"single:{c}", split="none", **(s if best[1] == 1 else s2)))

    gk = lambda g: list(GroupKFold(a.folds).split(X, groups=g))
    d["patent"] = d.screen.map(lambda s: s.split("/")[-1].split("_table_")[0])                 # OligoAI's released code splits by patent
    folds_gene, folds_screen, folds_patent = gk(d.group.values), gk(d.screen.values), gk(d.patent.values)
    folds_rand = list(KFold(a.folds, shuffle=True, random_state=0).split(X))

    print("\n2. LightGBM, gene groups held out, feature ablation (raw inhibition target)")
    for nm, cols in sets.items():
        t = time.time()
        d["p_" + nm] = oof(X[cols], d.y, folds_gene)
        record("gbm:" + nm, "p_" + nm, "gene")
        print(f"     {time.time() - t:.0f}s", flush=True)

    best = "base+thermo+access"
    print("\n3. same model, three split types")
    for sp, fo in (("patent", folds_patent), ("screen", folds_screen), ("random", folds_rand)):
        d[f"p_{best}_{sp}"] = oof(X[sets[best]], d.y, fo)
        record("gbm:" + best, f"p_{best}_{sp}", sp)

    print("\n4. within-screen rank-Gauss target (gene held out)")
    d["p_rank"] = oof(X[sets[best]], d.y, folds_gene, target="rank", screen=d.screen)
    record("gbm:" + best + " rankgauss", "p_rank", "gene")

    pcols = [c for c in d if c.startswith("p_")]
    d[["aso", "screen", "group", "y"] + pcols].to_csv(os.path.join(a.out, "aso_dev_oof.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "aso_dev_table.csv"), index=False)

    print("\npaired differences (gene split, bootstrap over gene groups)")
    for lab, c1, c2 in (("+thermo vs base", "p_base+thermo", "p_base"), ("+access vs +thermo", "p_base+thermo+access", "p_base+thermo"),
                        ("patent-split minus gene-split", f"p_{best}_patent", "p_" + best),
                        ("screen-split minus gene-split", f"p_{best}_screen", "p_" + best), ("random-split minus gene-split", f"p_{best}_random", "p_" + best)):
        pt, lo, hi = A.bootstrap(d, c1, c2, n=a.boot)
        print(f"  {lab:34s} {pt:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
