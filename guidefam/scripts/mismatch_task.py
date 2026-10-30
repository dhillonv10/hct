"""Mismatch-tolerance task: predict the LFC of single-mismatch variants of a Cas13d guide.

  dev    (default): leave-one-gene-out among the 4 development titration genes (AQR, EIF3B, NUP133, SNRNP200); prints only these.
  --final : train on the 4 development genes -> locked genes SF3B1 + WDR75; train on all 6 HEK genes -> external HAP1 titration (43k single mismatches, 1,082 genes),
            compared with the authors' released TIGER (needs --official-npz).

Metrics (activity = -LFC): within-gene Spearman over the gene's single-mismatch guides, and within-site Spearman among the variants of one perfect-match site
(the pure 'which mismatch hurts most here' signal). Block bootstrap over sites for paired differences.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import cv
from .. import data as D
from .. import mismatch as MM
from .. import metrics as M

DEV = ["AQR", "EIF3B", "NUP133", "SNRNP200"]
LOCKED_T = ["SF3B1", "WDR75"]


def site_rho(df, col, min_n=4):
    out = []
    for _, x in df.groupby("site"):
        if len(x) >= min_n:
            r = M.spearman(x.act, x[col])
            if np.isfinite(r):
                out.append(r)
    return float(np.mean(out)) if out else float("nan")


def gene_rho(df, col):
    return M.within_group(df, "gene", "act", col, min_n=20)


def paired(df, a, b, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    sites = df.site.unique()
    idx = {s: g.index.values for s, g in df.groupby("site")}
    d0 = fn(df, a) - fn(df, b)
    ds = []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(sites, len(sites), replace=True)])
        z = df.loc[ii].reset_index(drop=True)
        ds.append(fn(z, a) - fn(z, b))
    return d0, float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))


def hek_frames(a):
    pm = D.load_tiger(a.data, "off-target")
    allr = D.load_tiger(a.data, "off-target", pm_only=False)
    oof = pd.read_csv(a.oof, index_col=0)["gbm"]
    pm["pm_pred"] = oof
    sm = allr[allr.guide_type == "SM"].copy()
    key = pm.set_index(["gene", "target_seq"])
    sm["pm_pred"] = [key.pm_pred.get((g, t), np.nan) for g, t in zip(sm.gene, sm.target_seq)]
    g = pm.groupby("gene").y
    sm["gmed"], sm["gscale"] = sm.gene.map(g.median()), sm.gene.map(g.quantile(0.9) - g.quantile(0.1))
    sm["ya"] = (sm.gmed - sm.y) / sm.gscale
    sm["act"] = -sm.y
    sm["site"] = sm.gene + ":" + sm.target_seq
    return sm.dropna(subset=["pm_pred"]).reset_index(drop=True)


class PosMean:
    """Position-only baseline (+ optional linear term in the parent's predicted activity)."""
    def __init__(self, with_pm): self.with_pm = with_pm
    def fit(self, X, y):
        self.m = pd.Series(y).groupby(X.mm_pos.values).mean()
        if self.with_pm:
            r = y - X.mm_pos.map(self.m).values
            self.b = np.polyfit(X.pm_pred.values, r, 1)
        return self
    def predict(self, X):
        p = X.mm_pos.map(self.m).fillna(self.m.mean()).values
        return p + (np.polyval(self.b, X.pm_pred.values) if self.with_pm else 0)


def fit_models(Xtr, ytr, which):
    out = {}
    if "posmean" in which: out["posmean"] = PosMean(False).fit(Xtr, ytr)
    if "posmean+pm" in which: out["posmean+pm"] = PosMean(True).fit(Xtr, ytr)
    if "gbm_mm" in which:
        out["gbm_mm"] = cv.GBM(Xtr, n_estimators=600, num_leaves=15, n_jobs=4).fit(pd.DataFrame(index=Xtr.index), ytr)
    if "gbm_nopm" in which:
        cols = [c for c in Xtr.columns if c != "pm_pred"]
        out["gbm_nopm"] = cv.GBM(Xtr, cols=cols, n_estimators=600, num_leaves=15, n_jobs=4).fit(pd.DataFrame(index=Xtr.index), ytr)
    return out


def predict_all(models, X):
    r = {}
    for k, m in models.items():
        r[k] = m.predict(pd.DataFrame(index=X.index)) if hasattr(m, "lgb") else m.predict(X)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", required=True, help="hek_oof.csv (column gbm = leave-one-gene-out normalised PM activity)")
    ap.add_argument("--ext-hap", default=None, help="ext_preds_hap-titration.csv from final_cas13d (column gbm)")
    ap.add_argument("--official-npz", default=None)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    sm = hek_frames(a)
    X = MM.mm_features(sm, sm.pm_pred.values)
    which = ["posmean", "posmean+pm", "gbm_mm", "gbm_nopm"]
    sm = sm.reset_index(drop=True)

    def run(train_genes, test_df, Xtest, label):
        tr = sm.gene.isin(train_genes).values
        models = fit_models(X[tr], sm.ya.values[tr], which)
        for k, m in models.items():
            if hasattr(m, "lgb"):
                m.X = X
        preds = {}
        for k, m in models.items():
            if hasattr(m, "lgb"):
                m.X = Xtest
                preds[k] = m.predict(test_df)
            else:
                preds[k] = m.predict(Xtest)
        return preds

    print("=== DEV: leave-one-gene-out among", DEV, "(single-mismatch variants)")
    parts = []
    for g in DEV:
        te = (sm.gene == g).values
        tr_genes = [h for h in DEV if h != g]
        tr = sm.gene.isin(tr_genes).values
        models = fit_models(X[tr], sm.ya.values[tr], which)
        pr = {}
        for k, m in models.items():
            if hasattr(m, "lgb"):
                m.X = X
                pr[k] = m.predict(sm[te])
            else:
                pr[k] = m.predict(X[te])
        parts.append(sm[te].assign(**pr))
    dev = pd.concat(parts, ignore_index=True)
    for k in which:
        print(f"  {k:12s} within-gene {gene_rho(dev, k):.3f}   within-site {site_rho(dev, k):.3f}")
    if not a.final:
        print("\n(dev only; run with --final for locked genes and the external HAP1 comparison once the design is fixed)")
        return

    print("\n=== LOCKED HEK genes", LOCKED_T, "(trained on the 4 development genes)")
    te = sm.gene.isin(LOCKED_T).values
    tr = sm.gene.isin(DEV).values
    models = fit_models(X[tr], sm.ya.values[tr], which)
    pr = {}
    for k, m in models.items():
        if hasattr(m, "lgb"):
            m.X = X
            pr[k] = m.predict(sm[te])
        else:
            pr[k] = m.predict(X[te])
    lk = sm[te].assign(**pr).reset_index(drop=True)
    for k in which:
        print(f"  {k:12s} within-gene {gene_rho(lk, k):.3f}   within-site {site_rho(lk, k):.3f}")

    # ---- external HAP1
    print("\n=== EXTERNAL HAP1 titration single mismatches (trained on all 6 HEK titration genes)")
    h = D.load_tiger(a.data, "hap-titration", pm_only=False)
    hp = h[h.guide_type == "PM"].reset_index(drop=True)
    hs = h[h.guide_type == "SM"].reset_index(drop=True)
    ex = pd.read_csv(a.ext_hap)
    pmap = ex.drop_duplicates(["gene", "guide_id"]).set_index(["gene", "guide_id"]).gbm
    pm_by_site = {(g, t): pmap.get((g, i), np.nan) for g, t, i in zip(hp.gene, hp.target_seq, hp.guide_id)}
    hs["pm_pred"] = [pm_by_site.get((g, t), np.nan) for g, t in zip(hs.gene, hs.target_seq)]
    hs = hs.dropna(subset=["pm_pred"]).reset_index(drop=True)
    hs["act"] = -hs.y
    hs["site"] = hs.gene + ":" + hs.target_seq
    Xh = MM.mm_features(hs, hs.pm_pred.values).reindex(columns=X.columns)
    models = fit_models(X, sm.ya.values, which)
    for k, m in models.items():
        if hasattr(m, "lgb"):
            m.X = Xh
            hs[k] = m.predict(hs)
        else:
            hs[k] = m.predict(Xh)
    cols = list(which)
    if a.official_npz:
        from .. import official_tiger as OT
        hs["official"] = OT.predict(OT.load(a.official_npz), hs, guide_col="guide_seq")
        cols = ["official"] + cols
        rk = lambda c: hs.groupby("gene")[c].rank(pct=True)
        hs["gbm_mm+official"] = (rk("gbm_mm") + rk("official")) / 2
        cols += ["gbm_mm+official"]
    print(f"  {len(hs)} single-mismatch guides, {hs.gene.nunique()} genes, {hs.site.nunique()} sites")
    for k in cols:
        print(f"  {k:16s} within-gene {gene_rho(hs, k):.3f}   within-site {site_rho(hs, k):.3f}   pooled Spearman {M.spearman(hs.act, hs[k]):.3f}")
    if a.official_npz:
        for k in ("gbm_mm", "gbm_nopm", "posmean+pm", "gbm_mm+official"):
            for nm, fn in (("within-gene", gene_rho), ("within-site", site_rho)):
                d0, lo, hi = paired(hs, k, "official", fn, n=200)
                print(f"  paired {nm:11s} {k:16s} - official: {d0:+.3f} [{lo:+.3f},{hi:+.3f}]")
    hs[["gene", "guide_id", "act"] + cols].to_csv("mismatch_hap_preds.csv", index=False)


if __name__ == "__main__":
    main()
