"""Denoised-residual-guided correction: screen features on the variant-denoised site estimate (development titration genes only), fit a small linear
correction on the same sites, then test it on genes and screens that played no part in the screen or the fit.

  screen + fit : titration genes that are NOT locked: AQR, EIF3B, NUP133, SNRNP200 (~100 sites each, independent variant-based activity estimate)
  evaluation   : (a) the 2 locked titration genes SF3B1, WDR75 (all ~1,000 guides each)  (b) the other development genes (no variant panels, never used)
                 (c) external flow-cytometry (base = GBM + Tiger2D, no third-party tool)
The base score is the within-gene rank-average ensemble; the corrected score adds a within-gene-rank linear term in the selected features.

    python -m guidefam.scripts.residual_correction --data DATA --oof hek_oof.csv --pred PRED --struct STRUCT_DIR --ext-flow ext_preds_flow-cytometry.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from .. import comparators as C
from .. import data as D
from .. import features as F
from .. import metrics as M
from .feature_discovery import bh

FIT_GENES = ["AQR", "EIF3B", "NUP133", "SNRNP200"]


def pct(x):
    return rankdata(x) / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--struct", required=True, help="dir with struct_off-target.csv.gz and struct_flow-cytometry.csv.gz")
    ap.add_argument("--ext-flow", required=True)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    allr = D.load_tiger(a.data, "off-target", pm_only=False)
    pm = C.add_all(D.load_tiger(a.data, "off-target"), a.pred).join(pd.read_csv(a.oof, index_col=0)[["gbm", "tiger2d_port"]])
    rkw = lambda df, c: df.groupby("gene")[c].rank(pct=True)
    pm["ens3"] = (rkw(pm, "gbm") + rkw(pm, "tiger2d_port") + rkw(pm, "konermann")) / 3
    sm = allr[allr.guide_type == "SM"].copy()
    sm["pos"] = sm.guide_id.str.extract(r"_SM_P(\d+)-")[0].astype(int)
    sm["z"] = sm.groupby("gene").y.transform(lambda s: (s - s.mean()) / s.std())
    V = sm.groupby(["gene", "target_seq", "pos"]).z.mean().unstack("pos").dropna(thresh=20)
    V = V.apply(lambda c: c.fillna(c.mean()))

    def feats(frame, struct_csv):
        X = pd.concat([F.tabular(frame, "full", flank=2), F.thermo_frame(frame), F.motif_frame(frame)], axis=1)
        st = pd.read_csv(struct_csv).drop_duplicates(["gene", "guide_id"]).set_index(["gene", "guide_id"])
        S = st.reindex(pd.MultiIndex.from_arrays([frame.gene.values, frame.guide_id.values])).set_axis(frame.index)
        X = pd.concat([X, S], axis=1)
        return X.loc[:, ~X.columns.duplicated()]

    X = feats(pm, os.path.join(a.struct, "struct_off-target.csv.gz"))
    X = X.loc[:, X.std() > 0]
    pos_of = {k: i for i, k in enumerate(zip(pm.gene.values, pm.target_seq.values))}
    fit = {}
    for g in FIT_GENES:
        v = V.loc[g]
        v = (v - v.mean()) / v.std()
        u, s, _ = np.linalg.svd(v.values, full_matrices=False)
        lat = u[:, 0] * s[0]
        ix = np.array([pos_of[(g, t)] for t in v.index])
        if np.corrcoef(lat, pm.act.values[ix])[0, 1] < 0:
            lat = -lat
        fit[g] = (ix, lat)
    print(f"screen + fit on {sum(len(v[0]) for v in fit.values())} denoised sites in {FIT_GENES}")
    rows = []
    for c in X.columns:
        zs = []
        for g, (ix, lat) in fit.items():
            f = X.iloc[ix][c].values
            if np.nanstd(f) == 0:
                continue
            e, z = pct(pm.ens3.values[ix]), pct(lat)
            rz = z - np.polyval(np.polyfit(e, z, 1), e)
            r = spearmanr(f, rz, nan_policy="omit")[0]
            if np.isfinite(r):
                zs.append(np.arctanh(np.clip(r, -0.999, 0.999)) * np.sqrt(len(f) - 3))
        if zs:
            zc = np.sum(zs) / np.sqrt(len(zs))
            rows.append({"feature": c, "z": zc, "p": 2 * norm.sf(abs(zc))})
    t = pd.DataFrame(rows).dropna()
    t["q"] = bh(t.p.values)
    t = t.sort_values("p")
    sel = t[t.q < a.q].feature.tolist()
    print(t.head(8).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"selected at q < {a.q}: {sel}")
    if not sel:
        return
    # linear correction in WITHIN-GENE z-SCORE space (z-scores over all guides of the gene, so fit and application share one scale), accessibility family only, ridge
    fam = [c for c in sel if c.startswith(("acc", "pu_")) or c == "log_unpaired"]
    print("accessibility-family features used:", fam)

    def zs_gene(df, col_values):
        out = np.zeros(len(df))
        for g in df.gene.unique():
            m = (df.gene == g).values
            v = np.nan_to_num(col_values[m], nan=np.nanmedian(col_values[m]))
            out[m] = (v - v.mean()) / (v.std() + 1e-9)
        return out

    Z_e = zs_gene(pm, pm.ens3.values)
    Z_f = {c: zs_gene(pm, X[c].values) for c in fam}
    A, y = [], []
    for g, (ix, lat) in fit.items():
        A.append(np.column_stack([Z_e[ix]] + [Z_f[c][ix] for c in fam]))
        y.append((lat - lat.mean()) / lat.std())
    A, y = np.vstack(A), np.concatenate(y)
    lam = 20.0
    beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    print("ridge coefficients (z-score space):", dict(zip(["ensemble"] + fam, np.round(beta, 3))))

    def corrected(df, base_col, Xf):
        ze = zs_gene(df, df[base_col].values)
        new = beta[0] * ze
        for b, c in zip(beta[1:], fam):
            new = new + b * zs_gene(df, Xf[c].values)
        return new

    def report(label, df, base_col, new_col):
        df = df.reset_index(drop=True)
        d0, lo, hi = M.paired_within_gene(df, new_col, base_col, block="block", n=a.boot, min_n=20)
        print(f"  {label:44s} base {M.within_group(df, 'gene', 'act', base_col):.4f} -> corrected {M.within_group(df, 'gene', 'act', new_col):.4f}   paired {d0:+.4f} [{lo:+.4f},{hi:+.4f}]")

    pm["new"] = corrected(pm, "ens3", X)
    print("\nEvaluation (within-gene Spearman vs the ordinary perfect-match label; none of these guides' denoised estimates were used in the fit):")
    report("LOCKED titration genes SF3B1 + WDR75 (all guides)", pm[pm.gene.isin(["SF3B1", "WDR75"])], "ens3", "new")
    others = [g for g in pm.gene.unique() if g not in D.LOCKED_GENES and g not in FIT_GENES]
    report(f"development genes without variant panels (n={len(others)})", pm[pm.gene.isin(others)], "ens3", "new")
    report("other locked genes CDC5L + POLA1", pm[pm.gene.isin(["CDC5L", "POLA1"])], "ens3", "new")
    report("titration fit genes (in-sample, for reference only)", pm[pm.gene.isin(FIT_GENES)], "ens3", "new")
    fl = pd.read_csv(a.ext_flow)
    flow = D.load_tiger(a.data, "flow-cytometry").merge(fl[["gene", "guide_id", "gbm+tiger2d"]], on=["gene", "guide_id"])
    Xf = feats(flow, os.path.join(a.struct, "struct_flow-cytometry.csv.gz"))
    flow["new"] = corrected(flow, "gbm+tiger2d", Xf.reindex(columns=fam))
    report("EXTERNAL flow-cytometry (base = GBM + Tiger2D)", flow, "gbm+tiger2d", "new")
    pd.DataFrame({"feature": fam, "beta": beta[1:]}).to_csv("residual_correction_coefs.csv", index=False)


if __name__ == "__main__":
    main()
