"""FINAL Cas13d on-target evaluation (design frozen on the 12 development genes; the 4 locked genes and both external screens are read for the first time here).

  A. HEK293T, leave-one-gene-out predictions for all 16 genes -> reported on all 16, on the 12 development genes, and on the 4 locked genes.
  B. External: train on all 16 HEK293T genes, predict flow-cytometry CD46/CD55/TFRC (different lab/assay) and HAP1 (1,082 genes, different cell line).
     Konermann-rules and the random forest were developed on the CD46/CD55/TFRC flow screens, so those two are NOT external tests for them (marked).
Everything is scored with within-gene Spearman (the 'which guides do I order' metric) plus AUROC, with block-bootstrap CIs and paired deltas.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import comparators as C
from .. import cv
from .. import data as D
from .. import features as F
from .. import metrics as M


def rk_within(df, col):
    """Rank-average ensembling must happen within gene (scores are only comparable inside a gene)."""
    return df.groupby("gene")[col].rank(pct=True)


def table(ev, cols, ref, boot, label):
    rows = []
    for c in cols:
        x = ev[ev[c].notna()].reset_index(drop=True)
        r = M.block_bootstrap(x, lambda z, c=c: M.summarize(z, c), n=boot)
        row = {"set": label, "model": c, "n": r["n"][0], "rho_within": f"{r['rho_within'][0]:.3f} [{r['rho_within'][1]:.3f},{r['rho_within'][2]:.3f}]",
               "auroc_within": f"{r['auroc_within'][0]:.3f}"}
        if ref and c != ref and ref in x:
            both = x[x[ref].notna()].reset_index(drop=True)
            d0, lo, hi, _ = M.paired_bootstrap(both, c, ref, lambda z, col: M.within_group(z, "gene", "act", col), n=boot)
            row[f"delta vs {ref}"] = f"{d0:+.3f} [{lo:+.3f},{hi:+.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tiger-oof", required=True, help="oof_hek293t_logo.csv written by ladder_cas13d for tiger2d_port")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--ref", default="tiger2d_port")
    ap.add_argument("--skip-external-tiger", action="store_true")
    ap.add_argument("--official-npz", default=None, help="weights exported by export_official_tiger (NYGC non-commercial licence); adds the released TIGER to the external tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hek = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)
    X = F.tabular(hek, "full", flank=2)
    gbm_make = lambda: cv.GBM(X, n_jobs=4)
    path = os.path.join(a.out, "hek_oof.csv")
    oof = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame(index=hek.index)
    if "gbm" not in oof:
        oof["gbm"] = cv.logo_predict(hek, gbm_make, verbose=False)
    oof["tiger2d_port"] = pd.read_csv(a.tiger_oof, index_col=0)["tiger2d_port"].reindex(hek.index)
    oof.to_csv(path)
    ev = pd.concat([hek[["gene", "block", "act", "active", "deepcas13", "konermann", "rf"]], oof], axis=1)
    ev["gbm+konermann"] = (rk_within(ev, "gbm") + rk_within(ev, "konermann")) / 2             # fixed first, on the 12 development genes (0.636 there)
    ev["ens3"] = (rk_within(ev, "gbm") + rk_within(ev, "tiger2d_port") + rk_within(ev, "konermann")) / 3   # best of 5 pre-specified combos on dev (0.643 there)
    ev["gbm+tiger2d"] = (rk_within(ev, "gbm") + rk_within(ev, "tiger2d_port")) / 2             # supplementary: no third-party tool, so clean on every external set
    cols = ["deepcas13", "rf", "konermann", "tiger2d_port", "gbm", "gbm+tiger2d", "gbm+konermann", "ens3"]
    out = []
    for label, m in (("HEK293T all 16 genes", ev.gene.notna()), ("HEK293T 12 development genes", ~ev.gene.isin(D.LOCKED_GENES)), ("HEK293T 4 LOCKED genes", ev.gene.isin(D.LOCKED_GENES))):
        t = table(ev[m].reset_index(drop=True), cols, a.ref, a.boot, label)
        out.append(t)
        print(f"\n=== {label}  (n={int(m.sum())}); LOGO: the scored gene was never in training")
        print(t.drop(columns="set").to_string(index=False))
        sub = ev[m].reset_index(drop=True)
        for hi, lo_ in (("ens3", "tiger2d_port"), ("ens3", "konermann"), ("ens3", "gbm"), ("gbm+konermann", "tiger2d_port"), ("gbm+konermann", "konermann"), ("gbm+konermann", "gbm"), ("gbm+tiger2d", "tiger2d_port")):
            d0, lo, hi_, share = M.paired_bootstrap(sub, hi, lo_, lambda z, col: M.within_group(z, "gene", "act", col), n=a.boot)
            print(f"   paired: {hi} - {lo_}: {d0:+.3f} [{lo:+.3f},{hi_:+.3f}]")

    # ---- external: train on every HEK293T gene
    ya = cv.activity_target(hek)
    gbm = cv.GBM(X, n_jobs=4).fit(hek, ya)
    for name in ("flow-cytometry", "hap-titration"):
        ext = D.load_tiger(a.data, name)
        if name == "flow-cytometry":
            ext = C.add_all(ext, a.pred, sub="flow-cytometry")
            print("\nflow-cytometry: published-tool coverage", {c: round(float(ext[c].notna().mean()), 3) for c in ("deepcas13", "konermann", "rf")},
                  "| sign check (Spearman with measured activity, must be > 0):", {k: round(v, 3) for k, v in C.orient(ext, ["deepcas13", "konermann", "rf"]).items()})
        Xe = F.tabular(ext, "full", flank=2).reindex(columns=X.columns)
        if name == "hap-titration":
            Xe = Xe.copy()
            for c in [c for c in X.columns if c in ("log_gene_len",) or c.startswith(("rel_pos", "log_len", "gc_up", "gc_dn"))]:
                Xe[c] = np.nan
        gbm.X = Xe
        ext["gbm"] = gbm.predict(ext)
        if "konermann" in ext:
            ext["gbm+konermann"] = (rk_within(ext, "gbm") + rk_within(ext, "konermann")) / 2
        gbm.X = X
        if not a.skip_external_tiger:
            from ..tiger_torch import Tiger2DRegressor
            common = [c for c in F.scalar_frame(ext.iloc[:2]).columns if c in F.scalar_frame(hek.iloc[:2]).columns]
            t = Tiger2DRegressor(flank=1, scalars=True, cols=common)
            hb = hek.copy()
            t.fit(hb, ya)
            ext["tiger2d_port"] = t.predict(ext)
            ext["gbm+tiger2d"] = (rk_within(ext, "gbm") + rk_within(ext, "tiger2d_port")) / 2
            if "konermann" in ext:
                ext["ens3"] = (rk_within(ext, "gbm") + rk_within(ext, "tiger2d_port") + rk_within(ext, "konermann")) / 3
        ext = ext.reset_index(drop=True)
        if a.official_npz:
            from .. import official_tiger as OT
            ext["official"] = OT.predict(OT.load(a.official_npz), ext)
        ext[[c for c in ("gene", "guide_id", "y", "act", "active", "deepcas13", "konermann", "rf", "gbm", "tiger2d_port", "gbm+tiger2d", "gbm+konermann", "ens3") if c in ext]].to_csv(
            os.path.join(a.out, f"ext_preds_{name}.csv"), index=False)
        cs = [c for c in ("deepcas13", "rf", "konermann", "official", "tiger2d_port", "gbm", "gbm+tiger2d", "gbm+konermann", "ens3") if c in ext and ext[c].notna().mean() > 0.5]
        min_n = 30 if name == "flow-cytometry" else 5
        ext["block"] = ext["block"] if name == "flow-cytometry" else ext["gene"]
        tt = []
        for c in cs:
            x = ext[ext[c].notna()].reset_index(drop=True)
            r = M.block_bootstrap(x, lambda z, c=c: {"rho_within": M.within_group(z, "gene", "act", c, M.spearman, min_n=min_n),
                                                     "auroc_pooled": M.auroc(z["active"], z[c])}, n=a.boot)
            tt.append({"set": name, "model": c + (" (developed on these screens)" if name == "flow-cytometry" and c in ("konermann", "rf", "gbm+konermann", "ens3") else ""),
                       "n": len(x), "rho_within": f"{r['rho_within'][0]:.3f} [{r['rho_within'][1]:.3f},{r['rho_within'][2]:.3f}]", "auroc_pooled": f"{r['auroc_pooled'][0]:.3f}"})
        t = pd.DataFrame(tt)
        out.append(t)
        print(f"\n=== EXTERNAL {name}: trained on all 16 HEK293T genes (n={len(ext)}, {ext.gene.nunique()} genes)")
        print(t.drop(columns="set").to_string(index=False))
        if "official" in ext:
            for m_ in ("tiger2d_port", "gbm", "gbm+tiger2d"):
                d0, lo, hi = M.paired_within_gene(ext, m_, "official", block="block", n=min(a.boot, 300) if name != "flow-cytometry" else a.boot, min_n=min_n)
                print(f"   paired: {m_} - released TIGER (sequence-only): {d0:+.3f} [{lo:+.3f},{hi:+.3f}]")
    pd.concat(out).to_csv(os.path.join(a.out, "final_cas13d_tables.csv"), index=False)


if __name__ == "__main__":
    main()
