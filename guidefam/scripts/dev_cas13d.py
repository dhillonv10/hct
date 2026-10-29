"""Development experiments for the Cas13d on-target model. Scores ONLY the 12 development genes (LOCKED_GENES and the external screens are never touched here).

    python -m guidefam.scripts.dev_cas13d --data DATA --out RESULTS --exps base,rank,hap1,flow,hap1+flow
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .. import cv
from .. import data as D
from .. import features as F
from .. import metrics as M


def combined(data_dir):
    hek = D.load_tiger(data_dir, "off-target").assign(source="hek")
    hap = D.load_tiger(data_dir, "hap-titration").assign(source="hap1")
    flo = D.load_tiger(data_dir, "flow-cytometry").assign(source="flow")
    A = pd.concat([hek, hap, flo], ignore_index=True)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exps", default="base,rank,hap1,flow,hap1+flow")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--struct", default=None, help="dir with struct_off-target.csv.gz / struct_flow-cytometry.csv.gz from guidefam.structure")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    A = combined(a.data)
    X = F.tabular(A, "full", flank=20)
    for s in ("hap1", "flow"):
        X[f"src_{s}"] = (A.source == s).astype(float).values
    if a.struct:
        st = pd.concat([pd.read_csv(os.path.join(a.struct, f"struct_{n}.csv.gz")) for n in ("off-target", "flow-cytometry")])
        st = st.drop_duplicates(["gene", "guide_id"]).set_index(["gene", "guide_id"])
        S = st.reindex(pd.MultiIndex.from_arrays([A.gene.values, A.guide_id.values])).set_axis(A.index)
        S.columns = [c if c.startswith(("acc", "pu_")) else c for c in S.columns]
        X = pd.concat([X, S], axis=1)
    hek = A[A.source == "hek"]
    dev_genes = [g for g in sorted(hek.gene.unique()) if g not in D.LOCKED_GENES]
    path = os.path.join(a.out, "dev_oof.csv")
    oof = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame(index=hek.index)
    def flank_cols(k):
        """feature columns with k flanking nt on each side (f0..f19 = 5' flank far->near, f20..f39 = 3' flank near->far)"""
        keep = {f"f{p}_{n}" for p in list(range(20 - k, 20)) + list(range(20, 20 + k)) for n in F.NT}
        return [c for c in X.columns if not c.startswith(("src_", "acc", "pu_")) and not (c.startswith("f") and c[1].isdigit()) or c in keep]
    base_cols = flank_cols(2)

    for name in a.exps.split(","):
        if name in oof:
            print("have", name)
            continue
        mode = "rank" if "rank" in name else "tiger"
        srcs = [s for s in ("hap1", "flow") if s in name]
        extra = A[A.source.isin(srcs)] if srcs else None
        kf = [int(t[5:]) for t in name.split("+") if t.startswith("flank")]
        cols = (flank_cols(kf[0]) if kf else base_cols) + ([f"src_{s}" for s in srcs] if "flag" in name else [])
        if a.struct:
            agg = [c for c in X.columns if c.startswith("acc")]
            if name == "acc":
                cols = cols + agg + [c for c in X.columns if c.startswith("pu_")]
            elif name == "acc_agg":
                cols = cols + agg
            elif name.startswith("accW"):
                cols = cols + [c for c in agg if c.startswith("acc" + name[4:] + "_")]
        t0 = time.time()
        p = cv.logo_predict(hek, lambda: cv.GBM(X, cols=cols, n_jobs=a.threads), genes=dev_genes, extra=extra, mode=mode, verbose=False)
        oof[name] = p
        oof.to_csv(path)
        print(f"[{name}] extra={srcs or '-'} mode={mode}  {time.time() - t0:.0f}s", flush=True)

    ev = hek.loc[hek.gene.isin(dev_genes), ["gene", "block", "act", "active"]].join(oof)
    ref = "base"
    rows = []
    for c in oof.columns:
        r = M.block_bootstrap(ev, lambda x, c=c: M.summarize(x, c), n=a.boot)
        row = {"exp": c, "rho_within": f"{r['rho_within'][0]:.3f} [{r['rho_within'][1]:.3f},{r['rho_within'][2]:.3f}]",
               "auroc_within": f"{r['auroc_within'][0]:.3f}", "rho_pooled": f"{r['rho_pooled'][0]:.3f}"}
        if c != ref and ref in oof:
            d0, lo, hi, _ = M.paired_bootstrap(ev, c, ref, lambda x, col: M.within_group(x, "gene", "act", col), n=a.boot)
            row["delta_vs_base"] = f"{d0:+.3f} [{lo:+.3f},{hi:+.3f}]"
        rows.append(row)
    print(f"development genes only (n={len(dev_genes)}; locked genes and external screens untouched)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
