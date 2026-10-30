"""POST-HOC siRNA evaluation of TabDPT (a pretrained tabular model, not trained on biology) as the learner, same features as the frozen ridge + GBM design.

Mix and Takayuki had already been read when this was written, so treat it as exploratory, not as a clean held-out test.
    python -m guidefam.scripts.sirna_tabdpt_final --oligo OLIGOFORMER/data --ours RESULTS_DIR_with_preds_Mix.csv_and_preds_Taka.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import cv
from .. import sirna as SI
from .final_sirna import boot, metrics


def feats(d):
    X = SI.tabular(d)
    fl = SI.flank_frame(d, 8)
    return pd.concat([X, fl], axis=1), list(X.columns), list(fl.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oligo", required=True)
    ap.add_argument("--ours", required=True, help="dir with preds_Mix.csv / preds_Taka.csv from final_sirna.py")
    ap.add_argument("--n-ens", type=int, default=8)
    a = ap.parse_args()
    tr = SI.load(f"{a.oligo}/Hu.csv")
    Xtr, base, flc = feats(tr)
    rk = lambda v: rankdata(v) / len(v)
    out = []
    for flank in (False, True):
        cols = base + (flc if flank else [])
        m = cv.TabDPT(Xtr, cols=cols, n_ensembles=a.n_ens).fit(tr, tr.y.values)
        for name in ("Mix", "Taka"):
            te = SI.load(f"{a.oligo}/{name}.csv")
            Xte = feats(te)[0]
            m.X = Xte
            p = m.predict(te)
            ours = pd.read_csv(os.path.join(a.ours, f"preds_{name}.csv"))
            assert np.allclose(ours["label"].values, te["label"].values)
            y, yb = te.y.values, SI.binary(te, "inter")
            blocks = SI.target_groups(te.win) if name == "Mix" else np.arange(len(te)) // 25
            variants = {f"TabDPT{' + flank' if flank else ''}": p,
                        f"ens(frozen ridge+GBM, TabDPT{' + flank' if flank else ''})": (rk(ours["ridge+gbm ensemble (ours)"].values) + rk(p)) / 2}
            for k, v in variants.items():
                mt, ci = metrics(y, yb, v), boot(y, yb, v, blocks, n=500)
                print(f"{name:5s} {k:48s} AUC {mt['AUC']:.3f} [{ci['AUC'][0]:.3f},{ci['AUC'][1]:.3f}]  PRC {mt['PRC']:.3f}  PCC {mt['PCC']:.3f} [{ci['PCC'][0]:.3f},{ci['PCC'][1]:.3f}]", flush=True)
                out.append({"test": name, "model": k, **mt})
    pd.DataFrame(out).to_csv("sirna_tabdpt_posthoc.csv", index=False)


if __name__ == "__main__":
    main()
