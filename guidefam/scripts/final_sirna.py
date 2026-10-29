"""FINAL siRNA evaluation. Design frozen on Huesken contiguous-chunk CV (sirna_dev*.py); Mix and Takayuki are read here for the first time.

Protocol = OligoFormer's inter-dataset validation: train on Huesken, test on Mixset (and, additionally, on Takayuki).
Metrics = theirs: AUC / PRC against the binary label (>70 % of maximal inhibition) and Pearson r with the normalised efficacy.
Published OligoFormer numbers (Table 2 of Bai et al. 2024, Bioinformatics) are printed for reference; they come from their own runs, not ours.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import cv
from .. import sirna as SI

PUBLISHED_MIX = {"OligoFormer": dict(AUC=0.8163, PRC=0.8143, PCC=0.5879), "DSIR": dict(AUC=0.7483, PRC=0.7637, PCC=0.503),
                 "OligoWalk": dict(AUC=0.708, PRC=0.6998, PCC=0.5592), "Monopoli-RF": dict(AUC=0.7439, PRC=0.7388, PCC=0.4628),
                 "iScore": dict(AUC=0.7462, PRC=0.7565, PCC=0.3687), "s-Biopredsi": dict(AUC=0.7374, PRC=0.7557, PCC=0.3849),
                 "siRNAPred": dict(AUC=0.5782, PRC=0.618, PCC=0.2062)}


def metrics(y, yb, p):
    return {"PCC": float(np.corrcoef(y, p)[0, 1]), "SRCC": float(spearmanr(y, p)[0]), "AUC": float(roc_auc_score(yb, p)), "PRC": float(average_precision_score(yb, p))}


def boot(y, yb, p, blocks, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    draws = []
    for _ in range(n):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub), replace=True)])
        if len(np.unique(yb[ii])) < 2:
            continue
        draws.append(metrics(y[ii], yb[ii], p[ii]))
    return {k: (float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5))) for k in draws[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oligo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr = SI.load(f"{a.oligo}/Hu.csv")
    Xtr_b = SI.tabular(tr)
    base = list(Xtr_b.columns)
    rk = lambda v: rankdata(v) / len(v)
    gp = dict(n_jobs=2, n_estimators=300, num_leaves=7, min_child_samples=15)

    def feats(d):
        X = SI.tabular(d)
        fl = SI.flank_frame(d, 8)
        return pd.concat([X, fl], axis=1), list(fl.columns)

    Xtr, flc = feats(tr)
    models = {
        "ridge (no flank)": cv.Ridge(Xtr, cols=base, alpha=30.0).fit(tr, tr.y.values),
        "ridge": cv.Ridge(Xtr, cols=base + flc, alpha=100.0).fit(tr, tr.y.values),
    }
    gbm = cv.GBM(Xtr, cols=base + flc, **gp).fit(tr, tr.y.values)
    all_rows = []
    for name in ("Mix", "Taka"):
        te = SI.load(f"{a.oligo}/{name}.csv")
        Xte, _ = feats(te)
        y, yb = te.y.values, te["label"].gt(0.7).astype(int).values
        blocks = SI.target_groups(te.win) if name == "Mix" else np.arange(len(te)) // 25
        preds = {"Reynolds rules": Xte["reynolds"].values, "Ui-Tei rules": Xte["uitei"].values}
        for mn, m in models.items():
            m.X = Xte
            preds[mn] = m.predict(te)
        gbm.X = Xte
        preds["gbm"] = gbm.predict(te)
        preds["ridge+gbm ensemble (ours)"] = (rk(preds["ridge"]) + rk(preds["gbm"])) / 2
        pd.DataFrame(preds).assign(label=y, y_bin=yb, sirna=te.sirna.values).to_csv(os.path.join(a.out, f"preds_{name}.csv"), index=False)
        print(f"\n=== train Huesken ({len(tr)}) -> test {name} ({len(te)} siRNAs, {len(np.unique(blocks))} blocks; {yb.mean():.0%} effective)")
        for pn, p in preds.items():
            m = metrics(y, yb, p)
            ci = boot(y, yb, p, blocks)
            row = {"test": name, "model": pn, **m}
            all_rows.append(row)
            print(f"  {pn:28s} AUC {m['AUC']:.3f} [{ci['AUC'][0]:.3f},{ci['AUC'][1]:.3f}]  PRC {m['PRC']:.3f}  PCC {m['PCC']:.3f} [{ci['PCC'][0]:.3f},{ci['PCC'][1]:.3f}]  SRCC {m['SRCC']:.3f}")
        if name == "Mix":
            print("  -- published, same protocol (Bai et al. 2024, Table 2, 'Inter-dataset' = Huesken -> Mixset) --")
            for k, v in PUBLISHED_MIX.items():
                print(f"  {k:28s} AUC {v['AUC']:.3f}  PRC {v['PRC']:.3f}  PCC {v['PCC']:.3f}")
    pd.DataFrame(all_rows).to_csv(os.path.join(a.out, "final_sirna_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
