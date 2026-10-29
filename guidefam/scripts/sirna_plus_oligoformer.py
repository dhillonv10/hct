"""Combine our siRNA scores with OligoFormer's own predictions (run on your machine) and test whether the average beats both.

  1. Run OligoFormer (its README) trained on Huesken and predicting Mix and Taka; save a CSV per test set with columns  siRNA,score  (higher = more efficient).
     Its repo's `test.py` path (python scripts/main.py --datasets Hu Mix -t) evaluates Hu -> Mix; the same with `Hu Taka` for Takayuki.
  2. python -m guidefam.scripts.sirna_plus_oligoformer --ours RESULTS/preds_Mix.csv --theirs oligoformer_Mix.csv --name Mix

`--ours` is the file final_sirna.py wrote (columns include sirna, label and 'ridge+gbm ensemble (ours)'). Scores are rank-averaged (rank within the whole test set;
siRNA efficacy is compared across the set, unlike Cas13d where ranking is within gene). Reports PCC / AUC / PRC for each and for the average, with block-bootstrap CIs
and the paired difference average - OligoFormer.   --selftest checks the plumbing on synthetic scores.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .final_sirna import boot, metrics


def combine(ours: pd.DataFrame, theirs: pd.DataFrame, col: str = "ridge+gbm ensemble (ours)") -> pd.DataFrame:
    t = theirs.rename(columns={theirs.columns[0]: "sirna", theirs.columns[1]: "oligoformer"})
    t["sirna"] = t["sirna"].astype(str).str.upper().str.replace("T", "U")
    x = ours.copy()
    x["sirna"] = x["sirna"].astype(str).str.upper().str.replace("T", "U")
    x = x.merge(t.drop_duplicates("sirna"), on="sirna", how="inner")
    rk = lambda v: rankdata(v) / len(v)
    x["ours"] = x[col]
    x["average"] = (rk(x["ours"]) + rk(x["oligoformer"])) / 2
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours")
    ap.add_argument("--theirs")
    ap.add_argument("--name", default="Mix")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(0)
        n = 300
        y = rng.uniform(size=n)
        ours = pd.DataFrame({"sirna": [f"S{i}" for i in range(n)], "label": y, "ridge+gbm ensemble (ours)": y + rng.normal(size=n) * 0.6})
        theirs = pd.DataFrame({"siRNA": ours.sirna, "score": y + rng.normal(size=n) * 0.6})
    else:
        ours, theirs = pd.read_csv(a.ours), pd.read_csv(a.theirs)
    x = combine(ours, theirs)
    y = x["label"].values
    yb = (y > 0.7).astype(int)                                       # test.py convention for inter-dataset evaluation
    blocks = np.arange(len(x)) // 10
    print(f"{a.name}: {len(x)} siRNAs matched ({len(ours)} ours, {len(theirs)} theirs)")
    for c in ("ours", "oligoformer", "average"):
        m, ci = metrics(y, yb, x[c].values), boot(y, yb, x[c].values, blocks, n=500)
        print(f"  {c:12s} AUC {m['AUC']:.3f} [{ci['AUC'][0]:.3f},{ci['AUC'][1]:.3f}]  PRC {m['PRC']:.3f}  PCC {m['PCC']:.3f} [{ci['PCC'][0]:.3f},{ci['PCC'][1]:.3f}]")
    rng = np.random.default_rng(1)
    d = []
    for _ in range(500):
        ii = rng.integers(0, len(x), len(x))
        if len(np.unique(yb[ii])) < 2:
            continue
        d.append(metrics(y[ii], yb[ii], x["average"].values[ii])["PCC"] - metrics(y[ii], yb[ii], x["oligoformer"].values[ii])["PCC"])
    print(f"  paired PCC(average) - PCC(OligoFormer): {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
