"""Published Cas13d tools scored on the TIGER screens (no training; their own predictions).

    python -m guidefam.scripts.eval_published --data DATA_DIR --pred "TIGER_REPO/predictions (other models)"
"""
import argparse

import pandas as pd

from .. import comparators as C
from .. import data as D
from .. import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()

    d = D.load_tiger(a.data, "off-target")
    d = C.add_all(d, a.pred)
    cols = ["deepcas13", "konermann", "rf"]
    print(f"HEK293T PM guides: {len(d)}  genes: {d.gene.nunique()}  active threshold (LFC): {d.attrs['active_threshold']:.3f}  active: {d.active.mean():.3f}")
    print("coverage:", {c: round(float(d[c].notna().mean()), 3) for c in cols})
    print("orientation check (Spearman with measured activity, must be > 0):", {k: round(v, 3) for k, v in C.orient(d, cols).items()})
    rows = []
    for c in cols:
        r = M.block_bootstrap(d[d[c].notna()].reset_index(drop=True), lambda x, c=c: M.summarize(x, c), n=a.boot)
        rows.append({"tool": c, **{k: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for k, v in r.items() if k != "n"}, "n": r["n"][0]})
    print(pd.DataFrame(rows).to_string(index=False))
    # replicate ceiling on the same metric: one replicate vs the mean of the other two
    ceil = []
    for r in D.LFC:
        others = d[[x for x in D.LFC if x != r]].mean(axis=1)
        t = pd.DataFrame({"gene": d.gene, "act": -others, "p": -d[r]}).dropna()
        ceil.append(M.within_group(t, "gene", "act", "p"))
    print("replicate ceiling (within-gene Spearman, one replicate vs mean of others):", [round(x, 3) for x in ceil])


if __name__ == "__main__":
    main()
