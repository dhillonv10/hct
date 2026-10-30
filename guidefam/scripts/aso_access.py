"""Cache per-nucleotide unpaired probabilities of every unique ASO-Atlas target context (ViennaRNA partition function; label-free).

  python -m guidefam.scripts.aso_access --csv data_untrusted/aso_zenodo_20794660/aso.csv.gz --out data_untrusted/cache
Writes aso_access.npy (n_unique x 126, float32) and aso_access_ctx.txt (one context per line, same order). ~5-10 min on 4 cores.
"""
import argparse
import os
import time

import numpy as np

from .. import aso as A


def load_cache(out: str):
    import pandas as pd
    acc = np.load(os.path.join(out, "aso_access.npy"))
    ctx = [l.rstrip("\n") for l in open(os.path.join(out, "aso_access_ctx.txt"))]
    return acc, pd.Series(np.arange(len(ctx)), index=ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = A.load(a.csv)
    ctx = sorted(d.rna_context.dropna().unique())
    print(f"{len(ctx)} unique contexts", flush=True)
    t = time.time()
    acc = A.accessibility(ctx, a.procs)
    np.save(os.path.join(a.out, "aso_access.npy"), acc)
    with open(os.path.join(a.out, "aso_access_ctx.txt"), "w") as f:
        f.write("\n".join(ctx) + "\n")
    print(f"done in {time.time() - t:.0f}s, shape {acc.shape}", flush=True)


if __name__ == "__main__":
    main()
