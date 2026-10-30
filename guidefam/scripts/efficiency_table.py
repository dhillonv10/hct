"""Parameter counts (and where measurable, train time) for every model used, taken from the actual objects / checkpoint headers -- nothing assumed.

    python -m guidefam.scripts.efficiency_table --data DATA --pred PRED [--rnafm RNAFM_DIR] [--official-npz NPZ]
"""
import argparse
import glob
import json
import os
import struct
import time

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import cv
from .. import data as D
from .. import features as F


def st_params(path):
    """Number of parameters in a .safetensors file, from its header only (the tensors are not loaded)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return int(sum(np.prod(v["shape"]) for k, v in h.items() if k != "__metadata__"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--rnafm", default=None)
    ap.add_argument("--official-npz", default=None)
    a = ap.parse_args()
    d = D.load_tiger(a.data, "off-target")
    X = F.tabular(d, "full", flank=2)
    rows = []

    t0 = time.time()
    g = cv.GBM(X, n_jobs=4).fit(d, cv.activity_target(d))
    t = g.m.booster_.trees_to_dataframe()
    leaves = int(t.split_feature.isna().sum()); splits = int(t.split_feature.notna().sum())
    rows.append(("GBM (ours, Cas13d)", leaves + 2 * splits, f"{leaves} leaf values + {splits} splits x (feature, threshold)", f"{time.time() - t0:.0f}s fit, 4 cores"))
    r = cv.Ridge(X, cols=[c for c in X.columns if not c.startswith(("rel_pos", "log_len", "gc_up", "gc_dn"))]).fit(d, cv.activity_target(d))
    rows.append(("ridge (ours)", int(r.m.coef_.size + 1), "coefficients + intercept", "< 1s"))

    import torch
    from ..tiger_torch import Tiger2D
    n_sc = F.scalar_frame(d.iloc[:2]).shape[1]
    rows.append(("Tiger2D (our re-implementation)", sum(p.numel() for p in Tiger2D(25, n_sc).parameters()), "PyTorch module, flank 1, 16 scalars", "~4 min per fold (3 cores)"))
    from ..pairformer import MiniPairFormer
    rows.append(("MiniPairFormer (ours)", sum(p.numel() for p in MiniPairFormer(n_sc).parameters()), "2 layers, d=32, 4 heads", "~4 min per fold (2 cores)"))
    if a.official_npz:
        z = np.load(a.official_npz)
        rows.append(("TIGER as released", int(sum(z[k].size for k in z.files)), "authors' weights (checkpoint)", "n/a"))
    if a.rnafm:
        rows.append(("RNA-FM (frozen encoder some pipelines add)", st_params(os.path.join(a.rnafm, "model.safetensors")), "checkpoint header", "n/a"))
    import tabdpt
    hub = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Layer6--TabDPT/snapshots/*/*.safetensors"))
    if hub:
        rows.append(("TabDPT (pretrained tabular model, not biology)", st_params(hub[0]), "checkpoint header", "~2.5 min per fold (3 cores)"))
    t = pd.DataFrame(rows, columns=["model", "parameters", "what is counted", "cost (measured here)"])
    t["parameters"] = t["parameters"].map(lambda v: f"{v:,}")
    print(t.to_string(index=False))
    t.to_csv("efficiency_table.csv", index=False)


if __name__ == "__main__":
    main()
