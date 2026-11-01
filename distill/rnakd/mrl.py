"""Mean ribosome load (Sample et al. 2019, GEO GSE114002): download, the authors' splits, and the clean splits this study adds.

RiNALMo's own code (rinalmo/data/downstream/ribosome_loading/dataset.py) builds:
    drop NaN rl -> keep set in {random, human} -> total_reads >= 10 -> drop_duplicates('utr', keep=False)
    Random7600 = for each length 25..100, the top 100 rows by total_reads from the RANDOM library
    Human7600  = the same rule on the HUMAN library
    train      = every remaining random row
and then uses Random7600 as the VALIDATION set (checkpoint selection) and Human7600 as the test set. So the published
"Random7600 R2 = 0.925" is a number on the set used for early stopping, not a clean held-out test.

This module reproduces that exactly (`repro_val`, `ood_test`) so published numbers can be checked, and adds:
    iid_test   = the NEXT top-100-per-length block of random rows, built by the same selection rule, never used for
                 training or selection -> a clean in-distribution test matched in read depth to the other two
    val        = a random slice of what remains, used only for early stopping
    train      = the rest
Reporting IID and OOD on sets built by one rule is what makes the retention comparison in PILOT_PROTOCOL.md meaningful.
"""
from __future__ import annotations

import io
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd

GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114002&format=file"
MEMBER = "GSM4084997_varying_length_25to100.csv.gz"
LENGTHS = range(25, 101)
TOP_PER_LEN = 100


def download(workdir: str) -> str:
    """Fetch the GEO supplementary tar once and extract the one member we need. Returns the local .csv.gz path."""
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, MEMBER)
    if os.path.exists(out):
        return out
    tar_path = os.path.join(workdir, "GSE114002_RAW.tar")
    if not os.path.exists(tar_path):
        print(f"downloading {GEO_URL} ...")
        urllib.request.urlretrieve(GEO_URL, tar_path)
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        hit = [n for n in names if n.endswith(MEMBER)]
        if not hit:
            raise SystemExit(f"{MEMBER} not in the GEO archive; members are {names[:20]}")
        with tf.extractfile(hit[0]) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def _top_per_length(df: pd.DataFrame, k: int = TOP_PER_LEN) -> pd.Index:
    """Indices of the k rows with most reads at each UTR length, the authors' selection rule."""
    idx = []
    ln = df.utr.str.len()
    for L in LENGTHS:
        sub = df[ln == L]
        if len(sub):
            idx.extend(sub.nlargest(k, "total_reads").index.tolist())
    return pd.Index(idx)


def splits(csv_path: str, seed: int = 0, val_frac: float = 0.05, top_k: int = TOP_PER_LEN):
    """Returns a dict of DataFrames: train, val, iid_test, ood_test, repro_val. Columns: utr, rl, total_reads, set."""
    d = pd.read_csv(csv_path)
    need = {"utr", "rl", "total_reads", "set"}
    if not need.issubset(d.columns):
        raise SystemExit(f"expected columns {sorted(need)}; the file has {list(d.columns)[:20]}")
    d = d.dropna(subset=["rl"])
    d = d[d["set"].isin(["random", "human"])]
    d = d[d.total_reads >= 10]
    d = d.drop_duplicates("utr", keep=False).reset_index(drop=True)

    rnd, hum = d[d["set"] == "random"], d[d["set"] == "human"]
    repro_val = d.loc[_top_per_length(rnd, top_k)]                                      # the authors' "Random7600" (their validation set)
    rest = rnd.drop(index=repro_val.index)
    iid_test = d.loc[_top_per_length(rest, top_k)]                                      # ours: same rule, next block, never selected on
    rest = rest.drop(index=iid_test.index)
    ood_test = d.loc[_top_per_length(hum, top_k)]                                       # the authors' "Human7600"

    rs = np.random.default_rng(seed)
    perm = rs.permutation(len(rest))
    n_val = int(round(val_frac * len(rest)))
    val, train = rest.iloc[perm[:n_val]], rest.iloc[perm[n_val:]]
    out = dict(train=train, val=val, iid_test=iid_test, ood_test=ood_test, repro_val=repro_val)
    for k, v in out.items():
        print(f"  {k:10s} n={len(v):6d}  rl mean {v.rl.mean():.3f} sd {v.rl.std():.3f}  len {v.utr.str.len().min()}-{v.utr.str.len().max()}")
    assert not set(train.utr) & set(ood_test.utr) and not set(train.utr) & set(iid_test.utr), "split leak"
    return out


def synthetic(n=12000, seed=0):
    """A stand-in table with the same columns and a learnable signal, so the whole pipeline can be rehearsed offline."""
    rs = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        L = int(rs.integers(25, 101))
        s = "".join(rs.choice(list("ACGT"), L))
        # signal: upstream AUGs depress load, GC raises it; the 'human' library gets a shifted rule -> a real distribution shift
        kind = "random" if i % 5 else "human"
        gc = (s.count("G") + s.count("C")) / L
        y = 5 + 3 * gc - 1.2 * s.count("ATG") + (0.8 * s.count("TAA") if kind == "human" else 0.0) + rs.normal(0, 0.3)
        rows.append(dict(utr=s, rl=y, total_reads=int(rs.integers(10, 500)), set=kind))
    return pd.DataFrame(rows).drop_duplicates("utr", keep=False).reset_index(drop=True)
