#!/usr/bin/env python3
"""Pack everything the landmark model needs into one portable .npz.

Sequences are stored as uint8 codes (0-3 = ACGT, 4 = N) in one flat array with offsets, so the bundle stays
small and loads instantly. Junctions are stored the same ragged way. Nothing in the bundle needs the genome,
the GENCODE tables, or the 18.8 GB Saluki datapack -- the training script is self-contained once this exists.

  python3 make_bundle.py --work /home/user/hl_work --species human --out hl_bundle_human.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

CODE = {c: i for i, c in enumerate("ACGT")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/user/hl_work")
    ap.add_argument("--species", default="human")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = pd.read_pickle(os.path.join(a.work, f"tx_{a.species}.pkl"))
    X = np.load(os.path.join(a.work, f"X_{a.species}.npy"))
    cols = json.load(open(os.path.join(a.work, f"cols_{a.species}.json")))
    assert len(d) == len(X), (len(d), len(X))

    L = d.length.values.astype(np.int32)
    print(f"{len(d)} transcripts, total {L.sum() / 1e6:.1f} Mnt")
    for q in (50, 90, 95, 99, 100):
        print(f"  length p{q}: {int(np.percentile(L, q))}")
    for cap in (8192, 12288, 16384):
        print(f"  would be truncated at {cap}: {(L > cap).sum()} ({100 * (L > cap).mean():.2f}%)")

    seq = np.empty(int(L.sum()), np.uint8)
    soff = np.zeros(len(d) + 1, np.int64)
    jun_all, joff = [], np.zeros(len(d) + 1, np.int64)
    p = 0
    for i, r in enumerate(d.itertuples(index=False)):
        s = r.seq.upper()
        seq[p:p + len(s)] = np.frombuffer(
            bytes(CODE.get(c, 4) for c in s), dtype=np.uint8)
        p += len(s)
        soff[i + 1] = p
        j = [int(x) for x in r.junc.split(",")] if r.junc else []
        jun_all.extend(j)
        joff[i + 1] = len(jun_all)
    assert p == L.sum()

    np.savez_compressed(
        a.out,
        seq=seq, seq_off=soff,
        junc=np.array(jun_all, np.int32), junc_off=joff,
        length=L,
        cds_start=d.cds_start.values.astype(np.int32),
        cds_len=d.cds_len.values.astype(np.int32),
        n_exons=d.n_exons.values.astype(np.int32),
        y=d.y.values.astype(np.float32),
        gene=np.array(d.gene.values, dtype=object),
        X=X.astype(np.float32),
        feat_cols=np.array(cols, dtype=object),
    )
    mb = os.path.getsize(a.out) / 1e6
    print(f"\nwrote {a.out}  {mb:.1f} MB")
    z = np.load(a.out, allow_pickle=True)
    print("keys:", sorted(z.files))
    i = 0
    s = "".join("ACGTN"[c] for c in z["seq"][z["seq_off"][i]:z["seq_off"][i + 1]])
    assert len(s) == z["length"][i]
    print(f"roundtrip ok: gene {z['gene'][i]} len {len(s)} y {z['y'][i]:.3f}")
    cj = int(z["cds_start"][i])
    if cj >= 0 and z["cds_len"][i] > 0:
        print(f"  CDS starts '{s[cj:cj+3]}'")


if __name__ == "__main__":
    main()
