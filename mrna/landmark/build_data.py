#!/usr/bin/env python3
"""Build the extended training bundle: multi-task labels, the extra trainable genes, and either species.

Three things this adds over the first bundle, all of which are data we already had and were discarding:

  1. MULTI-TASK LABELS. The authors' PC1 is a compression of 54 individual decay datasets into one number.
     Those 54 columns hold 489,052 non-NA measurements against PC1's 19,503 -- 25x the supervision on the
     same sequences. Training 54 heads and reporting PC1 is strictly more information for the same compute.
  2. THE EXTRA GENES. 5,582 genes carry individual decay measurements but no PC1 (2,523 of them with >=3
     datasets). They cannot be SCORED on the primary benchmark but they can be TRAINED on, and they are
     marked so the evaluation never touches them.
  3. MOUSE. A second published benchmark (Saluki mouse r=0.7320) and ~14.5k more sequences.

  python3 build_data.py --species human --gencode /home/user/lncfam_data/ucsc --work /home/user/hl_work
  python3 build_data.py --species mouse --gencode /home/user/mm_data      --work /home/user/hl_work
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import halflife as hl                                            # noqa: E402  reuse the verified builders

REPO = "/home/user/vagarwal87/saluki_paper"
GENCODE_FILES = {
    "human": ("wgEncodeGencodeAttrsV50.txt.gz", "wgEncodeGencodeCompV50.txt.gz", "hg38.fa.gz"),
    "mouse": ("attrsVM37.txt.gz", "compVM37.txt.gz", "mm39.fa.gz"),
}


def labels(species):
    """PC1 (the scored target) plus the 54/27 individual decay datasets (extra supervision)."""
    pc1 = pd.read_csv(f"{REPO}/Fig2_S3/all_HLs_{species}_PC1.txt", sep="\t")
    pc1 = pc1.rename(columns={f"{species}_PC1": "y"})[["ensembl_gene_id", "y"]].dropna()
    allhl = pd.read_csv(f"{REPO}/Fig1_S1_S2/all_HLs_{species}.txt.gz", sep="\t")
    task_cols = [c for c in allhl.columns if c not in ("ensembl_gene_id", "external_gene_name")]
    print(f"{species}: PC1 on {len(pc1)} genes; {len(task_cols)} individual datasets on {len(allhl)} genes")
    print(f"  non-NA label cells: {int(allhl[task_cols].notna().to_numpy().sum()):,} "
          f"(vs {len(pc1):,} for PC1 alone)")
    return pc1, allhl, task_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="human", choices=["human", "mouse"])
    ap.add_argument("--gencode", required=True)
    ap.add_argument("--work", default="/home/user/hl_work")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sp = a.species
    os.makedirs(a.work, exist_ok=True)
    pc1, allhl, task_cols = labels(sp)

    # train on the union; score only where PC1 exists
    want = set(pc1.ensembl_gene_id) | set(allhl.ensembl_gene_id)
    print(f"  union gene set for sequence reconstruction: {len(want)}")

    attrs_f, comp_f, fa_f = GENCODE_FILES[sp]
    txp = os.path.join(a.work, f"tx2_{sp}.pkl")
    if os.path.exists(txp):
        d = pd.read_pickle(txp)
        print(f"  reusing {txp}: {len(d)} transcripts")
    else:
        attrs = hl.read_attrs(os.path.join(a.gencode, attrs_f), want)
        models = hl.read_models(os.path.join(a.gencode, comp_f), set(attrs))
        rep = hl.pick_representative(attrs, models)
        print(f"  {len(models)} transcript models -> {len(rep)} representatives; building sequences ...")
        tx = hl.build_transcripts(os.path.join(a.gencode, fa_f), rep)
        rows = []
        for gid, (seq, cds, junc) in tx.items():
            if len(seq) < 100 or seq.count("N") > 0.01 * len(seq):
                continue
            rows.append(dict(gene=gid, seq=seq,
                             cds_start=int(np.argmax(cds)) if cds.any() else -1,
                             cds_len=int(cds.sum()), n_exons=len(junc),
                             junc=",".join(map(str, junc[:-1])), length=len(seq)))
        d = pd.DataFrame(rows)
        d.to_pickle(txp)
        print(f"  wrote {txp}: {len(d)} transcripts")

    # align labels
    d = d.merge(pc1, on=None, how="left", left_on="gene", right_on="ensembl_gene_id").drop(
        columns=["ensembl_gene_id"], errors="ignore")
    M = allhl.set_index("ensembl_gene_id")[task_cols]
    Y = M.reindex(d.gene).to_numpy(dtype=np.float32)
    scored = d.y.notna().to_numpy()
    n_task = np.isfinite(Y).sum(1)
    keep = scored | (n_task >= 3)                           # a gene must be scorable OR carry real supervision
    d, Y, scored = d[keep].reset_index(drop=True), Y[keep], scored[keep]
    print(f"\n  kept {len(d)} transcripts: {scored.sum()} scorable on PC1, "
          f"{(~scored).sum()} train-only (>=3 individual datasets)")
    print(f"  multi-task matrix {Y.shape}, {int(np.isfinite(Y).sum()):,} observed cells "
          f"({100*np.isfinite(Y).mean():.1f}% dense)")

    # features (the same verified extractor as halflife.py)
    xp = os.path.join(a.work, f"X2_{sp}.npy")
    if os.path.exists(xp):
        X = np.load(xp); cols = json.load(open(os.path.join(a.work, f"cols2_{sp}.json")))
        print(f"  reusing {xp}: {X.shape}")
    else:
        from multiprocessing import Pool
        import time
        recs = d.to_dict("records")
        t0, rows = time.time(), []
        with Pool(max(1, os.cpu_count() or 1)) as p:
            for i, r in enumerate(p.imap(hl._one, recs, chunksize=16)):
                rows.append(r)
                if i and i % 2000 == 0:
                    el = time.time() - t0
                    print(f"    features {i}/{len(recs)}  eta {el/i*(len(recs)-i)/60:.1f} min", flush=True)
        F = pd.DataFrame(rows); cols = list(F.columns)
        X = np.nan_to_num(F.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.save(xp, X); json.dump(cols, open(os.path.join(a.work, f"cols2_{sp}.json"), "w"))
        print(f"  features {X.shape} in {time.time()-t0:.0f}s")

    # pack
    out = a.out or os.path.join(a.work, f"hl_bundle2_{sp}.npz")
    CODE = {c: i for i, c in enumerate("ACGT")}
    L = d.length.values.astype(np.int32)
    seq = np.empty(int(L.sum()), np.uint8); soff = np.zeros(len(d) + 1, np.int64)
    jun, joff = [], np.zeros(len(d) + 1, np.int64)
    p = 0
    for i, r in enumerate(d.itertuples(index=False)):
        s = r.seq.upper()
        seq[p:p + len(s)] = np.frombuffer(bytes(CODE.get(c, 4) for c in s), np.uint8)
        p += len(s); soff[i + 1] = p
        j = [int(x) for x in r.junc.split(",")] if r.junc else []
        jun.extend(j); joff[i + 1] = len(jun)
    np.savez_compressed(
        out, seq=seq, seq_off=soff, junc=np.array(jun, np.int32), junc_off=joff, length=L,
        cds_start=d.cds_start.values.astype(np.int32), cds_len=d.cds_len.values.astype(np.int32),
        n_exons=d.n_exons.values.astype(np.int32),
        y=np.nan_to_num(d.y.values.astype(np.float32), nan=0.0),
        scored=scored.astype(np.uint8), Y=Y, task_names=np.array(task_cols, dtype=object),
        gene=np.array(d.gene.values, dtype=object), X=X.astype(np.float32),
        feat_cols=np.array(cols, dtype=object), species=np.array([sp], dtype=object))
    print(f"\nwrote {out}  {os.path.getsize(out)/1e6:.1f} MB")
    z = np.load(out, allow_pickle=True)
    i = 0
    s = "".join("ACGTN"[c] for c in z["seq"][z["seq_off"][i]:z["seq_off"][i + 1]])
    assert len(s) == z["length"][i]
    print(f"roundtrip ok: {z['gene'][i]} len {len(s)} scored={bool(z['scored'][i])} "
          f"tasks observed={int(np.isfinite(z['Y'][i]).sum())}")


if __name__ == "__main__":
    main()
