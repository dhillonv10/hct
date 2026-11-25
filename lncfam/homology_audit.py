#!/usr/bin/env python3
"""T2 -- how much homology leakage can pruning actually remove from a lncRNA benchmark?

Why this exists. LncBench reports every metric twice, on all items and on a homology-pruned subset, and the
committed v2 run shows the two agree to within 0.003 everywhere. That looks like a clean null, but the counts
say otherwise: the run scored an 800-gene SUBSAMPLE, pruning removed 3 items out of 700, and 3 is what
subsampling alone predicts. With ~36.5k lncRNA genes in GENCODE v50 and a 700-gene sample, the chance that both
members of a homolog pair survive is (700/36476)^2, so only one or two pairs can possibly be present. The
committed comparison therefore cannot detect leakage in either direction, and the null it appears to show is an
artifact.

This script answers the precondition at full size, over every GENCODE v50 lncRNA gene:
  * what fraction of genes have a homolog at 10-mer containment >= C, for a range of C;
  * what fraction have one ACROSS the train/test boundary, which is the only kind pruning can remove.
The second number decides whether homology control is a real contribution of the benchmark or a no-op.

Splits are replicated exactly from lncrna_fm/lncfm/splits.py: chr8/13/18/21 are the locked test; the remaining
chromosomes are packed greedily (largest first onto the lightest fold) into 5 dev folds by gene count. A pair is
a cross-fold leak when its two genes sit in different folds, since each dev fold is a test fold in its turn.

Data comes from UCSC's mirror of GENCODE v50 because ftp.ebi.ac.uk is unreachable from this container.

  python3 homology_audit.py --data /home/user/lncfam_data/ucsc --out results/t2_homology.txt
"""
from __future__ import annotations

import argparse
import gzip
import os
import time

import numpy as np
from scipy.sparse import csr_matrix

K = 10                                     # k-mer size, as in lncfm/splits.py
MIN_SHARED = 40                            # candidate pairs need this many shared k-mers before containment is computed
THRESHOLDS = [0.10, 0.25, 0.40, 0.50, 0.70]
FLAG_CONT = 0.40                           # the threshold LncBench actually uses
TEST_CHROMS = {"chr8", "chr13", "chr18", "chr21"}
MAIN_CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}


def read_attrs(path):
    """transcript id -> (gene id, gene type). 15 tab-separated fields; indices verified against the file."""
    tx = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            tx[f[4]] = (f[0], f[2])
    return tx


def read_comp(path, keep_tx):
    """genePred rows for the transcripts we want: name -> (chrom, strand, exon starts, exon ends)."""
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11 or f[1] not in keep_tx:
                continue
            chrom = f[2]
            if chrom not in MAIN_CHROMS:                       # drop scaffolds and alt contigs
                continue
            starts = [int(x) for x in f[9].rstrip(",").split(",")]
            ends = [int(x) for x in f[10].rstrip(",").split(",")]
            out[f[1]] = (chrom, f[3], starts, ends)
    return out


def pick_representative(tx_attrs, models):
    """One transcript per lncRNA gene: the longest spliced one, as lncfm's representative_per_gene does."""
    best = {}
    for tid, (chrom, strand, starts, ends) in models.items():
        gid, gtype = tx_attrs[tid]
        if gtype != "lncRNA":
            continue
        length = sum(e - s for s, e in zip(starts, ends))
        if gid not in best or length > best[gid][0]:
            best[gid] = (length, tid, chrom, strand, starts, ends)
    return best


def extract_sequences(fa_gz, rep):
    """Spliced sequence per gene, read one chromosome at a time so the genome is never all in memory."""
    by_chrom = {}
    for gid, (_, _, chrom, strand, starts, ends) in rep.items():
        by_chrom.setdefault(chrom, []).append((gid, strand, starts, ends))
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    seqs, cur, name, buf = {}, None, None, []

    def flush():
        if name is None or name not in by_chrom:
            return
        s = "".join(buf).upper()
        for gid, strand, starts, ends in by_chrom[name]:
            e = "".join(s[a:b] for a, b in zip(starts, ends))
            seqs[gid] = e if strand == "+" else e.translate(comp)[::-1]
        print(f"    {name}: {len(by_chrom[name])} genes", flush=True)

    with gzip.open(fa_gz, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                flush()
                name, buf = line[1:].split()[0], []
            elif name in by_chrom:
                buf.append(line.strip())
    flush()
    return seqs


def fold_map(rep):
    """Greedy chromosome packing into 5 dev folds, largest chromosome onto the lightest fold (lncfm's rule)."""
    counts = {}
    for _, (_, _, chrom, _, _, _) in rep.items():
        if chrom not in TEST_CHROMS:
            counts[chrom] = counts.get(chrom, 0) + 1
    loads, fmap = np.zeros(5), {}
    for chrom, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        f = int(np.argmin(loads))
        fmap[chrom] = f
        loads[f] += counts[chrom]
    return fmap


def kmer_matrix(seqs, k=K):
    """Binary gene x k-mer sparse matrix over ACGT k-mers (k-mers containing N are dropped)."""
    base = np.full(256, -1, dtype=np.int64)
    for i, c in enumerate("ACGT"):
        base[ord(c)] = i
    indptr, indices = [0], []
    for s in seqs:
        a = base[np.frombuffer(s.encode(), dtype=np.uint8)]
        if len(a) < k:
            indptr.append(indptr[-1])
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, k)
        ok = (w >= 0).all(1)
        codes = np.unique((w[ok] * (4 ** np.arange(k - 1, -1, -1))).sum(1)) if ok.any() else np.empty(0, np.int64)
        indices.append(codes)
        indptr.append(indptr[-1] + len(codes))
    idx = np.concatenate(indices) if indices else np.empty(0, np.int64)
    return csr_matrix((np.ones(len(idx), dtype=np.float32), idx, np.array(indptr)), shape=(len(seqs), 4 ** k))


def edges(A, step=300):
    """Upper-triangle pairs with >= MIN_SHARED shared k-mers; containment = shared / min(set sizes)."""
    size = np.asarray(A.sum(1)).ravel()
    AT = A.T.tocsc()
    out, t0 = [], time.time()
    for s in range(0, A.shape[0], step):
        e = min(s + step, A.shape[0])
        S = (A[s:e] @ AT[:, s:]).tocoo()
        r, c, v = S.row + s, S.col + s, S.data
        m = (v >= MIN_SHARED) & (c > r)
        r, c, v = r[m], c[m], v[m]
        if len(r):
            cont = v / np.minimum(size[r], size[c])
            keep = cont >= min(THRESHOLDS)
            out.append(np.stack([r[keep], c[keep], cont[keep]], 1))
        if (s // step) % 20 == 0:
            done = sum(len(x) for x in out)
            print(f"    rows {e}/{A.shape[0]}  pairs {done}  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out) if out else np.empty((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/t2_homology.txt")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only this many genes (0 = all)")
    a = ap.parse_args()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    print("reading GENCODE v50 tables ...", flush=True)
    attrs = read_attrs(os.path.join(a.data, "wgEncodeGencodeAttrsV50.txt.gz"))
    lnc_tx = {t for t, (_, gt) in attrs.items() if gt == "lncRNA"}
    models = read_comp(os.path.join(a.data, "wgEncodeGencodeCompV50.txt.gz"), lnc_tx)
    rep = pick_representative(attrs, models)
    say(f"GENCODE v50: {len(lnc_tx)} lncRNA transcripts -> {len(rep)} lncRNA genes on chr1-22,X,Y")
    if a.limit:
        rep = {g: v for g, v in sorted(rep.items())[:a.limit]}
        say(f"SMOKE TEST: restricted to {len(rep)} genes -- rates below are NOT the real answer")

    print("extracting spliced sequences ...", flush=True)
    seqs = extract_sequences(os.path.join(a.data, "hg38.fa.gz"), rep)
    gids = [g for g in rep if g in seqs and len(seqs[g]) >= K]
    say(f"sequences recovered for {len(gids)} genes "
        f"(median length {int(np.median([len(seqs[g]) for g in gids]))} nt)")

    fmap = fold_map(rep)
    fold = np.array([-1 if rep[g][2] in TEST_CHROMS else fmap[rep[g][2]] for g in gids])
    say(f"folds: locked test (chr8/13/18/21) {int((fold == -1).sum())} genes; "
        + ", ".join(f"fold {f} {int((fold == f).sum())}" for f in range(5)))

    print("building k-mer matrix ...", flush=True)
    A = kmer_matrix([seqs[g] for g in gids])
    say(f"k-mer matrix: {A.shape[0]} genes x {A.shape[1]} possible {K}-mers, {A.nnz / 1e6:.1f}M nonzeros")

    print("finding homologous pairs ...", flush=True)
    E = edges(A)
    say(f"candidate pairs at containment >= {min(THRESHOLDS)}: {len(E)}")

    say()
    say("HOMOLOGY AT FULL SIZE  (all GENCODE v50 lncRNA genes, not a subsample)")
    say(f"  {'containment':>12s} {'pairs':>9s} {'genes with a homolog':>22s} {'genes with a CROSS-FOLD homolog':>33s}")
    n = len(gids)
    rows = []
    for c in THRESHOLDS:
        m = E[:, 2] >= c
        r, cc = E[m, 0].astype(int), E[m, 1].astype(int)
        any_hom = np.unique(np.concatenate([r, cc])) if m.any() else np.array([], int)
        cross = fold[r] != fold[cc]
        leaky = np.unique(np.concatenate([r[cross], cc[cross]])) if cross.any() else np.array([], int)
        rows.append((c, int(m.sum()), len(any_hom), len(leaky)))
        say(f"  {c:>12.2f} {int(m.sum()):>9d} {len(any_hom):>13d} ({100 * len(any_hom) / n:4.1f}%) "
            f"{len(leaky):>22d} ({100 * len(leaky) / n:4.1f}%)")

    flag = [r for r in rows if abs(r[0] - FLAG_CONT) < 1e-9][0]
    say()
    say(f"At the threshold LncBench uses ({FLAG_CONT:.2f}), homology pruning would remove "
        f"{flag[3]} of {n} genes ({100 * flag[3] / n:.1f}%).")
    say("VERDICT: " + ("pruning is a no-op; chromosome-group splitting already separates lncRNA homologs, so "
                      "homology control is not a contribution of this benchmark."
                      if 100 * flag[3] / n < 2.0 else
                      "pruning removes a material fraction, so the all-vs-pruned comparison is worth running at "
                      "full size on a labelled task."))
    say()
    say("Caveat: 10-mer containment is a crude homology measure. It catches paralogs and shared repeat families "
        "and will miss diverged homologs that an alignment would find.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
