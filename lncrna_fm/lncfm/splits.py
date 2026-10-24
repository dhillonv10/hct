"""Leakage-aware splits for LncBench v2.

* LOCKED TEST  = lncRNAs on chr8/13/18/21. Never used for model selection; evaluated once with `--final`.
* DEV FOLDS    = the other chromosomes, split into 5 chromosome groups balanced by gene count
                 (leave-chromosome-group-out CV).
* HOMOLOGY     = genes whose representative transcripts share many 10-mers (paralogs, same repeat family).
                 An item is flagged `leak` if it has a homolog on the training side of its split
                 (another dev fold for dev items; any dev gene for test items). Metrics are reported on all
                 items AND on the leak-free ("pruned") subset; a large drop means the model is memorising
                 families, not generalising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gencode, paths, retrieval

TEST_CHROMS = {"chr8", "chr13", "chr18", "chr21"}
K, MIN_SHARED, MIN_CONT = 10, 40, 0.25          # edges stored at >=25% containment (superset)
FLAG_CONT = 0.40                                # a gene is "homologous" if it shares >=40% of the shorter one's 10-mers.
# 25% flags 77% of genes (generic Alu/L1 sharing); 40% flags 14%; 50% flags 7%  (measured on GENCODE v50).


def rep_genes() -> pd.DataFrame:
    return gencode.representative_per_gene(gencode.build_table("human", "lnc"))


def dev_fold_map() -> dict:
    rep = rep_genes()
    counts = rep[~rep.chrom.isin(TEST_CHROMS)].chrom.value_counts()
    loads, fmap = np.zeros(5), {}
    for chrom, n in counts.items():                    # greedy: biggest chromosome -> lightest fold
        f = int(np.argmin(loads))
        fmap[chrom] = f
        loads[f] += n
    return fmap


def homology_edges(force=False) -> pd.DataFrame:
    """Pairs (a, b, containment) of lncRNA genes (gene_id_nov) with shared 10-mer content above threshold."""
    out = paths.ensure(paths.DATA / "processed") / "homology_edges.pkl"
    if out.exists() and not force:
        return pd.read_pickle(out)
    rep = rep_genes()
    A = retrieval.kmer_set_matrix(list(rep.seq), K).tocsr()
    size = np.asarray(A.sum(1)).ravel()
    AT = A.T.tocsc()
    edges = []
    step = 400
    for s in range(0, A.shape[0], step):
        e = min(s + step, A.shape[0])
        S = (A[s:e] @ AT[:, s:]).tocoo()                # only columns >= s (upper triangle incl. block diagonal)
        r, c, v = S.row + s, S.col + s, S.data
        m = (v >= MIN_SHARED) & (c > r)
        r, c, v = r[m], c[m], v[m]
        cont = v / np.minimum(size[r], size[c])
        m = cont >= MIN_CONT
        edges.append(np.stack([r[m], c[m], cont[m]], 1))
        if (s // step) % 10 == 0:
            print(f"  homology rows {e}/{A.shape[0]}  edges so far {sum(len(x) for x in edges)}", flush=True)
    E = np.concatenate(edges)
    ids = rep.gene_id_nov.values
    df = pd.DataFrame({"a": ids[E[:, 0].astype(int)], "b": ids[E[:, 1].astype(int)], "cont": E[:, 2]})
    df.to_pickle(out)
    return df


def attach(df: pd.DataFrame, gene_col="gene_id_nov", chrom_col="chrom", use_homology=True, min_cont=FLAG_CONT) -> pd.DataFrame:
    """Adds: fold (0-4 dev, -1 locked test), split ('dev'/'test'), leak (bool)."""
    df = df.copy()
    df["_row"] = np.arange(len(df))                    # original row index, to realign label matrices
    fmap = dev_fold_map()
    is_test = df[chrom_col].isin(TEST_CHROMS).values
    fold = np.array([-1 if t else fmap.get(c, -2) for c, t in zip(df[chrom_col], is_test)])
    df["fold"], df["split"] = fold, np.where(is_test, "test", "dev")
    df = df[df.fold != -2].reset_index(drop=True)      # chrM / scaffolds
    leak = np.zeros(len(df), bool)
    if use_homology:
        E = homology_edges()
        E = E[E.cont >= min_cont]
        pos = {g: i for i, g in enumerate(df[gene_col].values)}
        fl = df.fold.values
        for a, b in zip(E.a.values, E.b.values):
            ia, ib = pos.get(a), pos.get(b)
            if ia is None or ib is None:
                continue
            fa, fb = fl[ia], fl[ib]
            if fa == fb:
                continue                                  # same fold (or both test): not a train/test leak
            if fa == -1:                                  # a is test, b is dev -> test item leaks
                leak[ia] = True
            elif fb == -1:
                leak[ib] = True
            else:                                         # different dev folds: each is test for its own fold
                leak[ia] = leak[ib] = True
    df["leak"] = leak
    return df
