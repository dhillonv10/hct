"""T3: zero-shot positional-ortholog retrieval (human lncRNA -> mouse lncRNA pool).

For each human query gene, rank all mouse lncRNA genes by cosine similarity of embeddings
(each species mean-centred to remove species-level composition shift). A query is scored by the
rank of its best-ranked positional ortholog. No training on these labels => no leakage.

Subsets:  all (any synteny-defined partner) | strict (1:1 partner) | named (same gene symbol)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import gencode, orthology


def build_retrieval(max_queries: int | None = None, pool_size: int | None = None, seed: int = 0):
    H = gencode.representative_per_gene(gencode.build_table("human", "lnc"))
    M = gencode.representative_per_gene(gencode.build_table("mouse", "lnc"))
    m, _, _ = orthology.positional_orthologs()
    m = m[m.gene_id_h.isin(H.gene_id) & m.gene_id_m.isin(M.gene_id)].copy()
    nh = m.groupby("gene_id_h").gene_id_m.transform("nunique")
    nm = m.groupby("gene_id_m").gene_id_h.transform("nunique")
    m["strict"] = (nh == 1) & (nm == 1)
    m["named"] = m.gene_name_h.str.upper() == m.gene_name_m.str.upper()
    rng = np.random.default_rng(seed)
    qgenes = np.array(sorted(m.gene_id_h.unique()))
    if max_queries and len(qgenes) > max_queries:
        named = set(m[m.named].gene_id_h)
        rest = np.array([g for g in qgenes if g not in named])
        keep = list(named) + list(rng.choice(rest, max_queries - len(named), replace=False))
        qgenes = np.array(sorted(keep))
    m = m[m.gene_id_h.isin(qgenes)]
    pos_genes = set(m.gene_id_m)
    Mp = M
    if pool_size and len(M) > pool_size:
        others = M[~M.gene_id.isin(pos_genes)]
        extra = others.sample(max(0, pool_size - len(pos_genes)), random_state=seed)
        Mp = pd.concat([M[M.gene_id.isin(pos_genes)], extra]).sample(frac=1, random_state=seed)
    Mp = Mp.reset_index(drop=True)
    Q = H[H.gene_id.isin(qgenes)].reset_index(drop=True)
    midx = {g: i for i, g in enumerate(Mp.gene_id)}
    pos, strict, named = [], [], []
    by_q = {g: d for g, d in m.groupby("gene_id_h")}
    for g in Q.gene_id:
        d = by_q[g]
        pos.append(np.array([midx[x] for x in d.gene_id_m if x in midx]))
        strict.append(bool(d.strict.any()))
        named.append(bool(d.named.any()))
    return Q, Mp, pos, {"all": np.ones(len(Q), bool), "strict": np.array(strict), "named": np.array(named)}


def _norm(Z):
    Z = Z - Z.mean(0, keepdims=True)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def rank_metrics(S_chunks, pos):
    """S_chunks yields (start, S) with S (nq_chunk, n_pool) score matrices. Returns per-query best rank."""
    ranks = np.zeros(len(pos), dtype=np.int64)
    for start, S in S_chunks:
        for r in range(S.shape[0]):
            p = pos[start + r]
            best = S[r, p].max()
            ranks[start + r] = 1 + int((S[r] > best).sum())
    return ranks


def summarize(ranks, pos, n_pool, subsets):
    out = {}
    for name, mask in subsets.items():
        r = ranks[mask]
        if len(r) == 0:
            continue
        npos = np.array([len(p) for p, k in zip(pos, mask) if k])
        rand_h10 = float(np.mean(1 - (1 - 10 / n_pool) ** npos))
        out[name] = {"n": int(len(r)), "MRR": float(np.mean(1 / r)), "hit@1": float(np.mean(r <= 1)),
                     "hit@10": float(np.mean(r <= 10)), "hit@100": float(np.mean(r <= 100)),
                     "median_rank": float(np.median(r)), "rand_hit@10": rand_h10}
    return out


def eval_embeddings(Zq, Zp, pos, subsets, chunk=500):
    Qn, Pn = _norm(Zq.astype(np.float32)), _norm(Zp.astype(np.float32))

    def gen():
        for s in range(0, len(Qn), chunk):
            yield s, Qn[s:s + chunk] @ Pn.T

    return summarize(rank_metrics(gen(), pos), pos, len(Pn), subsets)


# ---- alignment-free 'BLAST-lite' baseline: shared 10-mers -----------------------------------
def kmer_set_matrix(seqs, k=10):
    from .kmer import encode_ints
    rows, cols = [], []
    for i, s in enumerate(seqs):
        a = encode_ints(s).astype(np.int64)
        n = len(a) - k + 1
        if n <= 0:
            continue
        idx = np.zeros(n, np.int64)
        ok = np.ones(n, bool)
        for j in range(k):
            x = a[j:j + n]
            ok &= x >= 0
            idx = idx * 4 + np.where(x >= 0, x, 0)
        u = np.unique(idx[ok])
        rows.append(np.full(len(u), i, np.int32))
        cols.append(u.astype(np.int32))
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(seqs), 4 ** k))


def eval_kmer_overlap(Qseqs, Pseqs, pos, subsets, k=10, chunk=300):
    Qm, Pm = kmer_set_matrix(Qseqs, k), kmer_set_matrix(Pseqs, k)
    qn = np.sqrt(np.asarray(Qm.sum(1)).ravel() + 1)
    pn = np.sqrt(np.asarray(Pm.sum(1)).ravel() + 1)
    PmT = Pm.T.tocsr()

    def gen():
        for s in range(0, Qm.shape[0], chunk):
            S = (Qm[s:s + chunk] @ PmT).toarray()
            yield s, S / qn[s:s + chunk, None] / pn[None, :]

    return summarize(rank_metrics(gen(), pos), pos, Pm.shape[0], subsets)
