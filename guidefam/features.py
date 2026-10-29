"""Features for PM guides. Two tiers so the same model code runs on every dataset:

  core : guide + flanking nucleotides + TIGER's scalar descriptors (location, junction distance, structure, accessibility).
         Present in all three Cas13d datasets (HAP1 has only 2 nt of flank and no gene length).
  full : core + context statistics computed from the whole transcript (relative position, local GC). HEK293T and flow-cytometry only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NT = "ACGT"
_IDX = {c: i for i, c in enumerate(NT)}

SCALARS = ["loc_utr_5p", "loc_cds", "loc_utr_3p", "log_gene_len", "junction_dist_5p", "junction_dist_3p", "direct_repeat", "g_quad",
           "mfe", "hybrid_mfe_1_23", "hybrid_mfe_15_9", "hybrid_mfe_3_12", "log_unpaired", "log_unpaired_11", "log_unpaired_19", "log_unpaired_25"]


def onehot(seqs, L: int, align: str = "left") -> np.ndarray:
    """(n, L, 4) float32; unknown/padding = all zeros."""
    out = np.zeros((len(seqs), L, 4), np.float32)
    for i, s in enumerate(seqs):
        s = s.upper().replace("U", "T")
        if align == "right":
            s = s[-L:]
            off = L - len(s)
        else:
            s = s[:L]
            off = 0
        for j, c in enumerate(s):
            k = _IDX.get(c)
            if k is not None:
                out[i, off + j, k] = 1.0
    return out


def scalar_frame(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SCALARS if c in d.columns]
    x = d[cols].astype(float).copy()
    for c in ("junction_dist_5p", "junction_dist_3p"):
        if c in x:
            x[c] = np.log10(1 + x[c].abs())
    return x


def target_window(d: pd.DataFrame, flank: int) -> np.ndarray:
    """(n, 23 + 2*flank, 4) in TARGET orientation, as TIGER feeds it (5' flank | target | 3' flank)."""
    left = onehot(d["5p_context"].tolist(), flank, "right") if flank else np.zeros((len(d), 0, 4), np.float32)
    mid = onehot(d["target_seq"].tolist(), 23)
    right = onehot(d["3p_context"].tolist(), flank, "left") if flank else np.zeros((len(d), 0, 4), np.float32)
    return np.concatenate([left, mid, right], axis=1)


def _gc(s: str) -> float:
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def context_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Whole-transcript context statistics (only meaningful where the full transcript is given)."""
    p5, p3 = d["5p_context"].astype(str), d["3p_context"].astype(str)
    n5, n3 = p5.str.len().values, p3.str.len().values
    out = pd.DataFrame({"rel_pos": n5 / np.maximum(n5 + n3 + 23, 1), "log_len5": np.log10(1 + n5), "log_len3": np.log10(1 + n3)}, index=d.index)
    for w in (25, 50, 100, 200):
        out[f"gc_up{w}"] = [_gc(s[-w:]) for s in p5]
        out[f"gc_dn{w}"] = [_gc(s[:w]) for s in p3]
    return out


def kmer_counts(seqs, k: int = 2) -> np.ndarray:
    import itertools
    keys = ["".join(p) for p in itertools.product(NT, repeat=k)]
    ix = {s: i for i, s in enumerate(keys)}
    out = np.zeros((len(seqs), len(keys)), np.float32)
    for r, s in enumerate(seqs):
        s = s.upper().replace("U", "T")
        for j in range(len(s) - k + 1):
            i = ix.get(s[j:j + k])
            if i is not None:
                out[r, i] += 1
    return out / max(len(seqs[0]) - k + 1, 1) if len(seqs) else out


def tabular(d: pd.DataFrame, tier: str = "core", flank: int = 2) -> pd.DataFrame:
    """Flat feature table for gradient boosting / linear models."""
    g = onehot(d["guide53"].tolist(), 23).reshape(len(d), -1)
    cols = [f"g{p}_{n}" for p in range(23) for n in NT]
    X = pd.DataFrame(g, columns=cols, index=d.index)
    w = target_window(d, flank)
    fl = np.concatenate([w[:, :flank], w[:, 23 + flank:]], axis=1).reshape(len(d), -1)
    X = pd.concat([X, pd.DataFrame(fl, columns=[f"f{p}_{n}" for p in range(2 * flank) for n in NT], index=d.index)], axis=1)
    k1 = kmer_counts(d["guide53"].tolist(), 1)
    k2 = kmer_counts(d["guide53"].tolist(), 2)
    X = pd.concat([X, pd.DataFrame(k1, columns=[f"k1_{c}" for c in NT], index=d.index),
                   pd.DataFrame(k2, columns=[f"k2_{i}" for i in range(16)], index=d.index), scalar_frame(d)], axis=1)
    X["gc_guide"] = d["guide53"].map(_gc)
    if tier == "full":
        ctx = context_frame(d)
        short = (d["5p_context"].astype(str).str.len() + d["3p_context"].astype(str).str.len()).values < 100     # no transcript context (HAP1)
        ctx.loc[short, :] = np.nan
        X = pd.concat([X, ctx], axis=1)
    return X
