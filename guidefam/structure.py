"""Position-level target accessibility from the transcript, RNAplfold-style, with ViennaRNA.

For a transcript we fold overlapping windows (size W, step S), take each position's unpaired probability 1 - sum_j P(i,j) from the partition
function of each window, and average over the windows that cover the position (centre-weighted, which damps window-edge artefacts).
Three scales (W = 60 / 150 / 300 nt) give a local, a medium and a long-range view. Features for a guide are then summaries of the
per-nucleotide profile over its 23-nt target and the flanks, and the (approximate) probability that a k-nt stretch is jointly unpaired.

Only transcripts whose full sequence is known can be profiled (HEK293T and flow-cytometry rows carry the whole transcript as context;
HAP1 rows do not).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SCALES = ((60, 10), (150, 15), (300, 30))


def transcripts(d: pd.DataFrame) -> dict:
    """gene -> full transcript sequence (RNA alphabet kept as DNA letters), reconstructed from 5' context + target + 3' context."""
    out = {}
    for g, x in d.groupby("gene"):
        seqs = (x["5p_context"] + x["target_seq"] + x["3p_context"]).str.upper().str.replace("U", "T")
        top = seqs.value_counts()
        if len(top) > 1 and top.iloc[1] > 0:
            # contexts of differing length would mean inconsistent coordinates; keep the most common reconstruction
            pass
        out[g] = top.index[0]
    return out


def _unpaired(sub: str) -> np.ndarray:
    import RNA
    fc = RNA.fold_compound(sub)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    M = np.array(fc.bpp(), dtype=float)[1:, 1:]
    return 1.0 - (M.sum(axis=0) + M.sum(axis=1))


def profile(seq: str, W: int, S: int) -> np.ndarray:
    n = len(seq)
    num, den = np.zeros(n), np.zeros(n)
    starts = list(range(0, max(n - W, 0) + 1, S))
    if starts[-1] + W < n:
        starts.append(n - W)
    w_c = np.minimum(np.arange(W) + 1, W - np.arange(W))          # triangular, peaks mid-window
    w_c = w_c / w_c.max()
    for s0 in starts:
        sub = seq[s0:s0 + W].replace("T", "U")
        u = _unpaired(sub)
        m = len(sub)
        num[s0:s0 + m] += u * w_c[:m]
        den[s0:s0 + m] += w_c[:m]
    return np.clip(num / np.maximum(den, 1e-9), 1e-6, 1.0)


def transcript_profiles(seq: str, cache: str | None = None) -> dict:
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return {int(k[1:]): z[k] for k in z.files}
    res = {W: profile(seq, W, S) for W, S in SCALES}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{f"w{W}": v for W, v in res.items()})
    return res


def guide_features(d: pd.DataFrame, cache_dir: str | None = None) -> pd.DataFrame:
    """Accessibility summaries for every row of `d` (needs the full-transcript context columns)."""
    tx = transcripts(d)
    prof = {g: transcript_profiles(s, os.path.join(cache_dir, f"{g}.npz") if cache_dir else None) for g, s in tx.items()}
    cols = {}
    rows = []
    for g, s0 in zip(d.gene, d.start.astype(int) - 1):          # 0-based index of the first target nt
        r = {}
        for W, p in prof[g].items():
            n = len(p)
            lo, hi = s0, s0 + 23
            t = p[lo:hi]
            lp = np.log(p)
            r[f"acc{W}_mean"] = float(t.mean())
            r[f"acc{W}_min"] = float(t.min())
            r[f"acc{W}_logjoint23"] = float(lp[lo:hi].sum())
            r[f"acc{W}_logjoint11"] = float(lp[lo + 6:lo + 17].sum())
            r[f"acc{W}_logjoint5c"] = float(lp[lo + 9:lo + 14].sum())
            r[f"acc{W}_5p8"] = float(t[:8].mean())
            r[f"acc{W}_3p8"] = float(t[-8:].mean())
            up = p[max(lo - 10, 0):lo]
            dn = p[hi:min(hi + 10, n)]
            r[f"acc{W}_up10"] = float(up.mean()) if len(up) else np.nan
            r[f"acc{W}_dn10"] = float(dn.mean()) if len(dn) else np.nan
            if W == 60:
                for i in range(23):
                    r[f"pu_{i}"] = float(t[i])
        rows.append(r)
    return pd.DataFrame(rows, index=d.index)
