"""k-mer count features: the baseline every foundation model has to beat."""
from __future__ import annotations

import numpy as np

_LUT = np.full(256, -1, dtype=np.int8)
for i, c in enumerate("ACGU"):
    _LUT[ord(c)] = i
_LUT[ord("T")] = 3


def encode_ints(seq: str) -> np.ndarray:
    """ACGU(T) -> 0..3, anything else -> -1."""
    return _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def kmer_counts(seq: str, k: int) -> np.ndarray:
    a = encode_ints(seq).astype(np.int64)
    n = len(a) - k + 1
    out = np.zeros(4 ** k, dtype=np.float32)
    if n <= 0:
        return out
    idx = np.zeros(n, dtype=np.int64)
    ok = np.ones(n, dtype=bool)
    for j in range(k):
        s = a[j:j + n]
        ok &= s >= 0
        idx = idx * 4 + np.where(s >= 0, s, 0)
    np.add.at(out, idx[ok], 1.0)
    return out


def kmer_features(seqs, ks=(1, 2, 3, 4, 5), log_len: bool = True) -> np.ndarray:
    """Frequency-normalised k-mer spectrum (per k) + log length. Returns (n, d) float32."""
    feats = []
    for s in seqs:
        parts = []
        for k in ks:
            c = kmer_counts(s, k)
            tot = c.sum()
            parts.append(c / tot if tot > 0 else c)
        if log_len:
            parts.append(np.array([np.log10(max(len(s), 1))], dtype=np.float32))
        feats.append(np.concatenate(parts))
    return np.stack(feats).astype(np.float32)


def gc_content(seq: str) -> float:
    a = encode_ints(seq)
    a = a[a >= 0]
    return float(((a == 1) | (a == 2)).mean()) if len(a) else 0.0
