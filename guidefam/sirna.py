"""siRNA efficacy datasets (OligoFormer's curated tables: Huesken 'Hu', 'Mix' of small published sets, Takayuki 'Taka') and features.

Each row = one 19-nt siRNA (guide strand, RNA) with the 57-nt mRNA window (19 nt flank | 19 nt site | 19 nt flank) around its target ('X' = padding past the transcript end) and a
normalised efficacy in [0, 1]. Rows are ordered along each mRNA, so consecutive overlapping windows belong to one target; `target_groups`
recovers those groups, which is what lets us split by TARGET gene (OligoFormer's own intra-dataset CV is a shuffled split over siRNAs, so
siRNAs against the same mRNA are in both train and test).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import features as F

TD_NAMES = [f"td{i}" for i in range(24)]


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["ybin_file"] = d["y"].astype(int)             # OligoFormer's own binary label (used by their intra-dataset path, train.py)
    d["sirna"] = d.siRNA.str.upper().str.replace("T", "U")
    d["win"] = d.mRNA.str.upper().str.replace("T", "U")
    d["y"] = d["label"].astype(float)
    td = d["td"].str.split(",", expand=True).astype(float)
    td.columns = TD_NAMES[: td.shape[1]]
    return pd.concat([d.reset_index(drop=True), td.reset_index(drop=True)], axis=1)


def target_groups(win: pd.Series, k: int = 20) -> np.ndarray:
    """Group id per row: a new group starts when a row's window shares no k-mer with the previous row's window."""
    gid, cur, prev = [], 0, None
    for w in win:
        s = w.replace("X", "")
        kmers = {s[i:i + k] for i in range(len(s) - k + 1)}
        if prev is not None and not (kmers & prev):
            cur += 1
        gid.append(cur)
        prev = kmers
    return np.array(gid)


def rc_dna(s: str) -> str:
    return s.upper().replace("U", "T").translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _gc(s):
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def reynolds_score(s: str) -> float:
    """Reynolds et al. 2004 (Nat Biotechnol) design criteria, 1 point each, evaluated on the sense strand (the reverse complement of the guide).
    Seven of the eight criteria: the 'no internal repeat (Tm < 20 C)' criterion needs a thermodynamic calculation and is left out."""
    g = s.upper().replace("T", "U")
    sense = rc_dna(g).replace("T", "U")                      # sense strand 5'->3', positions 1..19 = index 0..18
    sc = 0
    sc += 0.30 <= _gc(sense) <= 0.52                          # 30-52 % GC
    sc += sum(c in "AU" for c in sense[14:19]) >= 3           # >= 3 A/U at positions 15-19
    sc += sense[18] == "A"                                    # A at 19
    sc += sense[2] == "A"                                     # A at 3
    sc += sense[9] == "U"                                     # U at 10
    sc += sense[18] not in "GC"                               # not G/C at 19
    sc += sense[12] != "G"                                    # not G at 13
    return float(sc)


def uitei_ok(s: str) -> float:
    """Ui-Tei et al. 2004 rules: A/U at antisense 5' end (pos 1), G/C at 3' end (pos 19), >=4 A/U in the 5' 7 nt, no run of >9 G/C."""
    s = s.upper().replace("T", "U")
    c = 0
    c += s[0] in "AU"
    c += s[18] in "GC"
    c += sum(1 for x in s[:7] if x in "AU") >= 4
    run = max(len(r) for r in s.replace("A", " ").replace("U", " ").split()) if any(x in "GC" for x in s) else 0
    c += run <= 9
    return float(c)


def tabular(d: pd.DataFrame, flank: int = 20) -> pd.DataFrame:
    g = F.onehot(d["sirna"].tolist(), 19).reshape(len(d), -1)
    X = pd.DataFrame(g, columns=[f"s{p}_{n}" for p in range(19) for n in F.NT], index=d.index)
    X = pd.concat([X, d[[c for c in TD_NAMES if c in d]]], axis=1)
    X["gc"] = d["sirna"].map(_gc)
    X["gc_seed"] = d["sirna"].map(lambda s: _gc(s[1:8]))
    X["au_5p"] = d["sirna"].map(lambda s: sum(c in "AU" for c in s[:7]))
    X["reynolds"] = d["sirna"].map(reynolds_score)
    X["uitei"] = d["sirna"].map(uitei_ok)
    k1 = F.kmer_counts(d["sirna"].tolist(), 1)
    k2 = F.kmer_counts(d["sirna"].tolist(), 2)
    X = pd.concat([X, pd.DataFrame(k1, columns=[f"k1_{c}" for c in F.NT], index=d.index), pd.DataFrame(k2, columns=[f"k2_{i}" for i in range(16)], index=d.index)], axis=1)
    # flank of the mRNA window around the site (the site is always at 19..37 of the 57-nt window; flank bases are added in `flank_frame`)
    X["win_gc"] = d["win"].map(lambda w: _gc(w.replace("X", "")))
    return X


def accessibility(d: pd.DataFrame) -> pd.DataFrame:
    """Local unpaired probabilities of the target site inside OligoFormer's 57-nt window (19 nt flank | 19 nt site | 19 nt flank; 'X' = padding
    past the transcript end). Only the window is available for these tables, so this is a local-structure proxy, not full-mRNA accessibility."""
    from .structure import _unpaired
    rows = []
    for w in d["win"]:
        nl = len(w) - len(w.lstrip("X"))
        s = w.strip("X").replace("T", "U")
        lo = 19 - nl                                  # index of the site's first nt in the stripped window
        u = _unpaired(s)
        site = u[lo:lo + 19]
        lg = np.log(np.clip(u, 1e-6, 1.0))
        rows.append({"acc_mean": float(site.mean()), "acc_min": float(site.min()), "acc_5p8": float(site[:8].mean()), "acc_3p8": float(site[-8:].mean()),
                     "acc_seed": float(site[1:8].mean()), "acc_up": float(u[max(lo - 10, 0):lo].mean()) if lo > 0 else np.nan,
                     "acc_dn": float(u[lo + 19:lo + 29].mean()) if lo + 19 < len(u) else np.nan, "acc_logjoint": float(lg[lo:lo + 19].sum()),
                     **{f"pu_{i}": float(site[i]) for i in range(19)}})
    return pd.DataFrame(rows, index=d.index)


def flank_frame(d: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """One-hot of the n nt on each side of the target site (mRNA sense orientation; padding stays all-zero) and the flanks' GC."""
    up = F.onehot([w[19 - n:19].replace("X", "N") for w in d["win"]], n, "right").reshape(len(d), -1)
    dn = F.onehot([w[38:38 + n].replace("X", "N") for w in d["win"]], n, "left").reshape(len(d), -1)
    cols = [f"up{p}_{c}" for p in range(n) for c in F.NT] + [f"dn{p}_{c}" for p in range(n) for c in F.NT]
    out = pd.DataFrame(np.hstack([up, dn]), index=d.index, columns=cols)
    out["gc_up"] = [_gc(w[:19].replace("X", "")) for w in d["win"]]
    out["gc_dn"] = [_gc(w[38:].replace("X", "")) for w in d["win"]]
    return out


def binary(d: pd.DataFrame, protocol: str) -> np.ndarray:
    """The binary label each OligoFormer evaluation path uses.
    'inter' (test.py, train on one dataset / test on another): normalised efficacy > 0.7.
    'intra' (train.py, k-fold within a dataset): the file's own `y` column (for Huesken: unnormalised efficacy >= 0.7, ~50 % positives)."""
    return (d["label"] > 0.7).astype(int).values if protocol == "inter" else d["ybin_file"].values
