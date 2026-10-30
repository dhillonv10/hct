"""Mismatch tolerance: predict the LFC of a single-mismatch variant of a Cas13d guide.

A variant row has the same target site as its perfect-match (PM) parent; its `guide_seq` (TIGER's convention: complement of the target in target orientation) differs
from the perfect complement at one position. Features = the variant guide's own sequence features and scalar descriptors, the mismatch position and base change, and the
parent site's predicted activity from the on-target model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F

_COMP = str.maketrans("ACGT", "TGCA")
PURINE = set("AG")


def mismatch_info(d: pd.DataFrame) -> pd.DataFrame:
    """pos (0-based, target orientation), ref base (the perfect guide's base) and alt base for single-mismatch rows."""
    pos, ref, alt = [], [], []
    for t, g in zip(d.target_seq, d.guide_seq):
        perfect = t.upper().translate(_COMP)
        diff = [i for i, (a, b) in enumerate(zip(perfect, g)) if a != b]
        pos.append(diff[0] if len(diff) == 1 else -1)
        ref.append(perfect[diff[0]] if len(diff) == 1 else "N")
        alt.append(g[diff[0]] if len(diff) == 1 else "N")
    return pd.DataFrame({"mm_pos": pos, "mm_ref": ref, "mm_alt": alt}, index=d.index)


def mm_features(d: pd.DataFrame, pm_pred: np.ndarray | None = None) -> pd.DataFrame:
    """Feature table for single-mismatch variants. `d` needs target_seq, guide_seq, 5p_context, 3p_context and the scalar descriptors."""
    d = d.copy()
    d["guide53"] = d.guide_seq.map(lambda s: s[::-1])                       # the variant guide written 5'->3'
    X = F.tabular(d, "core", flank=2)
    mi = mismatch_info(d)
    X = pd.concat([X, mi[["mm_pos"]]], axis=1)
    for b in F.NT:
        X[f"ref_{b}"] = (mi.mm_ref == b).astype(float)
        X[f"alt_{b}"] = (mi.mm_alt == b).astype(float)
    X["transition"] = [(r in PURINE) == (a in PURINE) for r, a in zip(mi.mm_ref, mi.mm_alt)]
    X["transition"] = X["transition"].astype(float)
    for c in (4, 5, 6, 7):
        X[f"dist_c{c}"] = (mi.mm_pos - c).abs()
    X["seedish"] = mi.mm_pos.between(3, 8).astype(float)
    X["dist_5p"], X["dist_3p"] = mi.mm_pos, 22 - mi.mm_pos
    if pm_pred is not None:
        X["pm_pred"] = np.asarray(pm_pred, float)
    return X
