"""Longitudinal (twin-cohort) data for the dynamics stage: arrays on the model vocabulary, family-level splits, pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import subdir
from .epitraj import invert_vocab


def load_longitudinal(vocab):
    z = np.load(subdir("lb") / "longitudinal.npz"); X = z["X"].astype(np.float32); sid = z["sample_ids"]
    meta = pd.read_csv(subdir("lb") / "longitudinal_meta.csv").set_index("sample_id").loc[sid].reset_index()
    v2u = invert_vocab(vocab.u2m, len(vocab))
    Xv = np.where(v2u[None, :] >= 0, X[:, np.where(v2u >= 0, v2u, 0)], np.nan).astype(np.float32)
    return Xv, meta


def family_split(meta: pd.DataFrame, seed: int = 0, test: float = 0.3, val: float = 0.15):
    fams = np.unique(meta.family_id.astype(str)); rng = np.random.RandomState(seed); rng.shuffle(fams)
    n_te, n_va = int(test * len(fams)), int(val * len(fams))
    te, va = set(fams[:n_te]), set(fams[n_te:n_te + n_va])
    f = meta.family_id.astype(str)
    return (np.flatnonzero(~f.isin(te | va)), np.flatnonzero(f.isin(va)), np.flatnonzero(f.isin(te)))


def make_pairs(meta: pd.DataFrame, rows, min_dt: float = 1.0):
    rows = set(np.asarray(rows).tolist()); age = meta.age.to_numpy(float); out = []
    for s, g in meta.reset_index().groupby("subject_id"):
        idx = [i for i in g.sort_values("age")["index"].tolist() if i in rows]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if age[idx[b]] - age[idx[a]] >= min_dt:
                    out.append((idx[a], idx[b], float(age[idx[b]] - age[idx[a]])))
    return out
