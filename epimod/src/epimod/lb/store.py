"""LongevityBench-format training store and per-task input generators.

Everything is generated from real measured profiles: extremes are the true lowest/highest-100 CpGs
of a sample, pairs are the 160 CpGs that differ most between two real samples, fixed sets are
columns of the real profile. No values are simulated.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.annotation import load_probe_annotation
from ..data.computage import load_meta
from ..paths import subdir

MISSING = 255
N_EXT = 100


def lb_series(parsed: dict | None = None) -> set[str]:
    """Every GEO series that appears in any LongevityBench DNAm eval prompt."""
    parsed = parsed or pickle.load(open(subdir("lb") / "parsed.pkl", "rb"))
    s: set[str] = set()
    for t in parsed.values():
        for field in (t.gse, getattr(t, "gse_b", [])):
            s |= {g for g in field if g}
    return s


@dataclass
class LBStore:
    meta: pd.DataFrame            # one row per sample (SampleID index): DatasetID, Age, Gender, Condition, Platform, CellType ...
    universe: pd.Index            # CpG ids of the uint8 profile columns
    vocab: pd.Index               # CpG ids of the extremes indices (450K annotation order)
    low_idx: np.ndarray           # [S,100] vocabulary indices, ascending beta
    low_beta: np.ndarray          # [S,100] float16
    high_idx: np.ndarray          # [S,100] vocabulary indices, descending beta
    high_beta: np.ndarray
    prof: np.ndarray              # [S, |U|] uint8, beta*100, 255 = missing

    @property
    def n(self) -> int:
        return len(self.meta)

    def v2u(self) -> np.ndarray:
        """Map vocabulary index -> universe column (-1 if absent)."""
        return self.universe.get_indexer(self.vocab)

    def columns(self, cpg_ids) -> np.ndarray:
        """float32 [S,k] beta with NaN for missing, for the given CpG ids (must be in the universe)."""
        cols = self.universe.get_indexer(list(cpg_ids))
        assert (cols >= 0).all(), "CpG outside universe"
        x = self.prof[:, cols].astype(np.float32)
        x[x == MISSING] = np.nan
        return x / 100.0

    def subset(self, mask: np.ndarray) -> "LBStore":
        return LBStore(self.meta[mask].copy(), self.universe, self.vocab, self.low_idx[mask], self.low_beta[mask],
                       self.high_idx[mask], self.high_beta[mask], self.prof[mask])


def load_lb_store(exclude_series: set[str] | None = None, min_age: float = 0.0, max_age: float = 105.0) -> LBStore:
    exclude_series = set(exclude_series or ())
    root = subdir("lb", "store")
    universe = pd.Index(pd.read_parquet(subdir("lb") / "universe.parquet")["cpg"].values)
    vocab = pd.Index(load_probe_annotation().index)
    metas = []
    for split in ("train", "benchmark"):
        m = load_meta(split).copy(); m["split"] = split; metas.append(m)
    meta = pd.concat(metas)
    meta = meta[~meta.index.duplicated(keep="first")]
    meta = meta[meta.Age.between(min_age, max_age) & ~meta.DatasetID.isin(exclude_series)]
    counts = meta.groupby("DatasetID").size()
    S, U = len(meta), len(universe)
    low_idx = np.zeros((S, N_EXT), np.int32); high_idx = np.zeros((S, N_EXT), np.int32)
    low_beta = np.zeros((S, N_EXT), np.float16); high_beta = np.zeros((S, N_EXT), np.float16)
    prof = np.empty((S, U), np.uint8)
    ids: list[str] = []
    pos = 0
    for split in ("train", "benchmark"):
        for f in sorted((root / split).glob("GSE*.npz")):
            if f.stem not in counts.index:
                continue
            z = np.load(f, allow_pickle=False)
            samples = z["samples"].tolist()
            # the de-duplicated metadata assigns each sample to exactly one study
            keep = [i for i, s in enumerate(samples) if s in meta.index and meta.at[s, "DatasetID"] == f.stem]
            if not keep:
                continue
            n = len(keep)
            low_idx[pos:pos + n], high_idx[pos:pos + n] = z["low_idx"][keep], z["high_idx"][keep]
            low_beta[pos:pos + n], high_beta[pos:pos + n] = z["low_beta"][keep], z["high_beta"][keep]
            prof[pos:pos + n] = z["prof"][keep]
            ids.extend(samples[i] for i in keep); pos += n
    assert pos <= S
    meta = meta.loc[ids]
    return LBStore(meta, universe, vocab, low_idx[:pos], low_beta[:pos], high_idx[:pos], high_beta[:pos], prof[:pos])


# ----------------------------------------------------------------------------- task generators
def extremes_tokens(store: LBStore, rows: np.ndarray, v2u: np.ndarray):
    """(cpg universe index [B,200] with -1 pad, beta [B,200], role [B,200]: 0 = lowest list, 1 = highest)."""
    li, hi = store.low_idx[rows], store.high_idx[rows]
    cpg = np.concatenate([v2u[li], v2u[hi]], axis=1)
    beta = np.concatenate([store.low_beta[rows], store.high_beta[rows]], axis=1).astype(np.float32)
    role = np.concatenate([np.zeros_like(li), np.ones_like(hi)], axis=1).astype(np.int64)
    return cpg, beta, role


def pair_tokens(store: LBStore, ia: np.ndarray, ib: np.ndarray, k: int = 160):
    """The ``k`` CpGs with the largest |beta_A - beta_B| among those observed in both donors."""
    a = store.prof[ia].astype(np.int16); b = store.prof[ib].astype(np.int16)
    bad = (a == MISSING) | (b == MISSING)
    diff = np.where(bad, -1, np.abs(a - b))
    top = np.argpartition(-diff, k, axis=1)[:, :k]
    rows = np.arange(len(ia))[:, None]
    order = np.argsort(-diff[rows, top], axis=1)
    top = top[rows, order]
    return top.astype(np.int64), a[rows, top].astype(np.float32) / 100.0, b[rows, top].astype(np.float32) / 100.0


def sample_pairs(ages: np.ndarray, n: int, rng: np.random.RandomState, min_gap: float = 10.0, series: np.ndarray | None = None,
                 cross_series_frac: float = 0.94):
    """Random donor pairs with |age gap| >= ``min_gap`` (as in the benchmark); mostly cross-study."""
    S = len(ages)
    ia, ib = [], []
    while len(ia) < n:
        a = rng.randint(0, S, size=2 * n); b = rng.randint(0, S, size=2 * n)
        ok = np.abs(ages[a] - ages[b]) >= min_gap
        if series is not None:
            same = series[a] == series[b]
            want_cross = rng.rand(2 * n) < cross_series_frac
            ok &= ~(same & want_cross)
        ia.extend(a[ok]); ib.extend(b[ok])
    return np.array(ia[:n]), np.array(ib[:n])
