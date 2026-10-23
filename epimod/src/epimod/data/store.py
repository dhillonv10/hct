"""In-memory methylation store: panel-restricted beta matrix + aligned metadata + CpG context."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..age import age_to_t
from ..paths import subdir
from .annotation import GENE_GROUPS, load_probe_annotation
from .computage import load_meta


@dataclass
class CpGTable:
    """Per-CpG tensors the model needs (all aligned with the panel order)."""

    ids: np.ndarray        # [N] str
    chrom: np.ndarray      # [N] int
    pos: np.ndarray        # [N] int64 (hg19)
    island: np.ndarray     # [N] int
    reg: np.ndarray        # [N] int
    binary: np.ndarray     # [N, 9] float32: 6 gene-region flags, enhancer, DHS, probe design II
    mean: np.ndarray       # [N] training mean beta
    std: np.ndarray        # [N] training std of beta
    source: np.ndarray     # [N] str: clock | age_assoc | variance
    z_age: np.ndarray      # [N] meta-analytic within-study age association (training studies)

    def head(self, n: int) -> "CpGTable":
        return CpGTable(**{k: v[:n] for k, v in self.__dict__.items()})

    def subset(self, idx: np.ndarray) -> "CpGTable":
        return CpGTable(**{k: v[idx] for k, v in self.__dict__.items()})


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(subdir("store") / "panel.parquet")


def cpg_table(panel: pd.DataFrame, annotation: pd.DataFrame | None = None) -> CpGTable:
    ann = annotation if annotation is not None else load_probe_annotation()
    a = ann.loc[panel.index]
    binary = np.concatenate(
        [a[[f"gene_{g}" for g in GENE_GROUPS]].to_numpy(), a[["enhancer", "dhs", "design2"]].to_numpy()], axis=1
    ).astype(np.float32)
    def arr(x, dtype=None):  # writable copies (pandas >= 3 hands out read-only views)
        return np.array(x, dtype=dtype, copy=True)

    return CpGTable(
        ids=arr(panel.index.values, str),
        chrom=arr(a["chrom"]), pos=arr(a["pos"]), island=arr(a["island"]), reg=arr(a["reg"]),
        binary=binary, mean=arr(panel["mean"], np.float32), std=arr(panel["std"], np.float32),
        source=arr(panel["source"], str), z_age=arr(panel["z_meta"], np.float32),
    )


@dataclass
class MethylData:
    X: np.ndarray            # [S, N] float16 beta, NaN = missing
    meta: pd.DataFrame       # aligned rows
    cpgs: CpGTable

    @property
    def n(self) -> int:
        return len(self.meta)

    def tensors(self):
        """Return (age_t, has_age) in transformed age space."""
        age = self.meta["Age"].to_numpy(np.float64)
        has = np.isfinite(age)
        return np.where(has, age_to_t(np.where(has, age, 20.0)), 0.0).astype(np.float32), has

    def select(self, mask: np.ndarray) -> "MethylData":
        return MethylData(self.X[mask], self.meta.loc[mask].copy(), self.cpgs)


def load_split(split: str, n_cpgs: int | None = None, store_dir: Path | None = None) -> MethylData:
    """Concatenate all per-study arrays of one split. ``n_cpgs`` keeps the first N CpGs by priority."""
    panel = load_panel()
    n = len(panel) if n_cpgs is None else min(n_cpgs, len(panel))
    meta = load_meta(split)
    store_dir = Path(store_dir) if store_dir else subdir("store", split)
    xs, ids = [], []
    for f in sorted(store_dir.glob("GSE*.npz")):
        z = np.load(f, allow_pickle=False)
        samples = z["samples"].tolist()
        # keep a sample only if the (QC-filtered, de-duplicated) metadata assigns it to this study
        keep = [i for i, s in enumerate(samples) if s in meta.index and meta.at[s, "DatasetID"] == f.stem]
        xs.append(z["X"][keep, :n])
        ids.extend(samples[i] for i in keep)
    X = np.concatenate(xs, axis=0)
    meta = meta.loc[ids].copy()
    assert len(meta) == len(X) and meta.index.is_unique
    return MethylData(X=X, meta=meta, cpgs=cpg_table(panel.iloc[:n]))
