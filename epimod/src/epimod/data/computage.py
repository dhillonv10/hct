"""ComputAge Bench (Kriukov et al.) access: 46 healthy-control training studies and 65 benchmark
studies of human blood/saliva/buccal Illumina-array methylation, harmonised to beta values.

https://huggingface.co/datasets/computage/computage_bench  (CC BY-SA 4.0)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import subdir
from .download import download

HF_BASE = "https://huggingface.co/datasets/computage/computage_bench/resolve/main"
SPLITS = ("train", "benchmark")
_META_FILE = {"train": "computage_train_meta.tsv", "benchmark": "computage_bench_meta.tsv"}
_DATA_PREFIX = {"train": "computage_train_data_", "benchmark": "computage_bench_data_"}
PLATFORM_NAMES = {
    "GPL8490": "27K",
    "GPL13534": "450K",
    "GPL16304": "450K",
    "GPL21145": "EPIC",
    "GPL23976": "EPIC",
    "GPL29753": "EPICv2",
}


AGE_RANGE = (0.0, 105.0)  # labels outside this are unit errors (e.g. GSE80417 has a donor aged "891")


def load_meta(split: str, root: Path | None = None, qc: bool = True) -> pd.DataFrame:
    """Sample metadata indexed by GEO sample id (columns: DatasetID, PlatformID, Tissue, CellType,
    Gender, Age, Condition, Class). With ``qc`` samples with a missing or implausible age are dropped."""
    assert split in SPLITS, split
    meta_dir = (root or subdir("meta"))
    path = download(f"{HF_BASE}/{_META_FILE[split]}", Path(meta_dir) / _META_FILE[split])
    meta = pd.read_csv(path, sep="\t", index_col=0)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Platform"] = meta["PlatformID"].map(PLATFORM_NAMES).fillna("other")
    if qc:
        ok = meta["Age"].between(*AGE_RANGE)
        if (~ok).any():
            print(f"[qc] {split}: dropping {int((~ok).sum())} sample(s) with missing/implausible age")
        meta = meta[ok]
        dup = meta.index.duplicated(keep="first")      # e.g. 6 GSMs listed under both GSE118468 and GSE118469
        if dup.any():
            print(f"[qc] {split}: dropping {int(dup.sum())} duplicate registration(s) of the same GEO sample")
            meta = meta[~dup]
    return meta


def sibling_studies(max_gap: int = 3) -> list[str]:
    """Benchmark studies whose GEO series number is within ``max_gap`` of a training study.

    Neighbouring GSE numbers usually mean sub-series of one project (same lab, array batch and
    cohort), e.g. the Parkinson's series GSE72774/GSE72776 next to training series GSE72773/GSE72777.
    Individuals do not overlap (no shared GSM ids), but a learned model may benefit from the shared
    batch, so results are also reported with these studies removed."""
    num = lambda g: int("".join(c for c in g if c.isdigit()))
    tr = [num(g) for g in load_meta("train").DatasetID.unique()]
    return sorted({g for g in load_meta("benchmark").DatasetID.unique() if any(abs(num(g) - t) <= max_gap for t in tr)})


def dataset_url(split: str, gse: str) -> str:
    return f"{HF_BASE}/data/{split}/{_DATA_PREFIX[split]}{gse}.parquet"


def fetch_dataset(split: str, gse: str, raw_dir: Path | None = None) -> Path:
    """Download one study's parquet (samples are columns, CpGs are rows) and return its path."""
    raw_dir = Path(raw_dir) if raw_dir else subdir(f"raw_{split}")
    return download(dataset_url(split, gse), raw_dir / f"{gse}.parquet")
