"""Pass 2: pull the panel rows out of every study into compact float16 arrays.

Benchmark studies are downloaded one at a time and the multi-GB raw file is deleted straight after
extraction, so the whole benchmark (~15 GB raw) never has to fit on disk.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import subdir
from .computage import fetch_dataset, load_meta
from .parquet_io import MethylParquet


def extract_study(path: Path, panel_ids: np.ndarray, sample_ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Return (sample ids actually present, float16 ``[n_samples, n_panel]`` with NaN for missing)."""
    mp = MethylParquet(path)
    pos = pd.Index(mp.cpg_ids).get_indexer(panel_ids)
    have = pos >= 0
    present = [s for s in sample_ids if s in set(mp.sample_ids)]
    out = np.full((len(present), len(panel_ids)), np.nan, dtype=np.float16)
    col = {s: i for i, s in enumerate(present)}
    for names, block in mp.iter_blocks(batch=64, columns=present):
        sub = block[pos[have]]
        sub = np.where((sub >= 0) & (sub <= 1), sub, np.nan)
        rows = [col[n] for n in names]
        tmp = np.full((len(names), len(panel_ids)), np.nan, dtype=np.float32)
        tmp[:, have] = sub.T
        out[rows] = tmp.astype(np.float16)
    return present, out


def _job(args) -> tuple[str, int, int]:
    split, gse, panel_ids, sample_ids, raw_dir, out_dir, delete_raw = args
    out = Path(out_dir) / f"{gse}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=False)
        return gse, len(z["samples"]), int(np.isfinite(z["X"].astype(np.float32)).sum())
    path = fetch_dataset(split, gse, raw_dir)
    present, X = extract_study(path, panel_ids, sample_ids)
    np.savez(out.with_suffix(".tmp.npz"), X=X, samples=np.array(present))
    os.replace(out.with_suffix(".tmp.npz"), out)
    if delete_raw:
        Path(path).unlink(missing_ok=True)
    return gse, len(present), int(np.isfinite(X.astype(np.float32)).sum())


def run_extract(split: str, panel: pd.DataFrame, workers: int = 3, delete_raw: bool = False) -> Path:
    meta = load_meta(split)
    out_dir = subdir("store", split)
    raw_dir = subdir(f"raw_{split}")
    panel_ids = np.asarray(panel.index.values, dtype=str)
    jobs = []
    for gse, grp in meta.groupby("DatasetID"):
        jobs.append((split, gse, panel_ids, list(grp.index), str(raw_dir), str(out_dir), delete_raw))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for gse, n, n_obs in ex.map(_job, jobs):
            print(f"[extract:{split}] {gse}: {n} samples", flush=True)
    return out_dir
