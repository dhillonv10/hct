"""Compact training store: only the CpGs that can ever appear in a LongevityBench DNAm prompt.

Models read a few thousand CpGs (candidate pools, the fixed panel, clock sites). Keeping 200k columns per sample is
therefore 20x more than needed, so every study is stored with columns restricted to the union of CpG *identities*
(no values, no labels) over all benchmark prompts, intersected with the original universe. This makes it practical
to train on a much larger corpus. ``prof`` is uint8 (beta*100, 255 = missing), exactly as before."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import subdir


def lite_ids(parsed: dict, universe: pd.Index) -> pd.Index:
    """Union of CpG identities over every DNAm prompt (all tasks, both halves), restricted to ``universe``."""
    seen: set[str] = set()
    for t in parsed.values():
        for ids in t.ids:
            seen.update(map(str, ids))
    return pd.Index(sorted(seen & set(universe)))


def lite_universe_path():
    return subdir("lb") / "lite_universe.parquet"


def load_lite_universe() -> pd.Index:
    return pd.Index(pd.read_parquet(lite_universe_path())["cpg"].values)


def convert_store(parsed: dict) -> None:
    """Write ``lb/store_lite/<split>/<GSE>.npz`` from the 200k-column store, and the lite universe."""
    universe = pd.Index(pd.read_parquet(subdir("lb") / "universe.parquet")["cpg"].values)
    ids = lite_ids(parsed, universe)
    pd.DataFrame({"cpg": ids}).to_parquet(lite_universe_path())
    cols = universe.get_indexer(ids)
    print(f"lite universe: {len(ids)} CpGs (of {len(universe)})", flush=True)
    for split in ("train", "benchmark"):
        out = subdir("lb", "store_lite", split)
        for f in sorted((subdir("lb", "store") / split).glob("GSE*.npz")):
            if (out / f.name).exists():
                continue
            z = np.load(f, allow_pickle=False)
            np.savez(out / f.name, samples=z["samples"], prof=z["prof"][:, cols])
    print("converted", flush=True)


# ---------------------------------------------------------------------------------------------- direct extraction
def _lite_job(args):
    split, gse, sample_ids, lite_ids_, raw_dir, out_dir = args
    import os
    from pathlib import Path

    from ..data.computage import fetch_dataset
    from ..data.parquet_io import MethylParquet
    out = Path(out_dir) / f"{gse}.npz"
    if out.exists():
        return gse, 0
    path = fetch_dataset(split, gse, raw_dir)
    mp = MethylParquet(path)
    rows = pd.Index(mp.cpg_ids).get_indexer(pd.Index(lite_ids_))              # lite CpG -> file row (-1 = not on this array)
    have = np.flatnonzero(rows >= 0)
    present = [s for s in sample_ids if s in set(mp.sample_ids)]
    prof = np.full((len(present), len(lite_ids_)), 255, np.uint8)
    col = {s: i for i, s in enumerate(present)}
    for names, block in mp.iter_blocks(batch=48, columns=present):
        full = block[rows[have]]                                                # [n_have, batch]
        v = np.where((full >= 0) & (full <= 1), np.round(full * 100.0), 255.0)
        v[~np.isfinite(full)] = 255.0
        for j, s in enumerate(names):
            prof[col[s], have] = v[:, j].astype(np.uint8)
    np.savez(out.with_suffix(".tmp.npz"), samples=np.array(present), prof=prof)
    os.replace(out.with_suffix(".tmp.npz"), out)
    Path(path).unlink(missing_ok=True)
    return gse, len(present)


def extract_lite(parsed: dict, workers: int = 3, only: list[str] | None = None) -> None:
    """Stream every ComputAge study once and keep only the compact CpG panel (no 200k-CpG intermediate store)."""
    from concurrent.futures import ProcessPoolExecutor

    from ..data.annotation import load_probe_annotation
    from ..data.computage import load_meta
    ids = lite_ids(parsed, pd.Index(load_probe_annotation().index))
    pd.DataFrame({"cpg": ids}).to_parquet(lite_universe_path())
    jobs = []
    for split in ("train", "benchmark"):
        meta = load_meta(split)
        for gse, grp in meta.groupby("DatasetID"):
            if only is None or gse in only:
                jobs.append((split, gse, list(grp.index), list(ids), str(subdir(f"raw_{split}")), str(subdir("lb", "store_lite", split))))
    jobs.sort(key=lambda j: -len(j[2]))
    print(f"[lite-extract] {len(ids)} CpGs; {len(jobs)} studies", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for gse, n in ex.map(_lite_job, jobs):
            print(f"[lite-extract] {gse}: {n} samples", flush=True)
