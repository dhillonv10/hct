"""Second extraction pass producing LongevityBench-format training inputs from raw profiles.

For every ComputAge sample (raw parquet streamed, then deleted) store

* the **100 lowest and 100 highest methylated CpGs over the whole array** (probe index + exact beta),
  which is exactly how the benchmark's single-profile prompts (LB-0010 / LB-0002) are built, and
* a **uint8 profile (beta x 100, 255 = missing) over a 200k-CpG universe** containing every CpG shown
  in any LongevityBench DNAm prompt plus the age panel, from which fixed-set (LB-0190), clock-replica
  (LB-0195) and pairwise (LB-0006) inputs are generated on the fly. 0.01 is the precision of the
  benchmark's own displayed values.
"""
from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.annotation import load_probe_annotation
from ..data.computage import fetch_dataset, load_meta
from ..data.parquet_io import MethylParquet
from ..paths import subdir

N_EXT = 100
UNIVERSE_SIZE = 200_000


def lb_dir() -> Path:
    return subdir("lb")


def build_universe(parsed_path: Path | None = None, size: int = UNIVERSE_SIZE) -> pd.Index:
    """LB-shown CpGs first, then the age panel, then the highest within-study-variance CpGs."""
    ann = load_probe_annotation()
    vocab = pd.Index(ann.index)
    shown: dict[str, None] = {}
    tasks = pickle.load(open(parsed_path or lb_dir() / "parsed.pkl", "rb"))
    for t in tasks.values():
        for ids in t.ids:
            for i in ids:
                shown.setdefault(i, None)
    panel = pd.read_parquet(subdir("store") / "panel.parquet")
    stats = dict(np.load(subdir("store") / "panel_stats.npz"))
    wv = pd.Series(stats["within_var"], index=vocab).fillna(-1).sort_values(ascending=False)
    order: dict[str, None] = {}
    for src in (shown, dict.fromkeys(panel.index), dict.fromkeys(wv.index)):
        for i in src:
            if i in vocab and len(order) < size:
                order.setdefault(i, None)
    return pd.Index(list(order))


def _extract_sample_block(block: np.ndarray, rows_v: np.ndarray, rows_u: np.ndarray, u_pos: np.ndarray, n_u: int):
    """block [R, nb] raw betas -> extremes + uint8 profile for each column."""
    nb = block.shape[1]
    sub = block[rows_v]                                   # [Rv, nb]  (rows in the annotation vocabulary)
    lo_i = np.zeros((nb, N_EXT), np.int32); hi_i = np.zeros((nb, N_EXT), np.int32)
    lo_b = np.full((nb, N_EXT), np.nan, np.float16); hi_b = np.full((nb, N_EXT), np.nan, np.float16)
    prof = np.full((nb, n_u), 255, np.uint8)
    for j in range(nb):
        col = sub[:, j]
        ok = np.isfinite(col) & (col >= 0) & (col <= 1)
        if ok.sum() < 5 * N_EXT:
            continue
        lo_key = np.where(ok, col, np.inf); hi_key = np.where(ok, -col, np.inf)
        li = np.argpartition(lo_key, N_EXT)[:N_EXT]; li = li[np.argsort(lo_key[li])]
        hi = np.argpartition(hi_key, N_EXT)[:N_EXT]; hi = hi[np.argsort(hi_key[hi])]
        lo_i[j], hi_i[j] = li, hi
        lo_b[j], hi_b[j] = col[li], col[hi]
        full = block[rows_u, j]
        v = np.where((full >= 0) & (full <= 1), np.round(full * 100.0), 255.0)
        v[~np.isfinite(full)] = 255.0
        prof[j, u_pos] = v.astype(np.uint8)
    return lo_i, lo_b, hi_i, hi_b, prof


def _job(args):
    split, gse, sample_ids, vocab_ids, universe_ids, raw_dir, out_dir = args
    out = Path(out_dir) / f"{gse}.npz"
    if out.exists():
        return gse, 0
    path = fetch_dataset(split, gse, raw_dir)
    mp = MethylParquet(path)
    vocab = pd.Index(vocab_ids)
    rv = vocab.get_indexer(mp.cpg_ids)                    # file row -> vocab index
    rows_v = np.flatnonzero(rv >= 0)
    # extremes are reported as *vocabulary* indices, so remap sub-row positions
    vocab_of_row_v = rv[rows_v]
    uni = pd.Index(universe_ids)
    ru = uni.get_indexer(mp.cpg_ids)                      # file row -> universe index
    rows_u = np.flatnonzero(ru >= 0); u_pos = ru[rows_u]
    present = [s for s in sample_ids if s in set(mp.sample_ids)]
    n = len(present)
    LI = np.zeros((n, N_EXT), np.int32); HI = np.zeros((n, N_EXT), np.int32)
    LB = np.full((n, N_EXT), np.nan, np.float16); HB = np.full((n, N_EXT), np.nan, np.float16)
    PR = np.full((n, len(uni)), 255, np.uint8)
    col = {s: i for i, s in enumerate(present)}
    for names, block in mp.iter_blocks(batch=48, columns=present):
        li, lb_, hi, hb, pr = _extract_sample_block(block, rows_v, rows_u, u_pos, len(uni))
        idx = [col[s] for s in names]
        LI[idx], HI[idx] = vocab_of_row_v[li], vocab_of_row_v[hi]
        LB[idx], HB[idx], PR[idx] = lb_, hb, pr
    np.savez(out.with_suffix(".tmp.npz"), samples=np.array(present), low_idx=LI, low_beta=LB, high_idx=HI, high_beta=HB, prof=PR)
    os.replace(out.with_suffix(".tmp.npz"), out)
    Path(path).unlink(missing_ok=True)
    return gse, n


def run(workers: int = 3) -> Path:
    ann = load_probe_annotation()
    vocab_ids = np.asarray(ann.index.values, dtype=str)
    universe = build_universe()
    ud = lb_dir()
    pd.Series(universe.values).to_frame("cpg").to_parquet(ud / "universe.parquet")
    jobs = []
    for split in ("train", "benchmark"):
        meta = load_meta(split)
        out_dir = subdir("lb", "store", split); raw_dir = subdir(f"raw_{split}")
        for gse, grp in meta.groupby("DatasetID"):
            jobs.append((split, gse, list(grp.index), vocab_ids, np.asarray(universe.values, dtype=str), str(raw_dir), str(out_dir)))
    jobs.sort(key=lambda j: -len(j[2]))                  # big studies first
    print(f"[lb-extract] universe {len(universe)} CpGs; {len(jobs)} studies", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for gse, n in ex.map(_job, jobs):
            print(f"[lb-extract] {gse}: {n} samples", flush=True)
    return ud
