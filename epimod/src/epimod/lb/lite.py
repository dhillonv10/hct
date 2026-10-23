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
