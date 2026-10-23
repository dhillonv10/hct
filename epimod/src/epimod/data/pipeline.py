"""End-to-end data preparation: stats -> panel -> extraction (train + benchmark)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import subdir
from .annotation import load_probe_annotation
from .computage import fetch_dataset, load_meta
from .extract import run_extract
from .panel import collect_stats, select_panel


def prepare(n_total: int = 16000, n_age: int = 12000, workers: int = 3, with_clock_cpgs: bool = True,
            keep_benchmark_raw: bool = False) -> None:
    store = subdir("store")
    ann = load_probe_annotation()
    vocab = np.asarray(ann.index.values, dtype=str)
    train_meta = load_meta("train")
    train_meta = train_meta[np.isfinite(train_meta["Age"])]

    stats_path = store / "panel_stats.npz"
    if stats_path.exists():
        stats = dict(np.load(stats_path))
        print(f"[prepare] reusing {stats_path}")
    else:
        gses = sorted(train_meta.DatasetID.unique())
        paths = {g: fetch_dataset("train", g) for g in gses}
        stats = collect_stats(paths, vocab, train_meta, workers=workers)
        np.savez(stats_path, **stats)

    panel_path = store / "panel.parquet"
    if not panel_path.exists():
        must: list[str] = []
        if with_clock_cpgs:
            from ..eval.baselines import published_clock_cpgs

            must = published_clock_cpgs()
        panel = select_panel(vocab, stats, ann, must_include=must, n_total=n_total, n_age=n_age)
        panel.to_parquet(panel_path)
        print(f"[prepare] panel: {len(panel)} CpGs; by source: {panel.source.value_counts().to_dict()}")
    panel = pd.read_parquet(panel_path)

    run_extract("train", panel, workers=workers, delete_raw=not keep_benchmark_raw)
    run_extract("benchmark", panel, workers=workers, delete_raw=not keep_benchmark_raw)
    print("[prepare] done")
