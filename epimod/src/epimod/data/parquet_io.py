"""Streaming access to ComputAge parquet files (CpGs are rows, samples are columns)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq


class MethylParquet:
    """Column-batched reader. A 900k-row file is never fully materialised: samples are read
    ``batch`` columns at a time as float32 ``[n_cpg_rows, batch]`` blocks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._pf = pq.ParquetFile(self.path)
        names = self._pf.schema_arrow.names
        self.index_col = names[0]
        self.sample_ids: list[str] = list(names[1:])
        self.cpg_ids: np.ndarray = self._pf.read(columns=[self.index_col]).column(0).to_numpy(zero_copy_only=False).astype(str)

    def iter_blocks(self, batch: int = 64, columns: list[str] | None = None) -> Iterator[tuple[list[str], np.ndarray]]:
        cols = self.sample_ids if columns is None else columns
        for i in range(0, len(cols), batch):
            names = cols[i : i + batch]
            tbl = self._pf.read(columns=names)
            block = np.empty((tbl.num_rows, len(names)), dtype=np.float32)
            for j in range(len(names)):
                block[:, j] = tbl.column(j).to_numpy(zero_copy_only=False)
            yield names, block
