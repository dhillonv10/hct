"""Shared fixtures. The arrays below are *toy inputs for unit tests only* (shape / invariance /
gradient checks); they are never used to train or evaluate the released model."""
import numpy as np
import pytest

from epimod.data.store import CpGTable


def make_table(n: int, seed: int = 0) -> CpGTable:
    rng = np.random.RandomState(seed)
    return CpGTable(
        ids=np.array([f"cg{i:08d}" for i in range(n)]),
        chrom=rng.randint(0, 24, n), pos=rng.randint(1_000, 200_000_000, n).astype(np.int64),
        island=rng.randint(0, 6, n), reg=rng.randint(0, 9, n),
        binary=rng.randint(0, 2, (n, 9)).astype(np.float32),
        mean=rng.uniform(0.1, 0.9, n).astype(np.float32), std=rng.uniform(0.03, 0.2, n).astype(np.float32),
        source=np.array(["age_assoc"] * n), z_age=rng.randn(n).astype(np.float32),
    )


@pytest.fixture
def table():
    return make_table(300)


@pytest.fixture
def batch(table):
    rng = np.random.RandomState(1)
    x = np.clip(table.mean + rng.randn(8, len(table.ids)) * table.std, 0, 1).astype(np.float32)
    x[rng.rand(*x.shape) < 0.05] = np.nan
    return x
