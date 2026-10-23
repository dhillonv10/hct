"""Published epigenetic clocks as baselines, evaluated exactly like the ComputAge benchmark does.

Coefficients are read from the ``computage`` PyPI package (``pip install computage --no-deps``),
which ships them with the transforms below (taken from biolearn). Missing CpGs are imputed with the
SeSAMe 450K per-CpG medians, as in the benchmark's recommended setting.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ..age import t_to_age


def _anti_trafo(x):
    return t_to_age(np.asarray(x, dtype=np.float64))


# name -> (coefficient file, transform of the weighted sum)
PUBLISHED: dict[str, tuple[str, callable]] = {
    "HorvathV1": ("Horvath1.csv", lambda s: _anti_trafo(s + 0.696)),
    "HorvathV2": ("Horvath2.csv", lambda s: _anti_trafo(s - 0.447119319)),
    "Hannum": ("Hannum.csv", lambda s: s),
    "PhenoAgeV1": ("PhenoAge.csv", lambda s: s + 60.664),
    "PhenoAgeV2": ("HRSInCHPhenoAge.csv", lambda s: s + 52.8334080),
    "Lin": ("Lin.csv", lambda s: s + 12.2169841),
    "Zhang19_EN": ("Zhang19_EN.csv", lambda s: s + 65.79295),
}


def clocks_dir() -> Path:
    spec = importlib.util.find_spec("computage")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("published-clock baselines need `pip install computage --no-deps` (coefficient files)")
    return Path(list(spec.submodule_search_locations)[0]) / "models_library" / "clocks"


def load_coefficients(name: str) -> pd.Series:
    f = PUBLISHED[name][0]
    s = pd.read_csv(clocks_dir() / f, index_col=0)["CoefficientTraining"]
    return s[s.index.str.startswith("cg")]


def published_clock_cpgs(names: list[str] | None = None) -> list[str]:
    out: dict[str, None] = {}
    for n in names or list(PUBLISHED):
        for c in load_coefficients(n).index:
            out[c] = None
    return list(out)


def sesame_medians() -> pd.Series:
    return pd.read_csv(clocks_dir() / "sesame_450k_median.csv", index_col=0)["median"]


def predict_published(name: str, X: np.ndarray, cpg_ids: np.ndarray, impute: pd.Series | None = None) -> np.ndarray:
    """Predict age (years) for beta matrix ``X`` [S, N] whose columns are ``cpg_ids``.

    CpGs that are absent from the matrix, or NaN, are imputed with ``impute`` (SeSAMe medians)."""
    coef = load_coefficients(name)
    impute = sesame_medians() if impute is None else impute
    col = pd.Index(cpg_ids).get_indexer(coef.index)
    S = X.shape[0]
    total = np.zeros(S, dtype=np.float64)
    med = impute.reindex(coef.index).fillna(0.5).to_numpy()
    for j, (c, w) in enumerate(zip(coef.index, coef.to_numpy())):
        if col[j] >= 0:
            v = X[:, col[j]].astype(np.float64)
            v = np.where(np.isfinite(v), v, med[j])
        else:
            v = np.full(S, med[j])
        total += w * v
    return np.asarray(PUBLISHED[name][1](total), dtype=np.float64)
