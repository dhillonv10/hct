"""Re-implementation of the ComputAge Bench scoring protocol (Kriukov et al.).

A clock should (i) predict chronological age of healthy controls well (median absolute error), and
(ii) read *older* in people with aging-accelerating conditions (AACs):

* AA2 -- the condition's age acceleration (pred - age) exceeds that of same-study healthy controls
         (one-sided Welch t-test);
* AA1 -- (studies without controls) the acceleration is positive (one-sided one-sample t-test).

p-values are Benjamini-Hochberg corrected across all tests of a model; the headline score is
``AA2_hits + AA1_hits * (1 - relu(MedE) / MedAE)``. Ages are restricted to 18-90 y (progeroid
syndromes exempt). The curated test list is read from the ``computage`` package when installed,
otherwise derived from the metadata (every non-HC condition of every benchmark study).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CLASS_OF = {
    "AS": "CVD", "IHD": "CVD", "CVA": "CVD", "IBD": "ISD", "HIV": "ISD", "HIV_TB": "ISD", "XOB": "MBD", "T1D": "MBD",
    "T2D": "MBD", "OP": "MSD", "RA": "MSD", "MS": "NDD", "PD": "NDD", "AD": "NDD", "CJD": "NDD", "TB": "RSD",
    "COPD": "RSD", "CGL": "PGS", "WS": "PGS", "HGPS": "PGS",
}


def curated_tests(meta: pd.DataFrame) -> pd.DataFrame:
    """Rows (dataset, condition, test) -- the benchmark's curated list when available."""
    spec = importlib.util.find_spec("computage")
    if spec is not None and spec.submodule_search_locations:
        f = Path(list(spec.submodule_search_locations)[0]) / "data_library" / "BenchBSB_table.csv"
        if f.exists():
            t = pd.read_csv(f)
            return pd.DataFrame(
                {"dataset": t["Dataset ID"], "condition": t["Condition"], "test": np.where(t["HC content"] == "HC", "AA2", "AA1")}
            )
    rows = []
    for gse, g in meta.groupby("DatasetID"):
        has_hc = (g.Condition == "HC").any()
        for cond in sorted(set(g.Condition) - {"HC"}):
            rows.append((gse, cond, "AA2" if has_hc else "AA1"))
    return pd.DataFrame(rows, columns=["dataset", "condition", "test"])


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(p), p, 1.0)
    return stats.false_discovery_control(p, method="bh") if len(p) else p


def run_protocol(preds: pd.DataFrame, meta: pd.DataFrame, tests: pd.DataFrame | None = None,
                 age_limits: tuple[float, float] = (18, 90), alpha: float = 0.05):
    """``preds``: predicted age in years, index = sample id, one column per model."""
    meta = meta.loc[preds.index].copy()
    meta = meta[meta.Tissue.isin(["Blood", "Saliva", "Buccal"])]
    preds = preds.loc[meta.index]
    tests = curated_tests(meta) if tests is None else tests
    tests = tests[tests.dataset.isin(meta.DatasetID.unique())]
    records = []
    for _, t in tests.iterrows():
        g = meta[meta.DatasetID == t.dataset]
        exempt = CLASS_OF.get(t.condition) == "PGS"
        gg = g if exempt else g[(g.Age >= age_limits[0]) & (g.Age <= age_limits[1])]
        dis, hc = gg.index[gg.Condition == t.condition], gg.index[gg.Condition == "HC"]
        if len(dis) == 0 or (t.test == "AA2" and len(hc) == 0):
            continue
        for model in preds.columns:
            d_dis = (preds.loc[dis, model] - gg.loc[dis, "Age"]).to_numpy(float)
            if t.test == "AA2":
                d_hc = (preds.loc[hc, model] - gg.loc[hc, "Age"]).to_numpy(float)
                p = stats.ttest_ind(d_dis, d_hc, equal_var=False, alternative="greater").pvalue
                effect = d_dis.mean() - d_hc.mean()
            else:
                p = stats.ttest_1samp(d_dis, 0.0, alternative="greater").pvalue
                effect, d_hc = d_dis.mean(), np.array([])
            records.append(dict(model=model, dataset=t.dataset, condition=t.condition, test=t.test, p=p,
                                effect_years=effect, n_case=len(dis), n_hc=len(hc)))
    long = pd.DataFrame(records)
    if long.empty:
        return pd.DataFrame(), long
    long["q"] = np.nan
    for (model, test), idx in long.groupby(["model", "test"]).groups.items():
        long.loc[idx, "q"] = _bh(long.loc[idx, "p"].to_numpy())
    long["hit"] = long.q < alpha

    hc_all = meta[(meta.Condition == "HC") & meta.Age.notna()]
    rows = []
    for model in preds.columns:
        err = preds.loc[hc_all.index, model] - hc_all.Age
        medae, mede = float(np.median(np.abs(err))), float(np.median(err))
        adult = hc_all.Age.between(*age_limits)
        medae_adult = float(np.median(np.abs(err[adult])))
        m = long[long.model == model]
        aa2, aa1 = int(m[m.test == "AA2"].hit.sum()), int(m[m.test == "AA1"].hit.sum())
        rows.append(dict(
            model=model, MedAE=medae, MedAE_18_90=medae_adult, MedE=mede, AA2_hits=aa2, AA2_tests=int((m.test == "AA2").sum()),
            AA1_hits=aa1, AA1_tests=int((m.test == "AA1").sum()),
            score=aa2 + aa1 * (1 - max(0.0, mede) / medae),
            mean_effect_years_AA2=float(m[m.test == "AA2"].effect_years.mean()),
            frac_AA2_positive=float((m[m.test == "AA2"].effect_years > 0).mean()),
        ))
    return pd.DataFrame(rows).set_index("model").sort_values("score", ascending=False), long
