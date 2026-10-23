import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from epimod.age import age_to_t
from epimod.data.panel import select_panel, study_stats
from epimod.eval.computage_protocol import run_protocol


def _write_study(path, n_cpg=40, n=60, seed=0):
    rng = np.random.RandomState(seed)
    ages = rng.uniform(20, 80, n)
    X = rng.uniform(0.2, 0.8, (n_cpg, n))
    X[0] = 0.2 + 0.006 * ages + rng.randn(n) * 0.01          # strongly age-associated CpG
    X[1, :5] = np.nan                                        # a CpG with a few missing values
    ids = [f"cg{i:08d}" for i in range(n_cpg)]
    cols = {"index": ids}
    samples = [f"GSM{i}" for i in range(n)]
    for j, s in enumerate(samples):
        cols[s] = X[:, j].astype(np.float32)
    pq.write_table(pa.table(cols), path)
    return ids, dict(zip(samples, ages)), X, ages


def test_streaming_stats_match_direct_computation(tmp_path):
    ids, ages, X, a = _write_study(tmp_path / "s.parquet")
    st = study_stats(str(tmp_path / "s.parquet"), np.array(ids), ages)
    y = age_to_t(a)
    assert np.isclose(st.r[0], np.corrcoef(X[0], y)[0, 1], atol=1e-4)
    ok = np.isfinite(X[1])
    assert np.isclose(st.r[1], np.corrcoef(X[1][ok], y[ok])[0, 1], atol=1e-4)
    assert st.n[1] == ok.sum() and st.n[0] == len(a)


def test_panel_selection_rules():
    n = 600
    ids = np.array([f"cg{i:08d}" for i in range(n)])
    rng = np.random.RandomState(0)
    stats = dict(
        z_meta=rng.randn(n).astype(np.float32) * 5, r_meta=np.zeros(n, np.float32),
        frac_pos=np.where(rng.rand(n) < 0.5, 0.95, 0.5).astype(np.float32), n_studies_r=np.full(n, 20),
        coverage=np.ones(n, np.float32), within_var=rng.rand(n).astype(np.float32),
        mean=np.full(n, 0.5, np.float32), std=np.full(n, 0.1, np.float32),
    )
    ann = pd.DataFrame({"chrom": rng.randint(0, 24, n), "pos": 1000, "snp": (rng.rand(n) < 0.3).astype(int)}, index=ids)
    clock = list(ids[:5])
    p = select_panel(ids, stats, ann, must_include=clock, n_total=300, n_age=100, n_x=20)
    assert list(p.index[:5]) == clock and (p.source[:5] == "clock").all()          # clocks first
    rest = p.iloc[5:]
    assert (ann.loc[rest.index, "snp"] == 0).all()                                 # SNP probes never filler
    assert (ann.loc[rest.index, "chrom"] < 23).all()                               # no Y
    assert (rest.source == "xchrom").sum() <= 20 and len(p) <= 300
    assert p.index.is_unique and (p.priority.values == np.arange(len(p))).all()


def _bench_meta(effect, n_hc=40, n_case=40, seed=0):
    rng = np.random.RandomState(seed)
    age = rng.uniform(30, 70, n_hc + n_case)
    meta = pd.DataFrame({
        "DatasetID": "GSE1", "Tissue": "Blood", "Age": age,
        "Condition": ["HC"] * n_hc + ["XYZ"] * n_case, "Platform": "450K",
    }, index=[f"S{i}" for i in range(n_hc + n_case)])
    pred = age + rng.randn(len(age)) * 3 + np.where(meta.Condition == "XYZ", effect, 0.0)
    return meta, pred


def test_protocol_detects_true_acceleration_and_not_null():
    tests = pd.DataFrame({"dataset": ["GSE1"], "condition": ["XYZ"], "test": ["AA2"]})
    meta, pred = _bench_meta(effect=6.0)
    summ, long = run_protocol(pd.DataFrame({"m": pred}, index=meta.index), meta, tests)
    assert summ.loc["m", "AA2_hits"] == 1 and long.effect_years.iloc[0] > 3
    meta0, pred0 = _bench_meta(effect=0.0, seed=1)
    summ0, _ = run_protocol(pd.DataFrame({"m": pred0}, index=meta0.index), meta0, tests)
    assert summ0.loc["m", "AA2_hits"] == 0


def test_protocol_age_limits_and_aa1():
    meta, pred = _bench_meta(effect=5.0)
    tests = pd.DataFrame({"dataset": ["GSE1"], "condition": ["XYZ"], "test": ["AA1"]})
    summ, long = run_protocol(pd.DataFrame({"m": pred}, index=meta.index), meta, tests)
    assert long.n_case.iloc[0] == 40 and summ.loc["m", "AA1_hits"] == 1
    meta.loc[meta.Condition == "XYZ", "Age"] = 95           # outside 18-90 -> all cases excluded
    summ2, long2 = run_protocol(pd.DataFrame({"m": pred}, index=meta.index), meta, tests)
    assert long2.empty
