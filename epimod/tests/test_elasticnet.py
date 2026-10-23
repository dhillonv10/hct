import numpy as np
import pandas as pd

from epimod.age import age_to_t
from epimod.data.store import MethylData
from epimod.eval.elasticnet import ElasticNetClock

from conftest import make_table


def test_elasticnet_clock_fits_a_linear_signal_with_study_grouped_cv():
    """Toy fixture: a handful of CpGs track age; the clock must recover it and tolerate NaNs."""
    rng = np.random.RandomState(0)
    n, p = 160, 300
    tbl = make_table(p)
    age = rng.uniform(20, 80, n)
    X = np.clip(tbl.mean + rng.randn(n, p) * tbl.std * 0.5, 0, 1)
    X[:, :6] = np.clip(tbl.mean[:6] + (age[:, None] - 50) * 0.004 + rng.randn(n, 6) * 0.01, 0, 1)
    X[rng.rand(n, p) < 0.02] = np.nan
    meta = pd.DataFrame({"Age": age, "DatasetID": np.repeat([f"GSE{i}" for i in range(8)], n // 8)},
                        index=[f"S{i}" for i in range(n)])
    data = MethylData(X=X.astype(np.float16), meta=meta, cpgs=tbl)
    en = ElasticNetClock(l1_ratio=(0.5,), n_alphas=15, cv=4).fit(data)
    pred = en.predict(X.astype(np.float32))
    assert np.corrcoef(pred, age)[0, 1] > 0.9 and 0 < en.n_nonzero < p
    assert np.isfinite(en.predict(np.full((2, p), np.nan, np.float32))).all()
