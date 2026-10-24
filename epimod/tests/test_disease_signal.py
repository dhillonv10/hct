import numpy as np
import pandas as pd

from epimod.eval.disease_signal import group_report


def test_group_report_direction_and_auc():
    rng = np.random.RandomState(0)
    n = 60
    sc = pd.DataFrame({"age": rng.uniform(30, 70, n), "age_acceleration": np.r_[rng.randn(30), rng.randn(30) + 3.0],
                       "abs_age_acceleration": rng.rand(n), "deviation": rng.rand(n)}, index=[f"s{i}" for i in range(n)])
    grp = pd.Series(["ctrl"] * 30 + ["case"] * 30, index=sc.index)
    r = group_report(sc, grp, "ctrl").set_index("score")
    assert r.loc["age_acceleration", "auc"] > 0.95 and r.loc["age_acceleration", "cohens_d"] > 2
    assert 0.3 < r.loc["deviation", "auc"] < 0.7
    byst = group_report(sc, grp, "ctrl", pd.Series(["a", "b"] * 30, index=sc.index))
    assert set(byst.study) == {"a", "b"}
