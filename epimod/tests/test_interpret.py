import numpy as np
import pandas as pd

from epimod.eval.interpret import module_shift_table


def test_module_shift_is_age_adjusted_and_localised():
    """Toy fixture: cases are 8 y older than controls (a pure age confound on every module) and
    carry a true extra shift only in module 2. The adjusted estimate must find module 2 only."""
    rng = np.random.RandomState(0)
    rows, contrib = [], []
    for s in range(4):
        for i in range(120):
            case = i >= 60
            age = rng.uniform(30, 60) + (8.0 if case else 0.0)
            rows.append(dict(DatasetID=f"GSE{s}", Condition="RA" if case else "HC", Age=age))
            c = rng.randn(5) * 0.05 + (age - 45) / 10 * 0.2          # every module tracks age
            c[2] += 0.3 if case else 0.0                              # plus a real case effect in module 2
            contrib.append(c)
    meta = pd.DataFrame(rows, index=[f"S{i}" for i in range(len(rows))])
    est, p = module_shift_table(np.array(contrib), meta)
    e = est.loc["RA"].to_numpy()
    assert abs(e[2] - 0.3) < 0.03 and np.abs(np.delete(e, 2)).max() < 0.03   # age gap is not counted as a case effect
    assert p.loc["RA"].to_numpy()[2] < 1e-6
