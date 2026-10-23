import numpy as np
import pandas as pd
import torch

from epimod.age import age_to_t, t_to_age
from epimod.train.accel import AccelConfig, _t_to_age_torch, cross_fitted_age, fit_head


def test_torch_age_transform_matches_numpy():
    t = np.array([-2.0, -0.3, 0.0, 0.4, 2.5], dtype=np.float32)
    assert np.allclose(_t_to_age_torch(torch.from_numpy(t)).numpy(), t_to_age(t), atol=1e-4)


def _toy(n_studies=6, per=60, K=4, d=8, effect=1.0, seed=0):
    """Toy features where one hidden direction is shifted in cases (unit-test fixture only)."""
    rng = np.random.RandomState(seed)
    rows, h = [], []
    for s in range(n_studies):
        for i in range(per):
            case = i >= per // 2
            age = rng.uniform(30, 70)
            rows.append(dict(DatasetID=f"GSE{s}", Condition="RA" if case else "HC", Age=age))
            hv = rng.randn(K, d) * 0.5
            hv[0, 0] += effect if case else 0.0
            h.append(hv)
    meta = pd.DataFrame(rows, index=[f"S{i}" for i in range(len(rows))])
    h = np.stack(h).astype(np.float32)
    mu = age_to_t(meta.Age.to_numpy()).astype(np.float32) + np.random.RandomState(1).randn(len(meta)).astype(np.float32) * 0.05
    return {"h": h, "mu_t": mu, "contrib": np.zeros((len(meta), K), np.float32)}, meta


def test_head_learns_to_separate_cases_without_moving_controls():
    feats, meta = _toy()
    head = fit_head(feats, meta, np.arange(len(meta)), AccelConfig(steps=200))
    with torch.no_grad():
        shift = head(torch.from_numpy(feats["h"])).numpy()
    case = (meta.Condition != "HC").to_numpy()
    assert shift[case].mean() - shift[~case].mean() > 0.01
    assert abs(shift[~case].mean()) < abs(shift[case].mean()) + 0.05


def test_cross_fitting_scores_unseen_studies():
    feats, meta = _toy()
    pred, mod = cross_fitted_age(feats, meta, fold_by="study", n_folds=3, cfg=AccelConfig(steps=150))
    assert pred.shape == (len(meta),) and mod.shape == feats["contrib"].shape and np.isfinite(pred).all()
    case = (meta.Condition != "HC").to_numpy()
    base = t_to_age(feats["mu_t"])
    assert (pred - base)[case].mean() > (pred - base)[~case].mean()   # held-out studies still read older


def test_conditions_without_age_matched_controls_are_skipped():
    """Regression: a case group whose ages are all outside the window of every control gave 0 pairs."""
    feats, meta = _toy(n_studies=4)
    meta = meta.copy()
    meta.loc[(meta.DatasetID == "GSE0") & (meta.Condition != "HC"), "Age"] = 95.0   # no control within 7 y
    pred, mod = cross_fitted_age(feats, meta, fold_by="study", n_folds=2, cfg=AccelConfig(steps=20))
    assert np.isfinite(pred).all()
