import numpy as np
import torch

from epimod.age import age_to_t, dage_dt, t_to_age
from epimod.config import TrainConfig
from epimod.train.augment import augment_batch


def test_age_transform_round_trip_and_horvath_values():
    a = np.array([0.0, 0.5, 5.0, 20.0, 21.0, 45.0, 90.0])
    assert np.allclose(t_to_age(age_to_t(a)), a)
    # Horvath anti.trafo: x<0 -> 21*exp(x)-1 ; x>=0 -> 21*x+20
    assert np.isclose(t_to_age(-1.0), 21 * np.exp(-1.0) - 1)
    assert np.isclose(t_to_age(0.5), 21 * 0.5 + 20)
    assert np.isclose(age_to_t(20.0), 0.0)


def test_age_derivative_matches_finite_difference():
    t = np.array([-1.2, -0.1, 0.3, 2.0]); h = 1e-6
    fd = (t_to_age(t + h) - t_to_age(t - h)) / (2 * h)
    assert np.allclose(dage_dt(t), fd, atol=1e-4)


def test_augmentation_never_invents_measurements_and_masks_only_observed():
    g = torch.Generator().manual_seed(0)
    beta = torch.rand(16, 400, generator=g)
    beta[torch.rand(16, 400, generator=g) < 0.1] = float("nan")
    pool = beta.clone()
    cfg = TrainConfig(template_miss_prob=1.0)
    out, hidden = augment_batch(beta, cfg, g, pool)
    obs_in, obs_out = torch.isfinite(beta), torch.isfinite(out)
    assert not (obs_out & ~obs_in).any(), "augmentation turned a missing value into an observed one"
    assert (hidden <= obs_in).all(), "SSL mask must only cover measured values"
    assert not (hidden & obs_out).any(), "masked values must be hidden from the model"
    v = out[obs_out]
    assert v.min() >= 0.0 and v.max() <= 1.0
