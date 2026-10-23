"""Horvath-style log/linear age transform.

Methylation changes fast in childhood and roughly linearly in adulthood; modelling age in this
space (log below ``ADULT_AGE``, linear above) keeps child and adult errors on one scale.
``t_to_age`` is identical to Horvath's ``anti.trafo``, so published-clock outputs can use it too.
"""
from __future__ import annotations

import numpy as np

ADULT_AGE = 20.0


def age_to_t(age):
    age = np.asarray(age, dtype=np.float64)
    return np.where(
        age <= ADULT_AGE,
        np.log(np.clip(age, 0.0, None) + 1.0) - np.log(ADULT_AGE + 1.0),
        (age - ADULT_AGE) / (ADULT_AGE + 1.0),
    )


def t_to_age(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t <= 0.0, (ADULT_AGE + 1.0) * np.exp(np.minimum(t, 0.0)) - 1.0, (ADULT_AGE + 1.0) * t + ADULT_AGE)


def dage_dt(t):
    """Derivative of ``t_to_age`` (years per unit of transformed age); used to map sigma to years."""
    t = np.asarray(t, dtype=np.float64)
    return np.where(t <= 0.0, (ADULT_AGE + 1.0) * np.exp(np.minimum(t, 0.0)), ADULT_AGE + 1.0)
