"""Generators for LongevityBench-format inputs. The store below is a toy fixture for checking mechanics."""
import numpy as np
import pandas as pd

from epimod.lb.store import MISSING, LBStore, pair_tokens, pooled_extremes, sample_pairs
from epimod.lb.tasks import Vocab, balanced_accuracy, decade_of, decade_posterior, extremes_batch, pair_batch
from epimod.age import age_to_t


def toy_store(S=12, U=400, seed=0):
    rng = np.random.RandomState(seed)
    prof = rng.randint(0, 101, size=(S, U)).astype(np.uint8)
    prof[0, :5] = MISSING
    meta = pd.DataFrame({"Age": rng.uniform(5, 90, S), "Gender": rng.choice(["M", "F", "U"], S), "DatasetID": [f"GSE{i % 3}" for i in range(S)]},
                        index=[f"S{i}" for i in range(S)])
    return LBStore(meta, pd.Index([f"cg{i:08d}" for i in range(U)]), pd.Index([]), np.zeros((S, 1), np.int32), np.zeros((S, 1), np.float16),
                   np.zeros((S, 1), np.int32), np.zeros((S, 1), np.float16), prof)


def test_pooled_extremes_are_the_true_lowest_and_highest_within_the_pool():
    st = toy_store(); low, high = np.arange(0, 150), np.arange(150, 400)
    lc, lv, hc, hv = pooled_extremes(st, low, high, n=20)
    for s in range(st.n):
        x_lo = st.prof[s, low].astype(int); x_lo = np.where(x_lo == MISSING, 10_000, x_lo)
        assert sorted(x_lo[np.searchsorted(low, lc[s])]) == sorted(np.sort(x_lo)[:20]), "lowest-20 values wrong"
        assert set(lc[s]) <= set(low) and set(hc[s]) <= set(high)               # lists never leave their pool
        assert np.all(np.diff(lv[s]) >= -1e-9) and np.all(np.diff(hv[s]) <= 1e-9)  # ascending / descending
    assert not (lv == MISSING / 100).any()                                     # missing never enters a list


def test_pooled_extremes_pads_when_fewer_than_n_observed():
    st = toy_store(); st.prof[3, :] = MISSING
    lc, lv, hc, hv = pooled_extremes(st, np.arange(0, 100), np.arange(100, 200), n=10)
    assert (lc[3] == -1).all() and (hc[3] == -1).all() and (lv[3] == 0).all()


def test_pair_tokens_pick_largest_differences_and_respect_candidates():
    st = toy_store(); cand = np.arange(50, 250)
    ia, ib = np.array([1, 2]), np.array([3, 4])
    cols, ba, bb = pair_tokens(st, ia, ib, k=15, cand=cand)
    assert np.isin(cols, cand).all()
    for r in range(2):
        a, b = st.prof[ia[r], cand].astype(int), st.prof[ib[r], cand].astype(int)
        d = np.abs(a - b); d[(a == MISSING) | (b == MISSING)] = -1
        assert np.isclose(np.sort(np.abs(ba[r] - bb[r]))[::-1][:15].sum(), np.sort(d)[::-1][:15].sum() / 100.0, atol=1e-6)
        assert np.all(np.diff(np.abs(ba[r] - bb[r])) <= 1e-6)                  # sorted by |difference|


def test_sample_pairs_enforce_min_age_gap_and_mostly_cross_study():
    rng = np.random.RandomState(0)
    ages = np.random.RandomState(1).uniform(0, 90, 500); series = np.random.RandomState(2).randint(0, 20, 500)
    ia, ib = sample_pairs(ages, 2000, rng, 10.0, series)
    assert (np.abs(ages[ia] - ages[ib]) >= 10.0).all()
    assert (series[ia] == series[ib]).mean() < 0.15


def test_decade_helpers():
    assert list(decade_of([0, 9.9, 10, 59, 99.9, 100, 120])) == [0, 0, 1, 5, 9, 10, 10]
    post = decade_posterior(np.array([age_to_t(55.0)]), np.array([0.05]))
    assert post.shape == (1, 11) and post.argmax() == 5
    ba = balanced_accuracy(np.array([1, 1, 2, 2]), np.array([1, 2, 2, 2]))
    # class 1 recall = 1/1, class 2 recall = 2/3 -> balanced 5/6; plain accuracy = 3/4
    assert abs(ba["balanced_acc_present"] - 5 / 6) < 1e-9 and abs(ba["accuracy"] - 0.75) < 1e-9
