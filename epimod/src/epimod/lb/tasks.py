"""Tensor builders, vocabulary and metrics for the LongevityBench DNAm tasks."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..age import age_to_t
from ..paths import subdir
from .episet import N_DECADES, ROLES
from .store import MISSING, LBStore, pair_tokens, pooled_extremes

SEX_ID = {"": 0, "U": 0, "M": 1, "F": 2}
CLOCKS = ["Hannum", "Horvathv1", "Horvathv2", "Lin", "PEDBE", "VidalBralo", "YingAdaptAge", "YingCausAge", "YingDamAge"]
DECADE_LABELS = [f"{10 * i}-{10 * i + 9}" for i in range(10)] + ["100+"]


def decade_of(age) -> np.ndarray:
    return np.minimum(np.floor(np.asarray(age, dtype=np.float64) / 10.0), N_DECADES - 1).astype(np.int64)


def decade_label_to_id(lbl: str) -> int:
    return DECADE_LABELS.index(lbl)


def load_parsed() -> dict:
    return pickle.load(open(subdir("lb") / "parsed.pkl", "rb"))


@dataclass
class Pools:
    """Candidate CpG pools (universe columns) from which the benchmark builds its CpG lists."""

    low: np.ndarray       # candidates for the lowest-methylated list (LB-0010 / LB-0002)
    high: np.ndarray
    pair: np.ndarray      # candidates for the most-different-sites list (LB-0006)


def make_pools(parsed: dict, universe: pd.Index, ext_mask: np.ndarray | None = None, pair_mask: np.ndarray | None = None) -> Pools:
    """Pools from the CpG *identities* (never values or labels) shown in the selected eval prompts."""
    t = parsed["LB-0010"]
    sel = range(len(t.ids)) if ext_mask is None else np.flatnonzero(ext_mask)
    lo, hi = set(), set()
    for i in sel:
        flag = t.is_high[i].astype(bool)
        lo.update(t.ids[i][~flag]); hi.update(t.ids[i][flag])
    p = parsed["LB-0006"]
    sel = range(len(p.ids)) if pair_mask is None else np.flatnonzero(pair_mask)
    pr = set()
    for i in sel:
        pr.update(p.ids[i])
    cols = lambda ids: (lambda u: u[u >= 0])(universe.get_indexer(sorted(ids)))
    return Pools(cols(lo), cols(hi), cols(pr))


@dataclass
class Vocab:
    """CpG vocabulary of the model: universe column -> model id (-1 = not in vocabulary)."""

    u2m: np.ndarray
    cpg_ids: pd.Index          # model id -> probe id
    mean: np.ndarray           # per model id: mean beta over training profiles
    std: np.ndarray

    def __len__(self) -> int:
        return len(self.cpg_ids)


def build_vocab(store: LBStore, parsed: dict, pools: Pools) -> Vocab:
    """Vocabulary = candidate pools + the fixed 353 set + every published-clock CpG shown in clock prompts.

    CpGs outside it (e.g. seen only in held-out prompts) are unknown to the model and dropped."""
    uni = store.universe
    keep = np.zeros(len(uni), bool)
    for cols in (pools.low, pools.high, pools.pair):
        keep[cols] = True
    for ids in list(parsed["LB-0190"].ids[:1]) + list(parsed["LB-0195"].ids):
        u = uni.get_indexer(list(ids)); keep[u[u >= 0]] = True
    sel = np.flatnonzero(keep)
    u2m = np.full(len(uni), -1, np.int64); u2m[sel] = np.arange(len(sel))
    prof = store.prof[:, sel].astype(np.float32); prof[prof == MISSING] = np.nan
    prof /= 100.0
    mean = np.nanmean(prof, axis=0); std = np.nanstd(prof, axis=0)
    return Vocab(u2m, pd.Index(uni[sel]), np.nan_to_num(mean, nan=0.5), np.nan_to_num(std, nan=0.25))


def extremes_cache(store: LBStore, pools: Pools) -> dict:
    lc, lv, hc, hv = pooled_extremes(store, pools.low, pools.high)
    return {"low_col": lc, "low_val": lv, "high_col": hc, "high_val": hv}


# ---------------------------------------------------------------------------------------------
def _tensors(cpg, beta, role, sex, clock=None):
    out = dict(cpg=torch.as_tensor(cpg, dtype=torch.long), beta=torch.as_tensor(beta, dtype=torch.float32),
               role=torch.as_tensor(role, dtype=torch.long), sex=torch.as_tensor(sex, dtype=torch.long))
    if clock is not None:
        out["clock"] = torch.as_tensor(clock, dtype=torch.long)
    return out


def sex_ids(meta: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    return np.array([SEX_ID.get(g, 0) for g in meta.Gender.to_numpy()[rows]], dtype=np.int64)


def extremes_batch(store: LBStore, vocab: Vocab, cache: dict, rows: np.ndarray, drop: float = 0.0, rng=None, tail_shift: float = 0.0):
    ucpg = np.concatenate([cache["low_col"][rows], cache["high_col"][rows]], axis=1)
    beta = np.concatenate([cache["low_val"][rows], cache["high_val"][rows]], axis=1).astype(np.float32)
    n = cache["low_col"].shape[1]
    if tail_shift > 0:   # tail values depend on the preprocessing pipeline: lift the low list, lower the high list
        s = rng.uniform(0, tail_shift, size=(len(rows), 1)).astype(np.float32)
        beta = np.concatenate([beta[:, :n] + s, beta[:, n:] - rng.uniform(0, tail_shift, size=(len(rows), 1)).astype(np.float32)], axis=1)
    beta = np.round(np.clip(beta, 0.0, 1.0), 2).astype(np.float32)
    role = np.concatenate([np.zeros((len(rows), n)), np.ones((len(rows), n))], axis=1).astype(np.int64)
    mcpg = np.where(ucpg >= 0, vocab.u2m[np.maximum(ucpg, 0)], -1)
    if drop > 0:
        mcpg = np.where(rng.rand(*mcpg.shape) < rng.uniform(0, drop, size=(len(rows), 1)), -1, mcpg)
    return _tensors(mcpg, beta, role, sex_ids(store.meta, rows))


def fixed_batch(store: LBStore, vocab: Vocab, rows: np.ndarray, fixed_u: np.ndarray, drop: float = 0.0, rng=None):
    """Fixed CpG set given as universe columns; missing values become padding."""
    x = store.prof[rows][:, fixed_u]
    mcpg = np.broadcast_to(vocab.u2m[fixed_u], x.shape).copy()
    mcpg[x == MISSING] = -1
    if drop > 0:
        mcpg = np.where(rng.rand(*mcpg.shape) < rng.uniform(0, drop, size=(len(rows), 1)), -1, mcpg)
    return _tensors(mcpg, x.astype(np.float32) / 100.0, np.full(x.shape, ROLES["fixed"]), sex_ids(store.meta, rows))


def clock_batch(store: LBStore, vocab: Vocab, rows: np.ndarray, clock_sets: dict[str, np.ndarray], clock_names: list[str],
                max_cpgs: int, rng):
    """For each (sample, clock): up to ``max_cpgs`` of that clock's CpGs that are observed, in random order."""
    T = max_cpgs
    cpg = np.full((len(rows), T), -1, np.int64); beta = np.zeros((len(rows), T), np.float32)
    for r, (row, name) in enumerate(zip(rows, clock_names)):
        cols = clock_sets[name]
        x = store.prof[row, cols]
        ok = np.flatnonzero(x != MISSING)
        if len(ok) > max_cpgs:
            ok = rng.choice(ok, max_cpgs, replace=False)
        cpg[r, : len(ok)] = vocab.u2m[cols[ok]]; beta[r, : len(ok)] = x[ok] / 100.0
    clock = np.array([CLOCKS.index(c) for c in clock_names])
    return _tensors(cpg, beta, np.full(cpg.shape, ROLES["clock"]), sex_ids(store.meta, rows), clock)


def pair_batch(store: LBStore, vocab: Vocab, ia: np.ndarray, ib: np.ndarray, k: int = 160, cand: np.ndarray | None = None):
    """Two views over the shared k most-different CpGs (within ``cand``). Returns (view_a, view_b)."""
    ucols, ba, bb = pair_tokens(store, ia, ib, k, cand)
    mcpg = vocab.u2m[ucols]
    role = np.full(ucols.shape, ROLES["pair_a"])
    return (_tensors(mcpg, np.round(ba, 2), role, sex_ids(store.meta, ia)),
            _tensors(mcpg, np.round(bb, 2), role, sex_ids(store.meta, ib)))


# ------------------------------------------------------------------ LongevityBench eval items
def _map(vocab: Vocab, ids, universe: pd.Index) -> np.ndarray:
    u = universe.get_indexer(list(ids))
    return np.where(u >= 0, vocab.u2m[np.maximum(u, 0)], -1)


def _pad(rows: list[np.ndarray], fill, dtype) -> np.ndarray:
    T = max(len(r) for r in rows)
    out = np.full((len(rows), T), fill, dtype)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def eval_single(task, vocab: Vocab, universe: pd.Index):
    cpg = _pad([_map(vocab, i, universe) for i in task.ids], -1, np.int64)
    beta = _pad(list(task.beta), 0.0, np.float32)
    role = _pad(list(task.is_high), 0, np.int64)
    return _tensors(cpg, beta, role, np.array([SEX_ID.get(s, 0) for s in task.sex]))


def eval_fixed(task, vocab: Vocab, universe: pd.Index):
    cpg = _pad([_map(vocab, i, universe) for i in task.ids], -1, np.int64)
    return _tensors(cpg, _pad(list(task.beta), 0.0, np.float32), np.full(cpg.shape, ROLES["fixed"]), np.zeros(len(cpg), np.int64))


def eval_clock(task, vocab: Vocab, universe: pd.Index):
    cpg = _pad([_map(vocab, i, universe) for i in task.ids], -1, np.int64)
    clk = np.array([CLOCKS.index(c) if c in CLOCKS else CLOCKS.index({"Horvathv1": "Horvathv1"}.get(c, "Hannum")) for c in task.clock])
    return _tensors(cpg, _pad(list(task.beta), 0.0, np.float32), np.full(cpg.shape, ROLES["clock"]), np.zeros(len(cpg), np.int64), clk)


def eval_pair(task, vocab: Vocab, universe: pd.Index):
    cpg = _pad([_map(vocab, i, universe) for i in task.ids], -1, np.int64)
    role = np.full(cpg.shape, ROLES["pair_a"])
    a = _tensors(cpg, _pad(list(task.beta), 0.0, np.float32), role, np.array([SEX_ID.get(s, 0) for s in task.sex]))
    b = _tensors(cpg, _pad(list(task.beta_b), 0.0, np.float32), role, np.array([SEX_ID.get(s, 0) for s in task.sex_b]))
    return a, b


# ------------------------------------------------------------------------------- metrics
def mae(pred, age) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(age))))


def balanced_accuracy(pred_cls, true_cls, n_classes: int = N_DECADES) -> dict[str, float]:
    pred_cls, true_cls = np.asarray(pred_cls), np.asarray(true_cls)
    present = np.unique(true_cls)
    recalls = [float((pred_cls[true_cls == c] == c).mean()) for c in present]
    return {"balanced_acc_present": float(np.mean(recalls)), "balanced_acc_11": float(np.sum(recalls) / n_classes),
            "accuracy": float((pred_cls == true_cls).mean()), "n_classes_present": int(len(present))}


def decade_posterior(mu_t: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    """P(decade | N(mu_t, sigma_t) in the transformed age space) -> [B, 11]."""
    from scipy.stats import norm

    edges_age = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1e9], dtype=np.float64)
    edges_t = age_to_t(edges_age)
    cdf = norm.cdf((edges_t[None, :] - mu_t[:, None]) / sigma_t[:, None])
    return np.clip(np.diff(cdf, axis=1), 1e-9, 1.0)
