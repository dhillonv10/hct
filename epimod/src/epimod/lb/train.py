"""Training and evaluation of EpiSet on LongevityBench-format inputs generated from real profiles."""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..age import age_to_t, t_to_age
from .episet import EpiSet, SetConfig
from .store import LBStore, sample_pairs
from .tasks import (CLOCKS, Pools, Vocab, balanced_accuracy, clock_batch, decade_of, decade_posterior, extremes_batch, extremes_cache,
                    fixed_batch, mae, pair_batch)

ALL_TASKS = ("extremes", "fixed", "clock", "pair")


@dataclass
class LBTrainConfig:
    tasks: tuple = ALL_TASKS
    steps: int = 3000
    lr: float = 1e-3
    weight_decay: float = 0.02
    warmup: int = 150
    ema: float = 0.995
    bs: dict = field(default_factory=lambda: {"extremes": 32, "fixed": 32, "clock": 32, "pair": 24})
    w_huber: float = 1.0
    w_decade: float = 0.3
    w_pair_age: float = 0.3
    pair_temp: float = 0.25          # divides the t-space age difference
    drop: float = 0.15               # random token dropout on extremes / fixed sets
    study_power: float = 0.5
    age_balance: float = 0.0         # 1 = sample uniformly over 10-year age bins; 0 = natural age distribution
    tail_shift: float = 0.04         # augmentation of the extreme lists' tail values (pipeline differences)
    eval_every: int = 250
    seed: int = 0
    threads: int = 4


@dataclass
class TaskSets:
    fixed_u: np.ndarray                   # universe columns of the fixed 353-CpG set
    clock_u: dict[str, np.ndarray]        # per-clock universe columns
    clock_max: int = 150


def task_sets(parsed: dict, universe: pd.Index) -> TaskSets:
    fixed_ids = list(parsed["LB-0190"].ids[0])
    fixed_u = universe.get_indexer(fixed_ids)
    t = parsed["LB-0195"]
    clock_u = {}
    for c in CLOCKS:
        ids = sorted({i for k, name in enumerate(t.clock) if name == c for i in t.ids[k]})
        if ids:
            clock_u[c] = universe.get_indexer(ids)
    return TaskSets(fixed_u[fixed_u >= 0], {c: u[u >= 0] for c, u in clock_u.items()})


def _loss_single(model, b, age_t, age, cfg):
    mu, sigma, dec = model(b["cpg"], b["beta"], b["role"], b["sex"], b.get("clock"))
    nll = (0.5 * ((age_t - mu) / sigma) ** 2 + torch.log(sigma)).mean()
    hub = nn.functional.smooth_l1_loss(mu, age_t, beta=0.05)
    ce = nn.functional.cross_entropy(dec, torch.as_tensor(decade_of(age)))
    return nll + cfg.w_huber * hub + cfg.w_decade * ce, {"nll": float(nll.detach()), "hub": float(hub.detach()), "ce": float(ce.detach())}


class Trainer:
    def __init__(self, store_tr: LBStore, store_val: LBStore | None, vocab: Vocab, sets: TaskSets, pools: Pools, mcfg: SetConfig,
                 cfg: LBTrainConfig, log=print):
        self.cfg, self.vocab, self.sets, self.pools, self.log = cfg, vocab, sets, pools, log
        self.tr, self.val = store_tr, store_val
        torch.set_num_threads(cfg.threads); torch.manual_seed(cfg.seed)
        self.rng = np.random.RandomState(cfg.seed)
        mcfg = replace(mcfg, n_tokens_vocab=len(vocab))
        self.model = EpiSet(mcfg, vocab.mean, vocab.std)
        self.cache = extremes_cache(store_tr, pools) if "extremes" in cfg.tasks else None
        self.val_cache = extremes_cache(store_val, pools) if (store_val is not None and "extremes" in cfg.tasks) else None
        self.age = store_tr.meta.Age.to_numpy(np.float64)
        n_study = store_tr.meta.groupby("DatasetID")["Age"].transform("size").to_numpy()
        w = n_study.astype(np.float64) ** -cfg.study_power
        if cfg.age_balance > 0:
            bins = np.minimum((self.age // 10).astype(int), 10)
            dens = np.bincount(bins, minlength=11).astype(np.float64)
            w = w * (dens[bins] ** -cfg.age_balance)
        self.p = w / w.sum()
        self.series = store_tr.meta.DatasetID.to_numpy()
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.pair_temp)))
        self.clock_pool = [c for c in CLOCKS if c in sets.clock_u]

    # ----------------------------------------------------------------------------- batches
    def _rows(self, n):
        return self.rng.choice(self.tr.n, n, p=self.p)

    def step_loss(self, kind: str):
        cfg, m = self.cfg, self.model
        n = cfg.bs[kind]
        if kind == "pair":
            ia, ib = sample_pairs(self.age, n, self.rng, 10.0, self.series)
            va, vb = pair_batch(self.tr, self.vocab, ia, ib, 160, self.pools.pair)
            mua, siga, _ = m(va["cpg"], va["beta"], va["role"], va["sex"])
            mub, sigb, _ = m(vb["cpg"], vb["beta"], vb["role"], vb["sex"])
            y = torch.as_tensor((self.age[ia] > self.age[ib]).astype(np.float32))
            tau = self.log_tau.exp()
            bce = nn.functional.binary_cross_entropy_with_logits((mua - mub) / tau, y)
            ta, tb = torch.as_tensor(age_to_t(self.age[ia]), dtype=torch.float32), torch.as_tensor(age_to_t(self.age[ib]), dtype=torch.float32)
            aux = 0.5 * (nn.functional.smooth_l1_loss(mua, ta, beta=0.05) + nn.functional.smooth_l1_loss(mub, tb, beta=0.05))
            return bce + cfg.w_pair_age * aux, {"bce": float(bce.detach()), "aux": float(aux.detach())}
        rows = self._rows(n)
        if kind == "extremes":
            b = extremes_batch(self.tr, self.vocab, self.cache, rows, cfg.drop, self.rng, cfg.tail_shift)
        elif kind == "fixed":
            b = fixed_batch(self.tr, self.vocab, rows, self.sets.fixed_u, cfg.drop, self.rng)
        else:
            names = [self.clock_pool[i] for i in self.rng.randint(0, len(self.clock_pool), n)]
            b = clock_batch(self.tr, self.vocab, rows, self.sets.clock_u, names, self.sets.clock_max, self.rng)
        age = self.age[rows]
        return _loss_single(m, b, torch.as_tensor(age_to_t(age), dtype=torch.float32), age, cfg)

    # ----------------------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self, model: EpiSet, n_val: int = 1500) -> dict[str, float]:
        if self.val is None:
            return {}
        rng = np.random.RandomState(123)
        out = {}
        v = self.val
        age = v.meta.Age.to_numpy(np.float64)
        rows = rng.choice(v.n, min(n_val, v.n), replace=False)
        model.eval()
        if "extremes" in self.cfg.tasks:
            b = extremes_batch(v, self.vocab, self.val_cache, rows)
            pred, _, _ = model.predict_age(b["cpg"], b["beta"], b["role"], b["sex"])
            out["extremes_mae"] = mae(pred, age[rows])
        if "fixed" in self.cfg.tasks:
            b = fixed_batch(v, self.vocab, rows, self.sets.fixed_u)
            pred, _, _ = model.predict_age(b["cpg"], b["beta"], b["role"], b["sex"])
            out["fixed_mae"] = mae(pred, age[rows])
        if "clock" in self.cfg.tasks:
            names = [self.clock_pool[i] for i in rng.randint(0, len(self.clock_pool), len(rows))]
            b = clock_batch(v, self.vocab, rows, self.sets.clock_u, names, self.sets.clock_max, rng)
            pred, _, _ = model.predict_age(b["cpg"], b["beta"], b["role"], b["sex"], b["clock"])
            out["clock_mae"] = mae(pred, age[rows])
        if "pair" in self.cfg.tasks:
            ia, ib = sample_pairs(age, 1000, rng, 10.0, v.meta.DatasetID.to_numpy())
            va, vb = pair_batch(v, self.vocab, ia, ib, 160, self.pools.pair)
            pa, _, _ = model.predict_age(va["cpg"], va["beta"], va["role"], va["sex"])
            pb, _, _ = model.predict_age(vb["cpg"], vb["beta"], vb["role"], vb["sex"])
            out["pair_acc"] = float(np.mean((pa > pb) == (age[ia] > age[ib])))
        return out

    @staticmethod
    def val_score(m: dict[str, float]) -> float:
        """Lower is better: MAEs scaled by the mean-predictor error, pairwise error as is."""
        parts = [m[k] / s for k, s in (("extremes_mae", 16.0), ("fixed_mae", 19.0), ("clock_mae", 13.0)) if k in m]
        if "pair_acc" in m:
            parts.append(1.0 - m["pair_acc"])
        return float(np.mean(parts)) if parts else float("inf")

    # ----------------------------------------------------------------------------- loop
    def fit(self):
        cfg, model = self.cfg, self.model
        params = list(model.parameters()) + [self.log_tau]
        emb = [p for n, p in model.named_parameters() if n.endswith("emb.weight")]
        rest = [p for n, p in model.named_parameters() if not n.endswith("emb.weight")] + [self.log_tau]
        opt = torch.optim.AdamW([{"params": emb, "weight_decay": 0.0}, {"params": rest, "weight_decay": cfg.weight_decay}], lr=cfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        best, best_state, hist, t0 = float("inf"), None, [], time.time()
        for step in range(1, cfg.steps + 1):
            model.train()
            total, parts = 0.0, {}
            loss = torch.zeros(())
            for kind in cfg.tasks:
                l, d = self.step_loss(kind)
                loss = loss + l; parts.update({f"{kind}_{k}": v for k, v in d.items()})
            if model.cfg.moe_experts > 0:
                loss = loss + model.cfg.moe_balance * model.moe_aux()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for s, p in zip(ema.parameters(), model.parameters()):
                    s.mul_(cfg.ema).add_(p.detach(), alpha=1 - cfg.ema)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                val = self.validate(ema)
                sc = self.val_score(val)
                rec = {"step": step, "sec": round(time.time() - t0), "loss": round(float(loss), 4), "val_score": round(sc, 4),
                       **{k: round(v, 4) for k, v in val.items()}}
                if model.cfg.moe_experts > 0:
                    rec["moe_aux"] = round(float(model.moe_aux()), 3)
                hist.append(rec); self.log(json.dumps(rec))
                if sc < best:
                    best, best_state = sc, copy.deepcopy(ema.state_dict())
        ema.load_state_dict(best_state if best_state is not None else ema.state_dict())
        self.model = ema
        return {"history": hist, "best_val_score": best}
