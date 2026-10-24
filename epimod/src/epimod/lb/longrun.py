"""Long, unattended training: run while validation keeps improving, and never silently go bad.

What protects a multi-day run
  * plateau control   LR is held after a short warm-up; when the best validation score improves by < ``min_rel`` over
                      ``patience`` evaluations the LR is halved and the best weights restored; after ``max_lr_drops`` such
                      drops the run is declared converged and stops.  No fixed step budget, no cosine schedule.
  * step guard        a step with a non-finite loss / gradient norm, or a loss far above its recent level, is skipped;
                      too many in a row triggers a rollback to the last good checkpoint with a halved LR.
  * divergence        a validation score that is non-finite or far worse than the best triggers the same rollback.
  * checkpoints       atomic writes, one generation kept (``state_prev.pt``); the best EMA weights are kept separately.
  * device drift      every ``cpu_check_every`` evaluations the validation score is recomputed on the CPU; a mismatch with the
                      accelerator beyond ``cpu_check_tol`` is logged, beyond ``cpu_check_fail`` the run stops.
  * operator controls ``STOP`` file in the run directory ends the run cleanly; ``status.json`` shows progress at any time.
"""
from __future__ import annotations

import collections
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .train import Trainer


@dataclass
class LongConfig:
    eval_every: int = 1000
    warmup: int = 500
    lr: float = 3e-4
    ema: float = 0.999
    patience: int = 6              # evaluations
    min_rel: float = 0.005         # required relative improvement of the best score over one patience window
    max_lr_drops: int = 3
    lr_drop_factor: float = 0.5
    diverge_factor: float = 1.3    # validation score this much worse than the best -> rollback
    spike_k: float = 8.0           # loss spike threshold in robust standard deviations
    max_consecutive_bad: int = 25
    max_rollbacks_in_a_row: int = 3
    max_steps: int = 10**9
    max_hours: float = 0.0         # 0 = unlimited
    cpu_check_every: int = 25      # evaluations; 0 = never
    cpu_check_tol: float = 0.03
    cpu_check_fail: float = 0.10


class PlateauController:
    """Decides, from the sequence of validation scores (lower is better), whether to continue, drop the LR, roll back or stop."""

    def __init__(self, patience: int, min_rel: float, max_drops: int, diverge_factor: float):
        self.patience, self.min_rel, self.max_drops, self.diverge = patience, min_rel, max_drops, diverge_factor
        self.best_hist: list[float] = []
        self.drops = 0

    def update(self, score: float) -> str:
        """-> 'bad' | 'diverged' | 'improved' | 'same' | 'drop' | 'converged'."""
        if not math.isfinite(score):
            return "bad"
        if self.best_hist and score > self.best_hist[-1] * self.diverge:
            return "diverged"
        new_best = not self.best_hist or score < self.best_hist[-1]
        self.best_hist.append(min(score, self.best_hist[-1]) if self.best_hist else score)
        if len(self.best_hist) > self.patience and self.best_hist[-1] > self.best_hist[-1 - self.patience] * (1 - self.min_rel):
            if self.drops >= self.max_drops:
                return "converged"
            self.drops += 1
            self.best_hist = self.best_hist[-1:]                      # a new window starts after every drop
            return "drop"
        return "improved" if new_best else "same"

    def drop_now(self) -> bool:
        """Count a drop forced by divergence; False when the budget of drops is already used up."""
        if self.drops >= self.max_drops:
            return False
        self.drops += 1; self.best_hist = self.best_hist[-1:]
        return True

    def state(self) -> dict:
        return {"best_hist": self.best_hist, "drops": self.drops}

    def load(self, s: dict):
        self.best_hist, self.drops = list(s["best_hist"]), int(s["drops"])


class StepGuard:
    """Per-step sanity: skip non-finite or spiking steps; ask for a rollback after too many in a row."""

    def __init__(self, spike_k: float = 8.0, window: int = 200, min_hist: int = 50, max_consecutive: int = 25):
        self.hist = collections.deque(maxlen=window)
        self.spike_k, self.min_hist, self.max_consecutive = spike_k, min_hist, max_consecutive
        self.consecutive = 0
        self.total_bad = 0

    def check(self, loss: float, grad_norm: float | None = None) -> str:
        """-> 'ok' | 'skip' | 'rollback'."""
        bad = not math.isfinite(loss) or (grad_norm is not None and not math.isfinite(grad_norm))
        if not bad and len(self.hist) >= self.min_hist:
            h = np.asarray(self.hist); med = float(np.median(h)); mad = float(np.median(np.abs(h - med))) * 1.4826
            bad = loss > med + self.spike_k * max(mad, 0.05) + 1.0
        if bad:
            self.consecutive += 1; self.total_bad += 1
            return "rollback" if self.consecutive >= self.max_consecutive else "skip"
        self.consecutive = 0; self.hist.append(loss)
        return "ok"


class LongTrainer(Trainer):
    def __init__(self, store_tr, store_val, vocab, sets, pools, mcfg, cfg, long: LongConfig, run_dir, log=print, meta: dict | None = None):
        super().__init__(store_tr, store_val, vocab, sets, pools, mcfg, cfg, log=log, ckpt_path=None)
        self.lc, self.run_dir, self.meta = long, Path(run_dir), meta or {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------------------- persistence
    def _save(self, st: dict, ema, opt, ctrl, guard, best_state, best_score):
        payload = {"step": st["step"], "lr": st["lr"], "model": self.model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                   "log_tau": self.log_tau.detach(), "ctrl": ctrl.state(), "best_state": best_state, "best_score": best_score,
                   "hist": st["hist"], "rng": self.rng.get_state(), "total_bad": guard.total_bad, "rollbacks": st["rollbacks"]}
        p, prev, tmp = self.run_dir / "state.pt", self.run_dir / "state_prev.pt", self.run_dir / "state.tmp"
        torch.save(payload, tmp)
        if p.exists():
            os.replace(p, prev)
        os.replace(tmp, p)

    def _load(self):
        for name in ("state.pt", "state_prev.pt"):
            p = self.run_dir / name
            if p.exists():
                try:
                    return torch.load(p, map_location="cpu", weights_only=False)
                except Exception as e:                                   # truncated / corrupt file: fall back one generation
                    self.log(f"[warn] could not read {name}: {e}")
        return None

    def _write_status(self, st: dict, extra: dict):
        d = {"step": st["step"], "lr": st["lr"], "elapsed_h": round((time.time() - st["t0"]) / 3600, 3), "device": str(self.dev), **extra}
        tmp = self.run_dir / "status.tmp"; tmp.write_text(json.dumps(d, indent=1)); os.replace(tmp, self.run_dir / "status.json")

    def _export(self, best_state):
        """model.pt in the format every evaluation script loads (weights on CPU)."""
        sd = {k: v.detach().cpu() for k, v in best_state.items()}
        torch.save({"state": sd, "cfg": asdict(self.model.cfg), "vocab_ids": list(self.vocab.cpg_ids), "args": self.meta.get("args", {})}, self.run_dir / "model.pt")

    # -------------------------------------------------------------------------------------- training
    def fit_long(self, fault_hook=None) -> dict:
        lc, cfg, model = self.lc, self.cfg, self.model
        params = list(model.parameters()) + [self.log_tau]
        emb = [p for n, p in model.named_parameters() if n.endswith("emb.weight")]
        rest = [p for n, p in model.named_parameters() if not n.endswith("emb.weight")] + [self.log_tau]
        opt = torch.optim.AdamW([{"params": emb, "weight_decay": 0.0}, {"params": rest, "weight_decay": cfg.weight_decay}], lr=lc.lr)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        ctrl = PlateauController(lc.patience, lc.min_rel, lc.max_lr_drops, lc.diverge_factor)
        guard = StepGuard(lc.spike_k, max_consecutive=lc.max_consecutive_bad)
        st = {"step": 0, "lr": lc.lr, "hist": [], "t0": time.time(), "rollbacks": 0}
        best_state, best_score = None, float("inf")

        def restore(ck, lr_scale: float = 1.0, weights_from_best: bool = False):
            nonlocal best_state, best_score
            src = ck["best_state"] if (weights_from_best and ck.get("best_state") is not None) else None
            model.load_state_dict(src if src is not None else ck["model"]); ema.load_state_dict(src if src is not None else ck["ema"])
            opt.load_state_dict(ck["opt"]); self.log_tau.data.copy_(ck["log_tau"].to(self.dev))
            st["lr"] = ck["lr"] * lr_scale
            best_state, best_score = ck["best_state"], ck["best_score"]

        ck = self._load()
        if ck is not None:
            restore(ck)
            st.update(step=ck["step"], hist=ck["hist"], rollbacks=ck.get("rollbacks", 0)); ctrl.load(ck["ctrl"]); guard.total_bad = ck.get("total_bad", 0)
            self.rng.set_state(ck["rng"])
            self.log(f"[resume] continuing from step {st['step']} (best {best_score:.4f}, lr {st['lr']:.2e}, {ctrl.drops} lr drops)")
        t_run0, step_t0, steps_at_t0 = time.time(), time.time(), st["step"]
        reason = None
        while reason is None:
            if (self.run_dir / "STOP").exists():
                reason = "stop-file"; break
            if lc.max_hours and (time.time() - t_run0) / 3600 > lc.max_hours:
                reason = "max-hours"; break
            if st["step"] >= lc.max_steps:
                reason = "max-steps"; break
            st["step"] += 1; step = st["step"]
            lr_eff = st["lr"] * min(1.0, step / lc.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_eff
            model.train()
            loss = torch.zeros((), device=self.dev)
            for kind in cfg.tasks:
                l, _ = self.step_loss(kind); loss = loss + l
            if model.cfg.moe_experts > 0:
                loss = loss + model.cfg.moe_balance * model.moe_aux().to(self.dev)
            if fault_hook is not None:
                loss = fault_hook(step, loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = float(nn.utils.clip_grad_norm_(params, 1.0)); lv = float(loss.detach())
            verdict = guard.check(lv, gn)
            if verdict == "ok":
                opt.step()
                d = min(lc.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for s_, p_ in zip(ema.parameters(), model.parameters()):
                        s_.mul_(d).add_(p_.detach(), alpha=1 - d)
            else:
                opt.zero_grad(set_to_none=True)
                if verdict == "rollback":
                    ck = self._load()
                    st["rollbacks"] += 1
                    if ck is None or st["rollbacks"] > lc.max_rollbacks_in_a_row or not ctrl.drop_now():
                        reason = "unstable (repeated bad steps)"; break
                    self.log(f"[rollback] {guard.consecutive} bad steps in a row at step {step}: restoring step {ck['step']}, lr x{lc.lr_drop_factor}")
                    restore(ck, lr_scale=lc.lr_drop_factor); st["step"] = ck["step"]; guard.consecutive = 0
                else:
                    st["step"] -= 1                                  # a skipped step does not consume a step number
                continue
            if step % lc.eval_every:
                continue
            # ------------------------------------------------------------------ evaluation
            val = self.validate(ema); sc = self.val_score(val)
            action = ctrl.update(sc)
            rec = {"step": step, "sec": round(time.time() - st["t0"]), "loss": round(lv, 4), "val_score": round(sc, 4) if math.isfinite(sc) else None,
                   "lr": st["lr"], "action": action, "bad_steps": guard.total_bad, **{k: round(v, 4) for k, v in val.items()}}
            sps = (step - steps_at_t0) / max(time.time() - step_t0, 1e-9); rec["steps_per_sec"] = round(sps, 3)
            if math.isfinite(sc) and sc < best_score:
                best_score, best_state = sc, copy.deepcopy(ema.state_dict())
            st["hist"].append(rec); self.log(json.dumps(rec))
            if lc.cpu_check_every and self.dev.type != "cpu" and len(st["hist"]) % lc.cpu_check_every == 0 and math.isfinite(sc):
                sc_cpu = self.val_score(self.validate(copy.deepcopy(ema).to("cpu"))); rel = abs(sc - sc_cpu) / max(abs(sc_cpu), 1e-9)
                self.log(f"[cpu-check] {self.dev} {sc:.4f} vs cpu {sc_cpu:.4f} (rel diff {rel:.3%})")
                if rel > lc.cpu_check_fail:
                    reason = f"device mismatch {rel:.1%} (accelerator vs CPU validation)"
                elif rel > lc.cpu_check_tol:
                    self.log("[warn] accelerator and CPU validation scores differ by more than the tolerance")
            if action in ("bad", "diverged"):
                ck = self._load(); st["rollbacks"] += 1
                if ck is None or st["rollbacks"] > lc.max_rollbacks_in_a_row or not ctrl.drop_now():
                    reason = reason or f"unstable validation ({action})"
                else:
                    self.log(f"[rollback] validation {action} at step {step}: restoring step {ck['step']}, lr x{lc.lr_drop_factor}")
                    restore(ck, lr_scale=lc.lr_drop_factor, weights_from_best=True); st["step"] = ck["step"]
                self._write_status(st, {"best_score": best_score, "last_action": action}); continue
            st["rollbacks"] = 0
            if action == "drop" and best_state is not None:
                model.load_state_dict(best_state); ema.load_state_dict(best_state)
                st["lr"] *= lc.lr_drop_factor; self.log(f"[plateau] no {lc.min_rel:.1%} improvement in {lc.patience} evals: lr -> {st['lr']:.2e} (drop {ctrl.drops}/{lc.max_lr_drops}), best weights restored")
            if action == "converged":
                reason = f"converged (no {lc.min_rel:.1%} improvement in {lc.patience} evals after {lc.max_lr_drops} lr drops)"
            self._save(st, ema, opt, ctrl, guard, best_state, best_score)
            if best_state is not None:
                self._export(best_state)
            self._write_status(st, {"best_score": best_score, "last_score": sc, "last_action": action, "lr_drops": ctrl.drops, "steps_per_sec": round(sps, 3),
                                    "bad_steps": guard.total_bad, "state": "running"})
            if self.dev.type == "mps":
                torch.mps.empty_cache()
            step_t0, steps_at_t0 = time.time(), step
        if best_state is not None:
            self._export(best_state)
        if reason and not reason.startswith("unstable") and st["step"] > 0:      # never overwrite the last good state with a bad one
            self._save(st, ema, opt, ctrl, guard, best_state, best_score)
        summary = {"reason": reason, "step": st["step"], "best_val_score": best_score, "lr_drops": ctrl.drops, "bad_steps": guard.total_bad,
                   "elapsed_h": round((time.time() - st["t0"]) / 3600, 3)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        self._write_status(st, {**summary, "state": f"finished: {reason}"})
        self.log(f"[done] {summary}")
        return {**summary, "history": st["hist"]}


# ---------------------------------------------------------------------------------------------- run setup / export helpers
PRESETS = {
    "base": dict(d=128, d_emb=64, heads=4, local=2, glob=4, lr=5e-4),        # ~3M parameters
    "large": dict(d=256, d_emb=128, heads=8, local=3, glob=6, lr=3e-4),      # ~25M parameters
}
TASKS = "extremes,fixed,clock,pair,mask"


def build_inputs(a, device: str):
    """Everything a run needs from the shared command-line namespace ``a`` (see scripts/train_mps.py)."""
    from ..data.annotation import load_probe_annotation  # noqa: F401  (loaded by the trainer)
    from ..train.loop import split_studies
    from .epitraj import HierConfig
    from .episet import SetConfig
    from .store import lb_meta, lb_series, load_lb_store
    from .tasks import build_vocab, load_parsed, make_pools
    from .train import LBTrainConfig, task_sets

    parsed = load_parsed(); excl = lb_series(parsed)
    meta_all = lb_meta(excl); tr_m, va_m = split_studies(meta_all, 0.15, a.seed)
    tr_series = set(meta_all[tr_m].DatasetID)
    if a.unlabeled:
        full = lb_meta(excl, unlabeled=True); tr_series |= set(full[full.Age.isna()].DatasetID)
    tr = load_lb_store(exclude_series=excl, include_series=tr_series, unlabeled=a.unlabeled)
    va = load_lb_store(exclude_series=excl, include_series=set(meta_all[va_m].DatasetID))
    odd = np.arange(500) % 2 == 1
    pools = make_pools(parsed, tr.universe, ext_mask=odd, pair_mask=odd)
    vocab = build_vocab(tr, parsed, pools); sets = task_sets(parsed, tr.universe)
    p = PRESETS[a.preset]
    if a.model != "hier":
        a.tasks = ",".join(t for t in a.tasks.split(",") if t != "mask")            # masked reconstruction needs the hierarchical encoder
    if a.model == "hier":
        mcfg = HierConfig(d_model=p["d"], d_emb=p["d_emb"], n_heads=p["heads"], local_layers=p["local"], global_layers=p["glob"])
    else:
        mcfg = SetConfig(d_model=p["d"], d_emb=p["d_emb"], n_heads=p["heads"], n_layers=p["local"] + p["glob"])
    base_bs = {"extremes": 32, "fixed": 8, "clock": 32, "pair": 24, "mask": 16}
    cfg = LBTrainConfig(tasks=tuple(a.tasks.split(",")), bs={k: max(1, int(round(v * a.bs_mult))) for k, v in base_bs.items()},
                        age_balance=1.0, seed=a.seed, threads=max(1, (os.cpu_count() or 4) // 2), device=device)
    return dict(tr=tr, va=va, vocab=vocab, sets=sets, pools=pools, mcfg=mcfg, cfg=cfg, lr=a.lr or p["lr"])


def load_exported(path):
    """Rebuild the model stored in an exported ``model.pt`` (on the CPU)."""
    from ..data.annotation import load_probe_annotation
    from .episet import EpiSet, SetConfig
    from .epitraj import EpiHier, HierConfig, block_assignment
    import pandas as pd
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["args"].get("model") == "hier":
        hc = HierConfig(**ck["cfg"]); m = EpiHier(hc, block_assignment(pd.Index(ck["vocab_ids"]), load_probe_annotation(), hc.block_size))
    else:
        m = EpiSet(SetConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"]); return m.eval()
