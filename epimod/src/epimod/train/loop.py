"""Training loop: masked multi-task learning (age distribution + sex + masked-CpG reconstruction)."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..age import age_to_t
from ..config import ModelConfig, TrainConfig
from ..data.store import MethylData
from ..model.epimod import EpiMod
from .augment import augment_batch


def split_studies(meta, val_frac: float, seed: int, min_val_n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, val_mask) over rows of ``meta``, split by *study*, never by sample."""
    rng = np.random.RandomState(seed)
    counts = meta.groupby("DatasetID").size()
    eligible = [g for g, n in counts.items() if n >= min_val_n]
    rng.shuffle(eligible)
    target, got, val = val_frac * len(meta), 0, []
    for g in eligible:
        if got >= target:
            break
        val.append(g); got += counts[g]
    is_val = meta["DatasetID"].isin(val).to_numpy()
    return ~is_val, is_val


def age_metrics(pred_age: np.ndarray, age: np.ndarray, sigma_years: np.ndarray | None = None,
                lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> dict[str, float]:
    err = pred_age - age
    out = {
        "n": int(len(age)), "medae": float(np.median(np.abs(err))), "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))), "bias_med": float(np.median(err)),
        "r": float(np.corrcoef(pred_age, age)[0, 1]) if len(age) > 2 else float("nan"),
    }
    if lo is not None and hi is not None:
        out["cover90"] = float(np.mean((age >= lo) & (age <= hi)))
        out["mean_sigma_years"] = float(np.mean(sigma_years))
    return out


@torch.no_grad()
def evaluate_ages(model: nn.Module, data: MethylData, batch_size: int = 128) -> dict[str, float]:
    p = model.predict(data.X.astype(np.float32), batch_size=batch_size)
    return age_metrics(p["age"], data.meta["Age"].to_numpy(), p["sigma_years"], p["age_lo"], p["age_hi"])


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.shadow = decay, copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(b)


def multitask_loss(out, y_t, has_age, sex, has_sex, z_clean, ssl_mask, cfg: TrainConfig):
    parts: dict[str, torch.Tensor] = {}
    resid = (y_t - out.mu_t)[has_age]
    sig = out.sigma_t[has_age]
    parts["nll"] = (0.5 * (resid / sig) ** 2 + torch.log(sig)).mean()
    parts["huber"] = nn.functional.smooth_l1_loss(out.mu_t[has_age], y_t[has_age], beta=0.05)
    if has_sex.any():
        parts["sex"] = nn.functional.binary_cross_entropy_with_logits(out.sex_logit[has_sex], sex[has_sex])
    if out.z_hat is not None and ssl_mask.any():
        parts["ssl"] = ((out.z_hat - z_clean)[ssl_mask] ** 2).mean()
    if cfg.w_assign_entropy > 0 and out.assign is not None:
        A = out.assign
        parts["assign_entropy"] = (-(A * torch.log(A + 1e-12)).sum(-1) / math.log(A.shape[1])).mean()
    total = cfg.w_nll * parts["nll"] + cfg.w_huber * parts["huber"]
    if "assign_entropy" in parts:
        total = total + cfg.w_assign_entropy * parts["assign_entropy"]
    if "sex" in parts:
        total = total + cfg.w_sex * parts["sex"]
    if "ssl" in parts:
        total = total + cfg.w_ssl * parts["ssl"]
    return total, {k: float(v.detach()) for k, v in parts.items()}


def train(model: EpiMod, train_data: MethylData, val_data: MethylData | None, cfg: TrainConfig, out_dir: str | Path,
          log=print) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    X = train_data.X
    y_t, has_age = train_data.tensors()
    sex_f = (train_data.meta["Gender"] == "F").to_numpy().astype(np.float32)
    has_sex = train_data.meta["Gender"].isin(["F", "M"]).to_numpy()
    n_study = train_data.meta.groupby("DatasetID")["Age"].transform("size").to_numpy()
    w_np = n_study.astype(np.float64) ** -cfg.study_balance_power
    if cfg.celltype_balance_power > 0:
        cell = train_data.meta["CellType"].fillna("NA")
        w_np = w_np * cell.groupby(cell).transform("size").to_numpy().astype(np.float64) ** -cfg.celltype_balance_power
    w = torch.as_tensor(w_np)
    template_pool = torch.from_numpy(X[np.random.RandomState(cfg.seed).choice(len(X), min(2048, len(X)), replace=False)].astype(np.float32))

    prior = None
    if model.cfg.linear_skip and cfg.warm_start_linear:
        from ..model.linear_prior import fit_dual_ridge

        with torch.no_grad():
            Z = np.concatenate([model.standardise(torch.from_numpy(X[i : i + 1024].astype(np.float32)))[0].numpy()
                                for i in range(0, len(X), 1024)])
        w0, b0, lam, cv = fit_dual_ridge(Z, y_t, train_data.meta["DatasetID"].to_numpy())
        model.warm_start_linear(w0, b0)
        prior = {"w": w0, "b": b0, "lambda": lam, "cv_median_abs_err_t": cv}
        log(f"[train] ridge prior: lambda={lam:g}; grouped-CV median |err| (t-space) by lambda: "
            + ", ".join(f"{k:g}:{v:.4f}" for k, v in cv.items()))
        del Z

    steps_per_epoch = max(1, len(X) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    decay, no_decay, skip_w = [], [], []
    for n_, p in model.named_parameters():
        if n_ == "pool.s_free":
            skip_w.append(p)
        else:
            (no_decay if p.ndim < 2 or "norm" in n_ or "emb" in n_ else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if skip_w:
        groups.append({"params": skip_w, "weight_decay": 0.0, "lr": cfg.lr * cfg.skip_lr_mult})
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )
    ema = EMA(model, cfg.ema_decay)
    history, best, best_state, t0, step = [], float("inf"), None, time.time(), 0
    for epoch in range(cfg.epochs):
        model.train()
        agg: dict[str, float] = {}
        for _ in range(steps_per_epoch):
            idx = torch.multinomial(w, cfg.batch_size, replacement=True, generator=gen).numpy()
            beta = torch.from_numpy(X[idx].astype(np.float32))
            if cfg.aug:
                beta_in, hide = augment_batch(beta, cfg, gen, template_pool)
            else:
                beta_in, hide = beta, torch.zeros_like(beta, dtype=torch.bool)
            z_clean, _, _ = model.standardise(beta)
            out = model(beta_in, want_recon=bool(hide.any()))
            loss, parts = multitask_loss(
                out, torch.from_numpy(y_t[idx]), torch.from_numpy(has_age[idx]), torch.from_numpy(sex_f[idx]),
                torch.from_numpy(has_sex[idx]), z_clean, hide, cfg,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step(); ema.update(model); step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / steps_per_epoch
        rec = {"epoch": epoch + 1, "sec": round(time.time() - t0, 1), **{f"train_{k}": round(v, 4) for k, v in agg.items()}}
        if val_data is not None:
            vm = evaluate_ages(ema.shadow, val_data)
            rec.update({f"val_{k}": round(v, 4) for k, v in vm.items() if k != "n"})
            if vm["medae"] < best:
                best, best_state = vm["medae"], copy.deepcopy(ema.shadow.state_dict())
        history.append(rec)
        log(json.dumps(rec))
    final = best_state if best_state is not None else ema.shadow.state_dict()
    model.load_state_dict(final)
    return {"history": history, "best_val_medae": best if val_data is not None else None, "ridge_prior": prior}


@torch.no_grad()
def calibrate_sigma(model: EpiMod, data: MethylData, coverage: float = 0.9, batch_size: int = 128) -> float:
    """Rescale the predictive sd so that central ``coverage`` intervals are exact on ``data``
    (held-out *studies*, never the training samples). Returns the scale that was applied."""
    from scipy.stats import norm

    model.eval()
    y_t, has = data.tensors()
    X = torch.from_numpy(data.X.astype(np.float32))
    r = []
    for i in range(0, len(X), batch_size):
        o = model(X[i : i + batch_size])
        r.append(((torch.from_numpy(y_t[i : i + batch_size]) - o.mu_t) / o.sigma_t).abs())
    r = torch.cat(r)[torch.from_numpy(has)]
    scale = float(torch.quantile(r, coverage) / norm.ppf(0.5 + coverage / 2))
    model.sigma_scale.fill_(scale)
    return scale


def save_checkpoint(model: EpiMod, path: str | Path, train_cfg: TrainConfig | None = None, extra: dict | None = None):
    torch.save(
        {"state": model.state_dict(), "model_cfg": model.cfg.to_dict(), "cpg_ids": model.cpg_ids,
         "train_cfg": train_cfg.to_dict() if train_cfg else None, "extra": extra or {}},
        path,
    )


def load_checkpoint(path: str | Path) -> EpiMod:
    from ..data.store import CpGTable

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    N = cfg.n_cpgs
    dummy = CpGTable(
        ids=np.asarray(ck["cpg_ids"]), chrom=np.zeros(N, int), pos=np.zeros(N, np.int64), island=np.zeros(N, int),
        reg=np.zeros(N, int), binary=np.zeros((N, 9), np.float32), mean=np.zeros(N, np.float32),
        std=np.ones(N, np.float32), source=np.array([""] * N), z_age=np.zeros(N, np.float32),
    )
    model = EpiMod(dummy, cfg)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    if unexpected or set(missing) - {"sigma_scale"}:       # sigma_scale post-dates the first checkpoints
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()
