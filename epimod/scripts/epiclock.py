#!/usr/bin/env python3
"""EpiClock - a genome-aware transformer epigenetic clock, in one file.

  python epiclock.py train   --data <bundle>/data --out runs/clock1 --preset base      # pretrain -> fine-tune -> test report
  python epiclock.py predict --model runs/clock1/model.pt --beta my_betas.csv --out ages.csv
  python epiclock.py probe   --model runs/clock1/model.pt --data <bundle>/data         # second-level tasks on the learned model

What it learns
  Every CpG is a token: learned identity + genome context (chromosome, position at several scales, CpG-island / shore / shelf /
  open sea, gene region, regulatory flags) + its measured methylation (beta).  A sample is a random subset of its observed CpGs
  (so any array, any missing values).  Stage 1 (pre-training, no labels): hide 25% of the CpGs and reconstruct them from the
  visible ones -- the transformer must learn co-methylation, island/shore structure and the genome's layout.  Stage 2: the same
  network predicts age (Horvath-transformed) from a CpG subset; at test time predictions over several disjoint subsets are averaged.
  A per-CpG linear term (an unbiased estimate of the full-panel sum) lets it start at least as good as a linear clock.

Honest evaluation
  Test = the ComputAge benchmark studies (never used for training or model selection), healthy controls only, compared on the same
  samples with ridge and elastic-net trained on the same data and, if `pip install computage --no-deps` is available, with the
  published Horvath V1/V2, Hannum and Lin clocks (fully covered by the panel).  Early stopping uses held-out *series*.

Device: --device auto uses the Apple GPU (MPS) when present.  On a non-CPU device a self-test first compares the device with the CPU
(loss, gradients, a 60-step overfit) and stops if they disagree.  Training stops by itself when validation stops improving."""
from __future__ import annotations

import argparse
import collections
import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")           # must precede the torch import

import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore")

PRESETS = {                                                          # d_model, layers, heads, tokens per sample, batch, lr
    "tiny": dict(d=96, layers=3, heads=4, K=256, batch=64, lr=6e-4, dropout=0.0),
    "small": dict(d=128, layers=4, heads=4, K=384, batch=64, lr=5e-4, dropout=0.0),
    "base": dict(d=192, layers=6, heads=6, K=512, batch=128, lr=4e-4, dropout=0.1),
    "large": dict(d=256, layers=8, heads=8, K=768, batch=128, lr=3e-4, dropout=0.1),
}
ADULT = 20.0
LOGF = None


def log(*a):
    line = time.strftime("%F %T ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    if LOGF is not None:
        with open(LOGF, "a") as f:
            f.write(line + "\n")


def age_to_t(age):
    age = np.asarray(age, np.float64)
    return np.where(age <= ADULT, np.log(np.clip(age, 0, None) + 1) - np.log(ADULT + 1), (age - ADULT) / (ADULT + 1))


def t_to_age(t):
    t = np.asarray(t, np.float64)
    return np.where(t <= 0, (ADULT + 1) * np.exp(np.minimum(t, 0)) - 1, (ADULT + 1) * t + ADULT)


# ======================================================================================================== data
class Corpus:
    """Compact methylation corpus (uint8 beta*100, 255 = missing) over a fixed CpG panel, read from the offline bundle."""
    pass


def load_corpus(data: Path) -> Corpus:
    cache = data / "epiclock_corpus.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        c = Corpus(); c.cpgs = list(z["cpgs"]); c.X = z["X"]; c.meta = pd.DataFrame({k: z["m_" + k] for k in json.loads(str(z["cols"]))}, index=z["index"].astype(str))
        return c
    cpgs = pd.read_parquet(data / "lb/lite_universe.parquet")["cpg"].astype(str).tolist()
    def computage(split, fn):
        m = pd.read_csv(data / "meta" / fn, sep="\t", index_col=0); m["Age"] = pd.to_numeric(m["Age"], errors="coerce")
        m = m[m.Age.between(0, 105)]; m = m[~m.index.duplicated(keep="first")]; m["role"] = split; return m
    hold = set((data / "geo/holdout.txt").read_text().split()) if (data / "geo/holdout.txt").exists() else set()   # leukaemia/AML/MDS series
    def geo(sub, role):
        fs = sorted((data / "geo" / sub).glob("GSE*.parquet"))
        m = pd.concat([pd.read_parquet(f) for f in fs]) if fs else pd.DataFrame()
        if len(m):
            m = m[~m.DatasetID.isin(hold)]; m["role"] = role
        return m
    parts = [computage("train", "computage_train_meta.tsv"), computage("benchmark", "computage_bench_meta.tsv"), geo("meta", "geo"), geo("meta_unlabeled", "unlabeled")]
    cols = ["DatasetID", "Gender", "Age", "Condition", "Class", "Tissue", "Platform", "role"]
    meta = pd.concat([p.reindex(columns=cols) for p in parts if len(p)])
    meta = meta[~meta.index.duplicated(keep="first")]                              # ComputAge first, then labelled GEO, then unlabelled GEO
    ds = meta.DatasetID.to_dict(); blocks, ids = [], []
    for sub in ("train", "benchmark", "geo", "geo_unlabeled"):
        for f in sorted((data / "lb/store_lite" / sub).glob("GSE*.npz")):
            z = np.load(f, allow_pickle=False); s = z["samples"].astype(str)
            keep = [i for i, x in enumerate(s) if ds.get(x) == f.stem]               # each sample belongs to exactly one study
            if keep:
                blocks.append(z["prof"][keep]); ids += [s[i] for i in keep]
    c = Corpus(); c.cpgs = cpgs; c.X = np.concatenate(blocks); c.meta = meta.loc[ids]
    np.savez(cache, cpgs=np.array(cpgs), X=c.X, index=np.array(ids), cols=json.dumps(cols), **{"m_" + k: c.meta[k].to_numpy() for k in cols})
    return c


def make_splits(c: Corpus, seed: int, val_frac: float = 0.12):
    m = c.meta; lab = m.role.isin(["train", "geo"]).to_numpy() & m.Age.notna().to_numpy()
    rng = np.random.RandomState(seed); counts = m[lab].groupby("DatasetID").size()
    elig = [g for g, n in counts.items() if n >= 40]; rng.shuffle(elig)
    val_series, got = [], 0
    for g in elig:
        if got >= val_frac * lab.sum():
            break
        val_series.append(g); got += counts[g]
    is_val = m.DatasetID.isin(val_series).to_numpy() & lab
    ssl_ok = (lab & ~is_val) | (m.role == "unlabeled").to_numpy()
    test = (m.role == "benchmark").to_numpy() & (m.Condition == "HC").to_numpy()
    return dict(train=np.flatnonzero(lab & ~is_val), val=np.flatnonzero(is_val), ssl=np.flatnonzero(ssl_ok), test=np.flatnonzero(test),
                test_all=np.flatnonzero((m.role == "benchmark").to_numpy()), val_series=val_series)


def panel_stats(X, idx):
    sub = X[idx].astype(np.float32); miss = sub == 255; sub[miss] = np.nan; sub /= 100.0
    mean = np.nanmean(sub, 0); std = np.nanstd(sub, 0)
    return np.where(np.isfinite(mean), mean, 0.5).astype(np.float32), np.maximum(np.where(np.isfinite(std), std, 0.25), 0.02).astype(np.float32)


def genome_context(data: Path, cpgs):
    """Static per-CpG context: chromosome, island relation, regulatory class (categorical) + flags, gene regions, position features."""
    ann = pd.read_parquet(data / "annot/annotation_450k.parquet").reindex(cpgs)
    flags = ["enhancer", "dhs", "design2", "snp", "gene_TSS1500", "gene_TSS200", "gene_5'UTR", "gene_1stExon", "gene_Body", "gene_3'UTR"]
    pos = ann.pos.fillna(1).to_numpy(np.float64)
    four = np.concatenate([[np.sin(2 * np.pi * pos / P), np.cos(2 * np.pi * pos / P)] for P in (1e3, 1e4, 1e5, 1e6, 1e7)], 0).T
    cont = np.concatenate([ann[flags].fillna(0).to_numpy(np.float64), four, ((np.log10(pos) - 7.0) / 1.0)[:, None]], 1).astype(np.float32)
    return dict(chrom=ann.chrom.fillna(0).astype(int).clip(0, 23).to_numpy(), island=ann.island.fillna(0).astype(int).clip(0, 5).to_numpy(),
                reg=ann.reg.fillna(0).astype(int).clip(0, 8).to_numpy(), cont=cont)


# ======================================================================================================== model
class Block(nn.Module):
    """Pre-norm transformer block.  No dropout on the attention weights (generating that mask dominated the run time on CPUs)."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d)); self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0])
        return x + self.drop(self.ff(self.n2(x)))


class EpiClock(nn.Module):
    def __init__(self, n_cpg, mean, std, ctx, cfg):
        super().__init__()
        d = cfg["d"]; self.cfg = cfg; self.n = n_cpg
        self.register_buffer("mean", torch.as_tensor(mean)); self.register_buffer("std", torch.as_tensor(std))
        self.id_emb = nn.Embedding(n_cpg, d); self.slope = nn.Embedding(n_cpg, d)
        nn.init.normal_(self.id_emb.weight, std=0.1); nn.init.normal_(self.slope.weight, std=0.1)
        self.use_ctx = ctx is not None and cfg.get("context", True)
        if self.use_ctx:
            for k in ("chrom", "island", "reg"):
                self.register_buffer("c_" + k, torch.as_tensor(ctx[k]))
            self.register_buffer("c_cont", torch.as_tensor(ctx["cont"]))
            self.chrom_emb, self.island_emb, self.reg_emb = nn.Embedding(24, d), nn.Embedding(6, d), nn.Embedding(9, d)
            self.cont_mlp = nn.Sequential(nn.Linear(ctx["cont"].shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(6))
        self.val_mlp = nn.Sequential(nn.Linear(13, d), nn.GELU(), nn.Linear(d, d))
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02); self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["heads"], cfg.get("dropout", 0.0)) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(d); self.age_head = nn.Linear(d, 1)
        self.lin_w = nn.Embedding(n_cpg, 1); nn.init.zeros_(self.lin_w.weight)
        self.lin_on = cfg.get("linear_skip", True); self.bias = nn.Parameter(torch.zeros(()))
        self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def context(self, idx):
        if not self.use_ctx:
            return 0.0
        return self.chrom_emb(self.c_chrom[idx]) + self.island_emb(self.c_island[idx]) + self.reg_emb(self.c_reg[idx]) + self.cont_mlp(self.c_cont[idx])

    def tokens(self, idx, beta, is_mask):
        z = ((beta - self.mean[idx]) / self.std[idx]).clamp(-6, 6)
        ang = beta.unsqueeze(-1) * self.freqs
        val = self.val_mlp(torch.cat([torch.sin(ang), torch.cos(ang), beta.unsqueeze(-1)], -1)) + self.slope(idx) * z.unsqueeze(-1)
        m = is_mask.unsqueeze(-1).to(val.dtype)
        return self.id_emb(idx) + self.context(idx) + val * (1 - m) + self.mask_emb * m, z

    def encode(self, idx, beta, valid, is_mask=None):
        dev = self.cls.device
        idx, beta, valid = idx.to(dev), beta.to(dev), valid.to(dev)
        is_mask = torch.zeros_like(valid) if is_mask is None else is_mask.to(dev)
        x, z = self.tokens(idx, beta, is_mask)
        x = torch.cat([self.cls.expand(len(x), 1, -1), x], 1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=dev), ~valid], 1)
        for blk in self.blocks:
            x = blk(x, pad)
        return self.norm(x), z

    def age_t(self, idx, beta, valid, n_obs):
        """Age in the transformed space: head on [CLS] + unbiased estimate of the full-panel linear term."""
        valid = valid.to(self.cls.device); h, z = self.encode(idx, beta, valid)
        mu = self.age_head(h[:, 0]).squeeze(-1) + self.bias
        if self.lin_on:
            w = self.lin_w(idx.to(z.device)).squeeze(-1)
            nv = valid.sum(1).clamp(min=1).float()
            mu = mu + (w * z * valid.float()).sum(1) * (n_obs.to(z.device).float() / nv)
        return mu

    def reconstruct(self, idx, beta, valid, is_mask):
        h, _ = self.encode(idx, beta, valid, is_mask)
        return torch.sigmoid(self.recon(h[:, 1:]).squeeze(-1))

    def embed(self, idx, beta, valid):
        h, _ = self.encode(idx, beta, valid)
        v = valid.to(h.device).unsqueeze(-1).float()
        return torch.cat([h[:, 0], (h[:, 1:] * v).sum(1) / v.sum(1).clamp(min=1)], -1)


# ======================================================================================================== batches
def draw(X_rows, K, rng, order=None):
    """K random observed CpGs per row -> (idx, beta, valid) as torch tensors; rows with fewer than K observed are padded."""
    obs = X_rows != 255
    keys = rng.random(X_rows.shape, dtype=np.float32) if order is None else order
    keys = np.where(obs, keys, 2.0)
    idx = np.argpartition(keys, K - 1, axis=1)[:, :K] if X_rows.shape[1] > K else np.tile(np.arange(X_rows.shape[1]), (len(X_rows), 1))
    idx = rng.permuted(idx, axis=1)
    valid = np.take_along_axis(obs, idx, 1)
    beta = np.where(valid, np.take_along_axis(X_rows, idx, 1) / 100.0, 0.5).astype(np.float32)
    return torch.as_tensor(idx, dtype=torch.long), torch.as_tensor(beta), torch.as_tensor(valid), torch.as_tensor(obs.sum(1))


def ssl_batch(X_rows, K, rng, mask_frac=0.25):
    idx, beta, valid, _ = draw(X_rows, K, rng)
    nt = int(K * mask_frac); is_mask = torch.zeros_like(valid); is_mask[:, :nt] = True
    inp_beta = torch.where(is_mask, torch.full_like(beta, 0.5), beta)
    return idx, inp_beta, valid, is_mask, beta


@torch.no_grad()
def predict_t(model, X_rows, K, R, rng, batch=256):
    """Average over R disjoint random CpG subsets of size K (all observed CpGs when R*K covers them)."""
    model.eval(); out = np.zeros(len(X_rows)); dev = model.cls.device
    for s in range(0, len(X_rows), batch):
        rows = X_rows[s:s + batch]; obs = rows != 255
        order = rng.random(rows.shape, dtype=np.float32); order = np.where(obs, order, 2.0)
        rank = np.argsort(order, axis=1)                       # observed CpGs first, in random order
        acc = np.zeros(len(rows)); cnt = np.zeros(len(rows))
        for r in range(R):
            sl = rank[:, r * K:(r + 1) * K]
            if sl.shape[1] < K:
                break
            valid = np.take_along_axis(obs, sl, 1)
            if r > 0 and not valid.any():
                break
            beta = np.where(valid, np.take_along_axis(rows, sl, 1) / 100.0, 0.5).astype(np.float32)
            mu = model.age_t(torch.as_tensor(sl, dtype=torch.long), torch.as_tensor(beta), torch.as_tensor(valid), torch.as_tensor(obs.sum(1))).cpu().double().numpy()
            w = (valid.sum(1) > K // 4).astype(float) if r > 0 else np.ones(len(rows))
            acc += mu * w; cnt += w
        out[s:s + batch] = acc / np.maximum(cnt, 1)
    return out


# ======================================================================================================== training machinery
class Plateau:
    def __init__(self, patience=6, min_rel=0.005, max_drops=3):
        self.patience, self.min_rel, self.max_drops = patience, min_rel, max_drops; self.hist, self.drops = [], 0

    def update(self, score):
        if not math.isfinite(score):
            return "bad"
        self.hist.append(min(score, self.hist[-1]) if self.hist else score)
        if len(self.hist) > self.patience and self.hist[-1] > self.hist[-1 - self.patience] * (1 - self.min_rel):
            if self.drops >= self.max_drops:
                return "converged"
            self.drops += 1; self.hist = self.hist[-1:]
            return "drop"
        return "improved" if len(self.hist) < 2 or score < self.hist[-2] else "same"


def run_phase(name, model, step_fn, eval_fn, out: Path, dev, lr, wd=0.05, warmup=300, ema_decay=0.999, max_steps=10**9, eval_every=500,
              patience=6, min_rel=0.005, max_drops=3, max_hours=0.0, init_from=None):
    """Train until the validation score (lower = better) stops improving.  Resumable; atomic checkpoints; non-finite steps skipped."""
    state_p, best_p, stop_p = out / f"{name}_state.pt", out / f"{name}_best.pt", out / "STOP"
    emb = [p for n, p in model.named_parameters() if "emb" in n or n in ("mask_emb", "cls")]
    rest = [p for n, p in model.named_parameters() if not ("emb" in n or n in ("mask_emb", "cls"))]
    opt = torch.optim.AdamW([{"params": emb, "weight_decay": 0.0}, {"params": rest, "weight_decay": wd}], lr=lr)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ctrl = Plateau(patience, min_rel, max_drops)
    st = dict(step=0, lr=lr, best=float("inf"), best_state=None, hist=[])
    if init_from is not None and not state_p.exists():
        sd = torch.load(init_from, map_location="cpu", weights_only=False)["state"]
        miss = model.load_state_dict(sd, strict=False); ema.load_state_dict(model.state_dict()); log(f"[{name}] initialised from {init_from} ({len(miss.missing_keys)} new tensors)")
    if state_p.exists():
        ck = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); ctrl.hist, ctrl.drops = ck["ctrl"]
        st.update(step=ck["step"], lr=ck["lr"], best=ck["best"], best_state=ck["best_state"], hist=ck["hist"]); log(f"[{name}] resumed at step {st['step']} (best {st['best']:.4f})")
    t0, bad, reason = time.time(), 0, None
    while reason is None:
        if stop_p.exists():
            reason = "stop-file"; break
        if max_hours and (time.time() - t0) / 3600 > max_hours:
            reason = "max-hours"; break
        if st["step"] >= max_steps:
            reason = "max-steps"; break
        st["step"] += 1; s = st["step"]
        for g in opt.param_groups:
            g["lr"] = st["lr"] * min(1.0, s / warmup)
        model.train(); loss = step_fn(model)
        if not torch.isfinite(loss):
            bad += 1; st["step"] -= 1; opt.zero_grad(set_to_none=True)
            if bad > 25:
                reason = "unstable (25 non-finite losses)"; break
            continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gn):
            bad += 1; st["step"] -= 1; opt.zero_grad(set_to_none=True); continue
        bad = 0; opt.step(); d = min(ema_decay, (1 + s) / (10 + s))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        if s % eval_every:
            continue
        sc = eval_fn(ema); act = ctrl.update(sc)
        st["hist"].append(dict(step=s, loss=round(float(loss), 4), val=round(sc, 4), lr=st["lr"], action=act, sec=round(time.time() - t0)))
        log(f"[{name}] {json.dumps(st['hist'][-1])}")
        if math.isfinite(sc) and sc < st["best"]:
            st["best"], st["best_state"] = sc, {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
            torch.save({"state": st["best_state"], "cfg": model.cfg, "val": sc}, best_p)
        if act == "bad" and st["best_state"] is not None:
            model.load_state_dict(st["best_state"]); ema.load_state_dict(st["best_state"]); st["lr"] *= 0.5; log(f"[{name}] non-finite validation: restored best, lr {st['lr']:.2e}")
        elif act == "drop" and st["best_state"] is not None:
            model.load_state_dict(st["best_state"]); ema.load_state_dict(st["best_state"]); st["lr"] *= 0.5
            log(f"[{name}] plateau: lr -> {st['lr']:.2e} ({ctrl.drops}/{max_drops}), best weights restored")
        elif act == "converged":
            reason = "converged"
        tmp = out / f"{name}_state.tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), ctrl=(ctrl.hist, ctrl.drops), step=s, lr=st["lr"], best=st["best"],
                        best_state=st["best_state"], hist=st["hist"]), tmp); os.replace(tmp, state_p)
        if dev.type == "mps":
            torch.mps.empty_cache()
    log(f"[{name}] finished: {reason} at step {st['step']}, best validation {st['best']:.4f}")
    return dict(reason=reason, steps=st["step"], best=st["best"], hist=st["hist"])


def sample_weights(meta, idx, power=0.5, age_power=0.5):
    sub = meta.iloc[idx]; n = sub.groupby("DatasetID")["DatasetID"].transform("size").to_numpy().astype(float)
    w = n ** -power
    if age_power > 0 and sub.Age.notna().all():
        b = np.minimum((sub.Age.to_numpy() // 10).astype(int), 10); dens = np.bincount(b, minlength=11).astype(float)
        w = w * np.where(dens[b] > 0, dens[b], 1.0) ** -age_power
    return w / w.sum()


# ======================================================================================================== evaluation
def summarize(pred, age, name, extra=None):
    err = pred - age
    d = dict(model=name, n=int(len(age)), MedAE=float(np.median(np.abs(err))), MAE=float(np.mean(np.abs(err))), MedE=float(np.median(err)),
             r=float(np.corrcoef(pred, age)[0, 1]) if len(age) > 2 else float("nan"))
    d.update(extra or {})
    return d


def standardize(X, mean, std):
    x = X.astype(np.float32); miss = x == 255; x = x / 100.0; x = (x - mean) / std; x[miss] = 0.0
    return x


def linear_baselines(c, sp, data_mean, data_std, log):
    from sklearn.linear_model import Ridge, enet_path
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Xtr, Xva, Xte = standardize(c.X[tr], data_mean, data_std), standardize(c.X[va], data_mean, data_std), standardize(c.X[te], data_mean, data_std)
    ytr, ava = age_to_t(c.meta.Age.to_numpy()[tr]), c.meta.Age.to_numpy()[va]
    out = {}
    best = (1e9, None)
    for a in (100.0, 1000.0, 10000.0, 30000.0, 100000.0):
        m = Ridge(alpha=a).fit(Xtr, ytr); e = np.median(np.abs(t_to_age(m.predict(Xva)) - ava))
        if e < best[0]:
            best = (e, m, a)
    out["ridge"] = t_to_age(best[1].predict(Xte)); log(f"[baseline] ridge alpha {best[2]} (val MedAE {best[0]:.2f})")
    alphas, coefs, _ = enet_path(Xtr, ytr - ytr.mean(), l1_ratio=0.5, n_alphas=20, eps=1e-3)
    ev = [np.median(np.abs(t_to_age(Xva @ coefs[:, k] + ytr.mean()) - ava)) for k in range(coefs.shape[1])]
    k = int(np.argmin(ev)); out["elastic net"] = t_to_age(Xte @ coefs[:, k] + ytr.mean())
    log(f"[baseline] elastic net alpha {alphas[k]:.2e} (val MedAE {ev[k]:.2f}, {int((coefs[:, k] != 0).sum())} non-zero CpGs)")
    return out


def published_clocks(c, sp, log):
    spec = importlib.util.find_spec("computage")
    if spec is None or not spec.submodule_search_locations:
        log("[published] `pip install computage --no-deps` to add Horvath/Hannum/Lin to the comparison"); return {}
    cd = Path(list(spec.submodule_search_locations)[0]) / "models_library" / "clocks"
    table = {"Horvath V1": ("Horvath1.csv", lambda s: t_to_age(s + 0.696)), "Horvath V2": ("Horvath2.csv", lambda s: t_to_age(s - 0.447119319)),
             "Hannum": ("Hannum.csv", lambda s: s), "Lin": ("Lin.csv", lambda s: s + 12.2169841)}
    med = pd.read_csv(cd / "sesame_450k_median.csv", index_col=0)["median"]; pos = {x: i for i, x in enumerate(c.cpgs)}
    Xt = c.X[sp["test"]]; out = {}
    for name, (fn, tf) in table.items():
        coef = pd.read_csv(cd / fn, index_col=0)["CoefficientTraining"]; coef = coef[coef.index.str.startswith("cg")]
        cov = np.mean([x in pos for x in coef.index])
        if cov < 1.0:
            log(f"[published] {name}: only {cov:.0%} of its CpGs are in the panel - skipped"); continue
        s = np.zeros(len(Xt))
        for cg, w in coef.items():
            v = Xt[:, pos[cg]].astype(np.float64) / 100.0; v = np.where(Xt[:, pos[cg]] == 255, med.get(cg, 0.5), v); s += w * v
        out[name] = tf(s)
    return out


def sibling_set(meta, sp):
    num = lambda g: int("".join(ch for ch in g if ch.isdigit()) or 0)
    tr = {num(g) for g in meta.iloc[np.r_[sp["train"], sp["val"]]].DatasetID.unique()}
    return {g for g in meta.iloc[sp["test"]].DatasetID.unique() if any(abs(num(g) - t) <= 3 for t in tr)}


def test_report(c, sp, preds: dict, out: Path):
    meta = c.meta.iloc[sp["test"]]; age = meta.Age.to_numpy(float); sib = sibling_set(c.meta, sp)
    blood = meta.Tissue.astype(str).str.contains("blood", case=False).to_numpy(); adult = (age >= 18) & (age <= 90); nosib = ~meta.DatasetID.isin(sib).to_numpy()
    rows = []
    for name, p in preds.items():
        base = summarize(p, age, name)
        per = pd.Series(np.abs(p - age)).groupby(meta.DatasetID.to_numpy()).median()
        rows.append(dict(model=name, MedAE_all=base["MedAE"], MedAE_18_90=float(np.median(np.abs(p - age)[adult])), MedAE_blood=float(np.median(np.abs(p - age)[blood])),
                         MedAE_no_sibling_studies=float(np.median(np.abs(p - age)[nosib])), MAE=base["MAE"], MedE=base["MedE"], r=base["r"],
                         mean_study_MedAE=float(per.mean())))
    df = pd.DataFrame(rows).set_index("model").sort_values("MedAE_all")
    hdr = f"Test: {len(meta)} healthy-control samples from {meta.DatasetID.nunique()} held-out ComputAge benchmark studies ({int(blood.sum())} blood; {len(sib)} studies adjacent to training series)"
    (out / "test_report.md").write_text(hdr + "\n\n```\n" + df.round(3).to_string() + "\n```\n"); df.to_csv(out / "test_report.csv")
    log(hdr + "\n" + df.round(2).to_string())
    return df


# ======================================================================================================== commands
def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def selftest(model, c, sp, K, dev, rng):
    """The device must agree with the CPU (loss + gradients) and be able to overfit a fixed batch; otherwise stop before a long run."""
    log(f"[selftest] comparing {dev} with cpu ...")
    rows = c.X[sp["train"][:16]]; idx, beta, valid, nobs = draw(rows, K, np.random.default_rng(1)); y = torch.as_tensor(age_to_t(c.meta.Age.to_numpy()[sp["train"][:16]]), dtype=torch.float32)
    res = {}
    for name, m in (("device", model), ("cpu", copy.deepcopy(model).cpu())):
        m.eval(); m.zero_grad(); loss = nn.functional.smooth_l1_loss(m.age_t(idx, beta, valid, nobs).cpu(), y); loss.backward()
        res[name] = (float(loss), torch.cat([p.grad.flatten().cpu().double() for p in m.parameters() if p.grad is not None]))
    cos = float((res["device"][1] * res["cpu"][1]).sum() / (res["device"][1].norm() * res["cpu"][1].norm() + 1e-12)); rel = abs(res["device"][0] - res["cpu"][0]) / max(res["cpu"][0], 1e-6)
    ok = rel < 5e-3 and cos > 0.99
    log(f"[selftest] loss {res['device'][0]:.5f} vs cpu {res['cpu'][0]:.5f} (rel {rel:.1e}), gradient cosine {cos:.5f}: {'PASS' if ok else 'FAIL'}")
    m = copy.deepcopy(model).train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); losses = []
    for _ in range(60):
        loss = nn.functional.smooth_l1_loss(m.age_t(idx, beta, valid, nobs).cpu(), y); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    ok2 = losses[-1] < 0.5 * losses[0]; log(f"[selftest] overfit one batch: {losses[0]:.4f} -> {losses[-1]:.4f}: {'PASS' if ok2 else 'FAIL'}")
    if not (ok and ok2):
        sys.exit("SELF-TEST FAILED - the device does not reproduce the CPU. Re-run with --device cpu to confirm the code, and report the lines above.")


def cmd_train(a):
    global LOGF
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); LOGF = out / "train.log"
    torch.set_num_threads(a.threads or max(1, (os.cpu_count() or 4) // 2))
    try:
        torch.backends.mha.set_fastpath_enabled(False)          # the eval-mode transformer fast path has had MPS bugs
    except AttributeError:
        pass
    P = PRESETS[a.preset]; cfg = dict(d=P["d"], layers=P["layers"], heads=P["heads"], context=not a.no_context, linear_skip=not a.no_linear_skip, dropout=P["dropout"] if a.dropout < 0 else a.dropout)
    K, B, lr = a.tokens or P["K"], a.batch or P["batch"], a.lr or P["lr"]; dev = get_device(a.device)
    data = Path(a.data); log(f"EpiClock | preset {a.preset} {cfg} | K={K} batch={B} lr={lr} | device {dev}")
    c = load_corpus(data); sp = make_splits(c, a.seed); mean, std = panel_stats(c.X, sp["train"]); ctx = genome_context(data, c.cpgs)
    log(f"corpus {len(c.X)} samples x {len(c.cpgs)} CpGs | train {len(sp['train'])} | val {len(sp['val'])} ({len(sp['val_series'])} series) | pretrain pool {len(sp['ssl'])} | test (HC) {len(sp['test'])}")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model = EpiClock(len(c.cpgs), mean, std, ctx, cfg).to(dev)
    log(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    if dev.type != "cpu" and not a.no_selftest:
        selftest(model, c, sp, K, dev, rng)
    age_all = c.meta.Age.to_numpy(float)
    # ---- stage 1: masked-CpG reconstruction (no labels)
    pre_best = out / "pretrain_best.pt"
    if not a.no_pretrain:
        w_ssl = sample_weights(c.meta, sp["ssl"], 0.5, 0.0)
        vrows = c.X[sp["val"][np.random.RandomState(5).choice(len(sp["val"]), min(1500, len(sp["val"])), replace=False)]]
        base_mae = float(np.nanmean(np.abs(np.where(vrows != 255, vrows / 100.0, np.nan) - mean)))
        def ssl_step(m):
            rows = c.X[rng.choice(sp["ssl"], B, p=w_ssl)]; idx, ib, valid, is_mask, beta = ssl_batch(rows, K, rng)
            pred = m.reconstruct(idx, ib, valid, is_mask); tgt = beta.to(pred.device); ok = (is_mask & valid).to(pred.device)
            err = (pred - tgt).abs(); return torch.where(err < 0.05, 0.5 * err ** 2 / 0.05, err - 0.025)[ok].mean()
        def ssl_eval(m):
            r = np.random.default_rng(11); tot, n = 0.0, 0; m.eval()
            with torch.no_grad():
                for s in range(0, len(vrows), 128):
                    idx, ib, valid, is_mask, beta = ssl_batch(vrows[s:s + 128], K, r); pred = m.reconstruct(idx, ib, valid, is_mask).cpu(); ok = is_mask & valid
                    tot += float((pred - beta).abs()[ok].sum()); n += int(ok.sum())
            return tot / n
        log(f"[pretrain] baseline (predict each CpG's training mean) MAE on hidden CpGs of held-out series: {base_mae:.4f}")
        r1 = run_phase("pretrain", model, ssl_step, ssl_eval, out, dev, lr, max_steps=a.ssl_steps, eval_every=a.eval_every, patience=a.patience, max_hours=a.max_hours)
        log(f"[pretrain] reconstruction MAE {r1['best']:.4f} vs mean-baseline {base_mae:.4f} ({1 - r1['best'] / base_mae:.0%} lower)")
        json.dump(dict(r1, baseline_mae=base_mae), open(out / "pretrain.json", "w"), indent=1)
    # ---- stage 2: age
    tr_rows = sp["train"]; w = sample_weights(c.meta, tr_rows, 0.5, 0.5)
    vsel = sp["val"][np.random.RandomState(6).choice(len(sp["val"]), min(2000, len(sp["val"])), replace=False)]
    def age_step(m):
        sel = rng.choice(tr_rows, B, p=w); idx, beta, valid, nobs = draw(c.X[sel], K, rng)
        y = torch.as_tensor(age_to_t(age_all[sel]), dtype=torch.float32, device=m.cls.device)
        return nn.functional.smooth_l1_loss(m.age_t(idx, beta, valid, nobs), y, beta=0.05)
    def age_eval(m):
        p = t_to_age(predict_t(m, c.X[vsel], K, a.val_draws, np.random.default_rng(13))); return float(np.median(np.abs(p - age_all[vsel])))
    r2 = run_phase("finetune", model, age_step, age_eval, out, dev, lr * (0.5 if pre_best.exists() and not a.no_pretrain else 1.0), max_steps=a.ft_steps, eval_every=a.eval_every,
                   patience=a.patience, max_hours=a.max_hours, init_from=pre_best if (pre_best.exists() and not a.no_pretrain) else None)
    best = torch.load(out / "finetune_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["state"]); model = model.to(dev)
    torch.save(dict(state=best["state"], cfg=cfg, K=K, cpgs=c.cpgs, mean=mean, std=std, ctx=ctx if cfg["context"] else None), out / "model.pt")
    # ---- test
    log(f"[test] predicting {len(sp['test'])} held-out healthy controls with {a.test_draws} subsets of {K} CpGs each ...")
    preds = {"EpiClock": t_to_age(predict_t(model, c.X[sp["test"]], K, a.test_draws, np.random.default_rng(17)))}
    if not a.no_baselines:
        preds.update(linear_baselines(c, sp, mean, std, log)); preds.update(published_clocks(c, sp, log))
    df = test_report(c, sp, preds, out)
    json.dump(dict(preset=a.preset, cfg=cfg, K=K, pretrained=not a.no_pretrain, finetune=dict(r2, hist=None), test=df.round(4).to_dict("index")), open(out / "results.json", "w"), indent=1)
    log(f"done. model: {out / 'model.pt'}; report: {out / 'test_report.md'}")


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    m = EpiClock(len(ck["cpgs"]), ck["mean"], ck["std"], ck["ctx"], cfg); m.load_state_dict(ck["state"]); return m.to(dev).eval(), ck


def cmd_predict(a):
    dev = get_device(a.device); m, ck = load_model(a.model, dev)
    B = pd.read_csv(a.beta, sep="\t" if a.beta.endswith((".tsv", ".txt")) else ",", index_col=0)
    if B.index.astype(str).str.startswith("cg").mean() > 0.5:
        B = B.T
    pos = pd.Index(ck["cpgs"]).get_indexer(B.columns.astype(str)); keep = pos >= 0
    X = np.full((len(B), len(ck["cpgs"])), 255, np.uint8); v = B.to_numpy(np.float64)[:, keep]
    X[:, pos[keep]] = np.where(np.isfinite(v) & (v >= 0) & (v <= 1), np.rint(v * 100), 255).astype(np.uint8)
    log(f"{len(B)} samples; {keep.sum()} of your {len(keep)} CpGs are in the model's panel ({keep.sum() / len(ck['cpgs']):.0%} of the panel)")
    age = t_to_age(predict_t(m, X, ck["K"], a.draws, np.random.default_rng(0)))
    pd.DataFrame({"sample": B.index, "epigenetic_age": age}).to_csv(a.out, index=False); log(f"wrote {a.out}")


def cmd_probe(a):
    """Second-level tasks on the learned model: (1) sex from frozen sample embeddings, (2) does the CpG identity embedding know the genome context?"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    dev = get_device(a.device); m, ck = load_model(a.model, dev); data = Path(a.data); c = load_corpus(data); sp = make_splits(c, 0); K = ck["K"]
    sex = c.meta.Gender.astype(str).to_numpy(); lab = np.isin(sex, ["M", "F"])
    tr = sp["train"][lab[sp["train"]]][:6000]; te = sp["test_all"][lab[sp["test_all"]]][:3000]
    def emb(rows):
        r = np.random.default_rng(3); out = []
        with torch.no_grad():
            for s in range(0, len(rows), 128):
                idx, beta, valid, _ = draw(c.X[rows[s:s + 128]], K, r); out.append(m.embed(idx, beta, valid).cpu().numpy())
        return np.concatenate(out)
    Etr, Ete = emb(tr), emb(te); ytr, yte = (sex[tr] == "F").astype(int), (sex[te] == "F").astype(int)
    sc = StandardScaler().fit(Etr); lr = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(Etr), ytr); p = lr.predict_proba(sc.transform(Ete))[:, 1]
    mean, std = ck["mean"], ck["std"]; pca = PCA(64, random_state=0).fit(standardize(c.X[tr], mean, std)); Ptr, Pte = pca.transform(standardize(c.X[tr], mean, std)), pca.transform(standardize(c.X[te], mean, std))
    sc2 = StandardScaler().fit(Ptr); lr2 = LogisticRegression(C=0.1, max_iter=500).fit(sc2.transform(Ptr), ytr); p2 = lr2.predict_proba(sc2.transform(Pte))[:, 1]
    res = {"sex_from_frozen_embedding": dict(acc=float(((p > .5) == yte).mean()), auc=float(roc_auc_score(yte, p))),
           "sex_from_PCA64_of_raw_betas": dict(acc=float(((p2 > .5) == yte).mean()), auc=float(roc_auc_score(yte, p2)))}
    log("second-level task - sex prediction on held-out cohorts (linear probe on frozen features):\n" + json.dumps(res, indent=1))
    ann = pd.read_parquet(data / "annot/annotation_450k.parquet").reindex(c.cpgs); E = m.id_emb.weight.detach().cpu().numpy()
    tm = np.nanmean(np.where(c.X[sp["train"][:4000]] == 255, np.nan, c.X[sp["train"][:4000]] / 100.0), 0); ts = np.nanstd(np.where(c.X[sp["train"][:4000]] == 255, np.nan, c.X[sp["train"][:4000]] / 100.0), 0)
    simple = np.stack([np.nan_to_num(tm, nan=0.5), np.nan_to_num(ts)], 1)
    for lbl, y in (("CpG island relation (6 classes)", ann.island.fillna(0).astype(int).to_numpy()), ("regulatory feature class", ann.reg.fillna(0).astype(int).to_numpy()), ("chromosome (24 classes)", ann.chrom.fillna(0).astype(int).to_numpy())):
        r = {"chance (majority class)": float(np.bincount(y).max() / len(y))}
        for nm, F in (("mean+sd of beta", simple), ("learned CpG embedding", E), ("embedding + mean/sd", np.concatenate([E, simple], 1))):
            r[nm] = float(cross_val_score(LogisticRegression(C=0.5, max_iter=300), StandardScaler().fit_transform(F), y, cv=4).mean())
        res["context: " + lbl] = r; log(f"context probe - {lbl}: " + json.dumps({k: round(v, 3) for k, v in r.items()}))
    if ck["cfg"].get("context", True):
        log("NOTE: this model was given the genome context as input, so the probe above is trivially easy; train with --no-context to ask whether the identity embeddings learn it from methylation alone.")
    json.dump(res, open(Path(a.model).parent / "probe.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", default=os.environ.get("EPIMOD_DATA")); t.add_argument("--out", required=True)
    t.add_argument("--preset", default="small", choices=list(PRESETS)); t.add_argument("--device", default="auto"); t.add_argument("--threads", type=int, default=0)
    t.add_argument("--tokens", type=int, default=0); t.add_argument("--batch", type=int, default=0); t.add_argument("--lr", type=float, default=0.0); t.add_argument("--dropout", type=float, default=-1.0, help="-1 = preset default")
    t.add_argument("--ssl-steps", type=int, default=10**9, help="maximum pre-training steps (it stops earlier when validation stops improving)")
    t.add_argument("--ft-steps", type=int, default=10**9); t.add_argument("--eval-every", type=int, default=500); t.add_argument("--patience", type=int, default=6)
    t.add_argument("--max-hours", type=float, default=0.0, help="per stage; 0 = until converged"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--val-draws", type=int, default=2); t.add_argument("--test-draws", type=int, default=16)
    t.add_argument("--no-pretrain", action="store_true"); t.add_argument("--no-context", action="store_true"); t.add_argument("--no-linear-skip", action="store_true")
    t.add_argument("--no-baselines", action="store_true"); t.add_argument("--no-selftest", action="store_true")
    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--beta", required=True); p.add_argument("--out", default="epigenetic_ages.csv")
    p.add_argument("--device", default="auto"); p.add_argument("--draws", type=int, default=16)
    q = sub.add_parser("probe"); q.add_argument("--model", required=True); q.add_argument("--data", default=os.environ.get("EPIMOD_DATA")); q.add_argument("--device", default="auto")
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
