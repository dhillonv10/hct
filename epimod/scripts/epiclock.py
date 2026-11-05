#!/usr/bin/env python3
"""EpiClock - a family of genome-aware transformer epigenetic clocks, trained ONLY on audited, clean data.  One script, no switches.

  python epiclock.py                     # audit and build the clean corpus -> pre-train -> fine-tune -> evaluate -> report.md   (resumable: run it again to continue)
  python epiclock.py predict --model runs/epiclock_clean/release/EpiClock-B/model.pt --beta my_betas.csv [--meta meta.csv --age-col age --group-col status --control Control]

What goes in (data_audit.md and the end of report.md give exact counts and the reason for every exclusion)
  Sources: the bundle (ComputAge Bench training studies, public GEO series) and the MethylGPT / EWAS Data Hub / ClockBase profiles (about 23 GB, downloaded once, resumable).
  A profile is used only if ALL of these hold: EWAS Data Hub's curated metadata marks it a healthy control (or part of an HIV cohort) - never cancer, never other disease, never a control that
  carries a condition field, never a cell culture; GEO's own sample description does not name a disease or cancer; its study is not cancer-themed; it is not from the same study as a
  benchmark dataset; it is not a re-deposited copy of a benchmark sample (found by comparing the methylation profiles themselves).  Profiles without a health annotation are not used.
  ComputAge Bench training studies are healthy controls by curation and are checked the same way.  The test set (ComputAge Bench healthy controls) is never trained on.

What it trains: for each size (T/S/B/L/XL) the same network with and without masked-CpG pre-training, so the report shows what pre-training is worth; a base-size model without the genome-context
inputs (to test whether the network discovers CpG islands / regulatory structure from methylation alone); a linear-only control; an ensemble.
How it is judged: ridge and elastic net on the same data, Horvath / Hannum / Lin (computage) and more published clocks (biolearn), the ComputAge Bench score, paired bootstrap error bars over
samples and over whole studies, probes of what the CpG embeddings know about islands / chromosome / regulatory class, the deviation score.

Model: every CpG is a token (learned identity + genome context + its beta); a sample is a random subset of its observed CpGs; pre-training hides 25% and reconstructs them; fine-tuning predicts
age (Horvath-transformed) = a linear clock over the whole row (ridge-initialised) + a transformer correction; test-time predictions over several CpG subsets are averaged.
Control while it runs (in the output folder): `touch STOP` pauses (delete it and re-run to resume); `touch NEXT` ends the current training stage early.
Device: the Apple GPU (MPS) when present, after a self-test against the CPU.  Every stage stops by itself when validation stops improving."""
from __future__ import annotations

import argparse
import collections
import copy
import importlib.util
import json
import math
import os
import re
import shutil
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

PRESETS = {                                                          # d_model, layers, heads, tokens per sample, batch, lr, time caps (hours) for pre-training / fine-tuning
    "tiny": dict(d=96, layers=3, heads=4, K=256, batch=64, lr=6e-4, dropout=0.0, pre_h=3.0, ft_h=3.0),
    "small": dict(d=128, layers=4, heads=4, K=384, batch=64, lr=5e-4, dropout=0.0, pre_h=6.0, ft_h=6.0),
    "base": dict(d=192, layers=6, heads=6, K=512, batch=128, lr=4e-4, dropout=0.1, pre_h=12.0, ft_h=8.0),
    "large": dict(d=256, layers=8, heads=8, K=768, batch=128, lr=3e-4, dropout=0.1, pre_h=24.0, ft_h=12.0),
    "xl": dict(d=512, layers=12, heads=8, K=1024, batch=128, lr=2e-4, dropout=0.1, pre_h=72.0, ft_h=24.0),         # ~48M parameters; for a 96+ GB machine
}
# ---- edit these instead of passing arguments -------------------------------------------------------------------------------
CONFIG = dict(
    memory_fraction=0.60,     # never plan to use more than this fraction of the machine's memory for training
    eval_every=500,           # steps between validations
    patience=6,               # validations without a 0.5% gain before the learning rate is halved (stops after 3 halvings)
    test_draws=16,            # CpG subsets averaged per test sample
    val_draws=4,              # ... and per validation sample
    boot=1000,                # bootstrap draws for the error bars
    out_dir="runs/epiclock",  # re-running the same command resumes from here
)
BUNDLE_SHA256 = "b7fa7c79a99107904d9eaba2f5c21ad516fc71c40279a3603c059fdd7ffc249b"      # epimod_bundle.tar.gz (the two-part data file)
ADULT = 20.0
LIN_SCALE = 0.01
LOGF = None
CTRL = None                                                          # study folder: a STOP / NEXT file here applies to whatever stage is running


def log(*a):
    line = time.strftime("%F %T ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    if LOGF is not None:
        with open(LOGF, "a") as f:
            f.write(line + "\n")


def ctl_file(out, name):
    for d_ in (out, CTRL):
        if d_ is not None and (Path(d_) / name).exists():
            return Path(d_) / name
    return None


def age_to_t(age):
    age = np.asarray(age, np.float64)
    return np.where(age <= ADULT, np.log(np.clip(age, 0, None) + 1) - np.log(ADULT + 1), (age - ADULT) / (ADULT + 1))


def t_to_age(t):
    t = np.asarray(t, np.float64)
    return np.where(t <= 0, (ADULT + 1) * np.exp(np.minimum(t, 0)) - 1, (ADULT + 1) * t + ADULT)


# ======================================================================================================== data discovery / memory
def find_data(arg=None) -> Path:
    """Locate the data folder; if only the bundle files (epimod_bundle.tar.gz[.part-00/.part-01]) are around, unpack them."""
    here = Path(__file__).resolve().parent
    for cand in (arg, os.environ.get("EPIMOD_DATA"), here / "epiclock_data/epimod_bundle/data", here / "epimod_bundle/data", Path.cwd() / "epimod_bundle/data", here / "data"):
        if cand and (Path(cand) / "lb/lite_universe.parquet").exists():
            return Path(cand)
    for d in (here, Path.cwd(), Path.home() / "Downloads", Path.home() / "Desktop", Path.home() / "Documents"):
        whole = d / "epimod_bundle.tar.gz"
        if whole.exists():
            return unpack_bundle([whole], here / "epiclock_data")
        found = {}
        for f in sorted(d.glob("epimod_bundle.tar.gz.part-*")):
            m = re.search(r"part-(\d+)", f.name)
            if m and int(m.group(1)) not in found:
                found[int(m.group(1))] = f
        if found:
            return unpack_bundle([found[k] for k in sorted(found)], here / "epiclock_data")
    sys.exit("Could not find the data. Put the two files epimod_bundle.tar.gz.part-00 and epimod_bundle.tar.gz.part-01 (sent earlier) in the same folder as this script "
             "(or in ~/Downloads) and run it again. They are unpacked automatically; nothing else is needed.")


def unpack_bundle(parts, dest: Path) -> Path:
    import hashlib, shutil, tarfile
    dest.mkdir(parents=True, exist_ok=True); joined = dest / "bundle.joined.tmp"; h = hashlib.sha256()
    log(f"unpacking the data bundle from {[p.name for p in parts]} ...")
    with open(joined, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk); out.write(blk)
    if h.hexdigest() != BUNDLE_SHA256:
        joined.unlink(missing_ok=True)
        sys.exit(f"The bundle files are incomplete or corrupted (checksum {h.hexdigest()[:12]}..., expected {BUNDLE_SHA256[:12]}...). Re-download both part files.")
    with tarfile.open(joined, "r:gz") as tar:
        for m in tar:
            if m.name.startswith("epimod_bundle/data/") and ".." not in m.name and not m.name.startswith("/"):
                tar.extract(m, dest)
    joined.unlink(missing_ok=True)
    log(f"data unpacked to {dest / 'epimod_bundle/data'}")
    return dest / "epimod_bundle/data"


def total_mem_gb() -> float:
    import subprocess
    try:
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return 16.0


def is_oom(e: BaseException) -> bool:
    t = str(e).lower()
    return "out of memory" in t or "invalid buffer size" in t or "insufficient memory" in t


def free_mem(dev):
    import gc
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def used_gb(dev) -> float:
    try:
        if dev.type == "mps":
            return torch.mps.driver_allocated_memory() / 2**30
        if dev.type == "cuda":
            return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        pass
    return 0.0


class Budget:
    """Effective batch = micro-batch x accumulation.  The micro-batch is the largest that fits the memory budget and shrinks on any
    out-of-memory error, so the run keeps going (with more accumulation) instead of crashing."""

    def __init__(self, target, dev, limit_gb):
        self.target, self.micro, self.dev, self.limit = target, min(target, int(os.environ.get("EPICLOCK_MAX_MICRO", target))), dev, limit_gb

    @property
    def accum(self):
        return math.ceil(self.target / self.micro)

    def fit(self, model, trial):
        """Halve the micro-batch until one forward+backward fits the budget."""
        while self.micro >= 1:
            try:
                free_mem(self.dev)
                if self.dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                loss = trial(model, self.micro); loss.backward(); model.zero_grad(set_to_none=True)
                u = used_gb(self.dev)
                if u <= self.limit:
                    log(f"[memory] micro-batch {self.micro} x {self.accum} accumulation steps = batch {self.target}; trial used {u:.1f} GB of the {self.limit:.0f} GB budget")
                    return
                log(f"[memory] micro-batch {self.micro} needs {u:.1f} GB > budget {self.limit:.0f} GB: halving")
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                log(f"[memory] micro-batch {self.micro}: out of memory: halving")
            model.zero_grad(set_to_none=True); free_mem(self.dev); self.micro //= 2
        sys.exit("Not even a micro-batch of 1 fits in memory. Close other applications or choose a smaller model (edit CONFIG['preset']).")

    def shrink(self):
        if self.micro <= 1:
            return False
        self.micro //= 2; free_mem(self.dev); log(f"[memory] out of memory during training: micro-batch -> {self.micro} ({self.accum} accumulation steps)")
        return True


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
        self.lin_u = nn.Parameter(torch.zeros(n_cpg))                 # linear age term over the WHOLE row (exact; initialised from ridge)
        self.lin_on = cfg.get("linear_skip", True); self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.age_head.weight); nn.init.zeros_(self.age_head.bias)      # start exactly at the linear clock
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

    def age_t(self, idx, beta, valid, fz):
        """Age in the transformed space = [CLS] correction learned by the transformer + linear clock over the full standardized row ``fz``."""
        valid = valid.to(self.cls.device); h, _ = self.encode(idx, beta, valid)
        mu = self.age_head(h[:, 0]).squeeze(-1) + self.bias
        if self.lin_on:
            mu = mu + LIN_SCALE * (fz.to(h.device) @ self.lin_u)
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


def rowz(X_rows, mean, std):
    """Standardized full rows (missing = 0, clipped to +-6) as a float32 torch tensor."""
    z = np.clip((X_rows.astype(np.float32) / 100.0 - mean) / std, -6, 6); z[X_rows == 255] = 0.0
    return torch.as_tensor(z.astype(np.float32))


def ssl_batch(X_rows, K, rng, mask_frac=0.25):
    idx, beta, valid, _ = draw(X_rows, K, rng)
    nt = int(K * mask_frac); is_mask = torch.zeros_like(valid); is_mask[:, :nt] = True
    inp_beta = torch.where(is_mask, torch.full_like(beta, 0.5), beta)
    return idx, inp_beta, valid, is_mask, beta


@torch.no_grad()
def predict_t(model, X_rows, K, R, rng, batch=0):
    batch = batch or max(8, 65536 // K)
    mean_, std_ = model.mean.cpu().numpy(), model.std.cpu().numpy()
    """Average over R disjoint random CpG subsets of size K (all observed CpGs when R*K covers them)."""
    model.eval(); out = np.zeros(len(X_rows)); dev = model.cls.device
    for s in range(0, len(X_rows), batch):
        rows = X_rows[s:s + batch]; obs = rows != 255; fz = rowz(rows, mean_, std_)
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
            mu = model.age_t(torch.as_tensor(sl, dtype=torch.long), torch.as_tensor(beta), torch.as_tensor(valid), fz).cpu().double().numpy()
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
              patience=6, min_rel=0.005, max_drops=3, max_hours=0.0, init_from=None, post_init=None, budget=None):
    """Train until the validation score (lower = better) stops improving.  Resumable; atomic checkpoints; non-finite steps skipped.
    Control files in ``out`` or the study folder: STOP pauses the whole run (delete it and re-run to resume); NEXT ends the current phase now and moves on."""
    state_p, best_p, done_p = out / f"{name}_state.pt", out / f"{name}_best.pt", out / f"{name}_done.json"
    if done_p.exists():                                                  # a finished phase is never trained again on a restart
        r = json.load(open(done_p)); log(f"[{name}] already finished ({r['reason']}, best validation {r['best']:.4f}); skipping"); return r
    nodecay = lambda n: "emb" in n or n in ("mask_emb", "cls", "lin_u", "bias")
    emb = [p for n, p in model.named_parameters() if nodecay(n)]
    rest = [p for n, p in model.named_parameters() if not nodecay(n)]
    opt = torch.optim.AdamW([{"params": emb, "weight_decay": 0.0}, {"params": rest, "weight_decay": wd}], lr=lr)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ctrl = Plateau(patience, min_rel, max_drops)
    st = dict(step=0, lr=lr, best=float("inf"), best_state=None, hist=[])
    if init_from is not None and not state_p.exists():
        sd = torch.load(init_from, map_location="cpu", weights_only=False)["state"]
        miss = model.load_state_dict(sd, strict=False); ema.load_state_dict(model.state_dict()); log(f"[{name}] initialised from {init_from} ({len(miss.missing_keys)} new tensors)")
    if post_init is not None and not state_p.exists():
        post_init(model); ema.load_state_dict(model.state_dict())
    if not state_p.exists():                                           # score the starting point: the result can never be worse than it on validation
        sc0 = eval_fn(ema); st["best"], st["best_state"] = sc0, {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        torch.save({"state": st["best_state"], "cfg": model.cfg, "val": sc0}, best_p); st["hist"].append(dict(step=0, val=round(sc0, 4))); ctrl.hist = [sc0]
        log(f"[{name}] starting point: validation {sc0:.4f}")
    if state_p.exists():
        ck = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); ctrl.hist, ctrl.drops = ck["ctrl"]
        st.update(step=ck["step"], lr=ck["lr"], best=ck["best"], best_state=ck["best_state"], hist=ck["hist"]); log(f"[{name}] resumed at step {st['step']} (best {st['best']:.4f})")
    t0, bad, reason = time.time(), 0, None
    while reason is None:
        if ctl_file(out, "STOP"):
            reason = "stop-file"; break
        if ctl_file(out, "NEXT"):
            reason = "next-file"; break
        if max_hours and (time.time() - t0) / 3600 > max_hours:
            reason = "max-hours"; break
        if st["step"] >= max_steps:
            reason = "max-steps"; break
        st["step"] += 1; s = st["step"]
        for g in opt.param_groups:
            g["lr"] = st["lr"] * min(1.0, s / warmup)
        model.train(); opt.zero_grad(set_to_none=True); loss_val, finite = 0.0, True
        try:
            if os.environ.get("EPICLOCK_FAKE_OOM") and st["step"] == 3 and not st.get("faked"):          # test hook for the out-of-memory back-off
                st["faked"] = True; raise RuntimeError("MPS backend out of memory (simulated)")
            for _ in range(budget.accum):
                l = step_fn(model, budget.micro)
                if not torch.isfinite(l):
                    finite = False; break
                (l / budget.accum).backward(); loss_val += float(l.detach()) / budget.accum
        except RuntimeError as e:
            if not is_oom(e) or not budget.shrink():
                raise
            opt.zero_grad(set_to_none=True); st["step"] -= 1; continue
        if not finite:
            bad += 1; st["step"] -= 1; opt.zero_grad(set_to_none=True)
            if bad > 25:
                reason = "unstable (25 non-finite losses)"; break
            continue
        loss = torch.tensor(loss_val)
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
    res = dict(reason=reason, steps=st["step"], best=st["best"], hist=st["hist"])
    if reason != "stop-file" and not reason.startswith("unstable"):
        json.dump(res, open(done_p, "w"))
    if reason == "next-file":
        ctl_file(out, "NEXT").unlink()
    return res


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
    x = X.astype(np.float32); miss = x == 255; x = np.clip((x / 100.0 - mean) / std, -6, 6); x[miss] = 0.0
    return x


def ridge_fit(c, sp, mean, std, log, alphas=(300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)):
    """Closed-form ridge path through the Gram matrix; alpha is chosen on the validation series.  -> dict(w, b, alpha, val_medae)."""
    tr, va = sp["train"], sp["val"]; age = c.meta.Age.to_numpy(float)
    Xtr = standardize(c.X[tr], mean, std); y = age_to_t(age[tr]); ym = float(y.mean())
    G = (Xtr.T @ Xtr).astype(np.float64); rhs = Xtr.T.astype(np.float64) @ (y - ym); del Xtr
    Xva = standardize(c.X[va], mean, std); best = None
    for a in alphas:
        w = np.linalg.solve(G + a * np.eye(len(G)), rhs); e = float(np.median(np.abs(t_to_age(Xva @ w + ym) - age[va])))
        if best is None or e < best["val_medae"]:
            best = dict(w=w.astype(np.float32), b=ym, alpha=a, val_medae=e)
    log(f"[ridge] alpha {best['alpha']:g}: validation MedAE {best['val_medae']:.2f} y (this is the linear clock the transformer starts from)")
    return best


# ======================================================================================================== comparators
def ensure_pkg(mod, pip_args, why):
    """Import check; if the package is missing, pip-install it once (set EPICLOCK_NO_INSTALL=1 to forbid)."""
    if importlib.util.find_spec(mod) is not None:
        return True
    if os.environ.get("EPICLOCK_NO_INSTALL"):
        return False
    import subprocess
    log(f"[setup] pip install {' '.join(pip_args)}   ({why})")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pip_args], capture_output=True, text=True)
    importlib.invalidate_caches()
    if importlib.util.find_spec(mod) is None:
        log(f"[setup] could not install {pip_args[0]} ({r.stderr.strip()[-200:]}); continuing without it")
        return False
    return True


def get_ridge(e):
    f = e.out / "ridge.npz"
    if f.exists():
        z = np.load(f); return dict(w=z["w"], b=float(z["b"]), alpha=float(z["alpha"]), val_medae=float(z["val_medae"]))
    rg = ridge_fit(e.c, e.sp, e.mean, e.std, log); np.savez(f, **rg); return rg


def enet_predict(e, X_rows):
    from sklearn.linear_model import enet_path
    c, sp = e.c, e.sp; tr, va = sp["train"], sp["val"]
    Xtr, Xva = standardize(c.X[tr], e.mean, e.std), standardize(c.X[va], e.mean, e.std)
    ytr, ava = age_to_t(e.age[tr]), e.age[va]
    alphas, coefs, _ = enet_path(Xtr, ytr - ytr.mean(), l1_ratio=0.5, n_alphas=5 if e.a.smoke else 20, eps=1e-3)
    ev = [np.median(np.abs(t_to_age(Xva @ coefs[:, k] + ytr.mean()) - ava)) for k in range(coefs.shape[1])]
    k = int(np.argmin(ev))
    log(f"[baseline] elastic net alpha {alphas[k]:.2e} (val MedAE {ev[k]:.2f}, {int((coefs[:, k] != 0).sum())} non-zero CpGs)")
    return t_to_age(standardize(X_rows, e.mean, e.std) @ coefs[:, k] + ytr.mean())


def published_clocks(c, X, log):
    """Horvath V1/V2, Hannum, Lin from the `computage` package (coefficients read from its files; SeSAMe-median imputation of missing CpGs)."""
    if not ensure_pkg("computage", ["computage", "--no-deps"], "Horvath / Hannum / Lin coefficients and the official benchmark list"):
        return {}
    spec = importlib.util.find_spec("computage")
    cd = Path(list(spec.submodule_search_locations)[0]) / "models_library" / "clocks"
    table = {"Horvath V1": ("Horvath1.csv", lambda s: t_to_age(s + 0.696)), "Horvath V2": ("Horvath2.csv", lambda s: t_to_age(s - 0.447119319)),
             "Hannum": ("Hannum.csv", lambda s: s), "Lin": ("Lin.csv", lambda s: s + 12.2169841)}
    med = pd.read_csv(cd / "sesame_450k_median.csv", index_col=0)["median"]; pos = {x: i for i, x in enumerate(c.cpgs)}; out = {}
    for name, (fn, tf) in table.items():
        coef = pd.read_csv(cd / fn, index_col=0)["CoefficientTraining"]; coef = coef[coef.index.str.startswith("cg")]
        cov = np.mean([x in pos for x in coef.index])
        if cov < 1.0:
            log(f"[published] {name}: only {cov:.0%} of its CpGs are in the panel - skipped"); continue
        s = np.zeros(len(X))
        for cg, w in coef.items():
            v = X[:, pos[cg]].astype(np.float64) / 100.0; v = np.where(X[:, pos[cg]] == 255, med.get(cg, 0.5), v); s += w * v
        out[name] = tf(s)
    return out


BL_CLOCKS = {"Horvathv1": "Horvath V1", "Horvathv2": "Horvath V2", "Hannum": "Hannum", "Lin": "Lin", "VidalBralo": "Vidal-Bralo",
             "PhenoAge": "PhenoAge (83% of CpGs in panel)", "YingCausAge": "Ying CausAge", "YingDamAge": "Ying DamAge", "YingAdaptAge": "Ying AdaptAge"}


def biolearn_clocks(c, X, log):
    """More published clocks through the `biolearn` package (its own coefficient files and SeSAMe-style imputation)."""
    if not ensure_pkg("biolearn", ["biolearn"], "extra published clocks: Vidal-Bralo, PhenoAge, Ying CausAge/DamAge/AdaptAge"):
        return {}
    try:
        from biolearn.data_library import GeoData
        from biolearn.model_gallery import ModelGallery
    except Exception as ex:
        log(f"[biolearn] installed but cannot be imported ({type(ex).__name__}: {str(ex)[:150]}); skipping it"); return {}
    dn = pd.DataFrame(np.where(X == 255, np.nan, X / 100.0).T, index=c.cpgs, columns=[f"s{i}" for i in range(len(X))])
    gd, gal, out = GeoData(pd.DataFrame(index=dn.columns), dn), ModelGallery(), {}
    for nm, pretty in BL_CLOCKS.items():
        try:
            p = gal.get(nm).predict(gd); p = p.iloc[:, 0] if getattr(p, "ndim", 1) == 2 else p
            out[pretty] = np.asarray(p, float)
        except Exception as ex:
            log(f"[biolearn] {nm}: {type(ex).__name__}: {str(ex)[:120]} - skipped")
    return out


def get_comparators(e):
    """Predictions of every comparator on all ComputAge benchmark samples (cached)."""
    f = e.out / "comparators.npz"
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    rg = get_ridge(e); X = e.c.X[e.sp["test_all"]]
    out = {"ridge": t_to_age(standardize(X, e.mean, e.std) @ rg["w"] + rg["b"]), "elastic net": enet_predict(e, X)}
    out.update(published_clocks(e.c, X, log))
    for pretty, p in biolearn_clocks(e.c, X, log).items():
        out[pretty if pretty not in out else "biolearn " + pretty] = p
    hc = np.searchsorted(e.sp["test_all"], e.sp["test"]); age = e.age[e.sp["test"]]
    for k in ("Horvath V1", "Horvath V2", "Hannum", "Lin"):
        if k in out and "biolearn " + k in out:
            log(f"[comparators] {k}: MedAE {np.median(np.abs(out[k][hc] - age)):.2f} (computage) vs {np.median(np.abs(out['biolearn ' + k][hc] - age)):.2f} (biolearn) - two independent implementations")
    np.savez(f, **out); return out


# ======================================================================================================== statistics
class Boot:
    """Shared bootstrap draws so every comparison is paired.  'samples' resamples individual samples; 'studies' resamples whole studies
    (the honest one: samples within a study share batch effects, so the study is the unit that generalises)."""

    def __init__(self, groups, B=1000, seed=0):
        rng = np.random.default_rng(seed); n = len(groups); self.B = B
        self.samp = rng.integers(0, n, (B, n)); _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        by = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
        self.clus = [np.concatenate([by[k] for k in rng.integers(0, len(by), len(by))]) for _ in range(B)]

    def med(self, x, mode):
        return np.median(x[self.samp], axis=1) if mode == "samples" else np.array([np.median(x[i]) for i in self.clus])


def ci(v, digits=2, signed=False):
    lo, hi = np.percentile(v, [2.5, 97.5]); f = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return "[" + f.format(lo) + ", " + f.format(hi) + "]"


# ======================================================================================================== the official ComputAge Bench score
def official_config():
    spec = importlib.util.find_spec("computage")
    if spec is None or not spec.submodule_search_locations:
        return None
    p = Path(list(spec.submodule_search_locations)[0]) / "configs" / "datasets_bench_config.py"
    if not p.exists():
        return None
    s = importlib.util.spec_from_file_location("computage_bench_cfg", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.datasets_config_main


def bh(p):
    p = np.where(np.isfinite(p), p, 1.0); n = len(p); o = np.argsort(p); q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0); return out


def official_scores(e, P):
    """The ComputAge Bench scoring (same list of studies, tests, FDR and total-score formula as the `computage` package), applied to our
    predictions: AA2 = age acceleration higher in disease than in healthy controls of the same study; AA1 = positive age acceleration in
    disease; plus median error / bias on healthy controls.  Needs `computage` installed for the study list."""
    cfg = official_config()
    if cfg is None:
        return None
    from scipy.stats import ttest_1samp, ttest_ind
    meta = e.c.meta.iloc[e.sp["test_all"]]; ds, cond, cls = meta.DatasetID.to_numpy(), meta.Condition.astype(str).to_numpy(), meta.Class.astype(str).to_numpy()
    age = meta.Age.to_numpy(float); keep = meta.Tissue.isin(["Blood", "Saliva", "Buccal"]).to_numpy() & np.isfinite(age)
    names = list(P); p2 = {n: [] for n in names}; p1 = {n: [] for n in names}
    for gse, conf in cfg.items():
        base = keep & (ds == gse)
        for cd in conf["conditions"]:
            sel = base & (cond == cd); klass = cls[sel][0] if sel.any() else None
            ok = base if klass == "PGS" else base & (age >= 18) & (age <= 90)
            d_idx, h_idx = np.flatnonzero(ok & (cond == cd)), np.flatnonzero(ok & (cond == "HC"))
            if len(d_idx) == 0 or (conf["test"] == "AA2" and len(h_idx) == 0):
                continue
            for n in names:
                dd = P[n][d_idx] - age[d_idx]
                if conf["test"] == "AA2":
                    p2[n].append(ttest_ind(dd, P[n][h_idx] - age[h_idx], equal_var=False, alternative="greater")[1])
                else:
                    p1[n].append(ttest_1samp(dd, 0.0, alternative="greater")[1])
    hc = keep & np.isin(ds, list(cfg)) & (cond == "HC"); rows = []
    for n in names:
        s2 = int((bh(np.array(p2[n])) < 0.05).sum()); s1 = int((bh(np.array(p1[n])) < 0.05).sum()); err = P[n][hc] - age[hc]
        mad, mede = float(np.median(np.abs(err))), float(np.median(err))
        rows.append(dict(model=n, AA2=f"{s2}/{len(p2[n])}", AA1=f"{s1}/{len(p1[n])}", HC_MedAE=mad, HC_MedE=mede, score=round(s2 + s1 * (1 - max(0.0, mede) / mad), 1)))
    return pd.DataFrame(rows).set_index("model").sort_values("score", ascending=False)


# ======================================================================================================== disease age acceleration (for your own cohorts)
def age_acceleration(pred, age, grp, control, B=2000, seed=0):
    """Age acceleration = predicted age minus the value expected from chronological age in the control group (a straight line fitted on controls).
    For every other group: mean AA difference to controls and the AUC of AA for separating the group from controls, with bootstrap 95% CIs."""
    from scipy.stats import mannwhitneyu
    pred, age, grp = np.asarray(pred, float), np.asarray(age, float), np.asarray(grp).astype(str); ok = np.isfinite(pred) & np.isfinite(age)
    ctl = ok & (grp == str(control)); A = np.stack([np.ones(ctl.sum()), age[ctl]], 1); b = np.linalg.lstsq(A, pred[ctl], rcond=None)[0]
    aa = pred - (b[0] + b[1] * age); rng = np.random.default_rng(seed); rows = []
    auc = lambda x, y: (np.mean(x[:, None] > y[None, :]) + 0.5 * np.mean(x[:, None] == y[None, :]))
    for g in sorted(set(grp[ok]) - {str(control)}):
        x, y = aa[ok & (grp == g)], aa[ctl]
        if len(x) < 3:
            continue
        d, u = [], []
        for _ in range(B if len(x) * len(y) < 4e6 else 300):
            xs, ys = rng.choice(x, len(x)), rng.choice(y, len(y)); d.append(xs.mean() - ys.mean()); u.append(auc(xs[:400], ys[:400]))
        rows.append(dict(group=g, n=len(x), n_control=len(y), AA_diff_years=x.mean() - y.mean(), AA_diff_CI=ci(d, 2, True), AUC=auc(x[:2000], y[:2000]), AUC_CI=ci(u, 3),
                         p_two_sided=mannwhitneyu(x, y).pvalue))
    return pd.DataFrame(rows).set_index("group")


# ======================================================================================================== deviation from the healthy methylome (second-level task)
@torch.no_grad()
def recon_error(model, X_rows, K, R, rng, batch=0):
    """Per-sample mean absolute error when 25% of the K visible CpGs are hidden and reconstructed (mean of R random draws).  A sample whose
    methylation does not follow the co-methylation structure learned from healthy data is reconstructed worse: a deviation score."""
    batch = batch or max(8, 65536 // K); model.eval(); out = np.zeros(len(X_rows))
    for s in range(0, len(X_rows), batch):
        acc = np.zeros(len(X_rows[s:s + batch]))
        for _ in range(R):
            idx, ib, valid, is_mask, beta = ssl_batch(X_rows[s:s + batch], K, rng)
            pred = model.reconstruct(idx, ib, valid, is_mask).cpu(); ok = (is_mask & valid).float()
            acc += ((pred - beta).abs() * ok).sum(1).numpy() / np.maximum(ok.sum(1).numpy(), 1)
        out[s:s + batch] = acc / R
    return out


# ======================================================================================================== device / self-test / loading
def sibling_set(meta, sp):
    num = lambda g: int("".join(ch for ch in g if ch.isdigit()) or 0)
    tr = {num(g) for g in meta.iloc[np.r_[sp["train"], sp["val"]]].DatasetID.unique()}
    return {g for g in meta.iloc[sp["test"]].DatasetID.unique() if any(abs(num(g) - t) <= 3 for t in tr)}



def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def selftest(model, c, sp, K, dev, rng):
    """The device must agree with the CPU (loss + gradients) and be able to overfit a fixed batch; otherwise stop before a long run."""
    log(f"[selftest] comparing {dev} with cpu ...")
    rows = c.X[sp["train"][:16]]; idx, beta, valid, _ = draw(rows, K, np.random.default_rng(1)); fz = rowz(rows, model.mean.cpu().numpy(), model.std.cpu().numpy())
    y = torch.as_tensor(age_to_t(c.meta.Age.to_numpy()[sp["train"][:16]]), dtype=torch.float32)
    base = copy.deepcopy(model).cpu(); g = torch.Generator().manual_seed(0)            # random non-zero weights in every part, identical on both devices
    for n, p in base.named_parameters():
        if n in ("age_head.weight", "lin_u"):
            p.data.copy_(torch.randn(p.shape, generator=g) * (0.05 if n == "age_head.weight" else 1.0))
    res = {}
    for name, m in (("device", copy.deepcopy(base).to(dev)), ("cpu", copy.deepcopy(base))):
        m.eval(); m.zero_grad(); loss = nn.functional.smooth_l1_loss(m.age_t(idx, beta, valid, fz).cpu(), y); loss.backward()
        res[name] = (float(loss), torch.cat([p.grad.flatten().cpu().double() for p in m.parameters() if p.grad is not None]))
    cos = float((res["device"][1] * res["cpu"][1]).sum() / (res["device"][1].norm() * res["cpu"][1].norm() + 1e-12)); rel = abs(res["device"][0] - res["cpu"][0]) / max(res["cpu"][0], 1e-6)
    ok = rel < 5e-3 and cos > 0.99
    log(f"[selftest] loss {res['device'][0]:.5f} vs cpu {res['cpu'][0]:.5f} (rel {rel:.1e}), gradient cosine {cos:.5f}: {'PASS' if ok else 'FAIL'}")
    m = copy.deepcopy(base).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); losses = []
    for _ in range(60):
        loss = nn.functional.smooth_l1_loss(m.age_t(idx, beta, valid, fz).cpu(), y); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    ok2 = losses[-1] < 0.5 * losses[0]; log(f"[selftest] overfit one batch: {losses[0]:.4f} -> {losses[-1]:.4f}: {'PASS' if ok2 else 'FAIL'}")
    if not (ok and ok2):
        sys.exit("SELF-TEST FAILED - the device does not reproduce the CPU. Re-run with --device cpu to confirm the code, and report the lines above.")



def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    m = EpiClock(len(ck["cpgs"]), ck["mean"], ck["std"], ck["ctx"], cfg); m.load_state_dict(ck["state"]); return m.to(dev).eval(), ck



# ======================================================================================================== the study: shared state, arms, stages
class Paused(Exception):
    pass


class Env:
    """What every stage shares: data, splits, standardisation, device and memory budget."""


SIZE_NAMES = {"tiny": "EpiClock-T", "small": "EpiClock-S", "base": "EpiClock-B", "large": "EpiClock-L", "xl": "EpiClock-XL"}
KINDS = {"pre": "pre-trained transformer + linear core (the main model)", "scratch": "the same network without pre-training",
         "rob": "pre-trained, fine-tuned with a cross-study robustness penalty (variance of the per-study mean error)",
         "noctx": "pre-trained, but the genome-context inputs (island, chromosome, regulatory class, position) are removed",
         "pure": "transformer alone: no linear core, no pre-training", "cons": "pre-trained, fine-tuned with a two-view consistency penalty"}
DEFAULT_QUEUE = ["linear-tuned", "tiny-scratch", "tiny-pre", "small-scratch", "small-pre", "base-scratch", "base-pre", "base-noctx", "large-scratch", "large-pre", "xl-scratch", "xl-pre"]
SMOKE_QUEUE = ["linear-tuned", "tiny-scratch", "tiny-pre", "tiny-noctx"]
CONS_LAMBDA = 0.5
ROB_LAMBDA = 5.0


# ======================================================================================================== data: ONE fixed, audited training corpus
# Training uses only profiles that pass every check below.  Nothing "dirty" is trained on; everything that is excluded is counted, with the reason, in data_audit.md.
EWAS_URL = "https://ngdc.cncb.ac.cn/ewas/datahub/repository/basic?field=&val=&relationship=&limit={limit}&offset={offset}&sort=&order=asc"
MG_URL = "https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=1"
MG_PROBES_URL = "https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=1"
DUP_RMSD = 0.04                                        # a profile this close (RMS difference over co-measured CpGs) to a benchmark profile is the same sample deposited again

# EWAS Data Hub's "control" label is NOT "healthy" (it includes HIV+ patients, cancer survivors, treated samples, controls that carry a disease label ...).
# A sample counts as healthy only if it is a control AND none of the condition-bearing metadata fields below is filled in AND it is not a cell culture.  HIV cohorts are accepted; cancer never.
EWAS_CONDITION_FIELDS = re.compile(r"disease|doid|infection|treatment|condition|tumor|cancer|histology|morpholog|metasta|survival|relapse|recurren|progression|sledai|psoriasis|psychiatric|"
                                   r"cognition|drug resistant|hypertension|hyperlipemia|pregnancy complication|gdm|obese|stage|mutation|respond|outcome|illness|anti ss|ocular|focus score|"
                                   r"mrc scale|national institutes|stroke|suicide|abuse|trauma|adversity|asthma|atopy|crp|future status|braak|toxicology|cause of death|passage|reprogramming|"
                                   r"differentiat|viral load|cd4|antiretroviral|art adherence|intravenous|drug use|surgery|vital status|genotype|expression|biopsy|group$|baseline|time point|"
                                   r"exposure|diagnosis|onset|risk|polybrominated|arsenic|lead level", re.I)
EWAS_CANCER = re.compile(r"cancer|carcinoma|tumou?r|leuk[a]?emia|lymphoma|myeloma|sarcoma|blastoma|neoplas|malignan|metasta|myelodysplas|\bmds\b|kaposi|oncolog|"
                         r"\b(?!glaucoma|trachoma|granuloma|stroma|coma\b)\w{3,}oma\b", re.I)
EWAS_CANCER_PROJECT = re.compile(r"^(TCGA|TARGET|CPTAC|HCMI|CGCI)")
EWAS_HIV = re.compile(r"\bHIV|\bAIDS\b|immunodeficiency virus", re.I)
EWAS_HIV_FIELDS_OK = re.compile(r"infection|viral load|cd4|antiretroviral|art adherence|intravenous|drug use|doid|^disease$|disease status|disease subtype|treatment|time point", re.I)
EWAS_CULTURE_TISSUE = re.compile(r"cell line|cultured|iPSC|induced pluripotent|embryonic stem|organoid|xenograft", re.I)
GOOD_TIERS = ("clean healthy", "HIV cohort (no cancer)")

# second opinion from GEO's own sample descriptions: a string is evidence of disease when it names a disease or cancer and is not a negation / control statement
GEO_DISEASE = re.compile(r"\b(disease|disorder|syndrome|patients?|cases?|diagnos\w*|infect\w*|diabet\w*|arthrit\w*|lupus|alzheimer\w*|parkinson\w*|schizophren\w*|psychos\w*|autis\w*|asthma\w*|copd|hypertens\w*|stroke|dementia|sclerosis|"
                         r"crohn\w*|colitis|psoria\w*|depress\w*|bipolar|epilep\w*|covid\w*|sars\w*|sepsis|fibrosis|cirrhosis|hepatitis|nephro\w*|cardiovascular|coronary|myocardial|heart failure|infarct\w*|"
                         r"atheroscler\w*|osteoporo\w*|endometriosis|pre-?eclampsia|obes\w*|cystic|dystroph\w*|palsy|ataxia|huntington\w*|tubercul\w*|malaria|leprosy|progeria|trisomy|tauopath\w*|"
                         r"neurodegener\w*|lesions?|inflammat\w*|psp|ad|ms|ra|sle|ptsd|mdd|bpd|adenoma|polyp\w*)\b", re.I)
GEO_HEALTHY = re.compile(r"\b(control\w*|healthy|normal|non[- ]?(case|patient|disease|demented|affected|cancer|tumou?r)\w*|unaffected|no|none|negative|wild[- ]?type|free|never|absent|without|not|reference|neg)\b|n/a", re.I)
GEO_CULTURE = re.compile(r"\b(cultured|in[- ]vitro|passage|cell[- ]?lines?|ipsc\w*|differentiat\w*|organoid\w*|xenograft\w*|reprogram\w*|stimulat\w*|transfect\w*|knock-?(down|out)|crispr|embryoid|hesc|culture\w*|day[ _-]?\d+|d\d{1,2}_|\w+_day\d+)\b", re.I)
GEO_NEG_VALUES = {"0", "n", "no", "na", "n/a", "nan", "none", "false", "-", "unknown", "exclude", "nd", "not applicable", "missing", "."}
CANCER_THEMED = re.compile(r"cancer\w*|carcinoma\w*|tumou?rs?|leuk[a]?emi\w*|lymphoma\w*|myeloma\w*|sarcoma\w*|blastoma\w*|neoplas\w*|malignan\w*|metasta\w*|oncolog\w*|\bMDS\b", re.I)


def ewas_tier(pg):
    """Disease-state tier of every sample on one page of EWAS Data Hub metadata."""
    cond = [k for k in pg.columns if EWAS_CONDITION_FIELDS.search(k)]
    txt = [k for k in pg.columns if re.search(r"disease|condition|histology|morpholog|tumor|metasta|status|type|diagnos|group|stage|survival|relapse", k, re.I)]
    st = pg["sample type"]; cancer = (st == "adjacent normal") | pg["project id"].astype(str).str.contains(EWAS_CANCER_PROJECT)
    for k in txt:
        cancer = cancer | pg[k].astype(str).str.contains(EWAS_CANCER, na=False)
    culture = pg["tissue"].astype(str).str.contains(EWAS_CULTURE_TISSUE)
    strict = (st == "control") & (pg[cond].isna().all(axis=1) if cond else True) & ~culture & ~cancer
    hiv = pd.Series(False, index=pg.index)
    for k in ("disease", "infection", "disease status", "disease subtype"):
        if k in pg:
            hiv = hiv | pg[k].astype(str).str.contains(EWAS_HIV, na=False)
    other = [k for k in cond if not EWAS_HIV_FIELDS_OK.search(k)]
    named = pd.Series(True, index=pg.index)                                  # no other disease is named next to HIV
    for k in ("disease", "disease subtype", "disease status"):
        if k in pg:
            v = pg[k].astype(str); named = named & (pg[k].isna() | v.str.contains(EWAS_HIV, na=False) | v.str.lower().isin(["negative", "control", "normal", "healthy"]))
    hiv_ok = hiv & ~cancer & ~culture & named & (pg[other].isna().all(axis=1) if other else True)
    tier = np.full(len(pg), "other", object)
    tier[((st == "control") & ~cancer & ~culture & ~strict & ~hiv_ok).to_numpy()] = "control carrying a condition field"
    tier[((st == "disease tissue") & ~cancer & ~culture & ~hiv_ok).to_numpy()] = "non-cancer disease"
    tier[(culture & ~cancer).to_numpy()] = "cell culture / line"
    tier[hiv_ok.to_numpy()] = "HIV cohort (no cancer)"
    tier[strict.to_numpy()] = "clean healthy"
    tier[cancer.to_numpy()] = "CANCER"
    tier[st.isna().to_numpy()] = "no sample type"
    return tier


def ewas_table(e):
    """EWAS Data Hub's curated metadata for all ~218k samples (public JSON endpoint, about 45 requests, cached) with a disease-state tier per sample.  Raises if it cannot be fetched."""
    import urllib.request
    f = e.data / "external/ewas_table_v2.parquet"
    if f.exists():
        return pd.read_parquet(f)
    f.parent.mkdir(parents=True, exist_ok=True); keep, off, total, limit = [], 0, None, 5000
    while total is None or off < total:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(EWAS_URL.format(limit=limit, offset=off), headers={"User-Agent": "epiclock"}), timeout=120) as r:
                    d = json.loads(r.read().decode("utf-8"))
                break
            except Exception as ex:
                if attempt == 4:
                    raise
                log(f"[ewas] {type(ex).__name__}: {str(ex)[:60]} - retrying"); time.sleep(5 * (attempt + 1))
        total = int(d["total"]); pg = pd.DataFrame(d["content"]); off += limit
        if len(pg):
            t = pg.reindex(columns=["sample id", "project id", "tissue", "platform", "sex", "age (year)", "sample type", "disease"]); t["tier"] = ewas_tier(pg); keep.append(t)
        log(f"[ewas] sample table {min(off, total)} / {total}"); time.sleep(1.0)
    H = pd.concat(keep, ignore_index=True); H["sample id"] = H["sample id"].astype(str); H["age (year)"] = pd.to_numeric(H["age (year)"], errors="coerce")
    H["disease"] = H["disease"].astype(str).where(H["disease"].notna(), None); H = H.drop_duplicates("sample id")
    H.to_parquet(f); log(f"[ewas] {len(H)} samples: {H.tier.value_counts().to_dict()}"); return H


def _http(url, rng=None, timeout=120):
    import urllib.request
    h = {"User-Agent": "epiclock"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 3:
                raise
            time.sleep(2 * (a + 1))


def geo_project(gse, cache):
    """Sample descriptions (title, source, characteristics) and series text of one GEO series, from the headers of its series-matrix files (cached)."""
    import zlib
    f = cache / f"{gse}.json"
    if f.exists():
        return json.load(open(f))
    d = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/"; files = re.findall(r'href="([^"]*series_matrix\.txt\.gz)"', _http(d).decode("utf-8", "ignore")); samples, series = {}, ""
    for fn in files:
        for n in (1 << 20, 6 << 20, 24 << 20):
            raw = _http(d + fn, (0, n - 1), timeout=180); z = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try:
                txt = z.decompress(raw).decode("utf-8", "ignore")
            except zlib.error:
                txt = ""
            if "!series_matrix_table_begin" in txt or len(raw) < n:
                break
        hdr = {}
        for line in txt.split("\n"):
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!"):
                k, _, v = line.partition("\t"); hdr.setdefault(k, []).append(v)
        acc = [x.strip('"') for x in hdr.get("!Sample_geo_accession", [""])[0].split("\t")]; per = [[] for _ in acc]
        for k in ("!Sample_title", "!Sample_source_name_ch1", "!Sample_characteristics_ch1", "!Sample_description", "!Sample_treatment_protocol_ch1"):
            for line in hdr.get(k, []):
                vals = [x.strip('"') for x in line.split("\t")]
                for i in range(min(len(acc), len(vals))):
                    per[i].append(vals[i])
        samples.update(dict(zip(acc, per)))
        series = series or " ".join(" ".join(hdr.get(k, [])) for k in ("!Series_title", "!Series_summary", "!Series_overall_design")).replace('"', "")
    res = dict(samples=samples, series=series); json.dump(res, open(f, "w")); return res


def geo_string_kind(s):
    key, sep, val = s.partition(":"); v = (val if sep else s).strip().lower()
    if sep and v in GEO_NEG_VALUES:
        return ""
    if EWAS_CANCER.search(s) and not GEO_HEALTHY.search(s):
        return "cancer"
    if GEO_DISEASE.search(s) and not GEO_HEALTHY.search(s) and not (EWAS_HIV.search(s) and not EWAS_CANCER.search(s)):
        return "disease"
    if GEO_CULTURE.search(s):                           # cell culture / in-vitro / time-course material is not a physiological human sample
        return "culture"
    return ""


def geo_review(e, projects):
    """-> ({sample id: '' | 'disease' | 'cancer'}, {project: cancer-themed?}).  Samples whose GEO description cannot be fetched are absent (= unverified = not used)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = e.data / "external/geo_descriptions"; cache.mkdir(parents=True, exist_ok=True)
    todo = [p for p in projects if p.startswith("GSE") and not (cache / f"{p}.json").exists()]; failed = []
    if todo:
        log(f"[geo] fetching the sample descriptions of {len(todo)} GEO series (second opinion on health status)")
        def job(g):
            try:
                geo_project(g, cache); return g, None
            except Exception as ex:
                return g, f"{type(ex).__name__}: {str(ex)[:60]}"
        with ThreadPoolExecutor(6) as pool:
            for i, fu in enumerate(as_completed([pool.submit(job, g) for g in todo])):
                g, err = fu.result()
                if err:
                    failed.append((g, err))
                if (i + 1) % 200 == 0:
                    log(f"[geo] {i + 1}/{len(todo)} series fetched, {len(failed)} failed")
        if failed:
            log(f"[geo] {len(failed)} series could not be fetched ({failed[:3]}); their samples are treated as unverified and not used")
    kinds, themed = {}, {}
    for p in projects:
        f = cache / f"{p}.json"
        if not f.exists():
            continue
        d = json.load(open(f)); themed[p] = bool(CANCER_THEMED.search(d["series"]))
        for gsm, strs in d["samples"].items():
            ks = [geo_string_kind(s) for s in strs]; k = "cancer" if "cancer" in ks else "disease" if "disease" in ks else "culture" if "culture" in ks else ""
            kinds[gsm] = k if (k or gsm not in kinds) else kinds[gsm]
    return kinds, themed


def masked_nn(Xa, Xb, min_overlap):
    """For every row of Xa: RMS difference to the closest row of Xb over the CpGs measured in both (uint8 rows, 255 = missing); rows sharing fewer than min_overlap CpGs are ignored."""
    def prep(X):
        return torch.as_tensor(np.where(X != 255, X / 100.0, 0.0).astype(np.float32)), torch.as_tensor((X != 255).astype(np.float32))
    xa, ma = prep(Xa); xb, mb = prep(Xb); best = np.full(len(xa), 9.0, np.float32); xa2, xb2 = xa ** 2, xb ** 2
    for s in range(0, len(xa), 1000):
        a, am, a2 = xa[s:s + 1000], ma[s:s + 1000], xa2[s:s + 1000]
        n = am @ mb.T; ss = a2 @ mb.T + am @ xb2.T - 2 * (a @ xb.T)
        rmsd = torch.sqrt(torch.clamp(ss, min=0) / torch.clamp(n, min=1)); rmsd[n < min_overlap] = 9.0
        best[s:s + 1000] = rmsd.min(1).values.numpy()
    return best


def download(url, dest, what):
    """Resumable download (HTTP Range); keeps a .part file so an interrupted transfer continues where it stopped."""
    import urllib.error
    import urllib.request
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_name(dest.name + ".part"); tries = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        try:
            req = urllib.request.Request(url, headers=dict({"User-Agent": "epiclock"}, **({"Range": f"bytes={have}-"} if have else {})))
            with urllib.request.urlopen(req, timeout=120) as r:
                clen = int(r.headers.get("Content-Length") or 0); resumed = r.status == 206 and have > 0
                if not resumed:
                    have = 0
                total = have + clen
                if clen and shutil.disk_usage(dest.parent).free < clen * 1.15:
                    sys.exit(f"Not enough disk space for {what}: need {clen / 2**30:.0f} GB, only {shutil.disk_usage(dest.parent).free / 2**30:.0f} GB free in {dest.parent}.")
                log(f"[download] {what}: {total / 2**30:.2f} GB" + (f" (resuming at {have / 2**30:.2f} GB)" if resumed else ""))
                done, nxt = have, have + (1 << 30)
                with open(part, "ab" if resumed else "wb") as f:
                    while True:
                        blk = r.read(1 << 23)
                        if not blk:
                            break
                        f.write(blk); done += len(blk)
                        if done >= nxt:
                            log(f"[download] {what}: {done / 2**30:.1f} / {total / 2**30:.1f} GB"); nxt += 1 << 30
            if total and part.stat().st_size < total:
                raise OSError("connection closed early")
            os.replace(part, dest); return dest
        except urllib.error.HTTPError as ex:
            if ex.code == 416 and part.exists():
                os.replace(part, dest); return dest
            raise
        except (OSError, EOFError) as ex:
            tries += 1
            if tries > 8:
                raise
            log(f"[download] {what}: {type(ex).__name__}: {str(ex)[:80]} - retrying ({tries}/8)"); time.sleep(5 * tries)


def heldout_ids(data):
    """Sample ids of the leukaemia/AML/MDS series that the bundle holds out of every training set."""
    hf = data / "geo/holdout.txt"; hold = set(hf.read_text().split()) if hf.exists() else set(); out = set()
    for sub in ("meta", "meta_unlabeled"):
        for f in sorted((data / "geo" / sub).glob("GSE*.parquet")):
            m = pd.read_parquet(f)
            if "DatasetID" in m:
                out |= set(m.index.astype(str)[m.DatasetID.isin(hold).to_numpy()])
    return out


def ingest_methylgpt(e, accept, tier_of, known_ids):
    """Stream the MethylGPT / EWAS Data Hub / ClockBase set (226k profiles x 49,156 CpGs; 23 GB) and keep ONLY the profiles in `accept`, restricted to the 3,709 CpGs shared with our panel.
    Counts every profile it sees by disease tier, so the audit covers the whole set, not only what is kept."""
    import io
    import tarfile
    import hashlib
    import pyarrow.parquet as pq
    d = e.data / "external"; key = hashlib.sha256("\n".join(sorted(accept)).encode()).hexdigest()[:16]; out = d / f"methylgpt_clean_{key}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True); return z["X"], z["ids"].astype(str), z["pidx"], json.loads(str(z["stats"]))
    src = os.environ.get("EPICLOCK_MG_SOURCE")
    if src is None:
        src = download(MG_URL, d / "processed_type3_parquet_shuffled.tar.gz", "MethylGPT pre-training set")
    src = Path(src)
    probes = pd.read_csv(download(MG_PROBES_URL, d / "probe_ids_type3.csv", "MethylGPT CpG list"), index_col=0).iloc[:, 0].astype(str)
    pos = pd.Index(probes).get_indexer(e.c.cpgs); pidx = np.flatnonzero(pos >= 0); pcols = pos[pidx]
    log(f"[ingest] {len(pidx)} of the {len(e.c.cpgs)} panel CpGs are among the {len(probes)} CpGs of the MethylGPT set; keeping only the {len(accept)} accepted ids")
    def shards():
        if src.is_dir():
            for f in sorted(src.rglob("*.parquet")):
                yield f.name, f
        else:
            with tarfile.open(src, "r|gz") as tar:
                for m in tar:
                    if m.isfile() and m.name.endswith(".parquet"):
                        yield m.name, io.BytesIO(tar.extractfile(m).read())
    Xs, ids_all, seen, st, by_tier = [], [], set(), collections.Counter(), collections.Counter(); t0 = time.time()
    try:
        for name, fobj in shards():
            try:
                pf = pq.ParquetFile(fobj)
                for batch in pf.iter_batches(batch_size=500, columns=["id", "data"]):
                    ids = np.array(batch.column("id").to_pylist(), dtype=object).astype(str); n = len(ids); st["profiles_seen"] += n
                    for i in ids:
                        by_tier["already in the bundle / benchmark / held-out series" if i in known_ids else tier_of.get(i, "not in the EWAS table (no health annotation)")] += 1
                    ok = np.array([i in accept for i in ids], dtype=bool)
                    if not ok.any():
                        continue
                    v = batch.column("data").flatten().to_numpy(zero_copy_only=False).reshape(n, -1)[:, pcols][ok]
                    X = np.where(np.isfinite(v) & (v >= 0) & (v <= 1), np.rint(v * 100), 255).astype(np.uint8); good = (X != 255).mean(1) >= 0.5
                    st["too_few_cpgs"] += int((~good).sum()); sel = ids[ok][good]; fresh = np.array([i not in seen for i in sel], dtype=bool)
                    Xs.append(X[good][fresh]); ids_all.append(sel[fresh]); seen.update(sel[fresh].tolist())
            except Exception as ex:                     # a damaged / truncated shard: keep what was read, go on
                log(f"[ingest] {name}: {type(ex).__name__}: {str(ex)[:80]} - skipped the rest of this shard")
            log(f"[ingest] {name[:40]}: kept {sum(len(x) for x in Xs)} so far of {st['profiles_seen']} read ({(time.time() - t0) / 60:.0f} min)")
    except (tarfile.ReadError, EOFError, OSError) as ex:
        log(f"[ingest] archive ended early ({type(ex).__name__}); using the shards read so far")
    if not Xs or not sum(len(x) for x in Xs):
        sys.exit("[ingest] none of the accepted profiles was found in the MethylGPT set.")
    X, ids = np.concatenate(Xs), np.concatenate(ids_all); stats = dict(st, by_tier=dict(by_tier))
    np.savez_compressed(out, X=X, ids=ids, pidx=pidx, stats=json.dumps(stats)); log(f"[ingest] kept {len(X)} clean profiles of {st['profiles_seen']} read"); return X, ids, pidx, stats


def build_clean_corpus(e):
    """The one training corpus: ComputAge training studies + bundle GEO samples + MethylGPT profiles, each kept only if every check passes; ComputAge benchmark rows (test) untouched."""
    c = load_corpus(e.data); e.c = c; m = c.meta; ids = m.index.astype(str); role = m.role.to_numpy(); proj = m.DatasetID.astype(str).to_numpy(); P = len(c.cpgs)
    H = ewas_table(e); tier_of = dict(zip(H["sample id"], H["tier"])); bench = np.flatnonzero(role == "benchmark")
    bench_proj, bench_ids, held = set(proj[bench]), set(ids[bench]), heldout_ids(e.data)
    cand = H[H.tier.isin(GOOD_TIERS)]; projects = sorted(set(proj[np.isin(role, ["train", "geo", "unlabeled"])]) | set(cand["project id"].astype(str)))
    kinds, themed = geo_review(e, projects)
    # ---- exact duplicates of benchmark samples inside the bundle (cached)
    bundle_rows = np.flatnonzero(np.isin(role, ["train", "geo", "unlabeled"])); dupf = e.data / "external/bundle_benchmark_duplicates.npz"
    if dupf.exists() and len(np.load(dupf)["rows"]) == len(bundle_rows):
        dup = np.load(dupf)["dup"]
    else:
        log(f"[audit] comparing {len(bundle_rows)} bundle profiles with the {len(bench)} benchmark profiles to find re-deposited copies"); dup = masked_nn(c.X[bundle_rows], c.X[bench], 2500) < DUP_RMSD; np.savez(dupf, rows=bundle_rows, dup=dup)
    # ---- decision per bundle row (the first failing check is the recorded reason)
    reason = np.full(len(m), "", object); dupfull = np.zeros(len(m), bool); dupfull[bundle_rows] = dup
    for i in bundle_rows:
        r = role[i]; sid = ids[i]; tier = tier_of.get(sid, "not in the EWAS table"); kind = kinds.get(sid); by_design = r == "train"; why = ""
        if tier == "CANCER": why = "EWAS: cancer / tumour / adjacent normal"
        elif tier in ("non-cancer disease", "cell culture / line"): why = "EWAS: non-cancer disease or cell culture"
        elif not by_design and tier == "control carrying a condition field": why = "EWAS: control that carries a condition field"
        elif not by_design and tier not in GOOD_TIERS: why = "no health annotation (not in the EWAS table)"
        elif kind is None: why = "GEO description unavailable (cannot cross-check)"
        elif kind in ("cancer", "culture") or (kind == "disease" and not by_design): why = "GEO description names cell culture / in-vitro material" if kind == "culture" else f"GEO description names {kind}"
        elif themed.get(proj[i], False): why = "cancer-themed study"
        elif proj[i] in bench_proj: why = "same study as a benchmark dataset"
        elif dupfull[i]: why = "exact copy of a benchmark sample"
        reason[i] = why
    keep_bundle = np.array([r in ("train", "geo", "unlabeled") and reason[i] == "" for i, r in enumerate(role)])
    # ---- candidates from the MethylGPT set
    known = set(ids) | bench_ids | held; cand_ok = cand[~cand["sample id"].isin(known)].copy()
    cand_ok["kind"] = cand_ok["sample id"].map(kinds); cand_ok["themed"] = cand_ok["project id"].astype(str).map(themed).fillna(False); cand_ok["bench"] = cand_ok["project id"].astype(str).isin(bench_proj)
    cand_ok = cand_ok[(cand_ok.kind == "") & ~cand_ok.themed.astype(bool) & ~cand_ok.bench & cand_ok["project id"].astype(str).str.startswith("GSE")]
    accept = set(cand_ok["sample id"]); Xe, ide, pidx, mg_stats = ingest_methylgpt(e, accept, tier_of, known)
    Xfull = np.full((len(Xe), P), 255, np.uint8); Xfull[:, pidx] = Xe
    dup_e = masked_nn(Xfull, c.X[bench], 1500) < DUP_RMSD; log(f"[audit] {int(dup_e.sum())} of {len(Xe)} extra profiles are copies of benchmark samples and are dropped"); Xfull, ide = Xfull[~dup_e], ide[~dup_e]
    Hi = H.set_index("sample id").loc[ide]; age = Hi["age (year)"].where(Hi["age (year)"].between(0, 110)).to_numpy(float)
    sex = Hi["sex"].astype(str).str.upper().str[0]
    meta_e = pd.DataFrame({"DatasetID": Hi["project id"].astype(str).to_numpy(), "Gender": sex.where(sex.isin(["M", "F"])).to_numpy(),
                           "Age": age, "Condition": np.where(Hi.tier.to_numpy() == "HIV cohort (no cancer)", "HIV", "HC"), "Class": np.where(Hi.tier.to_numpy() == "HIV cohort (no cancer)", "ISD", "HC"),
                           "Tissue": Hi["tissue"].to_numpy(), "Platform": Hi["platform"].to_numpy(), "role": np.where(np.isfinite(age), "geo", "unlabeled")}, index=ide)
    keep = keep_bundle | (role == "benchmark")
    c.X = np.concatenate([c.X[keep], Xfull]); c.meta = pd.concat([m[keep], meta_e.reindex(columns=m.columns)])
    e.audit = dict(reason=pd.DataFrame({"role": role, "reason": reason, "tier": [tier_of.get(s, "not in the EWAS table") for s in ids]}, index=ids), H=H, mg=mg_stats, n_ext=int(len(ide)), n_ext_age=int(np.isfinite(age).sum()),
                   cand=len(cand), cand_ok=len(cand_ok), role_final=c.meta.role.value_counts().to_dict())
    write_audit(e, c); return c


def write_audit(e, c):
    A = e.audit; H = A["H"]; R = A["reason"]; L = ["# Data audit - what the training corpus is made of", "",
         "Training uses ONLY profiles that pass every check. Everything else is counted below with its reason and is not used. The test set (ComputAge benchmark healthy controls) is never trained on.", "",
         "## 1. The universe: EWAS Data Hub curated metadata (the master list behind the MethylGPT set), by disease state", "", "| disease-state tier | samples | with an age |", "|---|---|---|"]
    for t, n in H.tier.value_counts().items():
        L.append(f"| {t} | {n} | {int(H[H.tier == t]['age (year)'].between(0, 110).sum())} |")
    L += ["", "Tier rules: 'clean healthy' = curated control with none of ~150 condition-bearing fields filled and not a cell culture; 'HIV cohort' = HIV-only cohorts (accepted); 'CANCER' = tumour, adjacent normal, cancer projects or any cancer word; "
          "'control carrying a condition field' = labelled control but with a treatment / disease / stage / ... field filled in (not used).", "", "## 2. The bundle (45,280 profiles): attrition by source", ""]
    for role, nm in (("train", "ComputAge Bench training studies"), ("geo", "bundle: labelled GEO"), ("unlabeled", "bundle: unlabelled GEO")):
        r = R[R.role == role]; L += [f"**{nm}**: {len(r)} profiles", ""] + [f"- removed - {k}: {v}" for k, v in r.reason[r.reason != ""].value_counts().items()] + [f"- **kept: {int((r.reason == '').sum())}**", ""]
    mg = A["mg"]; L += ["## 3. The MethylGPT / EWAS Data Hub / ClockBase set (%d profiles read), by what we know about each" % mg["profiles_seen"], "", "| status | profiles |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(mg["by_tier"].items(), key=lambda x: -x[1])]
    L += ["", f"Candidates passing the EWAS tier, the GEO-description check, the cancer-themed-study check and the benchmark-study check: {A['cand_ok']} ids; found in the set: {A['n_ext']} kept after removing copies of benchmark samples ({A['n_ext_age']} with an age).", "",
          "## 4. The training corpus", "", f"- labelled (have an age; fine-tuning and ridge/elastic-net baselines): {int(((c.meta.role.isin(['train', 'geo'])) & c.meta.Age.notna()).sum())}",
          f"- label-free only (pre-training): {int((c.meta.role == 'unlabeled').sum())}", f"- test (ComputAge benchmark rows, never trained on): {int((c.meta.role == 'benchmark').sum())}",
          f"- profiles that are partly measured (3,709 of the 9,098 panel CpGs): {A['n_ext']}", "",
          "Not used, by design: cancer; non-cancer disease; controls carrying a condition field; cell cultures; anything without a health annotation; anything whose GEO description names a disease; cancer-themed studies; copies of test samples.", ""]
    txt = "\n".join(L); (e.out / "data_audit.md").write_text(txt); R.to_csv(e.out / "provenance_bundle.csv.gz"); e.audit_text = txt; log("[audit] wrote data_audit.md")


def dedupe_validation(e):
    """Validation must not contain copies of training profiles (a copy would make early stopping look better than it is)."""
    va, tr = e.sp["val"], e.sp["train"]; drop = masked_nn(e.c.X[va], e.c.X[tr], 1500) < DUP_RMSD
    if drop.any():
        log(f"[audit] {int(drop.sum())} validation profiles are copies of training profiles and are removed from validation"); e.sp["val"] = va[~drop]


def make_env(a):
    global LOGF, CTRL
    e = Env(); e.a = a; e.out = Path(a.out); e.out.mkdir(parents=True, exist_ok=True); LOGF = e.out / "study.log"; CTRL = e.out
    torch.set_num_threads(a.threads or max(1, (os.cpu_count() or 4) // 2))
    try:
        torch.backends.mha.set_fastpath_enabled(False)          # the eval-mode transformer fast path has had MPS bugs
    except AttributeError:
        pass
    e.dev = get_device(a.device); e.mem = total_mem_gb(); e.limit = a.memory_fraction * e.mem; e.selftested = False
    e.data = find_data(a.data); e.c = build_clean_corpus(e); e.sp = make_splits(e.c, a.seed); dedupe_validation(e); e.mean, e.std = panel_stats(e.c.X, e.sp["train"])
    e.ctx = genome_context(e.data, e.c.cpgs); e.age = e.c.meta.Age.to_numpy(float); e.rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    sp = e.sp
    log(f"EpiClock study | device {e.dev} | machine memory {e.mem:.0f} GB (training budget {e.limit:.0f} GB) | data {e.data}\n"
        f"corpus {len(e.c.X)} samples x {len(e.c.cpgs)} CpGs | train {len(sp['train'])} | val {len(sp['val'])} ({len(sp['val_series'])} series) | "
        f"pretrain pool {len(sp['ssl'])} | test: {len(sp['test'])} healthy controls of {e.c.meta.iloc[sp['test']].DatasetID.nunique()} held-out studies "
        f"(+{len(sp['test_all']) - len(sp['test'])} disease samples for the age-acceleration benchmark)")
    return e


def preset(e, size):
    return PRESETS["tiny" if e.a.smoke else size]


def parse_arm(name):
    p = name.split("-")
    if name == "linear-tuned":
        return dict(name=name, size="-", kind="linear", seed=0, context=False, linear=True, pretrain=False)
    if len(p) < 2 or p[0] not in PRESETS or p[1] not in KINDS or (len(p) > 2 and not re.fullmatch(r"s\d+", p[2])):
        sys.exit(f"Unknown arm '{name}'. Arms are <size>-<kind>[-s<seed>] with size in {list(PRESETS)} and kind in {list(KINDS)}, or 'linear-tuned'.")
    k = p[1]
    return dict(name=name, size=p[0], kind=k, seed=int(p[2][1:]) if len(p) > 2 else 0, context=k != "noctx", linear=k != "pure", pretrain=k in ("pre", "noctx", "cons", "rob"))


def legacy_preset(e):
    """Runs made by the first version of this script keep their files directly in the output folder."""
    try:
        return json.load(open(e.out / "results.json"))["preset"] if (e.out / "model.pt").exists() else None
    except Exception:
        return None


def arm_dir(e, ar):
    return e.out if (ar["kind"] == "pre" and ar["seed"] == 0 and legacy_preset(e) == ar["size"]) else e.out / "arms" / ar["name"]


def pre_dir(e, size, ctx):
    return e.out if (ctx and legacy_preset(e) == size and (e.out / "pretrain_best.pt").exists()) else e.out / "pre" / (size if ctx else size + "-noctx")


def mark_done_from_old_files(d):
    """A folder written by the first version holds pretrain.json / results.json instead of the *_done.json markers."""
    for src, key, dst in (("pretrain.json", None, "pretrain_done.json"), ("results.json", "finetune", "finetune_done.json")):
        f = d / src
        if f.exists() and not (d / dst).exists():
            r = json.load(open(f)); r = r.get(key) if key else r
            if r and r.get("reason") not in (None, "stop-file"):
                json.dump(r, open(d / dst, "w"))


def sanity_check_device(e, model, K):
    if e.dev.type != "cpu" and not e.a.no_selftest and not e.selftested:
        selftest(model, e.c, e.sp, K, e.dev, e.rng)
    e.selftested = True


def make_cfg(e, ar):
    P = preset(e, ar["size"]); return dict(d=P["d"], layers=P["layers"], heads=P["heads"], context=ar["context"], linear_skip=ar["linear"], dropout=P["dropout"])


def ensure_pretrained(e, size, ctx):
    """Masked-CpG reconstruction on the label-free pool (shared by every arm of the same size and context setting)."""
    d = pre_dir(e, size, ctx); d.mkdir(parents=True, exist_ok=True); mark_done_from_old_files(d)
    if (d / "pretrain_done.json").exists():
        return d / "pretrain_best.pt"
    P = preset(e, size); K = P["K"]; cfg = make_cfg(e, dict(size=size, context=ctx, linear=True))
    log(f"=== pre-training {SIZE_NAMES[size]}{'' if ctx else ' (no genome-context inputs)'}: d={cfg['d']} layers={cfg['layers']} K={K} batch={P['batch']}")
    model = EpiClock(len(e.c.cpgs), e.mean, e.std, e.ctx, cfg).to(e.dev); log(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    sanity_check_device(e, model, K)
    budget = Budget(P["batch"], e.dev, e.limit); rng = np.random.default_rng(e.a.seed + 3)
    def trial(m, n):
        sel = rng.choice(e.sp["ssl"], n); idx, ib, valid, is_mask, beta = ssl_batch(e.c.X[sel], K, rng)
        return m.reconstruct(idx, ib, valid, is_mask).mean()
    model.train(); budget.fit(model, trial)
    w_ssl = sample_weights(e.c.meta, e.sp["ssl"], 0.5, 0.0)
    vrows = e.c.X[e.sp["val"][np.random.RandomState(5).choice(len(e.sp["val"]), min(1500, len(e.sp["val"])), replace=False)]]
    base_mae = float(np.nanmean(np.abs(np.where(vrows != 255, vrows / 100.0, np.nan) - e.mean)))
    def ssl_step(m, n):
        rows = e.c.X[rng.choice(e.sp["ssl"], n, p=w_ssl)]; idx, ib, valid, is_mask, beta = ssl_batch(rows, K, rng)
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
    r1 = run_phase("pretrain", model, ssl_step, ssl_eval, d, e.dev, P["lr"], max_steps=e.a.ssl_steps, eval_every=e.a.eval_every, patience=e.a.patience, max_hours=P["pre_h"], budget=budget)
    json.dump(dict(r1, baseline_mae=base_mae), open(d / "pretrain.json", "w"), indent=1)
    if r1["reason"] == "stop-file":
        raise Paused()
    log(f"[pretrain] reconstruction MAE {r1['best']:.4f} vs mean-baseline {base_mae:.4f} ({1 - r1['best'] / base_mae:.0%} lower)")
    del model; free_mem(e.dev)
    return d / "pretrain_best.pt"


def age_fns(e, K, rng, kind):
    cons, rob = kind == "cons", kind == "rob"
    tr_rows = e.sp["train"]; w = sample_weights(e.c.meta, tr_rows, 0.5, 0.5); vsel = e.sp["val"]; sl1 = nn.functional.smooth_l1_loss
    ust, inv = np.unique(e.c.meta.DatasetID.to_numpy()[tr_rows].astype(str), return_inverse=True); by_study = [tr_rows[inv == k] for k in range(len(ust))]
    p_study = np.sqrt([len(r) for r in by_study]); p_study /= p_study.sum()
    def step(m, n):
        if rob and n >= 8:                          # G studies x n/G samples each; penalise the spread of the per-study MEAN error (the study offset)
            G = min(8, n // 2); per = n // G; ks = rng.choice(len(by_study), G, replace=False, p=p_study)
            sel = np.concatenate([rng.choice(by_study[k], per) for k in ks]); rows = e.c.X[sel]; idx, beta, valid, _ = draw(rows, K, rng); fz = rowz(rows, e.mean, e.std)
            y = torch.as_tensor(age_to_t(e.age[sel]), dtype=torch.float32, device=m.cls.device); pred = m.age_t(idx, beta, valid, fz); err = pred - y
            grp = torch.as_tensor(np.repeat(np.arange(G), per), device=err.device); gm = torch.stack([err[grp == g].mean() for g in range(G)])
            return sl1(pred, y, beta=0.05) + ROB_LAMBDA * gm.var()
        if cons and n >= 2:                         # two independent CpG subsets of the same people: both fit the age, and they must agree
            sel = rng.choice(tr_rows, n // 2, p=w); rows = e.c.X[sel]; fz = rowz(rows, e.mean, e.std)
            y = torch.as_tensor(age_to_t(e.age[sel]), dtype=torch.float32, device=m.cls.device)
            i1, b1, v1, _ = draw(rows, K, rng); i2, b2, v2, _ = draw(rows, K, rng); p1, p2 = m.age_t(i1, b1, v1, fz), m.age_t(i2, b2, v2, fz)
            return 0.5 * (sl1(p1, y, beta=0.05) + sl1(p2, y, beta=0.05)) + CONS_LAMBDA * sl1(p1, p2, beta=0.05)
        sel = rng.choice(tr_rows, n, p=w); rows = e.c.X[sel]; idx, beta, valid, _ = draw(rows, K, rng); fz = rowz(rows, e.mean, e.std)
        y = torch.as_tensor(age_to_t(e.age[sel]), dtype=torch.float32, device=m.cls.device)
        return sl1(m.age_t(idx, beta, valid, fz), y, beta=0.05)
    def ev(m):
        p = t_to_age(predict_t(m, e.c.X[vsel], K, e.a.val_draws, np.random.default_rng(13))); return float(np.median(np.abs(p - e.age[vsel])))
    return step, ev


def train_arm(e, ar, d):
    """Fine-tune on age (Horvath-transformed) from the ridge-initialised linear core; -> path of model.pt"""
    mp = d / "model.pt"; mark_done_from_old_files(d)
    if mp.exists() and (d / "finetune_done.json").exists():
        return mp
    P = preset(e, ar["size"]); K = P["K"]; cfg = make_cfg(e, ar); pre = ensure_pretrained(e, ar["size"], ar["context"]) if ar["pretrain"] else None
    log(f"=== fine-tuning {ar['name']}: {KINDS[ar['kind']]}  (seed {ar['seed']})")
    torch.manual_seed(e.a.seed + 101 * ar["seed"]); rng = np.random.default_rng(e.a.seed + 101 * ar["seed"] + 7)
    model = EpiClock(len(e.c.cpgs), e.mean, e.std, e.ctx, cfg).to(e.dev); log(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    sanity_check_device(e, model, K)
    budget = Budget(P["batch"], e.dev, e.limit)
    def trial(m, n):
        sel = rng.choice(e.sp["ssl"], n); idx, ib, valid, is_mask, beta = ssl_batch(e.c.X[sel], K, rng)
        return m.reconstruct(idx, ib, valid, is_mask).mean()
    model.train(); budget.fit(model, trial); rg = get_ridge(e)
    def warm_start(m):
        m.bias.data.fill_(rg["b"])
        if m.lin_on:
            m.lin_u.data.copy_(torch.as_tensor(rg["w"] / LIN_SCALE)); nn.init.zeros_(m.age_head.weight); nn.init.zeros_(m.age_head.bias)
            log("[finetune] linear clock initialised from ridge; transformer correction starts at zero")
    step, ev = age_fns(e, K, rng, ar["kind"])
    r2 = run_phase("finetune", model, step, ev, d, e.dev, P["lr"] * (0.5 if pre else 1.0), max_steps=e.a.ft_steps, eval_every=e.a.eval_every, patience=e.a.patience,
                   max_hours=P["ft_h"], init_from=pre, post_init=warm_start, budget=budget)
    if r2["reason"] == "stop-file":
        raise Paused()
    best = torch.load(d / "finetune_best.pt", map_location="cpu", weights_only=False)
    torch.save(dict(state=best["state"], cfg=cfg, K=K, cpgs=e.c.cpgs, mean=e.mean, std=e.std, ctx=e.ctx if cfg["context"] else None, arm=ar["name"], pretrained=pre is not None), mp)
    del model; free_mem(e.dev)
    return mp


class LinearClock(nn.Module):
    """The linear core on its own: same parameters, loss and training recipe as the fine-tuning stage, but no transformer.  It is the control
    that separates 'the transformer helps' from 'training the linear weights a bit more helps'."""

    def __init__(self, rg):
        super().__init__(); self.cfg = {}
        self.lin_u = nn.Parameter(torch.as_tensor(rg["w"] / LIN_SCALE)); self.bias = nn.Parameter(torch.tensor(float(rg["b"])))

    def forward(self, fz):
        return self.bias + LIN_SCALE * (fz @ self.lin_u)


class FixedBudget:
    def __init__(self, n):
        self.micro, self.accum = n, 1

    def shrink(self):
        return False


def fit_linear_tuned(e, d):
    log("=== linear-tuned control: the linear core trained with the fine-tuning recipe, no transformer")
    rg = get_ridge(e); rng = np.random.default_rng(e.a.seed + 5); tr = e.sp["train"]; w = sample_weights(e.c.meta, tr, 0.5, 0.5); vsel = e.sp["val"]
    def step(m, n):
        sel = rng.choice(tr, n, p=w); y = torch.as_tensor(age_to_t(e.age[sel]), dtype=torch.float32)
        return nn.functional.smooth_l1_loss(m(rowz(e.c.X[sel], e.mean, e.std)), y, beta=0.05)
    def ev(m):
        with torch.no_grad():
            p = t_to_age(m(rowz(e.c.X[vsel], e.mean, e.std)).double().numpy())
        return float(np.median(np.abs(p - e.age[vsel])))
    m = LinearClock(rg); cpu = torch.device("cpu")
    r = run_phase("linear", m, step, ev, d, cpu, 1e-4, max_hours=0.5, budget=FixedBudget(128))
    m.load_state_dict(torch.load(d / "linear_best.pt", map_location="cpu", weights_only=False)["state"]); out = []
    with torch.no_grad():
        for s in range(0, len(e.sp["test_all"]), 2000):
            out.append(t_to_age(m(rowz(e.c.X[e.sp["test_all"][s:s + 2000]], e.mean, e.std)).double().numpy()))
    np.savez(d / "preds.npz", pred=np.concatenate(out))
    if r["reason"] == "stop-file":
        raise Paused()


def predict_arm(e, ar, mp, d):
    model, ck = load_model(mp, e.dev); K = ck["K"]; rows = e.c.X[e.sp["test_all"]]
    log(f"[predict] {ar['name']}: {len(rows)} benchmark samples x {e.a.test_draws} CpG subsets of {K}")
    out = dict(pred=t_to_age(predict_t(model, rows, K, e.a.test_draws, np.random.default_rng(17))))
    if model.lin_on:                                          # decomposition: what the trained linear part alone predicts (no transformer correction)
        lin = []
        with torch.no_grad():
            for s in range(0, len(rows), 2000):
                fz = rowz(rows[s:s + 2000], ck["mean"], ck["std"]).to(e.dev); lin.append(t_to_age((model.bias + LIN_SCALE * (fz @ model.lin_u)).detach().cpu().double().numpy()))
        out["lin"] = np.concatenate(lin)
    if ar["pretrain"]:                                        # the reconstruction head only means something after pre-training
        out["recon"] = recon_error(model, rows, K, 2, np.random.default_rng(19))
    np.savez(d / "preds.npz", **out); del model; free_mem(e.dev)


def run_arm(e, name):
    ar = parse_arm(name); d = arm_dir(e, ar); d.mkdir(parents=True, exist_ok=True)
    if (d / "preds.npz").exists():
        return
    if ar["kind"] == "linear":
        fit_linear_tuned(e, d); return
    mp = train_arm(e, ar, d); predict_arm(e, ar, mp, d)


# ======================================================================================================== report
def finished_arms(e):
    names = set(); arms = e.out / "arms"
    if arms.exists():
        names |= {p.name for p in arms.iterdir() if (p / "preds.npz").exists()}
    lp = legacy_preset(e)
    if lp and (e.out / "preds.npz").exists():
        names.add(f"{lp}-pre")
    order = {s: i for i, s in enumerate(["linear-tuned"] + list(PRESETS))}
    return sorted(names, key=lambda n: (order.get(n.split("-")[0], 99) if n != "linear-tuned" else -1, n))


def load_arm_preds(e, name):
    z = np.load(arm_dir(e, parse_arm(name)) / "preds.npz"); return {k: z[k] for k in z.files}


def param_count(e, size):
    cfg = make_cfg(e, dict(size=size, context=True, linear=True))
    return sum(p.numel() for p in EpiClock(len(e.c.cpgs), e.mean, e.std, e.ctx, cfg).parameters())


PAIRS = [("{s}-pre", "linear-tuned", "transformer (pre-trained) vs the linear core alone"),
         ("{s}-scratch", "linear-tuned", "transformer (no pre-training) vs the linear core alone"),
         ("{s}-pre", "{s}-pre [linear part only]", "the transformer's correction itself (same model, correction removed)"),
         ("{s}-pre", "{s}-scratch", "value of pre-training"),
         ("{s}-pre", "{s}-noctx", "value of the genome-context inputs"),
         ("{s}-scratch", "{s}-pure", "value of the ridge-initialised linear core"),
         ("{s}-cons", "{s}-pre", "consistency fine-tuning"), ("{s}-rob", "{s}-pre", "cross-study robustness fine-tuning (penalty on the spread of per-study mean error)"),
         ("{s}-pre-s1", "{s}-pre", "seed 1 vs seed 0 (training noise)"), ("{s}-pre-s2", "{s}-pre", "seed 2 vs seed 0 (training noise)")]


def disease_table(e, scores):
    """Within every (study, condition) with >= 10 patients and >= 10 healthy controls of the same study: remove the age trend of the score (line fitted on
    that study's controls), then the AUC for patients vs controls (0.5 = no signal).  Reported per disease class with a bootstrap over (study, condition) units."""
    from scipy.stats import rankdata
    meta = e.c.meta.iloc[e.sp["test_all"]]; ds, cond, cls = meta.DatasetID.to_numpy(), meta.Condition.astype(str).to_numpy(), meta.Class.astype(str).to_numpy()
    age = meta.Age.to_numpy(float); ok = meta.Tissue.isin(["Blood", "Saliva", "Buccal"]).to_numpy() & np.isfinite(age) & (age >= 18) & (age <= 90)
    rows = []
    for gse, cd in sorted({(g, c_) for g, c_ in zip(ds[ok], cond[ok]) if c_ != "HC"}):
        d_idx, h_idx = np.flatnonzero(ok & (ds == gse) & (cond == cd)), np.flatnonzero(ok & (ds == gse) & (cond == "HC"))
        if len(d_idx) < 10 or len(h_idx) < 10 or np.ptp(age[h_idx]) < 5:
            continue
        r = dict(study=gse, condition=cd, cls=cls[d_idx[0]], n_disease=len(d_idx), n_control=len(h_idx)); A_ = np.stack([np.ones(len(h_idx)), age[h_idx]], 1)
        for n, v in scores.items():
            b = np.linalg.lstsq(A_, v[h_idx], rcond=None)[0]; res_d, res_h = v[d_idx] - (b[0] + b[1] * age[d_idx]), v[h_idx] - (b[0] + b[1] * age[h_idx])
            rk = rankdata(np.concatenate([res_d, res_h])); r[n] = (rk[:len(d_idx)].sum() - len(d_idx) * (len(d_idx) + 1) / 2) / (len(d_idx) * len(h_idx))
        rows.append(r)
    if not rows:
        return None
    U = pd.DataFrame(rows); rng = np.random.default_rng(0); out = {}
    for n in scores:
        bs = [U[n].to_numpy()[rng.integers(0, len(U), len(U))].mean() for _ in range(1000)]
        out[n] = {**{c_: f"{U.loc[U.cls == c_, n].mean():.2f} (n={int((U.cls == c_).sum())})" for c_ in sorted(U.cls.unique())}, "all units": f"{U[n].mean():.3f} {ci(bs, 3)}"}
    return pd.DataFrame(out).T, len(U)


def build_report(e):
    comp = get_comparators(e); arms = finished_arms(e); P = dict(comp); ours = []
    for n in arms:
        d = load_arm_preds(e, n); P[n] = d["pred"]; ours.append(n)
        if "lin" in d:
            P[f"{n} [linear part only]"] = d["lin"]; ours.append(f"{n} [linear part only]")
    RC = {n: load_arm_preds(e, n)["recon"] for n in arms if "recon" in load_arm_preds(e, n)}
    members = [n for n in arms if n != "linear-tuned" and parse_arm(n)["kind"] == "pre" and parse_arm(n)["seed"] == 0]
    if len(members) >= 2:
        ens = "ensemble of " + "+".join(parse_arm(n)["size"] for n in members); P[ens] = np.mean([P[n] for n in members], 0); ours.append(ens)
    te_all = e.sp["test_all"]; hc = np.searchsorted(te_all, e.sp["test"]); meta = e.c.meta.iloc[e.sp["test"]]; age = meta.Age.to_numpy(float); grp = meta.DatasetID.to_numpy()
    blood = meta.Tissue.astype(str).str.contains("blood", case=False).to_numpy(); adult = (age >= 18) & (age <= 90); sib = sibling_set(e.c.meta, e.sp)
    nosib = ~meta.DatasetID.isin(sib).to_numpy(); boot = Boot(grp, e.a.boot, 0)
    A = {n: np.abs(P[n][hc] - age) for n in P}; S = {n: P[n][hc] - age for n in P}
    Ms = {n: boot.med(A[n], "samples") for n in P}; Mc = {n: boot.med(A[n], "studies") for n in P}
    rows = []
    for n in P:
        d = Mc[n] - Mc["ridge"]; off = pd.Series(S[n]).groupby(grp).transform("mean").to_numpy(); offm = pd.Series(S[n]).groupby(grp).transform("median").to_numpy()
        rows.append(dict(model=n, MedAE=np.median(A[n]), offset_share=1 - np.mean((S[n] - off) ** 2) / np.mean(S[n] ** 2), MedAE_no_offsets=np.median(np.abs(S[n] - offm)), CI_samples=ci(Ms[n]), CI_studies=ci(Mc[n]), MAE=A[n].mean(), r=np.corrcoef(P[n][hc], age)[0, 1],
                         MedAE_blood=np.median(A[n][blood]), MedAE_18_90=np.median(A[n][adult]), MedAE_no_sibling=np.median(A[n][nosib]),
                         study_avg_MedAE=pd.Series(A[n]).groupby(grp).median().mean(), vs_ridge=(f"{np.median(A[n]) - np.median(A['ridge']):+.2f} {ci(d, 2, True)}" if n != "ridge" else ""),
                         P_beats_ridge=float((d < 0).mean()) if n != "ridge" else np.nan, ours=n in ours))
    main = pd.DataFrame(rows).set_index("model").sort_values("MedAE")
    pr = []
    for s in list(PRESETS):
        for a_, b_, why in PAIRS:
            a_, b_ = a_.format(s=s), b_.format(s=s)
            if a_ in P and b_ in P and a_ != b_:
                d_s, d_c = Ms[a_] - Ms[b_], Mc[a_] - Mc[b_]
                pr.append(dict(comparison=f"{a_}  minus  {b_}", meaning=why, delta_MedAE=np.median(A[a_]) - np.median(A[b_]), CI_samples=ci(d_s, 2, True), CI_studies=ci(d_c, 2, True), P_first_better_studies=float((d_c < 0).mean())))
    pairs = pd.DataFrame(pr)
    sc = []
    for n in members:
        sz = parse_arm(n)["size"]; sc.append(dict(model=SIZE_NAMES[sz], arm=n, parameters=f"{param_count(e, sz):,}", MedAE=np.median(A[n]), CI_studies=ci(Mc[n]), MAE=A[n].mean(), r=np.corrcoef(P[n][hc], age)[0, 1]))
    scaling = pd.DataFrame(sc)
    off = official_scores(e, P)
    corp = (f"Training corpus: {int(((e.c.meta.role.isin(['train', 'geo'])) & e.c.meta.Age.notna()).sum())} labelled and {int((e.c.meta.role == 'unlabeled').sum())} label-free profiles, ALL of them passing the checks "
            f"in the data audit at the end of this report (no cancer, no disease, no copies of test samples, nothing without a health annotation).")
    lines = [f"# EpiClock study report", "", corp, "", f"Test set: {len(hc)} healthy controls from {meta.DatasetID.nunique()} held-out ComputAge benchmark studies ({int(blood.sum())} blood; "
             f"{len(sib)} studies adjacent to training series). None of these studies was used for training, model selection or early stopping.", ""]
    best = [n for n in main.index if main.loc[n, "ours"] and "[linear part only]" not in n and n != "linear-tuned"]
    if best:
        b = best[0]; d = Mc[b] - Mc["ridge"]
        lines += ["## Headline", f"- Best EpiClock result: **{b}**, median absolute error **{main.loc[b, 'MedAE']:.2f} y** (95% CI over studies {main.loc[b, 'CI_studies']}) versus ridge "
                  f"{main.loc['ridge', 'MedAE']:.2f} y {main.loc['ridge', 'CI_studies']}. Paired difference {main.loc[b, 'vs_ridge']}; it beats ridge in {float((d < 0).mean()):.0%} of study-level bootstrap draws."]
        if "linear-tuned" in P and b != "linear-tuned":
            d2 = Mc[b] - Mc["linear-tuned"]; lo, hi = np.percentile(d2, [2.5, 97.5])
            lines += [f"- Against the linear core retrained with the same recipe (the right control for 'is it the transformer?'): {np.median(A[b]) - np.median(A['linear-tuned']):+.2f} y, "
                      f"study-level 95% CI [{lo:+.2f}, {hi:+.2f}] - " + ("the interval excludes zero." if hi < 0 or lo > 0 else "the interval includes zero: not distinguishable at the study level (the sample-level interval is narrower).")]
        lines += [""]
    lines += ["## All models on the same test samples (median absolute error in years; lower is better)", "```", main.drop(columns="ours").round(3).to_string(), "```",
              "offset_share = fraction of the squared error that is a per-study offset; MedAE_no_offsets = the error if each study's median error were removed (a diagnostic, not an achievable score). CI_samples resamples individual samples; CI_studies resamples whole studies (the honest one: samples within a study share batch effects). vs_ridge is the paired "
              "difference in MedAE (negative = better than ridge) with its study-level interval. `[linear part only]` rows show what the trained linear weights predict with the transformer correction removed.", ""]
    if len(pairs):
        lines += ["## What does each ingredient add? (paired, same test samples; negative = the first model is better)", "```", pairs.round(3).to_string(index=False), "```", ""]
    if len(scaling):
        lines += ["## Model family", "```", scaling.round(3).to_string(index=False), "```", ""]
    if off is not None:
        lines += ["## ComputAge Bench official-style score (all 65 benchmark studies incl. disease cohorts)",
                  "AA2: significant age acceleration in disease vs the same study's healthy controls; AA1: significant positive acceleration in disease (both FDR-corrected); "
                  "score = AA2 + AA1 x (1 - positive median bias / MedAE). Same study list, tests and formula as the `computage` package, applied to our predictions. "
                  "Scores differ by whole tests, so differences of 1-2 points are not meaningful.", "```", off.round(2).to_string(), "```", ""]
    if RC:
        big = max(RC, key=lambda n: (list(PRESETS).index(parse_arm(n)["size"]), n)); sc_ = {"ridge: age acceleration": P["ridge"]}
        if "Horvath V2" in P:
            sc_["Horvath V2: age acceleration"] = P["Horvath V2"]
        best_ = [n for n in main.index if n in arms and n != "linear-tuned"][0]; sc_[f"EpiClock {best_}: age acceleration"] = P[best_]; sc_[f"EpiClock {big}: deviation (masked-CpG reconstruction error)"] = RC[big]
        dt = disease_table(e, sc_)
        if dt is not None:
            lines += ["## Disease signal across the benchmark studies (AUC patients vs same-study healthy controls, age trend removed; 0.5 = none)",
                      f"{dt[1]} (study, condition) units with >= 10 patients and >= 10 controls. Classes: ISD immune/inflammatory, NDD neurodegenerative, CVD cardiovascular, MSD metabolic, MBD, RSD, PGS progeroid. "
                      "The deviation score needs no age label and no disease label: it asks how badly the pre-trained model reconstructs the sample's hidden CpGs.", "```", dt[0].to_string(), "```", ""]
    pj = sorted((e.out / "probes").glob("*.json")) if (e.out / "probes").exists() else []
    for f in pj:
        r = json.load(open(f))
        if "results" not in r:
            continue
        tab = pd.DataFrame({fs: {t: f"{v['score']:.3f}" for t, v in res.items()} for fs, res in r["results"].items()})
        chance = {t: v["chance"] for t, v in next(iter(r["results"].values())).items()}; tab.insert(0, "chance", {t: f"{c:.3f}" for t, c in chance.items()})
        lines += [f"## Does the CpG identity embedding know the genome? - {f.stem}" + ("  (the model was GIVEN these annotations as input, so this is not evidence of discovery)" if r["context_input"] else "  (no annotation input: anything above the baselines was learned from methylation alone)"),
                  "Linear probe, 5-fold cross-validation over CpGs; balanced accuracy for multi-class targets, AUC for yes/no targets.", "```", tab.to_string(), "```", ""]
    for f in sorted((e.out / "probes").glob("*_sex.json")) if (e.out / "probes").exists() else []:
        lines += [f"## Second-level task: sex from the frozen sample embedding - {f.stem}", "```", json.dumps(json.load(open(f)), indent=1), "```", ""]
    lines += ["## Notes", "- Rows starting with 'biolearn' are the same published clock run through the biolearn package; they differ from the computage versions only in how missing CpGs are imputed.",
              "- The score table follows the `computage` source code but was not cross-checked against the authors' published table; it rewards clocks that show acceleration in disease, which is not the same as being accurate in healthy people.",
              "- The panel has 9,098 CpGs; clocks that need other CpGs (PhenoAge: 83% covered) are imputed with SeSAMe reference medians, so they are handicapped.",
              "- EpiClock and ridge/elastic net were trained on this data mix and the published clocks were not: ridge is the fair opponent, the published clocks give context.",
              "- Bootstrap intervals describe test-set sampling uncertainty. Training-run noise is shown by the -s1/-s2 seed arms.",
              "- Chronological-age accuracy is not the same as usefulness as a biomarker: see the disease age-acceleration score above and use `predict --meta ...` on your own cohorts."]
    lines += ["## Data audit", "", e.audit_text]; (e.out / "report.md").write_text("\n".join(lines) + "\n"); main.to_csv(e.out / "report_main.csv"); pairs.to_csv(e.out / "report_pairs.csv", index=False)
    rep = dict(main={n: dict(MedAE=float(main.loc[n, "MedAE"]), CI_studies=main.loc[n, "CI_studies"], MAE=float(main.loc[n, "MAE"]), r=float(main.loc[n, "r"])) for n in main.index},
               official=None if off is None else off.round(2).to_dict("index"), n_test=int(len(hc)), n_studies=int(meta.DatasetID.nunique()))
    json.dump(rep, open(e.out / "report.json", "w"), indent=1)
    log(f"report written: {e.out / 'report.md'}\n" + "\n".join(lines[:12])); return rep


# ======================================================================================================== probes of what the model learned about the genome
def probe_scores(F, targets, seed=0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.nan_to_num(F)); out = {}; cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, (kind, y) in targets.items():
        y = np.asarray(y)
        if kind == "multi":
            cnt = pd.Series(y).value_counts(); keep = np.isin(y, cnt[cnt >= 10].index)
            if keep.sum() < 100 or len(cnt[cnt >= 10]) < 2:
                continue
            p = cross_val_predict(RidgeClassifier(alpha=100.0), X[keep], y[keep], cv=cv)
            out[name] = dict(score=float(balanced_accuracy_score(y[keep], p)), chance=float(1 / len(np.unique(y[keep]))))
        else:
            yb = y.astype(int)
            if yb.sum() < 20 or (1 - yb).sum() < 20:
                continue
            dfn = cross_val_predict(RidgeClassifier(alpha=100.0), X, yb, cv=cv, method="decision_function")
            out[name] = dict(score=float(roc_auc_score(yb, dfn)), chance=0.5)
    return out


def probe_architecture(e):
    """Does the free per-CpG embedding (the model's learned 'identity' for each CpG) encode CpG-island relation, regulatory class, chromosome ...?
    Compared with what plain methylation statistics and an SVD embedding of the same size already give, and with random vectors."""
    pdir = e.out / "probes"; pdir.mkdir(exist_ok=True); ann = pd.read_parquet(e.data / "annot/annotation_450k.parquet").reindex(e.c.cpgs)
    g = lambda col: ann[col].fillna(0).astype(int).to_numpy() if col in ann else np.zeros(len(ann), int)
    prom = ((g("gene_TSS200") + g("gene_TSS1500")) > 0).astype(int)
    targets = {"CpG-island relation": ("multi", g("island")), "regulatory class": ("multi", g("reg")), "chromosome": ("multi", g("chrom")), "chromosome X": ("bin", (g("chrom") == 23).astype(int)),
               "enhancer": ("bin", (g("enhancer") > 0).astype(int)), "DNase-hypersensitive": ("bin", (g("dhs") > 0).astype(int)), "promoter (TSS200/TSS1500)": ("bin", prom),
               "gene body": ("bin", (g("gene_Body") > 0).astype(int)), "probe design II (a technical feature)": ("bin", (g("design2") > 0).astype(int))}
    tr = e.sp["train"]; sub = e.c.X[tr[np.random.RandomState(0).choice(len(tr), min(4000, len(tr)), replace=False)]]; Z = standardize(sub, e.mean, e.std)
    xs = np.where(sub == 255, np.nan, sub / 100.0)
    stats = np.stack([np.nan_to_num(np.nanmean(xs, 0), nan=0.5), np.nan_to_num(np.nanstd(xs, 0))], 1)
    Sv, Vt = None, None; base_cache = {}; done = []
    for n in finished_arms(e):
        ar = parse_arm(n)
        if ar["kind"] not in ("pre", "noctx") or ar["seed"] != 0:
            continue
        f = pdir / f"{n}.json"
        if f.exists():
            continue
        ck = torch.load(arm_dir(e, ar) / "model.pt", map_location="cpu", weights_only=False); E = ck["state"]["id_emb.weight"].numpy(); d = E.shape[1]
        if d not in base_cache:
            if Vt is None:
                _, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
            k = min(d, len(Sv)); base_cache[d] = dict(
                svd=probe_scores((Vt[:k].T * Sv[:k]).astype(np.float32), targets), rand=probe_scores(np.random.default_rng(0).standard_normal((len(e.c.cpgs), d)).astype(np.float32), targets),
                stats=probe_scores(stats, targets))
        res = {"learned id embedding": probe_scores(E, targets), "SVD of methylation matrix, same size": base_cache[d]["svd"], "mean+sd of beta": base_cache[d]["stats"], "random vectors": base_cache[d]["rand"]}
        json.dump(dict(context_input=bool(ck["cfg"].get("context", True)), dim=int(d), results=res), open(f, "w"), indent=1); done.append(n)
    if done:
        log(f"[probe] CpG-architecture probes written for {done}")


def probe_sex(e, name):
    """Second-level task on the frozen model: sex from the sample embedding (vs the same classifier on PCA of the raw betas)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    pdir = e.out / "probes"; pdir.mkdir(exist_ok=True); f = pdir / f"{name}_sex.json"
    if f.exists():
        return
    ar = parse_arm(name); m, ck = load_model(arm_dir(e, ar) / "model.pt", e.dev); K = ck["K"]; c = e.c
    sex = c.meta.Gender.astype(str).to_numpy(); lab = np.isin(sex, ["M", "F"]); sp = e.sp
    tr = sp["train"][lab[sp["train"]]][:4000]; te = sp["test_all"][lab[sp["test_all"]]][:2000]
    if len(tr) < 200 or len(te) < 100:
        return
    def emb(rows):
        r = np.random.default_rng(3); out = []
        with torch.no_grad():
            for s in range(0, len(rows), 64):
                idx, beta, valid, _ = draw(c.X[rows[s:s + 64]], K, r); out.append(m.embed(idx, beta, valid).cpu().numpy())
        return np.concatenate(out)
    Etr, Ete = emb(tr), emb(te); ytr, yte = (sex[tr] == "F").astype(int), (sex[te] == "F").astype(int)
    def fit(Ftr, Fte):
        sc = StandardScaler().fit(Ftr); p = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(Ftr), ytr).predict_proba(sc.transform(Fte))[:, 1]
        return dict(accuracy=round(float(((p > .5) == yte).mean()), 4), auc=round(float(roc_auc_score(yte, p)), 4))
    pca = PCA(min(64, len(tr) - 1), random_state=0).fit(standardize(c.X[tr], e.mean, e.std))
    res = {"frozen embedding": fit(Etr, Ete), "PCA-64 of raw betas": fit(pca.transform(standardize(c.X[tr], e.mean, e.std)), pca.transform(standardize(c.X[te], e.mean, e.std)))}
    json.dump(res, open(f, "w"), indent=1); del m; free_mem(e.dev)


# ======================================================================================================== release
def release(e, rep):
    rel = e.out / "release"; rel.mkdir(exist_ok=True); members = []
    for n in finished_arms(e):
        ar = parse_arm(n)
        if n != "linear-tuned" and ar["kind"] == "pre" and ar["seed"] == 0 and (arm_dir(e, ar) / "model.pt").exists():
            members.append((n, ar))
    if not members:
        return
    fam = ["# EpiClock model family", "", "| model | parameters | tokens per sample | MedAE (y) [95% CI over studies] | MAE (y) | r |", "|---|---|---|---|---|---|"]
    for n, ar in members:
        nm = SIZE_NAMES[ar["size"]]; d = rel / nm; d.mkdir(exist_ok=True); shutil.copy(arm_dir(e, ar) / "model.pt", d / "model.pt"); P = preset(e, ar["size"]); r = rep["main"].get(n, {})
        npar = param_count(e, ar["size"]); cfg = make_cfg(e, ar)
        json.dump(dict(name=nm, parameters=npar, panel_cpgs=len(e.c.cpgs), tokens_per_sample=P["K"], config=cfg, age_transform=f"Horvath-style, adult age {ADULT}", test=r), open(d / "config.json", "w"), indent=1)
        (d / "MODEL_CARD.md").write_text(f"""# {nm}

Genome-aware transformer epigenetic clock ({npar:,} parameters). Input: beta values at up to {len(e.c.cpgs):,} CpGs (any subset; missing values allowed). Output: epigenetic age in years.

**How it works.** Each CpG is a token (learned identity + chromosome, CpG-island relation, regulatory class, position, gene-region flags + its beta). A sample is a random subset of {P['K']} observed CpGs;
predictions over several disjoint subsets are averaged. The model is pre-trained without labels by hiding 25% of CpGs and reconstructing them, then fine-tuned on chronological age.
The age output is a linear clock over the whole CpG row (initialised from ridge) plus a correction learned by the transformer.

**Evaluation** (healthy controls of {rep['n_studies']} held-out ComputAge benchmark studies, n={rep['n_test']}): median absolute error {r.get('MedAE', float('nan')):.2f} y
(95% CI over studies {r.get('CI_studies', '')}), mean absolute error {r.get('MAE', float('nan')):.2f} y, r = {r.get('r', float('nan')):.3f}. See report.md for comparators and paired differences.

**Training data.** Public ComputAge Bench training studies and public GEO methylation series restricted to a {len(e.c.cpgs):,}-CpG panel; leukaemia/AML/MDS series were excluded from all training.

**Intended use.** Research on chronological/biological-age estimation and age acceleration. Not a medical device; not validated for clinical decisions.
**Limitations.** Panel-restricted (9,098 CpGs); mostly blood; adult ages dominate; chronological-age accuracy does not imply mortality prediction; 450K/EPIC-style betas expected.
**Licence / provenance.** Check the terms of the source datasets (ComputAge Bench on Hugging Face; each GEO series) before redistributing weights or training data.
""")
        fam.append(f"| {nm} | {npar:,} | {P['K']} | {r.get('MedAE', float('nan')):.2f} {r.get('CI_studies', '')} | {r.get('MAE', float('nan')):.2f} | {r.get('r', float('nan')):.3f} |")
    fam += ["", "Ensemble (mean of the family) is reported in report.md.", "", "## Use", "```python", "# put epiclock.py next to the weights", "from epiclock import epiclock_predict",
            "res = epiclock_predict('EpiClock-B/model.pt', betas_dataframe)   # rows = samples, columns = CpG ids (cg...), values = beta in [0, 1]", "ages = res['epigenetic_age']; deviation = res['deviation']", "```"]
    (rel / "FAMILY.md").write_text("\n".join(fam) + "\n"); shutil.copy(e.out / "report.md", rel / "report.md")
    log(f"[release] family package written to {rel}")


# ======================================================================================================== commands
def default_queue(e):
    if e.a.smoke:
        return list(SMOKE_QUEUE)
    return list(DEFAULT_QUEUE)


def finish(e):
    rep = build_report(e)
    probe_architecture(e)
    for n in finished_arms(e):
        ar = parse_arm(n)
        if n != "linear-tuned" and ar["kind"] == "pre" and ar["seed"] == 0:
            probe_sex(e, n)
    rep = build_report(e); release(e, rep)
    log(f"done. Read {e.out / 'report.md'}; the model family is in {e.out / 'release'}")


def cmd_study(a):
    e = make_env(a); queue = [x for x in a.arms.split(",") if x] or default_queue(e)
    log(f"queue: {queue}. Results are written after every model; Ctrl-C or `touch {e.out / 'STOP'}` pauses, the same command resumes. `touch {e.out / 'NEXT'}` ends the current training stage early.")
    try:
        get_comparators(e)
        for name in queue:
            log(f"##### {name}")
            run_arm(e, name); build_report(e)
        finish(e)
    except Paused:
        log("paused by the STOP file. Delete it and run the same command again to resume.")


def cmd_report(a):
    e = make_env(a)
    try:
        finish(e)
    except Paused:
        pass


def epiclock_predict(model_path, betas, device="auto", draws=16):
    """Ages for a DataFrame of beta values (rows = samples, columns = CpG ids; rows may also be CpGs, it is detected).
    -> DataFrame with `epigenetic_age` and, for pre-trained models, `deviation` (masked-CpG reconstruction error: how unlike healthy methylation the sample is)."""
    dev = get_device(device); m, ck = load_model(model_path, dev)
    B = betas.T if pd.Index(betas.index.astype(str)).str.startswith("cg").mean() > 0.5 else betas
    pos = pd.Index(ck["cpgs"]).get_indexer(B.columns.astype(str)); keep = pos >= 0
    X = np.full((len(B), len(ck["cpgs"])), 255, np.uint8); v = B.to_numpy(np.float64)[:, keep]
    X[:, pos[keep]] = np.where(np.isfinite(v) & (v >= 0) & (v <= 1), np.rint(v * 100), 255).astype(np.uint8)
    log(f"{len(B)} samples; {keep.sum()} of your {len(keep)} CpGs are in the model's panel ({keep.sum() / len(ck['cpgs']):.0%} of the panel)")
    out = pd.DataFrame({"epigenetic_age": t_to_age(predict_t(m, X, ck["K"], draws, np.random.default_rng(0)))}, index=B.index)
    if ck.get("pretrained", True):
        out["deviation"] = recon_error(m, X, ck["K"], 4, np.random.default_rng(1))
    return out


def cmd_predict(a):
    B = pd.read_csv(a.beta, sep="\t" if a.beta.endswith((".tsv", ".txt")) else ",", index_col=0)
    res = epiclock_predict(a.model, B, a.device, a.draws); out = res.reset_index().rename(columns={res.index.name or "index": "sample"})
    if a.meta:
        M = pd.read_csv(a.meta, sep="\t" if a.meta.endswith((".tsv", ".txt")) else ",", index_col=0).reindex(res.index)
        out["age"] = M[a.age_col].to_numpy(float); out["age_acceleration_raw"] = out.epigenetic_age - out.age
        if a.group_col and a.control:
            grp = M[a.group_col].astype(str).to_numpy(); tab = age_acceleration(out.epigenetic_age.to_numpy(), out.age.to_numpy(), grp, a.control)
            log("age acceleration versus the control group (AA = predicted age minus the control-group fit on chronological age):\n" + tab.round(4).to_string())
            tab.to_csv(Path(a.out).with_suffix(".aa.csv"))
            if "deviation" in out:
                tab2 = age_acceleration(out.deviation.to_numpy(), out.age.to_numpy(), grp, a.control)
                log("deviation score (masked-CpG reconstruction error, age trend removed) versus the control group:\n" + tab2.round(4).to_string()); tab2.to_csv(Path(a.out).with_suffix(".deviation.csv"))
    out.to_csv(a.out, index=False); log(f"wrote {a.out}")


def cmd_probe(a):
    a.arms = ""; e = make_env(a); probe_architecture(e)
    for n in finished_arms(e):
        if n != "linear-tuned" and parse_arm(n)["kind"] == "pre" and parse_arm(n)["seed"] == 0:
            probe_sex(e, n)
    build_report(e)


def main():
    import types
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--model", required=True); ap.add_argument("--beta", required=True); ap.add_argument("--out", default="epigenetic_ages.csv")
        ap.add_argument("--device", default="auto"); ap.add_argument("--draws", type=int, default=16); ap.add_argument("--meta", default=None, help="csv/tsv indexed by sample id")
        ap.add_argument("--age-col", default="age"); ap.add_argument("--group-col", default=None); ap.add_argument("--control", default=None, help="value of --group-col that marks the controls")
        cmd_predict(ap.parse_args()); return
    smoke = os.environ.get("EPICLOCK_SMOKE") == "1"                          # developer switch: 30-step functional check of every code path (tiny models)
    a = types.SimpleNamespace(data=os.environ.get("EPICLOCK_DATA"), out=os.environ.get("EPICLOCK_OUT", "runs/epiclock_clean"), device=os.environ.get("EPICLOCK_DEVICE", "auto"), threads=0,
                              memory_fraction=CONFIG["memory_fraction"], seed=0, eval_every=CONFIG["eval_every"], patience=CONFIG["patience"], ssl_steps=10**9, ft_steps=10**9,
                              val_draws=CONFIG["val_draws"], test_draws=CONFIG["test_draws"], boot=CONFIG["boot"], no_selftest=False, smoke=smoke, arms="")
    if smoke:
        a.ssl_steps, a.ft_steps, a.eval_every, a.test_draws, a.val_draws, a.boot, a.no_selftest = 30, 30, 15, 2, 1, 200, True
    cmd_study(a)


if __name__ == "__main__":
    main()
