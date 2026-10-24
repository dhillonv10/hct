"""Multi-objective pretraining for LncFM.

Objectives (all label-free or annotation-derived; no benchmark label is ever used):
  1. span-masked LM        -- 15% of nucleotides in spans (mean 3 nt), BERT 80/10/10 corruption
  2. exon-junction tagging -- predict, per nucleotide, whether an exon-exon junction follows it
                              (labels from the GENCODE GTF; teaches splice-aware features)
  3. isoform contrastive   -- two views of the same *gene* (different isoform, or two crops) are
                              positives, other genes in the batch negatives (NT-Xent)

    python -m lncfm.pretrain --preset tiny  --steps 3000  --out runs/pilot_full
    python -m lncfm.pretrain --preset small --steps 60000 --out runs/small_v1     # on the M4

Ablations: --lambda-j 0 --lambda-c 0  gives plain span-MLM with identical batches.
Resumable: re-running the same command continues from <out>/ckpt_last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import gencode, paths
from .model import MASK, NT0, PAD, VOCAB, LncFM, LncFMConfig, build, encode, pick_device


# ----------------------------------------------------------------------------- corpus
class Corpus:
    """Gene-balanced sampler over transcripts of one or more species."""

    def __init__(self, species=("human", "mouse"), exclude_heldout=True):
        self.seqs, self.juncs, self.gene_rows = [], [], []
        gene_to_rows = {}
        for sp in species:
            df = gencode.build_table(sp, "lnc")
            if exclude_heldout and sp == "human":
                keep = ~df.chrom.isin(gencode.TEST_CHROMS | gencode.VAL_CHROMS)
                df = df[keep]
            for gid, seq, ex in zip(df.gene_id, df.seq, df.exon_lens):
                gene_to_rows.setdefault(sp + ":" + gid, []).append(len(self.seqs))
                self.seqs.append(seq)
                c = np.cumsum(ex)[:-1] - 1
                self.juncs.append(c.astype(np.int32))
        self.genes = [np.array(v) for v in gene_to_rows.values()]

    def __len__(self):
        return len(self.genes)

    def n_nt(self):
        return sum(len(s) for s in self.seqs)


def val_corpus(n_genes=256, seed=0):
    """Held-out human genes (val chromosomes) with deterministic order, for monitoring MLM loss."""
    df = gencode.build_table("human", "lnc")
    df = gencode.representative_per_gene(df)
    df = df[df.chrom.isin(gencode.VAL_CHROMS)].sample(n_genes, random_state=seed)
    return list(df.seq), [np.cumsum(e)[:-1].astype(np.int32) - 1 for e in df.exon_lens]


# ----------------------------------------------------------------------------- batching
def span_mask(n, rng, frac=0.15, mean_span=3, max_span=8):
    m = np.zeros(n, bool)
    target = max(1, int(round(frac * n)))
    while m.sum() < target:
        s = int(rng.integers(0, n))
        l = min(int(rng.geometric(1.0 / mean_span)), max_span)
        m[s:s + l] = True
    return m


def corrupt(ids, rng):
    """Returns (corrupted ids, labels with -100 on unmasked positions)."""
    n = len(ids)
    m = span_mask(n, rng)
    labels = np.where(m, ids, -100)
    x = ids.copy()
    r = rng.random(n)
    x[m & (r < 0.8)] = MASK
    rnd = m & (r >= 0.8) & (r < 0.9)
    x[rnd] = rng.integers(NT0, NT0 + 4, rnd.sum())
    return x, labels


def make_view(seq, junc, rng, max_len, crop_aug, stride=4):
    ids = encode(seq).astype(np.int64)
    n = len(ids)
    L = n
    if crop_aug and n > 64:
        L = int(n * rng.uniform(0.75, 1.0))
    L = min(L, max_len)
    a = int(rng.integers(0, n - L + 1)) if n > L else 0
    ids = ids[a:a + L]
    jl = np.zeros(L, np.float32)
    j = junc[(junc >= a) & (junc < a + L)] - a
    jl[j] = 1.0
    x, lab = corrupt(ids, rng)
    return x, lab, jl


def collate(views, stride=4):
    L = max(len(v[0]) for v in views)
    L = int(math.ceil(L / stride) * stride)
    B = len(views)
    X = np.zeros((B, L), np.int64)
    Y = np.full((B, L), -100, np.int64)
    J = np.zeros((B, L), np.float32)
    for i, (x, y, j) in enumerate(views):
        X[i, :len(x)], Y[i, :len(y)], J[i, :len(j)] = x, y, j
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(J)


def sample_batch(corpus: Corpus, B, rng, max_len):
    """B genes -> 2B views ordered [view1 of g0..gB-1, view2 of g0..gB-1]."""
    gidx = rng.integers(0, len(corpus), B)
    v1, v2 = [], []
    for g in gidx:
        rows = corpus.genes[g]
        if len(rows) >= 2:
            r1, r2 = rng.choice(rows, 2, replace=False)
        else:
            r1 = r2 = rows[0]
        v1.append(make_view(corpus.seqs[r1], corpus.juncs[r1], rng, max_len, True))
        v2.append(make_view(corpus.seqs[r2], corpus.juncs[r2], rng, max_len, True))
    return collate(v1 + v2)


# ----------------------------------------------------------------------------- losses
def nt_xent(z, tau=0.1):
    B2 = z.shape[0]
    B = B2 // 2
    sim = z @ z.T / tau
    sim = sim.masked_fill(torch.eye(B2, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(B, B2), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, tgt)
    acc = (sim.argmax(1) == tgt).float().mean()
    return loss, acc


def compute_loss(model: LncFM, X, Y, J, lam_j, lam_c, pos_weight=20.0):
    out = model(X)
    m = Y != -100
    l_mlm = F.cross_entropy(out["nt_logits"][m], Y[m])
    acc = (out["nt_logits"][m].argmax(-1) == Y[m]).float().mean()
    valid = (X != PAD)
    parts = {"mlm": l_mlm.detach(), "mlm_acc": acc.detach()}
    loss = l_mlm
    if lam_j > 0:
        l_j = F.binary_cross_entropy_with_logits(out["junction_logits"][valid], J[valid],
                                                 pos_weight=torch.tensor(pos_weight, device=X.device))
        loss = loss + lam_j * l_j
        parts["junc"] = l_j.detach()
    if lam_c > 0:
        l_c, c_acc = nt_xent(model.project(out["pooled"]))
        loss = loss + lam_c * l_c
        parts["con"], parts["con_acc"] = l_c.detach(), c_acc.detach()
    return loss, parts


@torch.no_grad()
def evaluate(model, device, vseqs, vjuncs, max_len, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    tot, n, correct = 0.0, 0, 0
    for i in range(0, len(vseqs), 32):
        views = [make_view(s, j, rng, max_len, False) for s, j in zip(vseqs[i:i + 32], vjuncs[i:i + 32])]
        X, Y, J = (t.to(device) for t in collate(views))
        out = model(X)
        m = Y != -100
        lg = out["nt_logits"][m]
        tot += F.cross_entropy(lg, Y[m], reduction="sum").item()
        correct += (lg.argmax(-1) == Y[m]).sum().item()
        n += int(m.sum())
    model.train()
    return {"val_mlm": tot / n, "val_mlm_acc": correct / n}


# ----------------------------------------------------------------------------- main loop
def lr_at(step, total, base, warm):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def save(path, model, opt, step, args):
    tmp = Path(str(path) + ".tmp")
    torch.save({"cfg": model.cfg.to_dict(), "model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "args": vars(args)}, tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32, help="genes per step (2 views each)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lambda-j", type=float, default=0.3)
    ap.add_argument("--lambda-c", type=float, default=0.3)
    ap.add_argument("--species", nargs="+", default=["human", "mouse"])
    ap.add_argument("--include-heldout", action="store_true", help="also pretrain on val/test chromosomes")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--snap-every", type=int, default=0, help="keep ckpt_step<N>.pt every N steps (learning curves)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model = build(a.preset, max_len=max(a.max_len, 1024)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    step0 = 0
    last = out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"resumed from step {step0}")
    print(f"device={device} params={model.n_params()/1e6:.2f}M cfg={model.cfg.to_dict()}")

    corpus = Corpus(a.species, exclude_heldout=not a.include_heldout)
    vseqs, vjuncs = val_corpus()
    print(f"corpus: {len(corpus)} genes, {len(corpus.seqs)} transcripts, {corpus.n_nt()/1e6:.1f} Mnt  | val genes {len(vseqs)}")
    rng = np.random.default_rng(a.seed + step0)
    logf = open(out / "log.jsonl", "a")
    t0, tok = time.time(), 0
    model.train()
    for step in range(step0, a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        X, Y, J = (t.to(device) for t in sample_batch(corpus, a.batch, rng, a.max_len))
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if a.dtype == "bf16" else torch.autocast(device.type, enabled=False)
        with ctx:
            loss, parts = compute_loss(model, X, Y, J, a.lambda_j, a.lambda_c)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tok += int((X != PAD).sum())
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "loss": float(loss), **{k: float(v) for k, v in parts.items()},
                   "lr": opt.param_groups[0]["lr"], "nt_per_s": tok / (time.time() - t0), "elapsed": time.time() - t0}
            if (step + 1) % a.eval_every == 0:
                rec.update(evaluate(model, device, vseqs, vjuncs, a.max_len))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if (step + 1) % 100 == 0 or "val_mlm" in rec:
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save(last, model, opt, step + 1, a)
        if a.snap_every and (step + 1) % a.snap_every == 0:
            save(out / f"ckpt_step{step+1}.pt", model, opt, step + 1, a)
    save(out / "ckpt_final.pt", model, opt, a.steps, a)
    print("done")


if __name__ == "__main__":
    main()
