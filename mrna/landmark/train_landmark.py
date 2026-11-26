#!/usr/bin/env python3
"""Landmark-anchored mRNA half-life model. Runs on an Apple-silicon Mac (MPS), CUDA, or CPU.

THE IDEA
--------
mRNA function is organised around landmarks, not around absolute position. Two measurements from this
project point the same way:

  * On mean-ribosome-load, replacing mean-pooling of frozen embeddings with frame-aware pooling gained
    +0.145 r2, and a stack given the choice assigned the mean-pooled vector a weight of 0.028 -- it threw
    the global average away.
  * On half-life, exon-junction DENSITY dominated 609 engineered features by 6.5x, yet collapsing junction
    positions into that one scalar left us 0.070 short of Saluki.

Existing models get landmark structure only indirectly. Saluki feeds frame and splice as input channels and
must learn from convolutions that distance-to-stop matters. Orthrus, NT and RiNALMo encode position relative
to the sequence start. Optimus/FramePool are frame-aware but 5'UTR-only.

This model encodes, for every nucleotide, the SIGNED DISTANCE to six biological landmarks -- cap, start
codon, stop codon, poly(A) site, previous exon junction, next exon junction -- as multi-scale Fourier
features. One period is exactly 3 nt, so reading frame is an explicit input rather than something to infer.
The consequence that matters mechanistically: nonsense-mediated decay fires when a junction sits >50 nt
DOWNSTREAM OF A STOP CODON, and this model reads that off its input instead of learning it.

The read-out then pools over regions, over distance bands around each landmark, by frame within the CDS,
and over junctions downstream of the stop -- the generalisation of the pooling that gained +0.145 above.

The 609 engineered features go in at the head, so the network starts from a representation that already
reaches r=0.702 on its own and can only add to it. That is a deliberate floor, not a crutch.

TARGET
------
Saluki, human, mean over the authors' 10 CV folds: r = 0.7726  (their Fig5_S6/experimentResults.csv)
Their engineered baseline BEeM: 0.7681.   Our features + GBM: 0.7023.

  python3 train_landmark.py --bundle hl_bundle_human.npz --selftest     # ~2 min, checks the whole pipeline
  python3 train_landmark.py --bundle hl_bundle_human.npz --quick        # 3 folds, ~1-2 h on an M4
  python3 train_landmark.py --bundle hl_bundle_human.npz                # 10 folds, ~8-12 h on an M4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

SALUKI = {"human": 0.7726, "mouse": 0.7320}      # their Fig5_S6/experimentResults.csv, mean of 10 folds
SALUKI_HUMAN = SALUKI["human"]
BEEM_HUMAN = 0.7681
FEATURES_GBM = 0.7023
SEQ_ONLY_SALUKI = 0.608                          # their sequence-only ablation, the phase-D reference
PERIODS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)   # 3 nt => reading frame, explicitly
N_LANDMARK = 6

# The released family. Depth and width scale together; dilations grow so the receptive field keeps pace.
# With only ~14k scorable genes the expectation is that this SATURATES early -- where it saturates is the
# result, because it tells a practitioner what size they actually need.
SIZES = {
    "tiny":  dict(c1=16, c=32,  s1=(1, 2),    s2=(1, 2, 4),             s3=(1, 2, 4, 8),             head=128),
    "small": dict(c1=24, c=64,  s1=(1, 2),    s2=(1, 2, 4, 8),          s3=(1, 2, 4, 8, 16),         head=256),
    "base":  dict(c1=32, c=96,  s1=(1, 2),    s2=(1, 2, 4, 8, 16),      s3=(1, 2, 4, 8, 16, 32),     head=512),
    "large": dict(c1=48, c=160, s1=(1, 2, 4), s2=(1, 2, 4, 8, 16),      s3=(1, 2, 4, 8, 16, 32),     head=768),
    "xl":    dict(c1=64, c=256, s1=(1, 2, 4), s2=(1, 2, 4, 8, 16, 32),  s3=(1, 2, 4, 8, 16, 32, 64), head=1024),
}


def pick_device(name=""):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------------------------- data

class Bundle:
    def __init__(self, path, lmax):
        z = np.load(path, allow_pickle=True)
        self.seq, self.soff = z["seq"], z["seq_off"]
        self.junc, self.joff = z["junc"], z["junc_off"]
        self.L = z["length"].astype(np.int64)
        self.cs = z["cds_start"].astype(np.int64)
        self.cl = z["cds_len"].astype(np.int64)
        self.y = z["y"].astype(np.float32)
        self.gene = z["gene"]
        self.X = z["X"].astype(np.float32)
        self.cols = list(z["feat_cols"])
        self.lmax = lmax
        self.n = len(self.L)
        # extended bundle: `scored` marks genes with a PC1 label (the only ones ever evaluated), and Y holds
        # the 54 individual decay datasets -- 25x the supervision of PC1 alone, on the same sequences.
        self.scored = z["scored"].astype(bool) if "scored" in z.files else np.ones(self.n, bool)
        self.Y = z["Y"].astype(np.float32) if "Y" in z.files else None
        self.tasks = list(z["task_names"]) if "task_names" in z.files else []
        self.species = str(z["species"][0]) if "species" in z.files else "human"

    def item(self, i):
        """One transcript as fixed-length arrays. Everything is 5'-anchored and right-padded."""
        lm = self.lmax
        s = self.seq[self.soff[i]:self.soff[i + 1]]
        L = int(min(self.L[i], lm))
        codes = np.full(lm, 4, np.uint8)
        codes[:L] = s[:L]
        start = int(self.cs[i]) if self.cl[i] > 0 else 0
        stop = int(self.cs[i] + self.cl[i]) if self.cl[i] > 0 else L
        j = self.junc[self.joff[i]:self.joff[i + 1]].astype(np.int64)
        j = j[j < L]
        isj = np.zeros(lm, np.uint8)
        isj[j] = 1
        # signed distance to the nearest junction upstream / downstream, clipped to int16 range
        pos = np.arange(lm)
        if len(j):
            k = np.searchsorted(j, pos, side="right")
            up = np.where(k > 0, pos - j[np.clip(k - 1, 0, len(j) - 1)], 30000)
            k2 = np.searchsorted(j, pos, side="left")
            dn = np.where(k2 < len(j), j[np.clip(k2, 0, len(j) - 1)] - pos, 30000)
        else:
            up = np.full(lm, 30000); dn = np.full(lm, 30000)
        ym = self.Y[i] if self.Y is not None else np.zeros(0, np.float32)
        return (codes, np.int32(L), np.int32(min(start, lm)), np.int32(min(stop, lm)),
                isj, up.astype(np.int16), dn.astype(np.int16), self.X[i], self.y[i],
                ym, np.float32(self.scored[i]))


def collate(bundle, idx):
    out = [bundle.item(i) for i in idx]
    t = lambda k, dt: torch.from_numpy(np.stack([o[k] for o in out]).astype(dt))
    return dict(codes=t(0, np.int64),
                L=torch.tensor([int(o[1]) for o in out]),
                start=torch.tensor([int(o[2]) for o in out]),
                stop=torch.tensor([int(o[3]) for o in out]),
                isj=t(4, np.float32), dup=t(5, np.float32), ddn=t(6, np.float32),
                aux=t(7, np.float32),
                y=torch.tensor([float(o[8]) for o in out], dtype=torch.float32),
                ytask=t(9, np.float32),
                scored=torch.tensor([float(o[10]) for o in out], dtype=torch.float32))


# ------------------------------------------------------------------------------------------- model

def fourier(d, periods):
    """(B,1,T) signed distance -> (B, 2*len(periods), T) sin/cos features. Period 3 encodes frame."""
    outs = []
    for p in periods:
        w = 2.0 * math.pi / p
        outs.append(torch.sin(d * w))
        outs.append(torch.cos(d * w))
    return torch.cat(outs, 1)


class Block(nn.Module):
    """Dilated residual block. GroupNorm rather than LayerNorm: better supported on MPS for (B,C,T)."""

    def __init__(self, c, k, dil):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.norm = nn.GroupNorm(8, c)
        self.pw = nn.Conv1d(c, c, 1)

    def forward(self, x):
        return x + self.pw(F.gelu(self.norm(self.conv(x))))


class LandmarkNet(nn.Module):
    def __init__(self, n_aux, size="base", periods=PERIODS, dropout=0.2, aux_drop=0.3, n_task=0):
        super().__init__()
        cfg = SIZES[size]
        c1, c, head = cfg["c1"], cfg["c"], cfg["head"]
        self.size, self.n_task = size, n_task
        self.periods = periods
        in_ch = 4 + 3 + 1 + 1 + N_LANDMARK * 2 * len(periods)
        self.in_ch = in_ch
        self.stem = nn.Conv1d(in_ch, c1, 9, padding=4)
        self.s1 = nn.Sequential(*[Block(c1, 5, d) for d in cfg["s1"]])
        self.d1 = nn.Conv1d(c1, c, 4, stride=4)
        self.s2 = nn.Sequential(*[Block(c, 5, d) for d in cfg["s2"]])
        self.d2 = nn.Conv1d(c, c, 4, stride=4)
        self.s3 = nn.Sequential(*[Block(c, 5, d) for d in cfg["s3"]])
        self.attn = nn.Conv1d(c, 1, 1)
        n_cells = 6 + 6 + 2 + 2 + 1          # regions(3 mean+3 max) + 6 bands + 2 junction + 2 global + attn
        self.pool_dim = n_cells * c + 3 * c1
        self.norm_pool = nn.LayerNorm(self.pool_dim)
        self.norm_aux = nn.LayerNorm(n_aux)
        # Aux-block dropout. The 609 engineered features already reach r=0.702 alone, so without this the
        # head can ignore the convolutional path entirely and the sequence model never learns anything.
        # Zeroing the WHOLE aux block on a fraction of training steps forces the conv path to carry weight.
        self.aux_drop = aux_drop
        self.head = nn.Sequential(
            nn.Linear(self.pool_dim + n_aux, head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head, head // 2), nn.GELU(), nn.Dropout(dropout))
        self.out_pc1 = nn.Linear(head // 2, 1)
        # One head per individual decay dataset. These are never reported; they exist so the trunk sees
        # 482k label cells instead of 19.5k, and so the 2,265 genes that lack a PC1 label can still train.
        self.out_task = nn.Linear(head // 2, n_task) if n_task else None

    def build_input(self, b):
        codes, isj = b["codes"], b["isj"]
        B, T = codes.shape
        dev = codes.device
        oh = F.one_hot(codes.clamp(max=4), 5).permute(0, 2, 1).float()[:, :4]   # drop the N channel
        pos = torch.arange(T, device=dev).view(1, 1, T).float().expand(B, 1, T)
        L = b["L"].view(B, 1, 1).float()
        st = b["start"].view(B, 1, 1).float()
        sp = b["stop"].view(B, 1, 1).float()
        valid = (pos < L).float()
        region = torch.cat([(pos < st) * valid, ((pos >= st) & (pos < sp)).float() * valid,
                            (pos >= sp).float() * valid], 1)
        dists = [pos, pos - st, pos - sp, pos - L,
                 b["dup"].unsqueeze(1), b["ddn"].unsqueeze(1)]
        fe = torch.cat([fourier(d, self.periods) for d in dists], 1)
        x = torch.cat([oh, region, isj.unsqueeze(1), valid, fe], 1)
        return x, valid, region, pos, st, sp, L

    @staticmethod
    def mmean(h, m):
        """Masked mean over time. m is (B,1,T) and may be fractional after downsampling."""
        return (h * m).sum(-1) / m.sum(-1).clamp(min=1e-3)

    @staticmethod
    def mmax(h, m):
        return torch.where(m > 0, h, torch.full_like(h, -1e4)).max(-1).values

    def forward(self, b):
        x, valid, region, pos, st, sp, L = self.build_input(b)
        h1 = self.s1(self.stem(x))                                  # full resolution: frame lives here
        h = self.s3(self.d2(self.s2(self.d1(h1))))                  # /16
        r = 16
        dn = lambda m: F.avg_pool1d(F.avg_pool1d(m, 4), 4)
        v16 = dn(valid)
        reg16 = [dn(region[:, k:k + 1]) for k in range(3)]
        cells = []
        for m in reg16:
            cells.append(self.mmean(h, m))
        for m in reg16:
            cells.append(self.mmax(h, m))
        # distance bands around the start and stop codons, the cap and the poly(A) site
        bands = [(pos - st).abs() <= 50, (pos - st).abs() <= 200,
                 (pos - sp).abs() <= 50, (pos - sp).abs() <= 200,
                 pos < 100, (pos - L).abs() <= 200]
        for bm in bands:
            cells.append(self.mmean(h, dn(bm.float() * valid)))
        # junctions, and junctions >50 nt past the stop codon -- the NMD rule, handed over directly
        isj = b["isj"].unsqueeze(1)
        cells.append(self.mmean(h, dn(isj)))
        cells.append(self.mmean(h, dn(isj * (pos > sp + 50).float())))
        cells.append(self.mmean(h, v16))
        cells.append(self.mmax(h, v16))
        a = self.attn(h).masked_fill(v16 <= 0, -1e4).softmax(-1)
        cells.append((h * a).sum(-1))
        # frame cells at FULL resolution, where mod-3 structure still exists
        inc = ((pos >= st) & (pos < sp)).float() * valid
        fr = (pos - st) % 3
        for f in range(3):
            cells.append(self.mmean(h1, inc * (fr == f).float()))
        z = self.norm_pool(torch.cat(cells, 1))
        a_in = self.norm_aux(b["aux"])
        if self.training and self.aux_drop > 0:
            keep = (torch.rand(a_in.shape[0], 1, device=a_in.device) >= self.aux_drop).float()
            a_in = a_in * keep
        h = self.head(torch.cat([z, a_in], 1))
        return self.out_pc1(h).squeeze(-1), (self.out_task(h) if self.out_task is not None else None)


# ---------------------------------------------------------------------------------------- training

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def run_fold(bu, tr, va, te, dev, args, seed):
    """tr may include train-only genes (no PC1 label); va and te are always scorable genes only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = bu.X[tr].mean(0), bu.X[tr].std(0)
    sd[sd < 1e-6] = 1.0
    sc_tr = tr[bu.scored[tr]]
    ymu, ysd = float(bu.y[sc_tr].mean()), float(bu.y[sc_tr].std())
    # per-task standardisation from training rows only; tasks are sparsely observed so use nan-aware stats
    n_task = bu.Y.shape[1] if (bu.Y is not None and args.multitask) else 0
    if n_task:
        Yt = bu.Y[tr]
        tmu = np.nanmean(np.where(np.isfinite(Yt), Yt, np.nan), 0)
        tsd = np.nanstd(np.where(np.isfinite(Yt), Yt, np.nan), 0)
        tmu = np.nan_to_num(tmu); tsd = np.where(np.isfinite(tsd) & (tsd > 1e-6), tsd, 1.0)
        tmu_t = torch.from_numpy(tmu.astype(np.float32)).to(dev)
        tsd_t = torch.from_numpy(tsd.astype(np.float32)).to(dev)
    net = LandmarkNet(bu.X.shape[1], size=args.size, aux_drop=args.aux_drop, n_task=n_task).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.25)

    def batches(idx, bs, shuffle):
        o = np.random.permutation(len(idx)) if shuffle else np.arange(len(idx))
        for i in range(0, len(idx), bs):
            sel = idx[o[i:i + bs]]
            b = collate(bu, sel)
            b["aux"] = (b["aux"] - torch.from_numpy(mu)) / torch.from_numpy(sd)
            yield {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}, sel

    def predict(idx):
        net.eval()
        out = np.zeros(len(idx), np.float32)
        p = 0
        with torch.no_grad():
            for b, sel in batches(idx, args.bs, False):
                out[p:p + len(sel)] = (net(b)[0].float().cpu().numpy() * ysd + ymu)
                p += len(sel)
        net.train()
        return out

    best, best_te, best_va, best_ep = -np.inf, None, None, -1
    for ep in range(args.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        for b, sel in batches(tr, args.bs, True):
            opt.zero_grad()
            pred, tpred = net(b)
            # PC1 loss only where a PC1 label exists; the train-only genes contribute through the task heads
            m = b["scored"]
            tgt = (b["y"] - ymu) / ysd
            loss = (F.huber_loss(pred, tgt, delta=1.5, reduction="none") * m).sum() / m.sum().clamp(min=1.0)
            if tpred is not None:
                yt = b["ytask"]
                obs = torch.isfinite(yt).float()
                ytn = torch.nan_to_num((yt - tmu_t) / tsd_t, nan=0.0, posinf=0.0, neginf=0.0)
                tl = F.huber_loss(tpred, ytn, delta=1.5, reduction="none") * obs
                loss = loss + args.task_w * tl.sum() / obs.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        pv = predict(va)
        rv = pearson(bu.y[va], pv)
        if rv > best:
            best, best_ep, best_va = rv, ep, pv
            best_te = predict(te)
        print(f"    ep {ep + 1:3d}/{args.epochs}  loss {tot / max(nb,1):.4f}  val r {rv:.4f}"
              f"{'  *' if ep == best_ep else ''}  {time.time() - t0:.0f}s", flush=True)
    print(f"    best val r {best:.4f} at epoch {best_ep + 1}  ({nparam/1e6:.2f}M params)")

    # Late fusion instead of a shared head. The shared head let the 609 features reach ~0.67 in a single
    # epoch, so the convolutional trunk never received a useful gradient and learned nothing. Worse, an MLP
    # on those features is weaker than gradient boosting on them (0.67 vs 0.702), so the "floor" the first
    # design claimed never existed. Training the two separately and combining them on the inner validation
    # split removes the shortcut and makes the floor real.
    if args.stack_gbm:
        from sklearn.ensemble import HistGradientBoostingRegressor
        g = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.12,
                                          random_state=seed).fit(bu.X[sc_tr], bu.y[sc_tr])
        gv, gt = g.predict(bu.X[va]), g.predict(bu.X[te])
        w = np.linalg.lstsq(np.c_[best_va, gv, np.ones(len(va))], bu.y[va], rcond=None)[0]
        st = np.c_[best_te, gt, np.ones(len(te))] @ w
        print(f"    nn {pearson(bu.y[te], best_te):.4f} | gbm {pearson(bu.y[te], gt):.4f} | "
              f"stack {pearson(bu.y[te], st):.4f}   (weights nn={w[0]:.2f} gbm={w[1]:.2f})")
        return st, best, nparam, dict(nn=pearson(bu.y[te], best_te), gbm=pearson(bu.y[te], gt),
                                      stack=pearson(bu.y[te], st))
    return best_te, best, nparam, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--lmax", type=int, default=12288)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1, help="predictions averaged over this many seeds per fold")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="landmark_results.json")
    ap.add_argument("--aux-drop", type=float, default=0.3,
                    help="1.0 = sequence-only arm, the mechanism control (Saluki sequence-only is 0.608)")
    ap.add_argument("--size", default="base", choices=list(SIZES))
    ap.add_argument("--multitask", action="store_true",
                    help="train the 54 individual-decay-dataset heads too (25x the label cells)")
    ap.add_argument("--task-w", type=float, default=1.0, help="weight on the auxiliary task loss")
    ap.add_argument("--tag", default="", help="label for this run in the results file")
    ap.add_argument("--stack-gbm", action="store_true",
                    help="late fusion: train a GBM on the 609 features inside each fold and combine it with "
                         "the network on the inner validation split. Use with --aux-drop 1.0 so the network "
                         "sees sequence only and the two stay independent.")
    ap.add_argument("--quick", action="store_true", help="3 folds, lmax 8192, 25 epochs")
    ap.add_argument("--bench", action="store_true",
                    help="time 20 training steps and extrapolate the full run, then exit")
    ap.add_argument("--selftest", action="store_true", help="tiny run that exercises every code path")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.lmax, args.epochs = 3, 8192, 25
    if args.selftest:
        args.folds, args.lmax, args.epochs, args.bs = 2, 2048, 2, 4

    dev = pick_device(args.device)
    bu = Bundle(args.bundle, args.lmax)
    nt = bu.Y.shape[1] if (bu.Y is not None and args.multitask) else 0
    print(f"device {dev} | {bu.n} transcripts ({int(bu.scored.sum())} scorable on PC1, "
          f"{int((~bu.scored).sum())} train-only) | species {bu.species}")
    print(f"size '{args.size}' | lmax {args.lmax} | {bu.X.shape[1]} aux features | "
          f"{nt} auxiliary task heads | aux_drop {args.aux_drop}")
    net0 = LandmarkNet(bu.X.shape[1], size=args.size, n_task=nt)
    print(f"input channels {net0.in_ch}  pooled dim {net0.pool_dim}  "
          f"params {sum(p.numel() for p in net0.parameters())/1e6:.2f}M")
    del net0

    if args.bench:
        net = LandmarkNet(bu.X.shape[1], size=args.size, n_task=nt).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        sel = np.random.default_rng(0).permutation(bu.n)[:args.bs * 21]
        t0 = None
        for i in range(21):
            b = collate(bu, sel[i * args.bs:(i + 1) * args.bs])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad()
            F.huber_loss(net(b)[0], b["y"] * 0).backward()
            opt.step()
            if i == 0:
                if dev.type == "mps": torch.mps.synchronize()
                t0 = time.time()                      # skip step 0: it pays the compile/alloc cost
        if dev.type == "mps": torch.mps.synchronize()
        per_step = (time.time() - t0) / 20
        spe = per_step * math.ceil(bu.n * 0.9 / args.bs)
        print(f"\n{per_step*1000:.0f} ms/step at bs={args.bs}, lmax={args.lmax}")
        print(f"  ~{spe/60:.1f} min per epoch")
        print(f"  --quick  (3 folds x 25 ep, lmax 8192): ~{3*25*spe*(8192/args.lmax)/3600:.1f} h")
        print(f"  full     (10 folds x {args.epochs} ep, lmax {args.lmax}): ~{10*args.epochs*spe/3600:.1f} h")
        print("\nIf the full run is too long: raise --bs, lower --lmax (8192 truncates 9.7% of transcripts,")
        print("12288 truncates 2.2%), or run --folds 5. Fewer folds only adds noise to the mean, it is not a cheat.")
        return

    scorable = np.flatnonzero(bu.scored)
    trainonly = np.flatnonzero(~bu.scored)
    if args.selftest:
        scorable = np.random.default_rng(0).permutation(scorable)[:240]
        trainonly = np.random.default_rng(1).permutation(trainonly)[:60]
    idx = scorable
    posmap = {int(g): i for i, g in enumerate(idx)}
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(idx)) % 10
    print(f"folds built over {len(idx)} scorable genes; {len(trainonly)} train-only genes "
          f"added to every training set")
    preds = np.full(len(idx), np.nan)
    per_fold, components = [], []
    for k in range(args.folds):
        te = idx[fold == k]
        inner = idx[fold != k]
        vcut = max(2, len(inner) // 9)
        perm = np.random.default_rng(100 + k).permutation(len(inner))
        va, tr = inner[perm[:vcut]], np.concatenate([inner[perm[vcut:]], trainonly])
        print(f"\n=== fold {k}  train {len(tr)}  val {len(va)}  test {len(te)} ===", flush=True)
        acc = np.zeros(len(te))
        nparam = 0
        for s in range(args.seeds):
            pt, bv, nparam, comp = run_fold(bu, tr, va, te, dev, args, seed=1234 + 17 * s)
            if comp:
                components.append(comp)
            acc += pt
        pt = acc / args.seeds
        preds[np.array([posmap[t] for t in te])] = pt
        r = pearson(bu.y[te], pt)
        per_fold.append(r)
        print(f"  fold {k}: test r = {r:.4f}   (running mean {np.mean(per_fold):.4f})", flush=True)

    done = ~np.isnan(preds)
    if components:
        print(f"\n{'component':26s}{'mean fold r':>13s}")
        for k in ("nn", "gbm", "stack"):
            v = [c[k] for c in components]
            print(f"{k:26s}{np.mean(v):13.4f}")
    print(f"\n{'arm':26s}{'mean fold r':>13s}{'sd':>8s}{'pooled r':>11s}")
    print(f"{'landmark model':26s}{np.mean(per_fold):13.4f}{np.std(per_fold):8.4f}"
          f"{pearson(bu.y[idx[done]], preds[done]):11.4f}")
    ref = SALUKI.get(bu.species, SALUKI["human"])
    print(f"{'Saluki ' + bu.species + ' (published)':26s}{ref:13.4f}")
    if bu.species == "human":
        print(f"{'BEeM (their features)':26s}{BEEM_HUMAN:13.4f}")
        print(f"{'our features + GBM':26s}{FEATURES_GBM:13.4f}")
    if args.aux_drop >= 1.0:
        print(f"{'Saluki sequence-only':26s}{SEQ_ONLY_SALUKI:13.4f}   <- the relevant reference for this arm")
    d = float(np.mean(per_fold)) - ref
    print(f"\nvs Saluki: {d:+.4f}  ->  {'BEATS IT' if d > 0 else 'does not beat it'}")
    if args.folds < 10:
        print(f"NOTE: only {args.folds} of 10 folds run, so this mean is noisier than the published one.")
    if args.selftest:
        print("\nSELFTEST ONLY -- 240 transcripts, 2 epochs. Numbers are meaningless; the point is that it ran.")
    json.dump(dict(tag=args.tag or args.size, size=args.size, species=bu.species,
                   params=int(nparam), multitask=bool(nt), aux_drop=args.aux_drop,
                   per_fold=per_fold, mean=float(np.mean(per_fold)), sd=float(np.std(per_fold)),
                   pooled=pearson(bu.y[idx[done]], preds[done]),
                   saluki=ref, folds=args.folds, epochs=args.epochs,
                   lmax=args.lmax, seeds=args.seeds, n_scorable=int(len(idx)),
                   components={k: float(np.mean([c[k] for c in components])) for k in ("nn", "gbm", "stack")}
                   if components else None),
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
