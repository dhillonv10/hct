#!/usr/bin/env python3
"""Is the read-out the bottleneck in mRNA property prediction, rather than the backbone?

Every model on mRNABench is scored the same way: run the transcript through the backbone, MEAN-POOL the token
embeddings into one vector, fit a linear probe. For mRNA that step discards region, frame and position -- and
frame in particular decides whether an upstream AUG gives a harmless N-terminal extension or a competing ORF
that represses translation hard. The information is in the token embeddings; the read-out averages it away.

This script holds the backbone FROZEN and changes only the read-out, so the comparison is airtight: same
weights, same probe, same splits, same selection rule. Mean-pooling is one cell of the structured basis, so
the baseline is nested inside the treatment and cannot be beaten by accident.

Arms (all selected on `val` only; reported on iid_test / ood_test / repro_val):
  B1  mean-pool + ridge               the mRNABench protocol -- the thing to beat
  B2  explicit 5'UTR features + ridge no language model at all
  B3  scratch CNN on one-hot          the Optimus-5-Prime-style play
  R   structured pooling + ridge      the hypothesis (joint, and as a per-cell stack)
  S   stack of B3 + R                 the attempt on the published number

Published target, already reproduced by this repo's harness to 4 d.p.: RiNALMo fine-tuned
r2 = 0.925 on Random7600 (the authors' validation set) and 0.860 on Human7600.

  python3 readout.py selftest                      # whole pipeline, synthetic data, 2-block model, ~2 min
  python3 readout.py data    --work WORK
  python3 readout.py embed   --work WORK --lm micro --ckpt /path/rinalmo_micro_pretrained.pt
  python3 readout.py train   --work WORK
  python3 readout.py report  --work WORK
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distill"))

PUBLISHED = {"repro_val": 0.925, "ood_test": 0.860}       # RiNALMo fine-tuned, Sample et al. splits
SPLITS = ["train", "val", "iid_test", "ood_test", "repro_val"]
GATE = 0.01                                               # the r2 margin DESIGN.md pre-registers for "R beats B1"

# ------------------------------------------------------------------------------------ structured pooling (the contribution)

# Cell meanings are fixed here and never changed after seeing a result. `mean` first so that B1 is literally
# the first slice of the basis.
CELLS = ["mean", "max", "std",
         "frame0", "frame1", "frame2",          # distance to the main AUG mod 3, over the whole UTR
         "fnear0", "fnear1", "fnear2",          # same, last 30 nt only
         "cap10", "aug10",                      # first 10 nt (cap-proximal) and last 10 nt (AUG-proximal)
         "q1", "q2", "q3",                      # positional thirds
         "first", "last"]                        # boundary tokens


def pool_cells(H, lens):
    """Structured pooling of per-token reps.

    H    : (B, Lmax, d) reps for the SEQUENCE tokens only -- <cls>/<eos> already stripped.
    lens : (B,) true lengths.
    Returns {cell: (B, d)} for every name in CELLS, in that order.

    Frame is defined by distance to the main AUG, which sits immediately 3' of the UTR: token i is at
    distance L-i, so (L-i) % 3 == 0 means in-frame with the main ORF.
    """
    import torch
    B, Lmax, d = H.shape
    dev = H.device
    pos = torch.arange(Lmax, device=dev).unsqueeze(0).expand(B, Lmax)      # (B,Lmax)
    L = lens.unsqueeze(1)
    valid = pos < L
    dist = L - pos                                                          # distance from token i to the AUG

    def mmean(m):
        m = (m & valid).unsqueeze(-1).to(H.dtype)
        return (H * m).sum(1) / m.sum(1).clamp(min=1.0)

    out = {}
    out["mean"] = mmean(torch.ones_like(valid))
    out["max"] = torch.where(valid.unsqueeze(-1), H, torch.full_like(H, -1e30)).max(1).values
    cen = (H - out["mean"].unsqueeze(1)) * valid.unsqueeze(-1).to(H.dtype)
    out["std"] = (cen.pow(2).sum(1) / L.to(H.dtype).clamp(min=1.0)).clamp(min=0).sqrt()
    for f in range(3):
        out[f"frame{f}"] = mmean((dist % 3) == f)
    for f in range(3):
        out[f"fnear{f}"] = mmean(((dist % 3) == f) & (dist <= 30))
    out["cap10"] = mmean(pos < 10)
    out["aug10"] = mmean(dist <= 10)
    th = L.to(H.dtype) / 3.0
    p = pos.to(H.dtype)
    out["q1"] = mmean(p < th)
    out["q2"] = mmean((p >= th) & (p < 2 * th))
    out["q3"] = mmean(p >= 2 * th)
    out["first"] = H[:, 0, :]
    out["last"] = H[torch.arange(B, device=dev), (lens - 1).clamp(min=0)]
    assert list(out) == CELLS, (list(out), CELLS)
    return out


# ------------------------------------------------------------------------------------ explicit features (arm B2, no LM)

_K1 = list("ACGT")
_K2 = [a + b for a in _K1 for b in _K1]
_K3 = [a + b + c for a in _K1 for b in _K1 for c in _K1]


def utr_features(s):
    """Known 5'UTR translation biology as explicit numbers. No learned component, no language model."""
    import RNA
    s = s.upper().replace("U", "T")
    L = len(s)
    f = {"len": L, "log_len": math.log(L), "gc": sum(s.count(c) for c in "GC") / L}
    for k in _K1: f[f"m1_{k}"] = s.count(k) / L
    for k in _K2: f[f"m2_{k}"] = s.count(k) / max(L - 1, 1)
    for k in _K3: f[f"m3_{k}"] = s.count(k) / max(L - 2, 1)

    # upstream AUGs, split by frame relative to the main ORF, and by whether the uORF closes before it
    augs = [i for i in range(L - 2) if s[i:i + 3] == "ATG"]
    f["n_uaug"] = len(augs)
    for fr in range(3):
        f[f"n_uaug_f{fr}"] = sum(1 for i in augs if (L - i) % 3 == fr)
    closed = oop = 0
    for i in augs:
        stop = next((j for j in range(i + 3, L - 2, 3) if s[j:j + 3] in ("TAA", "TAG", "TGA")), None)
        if stop is not None: closed += 1
        else: oop += 1 if (L - i) % 3 else 0
    f["n_uorf_closed"] = closed          # terminates inside the UTR -> reinitiation possible
    f["n_uaug_oof_open"] = oop           # out-of-frame and overlaps the main ORF -> the strong repressor
    f["first_uaug_pos"] = augs[0] / L if augs else 1.0
    f["last_uaug_dist"] = (L - augs[-1]) / L if augs else 1.0
    f["kozak_m3_purine"] = float(bool(augs) and augs[0] >= 3 and s[augs[0] - 3] in "AG")

    md = RNA.md(); md.max_bp_span = 100
    fc = RNA.fold_compound(s.replace("T", "U"), md)
    st, mfe = fc.mfe()
    f["mfe_per_nt"] = mfe / L
    f["unpaired"] = st.count(".") / L
    f["unpaired_cap"] = st[:15].count(".") / min(15, L)
    f["unpaired_aug"] = st[-15:].count(".") / min(15, L)
    return f


def feature_matrix(seqs, nproc=None):
    from multiprocessing import Pool
    nproc = nproc or max(1, os.cpu_count() or 1)
    with Pool(nproc) as p:
        rows = p.map(utr_features, list(seqs), chunksize=256)
    F = pd.DataFrame(rows)
    return np.nan_to_num(F.values.astype(np.float32)), list(F.columns)


# ------------------------------------------------------------------------------------ probes

def ridge_multi(Xtr, ytr, Xs, alphas):
    """Ridge over many alphas at the cost of one eigendecomposition.

    Solves w(a) = (X'X + aI)^-1 X'y by eigendecomposing the Gram matrix once, so the alpha sweep is free.
    That is what makes a 1500-7000 dim basis affordable on CPU. Returns {alpha: [preds per X in Xs]}.
    """
    n, d = Xtr.shape
    mu = Xtr.mean(0, dtype=np.float64)
    sd = Xtr.std(0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    y = np.asarray(ytr, np.float64)
    ym = float(y.mean())
    G = np.zeros((d, d)); b = np.zeros(d)
    for i in range(0, n, 8192):                       # chunked: peak memory is one chunk, not the whole matrix
        A = (Xtr[i:i + 8192].astype(np.float64) - mu) / sd
        G += A.T @ A
        b += A.T @ (y[i:i + 8192] - ym)
    ev, V = np.linalg.eigh(G)
    Vb = V.T @ b
    out = {}
    for a in alphas:
        w = V @ (Vb / (ev + a))
        out[a] = [((np.asarray(X, np.float64) - mu) / sd) @ w + ym for X in Xs]
    return out


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_select(Xtr, ytr, blocks, yval, alphas=None):
    """Fit on train, pick alpha on validation r2, return (preds per block, chosen alpha, val r2).

    `blocks` is an ordered dict of name -> X; the first MUST be the validation block, since alpha is chosen
    on it. Selection never touches a test set.
    """
    alphas = alphas if alphas is not None else np.logspace(-2, 6, 25)
    names = list(blocks)
    assert names[0] == "val", names
    got = ridge_multi(Xtr, ytr, [blocks[n] for n in names], alphas)
    best_a = max(alphas, key=lambda a: r2(yval, got[a][0]))
    return dict(zip(names, got[best_a])), float(best_a), r2(yval, got[best_a][0])


def pca_fit(X, k):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    C = np.cov((X - mu).T)
    ev, V = np.linalg.eigh(C)
    W = V[:, ::-1][:, :k]
    return mu, W


# ------------------------------------------------------------------------------------ scratch CNN (arm B3)

def train_cnn(tr_seqs, tr_y, blocks, val_y, max_len=100, epochs=24, bs=256, seed=0, width=120, verbose=True):
    """Optimus-5-Prime-style 1D CNN on one-hot, trained from scratch. No language model.

    Sequences are RIGHT-ALIGNED (padded on the left) so the main AUG is always at the same index and frame is
    consistent across lengths -- the same reason the structured pooling defines frame by distance to the AUG.
    """
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    def onehot(seqs):
        X = np.zeros((len(seqs), 4, max_len), np.float32)
        ix = {c: i for i, c in enumerate("ACGT")}
        for r, s in enumerate(seqs):
            s = s.upper().replace("U", "T")[-max_len:]
            off = max_len - len(s)                                 # right-align
            for j, c in enumerate(s):
                if c in ix: X[r, ix[c], off + j] = 1.0
        return X

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv1d(4, width, 8, padding="same"), nn.ReLU(),
                nn.Conv1d(width, width, 8, padding="same"), nn.ReLU(),
                nn.Conv1d(width, width, 8, padding="same"), nn.ReLU())
            self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.2),
                                   nn.Linear(width * max_len, 40), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(40, 1))

        def forward(self, x): return self.h(self.c(x)).squeeze(-1)

    Xtr, ytr = torch.from_numpy(onehot(tr_seqs)), torch.tensor(np.asarray(tr_y, np.float32))
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    ytr = (ytr - ymu) / ysd
    Bl = {k: torch.from_numpy(onehot(v)) for k, v in blocks.items()}
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xtr)
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    rs = np.random.default_rng(seed)
    best, best_preds, done = -np.inf, None, 0

    def predict(X):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                out.append(net(X[i:i + 512]).numpy())
        net.train()
        return np.concatenate(out) * ysd + ymu

    for ep in range(epochs):
        order = rs.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step(); sched.step(); done += 1
            tot += float(loss) * len(idx)
        pv = predict(Bl["val"])
        v = r2(val_y, pv)
        if v > best:
            best, best_preds = v, {k: predict(X) for k, X in Bl.items()}
        if verbose:
            print(f"    cnn epoch {ep + 1}/{epochs} train_mse {tot / n:.4f} val_r2 {v:.4f}"
                  f"{'  *' if v == best else ''}", flush=True)
    assert done == steps, (done, steps)
    return best_preds, best


# ------------------------------------------------------------------------------------ stages

def stage_data(work, synthetic=False, seed=0, top_k=100, n_syn=14000):
    """top_k is the authors' selection rule: the top `k` rows by read depth at each length, 25..100.

    k=100 reproduces their Random7600/Human7600 exactly and is what the real run uses. The selftest needs a
    smaller k, because three blocks of 7600 would consume a synthetic table whole and leave no training rows.
    """
    from rnakd import mrl
    os.makedirs(work, exist_ok=True)
    if synthetic:
        df = mrl.synthetic(n=n_syn, seed=seed)
        csv = os.path.join(work, "synthetic.csv")
        df.to_csv(csv, index=False)
    else:
        csv = mrl.download(work)
    sp = mrl.splits(csv, seed=seed, top_k=top_k)
    for k, v in sp.items():
        if len(v) == 0:
            raise SystemExit(f"split '{k}' is empty -- top_k={top_k} is too large for {len(pd.read_csv(csv))} rows")
    for k, v in sp.items():
        v[["utr", "rl", "total_reads", "set"]].to_csv(os.path.join(work, f"{k}.csv"), index=False)
    print(f"wrote {len(sp)} split files to {work}")
    return sp


def load_splits(work):
    out = {}
    for k in SPLITS:
        p = os.path.join(work, f"{k}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run the `data` stage first")
        out[k] = pd.read_csv(p)
    return out


def stage_embed(work, lm, ckpt, limit=0, bs=64, max_len=100, permute=False):
    """Frozen backbone -> structured pooled cells, cached per split. Per-token reps are never stored.

    `permute` is THE control for the obvious objection to the headline: R-joint has 16x B1's dimensionality,
    so it might win on capacity rather than on region/frame structure. With permute=True the token reps are
    randomly shuffled along the length axis WITHIN each sequence before pooling. Every cell still exists,
    the basis is still 7680 dims, the alpha sweep is identical -- but each cell is now a mean over a random
    subset of the same tokens, so position and frame carry no information.

      control ~= R-joint  -> the gain was dimensionality, and the structural story is dead
      control ~= B1       -> the gain is the structure, which is the claim

    Note the permutation is applied to the OUTPUT reps, not the input tokens: the backbone still sees the
    real sequence in the real order, so this isolates the read-out and nothing else.
    """
    import torch
    from rnakd import teacher
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = teacher.LM_CONFIGS[lm]
    model = teacher.RiNALMo(**cfg).eval()
    if ckpt:
        teacher.load_strict(model, ckpt)
        print(f"loaded {ckpt} (strict key match = the correctness test)")
    else:
        print(f"NO CHECKPOINT -- random init, for pipeline rehearsal only")
    d = cfg["embed"]
    sp = load_splits(work)
    tag = f"{lm}perm" if permute else lm
    for name, df in sp.items():
        out = os.path.join(work, f"cells_{tag}_{name}.npz")
        if os.path.exists(out):
            print(f"  {name}: cached"); continue
        seqs = df.utr.tolist()
        if limit and name == "train":
            seqs = seqs[:limit]
        n = len(seqs)
        lens = np.array([min(len(s), max_len) for s in seqs])
        order = np.argsort(lens, kind="mergesort")              # length buckets: pad to the batch max, not 100
        store = {c: np.zeros((n, d), np.float32) for c in CELLS}
        t0 = time.time()
        with torch.inference_mode():
            for bi in range(0, n, bs):
                idx = order[bi:bi + bs]
                bl = lens[idx]
                pad_to = int(bl.max()) + 2
                toks = torch.tensor([teacher.encode(seqs[i][:max_len], pad_to) for i in idx])
                rep, _ = model(toks)
                H = rep[:, 1:, :]                                # drop <cls>; <eos>/<pad> are masked out by lens
                if permute:
                    H = H.clone()
                    g = torch.Generator().manual_seed(917 + bi)
                    for r in range(H.shape[0]):
                        Lr = int(bl[r])
                        H[r, :Lr] = H[r, :Lr][torch.randperm(Lr, generator=g)]
                cells = pool_cells(H, torch.tensor(bl))
                for c in CELLS:
                    store[c][idx] = cells[c].float().numpy()
                if bi % (bs * 20) == 0:
                    el = time.time() - t0
                    print(f"  {name} {bi}/{n}  {el:.0f}s  eta {el / max(bi + bs, 1) * (n - bi) / 60:.1f} min",
                          flush=True)
        np.savez_compressed(out, lens=lens, y=df.rl.values[:n], **store)
        print(f"  {name}: {n} x {len(CELLS)} cells x {d}  in {time.time() - t0:.0f}s -> {out}")


def _cells(work, lm, name):
    z = np.load(os.path.join(work, f"cells_{lm}_{name}.npz"))
    return {c: z[c] for c in CELLS}, z["y"]


def stage_train(work, lm, pca_k=96, cnn_epochs=24, skip_cnn=False, seed=0):
    import torch
    sp = load_splits(work)
    cellset, Y = {}, {}
    for name in SPLITS:
        cellset[name], Y[name] = _cells(work, lm, name)
    ntr = len(Y["train"])
    seqs = {k: sp[k].utr.tolist()[: (ntr if k == "train" else None)] for k in SPLITS}
    for k in SPLITS:
        assert len(seqs[k]) == len(Y[k]), (k, len(seqs[k]), len(Y[k]))
    evals = [k for k in SPLITS if k != "train"]
    res, preds = {}, {}

    def record(arm, p, va, extra=None):
        res[arm] = dict(val_r2=va, **{k: r2(Y[k], p[k]) for k in evals}, **(extra or {}))
        preds[arm] = p
        print(f"  {arm:18s} val {va:.4f}  " + "  ".join(f"{k} {res[arm][k]:.4f}" for k in evals), flush=True)

    print(f"\n[B1] mean-pool + ridge  (the mRNABench protocol)   n_train={ntr}  d={cellset['train']['mean'].shape[1]}")
    blocks = {k: cellset[k]["mean"] for k in evals}
    p, a, va = fit_select(cellset["train"]["mean"], Y["train"], blocks, Y["val"])
    record("B1-meanpool", p, va, {"alpha": a, "dims": int(cellset["train"]["mean"].shape[1])})

    # Full dimension by default. This matters for the central claim: with every cell at its native width,
    # `mean` occupies the first 480 columns, so B1's hypothesis class is a strict SUBSET of R-joint's and a
    # loss cannot be blamed on compression. pca_k > 0 compresses each cell instead, as a robustness check.
    print(f"\n[R-joint] structured pooling + ridge  ({'full dims' if not pca_k else f'per-cell PCA({pca_k})'})")
    Xj = {k: [] for k in SPLITS}
    for c in CELLS:
        if pca_k:
            mu, W = pca_fit(cellset["train"][c], pca_k)
            for k in SPLITS: Xj[k].append((cellset[k][c] - mu) @ W)
        else:
            for k in SPLITS: Xj[k].append(cellset[k][c])
    for k in SPLITS: Xj[k] = np.hstack(Xj[k]).astype(np.float32)
    assert np.allclose(Xj["val"][:, :cellset["val"]["mean"].shape[1]], cellset["val"]["mean"]), \
        "B1 must be nested as the leading columns of the R-joint basis"
    print(f"  basis dims {Xj['train'].shape[1]} ({len(CELLS)} cells)")
    p, a, va = fit_select(Xj["train"], Y["train"], {k: Xj[k] for k in evals}, Y["val"])
    record("R-joint", p, va, {"alpha": a, "dims": int(Xj["train"].shape[1])})

    print(f"\n[R-stack] one ridge per cell, level-2 on validation predictions (gives the ablation free)")
    L1 = {}
    for c in CELLS:
        pc, ac, vc = fit_select(cellset["train"][c], Y["train"], {k: cellset[k][c] for k in evals}, Y["val"])
        L1[c] = (pc, vc)
        print(f"    cell {c:8s} val r2 {vc:.4f}", flush=True)
    Zv = np.stack([L1[c][0]["val"] for c in CELLS], 1)
    lvl2 = np.linalg.lstsq(np.hstack([Zv, np.ones((len(Zv), 1))]), Y["val"], rcond=None)[0]
    pst = {}
    for k in evals:
        Z = np.stack([L1[c][0][k] for c in CELLS], 1)
        pst[k] = np.hstack([Z, np.ones((len(Z), 1))]) @ lvl2
    record("R-stack", pst, r2(Y["val"], pst["val"]),
           {"level2_weights": {c: float(w) for c, w in zip(CELLS, lvl2[:-1])}})
    res["R-stack"]["per_cell_val_r2"] = {c: L1[c][1] for c in CELLS}

    print(f"\n[B2] explicit 5'UTR features + ridge  (no language model)")
    Fm = {}
    for k in SPLITS:
        fp = os.path.join(work, f"feat_{k}.npy")
        if os.path.exists(fp):
            Fm[k] = np.load(fp)
        else:
            Fm[k], cols = feature_matrix(seqs[k])
            np.save(fp, Fm[k])
            if k == "train": json.dump(cols, open(os.path.join(work, "feat_cols.json"), "w"))
        print(f"    {k}: {Fm[k].shape}", flush=True)
    p, a, va = fit_select(Fm["train"], Y["train"], {k: Fm[k] for k in evals}, Y["val"])
    record("B2-features", p, va, {"alpha": a, "dims": int(Fm["train"].shape[1])})

    print(f"\n[R+B2] does the language model add anything on top of explicit biology?")
    parts2 = ["B2-features", "R-joint"]
    Zv = np.stack([preds[a_]["val"] for a_ in parts2], 1)
    w2 = np.linalg.lstsq(np.hstack([Zv, np.ones((len(Zv), 1))]), Y["val"], rcond=None)[0]
    p2 = {}
    for k in evals:
        Z = np.stack([preds[a_][k] for a_ in parts2], 1)
        p2[k] = np.hstack([Z, np.ones((len(Z), 1))]) @ w2
    record("R+B2", p2, r2(Y["val"], p2["val"]),
           {"weights": {a_: float(x) for a_, x in zip(parts2, w2[:-1])},
            "marginal_vs_B2": {k: float(r2(Y[k], p2[k]) - res["B2-features"][k]) for k in evals}})

    if not skip_cnn:
        print(f"\n[B3] scratch CNN on one-hot  ({cnn_epochs} epochs, {ntr} train rows)")
        pc, vc = train_cnn(seqs["train"], Y["train"], {k: seqs[k] for k in evals}, Y["val"],
                           epochs=cnn_epochs, seed=seed)
        record("B3-cnn", pc, vc)

        print(f"\n[S] stack: B3 + R-joint (+B2), level-2 on validation predictions")
        parts = ["B3-cnn", "R-joint", "B2-features"]
        Zv = np.stack([preds[a_]["val"] for a_ in parts], 1)
        w = np.linalg.lstsq(np.hstack([Zv, np.ones((len(Zv), 1))]), Y["val"], rcond=None)[0]
        ps = {}
        for k in evals:
            Z = np.stack([preds[a_][k] for a_ in parts], 1)
            ps[k] = np.hstack([Z, np.ones((len(Z), 1))]) @ w
        record("S-stack", ps, r2(Y["val"], ps["val"]),
               {"parts": parts, "weights": {a_: float(x) for a_, x in zip(parts, w[:-1])}})

    out = os.path.join(work, f"results_{lm}.json")
    json.dump({"lm": lm, "n_train": ntr, "arms": res}, open(out, "w"), indent=1, default=float)
    pp = os.path.join(work, f"preds_{lm}.npz")
    np.savez_compressed(pp, **{f"{arm}__{k}": v for arm, p_ in preds.items() for k, v in p_.items()},
                        **{f"y__{k}": Y[k] for k in evals})
    print(f"\nwrote {out} and {pp}")
    return res


def stage_report(work, lm):
    R = json.load(open(os.path.join(work, f"results_{lm}.json")))
    arms = R["arms"]
    ev = ["val_r2", "iid_test", "ood_test", "repro_val"]
    print(f"\nMRL, frozen RiNALMo-{lm}, n_train={R['n_train']}\n")
    print(f"{'arm':20s}" + "".join(f"{e:>12s}" for e in ev))
    print("-" * (20 + 12 * len(ev)))
    for a, v in arms.items():
        print(f"{a:20s}" + "".join(f"{v.get(e, float('nan')):12.4f}" for e in ev))
    print(f"\n{'published (RiNALMo 651M fine-tuned)':20s}"
          f"{'':12s}{'':12s}{PUBLISHED['ood_test']:12.4f}{PUBLISHED['repro_val']:12.4f}")

    if "B1-meanpool" in arms:
        print("\n=== the pre-registered contrast: does the read-out matter? ===")
        b1 = arms["B1-meanpool"]
        for arm in ("R-joint", "R-stack"):
            if arm not in arms: continue
            d = {k: arms[arm][k] - b1[k] for k in ("iid_test", "ood_test", "repro_val")}
            both = d["iid_test"] >= GATE and d["ood_test"] >= GATE
            print(f"  {arm} - B1:  " + "  ".join(f"{k} {v:+.4f}" for k, v in d.items()))
            print(f"    pre-registered gate (>= +{GATE} on BOTH test sets): "
                  f"{'MET -- read-out claim holds' if both else 'NOT MET'}")
        if "B2-features" in arms:
            gap = b1["ood_test"] - arms["B2-features"]["ood_test"]
            print(f"\n  B1 - B2 on ood_test: {gap:+.4f}"
                  + ("   <- the backbone is not earning its parameters here" if abs(gap) < GATE else ""))
        for arm in ("S-stack", "B3-cnn"):
            if arm in arms:
                print(f"\n  {arm} vs published: ood_test {arms[arm]['ood_test']:.4f} vs "
                      f"{PUBLISHED['ood_test']:.3f}  ({arms[arm]['ood_test'] - PUBLISHED['ood_test']:+.4f}),  "
                      f"repro_val {arms[arm]['repro_val']:.4f} vs {PUBLISHED['repro_val']:.3f}  "
                      f"({arms[arm]['repro_val'] - PUBLISHED['repro_val']:+.4f})")

    if "R-stack" in arms and "level2_weights" in arms["R-stack"]:
        print("\n=== mechanism: which pooling cells carry the signal ===")
        pc = arms["R-stack"].get("per_cell_val_r2", {})
        w = arms["R-stack"]["level2_weights"]
        print(f"  {'cell':10s}{'solo val r2':>14s}{'level-2 weight':>16s}")
        for c in sorted(pc, key=lambda c: -abs(w.get(c, 0))):
            print(f"  {c:10s}{pc[c]:14.4f}{w.get(c, 0):16.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "data", "embed", "train", "report"])
    ap.add_argument("--work", default="/home/user/mrl_work")
    ap.add_argument("--lm", default="micro", choices=["selftest", "micro", "mega", "giga"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit", type=int, default=0, help="cap the training rows (0 = all)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--pca-k", type=int, default=0, help="0 = full per-cell dims (default); k>0 = PCA each cell to k")
    ap.add_argument("--cnn-epochs", type=int, default=24)
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--permute", action="store_true",
                    help="dimensionality control: shuffle token reps within each sequence before pooling")
    a = ap.parse_args()

    if a.stage == "selftest":
        w = os.path.join(a.work, "selftest")
        print("=== selftest: synthetic data, 2-block random-init model, whole pipeline ===")
        stage_data(w, synthetic=True, top_k=10)
        stage_embed(w, "selftest", "", limit=3000, bs=64)
        stage_train(w, "selftest", pca_k=0, cnn_epochs=3, seed=0)
        stage_report(w, "selftest")
        print("\nselftest done -- the pipeline runs end to end. Numbers here are meaningless by construction.")
        return
    if a.stage == "data":   stage_data(a.work, synthetic=a.synthetic); return
    if a.stage == "embed":  stage_embed(a.work, a.lm, a.ckpt, limit=a.limit, bs=a.bs,
                                        permute=a.permute); return
    if a.stage == "train":  stage_train(a.work, a.lm, pca_k=a.pca_k, cnn_epochs=a.cnn_epochs,
                                        skip_cnn=a.skip_cnn); return
    if a.stage == "report": stage_report(a.work, a.lm); return


if __name__ == "__main__":
    main()
