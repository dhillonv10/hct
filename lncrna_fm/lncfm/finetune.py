"""Adaptation ladder for RiNALMo: how much of the backbone should move?

    python -m lncfm.finetune --task loc_reg --modes frozen lora topk full --folds 0 1 2 3 4 --epochs 4
    python -m lncfm.finetune --task rbp_K562 --modes frozen lora --max-nt 512 --device auto

Rungs (same data, same folds, same head, same early stopping; only what is trainable differs):
  frozen  backbone fixed, linear head on mean-pooled embeddings (embeddings cached; this is the "probe")
  lora    backbone fixed + low-rank adapters on query/value (own implementation, r=8, alpha=16) + head
  topk    last K transformer layers + head
  full    everything, low learning rate
For every held-out fold: train on the other dev folds (20% of them held out for epoch selection), predict the fold.
Outputs pooled out-of-fold metrics, paired differences vs `frozen`, trainable-parameter counts and seconds/step
(use the latter to budget your M4 run). The locked test set is never touched here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import evalkit, lncbench
from .embed import HF_MODELS
from .model import pick_device


# ------------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))       # zero init: starts exactly at the pretrained model
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(model, targets=("query", "value"), r=8, alpha=16, dropout=0.05):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets:
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout)); n += 1
    return n


def set_trainable(model, mode, k=2):
    for p in model.parameters():
        p.requires_grad = False
    if mode == "lora":
        for n, p in model.named_parameters():
            if n.endswith(".A") or n.endswith(".B"):
                p.requires_grad = True
    elif mode == "topk":
        L = len(model.encoder.layer)
        for i in range(L - k, L):
            for p in model.encoder.layer[i].parameters():
                p.requires_grad = True
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Net(nn.Module):
    def __init__(self, backbone, n_out, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.config.hidden_size, n_out)

    def pooled(self, ids, am):
        h = self.backbone(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, ids, am):
        return self.head(self.drop(self.pooled(ids, am)))


# ------------------------------------------------------------------------------------------- training
def _loss(out, y, kind):
    if kind == "reg":
        return F.mse_loss(out.squeeze(-1), y)
    return F.binary_cross_entropy_with_logits(out.squeeze(-1) if kind == "clf" else out, y)


def _val_score(y, p, kind):
    if kind == "multi":
        return float(np.nanmean([evalkit._auc_fast(y[:, j], p[:, j]) for j in range(y.shape[1])]))
    return evalkit._spearman(y, p) if kind == "reg" else evalkit._auc_fast(y, p)


def _batches(idx, bs, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def _enc(tok, seqs, max_nt, device):
    e = tok([s[:max_nt] for s in seqs], padding=True, truncation=True, return_tensors="pt", max_length=max_nt + 2)
    return e["input_ids"].to(device), e["attention_mask"].to(device)


@torch.no_grad()
def _predict(net, tok, seqs, idx, max_nt, bs, device, cache=None):
    net.eval()
    outs = []
    for b in _batches(idx, bs, False, None):
        ids, am = _enc(tok, [seqs[i] for i in b], max_nt, device)
        outs.append(net(ids, am).float().cpu().numpy())
    return np.concatenate(outs)


def run_fold(mode, seqs, y, kind, train_idx, test_idx, args, tok, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    from transformers import AutoModel
    import multimolecule  # noqa: F401
    backbone = AutoModel.from_pretrained(HF_MODELS[args.model][0])
    if mode == "lora":
        apply_lora(backbone, r=args.lora_r, alpha=2 * args.lora_r)
    n_train = set_trainable(backbone, mode, args.topk)
    n_out = y.shape[1] if kind == "multi" else 1
    net = Net(backbone, n_out).to(device)
    ys = y.astype(np.float32)
    if kind == "reg":
        mu, sd = ys[train_idx].mean(), ys[train_idx].std() + 1e-6
        ys = (ys - mu) / sd
    perm = rng.permutation(train_idx)
    nva = max(int(0.2 * len(perm)), 16)
    va_idx, tr_idx = perm[:nva], perm[nva:]
    lr_bb = {"lora": args.lr_lora, "topk": args.lr_topk, "full": args.lr_full, "frozen": 0.0}[mode]
    groups = [{"params": net.head.parameters(), "lr": args.lr_head}]
    bb = [p for p in net.backbone.parameters() if p.requires_grad]
    if bb:
        groups.append({"params": bb, "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch)
    total = steps_per_epoch * args.epochs
    step, best, best_state, t_steps = 0, -9, None, []

    # frozen: embeddings never change -> compute once, train the head on cached vectors
    cache = None
    if mode == "frozen":
        net.eval()
        with torch.no_grad():
            cache = np.zeros((len(seqs), backbone.config.hidden_size), np.float32)
            allidx = np.concatenate([train_idx, test_idx])
            for b in _batches(allidx, args.batch * 2, False, None):
                ids, am = _enc(tok, [seqs[i] for i in b], args.max_nt, device)
                cache[b] = net.pooled(ids, am).float().cpu().numpy()
        mu_c, sd_c = cache[train_idx].mean(0), cache[train_idx].std(0) + 1e-6
        cache = (cache - mu_c) / sd_c

    for ep in range(args.epochs):
        net.train()
        if mode == "frozen":
            net.backbone.eval()
        for b in _batches(tr_idx, args.batch, True, rng):
            t0 = time.time()
            warm = min(1.0, (step + 1) / max(1, 0.1 * total))
            decay = max(0.0, 1 - step / total)
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * warm * decay
            yb = torch.from_numpy(ys[b]).to(device)
            if mode == "frozen":
                out = net.head(net.drop(torch.from_numpy(cache[b]).to(device)))
            else:
                ids, am = _enc(tok, [seqs[i] for i in b], args.max_nt, device)
                out = net(ids, am)
            loss = _loss(out, yb, kind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            step += 1
            t_steps.append(time.time() - t0)
        # epoch selection on inner validation
        if mode == "frozen":
            net.eval()
            with torch.no_grad():
                pv = net.head(torch.from_numpy(cache[va_idx]).to(device)).float().cpu().numpy()
        else:
            pv = _predict(net, tok, seqs, va_idx, args.max_nt, args.batch * 2, device)
        pv = pv if kind == "multi" else pv.ravel()
        sc = _val_score(ys[va_idx] if kind != "multi" else y[va_idx], pv, kind)
        if sc > best:
            best = sc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if (k.startswith("head") or "lora" in k.lower() or ".A" in k or ".B" in k
                          or any(p.requires_grad for n, p in net.named_parameters() if n == k))}
    net.load_state_dict({**net.state_dict(), **best_state})
    if mode == "frozen":
        net.eval()
        with torch.no_grad():
            pt = net.head(torch.from_numpy(cache[test_idx]).to(device)).float().cpu().numpy()
    else:
        pt = _predict(net, tok, seqs, test_idx, args.max_nt, args.batch * 2, device)
    pt = pt if kind == "multi" else pt.ravel()
    return pt, {"trainable_params": int(n_train + sum(p.numel() for p in net.head.parameters())), "best_val": float(best),
                "sec_per_step": float(np.median(t_steps)) if t_steps else 0.0, "steps": step}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="loc_reg")
    ap.add_argument("--model", default="rinalmo-micro")
    ap.add_argument("--modes", nargs="+", default=["frozen", "lora", "topk", "full"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-nt", type=int, default=512)
    ap.add_argument("--max-n", type=int, default=None, help="subsample the task (CPU smoke tests)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=3e-4)
    ap.add_argument("--lr-topk", type=float, default=5e-5)
    ap.add_argument("--lr-full", type=float, default=2e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import multimolecule  # noqa: F401
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODELS[args.model][0])
    task = lncbench.subsample(lncbench.load_task(args.task), args.max_n)
    seqs, y, kind = list(task.df.seq), task.y, task.kind
    fold = task.df.fold.values
    print(f"task {args.task} ({kind}) n={len(seqs)} dev={int((fold>=0).sum())} | device={device} model={args.model} | folds {args.folds}", flush=True)
    results, oofs = {}, {}
    for mode in args.modes:
        oof = np.full(y.shape if kind == "multi" else len(y), np.nan)
        info = []
        for f in args.folds:
            te = np.where(fold == f)[0]
            tr = np.where((fold >= 0) & (fold != f))[0]
            t0 = time.time()
            p, meta = run_fold(mode, seqs, y, kind, tr, te, args, tok, device, args.seed + f)
            oof[te] = p
            meta["secs"] = round(time.time() - t0, 1); info.append(meta)
            print(f"  [{mode:<6s} fold {f}] trainable={meta['trainable_params']:,} steps={meta['steps']} {meta['sec_per_step']:.2f}s/step "
                  f"best_val={meta['best_val']:.3f} ({meta['secs']}s)", flush=True)
        m = ~np.isnan(oof if kind != "multi" else oof[:, 0])
        if kind == "multi":
            r, _ = evalkit.summarize_multi(y, np.where(m[:, None], oof, np.nan), task.strata, task.names, n_boot=100)
        else:
            r = evalkit.summarize_single(y[m], oof[m], kind, 200)
        results[mode] = {**r, "folds": info}
        oofs[mode] = oof
        print("  ->", mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("spearman", "auroc", "macro_auroc", "macro_auroc_strat_expr_len", "n")}, flush=True)
    if "frozen" in oofs and kind != "multi":
        print("\npaired difference vs frozen probe (same items):")
        for mode in args.modes:
            if mode == "frozen":
                continue
            m = ~np.isnan(oofs[mode]) & ~np.isnan(oofs["frozen"])
            d, lo, hi, p = evalkit.paired_diff(y[m], oofs[mode][m], oofs["frozen"][m], kind)
            print(f"  {mode:<6s} - frozen: {d:+.3f} [{lo:+.3f},{hi:+.3f}]  p(<=0)={p:.3f}")
    if args.out:
        json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
