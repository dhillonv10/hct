"""Option B: continual pretraining (CPT) of an existing pretrained RNA/DNA masked-LM on lncRNA.

    python -m lncfm.cpt --model rinalmo-micro --steps 3000 --out runs/cpt_rinalmo_micro
    python -m lncfm.cpt --model rinalmo-micro --freeze-bottom 6 --lr 3e-5 --replay 0.2 ...

What it does
  * loads the pretrained masked-LM (default: multimolecule/rinalmo-micro, 33M, trained on ncRNA sequence)
  * trains BERT-style MLM on windows of lncRNA transcripts from the *training* chromosomes only
    (human + mouse, one random isoform per sampled gene), plus a `--replay` fraction of protein-coding
    windows to limit catastrophic forgetting
  * reports held-out-chromosome MLM loss BEFORE and DURING training (the loss must go down, and
    benchmark probes must improve vs the frozen model -- otherwise CPT bought nothing)
  * saves a HuggingFace directory usable as the provider  `local:<dir>`  in lncfm.bench

Only models that load without custom remote code are supported here (multimolecule/* ).
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

from . import gencode
from .embed import HF_MODELS, REMOTE_CODE
from .model import pick_device


def _windows(df, rng, n, max_nt):
    idx = rng.integers(0, len(df), n)
    out = []
    for i in idx:
        s = df.seq.iloc[i]
        a = int(rng.integers(0, max(1, len(s) - max_nt + 1)))
        out.append(s[a:a + max_nt])
    return out


class Sampler:
    def __init__(self, replay: float, max_nt: int):
        frames = []
        for sp in ("human", "mouse"):
            df = gencode.representative_per_gene(gencode.build_table(sp, "lnc"))
            if sp == "human":
                df = df[~df.chrom.isin(gencode.TEST_CHROMS | gencode.VAL_CHROMS)]
            frames.append(df[["seq"]])
        import pandas as pd
        self.lnc = pd.concat(frames, ignore_index=True)
        pc = gencode.representative_per_gene(gencode.build_table("human", "pc"))
        self.pc = pc[~pc.chrom.isin(gencode.TEST_CHROMS | gencode.VAL_CHROMS)][["seq"]].reset_index(drop=True)
        val = gencode.representative_per_gene(gencode.build_table("human", "lnc"))
        self.val = list(val[val.chrom.isin(gencode.VAL_CHROMS)].sample(256, random_state=0).seq.str.slice(0, max_nt))
        self.replay, self.max_nt = replay, max_nt

    def batch(self, rng, B):
        n_pc = int(round(B * self.replay))
        return _windows(self.lnc, rng, B - n_pc, self.max_nt) + _windows(self.pc, rng, n_pc, self.max_nt)


def mask_batch(tok, texts, rng, device, frac=0.15):
    enc = tok(texts, return_tensors="pt", padding=True, return_special_tokens_mask=True)
    ids = enc["input_ids"].clone()
    special = enc["special_tokens_mask"].bool() | (enc["attention_mask"] == 0)
    cand = ~special
    m = torch.from_numpy(rng.random(ids.shape) < frac) & cand
    labels = torch.where(m, ids, torch.full_like(ids, -100))
    r = torch.from_numpy(rng.random(ids.shape))
    ids[m & (r < 0.8)] = tok.mask_token_id
    rnd = m & (r >= 0.8) & (r < 0.9)
    if rnd.any():
        pool = torch.tensor([i for i in range(tok.vocab_size) if i not in set(tok.all_special_ids)])
        ids[rnd] = pool[torch.randint(len(pool), (int(rnd.sum()),))]
    return ids.to(device), enc["attention_mask"].to(device), labels.to(device)


@torch.no_grad()
def eval_loss(model, tok, texts, device, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    tot, n = 0.0, 0
    for i in range(0, len(texts), 16):
        ids, am, lab = mask_batch(tok, texts[i:i + 16], rng, device)
        lg = model(input_ids=ids, attention_mask=am).logits
        keep = lab != -100
        tot += F.cross_entropy(lg[keep].float(), lab[keep], reduction="sum").item()
        n += int(keep.sum())
    model.train()
    return tot / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="rinalmo-micro")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-nt", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--freeze-bottom", type=int, default=0, help="freeze embeddings + this many lowest layers")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.model in REMOTE_CODE:
        raise SystemExit(f"{a.model} needs remote code; CPT here supports multimolecule/* models only.")
    import multimolecule  # noqa: F401
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    hub = HF_MODELS[a.model][0]
    device = pick_device(a.device)
    tok = AutoTokenizer.from_pretrained(hub)
    model = AutoModelForMaskedLM.from_pretrained(hub).to(device).train()
    if a.freeze_bottom:
        enc = getattr(model, [n for n, _ in model.named_children()][0])
        for p in enc.embeddings.parameters():
            p.requires_grad = False
        for layer in enc.encoder.layer[:a.freeze_bottom]:
            for p in layer.parameters():
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"device={device} trainable={sum(p.numel() for p in params)/1e6:.1f}M / {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    S = Sampler(a.replay, a.max_nt)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    v0 = eval_loss(model, tok, S.val, device)
    print(f"held-out lncRNA MLM loss BEFORE cpt: {v0:.4f}")
    log.write(json.dumps({"step": 0, "val_mlm": v0}) + "\n")
    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        ids, am, lab = mask_batch(tok, S.batch(rng, a.batch), rng, device)
        lg = model(input_ids=ids, attention_mask=am).logits
        keep = lab != -100
        loss = F.cross_entropy(lg[keep].float(), lab[keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "mlm": float(loss), "elapsed": time.time() - t0}
            if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
                rec["val_mlm"] = eval_loss(model, tok, S.val, device)
                print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    model.save_pretrained(out / "hf"); tok.save_pretrained(out / "hf")
    print("saved", out / "hf", " -> evaluate with provider  local:" + str(out / "hf"))


if __name__ == "__main__":
    main()
