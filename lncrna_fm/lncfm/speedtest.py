"""Measure real training throughput on THIS machine and project wall-clock time.

    python -m lncfm.speedtest --preset small --batch 32 --max-len 1024
    python -m lncfm.speedtest --preset base  --batch 16 --max-len 1024 --dtype bf16

Run it on the M4 before committing to a long run. It uses the real model, real losses and
random (synthetic) nucleotide batches ONLY for timing -- no results are produced from them.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .model import build, pick_device
from .pretrain import compute_loss


def train_flops_per_nt(model, L=1024, B=2):
    """Counted (fwd+bwd) FLOPs per nucleotide, using torch's FlopCounterMode on a CPU copy."""
    from torch.utils.flop_counter import FlopCounterMode
    import copy
    m = copy.deepcopy(model).cpu()
    x = torch.randint(2, 6, (B, L))
    with FlopCounterMode(display=False) as fc:
        out = m(x)
        (out["nt_logits"].sum() + out["junction_logits"].sum() + out["pooled"].sum()).backward()
    return fc.get_total_flops() / (B * L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch", type=int, default=32, help="genes per step (x2 views)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--target-nt", type=float, default=1e9, help="nucleotides you want to train on")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    if dev.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model = build(a.preset, max_len=max(a.max_len, 1024)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B2 = 2 * a.batch
    X = torch.randint(2, 6, (B2, a.max_len), device=dev)
    Y = torch.where(torch.rand(B2, a.max_len, device=dev) < 0.15, X, torch.full_like(X, -100))
    J = (torch.rand(B2, a.max_len, device=dev) < 0.004).float()

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    times = []
    for i in range(a.steps):
        t = time.time()
        ctx = torch.autocast(dev.type, dtype=torch.bfloat16) if a.dtype == "bf16" else torch.autocast(dev.type, enabled=False)
        with ctx:
            loss, _ = compute_loss(model, X, Y, J, 0.3, 0.3)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sync()
        times.append(time.time() - t)
    t_step = float(np.median(times[3:]))      # skip warm-up
    fl = train_flops_per_nt(model, a.max_len)
    nt_s = B2 * a.max_len / t_step
    print(f"device={dev} preset={a.preset} params={model.n_params()/1e6:.1f}M batch={B2}x{a.max_len}nt dtype={a.dtype}")
    print(f"median step {t_step:.3f}s  ->  {nt_s:,.0f} nt/s")
    print(f"counted train FLOPs/nt = {fl:,.0f}  ->  achieved {fl*nt_s/1e12:.2f} TFLOP/s on this device")
    hrs = a.target_nt / nt_s / 3600
    print(f"projected time for {a.target_nt:.2e} nt: {hrs:.1f} h ({hrs/24:.1f} days)")
    if dev.type == "mps":
        print("note: if this is far below expectation, check for CPU fallbacks (PYTORCH_ENABLE_MPS_FALLBACK warnings above).")


if __name__ == "__main__":
    main()
