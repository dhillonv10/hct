"""Pre-flight checks before a long run.  Takes minutes; its job is to find in minutes what would otherwise cost days.

  1. the device computes (a matmul is finite);
  2. forward + backward parity: for every task, the same weights and the same real batch give the same loss and the same
     gradients on the accelerator and on the CPU (catches unsupported / wrongly implemented ops);
  3. overfitting one fixed batch: the loss must fall below 60% of its start;
  4. a real mini-run on real data: validation must not diverge, an injected NaN must be skipped, the run must resume from its
     checkpoint, the exported model must reproduce the validation score on the CPU, and the CPU cross-check must agree;
  5. speed and memory, so the length of the run can be planned."""
from __future__ import annotations

import copy
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .longrun import LongConfig, LongTrainer, load_exported


def _swap(T, model, dev, log_tau):
    old = (T.model, T.dev, T.log_tau)
    T.model, T.dev, T.log_tau = model, dev, log_tau
    return old


def _one_pass(T, kind: str, seed: int):
    T.rng = np.random.RandomState(seed)
    T.model.eval(); T.model.zero_grad(set_to_none=True)
    loss, _ = T.step_loss(kind)
    loss.backward()
    return float(loss.detach()), {n: p.grad.detach().cpu().double().clone() for n, p in T.model.named_parameters() if p.grad is not None}


def parity(T, dev: torch.device, kinds) -> list[tuple[str, bool, str]]:
    out = []
    cpu = torch.device("cpu")
    for kind in kinds:
        try:
            l_dev, g_dev = _one_pass(T, kind, 1)
            m_cpu = copy.deepcopy(T.model).to(cpu); tau_cpu = torch.nn.Parameter(T.log_tau.detach().cpu().clone())
            old = _swap(T, m_cpu, cpu, tau_cpu)
            try:
                l_cpu, g_cpu = _one_pass(T, kind, 1)
            finally:
                T.model, T.dev, T.log_tau = old
            rel = abs(l_dev - l_cpu) / max(abs(l_cpu), 1e-6)
            names = [n for n in g_cpu if n in g_dev]
            vc = torch.cat([g_cpu[n].flatten() for n in names]); vd = torch.cat([g_dev[n].flatten() for n in names])
            gcos = float((vc * vd).sum() / (float(vc.norm()) * float(vd.norm()) + 1e-12)); total = float(vc.norm())
            bad = [] if gcos > 0.999 else [("ALL (global gradient)", f"cos {gcos:.4f}")]
            for n, gc in g_cpu.items():                  # parameters that carry signal (>= 2% of the gradient norm) must agree individually;
                nc = float(gc.norm())                    # near-zero gradients are float noise and are only covered by the global check
                if nc < 0.02 * total:
                    continue
                gd = g_dev.get(n)
                if gd is None:
                    bad.append((n, "no gradient on device")); continue
                cos = float((gc * gd).sum() / (nc * float(gd.norm()) + 1e-12)); ratio = float(gd.norm()) / nc
                if cos < 0.99 or not 0.9 < ratio < 1.1:
                    bad.append((n, f"cos {cos:.3f}, norm ratio {ratio:.2f}"))
            ok = rel < 5e-3 and not bad
            out.append((f"parity[{kind}]", ok, f"loss {l_dev:.5f} vs cpu {l_cpu:.5f} (rel {rel:.2e}); global gradient cos {gcos:.5f}; {len(g_cpu)} grads" + (f"; worst: {bad[:3]}" if bad else "")))
        except Exception as e:
            out.append((f"parity[{kind}]", False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
    return out


def overfit(T, lr: float, steps: int = 120):
    """Train on ONE fixed batch per task (identical every step): the multi-task loss must fall sharply if gradients, optimizer and
    device all work.  Independent of generalisation, so it is reliable within a couple of minutes."""
    from torch import nn
    model = T.model; model.eval()                                   # dropout off so the loss can fall smoothly
    params = list(model.parameters()) + [T.log_tau]
    opt = torch.optim.AdamW(params, lr=lr * 2, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        T.rng = np.random.RandomState(123)
        loss = sum(T.step_loss(k)[0] for k in T.cfg.tasks)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses[:3])), float(np.mean(losses[-3:])), losses


def run(make_trainer, work: Path, lr: float, dev: torch.device, steps: int = 240, log=print) -> bool:
    """``make_trainer(run_dir, long_cfg)`` -> LongTrainer.  Returns True when every check passes."""
    results: list[tuple[str, bool, str]] = []
    def rec(name, ok, detail=""):
        results.append((name, ok, detail)); log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    # 1 ---------------------------------------------------------------- device
    try:
        x = torch.randn(512, 512, device=dev); v = float((x @ x.T).sum().cpu())
        rec("device", math.isfinite(v), f"{dev}, torch {torch.__version__}")
    except Exception as e:
        rec("device", False, f"{type(e).__name__}: {e}"); return False
    # 2 ---------------------------------------------------------------- parity
    T = make_trainer(work / "parity", LongConfig(lr=lr))
    for name, ok, detail in parity(T, dev, T.cfg.tasks):
        rec(name, ok, detail)
    # 2b --------------------------------------------------------------- can it learn at all on this device?
    try:
        t0 = time.time(); a0, a1, ls = overfit(T, lr); ok = math.isfinite(a1) and a1 < 0.6 * a0
        rec("overfit one batch", ok, f"multi-task loss {a0:.3f} -> {a1:.3f} in {len(ls)} steps ({a1 / a0:.0%} of the start; need < 60%); {len(ls) / (time.time() - t0):.2f} steps/s")
    except Exception as e:
        rec("overfit one batch", False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    del T; import gc; gc.collect()
    # 3 ---------------------------------------------------------------- mini run
    try:
        ev = max(10, steps // 4)
        lc = LongConfig(eval_every=ev, warmup=max(10, steps // 8), lr=lr, max_steps=steps, patience=1000, cpu_check_every=1)
        hit = []
        def hook(step, loss):                                     # one NaN loss in the middle of the run
            if step == steps // 2 + 1 and not hit:
                hit.append(1); return loss * float("nan")
            return loss
        t0 = time.time(); T = make_trainer(work / "mini", lc); logs = []; T.log = logs.append
        res = T.fit_long(fault_hook=hook); dt = time.time() - t0
        h = res["history"]; scores = [r["val_score"] for r in h]
        rec("mini-run finite", len(h) >= 3 and all(s is not None and math.isfinite(s) for s in scores), f"val scores {scores}")
        gain = (scores[0] - scores[-1]) / scores[0] if scores and scores[0] else float("nan")
        rec("mini-run not diverging", scores[-1] <= scores[0] * 1.05, f"validation score changed {gain:+.1%} over {res['step']} steps (informational: the big drop usually comes later; must not get worse by > 5%)")
        rec("NaN guard", res["bad_steps"] >= 1 and res["step"] == steps, f"{res['bad_steps']} bad step(s) skipped, run reached step {res['step']}")
        cc = [l for l in logs if "[cpu-check]" in l]
        rel = float(cc[-1].split("rel diff ")[1].rstrip("%)")) / 100 if cc else float("nan")
        if dev.type == "cpu":
            rec("accelerator vs CPU validation", True, "skipped (the device is the CPU)")
        else:
            rec("accelerator vs CPU validation", cc != [] and rel < 0.03, cc[-1] if cc else "no cpu-check line")
        # resume: a new trainer continues from the checkpoint
        T2 = make_trainer(work / "mini", LongConfig(eval_every=ev, warmup=lc.warmup, lr=lr, max_steps=steps + ev, patience=1000, cpu_check_every=0)); l2 = []; T2.log = l2.append
        r2 = T2.fit_long()
        rec("resume", any("[resume]" in l for l in l2) and r2["step"] == steps + ev, f"continued {steps} -> {r2['step']}")
        # exported model reproduces the score on the CPU
        m = load_exported(work / "mini" / "model.pt"); best = r2["best_val_score"]
        sc_cpu = T2.val_score(T2.validate(m)); rel = abs(sc_cpu - best) / max(best, 1e-9)
        rec("export round-trip", rel < 0.03, f"best val {best:.4f}, exported model on CPU {sc_cpu:.4f} (rel {rel:.2%})")
        sps = steps / dt
        mem = ""
        if dev.type == "mps":
            try:
                mem = f", GPU memory {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB"
            except Exception:
                pass
        rec("speed", True, f"{sps:.2f} steps/s incl. validation -> ~{1e4 / sps / 3600:.1f} h per 10,000 steps{mem}")
    except Exception as e:
        rec("mini-run", False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
    ok = all(r[1] for r in results)
    log("PREFLIGHT " + ("PASSED" if ok else "FAILED: " + ", ".join(n for n, k, _ in results if not k)))
    return ok
