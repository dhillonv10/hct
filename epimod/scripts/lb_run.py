"""Train one EpiSet variant and score it on the LongevityBench DNAm eval items.

    EPIMOD_DATA=... python scripts/lb_run.py --name extremes_dense --tasks extremes --steps 3000

Protocol: training data = ComputAge samples from series that do NOT appear in any LongevityBench DNAm
eval prompt; early stopping on 15% held-out *series* of that data; candidate CpG pools for the extreme /
pairwise lists are estimated from CpG identities (no values, no labels) in the ODD-numbered eval prompts, and the
headline numbers are on the EVEN-numbered prompts. Every LongevityBench evaluation is appended to
``lb_eval_log.jsonl`` so the number of variants scored is on record.
"""
import argparse, json, os, pickle, sys, time
if "mps" in sys.argv or "auto" in sys.argv:        # must be set before torch is imported
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from epimod.lb.episet import SetConfig
from epimod.lb.evaluate import evaluate
from epimod.lb.store import lb_meta, load_lb_store, lb_series
from epimod.lb.tasks import build_vocab, load_parsed, make_pools
from epimod.lb.train import ALL_TASKS, LBTrainConfig, Trainer, task_sets
from epimod.train.loop import split_studies

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--out", default="runs/lb")
ap.add_argument("--tasks", default=",".join(ALL_TASKS)); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--moe", type=int, default=0); ap.add_argument("--topk", type=int, default=2)
ap.add_argument("--d", type=int, default=96); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--age-balance", type=float, default=0.0); ap.add_argument("--tail-shift", type=float, default=0.04)
ap.add_argument("--bs-fixed", type=int, default=32, help="batch size of the (attention-heavy) fixed-panel task")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--full-pools", action="store_true",
                help="derive pools from ALL eval prompts (secondary check; not the headline protocol)")
ap.add_argument("--model", default="set", choices=["set", "hier"], help="set = flat set transformer (EpiSet); hier = EpiTrajectory-L")
ap.add_argument("--local-layers", type=int, default=2); ap.add_argument("--global-layers", type=int, default=3)
ap.add_argument("--w-mask", type=float, default=1.0)
ap.add_argument("--max-minutes", type=float, default=0.0, help="stop cleanly (checkpoint written) after this many minutes; rerun to resume")
ap.add_argument("--moe-level", default="token", choices=["token", "sample"])
ap.add_argument("--linear", action="store_true", help="add the explicit per-CpG linear route to the set transformer")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
ap.add_argument("--unlabeled", action="store_true", help="also train the masked-reconstruction task on the label-free GEO series")
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--d-emb", type=int, default=64); ap.add_argument("--bs-mult", type=float, default=1.0)
ap.add_argument("--no-lb-eval", action="store_true")
ap.add_argument("--eval-only", action="store_true", help="load runs/lb/<name>/model.pt and only run the LongevityBench evaluation")
a = ap.parse_args()

out = Path(a.out) / a.name; out.mkdir(parents=True, exist_ok=True)
parsed = load_parsed(); excl = lb_series(parsed)
meta_all = lb_meta(excl); tr_m, va_m = split_studies(meta_all, 0.15, 0)
tr_series = set(meta_all[tr_m].DatasetID)
if a.unlabeled:
    full = lb_meta(excl, unlabeled=True); tr_series |= set(full[full.Age.isna()].DatasetID)
tr = load_lb_store(exclude_series=excl, include_series=tr_series, unlabeled=a.unlabeled)     # train and validation loaded separately:
va = load_lb_store(exclude_series=excl, include_series=set(meta_all[va_m].DatasetID))     # peak memory = steady-state memory
uni = tr.universe
odd = np.arange(500) % 2 == 1
pools = make_pools(parsed, uni) if a.full_pools else make_pools(parsed, uni, ext_mask=odd, pair_mask=odd)
vocab = build_vocab(tr, parsed, pools); sets = task_sets(parsed, uni)
print(f"train {tr.n} samples | val {va.n} samples ({va.meta.DatasetID.nunique()} series) | vocab {len(vocab)} | pools low/high/pair {len(pools.low)}/{len(pools.high)}/{len(pools.pair)}", flush=True)
bs = {k: max(1, int(round(v * a.bs_mult))) for k, v in {"extremes": 32, "fixed": a.bs_fixed, "clock": 32, "pair": 24, "mask": 16}.items()}
cfg = LBTrainConfig(tasks=tuple(a.tasks.split(",")), bs=bs, steps=a.steps, lr=a.lr, age_balance=a.age_balance, tail_shift=a.tail_shift, seed=a.seed, threads=a.threads, device=a.device, w_mask=a.w_mask, max_minutes=a.max_minutes)
if a.model == "hier":
    from epimod.lb.epitraj import HierConfig
    mcfg = HierConfig(d_model=a.d, d_emb=a.d_emb, n_heads=a.heads, local_layers=a.local_layers, global_layers=a.global_layers, moe_experts=a.moe, moe_top_k=a.topk, moe_level=a.moe_level)
else:
    mcfg = SetConfig(d_model=a.d, d_emb=a.d_emb, n_heads=a.heads, n_layers=a.layers, moe_experts=a.moe, moe_top_k=a.topk, moe_level=a.moe_level, linear_route=a.linear)
if a.eval_only:
    ck = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    assert list(vocab.cpg_ids) == ck["vocab_ids"], "vocabulary differs from the trained model's (pools/flags changed?)"
    from epimod.lb.episet import EpiSet
    if ck["args"].get("model") == "hier":
        from epimod.data.annotation import load_probe_annotation
        from epimod.lb.epitraj import EpiHier, HierConfig, block_assignment
        hc = HierConfig(**ck["cfg"]); model = EpiHier(hc, block_assignment(vocab.cpg_ids, load_probe_annotation(), hc.block_size), vocab.mean, vocab.std)
    else:
        model = EpiSet(SetConfig(**ck["cfg"]), vocab.mean, vocab.std)
    model.load_state_dict(ck["state"]); model.eval()
    cfg = LBTrainConfig(tasks=tuple(ck["args"]["tasks"].split(",")))
else:
    tr_ = Trainer(tr, va, vocab, sets, pools, mcfg, cfg, log=lambda s: print(s, flush=True), ckpt_path=out / "ckpt.pt")
    t0 = time.time(); res = tr_.fit(); res["train_sec"] = round(time.time() - t0)
    if "stopped_at" in res:
        torch.save({"state": tr_.model.state_dict(), "cfg": asdict(tr_.model.cfg), "vocab_ids": list(vocab.cpg_ids), "args": vars(a)}, out / "model_partial.pt")
        print(f"segment finished at step {res['stopped_at']} (EMA weights saved to model_partial.pt); rerun the same command to resume", flush=True); raise SystemExit(0)
    model = tr_.model
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | best val score {res['best_val_score']:.4f}", flush=True)
    torch.save({"state": model.state_dict(), "cfg": asdict(model.cfg), "vocab_ids": list(vocab.cpg_ids), "args": vars(a)}, out / "model.pt")
    (out / "train_hist.json").write_text(json.dumps(res, indent=1))
if not a.no_lb_eval:
    lb = evaluate(model, vocab, parsed, uni, tasks=tuple(t for t in ("LB-0010", "LB-0002", "LB-0006", "LB-0190", "LB-0195")
                  if {"LB-0010": "extremes", "LB-0002": "extremes", "LB-0006": "pair", "LB-0190": "fixed", "LB-0195": "clock"}[t] in cfg.tasks))
    (out / "lb_eval.json").write_text(json.dumps(lb, indent=1))
    with open(Path(a.out) / "lb_eval_log.jsonl", "a") as f:
        f.write(json.dumps({"name": a.name, "time": time.strftime("%F %T"), "args": vars(a), "lb": lb}) + "\n")
    print(json.dumps(lb, indent=1), flush=True)
