"""Per-item LongevityBench predictions of a trained EpiSet (no training data needed: the vocabulary and CpG statistics
are restored from the checkpoint).   python scripts/lb_predict.py runs/lb/<name>/model.pt out.npz"""
import sys
import numpy as np, pandas as pd, torch

from epimod.age import t_to_age
from epimod.lb.episet import EpiSet, SetConfig
from epimod.lb.tasks import Vocab, eval_clock, eval_fixed, eval_pair, eval_single, load_parsed
from epimod.paths import subdir

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
parsed = load_parsed()
from epimod.lb.lite import load_lite_universe
uni = load_lite_universe()          # model vocabularies are subsets of the compact universe (all benchmark-prompt CpGs)
ids = pd.Index(ck["vocab_ids"]); u2m = np.full(len(uni), -1, np.int64); pos = uni.get_indexer(ids); u2m[pos[pos >= 0]] = np.flatnonzero(pos >= 0)
if ck["args"].get("model") == "hier":
    from epimod.data.annotation import load_probe_annotation
    from epimod.lb.epitraj import EpiHier, HierConfig, block_assignment
    hc = HierConfig(**ck["cfg"]); model = EpiHier(hc, block_assignment(ids, load_probe_annotation(), hc.block_size))
else:
    model = EpiSet(SetConfig(**ck["cfg"]))
model.load_state_dict(ck["state"]); model.eval()
vocab = Vocab(u2m, ids, model.cpg_mean[:-1].numpy(), model.cpg_std[:-1].numpy())
tasks = set(ck["args"]["tasks"].split(","))
out = {}
if "extremes" in tasks:
    for lb in ("LB-0010", "LB-0002"):
        b = eval_single(parsed[lb], vocab, uni)
        mu, sig, dec = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])
        out.update({f"{lb}/mu": mu, f"{lb}/sigma": sig, f"{lb}/head": dec, f"{lb}/pred": t_to_age(mu)})
if "pair" in tasks:
    a, b = eval_pair(parsed["LB-0006"], vocab, uni)
    ma = model.predict_all(a["cpg"], a["beta"], a["role"], a["sex"])[0]; mb = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])[0]
    out["LB-0006/logit"] = ma - mb
if "fixed" in tasks:
    b = eval_fixed(parsed["LB-0190"], vocab, uni)
    out["LB-0190/pred"] = t_to_age(model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])[0])
if "clock" in tasks:
    b = eval_clock(parsed["LB-0195"], vocab, uni)
    out["LB-0195/pred"] = t_to_age(model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"], b["clock"])[0])
np.savez(sys.argv[2], **out)
import json, time
from pathlib import Path
with open(Path(sys.argv[2]).parent / "predict_log.jsonl", "a") as f:        # every benchmark scoring is on record
    f.write(json.dumps({"time": time.strftime("%F %T"), "model": sys.argv[1], "out": sys.argv[2], "keys": sorted(out)}) + "\n")
print({k: v.shape for k, v in out.items()})
