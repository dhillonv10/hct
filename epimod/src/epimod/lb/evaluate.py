"""Score an EpiSet model on the LongevityBench DNAm eval items (numbers comparable to Table S6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .episet import EpiSet
from .tasks import (Vocab, balanced_accuracy, decade_label_to_id, decade_posterior, eval_clock, eval_fixed, eval_pair, eval_single, mae)


def _split(n: int) -> dict[str, np.ndarray]:
    odd = np.arange(n) % 2 == 1
    return {"all": np.ones(n, bool), "even": ~odd, "odd": odd}


def evaluate(model: EpiSet, vocab: Vocab, parsed: dict, universe: pd.Index, tasks=("LB-0010", "LB-0002", "LB-0006", "LB-0190", "LB-0195")) -> dict:
    out: dict = {}
    if "LB-0010" in tasks or "LB-0002" in tasks:
        for lb in ("LB-0010", "LB-0002"):
            if lb not in tasks:
                continue
            t = parsed[lb]; b = eval_single(t, vocab, universe)
            mu, sig, dec = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])
            age = np.array(t.age)
            if lb == "LB-0010":
                from ..age import t_to_age
                pred = t_to_age(mu)
                out[lb] = {k: mae(pred[m], age[m]) for k, m in _split(len(age)).items()}
            else:
                true = np.array([decade_label_to_id(l) for l in t.label])
                post = decade_posterior(mu, sig).argmax(1); head = dec.argmax(1)
                out[lb] = {f"{src}_{k}": balanced_accuracy(p[m], true[m]) for src, p in (("posterior", post), ("head", head)) for k, m in _split(len(true)).items()}
    if "LB-0006" in tasks:
        t = parsed["LB-0006"]; a, b = eval_pair(t, vocab, universe)
        ma, _, _ = model.predict_all(a["cpg"], a["beta"], a["role"], a["sex"]); mb, _, _ = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])
        correct = (ma > mb) == (np.array(t.label) == "A")
        out["LB-0006"] = {k: float(correct[m].mean()) for k, m in _split(len(correct)).items()}
    if "LB-0190" in tasks:
        t = parsed["LB-0190"]; b = eval_fixed(t, vocab, universe)
        from ..age import t_to_age
        mu, _, _ = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"])
        out["LB-0190"] = {"all": mae(t_to_age(mu), np.array(t.age))}
    if "LB-0195" in tasks:
        t = parsed["LB-0195"]; b = eval_clock(t, vocab, universe)
        from ..age import t_to_age
        mu, _, _ = model.predict_all(b["cpg"], b["beta"], b["role"], b["sex"], b["clock"])
        age = np.array(t.age); pred = t_to_age(mu); clk = np.array(t.clock)
        out["LB-0195"] = {"all": mae(pred, age), **{f"clock:{c}": mae(pred[clk == c], age[clk == c]) for c in sorted(set(clk))}}
    return out
