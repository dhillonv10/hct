"""Parsers for the LongevityBench DNA-methylation prompts (benchmark + extra configs)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HF_BASE = "https://huggingface.co/datasets/insilicomedicine/longebench/resolve/main"
CG = re.compile(r"(cg\d{8}) \(chr")
BETA = re.compile(r"Beta: ([0-9.]+)\.")
PAIR = re.compile(r"A: ([0-9.]+), B: ([0-9.]+),")
SEX = re.compile(r"(?:donor is|individual is) A (male|female)")
SEX_OPT = re.compile(r"(?:^|; )?([AB]): A (male|female)")
RAW = re.compile(r"(cg\d{8}): ([0-9.]+)")
PCT = re.compile(r"(cg\d{8}): ([0-9.]+)%")
ID = re.compile(r"(GSM\d+)_(GSE\d+)")
AGE_NOTE = re.compile(r"(\d+(?:\.\d+)?) years")


def user_text(messages) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def answer_text(messages) -> str:
    return messages[-1]["content"].strip()


def _meta(metadata) -> dict:
    if metadata is None or (isinstance(metadata, float) and np.isnan(metadata)):
        return {}
    try:
        return json.loads(metadata)
    except (TypeError, ValueError):
        return {}


@dataclass
class SetTask:
    """One task: ``ids[i]`` / ``beta[i]`` are the CpGs shown for sample ``i``."""

    name: str
    ids: list[np.ndarray] = field(default_factory=list)       # probe ids per sample
    beta: list[np.ndarray] = field(default_factory=list)      # beta values per sample (0..1)
    beta_b: list[np.ndarray] = field(default_factory=list)    # pairwise only: donor B values
    is_high: list[np.ndarray] = field(default_factory=list)   # single-profile: from the "highest methylated" list
    sex: list[str] = field(default_factory=list)              # 'M' | 'F' | '' (pairwise: donor A)
    sex_b: list[str] = field(default_factory=list)
    age: list[float] = field(default_factory=list)            # true age (pairwise: donor A)
    age_b: list[float] = field(default_factory=list)
    label: list[object] = field(default_factory=list)         # decade string / 'A'|'B'
    gsm: list[str] = field(default_factory=list)              # donor A
    gse: list[str] = field(default_factory=list)
    gsm_b: list[str] = field(default_factory=list)
    gse_b: list[str] = field(default_factory=list)
    clock: list[str] = field(default_factory=list)


def parse_single(df: pd.DataFrame, lb_id: str) -> SetTask:
    """LB-0010 (regression) and LB-0002 (age-decade multiclass): 100 lowest + 100 highest CpGs."""
    t = SetTask(lb_id)
    for _, r in df[df.lb_id == lb_id].iterrows():
        u = user_text(r["messages"])
        low, high = u.split("Highest methylated")
        ids = CG.findall(low) + CG.findall(high)
        beta = [float(x) for x in BETA.findall(low) + BETA.findall(high)]
        if len(ids) != len(beta) or len(ids) < 50:
            continue
        m = SEX.search(u)
        meta = _meta(r["metadata"]).get("follow_up", "")
        age = AGE_NOTE.search(meta)
        ident = ID.search(meta)
        t.ids.append(np.array(ids)); t.beta.append(np.array(beta, dtype=np.float32))
        t.is_high.append(np.array([0] * len(CG.findall(low)) + [1] * len(CG.findall(high)), dtype=np.int8))
        t.sex.append({"male": "M", "female": "F"}.get(m.group(1), "") if m else "")
        t.age.append(float(age.group(1)) if age else np.nan)
        gm = re.search(r"age group (\d+-\d+|100\+)", meta)
        t.label.append(gm.group(1) if gm else (answer_text(r["messages"]) if lb_id == "LB-0010" else ""))
        t.gsm.append(ident.group(1) if ident else ""); t.gse.append(ident.group(2) if ident else "")
    return t


def parse_pairwise(df: pd.DataFrame, lb_id: str = "LB-0006") -> SetTask:
    t = SetTask(lb_id)
    for _, r in df[df.lb_id == lb_id].iterrows():
        u = user_text(r["messages"])
        ids = CG.findall(u)
        ab = PAIR.findall(u)
        if len(ids) != len(ab) or len(ids) < 20:
            continue
        meta = _meta(r["metadata"]).get("follow_up", "")
        a_part, _, b_part = meta.partition("Patient-B")
        ia, ib = ID.search(a_part), ID.search(b_part)
        ages = AGE_NOTE.findall(meta)
        opts = u.split("Options:")[1].split("DNA methylation data")[0]
        sx = dict(SEX_OPT.findall(opts))
        t.ids.append(np.array(ids))
        t.beta.append(np.array([float(a) for a, _ in ab], dtype=np.float32))
        t.beta_b.append(np.array([float(b) for _, b in ab], dtype=np.float32))
        t.sex.append({"male": "M", "female": "F"}.get(sx.get("A", ""), "")); t.sex_b.append({"male": "M", "female": "F"}.get(sx.get("B", ""), ""))
        t.age.append(float(ages[0]) if len(ages) > 0 else np.nan); t.age_b.append(float(ages[1]) if len(ages) > 1 else np.nan)
        t.label.append(answer_text(r["messages"]))
        t.gsm.append(ia.group(1) if ia else ""); t.gse.append(ia.group(2) if ia else "")
        t.gsm_b.append(ib.group(1) if ib else ""); t.gse_b.append(ib.group(2) if ib else "")
    return t


def parse_raw(df: pd.DataFrame, lb_id: str = "LB-0190") -> SetTask:
    """LB-0190: fixed 353 consensus CpGs, beta values, true age as the answer."""
    t = SetTask(lb_id)
    for _, r in df[df.lb_id == lb_id].iterrows():
        pairs = RAW.findall(user_text(r["messages"]))
        if len(pairs) < 100:
            continue
        t.ids.append(np.array([p[0] for p in pairs])); t.beta.append(np.array([float(p[1]) for p in pairs], dtype=np.float32))
        t.age.append(float(answer_text(r["messages"])))
    return t


def parse_clock(df: pd.DataFrame, lb_id: str = "LB-0195") -> SetTask:
    """LB-0195: CpGs of one published clock (percent), chronological age as the answer."""
    t = SetTask(lb_id)
    for _, r in df[df.lb_id == lb_id].iterrows():
        pairs = PCT.findall(user_text(r["messages"]))
        if len(pairs) < 5:
            continue
        meta = _meta(r["metadata"])
        t.ids.append(np.array([p[0] for p in pairs])); t.beta.append(np.array([float(p[1]) / 100.0 for p in pairs], dtype=np.float32))
        t.age.append(float(answer_text(r["messages"])))
        t.gsm.append(meta.get("gsm", "")); t.gse.append(meta.get("gse", "")); t.clock.append(meta.get("clock", ""))
    return t
