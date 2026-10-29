"""Guide/target datasets for the guide-efficacy family.

Cas13d (TIGER, Wessels/Stirn et al. 2023, Nat Biotechnol): HEK293T survival screen `off-target` (PM + mismatch/indel guides, 16 genes),
HAP1 `hap-titration` (PM + single-mismatch, 1,082 genes), `flow-cytometry` (CD46/CD55/TFRC). The authors ship them as pandas-1.x pickles
inside .bz2; `export_tiger` loads them through an exact allowlist (safepickle) and writes plain CSV, so nothing else here unpickles.

Conventions
  target_seq  : 23 nt of the transcript (sense).      guide53 : its reverse complement = the guide as written 5'->3' (the strings other
  tools use). TIGER's own `guide_seq` column is the plain complement in target orientation, which we do not use.
  y           : mean of the replicate log2 fold-changes (more negative = stronger knockdown).  Sign is flipped in `act` (higher = more active).
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

LFC = ["lfc_r1", "lfc_r2", "lfc_r3"]
# HEK293T genes held out of every development decision (every 4th gene in alphabetical order). The untuned baseline ladder (default
# settings, no choices made from it) had already been run on all 16 genes when this rule was fixed; from here on, design choices are scored on
# the other 12 genes, and the locked genes and the external screens are evaluated once, at the end.
LOCKED_GENES = ("CDC5L", "POLA1", "SF3B1", "WDR75")
_COMP = str.maketrans("ACGTUN", "TGCAAN")


def rc(s: str) -> str:
    return s.upper().translate(_COMP)[::-1]


def export_tiger(src_dir: str, dst_dir: str):
    """TIGER data-processed/*.bz2 (pickles) -> dst_dir/*.csv.gz, through the strict loader."""
    from .safepickle import load_strict
    os.makedirs(dst_dir, exist_ok=True)
    for f in ["off-target", "off-target-nt", "hap-titration", "hap-titration-nt", "flow-cytometry"]:
        d = load_strict(os.path.join(src_dir, f + ".bz2"))
        d = d.reset_index() if not isinstance(d.index, pd.RangeIndex) else d
        d.to_csv(os.path.join(dst_dir, f + ".csv.gz"), index=False)
        print(f, d.shape)


def active_threshold(nt: pd.DataFrame, q: float = 0.01) -> float:
    """TIGER's definition: guides below the q-quantile of a normal fitted to the non-targeting mean-LFC distribution are 'active'."""
    m = nt[LFC].mean(axis=1)
    return float(norm.ppf(q, loc=m.mean(), scale=m.std()))


def load_tiger(data_dir: str, name: str = "off-target", pm_only: bool = True) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(data_dir, name + ".csv.gz"))
    if pm_only:
        d = d[d.guide_type == "PM"]
    d = d.copy()
    for c in ("5p_context", "3p_context", "target_seq"):
        d[c] = d[c].fillna("").astype(str)             # empty flanks round-trip through CSV as NaN
    d["y"] = d[LFC].mean(axis=1)
    d = d[d.y.notna()].reset_index(drop=True)
    d["act"] = -d["y"]
    d["guide53"] = d.target_seq.map(rc)
    nt_path = os.path.join(data_dir, name + "-nt.csv.gz")
    thr = active_threshold(pd.read_csv(nt_path)) if os.path.exists(nt_path) else -0.5
    d["active"] = (d.y < thr).astype(int)
    pos = d.guide_id.str.extract(r":(\d+)-(\d+)")
    d["start"] = pd.to_numeric(pos[0], errors="coerce")
    d["block"] = d.gene.astype(str) + ":" + (d.start // 50).fillna(-1).astype(int).astype(str)    # 50-nt target region, the bootstrap/CV unit
    d.attrs["active_threshold"] = thr
    return d


def gene_norm_params(d: pd.DataFrame, q_loc=50, q_neg=10, q_pos=90) -> pd.DataFrame:
    """TIGER's 'FrequentistQuantile' per-gene location/scale (median and q90-q10 of the gene's own PM LFCs)."""
    g = d[d.guide_type == "PM"].groupby("gene").y
    p = pd.DataFrame({"loc": g.quantile(q_loc / 100), "scale": g.quantile(q_pos / 100) - g.quantile(q_neg / 100)})
    return p


def read_oligoformer(path: str) -> pd.DataFrame:
    """OligoFormer siRNA tables (Hu / Mix / Taka): siRNA (19 nt, RNA), mRNA (59-nt window around the site; X = padding), label in [0,1]."""
    d = pd.read_csv(path)
    d["sirna"] = d.siRNA.str.upper().str.replace("T", "U")
    d["y"] = d["label"].astype(float)
    return d
