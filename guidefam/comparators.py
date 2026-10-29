"""Scores of published Cas13d guide tools on our guides, from the prediction files shipped in the TIGER repo
(`predictions (other models)/`): DeepCas13 (Wei 2021), a Konermann-rules 30-mer model, and the Wessels 2020 random forest.

Each function adds one column, oriented so that higher = more active, and returns the dataframe. Guides a tool did not score stay NaN.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .metrics import spearman


def _per_gene(P: str, tool: str, sub: str) -> pd.DataFrame:
    fr = [pd.read_csv(f).assign(gene=os.path.basename(f)[:-4]) for f in sorted(glob.glob(os.path.join(P, tool, sub, "*.csv")))]
    return pd.concat(fr, ignore_index=True)


def add_deepcas13(d: pd.DataFrame, P: str, sub: str = "off-target") -> pd.DataFrame:
    t = _per_gene(P, "DeepCas13", sub)
    t["seq"] = t["sgRNA Sequence"].str.upper().str.replace("U", "T")
    t = t.drop_duplicates(["gene", "seq"]).set_index(["gene", "seq"])["Score"].astype(float)
    d["deepcas13"] = [t.get((g, s), np.nan) for g, s in zip(d.gene, d.guide53)]
    return d


def add_konermann(d: pd.DataFrame, P: str, sub: str = "off-target") -> pd.DataFrame:
    """Konermann-rules predictions are for 30-mers; our 23-mer is the 5' prefix of exactly the 30-mer scoring the same site."""
    t = _per_gene(P, "Konermann", sub)
    t["guide"] = t.guide.str.upper().str.replace("U", "T")
    by_gene = {g: x for g, x in t.groupby("gene")}
    out = np.full(len(d), np.nan)
    for gene, ix in d.groupby("gene").groups.items():
        x = by_gene.get(gene)
        if x is None:
            continue
        m = {}
        for g30, v in zip(x.guide, x.predicted_value_sigmoid):
            m.setdefault(g30[:23], v)
        out[d.index.get_indexer(ix)] = [m.get(s, np.nan) for s in d.loc[ix, "guide53"]]
    d["konermann"] = out
    return d


def add_rf(d: pd.DataFrame, P: str) -> pd.DataFrame:
    """Wessels 2020 random forest, leave-gene-out predictions for the 16 HEK293T genes; oriented by its own observed column."""
    t = pd.read_csv(os.path.join(P, "RandomForest", "off-target", "OFF-target-Lib_L1OVC_RFnbt.csv"))
    key = t.set_index("Guide_Name")["predicted"].astype(float)
    ids = d.gene + "_" + d.guide_id
    d["rf"] = -ids.map(key)                      # normCS: more negative = more depleted = more active
    return d


def add_rf_flow(d: pd.DataFrame, P: str) -> pd.DataFrame:
    """The random forest's own CasRx predictions for CD46/CD55/TFRC (the screens it was developed on, so NOT an external test for it)."""
    fr = []
    for f in sorted(glob.glob(os.path.join(P, "RandomForest", "flow-cytometry", "*.csv"))):
        t = pd.read_csv(f)
        t["gene"] = os.path.basename(f).split(".")[0].split("_")[0]
        fr.append(t)
    t = pd.concat(fr).drop_duplicates(["gene", "GuideName"]).set_index(["gene", "GuideName"])["GuideScores"].astype(float)
    d["rf"] = [t.get((g, i), np.nan) for g, i in zip(d.gene, d.guide_id)]
    return d


def add_all(d: pd.DataFrame, P: str, sub: str = "off-target", rf: bool = True) -> pd.DataFrame:
    d = add_deepcas13(d, P, sub)
    d = add_konermann(d, P, sub)
    if rf:
        d = add_rf(d, P) if sub == "off-target" else add_rf_flow(d, P)
    return d


def orient(d: pd.DataFrame, cols) -> dict:
    """Sanity check on the sign convention: Spearman of each tool's score with measured activity must be positive."""
    return {c: spearman(d["act"], d[c]) for c in cols if c in d}
