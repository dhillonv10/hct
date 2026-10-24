"""LncBench task builders. Every task returns a tidy DataFrame with at least:

    tx_id, gene_id_nov, gene_name, chrom, seq, y            (+ task specific columns)

Labels come only from public data (ENCODE fractionation, GTEx, GENCODE annotation).
Gene-level labels are attached to ONE representative transcript per gene
(Ensembl-canonical/MANE if flagged, else the longest isoform).

Folds are chromosome-grouped (`chrom_folds`) so that neighbouring/duplicated loci never
straddle train and test; this is the minimum homology control. See PLAN.md for the
stricter near-duplicate analysis.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import gencode, paths

# ----------------------------------------------------------------------------- folds
# 5 leave-chromosome-group-out folds. NOT balanced: fold sizes range ~15%-31% of genes (fold 4 holds 8 chromosomes).
_FOLD_GROUPS = [
    ["chr1", "chr10", "chr21", "chrY"],
    ["chr2", "chr12", "chr19", "chr22"],
    ["chr3", "chr9", "chr16", "chr18"],
    ["chr4", "chr7", "chr11", "chr15"],
    ["chr5", "chr6", "chr8", "chr13", "chr14", "chr17", "chr20", "chrX"],
]
CHROM_FOLD = {c: i for i, g in enumerate(_FOLD_GROUPS) for c in g}


def chrom_folds(chroms) -> np.ndarray:
    return np.array([CHROM_FOLD.get(c, 4) for c in chroms])


def _human_rep() -> pd.DataFrame:
    lnc = gencode.build_table("human", "lnc")
    return gencode.representative_per_gene(lnc)


# ----------------------------------------------------------------------------- T1 localization
def encode_cnrci(min_total_tpm: float = 2.0, pseudo: float = 0.1) -> pd.DataFrame:
    """Per-gene mean log2(cytosol/nucleus) across cell lines (a CNRCI-like index).

    A gene counts as 'expressed' in a cell line if cytosol+nucleus TPM > min_total_tpm.
    Returns columns: gene_id_nov, cnrci, n_lines.
    """
    d = paths.DATA / "encode_fractions"
    man = pd.DataFrame(json.load(open(d / "manifest.json")))
    tp = {}
    for (cell, frac), g in man.groupby(["cell", "fraction"]):
        arr = [pd.read_csv(d / x, sep="\t", usecols=["gene_id", "TPM"]).set_index("gene_id").TPM for x in g.file]
        tp[(cell, frac)] = pd.concat(arr, axis=1).mean(axis=1)
    T = pd.DataFrame(tp)
    T.index = T.index.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    T = T[~T.index.duplicated()]
    cells = sorted({c for c, _ in tp})
    cn = pd.DataFrame({c: np.log2((T[(c, "cytosol")] + pseudo) / (T[(c, "nucleus")] + pseudo)) for c in cells})
    expr = pd.DataFrame({c: (T[(c, "cytosol")] + T[(c, "nucleus")]).gt(min_total_tpm) for c in cells})
    cn = cn.where(expr)
    out = pd.DataFrame({"cnrci": cn.mean(axis=1), "n_lines": expr.sum(axis=1)})
    return out.rename_axis("gene_id_nov").reset_index()


def task_localization(min_lines: int = 2) -> pd.DataFrame:
    """y = mean log2(cyto/nuc). `y_bin` = 1 for most-cytoplasmic tercile, 0 for most-nuclear tercile, else -1."""
    rep = _human_rep()
    lab = encode_cnrci()
    df = rep.merge(lab[lab.n_lines >= min_lines], on="gene_id_nov", how="inner")
    df = df.rename(columns={"cnrci": "y"})
    lo, hi = df.y.quantile([1 / 3, 2 / 3])
    df["y_bin"] = np.where(df.y <= lo, 0, np.where(df.y >= hi, 1, -1))
    df["fold"] = chrom_folds(df.chrom)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- T2 tissue specificity
BRAIN_PREFIX = "Brain"


def gtex_lnc_matrix() -> pd.DataFrame:
    g = pd.read_csv(paths.DATA / "gtex" / "gene_median_tpm.gct.gz", sep="\t", skiprows=2)
    g["gid"] = g.Name.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    g = g.drop_duplicates("gid").set_index("gid")
    return g.drop(columns=["Name", "Description"])


def task_tissue(min_max_tpm: float = 1.0) -> pd.DataFrame:
    """y = tau tissue-specificity index (0 ubiquitous .. 1 single-tissue) over log2(TPM+1).
    Also y_testis / y_brain: is the top tissue testis / any brain region (classification)."""
    rep = _human_rep()
    G = gtex_lnc_matrix()
    G = G.loc[G.index.isin(set(rep.gene_id_nov))]
    G = G[G.max(axis=1) >= min_max_tpm]
    X = np.log2(G + 1)
    tau = (1 - X.div(X.max(axis=1), axis=0)).sum(axis=1) / (X.shape[1] - 1)
    top = G.idxmax(axis=1)
    lab = pd.DataFrame({"y": tau, "top_tissue": top, "max_tpm": G.max(axis=1), "mean_log_tpm": X.mean(axis=1)})
    lab["y_testis"] = (lab.top_tissue == "Testis").astype(int)
    lab["y_brain"] = lab.top_tissue.str.startswith(BRAIN_PREFIX).astype(int)
    lab = lab.rename_axis("gene_id_nov").reset_index()
    df = rep.merge(lab, on="gene_id_nov", how="inner")
    df["fold"] = chrom_folds(df.chrom)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- T4 sanity: coding vs lncRNA
def task_coding_vs_lnc(n_per_class: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Sanity check only: lncRNA vs protein-coding mature transcripts (balanced subsample)."""
    lnc = _human_rep()
    pc = gencode.representative_per_gene(gencode.build_table("human", "pc"))
    rng = np.random.default_rng(seed)
    a = lnc.sample(min(n_per_class, len(lnc)), random_state=int(rng.integers(1 << 30))).assign(y=0)
    b = pc.sample(min(n_per_class, len(pc)), random_state=int(rng.integers(1 << 30))).assign(y=1)
    df = pd.concat([a, b], ignore_index=True)
    df["fold"] = chrom_folds(df.chrom)
    return df
