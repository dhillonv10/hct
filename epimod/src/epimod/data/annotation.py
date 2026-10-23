"""Genomic context for CpG probes, parsed from the GEO platform table of the Illumina 450K array
(GPL13534; hg19). The context is what lets EpiMod tokenise a CpG by *where it is and what it sits
in* rather than only by a learned id.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import subdir
from .download import download

GPL450K_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13534&targ=self&form=text&view=full"

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
ISLAND = ["OpenSea", "N_Shelf", "N_Shore", "Island", "S_Shore", "S_Shelf"]
GENE_GROUPS = ["TSS1500", "TSS200", "5'UTR", "1stExon", "Body", "3'UTR"]
REG_GROUPS = [
    "None",
    "Unclassified",
    "Unclassified_Cell_type_specific",
    "NonGene_Associated",
    "NonGene_Associated_Cell_type_specific",
    "Gene_Associated",
    "Gene_Associated_Cell_type_specific",
    "Promoter_Associated",
    "Promoter_Associated_Cell_type_specific",
]
_COLS = [
    "ID",
    "Infinium_Design_Type",
    "CHR",
    "MAPINFO",
    "UCSC_RefGene_Name",
    "UCSC_RefGene_Group",
    "Relation_to_UCSC_CpG_Island",
    "Enhancer",
    "Regulatory_Feature_Group",
    "DHS",
    "Probe_SNPs",
    "Probe_SNPs_10",
]


def _table_bounds(path: Path) -> tuple[int, int]:
    start = end = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith("!platform_table_begin"):
                start = i + 1
            elif line.startswith("!platform_table_end"):
                end = i
                break
    if start is None or end is None:
        raise ValueError(f"{path} has no platform table")
    return start, end


def parse_platform_table(path: Path) -> pd.DataFrame:
    """Parse a GEO SOFT platform file into a compact, model-ready annotation frame."""
    start, end = _table_bounds(path)
    raw = pd.read_csv(
        path, sep="\t", skiprows=start, nrows=end - start - 1, usecols=_COLS, dtype=str, keep_default_na=False,
        low_memory=False,
    ).set_index("ID")
    raw = raw[raw.index.str.startswith("cg")]
    chrom = raw["CHR"].map({c: i for i, c in enumerate(CHROMS)})
    pos = pd.to_numeric(raw["MAPINFO"], errors="coerce")
    island = raw["Relation_to_UCSC_CpG_Island"].replace("", "OpenSea").map({c: i for i, c in enumerate(ISLAND)})
    grp = raw["UCSC_RefGene_Group"].fillna("")
    gene_hot = np.stack([grp.str.contains(g, regex=False).to_numpy() for g in GENE_GROUPS], axis=1).astype(np.int8)
    reg = raw["Regulatory_Feature_Group"].replace("", "None").map({c: i for i, c in enumerate(REG_GROUPS)})
    out = pd.DataFrame(
        {
            "chrom": chrom,
            "pos": pos,
            "island": island,
            "reg": reg,
            "enhancer": (raw["Enhancer"].str.upper() == "TRUE").astype(np.int8),
            "dhs": (raw["DHS"].str.upper() == "TRUE").astype(np.int8),
            "design2": (raw["Infinium_Design_Type"] == "II").astype(np.int8),
            # common SNP in the probe body or within 10 bp of the CpG: values partly reflect genotype
            "snp": ((raw["Probe_SNPs"] != "") | (raw["Probe_SNPs_10"] != "")).astype(np.int8),
            "genes": raw["UCSC_RefGene_Name"].str.split(";").str[0].replace("", "-"),
        },
        index=raw.index,
    )
    for j, g in enumerate(GENE_GROUPS):
        out[f"gene_{g}"] = gene_hot[:, j]
    out = out.dropna(subset=["chrom", "pos", "island", "reg"])
    for c in ("chrom", "island", "reg"):
        out[c] = out[c].astype(np.int16)
    out["pos"] = out["pos"].astype(np.int64)
    return out


def load_probe_annotation() -> pd.DataFrame:
    """450K probe annotation, cached as parquet under the data root."""
    cache = subdir("annot") / "annotation_450k.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    soft = download(GPL450K_URL, subdir("annot") / "GPL13534_full.txt")
    ann = parse_platform_table(soft)
    ann.to_parquet(cache)
    return ann
