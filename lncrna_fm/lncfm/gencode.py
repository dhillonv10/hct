"""Parse GENCODE transcript FASTA + GTF into one tidy transcript table.

Table columns (one row per transcript):
  tx_id, gene_id, gene_name, gene_type, chrom, strand, n_exons, length,
  exon_lens (tuple of exon lengths in 5'->3' transcript order), canonical (bool), seq (RNA alphabet, ACGU)

`build_table("human", "lnc")` is cached to $LNCFM_DATA/processed/.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import download, paths

_ATTR = re.compile(r'(\w+) "([^"]*)"')


def read_fasta_gz(path):
    hdr, buf = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr, buf = line[1:].rstrip("\n"), []
            else:
                buf.append(line.strip())
    if hdr is not None:
        yield hdr, "".join(buf)


def parse_gtf_exons(path):
    """Return (tx_exons, tx_meta). tx_exons[tx_id] = list[(start,end)] genomic; tx_meta[tx_id]=dict."""
    tx_exons = defaultdict(list)
    tx_meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            feat = f[2]
            if feat not in ("exon", "transcript"):
                continue
            attrs = {}
            tags = []
            for k, v in _ATTR.findall(f[8]):
                if k == "tag":
                    tags.append(v)
                else:
                    attrs[k] = v
            tid = attrs["transcript_id"]
            if feat == "transcript":
                tx_meta[tid] = {
                    "gene_id": attrs["gene_id"], "gene_name": attrs.get("gene_name", ""),
                    "gene_type": attrs.get("gene_type", ""), "chrom": f[0], "strand": f[6],
                    "tx_start": int(f[3]), "tx_end": int(f[4]),
                    "canonical": ("Ensembl_canonical" in tags) or ("MANE_Select" in tags),
                    "tsl": attrs.get("transcript_support_level", ""),
                }
            else:
                tx_exons[tid].append((int(f[3]), int(f[4])))
    return tx_exons, tx_meta


def _cached_gtf(species, gtf):
    import pickle
    cache = paths.ensure(paths.DATA / "processed") / f"{species}_gtf_exons.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    res = parse_gtf_exons(gtf)
    res = (dict(res[0]), res[1])
    cache.write_bytes(pickle.dumps(res))
    return res


def gene_table(species: str = "human") -> pd.DataFrame:
    """Gene-level coordinates/type for ALL genes (used for synteny + flanking-gene logic)."""
    out = paths.ensure(paths.DATA / "processed") / f"{species}_genes.pkl"
    if out.exists():
        return pd.read_pickle(out)
    gtf = next(d for _, d in download.gencode_files(species) if str(d).endswith("annotation.gtf.gz"))
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            if f[2] != "gene":
                continue
            a = dict(_ATTR.findall(f[8]))
            rows.append({"gene_id": a["gene_id"], "gene_name": a.get("gene_name", ""), "gene_type": a.get("gene_type", ""),
                         "chrom": f[0], "start": int(f[3]), "end": int(f[4]), "strand": f[6]})
    df = pd.DataFrame(rows)
    df["gene_id_nov"] = df.gene_id.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    df.to_pickle(out)
    return df


def _exon_lens_5to3(exons, strand):
    ex = sorted(exons)
    if strand == "-":
        ex = ex[::-1]
    return tuple(e - s + 1 for s, e in ex)


def build_table(species: str = "human", kind: str = "lnc", force: bool = False) -> pd.DataFrame:
    """kind: 'lnc' (lncRNA transcripts) or 'pc' (protein-coding transcripts)."""
    out = paths.ensure(paths.DATA / "processed") / f"{species}_{kind}.pkl"
    if out.exists() and not force:
        return pd.read_pickle(out)
    files = {Path(d).name: d for _, d in download.gencode_files(species)}
    fa = next(d for n, d in files.items() if ("lncRNA_transcripts" if kind == "lnc" else "pc_transcripts") in n)
    gtf = next(d for n, d in files.items() if n.endswith("annotation.gtf.gz"))
    tx_exons, tx_meta = _cached_gtf(species, gtf)
    rows = []
    for hdr, seq in read_fasta_gz(fa):
        p = hdr.split("|")
        tid = p[0]
        m = tx_meta.get(tid)
        if m is None:
            continue
        seq = seq.upper().replace("T", "U")
        rows.append({
            "tx_id": tid, "gene_id": m["gene_id"], "gene_name": m["gene_name"], "gene_type": m["gene_type"],
            "chrom": m["chrom"], "strand": m["strand"], "tx_start": m["tx_start"], "tx_end": m["tx_end"],
            "n_exons": len(tx_exons[tid]), "length": len(seq), "canonical": m["canonical"], "tsl": m["tsl"],
            "exon_lens": _exon_lens_5to3(tx_exons[tid], m["strand"]), "seq": seq,
        })
    df = pd.DataFrame(rows)
    df["gene_id_nov"] = df.gene_id.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    df.to_pickle(out)
    return df


def representative_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    """One transcript per gene: canonical/MANE if flagged, else the longest."""
    d = df.assign(_c=df.canonical.astype(int)).sort_values(["gene_id", "_c", "length"], ascending=[True, False, False])
    return d.drop_duplicates("gene_id").drop(columns="_c").reset_index(drop=True)


# Chromosome-held-out splits (gene-level; all isoforms of a gene share a split).
TEST_CHROMS = {"chr8", "chr13", "chr18", "chr21"}
VAL_CHROMS = {"chr9", "chr16", "chr22"}


def split_of(chrom: str) -> str:
    if chrom in TEST_CHROMS:
        return "test"
    if chrom in VAL_CHROMS:
        return "val"
    return "train"
