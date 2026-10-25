"""ncRNA variant tables: map SNVs onto spliced non-coding transcripts and build labelled evaluation sets.

Everything the variant-effect code needs sits here: the GENCODE exon index, a ClinVar SNV parser, the
mapping from genomic coordinates to a (transcript, spliced offset, strand-corrected ref/alt) triple, and
the label sets.

Label sets
  clinvar : ClinVar pathogenic / likely pathogenic (label 1) vs benign / likely benign (label 0), SNVs that fall
            inside exons of a non-coding gene (lncRNA, snRNA, snoRNA, Mt_tRNA, ...), that ClinVar attributes to THAT
            gene (GENEINFO), and that carry no coding consequence (missense, nonsense, splice site, ...) for an
            overlapping protein-coding gene. Without these two filters the label belongs to the coding gene (TTN-AS1)
            or to the opposite-strand gene (TERC vs its antisense transcript), not to the ncRNA being scored.
            UTR and intronic consequences are allowed: RNU2-2 sits inside the 5' UTR exon of WDR74.
            For lncRNA genes only: a pathogenic record that ClinVar also attributes to a protein-coding gene is dropped
            unless the disease name mentions the lncRNA ("RBM8A-related_disorder" on the antisense lncRNA LIX1L-AS1
            goes). Antisense lncRNAs overlapping coding genes are the documented failure mode. The rule is NOT applied
            to snRNA / snoRNA / tRNA genes: the same co-attribution there marks genuine ncRNA disease genes (RNU4-2,
            SNORD118), and a first version applied to all types wrongly removed them. The lncRNA rule was added after the
            first scoring run showed LIX1L-AS1 / HNRNPK-AS1 in the panel (3 pathogenic records each, both really
            RBM8A / HNRNPK disease); results are reported with and without it.
  gnomad  : (optional, `gnomad.py`) common population variants in the same genes, used as an extra benign set.

The scoring code never sees the label; evaluation is WITHIN-GENE (see `varscore.within_gene_auroc`) because
gene identity alone separates pathogenic from benign in ClinVar (a gene with a known disease has P/LP, a big
benign-only gene does not).
"""
from __future__ import annotations

import gzip
import pickle
import re

import numpy as np
import pandas as pd

from . import gencode, paths

NC_TYPES = {"lncRNA", "misc_RNA", "snRNA", "snoRNA", "scaRNA", "miRNA", "rRNA", "Mt_tRNA", "Mt_rRNA", "ribozyme",
            "vault_RNA", "sRNA"}
PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BEN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
_COMP = str.maketrans("ACGT", "TGCA")
CODING_MC = ("missense", "synonymous", "nonsense", "stop_gained", "stop_lost", "frameshift", "initiator_codon",
             "start_lost", "inframe", "splice_donor", "splice_acceptor", "splice_region", "coding_sequence")
_STARS = {"practice_guideline": 4, "reviewed_by_expert_panel": 3, "criteria_provided,_multiple_submitters,_no_conflicts": 2,
          "criteria_provided,_single_submitter": 1, "criteria_provided,_conflicting_classifications": 0,
          "criteria_provided,_conflicting_interpretations": 0}


def _chrom(c: str) -> str:
    return "chrM" if c in ("MT", "M") else "chr" + c


def load_annotation():
    return pickle.loads((paths.DATA / "processed" / "human_gtf_exons.pkl").read_bytes())


class ExonIndex:
    """Sorted exon intervals per chromosome with a prefix-maximum of ends for overlap queries."""

    def __init__(self, items):
        by = {}
        for chrom, s, e, tid in items:
            by.setdefault(chrom, []).append((s, e, tid))
        self.c = {}
        for chrom, v in by.items():
            v.sort()
            starts = np.array([x[0] for x in v])
            ends = np.array([x[1] for x in v])
            self.c[chrom] = (starts, ends, np.maximum.accumulate(ends), [x[2] for x in v])

    def hits(self, chrom, pos):
        if chrom not in self.c:
            return []
        starts, ends, pmax, tids = self.c[chrom]
        j = int(np.searchsorted(starts, pos, "right")) - 1
        out = []
        while j >= 0 and pmax[j] >= pos:
            if ends[j] >= pos:
                out.append(tids[j])
            j -= 1
        return out


def spliced_len(exons):
    return sum(e - s + 1 for s, e in exons)


def tx_offset(exons, strand, pos):
    """0-based offset of genomic `pos` in the spliced transcript, or None if intronic."""
    cum = 0
    for s, e in sorted(exons, reverse=(strand == "-")):
        if s <= pos <= e:
            return cum + ((pos - s) if strand == "+" else (e - pos))
        cum += e - s + 1
    return None


def read_clinvar_snvs(path=None, min_stars=0) -> pd.DataFrame:
    """P/LP/B/LB single-nucleotide variants from the ClinVar VCF (GRCh38)."""
    path = path or paths.DATA / "clinvar" / "clinvar.vcf.gz"
    rows, n_coding = [], 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            if len(f[3]) != 1 or len(f[4]) != 1 or f[4] not in "ACGT" or f[3] not in "ACGT":
                continue
            info = f[7]
            m = re.search(r"CLNSIG=([^;]+)", info)
            if not m or (m.group(1) not in PATH and m.group(1) not in BEN):
                continue
            rs = re.search(r"CLNREVSTAT=([^;]+)", info)
            stars = _STARS.get(rs.group(1), 0) if rs else 0
            if stars < min_stars:
                continue
            d = re.search(r"CLNDN=([^;]+)", info)
            mc = re.search(r"MC=([^;]+)", info)
            mc = mc.group(1) if mc else ""
            if any(c in mc for c in CODING_MC):          # consequence for a protein-coding gene: label is not about the ncRNA
                n_coding += 1
                continue
            gi = re.search(r"GENEINFO=([^;]+)", info)
            sym = ",".join(x.split(":")[0] for x in gi.group(1).split("|")) if gi else ""
            rows.append((_chrom(f[0]), int(f[1]), f[3], f[4], int(m.group(1) in PATH), m.group(1), stars,
                         d.group(1)[:400] if d else "", f[2], sym))
    df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "label", "clnsig", "stars", "disease", "clinvar_id",
                                     "geneinfo"])
    df.attrs["coding_consequence_dropped"] = n_coding
    return df


def map_to_transcripts(df: pd.DataFrame, types=NC_TYPES, attributed_only=True):
    """Attach gene / transcript / offset / strand-corrected alleles to genomic SNVs (columns chrom,pos,ref,alt).

    Returns (mapped_df, report) where mapped_df has gene_id, gene, gene_type, tx_id, strand, offset, ref_t, alt_t.
    With `attributed_only`, a variant is kept only for a gene whose symbol appears in ClinVar's GENEINFO (or for any
    gene when the variant has no GENEINFO symbol and the gene has no symbol of its own).
    """
    tx_exons, tx_meta = load_annotation()
    pc_names = {m["gene_name"] for m in tx_meta.values() if m["gene_type"] == "protein_coding"}
    items = []
    for tid, ex in tx_exons.items():
        m = tx_meta.get(tid)
        if m and m["gene_type"] in types:
            for s, e in ex:
                items.append((m["chrom"], s, e, tid))
    idx = ExonIndex(items)
    rep = {"input": len(df), "no_ncrna_exon": 0, "not_attributed": 0, "pathogenic_attributed_to_coding_gene": 0}
    out = []
    for r in df.itertuples():
        tids = idx.hits(r.chrom, r.pos)
        if not tids:
            rep["no_ncrna_exon"] += 1
            continue
        syms = set(r.geneinfo.split(",")) if r.geneinfo else set()
        best = {}
        for t in tids:                                   # one transcript per gene: canonical first, then longest
            m = tx_meta[t]
            if attributed_only and not (m["gene_name"] in syms or (not syms and m["gene_name"].startswith("ENSG"))):
                continue
            key = (not m["canonical"], -spliced_len(tx_exons[t]))
            if m["gene_id"] not in best or key < best[m["gene_id"]][0]:
                best[m["gene_id"]] = (key, t)
        if not best:
            rep["not_attributed"] += 1
        for _, t in best.values():
            m = tx_meta[t]
            if r.label == 1 and attributed_only and m["gene_type"] == "lncRNA":   # pathogenic, but really the overlapping coding gene's disease?
                others = (syms - {m["gene_name"]}) & pc_names
                if others and m["gene_name"].lower() not in r.disease.lower():
                    rep["pathogenic_attributed_to_coding_gene"] += 1
                    continue
            off = tx_offset(tx_exons[t], m["strand"], r.pos)
            ref, alt = (r.ref, r.alt) if m["strand"] == "+" else (r.ref.translate(_COMP), r.alt.translate(_COMP))
            out.append({**r._asdict(), "gene_id": m["gene_id"], "gene": m["gene_name"], "gene_type": m["gene_type"],
                        "tx_id": t, "strand": m["strand"], "offset": off, "ref_t": ref, "alt_t": alt})
    res = pd.DataFrame(out).drop(columns=["Index"], errors="ignore")
    return res, rep


def attach_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Add the spliced transcript sequence (RNA alphabet) and drop variants whose reference base disagrees."""
    want = set(df.tx_id)
    fa = paths.human(f"gencode.v{paths.GENCODE_HUMAN_RELEASE}.transcripts.fa.gz")
    seqs = {}
    for hdr, seq in gencode.read_fasta_gz(fa):
        tid = hdr.split("|")[0]
        if tid in want:
            seqs[tid] = seq.upper().replace("T", "U")
    df = df[df.tx_id.isin(seqs)].copy()
    ok = [seqs[t][o] == r.replace("T", "U") for t, o, r in zip(df.tx_id, df.offset, df.ref_t)]
    df.attrs["dropped_ref_mismatch"] = int((~np.array(ok)).sum())
    df = df[ok].copy()
    df["seq"] = [seqs[t] for t in df.tx_id]
    df["tx_len"] = df.seq.str.len()
    return df.reset_index(drop=True)


def build_clinvar_table(min_stars=0, refresh=False) -> pd.DataFrame:
    """Labelled ClinVar ncRNA SNVs with sequences. Cached; `min_stars` filters on ClinVar review status."""
    out = paths.ensure(paths.DATA / "processed") / f"ncvar_clinvar_v3_s{min_stars}.pkl"
    if out.exists() and not refresh:
        return pd.read_pickle(out)
    raw = read_clinvar_snvs(min_stars=min_stars)
    mapped, rep = map_to_transcripts(raw)
    rep["coding_consequence_dropped"] = raw.attrs.get("coding_consequence_dropped", 0)
    df = attach_sequences(mapped)
    df = df.drop_duplicates(["tx_id", "offset", "alt_t", "label"]).reset_index(drop=True)
    df.attrs.update(rep)
    df.to_pickle(out)
    return df


def eligible_genes(df: pd.DataFrame, min_each=3) -> pd.DataFrame:
    """Genes with at least `min_each` pathogenic and `min_each` benign variants (the within-gene design needs both)."""
    g = df.groupby(["gene_id", "gene", "gene_type"]).label.agg(n_path="sum", n="size").reset_index()
    g["n_ben"] = g.n - g.n_path
    return g[(g.n_path >= min_each) & (g.n_ben >= min_each)].sort_values("n", ascending=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stars", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    d = build_clinvar_table(a.min_stars, a.refresh)
    print(f"{len(d)} SNVs; {int(d.label.sum())} P/LP, {int((1 - d.label).sum())} B/LB; {d.gene.nunique()} genes; "
          f"dropped (ref mismatch) {d.attrs.get('dropped_ref_mismatch', 'n/a')}")
    print(d.groupby("gene_type").label.agg(n="size", n_path="sum").to_string())
    print(eligible_genes(d).to_string(index=False))
