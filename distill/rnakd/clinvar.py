"""Clinical benchmark: ClinVar pathogenic vs benign single-nucleotide variants inside non-coding RNA genes.

Why this task and not an expression-based one: it is the only clinical endpoint we can put a SEQUENCE model on
directly. Patient-level AML tasks (which this repo also has) read expression matrices, so a 650M RNA language model
and a 5M student cannot be compared on them without an arbitrary embedding bridge. Here both score the same string.

The genes that survive the filters are real disease genes: RNU4-2 (neurodevelopmental disorder), RNU4ATAC
(Roifman / Taybi-Linder), RNU12, RNU2-2, RMRP (cartilage-hair hypoplasia), SNORD118 (leukoencephalopathy), RNU7-1,
and mitochondrial tRNAs (MT-TL1 is the MELAS gene).

Label hygiene, ported from `lncrna_fm/lncfm/ncvar.py` where these traps were found the hard way:
  * keep only SNVs inside exons of a non-coding gene that ClinVar ATTRIBUTES to that gene (GENEINFO). Without this,
    TERC variants land on the antisense lncRNA at the same locus.
  * drop anything carrying a CODING molecular consequence (missense, nonsense, splice donor/acceptor, ...): the label
    then belongs to an overlapping protein-coding gene, not to the ncRNA. UTR and intronic consequences are kept,
    because RNU2-2 sits inside the 5' UTR exon of WDR74 and excluding it by position removes a genuine disease gene.
  * for lncRNA genes only, drop a pathogenic record co-attributed to a protein-coding gene unless the disease name
    names the lncRNA (LIX1L-AS1 records are really RBM8A/TAR syndrome). Applying that rule to snRNA/snoRNA genes
    wrongly removed RNU4-2 and SNORD118, so it is restricted.

Evaluation is WITHIN-GENE (see `clinical.py`): gene identity alone separates pathogenic from benign in ClinVar, so a
pooled AUROC measures which genes are famous, not which variants are damaging.
"""
from __future__ import annotations

import gzip
import os
import re
import urllib.request

import numpy as np
import pandas as pd

GENCODE_RELEASE = "50"
GTF_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{r}/gencode.v{r}.basic.annotation.gtf.gz"
FA_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{r}/gencode.v{r}.transcripts.fa.gz"
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"

NC_TYPES = {"lncRNA", "misc_RNA", "snRNA", "snoRNA", "scaRNA", "miRNA", "rRNA", "Mt_tRNA", "Mt_rRNA", "ribozyme",
            "vault_RNA", "sRNA"}
PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BEN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
CODING_MC = ("missense", "synonymous", "nonsense", "stop_gained", "stop_lost", "frameshift", "initiator_codon",
             "start_lost", "inframe", "splice_donor", "splice_acceptor", "splice_region", "coding_sequence")
_COMP = str.maketrans("ACGT", "TGCA")


def fetch(url, dest):
    if not os.path.exists(dest):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def download(workdir, release=GENCODE_RELEASE):
    os.makedirs(workdir, exist_ok=True)
    return (fetch(GTF_URL.format(r=release), os.path.join(workdir, f"gencode.v{release}.gtf.gz")),
            fetch(FA_URL.format(r=release), os.path.join(workdir, f"gencode.v{release}.tx.fa.gz")),
            fetch(CLINVAR_URL, os.path.join(workdir, "clinvar.vcf.gz")))


# ------------------------------------------------------------------------------------------------------------ GENCODE
def read_gtf(path):
    """Exons of non-coding transcripts: {tx_id: dict(chrom, strand, gene, gtype, exons=[(start,end), ...])}."""
    tx = {}
    attr = re.compile(r'(\S+) "([^"]*)"')
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "exon":
                continue
            a = dict(attr.findall(f[8]))
            if a.get("gene_type") not in NC_TYPES:
                continue
            t = a["transcript_id"]
            rec = tx.setdefault(t, dict(chrom=f[0], strand=f[6], gene=a.get("gene_name", ""),
                                        gtype=a["gene_type"], exons=[]))
            rec["exons"].append((int(f[3]), int(f[4])))
    for r in tx.values():
        r["exons"].sort()
    print(f"  {len(tx)} non-coding transcripts in the annotation")
    return tx


def read_fasta(path, want):
    """Spliced transcript sequences for the ids in `want`. GENCODE headers are pipe-separated, field 0 is the id."""
    seqs, tid, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if tid in want:
                    seqs[tid] = "".join(buf)
                tid, buf = line[1:].split("|")[0], []
            else:
                buf.append(line.strip())
    if tid in want:
        seqs[tid] = "".join(buf)
    return seqs


class ExonIndex:
    """Sorted exon intervals per chromosome with a prefix maximum of ends, so overlap is a binary search."""

    def __init__(self, tx):
        by = {}
        for tid, r in tx.items():
            for s, e in r["exons"]:
                by.setdefault(r["chrom"], []).append((s, e, tid))
        self.c = {}
        for chrom, v in by.items():
            v.sort()
            self.c[chrom] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                             np.maximum.accumulate(np.array([x[1] for x in v])), [x[2] for x in v])

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


def tx_offset(exons, strand, pos):
    """0-based offset of a genomic position in the spliced transcript, or None if it is not exonic."""
    cum = 0
    for s, e in sorted(exons, reverse=(strand == "-")):
        if s <= pos <= e:
            return cum + ((pos - s) if strand == "+" else (e - pos))
        cum += e - s + 1
    return None


# ------------------------------------------------------------------------------------------------------------ ClinVar
def read_clinvar(path):
    """P/LP and B/LB single-nucleotide variants from the ClinVar GRCh38 VCF."""
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            ref, alt = f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            info = dict(kv.split("=", 1) for kv in f[7].split(";") if "=" in kv)
            sig = info.get("CLNSIG", "")
            label = 1 if sig in PATH else 0 if sig in BEN else None
            if label is None:
                continue
            genes = {g.split(":")[0] for g in info.get("GENEINFO", "").split("|") if g}
            rows.append(dict(chrom="chrM" if f[0] in ("MT", "M") else "chr" + f[0], pos=int(f[1]), ref=ref, alt=alt,
                             label=label, genes=genes, mc=info.get("MC", ""), dn=info.get("CLNDN", "")))
    d = pd.DataFrame(rows)
    print(f"  {len(d)} ClinVar P/LP or B/LB SNVs ({int(d.label.sum())} pathogenic)")
    return d


def build_panel(workdir, min_each=3, release=GENCODE_RELEASE):
    """Returns the labelled table and a per-gene summary. One row = one variant on one transcript."""
    gtf, fa, cv = download(workdir, release)
    tx = read_gtf(gtf)
    idx = ExonIndex(tx)
    var = read_clinvar(cv)

    n_coding = n_unattrib = 0
    hits = []
    for v in var.itertuples(index=False):
        if any(c in v.mc for c in CODING_MC):                                   # the label belongs to a coding gene
            n_coding += 1
            continue
        for tid in idx.hits(v.chrom, v.pos):
            r = tx[tid]
            if r["gene"] not in v.genes:                                        # ClinVar does not attribute it here
                n_unattrib += 1
                continue
            off = tx_offset(r["exons"], r["strand"], v.pos)
            if off is None:
                continue
            ref, alt = (v.ref, v.alt) if r["strand"] == "+" else (v.ref.translate(_COMP), v.alt.translate(_COMP))
            hits.append(dict(tx=tid, gene=r["gene"], gtype=r["gtype"], chrom=v.chrom, pos=v.pos, offset=off,
                             ref=ref, alt=alt, label=v.label, dn=v.dn, co_coding=len(v.genes) > 1))
    d = pd.DataFrame(hits)
    print(f"  {len(d)} variant-transcript pairs ({n_coding} dropped for a coding consequence, "
          f"{n_unattrib} for gene attribution)")

    # lncRNA-only rule: a pathogenic record co-attributed to a coding gene belongs to that gene unless the disease
    # names the lncRNA. Restricted to lncRNA because applying it everywhere removed RNU4-2 and most of SNORD118.
    names_disease = np.array([g.split("-")[0].upper() in str(dn).upper() for g, dn in zip(d.gene, d.dn)])
    drop = (d.gtype == "lncRNA") & (d.label == 1) & d.co_coding & ~names_disease
    print(f"  {int(drop.sum())} lncRNA pathogenic records dropped as belonging to an overlapping coding gene")
    d = d[~drop].reset_index(drop=True)

    seqs = read_fasta(fa, set(d.tx))
    d["seq"] = d.tx.map(seqs)
    d = d[d.seq.notna()].reset_index(drop=True)
    ok = [s[o:o + 1].upper().replace("U", "T") == r for s, o, r in zip(d.seq, d.offset, d.ref)]
    print(f"  reference base agrees with the transcript for {sum(ok)}/{len(ok)} variants")
    d = d[ok].reset_index(drop=True)

    # one transcript per gene (the longest), so a gene is not counted several times
    d["_len"] = d.seq.str.len()
    pick = d.sort_values("_len").groupby("gene").tx.last().to_dict()            # longest transcript per gene
    d = d[[t == pick[g] for t, g in zip(d.tx, d.gene)]].drop(columns="_len").reset_index(drop=True)
    d = d.drop_duplicates(["gene", "offset", "alt"]).reset_index(drop=True)

    g = d.groupby("gene").label.agg(["sum", "count"])
    keep = g[(g["sum"] >= min_each) & (g["count"] - g["sum"] >= min_each)].index
    panel = d[d.gene.isin(keep)].reset_index(drop=True)
    summary = panel.groupby(["gene", "gtype"]).label.agg(path="sum", n="count").reset_index()
    summary["benign"] = summary.n - summary.path
    print(f"\n  PANEL: {len(panel)} variants, {panel.label.sum()} pathogenic, {panel.gene.nunique()} genes "
          f"(>= {min_each} of each class)")
    print(summary.to_string(index=False))
    return panel, summary


def synthetic(n_genes=6, seed=0):
    """A stand-in panel with the same columns, so the whole path can be rehearsed with no downloads."""
    rs = np.random.default_rng(seed)
    rows = []
    for gi in range(n_genes):
        L = int(rs.integers(70, 200))
        seq = "".join(rs.choice(list("ACGT"), L))
        hot = set(rs.choice(L, max(8, L // 3), replace=False))                   # a "functional core" of the gene
        for _ in range(40):
            o = int(rs.integers(0, L))
            alt = rs.choice([c for c in "ACGT" if c != seq[o]])
            rows.append(dict(tx=f"T{gi}", gene=f"GENE{gi}", gtype="snRNA", chrom="chr1", pos=1000 + o, offset=o,
                             ref=seq[o], alt=str(alt), label=int(o in hot), dn="", co_coding=False, seq=seq))
    d = pd.DataFrame(rows).drop_duplicates(["gene", "offset", "alt"]).reset_index(drop=True)
    g = d.groupby("gene").label.agg(["sum", "count"])
    keep = g[(g["sum"] >= 3) & (g["count"] - g["sum"] >= 3)].index
    return d[d.gene.isin(keep)].reset_index(drop=True)


def transfer_sequences(fa_path, exclude_genes, width=256, max_n=None, seed=0):
    """Unlabelled windows of human non-coding transcripts, for distillation. The panel's genes are removed first.

    This is the free transfer set: no labels are needed because the targets are the teacher's own predictions (and,
    for the baseline arm, the masked nucleotide itself). Excluding the panel genes is what keeps the later zero-shot
    clinical evaluation honest -- without it the student would be trained on the very sequences it is tested on."""
    rs = np.random.default_rng(seed)
    out, dropped = [], 0
    with gzip.open(fa_path, "rt") as fh:
        tid = gene = gtype = None
        buf = []

        def flush():
            nonlocal dropped
            if not buf or gtype not in NC_TYPES:
                return
            if gene in exclude_genes:
                dropped += 1
                return
            s = "".join(buf).upper().replace("U", "T")
            for i in range(0, len(s), width):
                w = s[i:i + width]
                if len(w) >= 32 and set(w) <= set("ACGT"):
                    out.append(w)
        for line in fh:
            if line[0] == ">":
                flush()
                f = line[1:].rstrip().split("|")
                tid, gene, gtype, buf = f[0], (f[5] if len(f) > 5 else ""), (f[7] if len(f) > 7 else ""), []
            else:
                buf.append(line.strip())
        flush()
    rs.shuffle(out)
    if max_n:
        out = out[:max_n]
    print(f"  transfer set: {len(out)} windows of <= {width} nt ({dropped} transcripts of panel genes excluded)")
    return out


RFAM_SEED_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz"

# Rfam accessions of the families the panel's genes belong to. Best effort, NOT verified against Rfam -- used only
# when --exclude-families is passed. The default is to KEEP them, because the teacher was pretrained on RNAcentral,
# which contains these families: removing them from the student's data would compare a model that has seen the
# family against one that has not, which is not the comparison we want to make.
PANEL_FAMILIES = {"RNU4-2": "RF00015", "RNU4ATAC": "RF00618", "RNU12": "RF00007", "RNU2-2": "RF00004",
                  "RNU7-1": "RF00066", "RMRP": "RF00030", "SNORD118": "RF00137"}
MT_TRNA_FAMILY = "RF00005"


def rfam_sequences(workdir, width=256, max_n=None, exclude_acc=(), seed=0, min_len=32):
    """Seed sequences of every Rfam family, gaps removed. ~4,200 families of structured ncRNA.

    This exists because GENCODE's non-coding set is overwhelmingly lncRNA, while the clinical panel is snRNA,
    snoRNA, ribozyme and mitochondrial tRNA. A student trained on GENCODE alone never meets the families it is
    tested on, which is the most likely reason the first run's students scored near chance. Rfam seeds are small,
    curated and span the families directly.
    """
    path = fetch(RFAM_SEED_URL, os.path.join(workdir, "Rfam.seed.gz"))
    out, acc, kept_fams, skipped = [], None, set(), 0
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#=GF AC"):
                acc = line.split()[-1].split(".")[0]
            elif line.startswith("//"):
                acc = None
            elif line and not line.startswith("#"):
                if acc in exclude_acc:
                    skipped += 1
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                s = parts[1].upper().replace("U", "T").replace(".", "").replace("-", "").replace("~", "")
                if len(s) >= min_len and set(s) <= set("ACGT"):
                    kept_fams.add(acc)
                    out.append(s[:width])
    rs = np.random.default_rng(seed)
    rs.shuffle(out)
    if max_n:
        out = out[:max_n]
    print(f"  Rfam seeds: {len(out)} sequences from {len(kept_fams)} families"
          + (f" ({skipped} rows skipped from excluded families)" if skipped else ""))
    return out


def gencode_composition(fa_path, exclude_genes, width=256):
    """Windows per gene_type in the GENCODE transfer set. Prints the thing the first run did not check."""
    counts = {}
    with gzip.open(fa_path, "rt") as fh:
        gtype, gene, n = None, "", 0
        for line in fh:
            if line[0] == ">":
                if gtype in NC_TYPES and gene not in exclude_genes and n:
                    counts[gtype] = counts.get(gtype, 0) + max(1, n // width)
                f = line[1:].rstrip().split("|")
                gene, gtype, n = (f[5] if len(f) > 5 else ""), (f[7] if len(f) > 7 else ""), 0
            else:
                n += len(line.strip())
        if gtype in NC_TYPES and gene not in exclude_genes and n:
            counts[gtype] = counts.get(gtype, 0) + max(1, n // width)
    tot = sum(counts.values()) or 1
    print("  GENCODE transfer-set composition by gene type (windows):")
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {t:12s} {c:8,d}  {100 * c / tot:5.1f}%")
    return counts
