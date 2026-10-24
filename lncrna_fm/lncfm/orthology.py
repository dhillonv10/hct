"""Synteny-based positional orthologs between human and mouse lncRNA genes.

Why: most lncRNAs are not alignable across species, but a subset keep their genomic
neighbourhood (e.g. XIST/Xist, MALAT1/Malat1, H19/H19, MEG3/Meg3 ...). 'Positional
orthology' is the standard weak gold-standard for functional conservation of lncRNAs, and
tests whether a model embeds *function-like* similarity beyond alignable sequence.

Rule (all from GENCODE annotation, no external orthology table):
  * anchors = protein-coding genes whose upper-cased symbol exists in both species
  * intergenic lncRNA key = (left anchor, right anchor, strand relative to the block)
  * overlapping lncRNA key = (host anchor, same/opposite strand)    [antisense / intronic]
  * human & mouse lncRNAs sharing a key are positional-ortholog candidates
Many-to-many matches are kept (all pairs sharing a key are positives).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gencode

MAX_FLANK_DIST = 1_000_000


def _anchors(genes: pd.DataFrame, shared: set) -> pd.DataFrame:
    a = genes[(genes.gene_type == "protein_coding")].copy()
    a["sym"] = a.gene_name.str.upper()
    a = a[a.sym.isin(shared)]
    return a.sort_values(["chrom", "start"]).reset_index(drop=True)


def _keys(genes: pd.DataFrame, shared: set) -> pd.DataFrame:
    anc = _anchors(genes, shared)
    lnc = genes[genes.gene_type == "lncRNA"].copy()
    rows = []
    for chrom, L in lnc.groupby("chrom"):
        A = anc[anc.chrom == chrom]
        if A.empty:
            continue
        starts, ends = A.start.values, A.end.values
        syms, strands = A.sym.values, A.strand.values
        for gid, gname, s, e, st in zip(L.gene_id, L.gene_name, L.start, L.end, L.strand):
            ov = np.where((starts <= e) & (ends >= s))[0]
            if len(ov):
                for j in ov:
                    rows.append((gid, gname, ("host", syms[j], "same" if strands[j] == st else "opp")))
                continue
            li = np.where(ends < s)[0]
            ri = np.where(starts > e)[0]
            if not len(li) or not len(ri):
                continue
            l = li[np.argmax(ends[li])]
            r = ri[np.argmin(starts[ri])]
            if s - ends[l] > MAX_FLANK_DIST or starts[r] - e > MAX_FLANK_DIST:
                continue
            a, b = syms[l], syms[r]
            d = 1 if st == "+" else -1
            if a > b:
                a, b, d = b, a, -d
            rows.append((gid, gname, ("flank", a, b, d)))
    df = pd.DataFrame(rows, columns=["gene_id", "gene_name", "key"])
    return df


def positional_orthologs() -> pd.DataFrame:
    gh, gm = gencode.gene_table("human"), gencode.gene_table("mouse")
    sh = set(gh[gh.gene_type == "protein_coding"].gene_name.str.upper())
    sm = set(gm[gm.gene_type == "protein_coding"].gene_name.str.upper())
    shared = sh & sm
    kh, km = _keys(gh, shared), _keys(gm, shared)
    m = kh.merge(km, on="key", suffixes=("_h", "_m"))
    m["key_kind"] = m.key.map(lambda k: k[0])
    return m.drop_duplicates(["gene_id_h", "gene_id_m"]).reset_index(drop=True), kh, km
