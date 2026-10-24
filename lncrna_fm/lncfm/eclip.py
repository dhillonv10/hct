"""RNA-binding-protein (RBP) binding task for lncRNAs from ENCODE eCLIP.

    python -m lncfm.eclip download            # ~250 small IDR peak files (~25 MB)

Label for (lncRNA gene g, RBP r, cell line c):
    1  if a reproducible (IDR) eCLIP peak of r in c overlaps an exon of g's representative transcript on the same strand
    0  if g is *expressed in c* (mean cytosol/nucleus TPM > 1) and has no such peak
  (genes not expressed in c are excluded: they cannot show binding, so including them as negatives would
   just teach the model "expressed vs silent")
Because binding is strongly expression dependent, every result is also reported with an expression-
stratified AUROC and compared with an expression + length baseline that sees no sequence.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from . import download, gencode, paths

ENCODE = "https://www.encodeproject.org"
H = {"accept": "application/json"}
CELLS = ("K562", "HepG2")


def download_peaks(cells=CELLS, max_exp=None):
    d = paths.ensure(paths.DATA / "eclip")
    man = d / "manifest.json"
    if man.exists():
        print("eclip peaks already downloaded:", len(json.load(open(man))))
        return
    r = requests.get(ENCODE + "/search/", params=dict(type="Experiment", status="released", limit="all", format="json",
                     assay_title="eCLIP", assembly="GRCh38"), headers=H, timeout=120).json()["@graph"]
    exps = [e for e in r if e["biosample_ontology"]["term_name"] in cells]
    if max_exp:
        exps = exps[:max_exp]
    rows = []
    for k, e in enumerate(exps):
        j = requests.get(ENCODE + e["@id"] + "?format=json", headers=H, timeout=90).json()
        files = [f for f in j["files"] if f.get("output_type") == "peaks" and f.get("file_format") == "bed"
                 and f.get("file_format_type") == "narrowPeak" and f.get("assembly") == "GRCh38"
                 and len(f.get("biological_replicates") or []) == 2 and f.get("status") == "released"]
        if not files:
            continue
        f = files[0]
        cell, rbp = j["biosample_ontology"]["term_name"], j["target"]["label"]
        dest = d / f"{cell}__{rbp}__{f['accession']}.bed"
        download.fetch(ENCODE + f["href"], dest, verify_gz=False)
        rows.append({"cell": cell, "rbp": rbp, "experiment": j["accession"], "file": dest.name, "bytes": f.get("file_size")})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(exps)}", flush=True)
    man.write_text(json.dumps(rows, indent=1))
    print("eclip peak sets:", len(rows))


def _load_bed(path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"                  # ENCODE serves .bed.gz; we saved it under a .bed name
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 5], names=["chrom", "s", "e", "strand"],
                       compression="gzip" if gz else None)


def _peak_index(df):
    """chrom/strand -> (sorted starts, prefix-max of ends)."""
    idx = {}
    for (c, st), g in df.groupby(["chrom", "strand"]):
        g = g.sort_values("s")
        idx[(c, st)] = (g.s.values, np.maximum.accumulate(g.e.values))
    return idx


def encode_expression(cell: str) -> pd.Series:
    """Mean total (cytosol+nucleus)/2 TPM per ENSG (no version) in an ENCODE cell line."""
    d = paths.DATA / "encode_fractions"
    man = pd.DataFrame(json.load(open(d / "manifest.json")))
    man = man[man.cell == cell]
    arr = [pd.read_csv(d / x, sep="\t", usecols=["gene_id", "TPM"]).set_index("gene_id").TPM for x in man.file]
    s = pd.concat(arr, axis=1).mean(axis=1)
    s.index = s.index.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    return s[~s.index.duplicated()]


def build_binding_table(cell: str, min_pos: int = 15):
    """Returns (df, Y, rbps): df rows = expressed lncRNA genes (representative transcript), Y (n x R) 0/1."""
    out = paths.ensure(paths.DATA / "processed") / f"rbp_{cell}.pkl"
    if out.exists():
        return pickle.loads(out.read_bytes())
    tx_exons, tx_meta = gencode._cached_gtf("human", None)
    rep = gencode.representative_per_gene(gencode.build_table("human", "lnc"))
    expr = encode_expression(cell)
    rep = rep.assign(tpm=rep.gene_id_nov.map(expr)).dropna(subset=["tpm"])
    rep = rep[rep.tpm > 1.0].reset_index(drop=True)
    # exons of the representative transcript, 0-based half-open
    ex_rows = []
    for gi, (tid, chrom, strand) in enumerate(zip(rep.tx_id, rep.chrom, rep.strand)):
        for s, e in tx_exons[tid]:
            ex_rows.append((gi, chrom, strand, s - 1, e))
    EX = pd.DataFrame(ex_rows, columns=["g", "chrom", "strand", "s", "e"])
    man = pd.DataFrame(json.load(open(paths.DATA / "eclip" / "manifest.json")))
    man = man[man.cell == cell]
    cols = {}
    for rbp, grp in man.groupby("rbp"):
        hit = np.zeros(len(EX), bool)
        for f in grp.file:
            idx = _peak_index(_load_bed(paths.DATA / "eclip" / f))
            for (c, st), sub in EX.groupby(["chrom", "strand"]):
                if (c, st) not in idx:
                    continue
                starts, pmax = idx[(c, st)]
                i = np.searchsorted(starts, sub.e.values, side="left")         # peaks with start < exon end
                ok = (i > 0) & (pmax[np.maximum(i - 1, 0)] > sub.s.values)     # some of them end after exon start
                hit[sub.index.values] |= ok
        y = np.zeros(len(rep), np.float32)
        y[np.unique(EX.g.values[hit])] = 1.0
        cols[rbp] = y
    Y = pd.DataFrame(cols)
    keep = Y.columns[Y.sum() >= min_pos]
    Y = Y[keep]
    res = (rep, Y.values.astype(np.float32), list(keep))
    out.write_bytes(pickle.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build"])
    ap.add_argument("--max-exp", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download_peaks(max_exp=a.max_exp)
    else:
        for c in CELLS:
            df, Y, names = build_binding_table(c)
            print(c, "expressed lncRNAs:", len(df), "| RBPs with >=15 positives:", len(names),
                  "| median positives per RBP:", int(np.median(Y.sum(0))), "| genes with >=1 RBP:", int((Y.sum(1) > 0).sum()))


if __name__ == "__main__":
    main()
