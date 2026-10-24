"""Clinical variant-effect check on small ncRNA disease genes (ClinVar), scored zero-shot.

Why this shape: ClinVar variants inside *lncRNA* exons are almost all (a) overlapping protein-coding
exons (TTN-AS1, PRPF31-AS1, ...) or (b) gene-confounded (50 pathogenic in TERC vs 700 benign in
KCNQ1OT1). A pooled AUROC would measure gene identity. The only defensible design is
WITHIN-GENE discrimination (pathogenic vs benign in the same gene), which needs genes that have both
classes: RMRP, RNU4ATAC, RNU4-2, SNORD118, RNU12, RNU7-1, TERC ... (~hundreds of variants in total).

Score = log p(alt) - log p(ref) at the masked variant position (masked-LM log-likelihood ratio);
lower = more damaging. Metric = size-weighted mean of per-gene AUROC (-score vs pathogenic).
"""
from __future__ import annotations

import gzip
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import download, gencode, paths
from .model import MASK, NT0, encode

GENES = ["RMRP", "RNU4ATAC", "RNU4-2", "RNU12", "RNU7-1", "SNORD118", "TERC", "RNU2-2"]
PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BEN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
_COMP = str.maketrans("ACGT", "TGCA")


def build_variant_table(genes=GENES) -> pd.DataFrame:
    out = paths.ensure(paths.DATA / "processed") / "ncrna_variants.pkl"
    if out.exists():
        return pd.read_pickle(out)
    tx_exons, tx_meta = pickle.loads((paths.DATA / "processed" / "human_gtf_exons.pkl").read_bytes())
    want = {t: m for t, m in tx_meta.items() if m["gene_name"] in genes}
    fa = paths.human(f"gencode.v{paths.GENCODE_HUMAN_RELEASE}.transcripts.fa.gz")
    seqs = {}
    for hdr, seq in gencode.read_fasta_gz(fa):
        tid = hdr.split("|")[0]
        if tid in want:
            seqs[tid] = seq.upper()
    by_gene = {}
    for t, m in want.items():
        if t in seqs:
            by_gene.setdefault(m["gene_name"], []).append(t)
    rows, drop_ref = [], 0
    with gzip.open(paths.DATA / "clinvar" / "clinvar.vcf.gz", "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            if len(f[3]) != 1 or len(f[4]) != 1:
                continue
            mm = re.search(r"CLNSIG=([^;]+)", f[7])
            if not mm or (mm.group(1) not in PATH and mm.group(1) not in BEN):
                continue
            chrom, pos = "chr" + f[0], int(f[1])
            for g, tids in by_gene.items():
                done = False
                for tid in sorted(tids, key=lambda t: -len(seqs[t])):
                    m = tx_meta[tid]
                    if m["chrom"] != chrom:
                        continue
                    ex = sorted(tx_exons[tid], reverse=(m["strand"] == "-"))
                    cum = 0
                    for s, e in ex:
                        if s <= pos <= e:
                            off = cum + ((pos - s) if m["strand"] == "+" else (e - pos))
                            ref, alt = f[3], f[4]
                            if m["strand"] == "-":
                                ref, alt = ref.translate(_COMP), alt.translate(_COMP)
                            if seqs[tid][off] != ref:
                                drop_ref += 1
                            else:
                                rows.append({"gene": g, "tx_id": tid, "offset": off, "ref": ref, "alt": alt,
                                             "label": int(mm.group(1) in PATH), "clnsig": mm.group(1)})
                            done = True
                            break
                        cum += e - s + 1
                    if done:
                        break
    df = pd.DataFrame(rows).drop_duplicates(["tx_id", "offset", "alt", "label"])
    df.attrs["dropped_ref_mismatch"] = drop_ref
    df["seq"] = [seqs[t].replace("T", "U") for t in df.tx_id]
    df.to_pickle(out)
    return df


@torch.no_grad()
def mlm_llr(model, device, df: pd.DataFrame, window: int = 512) -> np.ndarray:
    """log p(alt) - log p(ref) at the masked variant position, window centred on the variant."""
    model.eval()
    out = np.zeros(len(df))
    s = model.cfg.stride
    for i, r in enumerate(df.itertuples()):
        ids = encode(r.seq).astype(np.int64)
        a = max(0, min(r.offset - window // 2, len(ids) - window)) if len(ids) > window else 0
        w = ids[a:a + window].copy()
        p = r.offset - a
        w[p] = MASK
        L = int(np.ceil(len(w) / s) * s)
        x = np.zeros((1, L), np.int64)
        x[0, :len(w)] = w
        lg = model(torch.from_numpy(x).to(device))["nt_logits"][0, p, NT0:NT0 + 4]
        lp = F.log_softmax(lg.float(), -1).cpu().numpy()
        out[i] = lp["ACGU".index(r.alt.replace("T", "U"))] - lp["ACGU".index(r.ref.replace("T", "U"))]
    return out


def within_gene_auroc(df: pd.DataFrame, score: np.ndarray, min_each: int = 3):
    """Size-weighted mean of per-gene AUROC for (-score) vs pathogenic; also returns per-gene table."""
    rows = []
    for g, d in df.assign(score=score).groupby("gene"):
        nP, nB = int(d.label.sum()), int((1 - d.label).sum())
        if nP >= min_each and nB >= min_each:
            rows.append({"gene": g, "n_path": nP, "n_ben": nB, "auroc": roc_auc_score(d.label, -d.score)})
    t = pd.DataFrame(rows)
    if t.empty:
        return float("nan"), t
    w = np.minimum(t.n_path, t.n_ben)
    return float((t.auroc * w).sum() / w.sum()), t
