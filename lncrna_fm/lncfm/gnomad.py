"""Population variants (gnomAD) as an extra *benign* set for ncRNA genes.

ClinVar has many pathogenic variants but almost no benign ones for the genes that matter most (TERC: 30 P/LP,
2 B/LB; RNU12: 7 vs 2). A within-gene comparison needs both classes, so common variants are added as presumed
benign: a variant seen at allele frequency >= `af_min` in gnomAD v4 (nuclear) or v3.1 (mitochondrial) is very
unlikely to cause a severe rare disease. This is the standard BA1/BS1 logic in clinical variant interpretation.

`af_min=0` switches to the weaker "observed in gnomAD at any frequency" background (adds TERC, whose ClinVar benign set
is only 2 variants); most of those variants are singletons, so it is a secondary analysis, never the headline.

What this is NOT: "absent from gnomAD" is never used as a label, and a common variant is only presumed benign for
severe early-onset phenotypes. Evaluations that use these variants say so (`source` column) and ClinVar-only
numbers are always reported next to the combined ones.

Network: gnomAD's public GraphQL endpoint, one small region query per gene, cached under DATA/processed/gnomad/.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from . import ncvar, paths

API = "https://gnomad.broadinstitute.org/api"
_Q_NUC = """query($chrom:String!,$start:Int!,$stop:Int!){ region(chrom:$chrom,start:$start,stop:$stop,reference_genome:GRCh38){
  variants(dataset: gnomad_r4){ pos ref alt exome{ ac an } genome{ ac an } } } }"""
_Q_MT = """query($start:Int!,$stop:Int!){ region(chrom:"M",start:$start,stop:$stop,reference_genome:GRCh38){
  mitochondrial_variants(dataset: gnomad_r3){ variant_id pos ac_hom ac_het an } } }"""


def _post(q, variables, tries=4):
    for i in range(tries):
        try:
            r = requests.post(API, json={"query": q, "variables": variables}, timeout=120)
            j = r.json()
            if "errors" in j and "data" not in j:
                raise RuntimeError(j["errors"][0]["message"])
            return j["data"]["region"]
        except Exception as e:                                        # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def region_snvs(chrom: str, start: int, stop: int) -> pd.DataFrame:
    """SNVs in a region with pooled allele counts. chrom like 'chr3' / 'chrM'. Columns: pos, ref, alt, ac, an, af."""
    cache = paths.ensure(paths.DATA / "processed" / "gnomad") / f"{chrom}_{start}_{stop}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        rows = []
        if chrom == "chrM":
            for v in _post(_Q_MT, {"start": start, "stop": stop})["mitochondrial_variants"]:
                _, _, ref, alt = v["variant_id"].split("-")                # "M-3243-A-G"
                if len(ref) == 1 and len(alt) == 1:
                    rows.append({"pos": v["pos"], "ref": ref, "alt": alt, "ac": (v["ac_hom"] or 0) + (v["ac_het"] or 0),
                                 "an": v["an"] or 0})
        else:
            for v in _post(_Q_NUC, {"chrom": chrom.replace("chr", ""), "start": start, "stop": stop})["variants"]:
                if len(v["ref"]) != 1 or len(v["alt"]) != 1:
                    continue
                ac = sum((v[k] or {}).get("ac", 0) for k in ("exome", "genome"))
                an = sum((v[k] or {}).get("an", 0) for k in ("exome", "genome"))
                rows.append({"pos": v["pos"], "ref": v["ref"], "alt": v["alt"], "ac": ac, "an": an})
        cache.write_text(json.dumps(rows))
    df = pd.DataFrame(rows, columns=["pos", "ref", "alt", "ac", "an"])
    df["af"] = df.ac / df.an.clip(lower=1)
    return df


def common_benign(gene_ids, af_min=1e-4, min_an=20000) -> pd.DataFrame:
    """Common SNVs inside the exons of the given GENCODE gene ids, in the evaluation-table schema (label 0)."""
    tx_exons, tx_meta = ncvar.load_annotation()
    parts = []
    for gid in gene_ids:
        tids = [t for t, m in tx_meta.items() if m["gene_id"] == gid]
        if not tids:
            continue
        m0 = tx_meta[tids[0]]
        lo = min(tx_meta[t]["tx_start"] for t in tids)
        hi = max(tx_meta[t]["tx_end"] for t in tids)
        v = region_snvs(m0["chrom"], lo, hi)
        v = v[(v.ac >= 1) & (v.af >= af_min) & (v.an >= min_an)]          # af_min=0: every observed variant
        if v.empty:
            continue
        d = pd.DataFrame({"chrom": m0["chrom"], "pos": v.pos.values, "ref": v.ref.values, "alt": v.alt.values,
                          "label": 0, "clnsig": "gnomad_common", "stars": -1, "disease": "", "clinvar_id": "",
                          "geneinfo": m0["gene_name"] if not m0["gene_name"].startswith("ENSG") else "", "af": v.af.values})
        mapped, _ = ncvar.map_to_transcripts(d, attributed_only=True)
        if len(mapped):
            parts.append(mapped[mapped.gene_id == gid])
    if not parts:
        return pd.DataFrame()
    return ncvar.attach_sequences(pd.concat(parts, ignore_index=True))


def build_eval_table(min_stars=0, af_min=1e-4, min_path=3, use_gnomad=True, refresh=False) -> pd.DataFrame:
    """ClinVar ncRNA SNVs + (optionally) common gnomAD variants for genes with >= `min_path` pathogenic variants."""
    out = paths.ensure(paths.DATA / "processed") / f"ncvar_eval_v3_s{min_stars}_af{af_min:g}_{int(use_gnomad)}.pkl"
    if out.exists() and not refresh:
        return pd.read_pickle(out)
    cv = ncvar.build_clinvar_table(min_stars).copy()
    cv["source"], cv["af"] = "clinvar", float("nan")
    if use_gnomad:
        g = cv.groupby("gene_id").label.sum()
        panel = list(g[g >= min_path].index)
        gn = common_benign(panel, af_min)
        if len(gn):
            gn["source"] = "gnomad"
            key = set(zip(cv.chrom, cv.pos, cv.alt))                       # never relabel a ClinVar-classified variant
            gn = gn[[(c, p, a) not in key for c, p, a in zip(gn.chrom, gn.pos, gn.alt)]]
            cv = pd.concat([cv, gn], ignore_index=True)
    cv = cv.drop_duplicates(["tx_id", "offset", "alt_t", "label"]).reset_index(drop=True)
    cv.to_pickle(out)
    return cv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--af-min", type=float, default=1e-4)
    ap.add_argument("--min-stars", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    d = build_eval_table(a.min_stars, a.af_min, refresh=a.refresh)
    t = d.groupby(["gene", "gene_type", "source"]).label.agg(n="size", n_path="sum").reset_index()
    print(t.pivot_table(index=["gene", "gene_type"], columns="source", values="n", fill_value=0).join(
        d.groupby("gene").label.sum().rename("P/LP")).sort_values("P/LP", ascending=False).head(25).to_string())
