"""External localization check: literature-curated labels (e.g. RNALocate) + outputs of a published predictor, on OUR split.

    # 1. you download the RNALocate human table in a browser (the site did not answer from the sandbox)
    python -m lncfm.locext --labels RNALocate_human.tsv --id-col "RNA Symbol" --loc-col "Subcellular Localization" \
           --providers kmer rinalmo-micro --predictions published_tool.tsv

What it does
  1. Maps the file's lncRNAs to GENCODE v50 lncRNA genes (by Ensembl id or gene symbol) and builds nucleus-vs-cytoplasm labels
     (items with both, or any other class, are dropped; the class strings it accepts are listed in NUC / CYTO below).
  2. LABEL CROSS-CHECK: how well does our ENCODE fractionation index (the label used by `loc_clf`) predict this independent
     literature label? A low value means one of the two labels is not measuring what we think. This number is reported first.
  3. Trains the same frozen-feature probes with the benchmark's chromosome folds and reports AUROC with bootstrap CIs.
  4. `--predictions`: a two-column file (gene id or symbol, score; higher = nuclear) written by a published predictor
     (iLoc-lncRNA, lncLocator, DeepLncLoc, GraphLncLoc, Loc4Lnc, ... run by you, locally or through its web server) is scored on exactly the
     same items, so the published tool and our models are compared on one set. Published tools were trained on their own data;
     overlap with the test items cannot be checked here, so a published tool that scores very high should be read with that in mind.

STATUS: tested only on a synthetic file built from our own labels (mechanics), NOT on the real RNALocate table. The column names and
class strings are parameters because I could not inspect the real file.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import bench, evalkit, gencode, splits, tasks as T

NUC = {"nucleus", "nuclear", "nucleoplasm", "chromatin", "nucleolus", "nuclear speckle", "nuclear speckles", "paraspeckle"}
CYTO = {"cytoplasm", "cytoplasmic", "cytosol", "ribosome", "exosome", "mitochondrion", "endoplasmic reticulum"}


def load_labels(path, id_col, loc_col, species_col=None, human="Homo sapiens"):
    t = pd.read_csv(path, sep=None, engine="python")
    for c in (id_col, loc_col):
        if c not in t.columns:
            raise SystemExit(f"column {c!r} not in {list(t.columns)}; pass --id-col / --loc-col")
    if species_col and species_col in t.columns:
        t = t[t[species_col].astype(str).str.contains(human, case=False)]
    t = t.assign(_id=t[id_col].astype(str).str.strip(), _loc=t[loc_col].astype(str).str.lower().str.strip())
    t["_cls"] = np.where(t._loc.isin(NUC), 0, np.where(t._loc.isin(CYTO), 1, -1))     # 0 nuclear, 1 cytoplasmic
    g = t[t._cls >= 0].groupby("_id")._cls.agg(["min", "max"])
    ok = g[g["min"] == g["max"]]                                                     # drop RNAs reported in both
    return pd.Series(ok["min"].values, index=ok.index, name="y")


def map_to_genes(lab: pd.Series) -> pd.DataFrame:
    rep = splits.rep_genes()
    key = {}
    for col in ("gene_id_nov", "gene_name"):
        for k, i in zip(rep[col], rep.index):
            key.setdefault(str(k), i)
    def find(i):
        return key.get(i, key.get(i.split(".")[0]))
    hit = [(find(i), y) for i, y in lab.items() if find(i) is not None]
    df = rep.loc[[h[0] for h in hit]].copy()
    df["y_ext"] = [h[1] for h in hit]
    return df.drop_duplicates("gene_id").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--loc-col", required=True)
    ap.add_argument("--species-col", default=None)
    ap.add_argument("--providers", nargs="+", default=["lengc", "kmer"])
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--trunc", type=int, default=None)
    a = ap.parse_args(argv)

    df = map_to_genes(load_labels(a.labels, a.id_col, a.loc_col, a.species_col))
    print(f"{len(df)} lncRNAs with a one-sided literature localization label ({int((df.y_ext == 0).sum())} nuclear, {int((df.y_ext == 1).sum())} cytoplasmic)")
    df = splits.attach(df)
    enc = T.encode_cnrci().set_index("gene_id_nov").cnrci
    df["encode"] = df.gene_id_nov.map(enc)
    m = df.encode.notna()
    if m.sum() > 20 and df.y_ext[m].nunique() == 2:
        print(f"[label cross-check] ENCODE fractionation index vs literature label: AUROC {roc_auc_score(df.y_ext[m], df.encode[m]):.3f} on {int(m.sum())} lncRNAs "
              f"(0.5 = unrelated; the two labels should agree well if both measure localization)")
    y = df.y_ext.values.astype(float)
    for pv in a.providers:
        pn, prov = bench.make_provider(pv, a.trunc)
        X = bench.cached_embed(pn, prov, df, "locext")
        p = evalkit.dev_oof(X, y, df.fold.values, "clf")
        ok = (df.fold.values >= 0) & ~np.isnan(p)
        r = evalkit.summarize_single(y[ok], p[ok], "clf")
        print(f"  {pn:<22s} AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}] n={r['n']}")
    if a.predictions:
        pr = pd.read_csv(a.predictions, sep=None, engine="python")
        pr.columns = ["id", "score"] + list(pr.columns[2:])
        s = {str(i): v for i, v in zip(pr.id, pr.score)}
        sc = np.array([s.get(g, s.get(n, np.nan)) for g, n in zip(df.gene_id_nov, df.gene_name)], float)
        ok = ~np.isnan(sc)
        if ok.sum() > 20:
            r = evalkit.summarize_single(1 - y[ok], sc[ok], "clf")                     # predictions: higher = more NUCLEAR
            print(f"  published predictor  AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}] on {int(ok.sum())} of {len(df)} items "
                  f"(scores are taken as 'higher = more nuclear'; an AUROC far below 0.5 means your file is oriented the other way)")
        else:
            print("  published predictor: fewer than 20 items matched; check the id column")


if __name__ == "__main__":
    main()
