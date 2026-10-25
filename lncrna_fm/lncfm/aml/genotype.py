"""AML genotype panel: which driver mutations can be read out of lncRNA expression alone?

    python -m lncfm.aml.genotype --out results/aml_genotype.json

For every gene mutated in >= `min_dev` BeatAML patients (with RNA-seq) and >= `min_ext` TCGA-LAML patients: CV AUROC in BeatAML
(5-fold stratified x repeats) for age only / lncRNA expression / mRNA expression, then fit on all of BeatAML and score
TCGA-LAML (external). Mutation = any non-silent coding variant in the gene at any time point recorded for that patient
(BeatAML mutation calls are patient-level across specimens; this is a limitation: the mutation may post-date the RNA-seq specimen).
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import data as D
from . import patient as P

warnings.filterwarnings("ignore")
GENES = {"NPM1": 4869, "FLT3": 2322, "DNMT3A": 1788, "IDH1": 3417, "IDH2": 3418, "TP53": 7157, "RUNX1": 861, "NRAS": 4893, "KRAS": 3845,
         "TET2": 54790, "CEBPA": 1050, "ASXL1": 171023, "SRSF2": 6427, "WT1": 7490, "PTPN11": 5781, "STAG2": 10735, "U2AF1": 7307,
         "BCOR": 54880, "EZH2": 2146, "SF3B1": 23451, "GATA2": 2624, "KIT": 3815, "CBL": 867, "SETBP1": 26040}
SILENT = "Silent|Intron|UTR|Flank|IGR|RNA"


def _mutated_patients(study: str, profile: str, prefix: str = ""):
    r = requests.post(f"{D.CBIO}/molecular-profiles/{profile}/mutations/fetch", params={"projection": "SUMMARY"},
                      json={"sampleListId": f"{study}_all"}, headers={"accept": "application/json", "Content-Type": "application/json"}, timeout=240)
    r.raise_for_status()
    m = pd.DataFrame(r.json())
    m = m[~m.mutationType.astype(str).str.contains(SILENT, regex=True)]
    m["pid"] = m.patientId.str.replace(prefix, "", regex=False)
    return {g: set(m[m.entrezGeneId == e].pid) for g, e in GENES.items()}


def labels():
    f = D.paths.ensure(D.paths.DATA / "aml") / "genotype_labels.pkl"
    if f.exists():
        return pd.read_pickle(f)
    out = {"beataml": _mutated_patients("aml_ohsu_2022", "aml_ohsu_2022_mutations", "aml_ohsu_2022_"),
           "tcga_laml": _mutated_patients("laml_tcga_pan_can_atlas_2018", "laml_tcga_pan_can_atlas_2018_mutations")}
    pd.to_pickle(out, f)
    return out


def run(min_dev=20, min_ext=8, reps=3, folds=5, verbose=True):
    L = labels()
    Xb, Cb, gtb, varb = P.load("beataml")
    Xe, Ce, gte, vare = P.load("tcga_laml")
    pb, pe = list(Xb.index), list(Xe.index)
    eset = set(Xe.columns); bpos = {g: i for i, g in enumerate(Xb.columns)}; epos = {g: i for i, g in enumerate(Xe.columns)}
    common = [g for g in Xb.columns if g in eset]
    ib, ie = [bpos[g] for g in common], [epos[g] for g in common]
    Xv = Xb.values.astype(np.float32); Xt = Xe.values[:, ie].astype(np.float32)
    rows = []
    if verbose:
        print(f"BeatAML n={len(pb)} | TCGA-LAML n={len(pe)}", flush=True)
    for gene in GENES:
        yb = np.array([1 if p in L["beataml"][gene] else 0 for p in pb]); ye = np.array([1 if p in L["tcga_laml"][gene] else 0 for p in pe])
        if yb.sum() < min_dev or (len(yb) - yb.sum()) < min_dev:
            continue
        r = {"gene": gene, "n_pos_beataml": int(yb.sum()), "n_pos_tcga": int(ye.sum())}
        for fs in ("age", "lnc", "mrna"):
            sc = []
            for rep in range(reps):
                for tr, te in StratifiedKFold(folds, shuffle=True, random_state=rep).split(Xv, yb):
                    A, B = P._design(fs, Xv[tr], Xv[te], Cb.iloc[tr], Cb.iloc[te], gtb, varb)
                    sc.append(roc_auc_score(yb[te], P._fit_predict_clf(A, yb[tr], B, rep)))
            r[f"cv_{fs}"] = float(np.mean(sc))
            if ye.sum() >= min_ext:
                A, B = P._design(fs, Xv[:, ib], Xt, Cb, Ce, gtb[ib], varb[ib])
                r[f"ext_{fs}"] = float(roc_auc_score(ye, P._fit_predict_clf(A, yb, B)))
        rows.append(r)
        if verbose:
            print("  {gene:<7s} BeatAML pos {n_pos_beataml:>3d} | CV AUROC age {cv_age:.3f} lncRNA {cv_lnc:.3f} mRNA {cv_mrna:.3f}".format(**r)
                  + (" | TCGA ({n_pos_tcga} pos) lncRNA {ext_lnc:.3f} mRNA {ext_mrna:.3f}".format(**r) if "ext_lnc" in r else ""), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dev", type=int, default=20)
    ap.add_argument("--min-ext", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = run(a.min_dev, a.min_ext, a.reps)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
