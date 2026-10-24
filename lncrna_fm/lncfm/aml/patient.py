"""Layer 2a: patient-level AML tasks from lncRNA expression, with external validation.

    python -m lncfm.aml.patient --tasks npm1 flt3_itd eln_adverse induction_cr os --out results/aml_patient.json
    python -m lncfm.aml.patient --tasks os --emb random kmer rinalmo-micro     # + sequence-tied gene-embedding models

Development cohort : BeatAML (adult, initial-diagnosis specimen). 5-fold stratified CV x repeats.
External cohorts   : TCGA-LAML (adult)  -> npm1, os
                     TARGET-AML (paediatric) -> os, induction_cr   (a shifted population: tests robustness, not "the same disease")
Every gene is z-scored within its own cohort (label-free) before fitting, to blunt batch effects. Top-K variable
genes are chosen on the training data only. Feature sets:
    age                clinical covariate (available in all cohorts)
    age+eln            age + ELN-2017 risk (BeatAML OS/internal only: the strong clinical baseline)
    lnc / mrna         ridge model on top-K variable lncRNAs / protein-coding genes
    lnc+age, mrna+age, lnc+age+eln
    lnc_tied:<emb>     patient = expression-weighted sum of d-dim lncRNA sequence embeddings (64 parameters)
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .. import evalkit
from ..clinical import survival as SV
from . import data as D

warnings.filterwarnings("ignore")
K_GENES, CLIN_SCALE = 1000, 5.0          # clinical columns are up-weighted so ridge does not shrink them away


def load(cohort):
    X, S = D.load_cohort(cohort)
    meta = pd.read_pickle(D.paths.DATA / "aml" / "gene_meta.pkl")
    C = {"beataml": D.beataml_clinical, "tcga_laml": D.tcga_laml_clinical, "target_aml": D.target_clinical}[cohort]()
    C = C.reindex(S.case_id.values)
    X = X.copy(); X.index = S.case_id.values
    var = X.var(0).values                                      # raw log2-TPM variance: used ONLY to pick the top-K genes
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)                   # per-cohort gene z-score (label-free), used as features
    gt = meta.reindex(X.columns).gene_type.fillna("").values
    return X, C, gt, var


def _eln_onehot(C):
    e = C.eln2017.astype(str) if "eln2017" in C else pd.Series("NA", index=C.index)
    return np.stack([(e == "Favorable"), (e == "Intermediate") | (e == "FavorableOrIntermediate"), (e == "Adverse") | (e == "IntermediateOrAdverse"),
                     ~e.isin(["Favorable", "Intermediate", "Adverse", "FavorableOrIntermediate", "IntermediateOrAdverse"])], 1).astype(np.float32)


def _clin(C, with_eln):
    age = C.age.fillna(C.age.median()).values[:, None].astype(np.float32)
    return np.hstack([age, _eln_onehot(C)]) if with_eln else age


def _topk(var, cols, k):
    """Top-k most variable genes by RAW expression variance (unsupervised, label-free). Selecting on z-scored data would be
    random: every z-scored gene has variance 1."""
    return cols[np.argsort(-var[cols])[:k]]


def _fit_predict_clf(A, y, B, seed=0):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A, B = (A - mu) / sd, (B - mu) / sd
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, ba = 1e-3, -1
    for C in (1e-4, 1e-3, 1e-2, 1e-1):
        s = []
        for a, b in skf.split(A, y):
            if y[b].min() == y[b].max():
                continue
            s.append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(A[a], y[a]).decision_function(A[b])))
        if s and np.mean(s) > ba:
            best, ba = C, float(np.mean(s))
    return LogisticRegression(C=best, max_iter=2000, class_weight="balanced").fit(A, y).decision_function(B)


def _design(fs, Xtr, Xte, Ctr, Cte, gt, var, emb=None, k=K_GENES, d=64):
    """Build train/test matrices for a feature-set name."""
    parts_tr, parts_te = [], []
    base = fs.split("+")
    if "age" in base or "eln" in base:
        with_eln = "eln" in base
        a, b = _clin(Ctr, with_eln), _clin(Cte, with_eln)
        mu, sd = a.mean(0), a.std(0) + 1e-6
        parts_tr.append((a - mu) / sd * CLIN_SCALE); parts_te.append((b - mu) / sd * CLIN_SCALE)
    for g in base:
        if g in ("lnc", "mrna"):
            cols = np.where(gt == ("lncRNA" if g == "lnc" else "protein_coding"))[0]
            sel = _topk(var, cols, k)
            parts_tr.append(Xtr[:, sel]); parts_te.append(Xte[:, sel])
        elif g.startswith("lnc_tied:"):
            E = emb[g.split(":", 1)[1]]
            cols = np.where(gt == "lncRNA")[0]
            sel = _topk(var, cols, k * 2)
            genes = emb["_genes"][sel]
            M = E.reindex(genes)
            ok = M.notna().all(axis=1).values
            Em = M.values[ok].astype(np.float32)
            Em = (Em - Em.mean(0)) / (Em.std(0) + 1e-6)
            if Em.shape[1] > d:
                Em = PCA(d, random_state=0).fit_transform(Em).astype(np.float32)
            parts_tr.append(Xtr[:, sel][:, ok] @ Em / np.sqrt(ok.sum())); parts_te.append(Xte[:, sel][:, ok] @ Em / np.sqrt(ok.sum()))
    return np.hstack(parts_tr), np.hstack(parts_te)


def _score(task, kind, y_or_te, risk, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "clf":
        y = y_or_te
        v = roc_auc_score(y, risk)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                bs.append(roc_auc_score(y[i], risk[i]))
        return {"metric": "AUROC", "value": float(v), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    t, e = y_or_te
    v = concordance_index(t, -risk, e)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(t), len(t))
        if e[i].sum() > 3:
            bs.append(concordance_index(t[i], -risk[i], e[i]))
    return {"metric": "C-index", "value": float(v), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


# (kind, label function on clinical table, external cohorts, feature sets)
TASKS = {
    "npm1":         ("clf", lambda C: C.npm1, ["tcga_laml"], ["age", "lnc", "mrna", "lnc+age"]),
    "flt3_itd":     ("clf", lambda C: C.flt3_itd, [], ["age", "lnc", "mrna", "lnc+age"]),
    "eln_adverse":  ("clf", lambda C: C.eln2017.map({"Adverse": 1.0, "Favorable": 0.0}), [], ["age", "lnc", "mrna", "lnc+age"]),
    "induction_cr": ("clf", lambda C: C.induction_cr, ["target_aml"], ["age", "lnc", "mrna", "lnc+age"]),
    "os":           ("surv", None, ["tcga_laml", "target_aml"], ["age", "age+eln", "lnc", "mrna", "lnc+age", "mrna+age", "lnc+age+eln"]),
}


def run_task(name, embs=None, reps=3, folds=5, n_boot=300, verbose=True):
    kind, lab, ext, sets = TASKS[name]
    sets = list(sets) + [f"lnc_tied:{e}" for e in (embs or {}) if not e.startswith("_")]
    Xb, Cb, gtb, varb = load("beataml")
    if kind == "clf":
        y_all = lab(Cb).values.astype(float)
        keep = ~np.isnan(y_all)
    else:
        y_all = None
        keep = (Cb.os_days.notna() & (Cb.os_days > 0) & Cb.os_event.notna()).values
    Xv = Xb.values[keep].astype(np.float32); Cv = Cb[keep].reset_index(drop=True)
    if kind == "clf":
        y = y_all[keep].astype(int)
    else:
        t, e = Cv.os_days.values.astype(float), Cv.os_event.values.astype(int)
    rows = []
    n = len(Cv)
    info = f"n={n}" + (f", positives={int(y.sum())}" if kind == "clf" else f", deaths={int(e.sum())}")
    if verbose:
        print(f"\n[2a:{name}] BeatAML {info}; external: {ext or 'none'}", flush=True)
    for fs in sets:
        if "eln" in fs.split("+") and name != "os":
            continue
        oof_scores = []
        for rep in range(reps):
            skf = StratifiedKFold(folds, shuffle=True, random_state=rep)
            strat = y if kind == "clf" else e
            for tr, te in skf.split(Xv, strat):
                A, B = _design(fs, Xv[tr], Xv[te], Cv.iloc[tr], Cv.iloc[te], gtb, varb, embs)
                if kind == "clf":
                    risk = _fit_predict_clf(A, y[tr], B, rep)
                    oof_scores.append(roc_auc_score(y[te], risk))
                else:
                    risk = SV.fit_predict(A, t[tr], e[tr], B, seed=rep)
                    oof_scores.append(SV.cidx(t[te], e[te], risk))
        m, se = float(np.mean(oof_scores)), float(np.std(oof_scores) / np.sqrt(len(oof_scores)))
        rows.append({"task": name, "features": fs, "cohort": "beataml_cv", "metric": "AUROC" if kind == "clf" else "C-index",
                     "value": m, "se": se, "n": n})
        line = f"  {fs:<26s} BeatAML CV {rows[-1]['metric']} {m:.3f} +- {se:.3f}"
        # external validation: fit on ALL BeatAML, evaluate on each external cohort
        for ec in ext:
            if fs.startswith("lnc_tied") and False:
                continue
            Xe, Ce, gte, vare = load(ec)
            eset = set(Xe.columns)
            common = [g for g in Xb.columns if g in eset]
            bpos = {g: i for i, g in enumerate(Xb.columns)}; epos = {g: i for i, g in enumerate(Xe.columns)}
            ib = [bpos[g] for g in common]; ie = [epos[g] for g in common]
            Xtr_c, Xte_c = Xv[:, ib], Xe.values[:, ie].astype(np.float32)
            gt_c = gtb[ib]
            if "eln" in fs.split("+"):
                continue                                    # ELN is not recorded in the external cohorts
            if kind == "clf":
                if name not in Ce.columns:
                    continue
                ye = Ce[name].values.astype(float)
                ok = ~np.isnan(ye)
                if ok.sum() < 20 or len(set(ye[ok])) < 2:
                    continue
                A, B = _design(fs, Xtr_c, Xte_c[ok], Cv, Ce[ok], gt_c, varb[ib], _sub_emb(embs, common))
                risk = _fit_predict_clf(A, y, B)
                r = _score(name, "clf", ye[ok].astype(int), risk, n_boot)
                nn = int(ok.sum())
            else:
                ok = (Ce.os_days.notna() & (Ce.os_days > 0) & Ce.os_event.notna()).values
                if ok.sum() < 20:
                    continue
                A, B = _design(fs, Xtr_c, Xte_c[ok], Cv, Ce[ok], gt_c, varb[ib], _sub_emb(embs, common))
                risk = SV.fit_predict(A, t, e, B)
                r = _score(name, "surv", (Ce.os_days.values[ok].astype(float), Ce.os_event.values[ok].astype(int)), risk, n_boot)
                nn = int(ok.sum())
            rows.append({"task": name, "features": fs, "cohort": ec, "n": nn, **r})
            line += f" | {ec} {r['value']:.3f} [{r['lo']:.2f},{r['hi']:.2f}] n={nn}"
        if verbose:
            print(line, flush=True)
    return rows


def _sub_emb(embs, common):
    if not embs:
        return embs
    return {**{k: v for k, v in embs.items() if k != "_genes"}, "_genes": np.array(common)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--emb", nargs="*", default=[], help="gene-embedding providers for the sequence-tied models")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    embs = None
    if a.emb:
        Xb, _, _, _ = load("beataml")
        embs = {n: SV.gene_embeddings(n, genes=set(Xb.columns)) for n in a.emb}
        embs["_genes"] = np.array(Xb.columns)
    allrows = []
    for t in a.tasks:
        allrows += run_task(t, embs, reps=a.reps)
    if a.out:
        json.dump(allrows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
