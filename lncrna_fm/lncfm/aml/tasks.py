"""Layer 2b: gene-level AML tasks. The unit is a lncRNA gene; the question is whether its *sequence* predicts
whether it is leukaemia-associated. Labels come from patient RNA-seq (so they are real measurements), and
label reliability is reported as the split-half correlation over patients.

  aml_vs_normal_reg          raw log2 fold change AML (TARGET marrow) vs normal marrow  -- confounded with expression level
  aml_vs_normal_resid_reg    the same after removing what mean expression + transcript length explain (cubic fit; uses no sequence)
  aml_vs_normal_resid_clf    top vs bottom tercile of that residual
  aml_npm1_reg               NPM1-mutant vs wild-type AML: Stouffer-combined Welch t over BeatAML + TCGA-LAML. Reliability is
                             the BeatAML-vs-TCGA agreement (independent cohorts): Spearman 0.58 on 4.3k lncRNAs; the strongest
                             hits (HOXB-AS3, HOXA-AS2/3, HOXB-AS2, BAALC-AS1) are the known NPM1-mutant HOX-antisense lncRNAs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, ttest_ind

from . import data as D


def _lnc_columns(X: pd.DataFrame, rep: pd.DataFrame):
    meta = pd.read_pickle(D.paths.DATA / "aml" / "gene_meta.pkl")
    is_lnc = meta.reindex(X.columns).gene_type.fillna("").values == "lncRNA"
    known = set(rep.gene_id_nov)
    return [g for g, ok in zip(X.columns, is_lnc) if ok and g in known]


def _reliability(M1: np.ndarray, M2: np.ndarray, f):
    """Split-half Spearman of a per-gene statistic f(matrix, mask-free) computed on two independent halves."""
    return float(spearmanr(f(M1), f(M2)).statistic)


def _auc_cols(Xa, Xb):
    """Per-gene AUROC (group a vs b) via rank-sum, vectorised over columns."""
    n1, n0 = len(Xa), len(Xb)
    R = rankdata(np.vstack([Xa, Xb]), axis=0)
    return (R[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_gene_task(name: str, splits):
    from ..lncbench import Task
    rep = splits.rep_genes()
    rng = np.random.default_rng(0)
    if name.startswith("aml_vs_normal"):
        resid = "_resid_" in name
        Xa, _ = D.load_cohort("target_aml"); Xn, _ = D.load_cohort("target_normal")
        genes = [g for g in _lnc_columns(Xa, rep) if g in Xn.columns]
        A, N = Xa[genes].values.astype(np.float64), Xn[genes].values.astype(np.float64)
        ma, mn = A.mean(0), N.mean(0)
        ok = np.maximum(ma, mn) >= 1.0                               # expressed: mean log2(TPM+1) >= 1 in either group
        genes = [g for g, k in zip(genes, ok) if k]
        A, N, ma, mn = A[:, ok], N[:, ok], ma[ok], mn[ok]
        lfc = ma - mn
        auc = _auc_cols(A, N)
        ia, in_ = rng.permutation(len(A)), rng.permutation(len(N))
        h = lambda idx, M: np.array_split(idx, 2)
        a1, a2 = np.array_split(ia, 2); n1, n2 = np.array_split(in_, 2)
        rel = float(spearmanr(A[a1].mean(0) - N[n1].mean(0), A[a2].mean(0) - N[n2].mean(0)).statistic)
        lab = pd.DataFrame({"gene_id_nov": genes, "y_lfc": lfc, "auc": auc, "mean_expr": (ma + mn) / 2})
        meta = {"split_half_rho": round(rel, 3), "n_aml": len(A), "n_normal": len(N)}
        ycol = "y_lfc"
    elif name == "aml_npm1_reg":
        Xb, Sb = D.load_cohort("beataml"); Xt, St = D.load_cohort("tcga_laml")
        Cb = D.beataml_clinical().reindex(Sb.case_id.values); Ct = D.tcga_laml_clinical().reindex(St.case_id.values)
        tset = set(Xt.columns)
        genes = [g for g in _lnc_columns(Xb, rep) if g in tset]

        def tstat(X, npm1):
            k = ~np.isnan(npm1)
            M = X[genes].values[k].astype(np.float64); g = npm1[k] == 1
            return ttest_ind(M[g], M[~g], equal_var=False).statistic, M.mean(0)
        tb, mb = tstat(Xb, Cb.npm1.values); tt, mt = tstat(Xt, Ct.npm1.values)
        ok = (mb >= 0.5) & (mt >= 0.5) & np.isfinite(tb) & np.isfinite(tt)
        rel = float(spearmanr(tb[ok], tt[ok]).statistic)
        lab = pd.DataFrame({"gene_id_nov": np.array(genes)[ok], "y_t": ((tb + tt) / np.sqrt(2))[ok], "mean_expr": ((mb + mt) / 2)[ok]})
        ycol, meta = "y_t", {"cross_cohort_rho": round(rel, 3), "n_genes": int(ok.sum())}
    else:
        raise KeyError(name)
    df = rep.merge(lab, on="gene_id_nov", how="inner")
    df["loglen"] = np.log10(df.length.values)
    if name.startswith("aml_vs_normal") and resid:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        Z = PolynomialFeatures(3).fit_transform(df[["mean_expr", "loglen"]].values)
        df["y_resid"] = df.y_lfc.values - LinearRegression().fit(Z, df.y_lfc.values).predict(Z)
        lo, hi = df.y_resid.quantile([1 / 3, 2 / 3])
        df["y_cls"] = np.where(df.y_resid <= lo, 0, np.where(df.y_resid >= hi, 1, -1))
        ycol = "y_resid" if name.endswith("_reg") else "y_cls"
    elif name.startswith("aml_vs_normal"):
        ycol = "y_lfc"
    df = splits.attach(df)
    kind = "reg" if name.endswith("_reg") else "clf"
    if kind == "clf":
        df = df[df[ycol] >= 0].reset_index(drop=True)
    return Task(name, "2b", kind, df, df[ycol].values.astype(float), cov_cols=["mean_expr", "loglen"], meta=meta)
