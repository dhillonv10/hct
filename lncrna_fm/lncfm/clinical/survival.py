"""Survival benchmark on TCGA: does lncRNA expression carry prognostic signal, and does
tying the model to lncRNA *sequence embeddings* help when patients are few?

Feature sets (all evaluated with identical repeated, event-stratified K-fold CV):
  clin          age (+ stage, gender when available)
  lnc_ridge     top-K variable lncRNAs, ridge-Cox with free per-gene coefficients
  mrna_ridge    same with protein-coding genes (control: is lncRNA as good as mRNA?)
  lnc_pca       PCA(d) of lncRNA expression (fit inside the training fold)
  lnc_tied:<E>  Z = X @ E : patient = expression-weighted sum of d-dim gene embeddings E
                (E = random | kmer | model name). Only d Cox parameters instead of K.
Every feature selection / PCA / scaler is fit inside the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(1)


def cox_fit(X, t, e, l2, iters=60):
    X = torch.tensor(X, dtype=torch.float32)
    order = torch.argsort(torch.tensor(-t, dtype=torch.float32))
    X, ev = X[order], torch.tensor(e, dtype=torch.float32)[order]
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = X @ w
        nll = -((eta - torch.logcumsumexp(eta, 0)) * ev).sum() / ev.sum().clamp(min=1) + l2 * (w ** 2).sum()
        nll.backward()
        return nll

    opt.step(closure)
    return w.detach().numpy()


def cidx(t, e, risk):
    return concordance_index(t, -risk, e)


def fit_predict(Xtr, ttr, etr, Xte, l2_grid=(1e-3, 1e-2, 1e-1, 1.0), seed=0):
    """Pick l2 by inner 3-fold C-index, refit on all train, return test risk."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_c = l2_grid[-1], -1
    for l2 in l2_grid:
        cs = []
        for a, b in skf.split(A, etr):
            if etr[a].sum() < 3 or etr[b].sum() < 2:
                continue
            w = cox_fit(A[a], ttr[a], etr[a], l2)
            cs.append(cidx(ttr[b], etr[b], A[b] @ w))
        c = np.mean(cs) if cs else -1
        if c > best_c:
            best, best_c = l2, c
    w = cox_fit(A, ttr, etr, best)
    return B @ w


def clin_matrix(C: pd.DataFrame):
    """Age (+ stage with a missing-indicator when >=50% of patients have it, + gender).
    A strong clinical baseline matters: a weak one makes any molecular feature set look useful."""
    X = pd.DataFrame({"age": C.age.fillna(C.age.median())})
    st = C.stage.map(lambda s: {"I": 1, "II": 2, "III": 3, "IV": 4}.get(str(s).replace("Stage ", "").rstrip("ABC "), np.nan))
    if st.notna().mean() >= 0.5:
        X["stage"] = st.fillna(st.median())
        X["stage_missing"] = st.isna().astype(float)
    if C.gender.nunique() > 1:
        X["male"] = (C.gender == "male").astype(float)
    return X.values.astype(np.float32)


def select_top_var(X, k):
    v = X.var(0)
    return np.argsort(-v)[:k]


def run_cohort(X: pd.DataFrame, C: pd.DataFrame, meta: pd.DataFrame, emb: dict, K=2000, d=64,
               n_rep=3, folds=5, n_train=None, seed=0, max_time_days=None, skip_base=False):
    """emb: {name: DataFrame(index=ENSG, columns=dims)} sequence embeddings for tied models."""
    t = C.os_time.values.astype(np.float64)
    e = C.os_event.values.astype(int)
    if max_time_days:
        e = np.where(t > max_time_days, 0, e)
        t = np.minimum(t, max_time_days)
    gtype = meta.reindex(X.columns).gene_type.fillna("")
    lnc_cols = np.where(gtype.values == "lncRNA")[0]
    pc_cols = np.where(gtype.values == "protein_coding")[0]
    Xv = X.values
    clin = clin_matrix(C)
    res = {}
    rng = np.random.default_rng(seed)
    for rep in range(n_rep):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(Xv, e):
            if n_train and len(tr) > n_train:
                tr = rng.choice(tr, n_train, replace=False)
            if e[tr].sum() < 8:
                continue
            sets = {} if skip_base else {"clin": (clin[tr], clin[te])}
            lt = lnc_cols[select_top_var(Xv[np.ix_(tr, lnc_cols)], K)]
            pt = pc_cols[select_top_var(Xv[np.ix_(tr, pc_cols)], K)]
            if not skip_base:
                sets["lnc_ridge"] = (Xv[np.ix_(tr, lt)], Xv[np.ix_(te, lt)])
                sets["mrna_ridge"] = (Xv[np.ix_(tr, pt)], Xv[np.ix_(te, pt)])
            Atr = Xv[np.ix_(tr, lt)]
            mu, sd = Atr.mean(0), Atr.std(0) + 1e-6
            pca = PCA(min(d, len(tr) - 1), random_state=0).fit((Atr - mu) / sd)
            if not skip_base:
                sets["lnc_pca"] = (pca.transform((Atr - mu) / sd), pca.transform((Xv[np.ix_(te, lt)] - mu) / sd))
            genes = X.columns[lt]
            for name, E in emb.items():
                M = E.reindex(genes)
                ok = M.notna().all(axis=1).values
                if ok.sum() < 50:
                    continue
                Em = M.values[ok].astype(np.float32)
                Em = (Em - Em.mean(0)) / (Em.std(0) + 1e-6)
                if Em.shape[1] > d:
                    Em = PCA(d, random_state=0).fit_transform(Em).astype(np.float32)
                Xs_tr = ((Atr - mu) / sd)[:, ok]
                Xs_te = ((Xv[np.ix_(te, lt)] - mu) / sd)[:, ok]
                sets[f"lnc_tied:{name}"] = (Xs_tr @ Em / np.sqrt(ok.sum()), Xs_te @ Em / np.sqrt(ok.sum()))
            for name, (A, B) in sets.items():
                risk = fit_predict(A, t[tr], e[tr], B, seed=seed + rep)
                res.setdefault(name, []).append(cidx(t[te], e[te], risk))
            if not skip_base:       # combined: clinical + lncRNA PCA
                A = np.hstack([sets["clin"][0], sets["lnc_pca"][0]]); B = np.hstack([sets["clin"][1], sets["lnc_pca"][1]])
                res.setdefault("clin+lnc_pca", []).append(cidx(t[te], e[te], fit_predict(A, t[tr], e[tr], B, seed=seed + rep)))
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))), len(v)) for k, v in res.items()}


# --------------------------------------------------------------------------- CLI
def gene_embeddings(name: str, genes=None, trunc=None, d_rand: int = 64, seed: int = 0) -> pd.DataFrame:
    """ENSG -> embedding of the gene's representative lncRNA transcript.
    name: random | kmer | lncfm:<ckpt> | <hf name>.  genes: optional ENSG subset (limits cost of slow models)."""
    from .. import bench, gencode
    rep = gencode.representative_per_gene(gencode.build_table("human", "lnc"))
    if genes is not None:
        rep = rep[rep.gene_id_nov.isin(set(genes))].reset_index(drop=True)
    if name == "random":
        rng = np.random.default_rng(seed)
        return pd.DataFrame(rng.standard_normal((len(rep), d_rand)).astype(np.float32), index=rep.gene_id_nov.values)
    pn, prov = bench.make_provider(name, trunc)
    Z = bench.cached_embed(pn, prov, rep, "genes")
    return pd.DataFrame(Z, index=rep.gene_id_nov.values)


def main(argv=None):
    import argparse, json
    from . import tcga
    from .. import paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+", required=True)
    ap.add_argument("--emb", nargs="*", default=["random", "kmer"])
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-train", type=int, nargs="*", default=[None])
    ap.add_argument("--emb-top", type=int, default=3000, help="embed only the top-N most variable lncRNAs per cohort")
    ap.add_argument("--trunc", type=int, default=None, help="clip sequences to first N nt for ALL embedders")
    ap.add_argument("--skip-base", action="store_true", help="only the tied-embedding variants (reuse base results)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    meta = pd.read_pickle(paths.DATA / "tcga" / "gene_meta.pkl")
    cohorts = {p: tcga.load(p) for p in a.projects}
    keep = set()
    for X, _ in cohorts.values():
        gt = meta.reindex(X.columns).gene_type.fillna("")
        v = X.loc[:, gt.values == "lncRNA"].var(0).sort_values(ascending=False)
        keep |= set(v.index[:a.emb_top])
    embs = {n: gene_embeddings(n, genes=keep, trunc=a.trunc) for n in a.emb}
    print(f"embedded {len(keep)} candidate lncRNAs per provider (before sequence-table overlap)", flush=True)
    allres = []
    for proj in a.projects:
        X, C = cohorts[proj]
        for nt in a.n_train:
            res = run_cohort(X, C, meta, embs, K=a.K, d=a.d, n_rep=a.reps, n_train=nt, skip_base=a.skip_base)
            print(f"\n{proj}  n={len(C)} events={int(C.os_event.sum())}  n_train={nt or 'all'}", flush=True)
            for k, (m, se, n) in sorted(res.items(), key=lambda kv: -kv[1][0]):
                print(f"   {k:<28s} C-index {m:.3f} ± {se:.3f}  (n_folds={n})", flush=True)
                allres.append({"project": proj, "n_train": nt, "features": k, "c_index": m, "se": se, "n_folds": n})
    if a.out:
        pd.DataFrame(allres).to_json(a.out, orient="records", indent=1)


if __name__ == "__main__":
    main()
