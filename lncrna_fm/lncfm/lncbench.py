"""LncBench v2: two-layer benchmark for lncRNA models.

LAYER 1  molecular function (public lab measurements)
    loc_reg / loc_clf     nuclear-vs-cytosolic localization (ENCODE fractionation, 15 cell lines)
    tau_reg / testis_clf / brain_clf   tissue specificity (GTEx v8)
    rbp_K562 / rbp_HepG2  RNA-binding-protein targets (ENCODE eCLIP), expression-controlled, multi-label
    coding_clf            sanity tier only
LAYER 2  clinical: acute myeloid leukaemia  (see lncfm/aml/)
    2b gene level  : aml_vs_normal_reg/clf, aml_npm1_reg   -- which lncRNAs are leukaemia / NPM1-associated?
    2a patient level: python -m lncfm.aml.patient            -- genotype, ELN risk, induction response, survival, external validation

Protocol: dev = chromosome-group 5-fold CV on non-test chromosomes; locked test = chr8/13/18/21 (only with --final);
metrics on all items and on the homology-pruned subset; bootstrap CIs; paired comparisons between models.

    python -m lncfm.lncbench run --providers expr_len lengc kmer rinalmo-micro --tasks all --trunc 400 --max-n 1500
    python -m lncfm.lncbench run --providers kmer --tasks loc_reg --final
    python -m lncfm.lncbench compare results/x.json kmer rinalmo-micro@400
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bench, evalkit, splits, tasks as T


@dataclass
class Task:
    name: str
    layer: str
    kind: str                       # 'reg' | 'clf' | 'multi'
    df: pd.DataFrame
    y: np.ndarray
    cov_cols: list = field(default_factory=list)   # sequence-free covariates for the 'expr_len' baseline
    strata: object = None
    names: list | None = None
    meta: dict = field(default_factory=dict)


def _gene_task(name, builder, ycol, kind, layer="1"):
    df = splits.attach(builder())
    if kind == "clf":
        df = df[df[ycol] >= 0].reset_index(drop=True)
    df["loglen"] = np.log10(df.length.values)
    return Task(name, layer, kind, df, df[ycol].values.astype(float), cov_cols=["loglen"])


def _rbp_task(cell):
    from . import eclip
    rep, Y, names = eclip.build_binding_table(cell)
    df = splits.attach(rep.assign(log_tpm=np.log2(rep.tpm.values + 1), loglen=np.log10(rep.length.values)))
    Y = Y[df._row.values]
    e5 = pd.qcut(df.log_tpm, 5, labels=False, duplicates="drop").values
    l3 = pd.qcut(df.loglen, 3, labels=False, duplicates="drop").values
    strata = {"expr": e5, "expr_len": e5 * 3 + l3}      # binding depends on expression AND on transcript length
    return Task(f"rbp_{cell}", "1", "multi", df, Y, cov_cols=["log_tpm", "loglen"], strata=strata, names=names)


def load_task(name: str) -> Task:
    if name == "loc_reg":
        return _gene_task(name, T.task_localization, "y", "reg")
    if name == "loc_clf":
        return _gene_task(name, T.task_localization, "y_bin", "clf")
    if name == "tau_reg":
        return _gene_task(name, T.task_tissue, "y", "reg")
    if name == "testis_clf":
        return _gene_task(name, T.task_tissue, "y_testis", "clf")
    if name == "brain_clf":
        return _gene_task(name, T.task_tissue, "y_brain", "clf")
    if name == "coding_clf":
        return _gene_task(name, T.task_coding_vs_lnc, "y", "clf")
    if name.startswith("rbp_"):
        return _rbp_task(name.split("_", 1)[1])
    if name.startswith("aml_"):
        from .aml import tasks as AT
        return AT.load_gene_task(name, splits)
    raise KeyError(name)


LAYER1 = ["loc_reg", "loc_clf", "tau_reg", "testis_clf", "brain_clf", "rbp_K562", "rbp_HepG2", "coding_clf"]
LAYER2B = ["aml_vs_normal_reg", "aml_vs_normal_resid_reg", "aml_vs_normal_resid_clf", "aml_npm1_reg"]


def subsample(task: Task, max_n: int | None, seed=0) -> Task:
    if not max_n or len(task.df) <= max_n:
        return task
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(task.df), max_n, replace=False))
    return Task(task.name, task.layer, task.kind, task.df.iloc[idx].reset_index(drop=True), task.y[idx],
                task.cov_cols, None if task.strata is None else {k: v[idx] for k, v in task.strata.items()}, task.names, task.meta)


def features(task: Task, provider: str, trunc=None):
    if provider == "cnn":                            # trained network: no frozen features; handled in evaluate()
        return np.zeros((len(task.df), 1), np.float32), "cnn"
    if provider == "expr_len":                       # sequence-free baseline: expression/length covariates only
        return task.df[task.cov_cols].values.astype(np.float32), provider
    pn, prov = bench.make_provider(provider, trunc)
    return bench.cached_embed(pn, prov, task.df, task.name), pn


def evaluate(task: Task, X, final=False, head="auto", n_boot=300):
    """Returns (rows, oof). rows: one dict per scope (dev_all, dev_pruned, [test_all, test_pruned])."""
    df, fold = task.df, task.df.fold.values
    kind = task.kind
    rows, extra = [], {}
    scopes = [("dev_all", (fold >= 0)), ("dev_pruned", (fold >= 0) & ~df.leak.values)]
    if head == "cnn":
        from . import cnn
        out = cnn.cnn_predict(task, final)
        preds = {"dev": out["dev"]}
        if final:
            preds["test"] = out["test"]
    else:
        preds = {"dev": evalkit.dev_oof(X, task.y, fold, kind, head)}
        if final:
            preds["test"] = evalkit.final_pred(X, task.y, fold, kind, head)
    if final:
        scopes += [("test_all", fold == -1), ("test_pruned", (fold == -1) & ~df.leak.values)]
    for scope, m in scopes:
        p = preds["dev" if scope.startswith("dev") else "test"]
        if kind == "multi":
            Pm = np.where(m[:, None], p, np.nan)
            r, per = evalkit.summarize_multi(task.y, Pm, task.strata, task.names, n_boot=n_boot)
            r["per_rbp"] = per.to_dict("records")
        else:
            r = evalkit.summarize_single(task.y[m], p[m], kind, n_boot)
        r["scope"] = scope
        rows.append(r)
    return rows, preds


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*converge.*")


def run(providers, task_names, trunc=None, max_n=None, final=False, out=None, head="auto", n_boot=300, save_oof=False):
    results = []
    for tn in task_names:
        task = subsample(load_task(tn), max_n)
        n_dev, n_test = int((task.df.fold >= 0).sum()), int((task.df.fold == -1).sum())
        print(f"\n[{tn}] layer {task.layer} kind={task.kind} items={len(task.df)} (dev {n_dev}, locked test {n_test}, "
              f"homology-flagged {100*task.df.leak.mean():.1f}%)" + (f"  label reliability: {task.meta}" if task.meta else ""), flush=True)
        for pv in providers:
            t0 = time.time()
            X, pn = features(task, pv, trunc)
            tk = task
            if pv == "cnn" and trunc:                       # same input as every other provider
                d2 = task.df.copy(); d2["seq"] = d2.seq.str.slice(0, trunc)
                tk = Task(task.name, task.layer, task.kind, d2, task.y, task.cov_cols, task.strata, task.names, task.meta)
                pn = f"cnn@{trunc}"
            rows, preds = evaluate(tk, X, final, "cnn" if pv == "cnn" else "gbm" if pv.endswith("_gbm") else head, n_boot)
            for r in rows:
                r.update(task=tn, provider=pn, kind=task.kind, layer=task.layer, dim=int(X.shape[1]), secs=round(time.time() - t0, 1))
                if save_oof and r["scope"] == "dev_all":
                    r["oof"] = [None if np.isnan(v) else round(float(v), 5) for v in (preds["dev"] if task.kind != "multi" else preds["dev"][:, 0])]
                    r["y"] = [float(v) for v in (task.y if task.kind != "multi" else task.y[:, 0])]
                results.append(r)
            show = [r for r in rows if r["scope"] in ("dev_all", "dev_pruned", "test_pruned")]
            print("  " + f"{pn:<24s}" + "  ".join(_fmt(r) for r in show), flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return results


def _fmt(r):
    s = r["scope"]
    if "macro_auroc" in r:
        flag = " UNDERPOWERED" if r.get("underpowered") else ""
        return (f"{s}: macroAUROC {r['macro_auroc']:.3f} | expr-strat {r['macro_auroc_strat_expr']:.3f} | "
                f"expr+len-strat {r['macro_auroc_strat_expr_len']:.3f} (RBPs {r['n_rbps']}{flag})")
    if "auroc" in r:
        return f"{s}: AUROC {r['auroc']:.3f} [{r['auroc_lo']:.2f},{r['auroc_hi']:.2f}] n={r['n']}"
    if "spearman" in r:
        return f"{s}: rho {r['spearman']:.3f} [{r['spearman_lo']:.2f},{r['spearman_hi']:.2f}] n={r['n']}"
    return f"{s}: n={r.get('n')}"


def _val(r):
    if "macro_auroc_strat_expr_len" in r:
        return r["macro_auroc_strat_expr_len"], r.get("macro_auroc_strat_expr_len_lo"), r.get("macro_auroc_strat_expr_len_hi")
    if "auroc" in r and r["auroc"] == r["auroc"]:
        return r["auroc"], r.get("auroc_lo"), r.get("auroc_hi")
    if "spearman" in r:
        return r["spearman"], r.get("spearman_lo"), r.get("spearman_hi")
    return None, None, None


def report(files, scope="dev_pruned"):
    """Markdown pivot (provider x task) for one scope. Multi-label RBP tasks show expression+length-stratified macro AUROC."""
    rows = []
    for f in files:
        rows += json.load(open(f))
    df = pd.DataFrame([r for r in rows if r.get("scope") == scope])
    if df.empty:
        print("no rows for scope", scope); return
    order = [t for t in LAYER1 + LAYER2B if t in set(df.task)]
    out = {}
    for _, r in df.iterrows():
        v, lo, hi = _val(r)
        if v is None:
            continue
        flag = "*" if r.get("underpowered") else ""
        out.setdefault(r["provider"], {})[r["task"]] = f"{v:.3f} [{lo:.2f},{hi:.2f}]{flag}" if lo is not None else f"{v:.3f}{flag}"
    print(f"\n**scope: {scope}**  (rho for *_reg, AUROC for *_clf, expression+length-stratified macro-AUROC for rbp_*; * = underpowered)\n")
    print("| provider | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
    for p, v in out.items():
        print(f"| {p} | " + " | ".join(v.get(t, "") for t in order) + " |")


def compare(spec_a: str, spec_b: str, scope="dev_all"):
    """Paired bootstrap A - B on identical items. spec = 'file.json:provider-substring'. Needs runs made with --save-oof."""
    from . import evalkit as ek
    def load(spec):
        f, prov = spec.rsplit(":", 1)
        rows = [r for r in json.load(open(f)) if r.get("scope") == scope and prov in r["provider"] and "oof" in r]
        return {r["task"]: r for r in rows}
    A, B = load(spec_a), load(spec_b)
    print(f"\n### {spec_a}  minus  {spec_b}   ({scope})\n| task | metric | A | B | delta [95% CI] | p(A<=B) |\n|---|---|---|---|---|---|")
    for t in A:
        if t not in B or A[t]["kind"] == "multi":
            continue
        ra, rb = A[t], B[t]
        ok = [i for i, (x, z) in enumerate(zip(ra["oof"], rb["oof"])) if x is not None and z is not None]
        y = np.array(ra["y"])[ok]
        assert np.allclose(y, np.array(rb["y"])[ok]), f"items differ for {t}: run both with the same --max-n/--trunc"
        d, lo, hi, p = ek.paired_diff(y, np.array(ra["oof"], dtype=float)[ok], np.array(rb["oof"], dtype=float)[ok], ra["kind"])
        key = "auroc" if ra["kind"] == "clf" else "spearman"
        print(f"| {t} | {key} | {ra[key]:.3f} | {rb[key]:.3f} | {d:+.3f} [{lo:+.3f},{hi:+.3f}] | {p:.3f} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("a"); cp.add_argument("b"); cp.add_argument("--scope", default="dev_all")
    rp = sub.add_parser("report")
    rp.add_argument("files", nargs="+")
    rp.add_argument("--scope", default="dev_pruned", choices=["dev_all", "dev_pruned", "test_all", "test_pruned"])
    r = sub.add_parser("run")
    r.add_argument("--providers", nargs="+", default=["expr_len", "lengc", "kmer"])
    r.add_argument("--tasks", nargs="+", default=["layer1"])
    r.add_argument("--trunc", type=int, default=None)
    r.add_argument("--max-n", type=int, default=None)
    r.add_argument("--final", action="store_true", help="ALSO score the locked test set (do this once, at the end)")
    r.add_argument("--n-boot", type=int, default=300)
    r.add_argument("--save-oof", action="store_true")
    r.add_argument("--out", default=None)
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        compare(a.a, a.b, a.scope)
        return
    if a.cmd == "report":
        report(a.files, a.scope)
        return
    if a.cmd == "list":
        print("layer 1:", LAYER1, "\nlayer 2b:", LAYER2B)
        return
    names = []
    for t in a.tasks:
        names += LAYER1 if t == "layer1" else LAYER2B if t == "layer2b" else LAYER1 + LAYER2B if t == "all" else [t]
    run(a.providers, names, a.trunc, a.max_n, a.final, a.out, n_boot=a.n_boot, save_oof=a.save_oof)


if __name__ == "__main__":
    main()
