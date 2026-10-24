"""LncBench runner: evaluate any feature provider on every task with identical probes.

    python -m lncfm.bench --providers lengc kmer --tasks all
    python -m lncfm.bench --providers kmer lncfm:runs/pilot/best.pt rinalmo-micro

A *provider* maps a list of RNA sequences -> (n, d) float array. Embeddings are cached
under $LNCFM_DATA/emb/<provider>/<task>.npy so re-runs are instant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evalkit, kmer, paths, tasks as T

# (task_name, builder, label column, kind)
TASKS = {
    "loc_reg":   (T.task_localization, "y",       "reg"),
    "loc_clf":   (T.task_localization, "y_bin",   "clf"),   # nuclear tercile vs cytoplasmic tercile
    "tau_reg":   (T.task_tissue,       "y",       "reg"),
    "testis_clf": (T.task_tissue,      "y_testis", "clf"),
    "brain_clf": (T.task_tissue,       "y_brain",  "clf"),
    "coding_clf": (T.task_coding_vs_lnc, "y",      "clf"),   # sanity
}


def provider_lengc(seqs):
    return np.array([[np.log10(len(s)), kmer.gc_content(s)] for s in seqs], dtype=np.float32)


def provider_kmer(seqs):
    return kmer.kmer_features(seqs, ks=(1, 2, 3, 4, 5))


def provider_kmer3(seqs):
    return kmer.kmer_features(seqs, ks=(1, 2, 3))


def get_provider(name: str):
    if name == "lengc":
        return provider_lengc
    if name == "kmer":
        return provider_kmer
    if name == "kmer3_gbm":
        return provider_kmer3
    if name.startswith("lncfm:"):
        from .embed import LncFMProvider
        return LncFMProvider(name.split(":", 1)[1])
    from .embed import HFProvider  # rinalmo-micro, rnafm, nt-v2-50m, ...
    return HFProvider(name)


def make_provider(name: str, trunc: int | None = None):
    """Returns (cache_name, callable). `trunc` clips every sequence to its first N nt for ALL providers."""
    prov = get_provider(name)
    if not trunc:
        return name, prov
    return f"{name}@{trunc}", (lambda seqs, _p=prov, _n=trunc: _p([x[:_n] for x in seqs]))


def cached_embed(pname: str, provider, df: pd.DataFrame, tname: str = "") -> np.ndarray:
    """Per-provider store keyed by tx_id, so transcripts shared between tasks are embedded once."""
    import pickle
    f = paths.ensure(paths.DATA / "emb") / (pname.replace("/", "_").replace(":", "_") + ".pkl")
    store = pickle.loads(f.read_bytes()) if f.exists() else {}
    todo = [i for i, t in enumerate(df.tx_id) if t not in store]
    if todo:
        Z = provider([df.seq.iloc[i] for i in todo])
        for i, z in zip(todo, Z):
            store[df.tx_id.iloc[i]] = z
        f.write_bytes(pickle.dumps(store))
    return np.stack([store[t] for t in df.tx_id])


def run(provider_names, task_names, n_boot=500, out=None, max_n=None, trunc=None, save_oof=False):
    rows = []
    built = {}
    for pn0 in provider_names:
        pn, prov = make_provider(pn0, trunc)
        head = "gbm" if pn0.endswith("_gbm") else "auto"
        for tn in task_names:
            builder, col, kind = TASKS[tn]
            if builder not in built:
                built[builder] = builder()
            df = built[builder]
            if kind == "clf":
                df = df[df[col] >= 0]
            if max_n and len(df) > max_n:
                df = df.sample(max_n, random_state=0)
            df = df.reset_index(drop=True)
            t0 = time.time()
            X = cached_embed(pn, prov, df, tn)
            r = evalkit.cv_eval(X, df[col].values, df.fold.values, kind, head=head, n_boot=n_boot)
            oof, yy = r.pop("_oof"), r.pop("_y")
            r.update(provider=pn, task=tn, kind=kind, dim=int(X.shape[1]), secs=round(time.time() - t0, 1))
            if save_oof:
                r["oof"], r["y"] = [round(float(v), 5) for v in oof], [float(v) for v in yy]
                r["tx_id"] = list(df.tx_id)
            rows.append(r)
            metric = evalkit.fmt_auc(r) if kind == "clf" else evalkit.fmt_rho(r)
            print(f"{pn:>22s}  {tn:<11s} n={r['n']:<6d} {('AUROC' if kind=='clf' else 'rho'):>5s} {metric}", flush=True)
    res = pd.DataFrame(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        res.to_json(out, orient="records", indent=1)
    return res


def run_retrieval(provider_names, max_queries=None, pool_size=None, out=None, trunc=None):
    """T3: zero-shot positional-ortholog retrieval, human query -> mouse pool."""
    from . import retrieval
    Q, P, pos, subsets = retrieval.build_retrieval(max_queries, pool_size)
    print(f"retrieval: {len(Q)} human queries, pool of {len(P)} mouse genes; "
          + ", ".join(f"{k}={int(v.sum())}" for k, v in subsets.items()), flush=True)
    rows = []

    def report(pn, res):
        for sub, r in res.items():
            rows.append({"provider": pn, "subset": sub, **r})
            print(f"{pn:>22s}  retr/{sub:<6s} n={r['n']:<5d} MRR={r['MRR']:.3f} hit@10={r['hit@10']:.3f} "
                  f"hit@100={r['hit@100']:.3f} medrank={r['median_rank']:.0f} (random hit@10={r['rand_hit@10']:.4f})", flush=True)

    for pn0 in provider_names:
        if pn0 == "shared10mer":
            report(pn0, retrieval.eval_kmer_overlap([x[:trunc] if trunc else x for x in Q.seq],
                                                    [x[:trunc] if trunc else x for x in P.seq], pos, subsets))
            continue
        pn, prov = make_provider(pn0, trunc)
        Zq = cached_embed(pn, prov, Q, "retrQ")
        Zp = cached_embed(pn, prov, P, "retrP")
        report(pn, retrieval.eval_embeddings(Zq, Zp, pos, subsets))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_json(out, orient="records", indent=1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true", help="run the ortholog-retrieval task instead of the CV tasks")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--providers", nargs="+", default=["lengc", "kmer"])
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-n", type=int, default=None, help="cap items per task (CPU smoke tests)")
    ap.add_argument("--save-oof", action="store_true", help="store out-of-fold predictions (needed for paired comparisons)")
    ap.add_argument("--trunc", type=int, default=None, help="clip every sequence to its first N nt, for all providers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.retrieval:
        run_retrieval(a.providers, a.max_queries, a.pool_size, a.out, a.trunc)
        return
    names = list(TASKS) if a.tasks == ["all"] else a.tasks
    run(a.providers, names, a.n_boot, a.out, a.max_n, a.trunc, a.save_oof)


if __name__ == "__main__":
    main()
