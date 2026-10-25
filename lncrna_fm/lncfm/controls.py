"""Harness controls: can the feature -> probe -> metric path find a planted signal, and does it refuse to invent one?

    python -m lncfm.controls                       # k-mers + sequence-free baseline (seconds)
    python -m lncfm.controls --providers kmer rinalmo-micro --trunc 300

Four synthetic tasks on random RNA (200-400 nt), evaluated with the SAME code as the real benchmark
(`lncbench.evaluate`, 5-fold, bootstrap CIs):

  S1 planted     half of the positives carry a 15-nt AU-rich element (UAUUUAUUUAUUUAU). A working harness recovers most of
                 the signal that exists (the oracle "contains the element" gives the ceiling).
  S2 weak        the same element in 15% of positives: tests sensitivity.
  S3 shuffled    S1 with the labels permuted: AUROC must be 0.5 (its CI must contain 0.5).
  S4 length      labels = "longer than median", sequences otherwise random: the sequence-free baseline must find it;
                 any sequence feature that includes length will too, which is exactly why covariate baselines are reported.

For a language model the planted element also shows whether mean-pooled embeddings keep a local motif at all.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import bench, lncbench

MOTIF = "UAUUUAUUUAUUUAU"


def random_seqs(n, rng, lo=200, hi=400):
    L = rng.integers(lo, hi + 1, n)
    return ["".join(rng.choice(list("ACGU"), l)) for l in L]


def plant(seq, rng):
    p = int(rng.integers(0, len(seq) - len(MOTIF)))
    return seq[:p] + MOTIF + seq[p + len(MOTIF):]


def make_task(name, seqs, y, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"tx_id": [f"{name}:{i}" for i in range(len(seqs))], "seq": seqs})
    df["length"] = df.seq.str.len()
    df["loglen"] = np.log10(df.length)
    df["fold"] = rng.integers(0, 5, len(df))
    df["leak"] = False
    return lncbench.Task(name, "ctrl", "clf", df, np.asarray(y, float), cov_cols=["loglen"])


def scenarios(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    seqs = random_seqs(n, rng)
    y = (rng.random(n) < 0.5).astype(int)

    def planted(prob):
        out = []
        for s, l in zip(seqs, y):
            out.append(plant(s, rng) if l == 1 and rng.random() < prob else s)
        return out
    s1 = planted(0.5)
    s2 = planted(0.15)
    oracle1 = roc_auc_score(y, [MOTIF in s for s in s1])
    oracle2 = roc_auc_score(y, [MOTIF in s for s in s2])
    perm = rng.permutation(n)
    ylen = (np.array([len(s) for s in seqs]) > 300).astype(int)
    return {
        "S1 planted (50% of positives)": (make_task("S1", s1, y), oracle1),
        "S2 weak (15% of positives)": (make_task("S2", s2, y), oracle2),
        "S3 shuffled labels": (make_task("S3", s1, y[perm]), 0.5),
        "S4 label = length": (make_task("S4", seqs, ylen), 1.0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", nargs="+", default=["kmer", "expr_len"])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--trunc", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sc = scenarios(a.n, a.seed)
    res, ok = {}, True
    print(f"{'scenario':<34s}{'oracle':>8s}" + "".join(f"{p:>22s}" for p in a.providers))
    for name, (task, oracle) in sc.items():
        row = []
        for pv in a.providers:
            if pv == "expr_len":
                X = task.df[task.cov_cols].values.astype(np.float32)
            else:
                _, prov = bench.make_provider(pv, a.trunc)
                X = np.asarray(prov(list(task.df.seq)), np.float32)
            r, _ = lncbench.evaluate(task, X, n_boot=200)
            d = [x for x in r if x["scope"] == "dev_all"][0]
            res[(name, pv)] = d
            row.append(f"{d['auroc']:.3f} [{d['auroc_lo']:.2f},{d['auroc_hi']:.2f}]")
        print(f"{name:<34s}{oracle:8.3f}" + "".join(f"{x:>22s}" for x in row))
    # verdicts
    print()
    for pv in a.providers:
        s3 = res[("S3 shuffled labels", pv)]
        neg = s3["auroc_lo"] <= 0.5 <= s3["auroc_hi"]
        print(f"[{pv}] negative control (shuffled labels): AUROC {s3['auroc']:.3f}, CI contains 0.5: {'PASS' if neg else 'FAIL'}")
        ok &= neg
    if "kmer" in a.providers:
        s1, orc = res[("S1 planted (50% of positives)", "kmer")], sc["S1 planted (50% of positives)"][1]
        frac = (s1["auroc"] - 0.5) / (orc - 0.5)
        print(f"[kmer] positive control: recovers {100*frac:.0f}% of the above-chance signal (needs >= 80%): {'PASS' if frac >= 0.8 else 'FAIL'}")
        ok &= frac >= 0.8
    if "expr_len" in a.providers:
        s4 = res[("S4 label = length", "expr_len")]
        print(f"[expr_len] covariate baseline finds a pure length label: AUROC {s4['auroc']:.3f} (needs >= 0.95): {'PASS' if s4['auroc'] >= 0.95 else 'FAIL'}")
        ok &= s4["auroc"] >= 0.95
    print("\nCONTROLS " + ("OK" if ok else "FAILED: do not trust benchmark numbers until this is fixed"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
