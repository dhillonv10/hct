#!/usr/bin/env python
"""Paired bootstrap comparison of providers on identical items (needs bench --save-oof outputs).

    python scripts/compare.py A=results/a_cv.json B=results/b_cv.json [C=...]  --pairs A:B A:C
Prints, per task, delta = metric(A) - metric(B) with a 95% CI and the one-sided bootstrap p(A <= B).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from lncfm import evalkit  # noqa: E402


def load(spec):
    name, path = spec.split("=", 1)
    rows = json.load(open(path))
    return name, {(r["task"]): r for r in rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--pairs", nargs="+", required=True, help="e.g. A:B means metric(A)-metric(B)")
    ap.add_argument("--provider", default=None, help="if a file holds several providers pick one (substring)")
    a = ap.parse_args()
    S = dict(load(s) for s in a.sources)
    for pair in a.pairs:
        A, B = pair.split(":")
        print(f"\n### {A} minus {B}")
        print("| task | metric | A | B | delta [95% CI] | p(A<=B) |\n|---|---|---|---|---|---|")
        for t in S[A]:
            if t not in S[B]:
                continue
            ra, rb = S[A][t], S[B][t]
            assert np.allclose(ra["y"], rb["y"]), f"items differ for task {t}"
            d, lo, hi, p = evalkit.paired_diff(ra["y"], ra["oof"], rb["oof"], ra["kind"])
            m = "AUROC" if ra["kind"] == "clf" else "rho"
            va = ra.get("auroc", ra.get("spearman")); vb = rb.get("auroc", rb.get("spearman"))
            print(f"| {t} | {m} | {va:.3f} | {vb:.3f} | {d:+.3f} [{lo:+.3f},{hi:+.3f}] | {p:.3f} |")
