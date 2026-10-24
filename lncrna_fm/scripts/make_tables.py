#!/usr/bin/env python
"""Merge LncBench result JSONs into markdown tables.

    python scripts/make_tables.py results/*_cv.json            # CV tasks: one row per provider
    python scripts/make_tables.py --retrieval results/*_retrieval.json
"""
import argparse
import json

import pandas as pd

TASK_ORDER = ["loc_reg", "loc_clf", "tau_reg", "testis_clf", "brain_clf", "coding_clf"]


def cv_table(rows):
    df = pd.DataFrame(rows)
    out = {}
    for _, r in df.iterrows():
        if "spearman" in r and pd.notna(r.get("spearman")):
            cell = f"{r['spearman']:.3f} [{r['spearman_lo']:.2f},{r['spearman_hi']:.2f}]"
        else:
            cell = f"{r['auroc']:.3f} [{r['auroc_lo']:.2f},{r['auroc_hi']:.2f}]"
        out.setdefault(r["provider"], {})[r["task"]] = cell
    cols = [t for t in TASK_ORDER if any(t in v for v in out.values())]
    lines = ["| provider | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for p, v in out.items():
        lines.append(f"| {p} | " + " | ".join(v.get(c, "") for c in cols) + " |")
    return "\n".join(lines)


def retrieval_table(rows, subset="named", metrics=("MRR", "hit@10", "hit@100")):
    df = pd.DataFrame(rows)
    lines = []
    for sub in ("all", "strict", "named"):
        d = df[df.subset == sub]
        if d.empty:
            continue
        lines.append(f"\n**{sub}** (n={int(d.n.iloc[0])}; random hit@10 = {d.rand_hit_at_10.iloc[0] if 'rand_hit_at_10' in d else d['rand_hit@10'].iloc[0]:.4f})\n")
        lines.append("| provider | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---|" * len(metrics))
        for _, r in d.iterrows():
            lines.append(f"| {r['provider']} | " + " | ".join(f"{r[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--retrieval", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in a.files:
        rows += json.load(open(f))
    print(retrieval_table(rows) if a.retrieval else cv_table(rows))
