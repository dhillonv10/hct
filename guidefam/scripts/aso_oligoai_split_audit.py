"""Audit: how gene-independent is the held-out set produced by OligoAI's released training code?

Their `ASODataset.train_val_test_split` (rinalmo/data/downstream/aso/dataset.py, Apache-2.0) assigns whole PATENTS to train/val/test
(80/10/10 of the unique patents in file order, np.random.seed(42) + permutation). A patent is not a gene: the same gene is screened in many
patents (F12 in 82 screens, DGAT2 in 67). This script re-creates that exact split on the same file and reports how much of the TEST set
shares a gene group, or an identical (gene, ASO) pair, with TRAIN. It reads data only; no OligoAI code is run.

  python -m guidefam.scripts.aso_oligoai_split_audit --csv data_untrusted/aso_zenodo_20794660/aso.csv.gz --out guidefam/results/aso
"""
import argparse
import os

import numpy as np
import pandas as pd

from .. import aso as A


def patent_id(custom_id: str) -> str:
    return custom_id.split("/")[-1].split("_table_")[0]


def oligoai_split(raw: pd.DataFrame, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> pd.Series:
    """Replicates train_val_test_split on the frame as their ASODataset builds it (rows with a label, original order)."""
    df = raw.dropna(subset=["inhibition_percent"]).reset_index(drop=True)
    pid = df.custom_id.map(patent_id)
    np.random.seed(seed)
    perm = np.random.permutation(pid.unique())
    n = len(perm)
    tr, va = int(train_ratio * n), int(val_ratio * n)
    part = {p: "train" for p in perm[:tr]} | {p: "val" for p in perm[tr:tr + va]} | {p: "test" for p in perm[tr + va:]}
    return pd.Series(pid.map(part).values, index=df.index), pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = pd.read_csv(a.csv, low_memory=False)
    sp, pid = oligoai_split(raw)
    raw = raw.dropna(subset=["inhibition_percent"]).reset_index(drop=True)
    raw["oa_split"], raw["patent"] = sp.values, pid.values
    print(f"{raw.patent.nunique()} patents; split (patents / screens / rows):")
    for s in ("train", "val", "test"):
        x = raw[raw.oa_split == s]
        print(f"  {s:5s} {x.patent.nunique():4d} patents  {x.custom_id.nunique():5d} screens  {len(x):7d} rows")

    d = A.load(a.csv)                                                     # for the gene-group of each screen
    g_of = d.drop_duplicates("screen").set_index("screen").group
    raw["group"] = raw.custom_id.map(g_of)
    raw["aso"] = raw.aso_sequence_5_to_3.str.upper().str.replace("U", "T")
    tr, te = raw[raw.oa_split == "train"], raw[raw.oa_split == "test"]
    tg = set(tr.group)
    seen_pair = set(zip(tr.group, tr.aso))
    te = te.assign(in_train_group=te.group.isin(tg), in_train_pair=[(g, s) in seen_pair for g, s in zip(te.group, te.aso)])
    sc = te.groupby("custom_id").agg(n=("y" if "y" in te else "aso", "size"), grp=("in_train_group", "first"))
    n20 = sc[sc.n >= 20]
    print("\nTEST patents' rows that share a GENE GROUP with TRAIN patents:  {:.1%}".format(te.in_train_group.mean()))
    print("TEST rows whose exact (gene group, ASO sequence) occurs in TRAIN:  {:.1%}".format(te.in_train_pair.mean()))
    print("TEST screens (n>=20) whose gene group occurs in TRAIN:  {:.1%} of {}".format(n20.grp.mean(), len(n20)))
    print("test gene groups: {}, of which also in train: {}".format(te.group.nunique(), len(set(te.group) & tg)))
    # same statistics under OUR split rule, for contrast (must be 0 by construction)
    out = pd.DataFrame({"rows_test": [len(te)], "frac_rows_gene_in_train": [te.in_train_group.mean()], "frac_rows_pair_in_train": [te.in_train_pair.mean()],
                        "test_screens_n20": [len(n20)], "frac_screens_gene_in_train": [n20.grp.mean()], "test_groups": [te.group.nunique()],
                        "test_groups_in_train": [len(set(te.group) & tg)]})
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, "oligoai_split_audit.csv"), index=False)
    raw[["custom_id", "aso", "group", "oa_split"]].to_csv(os.path.join(a.out, "oligoai_split_assignment.csv.gz"), index=False)


if __name__ == "__main__":
    main()
