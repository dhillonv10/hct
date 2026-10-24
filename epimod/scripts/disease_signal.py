"""Age acceleration / deviation-from-healthy for your own methylation dataset (e.g. AML cohorts).

  python scripts/disease_signal.py --beta betas.csv --meta meta.csv --age-col age --group-col status --control Control \\
         [--study-col cohort] [--id-col sample_id] --out results/aml

--beta : samples x CpGs (columns = cg ids) or CpGs x samples (rows = cg ids); csv / tsv / parquet. Values must be beta in [0,1].
--meta : one row per sample with the sample id, age, and a group label (the control label is given by --control).
Writes <out>_scores.csv (per sample) and <out>_report.csv (per group [and study]: Cohen's d, Mann-Whitney p, AUC)."""
import argparse

import numpy as np
import pandas as pd

from epimod.eval.disease_signal import HealthyReference, group_report

ap = argparse.ArgumentParser()
ap.add_argument("--beta", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--id-col", default=None); ap.add_argument("--age-col", default="age"); ap.add_argument("--group-col", default="group")
ap.add_argument("--control", required=True); ap.add_argument("--study-col", default=None)
a = ap.parse_args()


def read(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".txt")) else ",", index_col=0)


B = read(a.beta); meta = pd.read_csv(a.meta, sep="\t" if a.meta.endswith((".tsv", ".txt")) else ",")
meta = meta.set_index(a.id_col or meta.columns[0])
if B.index.astype(str).str.startswith("cg").mean() > 0.5:                 # CpGs x samples -> transpose
    B = B.T
B.index = B.index.astype(str); common = B.index.intersection(meta.index.astype(str)); meta.index = meta.index.astype(str)
print(f"{len(common)} samples in common of {len(B)} beta rows / {len(meta)} metadata rows")
B, meta = B.loc[common], meta.loc[common]
ref = HealthyReference.load()
sc = ref.score(B.to_numpy(np.float32), B.columns.astype(str), meta[a.age_col].to_numpy(float)); sc.index = common
print(f"panel coverage {sc.attrs['panel_coverage']:.2f} (fraction of the {len(ref.cpgs)} reference CpGs present in your data); "
      f"reference: {ref.n_ref} healthy controls")
if sc.attrs["panel_coverage"] < 0.5:
    print("WARNING: low panel coverage (27K-style array?) - clock and deviation are mean-imputed for the missing CpGs")
rep = group_report(sc, meta[a.group_col], a.control, meta[a.study_col] if a.study_col else None)
sc.join(meta[[a.group_col] + ([a.study_col] if a.study_col else [])]).to_csv(a.out + "_scores.csv"); rep.to_csv(a.out + "_report.csv", index=False)
print(rep.round(3).to_string(index=False))
