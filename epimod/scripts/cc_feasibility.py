"""Feasibility test for composition-centering with a LINEAR clock (cheap, interpretable)."""
import os, sys, json, numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from epimod.age import age_to_t, t_to_age
from epimod.eval.baselines import clocks_dir
from epimod.lb.store import load_lb_columns, lb_series
from epimod.lb.tasks import load_parsed

D = os.environ["EPIMOD_DATA"]
panel = pd.read_parquet(D + "/store/panel.parquet")
ref = pd.read_csv(clocks_dir() / "450K_reinius_12_reference.csv", index_col=0)           # 600 CpGs x 6 cell types
CT = list(ref.columns)
want = list(dict.fromkeys(list(panel.index[:12000]) + list(ref.index)))
meta, X, ids = load_lb_columns(want, exclude_series=lb_series(load_parsed()))
col = {c: i for i, c in enumerate(ids)}
print("samples:", len(meta), "| columns:", X.shape[1], "| reference CpGs available:", sum(c in col for c in ref.index), flush=True)

# ---- 1) composition by non-negative least squares on the reference CpGs (per sample, missing CpGs skipped)
rc = [c for c in ref.index if c in col]; B = ref.loc[rc].to_numpy(np.float64); R = X[:, [col[c] for c in rc]].astype(np.float64)
W = np.zeros((len(meta), len(CT)))
for i in range(len(meta)):
    ok = np.isfinite(R[i])
    if ok.sum() < 100: W[i] = np.nan; continue
    w, _ = nnls(B[ok], R[i, ok]); W[i] = w / max(w.sum(), 1e-9)
print("composition estimated; NaN rows:", int(np.isnan(W[:, 0]).sum()), flush=True)

# ---- 2) do estimates agree with annotated cell types? (purified samples only)
lin = {"CD14+ Monocytes": "monocyte", "CD4+ T cells": "cd4_t_cell", "CD4+ T-cells": "cd4_t_cell", "CD4 +T cells": "cd4_t_cell", "CD4+ Naive T-cells": "cd4_t_cell", "CD4+ Memory T-cells": "cd4_t_cell",
       "CD8+ T cells": "cd8_t_cell", "CD8+ T-cells": "cd8_t_cell", "CD8+ Naive T-cells": "cd8_t_cell", "CD19+ B cells": "b_cell", "CD19+ B-cells": "b_cell", "CD56+ NK cells": "nk_cell",
       "Neutrophils": "neutrophil", "CD15+ Neutrophils": "neutrophil", "Granulocytes": "neutrophil", "CD15+ Granulocytes": "neutrophil"}
meta["lineage"] = meta.CellType.map(lin)
pur = meta.lineage.notna().to_numpy() & ~np.isnan(W[:, 0])
pred = np.array(CT)[W[pur].argmax(1)]
print(f"\npurified samples with a mapped lineage: {int(pur.sum())} | NNLS argmax agrees with annotation: {np.mean(pred == meta.lineage.to_numpy()[pur]):.3f}")
print(pd.crosstab(meta.lineage[pur].to_numpy(), pred).to_string())
whole = meta.CellType.fillna("NA").str.contains("hole|PBL|buffy|PBMC", case=False).to_numpy() & ~np.isnan(W[:, 0])
print("\nmean estimated composition, whole-blood-like samples:", dict(zip(CT, W[whole].mean(0).round(3))))
np.savez(D + "/lb/composition.npz", W=W, ids=np.array(meta.index), ct=np.array(CT))
