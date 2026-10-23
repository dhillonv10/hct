"""Does composition-centering help a LINEAR clock? Plain vs +composition vs centered (+composition)."""
import os, json, numpy as np, pandas as pd
from epimod.age import age_to_t, t_to_age
from epimod.eval.baselines import clocks_dir
from epimod.lb.store import load_lb_columns, lb_series
from epimod.lb.tasks import load_parsed
from epimod.model.linear_prior import fit_dual_ridge

D = os.environ["EPIMOD_DATA"]
panel = pd.read_parquet(D + "/store/panel.parquet"); ref = pd.read_csv(clocks_dir() / "450K_reinius_12_reference.csv", index_col=0)
want = list(dict.fromkeys(list(panel.index[:12000]) + list(ref.index)))
meta, X, ids = load_lb_columns(want, exclude_series=lb_series(load_parsed()))
z = np.load(D + "/lb/composition.npz", allow_pickle=True); assert list(z["ids"]) == list(meta.index); W = z["W"]; CT = list(z["ct"])
col = {c: i for i, c in enumerate(ids)}
pcols = [col[c] for c in panel.index[:12000] if c in col]; Xp = X[:, pcols]
lin = {"CD14+ Monocytes": "monocyte", "CD4+ T cells": "cd4_t_cell", "CD4+ T-cells": "cd4_t_cell", "CD4 +T cells": "cd4_t_cell", "CD4+ Naive T-cells": "cd4_t_cell", "CD4+ Memory T-cells": "cd4_t_cell",
       "CD8+ T cells": "cd8_t_cell", "CD8+ T-cells": "cd8_t_cell", "CD8+ Naive T-cells": "cd8_t_cell", "CD19+ B cells": "b_cell", "CD19+ B-cells": "b_cell", "CD56+ NK cells": "nk_cell",
       "Neutrophils": "neutrophil", "CD15+ Neutrophils": "neutrophil", "Granulocytes": "neutrophil", "CD15+ Granulocytes": "neutrophil"}
meta["lineage"] = meta.CellType.map(lin)
ok = ~np.isnan(W[:, 0])
tr = (meta.split == "train").to_numpy() & ok; te = (meta.split == "benchmark").to_numpy() & ok & (meta.Condition == "HC").to_numpy()
age = meta.Age.to_numpy(float)
print(f"train {tr.sum()} (purified with lineage: {int(meta.lineage[tr].notna().sum())}) | test = benchmark HC {te.sum()} (purified: {int(meta.lineage[te].notna().sum())})", flush=True)

# lineage signatures on the panel CpGs from TRAINING purified samples only (mean per lineage)
SIG = os.environ.get("SIG", "train")      # "oracle" = signatures from ALL purified samples incl. test (a ceiling diagnostic, not a usable model)
B = np.full((len(CT), Xp.shape[1]), np.nan)
print("signature source:", SIG)
for k, c in enumerate(CT):
    sel = (tr if SIG == "train" else ok) & (meta.lineage == c).to_numpy()
    if sel.sum() >= 5: B[k] = np.nanmean(Xp[sel], axis=0)
    print(f"  signature {c:11s}: {int(sel.sum()):4d} training samples", flush=True)
have = ~np.isnan(B).any(0); print("CpGs with a signature for all 6 lineages:", int(have.sum()), "of", Xp.shape[1])
Wn = np.nan_to_num(W)
center = Xp - np.where(have, Wn @ np.nan_to_num(B), 0.0)                     # subtract composition-predicted profile where known

def design(mat, extra=None):
    mu = np.nanmean(mat[tr], 0); sd = np.maximum(np.nanstd(mat[tr], 0), 0.02)
    Zm = np.where(np.isfinite(mat), (mat - mu) / sd, 0.0).astype(np.float32); Zm = np.clip(Zm, -6, 6)
    if extra is not None:
        e = (extra - extra[tr].mean(0)) / np.maximum(extra[tr].std(0), 1e-3); Zm = np.concatenate([Zm, (3.0 * e).astype(np.float32)], 1)
    return Zm
variants = {"plain": design(Xp), "centered": design(center), "centered + composition": design(center, Wn)} if SIG == "oracle" else {"plain": design(Xp), "plain + composition": design(Xp, Wn), "centered": design(center), "centered + composition": design(center, Wn)}
groups = meta.DatasetID.to_numpy()[tr]; y = age_to_t(age[tr])
res = {}
lineage_te = meta.lineage.to_numpy()[te]; mixed = pd.isna(lineage_te)
for name, Z in variants.items():
    w, b, lam, cv = fit_dual_ridge(Z[tr], y, groups, lambdas=(1e3, 3e3, 1e4, 3e4, 1e5, 3e5))
    pred = t_to_age(Z[te] @ w + b); err = np.abs(pred - age[te]); bias = pred - age[te]
    r = {"lambda": lam, "all": float(np.median(err)), "mixed(whole-blood-like)": float(np.median(err[mixed])), "purified": float(np.median(err[~mixed])),
         **{f"pur:{c}": float(np.median(err[lineage_te == c])) for c in CT if (lineage_te == c).sum() >= 20}, "bias_purified": float(np.median(bias[~mixed])), "bias_mixed": float(np.median(bias[mixed]))}
    res[name] = r; print(f"{name:24s} lambda={lam:8.0f} | MedAE all {r['all']:.2f} | mixed {r['mixed(whole-blood-like)']:.2f} | purified {r['purified']:.2f} | bias purified {r['bias_purified']:+.2f} mixed {r['bias_mixed']:+.2f} | " + " ".join(f"{c[:4]}={r[f'pur:{c}']:.1f}" for c in CT if f'pur:{c}' in r), flush=True)
json.dump(res, open(D + f"/lb/cc_linear_{SIG}.json", "w"), indent=1)
print("n test purified by lineage:", pd.Series(lineage_te[~mixed]).value_counts().to_dict())
