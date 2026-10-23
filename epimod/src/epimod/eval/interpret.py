"""Interpretation of EpiMod's learned epigenetic modules.

Because a module's CpG assignment is sample-independent, a module *is* a weighted CpG set and can be
described directly: its effective size, which genes / islands / regulatory classes it is enriched
for, and how much of the panel's published-clock CpGs it covers. Age is an exact sum of per-module
contributions, so condition-specific age acceleration can be decomposed over modules too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..data.annotation import CHROMS, GENE_GROUPS, ISLAND, REG_GROUPS
from ..data.store import CpGTable
from ..model.epimod import EpiMod


@torch.no_grad()
def assignment_matrix(model: EpiMod) -> np.ndarray:
    model.eval()
    return model.assignment().numpy()


def _enrichment(weights: np.ndarray, member: np.ndarray) -> tuple[float, float]:
    """Fold enrichment of assignment mass in ``member`` vs. its panel frequency, and a binomial-free
    permutation-free z (mass share vs. expected share under the effective sample size)."""
    base = member.mean()
    if base in (0.0, 1.0):
        return 1.0, 0.0
    share = float((weights * member).sum() / weights.sum())
    n_eff = 1.0 / float((weights**2).sum() / weights.sum() ** 2)
    z = (share - base) / np.sqrt(base * (1 - base) / n_eff)
    return share / base, float(z)


def describe_modules(model: EpiMod, cpgs: CpGTable, annotation: pd.DataFrame, top_genes: int = 6, top_cpgs: int = 5) -> pd.DataFrame:
    """One row per module: size, strongest enrichments, dominant genes and clock content."""
    A = assignment_matrix(model)
    ids = np.asarray(cpgs.ids)[: A.shape[1]]
    ann = annotation.loc[ids]
    cats: dict[str, np.ndarray] = {}
    for j, name in enumerate(ISLAND):
        cats[f"island:{name}"] = (cpgs.island[: A.shape[1]] == j)
    for j, name in enumerate(REG_GROUPS[1:], start=1):
        cats[f"reg:{name}"] = (cpgs.reg[: A.shape[1]] == j)
    for j, g in enumerate(GENE_GROUPS):
        cats[f"gene:{g}"] = cpgs.binary[: A.shape[1], j] > 0
    cats["enhancer"] = cpgs.binary[: A.shape[1], 6] > 0
    cats["DHS"] = cpgs.binary[: A.shape[1], 7] > 0
    cats["chrX"] = cpgs.chrom[: A.shape[1]] == CHROMS.index("X")
    cats["published_clock_cpg"] = cpgs.source[: A.shape[1]] == "clock"
    rows = []
    for k in range(A.shape[0]):
        w = A[k].astype(np.float64)
        enr = {n: _enrichment(w, m) for n, m in cats.items()}
        best = sorted(enr.items(), key=lambda kv: -abs(kv[1][1]))[:3]
        order = np.argsort(-w)
        genes = pd.Series(ann["genes"].to_numpy()[order[:400]])
        genes = genes[genes != "-"].value_counts().head(top_genes)
        rows.append(
            dict(
                module=k,
                effective_cpgs=float(1.0 / (w**2).sum()),
                top1pct_mass=float(np.sort(w)[::-1][: max(1, len(w) // 100)].sum()),
                top_enrichments="; ".join(f"{n} x{fe:.2f} (z={z:+.0f})" for n, (fe, z) in best),
                clock_cpg_fold=enr["published_clock_cpg"][0],
                top_genes=",".join(genes.index),
                top_cpgs=",".join(ids[order[:top_cpgs]]),
                mean_abs_age_assoc_z=float((w * np.abs(cpgs.z_age[: A.shape[1]])).sum() / w.sum()),
            )
        )
    return pd.DataFrame(rows).set_index("module")


def module_shift_table(contrib_t: np.ndarray, meta: pd.DataFrame, age_limits=(18, 90), min_n: int = 10):
    """Per condition: how much more each module contributes in cases than in healthy controls,
    **adjusted for chronological age and study** (OLS: contribution ~ study + age + case).

    Returns (estimates, p-values), rows = conditions, columns = modules, in transformed-age units
    (about 21 years per unit in adulthood). Cases are often older than controls, so an unadjusted
    difference would mostly measure the age gap rather than accelerated aging. Only studies with
    >= 3 cases and >= 3 controls enter, so batch effects cancel within study."""
    m = meta.copy(); m["_i"] = np.arange(len(m))
    m = m[m.Age.between(*age_limits)]
    est, pv = {}, {}
    for cond in sorted(set(m.Condition) - {"HC"}):
        keep = [g for g, gg in m.groupby("DatasetID") if (gg.Condition == cond).sum() >= 3 and (gg.Condition == "HC").sum() >= 3]
        sub = m[m.DatasetID.isin(keep) & m.Condition.isin([cond, "HC"])]
        if len(sub) < min_n:
            continue
        design = np.column_stack([
            pd.get_dummies(sub.DatasetID).to_numpy(float),
            (sub.Age.to_numpy(float) - sub.Age.mean()) / 10.0,
            (sub.Condition == cond).to_numpy(float),
        ])
        Y = contrib_t[sub._i.to_numpy()]
        beta = np.linalg.lstsq(design, Y, rcond=None)[0]
        dof = len(sub) - np.linalg.matrix_rank(design)
        s2 = ((Y - design @ beta) ** 2).sum(0) / max(dof, 1)
        se = np.sqrt(np.linalg.pinv(design.T @ design)[-1, -1] * s2)
        est[cond] = beta[-1]
        pv[cond] = 2 * stats.t.sf(np.abs(beta[-1] / np.maximum(se, 1e-12)), max(dof, 1))
    return pd.DataFrame(est).T, pd.DataFrame(pv).T


def integrated_gradients(model: EpiMod, x: np.ndarray, baseline: np.ndarray | None = None, steps: int = 12) -> np.ndarray:
    """Per-CpG attribution of predicted (transformed) age by integrated gradients along the straight
    path from ``baseline`` (default: panel mean beta) to the sample. Missing CpGs get zero attribution."""
    model.eval()
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    base = model.mean.detach().clone() if baseline is None else torch.as_tensor(baseline, dtype=torch.float32)
    base = base.expand_as(x)
    miss = ~torch.isfinite(x)
    x0 = torch.where(miss, base, x)
    total = torch.zeros_like(x0)
    for a in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:
        pt = (base + a * (x0 - base)).requires_grad_(True)
        model(pt).mu_t.sum().backward()
        total += pt.grad
    return (total / steps * (x0 - base)).masked_fill(miss, 0.0).numpy()
