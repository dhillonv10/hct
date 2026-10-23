"""CpG panel construction from training studies only.

Pass 1 streams every healthy-control training study once and records, for every 450K-annotated CpG,
(a) its *within-study* correlation with transformed age, so batch differences between studies
cannot masquerade as an age signal, (b) pooled mean / variance, and (c) coverage. The benchmark
studies are never read here, so panel selection cannot leak test information.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..age import age_to_t
from .parquet_io import MethylParquet

MIN_STUDY_N = 30        # a study contributes an age correlation only with this many samples ...
MIN_STUDY_AGE_SD = 5.0  # ... and this much age spread (years)


@dataclass
class StudyStats:
    gse: str
    gidx: np.ndarray      # global CpG index for each retained row
    n: np.ndarray         # observed samples per CpG
    sx: np.ndarray
    sxx: np.ndarray
    r: np.ndarray | None  # within-study Pearson r with transformed age (None if study unusable)
    n_samples: int


def study_stats(path: str, vocab_ids: np.ndarray, ages: dict[str, float]) -> StudyStats:
    """Sufficient statistics for one study (runs in a worker process)."""
    mp = MethylParquet(path)
    vocab = pd.Index(vocab_ids)
    row_g = vocab.get_indexer(mp.cpg_ids)
    keep = np.flatnonzero(row_g >= 0)
    gidx = row_g[keep]
    samples = [s for s in mp.sample_ids if s in ages and np.isfinite(ages[s])]
    y_all = age_to_t(np.array([ages[s] for s in samples]))
    usable = len(samples) >= MIN_STUDY_N and np.std([ages[s] for s in samples]) >= MIN_STUDY_AGE_SD
    R = len(keep)
    n = np.zeros(R); sx = np.zeros(R); sxx = np.zeros(R); sxy = np.zeros(R); sy = np.zeros(R); syy = np.zeros(R)
    ypos = {s: i for i, s in enumerate(samples)}
    for names, block in mp.iter_blocks(batch=64, columns=samples):
        X = block[keep]
        obs = np.isfinite(X) & (X >= 0) & (X <= 1)
        X0 = np.where(obs, X, 0.0).astype(np.float32)
        M = obs.astype(np.float32)
        y = y_all[[ypos[s] for s in names]].astype(np.float32)
        n += M.sum(1); sx += X0.sum(1); sxx += (X0 * X0).sum(1)
        sxy += X0 @ y; sy += M @ y; syy += M @ (y * y)
    r = None
    if usable:
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = n * sxy - sx * sy
            den = np.sqrt(np.clip(n * sxx - sx**2, 0, None) * np.clip(n * syy - sy**2, 0, None))
            r = np.where((den > 0) & (n >= MIN_STUDY_N), cov / den, np.nan).astype(np.float32)
    return StudyStats(Path(path).stem, gidx.astype(np.int64), n, sx, sxx, r, len(samples))


def _worker(args):
    return study_stats(*args)


def collect_stats(paths: dict[str, Path], vocab_ids: np.ndarray, meta: pd.DataFrame, workers: int = 3) -> dict[str, np.ndarray]:
    """Aggregate StudyStats over all training studies into per-CpG arrays."""
    G = len(vocab_ids)
    ages_by_gse = {g: meta.loc[meta.DatasetID == g, "Age"].to_dict() for g in paths}
    jobs = [(str(p), vocab_ids, ages_by_gse[g]) for g, p in paths.items()]
    n_tot = np.zeros(G); sx_tot = np.zeros(G); sxx_tot = np.zeros(G)
    within_var_num = np.zeros(G); within_var_den = np.zeros(G)
    zw = np.zeros(G); w_sum = np.zeros(G); n_studies_r = np.zeros(G, dtype=np.int32)
    n_pos = np.zeros(G, dtype=np.int32); n_studies_obs = np.zeros(G, dtype=np.int32)
    total_samples = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for st in ex.map(_worker, jobs):
            total_samples += st.n_samples
            g = st.gidx
            n_tot[g] += st.n; sx_tot[g] += st.sx; sxx_tot[g] += st.sxx
            with np.errstate(divide="ignore", invalid="ignore"):
                var = np.where(st.n > 1, st.sxx / st.n - (st.sx / st.n) ** 2, 0.0)
            within_var_num[g] += st.n * np.clip(var, 0, None); within_var_den[g] += st.n
            n_studies_obs[g] += (st.n >= 0.9 * st.n_samples).astype(np.int32)
            if st.r is not None:
                ok = np.isfinite(st.r)
                gi, ri, ni = g[ok], np.clip(st.r[ok], -0.999, 0.999), st.n[ok]
                w = np.clip(ni - 3, 0, None)
                zw[gi] += w * np.arctanh(ri); w_sum[gi] += w
                n_studies_r[gi] += 1; n_pos[gi] += (ri > 0).astype(np.int32)
            print(f"[stats] {st.gse}: {st.n_samples} samples, age-informative={st.r is not None}", flush=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n_tot > 0, sx_tot / n_tot, np.nan)
        std = np.sqrt(np.clip(np.where(n_tot > 0, sxx_tot / n_tot - mean**2, np.nan), 0, None))
        wvar = np.where(within_var_den > 0, within_var_num / within_var_den, np.nan)
        z_meta = np.where(w_sum > 0, zw / np.sqrt(w_sum), 0.0)
        r_meta = np.where(w_sum > 0, np.tanh(zw / w_sum), 0.0)
    return dict(
        mean=mean.astype(np.float32), std=std.astype(np.float32), within_var=wvar.astype(np.float32),
        z_meta=z_meta.astype(np.float32), r_meta=r_meta.astype(np.float32), n_studies_r=n_studies_r,
        frac_pos=np.where(n_studies_r > 0, n_pos / np.maximum(n_studies_r, 1), 0.5).astype(np.float32),
        n_studies_obs=n_studies_obs, coverage=(n_tot / max(total_samples, 1)).astype(np.float32),
        total_samples=np.array(total_samples),
    )


def select_panel(
    vocab_ids: np.ndarray,
    stats: dict[str, np.ndarray],
    annotation: pd.DataFrame,
    must_include: list[str] | None = None,
    n_total: int = 16000,
    n_age: int = 12000,
    n_x: int = 400,
    min_coverage: float = 0.85,
    x_min_coverage: float = 0.70,
    min_study_frac: float = 0.5,
) -> pd.DataFrame:
    """Rank-ordered CpG panel.

    Priority order (so that ``panel.head(k)`` is itself a sensible smaller panel):
      1. published-clock CpGs (``must_include``), so published clocks can be evaluated fairly;
      2. CpGs by within-study age association (|meta-analytic Z|), requiring a consistent sign;
      3. the ``n_x`` most variable X-chromosome CpGs (sex-driven; supports the sex auxiliary task);
      4. high within-study-variance autosomal CpGs (cell composition, smoking ...) to fill the panel.
    Everything except (1) must be observed in >= ``min_coverage`` of training samples, lie on an
    autosome or X, and be free of common SNPs in/near the probe (genotype-driven probes encode
    ancestry and cohort identity rather than aging).
    """
    df = pd.DataFrame({"cpg": vocab_ids, "gidx": np.arange(len(vocab_ids))}).set_index("cpg")
    for k in ("z_meta", "r_meta", "frac_pos", "n_studies_r", "coverage", "within_var", "mean", "std"):
        df[k] = stats[k]
    df = df.join(annotation[["chrom", "pos", "snp"]], how="inner")
    base = (df.chrom < 23) & df["std"].gt(1e-3) & np.isfinite(df["mean"]) & (df.snp == 0)
    eligible = base & (df.coverage >= min_coverage)
    # some GEO depositors uploaded autosomes only, so X is present in just ~73% of training samples
    eligible_x = base & (df.chrom == 22) & (df.coverage >= x_min_coverage)
    min_studies = max(3, int(min_study_frac * stats["n_studies_r"].max()))
    consistent = eligible & ((df.frac_pos - 0.5).abs() >= 0.2) & (df.n_studies_r >= min_studies)

    source: dict[str, str] = {}  # insertion order == priority order

    def add(ids, tag: str):
        for c in ids:
            if c not in source and len(source) < n_total:
                source[c] = tag

    # clock CpGs bypass the coverage / variance filters: baselines are evaluated on them regardless
    add([c for c in (must_include or []) if c in df.index], "clock")
    ranked = df[consistent].assign(az=lambda d: d.z_meta.abs()).sort_values("az", ascending=False)
    add(ranked.index[: max(0, n_age - len(source))], "age_assoc")
    xs = df[eligible_x & ~df.index.isin(list(source))].sort_values("within_var", ascending=False)
    add(xs.index[:n_x], "xchrom")
    rest = df[eligible & (df.chrom < 22) & ~df.index.isin(list(source))].sort_values("within_var", ascending=False)
    add(rest.index[: n_total - len(source)], "variance")

    panel = df.loc[list(source)].copy()
    panel["mean"] = panel["mean"].fillna(0.5)                     # clock CpGs never seen in training
    panel["std"] = panel["std"].fillna(0.2).clip(lower=0.02)
    panel["priority"] = np.arange(len(panel))
    panel["source"] = list(source.values())
    return panel
