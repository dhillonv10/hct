"""High-level experiment entry points used by the CLI: train a run, evaluate runs + baselines."""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig, TrainConfig
from .data.computage import sibling_studies
from .data.store import load_split
from .eval.baselines import PUBLISHED, predict_published
from .eval.computage_protocol import run_protocol
from .eval.elasticnet import ElasticNetClock
from .model.epimod import EpiMod
from .train.loop import calibrate_sigma, evaluate_ages, load_checkpoint, save_checkpoint, split_studies, train


def _apply(cfg, overrides: dict):
    names = {f.name for f in fields(cfg)}
    bad = set(overrides) - names
    if bad:
        raise ValueError(f"unknown config keys {sorted(bad)}; valid: {sorted(names)}")
    return replace(cfg, **overrides)


def cpg_half(n: int, which: str, seed: int) -> np.ndarray:
    """Indices of a deterministic random half ("A" or "B") of ``n`` CpGs; A and B are disjoint."""
    in_a = np.random.RandomState(seed).rand(n) < 0.5
    return np.flatnonzero(in_a if which == "A" else ~in_a)


def run_training(out_dir: str | Path, model_overrides: dict | None = None, train_overrides: dict | None = None,
                 log=print) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mc = _apply(ModelConfig(), model_overrides or {})
    tc = _apply(TrainConfig(), train_overrides or {})
    data = load_split("train", n_cpgs=mc.n_cpgs)
    if mc.cpg_split:
        idx = cpg_half(mc.n_cpgs, mc.cpg_split, mc.cpg_split_seed)
        data = type(data)(X=data.X[:, idx], meta=data.meta, cpgs=data.cpgs.subset(idx))
        mc = replace(mc, n_cpgs=len(idx))
        log(f"[train] CpG split {mc.cpg_split}: training on {len(idx)} of the panel's CpGs")
    tr, va = split_studies(data.meta, tc.val_study_frac, tc.split_seed)
    train_d, val_d = data.select(tr), data.select(va)
    log(f"[train] {train_d.n} train / {val_d.n} val samples; val studies: {sorted(val_d.meta.DatasetID.unique())}")
    model = EpiMod(data.cpgs, mc)
    log(f"[train] parameters: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    res = train(model, train_d, val_d, tc, out_dir, log=log)
    res["wall_sec"] = round(time.time() - t0, 1)
    if mc.distributional:
        res["sigma_scale"] = calibrate_sigma(model, val_d)
        res["val_after_calibration"] = evaluate_ages(model, val_d)
        log(f"[train] sigma calibrated on validation studies: scale={res['sigma_scale']:.3f}; "
            f"val metrics: {json.dumps({k: round(v, 3) for k, v in res['val_after_calibration'].items()})}")
    save_checkpoint(model, out_dir / "model.pt", tc, extra={"val_studies": sorted(val_d.meta.DatasetID.unique())})
    prior = res.pop("ridge_prior", None)
    if prior is not None:
        np.savez(out_dir / "ridge_prior.npz", w=prior["w"], b=prior["b"], lam=prior["lambda"])
        res["ridge_lambda"] = prior["lambda"]; res["ridge_cv"] = prior["cv_median_abs_err_t"]
    (out_dir / "history.json").write_text(json.dumps(res, indent=1))
    (out_dir / "config.json").write_text(json.dumps({"model": mc.to_dict(), "train": tc.to_dict()}, indent=1))
    return res


def _elasticnet(train_dir: Path, n_cpgs: int, cache: Path) -> ElasticNetClock:
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    ck = load_checkpoint(train_dir / "model.pt")
    import torch

    val = torch.load(train_dir / "model.pt", map_location="cpu", weights_only=False)["extra"]["val_studies"]
    data = load_split("train", n_cpgs=n_cpgs)
    keep = ~data.meta["DatasetID"].isin(val).to_numpy()
    en = ElasticNetClock().fit(data.select(keep))
    cache.write_bytes(pickle.dumps(en))
    del ck
    return en


def breakdown(preds: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Healthy-control MedAE by platform / tissue / age band, per model."""
    hc = meta[(meta.Condition == "HC")]
    rows = []
    bands = pd.cut(hc.Age, [-1, 18, 40, 60, 90, 120], labels=["<18", "18-40", "40-60", "60-90", ">90"])
    for name, grp in (("platform", hc.Platform), ("tissue", hc.Tissue), ("age_band", bands)):
        for level in pd.unique(grp.dropna()):
            idx = grp.index[grp == level]
            for m in preds.columns:
                rows.append(dict(by=name, level=str(level), model=m, n=len(idx),
                                 MedAE=float(np.median(np.abs(preds.loc[idx, m] - hc.loc[idx, "Age"])))))
    return pd.DataFrame(rows).pivot_table(index=["by", "level", "n"], columns="model", values="MedAE").reset_index()


def run_evaluation(runs: dict[str, str | Path], out_dir: str | Path, elasticnet: bool = True,
                   published: bool = True, log=print) -> pd.DataFrame:
    """``runs``: display name -> run directory containing ``model.pt``."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models = {k: load_checkpoint(Path(v) / "model.pt") for k, v in runs.items()}
    bench = load_split("benchmark")                      # full 16k-CpG panel; models pick columns by CpG id
    X = bench.X.astype(np.float32)
    preds = pd.DataFrame(index=bench.meta.index)
    extras: dict = {}
    for name, m in models.items():
        cols = pd.Index(bench.cpgs.ids).get_indexer(m.cpg_ids)
        assert (cols >= 0).all(), f"{name}: model uses CpGs missing from the benchmark store"
        p = m.predict(X[:, cols])
        preds[name] = p["age"]
        extras[name] = {"sigma_years": p["sigma_years"], "lo": p["age_lo"], "hi": p["age_hi"]}
        log(f"[eval] predicted with {name}")
    base_n = ModelConfig().n_cpgs                        # baselines use the default panel prefix
    first = Path(next(iter(runs.values())))
    if elasticnet:
        en = _elasticnet(first, base_n, out_dir / "elasticnet.pkl")
        preds["ElasticNet(same panel)"] = en.predict(X[:, :base_n])
        log(f"[eval] ElasticNet: {en.n_nonzero} non-zero coefficients")
    prior_f = first / "ridge_prior.npz"
    if prior_f.exists() and len(np.load(prior_f)["w"]) == base_n:
        from .age import t_to_age

        rp = np.load(prior_f)
        mean, std = bench.cpgs.mean[:base_n], np.maximum(bench.cpgs.std[:base_n], 1e-2)
        z = np.where(np.isfinite(X[:, :base_n]), np.clip((X[:, :base_n] - mean) / std, -6, 6), 0.0)
        preds["Ridge(same panel)"] = t_to_age(z @ rp["w"] + float(rp["b"]))
    if published:
        for name in PUBLISHED:
            preds[name] = predict_published(name, X, bench.cpgs.ids)
    summary, long = run_protocol(preds, bench.meta)
    summary.to_csv(out_dir / "summary.csv"); long.to_csv(out_dir / "tests_long.csv", index=False)
    sib = sibling_studies()
    keep = ~bench.meta.DatasetID.isin(sib).to_numpy()
    run_protocol(preds[keep], bench.meta[keep])[0].to_csv(out_dir / "summary_no_siblings.csv")
    (out_dir / "sibling_studies_excluded.json").write_text(json.dumps(sib))
    preds.to_csv(out_dir / "predictions.csv.gz")
    breakdown(preds, bench.meta).to_csv(out_dir / "hc_breakdown.csv", index=False)
    # calibration of the predictive interval on healthy controls
    cal = {}
    hc = (bench.meta.Condition == "HC").to_numpy()
    age = bench.meta.Age.to_numpy()
    unc = {}
    for name, e in extras.items():
        inside = (age >= e["lo"]) & (age <= e["hi"])
        d = {"cover90_hc": float(inside[hc].mean()), "mean_sigma_years_hc": float(np.mean(e["sigma_years"][hc]))}
        for col in ("Platform", "Tissue"):
            for lvl in sorted(bench.meta[col].unique()):
                m = hc & (bench.meta[col] == lvl).to_numpy()
                if m.sum() >= 30:
                    d[f"cover90_hc[{col}={lvl}]"] = float(inside[m].mean())
        core = hc & bench.meta.Tissue.eq("Blood").to_numpy() & bench.meta.Platform.isin(["450K", "EPIC", "EPICv2"]).to_numpy()
        d["cover90_hc[blood,450K/EPIC]"] = float(inside[core].mean())
        cal[name] = d
        unc[f"{name}:sigma_years"], unc[f"{name}:lo"], unc[f"{name}:hi"] = e["sigma_years"], e["lo"], e["hi"]
    (out_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    pd.DataFrame(unc, index=bench.meta.index).to_csv(out_dir / "uncertainty.csv.gz")
    return summary
