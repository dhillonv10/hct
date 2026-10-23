"""Post-hoc analysis of a trained run on the held-out benchmark: modules, per-condition module
shifts, and the cross-fitted acceleration head scored with the ComputAge protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data.annotation import load_probe_annotation
from .data.store import load_split
from .eval.baselines import PUBLISHED, predict_published
from .eval.computage_protocol import run_protocol
from .eval.interpret import describe_modules, module_shift_table
from .train.accel import AccelConfig, cross_fitted_age, extract_features
from .train.loop import load_checkpoint


def run_analysis(run_dir: str | Path, out_dir: str | Path, log=print) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(run_dir / "model.pt")
    bench = load_split("benchmark", n_cpgs=model.cfg.n_cpgs)
    X = bench.X.astype(np.float32)
    feats = extract_features(model, X)
    from .age import t_to_age

    base = t_to_age(feats["mu_t"].astype(np.float64))
    log(f"[analysis] features: h{feats['h'].shape}")

    # 1. what are the modules?
    ann = load_probe_annotation()
    mods = describe_modules(model, bench.cpgs, ann)
    mods.to_csv(out_dir / "modules.csv")

    # 2. which modules carry condition-specific acceleration (cases vs same-study controls)?
    shift, p = module_shift_table(feats["contrib"], bench.meta)
    shift.to_csv(out_dir / "module_shift_by_condition.csv"); p.to_csv(out_dir / "module_shift_pvalues.csv")

    # 3. acceleration head, cross-fitted by study and by whole disease class
    preds = pd.DataFrame({"EpiMod (age only)": base}, index=bench.meta.index)
    cfg = AccelConfig()
    for fold_by in ("study", "class"):
        pred, _ = cross_fitted_age(feats, bench.meta, fold_by=fold_by, cfg=cfg)
        preds[f"EpiMod + accel head ({fold_by}-held-out)"] = pred
    for name in ("HorvathV1", "HorvathV2", "PhenoAgeV1", "PhenoAgeV2"):
        preds[name] = predict_published(name, X, bench.cpgs.ids)
    summary, long = run_protocol(preds, bench.meta)
    summary.to_csv(out_dir / "accel_summary.csv"); long.to_csv(out_dir / "accel_tests_long.csv", index=False)
    log(summary.round(2).to_string())
    return {"summary": summary, "modules": mods, "shift": shift}
