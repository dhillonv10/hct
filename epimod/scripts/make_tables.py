"""Turn evaluation CSVs into markdown tables.

    python scripts/make_tables.py RUNS_DIR OUT.md [ANALYSIS_DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

KEEP = ["MedAE", "MedAE_18_90", "MedE", "AA2_hits", "AA2_tests", "AA1_hits", "AA1_tests", "score"]


def md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, r in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(runs: str, out: str, analysis: str | None = None) -> None:
    runs_p, ev = Path(runs), Path(runs) / "eval"
    parts: list[str] = []
    s = pd.read_csv(ev / "summary.csv", index_col=0)
    parts += ["## Held-out benchmark: all models", "", md(s[KEEP].astype({"AA2_hits": int, "AA2_tests": int, "AA1_hits": int, "AA1_tests": int})), ""]
    if (ev / "summary_no_siblings.csv").exists():
        sib = json.loads((ev / "sibling_studies_excluded.json").read_text())
        s2 = pd.read_csv(ev / "summary_no_siblings.csv", index_col=0)
        parts += [f"## Sensitivity: without sibling series of training studies ({', '.join(sib)})", "",
                  md(s2[KEEP].astype({"AA2_hits": int, "AA2_tests": int, "AA1_hits": int, "AA1_tests": int})), ""]
    # ablations vs full
    abl = [i for i in s.index if i in {"full", "full_seed1", "no_context", "no_id", "no_module_mixing", "point_estimate",
                                       "no_aug", "no_ssl", "no_linear_route", "sparse_modules"}]
    if "full" in abl:
        a = s.loc[abl, ["MedAE", "MedAE_18_90", "MedE", "AA2_hits", "AA1_hits", "score"]].copy()
        a["dMedAE_vs_full"] = a.MedAE - a.loc["full", "MedAE"]
        a["dScore_vs_full"] = a.score - a.loc["full", "score"]
        parts += ["## Ablations (identical budget; `full_seed1` is the same model with another init seed = noise floor)", "", md(a), ""]
    bd = pd.read_csv(ev / "hc_breakdown.csv")
    cols = [c for c in ["full", "ElasticNet(same panel)", "Ridge(same panel)", "HorvathV1", "HorvathV2", "PhenoAgeV2"] if c in bd.columns]
    parts += ["## Healthy-control MedAE by subgroup (years)", "", md(bd[["by", "level", "n"] + cols].set_index("by")), ""]
    cal = ev / "calibration.json"
    if cal.exists():
        c = pd.DataFrame(json.loads(cal.read_text())).T
        parts += ["## Predictive-interval calibration on benchmark healthy controls (nominal 90%)", "", md(c, "{:.3f}"), ""]
    hist = []
    for d in sorted(p for p in runs_p.iterdir() if (p / "history.json").exists()):
        h = json.loads((d / "history.json").read_text())
        hist.append(dict(run=d.name, best_val_medae=h.get("best_val_medae"), epochs=len(h["history"]), wall_min=round(h["wall_sec"] / 60, 1)))
    if hist:
        parts += ["## Training runs", "", md(pd.DataFrame(hist).set_index("run")), ""]
    if analysis and (Path(analysis) / "accel_summary.csv").exists():
        a = pd.read_csv(Path(analysis) / "accel_summary.csv", index_col=0)
        parts += ["## Acceleration head (cross-fitted)", "", md(a[KEEP].astype({"AA2_hits": int, "AA2_tests": int, "AA1_hits": int, "AA1_tests": int})), ""]
    Path(out).write_text("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
