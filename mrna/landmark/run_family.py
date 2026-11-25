#!/usr/bin/env python3
"""Run the whole model family, in order of information per hour. Resumable: re-run and it skips what is done.

Four phases, and the order matters -- each one's result changes whether the next is worth its compute.

  A  SCALING SWEEP     5 sizes (tiny..xl), human, 5 folds. Produces the scaling curve.
                       Pre-registered expectation: this SATURATES early. With ~14k scorable genes, capacity
                       is not the binding constraint, and where the curve flattens is the deliverable --
                       it tells a practitioner what size they actually need, which nobody has published
                       for mRNA half-life.
  B  HEADLINE          best size by validation, human, 10 folds, 2 seeds. The number compared to Saluki.
  C  MOUSE             same size, mouse bundle, 10 folds. Second published benchmark (Saluki 0.7320).
  D  MECHANISM         --aux-drop 1.0 at the best size: sequence path only, engineered features switched
                       off entirely. Saluki's own sequence-only ablation is 0.608, so BEATING 0.608 here is
                       what shows the landmark encoding contributes something their convolutions miss.
                       If phase B looks good but D lands at ~0.60, the win came from the features, not the
                       architecture, and the paper has to say so.

  python3 run_family.py --bundle hl_bundle2_human.npz --mouse-bundle hl_bundle2_mouse.npz
  python3 run_family.py --bundle hl_bundle2_human.npz --phases A        # just the sweep
  python3 run_family.py --bundle hl_bundle2_human.npz --dry-run         # print the plan and exit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_landmark.py")
SIZES = ["tiny", "small", "base", "large", "xl"]
SALUKI = {"human": 0.7726, "mouse": 0.7320}
SEQ_ONLY_SALUKI = 0.608


def run(tag, bundle, outdir, extra, dry):
    out = os.path.join(outdir, f"{tag}.json")
    if os.path.exists(out):
        print(f"[skip] {tag} already done -> {out}")
        return json.load(open(out))
    cmd = [sys.executable, TRAIN, "--bundle", bundle, "--out", out, "--tag", tag] + extra
    print(f"\n{'=' * 78}\n[run] {tag}\n  {' '.join(cmd)}\n{'=' * 78}", flush=True)
    if dry:
        return None
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[FAIL] {tag} exited {r.returncode}. Fix it and re-run; finished phases are skipped.")
        return None
    print(f"[done] {tag} in {(time.time() - t0) / 3600:.2f} h")
    return json.load(open(out)) if os.path.exists(out) else None


def table(rows, saluki):
    print(f"\n{'run':22s}{'params':>10s}{'mean r':>9s}{'sd':>8s}{'folds':>7s}{'vs Saluki':>11s}")
    print("-" * 67)
    for r in rows:
        if not r:
            continue
        print(f"{r['tag']:22s}{r['params']/1e6:9.2f}M{r['mean']:9.4f}{r['sd']:8.4f}"
              f"{r['folds']:7d}{r['mean'] - saluki.get(r['species'], saluki['human']):+11.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="human bundle (hl_bundle2_human.npz)")
    ap.add_argument("--mouse-bundle", default="")
    ap.add_argument("--outdir", default="family_results")
    ap.add_argument("--phases", default="ABCD")
    ap.add_argument("--sweep-folds", type=int, default=5)
    ap.add_argument("--sweep-epochs", type=int, default=30)
    ap.add_argument("--final-folds", type=int, default=10)
    ap.add_argument("--final-epochs", type=int, default=40)
    ap.add_argument("--lmax", type=int, default=12288)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    common = ["--lmax", str(a.lmax), "--bs", str(a.bs), "--multitask"]
    if a.device:
        common += ["--device", a.device]

    sweep = []
    if "A" in a.phases:
        print("\n### PHASE A -- scaling sweep ###")
        for sz in SIZES:
            sweep.append(run(f"A_{sz}_human", a.bundle, a.outdir,
                             common + ["--size", sz, "--folds", str(a.sweep_folds),
                                       "--epochs", str(a.sweep_epochs)], a.dry_run))
        got = [r for r in sweep if r]
        if got:
            table(got, SALUKI)
            best = max(got, key=lambda r: r["mean"])
            print(f"\nbest by mean r: {best['size']} ({best['params']/1e6:.2f}M params, r={best['mean']:.4f})")
            flat = [r for r in got if r["mean"] >= best["mean"] - 0.005]
            cheapest = min(flat, key=lambda r: r["params"])
            print(f"cheapest within 0.005 of best: {cheapest['size']} ({cheapest['params']/1e6:.2f}M)")
            print("  -> if those differ, the curve has saturated and the smaller model is the honest release.")
            json.dump({"best": best["size"], "cheapest_equiv": cheapest["size"]},
                      open(os.path.join(a.outdir, "_best.json"), "w"), indent=1)

    bp = os.path.join(a.outdir, "_best.json")
    best_size = json.load(open(bp))["cheapest_equiv"] if os.path.exists(bp) else "base"
    print(f"\nusing size '{best_size}' for phases B-D")

    out = []
    if "B" in a.phases:
        print("\n### PHASE B -- headline, human, 10 folds, 2 seeds ###")
        out.append(run(f"B_{best_size}_human_final", a.bundle, a.outdir,
                       common + ["--size", best_size, "--folds", str(a.final_folds),
                                 "--epochs", str(a.final_epochs), "--seeds", "2"], a.dry_run))
    if "C" in a.phases and a.mouse_bundle:
        print("\n### PHASE C -- mouse, second published benchmark ###")
        out.append(run(f"C_{best_size}_mouse", a.mouse_bundle, a.outdir,
                       common + ["--size", best_size, "--folds", str(a.final_folds),
                                 "--epochs", str(a.final_epochs)], a.dry_run))
    if "D" in a.phases:
        print("\n### PHASE D -- mechanism control: sequence path only ###")
        d = run(f"D_{best_size}_seqonly", a.bundle, a.outdir,
                common + ["--size", best_size, "--folds", str(a.sweep_folds),
                          "--epochs", str(a.final_epochs), "--aux-drop", "1.0"], a.dry_run)
        out.append(d)
        if d:
            print(f"\nsequence-only: r = {d['mean']:.4f}  vs Saluki's sequence-only ablation {SEQ_ONLY_SALUKI}"
                  f"  ({d['mean'] - SEQ_ONLY_SALUKI:+.4f})")
            print("  above 0.608 -> the landmark encoding is extracting something their convolutions do not."
                  "\n  at or below  -> any headline win came from the engineered features, and must be reported"
                  " that way.")

    rows = [r for r in sweep + out if r]
    if rows:
        print("\n### FAMILY SUMMARY ###")
        table(rows, SALUKI)
        print(f"\nreference: Saluki human {SALUKI['human']}, mouse {SALUKI['mouse']}, "
              f"sequence-only {SEQ_ONLY_SALUKI}; BEeM 0.7681; our features+GBM 0.7023")
        json.dump(rows, open(os.path.join(a.outdir, "_family_summary.json"), "w"), indent=1)
        print(f"wrote {a.outdir}/_family_summary.json")


if __name__ == "__main__":
    main()
