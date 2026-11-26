#!/usr/bin/env python3
"""Run the whole model family, in order of information per hour. Resumable: re-run and it skips what is done.

Four phases, and the order matters -- each one's result changes whether the next is worth its compute.

REVISED after the first real run. The original design concatenated the 609 engineered features into the
network's head. That let the head reach val r=0.674 in ONE epoch and never improve across 25 more, because
the trunk had a shortcut and never received a useful gradient; the hybrid scored 0.693, BELOW the 0.702 that
gradient boosting gets on those features alone. So every phase now runs the network on SEQUENCE ONLY
(--aux-drop 1.0) and fuses late with a GBM trained inside the same fold (--stack-gbm). Each run therefore
reports three numbers: `nn` (sequence only, compare to Saluki's sequence-only ablation 0.608), `gbm`
(features alone, our 0.7023), and `stack` (compare to Saluki 0.7726). The old phase D is gone because the
control it provided is now part of every single run.

  0  READ-OUT ABLATION  3 read-outs at `base`, 3 folds, ~2.6 h. THE DECIDING PHASE.
                       Structured pooling alone scored nn=0.631 twice -- before and after the LR fix and
                       the structural scalars -- against Saluki's 0.744 for the same input class. The
                       diagnosis: collapsing ~768 positions into 20 pooled cells destroys the long-range
                       arrangement, so the network cannot use junction positions and scores like a model
                       with no splice track. Saluki downsamples to a short sequence and runs a GRU over it;
                       recurrent integration IN ORDER is the one mechanism it has that this model lacked.
                       `pool` reproduces the 0.631 result, `gru` replaces pooling with recurrence, `both`
                       concatenates. Falsifiable expectation: if the read-out was the bottleneck, `gru` and
                       `both` land at nn 0.70-0.75. If they land at 0.65, the bottleneck is elsewhere and
                       the later phases are not worth their compute.

  A  SCALING SWEEP     5 sizes (tiny..xl), human, 5 folds. Produces the scaling curve.
                       Pre-registered expectation: this SATURATES early. With ~14k scorable genes, capacity
                       is not the binding constraint, and where the curve flattens is the deliverable --
                       it tells a practitioner what size they actually need, which nobody has published
                       for mRNA half-life.
  B  HEADLINE          best size by validation, human, 10 folds, 2 seeds. The number compared to Saluki.
  C  MOUSE             same size, mouse bundle, 10 folds. Second published benchmark (Saluki 0.7320).

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
    print(f"\n{'run':22s}{'read-out':>9s}{'params':>9s}{'seq-only':>10s}{'gbm':>8s}{'stack':>8s}"
          f"{'vs Saluki':>11s}")
    print("-" * 77)
    for r in rows:
        if not r:
            continue
        c = r.get("components") or {}
        nn = f"{c['nn']:10.4f}" if c else f"{'-':>10s}"
        gb = f"{c['gbm']:8.4f}" if c else f"{'-':>8s}"
        print(f"{r['tag']:22s}{r.get('readout','-'):>9s}{r['params']/1e6:8.2f}M{nn}{gb}{r['mean']:8.4f}"
              f"{r['mean'] - saluki.get(r['species'], saluki['human']):+11.4f}")
    print(f"\nseq-only column compares to Saluki's sequence-only ablation {SEQ_ONLY_SALUKI}; "
          f"stack compares to full Saluki.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="human bundle (hl_bundle2_human.npz)")
    ap.add_argument("--mouse-bundle", default="")
    ap.add_argument("--outdir", default="family_results")
    ap.add_argument("--phases", default="0ABC")
    ap.add_argument("--sweep-folds", type=int, default=5)
    ap.add_argument("--sweep-epochs", type=int, default=25)
    ap.add_argument("--final-folds", type=int, default=10)
    ap.add_argument("--final-epochs", type=int, default=35)
    ap.add_argument("--lmax", type=int, default=12288)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    common = ["--lmax", str(a.lmax), "--bs", str(a.bs), "--multitask",
              "--aux-drop", "1.0", "--stack-gbm"]
    if a.device:
        common += ["--device", a.device]

    # ---- phase 0: which read-out? -------------------------------------------------------------
    readout = "both"
    rp = os.path.join(a.outdir, "_readout.json")
    if "0" in a.phases:
        print("\n### PHASE 0 -- read-out ablation (the deciding phase) ###")
        abl = []
        for ro in ("pool", "gru", "both"):
            abl.append(run(f"0_readout_{ro}", a.bundle, a.outdir,
                           common + ["--size", "base", "--readout", ro, "--folds", "3",
                                     "--epochs", "20"], a.dry_run))
        got = [r for r in abl if r]
        if got:
            table(got, SALUKI)
            best = max(got, key=lambda r: (r.get("components") or {}).get("nn", r["mean"]))
            readout = best.get("readout", "both")
            nn = (best.get("components") or {}).get("nn")
            print(f"\nbest read-out by the SEQUENCE-ONLY component: '{readout}'"
                  + (f"  (nn = {nn:.4f})" if nn else ""))
            print(f"  reference: Saluki sequence+splice 0.7443, full Saluki {SALUKI['human']}, "
                  f"pooling-only measured 0.631")
            if nn is not None and nn < 0.66:
                print("\n  *** nn is still below 0.66. The read-out was NOT the bottleneck. Phases A-C would")
                print("      spend ~30 h scaling a trunk that a validation-fitted stack weights near zero.")
                print("      Stop here and send me this table rather than continuing. ***")
            json.dump({"readout": readout, "nn": nn}, open(rp, "w"), indent=1)
    elif os.path.exists(rp):
        readout = json.load(open(rp))["readout"]
    common += ["--readout", readout]
    print(f"\nusing read-out '{readout}' for phases A-C")

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
