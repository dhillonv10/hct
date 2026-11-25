#!/usr/bin/env python3
"""Do RNA language models beat the in-silico baselines on functionally measured RNU4-2 variants?

The benchmark. Nature 2026;654:429-436 (doi 10.1038/s41586-026-10334-9) saturation-genome-edited RNU4-2, the
ReNU syndrome gene, and measured a growth function score for 453 single-nucleotide variants -- essentially every
possible SNV in the transcript. On the 12 ReNU variants versus 346 population variants, the paper reports its own
assay at AUC 0.93, CADD v1.7 at 0.65 and ViennaRNA delta-delta-G at 0.72. No RNA language model was tested.

Why it is worth doing. Our own ClinVar panel has 29 variants in this gene and wide intervals; this has 453 with a
quantitative, functional readout and no ClinVar label circularity. And the paper ships CADD and ddG per variant,
so the published baselines can be reproduced exactly before anything new is scored -- which is the gate below.

  python3 rnu42_sge.py prep   --xlsx sge.xlsx --genome hg38.fa.gz --work work
  python3 rnu42_sge.py score  --work work --ckpt rinalmo_micro_pretrained.pt --cfg micro
  python3 rnu42_sge.py report --work work
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distill"))

LO, HI = 120291753, 120291903          # GRCh38 chr12, the region the paper scored; RNU4-2 is on the minus strand
PUBLISHED = {"CADD_score": 0.65, "ddG_bind": 0.72, "sge": 0.93}
GATE = 0.02


# --------------------------------------------------------------------------------------------------- evaluation
def auroc(y, x):
    """Rank AUROC with average ranks for ties (CADD and ddG both tie heavily)."""
    y, x = np.asarray(y), np.asarray(x, float)
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return np.nan
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    ranks = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    r = np.empty(len(x))
    r[o] = ranks
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def boot_auroc(y, x, n=2000, seed=0):
    rs = np.random.default_rng(seed)
    y, x = np.asarray(y), np.asarray(x, float)
    v = [a for a in (auroc(y[i], x[i]) for i in (rs.integers(0, len(y), len(y)) for _ in range(n)))
         if not np.isnan(a)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_spearman(a, b, n=2000, seed=0):
    rs = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = [spearman(a[i], b[i]) for i in (rs.integers(0, len(a), len(a)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# -------------------------------------------------------------------------------------------------------- prep
def read_transcript(genome, lo=LO, hi=HI):
    """Spliced RNU4-2 sequence, reverse-complemented off the minus strand. RNU4-2 is single-exon."""
    out, take, pos = [], False, 0
    with gzip.open(genome, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if take:
                    break
                take = line[1:].split()[0] == "chr12"
                pos = 0
                continue
            if not take:
                continue
            s = line.strip()
            if pos + len(s) >= lo - 1 and pos < hi:
                out.append(s[max(0, lo - 1 - pos):min(len(s), hi - pos)])
            pos += len(s)
            if pos > hi:
                break
    g = "".join(out).upper()
    return g.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def stage_prep(xlsx, genome, work):
    os.makedirs(work, exist_ok=True)
    d = pd.read_excel(xlsx, sheet_name=0, skiprows=1)
    s = d[d.Type == "SNV"].copy()
    tx = read_transcript(genome)
    print(f"transcript: {len(tx)} nt (145 transcribed + 6 beyond the 3' end, as the paper describes)")

    pos, ref, alt = [], [], []
    for hg in s.HGVS_transcript:                               # pandas 3 keeps NA through astype(str)
        m = re.match(r"n\.(\d+)([ACGT])>([ACGT])", "" if pd.isna(hg) else str(hg))
        pos.append(int(m.group(1)) if m else -1)
        ref.append(m.group(2) if m else "")
        alt.append(m.group(3) if m else "")
    s["pos"], s["ref"], s["alt"] = pos, ref, alt
    s = s[s.pos > 0].reset_index(drop=True)

    bad = [(p, r, tx[p - 1]) for p, r in zip(s.pos, s.ref) if tx[p - 1] != r]
    if bad:
        raise SystemExit(f"reference base disagrees with the extracted transcript at {len(bad)} variants: {bad[:5]}")
    print(f"reference base matches the extracted transcript for all {len(s)} SNVs")

    s["y"] = np.where(s.category == "ReNU_syndrome", 1,
                      np.where(s.category == "UKBB/AllofUs", 0, np.nan))
    keep = ["HGVS_transcript", "pos", "ref", "alt", "category", "y",
            "function_score_mean", "qvalues", "CADD_score", "ddG_bind"]
    s[keep].to_csv(os.path.join(work, "variants.csv"), index=False)
    open(os.path.join(work, "transcript.txt"), "w").write(tx + "\n")
    print(f"{len(s)} SNVs written | ReNU {int((s.y == 1).sum())} | population {int((s.y == 0).sum())} | "
          f"unobserved {int(s.y.isna().sum())}")


# ------------------------------------------------------------------------------------------------------- score
def stage_score(work, ckpt, cfg, device="auto", bs=16):
    import torch
    from rnakd.lm import masked_log_odds
    from rnakd.teacher import LM_CONFIGS, RiNALMo, load_strict

    v = pd.read_csv(os.path.join(work, "variants.csv"))
    tx = open(os.path.join(work, "transcript.txt")).read().strip()
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RiNALMo(**LM_CONFIGS[cfg])
    load_strict(model, ckpt)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {cfg} ({n:,d} params) on {dev}")

    rows = v.assign(seq=tx, offset=v.pos - 1)
    s = masked_log_odds(model, rows, dev, width=len(tx) + 2, bs=bs)
    col = f"lm_{cfg}"
    out = os.path.join(work, "variants.csv")
    v[col] = s
    v.to_csv(out, index=False)
    print(f"wrote column {col} to {out}")


# ------------------------------------------------------------------------------------------------------ report
def stage_report(work, out=None):
    v = pd.read_csv(os.path.join(work, "variants.csv"))
    lines = []

    def say(t=""):
        print(t)
        lines.append(t)

    lab = v[v.y.notna()].copy()
    y = lab.y.values.astype(int)
    say(f"RNU4-2 saturation genome editing, Nature 2026;654:429-436 (CC BY 4.0)")
    say(f"  {len(v)} SNVs scored by the assay; the labelled contrast is "
        f"{int(y.sum())} ReNU syndrome vs {int((1 - y).sum())} population variants")
    say()

    say("GATE -- reproduce the paper's own numbers before trusting anything new")
    ok = True
    for col, name in (("CADD_score", "CADD v1.7"), ("ddG_bind", "ViennaRNA ddG")):
        m = lab[lab[col].notna()]
        a = auroc(m.y.values.astype(int), m[col].values)
        hit = abs(a - PUBLISHED[col]) <= GATE
        ok &= hit
        say(f"  {name:16s} AUC {a:.3f}   published {PUBLISHED[col]:.2f}   {'MATCH' if hit else 'MISMATCH'}")
    m = lab[lab.function_score_mean.notna()]
    a = auroc(m.y.values.astype(int), -m.function_score_mean.values)
    hit = abs(a - PUBLISHED["sge"]) <= GATE
    ok &= hit
    say(f"  {'SGE assay':16s} AUC {a:.3f}   published {PUBLISHED['sge']:.2f}   {'MATCH' if hit else 'MISMATCH'}")
    if not ok:
        say("\n  GATE FAILED -- the harness does not reproduce the published baselines. Stop here.")
        open(out or os.path.join(work, "report.txt"), "w").write("\n".join(lines) + "\n")
        return
    say("  gate passed: every number below is computed the same way")

    models = [c for c in v.columns if c.startswith("lm_")]
    if not models:
        say("\nno language-model scores yet (run `score`)")
        open(out or os.path.join(work, "report.txt"), "w").write("\n".join(lines) + "\n")
        return

    say()
    say("A. SEPARATING ReNU SYNDROME FROM POPULATION VARIANTS  (the paper's own contrast)")
    say(f"  {'score':22s} {'AUC':>6s}  {'95% CI':>15s}   vs CADD 0.65")
    base = auroc(y, lab.CADD_score.values)
    for col, name in [("CADD_score", "CADD v1.7"), ("ddG_bind", "ViennaRNA ddG")] + \
                     [(c, c.replace("lm_", "RiNALMo-")) for c in models]:
        m = lab[lab[col].notna()]
        a = auroc(m.y.values.astype(int), m[col].values)
        lo, hi = boot_auroc(m.y.values.astype(int), m[col].values)
        say(f"  {name:22s} {a:6.3f}  [{lo:.3f},{hi:.3f}]   {a - base:+.3f}")
    say(f"  {'SGE assay (ceiling)':22s} {auroc(y, -lab.function_score_mean.values):6.3f}")

    say()
    say("B. AGREEMENT WITH THE CONTINUOUS FUNCTION SCORE  (all SNVs, not just the 358 labelled ones)")
    say("   Spearman against the measured effect, oriented so positive = agrees with the assay.")
    say(f"  {'score':22s} {'rho':>7s}  {'95% CI':>16s}  {'n':>5s}")
    f = v[v.function_score_mean.notna()]
    for col, name, sign in [("CADD_score", "CADD v1.7", -1), ("ddG_bind", "ViennaRNA ddG", -1)] + \
                           [(c, c.replace("lm_", "RiNALMo-"), -1) for c in models]:
        m = f[f[col].notna()]
        r = sign * spearman(m[col].values, m.function_score_mean.values)
        lo, hi = boot_spearman(m[col].values, m.function_score_mean.values)
        lo, hi = sorted([sign * lo, sign * hi])
        say(f"  {name:22s} {r:7.3f}  [{lo:.3f},{hi:.3f}]  {len(m):5d}")

    say()
    say("C. SEPARATING THE PAPER'S OWN DEPLETED CALLS  (function score < -0.302 at q < 0.01)")
    say("   The best-powered contrast available: it uses the assay's own hit calls rather than the 12 clinically")
    say("   ascertained variants, so the intervals are a third as wide as in A.")
    dep = v[v.function_score_mean.notna()].copy()
    dep["yd"] = (dep.function_score_mean < -0.302).astype(int)
    say(f"   {int(dep.yd.sum())} depleted vs {int((1 - dep.yd).sum())} not depleted")
    say(f"  {'score':22s} {'AUC':>6s}  {'95% CI':>15s}")
    for col, name in [("CADD_score", "CADD v1.7"), ("ddG_bind", "ViennaRNA ddG")] + \
                     [(c, c.replace("lm_", "RiNALMo-")) for c in models]:
        m = dep[dep[col].notna()]
        a = auroc(m.yd.values, m[col].values)
        lo, hi = boot_auroc(m.yd.values, m[col].values)
        say(f"  {name:22s} {a:6.3f}  [{lo:.3f},{hi:.3f}]")

    say()
    say("Note on direction: a language model's masked log-odds is oriented so that higher means more damaging, "
        "and the assay's function score is lower when a variant is more depleted, so the two are expected to "
        "correlate negatively; the sign is flipped above so that positive means agreement.")
    path = out or os.path.join(work, "report.txt")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "score", "report"])
    ap.add_argument("--xlsx")
    ap.add_argument("--genome")
    ap.add_argument("--work", default="work_rnu42")
    ap.add_argument("--ckpt")
    ap.add_argument("--cfg", default="micro")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "prep":
        stage_prep(a.xlsx, a.genome, a.work)
    elif a.stage == "score":
        stage_score(a.work, a.ckpt, a.cfg, a.device, a.bs)
    else:
        stage_report(a.work)


if __name__ == "__main__":
    main()
