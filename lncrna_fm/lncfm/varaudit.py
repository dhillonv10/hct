"""Audit of the ncRNA variant panel: is the within-gene AUROC measuring pathogenicity, or something cheaper?

    python -m lncfm.varaudit --models rinalmo-micro rnafm --out results/v4/audit_common.txt
    python -m lncfm.varaudit --bg any --models rinalmo-micro rnafm        # also adds TERC; benign = every gnomAD-observed variant

Every check below targets a specific way the headline number could be nonsense:

  1 integrity            a variant labelled both ways; duplicated sequences; reference-base disagreement
  2 permutation          label permutation within gene: how often does chance alone give the observed weighted AUROC?
  3 influence            drop one gene at a time; unweighted mean; median. Is the result carried by two or three genes?
  4 substitution class   pathogenic and benign variants differ in mutation type (transitions are far more common among
                         population variants). Does the LM add anything once substitution class is known? AUROC inside strata.
  5 position vs allele   mask score = log p(ref) - log p(alt). Is the signal "this position is constrained" (p(ref)) or
                         "this allele is implausible" (p(alt))?
  6 frequency            do rare population variants look more damaging than common ones? If the score tracks allele
                         frequency it can separate pathogenic from common variants without knowing anything about disease.
                         Pathogenic vs RARE-observed variants is the harder, fairer test.
  7 spot checks          top / bottom scored variants per gene, for a human to read
  8 CADD                 re-query a sample directly and compare with the cached values
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from . import gnomad, ncvar, varscore as V


def _tt(ref, alt):
    ref, alt = ref.replace("T", "U"), alt.replace("T", "U")
    return ref + ">" + alt


def subst_class(d):
    return np.array([_tt(r, a) for r, a in zip(d.ref_t, d.alt_t)])


def wauc(d, s, min_each=3):
    return V.within_gene(d.reset_index(drop=True), np.asarray(s, float), min_each)


def strat_wauc(d, s, strata, min_each=3):
    """Within-gene AUROC computed separately inside each stratum (e.g. substitution class), averaged with weights min(n_p, n_b)."""
    num = den = 0.0
    s = np.asarray(s, float)
    for g, ix in d.reset_index(drop=True).groupby("gene_id").indices.items():
        for st in np.unique(strata[ix]):
            j = ix[strata[ix] == st]
            y = d.label.values[j]
            npos, nneg = int(y.sum()), int((1 - y).sum())
            if npos >= 1 and nneg >= 1 and min(npos, nneg) >= 1:
                w = min(npos, nneg)
                num += V.fast_auc(y, s[j]) * w
                den += w
    return num / den if den else np.nan, den


def run(models, bg="common", af_min=1e-4, min_each=3, nperm=2000, cadd=True, out=None, seed=0):
    lines = []

    def P(*a):
        t = " ".join(str(x) for x in a)
        print(t, flush=True)
        lines.append(t)

    d = gnomad.build_eval_table(0, 0.0 if bg == "any" else af_min)
    panel = ncvar.eligible_genes(d, min_each)
    d = d[d.gene_id.isin(panel.gene_id)].reset_index(drop=True)
    P(f"=== AUDIT  background={bg}  {len(d)} variants, {int(d.label.sum())} pathogenic, {d.gene_id.nunique()} genes")

    # ---- 1 integrity
    key = d.tx_id + ":" + d.offset.astype(str) + ":" + d.alt_t
    conflict = d.groupby(key).label.nunique()
    P(f"[1] integrity: variants labelled both ways: {int((conflict > 1).sum())}; duplicated (transcript,offset,alt): {int(key.duplicated().sum())}; "
      f"reference-base disagreements: {sum(r.seq[r.offset] != r.ref_t.replace('T','U') for r in d.itertuples())}")
    sc_ = subst_class(d)
    tab = pd.crosstab(sc_, d.label)
    tab["P/LP share %"] = (100 * tab[1] / tab.sum(1)).round(0)
    P("    substitution classes (rows) by label (0 benign, 1 P/LP):")
    P("    " + tab.to_string().replace("\n", "\n    "))
    ts = np.array([(a in "AG") == (b in "AG") for a, b in zip(d.ref_t.str.replace("T", "U"), d.alt_t.str.replace("T", "U"))])
    P(f"    transitions: {100*ts[d.label.values==1].mean():.0f}% of P/LP vs {100*ts[d.label.values==0].mean():.0f}% of benign")

    # ---- scores
    X = V.features(d, models, with_struct=False, with_base=True, with_cadd=cadd, emb=False)
    cols = [f"lm:{V.model_tag(m)}:mask" for m in models]
    S = X[cols + (["cadd"] if cadd else []) + ["b:transversion"]].copy()

    # ---- 2 permutation test (zero-shot score, so the null is exactly 'no association inside gene')
    rng = np.random.default_rng(seed)
    P(f"[2] permutation test ({nperm} within-gene label shuffles), weighted within-gene AUROC:")
    gi = [ix for _, ix in d.groupby("gene_id").indices.items()]
    y0 = d.label.values.copy()
    for c in S:
        obs, _ = wauc(d, S[c].values, min_each)
        s = S[c].values
        null = []
        for _ in range(nperm):
            y = y0.copy()
            for ix in gi:
                y[ix] = rng.permutation(y[ix])
            dd = d.assign(label=y)
            null.append(V.within_gene(dd, s, min_each)[0])
        null = np.array(null)
        p = (1 + (null >= obs).sum()) / (nperm + 1)
        P(f"    {c:<26s} observed {obs:.3f}   null mean {np.nanmean(null):.3f} (sd {np.nanstd(null):.3f})   one-sided p = {p:.4f}")

    # ---- 3 influence
    P("[3] influence of single genes on the primary score(s):")
    for c in cols:
        a, t = wauc(d, S[c].values, min_each)
        loo = {g: wauc(d[d.gene != g], S[c].values[(d.gene != g).values], min_each)[0] for g in t.gene}
        w = np.minimum(t.n_path, t.n_ben)
        P(f"    {c}: weighted {a:.3f} | unweighted mean {t.auroc.mean():.3f} | median {t.auroc.median():.3f} | genes with AUROC>0.5: {(t.auroc>0.5).sum()}/{len(t)}")
        P("      leave-one-gene-out aggregate: " + ", ".join(f"-{g} {v:.3f}" for g, v in sorted(loo.items(), key=lambda kv: kv[1])[:4]) +
          " ... " + ", ".join(f"-{g} {v:.3f}" for g, v in sorted(loo.items(), key=lambda kv: kv[1])[-2:]))

    # ---- 4 substitution class
    P("[4] substitution class (does the LM add beyond mutation type?)")
    base, _ = V.logo_combiner(d, pd.get_dummies(pd.Series(sc_)).astype(float), min_each)
    P(f"    substitution-class model (leave-one-gene-out logistic on the 12 classes): {wauc(d, base, min_each)[0]:.3f}")
    for c in cols + (["cadd"] if cadd else []):
        s = S[c].values
        a_all = wauc(d, s, min_each)[0]
        a_str, w = strat_wauc(d, s, sc_, min_each)
        a_ts = strat_wauc(d, s, np.where(ts, "ts", "tv"), min_each)[0]
        comb, _ = V.logo_combiner(d, pd.concat([pd.get_dummies(pd.Series(sc_)).astype(float), S[[c]].reset_index(drop=True)], axis=1), min_each)
        P(f"    {c:<26s} overall {a_all:.3f} | inside transition/transversion strata {a_ts:.3f} | inside the 12 substitution classes {a_str:.3f} | "
          f"class + score (LOGO) {wauc(d, comb, min_each)[0]:.3f}")

    # ---- 5 position vs allele
    P("[5] position constraint p(ref) vs allele plausibility p(alt)  (masked LM, first model)")
    sc = V.MLMScorer(models[0], "cpu", 1000, 16, emb=False)
    lpr, lpa = np.full(len(d), np.nan), np.full(len(d), np.nan)
    wins = [V.window(s, o, sc.max_nt) for s, o in zip(d.seq, d.offset)]
    ba = {c: i for i, c in enumerate("ACGU")}
    order = np.argsort([len(w[0]) for w in wins])
    for b0 in range(0, len(d), 16):
        ix = order[b0:b0 + 16]
        lg, _ = sc._run([wins[i][0] for i in ix], [wins[i][1] for i in ix])
        lp = lg[np.arange(len(ix)), np.array([wins[i][1] for i in ix]) + 1][:, sc.base].log_softmax(-1)
        for j, i in enumerate(ix):
            lpr[i] = float(lp[j, ba[d.ref_t.iloc[i].replace("T", "U")]])
            lpa[i] = float(lp[j, ba[d.alt_t.iloc[i].replace("T", "U")]])
    for name, s in (("position constraint  log p(ref)", lpr), ("allele implausibility -log p(alt)", -lpa), ("difference (the mask score)", lpr - lpa)):
        P(f"    {name:<36s} within-gene AUROC {wauc(d, s, min_each)[0]:.3f}")

    # ---- 6 frequency
    if bg == "any":
        P("[6] population frequency: is the score just tracking how common a variant is?")
        gn = d.source.values == "gnomad"
        af = d.af.values
        for c in cols + (["cadd"] if cadd else []):
            ok = gn & ~np.isnan(S[c].values)
            r, p = spearmanr(np.log10(af[ok]), S[c].values[ok])
            P(f"    {c:<26s} Spearman(log10 AF, score) among gnomAD variants: {r:+.3f} (p={p:.1e}, n={ok.sum()})  [negative = rarer variants look more damaging]")
        bins = {"common AF>=1e-4": gn & (af >= 1e-4), "rare 1e-5<=AF<1e-4": gn & (af >= 1e-5) & (af < 1e-4), "ultra-rare AF<1e-5": gn & (af < 1e-5)}
        P("    P/LP versus each benign background (within-gene AUROC, genes with >=3 of each class):")
        for nm, m in bins.items():
            sub = d[(d.source == "clinvar") & (d.label == 1) | m].reset_index(drop=True)
            ss = S[(d.source == "clinvar").values & (d.label.values == 1) | m].reset_index(drop=True)
            row = [f"{c.split(':')[1] if c.startswith('lm:') else c} {wauc(sub, ss[c].values, min_each)[0]:.3f}" for c in S]
            P(f"      {nm:<22s} n_benign={int(m.sum()):5d} | " + " | ".join(row))
    else:
        P("[6] skipped (run with --bg any for the frequency analysis)")

    # ---- 7 spot checks
    P("[7] spot checks (primary score; higher = more damaging)")
    c0 = cols[0]
    for g, ix in d.groupby("gene").indices.items():
        sub = d.iloc[ix].assign(score=S[c0].values[ix])
        pp = sub[sub.label == 1].sort_values("score")
        bb = sub[sub.label == 0].sort_values("score")
        if len(pp) >= 3 and len(bb) >= 3:
            P(f"    {g}: P/LP lowest-scored {[(int(r.pos), r.ref + '>' + r.alt, round(r.score, 2)) for r in pp.head(2).itertuples()]}  "
              f"highest {[(int(r.pos), r.ref + '>' + r.alt, round(r.score, 2)) for r in pp.tail(2).itertuples()]} | benign highest "
              f"{[(int(r.pos), r.ref + '>' + r.alt, round(r.score, 2), r.source[:2]) for r in bb.tail(2).itertuples()]}")
    if "MT-TL1" in set(d.gene):
        r = d[(d.gene == "MT-TL1") & (d.pos == 3243)]
        if len(r):
            P(f"    known MELAS variant m.3243A>G: label={int(r.label.iloc[0])}, score {S[c0].values[r.index[0]]:.2f}, "
              f"percentile within MT-TL1 {100*(rankdata(S[c0].values[(d.gene=='MT-TL1').values]) / (d.gene=='MT-TL1').sum())[list(d[d.gene=='MT-TL1'].index).index(r.index[0])]:.0f}")

    # ---- 8 CADD
    if cadd:
        import requests
        nuc = d[(d.chrom != "chrM")].sample(8, random_state=1)
        bad = 0
        for r in nuc.itertuples():
            j = requests.get(f"https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7/{r.chrom.replace('chr','')}:{r.pos}", timeout=60).json()
            v = {x["Ref"] + ">" + x["Alt"]: float(x["PHRED"]) for x in j}.get(f"{r.ref}>{r.alt}", np.nan)
            cached = X["cadd"].loc[r.Index] if "cadd" in X else np.nan
            bad += int(not (np.isnan(v) and np.isnan(cached)) and abs(v - cached) > 1e-6)
        P(f"[8] CADD: re-queried 8 random nuclear variants; mismatches with cached values: {bad}")
        P(f"    CADD coverage of nuclear variants: {100*X['cadd'][(d.chrom!='chrM').values].notna().mean():.0f}%")
    if out:
        open(out, "w").write("\n".join(lines))
        print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rinalmo-micro", "rnafm"])
    ap.add_argument("--bg", choices=["common", "any"], default="common")
    ap.add_argument("--af-min", type=float, default=1e-4)
    ap.add_argument("--min-each", type=int, default=3)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--no-cadd", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.models, a.bg, a.af_min, a.min_each, a.nperm, not a.no_cadd, a.out)


if __name__ == "__main__":
    main()
