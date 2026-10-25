"""Population-constraint test: does predicted damage predict which variants are ABSENT from the human population?

    python -m lncfm.constraint --set lncrna --n 150 --models rinalmo-micro --out results/v4/constraint_lncrna.txt
    python -m lncfm.constraint --set core   --models rinalmo-micro            # positive control: abundant functional ncRNAs (RNU*, RN7SK, RMRP, TERC, ...)

Why this task exists
  ClinVar has almost no lncRNA ground truth (only TERC), so a clinical label cannot validate a lncRNA variant score. What every lncRNA
  does have is population data: if a stretch of the RNA matters, variants that damage it are removed by selection and are seen less
  often than mutation alone predicts. This is the logic behind genome-wide constraint maps (e.g. gnomAD's Gnocchi). A score that
  captures function should therefore be depleted among observed variants relative to the set of all possible single-nucleotide changes.

Design
  * All 3 x L possible SNVs of the spliced representative transcript of each gene; observed = seen in gnomAD v4 (AC >= 1) exons.
  * Mutation rate is the confound (CpG transitions are seen far more often). Expected probability of being observed comes from a
    96-class trinucleotide-context model fitted on the same data, then calibrated per gene so that the gene's expected count equals its
    observed count (this absorbs gene-level coverage and overall constraint).
  * Test: is within-gene damage related to the residual (observed - expected)?  A score test with gene-clustered variance:
    U_g = sum_i (y_i - p_i) z_i,  beta = sum U_g / sum p(1-p) z^2,  se = sqrt(sum U_g^2) / sum p(1-p) z^2.  z = within-gene centred
    damage / global SD.  beta < 0 means higher damage -> fewer observed variants. O/E by damage decile is reported too.
  * Controls: shuffled-score control (beta ~ 0), and the positive-control set of small ncRNAs where selection is expected.
  * Score used: the unmasked ("wild-type marginal") log-odds log p(ref) - log p(alt) from one forward pass per window, because scoring
    every possible SNV with the masked score would need ~L passes per gene.

What a result means: a negative beta with an interval below 0 says the model's damage score tracks selection in that gene set. It does
not say the variants are clinically pathogenic, and an effect on lncRNAs may be small (lncRNAs are weakly constrained overall).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from . import gencode, gnomad, ncvar, paths, splits, varscore as V

COMP = str.maketrans("ACGU", "UGCA")
BASES = "ACGU"


CORE = ["RNU1-1", "RNU1-2", "RNU1-3", "RNU1-4", "RNU2-1", "RNU2-2", "RNU4-1", "RNU4-2", "RNU5A-1", "RNU5B-1", "RNU6-1", "RNU6-2",
        "RNU6-7", "RNU6-8", "RNU6-9", "RNU7-1", "RNU11", "RNU12", "RNU4ATAC", "RNU6ATAC", "RN7SK", "RPPH1", "RMRP", "TERC", "VTRNA1-1",
        "VTRNA1-2", "VTRNA1-3", "VTRNA2-1", "RN7SL1", "RN7SL2", "RN7SL3", "RNY1", "RNY3", "RNY4", "RNY5", "SNORD118", "RNU5D-1", "RNU5E-1", "RNU5F-1"]


def ctx_class(prev, ref, nxt, alt):
    """Pyrimidine-normalised trinucleotide context class (strand symmetric), e.g. 'A[C>U]G'."""
    if ref in "AG":
        prev, ref, nxt, alt = nxt.translate(COMP), ref.translate(COMP), prev.translate(COMP), alt.translate(COMP)
    return f"{prev}[{ref}>{alt}]{nxt}"


def pick_genes(kind, n, seed, min_len=300, max_len=3000):
    tx_exons, tx_meta = ncvar.load_annotation()
    rep = splits.rep_genes()
    if kind == "lncrna":
        r = rep[(rep.length >= min_len) & (rep.length <= max_len)]
        r = r.sample(min(n, len(r)), random_state=seed)
    elif kind == "core":                                          # positive control chosen BEFORE looking at any result: abundant, functional, well-known ncRNAs
        want = set(CORE)
        t = pd.concat([gencode.build_ncrna_table(), gencode.build_table("human", "lnc")], ignore_index=True)
        r = t[t.gene_name.isin(want)]
        r = r.sort_values(["gene_id", "canonical", "length"], ascending=[True, False, False]).drop_duplicates("gene_id")
    else:                                                         # positive control: nuclear small ncRNAs with known disease variants
        d = gnomad.build_eval_table(0, 1e-4)
        names = [g for g in ncvar.eligible_genes(d, 3).gene if not g.startswith("MT-")]
        t = gencode.build_ncrna_table()
        r = t[t.gene_name.isin(names)]
        r = r[r.canonical | ~r.duplicated("gene_id", keep=False)].drop_duplicates("gene_id")
    return r.reset_index(drop=True), tx_exons


def observed_variants(chrom, exons):
    """{(pos, alt): max AF} for SNVs in gnomAD v4 inside the given exons, plus the reference base gnomAD reports at each position."""
    got = {}
    for s, e in exons:
        v = gnomad.region_snvs(chrom, int(s), int(e))
        for pos, ref, alt, ac, af in zip(v.pos.values, v.ref.values, v.alt.values, v.ac.values, v.af.values):
            if ac >= 1:
                got[(int(pos), alt)] = (max(af, got.get((int(pos), alt), (0, ""))[0]), ref)
    return got


def build_gene_rows(row, tx_exons, scorer):
    tid, chrom, strand, seq = row.tx_id, row.chrom, row.strand, row.seq
    ex = tx_exons[tid]
    L = len(seq)
    obs = observed_variants(chrom, ex)
    lo = V_wt_logodds(scorer, seq)                                   # (L, 4) log p(base) per position (unmasked)
    rows, refmis = [], 0
    ba = {c: i for i, c in enumerate(BASES)}
    bounds, cum = [], 0                                           # transcript offsets of exon-exon junctions
    for st, en in sorted(ex, reverse=(strand == "-")):
        cum += en - st + 1
        bounds.append(cum)
    bounds = np.array(bounds[:-1]) if len(bounds) > 1 else np.array([])
    run = np.ones(L)                                              # homopolymer-run length containing each position
    i = 0
    while i < L:
        j = i
        while j + 1 < L and seq[j + 1] == seq[i]:
            j += 1
        run[i:j + 1] = j - i + 1
        i = j + 1
    ent = np.zeros(L)                                             # Shannon entropy of the +-10 nt window (low = repeat / low complexity)
    for k in range(L):
        w = seq[max(0, k - 10):k + 11]
        f = np.array([w.count(b) for b in "ACGU"], float) / max(len(w), 1)
        ent[k] = -(f[f > 0] * np.log2(f[f > 0])).sum()
    isgc = np.array([c in "GC" for c in seq], float)
    gc25 = np.convolve(isgc, np.ones(51), "same") / np.convolve(np.ones(L), np.ones(51), "same")
    for off in range(1, L - 1):
        ref = seq[off]
        if ref not in ba:
            continue
        g = V.genomic_of_offset(ex, strand, off)
        for alt in BASES:
            if alt == ref or alt not in ba or seq[off - 1] not in ba or seq[off + 1] not in ba:
                continue
            gref, galt = (ref, alt) if strand == "+" else (ref.translate(COMP), alt.translate(COMP))
            hit = obs.get((g, galt.replace("U", "T")))
            if hit is not None and hit[1] != gref.replace("U", "T"):
                refmis += 1
                hit = None
            dj = float(np.abs(bounds - off).min()) if len(bounds) else 999.0
            rows.append((row.gene_id, off, ctx_class(seq[off - 1], ref, seq[off + 1], alt), float(lo[off, ba[ref]] - lo[off, ba[alt]]),
                         int(hit is not None), hit[0] if hit else 0.0, float(gc25[off]), np.log1p(min(dj, 50.0)), off / L,
                         float(min(off, L - 1 - off) < 25), min(run[off], 8.0), ent[off]))
    return rows, refmis


def V_wt_logodds(scorer, seq, win=1000, stride=500):
    """log p(base) at every position from unmasked windows (positions take the window in which they are most central)."""
    L = len(seq)
    out = np.zeros((L, 4))
    best = np.full(L, -1.0)
    starts = list(range(0, max(L - win, 0) + 1, stride))
    if starts[-1] + win < L:
        starts.append(L - win)
    for a in starts:
        w = seq[a:a + win]
        lg, _ = scorer._run([w])
        lp = lg[0, 1:len(w) + 1][:, scorer.base].log_softmax(-1).numpy()
        centrality = np.minimum(np.arange(len(w)), len(w) - 1 - np.arange(len(w))).astype(float)
        sel = centrality > best[a:a + len(w)]
        out[a:a + len(w)][sel] = lp[sel]
        best[a:a + len(w)][sel] = centrality[sel]
    return out


def score_test(D, shuffle_seed=None, adjust=(), score_col="dmg"):
    """Context model (+ optional numeric covariates) + per-gene calibration, then the cluster-robust score test for `score_col`."""
    classes = {c: i for i, c in enumerate(sorted(D.ctx.unique()))}
    X = sp.csr_matrix((np.ones(len(D)), (np.arange(len(D)), D.ctx.map(classes).values)), shape=(len(D), len(classes)))
    if adjust:
        Z = D[list(adjust)].values.astype(float)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        X = sp.hstack([X, sp.csr_matrix(np.hstack([Z, Z ** 2]))]).tocsr()
    y = D.y.values.astype(float)
    m = LogisticRegression(C=100.0, max_iter=500).fit(X, y)
    eta = X @ m.coef_[0] + m.intercept_[0]
    p = np.empty(len(D))
    for g, ix in D.groupby("gene").indices.items():                    # per-gene calibration: expected count = observed count
        d = 0.0
        for _ in range(30):
            q = 1 / (1 + np.exp(-(eta[ix] + d)))
            grad, hess = q.sum() - y[ix].sum(), (q * (1 - q)).sum()
            if hess < 1e-9:
                break
            d -= grad / hess
        p[ix] = 1 / (1 + np.exp(-(eta[ix] + d)))
    s = D[score_col].values.astype(float).copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for g, ix in D.groupby("gene").indices.items():
            s[ix] = rng.permutation(s[ix])
    z = s - pd.Series(s).groupby(D.gene.values).transform("mean").values
    z = z / z.std()
    U = pd.Series((y - p) * z).groupby(D.gene.values).sum()
    I = (p * (1 - p) * z ** 2).sum()
    beta, se = U.sum() / I, np.sqrt((U ** 2).sum()) / I
    # O/E by within-gene damage decile
    pct = pd.Series(s).groupby(D.gene.values).rank(pct=True).values
    dec = np.minimum((pct * 10).astype(int), 9)
    oe = [(y[dec == k].sum() / p[dec == k].sum()) for k in range(10)]
    return beta, se, oe, p, int(len(U))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["lncrna", "panel", "core"], default="lncrna")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--models", nargs="+", default=["rinalmo-micro"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lines = []

    def P(*x):
        t = " ".join(str(v) for v in x)
        print(t, flush=True)
        lines.append(t)

    genes, tx_exons = pick_genes(a.set, a.n, a.seed)
    P(f"=== population-constraint test, set={a.set}: {len(genes)} genes, {int(genes.length.sum()) if 'length' in genes else ''} nt of spliced transcript")
    for m in a.models:
        sc = V.MLMScorer(m, "cpu", 1000, 1, emb=False)
        rows, mis = [], 0
        with ThreadPoolExecutor(a.workers) as ex:                       # prefetch gnomAD (network-bound, cached on disk)
            list(ex.map(lambda r: observed_variants(r.chrom, tx_exons[r.tx_id]), genes.itertuples()))
        for r in genes.itertuples():
            rr, mm = build_gene_rows(r, tx_exons, sc)
            rows += rr
            mis += mm
        D = pd.DataFrame(rows, columns=["gene", "off", "ctx", "dmg", "y", "af", "gc25", "djunc", "relpos", "tx_end", "runlen", "ent21"])
        P(f"[{V.model_tag(m)}] {len(D)} possible SNVs; {int(D.y.sum())} observed in gnomAD ({100*D.y.mean():.1f}%); "
          f"reference-base disagreements with gnomAD: {mis}")
        beta, se, oe, p, ng = score_test(D)
        P(f"    damage score (per +1 SD within gene): beta = {beta:+.3f} (se {se:.3f}, z = {beta/se:+.2f}, {ng} genes)  "
          f"odds ratio of being observed = {np.exp(beta):.3f} [{np.exp(beta-1.96*se):.3f},{np.exp(beta+1.96*se):.3f}]")
        P("    observed / expected by within-gene damage decile (1 = least damaging ... 10 = most): " + " ".join(f"{x:.2f}" for x in oe))
        adj = ("gc25", "djunc", "relpos", "tx_end")
        b3, s3, oe3, _, _ = score_test(D, adjust=adj)
        P(f"    ADJUSTED for local GC, distance to exon junction, relative position and transcript ends: beta = {b3:+.3f} (se {s3:.3f}, z = {b3/s3:+.2f}); "
          f"O/E deciles " + " ".join(f"{x:.2f}" for x in oe3))
        adj2 = adj + ("runlen", "ent21")
        b4, s4, oe4, _, _ = score_test(D, adjust=adj2)
        P(f"    ADJUSTED ALSO for homopolymer run length and local sequence entropy (repeat / low-complexity proxies): beta = {b4:+.3f} (se {s4:.3f}, z = {b4/s4:+.2f})")
        pct = D.groupby("gene").dmg.rank(pct=True).values
        D5 = D[pct > 0.10].reset_index(drop=True)
        b5, s5, _, _, _ = score_test(D5, adjust=adj2)
        P(f"    same, dropping each gene's 10% least-damaging positions: beta = {b5:+.3f} (se {s5:.3f}, z = {b5/s5:+.2f})")
        P("    each covariate used alone as the 'damage score' (to see what the language model could be mimicking): " +
          "; ".join(f"{c}: beta {score_test(D, score_col=c)[0]:+.3f} (z {score_test(D, score_col=c)[0]/score_test(D, score_col=c)[1]:+.1f})"
                    for c in ("gc25", "djunc", "relpos")))
        null = []
        for k in range(3):
            b_, s_, _, _, _ = score_test(D, shuffle_seed=k)
            null.append(b_ / s_)
        P("    control, damage shuffled within gene (z-statistics of 3 shuffles): " + " ".join(f"{x:+.2f}" for x in null))
        for nm, thr in (("common (AF >= 1e-4)", 1e-4),):
            D2 = D.assign(y=(D.af >= thr).astype(int))
            if D2.y.sum() > 50:
                b2, s2, oe2, _, _ = score_test(D2)
                P(f"    {nm} only: beta = {b2:+.3f} (se {s2:.3f}, z = {b2/s2:+.2f}); O/E deciles " + " ".join(f"{x:.2f}" for x in oe2))
    if a.out:
        open(a.out, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
