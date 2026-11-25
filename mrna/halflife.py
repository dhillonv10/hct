#!/usr/bin/env python3
"""Beat Saluki on mRNA half-life -- a benchmark that is NOT saturated.

Why this task. On MRL we measured the ceiling directly: two replicates of the same 50-mer library agree at
r2=0.94 even at 4000+ reads, and published SOTA (0.925) sits within ~0.01-0.03 of it. Nothing to win there.
Half-life is different: Saluki reports r=0.777 (human), so roughly 40% of the variance is still unexplained.

Why a feature model has a real shot. Saluki's own ablation table says the splice-site track is worth more
than everything else it adds (sequence 0.608 -> +splice 0.744 -> +codon 0.783), and exon-junction density is
a SCALAR -- a hand-built feature captures it exactly, no convolution needed. Their own engineered baseline
"BEeM" reaches 0.772, within 0.005 of the deep model. So the gap between engineered features and Saluki was
already nearly closed in their own paper; this script tries to close and cross it.

Target to beat (their Fig5_S6/experimentResults.csv, 10 CV folds):
    Saluki human  mean r = 0.7768   (min 0.7346, max 0.7870)
    Saluki mouse  mean r = 0.7320
    BEeM   human  mean r = 0.7721
    BC3MS  human  mean r = 0.6732

Target values come from the authors' own PC1 of their 39-dataset human decay compendium, so the labels are
theirs, not ours. Transcript sequences are rebuilt from GENCODE V50 + hg38 because the authors' sequence
datapack is 18.8 GB; the representative-transcript rule is therefore OURS, not theirs, and that is the one
material difference in the comparison. It is stated in the report rather than buried.

  python3 halflife.py prep  --work WORK --gencode /home/user/lncfam_data/ucsc
  python3 halflife.py feat  --work WORK
  python3 halflife.py train --work WORK
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

SALUKI = {"human": [0.7830071, 0.7837705, 0.7870404, 0.7346236, 0.7807139,
                    0.7775869, 0.7666451, 0.7766784, 0.7603674, 0.7759681],
          "mouse": [0.7377259, 0.7410566, 0.7244405, 0.7236191, 0.7196615,
                    0.7450399, 0.7359199, 0.7327611, 0.7238828, 0.7352646]}
BEEM = [0.776160537470687, 0.776464726726784, 0.77830944355361, 0.767121667422453, 0.748941667886045,
        0.765163247225974, 0.770342069357331, 0.77569410641493, 0.758897211458665, 0.763588581325803]
MAIN_CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}
NFOLD = 10

# ------------------------------------------------------------------------------------------- transcript reconstruction


def read_attrs(path, want):
    """transcript id -> (gene id, gene type), restricted to the genes we need."""
    tx = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            g = f[0].split(".")[0]
            if g in want:
                tx[f[4]] = (g, f[2])
    return tx


def read_models(path, keep_tx):
    """genePredExt rows -> per transcript: chrom, strand, cdsStart, cdsEnd, exon starts, exon ends."""
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11 or f[1] not in keep_tx:
                continue
            if f[2] not in MAIN_CHROMS:
                continue
            out[f[1]] = (f[2], f[3], int(f[6]), int(f[7]),
                         [int(x) for x in f[9].rstrip(",").split(",")],
                         [int(x) for x in f[10].rstrip(",").split(",")])
    return out


def pick_representative(tx_attrs, models):
    """One transcript per gene: longest CDS, then longest total length. Approximates the canonical choice."""
    best = {}
    for tid, (chrom, strand, cs, ce, st, en) in models.items():
        gid, gtype = tx_attrs[tid]
        total = sum(e - s for s, e in zip(st, en))
        cds = sum(max(0, min(e, ce) - max(s, cs)) for s, e in zip(st, en))
        key = (cds, total)
        if gid not in best or key > best[gid][0]:
            best[gid] = (key, tid, chrom, strand, cs, ce, st, en, gtype)
    return best


def build_transcripts(fa_gz, rep):
    """Spliced mRNA plus a CDS mask and exon-junction offsets, read one chromosome at a time."""
    by_chrom = {}
    for gid, v in rep.items():
        by_chrom.setdefault(v[2], []).append((gid, v))
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    out, name, buf = {}, None, []

    def flush():
        if name is None or name not in by_chrom:
            return
        g = "".join(buf).upper()
        for gid, (_key, tid, chrom, strand, cs, ce, st, en) in [(a, b[:8]) for a, b in by_chrom[name]]:
            seq, cds, junc, pos = [], [], [], 0
            for s, e in zip(st, en):
                seq.append(g[s:e])
                cds.append(np.array([1 if (s + i) >= cs and (s + i) < ce else 0 for i in range(e - s)],
                                    dtype=np.int8))
                pos += e - s
                junc.append(pos)
            s_all = "".join(seq)
            c_all = np.concatenate(cds) if cds else np.zeros(0, np.int8)
            L = len(s_all)
            if strand == "-":
                s_all = s_all.translate(comp)[::-1]
                c_all = c_all[::-1].copy()
                junc = sorted(L - j for j in junc[:-1]) + [L]
            out[gid] = (s_all, c_all, junc)
        print(f"    {name}: {len(by_chrom[name])} genes", flush=True)

    with gzip.open(fa_gz, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                flush()
                name, buf = line[1:].split()[0], []
            elif name in by_chrom:
                buf.append(line.strip())
    flush()
    return out


def stage_prep(work, gencode, species="human"):
    os.makedirs(work, exist_ok=True)
    pc1 = pd.read_csv(f"/home/user/vagarwal87/saluki_paper/Fig2_S3/all_HLs_{species}_PC1.txt", sep="\t")
    col = f"{species}_PC1"
    pc1 = pc1.dropna(subset=[col])
    want = set(pc1.ensembl_gene_id)
    print(f"{species}: {len(want)} genes with a PC1 half-life")
    attrs = read_attrs(os.path.join(gencode, "wgEncodeGencodeAttrsV50.txt.gz"), want)
    models = read_models(os.path.join(gencode, "wgEncodeGencodeCompV50.txt.gz"), set(attrs))
    rep = pick_representative(attrs, models)
    print(f"  {len(models)} transcript models -> {len(rep)} representative transcripts")
    print("building spliced transcripts ...")
    tx = build_transcripts(os.path.join(gencode, "hg38.fa.gz"), rep)
    rows = []
    for gid, (seq, cds, junc) in tx.items():
        if len(seq) < 100 or seq.count("N") > 0.01 * len(seq):
            continue
        rows.append(dict(gene=gid, seq=seq, cds_start=int(np.argmax(cds)) if cds.any() else -1,
                         cds_len=int(cds.sum()), n_exons=len(junc),
                         junc=",".join(map(str, junc[:-1])), length=len(seq)))
    d = pd.DataFrame(rows).merge(pc1[["ensembl_gene_id", col]], left_on="gene",
                                 right_on="ensembl_gene_id").drop(columns=["ensembl_gene_id"])
    d = d.rename(columns={col: "y"})
    out = os.path.join(work, f"tx_{species}.pkl")
    d.to_pickle(out)
    print(f"\n{len(d)} genes with sequence and label -> {out}")
    print(f"  length median {int(d.length.median())} nt, max {int(d.length.max())}")
    print(f"  coding {(d.cds_len > 0).sum()}  noncoding {(d.cds_len == 0).sum()}")
    print(f"  exons median {int(d.n_exons.median())}")
    print(f"  y mean {d.y.mean():.3f} sd {d.y.std():.3f}")
    return d


# ------------------------------------------------------------------------------------------------------- features

B = list("ACGT")
K2 = [a + b for a in B for b in B]
K3 = [a + b + c for a in B for b in B for c in B]
K4 = [a + b + c + d for a in B for b in B for c in B for d in B]
BASE_ORDER = "TCAG"
CODONS = [a + b + c for a in BASE_ORDER for b in BASE_ORDER for c in BASE_ORDER]
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_AA = {c: AA[i] for i, c in enumerate(CODONS)}
STOPS = {"TAA", "TAG", "TGA"}
_CSC = None


def csc_table():
    """Codon stability coefficients, Forrest et al., shipped in the Saluki repo. 8 species columns."""
    global _CSC
    if _CSC is None:
        t = pd.read_csv("/home/user/vagarwal87/saluki_paper/Fig5_S6/codon_stability_coefs_Forrest_et_al.txt",
                        sep="\t").set_index("Codon")
        _CSC = {c: t[c].to_dict() for c in t.columns}
    return _CSC


def kmer_freq(s, ks, prefix, out):
    n = len(s)
    for k, names in ks:
        if n < k:
            for nm in names:
                out[f"{prefix}{k}_{nm}"] = 0.0
            continue
        cnt = Counter(s[i:i + k] for i in range(n - k + 1))
        tot = max(n - k + 1, 1)
        for nm in names:
            out[f"{prefix}{k}_{nm}"] = cnt.get(nm, 0) / tot


def fold_win(s, span=100):
    import RNA
    if len(s) < 15:
        return 0.0, 0.0
    md = RNA.md(); md.max_bp_span = span
    st, mfe = RNA.fold_compound(s.replace("T", "U"), md).mfe()
    return mfe / len(s), st.count(".") / len(st)


def count_overlap(s, pat):
    """Occurrences of a motif given as a list of allowed-character sets."""
    k = len(pat)
    return sum(1 for i in range(len(s) - k + 1) if all(s[i + j] in pat[j] for j in range(k)))


DRACH = [set("AGT"), set("AG"), set("A"), set("C"), set("ACT")]       # m6A consensus
PUM = [set("T"), set("G"), set("T"), set("A"), set("ACGT"), set("A"), set("T"), set("A")]


def features(rec):
    s = rec["seq"]
    L = len(s)
    cs, cl = rec["cds_start"], rec["cds_len"]
    junc = [int(x) for x in rec["junc"].split(",")] if rec["junc"] else []
    coding = cl > 0
    u5 = s[:cs] if coding else ""
    cd = s[cs:cs + cl] if coding else ""
    u3 = s[cs + cl:] if coding else s
    f = {"is_coding": float(coding)}

    # geometry -- Saluki's ablation says the splice track is its single biggest addition, and exon
    # structure is a handful of scalars, so this block is where a feature model should be strongest.
    f["len"] = L; f["log_len"] = math.log(L)
    for nm, r in (("u5", u5), ("cds", cd), ("u3", u3)):
        f[f"len_{nm}"] = len(r); f[f"log_len_{nm}"] = math.log1p(len(r))
        f[f"frac_{nm}"] = len(r) / L
        f[f"gc_{nm}"] = sum(r.count(c) for c in "GC") / max(len(r), 1)
    f["gc"] = sum(s.count(c) for c in "GC") / L
    f["gc3"] = sum(1 for i in range(2, len(cd), 3) if cd[i] in "GC") / max(len(cd) // 3, 1)
    ne = rec["n_exons"]
    f["n_exons"] = ne
    f["exon_density_kb"] = 1000.0 * ne / L
    f["log_exon_density"] = math.log1p(1000.0 * ne / L)
    ex = np.diff([0] + junc + [L]) if junc else np.array([L])
    f["exon_len_mean"] = float(ex.mean()); f["exon_len_median"] = float(np.median(ex))
    f["exon_len_max"] = float(ex.max()); f["exon_len_min"] = float(ex.min())
    f["exon_len_sd"] = float(ex.std())
    f["first_exon_len"] = float(ex[0]); f["last_exon_len"] = float(ex[-1])
    f["n_junc_u5"] = sum(1 for j in junc if j < len(u5))
    f["n_junc_cds"] = sum(1 for j in junc if len(u5) <= j < len(u5) + len(cd))
    f["n_junc_u3"] = sum(1 for j in junc if j >= len(u5) + len(cd))
    f["junc_density_cds"] = f["n_junc_cds"] / max(len(cd) / 1000.0, 1e-3)
    f["junc_density_u3"] = f["n_junc_u3"] / max(len(u3) / 1000.0, 1e-3)
    f["last_junc_to_end"] = (L - junc[-1]) / L if junc else 1.0

    # codon usage and codon stability coefficients
    cods = [cd[i:i + 3] for i in range(0, len(cd) - 2, 3)]
    cods = [c for c in cods if c in CODON_AA]
    cnt = Counter(cods); tot = max(len(cods), 1)
    for c in CODONS:
        f[f"cod_{c}"] = cnt[c] / tot
    f["n_codons"] = len(cods)
    f["premature_stop"] = sum(1 for c in cods[:-1] if c in STOPS)
    tab = csc_table()
    for col, w in tab.items():
        f[f"csc_{col}"] = sum(cnt[c] * w.get(c, 0.0) for c in cnt) / tot
        f[f"csc_{col}_opt"] = sum(cnt[c] for c in cnt if w.get(c, 0.0) > 0) / tot

    # composition, per region
    kmer_freq(cd if coding else "", [(3, K3)], "cds_m", f)
    kmer_freq(u3, [(3, K3), (4, K4)], "u3_m", f)
    kmer_freq(u5, [(2, K2), (3, K3)], "u5_m", f)

    # cis elements
    f["are_u3"] = u3.count("ATTTA"); f["are_u3_kb"] = 1000.0 * u3.count("ATTTA") / max(len(u3), 1)
    f["are_strong_u3"] = u3.count("TATTTAT")
    f["pas_u3"] = u3.count("AATAAA") + u3.count("ATTAAA")
    f["drach_u3"] = count_overlap(u3, DRACH); f["drach_u3_kb"] = 1000.0 * f["drach_u3"] / max(len(u3), 1)
    f["drach_cds"] = count_overlap(cd, DRACH); f["drach_cds_kb"] = 1000.0 * f["drach_cds"] / max(len(cd), 1)
    f["pum_u3"] = count_overlap(u3, PUM); f["pum_u3_kb"] = 1000.0 * f["pum_u3"] / max(len(u3), 1)
    f["gquad_u3"] = count_overlap(u3, [set("G")] * 3) / max(len(u3), 1) * 1000
    augs = [i for i in range(len(u5) - 2) if u5[i:i + 3] == "ATG"]
    f["uaug"] = len(augs)
    for fr in range(3):
        f[f"uaug_f{fr}"] = sum(1 for i in augs if (len(u5) - i) % 3 == fr)
    f["uorf_closed"] = sum(1 for i in augs
                           if any(u5[j:j + 3] in STOPS for j in range(i + 3, len(u5) - 2, 3)))
    f["top_motif"] = float(len(u5) > 5 and all(c in "CT" for c in u5[1:6]))
    f["kozak_m3_purine"] = float(len(u5) >= 3 and u5[-3] in "AG")

    # local structure, capped windows so cost is flat in transcript length
    for nm, w in (("cap", s[:70]), ("u5", u5[-200:]), ("cds_start", cd[:200]),
                  ("cds_end", cd[-200:]), ("u3_start", u3[:300]), ("u3_end", u3[-300:])):
        m, up = fold_win(w)
        f[f"mfe_{nm}"] = m; f[f"unp_{nm}"] = up
    return f


def _one(rec):
    return features(rec)


def stage_feat(work, species="human", nproc=None):
    from multiprocessing import Pool
    d = pd.read_pickle(os.path.join(work, f"tx_{species}.pkl"))
    recs = d.to_dict("records")
    nproc = nproc or max(1, (os.cpu_count() or 2))
    print(f"features for {len(recs)} genes on {nproc} processes")
    t0 = time.time()
    rows = []
    with Pool(nproc) as p:
        for i, r in enumerate(p.imap(_one, recs, chunksize=16)):
            rows.append(r)
            if i and i % 1000 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(recs)}  {el:.0f}s  eta {el / i * (len(recs) - i) / 60:.1f} min", flush=True)
    F = pd.DataFrame(rows)
    cols = list(F.columns)
    X = np.nan_to_num(F.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.save(os.path.join(work, f"X_{species}.npy"), X)
    json.dump(cols, open(os.path.join(work, f"cols_{species}.json"), "w"))
    d[["gene", "y"]].to_pickle(os.path.join(work, f"y_{species}.pkl"))
    print(f"\n{X.shape} -> X_{species}.npy  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------- train

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def stage_train(work, species="human", seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    X = np.load(os.path.join(work, f"X_{species}.npy"))
    cols = json.load(open(os.path.join(work, f"cols_{species}.json")))
    yd = pd.read_pickle(os.path.join(work, f"y_{species}.pkl"))
    y = yd.y.values.astype(float)
    n = len(y)
    print(f"{species}: {n} genes x {X.shape[1]} features")

    rs = np.random.default_rng(seed)
    fold = rs.permutation(n) % NFOLD
    arms = {k: np.zeros(n) for k in ("gbm", "ridge", "stack")}
    for k in range(NFOLD):
        te = fold == k
        inner = np.flatnonzero(~te)
        rs2 = np.random.default_rng(1000 + k)
        va = inner[rs2.permutation(len(inner))[:len(inner) // 9]]       # inner validation for the stack weights
        tr = np.setdiff1d(inner, va)
        t0 = time.time()
        g = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.12,
                                          random_state=seed).fit(X[tr], y[tr])
        sc = StandardScaler().fit(X[tr])
        r = RidgeCV(alphas=np.logspace(-2, 5, 20)).fit(sc.transform(X[tr]), y[tr])
        pg_va, pr_va = g.predict(X[va]), r.predict(sc.transform(X[va]))
        pg_te, pr_te = g.predict(X[te]), r.predict(sc.transform(X[te]))
        w = np.linalg.lstsq(np.c_[pg_va, pr_va, np.ones(len(va))], y[va], rcond=None)[0]
        arms["gbm"][te] = pg_te
        arms["ridge"][te] = pr_te
        arms["stack"][te] = np.c_[pg_te, pr_te, np.ones(te.sum())] @ w
        print(f"  fold {k}: gbm r={pearson(y[te], pg_te):.4f}  ridge r={pearson(y[te], pr_te):.4f}  "
              f"stack r={pearson(y[te], arms['stack'][te]):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    print(f"\n{'arm':10s}{'per-fold mean r':>18s}{'sd':>8s}{'pooled r':>11s}")
    for a, p in arms.items():
        per = [pearson(y[fold == k], p[fold == k]) for k in range(NFOLD)]
        res[a] = dict(per_fold=per, mean=float(np.mean(per)), sd=float(np.std(per)),
                      pooled=pearson(y, p))
        print(f"{a:10s}{np.mean(per):18.4f}{np.std(per):8.4f}{pearson(y, p):11.4f}")

    sal = SALUKI[species]
    print(f"\n{'Saluki (published)':10s}{np.mean(sal):18.4f}{np.std(sal):8.4f}")
    if species == "human":
        print(f"{'BEeM':10s}{np.mean(BEEM):18.4f}{np.std(BEEM):8.4f}")
    best = max(res, key=lambda a: res[a]["mean"])
    d = res[best]["mean"] - float(np.mean(sal))
    print(f"\nbest arm '{best}': mean r {res[best]['mean']:.4f} vs Saluki {np.mean(sal):.4f}  ({d:+.4f})")
    print("VERDICT:", "BEATS the published number" if d > 0 else "does NOT beat the published number")
    print("\nCaveat that belongs next to any claim here: labels and gene set are the authors' own, but the"
          "\ntranscript sequences are rebuilt from GENCODE V50 with OUR representative-transcript rule, and"
          "\nthe CV fold assignment is ours. Same task and labels, not a byte-identical replication.")
    json.dump({"species": species, "n": n, "arms": res,
               "saluki_mean": float(np.mean(sal))},
              open(os.path.join(work, f"hl_results_{species}.json"), "w"), indent=1)

    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                      random_state=0).fit(X, y)
    try:
        from sklearn.inspection import permutation_importance
        sub = np.random.default_rng(0).permutation(n)[:3000]
        imp = permutation_importance(g, X[sub], y[sub], n_repeats=3, random_state=0, n_jobs=1)
        order = np.argsort(-imp.importances_mean)[:25]
        print("\ntop 25 features by permutation importance:")
        for i in order:
            print(f"  {cols[i]:24s}{imp.importances_mean[i]:9.4f}")
    except Exception as e:
        print("importance skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "feat", "train"])
    ap.add_argument("--work", default="/home/user/hl_work")
    ap.add_argument("--gencode", default="/home/user/lncfam_data/ucsc")
    ap.add_argument("--species", default="human")
    a = ap.parse_args()
    if a.stage == "prep":  stage_prep(a.work, a.gencode, a.species)
    if a.stage == "feat":  stage_feat(a.work, a.species)
    if a.stage == "train": stage_train(a.work, a.species)


if __name__ == "__main__":
    main()
