"""Variant-effect scoring for ncRNA single-nucleotide variants: scorers, within-gene evaluation, a VCF scorer.

    python -m lncfm.varscore eval  --models rinalmo-micro --out $LNCFM_RUNS/v2/varscore.json
    python -m lncfm.varscore score --input my_variants.vcf --models rinalmo-micro --combiner $LNCFM_RUNS/v2/combiner_micro.json --out scored.tsv
    python -m lncfm.varscore saturate --gene RNU4-2 --models rinalmo-micro --out rnu4-2_saturation.tsv

Every score is oriented "higher = more damaging" and is never flipped after the fact.

Scores (one column per score)
  lm:<model>:mask      masked-LM log-odds log p(ref) - log p(alt) with the variant position masked   <- primary LM score
  lm:<model>:wt        same, but the position is left visible (wild-type marginal)
  lm:<model>:embpos    1 - cosine(hidden state at the variant, ref vs alt sequence)
  lm:<model>:embwin    same, averaged over +-8 nt
  st:*                 ViennaRNA structure change in a +-75 nt window: delta MFE, delta ensemble free energy,
                       pairing probability of the reference base, base-pair distance between MFE structures
  b:*                  baselines with no sequence model: transversion, CpG, distance to transcript end, local GC
  cadd                 CADD PHRED (GRCh38 v1.7) from the public API, nuclear genome only; a strong, conservation-aware
                       reference that any new model has to be compared with (missing for chrM; enters the combiner as the
                       training mean there)
  combined             leave-one-gene-out logistic regression over all of the above (per-gene intercepts, so only within-gene
                       variation sets the weights). Pre-specified as "everything"; a ranking score, not a probability

Evaluation = WITHIN-GENE: for every gene with >= `min_each` pathogenic and benign variants, the AUROC of the score for
separating the two classes inside that gene; the headline is the mean over genes weighted by min(n_path, n_benign).
Gene identity must not be able to carry the result, which is why the pooled AUROC is not reported. Confidence intervals
resample genes and then variants within gene (hierarchical bootstrap), because variants within a gene are not
independent and there are only ~10-20 genes.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from . import embed, gnomad, ncvar, paths
from .model import pick_device

# --------------------------------------------------------------------------------------------- windows


def window(seq: str, off: int, max_nt: int):
    """Window of at most `max_nt` nt around `off`; returns (substring, position of the variant inside it)."""
    if len(seq) <= max_nt:
        return seq, off
    a = max(0, min(off - max_nt // 2, len(seq) - max_nt))
    return seq[a:a + max_nt], off - a


def model_tag(name: str) -> str:
    if name.startswith("local:"):
        p = name.split(":", 1)[1].rstrip("/").split("/")
        return "local_" + (p[-2] if p[-1] == "hf" and len(p) > 1 else p[-1])
    if name.startswith("lncfm:"):
        p = name.split(":", 1)[1].split("/")
        return "lncfm_" + "_".join(p[-2:]).replace(".pt", "")
    return name


# --------------------------------------------------------------------------------------------- scorers


class MLMScorer:
    """Hugging Face masked LM (RiNALMo, RNA-FM, or a directory written by lncfm.cpt)."""

    def __init__(self, name: str, device="auto", max_nt=1000, batch=16, emb=True, radius=8):
        import multimolecule  # noqa: F401  (registers RiNALMo / RNA-FM; normal PyPI package, no remote code)
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        if name.startswith("local:"):
            hub = name.split(":", 1)[1]
        else:
            hub = embed.HF_MODELS[name][0]
            if name in embed.REMOTE_CODE:
                raise SystemExit(f"{name} needs remote code; only multimolecule/* models are supported for variant scoring")
        self.tok = AutoTokenizer.from_pretrained(hub)
        self.model = AutoModelForMaskedLM.from_pretrained(hub).eval()
        self.device = pick_device(device)
        self.model.to(self.device)
        self.base = self.tok.convert_tokens_to_ids(list("ACGU"))
        self.max_nt = min(max_nt, self.model.config.max_position_embeddings - 2)
        self.batch, self.emb, self.radius, self.tag = batch, emb, radius, model_tag(name)

    @torch.no_grad()
    def _run(self, texts, mask_pos=None):
        enc = self.tok(texts, return_tensors="pt", padding=True)
        ids, am = enc["input_ids"].clone(), enc["attention_mask"]
        if mask_pos is not None:
            ids[torch.arange(len(texts)), torch.as_tensor(mask_pos) + 1] = self.tok.mask_token_id
        out = self.model(input_ids=ids.to(self.device), attention_mask=am.to(self.device), output_hidden_states=self.emb)
        return out.logits.float().cpu(), (out.hidden_states[-1].float().cpu() if self.emb else None)

    def score(self, df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        n = len(df)
        cols = {f"lm:{self.tag}:mask": np.full(n, np.nan), f"lm:{self.tag}:wt": np.full(n, np.nan)}
        if self.emb:
            cols[f"lm:{self.tag}:embpos"] = np.full(n, np.nan)
            cols[f"lm:{self.tag}:embwin"] = np.full(n, np.nan)
        wins = [window(s, o, self.max_nt) for s, o in zip(df.seq, df.offset)]
        order = np.argsort([len(w[0]) for w in wins])
        ba = {c: i for i, c in enumerate("ACGU")}
        t0 = time.time()
        for b0 in range(0, n, self.batch):
            ix = order[b0:b0 + self.batch]
            texts = [wins[i][0] for i in ix]
            ps = [wins[i][1] for i in ix]
            r = np.array([ba[df.ref_t.iloc[i].replace("T", "U")] for i in ix])
            a = np.array([ba[df.alt_t.iloc[i].replace("T", "U")] for i in ix])
            lg_m, _ = self._run(texts, ps)
            rows = np.arange(len(ix))
            lm = lg_m[rows, np.array(ps) + 1][:, self.base]
            cols[f"lm:{self.tag}:mask"][ix] = (lm[rows, r] - lm[rows, a]).numpy()
            if self.emb:
                lg_r, h_r = self._run(texts)
                lw = lg_r[rows, np.array(ps) + 1][:, self.base]
                cols[f"lm:{self.tag}:wt"][ix] = (lw[rows, r] - lw[rows, a]).numpy()
                alt_texts = [t[:p] + "ACGU"[k] + t[p + 1:] for t, p, k in zip(texts, ps, a)]
                _, h_a = self._run(alt_texts)
                for j, (p, i) in enumerate(zip(ps, ix)):
                    hr, ha = h_r[j, p + 1], h_a[j, p + 1]
                    cols[f"lm:{self.tag}:embpos"][i] = 1 - float(torch.cosine_similarity(hr, ha, dim=0))
                    lo, hi = max(1, p + 1 - self.radius), min(len(texts[j]), p + 1 + self.radius) + 1
                    cols[f"lm:{self.tag}:embwin"][i] = 1 - float(torch.cosine_similarity(h_r[j, lo:hi].mean(0), h_a[j, lo:hi].mean(0), dim=0))
            else:
                cols[f"lm:{self.tag}:wt"][ix] = np.nan
            if verbose and (b0 // self.batch) % 20 == 0:
                print(f"  {self.tag}: {min(b0 + self.batch, n)}/{n}  {time.time() - t0:.0f}s", flush=True)
        return pd.DataFrame(cols, index=df.index)


class LncFMScorer:
    """An LncFM checkpoint trained by lncfm.pretrain (masked-LM log-odds only)."""

    def __init__(self, spec: str, device="cpu"):
        from . import variant_eval
        self.model = variant_eval.load(spec.split(":", 1)[1] if spec.startswith("lncfm:") else spec).to(pick_device(device))
        self.device, self.tag = pick_device(device), model_tag(spec if spec.startswith("lncfm:") else "lncfm:" + spec)

    def score(self, df, verbose=True):
        from . import variants
        view = pd.DataFrame({"seq": df.seq.values, "offset": df.offset.values, "ref": df.ref_t.values, "alt": df.alt_t.values})
        llr = variants.mlm_llr(self.model, self.device, view, window=512)
        return pd.DataFrame({f"lm:{self.tag}:mask": -llr}, index=df.index)


# --------------------------------------------------------------------------------------------- structure


def structure_features(df: pd.DataFrame, flank=75) -> pd.DataFrame:
    """ViennaRNA: how much does the variant change the local fold? (whole transcript if it is short)"""
    import RNA

    def fold(s):
        fc = RNA.fold_compound(s)
        ss, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
        _, g = fc.pf()
        return ss, mfe, g, np.array(fc.bpp())

    out = []
    for r in df.itertuples():
        a = max(0, r.offset - flank)
        b = min(len(r.seq), r.offset + flank + 1)
        w, p = r.seq[a:b], r.offset - a
        alt = w[:p] + r.alt_t.replace("T", "U") + w[p + 1:]
        ss0, m0, g0, b0 = fold(w)
        ss1, m1, g1, b1 = fold(alt)
        pp = lambda B: float(B[1:p + 1, p + 1].sum() + B[p + 1, p + 2:].sum())      # 1-based upper-triangular bpp matrix
        out.append({"st:dmfe": m1 - m0, "st:dg": g1 - g0, "st:absdg": abs(g1 - g0), "st:pair_ref": pp(b0),
                    "st:dpair": pp(b0) - pp(b1), "st:bpdist": RNA.bp_distance(ss0, ss1)})
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------------------------- baselines


def baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    purines = set("AG")
    tv, cpg, endd, gc = [], [], [], []
    for r in df.itertuples():
        ref, alt = r.ref_t.replace("T", "U"), r.alt_t.replace("T", "U")
        tv.append(int((ref in purines) != (alt in purines)))
        s, o = r.seq, r.offset
        cpg.append(int((ref == "C" and s[o + 1:o + 2] == "G") or (ref == "G" and s[o - 1:o] == "C")))
        endd.append(min(o, len(s) - 1 - o))
        w = s[max(0, o - 15):o + 16]
        gc.append((w.count("G") + w.count("C")) / len(w))
    return pd.DataFrame({"b:transversion": tv, "b:cpg": cpg, "b:end_dist": endd, "b:gc": gc}, index=df.index)


_CADD_URL = "https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7/{chrom}:{pos}"


def cadd_scores(df: pd.DataFrame, sleep=0.25) -> pd.Series:
    """CADD PHRED per variant through the public API (cached; one request per genomic position; no chrM)."""
    import requests
    cache_f = paths.ensure(paths.DATA / "processed") / "cadd_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    vals, new = [], 0
    for r in df.itertuples():
        if r.chrom == "chrM":
            vals.append(np.nan)
            continue
        key = f"{r.chrom}:{r.pos}"
        if key not in cache:
            for t in range(4):
                try:
                    j = requests.get(_CADD_URL.format(chrom=r.chrom.replace("chr", ""), pos=r.pos), timeout=60).json()
                    cache[key] = {x["Ref"] + ">" + x["Alt"]: float(x["PHRED"]) for x in j}
                    break
                except Exception:                                           # noqa: BLE001
                    time.sleep(2 ** (t + 1))
            else:
                cache[key] = None
            new += 1
            time.sleep(sleep)
            if new % 50 == 0:
                cache_f.write_text(json.dumps(cache))
        c = cache.get(key)
        vals.append(c.get(f"{r.ref}>{r.alt}", np.nan) if c else np.nan)
    cache_f.write_text(json.dumps(cache))
    return pd.Series(vals, index=df.index, name="cadd")


# --------------------------------------------------------------------------------------------- score cache


def _keys(df):
    return (df.tx_id + ":" + df.offset.astype(str) + ":" + df.alt_t).values


def cached(tag: str, fn, df: pd.DataFrame) -> pd.DataFrame:
    """Score only the rows not yet cached for `tag`; the cache makes re-runs and extra genes cheap."""
    f = paths.ensure(paths.DATA / "processed" / "varscores") / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', tag)}.pkl"
    have = pd.read_pickle(f) if f.exists() else pd.DataFrame()
    k = _keys(df)
    miss = df[[x not in have.index for x in k]] if len(have) else df
    if len(miss):
        new = fn(miss)
        new.index = _keys(miss)
        have = pd.concat([have, new])
        have = have[~have.index.duplicated()]
        have.to_pickle(f)
    return have.loc[k].set_axis(df.index)


# --------------------------------------------------------------------------------------------- evaluation


def fast_auc(y, s):
    y = np.asarray(y)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def within_gene(df: pd.DataFrame, s, min_each=3):
    """(weighted mean of per-gene AUROC, per-gene table). `s` is oriented higher = more damaging; NaN rows are dropped."""
    s = np.asarray(s, float)
    rows = []
    for gid, ix in df.groupby("gene_id").indices.items():
        ok = ix[~np.isnan(s[ix])]
        y = df.label.values[ok]
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos >= min_each and nneg >= min_each:
            rows.append({"gene": df.gene.values[ok[0]], "gene_type": df.gene_type.values[ok[0]], "n_path": npos, "n_ben": nneg,
                         "auroc": fast_auc(y, s[ok])})
    t = pd.DataFrame(rows)
    if t.empty:
        return np.nan, t
    w = np.minimum(t.n_path, t.n_ben)
    return float((t.auroc * w).sum() / w.sum()), t


def hier_bootstrap(df: pd.DataFrame, S: pd.DataFrame, min_each=3, B=500, seed=0):
    """Resample genes, then variants within gene. Returns the B x n_scores matrix of weighted within-gene AUROCs."""
    rng = np.random.default_rng(seed)
    S = S.values.astype(float)
    y = df.label.values
    idx = {g: ix for g, ix in df.groupby("gene_id").indices.items()
           if y[ix].sum() >= min_each and (1 - y[ix]).sum() >= min_each}
    genes = list(idx)
    out = np.full((B, S.shape[1]), np.nan)
    for b in range(B):
        num, den = np.zeros(S.shape[1]), np.zeros(S.shape[1])
        for g in rng.choice(len(genes), len(genes)):
            ix = idx[genes[g]]
            ix = ix[rng.integers(0, len(ix), len(ix))]
            yy = y[ix]
            npos, nneg = int(yy.sum()), int((1 - yy).sum())
            if npos == 0 or nneg == 0:
                continue
            w = min(npos, nneg)
            for j in range(S.shape[1]):
                a = fast_auc(yy, S[ix, j])
                if a == a:
                    num[j] += a * w
                    den[j] += w
        out[b] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return out


def report(df, S: pd.DataFrame, min_each=3, boot=500, ref: str | None = None, title="") -> dict:
    """Print a table (score, within-gene AUROC [95% CI], genes, coverage) and, if `ref`, paired differences to `ref`."""
    point = {c: within_gene(df, S[c].values, min_each) for c in S}
    bs = hier_bootstrap(df, S, min_each, boot) if boot else None
    print(f"\n== {title}  ({len(df)} variants, {int(df.label.sum())} pathogenic, {df.gene_id.nunique()} genes)")
    print(f"{'score':<34s}{'within-gene AUROC [95% CI]':<30s}{'genes':>6s}{'coverage':>10s}" + (f"{'delta vs ' + ref:>34s}" if ref and ref in S else ""))
    res = {}
    for j, c in enumerate(S):
        a, t = point[c]
        lo, hi = (np.nanpercentile(bs[:, j], [2.5, 97.5]) if bs is not None and not np.all(np.isnan(bs[:, j])) else (np.nan, np.nan))
        line = f"{c:<34s}{a:7.3f} [{lo:.3f},{hi:.3f}]".ljust(64) + f"{len(t):>6d}{S[c].notna().mean():>10.2f}"
        rec = {"auroc": a, "lo": lo, "hi": hi, "genes": int(len(t)), "coverage": float(S[c].notna().mean()),
               "per_gene": t.round(4).to_dict("records")}
        if ref and ref in S and c != ref and bs is not None:
            d = bs[:, j] - bs[:, list(S).index(ref)]
            dl, dh = np.nanpercentile(d, [2.5, 97.5])
            line += f"{point[c][0] - point[ref][0]:+9.3f} [{dl:+.3f},{dh:+.3f}]"
            rec.update({"delta_vs_ref": point[c][0] - point[ref][0], "delta_lo": float(dl), "delta_hi": float(dh)})
        print(line)
        res[c] = rec
    return res


def logo_combiner(df: pd.DataFrame, X: pd.DataFrame, min_each=3, C=0.3):
    """Leave-one-gene-out logistic regression with per-gene intercepts (so only WITHIN-gene variation sets the slopes).

    Returns (held-out ranking score for every variant of an eligible gene, final coefficients fit on all eligible genes).
    A ranking score, not a probability.
    """
    from sklearn.linear_model import LogisticRegression
    y = df.label.values
    ix = {g: i for g, i in df.groupby("gene_id").indices.items()
          if y[i].sum() >= min_each and (1 - y[i]).sum() >= min_each}
    genes = list(ix)
    Xv = X.values.astype(float)

    def fit(train_genes):
        tr = np.concatenate([ix[g] for g in train_genes])
        mu, sd = np.nanmean(Xv[tr], 0), np.nanstd(Xv[tr], 0) + 1e-9
        Z = np.nan_to_num((Xv[tr] - mu) / sd)
        G = np.zeros((len(tr), len(train_genes)))
        pos = 0
        for k, g in enumerate(train_genes):
            G[pos:pos + len(ix[g]), k] = 1.0
            pos += len(ix[g])
        clf = LogisticRegression(C=C, max_iter=2000).fit(np.hstack([Z, G]), y[tr])
        return mu, sd, clf.coef_[0][:Z.shape[1]]

    out = np.full(len(df), np.nan)
    for g in genes:
        mu, sd, w = fit([h for h in genes if h != g])
        out[ix[g]] = np.nan_to_num((Xv[ix[g]] - mu) / sd) @ w
    mu, sd, w = fit(genes)
    return out, {"features": list(X.columns), "mean": mu.tolist(), "std": sd.tolist(), "coef": w.tolist(),
                 "trained_on_genes": [str(g) for g in genes]}


# --------------------------------------------------------------------------------------------- feature assembly


def make_scorer(name, device, max_nt, batch, emb):
    if name.startswith("lncfm:") or name.endswith(".pt"):
        return LncFMScorer(name, "cpu")
    return MLMScorer(name, device, max_nt, batch, emb)


def features(df: pd.DataFrame, models, with_struct=True, with_base=True, with_cadd=False, device="auto", max_nt=1000,
             batch=16, emb=True) -> pd.DataFrame:
    parts = []
    for m in models:
        sc = make_scorer(m, device, max_nt, batch, emb)
        parts.append(cached(f"{sc.tag}_n{max_nt}_e{int(emb)}", sc.score, df))
        del sc
    if with_struct:
        parts.append(cached("struct75", structure_features, df))
    if with_base:
        parts.append(baseline_features(df))
    if with_cadd:
        parts.append(cadd_scores(df).to_frame())
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------------------------- CLI: eval


def cmd_eval(a):
    if a.bg == "clinvar":
        d = gnomad.build_eval_table(a.min_stars, use_gnomad=False)
    else:
        d = gnomad.build_eval_table(a.min_stars, 0.0 if a.bg == "any" else a.af_min)
    panel = ncvar.eligible_genes(d, a.min_each)
    print(f"background = {a.bg} (af_min={a.af_min if a.bg == 'common' else '-'}), min_stars={a.min_stars}")
    print(panel.to_string(index=False))
    if a.genes == "panel":
        d = d[d.gene_id.isin(panel.gene_id)]
    d = d.reset_index(drop=True)
    if a.limit:
        d = d.sample(min(a.limit, len(d)), random_state=0).reset_index(drop=True)
    print(f"{len(d)} variants to score ({d.source.value_counts().to_dict()})")
    X = features(d, a.models, not a.no_struct, True, a.cadd, a.device, a.max_nt, a.batch, not a.no_emb)
    out = {"args": vars(a), "n": len(d), "panel": panel.to_dict("records")}
    ref = a.ref or (f"lm:{model_tag(a.models[0])}:mask" if a.models else None)
    S = X.copy()
    primary = [c for c in S if c.endswith(":mask")] + [c for c in ("cadd",) if c in S]
    if a.combine and len(a.models):
        feats = [c for c in X if (c.startswith(("lm:", "st:", "b:")) and not c.endswith(":wt")) or c == "cadd"]
        Xc = X[feats]
        name = "combined(lm+struct+base" + ("+cadd)" if "cadd" in feats else ")")
        S[name], comb = logo_combiner(d, Xc, a.min_each)
        out["combiner"] = comb
        if a.save_combiner:
            from pathlib import Path
            Path(a.save_combiner).parent.mkdir(parents=True, exist_ok=True)
            Path(a.save_combiner).write_text(json.dumps(comb))
            print("combiner saved ->", a.save_combiner)
    out["all"] = report(d, S, a.min_each, a.boot, ref, "ALL (ClinVar" + (" + gnomAD common" if a.bg == "common" else "") + ")")
    nuc = d.chrom != "chrM"
    if (~nuc).any():
        out["nuclear"] = report(d[nuc].reset_index(drop=True), S[nuc.values].reset_index(drop=True), a.min_each, a.boot, ref, "NUCLEAR genes only")
        mt = ~nuc
        out["mito"] = report(d[mt].reset_index(drop=True), S[mt.values].reset_index(drop=True), a.min_each, a.boot, ref, "MITOCHONDRIAL tRNA/rRNA genes only")
    if a.bg == "common" and (d.source == "gnomad").any():
        cv = (d.source == "clinvar").values
        out["clinvar_only"] = report(d[cv].reset_index(drop=True), S[cv].reset_index(drop=True), a.min_each, a.boot, ref, "ClinVar benign only (no gnomAD)")
    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, default=float))
        keep = ["gene", "gene_type", "source", "chrom", "pos", "ref", "alt", "label", "clnsig", "stars", "af", "disease"]
        pd.concat([d[keep], S], axis=1).to_csv(str(a.out).replace(".json", "_scores.tsv"), sep="\t", index=False)
        print("wrote", a.out)


# --------------------------------------------------------------------------------------------- CLI: score a VCF / TSV


def read_variants(path: str) -> pd.DataFrame:
    rows = []
    op = __import__("gzip").open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t") if "\t" in line else line.split()
            c, pos, ref, alt = (f[0], f[1], f[3], f[4]) if re.sub(r"\.gz$", "", path).endswith(".vcf") else (f[0], f[1], f[2], f[3])
            if len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT":
                rows.append((ncvar._chrom(c.replace("chr", "")), int(pos), ref, alt))
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])


def cmd_score(a):
    v = read_variants(a.input)
    v["label"], v["geneinfo"] = 0, ""
    mapped, rep = ncvar.map_to_transcripts(v, attributed_only=False)
    print(f"{len(v)} SNVs read; {len(mapped)} fall in exons of annotated non-coding genes ({rep})")
    if mapped.empty:
        return
    d = ncvar.attach_sequences(mapped)
    comb = json.loads(open(a.combiner).read()) if a.combiner else None
    X = features(d, a.models, not a.no_struct, True, a.cadd, a.device, a.max_nt, a.batch, not a.no_emb)
    out = pd.concat([d[["chrom", "pos", "ref", "alt", "gene", "gene_type", "tx_id", "offset"]], X], axis=1)
    if comb:
        miss = [c for c in comb["features"] if c not in X]
        if miss:
            raise SystemExit(f"combiner needs features {miss}; pass the same --models/--no-struct/--no-emb as when it was trained")
        Z = np.nan_to_num((X[comb["features"]].values - np.array(comb["mean"])) / np.array(comb["std"]))
        out["combined"] = Z @ np.array(comb["coef"])
        out["pct_in_gene"] = out.groupby("gene").combined.rank(pct=True)
    out.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out}  (scores: higher = more damaging; 'combined' is a ranking score, not a probability)")


def genomic_of_offset(exons, strand, off):
    """Inverse of ncvar.tx_offset: genomic position of the spliced-transcript offset."""
    cum = 0
    for st, en in sorted(exons, reverse=(strand == "-")):
        n = en - st + 1
        if off < cum + n:
            return st + (off - cum) if strand == "+" else en - (off - cum)
        cum += n
    raise ValueError(off)


def cmd_saturate(a):
    """Score every possible SNV of one ncRNA (3 x length variants): a per-position intolerance map."""
    tx_exons, tx_meta = ncvar.load_annotation()
    cands = [t for t, m in tx_meta.items() if m["gene_name"] == a.gene and m["gene_type"] in ncvar.NC_TYPES]
    if not cands:
        raise SystemExit(f"{a.gene}: not a non-coding gene in GENCODE v{paths.GENCODE_HUMAN_RELEASE}")
    tid = sorted(cands, key=lambda t: (not tx_meta[t]["canonical"], -ncvar.spliced_len(tx_exons[t])))[0]
    m = tx_meta[tid]
    fa = paths.human(f"gencode.v{paths.GENCODE_HUMAN_RELEASE}.transcripts.fa.gz")
    from . import gencode
    seq = next(sq.upper().replace("T", "U") for h, sq in gencode.read_fasta_gz(fa) if h.split("|")[0] == tid)
    comp = str.maketrans("ACGU", "UGCA")
    rows = []
    for off, ref in enumerate(seq):
        g = genomic_of_offset(tx_exons[tid], m["strand"], off)
        for alt in "ACGU":
            if alt == ref:
                continue
            gr, ga = (ref, alt) if m["strand"] == "+" else (ref.translate(comp), alt.translate(comp))
            rows.append({"chrom": m["chrom"], "pos": g, "ref": gr.replace("U", "T"), "alt": ga.replace("U", "T"),
                         "gene_id": m["gene_id"], "gene": a.gene, "gene_type": m["gene_type"], "tx_id": tid,
                         "strand": m["strand"], "offset": off, "ref_t": ref, "alt_t": alt, "label": 0, "seq": seq,
                         "tx_len": len(seq)})
    d = pd.DataFrame(rows)
    print(f"{a.gene} ({m['gene_type']}, {tid}): {len(seq)} nt -> {len(d)} possible SNVs")
    X = features(d, a.models, not a.no_struct, True, a.cadd, a.device, a.max_nt, a.batch, not a.no_emb)
    out = pd.concat([d[["chrom", "pos", "ref", "alt", "gene", "tx_id", "offset", "ref_t", "alt_t"]], X], axis=1)
    main_col = f"lm:{model_tag(a.models[0])}:mask" if a.models else None
    if main_col:
        out["site_mean_" + main_col] = out.groupby("offset")[main_col].transform("mean")
    out.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out}")
    if main_col:
        site = out.groupby("offset").agg(ref=("ref_t", "first"), pos=("pos", "first"), mean_damage=(main_col, "mean"))
        print("most intolerant positions (mean masked-LM damage over the 3 substitutions):")
        print(site.sort_values("mean_damage", ascending=False).head(12).round(2).to_string())


def cmd_check(a):
    """Regression tests for the scoring path: batch/padding invariance, score orientation, window handling."""
    d = gnomad.build_eval_table(0, use_gnomad=False)
    d = d[d.gene_id.isin(ncvar.eligible_genes(d, 3).gene_id)].sample(24, random_state=1).reset_index(drop=True)
    d.loc[0, "offset"] = d.loc[0, "offset"]                       # keep the frame explicit
    m1 = MLMScorer(a.models[0], a.device, a.max_nt, batch=1, emb=True)
    m16 = MLMScorer(a.models[0], a.device, a.max_nt, batch=16, emb=True)
    s1, s16 = m1.score(d, verbose=False), m16.score(d, verbose=False)
    diff = float((s1 - s16).abs().max().max())
    print(f"[1] batch=1 vs batch=16, max |diff| over all LM scores = {diff:.2e}  (must be < 1e-3)")
    assert diff < 1e-3
    # orientation: for the same position, the alt base the model finds least likely must look more damaging than the most likely
    row = d.iloc[2]
    w, p = window(row.seq, row.offset, m1.max_nt)
    lg, _ = m1._run([w], [p])
    lp = lg[0, p + 1][m1.base].log_softmax(-1)
    ref = "ACGU".index(row.ref_t.replace("T", "U"))
    others = [i for i in range(4) if i != ref]
    hi, lo = max(others, key=lambda i: lp[i]), min(others, key=lambda i: lp[i])
    dmg = lambda i: float(lp[ref] - lp[i])
    print(f"[2] damage(least likely alt) = {dmg(lo):.3f}  >=  damage(most likely alt) = {dmg(hi):.3f}")
    assert dmg(lo) >= dmg(hi)
    # window: a variant near the end of a long sequence must keep its position
    long_seq = "ACGU" * 600
    ww, pp = window(long_seq, len(long_seq) - 5, 1000)
    assert len(ww) == 1000 and ww[pp] == long_seq[len(long_seq) - 5], "window position bookkeeping"
    ww, pp = window(long_seq, 3, 1000)
    assert ww[pp] == long_seq[3]
    print("[3] long-sequence window keeps the variant position")
    # reference base in the table agrees with the transcript (the strand / splice mapping)
    bad = sum(r.seq[r.offset] != r.ref_t.replace("T", "U") for r in d.itertuples())
    print(f"[4] reference base disagreements in {len(d)} sampled variants: {bad}")
    assert bad == 0
    # offset <-> genomic position round trip on both strands, multi-exon transcripts included
    tx_exons, tx_meta = ncvar.load_annotation()
    rng = np.random.default_rng(0)
    n = 0
    for name in ("RMRP", "RNU4-2", "TERC", "MT-TL1", "MALAT1", "NEAT1"):
        for t in [t for t, m in tx_meta.items() if m["gene_name"] == name][:2]:
            ex, st = tx_exons[t], tx_meta[t]["strand"]
            for off in rng.choice(ncvar.spliced_len(ex), min(40, ncvar.spliced_len(ex)), replace=False):
                assert ncvar.tx_offset(ex, st, genomic_of_offset(ex, st, int(off))) == off
                n += 1
    print(f"[5] offset <-> genomic round trip OK on {n} positions (+ and - strand)")
    print("varscore check OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--models", nargs="+", default=["rinalmo-micro"])
    ck.add_argument("--device", default="cpu")
    ck.add_argument("--max-nt", type=int, default=1000)
    for name in ("eval", "score", "saturate"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="*", default=[], help="rinalmo-micro rinalmo-mega rnafm local:<dir> lncfm:<ckpt.pt>")
        p.add_argument("--device", default="auto")
        p.add_argument("--max-nt", type=int, default=1000)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--no-emb", action="store_true", help="skip the (3x slower) embedding-distance scores")
        p.add_argument("--no-struct", action="store_true")
        p.add_argument("--cadd", action="store_true", help="add CADD PHRED via the public API (needs internet)")
        p.add_argument("--out", required=True)
    e = sub.choices["eval"]
    e.add_argument("--bg", choices=["common", "clinvar", "any"], default="common",
                   help="benign set: ClinVar B/LB + gnomAD common (default), ClinVar only, or ClinVar + any gnomAD-observed")
    e.add_argument("--af-min", type=float, default=1e-4)
    e.add_argument("--min-stars", type=int, default=0)
    e.add_argument("--min-each", type=int, default=3)
    e.add_argument("--genes", choices=["panel", "all"], default="panel")
    e.add_argument("--limit", type=int, default=0, help="debug: score a random subset")
    e.add_argument("--boot", type=int, default=500)
    e.add_argument("--ref", default=None, help="score the others are compared with (default: first model's :mask)")
    e.add_argument("--combine", action="store_true", help="also fit the leave-one-gene-out combiner")
    e.add_argument("--save-combiner", default=None)
    st = sub.choices["saturate"]
    st.add_argument("--gene", required=True, help="GENCODE gene symbol of a non-coding gene, e.g. RMRP, RNU4-2, TERC, MT-TL1")
    s = sub.choices["score"]
    s.add_argument("--input", required=True, help="VCF (GRCh38) or 4-column TSV: chrom pos ref alt")
    s.add_argument("--combiner", default=None, help="combiner.json from `eval --combine --save-combiner`")
    a = ap.parse_args(argv)
    {"eval": cmd_eval, "score": cmd_score, "check": cmd_check, "saturate": cmd_saturate}[a.cmd](a)


if __name__ == "__main__":
    main()
