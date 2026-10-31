"""RNase-H gapmer ASO efficacy (ASO Atlas, the frozen table shipped with OligoAI; Zenodo 10.5281/zenodo.20794660, CC-BY-4.0).

Each row = one ASO measured in one screen (a patent table): one gene, one cell line, one dose, usually ~78 ASOs tiled along the transcript.
`rna_context` is the target RNA with ~50 nt of flank on each side. Label = % inhibition (0-100), kept as y in [0, 1].

Unit of generalisation = the GENE (after merging aliases), never the screen or the ASO:
  * heavy genes carry 22-82 screens and thousands of overlapping tiled ASOs, and ~8.6k (gene, sequence) pairs are measured in more than one screen,
    so a screen-level split puts the same site on both sides;
  * gene names are free text ('K-Ras'/'KRAS', 'C9ORF72'/'C9orf72', 'huntingtin interacting protein 2' for UBE2K), so genes are merged by normalised name
    AND by shared ASO sequences AND by shared 40-nt target stretches (union-find); every row in a merged group stays on one side.
Metric = within-screen Spearman (which ASOs of a screen to order), mean over screens with n >= 20, CI by bootstrap over gene groups.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from . import metrics as M

LOCK_EVERY = 4                                   # every 4th gene group (alphabetical by group key) is LOCKED, as for the Cas13d genes
MAX_GENES_PER_SEQ = 5                            # an ASO seen under more genes than this is a control, not evidence that the genes are the same
KMER, KMER_MOD = 40, 16                          # shared 40-nt stretch (sampled by hash) links two gene labels to one transcript
MIN_SHARED_SEQ, MIN_SHARED_KMERS = 3, 5          # evidence needed to merge two gene labels (see _merge_genes)
COMP = str.maketrans("ACGT", "TGCA")


def rc_dna(s: str) -> str:
    return s.translate(COMP)[::-1]


def _norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _tokens(s) -> list:
    return re.findall(r"'([^']+)'", s) if isinstance(s, str) else []     # the lists are stored as Python-repr strings; regex, never eval


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _merge_genes(d: pd.DataFrame) -> pd.Series:
    """Group key per row. Union-find over gene keys, linking two keys only on STRONG evidence: >= MIN_SHARED_SEQ shared ASO sequences or
    >= MIN_SHARED_KMERS shared (hash-sampled) 40-nt target stretches. A single shared hit is not enough: one chance hit chained
    ATXN2, BAX, CDK4, CRP, F12 ... into one 52k-row 'APOL1' group in a first version. Measured on this table the evidence is bimodal:
    real aliases/paralogs share many (STAT5A/STAT5B 78 ASOs and 32 stretches, KLKB1/PKK 79 ASOs, GCCR/NR3C1 15 ASOs, KIF11/KINESINLIKE1 3 ASOs,
    ANGPTL3 and an unnamed screen of it 7 ASOs), chance links share exactly one (ATXN2/ATXN3, CAG-repeat genes, share 3 stretches)."""
    uf = _UF()
    for k in d.gene_key.unique():
        uf.find(k)
    pair_seq, pair_kmer = Counter(), Counter()
    for genes in d.groupby("aso").gene_key.agg(lambda s: sorted(set(s))):
        if 1 < len(genes) <= MAX_GENES_PER_SEQ:
            pair_seq.update(combinations(genes, 2))
    kg = {}
    ctx = d.dropna(subset=["rna_context"]).drop_duplicates("rna_context")[["rna_context", "gene_key"]]
    for c, g in zip(ctx.rna_context.values, ctx.gene_key.values):
        for i in range(len(c) - KMER + 1):
            k = c[i:i + KMER]
            if not zlib.crc32(k.encode()) % KMER_MOD:
                kg.setdefault(k, set()).add(g)
    for gs in kg.values():
        if 1 < len(gs) <= MAX_GENES_PER_SEQ:
            pair_kmer.update(combinations(sorted(gs), 2))
    for (a, b), n in pair_seq.items():
        if n >= MIN_SHARED_SEQ:
            uf.union(a, b)
    for (a, b), n in pair_kmer.items():
        if n >= MIN_SHARED_KMERS:
            uf.union(a, b)
    return d.gene_key.map(uf.find)


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["raw_id"] = np.arange(len(d))                                           # position in the original table, before de-duplication (joins external predictions)
    d["aso"] = d.aso_sequence_5_to_3.str.upper().str.replace("U", "T")
    d["y"] = d.inhibition_percent.astype(float) / 100.0
    d["screen"] = d.custom_id
    name = d.target_gene.fillna(d.target_mrna)
    d["gene_key"] = np.where(name.isna(), "UNNAMED:" + d.screen, name.map(_norm))      # unnamed screens stay separate unless the evidence links them
    d["rna_context"] = d.rna_context.str.upper().str.replace("T", "U")
    d = d.drop_duplicates(subset=["aso", "screen", "chemistry", "dosage", "inhibition_percent"]).reset_index(drop=True)
    d["group"] = _merge_genes(d)
    keys = sorted(d.group.unique())
    locked = set(keys[LOCK_EVERY - 1::LOCK_EVERY])
    d["split"] = np.where(d.group.isin(locked), "locked", "dev")
    d["block"] = d.group                                       # bootstrap unit (whole gene groups)
    return d


ACC_LEN = 125                                    # padded context length in the accessibility table (contexts are 66-121 nt)


def _fold_one(ctx: str) -> np.ndarray:
    """Per-nucleotide unpaired probability (partition function of the whole context, ViennaRNA) + MFE per nt, padded to ACC_LEN+1 with NaN."""
    import RNA
    fc = RNA.fold_compound(ctx)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    B = np.array(fc.bpp(), dtype=np.float32)[1:, 1:]
    B = B + B.T
    out = np.full(ACC_LEN + 1, np.nan, np.float32)
    out[:len(ctx)] = np.clip(1.0 - B.sum(axis=1), 0.0, 1.0)
    out[ACC_LEN] = mfe / len(ctx)
    return out


def accessibility(ctxs: list, procs: int = 4) -> np.ndarray:
    """(len(ctxs), ACC_LEN + 1) table; row i belongs to ctxs[i]. Label-free, so it may be computed for dev and locked genes alike."""
    from multiprocessing import Pool
    with Pool(procs) as p:
        return np.vstack(p.map(_fold_one, ctxs, chunksize=256))


_SUG = {}                                        # parsed-chemistry cache


def _gapmer(sugar: str):
    """(wing5, gap, wing3, {sugar name: count}) from the sugar list: wings = leading / trailing non-DNA residues, gap = DNA residues."""
    if sugar not in _SUG:
        t = [x.upper() for x in _tokens(sugar)]                                   # Atlas 2.0 spells the token 'cET', Atlas 1 'CET'
        n5 = next((i for i, x in enumerate(t) if x == "DNA"), len(t))
        n3 = next((i for i, x in enumerate(t[::-1]) if x == "DNA"), len(t))
        _SUG[sugar] = (n5, t.count("DNA"), n3, Counter(t))
    return _SUG[sugar]


def features(d: pd.DataFrame, acc: np.ndarray | None = None, acc_ctx: pd.Series | None = None, thermo: bool = True) -> pd.DataFrame:
    """Label-free feature table. acc/acc_ctx: output of `accessibility` and a Series mapping context string -> row of acc."""
    n, P = len(d), 28
    seq = d.aso.values
    L = np.array([len(s) for s in seq])
    code = np.zeros((n, P), np.int8)
    tr = {ord("A"): 1, ord("C"): 2, ord("G"): 3, ord("T"): 4}
    for i, s in enumerate(seq):
        b = np.frombuffer(s[:P].encode(), np.uint8)
        code[i, :len(b)] = [tr.get(x, 0) for x in b]
    f = {"len": L}
    for k, nm in enumerate("ACGT", 1):
        f[f"n{nm}"] = (code == k).sum(1)
    f["gc"] = (f["nC"] + f["nG"]) / L
    f["cpg"] = np.array([s.count("CG") for s in seq])
    f["run"] = np.array([max(len(m.group()) for m in re.finditer(r"(.)\1*", s)) for s in seq])
    for m in ("GGGG", "CCCC", "TTTT", "GCGC"):
        f["has_" + m] = np.array([m in s for s in seq], dtype=np.int8)
    f["first"], f["last"] = code[:, 0], code[np.arange(n), np.clip(L - 1, 0, P - 1)]
    for j in range(P):
        for k, nm in enumerate("ACGT", 1):
            f[f"p{j}{nm}"] = (code[:, j] == k).astype(np.int8)
    g = [_gapmer(s) for s in d.sugar_mods.values]
    f["wing5"], f["gap"], f["wing3"] = (np.array([x[i] for x in g]) for i in range(3))
    SUGARS = ("MOE", "CET", "OME", "F", "DNA")                 # the only sugar tokens in the table (counted over 178k rows: MOE 1.04M, DNA 1.79M, CET 0.41M, OME 686, F 8)
    for nm in SUGARS:
        f["sug_" + nm] = np.array([x[3].get(nm, 0) for x in g])
    f["sug_other"] = L - sum(f["sug_" + nm] for nm in SUGARS)
    bb = d.backbone_mods.map(lambda s: _tokens(s)).values
    f["n_PS"] = np.array([t.count("PS") for t in bb])
    f["n_PO"] = np.array([t.count("PO") for t in bb])
    f["frac_PS"] = f["n_PS"] / np.maximum(f["n_PS"] + f["n_PO"], 1)
    f["log_dose"] = np.log10(d.dosage.values)
    f["log_cells"] = np.log10(d.cells_per_well.values)
    f["transfection"] = pd.Categorical(d.transfection_method).codes
    top = d.cell_line.value_counts().index[:20]
    f["cell_line"] = pd.Categorical(d.cell_line.where(d.cell_line.isin(top)), categories=list(top)).codes
    df = pd.DataFrame(f, index=d.index)
    if thermo:
        import RNA
        un = pd.unique(d.aso.values)
        dg = {}
        for s in un:
            t = rc_dna(s).replace("T", "U")
            dg[s] = (RNA.duplexfold(t, s.replace("T", "U")).energy, RNA.fold(s.replace("T", "U"))[1])
        df["dg_duplex"] = d.aso.map(lambda s: dg[s][0]).values
        df["dg_self"] = d.aso.map(lambda s: dg[s][1]).values
    if acc is not None:
        site = np.array([(c.find(rc_dna(a).replace("T", "U")) if isinstance(c, str) else -1) for c, a in zip(d.rna_context.values, seq)])
        df["site_found"] = (site >= 0).astype(np.int8)
        rows = d.rna_context.map(acc_ctx).values
        feats = {k: np.full(n, np.nan, np.float32) for k in ("u_site", "u_site5", "u_site3", "u_min", "u_fl5", "u_fl3", "u_ctx", "mfe_ctx")}
        for i in range(n):
            r = rows[i]
            if np.isnan(r) or site[i] < 0:
                continue
            u, p, q = acc[int(r)], site[i], site[i] + L[i]
            seg = u[p:q]
            feats["u_site"][i], feats["u_min"][i] = seg.mean(), seg.min()
            feats["u_site5"][i], feats["u_site3"][i] = seg[:5].mean(), seg[-5:].mean()
            feats["u_fl5"][i] = u[max(0, p - 10):p].mean() if p > 0 else np.nan
            ctx_len = int((~np.isnan(u[:ACC_LEN])).sum())
            feats["u_fl3"][i] = u[q:min(ctx_len, q + 10)].mean() if q < ctx_len else np.nan
            feats["u_ctx"][i], feats["mfe_ctx"][i] = np.nanmean(u[:ACC_LEN]), u[ACC_LEN]
        for k, v in feats.items():
            df[k] = v
    return df


def screen_rhos(d: pd.DataFrame, col: str, min_n: int = 20) -> pd.Series:
    """Spearman between y and `col` inside each screen with at least min_n ASOs (index = screen id)."""
    r = M._gene_rhos(d.y.values, d[col].values.astype(float), d.screen.values.astype(str), min_n)
    return pd.Series(r, name=col)


def summarize(d: pd.DataFrame, col: str, min_n: int = 20) -> dict:
    """rho_screen = mean over screens; rho_group = mean over gene groups of their mean screen rho (so a gene with 82 screens counts once)."""
    r = screen_rhos(d, col, min_n)
    grp = d.drop_duplicates("screen").set_index("screen").group.reindex(r.index)
    return {"n_screens": int(len(r)), "n_groups": int(grp.nunique()),
            "rho_screen": float(r.mean()), "rho_group": float(r.groupby(grp).mean().mean())}


def _boot_frames(d: pd.DataFrame, n: int, seed: int):
    """Bootstrap resamples of whole gene groups; a group drawn j times appears j times, with its screens relabelled 'id#draw' so each copy counts."""
    rng = np.random.default_rng(seed)
    ub = d.group.unique()
    idx = {k: np.flatnonzero(d.group.values == k) for k in ub}
    for _ in range(n):
        picks = rng.choice(ub, len(ub), replace=True)
        ii = np.concatenate([idx[k] for k in picks])
        tag = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(picks)])
        x = d.iloc[ii].reset_index(drop=True)
        x["screen"] = x.screen.astype(str) + "#" + pd.Series(tag).astype(str)
        yield x


def bootstrap(d: pd.DataFrame, a: str, b: str | None = None, n: int = 300, min_n: int = 20, seed: int = 0):
    """(point, lo, hi) of rho_screen for column a, or of rho_screen(a) - rho_screen(b) when b is given; resamples whole gene groups.
    """
    def stat(x):
        ra = screen_rhos(x, a, min_n)
        if b is None:
            return float(ra.mean())
        rb = screen_rhos(x, b, min_n)
        k = ra.index.intersection(rb.index)
        return float((ra[k] - rb[k]).mean())
    pt = stat(d)
    ds = [stat(x) for x in _boot_frames(d, n, seed)]
    return pt, float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))
