"""Peak-level RBP binding in lncRNAs: the RNA field's standard formulation, with its main pitfall controlled.

Why this replaces the gene-level RBP task as the headline
  The gene-level task ("does lncRNA X have ANY eCLIP peak for RBP Y") is driven by expression and transcript length even after
  stratification. The published formulation (GraphProt, DeepRiPe, iDeepS, RBPNet, ...) is: a 101-nt window centred on a peak
  versus windows that are not. How the negatives are drawn decides the result (a recent systematic benchmark of 37 methods on CLIP-seq
  data says so, and recommends sites bound by OTHER RBPs as a bias-aware negative), so both are provided:

    --neg unbound   windows from the SAME transcripts as the positives, same relative-position decile, >= 100 nt from any peak of
                    this RBP.  Controls expression, transcript length, and where in the transcript a window lies.
    --neg other     windows centred on peaks of OTHER RBPs in the same cell line (>= 100 nt from any peak of this RBP), from the same
                    transcript when possible.  Controls "is this a generally accessible, protein-bound site"; asks only for RBP specificity.

  Windows come from the spliced representative transcript of expressed lncRNAs (ENCODE IDR peaks, K562 and HepG2); at most 12
  positive windows per gene and RBP, because a handful of giant lncRNAs (NEAT1, MALAT1) otherwise carry most of the peaks.
  Folds are the chromosome groups of the benchmark (a gene never straddles folds); chr8/13/18/21 are the locked test.

Models: k-mer spectrum, length+GC, a small one-hot CNN trained from scratch (the BEACON lesson: always include a trained non-language
model), and any frozen language model (rinalmo-micro, rnafm, local:<cpt dir>, ...). Metric: macro-AUROC over RBPs with enough positives,
bootstrap CI over RBPs, paired differences over RBPs.  `--transfer` trains on one cell line and tests on the other (different genes by fold).

    python -m lncfm.rbppeak --cell K562 --neg unbound --providers kmer cnn rinalmo-micro --out $LNCFM_RUNS/v2/rbp_peak_K562.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import bench, eclip, evalkit, gencode, ncvar, paths, splits

HALF = 50                      # 101-nt windows
EXCL = 100                     # negatives at least this far from any peak of the RBP


def _windows_for_cell(cell, rbps=None, min_peaks=60):
    """Peak-centred windows per RBP: DataFrame [rbp, g, tx_id, off, key]; plus rep (gene table) and all-peak table."""
    rep, _, _ = eclip._raw_table(cell)
    tx_exons, _ = gencode._cached_gtf("human", None)
    items = [(f"{c}{st}", s, e, gi) for gi, (tid, c, st) in enumerate(zip(rep.tx_id, rep.chrom, rep.strand))
             for s, e in tx_exons[tid]]
    idx = ncvar.ExonIndex(items)
    man = pd.DataFrame(json.load(open(paths.DATA / "eclip" / "manifest.json")))
    man = man[man.cell == cell]
    rows = []
    for rbp, grp in man.groupby("rbp"):
        if rbps and rbp not in rbps:
            continue
        seen = set()
        for f in grp.file:
            bed = eclip._load_bed(paths.DATA / "eclip" / f)
            for c, s, e, st in zip(bed.chrom.values, bed.s.values, bed.e.values, bed.strand.values):
                ctr = (int(s) + int(e)) // 2 + 1                      # 1-based centre
                for gi in idx.hits(f"{c}{st}", ctr):
                    tid = rep.tx_id.iloc[gi]
                    off = ncvar.tx_offset(tx_exons[tid], st, ctr)
                    L = len(rep.seq.iloc[gi])
                    if off is None or off < HALF or off + HALF >= L:
                        continue
                    k = (gi, off // 25)
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append((rbp, gi, off))
    W = pd.DataFrame(rows, columns=["rbp", "g", "off"])
    keep = W.rbp.value_counts()
    return W[W.rbp.isin(keep[keep >= min_peaks].index)].reset_index(drop=True), rep


def build(cell, neg="unbound", rbps=None, min_peaks=60, max_pos=800, ratio=2, seed=0, per_gene=12):
    """Windows table: [rbp, gene_id_nov, chrom, seq, y, fold, leak, g, off]."""
    cache = paths.ensure(paths.DATA / "processed") / f"rbppeak_{cell}_{neg}_m{min_peaks}_p{max_pos}_r{ratio}_g{per_gene}.pkl"
    if cache.exists() and not rbps:
        return pd.read_pickle(cache)
    rng = np.random.default_rng(seed)
    W, rep = _windows_for_cell(cell, rbps, min_peaks)
    allpk = {r: g for r, g in W.groupby("rbp")}
    out = []
    for rbp, g in allpk.items():
        # a few giant lncRNAs (NEAT1, MALAT1, ...) carry thousands of peaks: cap windows per gene so they cannot dominate
        gc = g.sample(frac=1, random_state=seed).groupby("g").head(per_gene)
        pos = gc.sample(min(max_pos, len(gc)), random_state=seed)
        near = {}                                                     # gene -> sorted offsets of ALL this RBP's peaks
        for gi, o in zip(g.g.values, g.off.values):
            near.setdefault(gi, []).append(o)
        near = {k: np.sort(v) for k, v in near.items()}
        others = W[(W.rbp != rbp)]
        by_gene = {k: v for k, v in others.groupby("g")}
        pool = others.sample(min(len(others), 20000), random_state=seed) if neg == "other" else None
        rows = [(rbp, gi, o, 1) for gi, o in zip(pos.g.values, pos.off.values)]
        for gi, o in zip(pos.g.values, pos.off.values):
            L = len(rep.seq.iloc[gi])
            for _ in range(ratio):
                cand = None
                for _try in range(30):
                    if neg == "unbound":
                        dec = min(int(10 * o / L), 9)
                        a, b = max(HALF, int(L * dec / 10)), min(L - HALF - 1, int(L * (dec + 1) / 10))
                        if b <= a:
                            a, b = HALF, L - HALF - 1
                        c = int(rng.integers(a, b + 1)) if b > a else None
                        gj = gi
                    else:
                        if gi in by_gene and rng.random() < 0.7:
                            r2 = by_gene[gi].iloc[int(rng.integers(len(by_gene[gi])))]
                        else:
                            r2 = pool.iloc[int(rng.integers(len(pool)))]
                        gj, c = int(r2.g), int(r2.off)
                    if c is None:
                        continue
                    pk = near.get(gj)
                    if pk is not None and np.abs(pk - c).min() < EXCL:
                        continue
                    cand = (rbp, gj, c, 0)
                    break
                if cand:
                    rows.append(cand)
        out.extend(rows)
    D = pd.DataFrame(out, columns=["rbp", "g", "off", "y"]).drop_duplicates(["rbp", "g", "off"])
    D["seq"] = [rep.seq.iloc[g][o - HALF:o + HALF + 1] for g, o in zip(D.g, D.off)]
    D["chrom"] = rep.chrom.values[D.g.values]
    D["gene_id_nov"] = rep.gene_id_nov.values[D.g.values]
    D = splits.attach(D.reset_index(drop=True))
    if not rbps:
        D.to_pickle(cache)
    return D


# ------------------------------------------------------------------------------------------------ features


def _store(pn):
    f = paths.ensure(paths.DATA / "emb") / ("peak_" + pn.replace("/", "_").replace(":", "_") + ".pkl")
    return f, (pickle.loads(f.read_bytes()) if f.exists() else {})


def features(seqs, provider, trunc=None):
    if provider == "cnn":
        return None
    pn, prov = bench.make_provider(provider, trunc)
    f, st = _store(pn)
    key = lambda s: hashlib.md5(s.encode()).hexdigest()[:16]
    todo = sorted({s for s in seqs if key(s) not in st})
    if todo:
        for i in range(0, len(todo), 4096):
            Z = prov(todo[i:i + 4096])
            for s, z in zip(todo[i:i + 4096], Z):
                st[key(s)] = z
        f.write_bytes(pickle.dumps(st))
    return np.stack([st[key(s)] for s in seqs])


_ONE = np.eye(4, dtype=np.float32)


def onehot(seqs):
    lut = {c: i for i, c in enumerate("ACGU")}
    lut["T"] = 3
    out = np.zeros((len(seqs), 4, HALF * 2 + 1), np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:HALF * 2 + 1]):
            if c in lut:
                out[i, lut[c], j] = 1.0
    return out


def cnn_fit_predict(Xtr, ytr, Xte, epochs=12, seed=0):
    """Small DeepRiPe-style CNN from scratch: conv(8) - pool - conv(5) - global max - dense."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv1d(4, 48, 8, padding=4), nn.ReLU(), nn.MaxPool1d(3), nn.Dropout(0.2),
                        nn.Conv1d(48, 48, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                        nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(xt)), max(len(xt) // 64, 1)):
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[b]).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(Xte)).squeeze(1).numpy()


# ------------------------------------------------------------------------------------------------ evaluation


def eval_rbp(d, X, provider, final):
    """OOF dev predictions (chromosome-group folds) [+ locked-test predictions] for one RBP. Returns dict of arrays."""
    fold, y = d.fold.values, d.y.values
    pred = np.full(len(d), np.nan)
    use_cnn = provider == "cnn"
    Xc = onehot(list(d.seq)) if use_cnn else None
    for f in range(5):
        tr, te = (fold >= 0) & (fold != f), fold == f
        if te.sum() < 5 or min(y[tr].sum(), (1 - y[tr]).sum()) < 15:      # too little training data in this fold
            continue
        pred[te] = (cnn_fit_predict(Xc[tr], y[tr], Xc[te]) if use_cnn else
                    evalkit._fit_predict(X[tr], y[tr], X[te], "clf", np.random.default_rng(0)))
    out = {"dev": pred}
    if final:
        tr, te = fold >= 0, fold == -1
        p = np.full(len(d), np.nan)
        if te.sum() >= 5 and min(y[tr].sum(), (1 - y[tr]).sum()) >= 15:
            p[te] = (cnn_fit_predict(Xc[tr], y[tr], Xc[te]) if use_cnn else
                     evalkit._fit_predict(X[tr], y[tr], X[te], "clf", np.random.default_rng(0)))
        out["test"] = p
    return out


def _auc(y, p, m):
    m = m & ~np.isnan(p)
    return roc_auc_score(y[m], p[m]) if m.sum() > 10 and len(np.unique(y[m])) == 2 else np.nan


def run(cell, neg, providers, trunc=None, max_rbps=None, final=False, min_peaks=60, max_pos=800, out=None, n_boot=1000):
    D = build(cell, neg, None, min_peaks, max_pos)
    rbps = sorted(D.rbp.unique())
    if max_rbps:
        rbps = rbps[:max_rbps]
    print(f"[rbp_peak {cell} neg={neg}] {len(rbps)} RBPs, {len(D)} windows ({100*D.y.mean():.0f}% positive); "
          f"homology-flagged {100*D.leak.mean():.1f}%", flush=True)
    res = {}
    for pv in providers:
        t0 = time.time()
        feats = features(list(D.seq), pv, trunc) if pv != "cnn" else None
        rows = []
        for r in rbps:
            ix = np.where(D.rbp.values == r)[0]
            d = D.iloc[ix]
            X = None if feats is None else feats[ix]
            o = eval_rbp(d, X, pv, final)
            y, leak = d.y.values, d.leak.values
            row = {"rbp": r, "n": len(d), "n_pos": int(y.sum()),
                   "dev_all": _auc(y, o["dev"], d.fold.values >= 0),
                   "dev_pruned": _auc(y, o["dev"], (d.fold.values >= 0) & ~leak)}
            if final:
                row["test"] = _auc(y, o["test"], d.fold.values == -1)
            rows.append(row)
        res[pv] = pd.DataFrame(rows)
        a = res[pv].dev_all.dropna().values
        rng = np.random.default_rng(0)
        bs = [rng.choice(a, len(a)).mean() for _ in range(n_boot)]
        print(f"  {pv:<26s} macro-AUROC {a.mean():.3f} [{np.percentile(bs, 2.5):.3f},{np.percentile(bs, 97.5):.3f}] over {len(a)} RBPs"
              f"   pruned {res[pv].dev_pruned.mean():.3f}   ({time.time() - t0:.0f}s)", flush=True)
    names = list(providers)
    if len(names) > 1:
        print("  paired differences over RBPs (dev_all):")
        base = res[names[0]].set_index("rbp").dev_all
        for pv in names[1:]:
            dlt = (res[pv].set_index("rbp").dev_all - base).dropna()
            rng = np.random.default_rng(1)
            bs = [rng.choice(dlt.values, len(dlt)).mean() for _ in range(n_boot)]
            print(f"    {pv} - {names[0]}: {dlt.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]  "
                  f"better on {int((dlt > 0).sum())}/{len(dlt)} RBPs")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"cell": cell, "neg": neg, "providers": {k: v.to_dict("records") for k, v in res.items()}},
                                        default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="K562", choices=["K562", "HepG2"])
    ap.add_argument("--neg", default="unbound", choices=["unbound", "other"])
    ap.add_argument("--providers", nargs="+", default=["lengc", "kmer", "cnn"])
    ap.add_argument("--trunc", type=int, default=None)
    ap.add_argument("--max-rbps", type=int, default=None)
    ap.add_argument("--min-peaks", type=int, default=40)
    ap.add_argument("--max-pos", type=int, default=800)
    ap.add_argument("--final", action="store_true", help="also score the locked test chromosomes (once)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.cell, a.neg, a.providers, a.trunc, a.max_rbps, a.final, a.min_peaks, a.max_pos, a.out)


if __name__ == "__main__":
    main()
