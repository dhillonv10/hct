"""Feature discovery against the DENOISED residual.

The mismatch-variant panel gives each of ~600 perfect-match sites an independent, low-noise activity estimate (PC1 of the 23 single-mismatch LFCs; corr 0.85 with the
perfect-match label). We remove what the ensemble already predicts from that estimate (within gene) and ask which candidate features still correlate with what is left.
Because the target is far less noisy than the raw label, weak real signals that vanish in a noisy-label model comparison can show up here.
Per-gene Spearman correlations are combined across genes (Stouffer), then Benjamini-Hochberg over all features.

    python -m guidefam.scripts.feature_discovery --data DATA --oof hek_oof.csv --pred PRED [--struct struct_off-target.csv.gz]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from .. import comparators as C
from .. import data as D
from .. import features as F


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--emb-cache", default=None)
    a = ap.parse_args()
    allr = D.load_tiger(a.data, "off-target", pm_only=False)
    pm = C.add_all(D.load_tiger(a.data, "off-target"), a.pred).join(pd.read_csv(a.oof, index_col=0)[["gbm", "tiger2d_port"]])
    rk = lambda c: pm.groupby("gene")[c].rank(pct=True)
    pm["ens3"] = (rk("gbm") + rk("tiger2d_port") + rk("konermann")) / 3
    sm = allr[allr.guide_type == "SM"].copy()
    sm["pos"] = sm.guide_id.str.extract(r"_SM_P(\d+)-")[0].astype(int)
    sm["z"] = sm.groupby("gene").y.transform(lambda s: (s - s.mean()) / s.std())
    V = sm.groupby(["gene", "target_seq", "pos"]).z.mean().unstack("pos").dropna(thresh=20)
    V = V.apply(lambda c: c.fillna(c.mean()))
    # candidate features for every perfect-match row
    X = F.tabular(pm, "full", flank=2)
    X = pd.concat([X, F.thermo_frame(pm), F.motif_frame(pm)], axis=1)
    if a.struct:
        st = pd.read_csv(a.struct).drop_duplicates(["gene", "guide_id"]).set_index(["gene", "guide_id"])
        S = st.reindex(pd.MultiIndex.from_arrays([pm.gene.values, pm.guide_id.values])).set_axis(pm.index)
        X = pd.concat([X, S], axis=1)
    X = X.loc[:, X.std() > 0]
    key = pd.MultiIndex.from_arrays([pm.gene.values, pm.target_seq.values])
    pos_of = {k: i for i, k in enumerate(key)}
    rows = []
    res = {}
    for g in V.index.get_level_values(0).unique():
        v = V.loc[g]
        v = (v - v.mean()) / v.std()
        u, s, _ = np.linalg.svd(v.values, full_matrices=False)
        lat = u[:, 0] * s[0]
        ix = np.array([pos_of[(g, t)] for t in v.index])
        sub = pm.iloc[ix]
        if np.corrcoef(lat, sub.act.values)[0, 1] < 0:
            lat = -lat
        p = rankdata(sub.ens3.values)
        z = rankdata(lat)
        rz = z - np.polyval(np.polyfit(p, z, 1), p)               # what the ensemble does not explain of the independent site estimate
        res[g] = (ix, rz)
    for c in X.columns:
        zs, n = [], 0
        for g, (ix, rz) in res.items():
            f = X.iloc[ix][c].values
            if np.nanstd(f) == 0:
                continue
            r = spearmanr(f, rz, nan_policy="omit")[0]
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999)) * np.sqrt(len(f) - 3))
            n += 1
        if n:
            zc = np.sum(zs) / np.sqrt(n)
            rows.append({"feature": c, "stouffer_z": zc, "p": 2 * norm.sf(abs(zc)), "mean_r": np.tanh(np.mean(zs) / np.sqrt(np.mean([len(res[g][0]) for g in res]) - 3))})
    t = pd.DataFrame(rows).dropna(subset=["p"]).reset_index(drop=True)          # undefined correlations (constant within every gene) are not tests
    t["q_BH"] = bh(t.p.values)
    t = t.sort_values("p")
    print(f"{len(t)} candidate features tested against the denoised residual ({sum(len(v[0]) for v in res.values())} sites, {len(res)} genes)")
    print(t.head(20).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nsignificant at q < 0.05: {int((t.q_BH < 0.05).sum())};  q < 0.10: {int((t.q_BH < 0.10).sum())}")
    t.to_csv("feature_discovery.csv", index=False)


if __name__ == "__main__":
    main()
