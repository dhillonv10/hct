"""Is the held-out accuracy plateau label noise, or biology the model does not see?

~600 perfect-match sites (100 in each of 6 genes) were also measured with ~181 single/multi-mismatch variants of the SAME site. Those variants carry an
independent estimate of the site's activity (their noise is independent of the perfect-match replicates). We build that estimate (first principal component of the
23 per-position single-mismatch LFCs, gene-standardised), check it against the perfect-match label, and then ask the decisive question:

   after removing what the model predicts, does the perfect-match residual still correlate with the independent variant-based estimate?
   yes -> real, reproducible site biology is left unexplained (missing information);   no -> the residual is noise (we are at the ceiling).

    python -m guidefam.scripts.noise_vs_information --data DATA --oof hek_oof.csv --pred PRED
"""
import argparse

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import data as D
from .. import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", required=True, help="leave-one-gene-out predictions (columns gbm, tiger2d_port)")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    allr = D.load_tiger(a.data, "off-target", pm_only=False)
    pm = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)               # same frame (and index) the saved predictions are keyed to
    o = pd.read_csv(a.oof, index_col=0)
    sm = allr[allr.guide_type == "SM"].copy()
    sm["pos"] = sm.guide_id.str.extract(r"_SM_P(\d+)-")[0].astype(int)
    sm["z"] = sm.groupby("gene").y.transform(lambda s: (s - s.mean()) / s.std())          # gene-standardised LFC
    V = sm.groupby(["gene", "target_seq", "pos"]).z.mean().unstack("pos")                  # site x position, mean of the 3 substitutions
    V = V.dropna(thresh=20).apply(lambda c: c.fillna(c.mean()))
    pm = pm.join(o[["gbm", "tiger2d_port"]])
    rk = lambda c: pm.groupby("gene")[c].rank(pct=True)
    pm["ens3"] = (rk("gbm") + rk("tiger2d_port") + rk("konermann")) / 3
    pm["zact"] = pm.groupby("gene").act.transform(lambda s: (s - s.mean()) / s.std())
    site = pm.set_index(["gene", "target_seq"]).loc[V.index]
    print(f"{len(V)} sites with a full variant panel in genes {sorted(V.index.get_level_values(0).unique())}")
    # latent site activity = first principal component of the variant panel (within gene), oriented to agree with the perfect-match label
    zs = []
    for g in V.index.get_level_values(0).unique():
        v = V.loc[g]
        v = (v - v.mean()) / v.std()
        u, s, vt = np.linalg.svd(v.values, full_matrices=False)
        pc = pd.Series(u[:, 0] * s[0], index=v.index)
        if np.corrcoef(pc.values, site.loc[g].zact.reindex(v.index).values)[0, 1] < 0:
            pc = -pc
        zs.append(pd.Series(pc.values, index=pd.MultiIndex.from_product([[g], v.index])))
        print(f"  {g:9s} PC1 explains {s[0]**2 / (s**2).sum():.1%} of variant-panel variance; corr(PC1, perfect-match activity) = {np.corrcoef(pc.values, site.loc[g].zact.reindex(v.index).values)[0, 1]:.3f}")
    site["lat"] = pd.concat(zs).reindex(site.index).values
    site["lat"] = site.groupby(level=0).lat.transform(lambda s: (s - s.mean()) / s.std())
    print("\nOn these sites (within-gene, Spearman): measured PM label vs model prediction, and vs the independent variant-based estimate")
    for c in ("gbm", "tiger2d_port", "konermann", "ens3"):
        print(f"  {c:13s} corr(prediction, PM label) {M.within_group(site.reset_index(), 'gene', 'zact', c, min_n=30):.3f}    corr(prediction, variant-based estimate) {M.within_group(site.reset_index(), 'gene', 'lat', c, min_n=30):.3f}")
    print(f"  corr(PM label, variant-based estimate) {M.within_group(site.reset_index(), 'gene', 'zact', 'lat', min_n=30):.3f}")
    # residual test: partial correlation of the PM residual with the variant estimate, controlling for the prediction
    print("\nResidual test: partial correlation (within gene) between the PM label and the independent estimate, controlling for the model prediction")
    rng = np.random.default_rng(0)
    for c in ("gbm", "ens3"):
        def partial(df):
            out = []
            for g, x in df.groupby("gene"):
                p, y, z = x[c].rank().values, x.zact.rank().values, x.lat.rank().values
                ry = y - np.polyval(np.polyfit(p, y, 1), p)
                rz = z - np.polyval(np.polyfit(p, z, 1), p)
                out.append(np.corrcoef(ry, rz)[0, 1])
            return float(np.mean(out))
        s = site.reset_index()
        pt = partial(s)
        sites = s.groupby("gene").apply(lambda x: x.index.values).to_dict()
        ds = []
        for _ in range(a.boot):
            ix = np.concatenate([rng.choice(v, len(v), replace=True) for v in sites.values()])
            b = s.loc[ix].copy(); b["gene"] = np.concatenate([[g] * len(v) for g, v in sites.items()])
            ds.append(partial(b.reset_index(drop=True)))
        print(f"  controlling for {c:5s}: partial corr {pt:+.3f}  [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
