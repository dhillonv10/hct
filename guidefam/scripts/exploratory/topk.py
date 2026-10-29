import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import comparators as C, data as D, metrics as M
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = C.add_all(D.load_tiger(f"{S}/data/tiger", "off-target"), f"{S}/ext/tiger/predictions (other models)")
o = pd.read_csv(f"{S}/results/final_cas13d/hek_oof.csv", index_col=0)
x = d.join(o[["gbm", "tiger2d_port"]])
r = lambda c: x.groupby("gene")[c].rank(pct=True)
x["gbm+tiger2d"] = (r("gbm") + r("tiger2d_port")) / 2
x["ens3"] = (r("gbm") + r("tiger2d_port") + r("konermann")) / 3
x["random"] = np.random.default_rng(0).random(len(x))
x["oracle"] = x.act
rng = np.random.default_rng(0)
print("Top-k guides per gene (LOGO, 16 genes): what a designer gets by ordering the top k predicted guides")
print(f"{'model':14s} {'k':>3s} {'mean z-activity':>16s} {'frac active':>12s} {'best-of-k > 90th pct':>22s}")
x["z"] = x.groupby("gene").act.transform(lambda s: (s - s.mean()) / s.std())
x["pct"] = x.groupby("gene").act.rank(pct=True)
for k in (5, 10, 20):
    for c in ("random", "deepcas13", "konermann", "tiger2d_port", "gbm", "gbm+tiger2d", "ens3", "oracle"):
        top = x.sort_values(c, ascending=False).groupby("gene").head(k)
        best90 = top.groupby("gene").pct.max().ge(0.9).mean()
        print(f"{c:14s} {k:>3d} {top.z.mean():16.3f} {top.active.mean():12.3f} {best90:22.3f}")
    print()
# bootstrap over genes for the k=10 mean z difference ens3 - tiger2d
def kz(df, c, k=10): return df.sort_values(c, ascending=False).groupby("gene").head(k).z.mean()
genes = x.gene.unique(); ds = []
for _ in range(500):
    gs = rng.choice(genes, len(genes)); b = pd.concat([x[x.gene == g].assign(gene=f"{g}_{i}") for i, g in enumerate(gs)])
    ds.append(kz(b, "ens3") - kz(b, "tiger2d_port"))
print(f"k=10 mean z-activity, ens3 - tiger2d (gene bootstrap): {kz(x,'ens3')-kz(x,'tiger2d_port'):+.3f} [{np.percentile(ds,2.5):+.3f},{np.percentile(ds,97.5):+.3f}]")
