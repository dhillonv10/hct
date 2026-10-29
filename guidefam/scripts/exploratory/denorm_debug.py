import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import data as D, metrics as M
d = D.load_tiger(sys.argv[1], "off-target"); oof = pd.read_csv(sys.argv[2], index_col=0)
x = d.join(oof[["gbm_core"]], how="inner")
p = D.gene_norm_params(d)
g = x.groupby("gene")
tab = pd.DataFrame({"y_median": g.y.median(), "y_scale(q90-q10)": g.y.apply(lambda s: s.quantile(.9) - s.quantile(.1)), "y_std": g.y.std(),
                    "pred_mean(norm)": g.gbm_core.mean(), "pred_std(norm)": g.gbm_core.std()})
print(tab.round(2).to_string())
print("\nstd across genes of true gene median: %.3f   mean within-gene std of y: %.3f" % (tab.y_median.std(), tab.y_std.mean()))
print("std across genes of mean prediction (norm units): %.3f   mean within-gene std of prediction: %.3f" % (tab["pred_mean(norm)"].std(), tab["pred_std(norm)"].mean()))
x["pc"] = x.gbm_core - g.gbm_core.transform("mean")            # remove the model's own gene-level offset
for nm, col in (("fair: constant map of raw prediction", "gbm_core"), ("fair: constant map, gene offset removed", "pc")):
    out = []
    for gg, xg in x.groupby("gene"):
        o = p.drop(index=gg); out.append(pd.Series(o["loc"].mean() - o["scale"].mean() * xg[col], index=xg.index))
    pr = pd.concat(out).reindex(x.index); print(f"{nm:44s} pooled Pearson {M.pearson(x.y, pr):.3f}  Spearman {M.spearman(x.y, pr):.3f}")
pr = x.gene.map(p["loc"]) - x.gene.map(p["scale"]) * x.gbm_core
print(f"{'oracle (TIGER recipe)':44s} pooled Pearson {M.pearson(x.y, pr):.3f}  Spearman {M.spearman(x.y, pr):.3f}")
pr = x.gene.map(p["loc"]) - x.gene.map(p["scale"]) * 0
print(f"{'oracle gene median ALONE (no model at all)':44s} pooled Pearson {M.pearson(x.y, pr):.3f}  Spearman {M.spearman(x.y, pr):.3f}")
