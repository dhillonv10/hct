import sys
sys.path.insert(0, "/home/user/hct")
import pandas as pd
from guidefam import data as D, metrics as M
d = D.load_tiger(sys.argv[1], "off-target")
oof = pd.read_csv(sys.argv[2], index_col=0)
dev = d[~d.gene.isin(D.LOCKED_GENES)].join(oof, how="inner")
rows = []
for c in oof.columns:
    x = dev[dev[c].notna()]
    rows.append({"exp": c, "rho_within": round(M.within_group(x, "gene", "act", c), 4), "auroc_within": round(M.within_group(x, "gene", "active", c, M.auroc), 4), "rho_pooled": round(M.spearman(x.act, x[c]), 4)})
print(pd.DataFrame(rows).to_string(index=False))
