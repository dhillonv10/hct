import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import data as D
S = "/tmp/claude-0/-home-user-hct/4c5637ac-6023-59b5-a907-35ccb9f71d81/scratchpad"
d = D.load_tiger(f"{S}/data/tiger", "off-target", pm_only=False)
print(d.guide_type.value_counts().to_dict())
pm = d[d.guide_type == "PM"]
print("PM target_seq unique within gene:", not pm.duplicated(["gene", "target_seq"]).any())
mm = d[d.guide_type != "PM"].copy()
key = set(zip(pm.gene, pm.target_seq))
mm["parent"] = [(g, t) in key for g, t in zip(mm.gene, mm.target_seq)]
print("MM rows whose (gene,target_seq) matches a PM parent:", mm.parent.mean().round(4), " of", len(mm))
print(mm[["gene", "guide_id", "guide_type", "target_seq", "guide_seq"]].head(4).to_string())
# mismatch positions: compare the variant guide with the complement of its target
comp = str.maketrans("ACGT", "TGCA")
def mmpos(t, g):
    ref = t.translate(comp)
    return [i for i, (a, b) in enumerate(zip(ref, g)) if a != b] if len(g) == len(ref) else None
smp = mm[mm.guide_type == "SM"].head(2000)
pos = [mmpos(t, g) for t, g in zip(smp.target_seq, smp.guide_seq)]
print("SM: fraction with exactly one differing position:", np.mean([p is not None and len(p) == 1 for p in pos]).round(3), " guide len", smp.guide_seq.str.len().value_counts().to_dict())
# how many variants per PM site, by gene
cnt = mm[mm.parent].groupby(["gene", "target_seq"]).size()
per_gene = cnt.groupby("gene").agg(["count", "mean", "max"]).round(1)
per_gene["PM"] = pm.groupby("gene").size()
print(per_gene.to_string())
print("replicates present on MM rows:", mm[D.LFC].notna().mean().round(3).to_dict())
