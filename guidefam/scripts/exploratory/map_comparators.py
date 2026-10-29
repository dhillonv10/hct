import glob
import os
import sys
import numpy as np
import pandas as pd

D = sys.argv[1]
P = sys.argv[2]
d = pd.read_csv(f"{D}/off-target.csv.gz")
pm = d[d.guide_type == "PM"].copy()
print("PM", len(pm))
print(pm[["gene", "guide_seq", "target_seq", "5p_context", "3p_context"]].head(3).to_string())


def rc(s):
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A", "N": "N"}
    return "".join(comp.get(c, "N") for c in reversed(s.upper()))


pm["gs"] = pm.guide_seq.str.upper().str.replace("U", "T")
pm["ts"] = pm.target_seq.str.upper().str.replace("U", "T")
print("guide == rc(target)?", (pm.gs == pm.ts.map(rc)).mean(), "  guide == target?", (pm.gs == pm.ts).mean())

# DeepCas13
frames = []
for f in sorted(glob.glob(f"{P}/DeepCas13/off-target/*.csv")):
    t = pd.read_csv(f)
    t["gene"] = os.path.basename(f)[:-4]
    frames.append(t)
dc = pd.concat(frames)
dc["seq"] = dc["sgRNA Sequence"].str.upper().str.replace("U", "T")
print("DeepCas13 rows", len(dc), "genes", dc.gene.nunique(), "seq len", dc.seq.str.len().value_counts().to_dict())
for name, key in [("guide", "gs"), ("target", "ts"), ("rc(guide)", None), ("rc(target)", None)]:
    s = set(dc.seq)
    if key:
        hit = pm[key].isin(s).mean()
    elif name == "rc(guide)":
        hit = pm.gs.map(rc).isin(s).mean()
    else:
        hit = pm.ts.map(rc).isin(s).mean()
    print(f"  fraction of PM guides found in DeepCas13 file via {name}: {hit:.3f}")

# Konermann
frames = []
for f in sorted(glob.glob(f"{P}/Konermann/off-target/*.csv")):
    t = pd.read_csv(f)
    t["gene"] = os.path.basename(f)[:-4]
    frames.append(t)
ko = pd.concat(frames)
print("Konermann rows", len(ko), ko.guide.str.len().value_counts().head(3).to_dict())
kg = ko.guide.str.upper().str.replace("U", "T")
for name, fn in [("guide in 30mer", lambda g: g), ("rc(guide) in 30mer", rc), ("rc(target) in 30mer", None)]:
    pass
sample = pm.sample(300, random_state=0)
def contained(seqs, pool):
    pool = list(pool)
    return np.mean([any(s in p for p in pool) for s in seqs])
print("  guide substring of a Konermann 30mer:", contained(sample.gs, kg))
print("  rc(guide) substring:", contained(sample.gs.map(rc), kg))
print("  target substring:", contained(sample.ts, kg))
print("  rc(target)=guide... same as first")

# RandomForest
rf = pd.read_csv(f"{P}/RandomForest/off-target/OFF-target-Lib_L1OVC_RFnbt.csv")
print("RF", rf.shape, list(rf.columns)[:12])
print(rf.head(3).to_string()[:800])
