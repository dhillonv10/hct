import sys
sys.path.insert(0, "/home/user/hct")
import numpy as np, pandas as pd
from guidefam import sirna as SI
O = sys.argv[1]
for n in ["Hu", "Mix", "Taka"]:
    d = SI.load(f"{O}/{n}.csv")
    L = d.win.str.len()
    tgt = d.sirna.map(lambda s: SI.rc_dna(s).replace("T", "U"))
    pos = [w.find(t) for w, t in zip(d.win, tgt)]
    pos = pd.Series(pos)
    print(f"{n}: win len {L.value_counts().head(4).to_dict()}  | site found {(pos>=0).mean():.3f}; position of site in window: {pos[pos>=0].value_counts().head(6).to_dict()}")
    bad = d[pos.values < 0]
    if len(bad): print("   not found example:", bad.sirna.iloc[0], bad.win.iloc[0], "| rc-not-complement check:", SI.rc_dna(bad.sirna.iloc[0]))
    # try plain complement (no reverse) in case windows are stored 3'->5'
    pos2 = pd.Series([w.find(s.translate(str.maketrans("ACGU", "UGCA"))) for w, s in zip(d.win, d.sirna)])
    print("   plain-complement found:", round((pos2 >= 0).mean(), 3))
