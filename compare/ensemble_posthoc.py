import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

d = pd.read_csv(sys.argv[1])
seen = set(d[d.oa_split == "train"].group)


def screen_table(df, ours, unit, ws=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """One row per screen (n >= 20): within-screen Spearman of ours, OligoAI and rank-average mixes. Screens sit wholly inside one bootstrap unit."""
    rows = []
    for sid, g in df.dropna(subset=[ours, "oligoai_pred"]).groupby("screen"):
        if len(g) < 20:
            continue
        y = rankdata(g.y.values)
        ra, rb = rankdata(g[ours].values) / len(g), rankdata(g.oligoai_pred.values) / len(g)
        if y.std() == 0:
            continue
        r = {"screen": sid, "unit": g[unit].iloc[0], "pred_corr": np.corrcoef(ra, rb)[0, 1]}
        for w in ws:
            m = w * ra + (1 - w) * rb
            r[f"w{w}"] = np.corrcoef(y, rankdata(m))[0, 1] if m.std() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).dropna()


def boot_mean(t, cols, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    units = t.unit.unique()
    idx = {u: np.flatnonzero(t.unit.values == u) for u in units}
    vals = {c: t[c].values for c in cols}
    out = {c: [] for c in cols}
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(units, len(units))])
        for c in cols:
            out[c].append(vals[c][ii].mean())
    return out


def report(name, df, ours, unit):
    t = screen_table(df, ours, unit)
    print(f"\n=== {name}: {len(t)} screens, {t.unit.nunique()} bootstrap units ({unit})")
    cols = ["w1.0", "w0.0", "w0.5"]
    b = boot_mean(t, cols)
    lab = {"w1.0": "ours", "w0.0": "OligoAI", "w0.5": "50/50 rank average"}
    for c in cols:
        lo, hi = np.percentile(b[c], [2.5, 97.5])
        print(f"  {lab[c]:20s} within-screen Spearman {t[c].mean():.3f} [{lo:.3f},{hi:.3f}]")
    for a_, b_, nm in (("w0.5", "w0.0", "MIX - OligoAI"), ("w0.5", "w1.0", "MIX - ours"), ("w1.0", "w0.0", "ours - OligoAI")):
        diff = np.array(b[a_]) - np.array(b[b_])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"  {nm:20s} {t[a_].mean() - t[b_].mean():+.3f} [{lo:+.3f},{hi:+.3f}]   P(MIX better) = {np.mean(diff > 0):.3f}" if nm == "MIX - OligoAI" else
              f"  {nm:20s} {t[a_].mean() - t[b_].mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"  agreement between the two models' rankings inside a screen (Pearson of rank-percentiles): {t.pred_corr.mean():.3f}")
    print("  descriptive weight sweep, weight on ours (only 0.5 was specified in advance):",
          {w: round(t[f'w{w}'].mean(), 3) for w in (0.0, 0.25, 0.5, 0.75, 1.0)})


report("S1  OligoAI's protocol (patent split)", d[d.oa_split == "test"], "ours", "patent")
report("S2  our locked gene groups (favours OligoAI)", d[d.split == "locked"], "ours2", "group")
report("S2b locked groups OligoAI never saw", d[(d.split == "locked") & (~d.group.isin(seen))], "ours2", "group")
