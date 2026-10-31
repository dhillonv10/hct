#!/usr/bin/env python3
"""cascade_sim.py -- the simulation fixed in CASCADE_PROTOCOL.md. Input: fold_compare_rows.csv.gz (columns screen, patent, y, ours, oligoai).

  python cascade_sim.py fold_compare_rows.csv.gz --out cascade_report.txt
"""
import argparse
import math

import numpy as np
import pandas as pd

FRACS = [0.05, 0.10, 0.25, 0.50, 1.00]
K = 5


def pct(v):
    r = pd.Series(v).rank(method="average").values
    return r / len(r)


def select(g, f, policy, rng):
    """Indices (positions in g) of the top-K ASOs under a policy. g has numpy columns ours, oligoai."""
    n = len(g["ours"])
    if policy == "compact":
        return np.argsort(-g["ours"])[:K]
    if policy == "large_only":
        return np.argsort(-g["oligoai"])[:K]
    m = max(1, math.ceil(f * n))
    if policy == "cascade":
        called = np.argsort(-g["ours"])[:m]
    elif policy == "random":
        called = rng.choice(n, m, replace=False)
    else:
        raise ValueError(policy)
    mix = 0.5 * pct(g["ours"][called]) + 0.5 * pct(g["oligoai"][called])
    order = called[np.argsort(-mix)]
    rest = np.setdiff1d(np.argsort(-g["ours"]), called, assume_unique=False)
    rest = rest[np.argsort(-g["ours"][rest])]
    return np.concatenate([order, rest])[:K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--out", default="cascade_report.txt")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = pd.read_csv(a.rows, usecols=["screen", "patent", "y", "ours", "oligoai"])
    rng = np.random.default_rng(0)
    rows = []
    for sid, g in d.groupby("screen"):
        if len(g) < 20 or g.y.std() == 0:
            continue
        arr = {"ours": g.ours.values, "oligoai": g.oligoai.values}
        y = g.y.values * 100.0
        rec = {"screen": sid, "patent": g.patent.iloc[0], "n": len(g), "mean": y.mean(), "oracle": np.sort(y)[-K:].mean()}
        rec["compact"] = y[select(arr, 0, "compact", rng)].mean()
        rec["large_only"] = y[select(arr, 1, "large_only", rng)].mean()
        for f in FRACS:
            rec[f"cascade_{f}"] = y[select(arr, f, "cascade", rng)].mean()
            rec[f"random_{f}"] = np.mean([y[select(arr, f, "random", np.random.default_rng(100 + s))].mean() for s in range(5)])
        rows.append(rec)
    t = pd.DataFrame(rows)
    units = t.patent.unique()
    idx = {u: np.flatnonzero(t.patent.values == u) for u in units}
    bg = np.random.default_rng(1)
    draws = [np.concatenate([idx[u] for u in bg.choice(units, len(units))]) for _ in range(a.boot)]

    def ci(x, y_=None):
        v = t[x].values if y_ is None else t[x].values - t[y_].values
        b = np.array([v[ii].mean() for ii in draws])
        return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

    out = []

    def say(s=""):
        print(s)
        out.append(s)
    say(f"{len(t)} screens, {t.patent.nunique()} patents, mean screen size {t.n.mean():.0f}. Quality = mean measured inhibition (percentage points) of the top-{K} selected ASOs per screen.")
    say(f"  screen mean (random pick) {t['mean'].mean():.1f} | oracle {t.oracle.mean():.1f}")
    for lab, c in (("compact only (no large-model calls)", "compact"), ("large model only (calls on 100%)", "large_only")):
        m, lo, hi = ci(c)
        say(f"  {lab:46s} {m:6.2f} [{lo:.2f},{hi:.2f}]")
    say()
    say(f"  {'large-model calls':>18s} | {'cascade (compact routes)':26s} | {'random routing':14s} | cascade - full mixture | cascade - large only")
    for f in FRACS:
        m, lo, hi = ci(f"cascade_{f}")
        mr, _, _ = ci(f"random_{f}")
        d1 = ci(f"cascade_{f}", "cascade_1.0")
        d2 = ci(f"cascade_{f}", "large_only")
        say(f"  {f:>17.0%} | {m:6.2f} [{lo:.2f},{hi:.2f}]       | {mr:6.2f}        | {d1[0]:+.2f} [{d1[1]:+.2f},{d1[2]:+.2f}] | {d2[0]:+.2f} [{d2[1]:+.2f},{d2[2]:+.2f}]")
    d1, d2 = ci("cascade_0.25", "cascade_1.0"), ci("cascade_0.25", "large_only")
    shortfall_ok = (-d1[0] <= 1.0) and (-d1[1] < 2.0)
    better_ok = d2[1] > 0
    rnd = ci("cascade_0.25", "random_0.25")
    say()
    say(f"DECISION at 25% large-model calls: (a) within 1.0 point of the full mixture (shortfall {-d1[0]:.2f}, upper bound {-d1[1]:.2f} < 2.0): {'MET' if shortfall_ok else 'NOT MET'}")
    say(f"                                     (b) better than the large model alone ({d2[0]:+.2f} [{d2[1]:+.2f},{d2[2]:+.2f}], lower bound > 0): {'MET' if better_ok else 'NOT MET'}")
    say(f"  control: cascade vs random routing at 25%: {rnd[0]:+.2f} [{rnd[1]:+.2f},{rnd[2]:+.2f}] ({'compact routing helps' if rnd[1] > 0 else 'no evidence the compact model routes better than chance'})")
    open(a.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
