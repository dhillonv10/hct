"""Figure: held-out-gene accuracy against model size (parameters), every number computed from the saved leave-one-gene-out predictions.

    python -m guidefam.scripts.make_figure --data DATA --pred PRED --results guidefam/results --out guidefam/figures
"""
import argparse
import os

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import data as D
from .. import metrics as M

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"            # slots 1-3 of the validated categorical palette (all-pairs checks pass)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = a.results
    d = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)
    dev = d[~d.gene.isin(D.LOCKED_GENES)].copy()
    rd = lambda f, c: pd.read_csv(os.path.join(R, f), index_col=0)[c].reindex(dev.index)
    dev["gbm"] = rd("cas13d_dev/dev_oof.csv", "base")
    dev["rnafm_probe"] = rd("cas13d_dev/dev_oof.csv", "ridge_rnafm23")
    dev["tiger2d"] = rd("cas13d_tiger/oof_hek293t_logo.csv", "tiger2d_port")
    dev["ridge"] = rd("cas13d_ladder/oof_hek293t_logo.csv", "seq_ridge")
    dev["pf_struct"] = rd("cas13d_novel/pf_struct_oof.csv", "pf_struct")
    dev["pf_nostruct"] = rd("cas13d_novel/pf_nostruct_oof.csv", "pf_nostruct")
    dev["tabdpt"] = rd("cas13d_novel/tabdpt_oof.csv", "tabdpt")
    rk = lambda c: dev.groupby("gene")[c].rank(pct=True)
    dev["ens3"] = (rk("gbm") + rk("tiger2d") + rk("konermann")) / 3
    # (label, column, parameters, family)  parameters from efficiency_table.py
    pts = [("ridge, sequence only", "ridge", 130, "classical"), ("gradient-boosted trees", "gbm", 17200, "classical"),
           ("small transformer + structure bias", "pf_struct", 25165, "scratch"), ("small transformer, no structure bias", "pf_nostruct", 25165, "scratch"),
           ("2-D CNN (TIGER-style)", "tiger2d", 236289, "scratch"), ("TabDPT (tabular foundation model)", "tabdpt", 63040624, "pretrained"),
           ("RNA-FM embedding + ridge", "rnafm_probe", 99523469, "pretrained")]
    rows = [{"model": l, "parameters": p, "family": f, "within_gene_spearman": round(M.within_group(dev, "gene", "act", c), 4)} for l, c, p, f in pts]
    ens = round(M.within_group(dev, "gene", "act", "ens3"), 4)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "params_vs_accuracy.csv"), index=False)
    print(t.to_string(index=False), "\nensemble:", ens)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = {"classical": (BLUE, "o", "classical (ridge, boosted trees)"), "scratch": (ORANGE, "s", "neural nets trained from scratch"), "pretrained": (AQUA, "D", "pretrained models used as feature extractor / learner")}
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    ax.set_facecolor("#fcfcfb")
    ax.axhline(ens, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(10 ** 1.65, ens + 0.004, f"3-model ensemble (boosted trees + CNN + published tool): {ens:.3f}", color=MUTED, fontsize=8.5, va="bottom")
    off = {"ridge": (8, -2, "left"), "gbm": (-9, -16, "right"), "pf_struct": (9, -13, "left"), "pf_nostruct": (9, 3, "left"), "tiger2d": (9, 3, "left"),
           "tabdpt": (72, -36, "right"), "rnafm_probe": (-9, 6, "right")}
    for (l, c, p, f), r in zip(pts, rows):
        col, mk, _ = fam[f]
        ax.scatter([p], [r["within_gene_spearman"]], s=64, marker=mk, color=col, edgecolor="#fcfcfb", linewidth=1.4, zorder=3)
        dx, dy, ha = off[c]
        ax.annotate(f"{l}  {r['within_gene_spearman']:.3f}", (p, r["within_gene_spearman"]), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=5) if c == "tabdpt" else None)
    ax.set_xscale("log")
    ax.set_xlim(10 ** 1.6, 10 ** 8.9)
    ax.set_ylim(0.42, 0.68)
    ax.set_xlabel("model parameters (log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("held-out-gene accuracy\n(within-gene Spearman)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    handles = [plt.Line2D([], [], marker=m, color="none", markerfacecolor=c, markeredgecolor="#fcfcfb", markersize=8, label=lab) for c, m, lab in fam.values()]
    ax.legend(handles=handles, loc="lower right", fontsize=7.8, frameon=False, labelcolor=MUTED)
    fig.suptitle("Beyond ~10,000 parameters, bigger models are not more accurate on held-out genes", x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.01, 0.925, "Cas13d guide activity; each gene predicted by models that never saw it (12 development genes).", fontsize=8.2, color=MUTED, ha="left")
    fig.text(0.01, 0.895, "The replicate-noise ceiling (correlation with the observed mean) is about 0.89, well above the top of this chart.", fontsize=8.2, color=MUTED, ha="left")
    fig.subplots_adjust(top=0.84, left=0.1, right=0.98, bottom=0.11)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(a.out, f"params_vs_accuracy.{ext}"), facecolor="#fcfcfb")
    print("saved", os.path.join(a.out, "params_vs_accuracy.png"))


if __name__ == "__main__":
    main()
