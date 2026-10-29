"""M4 extension stage: candidate improvements that need long compute. DEV-FIRST so the benchmark stays clean.

  python -m guidefam.scripts.m4_extend dev    --data DATA --pred PRED --out OUT     # trains everything, prints scores on the 12 DEVELOPMENT genes only
  python -m guidefam.scripts.m4_extend report --data DATA --pred PRED --out OUT --use gbm,tiger_bag,konermann   # locked genes + external, for ONE combination you chose

Candidates (each is a leave-one-gene-out model; ~1-2 h each on an M4 with 8-10 threads, Tiger2D-based ones dominate):
  gbm_tuned     LightGBM with a small random search, hyper-parameters chosen by an INNER leave-one-gene-out on the training genes only
  tiger_std     Tiger2D with standardised scalar inputs (the authors feed raw MFE/hybridisation values to sigmoid layers)
  tiger_bag     Tiger2D, mean of N seeds (variance reduction); N = --bag (default 5)
  tabdpt        TabDPT (pretrained tabular foundation model, open safetensors weights) as the learner on the tabular features
Published tool scores (konermann, rf, deepcas13) are attached for use in the combination.

Rules the script enforces: `dev` only ever prints the 12 development genes; combinations are rank-averages WITHIN gene; `report` refuses to run unless
--use is given (so the choice is made after looking at dev), and records it in OUT/chosen.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .. import comparators as C
from .. import cv
from .. import data as D
from .. import features as F
from .. import metrics as M


def rk(df, col):
    return df.groupby("gene")[col].rank(pct=True)


def random_search_gbm(X, train, ya, n_iter, seed):
    """Pick GBM hyper-parameters by inner leave-one-gene-out over the TRAINING genes only (never the scored gene)."""
    rng = np.random.default_rng(seed)
    grid = dict(n_estimators=[300, 600, 1000], learning_rate=[0.02, 0.03, 0.05], num_leaves=[7, 15, 31], min_child_samples=[10, 20, 40],
                colsample_bytree=[0.3, 0.5, 0.8], reg_lambda=[1.0, 5.0, 20.0])
    genes = sorted(train.gene.unique())
    best, best_s = None, -9
    for _ in range(n_iter):
        p = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        sc = []
        for g in genes[:: max(1, len(genes) // 4)]:                     # 4 inner folds are enough to rank settings
            te = train.gene == g
            m = cv.GBM(X, **p).fit(train[~te], cv.activity_target(train[~te]))
            sc.append(M.spearman(train[te].act, m.predict(train[te])))
        if np.mean(sc) > best_s:
            best, best_s = p, float(np.mean(sc))
    return best


def make(name, X, hek, a):
    from ..tiger_torch import Tiger2DRegressor
    if name == "tiger_std":
        return lambda: Tiger2DRegressor(flank=1, scalars=True, standardize=True)
    if name == "tiger_bag":
        class Bag:
            def fit(self, d, ya):
                self.ms = [Tiger2DRegressor(flank=1, scalars=True, seed=s).fit(d, ya) for s in range(a.bag)]
                return self

            def predict(self, d):
                return np.mean([m.predict(d) for m in self.ms], axis=0)
        return Bag
    if name == "tabdpt":
        return lambda: cv.TabDPT(X, n_ensembles=8)
    if name == "gbm_tuned":
        class Tuned:
            def fit(self, d, ya):
                self.p = random_search_gbm(X, d, ya, a.search, 0)
                self.m = cv.GBM(X, **self.p).fit(d, ya)
                return self

            def predict(self, d):
                return self.m.predict(d)
        return Tuned
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="gbm_tuned,tiger_std,tiger_bag,tabdpt")
    ap.add_argument("--base-oof", default=None, help="hek_oof.csv from final_cas13d (columns gbm, tiger2d_port) to reuse as members")
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--search", type=int, default=12)
    ap.add_argument("--use", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 genes, tiny budgets: checks the plumbing only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hek = C.add_all(D.load_tiger(a.data, "off-target"), a.pred)
    X = F.tabular(hek, "full", flank=2)
    path = os.path.join(a.out, "m4_oof.csv")
    oof = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame(index=hek.index)
    if a.base_oof:
        b = pd.read_csv(a.base_oof, index_col=0)
        for c in ("gbm", "tiger2d_port"):
            oof[c] = b[c].reindex(hek.index)
    genes = sorted(hek.gene.unique())
    if a.smoke:
        genes = genes[:2]
    if a.mode == "dev":
        for name in a.models.split(","):
            if name in oof and oof.loc[hek.gene.isin(genes), name].notna().all():
                continue
            t0 = time.time()
            mk = make(name, X, hek, a)
            if a.smoke and name.startswith("tiger"):
                from ..tiger_torch import Tiger2DRegressor
                mk = lambda: Tiger2DRegressor(max_epochs=3)
            p = cv.logo_predict(hek, mk, genes=genes, verbose=True)           # ALL genes are predicted so `report` can run later; dev prints only dev
            oof[name] = np.where(np.isnan(p), oof[name] if name in oof else np.nan, p)
            oof.to_csv(path)
            print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
        ev = hek.join(oof)
        ev = ev[~ev.gene.isin(D.LOCKED_GENES)]
        print("\n12 DEVELOPMENT genes only (locked genes and external screens are not printed in this mode)")
        for c in list(oof.columns) + ["konermann", "rf"]:
            if c in ev and ev[c].notna().any():
                x = ev[ev[c].notna()]
                print(f"  {c:14s} within-gene Spearman {M.within_group(x, 'gene', 'act', c):.4f}   AUROC {M.within_group(x, 'gene', 'active', c, M.auroc):.4f}")
        cands = [c for c in oof.columns if c in ev and ev[c].notna().any()] + ["konermann"]
        best = []
        for k in (2, 3, 4):
            import itertools
            for combo in itertools.combinations(cands, k):
                z = ev.copy()
                z["e"] = sum(rk(z, c) for c in combo) / len(combo)
                best.append((M.within_group(z, "gene", "act", "e"), combo))
        print("\ntop dev combinations (rank-average within gene):")
        for s, combo in sorted(best, reverse=True)[:6]:
            print(f"  {s:.4f}  {'+'.join(combo)}")
        print("\nPick ONE combination from this list, then:  m4_extend report ... --use <comma-separated members>")
        return

    if not a.use:
        raise SystemExit("report needs --use (choose a combination from the dev table first)")
    use = a.use.split(",")
    json.dump({"use": use, "chosen_after_dev_only": True}, open(os.path.join(a.out, "chosen.json"), "w"))
    ev = hek.join(oof)
    ev["chosen"] = sum(rk(ev, c) for c in use) / len(use)
    ev["ref_tiger"] = ev["tiger2d_port"] if "tiger2d_port" in ev else np.nan
    for label, m in (("all 16 genes", ev.gene.notna()), ("4 LOCKED genes", ev.gene.isin(D.LOCKED_GENES))):
        x = ev[m].reset_index(drop=True)
        print(f"\n=== {label}: {'+'.join(use)}")
        for c in ("chosen", "ref_tiger"):
            r = M.block_bootstrap(x, lambda z, c=c: M.summarize(z, c), n=500)
            print(f"  {c:10s} within-gene {r['rho_within'][0]:.3f} [{r['rho_within'][1]:.3f},{r['rho_within'][2]:.3f}]  AUROC_within {r['auroc_within'][0]:.3f}")
        d0, lo, hi, sh = M.paired_bootstrap(x, "chosen", "ref_tiger", lambda z, col: M.within_group(z, "gene", "act", col), n=500)
        print(f"  paired chosen - Tiger2D: {d0:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("\nExternal screens: re-run final_cas13d.py (it trains on all HEK293T genes); swap members there if you adopt new ones.")


if __name__ == "__main__":
    main()
