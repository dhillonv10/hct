"""Development: does a frozen RNA-FM embedding of the 23-nt target add anything to the GBM? (12 development genes only)"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .. import cv, lm
from .. import data as D
from .. import features as F
from .. import metrics as M


class PCAGBM(cv.GBM):
    """GBM on the tabular features plus PCA components of the embedding (PCA fit on the training rows only)."""

    def __init__(self, X, E, k, **kw):
        super().__init__(X, **kw)
        self.E, self.k = E, k

    def _x(self, d, fit=False):
        Z = self.E.loc[d.index].values
        if fit:
            self.pca = PCA(self.k, random_state=0).fit(Z)
        return pd.concat([self.X.loc[d.index, self.cols], pd.DataFrame(self.pca.transform(Z), index=d.index, columns=[f"pc{i}" for i in range(self.k)])], axis=1)

    def fit(self, d, ya):
        self.m = self.lgb.LGBMRegressor(**self.params).fit(self._x(d, True), ya)
        return self

    def predict(self, d):
        return self.m.predict(self._x(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rnafm", required=True)
    ap.add_argument("--emb-cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = D.load_tiger(a.data, "off-target")
    X = F.tabular(d, "full", flank=2)
    E = pd.DataFrame(lm.embed(d.target_seq.tolist(), a.rnafm, a.emb_cache, batch=64), index=d.index)
    dev_genes = [g for g in sorted(d.gene.unique()) if g not in D.LOCKED_GENES]
    path = os.path.join(a.out, "dev_oof.csv")
    oof = pd.read_csv(path, index_col=0)
    for k in (8, 32):
        nm = f"rnafm23_pca{k}"
        if nm not in oof:
            oof[nm] = cv.logo_predict(d, lambda k=k: PCAGBM(X, E, k, n_jobs=1), genes=dev_genes, verbose=False)
            oof.to_csv(path)
    # embedding alone (ridge) vs one-hot alone (ridge): is the LM telling us anything the sequence does not?
    oh = [c for c in X.columns if c.startswith("g") and "_" in c and c[1].isdigit()]
    Xe = pd.concat([X[oh], E.add_prefix("e")], axis=1)
    if "ridge_onehot" not in oof:
        oof["ridge_onehot"] = cv.logo_predict(d, lambda: cv.Ridge(X, cols=oh, alpha=30.0), genes=dev_genes, verbose=False)
        oof["ridge_rnafm23"] = cv.logo_predict(d, lambda: cv.Ridge(Xe, cols=list(E.add_prefix("e").columns), alpha=300.0), genes=dev_genes, verbose=False)
        oof.to_csv(path)
    dev = d[d.gene.isin(dev_genes)].join(oof, how="inner")
    for c in ("base", "rnafm23_pca8", "rnafm23_pca32", "ridge_onehot", "ridge_rnafm23"):
        print(f"{c:16s} within-gene rho {M.within_group(dev, 'gene', 'act', c):.4f}")


if __name__ == "__main__":
    main()
