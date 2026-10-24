"""Does deviation from a healthy-aging manifold carry disease signal beyond age acceleration?  (linear embedding, no network)

Reference = ComputAge TRAIN split (healthy controls) only.  Embedding = PCA(k) of the standardised compact-panel profile.
manifold(age) = spline ridge of the embedding on age; deviation = Mahalanobis distance (Ledoit-Wolf) of the residual.
age acceleration = ridge clock prediction - age, residualised on age in the reference.
Evaluation = ComputAge BENCHMARK split, within each study with controls and cases: AUC(case vs control) for
|AA|, signed AA and deviation; cluster (study) bootstrap over studies."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from epimod.age import age_to_t, t_to_age
from epimod.lb.store import lb_meta, load_lb_store

K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
meta = lb_meta(None, geo=False)
st = load_lb_store(include_series=set(meta.DatasetID), geo=False)
m = st.meta; X = np.asarray(st.prof, dtype=np.float32); X = np.where(X == 255, np.nan, X / 100.0)
ref = (m.split == "train").to_numpy() & (m.Class == "HC").to_numpy(); ben = (m.split == "benchmark").to_numpy()
print(f"reference HC (train split) {ref.sum()} samples / {m[ref].DatasetID.nunique()} studies | benchmark split {ben.sum()} samples / {m[ben].DatasetID.nunique()} studies", flush=True)
mu = np.nanmean(X[ref], 0); mu = np.where(np.isfinite(mu), mu, 0.5); sd = np.maximum(np.nanstd(X[ref], 0), 0.02)
Z = (np.where(np.isfinite(X), X, mu) - mu) / sd
age = m.Age.to_numpy(float)
clock = Ridge(alpha=3000.0).fit(Z[ref], age_to_t(age[ref])); A = t_to_age(clock.predict(Z))
lin = np.polyfit(age[ref], A[ref] - age[ref], 1); AA = A - age - np.polyval(lin, age)
pca = PCA(K, random_state=0).fit(Z[ref]); E = pca.transform(Z)
man = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(alpha=5.0)).fit(age[ref, None], E[ref])
R = E - man.predict(age[:, None]); P = LedoitWolf().fit(R[ref]).precision_
dev = np.sqrt(np.einsum("bi,ij,bj->b", R, P, R))
print(f"clock MAE on reference {np.abs(A[ref]-age[ref]).mean():.2f} y | deviation (reference) median {np.median(dev[ref]):.2f}, benchmark HC median {np.median(dev[ben & (m.Class=='HC').to_numpy()]):.2f}", flush=True)

rows = []
for (ds, cond), g in m[ben].groupby(["DatasetID", "Condition"]):
    if cond == "HC":
        continue
    c_idx = np.flatnonzero(ben & (m.DatasetID == ds).to_numpy() & (m.Condition == cond).to_numpy()); h_idx = np.flatnonzero(ben & (m.DatasetID == ds).to_numpy() & (m.Class == "HC").to_numpy())
    if len(c_idx) < 8 or len(h_idx) < 8:
        continue
    y = np.r_[np.ones(len(c_idx)), np.zeros(len(h_idx))]; ix = np.r_[c_idx, h_idx]
    rows.append({"study": ds, "condition": cond, "n_case": len(c_idx), "n_ctrl": len(h_idx),
                 "auc_dev": roc_auc_score(y, dev[ix]), "auc_AA": roc_auc_score(y, AA[ix]), "auc_absAA": roc_auc_score(y, np.abs(AA[ix])),
                 "mean_age_case": age[c_idx].mean(), "mean_age_ctrl": age[h_idx].mean()})
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False), flush=True)
rng = np.random.RandomState(0)
def summ(col):
    v = r[col].to_numpy(); b = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(2000)]
    return f"{v.mean():.3f} [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]"
print(f"\n{len(r)} study-condition tests | mean AUC  deviation {summ('auc_dev')} | signed AA {summ('auc_AA')} | |AA| {summ('auc_absAA')}", flush=True)
d = (r.auc_dev - r.auc_absAA).to_numpy(); b = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
print(f"deviation - |AA| mean AUC difference {d.mean():+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]; tests with AUC>0.6: dev {int((r.auc_dev>0.6).sum())}, AA {int((r.auc_AA>0.6).sum())}, |AA| {int((r.auc_absAA>0.6).sum())}", flush=True)
r.to_csv(sys.argv[1] if len(sys.argv) > 1 else "dev_linear.csv", index=False)
