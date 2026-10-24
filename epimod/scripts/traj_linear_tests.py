"""Do individual methylation trajectories exist beyond the population drift?  (E-MTAB-7309, linear tests, no neural network)

Test 1  persistent individual velocity (CpG level): for people with >= 4 visits, the velocity of the first interval
        (visit1->2) vs the velocity of a NON-overlapping later interval (visit3->4); with overlapping intervals measurement
        noise in the shared visit induces a negative correlation, so they are not used.  Velocity = (dBeta/dt) minus the
        population drift (estimated on training families only), on the 1000 CpGs with the largest training drift.
Test 2  the same at clock level (ridge age predictor trained on the cross-sectional corpus).
Test 3  twin concordance: correlation of co-twins' clock velocity (first -> last visit).
Test 4  out-of-sample forecast: next-interval beta from (persistence | + population drift | + shrunk individual velocity),
        lambda fitted on training families; error compared on held-out families (cluster bootstrap by family)."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

from epimod.age import age_to_t
from epimod.lb.store import lb_meta, lb_series, load_lb_store
from epimod.lb.tasks import load_parsed
from epimod.paths import subdir

rng = np.random.RandomState(0)
z = np.load(subdir("lb") / "longitudinal.npz"); X = z["X"].astype(np.float32); sid = z["sample_ids"]
meta = pd.read_csv(subdir("lb") / "longitudinal_meta.csv").set_index("sample_id").loc[sid].reset_index()
X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
fam = meta.family_id.astype(str).to_numpy(); subj = meta.subject_id.astype(str).to_numpy(); age = meta.age.to_numpy(float)
fams = np.unique(fam); rng.shuffle(fams); test_f = set(fams[: int(0.3 * len(fams))])
is_test = np.array([f in test_f for f in fam])
print(f"{len(meta)} arrays | {len(set(subj))} people | {len(fams)} twin pairs | test families {len(test_f)}", flush=True)

# --- clock-level quantity: ridge age predictor trained on the cross-sectional corpus (ComputAge + GEO, benchmark series excluded)
excl = lb_series(load_parsed()); st = load_lb_store(exclude_series=excl)
Xs = np.asarray(st.prof, dtype=np.float32); Xs = np.where(Xs == 255, np.nan, Xs / 100.0)
mu_c = np.nanmean(Xs, 0); mu_c = np.where(np.isfinite(mu_c), mu_c, 0.5); sd_c = np.maximum(np.nanstd(Xs, 0), 0.02)
Zs = (np.where(np.isfinite(Xs), Xs, mu_c) - mu_c) / sd_c
clock = Ridge(alpha=3000.0).fit(Zs, age_to_t(st.meta.Age.to_numpy(float)))
from epimod.age import t_to_age
A = t_to_age(clock.predict((X - mu_c) / sd_c))
print(f"clock on twin arrays: corr with age {np.corrcoef(A, age)[0,1]:.3f}, MAE {np.abs(A-age).mean():.2f} y, bias {np.mean(A-age):+.2f} y", flush=True)

# --- visits per person, sorted by age
order = {}
for s in np.unique(subj):
    idx = np.flatnonzero(subj == s); order[s] = idx[np.argsort(age[idx])]
sfam = {s: fam[order[s][0]] for s in order}
def intervals(ids, min_dt=1.0):
    return [(order[s][k], order[s][k + 1]) for s in ids for k in range(len(order[s]) - 1) if age[order[s][k + 1]] - age[order[s][k]] >= min_dt]
train_ids = [s for s in order if sfam[s] not in test_f]; test_ids = [s for s in order if sfam[s] in test_f]
tr_pairs = intervals(train_ids)
# population drift per CpG and for the clock (train people only)
drift = np.mean([(X[j] - X[i]) / (age[j] - age[i]) for i, j in tr_pairs], 0)
a_drift = np.mean([(A[j] - A[i]) / (age[j] - age[i]) for i, j in tr_pairs])
top = np.argsort(-np.abs(drift))[:1000]
print(f"population drift: clock {a_drift:.3f} y/y | top-1000 CpGs mean |drift| {np.abs(drift[top]).mean():.5f} /y", flush=True)

def boot_ci(vals, groups, fn, B=1000):
    g = pd.factorize(groups)[0]; G = g.max() + 1; out = []
    for _ in range(B):
        w = np.bincount(rng.randint(0, G, G), minlength=G)[g]; out.append(fn(w))
    return np.percentile(out, [2.5, 97.5])

# --- Test 1/2: non-overlapping intervals (visits 1->2 and 3->4) among people with >= 4 visits
quad = [s for s in order if len(order[s]) >= 4]
def vel(i, j, f): return (f[j] - f[i]) / (age[j] - age[i])
rows = []
for s in quad:
    o = order[s]
    if min(age[o[1]] - age[o[0]], age[o[3]] - age[o[2]]) >= 1.0:
        rows.append((s, o[0], o[1], o[2], o[3]))
print(f"people with >= 4 visits and both intervals >= 1 y: {len(rows)}", flush=True)
v1 = np.array([vel(a, b, X)[top] for _, a, b, _, _ in rows]) - drift[top]; v2 = np.array([vel(c, d, X)[top] for _, _, _, c, d in rows]) - drift[top]
c_cpg = np.array([np.corrcoef(v1[:, k], v2[:, k])[0, 1] for k in range(v1.shape[1])])
r_pool = float(np.corrcoef(v1.ravel() / v1.std(0).repeat(len(rows)).reshape(v1.shape[1], -1).T.ravel(), v2.ravel() / v2.std(0).repeat(len(rows)).reshape(v2.shape[1], -1).T.ravel())[0, 1])
print(f"Test 1 CpG velocity persistence: mean per-CpG correlation {np.nanmean(c_cpg):+.3f} (fraction > 0: {np.nanmean(c_cpg > 0):.2f}); pooled standardised r = {r_pool:+.3f}", flush=True)
a1 = np.array([vel(a, b, A) for _, a, b, _, _ in rows]) - a_drift; a2 = np.array([vel(c, d, A) for _, _, _, c, d in rows]) - a_drift
fam_rows = np.array([sfam[r[0]] for r in rows])
r_clock = float(np.corrcoef(a1, a2)[0, 1]); lo, hi = boot_ci(None, fam_rows, lambda w: np.cov(a1, a2, aweights=w + 1e-9)[0, 1] / np.sqrt(np.cov(a1, aweights=w + 1e-9) * np.cov(a2, aweights=w + 1e-9)))
print(f"Test 2 clock velocity persistence: r = {r_clock:+.3f}  (95% family-bootstrap CI [{lo:+.3f}, {hi:+.3f}], n={len(rows)}); velocity SD {a1.std():.2f} y/y", flush=True)

# --- Test 3: co-twin concordance of clock velocity (first -> last visit)
tv = {}
for s in order:
    o = order[s]
    if len(o) >= 2 and age[o[-1]] - age[o[0]] >= 2.0:
        tv[s] = (A[o[-1]] - A[o[0]]) / (age[o[-1]] - age[o[0]]) - a_drift
pairs = []
for f in fams:
    members = [s for s in tv if sfam[s] == f]
    if len(members) == 2:
        pairs.append((tv[members[0]], tv[members[1]], f))
if len(pairs) > 5:
    p = np.array([(a, b) for a, b, _ in pairs]); r_tw = np.corrcoef(p[:, 0], p[:, 1])[0, 1]
    lo, hi = boot_ci(None, np.array([f for *_, f in pairs]), lambda w: np.cov(p[:, 0], p[:, 1], aweights=w + 1e-9)[0, 1] / np.sqrt(np.cov(p[:, 0], aweights=w + 1e-9) * np.cov(p[:, 1], aweights=w + 1e-9)))
    print(f"Test 3 co-twin clock-velocity correlation: r = {r_tw:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(pairs)} twin pairs", flush=True)

# --- Test 4: forecast of the next visit from (x_prev, x_cur): persistence | +population drift | +shrunk individual velocity
def triples(ids):
    out = []
    for s in ids:
        o = order[s]
        for k in range(len(o) - 2):
            a, b, c = o[k], o[k + 1], o[k + 2]
            if age[b] - age[a] >= 1.0 and age[c] - age[b] >= 1.0:
                out.append((a, b, c, s))
    return out
trt, tet = triples(train_ids), triples(test_ids)
def feats(t):
    a, b, c = np.array([x[0] for x in t]), np.array([x[1] for x in t]), np.array([x[2] for x in t])
    dt1, dt2 = (age[b] - age[a])[:, None], (age[c] - age[b])[:, None]
    base = X[b][:, top] + dt2 * drift[top]; ind = dt2 * ((X[b][:, top] - X[a][:, top]) / dt1 - drift[top]); y = X[c][:, top]
    return base, ind, y, X[b][:, top]
b0, i0, y0, _ = feats(trt); lam = float(((y0 - b0) * i0).sum() / (i0 * i0).sum())
b1, i1, y1, pers = feats(tet)
e_pers, e_pop, e_ind = np.abs(y1 - pers).mean(1), np.abs(y1 - b1).mean(1), np.abs(y1 - (b1 + lam * i1)).mean(1)
fm = np.array([sfam[t[3]] for t in tet])
for nm, e, ref in (("population drift vs persistence", e_pop, e_pers), ("individual velocity vs population drift", e_ind, e_pop)):
    d = e - ref; lo, hi = boot_ci(None, fm, lambda w: (w * d).sum() / w.sum())
    print(f"Test 4 {nm}: dMAE = {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] (MAE {ref.mean():.5f}); lambda={lam:+.3f}; n test triples {len(tet)}", flush=True)
json.dump({"n_arrays": len(meta), "corr_age_clock": float(np.corrcoef(A, age)[0, 1]), "test1_mean_cpg_corr": float(np.nanmean(c_cpg)), "test1_pooled_r": r_pool,
           "test2_r": r_clock, "lambda": lam}, open(sys.argv[1] if len(sys.argv) > 1 else "traj_linear.json", "w"), indent=1)
