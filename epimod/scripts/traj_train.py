"""Dynamics stage for EpiTrajectory-L on the twin cohort (E-MTAB-7309).

Starts from a pretrained hierarchical checkpoint; trains the latent-ODE forecast head, the time-direction head and (at a low
learning rate) the encoder with an age loss that keeps the clock from drifting.  Families are split into train / validation /
test (twins stay together).  Reports, on held-out families with a family cluster bootstrap:
  forecast MAE of  persistence | persistence + population drift | model        (all vocabulary CpGs and the 1000 largest-drift CpGs)
  cosine between predicted and observed change, time-direction AUC, and the same model trained on subject-permuted targets (negative control)
    python scripts/traj_train.py runs/lb/hier_a/model.pt out_dir [--epochs 8] [--perm]"""
import argparse, copy, json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from torch import nn

from epimod.age import age_to_t, t_to_age
from epimod.data.annotation import load_probe_annotation
from epimod.lb.episet import ROLES
from epimod.lb.epitraj import EpiHier, HierConfig, block_assignment
from epimod.lb.tasks import Vocab
from epimod.lb.trajdata import family_split, load_longitudinal, make_pairs
from epimod.paths import subdir

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--vis", type=float, default=0.3); ap.add_argument("--perm", action="store_true")
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
uni = pd.Index(pd.read_parquet(subdir("lb") / "universe.parquet")["cpg"].values)          # legacy universe (unused columns ignored)
from epimod.lb.lite import load_lite_universe
uni = load_lite_universe()
ids = pd.Index(ck["vocab_ids"]); u2m = np.full(len(uni), -1, np.int64); pos = uni.get_indexer(ids); u2m[pos[pos >= 0]] = np.flatnonzero(pos >= 0)
hc = HierConfig(**ck["cfg"]); model = EpiHier(hc, block_assignment(ids, load_probe_annotation(), hc.block_size))
model.load_state_dict(ck["state"]); vocab = Vocab(u2m, ids, model.cpg_mean[:-1].numpy(), model.cpg_std[:-1].numpy())
X, meta = load_longitudinal(vocab); V = X.shape[1]; age = meta.age.to_numpy(float); subj = pd.factorize(meta.subject_id)[0]
tr, va, te = family_split(meta, a.seed)
pairs = {k: make_pairs(meta, r) for k, r in (("train", tr), ("val", va), ("test", te))}
print(f"{len(meta)} arrays, {V} vocabulary CpGs | pairs train/val/test {[len(v) for v in pairs.values()]} | families {meta.family_id.nunique()}", flush=True)
sex = torch.as_tensor(meta.sex.map({"M": 1, "F": 2}).fillna(0).to_numpy(np.int64))
Xt = torch.as_tensor(np.nan_to_num(X, nan=0.5)); obs = torch.as_tensor(np.isfinite(X))

if a.perm:   # negative control: pair every earlier sample with the later sample of a DIFFERENT subject
    P = pairs["train"]; later = np.array([j for _, j, _ in P]); si = np.array([subj[i] for i, _, _ in P]); sj = subj[later]
    perm = rng.permutation(len(P))
    for _ in range(100):
        bad = np.flatnonzero(si == sj[perm])
        if not len(bad): break
        perm[bad] = rng.permutation(perm[bad])
    pairs["train"] = [(i, int(later[p]), dt) for (i, _, dt), p in zip(P, perm)]


def tokens(rows, vis, rs):
    """visible subset of the observed CpGs of each row -> (cpg ids, beta, role, sex)."""
    o = obs[rows]; u = torch.as_tensor(rs.rand(*o.shape)); keep = o & (u < vis)
    t = int(keep.sum(1).max()); key = torch.where(keep, u, torch.full_like(u, 2.0)); idx = key.argsort(1)[:, :t]
    ok = keep.gather(1, idx)
    return torch.where(ok, idx, torch.full_like(idx, -1)), (Xt[rows].gather(1, idx) * ok), torch.full(idx.shape, ROLES["fixed"]), sex[rows]


def encode(rows, vis, rs):
    c, b, r, s = tokens(rows, vis, rs); return model.encode(c, b, r, s)[0]


def age_nll(h, rows):
    mu, raw = model.age_head(h).unbind(-1); sig = nn.functional.softplus(raw - 1.0) + 1e-2
    y = torch.as_tensor(age_to_t(age[rows]), dtype=torch.float32); return (0.5 * ((y - mu) / sig) ** 2 + torch.log(sig)).mean()


def batch_loss(P, rs):
    i = np.array([p[0] for p in P]); j = np.array([p[1] for p in P]); dt = torch.as_tensor([p[2] for p in P], dtype=torch.float32)
    h1, h2 = encode(i, a.vis, rs), encode(j, a.vis, rs)
    probe = torch.as_tensor(rs.permutation(V)[:1024])
    hf, xh = model.forecast(h1, dt, Xt[i], probe)
    ok = obs[j][:, probe] & obs[i][:, probe]
    err = (xh - Xt[j][:, probe]).abs(); l_fut = torch.where(err < 0.05, 0.5 * err ** 2 / 0.05, err - 0.025)[ok].mean()
    l_lat = nn.functional.smooth_l1_loss(hf, h2.detach())
    lf, lr_ = model.direction_logit(h1, h2), model.direction_logit(h2, h1)
    l_dir = 0.5 * (nn.functional.binary_cross_entropy_with_logits(lf, torch.ones_like(lf)) + nn.functional.binary_cross_entropy_with_logits(lr_, torch.zeros_like(lr_)))
    l_age = 0.5 * (age_nll(h1, i) + age_nll(h2, j))
    return 1.0 * l_fut * 20 + 0.25 * l_lat + 0.1 * l_dir + 0.3 * l_age


@torch.no_grad()
def evaluate(P, rs, n_draw=3):
    model.eval(); res = {k: [] for k in ("model", "persist", "cos", "fam", "dt", "lf", "lr", "dcl")}
    for s in range(0, len(P), 16):
        Q = P[s:s + 16]; i = np.array([p[0] for p in Q]); j = np.array([p[1] for p in Q]); dt = torch.as_tensor([p[2] for p in Q], dtype=torch.float32)
        hs = [(encode(i, a.vis, rs), encode(j, a.vis, rs)) for _ in range(n_draw)]
        h1 = torch.stack([h[0] for h in hs]).mean(0); h2 = torch.stack([h[1] for h in hs]).mean(0)
        _, xh = model.forecast(h1, dt, Xt[i], torch.arange(V)); w = (obs[i] & obs[j]).float(); n = w.sum(1).clamp_min(1)
        res["model"].append(((xh - Xt[j]).abs() * w).sum(1).numpy() / n.numpy()); res["persist"].append(((Xt[i] - Xt[j]).abs() * w).sum(1).numpy() / n.numpy())
        dtrue, dpred = (Xt[j] - Xt[i]) * w, (xh - Xt[i]) * w
        res["cos"].append(((dtrue * dpred).sum(1) / (dtrue.norm(dim=1) * dpred.norm(dim=1) + 1e-8)).numpy())
        res["fam"].append(meta.family_id.to_numpy()[i]); res["dt"].append(dt.numpy())
        res["lf"].append(model.direction_logit(h1, h2).numpy()); res["lr"].append(model.direction_logit(h2, h1).numpy())
        A1, A2 = [t_to_age(model.age_head(h).detach()[:, 0].numpy().astype(np.float64)) for h in (h1, h2)]
        res["dcl"].append(A2 - A1)
    return {k: np.concatenate(v) for k, v in res.items()}


opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
best, best_state, hist = 1e9, None, []
for ep in range(1, a.epochs + 1):
    model.train(); P = [pairs["train"][k] for k in rng.permutation(len(pairs["train"]))]; t0 = time.time(); tot = []
    for s in range(0, len(P) - a.bs + 1, a.bs):
        loss = batch_loss(P[s:s + a.bs], rng); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot.append(float(loss))
    r = evaluate(pairs["val"], np.random.RandomState(1)); vscore = float(r["model"].mean() / r["persist"].mean())
    hist.append({"epoch": ep, "train_loss": float(np.mean(tot)), "val_model_over_persistence": vscore, "min": round((time.time() - t0) / 60, 1)}); print(hist[-1], flush=True)
    if vscore < best:
        best, best_state = vscore, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)

# ---- held-out families: baselines and model
r = evaluate(pairs["test"], np.random.RandomState(2)); P = pairs["test"]
tri = [p for p in pairs["train"]]
drift = np.zeros(V); cnt = np.zeros(V)
for i, j, dt in tri:
    ok = np.isfinite(X[i]) & np.isfinite(X[j]); drift += np.where(ok, (X[j] - X[i]) / dt, 0.0); cnt += ok
drift /= np.maximum(cnt, 1); top = np.argsort(-np.abs(drift))[:1000]
slope_err = np.array([np.nanmean(np.abs(np.clip(X[i] + dt * drift, 0, 1) - X[j])) for i, j, dt in P])
slope_top = np.array([np.nanmean(np.abs(np.clip(X[i] + dt * drift, 0, 1) - X[j])[top]) for i, j, dt in P])
pers_top = np.array([np.nanmean(np.abs(X[i] - X[j])[top]) for i, j, dt in P])
fam = r["fam"]; code = pd.factorize(fam)[0]; G = code.max() + 1; brng = np.random.RandomState(3)
def ci(v):
    b = [(w := np.bincount(brng.randint(0, G, G), minlength=G)[code])[None] @ v / w.sum() for _ in range(1000)]; return float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
m_, p_ = r["model"], r["persist"]
summary = {"n_test_pairs": len(P), "future_MAE_model": ci(m_), "future_MAE_persistence": ci(p_), "future_MAE_population_drift": ci(slope_err),
           "model_minus_persistence": ci(m_ - p_), "model_minus_population_drift": ci(m_ - slope_err),
           "delta_cosine": ci(r["cos"]), "direction_auc": float(roc_auc_score(np.r_[np.ones(len(r["lf"])), np.zeros(len(r["lr"]))], 1 / (1 + np.exp(-np.r_[r["lf"], r["lr"]])))),
           "clock_change_vs_dt_corr": float(np.corrcoef(r["dcl"], r["dt"])[0, 1]), "history": hist, "perm_control": bool(a.perm)}
print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1), flush=True)
json.dump(summary, open(out / ("traj_perm.json" if a.perm else "traj.json"), "w"), indent=1)
torch.save({"state": model.state_dict(), "cfg": ck["cfg"], "vocab_ids": ck["vocab_ids"], "args": {**ck["args"], "traj": True}}, out / ("model_traj_perm.pt" if a.perm else "model_traj.pt"))
