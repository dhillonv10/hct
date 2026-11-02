"""Pilot driver: four stages, each resumable, each ending in an artifact that stands on its own.

  python -m rnakd.pilot selftest                      # whole path on synthetic data + a random tiny teacher (no network, no GPU)
  python -m rnakd.pilot data     --work W
  python -m rnakd.pilot teacher  --work W --ckpt-ft rinalmo_giga_mrl_ft.pt [--ckpt-lm rinalmo_giga_pretrained.pt]
  python -m rnakd.pilot train    --work W [--grid main|controls|frac|all]
  python -m rnakd.pilot report   --work W

Every trained run appends one row to `runs.csv`; a run whose key is already present is skipped, so an interrupted session
resumes by re-issuing the same command. Nothing in `train` touches the teacher: it reads cached features, so the expensive
pass happens once.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kd, mrl
from .student import PRESETS, build
from .teacher import PAD, TeacherMRL, dump_keys, encode, load_strict

SPLITS = ["train", "val", "iid_test", "ood_test", "repro_val"]
RATE = []                                                                       # teacher sequences/second, filled by teacher_pass
PUBLISHED = {"repro_val": 0.925, "ood_test": 0.860}                             # RiNALMo-giga, arXiv 2403.00043v2 Table S6
PUBLISHED_33M = {"repro_val": 0.884, "ood_test": 0.827}


# ------------------------------------------------------------------------------------------------------------------- helpers
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def spearman(y, p):
    a, b = pd.Series(y).rank().values, pd.Series(p).rank().values
    return float(np.corrcoef(a, b)[0, 1])


def boot_r2(y, p, n=1000, seed=0):
    rs = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.asarray(p, float)
    v = [r2(y[i], p[i]) for i in (rs.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def tokenise(seqs, pad_to):
    return np.asarray([encode(s, pad_to) for s in seqs], dtype=np.int64)


def device_of(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ledger:
    """Append-only record of finished work, so a disconnect costs minutes."""

    def __init__(self, path):
        self.path = path
        self.rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
        self.done = {r["key"] for r in self.rows}

    def add(self, row):
        self.rows.append(row)
        self.done.add(row["key"])
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


# ------------------------------------------------------------------------------------------------------------------- stage: data
def stage_data(work, synthetic=False, seed=0, top_k=100):
    os.makedirs(work, exist_ok=True)
    csv = os.path.join(work, "mrl_synth.csv")
    if synthetic:
        if not os.path.exists(csv):
            mrl.synthetic().to_csv(csv, index=False)
    else:
        csv = mrl.download(work)
    sp = mrl.splits(csv, seed=seed, top_k=top_k)
    pad_to = int(max(s.utr.str.len().max() for s in sp.values())) + 2
    np.savez_compressed(os.path.join(work, "data.npz"), pad_to=pad_to,
                        **{f"{k}_x": tokenise(v.utr.tolist(), pad_to) for k, v in sp.items()},
                        **{f"{k}_y": v.rl.values.astype(np.float32) for k, v in sp.items()},
                        **{f"{k}_s": np.asarray(v.utr.tolist()) for k, v in sp.items()})
    print(f"wrote {work}/data.npz  pad_to={pad_to}")


def load_data(work):
    z = np.load(os.path.join(work, "data.npz"))
    return {k: (z[f"{k}_x"], z[f"{k}_y"]) for k in SPLITS}, int(z["pad_to"])


# ---------------------------------------------------------------------------------------------------------------- stage: teacher
@torch.no_grad()
def teacher_pass(model, x, device, bs=64, amp=True):
    """Returns prediction, 32-d head feature, 1280-d pooled encoder feature. Also records sequences/second in RATE."""
    model.eval().to(device)
    P, Hf, Ef = [], [], []
    t0 = time.time()
    for i in range(0, len(x), bs):
        tk = torch.from_numpy(x[i:i + bs]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            y, hf, ef = model(tk, return_feat=True)
        P.append(y.float().cpu()), Hf.append(hf.float().cpu()), Ef.append(ef.float().cpu())
        if i % (bs * 50) == 0:
            done = i + len(tk)
            print(f"    {done}/{len(x)}  {done / max(time.time() - t0, 1e-9):.0f} seq/s", flush=True)
    RATE.append(len(x) / max(time.time() - t0, 1e-9))
    return torch.cat(P).numpy(), torch.cat(Hf).numpy(), torch.cat(Ef).numpy()


def stage_teacher(work, ckpt_ft=None, cfg="giga", device="auto", bs=64, untrained=True, selftest=False):
    data, pad_to = load_data(work)
    dev = device_of(device)
    model = TeacherMRL(cfg)
    if selftest:
        print("selftest: random teacher weights (no checkpoint)")
    else:
        if not ckpt_ft:
            raise SystemExit("--ckpt-ft is required (the fine-tuned MRL checkpoint)")
        load_strict(model, ckpt_ft)
        print(f"loaded {ckpt_ft} strictly into the re-implementation")
    cache, report = {}, []
    for k in SPLITS:
        x, y = data[k]
        print(f"  teacher on {k} (n={len(x)})", flush=True)
        p, hf, ef = teacher_pass(model, x, dev, bs)
        cache[f"{k}_pred"], cache[f"{k}_head"], cache[f"{k}_enc"] = p, hf.astype(np.float16), ef.astype(np.float16)
        if k in ("iid_test", "ood_test", "repro_val"):
            lo, hi = boot_r2(y, p)
            pub = PUBLISHED.get(k)
            report.append(dict(split=k, n=len(y), r2=r2(y, p), lo=lo, hi=hi, spearman=spearman(y, p),
                               published=pub, delta=None if pub is None else r2(y, p) - pub))
    if untrained and not selftest:
        print("  untrained teacher (control) ...", flush=True)
        rnd = TeacherMRL(cfg)
        for k in SPLITS:
            _, hf, _ = teacher_pass(rnd, data[k][0], dev, bs)
            cache[f"{k}_head_untrained"] = hf.astype(np.float16)
    np.savez_compressed(os.path.join(work, "teacher.npz"), **cache)
    rep = pd.DataFrame(report)
    rep["teacher_seq_per_s"] = round(float(np.mean(RATE)), 1) if RATE else np.nan
    rep["teacher_params"] = sum(p.numel() for p in model.parameters())
    rep.to_csv(os.path.join(work, "teacher_report.csv"), index=False)
    print(rep.to_string(index=False))
    bad = [r for r in report if r["delta"] is not None and abs(r["delta"]) > 0.02]
    if bad:
        print("\nGATE P1 NOT MET: our evaluation of the released checkpoint differs from the published R2 by more than 0.02 on "
              f"{[b['split'] for b in bad]}. Do not run the student grid on these teacher targets until this is understood; "
              "either fix the loader or fine-tune our own teacher and report our number beside the published one.")
    elif not selftest:
        print("\nGATE P1 MET: the released checkpoint reproduces the published R2 within 0.02.")
    return rep


# -------------------------------------------------------------------------------------------------------- stage: baselines
def kmer_features(seqs, k=4):
    """Counts of all 4^k ACGT k-mers, normalised by length, plus the length itself."""
    letters = "ACGT"
    index = {"".join(p): i for i, p in enumerate(__import__("itertools").product(letters, repeat=k))}
    X = np.zeros((len(seqs), len(index) + 1), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = str(s).upper().replace("U", "T")
        for i in range(len(s) - k + 1):
            j = index.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1.0
        X[r, :-1] /= max(len(s) - k + 1, 1)
        X[r, -1] = len(s)
    return X


def ridge(Xtr, ytr, Xte, lam=1.0):
    """Closed-form ridge on standardised features; no sklearn dependency so the package stays small."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return B @ w


def stage_baselines(work, lam=1.0):
    """Two reference points, both on CPU in seconds:
       * kmer_ridge  - a sequence model with no learned representation at all; if the students do not beat it, nothing else matters
       * probe_enc / probe_head - ridge on the FROZEN teacher features already cached, i.e. what the teacher's representation
         carries before any student is trained. This separates 'the teacher's features are informative' from 'the fine-tuned
         teacher is accurate', which the single published R2 cannot."""
    z = np.load(os.path.join(work, "data.npz"), allow_pickle=True)
    tz = np.load(os.path.join(work, "teacher.npz"))
    led = Ledger(os.path.join(work, "runs.csv"))
    ytr = z["train_y"]
    feats = {"kmer_ridge": {k: kmer_features(z[f"{k}_s"]) for k in SPLITS}}
    for name, suffix in (("probe_enc", "enc"), ("probe_head", "head")):
        if f"train_{suffix}" in tz.files:
            feats[name] = {k: tz[f"{k}_{suffix}"].astype(np.float32) for k in SPLITS}
    for name, F in feats.items():
        key = f"-|{name}|0|1.0"
        if key in led.done:
            continue
        t0 = time.time()
        row = dict(key=key, grid="baselines", preset="-", n_params=F["train"].shape[1] + 1, arm=name, seed=0, frac=1.0,
                   val_r2=r2(z["val_y"], ridge(F["train"], ytr, F["val"], lam)), secs=0.0)
        for sp in ("iid_test", "ood_test", "repro_val"):
            p = ridge(F["train"], ytr, F[sp], lam)
            row[f"{sp}_r2"], row[f"{sp}_rho"] = r2(z[f"{sp}_y"], p), spearman(z[f"{sp}_y"], p)
        row["secs"] = round(time.time() - t0, 1)
        led.add(row)
        print(f"  {name:12s} d={F['train'].shape[1]:5d}  repro_val {row['repro_val_r2']:.4f}  "
              f"iid_test {row['iid_test_r2']:.4f}  ood_test {row['ood_test_r2']:.4f}")
    return pd.DataFrame(led.rows)


# ------------------------------------------------------------------------------------------------------------------ stage: train
def fit_one(data, tcache, preset, arm, seed, frac, dev, epochs=30, bs=128, lr=3e-3, alpha=1.0, patience=6, verbose=False):
    floss, ftarget, use_logit = kd.ARMS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr = data["train"]
    rs = np.random.default_rng(seed)
    idx = rs.permutation(len(xtr))[:max(64, int(round(frac * len(xtr))))]
    xtr, ytr = xtr[idx], ytr[idx]
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-8)

    tt = None
    if ftarget:
        base = ftarget.replace("_shuffled", "").replace("_untrained", "")
        key = f"train_{base}_untrained" if ftarget.endswith("_untrained") else f"train_{base}"
        tt = tcache[key][idx].astype(np.float32)
        if ftarget.endswith("_shuffled"):
            tt = tt[rs.permutation(len(tt))]                                    # control: same statistics, no correspondence
    tl = (tcache["train_pred"][idx].astype(np.float32) - mu) / sd if use_logit else None

    model = build(preset).to(dev)
    proj = kd.Projector(model.dim, tt.shape[1]).to(dev) if tt is not None else None
    params = list(model.parameters()) + (list(proj.parameters()) if proj else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * (len(xtr) // bs + 1)))

    xv, yv = data["val"]
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train()
        order = rs.permutation(len(xtr))
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            tk = torch.from_numpy(xtr[j]).to(dev)
            y = torch.from_numpy((ytr[j] - mu) / sd).to(dev)
            out, feat, _ = model(tk)
            loss = F.mse_loss(out, y)
            if tt is not None:
                loss = loss + alpha * kd.feature_loss(proj(feat), torch.from_numpy(tt[j]).to(dev), floss)
            if use_logit:
                loss = loss + alpha * kd.logit_loss(out, torch.from_numpy(tl[j]).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(model, xv, dev, mu, sd)
        s = r2(yv, pv)
        if s > best + 1e-4:
            best, best_state, bad = s, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"      ep{ep} val R2 {s:.4f}")
    model.load_state_dict(best_state)
    return model, mu, sd, best


@torch.no_grad()
def predict(model, x, dev, mu, sd, bs=512):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        o, _, _ = model(torch.from_numpy(x[i:i + bs]).to(dev))
        out.append(o.float().cpu())
    return torch.cat(out).numpy() * sd + mu


GRIDS = {
    # the headline ladder: does distillation beat a tuned supervised twin, and does the margin survive the distribution shift
    "main":     dict(presets=["nano", "tiny", "small", "medium"], arms=["scratch", "kd_feat", "kd_logit", "kd_both"],
                     seeds=[0, 1, 2], fracs=[1.0]),
    # controls and loss variants: a gain that survives these is knowledge transfer, not regularisation
    "controls": dict(presets=["tiny", "small"], arms=["kd_fitnet", "kd_feat_enc", "ctrl_shuffled", "ctrl_untrained"],
                     seeds=[0, 1, 2], fracs=[1.0]),
    # data efficiency: distillation is supposed to help most when labels are scarce
    "frac":     dict(presets=["small"], arms=["scratch", "kd_feat"], seeds=[0, 1, 2], fracs=[0.05, 0.25]),
}


def stage_train(work, grid="main", device="auto", epochs=30, bs=128, lr=3e-3, alpha=1.0):
    data, _ = load_data(work)
    tz = np.load(os.path.join(work, "teacher.npz"))
    tcache = {k: tz[k] for k in tz.files}
    dev = device_of(device)
    led = Ledger(os.path.join(work, "runs.csv"))
    grids = GRIDS if grid == "all" else {grid: GRIDS[grid]}
    todo = [(g, p, a, s, f) for g, c in grids.items() for p in c["presets"] for a in c["arms"]
            for s in c["seeds"] for f in c["fracs"]]
    print(f"{len(todo)} runs in grid '{grid}', {sum(1 for t in todo if f'{t[1]}|{t[2]}|{t[3]}|{t[4]}' in led.done)} already done")
    for n, (g, preset, arm, seed, frac) in enumerate(todo, 1):
        key = f"{preset}|{arm}|{seed}|{frac}"
        if key in led.done:
            continue
        if arm == "ctrl_untrained" and "train_head_untrained" not in tcache:
            print(f"  skip {key}: no untrained-teacher cache (re-run the teacher stage with --untrained)")
            continue
        t0 = time.time()
        model, mu, sd, vbest = fit_one(data, tcache, preset, arm, seed, frac, dev, epochs, bs, lr, alpha)
        row = dict(key=key, grid=g, preset=preset, n_params=model.n_params, arm=arm, seed=seed, frac=frac,
                   val_r2=vbest, secs=round(time.time() - t0, 1))
        for sp in ("iid_test", "ood_test", "repro_val"):
            x, y = data[sp]
            p = predict(model, x, dev, mu, sd)
            row[f"{sp}_r2"], row[f"{sp}_rho"] = r2(y, p), spearman(y, p)
        led.add(row)
        print(f"  [{n}/{len(todo)}] {key:34s} params {model.n_params:>9,d}  IID {row['iid_test_r2']:.4f}  "
              f"OOD {row['ood_test_r2']:.4f}  ({row['secs']:.0f}s)", flush=True)
    return pd.DataFrame(led.rows)


# ----------------------------------------------------------------------------------------------------------------- stage: report
def stage_report(work, out=None):
    """Tables and the pre-registered verdict.

    Which split plays which role, and why (this is the part a reader must check):
      * OOD = `ood_test` (Human7600). The released teacher was fine-tuned on the RANDOM library only and reported Human7600 as
        its test set, so this split is clean for teacher and students alike. Every OOD number below is sound.
      * IID = `repro_val` (Random7600). The released teacher used it for CHECKPOINT SELECTION, so its IID score carries a mild
        upward bias; our students never touch it. The bias inflates the teacher's IID headroom, which shrinks R_IID, which
        pushes P4 toward "OOD transfers better". A NEGATIVE P4 is therefore robust to this; a positive one is not, and is
        reported as suggestive only.
      * `iid_test` is clean for students but the released teacher was fine-tuned on those rows, so no teacher value is shown
        for it; it appears as a student-only check that the Random7600 figures are not selection-inflated.
    """
    runs = pd.read_csv(os.path.join(work, "runs.csv"))
    trep = pd.read_csv(os.path.join(work, "teacher_report.csv"))
    IID, OOD = "repro_val", "ood_test"
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("TEACHER  RiNALMo-giga fine-tuned on MRL, released checkpoint, scored through our loader")
    for _, r in trep.iterrows():
        pub = "" if pd.isna(r.published) else f"   published {r.published:.3f}   delta {r.delta:+.3f}"
        flag = "  <- teacher trained on these rows; not usable as a teacher reference" if r.split == "iid_test" else ""
        say(f"  {r.split:10s} n={int(r.n):5d}  R2 {r.r2:.4f} [{r.lo:.4f},{r.hi:.4f}]  rho {r.spearman:.4f}{pub}{flag}")
    T = {r.split: r.r2 for _, r in trep.iterrows()}
    say(f"  roles: IID = {IID} (teacher selected on it), OOD = {OOD} (clean for both)")

    refs = runs[runs.grid == "baselines"]
    if len(refs):
        say("\nREFERENCE POINTS  no student trained")
        for _, r in refs.iterrows():
            say(f"  {r.arm:12s} d={int(r.n_params):5d}  IID R2 {r[IID + '_r2']:.4f}  OOD R2 {r[OOD + '_r2']:.4f}  "
                f"(clean iid_test {r.iid_test_r2:.4f})")
        say("  kmer_ridge = no learned representation at all; probe_* = ridge on the frozen teacher's cached features")
    runs = runs[runs.grid != "baselines"]
    if not len(runs):
        say("\nno student runs yet")
        open(out or os.path.join(work, "pilot_report.txt"), "w").write("\n".join(lines) + "\n")
        return

    agg = runs.groupby(["preset", "arm", "frac"]).agg(
        n_params=("n_params", "first"), seeds=("seed", "count"),
        iid=(IID + "_r2", "mean"), iid_sd=(IID + "_r2", "std"),
        ood=(OOD + "_r2", "mean"), ood_sd=(OOD + "_r2", "std"),
        clean=("iid_test_r2", "mean")).reset_index()
    # The yardstick is the noise of the BASELINE arm, so a collapsing KD arm cannot inflate the bar. It must also come from
    # the SAME train fraction as the comparisons it gates (every gate below is on frac 1.0): a cell trained on 5% of the
    # labels is far noisier, and pooling it in would silently raise the bar for the full-data comparison.
    base_cells = agg[(agg.arm == "scratch") & (agg.frac == 1.0)]
    spread = float(np.nanmax([base_cells.iid_sd.max(), base_cells.ood_sd.max()]))
    allmax = float(np.nanmax([agg.iid_sd.max(), agg.ood_sd.max()]))
    say(f"\nseed spread, max sd over the scratch cells at frac 1.0: {spread:.4f}   [max over every cell, any frac: {allmax:.4f}]")
    say("a margin counts as real only above the first number")

    order = [p for p in PRESETS if p in set(agg.preset)]
    for frac in sorted(agg.frac.unique()):
        arms = sorted(set(agg[agg.frac == frac].arm))
        say(f"\nSTUDENTS  train fraction {frac:g}   cell = IID R2 / OOD R2 +- max seed sd")
        say(f"  {'size':8s} {'params':>9s}  " + "  ".join(f"{a:>22s}" for a in arms))
        for p in order:
            sub = agg[(agg.preset == p) & (agg.frac == frac)]
            if not len(sub):
                continue
            cells = []
            for a in arms:
                r = sub[sub.arm == a]
                if not len(r):
                    cells.append(" " * 22)
                    continue
                sds = [v for v in (r.iid_sd.iloc[0], r.ood_sd.iloc[0]) if np.isfinite(v)]
                tag = f"+-{max(sds):.3f}" if sds else "(1 seed)"
                cells.append(f"{r.iid.iloc[0]:.3f}/{r.ood.iloc[0]:.3f} {tag}".rjust(22))
            say(f"  {p:8s} {int(sub.n_params.iloc[0]):>9,d}  " + "  ".join(cells))

    say("\nSTUDENT-ONLY CLEAN IN-DISTRIBUTION SET (iid_test; never used for selection by anyone but the released teacher trained on it)")
    for p in order:
        sub = agg[(agg.preset == p) & (agg.frac == 1.0)]
        say(f"  {p:8s} " + "  ".join(f"{r.arm}={r.clean:.3f}" for _, r in sub.iterrows()))

    say("\nRETENTION  R = (student - scratch) / (teacher - scratch), frac 1.0")
    say(f"  {'size':8s} {'arm':15s} {'R_IID':>8s} {'R_OOD':>8s} {'R_OOD-R_IID':>12s}")
    ret = []
    for p in order:
        b = agg[(agg.preset == p) & (agg.arm == "scratch") & (agg.frac == 1.0)]
        if not len(b):
            continue
        for a in sorted(set(agg.arm) - {"scratch"}):
            r = agg[(agg.preset == p) & (agg.arm == a) & (agg.frac == 1.0)]
            if not len(r):
                continue
            gi, go = T[IID] - b.iid.iloc[0], T[OOD] - b.ood.iloc[0]
            ri = (r.iid.iloc[0] - b.iid.iloc[0]) / gi if gi > 0 else np.nan
            ro = (r.ood.iloc[0] - b.ood.iloc[0]) / go if go > 0 else np.nan
            ret.append(dict(preset=p, arm=a, R_iid=ri, R_ood=ro))
            say(f"  {p:8s} {a:15s} {ri:8.3f} {ro:8.3f} {ro - ri:12.3f}")

    if "teacher_seq_per_s" in trep.columns and trep.teacher_seq_per_s.notna().any():
        dev = device_of("auto")
        tp, tr = int(trep.teacher_params.iloc[0]), float(trep.teacher_seq_per_s.iloc[0])
        say(f"\nINFERENCE COST, measured on {dev.type} in this run   (teacher rate is from the cached pass, batch as run)")
        say(f"  {'model':10s} {'params':>12s} {'seq/s':>10s} {'x smaller':>10s} {'x faster':>9s}")
        say(f"  {'teacher':10s} {tp:>12,d} {tr:>10.0f} {1.0:>10.1f} {1.0:>9.1f}")
        probe = torch.randint(5, 9, (128, 102), device=dev)
        for pr in order:
            m = build(pr).to(dev).eval()
            with torch.no_grad():
                m(probe)
                t0 = time.time()
                for _ in range(5):
                    m(probe)
                rate = 5 * 128 / max(time.time() - t0, 1e-9)
            say(f"  {pr:10s} {m.n_params:>12,d} {rate:>10.0f} {tp / m.n_params:>10.0f} {rate / tr:>9.0f}")

    say("\nPRE-REGISTERED GATES  (PILOT_PROTOCOL.md)")
    p1 = bool(trep.delta.dropna().abs().max() <= 0.02) if trep.delta.notna().any() else None
    say(f"  P1 reproduction, |delta| <= 0.02 on both published splits: {'MET' if p1 else 'NOT MET' if p1 is not None else 'n/a'}")
    sc = agg[(agg.arm == "scratch") & (agg.frac == 1.0)]
    head = bool(len(sc) and ((T[IID] - sc.iid.max() > 2 * spread) or (T[OOD] - sc.ood.max() > 2 * spread)))
    say(f"  P2 headroom over the best tuned scratch student, > 2x spread on IID or OOD: {'MET' if head else 'NOT MET'}"
        f"   (IID gap {T[IID] - sc.iid.max():+.4f}, OOD gap {T[OOD] - sc.ood.max():+.4f})")
    kdw = [f"{p}/{a}" for p in order for a in ("kd_feat", "kd_feat_enc", "kd_fitnet", "kd_logit", "kd_both")
           for b in [agg[(agg.preset == p) & (agg.arm == "scratch") & (agg.frac == 1.0)]]
           for r in [agg[(agg.preset == p) & (agg.arm == a) & (agg.frac == 1.0)]]
           if len(b) and len(r) and (r.iid.iloc[0] - b.iid.iloc[0]) > spread]
    say(f"  P3 distillation beats scratch on IID by > spread: {'MET' if kdw else 'NOT MET'}   {kdw}")
    if ret:
        d = pd.DataFrame(ret)
        d = d[d.arm.str.startswith("kd_")]
        m = float((d.R_ood - d.R_iid).mean()) if len(d) else float("nan")
        if not np.isfinite(m):
            say("  P4 mean R_OOD - R_IID: NOT COMPUTABLE -- retention needs the teacher to beat the tuned scratch student "
                "on that split, and it does not (see P2). With no headroom there is nothing to retain, and that absence is "
                "itself the pilot's finding for this task.")
        else:
            say(f"  P4 mean R_OOD - R_IID over the distillation arms: {m:+.3f}  -> "
                + ("the out-of-distribution advantage transfers at least as well as the in-distribution one "
                   "(SUGGESTIVE only: the teacher's IID score is selection-inflated, which biases this upward)"
                   if m >= 0 else
                   "distillation buys LESS out of distribution than in it (robust to the teacher's IID selection bias, "
                   "which pushes this quantity the other way)"))
    ctrls = [f"{p}/{a}" for p in order for a in ("ctrl_shuffled", "ctrl_untrained")
             for b in [agg[(agg.preset == p) & (agg.arm == "scratch") & (agg.frac == 1.0)]]
             for r in [agg[(agg.preset == p) & (agg.arm == a) & (agg.frac == 1.0)]]
             if len(b) and len(r) and (r.iid.iloc[0] - b.iid.iloc[0]) > spread]
    say(f"  P5 controls do NOT beat scratch: {'MET' if not ctrls else 'NOT MET -- a gain the controls reproduce is regularisation'}   {ctrls}")
    say("\n  GO to the full study iff P1, P3 and P5 are met and P4 is not strongly negative.")

    path = out or os.path.join(work, "pilot_report.txt")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


# ------------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "data", "teacher", "baselines", "train", "report", "dumpkeys"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--ckpt-ft")
    ap.add_argument("--cfg", default="giga")
    ap.add_argument("--grid", default="main", choices=list(GRIDS) + ["all"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-untrained", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.stage == "dumpkeys":
        dump_keys(a.ckpt_ft)
    elif a.stage == "selftest":
        w = a.work if a.work != "work" else "selftest_work"
        print("== selftest: synthetic data, random tiny teacher, 2 seeds, few epochs ==")
        stage_data(w, synthetic=True, top_k=20)
        stage_teacher(w, cfg="selftest", device=a.device, bs=32, untrained=False, selftest=True)
        tz = np.load(os.path.join(w, "teacher.npz"))
        cache = {k: tz[k] for k in tz.files}
        for k in list(cache):                                                    # give the control arms a cache to read
            if k.endswith("_head"):
                cache[k + "_untrained"] = cache[k][np.random.default_rng(0).permutation(len(cache[k]))]
        np.savez_compressed(os.path.join(w, "teacher.npz"), **cache)
        GRIDS["main"] = dict(presets=["nano", "tiny"], arms=["scratch", "kd_feat", "kd_logit", "ctrl_shuffled", "ctrl_untrained"],
                             seeds=[0, 1], fracs=[1.0])
        stage_baselines(w)
        stage_train(w, "main", a.device, epochs=4, bs=128)
        stage_report(w)
        print("\nselftest finished: every stage ran, the ledger resumes, the report applies the gates.")
    elif a.stage == "data":
        stage_data(a.work, synthetic=a.synthetic)
    elif a.stage == "teacher":
        stage_teacher(a.work, a.ckpt_ft, a.cfg, a.device, a.bs, not a.no_untrained)
    elif a.stage == "baselines":
        stage_baselines(a.work)
    elif a.stage == "train":
        stage_train(a.work, a.grid, a.device, a.epochs, a.bs, a.lr, a.alpha)
    elif a.stage == "report":
        stage_report(a.work)


if __name__ == "__main__":
    main()
