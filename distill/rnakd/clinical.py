"""Clinical benchmark driver: zero-shot ClinVar variant scoring, a teacher size ladder, and a distilled compact scorer.

  python -m rnakd.clinical selftest                                   # whole path, synthetic, no network, no GPU
  python -m rnakd.clinical panel    --work W
  python -m rnakd.clinical score    --work W --ckpt-lm <giga.pt> --cfg giga        # GATE C1 + C2, repeat per size
  python -m rnakd.clinical transfer --work W --ckpt-lm <giga.pt> --cfg giga        # free transfer set + teacher cache
  python -m rnakd.clinical distil   --work W                          # students: mlm / kd / mlm_kd, seeds
  python -m rnakd.clinical report   --work W

Every stage appends to a ledger and skips finished work, as in `pilot.py`. `distil` never loads the teacher.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import clinvar, lm
from .teacher import LM_CONFIGS, MASK, PAD, RiNALMo, load_strict

BASELINES = {
    # no-model scores, oriented so higher = more damaging. These exist because a language model that cannot beat
    # "transversions are worse than transitions" is not doing biology.
    "transversion": lambda d: np.array([0.0 if (r + a) in ("AG", "GA", "CT", "TC") else 1.0
                                        for r, a in zip(d.ref, d.alt)], dtype=np.float32),
    "gc_loss": lambda d: np.array([(r in "GC") - (a in "GC") for r, a in zip(d.ref, d.alt)], dtype=np.float32),
}


# ------------------------------------------------------------------------------------------------------- evaluation
def auroc(y, s):
    """Rank AUROC with average ranks for ties. Ties matter here: a no-model baseline takes two values."""
    y = np.asarray(y)
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return np.nan
    s = np.asarray(s, dtype=float)
    o = np.argsort(s, kind="mergesort")
    sv = s[o]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[o] = ranks
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * (len(y) - n1))


def _by_gene(d, col):
    """[(gene, labels, scores, weight), ...]; weight = min(pathogenic, benign), the usable pairs in that gene."""
    out = []
    for g, sub in d.groupby("gene", sort=True):
        y = sub.label.values.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        out.append((g, y, sub[col].values.astype(float), min(int(y.sum()), int(len(y) - y.sum()))))
    return out


def within_gene(d, col):
    """Mean AUROC inside each gene, weighted by min(pathogenic, benign). Pooled AUROC is not used: gene identity
    alone separates the classes in ClinVar, so a pooled number measures which genes are famous."""
    parts = _by_gene(d, col)
    num = den = 0.0
    per = {}
    for g, y, s, w in parts:
        a = auroc(y, s)
        if np.isnan(a):
            continue
        per[g] = a
        num, den = num + w * a, den + w
    return (num / den if den else np.nan), per


def boot_within_gene(d, col, n=2000, seed=0):
    """Hierarchical bootstrap: resample genes, then variants inside each resampled gene. Pure numpy, because the
    report runs this for every model and a pandas-concat version dominated the runtime."""
    parts = _by_gene(d, col)
    if not parts:
        return (np.nan, np.nan)
    rs = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        num = den = 0.0
        for gi in rs.integers(0, len(parts), len(parts)):
            _, y, s, _ = parts[gi]
            idx = rs.integers(0, len(y), len(y))
            yy, ss = y[idx], s[idx]
            a = auroc(yy, ss)
            if np.isnan(a):
                continue
            w = min(int(yy.sum()), int(len(yy) - yy.sum()))
            num, den = num + w * a, den + w
        if den:
            vals.append(num / den)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


class Ledger:
    def __init__(self, path):
        self.path = path
        self.rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
        self.done = {r["key"] for r in self.rows}

    def add(self, row):
        self.rows.append(row)
        self.done.add(row["key"])
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def device_of(name="auto"):
    return torch.device(name) if name != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------------------ stages
def stage_panel(work, synthetic=False, release=clinvar.GENCODE_RELEASE):
    os.makedirs(work, exist_ok=True)
    if synthetic:
        panel = clinvar.synthetic()
        print(f"  synthetic panel: {len(panel)} variants, {panel.gene.nunique()} genes")
    else:
        panel, summary = clinvar.build_panel(work, release=release)
        summary.to_csv(os.path.join(work, "panel_summary.csv"), index=False)
    panel.to_csv(os.path.join(work, "panel.csv.gz"), index=False)
    for name, fn in BASELINES.items():
        v, _ = within_gene(panel.assign(**{name: fn(panel)}), name)
        print(f"  baseline {name:14s} within-gene AUROC {v:.3f}")
    return panel


def load_panel(work):
    return pd.read_csv(os.path.join(work, "panel.csv.gz"))


def stage_score(work, ckpt=None, cfg="giga", device="auto", bs=16, width=512, selftest=False):
    """Zero-shot masked log-odds for one model size. GATE C1 is the recovery check; C2 is the published-free
    internal reference (RiNALMo-micro scored 0.72 within-gene AUROC on this panel in `lncrna_fm`)."""
    panel = load_panel(work)
    dev = device_of(device)
    model = RiNALMo(**LM_CONFIGS[cfg])
    if selftest:
        print(f"  selftest: random {cfg} weights")
    else:
        load_strict(model, ckpt)
        print(f"  loaded {ckpt} strictly as a bare RiNALMo ({sum(p.numel() for p in model.parameters()):,d} params)")
    acc = lm.recovery_accuracy(model, panel.drop_duplicates('gene').seq.tolist(), dev, width=width, bs=bs)
    print(f"  GATE C1 masked-base recovery accuracy: {acc:.3f} (chance 0.25)")
    if acc < 0.30 and not selftest:
        raise SystemExit("GATE C1 NOT MET: the model cannot recover a masked nucleotide above chance. The mask head "
                         "is wired wrongly or the checkpoint is not what we think. Fix before trusting any score.")
    t0 = time.time()
    s = lm.masked_log_odds(model, panel, dev, width=width, bs=bs)
    rate = len(panel) / max(time.time() - t0, 1e-9)
    led = Ledger(os.path.join(work, "scores.csv"))
    col = f"lm_{cfg}"
    panel[col] = s
    panel.to_csv(os.path.join(work, "panel.csv.gz"), index=False)
    v, per = within_gene(panel, col)
    lo, hi = boot_within_gene(panel, col)
    led.add(dict(key=col, model=col, params=sum(p.numel() for p in model.parameters()), auroc=v, lo=lo, hi=hi,
                 recovery=acc, var_per_s=round(rate, 1)))
    print(f"  {col}: within-gene AUROC {v:.3f} [{lo:.3f},{hi:.3f}]   {rate:.0f} variants/s")
    print("  per gene: " + "  ".join(f"{g}={a:.2f}" for g, a in sorted(per.items())))
    return v


def stage_transfer(work, ckpt=None, cfg="giga", device="auto", bs=32, width=256, max_n=60000,
                   mask_frac=0.15, selftest=False, release=clinvar.GENCODE_RELEASE):
    """Build the free transfer set and cache the teacher's distribution at a FIXED set of masked positions.

    Fixing the mask pattern makes every student arm see identical inputs and identical positions, so the only thing
    that differs between `mlm`, `kd` and `mlm_kd` is the target. It costs masking diversity; that is the trade."""
    panel = load_panel(work)
    if selftest:
        rs = np.random.default_rng(0)
        seqs = ["".join(rs.choice(list("ACGT"), int(rs.integers(64, width)))) for _ in range(400)]
    else:
        fa = os.path.join(work, f"gencode.v{release}.tx.fa.gz")
        seqs = clinvar.transfer_sequences(fa, set(panel.gene), width=width, max_n=max_n)
    dev = device_of(device)
    model = RiNALMo(**LM_CONFIGS[cfg])
    if not selftest:
        load_strict(model, ckpt)
    model.eval().to(dev)

    rs = np.random.default_rng(0)
    toks = lm.tokenise(seqs, pad_to=width + 2)
    k = max(1, int(mask_frac * width))
    pos = np.zeros((len(seqs), k), dtype=np.int64)
    for i, s in enumerate(seqs):                                                # positions inside the real sequence
        pos[i] = rs.choice(np.arange(1, len(s) + 1), k, replace=len(s) < k)
    lp = np.zeros((len(seqs), k, 22), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            t = torch.from_numpy(toks[i:i + bs]).to(dev)
            p = torch.from_numpy(pos[i:i + bs]).to(dev)
            t.scatter_(1, p, MASK)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                g = model.logits(t)
            sel = torch.gather(g.float(), 1, p[..., None].expand(-1, -1, 22))
            lp[i:i + bs] = F.log_softmax(sel, -1).cpu().numpy().astype(np.float16)
            if i % (bs * 50) == 0:
                print(f"    {i + len(t)}/{len(seqs)}  {(i + len(t)) / max(time.time() - t0, 1e-9):.0f} seq/s", flush=True)
    np.savez_compressed(os.path.join(work, "transfer.npz"), toks=toks, pos=pos, lp=lp,
                        truth=np.take_along_axis(toks, pos, 1))
    print(f"  cached {len(seqs)} windows x {k} masked positions")


def fit_student(work, preset, arm, seed, dev, epochs=3, bs=32, lr=3e-4, verbose=False):
    z = np.load(os.path.join(work, "transfer.npz"))
    toks, pos, lp, truth = z["toks"], z["pos"], z["lp"], z["truth"]
    n_val = max(1, len(toks) // 20)
    torch.manual_seed(seed)
    model = lm.Student(preset).to(dev)          # fixed max_len: the student is scored on panel windows,
                                                # which are longer than the transfer windows it trains on
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_tr = len(toks) - n_val
    steps = max(1, epochs * math.ceil(n_tr / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    rs = np.random.default_rng(seed)
    best, state = 1e9, None
    for ep in range(epochs):
        model.train()
        order = rs.permutation(n_tr)
        for i in range(0, n_tr, bs):
            j = order[i:i + bs]
            t = torch.from_numpy(toks[j]).to(dev)
            p = torch.from_numpy(pos[j]).to(dev)
            t = t.clone().scatter_(1, p, MASK)
            g = model.logits(t)
            sel = torch.gather(g, 1, p[..., None].expand(-1, -1, 22)).reshape(-1, 22)
            loss = lm.kd_loss(sel, torch.from_numpy(lp[j].astype(np.float32)).to(dev).reshape(-1, 22),
                              torch.from_numpy(truth[j]).to(dev).reshape(-1), arm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(toks[-n_val:]).to(dev)
            p = torch.from_numpy(pos[-n_val:]).to(dev)
            g = model.logits(t.clone().scatter_(1, p, MASK))
            sel = torch.gather(g, 1, p[..., None].expand(-1, -1, 22)).reshape(-1, 22)
            v = float(F.cross_entropy(sel, torch.from_numpy(truth[-n_val:]).to(dev).reshape(-1)))
        if verbose:
            print(f"      ep{ep} val CE {v:.4f}")
        if v < best:
            best, state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def stage_distil(work, presets=("s1", "s5"), arms=("mlm", "kd", "mlm_kd"), seeds=(0, 1, 2),
                 device="auto", epochs=3, bs=32, lr=3e-4, width=512):
    panel = load_panel(work)
    dev = device_of(device)
    led = Ledger(os.path.join(work, "scores.csv"))
    todo = [(p, a, s) for p in presets for a in arms for s in seeds]
    print(f"{len(todo)} student runs, {sum(1 for t in todo if f'{t[0]}|{t[1]}|{t[2]}' in led.done)} already done")
    for n, (preset, arm, seed) in enumerate(todo, 1):
        key = f"{preset}|{arm}|{seed}"
        if key in led.done:
            continue
        t0 = time.time()
        model, ce = fit_student(work, preset, arm, seed, dev, epochs, bs, lr)
        acc = lm.recovery_accuracy(model, panel.drop_duplicates("gene").seq.tolist(), dev, width=width, bs=16)
        s = lm.masked_log_odds(model, panel, dev, width=width, bs=32, progress=False)
        col = f"stu_{key}"
        panel[col] = s
        v, _ = within_gene(panel, col)
        lo, hi = boot_within_gene(panel, col, n=500)
        led.add(dict(key=key, model=col, params=model.n_params, auroc=v, lo=lo, hi=hi, recovery=acc,
                     val_ce=ce, preset=preset, arm=arm, seed=seed, secs=round(time.time() - t0, 1)))
        print(f"  [{n}/{len(todo)}] {key:16s} params {model.n_params:>9,d}  recovery {acc:.3f}  "
              f"AUROC {v:.3f} [{lo:.3f},{hi:.3f}]  ({time.time() - t0:.0f}s)", flush=True)
    panel.to_csv(os.path.join(work, "panel.csv.gz"), index=False)


def stage_report(work, out=None):
    panel = load_panel(work)
    sc = pd.read_csv(os.path.join(work, "scores.csv"))
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"CLINICAL PANEL  {len(panel)} ClinVar SNVs, {int(panel.label.sum())} pathogenic, {panel.gene.nunique()} genes")
    g = panel.groupby(["gene", "gtype"]).label.agg(path="sum", n="count").reset_index()
    for _, r in g.iterrows():
        say(f"  {r.gene:12s} {r.gtype:10s} {int(r.path):3d} pathogenic / {int(r.n - r.path):3d} benign")
    say("\n  metric: within-gene AUROC, weighted by min(pathogenic, benign); 95% CI by hierarchical bootstrap")

    say("\nNO-MODEL BASELINES")
    for name, fn in BASELINES.items():
        p2 = panel.assign(**{name: fn(panel)})
        v, _ = within_gene(p2, name)
        lo, hi = boot_within_gene(p2, name, n=500)
        say(f"  {name:16s} {v:.3f} [{lo:.3f},{hi:.3f}]")

    teach = sc[sc.key.str.startswith("lm_")]
    if len(teach):
        say("\nTEACHER SIZE LADDER  (zero-shot masked log-odds, no fine-tuning, no labels)")
        say(f"  {'model':12s} {'params':>13s} {'recovery':>9s} {'AUROC':>7s}  {'95% CI':>15s} {'var/s':>8s}")
        for _, r in teach.sort_values("params").iterrows():
            say(f"  {r.model:12s} {int(r.params):>13,d} {r.recovery:>9.3f} {r.auroc:>7.3f}  "
                f"[{r.lo:.3f},{r.hi:.3f}] {r.var_per_s:>8.1f}")

    stu = sc[~sc.key.str.startswith("lm_")]
    if len(stu):
        agg = stu.groupby(["preset", "arm"]).agg(params=("params", "first"), auroc=("auroc", "mean"),
                                                 sd=("auroc", "std"), rec=("recovery", "mean"),
                                                 seeds=("seed", "count")).reset_index()
        spread = float(np.nanmax(agg.sd)) if agg.sd.notna().any() else 0.0
        say(f"\nDISTILLED STUDENTS  mean over seeds; seed spread {spread:.3f}")
        say(f"  {'size':6s} {'params':>11s} " + "  ".join(f"{a:>16s}" for a in sorted(agg.arm.unique())))
        for p in sorted(agg.preset.unique(), key=lambda x: agg[agg.preset == x].params.iloc[0]):
            sub = agg[agg.preset == p]
            cells = []
            for a in sorted(agg.arm.unique()):
                r = sub[sub.arm == a]
                cells.append(" " * 16 if not len(r) else
                             f"{r.auroc.iloc[0]:.3f}+-{0 if pd.isna(r.sd.iloc[0]) else r.sd.iloc[0]:.3f}".rjust(16))
            say(f"  {p:6s} {int(sub.params.iloc[0]):>11,d} " + "  ".join(cells))
        say("  mlm = the student trained on the same sequences with no teacher; kd = teacher distribution only")

        say("\nGATES")
        best_t = teach.auroc.max() if len(teach) else np.nan
        c1 = bool((teach.recovery > 0.30).all()) if len(teach) else None
        say(f"  C1 every teacher recovers masked bases above chance: {'MET' if c1 else 'NOT MET' if c1 is not None else 'n/a'}")
        if len(teach) > 1:
            t = teach.sort_values("params")
            say(f"  C2 size ladder: {t.iloc[0].model} {t.iloc[0].auroc:.3f} -> {t.iloc[-1].model} {t.iloc[-1].auroc:.3f} "
                f"({t.iloc[-1].auroc - t.iloc[0].auroc:+.3f} for {t.iloc[-1].params / t.iloc[0].params:.0f}x the parameters)")
        for p in sorted(agg.preset.unique()):
            m = agg[(agg.preset == p) & (agg.arm == "mlm")]
            k = agg[(agg.preset == p) & (agg.arm == "kd")]
            if len(m) and len(k):
                d = k.auroc.iloc[0] - m.auroc.iloc[0]
                say(f"  C3 {p}: distillation minus own-LM training {d:+.3f} "
                    f"({'above' if abs(d) > spread else 'inside'} the seed spread)")
        if len(teach) and len(agg):
            b = agg.auroc.max()
            ratio = teach.params.max() / max(agg.params.min(), 1)
            say(f"  C4 best student {b:.3f} vs best teacher {best_t:.3f}: retains {b / best_t * 100:.0f}% of its "
                f"within-gene AUROC at {ratio:,.0f}x fewer parameters"
                + ("" if ratio >= 1 else "   (teacher is SMALLER here -- expected only in the selftest)"))

    path = out or os.path.join(work, "clinical_report.txt")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "panel", "score", "transfer", "distil", "report"])
    ap.add_argument("--work", default="clinwork")
    ap.add_argument("--ckpt-lm")
    ap.add_argument("--cfg", default="giga")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=60000)
    ap.add_argument("--release", default=clinvar.GENCODE_RELEASE)
    ap.add_argument("--presets", default="s1,s5")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    if a.stage == "selftest":
        w = a.work if a.work != "clinwork" else "clin_selftest"
        print("== selftest: synthetic panel, random tiny model, no network ==")
        stage_panel(w, synthetic=True)
        stage_score(w, cfg="selftest", device=a.device, bs=8, width=256, selftest=True)
        stage_transfer(w, cfg="selftest", device=a.device, bs=16, width=128, selftest=True)
        stage_distil(w, presets=("s1",), seeds=(0, 1), device=a.device, epochs=1, bs=16, width=256)
        stage_report(w)
        print("\nselftest finished: every stage ran and the report applied the gates.")
    elif a.stage == "panel":
        stage_panel(a.work, release=a.release)
    elif a.stage == "score":
        stage_score(a.work, a.ckpt_lm, a.cfg, a.device, a.bs, a.width)
    elif a.stage == "transfer":
        stage_transfer(a.work, a.ckpt_lm, a.cfg, a.device, max(a.bs, 32), max_n=a.max_n, release=a.release)
    elif a.stage == "distil":
        stage_distil(a.work, tuple(a.presets.split(",")), device=a.device, epochs=a.epochs,
                     seeds=tuple(int(s) for s in a.seeds.split(",")), width=a.width)
    elif a.stage == "report":
        stage_report(a.work)


if __name__ == "__main__":
    main()
