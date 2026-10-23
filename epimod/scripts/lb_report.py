"""LongevityBench DNAm report from per-item predictions: metrics with 95% bootstrap CIs (items resampled) and paired
differences against a reference model.

    python scripts/lb_report.py out.md  episet=preds/mt_dense.npz  ridge=preds/baselines_preds.npz ...   [--ref name]
Model files may contain any of: LB-0010/pred (+ /mu,/sigma), LB-0002/head or LB-0002/{mu,sigma}, LB-0006/logit[_C*],
LB-0190/pred[/ridge], LB-0195/pred[/ridge]. Keys are looked up per task (see KEYS)."""
import sys
import numpy as np

from epimod.lb.tasks import balanced_accuracy, decade_label_to_id, decade_posterior, load_parsed, mae

PAPER = {"LB-0010": "12.96 (ridge 15.90)", "LB-0002": "0.203 (NB 0.13)", "LB-0006": "0.868 (NB 0.61)", "LB-0190": "5.82 (ridge 4.30)", "LB-0195": "13.17 (EN 8.93)"}
KEYS = {"LB-0010": ("LB-0010/pred", "LB-0010/ridge"), "LB-0190": ("LB-0190/pred", "LB-0190/ridge"), "LB-0195": ("LB-0195/pred", "LB-0195/ridge"),
        "LB-0006": ("LB-0006/logit",)}
B = 2000


def load(path):
    return dict(np.load(path))


def get(d, task):
    if task == "LB-0002":
        if "LB-0002/head" in d:
            return d["LB-0002/head"].argmax(1)
        if "LB-0002/mu" in d:
            return decade_posterior(d["LB-0002/mu"], d["LB-0002/sigma"]).argmax(1)
        return None
    for k in KEYS[task]:
        if k in d:
            return d[k]
    return None


def metric(task, pred, y, idx):
    if task == "LB-0006":
        return float(((pred[idx] > 0) == y[idx]).mean())
    if task == "LB-0002":
        return balanced_accuracy(pred[idx], y[idx])["balanced_acc_11"]
    return mae(pred[idx], y[idx])


def main():
    out_path, rest = sys.argv[1], sys.argv[2:]
    ref = None
    if "--ref" in rest:
        i = rest.index("--ref"); ref = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
    models = {}
    for a in rest:
        name, path = a.split("=", 1)
        models[name] = load(path)
    parsed = load_parsed(); rng = np.random.RandomState(0)
    truth = {"LB-0010": np.array(parsed["LB-0010"].age), "LB-0190": np.array(parsed["LB-0190"].age), "LB-0195": np.array(parsed["LB-0195"].age),
             "LB-0006": np.array(parsed["LB-0006"].label) == "A", "LB-0002": np.array([decade_label_to_id(l) for l in parsed["LB-0002"].label])}
    lines = ["| task | metric | paper best LLM | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
    diffs = []
    for task in ("LB-0010", "LB-0002", "LB-0006", "LB-0190", "LB-0195"):
        y = truth[task]; n = len(y); idxs = rng.randint(0, n, size=(B, n)); full = np.arange(n); even = np.arange(n) % 2 == 0
        preds = {m: get(d, task) for m, d in models.items()}
        cells, boots = [], {}
        for m, p in preds.items():
            if p is None:
                cells.append("–"); continue
            pt = metric(task, p, y, full); bs = np.array([metric(task, p, y, ix) for ix in idxs]); boots[m] = bs
            cells.append(f"{pt:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] (even {metric(task, p, y, full[even]):.3f})")
        name = {"LB-0010": "age MAE ↓", "LB-0002": "decade bal-acc (11 cl.) ↑", "LB-0006": "pair concordance ↑", "LB-0190": "age MAE ↓", "LB-0195": "age MAE ↓"}[task]
        lines.append(f"| {task} | {name} | {PAPER[task]} | " + " | ".join(cells) + " |")
        if ref and ref in boots:
            for m, bs in boots.items():
                if m != ref:
                    d = bs - boots[ref]; diffs.append(f"| {task} | {m} − {ref} | {np.mean(d):+.3f} | [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
    text = "\n".join(lines)
    if diffs:
        text += "\n\nPaired bootstrap difference (same resampled items; metric as above):\n\n| task | comparison | mean diff | 95% CI |\n|---|---|---|---|\n" + "\n".join(diffs)
    open(out_path, "w").write(text + "\n"); print(text)


main()
