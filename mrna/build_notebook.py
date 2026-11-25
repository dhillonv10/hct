#!/usr/bin/env python3
"""Generate mrna/ensemble_colab.ipynb. Edit the CELLS list here, never the .ipynb by hand."""
import json, sys

CELLS = []
def md(s):   CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ───────────────────────────────── phase 0 ─────────────────────────────────
md(r'''
# Can an ensemble beat the best single model on mRNABench?

**Protocol is pre-registered in `mrna/PROTOCOL.md`. Read it before running.** Short version:

mRNABench scores every RNA foundation model **alone**, by linear probe, on 79 tasks. Nobody has published
the combination. Nine public backbones are trained on different objectives, and explicit biophysics
(folding energy, codon usage, UTR features) is in none of their inputs as a feature. If their errors are
decorrelated, a stacked combination wins. If they are not, that is a real finding about the field.

**Stated odds before running: ~50–60% of beating the best single model on a majority of tasks, ~15% on
every task.** This notebook is built to find that out in one run, not to talk us into it.

### Structure

| phase | what | time | gate |
|---|---|---|---|
| 1 | install, discover the real API, inspect one dataset | ~15 min | prints a STATUS block — **paste it back before running phase 2** |
| 2 | correctness gate G1: our probe must match `LinearProbeBuilder` to 0.01 | ~20 min | **stop if it fails** |
| 3 | embed every (model × dataset), cached to Drive | hours | resumable |
| 4 | biophysical feature block, cached | ~3 min / 10k seqs | resumable |
| 5 | probe all configurations on identical splits/seeds | ~30 min | — |
| 6 | win-count table, paired bootstrap, decorrelation map | ~5 min | K2/K3/K4 |

Phases 3–5 are **resumable**: every artifact is cached to Drive keyed by content, so a Colab disconnect
costs only the in-flight item. Re-run the cell and it picks up.

### Kill criteria, decided in advance
* **K1** gate G1 fails after one debugging pass → stop, report nothing.
* **K2** win rate under 60% → the across-the-board claim is dead; report the win rate as the result.
* **K3** pooled paired interval includes 0 → null; "nine backbones carry one signal" is the finding.
* **K4** `bio` alone within 0.01 of the best model on a task → that task's LMs aren't earning their parameters.
''')

md("## Phase 1 — environment, install, API discovery")

code(r'''
#@title 1.1 GPU + runtime check
import subprocess, sys, os, platform
print("python  :", sys.version.split()[0])
print("platform:", platform.platform())
try:
    print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "no GPU")
except FileNotFoundError:
    print("nvidia-smi not found -> NO GPU. Embedding will be unusably slow; switch runtime to a GPU.")
try:
    import torch; print("torch   :", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
except ImportError:
    print("torch   : not installed yet")
''')

code(r'''
#@title 1.2 Mount Drive and set up the cache tree
from google.colab import drive
drive.mount('/content/drive')

import os
ROOT = '/content/drive/MyDrive/mrna_ensemble'
DATA = f'{ROOT}/data'          # mrna-bench dataset cache (slow to download -> keep on Drive)
WTS  = f'{ROOT}/weights'       # model weights
EMB  = f'{ROOT}/emb'           # our embedding cache
FEAT = f'{ROOT}/feat'          # biophysical feature cache
OUT  = f'{ROOT}/out'           # results
for p in (DATA, WTS, EMB, FEAT, OUT): os.makedirs(p, exist_ok=True)
print("cache root:", ROOT)
!df -h /content/drive/MyDrive 2>/dev/null | tail -1
''')

code(r'''
#@title 1.3 Install core mrna-bench (datasets + probing only)
# Deliberately NOT [base_models] yet: that extra pins torch 2.2.2 + cu121 and can break the runtime.
# Core is enough for datasets, splitters and LinearProbeBuilder, which is what phase 1 and the G1 gate need.
!pip install -q mrna-bench 2>&1 | tail -5
!pip install -q ViennaRNA 2>&1 | tail -3
print("---")
import importlib
for m in ("mrna_bench", "RNA", "sklearn", "scipy"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} OK  {getattr(mod,'__version__','(no __version__)')}")
    except Exception as e:
        print(f"{m:12s} FAIL {type(e).__name__}: {e}")
''')

code(r'''
#@title 1.4 Point mrna-bench at Drive and print both catalogs
import mrna_bench as mb
mb.update_data_path(DATA)
try:
    mb.update_model_weights_path(WTS)
except Exception as e:
    print("update_model_weights_path:", type(e).__name__, e)

print("=== mrna_bench top-level names ===")
print(sorted(n for n in dir(mb) if not n.startswith('_')))

print("\n=== DATASET CATALOG ===")
try:
    from mrna_bench.datasets import DATASET_CATALOG
    for k, v in sorted(DATASET_CATALOG.items()):
        print(f"  {k:34s} {getattr(v,'__name__',v)}")
except Exception as e:
    print("import failed:", type(e).__name__, e)
    print("fallback, mb.load_dataset signature:")
    import inspect; print(inspect.signature(mb.load_dataset))

print("\n=== MODEL CATALOG ===")
try:
    from mrna_bench.models import MODEL_CATALOG, MODEL_VERSION_MAP
    for k in sorted(MODEL_CATALOG): print(f"  {k}")
    print("\n  versions per model:")
    for k, v in sorted(MODEL_VERSION_MAP.items()): print(f"    {k:22s} {v}")
except Exception as e:
    print("import failed (expected without [base_models]):", type(e).__name__, e)
''')

code(r'''
#@title 1.5 Load one dataset and inspect what the rows actually contain
# rnahl-human is Saluki's own half-life data: small, regression, and has a strong non-LM incumbent.
ds = mb.load_dataset("rnahl-human")
df = ds.data_df
print("dataset  :", type(ds).__name__)
print("shape    :", df.shape)
print("columns  :", list(df.columns))
print("\ndtypes:"); print(df.dtypes)
print("\nfirst row, truncated:")
for c in df.columns:
    v = df[c].iloc[0]
    s = str(v)
    print(f"  {c:22s} {type(v).__name__:10s} len={len(v) if hasattr(v,'__len__') else '-'}  {s[:90]}")

# attributes the probe builder will need
for attr in ("target_col","task","species","dataset_name","embedding_dim","split_type"):
    print(f"\nds.{attr} =", getattr(ds, attr, "<absent>"))
print("\nall non-private attrs:", sorted(a for a in dir(ds) if not a.startswith('_'))[:40])
''')

code(r'''
#@title 1.6 Introspect the probe / splitter / evaluator API (we must not guess this)
import inspect
from mrna_bench.linear_probe import LinearProbeBuilder
print("=== LinearProbeBuilder methods ===")
for n in sorted(n for n in dir(LinearProbeBuilder) if not n.startswith('_')):
    try: print(f"  {n}{inspect.signature(getattr(LinearProbeBuilder,n))}")
    except (TypeError,ValueError): print(f"  {n}")

for modname, label in [("mrna_bench.linear_probe","linear_probe pkg"),
                       ("mrna_bench.linear_probe.data_splitter","splitters"),
                       ("mrna_bench.linear_probe.evaluator","evaluators")]:
    print(f"\n=== {label} ({modname}) ===")
    try:
        import importlib; m = importlib.import_module(modname)
        print(" ", sorted(n for n in dir(m) if not n.startswith('_')))
    except Exception as e:
        print("  import failed:", type(e).__name__, e)

# The two things the G1 gate depends on: how splits come out, and which metrics are computed.
print("\n=== split catalog / signatures ===")
try:
    from mrna_bench.linear_probe.data_splitter import SPLIT_CATALOG
    for k,v in SPLIT_CATALOG.items():
        print(f"  {k:14s} {v.__name__}")
        for n in ("get_all_splits_idx","get_split_idx","split"):
            if hasattr(v,n):
                try: print(f"      .{n}{inspect.signature(getattr(v,n))}")
                except Exception: print(f"      .{n}")
except Exception as e:
    print("  ", type(e).__name__, e)
''')

code(r'''
#@title 1.7 Does mRNABench already ship an ensemble / multi-embedding baseline?
# If it does, our planned contribution is already in the benchmark and we need to know NOW, not after
# burning GPU hours. Search the installed package for any sign of it.
import mrna_bench, os, re
from mrna_bench.linear_probe import LinearProbeBuilder
pkg = os.path.dirname(mrna_bench.__file__)
print("package at", pkg, "\n")
pat = re.compile(r'ensemble|stack|concat|multi_?embedding|combine|fuse', re.I)
hits = 0
for dp, _, fns in os.walk(pkg):
    for fn in fns:
        if not fn.endswith('.py'): continue
        p = os.path.join(dp, fn)
        for i, line in enumerate(open(p, errors='replace'), 1):
            if pat.search(line):
                print(f"  {os.path.relpath(p,pkg)}:{i}: {line.strip()[:110]}"); hits += 1
print(f"\n{hits} matches.")
print("Read these before phase 2. 'concat' inside a single model's track handling is fine and expected;")
print("what would sink the plan is a documented multi-model ensemble baseline with published numbers.")
# also check how many embedding slots a probe can hold
print("\nfetch_* methods on the builder (more than one => multi-embedding is a supported path):")
print([n for n in dir(LinearProbeBuilder) if n.startswith('fetch')])
''')

code(r'''
#@title 1.8 STATUS BLOCK — paste this whole output back before running phase 2
import importlib, sys, json, platform
rep = {}
try:
    import torch; rep['torch'] = f"{torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}"
    rep['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e: rep['torch'] = f"FAIL {e}"
for m in ("mrna_bench","RNA","sklearn","scipy","numpy","pandas"):
    try: rep[m] = getattr(importlib.import_module(m), "__version__", "ok")
    except Exception as e: rep[m] = f"FAIL {type(e).__name__}"
try:
    from mrna_bench.datasets import DATASET_CATALOG; rep['n_datasets'] = len(DATASET_CATALOG)
except Exception: rep['n_datasets'] = 'import failed'
try:
    from mrna_bench.models import MODEL_CATALOG; rep['n_models'] = len(MODEL_CATALOG)
except Exception as e: rep['n_models'] = f'import failed ({type(e).__name__}) -- expected pre-base_models'
try:
    rep['rnahl_shape'] = list(df.shape); rep['rnahl_cols'] = list(df.columns)
except Exception: rep['rnahl_shape'] = 'dataset cell not run'
try:
    from mrna_bench.linear_probe import LinearProbeBuilder as L
    rep['builder_methods'] = sorted(n for n in dir(L) if not n.startswith('_'))
except Exception as e: rep['builder_methods'] = f'FAIL {e}'
print("=========== STATUS ===========")
print(json.dumps(rep, indent=2, default=str))
print("==============================")
print("\nIf n_models says import failed, that is expected -- run 1.9 next.")
''')

code(r'''
#@title 1.9 Install the model backends ([base_models]) — the risky step
# This extra pins torch==2.2.2 + cu121. On Colab that can downgrade torch and require a RESTART.
# Run it, then RESTART THE RUNTIME when prompted, then re-run 1.2 and 1.4, then continue.
# If it wrecks the runtime, phase 2's fallback loader (cell 2.0) uses plain HF transformers instead.
!pip install --no-build-isolation -q 'mrna-bench[base_models]' 2>&1 | tail -25
print("--- after install ---")
import importlib
for m in ("torch","transformers"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "FAIL", e)
print("\n>>> RESTART RUNTIME NOW (Runtime > Restart session), then re-run 1.2 and 1.4. <<<")
''')

code(r'''
#@title 1.10 After restart — confirm the model catalog loads and one model instantiates
import torch, mrna_bench as mb
from mrna_bench.models import MODEL_CATALOG, MODEL_VERSION_MAP
print("models:", sorted(MODEL_CATALOG))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = mb.load_model("Orthrus", "orthrus-large-6-track", dev)   # name reused by phase 2
    print("\nloaded Orthrus ok ->", type(model).__name__)
    for a in ("embedding_dim","max_length","is_sixtrack","short_name"):
        print(f"  {a} =", getattr(model, a, "<absent>"))
    print("  methods:", [n for n in dir(model) if n.startswith('embed')])
except Exception as e:
    import traceback; traceback.print_exc()
    print("\nOrthrus failed -> use the fallback loader in 2.0 and tell me this traceback.")
''')

# ───────────────────────────────── phase 2 ─────────────────────────────────
md(r'''
## Phase 2 — correctness gate G1

Stacking needs out-of-fold predictions, which `LinearProbeBuilder` does not expose, so we run our own probe.
That makes this gate mandatory: **our probe, on mRNABench's own split indices, must reproduce
`LinearProbeBuilder`'s metric for the same single model to within 0.01** — on one regression and one
multilabel dataset. If it does not, K1 fires and nothing downstream is believable.
''')

code(r'''
#@title 2.0 Shared helpers: cache, embed, splits, probes
import os, json, hashlib, time, numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from scipy.stats import spearmanr, pearsonr

SEEDS = [2541, 1234, 42, 7, 31337]          # fixed by protocol
ALPHAS = np.logspace(-3, 5, 17)             # ridge / inverse-C grid

def _key(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]

def cached_npy(path, fn):
    """Compute fn() once and memoise to an .npy on Drive."""
    if os.path.exists(path):
        return np.load(path, allow_pickle=False)
    a = np.asarray(fn())
    tmp = path + ".tmp.npy"
    np.save(tmp, a); os.replace(tmp, path)
    return a

def embed_dataset(model, ds, tag, batch_hint=None):
    """DatasetEmbedder -> (n, d) float32, cached on Drive by (model tag, dataset name)."""
    from mrna_bench.embedder import DatasetEmbedder
    name = getattr(ds, 'dataset_name', type(ds).__name__)
    path = f"{EMB}/{tag}__{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    t0 = time.time()
    emb = DatasetEmbedder(model, ds).embed_dataset()
    emb = emb.detach().cpu().float().numpy()
    np.save(path + ".tmp.npy", emb); os.replace(path + ".tmp.npy", path)
    print(f"  embedded {tag} x {name}: {emb.shape} in {time.time()-t0:.0f}s -> {path}")
    return emb

# ---- metrics, matched to task type -------------------------------------------------
def eval_regression(y, p):
    return {"r2": float(r2_score(y, p)),
            "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(pearsonr(y, p).statistic)}

def eval_multilabel(Y, P):
    """Column-wise AUROC/AUPRC, averaged over columns that have both classes."""
    aur, aup = [], []
    Y = np.asarray(Y); P = np.asarray(P)
    if Y.ndim == 1: Y, P = Y[:, None], P[:, None]
    for j in range(Y.shape[1]):
        yj = Y[:, j]
        if 0 < yj.sum() < len(yj):
            aur.append(roc_auc_score(yj, P[:, j]))
            aup.append(average_precision_score(yj, P[:, j]))
    return {"auroc": float(np.mean(aur)) if aur else np.nan,
            "auprc": float(np.mean(aup)) if aup else np.nan,
            "n_cols": len(aur)}

# ---- level-1 probes ---------------------------------------------------------------
def fit_predict_reg(Xtr, ytr, Xva, Xte, yva):
    """Ridge with alpha chosen on validation Spearman. Returns val and test predictions."""
    sc = StandardScaler().fit(Xtr)
    A, B, C = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, bp = -np.inf, None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(A, ytr)
        s = spearmanr(yva, m.predict(B)).statistic
        if np.isfinite(s) and s > best:
            best, bp = s, (m.predict(B), m.predict(C), a)
    return bp[0], bp[1], {"alpha": bp[2], "val_spearman": best}

def fit_predict_multi(Xtr, Ytr, Xva, Xte, Yva):
    """One-vs-rest logistic, C chosen on mean validation AUROC."""
    sc = StandardScaler().fit(Xtr)
    A, B, C = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    Ytr = np.asarray(Ytr); Yva = np.asarray(Yva)
    if Ytr.ndim == 1: Ytr, Yva = Ytr[:, None], Yva[:, None]
    best, bp = -np.inf, None
    for a in ALPHAS:
        Pv = np.zeros((len(B), Ytr.shape[1])); Pt = np.zeros((len(C), Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            yj = Ytr[:, j]
            if yj.sum() == 0 or yj.sum() == len(yj):
                Pv[:, j] = Pt[:, j] = 0.5; continue
            m = LogisticRegression(C=1.0/a, max_iter=2000, solver='liblinear').fit(A, yj)
            Pv[:, j] = m.predict_proba(B)[:, 1]; Pt[:, j] = m.predict_proba(C)[:, 1]
        s = eval_multilabel(Yva, Pv)["auroc"]
        if np.isfinite(s) and s > best:
            best, bp = s, (Pv, Pt, a)
    return bp[0], bp[1], {"alpha": bp[2], "val_auroc": best}

print("helpers loaded. SEEDS =", SEEDS)
''')

code(r'''
#@title 2.1 Get mRNABench's own split indices (do not reimplement the splitter)
import inspect
from mrna_bench.linear_probe.data_splitter import SPLIT_CATALOG

def get_splits(ds, seed, species="human"):
    """Return (tr, va, te) index arrays using mRNABench's homology splitter.

    The exact constructor/method names are printed by 1.6; this tries the plausible
    shapes and raises loudly with the real signature if none fit, rather than guessing.
    """
    cls = SPLIT_CATALOG["homology"]
    errs = []
    for kwargs in ({"species": species}, {}):
        try:
            sp = cls(**kwargs)
        except Exception as e:
            errs.append(f"ctor{kwargs}: {e}"); continue
        for meth in ("get_all_splits_idx", "get_split_idx", "split"):
            if not hasattr(sp, meth): continue
            f = getattr(sp, meth)
            for args in ((ds.data_df, seed), (ds.data_df,), (ds, seed), (ds,)):
                try:
                    r = f(*args)
                    if isinstance(r, (tuple, list)) and len(r) == 3:
                        return tuple(np.asarray(x) for x in r)
                    errs.append(f"{meth}{len(args)}args returned {type(r)} len "
                                f"{len(r) if hasattr(r,'__len__') else '-'}")
                except Exception as e:
                    errs.append(f"{meth}{len(args)}args: {type(e).__name__}: {e}")
    print("=== splitter introspection ===")
    print(" ctor:", inspect.signature(cls.__init__))
    for m in ("get_all_splits_idx","get_split_idx","split"):
        if hasattr(cls, m):
            print(f" .{m}:", inspect.signature(getattr(cls, m)))
    raise RuntimeError("could not obtain splits; attempts:\n  " + "\n  ".join(errs))

tr, va, te = get_splits(ds, SEEDS[0])
print("split sizes:", len(tr), len(va), len(te), "of", len(ds.data_df))
assert len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0, "SPLIT OVERLAP -- stop"
print("no train/val/test index overlap. ok.")
''')

code(r'''
#@title 2.2 G1 — reference number from LinearProbeBuilder vs our probe
from mrna_bench.linear_probe import LinearProbeBuilder

TASK_TYPE = {  # filled from the catalog; corrected in phase 1 if 1.5 says otherwise
 "rnahl-human":"reg","rnahl-mouse":"reg","mrl-sugimoto":"reg","translation-efficiency-human":"reg",
 "prot-loc":"multi","go-mf":"multi","eclip-binding-k562":"multi",
}

def target_matrix(ds):
    """y (n,) for regression or Y (n,k) for multilabel, from whatever column the dataset uses."""
    d = ds.data_df
    col = getattr(ds, "target_col", None)
    if col is None:
        cands = [c for c in d.columns if c in ("target","targets","label","labels","y")]
        if not cands: raise RuntimeError(f"no target column found in {list(d.columns)}")
        col = cands[0]
    v = d[col].values
    if isinstance(v[0], (list, np.ndarray)):
        return np.stack([np.asarray(x, float) for x in v]), col
    return np.asarray(v, float), col

name = getattr(ds, 'dataset_name', 'rnahl-human')
tt = TASK_TYPE.get(name, "reg")
y, ycol = target_matrix(ds)
print(f"{name}: task={tt} target col='{ycol}' y shape={y.shape}")

emb = embed_dataset(model, ds, "orthrus-large-6")      # from 1.10
print("embeddings:", emb.shape)

# --- their harness ---
prober = (LinearProbeBuilder(ds)
          .fetch_embedding_by_embedding_instance("orthrus-large-6", emb)
          .build_splitter("homology", species="human", eval_all_splits=False)
          .build_evaluator("regression" if tt == "reg" else "multilabel")
          .set_target(ycol).build())
theirs = prober.run_linear_probe(SEEDS[0])
print("\nLinearProbeBuilder ->", theirs)

# --- ours, same indices ---
tr, va, te = get_splits(ds, SEEDS[0])
if tt == "reg":
    pv, pt, info = fit_predict_reg(emb[tr], y[tr], emb[va], emb[te], y[va])
    ours = eval_regression(y[te], pt)
else:
    pv, pt, info = fit_predict_multi(emb[tr], y[tr], emb[va], emb[te], y[va])
    ours = eval_multilabel(y[te], pt)
print("ours               ->", ours, info)
print("\n>>> G1: compare the shared metric. Difference must be < 0.01.")
print(">>> If their dict uses a metric name ours lacks, tell me the key names and I will match them.")
''')

code(r'''
#@title 2.3 G1 on a multilabel dataset too (gate needs both task types)
ds_m = mb.load_dataset("prot-loc")
print("prot-loc:", ds_m.data_df.shape, list(ds_m.data_df.columns))
ym, ycol_m = target_matrix(ds_m)
print("Y:", ym.shape)
emb_m = embed_dataset(model, ds_m, "orthrus-large-6")

prober_m = (LinearProbeBuilder(ds_m)
            .fetch_embedding_by_embedding_instance("orthrus-large-6", emb_m)
            .build_splitter("homology", species="human", eval_all_splits=False)
            .build_evaluator("multilabel").set_target(ycol_m).build())
print("\ntheirs ->", prober_m.run_linear_probe(SEEDS[0]))

trm, vam, tem = get_splits(ds_m, SEEDS[0])
_, ptm, infom = fit_predict_multi(emb_m[trm], ym[trm], emb_m[vam], emb_m[tem], ym[vam])
print("ours   ->", eval_multilabel(ym[tem], ptm), infom)
print("\n>>> G1 passes only if BOTH datasets agree to 0.01. Otherwise K1: stop.")
''')

# ───────────────────────────────── phase 3 ─────────────────────────────────
md(r'''
## Phase 3 — embed every (model × dataset)

Resumable: each embedding is one `.npy` on Drive keyed by model tag and dataset name. A disconnect costs
only the in-flight item. Re-run the cell to continue. Models that fail to load are recorded and skipped —
the protocol's minimum viable set is **4 models spanning ≥3 pretraining objectives**, so check the tally at
the end against that before moving on.
''')

code(r'''
#@title 3.1 The model and dataset grid
# (catalog key, version, short tag, pretraining objective) -- objective matters for the diversity floor
MODELS = [
 ("Orthrus",              "orthrus-large-6-track",   "orthrus-l6",  "contrastive-isoform"),
 ("Orthrus",              "orthrus-base-4-track",    "orthrus-b4",  "contrastive-isoform"),
 ("NucleotideTransformer","500m-human-ref",          "nt-500m-hs",  "genomic-MLM"),
 ("DNABERT2",             "dnabert2",                "dnabert2",    "genomic-MLM-bpe"),
 ("SpliceBERT",           "SpliceBERT-human.510nt",  "splicebert",  "pre-mRNA-MLM"),
 ("RiNALMo",              "rinalmo",                 "rinalmo",     "ncRNA-nt-MLM"),
 ("RNAFM",                "mrna-fm",                 "mrna-fm",     "codon-MLM"),
 ("CodonBERT",            "codonbert",               "codonbert",   "codon-LM"),
 ("HyenaDNA",             "hyenadna-medium-160k-seqlen-hf","hyena-160k","genomic-AR"),
]
DATASETS = ["rnahl-human","rnahl-mouse","mrl-sugimoto","translation-efficiency-human",
            "prot-loc","go-mf","eclip-binding-k562"]
print(len(MODELS), "models x", len(DATASETS), "datasets =", len(MODELS)*len(DATASETS), "embedding jobs")
print("\nobjectives represented:", sorted({o for *_, o in MODELS}))
''')

code(r'''
#@title 3.2 Run the embedding grid (resumable — just re-run after a disconnect)
import gc, traceback, torch, json, os
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = f"{OUT}/embed_log.json"
log = json.load(open(LOG)) if os.path.exists(LOG) else {}

for cat, ver, tag, obj in MODELS:
    pending = [d for d in DATASETS if not os.path.exists(f"{EMB}/{tag}__{d}.npy")]
    if not pending:
        print(f"[{tag}] all {len(DATASETS)} cached, skip"); continue
    try:
        mdl = mb.load_model(cat, ver, dev)
    except Exception as e:
        log[tag] = f"LOAD FAIL {type(e).__name__}: {e}"
        print(f"[{tag}] LOAD FAIL {type(e).__name__}: {e}")
        json.dump(log, open(LOG,'w'), indent=1); continue
    print(f"[{tag}] loaded ({obj}), {len(pending)} to do")
    for dname in pending:
        try:
            d = mb.load_dataset(dname)
            embed_dataset(mdl, d, tag)
            log[f"{tag}__{dname}"] = "ok"
        except Exception as e:
            log[f"{tag}__{dname}"] = f"{type(e).__name__}: {e}"
            print(f"  [{tag} x {dname}] FAIL {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        json.dump(log, open(LOG,'w'), indent=1)
    del mdl; gc.collect(); torch.cuda.empty_cache()

print("\n=== embedding tally ===")
have = {}
for cat, ver, tag, obj in MODELS:
    n = sum(os.path.exists(f"{EMB}/{tag}__{d}.npy") for d in DATASETS)
    have[tag] = (n, obj)
    print(f"  {tag:14s} {n}/{len(DATASETS)}  {obj}")
ok_models = [t for t,(n,_) in have.items() if n == len(DATASETS)]
ok_objs = {o for t,(n,o) in have.items() if n == len(DATASETS)}
print(f"\ncomplete models: {len(ok_models)} -> {ok_models}")
print(f"distinct objectives: {len(ok_objs)} -> {sorted(ok_objs)}")
print("PROTOCOL FLOOR (>=4 models, >=3 objectives):",
      "MET" if len(ok_models) >= 4 and len(ok_objs) >= 3 else "NOT MET -- fix loads before phase 5")
''')

# ───────────────────────────────── phase 4 ─────────────────────────────────
md(r'''
## Phase 4 — the biophysical feature block

~100 explicit features that are in no backbone's input as a feature: folding energy on capped windows,
codon usage and CAI, UTR geometry, uAUG/Kozak, ARE and polyA signals. Everything deterministic, everything
cached. `max_bp_span=100` keeps RNAfold roughly linear in length, which is what makes this tractable on
multi-kb mRNAs.

**K4 lives here**: if this block alone lands within 0.01 of the best language model on a task, that task's
language models are not earning their parameters, and that is the headline for that task.
''')

code(r'''
#@title 4.1 Feature extraction
import RNA, numpy as np, re
from collections import Counter

BASE_ORDER = "TCAG"                        # order that matches the standard-code AA string below
CODONS = [a+b+c for a in BASE_ORDER for b in BASE_ORDER for c in BASE_ORDER]
CODON_IX = {c: i for i, c in enumerate(CODONS)}
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_AA = {c: AA[i] for i, c in enumerate(CODONS)}
assert CODON_AA["ATG"] == "M" and CODON_AA["TAA"] == "*" and CODON_AA["TGG"] == "W"

# Cost note, measured not guessed: fc.pf() is ~70% of folding time (110 ms/seq with it, 33 ms without on a
# 300 nt window) and the only feature it adds here is ensemble free energy per nt, which correlates ~0.99
# with MFE per nt. So USE_PF defaults off: ~35 min -> ~11 min per 10k sequences before parallelism.
# Set it True if you want efe_* columns; the cached block records the setting so the two cannot be mixed.
USE_PF = False
MAX_BP_SPAN = 100          # keeps folding ~linear in length; without it multi-kb mRNAs are intractable

def fold_window(s, max_span=MAX_BP_SPAN, use_pf=USE_PF):
    """(mfe/nt, efe/nt or None, unpaired fraction) for one window. Zeros for a window too short to fold."""
    if len(s) < 10:
        return 0.0, (0.0 if use_pf else None), 0.0
    md = RNA.md(); md.max_bp_span = max_span
    fc = RNA.fold_compound(s.replace("T", "U"), md)
    st, mfe = fc.mfe()
    efe = fc.pf()[1] / len(s) if use_pf else None
    return mfe / len(s), efe, st.count(".") / len(st)

AREP = re.compile("ATTTA"); PAS = re.compile("AATAAA|ATTAAA")

def bio_features(seq, cds=None, splice=None):
    """seq: str. cds: optional per-nt 0/1 track marking coding positions (mRNABench 6-track)."""
    s = seq.upper().replace("U","T")
    n = len(s)
    f = {}
    f['len'] = n; f['log_len'] = np.log1p(n)
    f['gc'] = (s.count("G")+s.count("C"))/max(n,1)
    for b in "ACGT": f[f'frac_{b}'] = s.count(b)/max(n,1)

    # region geometry from the CDS track when present, else treat whole sequence as one region
    if cds is not None and len(cds) == n and np.any(np.asarray(cds) > 0):
        c = np.asarray(cds) > 0
        idx = np.flatnonzero(c); a, z = int(idx[0]), int(idx[-1])+1
        u5, cd, u3 = s[:a], s[a:z], s[z:]
    else:
        u5, cd, u3 = "", s, ""
    f['len_5utr'], f['len_cds'], f['len_3utr'] = len(u5), len(cd), len(u3)
    f['frac_cds'] = len(cd)/max(n,1)
    f['log_len_5utr'] = np.log1p(len(u5)); f['log_len_3utr'] = np.log1p(len(u3))
    f['gc_5utr'] = (u5.count("G")+u5.count("C"))/max(len(u5),1)
    f['gc_3utr'] = (u3.count("G")+u3.count("C"))/max(len(u3),1)
    f['gc_cds']  = (cd.count("G")+cd.count("C"))/max(len(cd),1)
    f['gc3'] = sum(1 for i in range(2,len(cd),3) if cd[i] in "GC")/max(len(cd)//3,1)

    # folding on capped windows: cap-proximal, 5'UTR, CDS start, 3'UTR end
    for label, w in (('cap70', s[:70]), ('u5', u5[-200:]), ('cds200', cd[:200]), ('u3', u3[-300:])):
        mfe, efe, unp = fold_window(w)
        f[f'mfe_{label}'] = mfe; f[f'unpaired_{label}'] = unp
        if efe is not None: f[f'efe_{label}'] = efe

    # translation-initiation context
    f['uaug'] = u5.count("ATG")
    f['uaug_per_100'] = 100*u5.count("ATG")/max(len(u5),1)
    st = len(u5)                           # index of the A of the main AUG
    f['kozak_m3_purine'] = 1.0 if st >= 3 and s[st-3] in "AG" else 0.0
    f['kozak_p4_G'] = 1.0 if len(s) > st+3 and s[st+3] == "G" else 0.0
    f['has_atg_at_cds_start'] = 1.0 if s[st:st+3] == "ATG" else 0.0

    # stability elements
    f['are'] = len(AREP.findall(u3)); f['are_per_kb'] = 1000*len(AREP.findall(u3))/max(len(u3),1)
    f['pas'] = len(PAS.findall(u3))

    # codon usage over the in-frame CDS + CAI against the human reference
    codons = [cd[i:i+3] for i in range(0, len(cd)-2, 3)]
    codons = [c for c in codons if c in CODON_IX]
    cnt = Counter(codons); tot = max(len(codons), 1)
    for c in CODONS: f[f'cod_{c}'] = cnt[c]/tot
    f['n_codons'] = len(codons)
    f['stop_premature'] = sum(1 for c in codons[:-1] if CODON_AA[c] == "*")
    return f

# smoke test on a short synthetic mRNA with an explicit CDS track
import numpy as _np
_seq = "GGGACACGCATGCA" + "ATG" + "GCTACGGATCCA"*20 + "TAA" + "ATTTAATTTAAATAAA"
_tr = _np.zeros(len(_seq)); _tr[14:14+3+240+3] = 1
_t = bio_features(_seq, _tr)
assert all(_np.isfinite(list(_t.values()))), "non-finite feature in smoke test"
assert _t['has_atg_at_cds_start'] == 1.0 and _t['len_5utr'] == 14
print(f"{len(_t)} features, USE_PF={USE_PF}; spot check:",
      {k: round(float(_t[k]), 4) for k in ('len_5utr','len_cds','mfe_cap70','unpaired_u3','uaug','are','n_codons','gc3')})
''')

code(r'''
#@title 4.2 Build and cache the feature block per dataset (resumable)
import os, json, time, numpy as np, pandas as pd
from multiprocessing import Pool
NPROC = max(1, (os.cpu_count() or 2))
print(f"folding on {NPROC} processes")

def _bio_one(a):
    """Top-level so it is picklable by multiprocessing."""
    return bio_features(a[0], a[1])

def feature_block(dname):
    """(n, p) float32 feature matrix for a dataset, cached to Drive. Writes the column names alongside."""
    path = f"{FEAT}/{dname}.pf{int(USE_PF)}.npy"
    cpath = f"{FEAT}/{dname}.pf{int(USE_PF)}.cols.json"
    if os.path.exists(path) and os.path.exists(cpath):
        return np.load(path), json.load(open(cpath))
    d = mb.load_dataset(dname).data_df
    seqcol = next(c for c in d.columns if c.lower() in ("sequence","seq","sequences"))
    cdscol = next((c for c in d.columns if c.lower() in ("cds","cds_track")), None)
    print(f"  {dname}: {len(d)} rows, seq col '{seqcol}', cds col {cdscol!r}", flush=True)
    args = [(getattr(r, seqcol), getattr(r, cdscol) if cdscol else None)
            for r in d.itertuples(index=False)]
    t0 = time.time()
    with Pool(NPROC) as pool:
        rows = []
        for i, row in enumerate(pool.imap(_bio_one, args, chunksize=32)):
            rows.append(row)
            if i and i % 1000 == 0:
                rate = (time.time()-t0)/i
                print(f"    {i}/{len(args)}  {time.time()-t0:.0f}s  eta {rate*(len(args)-i)/60:.1f} min", flush=True)
    F = pd.DataFrame(rows)
    cols = list(F.columns)
    A = F.values.astype(np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(path + ".tmp.npy", A); os.replace(path + ".tmp.npy", path)
    json.dump(cols, open(cpath, "w"))
    print(f"  {dname}: {A.shape} in {time.time()-t0:.0f}s")
    return A, cols

for dn in DATASETS:
    try:
        A, cols = feature_block(dn)
        print(f"{dn:32s} {A.shape}")
    except Exception as e:
        import traceback; print(f"{dn:32s} FAIL {type(e).__name__}: {e}"); traceback.print_exc(limit=2)
''')

# ───────────────────────────────── phase 5 ─────────────────────────────────
md(r'''
## Phase 5 — probe every configuration on identical splits and seeds

Level-1 probes fit on **train**. The level-2 combiner fits on **validation** predictions. Test is touched
once per (configuration, seed). "Best single" is picked on validation, never on test — that is the whole
point of the paired contrast, and picking it on test would manufacture the win we are trying to measure.

Both `concat` and `stack` are run because which one wins is empirical, not decidable in advance. In a
synthetic check where the signal is exactly linear in the raw features, concat wins (0.92 vs 0.84); with
real embeddings the blocks are thousands of dimensions against a few thousand samples, which is the regime
where stacking's compression should pay off instead. If concat wins on the real data, that is the answer.
''')

code(r'''
#@title 5.1 Configurations and the run loop
import itertools, json, os, numpy as np

def load_blocks(dname, tags):
    out = {}
    for t in tags:
        p = f"{EMB}/{t}__{dname}.npy"
        if os.path.exists(p): out[t] = np.load(p)
    fp = f"{FEAT}/{dname}.pf{int(USE_PF)}.npy"
    if os.path.exists(fp): out['bio'] = np.load(fp)
    return out

def run_dataset(dname, tags, seeds=SEEDS):
    tt = TASK_TYPE[dname]
    ds_ = mb.load_dataset(dname)
    y, _ = target_matrix(ds_)
    blocks = load_blocks(dname, tags)
    model_tags = [t for t in blocks if t != 'bio']
    if len(model_tags) < 2:
        return []
    fitp = fit_predict_reg if tt == "reg" else fit_predict_multi
    evalf = eval_regression if tt == "reg" else eval_multilabel
    key = "spearman" if tt == "reg" else "auroc"
    rows = []
    for seed in seeds:
        tr, va, te = get_splits(ds_, seed)
        L1 = {}                                  # block -> (val preds, test preds, info)
        for b, X in blocks.items():
            L1[b] = fitp(X[tr], y[tr], X[va], X[te], y[va])
        # single-model and bio-only rows
        for b, (pv, pt, info) in L1.items():
            rows.append(dict(dataset=dname, seed=seed, config=("bio" if b=="bio" else f"single:{b}"),
                             val_score=info.get("val_spearman", info.get("val_auroc")),
                             **evalf(y[te], pt)))
        # val-selected best single (models only, bio excluded -- it is not a language model)
        best_tag = max(model_tags, key=lambda b: L1[b][2].get("val_spearman", L1[b][2].get("val_auroc")))
        rows.append(dict(dataset=dname, seed=seed, config="best-single-by-val", chosen=best_tag,
                         val_score=L1[best_tag][2].get("val_spearman", L1[best_tag][2].get("val_auroc")),
                         **evalf(y[te], L1[best_tag][1])))
        # concat of standardised blocks
        for label, use in (("concat-all", model_tags), ("concat-all+bio", model_tags + (['bio'] if 'bio' in blocks else []))):
            Xc = np.hstack([blocks[b] for b in use])
            pv, pt, info = fitp(Xc[tr], y[tr], Xc[va], Xc[te], y[va])
            rows.append(dict(dataset=dname, seed=seed, config=label,
                             val_score=info.get("val_spearman", info.get("val_auroc")),
                             **evalf(y[te], pt)))
        # stacking: level-2 on validation predictions
        for label, use in (("stack", model_tags), ("stack+bio", model_tags + (['bio'] if 'bio' in blocks else []))):
            col = lambda a: np.asarray(a).reshape(len(a), -1)
            Zv = np.hstack([col(L1[b][0]) for b in use])
            Zt = np.hstack([col(L1[b][1]) for b in use])
            # level-2 trains on val, so there is no held-out set for alpha; use a fixed mid alpha
            if tt == "reg":
                from sklearn.linear_model import RidgeCV
                sc = StandardScaler().fit(Zv)
                m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Zv), y[va])
                pt2 = m.predict(sc.transform(Zt))
            else:
                sc = StandardScaler().fit(Zv)
                A2, B2 = sc.transform(Zv), sc.transform(Zt)
                Yv = y[va].reshape(len(va), -1)
                pt2 = np.zeros((len(B2), Yv.shape[1]))
                for j in range(Yv.shape[1]):
                    yj = Yv[:, j]
                    if 0 < yj.sum() < len(yj):
                        mm = LogisticRegression(C=1.0, max_iter=2000, solver='liblinear').fit(A2, yj)
                        pt2[:, j] = mm.predict_proba(B2)[:, 1]
                    else:
                        pt2[:, j] = 0.5
            rows.append(dict(dataset=dname, seed=seed, config=label, **evalf(y[te], pt2)))
        print(f"  {dname} seed {seed}: best-single={best_tag}", flush=True)
    return rows

RES = f"{OUT}/results.json"
allrows = json.load(open(RES)) if os.path.exists(RES) else []
done = {(r['dataset'], r['seed']) for r in allrows}
for dn in DATASETS:
    if all((dn, s) in done for s in SEEDS):
        print(f"{dn}: cached, skip"); continue
    print(f"== {dn} ==")
    try:
        allrows += run_dataset(dn, [t for _, _, t, _ in MODELS])
        json.dump(allrows, open(RES,'w'), indent=1)
    except Exception as e:
        import traceback; print(f"{dn} FAILED: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print(f"\n{len(allrows)} result rows -> {RES}")
''')

# ───────────────────────────────── phase 6 ─────────────────────────────────
md(r'''
## Phase 6 — the verdict

Three numbers decide it, and they were fixed before the run:

* **K2** win rate of `stack+bio` over the val-selected best single model. Under 60% kills the
  across-the-board claim.
* **K3** pooled paired improvement with a hierarchical bootstrap over (dataset, seed). An interval
  containing 0 is a null, and the null gets reported as the result.
* **K4** per task, is `bio` alone within 0.01 of the best language model?

Plus the decorrelation map, which is what makes a win worth writing up: an ensemble that wins for no
identifiable reason is not a finding.
''')

code(r'''
#@title 6.1 Win rates, paired bootstrap, and the decorrelation map
import json, numpy as np, pandas as pd
R = pd.DataFrame(json.load(open(f"{OUT}/results.json")))
KEY = {"reg":"spearman","multi":"auroc"}
R['metric'] = [KEY[TASK_TYPE[d]] for d in R.dataset]
R['score']  = [r.get(k, np.nan) for r, k in zip(R.to_dict('records'), R.metric)]

piv = R.pivot_table(index=['dataset','seed'], columns='config', values='score')
print("=== mean score per configuration per dataset ===")
print(piv.groupby('dataset').mean().round(4).to_string())

base, test = 'best-single-by-val', 'stack+bio'
if base in piv and test in piv:
    d = (piv[test] - piv[base]).dropna()
    per_ds = d.groupby('dataset').mean()
    print(f"\n=== K2: {test} vs {base} ===")
    print(per_ds.round(4).to_string())
    wr = float((per_ds > 0).mean())
    print(f"\nwin rate over datasets: {wr:.1%}  ->  K2 {'MET' if wr >= 0.60 else 'NOT MET (claim dead)'}")

    # hierarchical bootstrap: resample datasets, then seeds within dataset
    rng = np.random.default_rng(0)
    dss = per_ds.index.to_numpy(); boots = []
    for _ in range(10000):
        pick = rng.choice(dss, len(dss), replace=True)
        vals = []
        for ds_ in pick:
            v = d.loc[ds_].to_numpy()
            vals.append(rng.choice(v, len(v), replace=True).mean())
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"pooled paired improvement: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"K3 {'MET (interval excludes 0)' if lo > 0 or hi < 0 else 'NOT MET -- null, report it as such'}")

# K4: is explicit biophysics as good as the best language model?
if 'bio' in piv and base in piv:
    k4 = (piv[base] - piv['bio']).groupby('dataset').mean()
    print("\n=== K4: best language model minus bio-only ===")
    for ds_, v in k4.items():
        print(f"  {ds_:32s} {v:+.4f}" + ("   <- LMs not earning their parameters" if abs(v) < 0.01 else ""))

# which single models get picked, and how often
if 'chosen' in R.columns:
    print("\n=== val-selected best single model, by dataset ===")
    print(R[R.config==base].groupby(['dataset','chosen']).size().to_string())
''')

code(r'''
#@title 6.2 Decorrelation map — where any win actually comes from
# Correlate per-model test-set residuals. Highly correlated models cannot help each other; the ensemble's
# gain should track the LOW-correlation pairs. If every pair sits above ~0.9, the honest headline is
# "nine RNA backbones carry one signal", which is a result in its own right.
import numpy as np, itertools, pandas as pd

def residual_corr(dname, seed=SEEDS[0]):
    tt = TASK_TYPE[dname]; ds_ = mb.load_dataset(dname); y, _ = target_matrix(ds_)
    tr, va, te = get_splits(ds_, seed)
    blocks = load_blocks(dname, [t for _,_,t,_ in MODELS])
    fitp = fit_predict_reg if tt == "reg" else fit_predict_multi
    res = {}
    for b, X in blocks.items():
        _, pt, _ = fitp(X[tr], y[tr], X[va], X[te], y[va])
        yt = y[te]
        res[b] = (np.asarray(yt, float).ravel() - np.asarray(pt, float).ravel())
    names = list(res)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        c = np.corrcoef(res[a], res[b])[0, 1]
        M.loc[a, b] = M.loc[b, a] = c
    return M

for dn in DATASETS[:3]:
    try:
        M = residual_corr(dn)
        print(f"\n=== residual correlation, {dn} ===")
        print(M.round(3).to_string())
        off = M.values[np.triu_indices(len(M), 1)]
        print(f"mean |r| off-diagonal: {np.abs(off).mean():.3f}   min {off.min():.3f}   max {off.max():.3f}")
    except Exception as e:
        print(f"{dn}: {type(e).__name__}: {e}")
''')

code(r'''
#@title 6.3 Export everything for figures and for me to read
import json, os, pandas as pd
R.to_csv(f"{OUT}/results_tidy.csv", index=False)
piv.to_csv(f"{OUT}/results_pivot.csv")
print("wrote:")
for f in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{f}  {os.path.getsize(os.path.join(OUT,f)):,} bytes")
print("\nPaste results_pivot.csv and the phase 6 output back and I will write it up -- including if it is a null.")
''')

code(r'''
#@title 6.4 Release the runtime
from google.colab import runtime
runtime.unassign()
''')

nb = {
 "cells": [{"cell_type": t,
            "metadata": {},
            "source": (s + "\n").splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if t == "code" else {})}
           for t, s in CELLS],
 "metadata": {"accelerator": "GPU",
              "colab": {"provenance": [], "toc_visible": True},
              "kernelspec": {"display_name": "Python 3", "name": "python3"},
              "language_info": {"name": "python"}},
 "nbformat": 4, "nbformat_minor": 0,
}
out = sys.argv[1] if len(sys.argv) > 1 else "ensemble_colab.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out}: {len(CELLS)} cells ({sum(1 for t,_ in CELLS if t=='code')} code)")
