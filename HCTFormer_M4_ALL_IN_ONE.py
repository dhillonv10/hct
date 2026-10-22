"""
HCT T-CELL FOUNDATION MODEL (v4)  --  single file, public data only
=====================================================================
A T-cell "baseline" model trained on healthy human T/NK cells, then used as a FROZEN feature extractor for GVHD-related
clinical prediction tasks defined on public transplant cohorts. It replaces the earlier HCTFormer M4 script, whose
problems (13-patient supervised set, single-cell/bulk mismatch, gene panel chosen by one 6-sample study, discarded
labels/patient ids, silent failure paths) were found by running it on the real data.
Whether the pretrained ENCODER adds anything beyond the baseline NORMALISATION is measured in every run (results/RUN_REPORT.md),
not assumed: the report compares encoder features with gene-level features and with an untrained encoder of the same size.

RUN       python3 HCTFormer_M4_ALL_IN_ONE.py            (all stages; resumable; first run downloads ~9 GB)
          python3 HCTFormer_M4_ALL_IN_ONE.py selftest   (offline, ~15 s: labels, parsers, leakage, grouped CV, exact resume)
          HCT_PROFILE=smoke ...                         (CPU-sized run; default profile is 'laptop' for Apple silicon)
          Outputs go to ~/HCT_TcellFM_Run  (HCT_ROOT to change): model/, results/RUN_REPORT.md, run_log.txt.
USE       from HCTFormer_M4_ALL_IN_ONE import predict_clinical     # guarded scoring of new expression tables
EXTEND    new cohort -> write a loader returning `Cohort` (strict label parsing, patient ids), add it to COHORT_LOADERS, then add
          a dict to TASKS (cohort, compartments, kind, confounders). New pretraining atlas -> add an entry to PRETRAIN_SOURCES.

DATA POLICY (enforced in code, not just stated)
  1. PRETRAINING uses only public HEALTHY human T/NK single-cell data (four CELLxGENE atlases, pinned by size+sha256).
     No GVHD/transplant/clinical-cohort sample, and no clinical label, enters pretraining.
  2. CLINICAL cohorts (GSE4624, GSE182610/12/78, GSE73809, GSE103569, GSE295359) define and evaluate tasks and fit small
     heads on frozen features. They are never pretraining data.
  3. No synthetic data are ever used for training (self-tests use arrays only to test code paths).
  4. A download whose size/sha256 differs from the pinned value, a label outside the verified vocabulary, or a sample
     count that differs from what was verified STOPS the run. Nothing is guessed or coerced.

THE T-CELL BASELINE
  Every profile (10x pseudobulk, bulk RNA-seq, microarray) is mapped to one space: log expression of a T-cell gene
  vocabulary (4,096 genes in the 'laptop' profile) -> within-sample quantile normalisation to the healthy-T-cell reference distribution -> per-gene z-score
  against the healthy-T-cell baseline. 0 = "typical healthy T cell"; clinical samples enter as DEVIATIONS from baseline.
  Training samples are multi-resolution PSEUDOBULKS (10-400 cells of one donor; natural, sorted-like and Dirichlet
  mixtures) with simulated platform gene offsets and partial gene coverage, so single-cell data train a model that is
  applied to bulk data (the earlier version mixed ~24-gene single cells with ~980-gene bulk and did not transfer).

PRETRAINING OBJECTIVES  (held-out DONORS evaluate them; each is reported against trivial and PCA/ridge baselines)
  P1 masked-gene reconstruction   10-40% of genes hidden; loss = MSE on hidden genes only (values are zeroed before they
                                  are embedded, so leakage is impossible and tested).
  P2 T-cell-state composition     predict the composition of the pseudobulk over 10 states (CD4 naive/memory/CTL, Treg, CD8
                                  naive/central/effector memory, MAIT, gdT, NK) and 5 lineages from the atlases' own
                                  annotations (soft-label cross-entropy). This is deconvolution learned from public labels.
  P3 activation level             fraction of in-vitro-activated cells (resting vs CD3/CD28-activated T cells, 4 tissues).

CLINICAL TASKS  (unit = patient; label and timing stated; all use frozen features + a regularised logistic head)
  K1 PROGNOSTIC   donor CD4 / CD8 T cells BEFORE transplant (GSE4624, 50 HLA-identical sibling donors) -> recipient acute GVHD.
  K2 DIAGNOSTIC   sorted CD8 / CD4-Tconv / Treg after post-transplant cyclophosphamide (GSE182610/12/78) -> GVHD at sampling.
  K3 DIAGNOSTIC   PBMC CD4+CD8 T cells, matched case-control, day 28-176 (GSE73809, 24 patients) -> aGVHD.
  K4 DIAGNOSTIC   three sorted CD8 subsets per patient (GSE103569, 20 patients) -> aGVHD (any organ).
  K5 DIAGNOSTIC   single-cell-derived pseudobulk (GSE295359, 10 patients; graft/D30/D100) -> aGVHD II-IV (grouped by patient).
  K5x (exploratory) infused-graft T cells -> later aGVHD II-IV (8 patients).

EVALUATION PROTOCOL
  * Patient-grouped, stratified, repeated K-fold; head strength chosen by nested patient-grouped CV; AUROC pooled out-of-fold.
  * 95% CIs by patient-level bootstrap; permutation null (patient-level label shuffles through the whole nested procedure) for
    the pre-specified primary family (`fmA_all`: anchored foundation-model features), the deployable family (`genes_z`:
    baseline-normalised genes) and the comparator (`genes_native`: every gene the platform measured, top-2000 variance).
  * Every result sits next to: gene-level baselines (all platform genes; vocabulary genes; vocabulary genes after the same
    normalisation), a randomly initialised encoder of identical architecture (does pretraining matter?), gene-by-gene residual
    profiles (exploratory), and a confounder-only model (batch / array order / site / day / timepoint).
  * Cross-study transfer: train on all other studies' patients, test on the held-out study.
  * Transfer check: can the composition head recognise SORTED bulk populations on other platforms (no GVHD label)?
  * The report states its own bottom line (how many tasks each family clears chance on) -- read it before trusting any head.

EXPORTED
  model/tcell_baseline_fm.pt   encoder, vocabulary, healthy-T-cell reference, pretraining report (loads with weights_only=True)
  model/clinical_heads.pt      per task and family (`fmA_all`, `genes_z`): head + its CV AUROC, permutation p and verdict
  results/RUN_REPORT.md        all tables;  predict_clinical(table, task_id, family='genes_z') scores new samples, refusing
                               tables that cover <35% of the vocabulary, and attaches the head's evidence to every output.

LIMITATIONS (also written into every exported head)
  * A few dozen patients per task: confidence intervals are wide, and a feature family that cannot beat chance here may
    still carry signal at larger n. Nothing here is a validated clinical device.
  * Heads are calibrated to one platform/compartment/protocol; the cohorts have no external validation set of their own.
  * GEO supplies no patient ids for the PTCy cohort (populations are modelled separately); GSE4624 donor groups are inferred
    from consecutive ids (45/45 pairs outcome-concordant); onset timing is absent from GSE73809/GSE295359 metadata.
  * Not run on Apple MPS by the author of this version: the code is device-agnostic (MPS -> CUDA -> CPU, logged) but the
    laptop-profile timings are estimates; resume is bit-exact on CPU (tested).
"""
# ---------------------------------------------------------------------------
# ENVIRONMENT BOOTSTRAP (runs only when executed as a script, BEFORE heavy imports)
# ---------------------------------------------------------------------------
import os as _os, sys as _sys
_os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # unsupported MPS ops fall back per-op instead of crashing

_REQUIRED = {  # import name -> (pip requirement, min version, exclusive max version)
    'torch': ('torch>=2.4', '2.4', None),
    'numpy': ('numpy>=1.26,<3', '1.26', '3'),
    'pandas': ('pandas>=2.1,<3', '2.1', '3'),
    'scipy': ('scipy>=1.11,<2', '1.11', '2'),
    'sklearn': ('scikit-learn>=1.4,<2', '1.4', '2'),
    'anndata': ('anndata>=0.10,<0.13', '0.10', '0.13'),
    'h5py': ('h5py>=3.10', '3.10', None),
}
_DIST = {'sklearn': 'scikit-learn'}


def _vtuple(v):
    out = []
    for p in v.replace('+', '.').split('.'):
        digits = ''.join(ch for ch in p if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _unsatisfied():
    import importlib.util
    from importlib import metadata
    bad = []
    for mod, (req, lo, hi) in _REQUIRED.items():
        if importlib.util.find_spec(mod) is None:
            bad.append(req)
            continue
        try:
            ver = _vtuple(metadata.version(_DIST.get(mod, mod)))
        except Exception:
            bad.append(req)
            continue
        if ver < _vtuple(lo) or (hi is not None and ver >= _vtuple(hi)):
            bad.append(req)       # installed but outside the tested range
    return bad


def _bootstrap_environment():
    """Re-run inside a private venv when packages are missing OR outside the tested version range."""
    import subprocess, venv
    from pathlib import Path
    if _sys.version_info < (3, 11) or _sys.version_info >= (3, 14):
        raise RuntimeError('Use Python 3.11-3.13 (tested on 3.13).')
    bad = _unsatisfied()
    if not bad:
        return
    root = Path.home() / '.cache' / 'hct_tcell_fm_env'
    py = root / 'bin' / 'python'
    if _os.environ.get('HCT_IN_VENV') == '1':
        raise RuntimeError(f'Private environment {root} still lacks: {bad}')
    if not py.exists():
        print(f'[setup] creating private environment {root}', flush=True)
        venv.create(root, with_pip=True)
    probe = subprocess.run([str(py), '-c', 'import sys;sys.exit(0)'], capture_output=True)
    if probe.returncode == 0:
        print(f'[setup] installing/upgrading to tested versions: {bad}', flush=True)
        subprocess.check_call([str(py), '-m', 'pip', 'install', '--quiet', '--upgrade', 'pip'])
        subprocess.check_call([str(py), '-m', 'pip', 'install', '--quiet', *[v[0] for v in _REQUIRED.values()]])
    _os.environ['HCT_IN_VENV'] = '1'
    _os.execv(str(py), [str(py), str(Path(__file__).resolve()), *_sys.argv[1:]])


if __name__ == '__main__':
    _bootstrap_environment()

import os, re, io, csv, gzip, json, time, math, hashlib, random, sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# ===========================================================================
# CONFIGURATION (everything lives here; HCT_PROFILE / HCT_ROOT / HCT_DEVICE env vars are the only overrides)
# ===========================================================================
PROFILES = {
    # Intended for an Apple-silicon laptop (M-series, >=24 GB unified memory). Several hours end to end.
    'laptop': dict(
        vocab_size=4096, d_model=256, heads=8, layers=4, latents=64, dropout=0.10,
        max_cells_per_source=150_000, stats_cells_per_source=30_000,
        pretrain_steps=24_000, batch_size=64, lr=3e-4, warmup_steps=1_000, weight_decay=0.01,
        eval_every=500, ckpt_every=500, patience_evals=14, val_pseudobulks=1_536,
        cv_repeats=10, n_perm=200, n_boot=2_000),
    # CPU-friendly end-to-end check (minutes to ~1 h). Same code paths, smaller everything.
    'smoke': dict(
        vocab_size=3072, d_model=128, heads=4, layers=2, latents=32, dropout=0.10,
        max_cells_per_source=24_000, stats_cells_per_source=10_000,
        pretrain_steps=4_000, batch_size=64, lr=5e-4, warmup_steps=100, weight_decay=0.01,
        eval_every=500, ckpt_every=500, patience_evals=8, val_pseudobulks=512,
        cv_repeats=4, n_perm=60, n_boot=500),
}
PROFILE = os.environ.get('HCT_PROFILE', 'laptop')
if PROFILE not in PROFILES:
    raise SystemExit(f'HCT_PROFILE must be one of {list(PROFILES)}')

CFG = {
    'seed': 20261022,
    'profile': PROFILE,
    **PROFILES[PROFILE],
    # --- pseudobulk (multi-resolution) sampler -------------------------------------------------
    'pb_min_cells': 10, 'pb_max_cells': 400,       # cells aggregated per training profile (log-uniform)
    'mask_frac_range': (0.10, 0.40),               # fraction of observed genes hidden per profile
    'platform_aug': 2.2,                           # max sd of simulated per-platform gene offsets in z-space (input only)
    'noise_aug': 0.15,                             # per-profile gaussian noise in z-space (input only)
    'z_clip': 6.0,
    # --- loss weights ---------------------------------------------------------------------------
    'w_recon': 1.0, 'w_fine': 1.0, 'w_lineage': 0.5, 'w_activation': 0.5,
    # --- vocabulary -----------------------------------------------------------------------------
    'min_detect_frac': 0.02,                       # gene must be detected in >=2% of T/NK cells
    'min_array_platforms': 2,                      # gene must be measurable on >=2 of the 3 array platforms
    # --- clinical evaluation --------------------------------------------------------------------
    'min_vocab_coverage': 0.35,                    # refuse to featurize a sample measuring <35% of the vocabulary
    'min_cohort_samples': 12,
    'head_C_grid': (0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
    'download_retries': 5,
    'allow_changed_sources': False,                # True downgrades a pinned-hash mismatch from error to warning
}
ROOT = Path(os.environ.get('HCT_ROOT', str(Path.home() / 'HCT_TcellFM_Run')))
DATA = ROOT / 'data'
WORK = ROOT / 'work'
OUT = ROOT / 'results'


def ensure_dirs():
    for p in (ROOT, DATA, WORK, OUT):
        p.mkdir(parents=True, exist_ok=True)


_DEVICE = None


def get_device():
    """MPS -> CUDA -> CPU. Never silent: the choice is logged once."""
    global _DEVICE
    if _DEVICE is None:
        pref = os.environ.get('HCT_DEVICE', 'auto').lower()
        mps_ok = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
        if pref in ('auto', 'mps') and mps_ok:
            _DEVICE = torch.device('mps')
        elif pref in ('auto', 'cuda') and torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
        else:
            if pref in ('mps', 'cuda'):
                print(f'[warn] requested device {pref!r} unavailable; using CPU', flush=True)
            _DEVICE = torch.device('cpu')
        print(f'[device] using {_DEVICE} (torch {torch.__version__}); profile={PROFILE}', flush=True)
    return _DEVICE


# ===========================================================================
# UTILITIES: logging, atomic JSON, hashing, provenance-checked resumable downloads
# ===========================================================================
def log(message):
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {message}"
    print(line, flush=True)
    try:
        ensure_dirs()
        with (ROOT / 'run_log.txt').open('a') as f:
            f.write(line + '\n')
    except OSError:
        pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    return str(o)


def _clean_nan(o):
    """JSON cannot hold NaN/inf; store them as null instead of crashing a finished run."""
    if isinstance(o, dict):
        return {str(k): _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(_clean_nan(obj), f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def torch_load(path):
    """Checkpoints hold only tensors/plain containers, so the safe unpickler suffices (no arbitrary code execution)."""
    return torch.load(path, map_location='cpu', weights_only=True)


# --- provenance --------------------------------------------------------------------------------
def _prov_path():
    return DATA / 'provenance.json'


def _prov_key(dest):
    try:
        return str(Path(dest).resolve().relative_to(DATA.resolve()))
    except ValueError:
        return str(dest)


def fetch(url, dest, *, expected_bytes=None, expected_sha256=None, min_bytes=1000, timeout=120):
    """Download `url` to `dest`: resumable, atomic, retries only transient errors.

    Integrity model: a file is accepted when (a) its size/sha256 match the values PINNED in this script
    (when given) and (b) its recorded source URL matches. Pinned values come from the files this pipeline
    was developed against; a mismatch means upstream changed and the run STOPS (unless
    CFG['allow_changed_sources']), instead of silently training on different data.
    """
    ensure_dirs()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prov = load_json(_prov_path(), {})
    key = _prov_key(dest)
    rec = prov.get(key)

    def _verify(path):
        size = path.stat().st_size
        if size < min_bytes:
            return False, f'implausibly small ({size} bytes)'
        if expected_bytes is not None and size != expected_bytes:
            return False, f'size {size} != pinned {expected_bytes}'
        if rec and rec.get('url') == url and rec.get('bytes') == size and rec.get('mtime_ns') == path.stat().st_mtime_ns:
            digest = rec['sha256']                        # unchanged since last verification
        else:
            digest = sha256_file(path)
        if expected_sha256 and digest != expected_sha256:
            return False, f'sha256 {digest[:12]}.. != pinned {expected_sha256[:12]}..'
        return True, digest

    if dest.exists():
        ok, info = _verify(dest)
        if ok:
            prov[key] = {'url': url, 'sha256': info, 'bytes': dest.stat().st_size, 'mtime_ns': dest.stat().st_mtime_ns,
                         'pinned': bool(expected_sha256), 'verified_utc': datetime.now(timezone.utc).isoformat()}
            save_json(_prov_path(), prov)
            log(f'cache verified: {dest.name} ({dest.stat().st_size:,} bytes{", sha256 pinned" if expected_sha256 else ""})')
            return dest
        if CFG['allow_changed_sources'] and 'pinned' in info:
            log(f'WARNING upstream differs from pinned value for {dest.name}: {info}')
            return dest
        log(f'cached file rejected ({info}); re-downloading {dest.name}')
        dest.unlink()

    tmp = dest.with_name(dest.name + '.partial')
    errors = []
    for attempt in range(CFG['download_retries']):
        try:
            start = tmp.stat().st_size if tmp.exists() else 0
            headers = {'User-Agent': 'HCT-TcellFM research pipeline/4.0', 'Accept': '*/*'}
            if start:
                headers['Range'] = f'bytes={start}-'
            with urlopen(Request(url, headers=headers), timeout=timeout) as res:
                status = getattr(res, 'status', 200)
                if start and status != 206:               # server ignored Range: restart cleanly
                    start = 0
                total = int(res.headers.get('Content-Length', 0) or 0) + start
                mode = 'ab' if start else 'wb'
                got = start
                next_report = got + 128 * 1024 * 1024
                with open(tmp, mode) as f:
                    while True:
                        chunk = res.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if got >= next_report:
                            log(f'downloading {dest.name}: {got / 1e9:.2f}' + (f' / {total / 1e9:.2f} GB' if total else ' GB'))
                            next_report = got + 128 * 1024 * 1024
            if total and tmp.stat().st_size != total:
                raise IOError(f'truncated transfer {tmp.stat().st_size} of {total} bytes')
            os.replace(tmp, dest)
            ok, info = _verify(dest)
            if not ok and not CFG['allow_changed_sources']:
                dest.rename(dest.with_name(dest.name + '.rejected'))
                raise RuntimeError(f'downloaded file failed integrity check: {info}')
            digest = info if ok else sha256_file(dest)
            prov[key] = {'url': url, 'sha256': digest, 'bytes': dest.stat().st_size, 'mtime_ns': dest.stat().st_mtime_ns,
                         'pinned': bool(expected_sha256) and ok, 'verified_utc': datetime.now(timezone.utc).isoformat()}
            save_json(_prov_path(), prov)
            log(f'downloaded {dest.name} ({dest.stat().st_size:,} bytes, sha256 {digest[:12]}..)')
            return dest
        except HTTPError as e:
            if e.code in (404, 403, 401, 410):            # permanent: do not hammer the server
                raise RuntimeError(f'{url} -> HTTP {e.code} (permanent)') from e
            errors.append(f'HTTP {e.code}')
        except (URLError, TimeoutError, ConnectionError, IOError) as e:
            errors.append(f'{type(e).__name__}: {str(e)[:100]}')
        log(f'download retry {attempt + 1}/{CFG["download_retries"]} for {dest.name}: {errors[-1]}')
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'download failed for {url}: {errors[-3:]}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    dev = get_device()
    if dev.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.manual_seed(seed)
    elif dev.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# REGISTRY OF PUBLIC SOURCES (pinned by size + sha256; every entry was downloaded and parsed during development)
# ===========================================================================
CXG = 'https://datasets.cellxgene.cziscience.com/'

# Pretraining corpora: public single-cell RNA-seq of HEALTHY (disease == 'normal') human T / NK cells.
# NO transplant, GVHD or other clinical label is used anywhere in pretraining; clinical cohorts are never pretraining data.
PRETRAIN_SOURCES = [
    dict(id='hao_pbmc', url=CXG + '4078abd1-063d-4b35-aa04-324dddf8244e.h5ad', bytes=2606736974,
         sha256='e33c8904af0c77333cb1c206ebd285ca620d2c13cdca6d4e79dd1b3d41880904',
         scheme='cl', heads=('fine', 'lineage'), counts='raw', donor='donor_id',
         cite='Hao et al., Cell 2021, doi:10.1016/j.cell.2021.04.048 (8 donors, 10x 3\' v3, CITE-seq-annotated PBMC)'),
    dict(id='blood65', url=CXG + 'e1ce4d49-a850-4167-a18c-9e9e68bbfe66.h5ad', bytes=2416003445,
         sha256='74566c3fa14613621583efc6c44cfe24ae01689e008cd68c676c8390a3a0387e',
         scheme='cl', heads=('fine', 'lineage'), counts='raw', donor='donor_id',
         cite='Cell 2023, doi:10.1016/j.cell.2023.11.026 (65 healthy donors, four 10x chemistries)'),
    dict(id='tilc', url=CXG + 'e0fc280d-c750-4c13-b7b8-3f01fcd53095.h5ad', bytes=1762461938,
         sha256='6996ccd3e63db4a936e58ff38dcfa2d5570ede7966d26a4afd747e08903ef5d4',
         scheme='cl', heads=('lineage',), counts='raw', donor='donor_id',
         cite='Dominguez Conde et al., Science 2022, doi:10.1126/science.abl5197 (T/ILC from 17 tissues incl. gut, liver, lung)'),
    dict(id='szabo_tissue_t', url=CXG + '75a4447d-e6f4-49a4-98d9-1ccd36a0b622.h5ad', bytes=318979430,
         sha256='ef1df9b11dfb64d3aca23a89465e2e2210b1b3ef10de2aa88ccc80ac5f50c28a',
         scheme='szabo', heads=('activation',), counts='raw', donor='donor_id',
         cite='Szabo et al., Nat Commun 2019, doi:10.1038/s41467-019-12464-3 (resting vs in-vitro-activated T cells, 4 tissues)'),
]

# T-cell state vocabulary for the composition heads (labels come from the public datasets' own annotations).
FINE_CLASSES = ['CD4_naive', 'CD4_memory', 'CD4_CTL', 'Treg', 'CD8_naive', 'CD8_central_memory',
                'CD8_effector_memory', 'MAIT', 'gdT', 'NK']
LINEAGE_CLASSES = ['CD4_conventional', 'Treg', 'CD8', 'unconventional_T', 'NK']
_FINE_TO_LINEAGE = {'CD4_naive': 'CD4_conventional', 'CD4_memory': 'CD4_conventional', 'CD4_CTL': 'CD4_conventional',
                    'Treg': 'Treg', 'CD8_naive': 'CD8', 'CD8_central_memory': 'CD8', 'CD8_effector_memory': 'CD8',
                    'MAIT': 'unconventional_T', 'gdT': 'unconventional_T', 'NK': 'NK'}
K_FINE, K_LIN = len(FINE_CLASSES), len(LINEAGE_CLASSES)


def map_cl_label(label):
    """Cell Ontology label -> (fine class or None, lineage class or None). Unmapped/generic labels are EXCLUDED, never guessed."""
    s = str(label).lower().strip()
    if 'natural killer' in s:
        return 'NK', 'NK'
    if 'regulatory t' in s:
        return 'Treg', 'Treg'
    if 'mucosal-associated invariant' in s:
        return 'MAIT', 'unconventional_T'
    if 'gamma-delta' in s:
        return 'gdT', 'unconventional_T'
    if 'cd4-positive' in s or 'follicular helper' in s:
        if 'cytotoxic' in s:
            return 'CD4_CTL', 'CD4_conventional'
        if 'naive' in s:
            return 'CD4_naive', 'CD4_conventional'
        if s in ('cd4-positive, alpha-beta t cell',):
            return None, None                              # generic label: state unknown
        return 'CD4_memory', 'CD4_conventional'            # central / effector memory, helper, Tfh
    if 'cd8-positive' in s:
        if 'naive' in s:
            return 'CD8_naive', 'CD8'
        if 'central memory' in s:
            return 'CD8_central_memory', 'CD8'
        if 'effector memory' in s:
            return 'CD8_effector_memory', 'CD8'
        if 'memory' in s:
            return None, 'CD8'                              # generic CD8 memory (lineage known, state ambiguous)
        return None, None
    return None, None


# Genes known to matter for T-cell state / alloreactivity; kept in the vocabulary whenever measurable.
MUST_HAVE_GENES = (
    'CD3D CD3E CD3G CD4 CD8A CD8B CD28 ICOS CTLA4 PDCD1 LAG3 TIGIT HAVCR2 TOX TOX2 EOMES TBX21 GATA3 RORC FOXP3 IL2RA IL7R '
    'CCR7 SELL TCF7 LEF1 CD27 CD69 CD44 ITGAE ITGA1 CXCR3 CXCR5 CXCR6 CCR5 CCR6 CCR4 CCR9 ITGB7 SELPLG GZMA GZMB GZMH GZMK GZMM '
    'PRF1 GNLY NKG7 KLRB1 KLRD1 KLRG1 FASLG FAS IFNG TNF IL2 IL4 IL5 IL10 IL13 IL17A IL21 IL22 CSF2 TGFB1 IL2RB STAT1 STAT3 '
    'STAT4 STAT5A IRF4 BATF PRDM1 BCL6 MKI67 TOP2A AURKA AURKB CDK1 MYC HLA-DRA HLA-DRB1 HLA-DQA1 CD38 ENTPD1 ISG15 IFIT1 '
    'IFIT3 MX1 OAS1 IFI44L CXCL13 CXCL9 CXCL10 TNFRSF9 TNFRSF4 TNFRSF18 TNFRSF1B LTB CD74 TRAC TRBC1 TRBC2 TRDC TRGC1 ZBTB16 '
    'NR4A1 NR4A2 JUN FOS DUSP1 TXNIP KLF2 S1PR1 CD5 CD7 CD2 CD6 JAK1 JAK2 JAK3 MTOR IL6ST IL12RB2 IL18R1 IL23R SOCS1 SOCS3').split()

# Genes that would add technical / sex / non-T-cell structure instead of T-cell state (excluded from the vocabulary).
EXCLUDE_GENE_RE = re.compile(
    r'^(MT-|RP[SL]\d|HB[ABDEGMQZ]\d?$|HBA|HBB|LINC\d|MIR\d|SNOR|SCARNA|LOC\d|MALAT1$|NEAT1$|XIST$|TSIX$|RPS4Y|DDX3Y$|KDM5D$|'
    r'UTY$|EIF1AY$|USP9Y$|ZFY$|NLGN4Y$|TXLNGY$|PPBP$|PF4$|ALAS2$)|(-AS\d$|-IT\d$|-OT\d$|\.\d+$|^AC\d{6}|^AL\d{6}|^AP\d{6})')

# ---------------------------------------------------------------------------------------------------------
# CLINICAL COHORT FILES (all public, none used for pretraining). Pinned where upstream is immutable.
# ---------------------------------------------------------------------------------------------------------
GEO_FTP = 'https://ftp.ncbi.nlm.nih.gov/geo'


def geo_series_dir(gse):
    return f'{GEO_FTP}/series/{gse[:-3]}nnn/{gse}'


def geo_matrix_url(gse, name=None):
    return f'{geo_series_dir(gse)}/matrix/{name or gse + "_series_matrix.txt.gz"}'


def gpl_table_url(gpl):
    # NCBI GEO accession viewer; serves the platform's own annotation table as text.
    return f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&view=data&form=text'


ZENODO_GSE295359 = dict(
    url='https://zenodo.org/records/15264188/files/total_t_cells.h5ad?download=1', bytes=1842459115,
    sha256='a609adcce512c1303f65ca1569490304ba60f7a84fffd0f8c7b1d50de7de9310')
PTCY_COUNTS = {   # GEO supplementary count tables (STAR/featureCounts, Ensembl ids); sha256 pinned
    'GSE182610': [('GSE182610_15065R_counts.txt.gz', 2448056, '7eeca2c64f2945dc96c9bc02a032cb71fa5f2d6267e7dee90442e8f0481a861e')],
    'GSE182612': [('GSE182612_17941R_counts.txt.gz', 1364293, '7faf789a5b8f57d3cf1f5ebfe80e2723abdd10621902b79c2dd74589a53c82f0'),
                  ('GSE182612_18740R_counts.txt.gz', 1372574, 'a98db31c2e813656941754f3baeb363ea8b054d00e80bea7b4c3fcb2c9422a32')],
    'GSE182678': [('GSE182678_17618R_counts.txt.gz', 1364329, 'eb848792f803025a5a7b7ad51e24753f009d3a35b59d9d0e9a3d6ed7c0944d5f')],
}


# ===========================================================================
# GENE HARMONIZATION: probe -> HGNC-style symbol (uppercase). Ambiguous probes are EXCLUDED, never guessed.
# ===========================================================================
_BAD_SYMBOLS = {'NA', 'NULL', 'NONE', '---', 'UNKNOWN', 'NAN', ''}
_SYMBOL_RE = re.compile(r'[A-Z][A-Z0-9._-]{1,29}')


def decode_gene_symbol(raw, column_name):
    """One probe's annotation field -> a single uppercase symbol, or None if missing/ambiguous.

    HTA-2.0 `gene_assignment` is 'ACC // SYMBOL // description // ... /// ACC // SYMBOL // ...': the symbol is the
    2nd ' // ' field of each transcript. A probe whose transcripts disagree on the gene is ambiguous -> None.
    Plain symbol columns are split on '///' / ';' and must contain exactly one distinct symbol.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    field = str(raw).strip()
    if field.upper() in _BAD_SYMBOLS:
        return None
    if re.sub('[^a-z]', '', column_name.lower()) == 'geneassignment':
        symbols = set()
        for part in re.split(r'\s*///\s*', field):
            fields = re.split(r'\s*//\s*', part)
            if len(fields) >= 2 and fields[1].strip().upper() not in _BAD_SYMBOLS:
                symbols.add(fields[1].strip().upper())
    else:
        symbols = {p.strip().upper() for p in re.split(r'\s*///\s*|;', field) if p.strip()}
        symbols -= _BAD_SYMBOLS
    if len(symbols) != 1:
        return None
    symbol = next(iter(symbols))
    return symbol if _SYMBOL_RE.fullmatch(symbol) else None


def gpl_probe_map(gpl):
    """GEO platform annotation table -> {probe_id: symbol}. Cached with provenance; refuses truncated/garbled tables."""
    cache = DATA / f'{gpl}_probe_to_symbol.json'
    cached = load_json(cache)
    if cached and cached.get('gpl') == gpl and cached.get('version') == 2:
        return cached['mapping']
    path = fetch(gpl_table_url(gpl), DATA / f'{gpl}.table.txt', min_bytes=10_000, timeout=300)
    with open(path, 'rt', encoding='utf-8', errors='replace', newline='') as h:
        physical = sum(1 for line in h if line.strip() and not line.startswith(('#', '!', '^'))) - 1
    csv.field_size_limit(16 * 1024 * 1024)
    columns, mapping, scanned = None, {}, 0
    with open(path, 'rt', encoding='utf-8', errors='replace', newline='') as h:
        for cells in csv.reader(h, delimiter='\t'):
            if not cells or cells[0].startswith(('#', '!', '^')):
                continue
            if columns is None:
                columns = [c.strip() for c in cells]
                norm = {re.sub('[^a-z]', '', c.lower()): i for i, c in enumerate(columns)}
                sym_i = next((norm[k] for k in ('genesymbol', 'genesymbols', 'symbol', 'genesymbol', 'geneassignment') if k in norm), None)
                id_i = next((i for i, c in enumerate(columns) if c.upper() == 'ID'), None)
                if sym_i is None or id_i is None:
                    raise RuntimeError(f'{gpl}: cannot find ID / gene-symbol columns in {columns[:25]}')
                continue
            scanned += 1
            if sym_i >= len(cells) or id_i >= len(cells):
                continue
            sym = decode_gene_symbol(cells[sym_i], columns[sym_i])
            if sym:
                mapping[cells[id_i].strip()] = sym
    if scanned < 0.98 * physical:
        raise RuntimeError(f'{gpl}: parsed {scanned} of {physical} annotation rows (quote-merging?); refusing to continue')
    if len(mapping) < 2000:
        raise RuntimeError(f'{gpl}: only {len(mapping)} probes mapped to a unique gene symbol')
    save_json(cache, {'gpl': gpl, 'version': 2, 'mapping': mapping, 'symbol_column': columns[sym_i],
                      'rows_scanned': scanned, 'n_mapped': len(mapping), 'n_symbols': len(set(mapping.values()))})
    log(f'{gpl}: {len(mapping):,} of {scanned:,} probes map to a unique symbol ({len(set(mapping.values())):,} genes) via {columns[sym_i]}')
    return mapping


def collapse_probes(values, probe_ids, mapping):
    """samples x probes (NaN allowed) -> samples x genes using the per-gene MEDIAN over its probes."""
    keep = [i for i, p in enumerate(probe_ids) if str(p) in mapping]
    if len(keep) < 1000:
        raise RuntimeError(f'only {len(keep)} probes map to genes')
    genes = np.array([mapping[str(probe_ids[i])] for i in keep])
    frame = pd.DataFrame(np.asarray(values)[:, keep].T, index=genes)
    out = frame.groupby(level=0, sort=True).median()
    return out.T.astype('float32')            # samples x genes (DataFrame)


# --- reading sparse matrices straight from HDF5 (no whole-matrix loads, constant memory) -------------------
def open_h5ad_meta(path):
    """anndata object in backed mode: gives obs/var cheaply; the matrix is read separately in chunks."""
    import anndata as ad
    return ad.read_h5ad(path, backed='r')


def csr_chunk(h5file, loc, start, stop):
    """Rows [start, stop) of an HDF5-stored CSR matrix (`loc` such as 'raw/X', 'X', 'layers/counts') as scipy CSR."""
    import scipy.sparse as sp
    node = h5file[loc]
    enc = node.attrs.get('encoding-type', b'')
    enc = enc.decode() if isinstance(enc, bytes) else enc
    if enc != 'csr_matrix':
        raise RuntimeError(f'{loc}: expected a CSR matrix, found {enc!r}')
    n_cols = int(node.attrs['shape'][1])
    indptr = node['indptr'][start:stop + 1]
    lo, hi = int(indptr[0]), int(indptr[-1])
    return sp.csr_matrix((node['data'][lo:hi], node['indices'][lo:hi], indptr - lo), shape=(stop - start, n_cols))


def source_gene_table(d, counts_loc):
    """(ensembl_ids, symbols) aligned with the columns of the count matrix at `counts_loc`."""
    var = d.raw.var if counts_loc == 'raw' and d.raw is not None else d.var
    ids = np.array(var.index.astype(str))
    col = next((c for c in ('feature_name', 'gene_symbols', 'gene_name', 'symbol') if c in var.columns), None)
    names = np.array(var[col].astype(str)) if col else ids
    return ids, np.char.upper(names.astype(str))


# ===========================================================================
# PRETRAINING CORPUS: public healthy T/NK single cells -> vocabulary -> compact count store
# ===========================================================================
ARRAY_PLATFORMS = ('GPL17586', 'GPL13497', 'GPL3639')       # platforms of the array-based clinical cohorts
SZABO_GROUPS = {'rest_CD4': 0, 'rest_CD8': 1, 'act_CD4': 2, 'act_CD8': 3}


def source_cell_table(src, d):
    """Per-cell: donor, fine/lineage class index (-1 = not supervised), enrichment group, activation (0/1/NaN), valid."""
    obs = d.obs
    n = len(obs)
    donor = obs[src['donor']].astype(str).to_numpy()
    fine = np.full(n, -1, dtype=np.int16)
    lineage = np.full(n, -1, dtype=np.int16)
    act = np.full(n, np.nan, dtype=np.float32)
    if src['scheme'] == 'cl':
        codes, uniques = pd.factorize(obs['cell_type'].astype(str))
        maps = [map_cl_label(u) for u in uniques]
        f_of = np.array([FINE_CLASSES.index(m[0]) if m[0] else -1 for m in maps], dtype=np.int16)
        l_of = np.array([LINEAGE_CLASSES.index(m[1]) if m[1] else -1 for m in maps], dtype=np.int16)
        lineage = l_of[codes]
        if 'fine' in src['heads']:
            fine = f_of[codes]
            group, valid = fine.copy(), fine >= 0
        else:
            group, valid = lineage.copy(), lineage >= 0
    elif src['scheme'] == 'szabo':
        status = obs['status'].astype(str).to_numpy()
        group = np.array([SZABO_GROUPS.get(s, -1) for s in status], dtype=np.int16)
        valid = group >= 0
        act = np.where(np.char.startswith(status.astype(str), 'act'), 1.0, 0.0).astype(np.float32)
        act[~valid] = np.nan
    else:
        raise ValueError(src['scheme'])
    return pd.DataFrame({'donor': donor, 'fine': fine, 'lineage': lineage, 'group': group.astype(np.int16),
                         'act': act, 'valid': valid})


def _row_windows(n_rows, chunk):
    return [(s, min(s + chunk, n_rows)) for s in range(0, n_rows, chunk)]


def scan_source(src, path, budget, rng):
    """Pass 1: per-gene detection and across-group variability of mean log-CP10k, from randomly chosen row windows."""
    import h5py
    import scipy.sparse as sp
    d = open_h5ad_meta(path)
    try:
        ens, sym = source_gene_table(d, src['counts'])
        tbl = source_cell_table(src, d)
    finally:
        if getattr(d, 'file', None) is not None:
            d.file.close()
    loc = 'raw/X' if src['counts'] == 'raw' else src['counts']
    chunk = 5000
    wins = _row_windows(len(tbl), chunk)
    valid_frac = max(tbl.valid.mean(), 1e-3)
    n_windows = int(min(len(wins), max(2, math.ceil(budget / (chunk * valid_frac)))))
    chosen = sorted(rng.choice(len(wins), size=n_windows, replace=False))
    keys = {}
    G = len(sym)
    det = np.zeros(G, dtype=np.float64)
    n_valid = 0
    acc_sum = {}
    acc_cnt = {}
    with h5py.File(path, 'r') as f:
        for wi in chosen:
            s, e = wins[wi]
            sub = tbl.iloc[s:e]
            rows = np.where(sub.valid.to_numpy())[0]
            if len(rows) == 0:
                continue
            X = csr_chunk(f, loc, s, e)[rows]
            lib = np.asarray(X.sum(1)).ravel()
            ok = lib > 0
            X, rows, lib = X[ok], rows[ok], lib[ok]
            det += np.asarray((X > 0).sum(0)).ravel()
            n_valid += X.shape[0]
            L = sp.diags(1e4 / lib) @ X
            L.data = np.log1p(L.data)
            kd = sub.iloc[rows]
            key = list(zip(kd.donor.to_numpy(), kd.group.to_numpy().tolist()))
            gid = np.array([keys.setdefault(k, len(keys)) for k in key])
            uniq, inv = np.unique(gid, return_inverse=True)
            ind = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
            S = (ind @ L).toarray()
            for j, k_ in enumerate(uniq):
                acc_sum[k_] = acc_sum.get(k_, 0) + S[j]
                acc_cnt[k_] = acc_cnt.get(k_, 0) + int((inv == j).sum())
    means = [acc_sum[k] / acc_cnt[k] for k in acc_sum if acc_cnt[k] >= 30]
    if len(means) < 3:
        raise RuntimeError(f'{src["id"]}: only {len(means)} donor x group cells types with >=30 cells in the scanned windows')
    return dict(ens=ens, sym=sym, det=det / max(n_valid, 1), between=np.var(np.stack(means), axis=0), n_valid=n_valid,
                n_groups=len(means))


def build_vocabulary():
    """Select V genes: expressed in T/NK cells, variable across T-cell states/donors, measurable on the clinical array
    platforms, never sex/technical genes. Uses ONLY public healthy single-cell data + platform annotation (gene
    identifiers) -- no clinical expression values or labels."""
    ensure_dirs()
    key = stable_hash({'src': [(s['id'], s['sha256']) for s in PRETRAIN_SOURCES], 'V': CFG['vocab_size'],
                       'stats': CFG['stats_cells_per_source'], 'det': CFG['min_detect_frac'],
                       'arr': CFG['min_array_platforms'], 'seed': CFG['seed'], 'v': 4,
                       'must': MUST_HAVE_GENES, 'excl': EXCLUDE_GENE_RE.pattern})
    cache = WORK / 'vocab.json'
    cached = load_json(cache)
    if cached and cached.get('key') == key:
        return cached
    rng = np.random.default_rng(CFG['seed'])
    scans = []
    for src in PRETRAIN_SOURCES:
        path = fetch(src['url'], DATA / f"{src['id']}.h5ad", expected_bytes=src['bytes'], expected_sha256=src['sha256'])
        t0 = time.time()
        sc = scan_source(src, path, CFG['stats_cells_per_source'], rng)
        log(f"vocab scan {src['id']}: {sc['n_valid']:,} T/NK cells, {sc['n_groups']} donor-x-state groups, {time.time() - t0:.0f}s")
        scans.append(sc)
    # align on symbols present in EVERY source
    index = []
    for sc in scans:
        first = {}
        for i, s in enumerate(sc['sym']):
            first.setdefault(s, i)
        index.append(first)
    shared = set(index[0])
    for ix in index[1:]:
        shared &= set(ix)
    shared = sorted(shared)
    det = np.mean([[sc['det'][ix[s]] for s in shared] for sc, ix in zip(scans, index)], axis=0)
    between = np.mean([[sc['between'][ix[s]] for s in shared] for sc, ix in zip(scans, index)], axis=0)
    arrays = [set(gpl_probe_map(g).values()) for g in ARRAY_PLATFORMS]
    n_arr = np.array([sum(s in a for a in arrays) for s in shared])
    shared_arr = np.array(shared)
    excluded = np.array([bool(EXCLUDE_GENE_RE.search(s)) for s in shared])
    eligible = (det >= CFG['min_detect_frac']) & (n_arr >= CFG['min_array_platforms']) & ~excluded
    must = np.isin(shared_arr, MUST_HAVE_GENES) & (n_arr >= 1) & ~excluded
    V = CFG['vocab_size']
    # half the score is variability across T-cell states/donors, half is how consistently the gene is expressed in T/NK cells:
    # a vocabulary chosen by state-variability alone discarded genes that matter for donor- and disease-level signals.
    score = 0.5 * pd.Series(between).rank(pct=True).to_numpy() + 0.5 * pd.Series(det).rank(pct=True).to_numpy()
    order = np.argsort(-score)
    chosen = [i for i in order if must[i]]
    in_chosen = set(chosen)
    for i in order:
        if len(chosen) >= V:
            break
        if eligible[i] and i not in in_chosen:
            chosen.append(i)
            in_chosen.add(i)
    chosen = chosen[:V]
    genes = sorted(shared_arr[chosen].tolist())
    if len(genes) < 512:
        raise RuntimeError(f'vocabulary too small ({len(genes)}); check sources')
    ens2sym = {}
    for sc in scans:
        for e, s in zip(sc['ens'], sc['sym']):
            ens2sym.setdefault(str(e), str(s))
    out = dict(key=key, genes=genes, n_shared=len(shared), n_eligible=int(eligible.sum()),
               n_must_present=int(must.sum()), array_coverage={g: float(np.mean([x in a for x in genes]))
                                                              for g, a in zip(ARRAY_PLATFORMS, arrays)})
    save_json(cache, out)
    save_json(WORK / 'ens2sym.json', ens2sym)
    log(f"vocabulary: {len(genes)} genes (of {len(shared):,} shared, {int(eligible.sum()):,} eligible); "
        f"array coverage {', '.join(f'{k} {v:.0%}' for k, v in out['array_coverage'].items())}")
    return out


def load_ens2sym():
    m = load_json(WORK / 'ens2sym.json')
    if m is None:
        build_vocabulary()
        m = load_json(WORK / 'ens2sym.json')
    return m


def _group_aware_sample(tbl, cap, rng):
    """Cap a source at `cap` cells, keeping rare donor x state groups (equalise toward sqrt-sized groups)."""
    idx = np.where(tbl.valid.to_numpy())[0]
    if len(idx) <= cap:
        return idx
    keys = pd.factorize(tbl.donor.iloc[idx].astype(str) + '|' + tbl.group.iloc[idx].astype(str))[0]
    sizes = np.bincount(keys)
    w = 1.0 / np.sqrt(sizes[keys])
    w = w / w.sum()
    return np.sort(rng.choice(idx, size=cap, replace=False, p=w))


def build_source_store(src, vocab_genes):
    """Pass 2: counts of the vocabulary genes (uint16) + whole-library size for the selected cells of one source."""
    import h5py
    key = stable_hash({'sha': src['sha256'], 'genes': vocab_genes, 'cap': CFG['max_cells_per_source'], 'seed': CFG['seed'], 'v': 4})
    out = WORK / 'corpus' / f"{src['id']}.npz"
    meta = WORK / 'corpus' / f"{src['id']}.json"
    if out.exists() and (load_json(meta) or {}).get('key') == key:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    path = fetch(src['url'], DATA / f"{src['id']}.h5ad", expected_bytes=src['bytes'], expected_sha256=src['sha256'])
    d = open_h5ad_meta(path)
    try:
        ens, sym = source_gene_table(d, src['counts'])
        tbl = source_cell_table(src, d)
    finally:
        if getattr(d, 'file', None) is not None:
            d.file.close()
    first = {}
    for i, s in enumerate(sym):
        first.setdefault(s, i)
    missing = [g for g in vocab_genes if g not in first]
    if missing:
        raise RuntimeError(f"{src['id']}: {len(missing)} vocabulary genes absent from source, e.g. {missing[:5]}")
    cols = np.array([first[g] for g in vocab_genes])
    rng = np.random.default_rng(CFG['seed'] + 17)
    sel = _group_aware_sample(tbl, CFG['max_cells_per_source'], rng)
    loc = 'raw/X' if src['counts'] == 'raw' else src['counts']
    V = len(vocab_genes)
    counts = np.zeros((len(sel), V), dtype=np.uint16)
    lib = np.zeros(len(sel), dtype=np.float32)
    chunk = 10_000
    pos = 0
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        for s, e in _row_windows(len(tbl), chunk):
            lo, hi = np.searchsorted(sel, [s, e])
            if hi == lo:
                continue
            X = csr_chunk(f, loc, s, e)
            rows = sel[lo:hi] - s
            Xr = X[rows]
            lib[lo:hi] = np.asarray(Xr.sum(1)).ravel()
            counts[lo:hi] = np.minimum(Xr[:, cols].toarray(), 65535).astype(np.uint16)
            pos = hi
            if (s // chunk) % 5 == 0:
                log(f"  {src['id']}: read rows {e:,}/{len(tbl):,} ({pos:,}/{len(sel):,} cells kept, {time.time() - t0:.0f}s)")
    good = lib > 0
    sub = tbl.iloc[sel].reset_index(drop=True)
    sub = sub[good]
    donors, donor_codes = np.unique(sub.donor.to_numpy(), return_inverse=True)
    np.savez(out, counts=counts[good], lib=lib[good], donor=donor_codes.astype(np.int32), donor_names=donors.astype(str),
             fine=sub.fine.to_numpy(), lineage=sub.lineage.to_numpy(), group=sub.group.to_numpy(), act=sub.act.to_numpy())
    save_json(meta, dict(key=key, n_cells=int(good.sum()), n_donors=len(donors), id=src['id']))
    log(f"corpus store {src['id']}: {int(good.sum()):,} cells x {V} genes, {len(donors)} donors")
    return out


def build_corpus_stores():
    vocab = build_vocabulary()
    return vocab, [build_source_store(src, vocab['genes']) for src in PRETRAIN_SOURCES]


# ===========================================================================
# THE T-CELL BASELINE: platform-robust normalisation + multi-resolution pseudobulk sampler
# ===========================================================================
# Every profile -- a 10x pseudobulk, a bulk RNA-seq library or a microarray -- is mapped into ONE space:
#   1. log-scale expression of the vocabulary genes (log1p CPM for counts; native log2 for arrays),
#   2. within-sample quantile normalisation to the healthy-T-cell reference distribution (removes platform scale),
#   3. per-gene z-score against the healthy-T-cell baseline (mean/sd of training pseudobulks).
# So a value of 0 means "typical healthy T cell"; clinical samples enter as DEVIATIONS from that baseline.
GRID = 1001


def _rankdata_rows(x):
    from scipy.stats import rankdata
    return rankdata(x, axis=1, method='average', nan_policy='omit')


def quantile_normalize(x, observed, ref_q):
    """Within-sample map of the observed values onto the reference quantile function (ties share the mean rank)."""
    x = np.where(observed, x, np.nan).astype(np.float64)
    ranks = _rankdata_rows(x)
    n = observed.sum(1, keepdims=True).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        q = (ranks - 0.5) / n
    out = np.interp(q, np.linspace(0.0, 1.0, GRID), ref_q)
    out[~observed] = np.nan
    return out.astype(np.float32)


def to_model_space(x, observed, baseline, clip=None):
    """x: samples x V log-scale values (NaN allowed where unmeasured) -> z (0 where unmeasured)."""
    clip = CFG['z_clip'] if clip is None else clip
    observed = np.asarray(observed, dtype=bool) & np.isfinite(x)
    xq = quantile_normalize(np.asarray(x, dtype=np.float64), observed, baseline['ref_q'])
    z = (xq - baseline['mu']) / baseline['sd']
    z = np.clip(np.nan_to_num(z, nan=0.0), -clip, clip).astype(np.float32)
    z[~observed] = 0.0
    return z, observed


def log_cpm(counts, lib, scale=1e6):
    return np.log1p(counts.astype(np.float64) / np.maximum(lib[:, None].astype(np.float64), 1.0) * scale).astype(np.float32)


class Corpus:
    """In-memory view of the cached single-cell stores with donor-level train/validation splits."""

    def __init__(self, vocab_genes):
        self.genes = list(vocab_genes)
        self.V = len(self.genes)
        self.src = []
        by_id = {s['id']: s for s in PRETRAIN_SOURCES}
        rng = np.random.default_rng(CFG['seed'] + 5)
        for path in sorted((WORK / 'corpus').glob('*.npz')):
            sid = path.stem
            if sid not in by_id:
                continue
            z = np.load(path, allow_pickle=False)
            s = dict(id=sid, cfg=by_id[sid], counts=z['counts'], lib=z['lib'], donor=z['donor'], fine=z['fine'],
                     lineage=z['lineage'], group=z['group'], act=z['act'], donor_names=z['donor_names'])
            nd = len(s['donor_names'])
            order = rng.permutation(nd)
            n_val = max(1, int(round(0.2 * nd))) if nd > 1 else 0
            s['val_donors'] = sorted(order[:n_val].tolist())
            s['train_donors'] = sorted(order[n_val:].tolist())
            s['cells'] = {}
            for dn in range(nd):
                idx = np.where(s['donor'] == dn)[0]
                groups = {}
                for g in np.unique(s['group'][idx]):
                    gi = idx[s['group'][idx] == g]
                    if len(gi) >= 5:
                        groups[int(g)] = gi
                if groups:
                    s['cells'][dn] = dict(all=np.concatenate(list(groups.values())), groups=groups)
            s['train_donors'] = [d for d in s['train_donors'] if d in s['cells']]
            s['val_donors'] = [d for d in s['val_donors'] if d in s['cells']]
            self.src.append(s)
        if not self.src:
            raise RuntimeError('no corpus stores found; run the corpus build first')
        log('corpus: ' + '; '.join(f"{s['id']}={len(s['donor']):,} cells/{len(s['donor_names'])} donors "
                                   f"(val donors {len(s['val_donors'])})" for s in self.src))

    def _pick_source(self, split, rng):
        ok = [i for i, s in enumerate(self.src) if s[f'{split}_donors']]
        w = np.array([math.sqrt(len(self.src[i]['donor'])) for i in ok])
        return ok[int(rng.choice(len(ok), p=w / w.sum()))]

    def _recipe(self, s, donor, m, rng):
        info = s['cells'][donor]
        groups = list(info['groups'].keys())
        kind = rng.choice(['natural', 'enriched', 'dirichlet'], p=[0.35, 0.45, 0.20])
        if kind == 'natural' or len(groups) == 1:
            pool = info['all']
            return rng.choice(pool, size=m, replace=len(pool) < m)
        if kind == 'enriched':      # sorted-population-like: 1-3 states dominate
            k = int(rng.integers(1, min(3, len(groups)) + 1))
            chosen = list(rng.choice(groups, size=k, replace=False))
            w = rng.dirichlet(np.ones(k))
        else:
            chosen = groups
            w = rng.dirichlet(0.5 * np.ones(len(groups)))
        n_each = rng.multinomial(m, w)
        parts = [rng.choice(info['groups'][g], size=n, replace=len(info['groups'][g]) < n)
                 for g, n in zip(chosen, n_each) if n > 0]
        return np.concatenate(parts)

    def draw(self, split, rng, batch_size):
        """Pseudobulk profiles with composition / activation targets. Returns numpy dict (log-CPM, not yet normalised)."""
        B, V = batch_size, self.V
        lc = np.zeros((B, V), dtype=np.float32)
        m_arr = np.zeros(B, dtype=np.float32)
        fine_t = np.zeros((B, K_FINE), dtype=np.float32)
        lin_t = np.zeros((B, K_LIN), dtype=np.float32)
        act_t = np.zeros(B, dtype=np.float32)
        valid = np.zeros((B, 3), dtype=bool)
        sid = np.zeros(B, dtype=np.int64)
        per_src = {}
        for b in range(B):
            si = self._pick_source(split, rng)
            s = self.src[si]
            donors = s[f'{split}_donors']
            donor = donors[int(rng.integers(len(donors)))]
            m = int(round(math.exp(rng.uniform(math.log(CFG['pb_min_cells']), math.log(CFG['pb_max_cells'])))))
            idx = self._recipe(s, donor, m, rng)
            per_src.setdefault(si, []).append((b, idx))
            heads = s['cfg']['heads']
            m_arr[b], sid[b] = len(idx), si
            if 'fine' in heads:
                fine_t[b] = np.bincount(s['fine'][idx], minlength=K_FINE)[:K_FINE] / len(idx)
                valid[b, 0] = True
            if 'lineage' in heads:
                li = s['lineage'][idx]
                ok = li >= 0
                if ok.any():
                    lin_t[b] = np.bincount(li[ok], minlength=K_LIN)[:K_LIN] / ok.sum()
                    valid[b, 1] = True
            if 'activation' in heads:
                act_t[b] = float(np.nanmean(s['act'][idx]))
                valid[b, 2] = True
        for si, items in per_src.items():
            s = self.src[si]
            rows = np.concatenate([i for _, i in items])
            off = np.cumsum([0] + [len(i) for _, i in items[:-1]])
            sums = np.add.reduceat(s['counts'][rows].astype(np.float32), off, axis=0)
            libs = np.add.reduceat(s['lib'][rows].astype(np.float64), off)
            out_rows = [b for b, _ in items]
            lc[out_rows] = log_cpm(sums, libs)
        return dict(x=lc, m=m_arr, fine=fine_t, lineage=lin_t, act=act_t, valid=valid, source=sid)


def fit_baseline(corpus, n_profiles=3000, seed=0):
    """Healthy-T-cell reference from TRAINING donors only: reference quantile function + per-gene mean/sd."""
    rng = np.random.default_rng(seed)
    xs = []
    done = 0
    while done < n_profiles:
        b = corpus.draw('train', rng, min(250, n_profiles - done))
        xs.append(b['x'])
        done += len(b['x'])
    x = np.concatenate(xs)
    V = x.shape[1]
    ref_sorted = np.sort(x, axis=1).mean(0)
    ref_q = np.interp(np.linspace(0, 1, GRID), (np.arange(V) + 0.5) / V, ref_sorted).astype(np.float64)
    obs = np.ones(x.shape, dtype=bool)
    xq = quantile_normalize(x, obs, ref_q)
    mu = xq.mean(0).astype(np.float64)
    sd = xq.std(0).astype(np.float64)
    sd = np.maximum(sd, 0.1 * np.median(sd))
    raw_sd = x.std(0).astype(np.float64)
    return dict(ref_q=ref_q, mu=mu, sd=sd, raw_sd=np.maximum(raw_sd, 1e-3), n_profiles=int(n_profiles))


def augment_coverage(x, observed, baseline, rng):
    """Before normalisation: simulate platforms that lack part of the vocabulary (shared within a 'platform' group of
    profiles) plus a small log-scale gene offset. Returns perturbed log-scale x and the observed mask."""
    B, V = x.shape
    x = x.copy()
    observed = observed.copy()
    n_groups = max(1, B // 16)
    gid = rng.integers(0, n_groups, size=B)
    for g in range(n_groups):
        rows = np.where(gid == g)[0]
        if len(rows) == 0:
            continue
        x[rows] += (rng.normal(0.0, 1.0, size=V) * baseline['raw_sd'] * 0.15).astype(np.float32)
        if rng.random() < 0.5:
            observed[rows] &= (rng.random(V) > rng.uniform(0.05, 0.55))[None, :]
    return x, observed


def augment_z(z, observed, rng, strength, noise):
    """After normalisation: shared per-'platform' gene offsets of up to `strength` baseline-sd (real post-transplant /
    other-platform cohorts sit 1.3-2.1 mean |z| from the healthy baseline vs 0.8 for held-out healthy donors) and
    per-profile noise. Applied to the INPUT only; reconstruction targets stay clean."""
    B, V = z.shape
    z = z.copy()
    n_groups = max(1, B // 16)
    gid = rng.integers(0, n_groups, size=B)
    for g in range(n_groups):
        rows = np.where(gid == g)[0]
        if len(rows):
            z[rows] += (rng.normal(0.0, 1.0, size=V) * strength * rng.uniform(0.0, 1.0)).astype(np.float32)
    z += rng.normal(0.0, noise, size=z.shape).astype(np.float32)
    z = np.clip(z, -CFG['z_clip'], CFG['z_clip'])
    z[~observed] = 0.0
    return z


# ===========================================================================
# MODEL: latent cross-attention (Perceiver-style) encoder over gene tokens
# ===========================================================================
STATE_OBSERVED, STATE_MASKED, STATE_MISSING = 0, 1, 2
N_RBF = 16


class TCellFM(nn.Module):
    """Gene tokens are read by L learned latents via cross-attention (cost O(V*L), not O(V^2)), mixed by a small
    transformer, and read back out by gene-specific queries.

    Efficiency: a token's key/value is  K(gene identity) + K(value) + K(state)  -- a linear projection of an additive
    embedding -- so the gene part is computed once per step and the value part is a 16->d linear map of a radial-basis
    encoding of the value. This avoids a d x d projection over every (profile x gene) token. The decoder is a gene-specific
    attention read-out (no per-token feed-forward).

    * Hidden (masked) and unmeasured (missing) genes keep their IDENTITY but their VALUE is zeroed BEFORE it is encoded, so
      the network cannot see a hidden value -- leakage is impossible by construction (asserted in the self-tests).
    * `log10_cells` conditions the latents on the effective number of cells behind a profile (bulk samples use the maximum).
    * Heads (read the pooled latent): fine T-cell-state composition, lineage composition, activation level.
    """

    def __init__(self, n_genes, d_model=256, heads=8, layers=4, latents=64, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.n_genes, self.d, self.h = n_genes, d_model, heads
        self.hp = dict(n_genes=n_genes, d_model=d_model, heads=heads, layers=layers, latents=latents, dropout=dropout)
        self.dropout = dropout
        d = d_model
        self.register_buffer('rbf_centers', torch.linspace(-CFG['z_clip'], CFG['z_clip'], N_RBF), persistent=False)
        self.rbf_width = 2 * CFG['z_clip'] / (N_RBF - 1)
        self.gene_emb = nn.Embedding(n_genes, d)
        self.k_gene, self.v_gene = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.k_val, self.v_val = nn.Linear(N_RBF, d, bias=False), nn.Linear(N_RBF, d, bias=False)
        self.k_state, self.v_state = nn.Embedding(3, d), nn.Embedding(3, d)
        self.latents = nn.Parameter(torch.randn(latents, d) * 0.02)
        self.res_proj = nn.Linear(1, d)
        self.q_norm, self.q_proj, self.o_proj = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, d)
        self.cross_ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.dec_q_norm = nn.LayerNorm(d)
        self.dec_q, self.dec_k, self.dec_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.readout = nn.Parameter(torch.randn(n_genes, d) * d ** -0.5)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.fine_head = nn.Linear(d, K_FINE)
        self.lineage_head = nn.Linear(d, K_LIN)
        self.act_head = nn.Linear(d, 1)

    def _split(self, x):                       # B,N,d -> B,H,N,dh
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d // self.h).transpose(1, 2)

    def forward(self, z, state, log10_cells):
        B, V = z.shape
        if V != self.n_genes:
            raise ValueError(f'expected {self.n_genes} genes, got {V}')
        value = torch.where(state == STATE_OBSERVED, z, torch.zeros_like(z))      # hidden values never enter the network
        rbf = torch.exp(-(((value.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2))
        g = self.gene_emb.weight
        k = self.k_gene(g)[None] + self.k_val(rbf) + self.k_state(state)
        v = self.v_gene(g)[None] + self.v_val(rbf) + self.v_state(state)
        lat = self.latents[None].expand(B, -1, -1) + self.res_proj(log10_cells.view(B, 1, 1))
        q = self.q_proj(self.q_norm(lat))
        att = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v))   # no attention dropout over V tokens
        lat = lat + self.o_proj(att.transpose(1, 2).reshape(B, -1, self.d))
        lat = lat + self.cross_ff(lat)
        lat = self.out_norm(self.encoder(lat))
        pooled = lat.mean(dim=1)
        dq = self.dec_q(self.dec_q_norm(g)).view(1, V, self.h, self.d // self.h).transpose(1, 2).expand(B, -1, -1, -1).contiguous()
        out = F.scaled_dot_product_attention(dq, self._split(self.dec_k(lat)), self._split(self.dec_v(lat)))
        out = out.transpose(1, 2).reshape(B, V, self.d)
        recon = (out * self.readout).sum(-1) + self.gene_bias
        return dict(pooled=pooled, recon=recon, fine=self.fine_head(pooled), lineage=self.lineage_head(pooled),
                    activation=self.act_head(pooled).squeeze(-1))

    @property
    def config(self):
        return dict(self.hp)


def build_model(n_genes, hp=None):
    c = hp or CFG
    return TCellFM(n_genes, c['d_model'], c['heads'], c['layers'], c['latents'], c['dropout'])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def embed_profiles(model, z, observed, log10_cells=None, batch_size=128, device=None):
    """Frozen-encoder features for clinical samples. Returns dict of numpy arrays (no masking applied)."""
    device = device or get_device()
    model.eval().to(device)
    z = np.asarray(z, dtype=np.float32)
    state = np.where(observed, STATE_OBSERVED, STATE_MISSING).astype(np.int64)
    logc = np.full(len(z), math.log10(CFG['pb_max_cells']) if log10_cells is None else log10_cells, dtype=np.float32)
    outs = {k: [] for k in ('pooled', 'fine', 'lineage', 'activation')}
    for i in range(0, len(z), batch_size):
        o = model(torch.from_numpy(z[i:i + batch_size]).to(device), torch.from_numpy(state[i:i + batch_size]).to(device),
                  torch.from_numpy(logc[i:i + batch_size]).to(device))
        for k in outs:
            outs[k].append(o[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


@torch.no_grad()
def deviation_scores(model, z, observed, n_rounds=4, frac=0.25, seed=0, batch_size=128, device=None):
    """How far a profile sits from the healthy-T-cell manifold: mean error when hidden genes are re-predicted from the
    visible ones (higher = less typical of healthy T cells). Fixed seeds -> deterministic."""
    device = device or get_device()
    model.eval().to(device)
    z = np.asarray(z, dtype=np.float32)
    obs = np.asarray(observed, dtype=bool)
    scores = np.zeros(len(z), dtype=np.float64)
    for r in range(n_rounds):
        gen = np.random.default_rng([seed, r])
        mask = (gen.random(z.shape) < frac) & obs
        state = np.where(~obs, STATE_MISSING, np.where(mask, STATE_MASKED, STATE_OBSERVED)).astype(np.int64)
        for i in range(0, len(z), batch_size):
            sl = slice(i, i + batch_size)
            zt = torch.from_numpy(z[sl]).to(device)
            logc = torch.full((zt.shape[0],), math.log10(CFG['pb_max_cells']), device=device)
            o = model(zt, torch.from_numpy(state[sl]).to(device), logc)
            mt = torch.from_numpy(mask[sl]).to(device)
            err = ((o['recon'] - zt) ** 2 * mt).sum(1) / mt.sum(1).clamp(min=1)
            scores[sl] += err.float().cpu().numpy() / n_rounds
    return scores


@torch.no_grad()
def residual_profiles(model, z, observed, n_partitions=2, n_folds=4, seed=0, batch_size=64, device=None):
    """Gene-by-gene deviation from the healthy-T-cell expectation: z - E[z | all other genes], where the expectation is the
    model's prediction of a gene while it is hidden. Genes are split into `n_folds` hidden-sets per random partition, so every
    observed gene is predicted `n_partitions` times from the others. Removes variance explained by healthy cell-state
    structure and keeps what that structure does not predict. Unmeasured genes get 0."""
    device = device or get_device()
    model.eval().to(device)
    z = np.asarray(z, dtype=np.float32)
    obs = np.asarray(observed, dtype=bool)
    n, V = z.shape
    pred_sum = np.zeros((n, V), dtype=np.float64)
    pred_cnt = np.zeros((n, V), dtype=np.float64)
    for part in range(n_partitions):
        rng = np.random.default_rng([seed, part])
        fold_of = rng.integers(0, n_folds, size=V)
        for k in range(n_folds):
            hidden = (fold_of == k)[None, :] & obs
            state = np.where(~obs, STATE_MISSING, np.where(hidden, STATE_MASKED, STATE_OBSERVED)).astype(np.int64)
            for i in range(0, n, batch_size):
                zt = torch.from_numpy(z[i:i + batch_size]).to(device)
                logc = torch.full((zt.shape[0],), math.log10(CFG['pb_max_cells']), device=device)
                o = model(zt, torch.from_numpy(state[i:i + batch_size]).to(device), logc)
                pred_sum[i:i + batch_size] += np.where(hidden[i:i + batch_size], o['recon'].float().cpu().numpy(), 0.0)
                pred_cnt[i:i + batch_size] += hidden[i:i + batch_size]
    pred = np.where(pred_cnt > 0, pred_sum / np.maximum(pred_cnt, 1), 0.0)
    return np.where(obs, z - pred, 0.0).astype(np.float32)


# ===========================================================================
# PRETRAINING: three objectives on public healthy-T-cell pseudobulks
#   P1 masked-gene reconstruction     (self-supervised; learns gene-gene structure of T-cell states)
#   P2 T-cell-state composition       (fine states + lineage; deconvolution learned from public annotations)
#   P3 activation level               (resting vs in-vitro-activated T cells; a TCR-activation axis)
# ===========================================================================
def make_batch(corpus, baseline, split, step, batch_size, augment, seed=None, fixed_frac=None):
    rng = np.random.default_rng([CFG['seed'] if seed is None else seed, int(step), 0 if split == 'train' else 1])
    raw = corpus.draw(split, rng, batch_size)
    x, B, V = raw['x'], batch_size, corpus.V
    full = np.ones((B, V), dtype=bool)
    z_clean, _ = to_model_space(x, full, baseline)
    if augment:
        xa, oa = augment_coverage(x, full, baseline, rng)
        z_in, obs = to_model_space(xa, oa, baseline)
        z_in = augment_z(z_in, obs, rng, CFG['platform_aug'], CFG['noise_aug'])
    else:
        z_in, obs = z_clean.copy(), full
    lo, hi = CFG['mask_frac_range']
    frac = np.full(B, fixed_frac) if fixed_frac is not None else rng.uniform(lo, hi, size=B)
    mask = (rng.random((B, V)) < frac[:, None]) & obs
    none = ~mask.any(1)
    if none.any():
        first = np.argmax(obs[none], axis=1)
        rows = np.where(none)[0]
        mask[rows, first] = obs[rows, first]
    state = np.where(~obs, STATE_MISSING, np.where(mask, STATE_MASKED, STATE_OBSERVED)).astype(np.int64)
    return dict(z_in=torch.from_numpy(z_in), z=torch.from_numpy(z_clean), state=torch.from_numpy(state),
                logc=torch.from_numpy(np.log10(raw['m']).astype(np.float32)), fine=torch.from_numpy(raw['fine']),
                lineage=torch.from_numpy(raw['lineage']), act=torch.from_numpy(raw['act']),
                valid=torch.from_numpy(raw['valid']))


def soft_ce(logits, target, valid):
    ce = -(target * F.log_softmax(logits, dim=-1)).sum(-1)
    return (ce * valid).sum() / valid.sum().clamp(min=1)


def pretrain_loss(out, b):
    masked = (b['state'] == STATE_MASKED).float()
    l_rec = (((out['recon'] - b['z']) ** 2) * masked).sum() / masked.sum().clamp(min=1)
    l_fine = soft_ce(out['fine'], b['fine'], b['valid'][:, 0].float())
    l_lin = soft_ce(out['lineage'], b['lineage'], b['valid'][:, 1].float())
    va = b['valid'][:, 2].float()
    l_act = (F.binary_cross_entropy_with_logits(out['activation'], b['act'], reduction='none') * va).sum() / va.sum().clamp(min=1)
    total = CFG['w_recon'] * l_rec + CFG['w_fine'] * l_fine + CFG['w_lineage'] * l_lin + CFG['w_activation'] * l_act
    return total, dict(recon=l_rec.detach(), fine=l_fine.detach(), lineage=l_lin.detach(), activation=l_act.detach())


def to_device(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


class Prefetcher:
    """Builds batches for steps [start, stop) in a worker thread (each batch depends only on (seed, step) -> reproducible)."""

    def __init__(self, fn, start, stop, depth=3):
        import queue, threading
        self.q = queue.Queue(maxsize=depth)
        self.fn, self.start, self.stop = fn, start, stop

        def run():
            try:
                for s in range(start, stop):
                    self.q.put((s, fn(s)))
            except BaseException as e:      # surface worker errors in the training thread
                self.q.put((None, e))
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def __iter__(self):
        for _ in range(self.start, self.stop):
            s, b = self.q.get()
            if s is None:
                raise b
            yield s, b


def js_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p /= p.sum(-1, keepdims=True); q /= q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)


@torch.no_grad()
def evaluate_pretrain(model, val, dev, batch_size=256):
    model.eval()
    n = len(val['z'])
    acc = dict(rec_num=0.0, rec_den=0.0, fine_n=0.0, fine_d=0.0, lin_n=0.0, lin_d=0.0, act_n=0.0, act_d=0.0)
    probs = {'fine': [], 'lineage': [], 'act': []}
    for i in range(0, n, batch_size):
        b = to_device({k: v[i:i + batch_size] for k, v in val.items()}, dev)
        o = model(b['z_in'], b['state'], b['logc'])
        masked = (b['state'] == STATE_MASKED).float()
        acc['rec_num'] += float((((o['recon'] - b['z']) ** 2) * masked).sum()); acc['rec_den'] += float(masked.sum())
        for key, head, vi in (('fine', 'fine', 0), ('lin', 'lineage', 1)):
            v = b['valid'][:, vi].float()
            acc[f'{key}_n'] += float((-(b[head] * F.log_softmax(o[head], -1)).sum(-1) * v).sum()); acc[f'{key}_d'] += float(v.sum())
        va = b['valid'][:, 2].float()
        acc['act_n'] += float((F.binary_cross_entropy_with_logits(o['activation'], b['act'], reduction='none') * va).sum())
        acc['act_d'] += float(va.sum())
        probs['fine'].append(F.softmax(o['fine'], -1).float().cpu().numpy())
        probs['lineage'].append(F.softmax(o['lineage'], -1).float().cpu().numpy())
        probs['act'].append(torch.sigmoid(o['activation']).float().cpu().numpy())
    m = dict(recon_mse=acc['rec_num'] / max(acc['rec_den'], 1), fine_ce=acc['fine_n'] / max(acc['fine_d'], 1),
             lineage_ce=acc['lin_n'] / max(acc['lin_d'], 1), activation_bce=acc['act_n'] / max(acc['act_d'], 1))
    m['total'] = (CFG['w_recon'] * m['recon_mse'] + CFG['w_fine'] * m['fine_ce'] + CFG['w_lineage'] * m['lineage_ce']
                  + CFG['w_activation'] * m['activation_bce'])
    m['_probs'] = {k: np.concatenate(v) for k, v in probs.items()}
    return m


def pretrain(corpus, baseline, vocab_key):
    """Train (or resume) the baseline model. Resumption is exact: batches and dropout depend only on (seed, step)."""
    dev = get_device()
    work = WORK / 'pretrain'
    work.mkdir(parents=True, exist_ok=True)
    latest, best_path = work / 'latest.pt', work / 'best.pt'
    seed_everything(CFG['seed'])
    model = build_model(corpus.V).to(dev)
    log(f'model: {count_params(model) / 1e6:.2f}M parameters, d={CFG["d_model"]}, latents={CFG["latents"]}, genes={corpus.V}')
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'], betas=(0.9, 0.98))
    identity = stable_hash({'arch': model.config, 'bs': CFG['batch_size'], 'lr': CFG['lr'], 'warm': CFG['warmup_steps'],
                            'vocab': vocab_key, 'seed': CFG['seed'], 'pb': (CFG['pb_min_cells'], CFG['pb_max_cells']),
                            'aug': (CFG['platform_aug'], CFG['noise_aug']), 'mask': CFG['mask_frac_range'],
                            'w': (CFG['w_recon'], CFG['w_fine'], CFG['w_lineage'], CFG['w_activation']),
                            'corpus': [(s['id'], len(s['donor'])) for s in corpus.src], 'v': 4})
    start, best_val, stale, history = 0, float('inf'), 0, []
    if latest.exists():
        snap = torch_load(latest)
        if snap['identity'] != identity:
            raise RuntimeError('checkpoint in work/pretrain was made with a different data/model configuration; '
                               'point HCT_ROOT at a new folder (or delete work/pretrain) to retrain')
        model.load_state_dict(snap['model']); opt.load_state_dict(snap['opt'])
        start, best_val, stale, history = int(snap['step']), float(snap['best_val']), int(snap['stale']), snap['history']
        log(f'resuming pretraining at step {start}/{CFG["pretrain_steps"]} (best val {best_val:.4f})')
    steps, B = CFG['pretrain_steps'], CFG['batch_size']
    val = make_batch(corpus, baseline, 'val', 10 ** 6, CFG['val_pseudobulks'], augment=False, seed=CFG['seed'] + 99, fixed_frac=0.25)
    if start >= steps or stale >= CFG['patience_evals']:
        if best_path.exists():
            model.load_state_dict(torch_load(best_path)['model'])
        return model, history

    def lr_at(s):
        warm = CFG['warmup_steps']
        if s < warm:
            return CFG['lr'] * (s + 1) / warm
        prog = (s - warm) / max(1, steps - warm)
        return CFG['lr'] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    t0 = time.time()
    run_loss = {'total': 0.0, 'recon': 0.0, 'fine': 0.0, 'lineage': 0.0, 'activation': 0.0}
    n_run = 0
    feeder = Prefetcher(lambda s: make_batch(corpus, baseline, 'train', s, B, augment=True), start, steps)
    for step, batch in feeder:
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        torch.manual_seed(CFG['seed'] * 7919 + step)
        if dev.type == 'mps':
            torch.mps.manual_seed(CFG['seed'] * 7919 + step)
        model.train()
        b = to_device(batch, dev)
        out = model(b['z_in'], b['state'], b['logc'])
        loss, parts = pretrain_loss(out, b)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'non-finite loss at step {step}; lower CFG["lr"] and resume')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        run_loss['total'] += float(loss.detach()); n_run += 1
        for k in parts:
            run_loss[k] += float(parts[k])
        done = step + 1
        if done % CFG['eval_every'] == 0 or done == steps:
            m = evaluate_pretrain(model, val, dev)
            rate = (done - start) / max(time.time() - t0, 1e-9)
            tr = {k: v / max(n_run, 1) for k, v in run_loss.items()}
            history.append(dict(step=done, train=tr, val={k: v for k, v in m.items() if k != '_probs'}, lr=lr_at(step)))
            improved = m['total'] < best_val - 1e-4
            if improved:
                best_val, stale = m['total'], 0
                atomic_torch_save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'step': done,
                                   'val': {k: v for k, v in m.items() if k != '_probs'}}, best_path)
            else:
                stale += 1
            log(f'step {done}/{steps} | train {tr["total"]:.3f} (rec {tr["recon"]:.3f}) | val total {m["total"]:.4f} '
                f'rec {m["recon_mse"]:.4f} fine {m["fine_ce"]:.3f} lin {m["lineage_ce"]:.3f} act {m["activation_bce"]:.3f} '
                f'| {rate:.2f} step/s | ETA {(steps - done) / max(rate, 1e-9) / 60:.0f} min | patience {stale}/{CFG["patience_evals"]}')
            run_loss = {k: 0.0 for k in run_loss}; n_run = 0
            save_json(work / 'history.json', history)
            if stale >= CFG['patience_evals']:
                log('early stopping (validation did not improve)')
                break
        if done % CFG['ckpt_every'] == 0 or done == steps:
            atomic_torch_save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'opt': opt.state_dict(),
                               'step': done, 'best_val': best_val, 'stale': stale, 'history': history, 'identity': identity}, latest)
    if not best_path.exists():
        atomic_torch_save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'step': steps, 'val': {}}, best_path)
    model.load_state_dict(torch_load(best_path)['model'])
    return model, history


def pretrain_baselines(model, corpus, baseline):
    """The comparison that makes the pretraining numbers meaningful: trivial and PCA/ridge predictors on the SAME
    held-out-donor profiles and masks."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    dev = get_device()
    n_tr = 2500
    tr = make_batch(corpus, baseline, 'train', 10 ** 6 + 1, n_tr, augment=False, seed=CFG['seed'] + 7)
    va = make_batch(corpus, baseline, 'val', 10 ** 6, CFG['val_pseudobulks'], augment=False, seed=CFG['seed'] + 99, fixed_frac=0.25)
    mm = evaluate_pretrain(model, va, dev)
    ztr, zva = tr['z'].numpy(), va['z'].numpy()
    masked = (va['state'] == STATE_MASKED).numpy()
    zero = float(((zva ** 2) * masked).sum() / masked.sum())
    k = min(48, ztr.shape[0] - 1)
    pca = PCA(n_components=k, random_state=0).fit(ztr)
    Vt = pca.components_                                          # k x V
    mu = pca.mean_
    pred = np.zeros_like(zva)
    for i in range(len(zva)):
        vis = ~masked[i]
        A = Vt[:, vis].T
        coef = np.linalg.solve(A.T @ A + 5.0 * np.eye(k), A.T @ (zva[i, vis] - mu[vis]))
        pred[i] = mu + Vt.T @ coef
    pca_mse = float(((pred - zva) ** 2)[masked].mean())
    res = dict(recon=dict(model=mm['recon_mse'], zero_predictor=zero, pca_ridge=pca_mse))
    scores_tr, scores_va = pca.transform(ztr), pca.transform(zva)
    for name, key, vi, targ in (('fine', 'fine', 0, 'fine'), ('lineage', 'lineage', 1, 'lineage')):
        vtr, vva = tr['valid'][:, vi].numpy(), va['valid'][:, vi].numpy()
        if vtr.sum() < 20 or vva.sum() < 10:
            continue
        rid = Ridge(alpha=10.0).fit(scores_tr[vtr], tr[targ].numpy()[vtr])
        p_pca = np.clip(rid.predict(scores_va[vva]), 0, None)
        prior = np.tile(tr[targ].numpy()[vtr].mean(0), (int(vva.sum()), 1))
        truth = va[targ].numpy()[vva]
        p_model = mm['_probs'][key][vva]
        res[name] = dict(js_model=float(js_divergence(p_model, truth).mean()), js_pca_ridge=float(js_divergence(p_pca, truth).mean()),
                         js_prior=float(js_divergence(prior, truth).mean()),
                         top1_accuracy_model=float((p_model.argmax(1) == truth.argmax(1)).mean()),
                         top1_accuracy_pca_ridge=float((p_pca.argmax(1) == truth.argmax(1)).mean()))
    vtr, vva = tr['valid'][:, 2].numpy(), va['valid'][:, 2].numpy()
    if vtr.sum() >= 20 and vva.sum() >= 10:
        rid = Ridge(alpha=10.0).fit(scores_tr[vtr], tr['act'].numpy()[vtr])
        truth, p_model = va['act'].numpy()[vva], mm['_probs']['act'][vva]
        p_pca = np.clip(rid.predict(scores_va[vva]), 0, 1)
        r2 = lambda p: float(1 - ((truth - p) ** 2).sum() / max(((truth - truth.mean()) ** 2).sum(), 1e-9))
        res['activation'] = dict(r2_model=r2(p_model), r2_pca_ridge=r2(p_pca))
    return res


# ===========================================================================
# CLINICAL COHORTS (public, labelled). None of these samples is ever used for pretraining.
# Every adapter validates sample counts, label vocabulary and ID matching against what was verified during
# development and STOPS on any surprise (no guessing, no silent label coercion).
# ===========================================================================
class Cohort:
    """Samples x vocabulary genes on a log scale (NaN = not measured) + the same samples on every gene the platform measured."""

    def __init__(self, cid, x_vocab, meta, native, native_genes, platform, scale, notes):
        self.id, self.x, self.meta, self.native, self.native_genes = cid, x_vocab, meta, native, native_genes
        self.platform, self.scale, self.notes = platform, scale, notes

    def subset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        return Cohort(self.id, self.x[mask], self.meta[mask].copy(), self.native[mask], self.native_genes,
                      self.platform, self.scale, self.notes)

    def coverage(self):
        return float(np.isfinite(self.x).any(0).mean())


def parse_series_matrix(path, want_table=True):
    """GEO series matrix -> (sample metadata, probe x sample table). EVERY !Sample_ row is kept; repeated
    `characteristics_ch1` rows ('key: value') become separate columns (the previous parser kept only the last row and
    thereby discarded patient IDs, batches and time points)."""
    rows, table, inside = {}, [], False
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_'):
                fields = next(csv.reader([line.rstrip('\r\n')], delimiter='\t', quotechar='"'))
                rows.setdefault(fields[0], []).append(fields[1:])
            elif line.startswith('!series_matrix_table_begin'):
                inside = True
            elif line.startswith('!series_matrix_table_end'):
                break
            elif inside and want_table:
                table.append(line)
    if '!Sample_geo_accession' not in rows:
        raise RuntimeError(f'{path}: no sample accessions')
    gsm = [g.strip() for g in rows['!Sample_geo_accession'][0]]
    meta = pd.DataFrame(index=gsm)
    for key, rr in rows.items():
        name = key[len('!Sample_'):]
        if name.startswith('characteristics'):
            for r in rr:
                if len(r) != len(gsm):
                    raise RuntimeError(f'{path}: ragged characteristics row')
                head = r[0].split(': ', 1)[0].strip().lower() if ': ' in r[0] else f'{name}_{len(meta.columns)}'
                col = head
                while col in meta.columns:
                    col += '_'
                meta[col] = [v.split(': ', 1)[1].strip() if ': ' in v else v.strip() for v in r]
        elif len(rr[0]) == len(gsm) and name not in meta.columns:
            meta[name] = rr[0]
    expr = None
    if want_table:
        if len(table) < 100:
            raise RuntimeError(f'{path}: expression table missing/too small ({len(table)} rows)')
        expr = pd.read_csv(io.StringIO(''.join(table)), sep='\t', index_col=0, low_memory=False)
        expr.columns = [str(c).strip().strip('"') for c in expr.columns]
        if list(expr.columns) != gsm:
            raise RuntimeError(f'{path}: sample order differs between metadata and expression table')
        expr = expr.apply(pd.to_numeric, errors='coerce').astype('float32')
        expr.index = expr.index.astype(str)
    return meta, expr


def _vocab_matrix(frame, genes):
    """samples x genes DataFrame -> numpy samples x vocabulary (NaN where the platform lacks the gene)."""
    return frame.reindex(columns=genes).to_numpy(dtype=np.float32)


def _cohort_cache(cid, vocab_key):
    return WORK / 'cohorts' / f'{cid}.npz', WORK / 'cohorts' / f'{cid}.csv', WORK / 'cohorts' / f'{cid}.json'


def save_cohort(c, key):
    npz, csvp, jp = _cohort_cache(c.id, key)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, x=c.x, native=c.native, native_genes=np.array(c.native_genes, dtype=str))
    c.meta.to_csv(csvp)
    save_json(jp, dict(key=key, platform=c.platform, scale=c.scale, notes=c.notes))


def load_cohort_cached(cid, key):
    npz, csvp, jp = _cohort_cache(cid, key)
    j = load_json(jp)
    if j and j.get('key') == key and npz.exists() and csvp.exists():
        z = np.load(npz, allow_pickle=False)
        meta = pd.read_csv(csvp, index_col=0, keep_default_na=False, na_values=[''])
        return Cohort(cid, z['x'], meta, z['native'], list(z['native_genes']), j['platform'], j['scale'], j['notes'])
    return None


# ---------------------------------------------------------------------------------------------------------
def load_gse73809(vocab):
    gse, gpl = 'GSE73809', 'GPL17586'
    path = fetch(geo_matrix_url(gse), DATA / f'{gse}_series_matrix.txt.gz')
    meta, expr = parse_series_matrix(path)
    need = {'group', 'patient_id', 'batch', 'day_post_hct'}
    if not need <= set(meta.columns) or len(meta) != 24:
        raise RuntimeError(f'{gse}: unexpected metadata {list(meta.columns)} / n={len(meta)}')
    lab = {'GVHD': 1, 'No GVHD': 0}
    bad = set(meta['group']) - set(lab)
    if bad:
        raise RuntimeError(f'{gse}: unexpected group labels {bad}')
    m = pd.DataFrame({'patient': meta['patient_id'], 'compartment': 'PBMC_CD4CD8_T', 'gvhd': meta['group'].map(lab).astype(int),
                      'day_post_hct': pd.to_numeric(meta['day_post_hct']), 'batch': meta['batch'],
                      'site': np.where(meta['patient_id'].str.startswith('UM'), 'Minnesota', 'Emory')}, index=meta.index)
    if m.patient.nunique() != 24 or m.gvhd.sum() != 11:
        raise RuntimeError(f'{gse}: expected 24 distinct patients (11 GVHD); got {m.patient.nunique()} / {m.gvhd.sum()}')
    genes = collapse_probes(expr.T.to_numpy(), list(expr.index), gpl_probe_map(gpl))
    genes.index = m.index
    notes = dict(source='GSE73809', design='case-matched aGVHD vs no GVHD, PBMC CD4+CD8 T cells, day 28-176 post-HCT',
                 label='concurrent aGVHD (onset timing not provided)', patient_independence='verified: 24 distinct patient_id')
    return Cohort(gse, _vocab_matrix(genes, vocab['genes']), m, genes.to_numpy(), list(genes.columns), gpl, 'log2 (RMA)', notes)


def load_gse103569(vocab):
    gse, gpl = 'GSE103569', 'GPL13497'
    path = fetch(geo_matrix_url(gse), DATA / f'{gse}_series_matrix.txt.gz')
    meta, expr = parse_series_matrix(path)
    rx = re.compile(r'^Ly/(CD8b\+/CLA\+|CD8b\+/ITGB7\+|CD8b\+/CLA-/ITGB7-) T cells_aGvHD developed: (.+)_Patient No\. (\d+)$')
    rows = []
    for t in meta['title']:
        mt = rx.match(t)
        if not mt:
            raise RuntimeError(f'{gse}: unrecognised title {t!r}')
        rows.append(mt.groups())
    comp = {'CD8b+/CLA+': 'CD8_skinhoming_CLA+', 'CD8b+/ITGB7+': 'CD8_guthoming_ITGB7+', 'CD8b+/CLA-/ITGB7-': 'CD8_reference_DN'}
    status = [r[1] for r in rows]
    allowed = {'Absent', 'Cutaneous', 'Gastrointestinal', 'Cutaneous + Gastrointestinal'}
    if set(status) - allowed:
        raise RuntimeError(f'{gse}: unexpected aGVHD labels {set(status) - allowed}')
    m = pd.DataFrame({'patient': [f'P{r[2]}' for r in rows], 'compartment': [comp[r[0]] for r in rows],
                      'gvhd': [int(s != 'Absent') for s in status], 'skin': [int('Cutaneous' in s) for s in status],
                      'gut': [int('Gastrointestinal' in s) for s in status], 'organ': status}, index=meta.index)
    if len(m) != 60 or m.patient.nunique() != 20 or (m.groupby('patient').gvhd.nunique() > 1).any():
        raise RuntimeError(f'{gse}: expected 60 samples from 20 patients with patient-constant labels')
    genes = collapse_probes(expr.T.to_numpy(), list(expr.index), gpl_probe_map(gpl))
    genes.index = m.index
    notes = dict(source='GSE103569', design='sorted CD8+ skin-/gut-homing/reference subsets from 20 aHSCT patients',
                 label='patient developed aGVHD (any organ) vs none; organ involvement', patient_independence='verified: 3 samples per patient (grouped)')
    return Cohort(gse, _vocab_matrix(genes, vocab['genes']), m, genes.to_numpy(), list(genes.columns), gpl, 'log (Agilent, normalised)', notes)


def load_gse4624(vocab):
    gse, gpl, name = 'GSE4624', 'GPL3639', 'GSE4624-GPL3639_series_matrix.txt.gz'
    path = fetch(geo_matrix_url(gse, name), DATA / name)
    meta, expr = parse_series_matrix(path)
    rx = re.compile(r'^(PERH\d+)_(CD[48])\+_(t\d)_aGVHD([+-])_cGVHD([+-])_(\d+)$')
    rows = []
    for t in meta['title']:
        mt = rx.match(t)
        if not mt:
            raise RuntimeError(f'{gse}: unrecognised title {t!r}')
        rows.append(mt.groups())
    d = pd.DataFrame(rows, columns=['perh', 'comp', 'time', 'a', 'c', 'suffix'], index=meta.index)
    d['num'] = d.perh.str[4:].astype(int)
    d = d[d.time == 't0']                                   # donor, before transplant; t3 = recipients at day 365
    if len(d) != 96:
        raise RuntimeError(f'{gse}: expected 96 pre-transplant donor samples, found {len(d)}')
    # donor grouping: a donor's CD4 sample (id n) and CD8 sample (id n+1) have consecutive ids and identical outcomes
    # (verified for 45/45 pairs). Unpaired samples are their own group.
    key = {}
    by_num = {(r.comp, r.num): (i, r) for i, r in d.iterrows()}
    n_pairs = 0
    for (comp, num), (i, r) in by_num.items():
        if comp == 'CD4':
            partner = by_num.get(('CD8', num + 1))
            if partner and partner[1].a == r.a and partner[1].c == r.c:
                key[partner[0]] = f'D{num}'
                n_pairs += 1
            key[i] = f'D{num}'
    for i in d.index:
        key.setdefault(i, f'D{d.loc[i, "num"]}')
    m = pd.DataFrame({'patient': [key[i] for i in d.index], 'compartment': d.comp.map({'CD4': 'CD4', 'CD8': 'CD8'}),
                      'gvhd': (d.a == '+').astype(int), 'cgvhd': (d.c == '+').astype(int), 'array_order': d.suffix.astype(int)}, index=d.index)
    genes = collapse_probes(expr.loc[:, d.index].T.to_numpy(), list(expr.index), gpl_probe_map(gpl))
    genes.index = d.index
    notes = dict(source='GSE4624 (H19K array, pre-HCT donor T cells)', design='50 HLA-identical sibling donors; CD4+/CD8+ T cells before transplant',
                 label='RECIPIENT acute GVHD within 100 days (prognostic: sample precedes the outcome)',
                 patient_independence=f'donor groups inferred from consecutive ids ({n_pairs} CD4/CD8 pairs, all outcome-concordant)',
                 caution='`array_order` (title suffix) is weakly outcome-associated; its AUROC is reported next to every result')
    return Cohort(gse, _vocab_matrix(genes, vocab['genes']), m, genes.to_numpy(), list(genes.columns), gpl, 'log2 ratio vs universal reference', notes)


def load_ptcy(vocab):
    """Sorted CD8 / CD4-Tconv / Treg / NK RNA-seq from patients after post-transplant cyclophosphamide (GSE182610/12/78)."""
    ens2sym = load_ens2sym()
    metas, mats = [], []
    for gse, files in PTCY_COUNTS.items():
        mpath = fetch(geo_matrix_url(gse), DATA / f'{gse}_series_matrix.txt.gz')
        meta, _ = parse_series_matrix(mpath, want_table=False)
        if 'disease state' not in meta.columns:
            raise RuntimeError(f'{gse}: no disease state')
        meta = meta.assign(gse=gse, sid=meta['title'].str.upper())
        counts = {}
        for fname, nbytes, sha in files:
            p = fetch(f'{geo_series_dir(gse)}/suppl/{fname}', DATA / fname, expected_bytes=nbytes, expected_sha256=sha)
            df = pd.read_csv(p, sep='\t', index_col=0)
            df.columns = [c.upper() for c in df.columns]
            counts[fname] = df
        allcols = {c: f for f, df in counts.items() for c in df.columns}
        missing = [s for s in meta.sid if s not in allcols]
        if missing:
            raise RuntimeError(f'{gse}: {len(missing)} deposited samples have no counts, e.g. {missing[:3]}')
        for fname, df in counts.items():
            use = [c for c in df.columns if c in set(meta.sid)]
            if not use:
                continue
            sub = df[use]
            lib = sub.sum(0).to_numpy(dtype=np.float64)
            sym = pd.Series(sub.index.map(lambda e: ens2sym.get(str(e).split('.')[0])), index=sub.index)
            ok = sym.notna().to_numpy()
            by_sym = sub[ok].groupby(sym[ok].to_numpy()).sum()
            logcpm = np.log1p(by_sym.T.to_numpy(dtype=np.float64) / lib[:, None] * 1e6).astype(np.float32)
            mats.append(pd.DataFrame(logcpm, index=use, columns=by_sym.index))
            metas.append(meta[meta.sid.isin(use)].assign(batch=re.search(r'_(\d+R)_', fname).group(1)).set_index('sid'))
    meta = pd.concat(metas)
    if meta.index.duplicated().any():
        raise RuntimeError('PTCy: duplicated sample ids across subseries')
    genes = pd.concat(mats, axis=0, join='outer')
    genes = genes.loc[meta.index]
    comp = {'Sorted CD8-positive T cells': 'CD8', 'Sorted CD4 Tconv T cells': 'CD4_Tconv', 'Sorted CD4 Treg T cells': 'Treg',
            'Sorted NK cells': 'NK'}
    if set(meta.source_name_ch1) - set(comp):
        raise RuntimeError(f'PTCy: unexpected sorted populations {set(meta.source_name_ch1) - set(comp)}')
    st = meta['disease state']
    if not st.str.match(r'^(No )?GVHD(, (No )?Relapse)?$').all():
        raise RuntimeError(f'PTCy: unexpected disease-state labels {set(st)}')
    m = pd.DataFrame({'patient': meta.index, 'compartment': meta.source_name_ch1.map(comp),
                      'gvhd': st.str.match(r'^GVHD').astype(int), 'relapse': np.where(st.str.contains('Relapse'), np.where(st.str.contains('No Relapse'), 0, 1), -1),
                      'batch': meta['batch'], 'series': meta['gse']}, index=meta.index)
    notes = dict(source='GSE182610 + GSE182612 + GSE182678 (bulk RNA-seq, Ensembl 92/98 -> symbols)',
                 design='FACS-sorted CD8 / CD4-Tconv / Treg / NK from allo-HCT patients treated with PTCy',
                 label='GVHD (vs no GVHD) at sampling', patient_independence='UNVERIFIED: GEO gives no patient ids; each population is modelled separately',
                 caution='batches (15065R/17618R/17941R/18740R) are reported; batch-only AUROC is shown next to results')
    return Cohort('PTCY', _vocab_matrix(genes, vocab['genes']), m, genes.to_numpy(), list(genes.columns), 'bulk RNA-seq', 'log1p CPM', notes)


GSE295359_CD4 = {'CD4 Tnv/cm', 'Th1', 'Th2', 'Th17', 'Tfh', 'Treg'}


def load_gse295359(vocab):
    """Single-cell HCT T cells -> one pseudobulk per (donor x timepoint) sample and compartment (T_all / CD4 / CD8)."""
    import h5py
    import scipy.sparse as sp
    z = ZENODO_GSE295359
    path = fetch(z['url'], DATA / 'GSE295359_total_t_cells.h5ad', expected_bytes=z['bytes'], expected_sha256=z['sha256'])
    d = open_h5ad_meta(path)
    try:
        obs = d.obs[['sample', 'donor', 'timepoint', 'aGvHD', 'remission_status', 'refined_anno_t']].copy()
        names = np.array([str(n).upper() for n in d.var_names])
    finally:
        if getattr(d, 'file', None) is not None:
            d.file.close()
    first = {}
    for i, s in enumerate(names):
        first.setdefault(s, i)
    present = [g for g in vocab['genes'] if g in first]
    cols = np.array([first[g] for g in present])
    samples = sorted(obs['sample'].astype(str).unique())
    sidx = {s: i for i, s in enumerate(samples)}
    comps = ['T_all', 'CD4', 'CD8']
    sums = np.zeros((len(samples), 3, len(cols)))
    libs = np.zeros((len(samples), 3))
    ncell = np.zeros((len(samples), 3), dtype=int)
    anno = obs['refined_anno_t'].astype(str).to_numpy()
    in_c = {'T_all': np.ones(len(obs), bool), 'CD4': np.isin(anno, list(GSE295359_CD4)), 'CD8': np.char.startswith(anno.astype(str), 'CD8 ')}
    srow = np.array([sidx[s] for s in obs['sample'].astype(str)])
    with h5py.File(path, 'r') as f:
        for a in range(0, len(obs), 10_000):
            b = min(a + 10_000, len(obs))
            X = csr_chunk(f, 'layers/counts', a, b)
            lib = np.asarray(X.sum(1)).ravel()
            Xv = X[:, cols]
            for ci, c in enumerate(comps):
                sel = in_c[c][a:b]
                if not sel.any():
                    continue
                ind = sp.csr_matrix((np.ones(sel.sum()), (srow[a:b][sel], np.arange(sel.sum()))), shape=(len(samples), int(sel.sum())))
                sums[:, ci, :] += (ind @ Xv[sel]).toarray()
                libs[:, ci] += np.bincount(srow[a:b][sel], weights=lib[sel], minlength=len(samples))
                ncell[:, ci] += np.bincount(srow[a:b][sel], minlength=len(samples))
    lab = obs.drop_duplicates('sample').set_index(obs.drop_duplicates('sample')['sample'].astype(str))
    rows, xs = [], []
    for s in samples:
        for ci, c in enumerate(comps):
            if ncell[sidx[s], ci] >= 50:
                r = lab.loc[s]
                x = np.full(len(vocab['genes']), np.nan, dtype=np.float32)
                pos = [vocab['genes'].index(g) for g in present]
                x[pos] = np.log1p(sums[sidx[s], ci] / libs[sidx[s], ci] * 1e6)
                xs.append(x)
                rows.append(dict(sample_id=f'{s}|{c}', patient=str(r['donor']) if 'donor' in r else s.rsplit('_', 1)[0],
                                 compartment=c, timepoint=str(r['timepoint']), gvhd=int(str(r['aGvHD']) == 'II-IV'),
                                 remission=str(r['remission_status']), n_cells=int(ncell[sidx[s], ci])))
    m = pd.DataFrame(rows).set_index('sample_id')
    if m.patient.nunique() != 10 or m.groupby('patient').gvhd.nunique().max() != 1:
        raise RuntimeError(f'GSE295359: expected 10 donors with donor-constant labels, found {m.patient.nunique()}')
    X = np.stack(xs).astype(np.float32)
    notes = dict(source='GSE295359 (Song et al.; Zenodo 15264188)', design='scRNA-seq T cells, 10 allo-HCT recipients, graft / day 30 / day 100',
                 label='recipient aGVHD grade II-IV (donor-constant)', patient_independence='verified: donor id; label constant within donor',
                 caution='only 10 patients; onset timing relative to sampling is not in the public metadata')
    return Cohort('GSE295359', X, m, X, list(vocab['genes']), 'scRNA pseudobulk', 'log1p CPM', notes)


COHORT_LOADERS = {'GSE73809': load_gse73809, 'GSE103569': load_gse103569, 'GSE4624': load_gse4624,
                  'PTCY': load_ptcy, 'GSE295359': load_gse295359}


def load_cohorts(vocab, which=None):
    key = stable_hash({'genes': vocab['key'], 'v': 4})
    out = {}
    for cid in (which or COHORT_LOADERS):
        c = load_cohort_cached(cid, key)
        if c is None:
            t0 = time.time()
            c = COHORT_LOADERS[cid](vocab)
            save_cohort(c, key)
            log(f'cohort {cid}: {len(c.meta)} samples, vocabulary coverage {c.coverage():.0%}, {time.time() - t0:.0f}s')
        out[cid] = c
    return out


# ===========================================================================
# PIPELINE STAGES: baseline reference -> pretraining -> exported foundation-model artifact
# ===========================================================================
def get_baseline(corpus, vocab):
    """Healthy-T-cell reference (fitted on training donors only), cached."""
    path = WORK / 'baseline.npz'
    key = stable_hash({'vocab': vocab['key'], 'corpus': [(s['id'], len(s['donor'])) for s in corpus.src],
                       'pb': (CFG['pb_min_cells'], CFG['pb_max_cells']), 'seed': CFG['seed'], 'v': 4})
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z['key']) == key:
            return dict(ref_q=z['ref_q'], mu=z['mu'], sd=z['sd'], raw_sd=z['raw_sd'], n_profiles=int(z['n_profiles']))
    b = fit_baseline(corpus, n_profiles=4000, seed=CFG['seed'] + 3)
    np.savez(path, key=key, **b)
    log(f'T-cell baseline fitted from {b["n_profiles"]} healthy pseudobulks (training donors only)')
    return b


def save_foundation_model(model, baseline, vocab, report, history, path, artifact_key):
    atomic_torch_save({
        'format': 'hct_tcell_fm_v4', 'artifact_key': artifact_key,
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'hp': model.config, 'genes': list(vocab['genes']), 'vocab_key': vocab['key'],
        'baseline': {k: torch.as_tensor(np.asarray(v, dtype=np.float64)) for k, v in baseline.items() if k != 'n_profiles'},
        'fine_classes': FINE_CLASSES, 'lineage_classes': LINEAGE_CLASSES,
        'pretrain_report': report, 'pretrain_history_tail': history[-5:],
        'sources': [{k: s[k] for k in ('id', 'url', 'sha256', 'cite')} for s in PRETRAIN_SOURCES],
        'clinical_data_in_pretraining': False,
    }, path)


def load_foundation_model(path):
    snap = torch_load(path)
    if snap.get('format') != 'hct_tcell_fm_v4':
        raise RuntimeError(f'{path} is not a v4 foundation-model artifact')
    model = build_model(len(snap['genes']), snap['hp'])
    model.load_state_dict(snap['model'])
    base = {k: v.numpy() for k, v in snap['baseline'].items()}
    base['n_profiles'] = 0
    return model.eval(), base, snap


def stage_pretrain():
    ensure_dirs()
    vocab, _ = build_corpus_stores()
    corpus = Corpus(vocab['genes'])
    baseline = get_baseline(corpus, vocab)
    art = ROOT / 'model' / 'tcell_baseline_fm.pt'
    key = stable_hash({'vocab': vocab['key'], 'cfg': {k: CFG[k] for k in ('d_model', 'heads', 'layers', 'latents', 'dropout', 'pretrain_steps',
                                                                          'batch_size', 'lr', 'seed', 'max_cells_per_source')}, 'v': 4})
    if art.exists():
        snap = torch_load(art)
        if snap.get('format') == 'hct_tcell_fm_v4' and snap.get('artifact_key') == key:
            log(f'pretrained foundation model found: {art}')
            return art, vocab
    model, history = pretrain(corpus, baseline, vocab['key'])
    report = pretrain_baselines(model, corpus, baseline)
    save_json(OUT / 'pretrain_report.json', report)
    log('PRETRAINING REPORT (held-out donors): ' + json.dumps(_clean_nan(report), default=_json_default))
    save_foundation_model(model, baseline, vocab, report, history, art, key)
    return art, vocab


# ===========================================================================
# CLINICAL TASKS AND EVALUATION
# ===========================================================================
# Each task fixes: cohort + T-cell compartment, the UNIT (patient), the LABEL, and what the sample's timing means.
# Features are FROZEN (the pretrained encoder is not fine-tuned on a few dozen patients); the clinical head is a
# regularised logistic regression whose strength is chosen by nested cross-validation.
TASKS = [
    dict(id='K1_donor_CD8_to_recipient_aGVHD', cohort='GSE4624', compartments=['CD8'], kind='prognostic', primary=True,
         confounders=['array_order'], question='Does a DONOR\'s pre-transplant CD8 T-cell transcriptome predict acute GVHD in the recipient?'),
    dict(id='K1_donor_CD4_to_recipient_aGVHD', cohort='GSE4624', compartments=['CD4'], kind='prognostic', primary=True,
         confounders=['array_order'], question='Same, from donor CD4 T cells.'),
    dict(id='K2_PTCy_CD8_GVHD', cohort='PTCY', compartments=['CD8'], kind='diagnostic', primary=True, confounders=['batch'],
         question='Is GVHD present, judged from sorted CD8 T cells after post-transplant cyclophosphamide?'),
    dict(id='K2_PTCy_CD4Tconv_GVHD', cohort='PTCY', compartments=['CD4_Tconv'], kind='diagnostic', primary=True, confounders=['batch'],
         question='Same, from sorted conventional CD4 T cells.'),
    dict(id='K2_PTCy_Treg_GVHD', cohort='PTCY', compartments=['Treg'], kind='diagnostic', primary=True, confounders=['batch'],
         question='Same, from sorted regulatory T cells.'),
    dict(id='K3_PBMC_T_aGVHD', cohort='GSE73809', compartments=['PBMC_CD4CD8_T'], kind='diagnostic', primary=True,
         confounders=['batch', 'site', 'day_post_hct'], question='aGVHD vs matched controls from PBMC CD4+CD8 T cells (day 28-176).'),
    dict(id='K4_CD8_subsets_aGVHD', cohort='GSE103569', compartments=['CD8_skinhoming_CLA+', 'CD8_guthoming_ITGB7+', 'CD8_reference_DN'],
         kind='diagnostic', primary=True, confounders=[], question='Did the patient develop aGVHD (any organ)? Three sorted CD8 subsets per patient, grouped by patient.'),
    dict(id='K5_scRNA_pseudobulk_aGVHD', cohort='GSE295359', compartments=['T_all', 'CD4', 'CD8'], kind='diagnostic', primary=True,
         confounders=['timepoint'], question='Recipient aGVHD grade II-IV from single-cell-derived T-cell pseudobulk (graft/D30/D100), grouped by patient.'),
    dict(id='K5x_graft_T_to_aGVHD', cohort='GSE295359', compartments=['T_all'], kind='prognostic', primary=False, confounders=[],
         select=lambda m: m.timepoint == 'Graft', question='EXPLORATORY: does the infused graft\'s T-cell pseudobulk predict later aGVHD II-IV? (8 patients)'),
]
FEATURE_SETS = ['fmA_all', 'fm_all', 'fmA_embed', 'fmA_comp', 'scratchA_embed', 'scratch_embed', 'genes_z', 'genes_vocab',
                'genes_native', 'hybrid_genes_fmA', 'fmA_resid', 'scratchA_resid']   # *_resid: exploratory, added after iteration 2
# Pre-specified BEFORE the second-iteration results were seen: the anchored foundation-model features are the primary family;
# the strongest gene-level family (all platform genes) is the comparator.
PRIMARY_SET, COMPARATOR_SET = 'fmA_all', 'genes_native'
DEPLOY_SET = 'genes_z'          # baseline-normalised vocabulary genes: platform-harmonised, needs no encoder at scoring time
MIN_ANCHOR_SAMPLES = 8


def anchor_to_baseline(z, obs, blocks):
    """Centre each (cohort, compartment) block on its own label-free mean in baseline space. Removes platform/cohort offsets so
    the encoder sees deviations that are in-distribution. Returns anchored z and the per-block mean vectors (stored in heads)."""
    za, means = z.copy(), {}
    for b in np.unique(blocks):
        idx = blocks == b
        if idx.sum() < MIN_ANCHOR_SAMPLES:
            continue
        cnt = obs[idx].sum(0)
        m = np.where(cnt > 0, (z[idx] * obs[idx]).sum(0) / np.maximum(cnt, 1), 0.0)
        means[str(b)] = m
        za[idx] = np.where(obs[idx], np.clip(z[idx] - m, -CFG['z_clip'], CFG['z_clip']), 0.0)
    return za, means


def comp_features(e, dev):
    return np.hstack([e['fine'], e['lineage'], e['activation'][:, None], dev[:, None]])


def featurize(cohort, model, baseline, scratch):
    """All feature families for a cohort's samples (frozen encoders; nothing here sees a label)."""
    obs = np.isfinite(cohort.x)
    nv = obs.sum(1) / obs.shape[1]
    if (nv < CFG['min_vocab_coverage']).any():
        raise RuntimeError(f'{cohort.id}: samples measure <{CFG["min_vocab_coverage"]:.0%} of the vocabulary')
    z, obs = to_model_space(cohort.x, obs, baseline)
    blocks = cohort.meta.compartment.astype(str).to_numpy()
    za, anchors = anchor_to_baseline(z, obs, blocks)
    e, ea = embed_profiles(model, z, obs), embed_profiles(model, za, obs)
    s_raw, s_anc = embed_profiles(scratch, z, obs), embed_profiles(scratch, za, obs)
    dev, deva = deviation_scores(model, z, obs), deviation_scores(model, za, obs)
    comp, compa = comp_features(e, dev), comp_features(ea, deva)
    nat = cohort.native.astype(np.float64)
    med = np.nanmedian(nat, axis=0)
    nat = np.where(np.isfinite(nat), nat, med[None, :])
    nat = nat[:, np.nanvar(nat, axis=0) > 0]
    top = np.argsort(-np.nanvar(nat, axis=0))[:2000]            # label-free variance filter on the cohort's own values
    xv = cohort.x.astype(np.float64)                              # vocabulary genes, native scale (isolates the vocabulary effect)
    vmed = np.nanmedian(np.where(np.isfinite(xv), xv, np.nan), axis=0)
    xv = np.where(np.isfinite(xv), xv, np.nan_to_num(vmed)[None, :])
    fmA_all = np.hstack([ea['pooled'], compa])
    resid = residual_profiles(model, za, obs)
    resid_scratch = residual_profiles(scratch, za, obs)
    return dict(fmA_all=fmA_all, fm_all=np.hstack([e['pooled'], comp]), fmA_embed=ea['pooled'], fmA_comp=compa,
                scratchA_embed=s_anc['pooled'], scratch_embed=s_raw['pooled'], genes_z=z, genes_vocab=xv.astype(np.float32),
                genes_native=nat[:, top].astype(np.float32), hybrid_genes_fmA=np.hstack([z, fmA_all]),
                fmA_resid=resid, scratchA_resid=resid_scratch,
                _comp_names=[f'fine:{c}' for c in FINE_CLASSES] + [f'lineage:{c}' for c in LINEAGE_CLASSES] + ['activation', 'deviation'],
                _components=dict(fine=e['fine'], lineage=e['lineage'], activation=e['activation'], deviation=dev), _anchors=anchors)


def _head(C):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000))


def fit_head(X, y, groups, seed=0):
    """Logistic head; C chosen by patient-grouped inner CV (fixed fallback when a class is too small)."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    grid = CFG['head_C_grid']
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    k = min(4, n_pos, n_neg)
    if k < 3:
        return _head(0.03).fit(X, y), 0.03
    best_c, best = grid[len(grid) // 2], -1.0
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))
    for c in grid:
        pred = np.zeros(len(y))
        ok = True
        for tr, te in folds:
            if len(set(y[tr])) < 2:
                ok = False
                break
            pred[te] = _head(c).fit(X[tr], y[tr]).decision_function(X[te])
        if ok and len(set(y)) == 2:
            a = roc_auc_score(y, pred)
            if a > best + 1e-9:
                best, best_c = a, c
    return _head(best_c).fit(X, y), best_c


def cv_oof(X, y, groups, repeat, n_splits=5):
    """One repeat of patient-grouped stratified K-fold -> out-of-fold scores."""
    from sklearn.model_selection import StratifiedGroupKFold
    n_pos_g = len(set(groups[y == 1])); n_neg_g = len(set(groups[y == 0]))
    k = max(2, min(n_splits, n_pos_g, n_neg_g))
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=1000 + repeat).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        head, _ = fit_head(X[tr], y[tr], groups[tr], seed=repeat)
        oof[te] = head.decision_function(X[te])
    return oof


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m])) if m.sum() > 3 and len(set(y[m])) == 2 else float('nan')


def average_precision(y, s):
    from sklearn.metrics import average_precision_score
    m = np.isfinite(s)
    return float(average_precision_score(y[m], s[m])) if m.sum() > 3 and len(set(y[m])) == 2 else float('nan')


def _perm_auc(args):
    X, y, groups, seed = args
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    lab = {g: y[groups == g][0] for g in ug}
    perm = dict(zip(ug, rng.permutation([lab[g] for g in ug])))
    yp = np.array([perm[g] for g in groups])
    return auroc(yp, cv_oof(X, yp, groups, repeat=int(seed % 997)))


def group_bootstrap_auc(y, scores_list, groups, n_boot, seed=0):
    """Bootstrap patients; returns AUROC samples for each score vector with the SAME resamples (paired)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ug}
    out = [[] for _ in scores_list]
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_of[g] for g in pick])
        if len(set(y[ii])) < 2:
            continue
        for k, s in enumerate(scores_list):
            out[k].append(auroc(y[ii], s[ii]))
    return [np.array(o) for o in out]


def confounder_auroc(meta, cols, y, groups, repeats):
    """Can the label be predicted from technical / design variables alone? (a high value warns of confounding)"""
    res = {}
    for c in cols:
        if c not in meta.columns:
            continue
        v = meta[c]
        X = pd.get_dummies(v.astype(str)).to_numpy(float) if v.dtype == object or v.nunique() < 6 else v.to_numpy(float)[:, None]
        if len(set(v.astype(str))) < 2:
            continue
        aucs = [auroc(y, cv_oof(X, y, groups, r)) for r in range(max(2, repeats // 2))]
        res[c] = float(np.nanmean(aucs))
    return res


def evaluate_task(task, cohort, feats, n_jobs=None):
    """Repeated patient-grouped CV for every feature family + permutation null + paired bootstrap against the comparator."""
    from joblib import Parallel, delayed
    sel = cohort.meta.compartment.isin(task['compartments']).to_numpy()
    if 'select' in task:
        sel &= np.asarray(task['select'](cohort.meta), dtype=bool)
    meta = cohort.meta[sel]
    y = meta.gvhd.to_numpy().astype(int)
    groups = meta.patient.astype(str).to_numpy()
    info = dict(id=task['id'], question=task['question'], kind=task['kind'], primary=task['primary'], n_samples=int(sel.sum()),
                n_patients=int(len(set(groups))), n_positive_samples=int(y.sum()), prevalence=float(y.mean()), cohort=task['cohort'],
                compartments=task['compartments'], patient_independence=cohort.notes.get('patient_independence'))
    if sel.sum() < CFG['min_cohort_samples'] or min(y.sum(), (1 - y).sum()) < 4:
        info['status'] = 'skipped: too few samples/positives'
        return info
    # block-wise centring (label-free) so pooled compartments are comparable
    blocks = meta.compartment.to_numpy()
    oof_avg, per_set = {}, {}
    for name in FEATURE_SETS:
        X = feats[name][sel].astype(np.float64)
        if len(task['compartments']) > 1:
            X = X.copy()
            for b in np.unique(blocks):
                X[blocks == b] -= X[blocks == b].mean(0)
        oofs = [cv_oof(X, y, groups, r) for r in range(CFG['cv_repeats'])]
        aucs = [auroc(y, o) for o in oofs]
        avg = np.nanmean(np.stack(oofs), axis=0)
        oof_avg[name] = avg
        per_set[name] = dict(auroc=float(np.nanmean(aucs)), auroc_sd_over_repeats=float(np.nanstd(aucs)),
                             average_precision=average_precision(y, avg))
    names = list(oof_avg)
    boots = group_bootstrap_auc(y, [oof_avg[n] for n in names], groups, CFG['n_boot'])
    for n, b in zip(names, boots):
        per_set[n]['auroc_ci95'] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    base_b = boots[names.index(COMPARATOR_SET)]
    for n, b in zip(names, boots):
        k = min(len(b), len(base_b))
        d = b[:k] - base_b[:k]
        per_set[n]['delta_vs_comparator'] = float(per_set[n]['auroc'] - per_set[COMPARATOR_SET]['auroc'])
        per_set[n]['delta_ci95'] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    # permutation nulls (group-level label shuffles through the FULL nested procedure)
    jobs = n_jobs or max(1, min(4, os.cpu_count() or 1))
    for name in (PRIMARY_SET, COMPARATOR_SET, DEPLOY_SET):
        X = feats[name][sel].astype(np.float64)
        if len(task['compartments']) > 1:
            for b in np.unique(blocks):
                X[blocks == b] -= X[blocks == b].mean(0)
        nulls = Parallel(n_jobs=jobs, backend='threading')(delayed(_perm_auc)((X, y, groups, 7000 + i)) for i in range(CFG['n_perm']))
        nulls = np.array([v for v in nulls if np.isfinite(v)])
        per_set[name]['perm_p'] = float((1 + (nulls >= per_set[name]['auroc']).sum()) / (1 + len(nulls)))
        per_set[name]['perm_null_mean'] = float(nulls.mean()) if len(nulls) else float('nan')
    info['status'] = 'ok'
    info['feature_sets'] = per_set
    info['confounder_only_auroc'] = confounder_auroc(meta, task['confounders'], y, groups, CFG['cv_repeats'])
    info['oof'] = {n: oof_avg[n].tolist() for n in (PRIMARY_SET,)}
    return info


def verdict(res, family=None):
    """A deliberately blunt reading of one task result for one feature family (default: the pre-specified primary)."""
    if res.get('status') != 'ok':
        return res.get('status', 'n/a')
    family = family or PRIMARY_SET
    fm = res['feature_sets'][family]
    if 'perm_p' not in fm:
        return 'not tested'
    sig = fm['perm_p'] < 0.05 and fm['auroc_ci95'][0] > 0.5
    if family == COMPARATOR_SET:
        return 'significant' if sig else 'NOT distinguishable from chance'
    beats = fm['delta_ci95'][0] > 0
    if sig and beats:
        return 'SUPPORTED: above chance AND better than the all-genes baseline'
    if sig:
        return 'above chance (not shown to beat the all-genes baseline)'
    return 'NOT distinguishable from chance'


# ===========================================================================
# TRANSFER CHECKS, CROSS-STUDY EVALUATION, EXPORT, INFERENCE API, REPORT
# ===========================================================================
def _softmax(a):
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


MARKERS = {'CD8': (['CD8A', 'CD8B'], ['CD4']), 'CD4_conventional': (['CD4', 'IL7R'], ['CD8A', 'CD8B']),
           'Treg': (['FOXP3', 'IL2RA', 'CTLA4'], ['IL7R']), 'NK': (['NKG7', 'GNLY', 'KLRD1'], ['CD3E'])}
COMPARTMENT_TO_LINEAGE = {'CD8': 'CD8', 'CD4_Tconv': 'CD4_conventional', 'Treg': 'Treg', 'NK': 'NK', 'CD4': 'CD4_conventional'}


def marker_score(cohort, cls):
    pos, neg = MARKERS[cls]
    cols = {g: i for i, g in enumerate(cohort.native_genes)}
    p = [cohort.native[:, cols[g]] for g in pos if g in cols]
    n = [cohort.native[:, cols[g]] for g in neg if g in cols]
    if not p:
        return None
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    s = np.mean([z(v) for v in p], axis=0)
    if n:
        s = s - np.mean([z(v) for v in n], axis=0)
    return s


def transfer_checks(cohorts, features):
    """Does the composition head, trained ONLY on single-cell pseudobulks, recognise sorted bulk populations on other
    platforms? One-vs-rest AUROC of the predicted lineage probability against known sorted identity (and a marker-gene
    score for reference). This is the test of the 'T-cell baseline' idea; it uses no GVHD label."""
    out = {}
    for cid, c in cohorts.items():
        comp = c.meta.compartment.astype(str).map(COMPARTMENT_TO_LINEAGE)
        P = _softmax(features[cid]['_components']['lineage'])
        res = {}
        for cls in set(comp.dropna()):
            y = (comp == cls).to_numpy().astype(int)
            others = comp.notna().to_numpy()
            if y.sum() < 3 or (others & (y == 0)).sum() < 3:
                continue
            k = LINEAGE_CLASSES.index(cls)
            ms = marker_score(c, cls)
            res[cls] = dict(n=int(y.sum()), n_other=int((others & (y == 0)).sum()),
                            auroc_model=auroc(y[others], P[others, k]),
                            auroc_marker_genes=auroc(y[others], ms[others]) if ms is not None else None)
        if comp.notna().any():
            pred = P.argmax(1)
            true = np.array([LINEAGE_CLASSES.index(x) if isinstance(x, str) else -1 for x in comp])
            ok = true >= 0
            res['_top1_accuracy'] = float((pred[ok] == true[ok]).mean()) if ok.any() else None
        out[cid] = res
    return out


def loco_transfer(cohorts, features, sets=('fmA_all', 'fmA_resid', 'hybrid_genes_fmA', 'genes_z', 'scratchA_embed')):
    """Leave-one-STUDY-out: train a head on all other studies' patients, test on the held-out study. Features are
    centred within (study, compartment) blocks using only label-free statistics of that block."""
    pool = []
    for cid, c in cohorts.items():
        m = c.meta
        keep = np.ones(len(m), bool)
        if cid == 'GSE295359':
            keep = m.timepoint.isin(['D30', 'D100']).to_numpy()                 # concurrent-GVHD semantics only
        if cid == 'GSE4624':
            continue                                                            # prognostic (donor) task: different label meaning
        if cid == 'PTCY':
            keep &= (m.compartment != 'NK').to_numpy()
        for i in np.where(keep)[0]:
            pool.append((cid, i))
    study = np.array([p[0] for p in pool])
    y = np.array([int(cohorts[c].meta.gvhd.iloc[i]) for c, i in pool])
    patient = np.array([f'{c}:{cohorts[c].meta.patient.iloc[i]}' for c, i in pool])
    block = np.array([f'{c}|{cohorts[c].meta.compartment.iloc[i]}' for c, i in pool])
    results = {}
    for name in sets:
        X = np.stack([features[c][name][i] for c, i in pool]).astype(np.float64)
        for b in np.unique(block):
            X[block == b] -= X[block == b].mean(0)
        for held in np.unique(study):
            tr, te = study != held, study == held
            if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
                continue
            head, C = fit_head(X[tr], y[tr], patient[tr], seed=3)
            s = head.decision_function(X[te])
            boot = group_bootstrap_auc(y[te], [s], patient[te], CFG['n_boot'], seed=5)[0]
            results.setdefault(held, {})[name] = dict(auroc=auroc(y[te], s), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                                                      n_test=int(te.sum()), n_train=int(tr.sum()), C=C)
    return results


# --- export + guarded inference ---------------------------------------------------------------------------
def _task_selection(task, cohort):
    sel = cohort.meta.compartment.isin(task['compartments']).to_numpy()
    if 'select' in task:
        sel &= np.asarray(task['select'](cohort.meta), dtype=bool)
    return sel


def export_heads(tasks_done, cohorts, features, path):
    """One head per (task, family) for single-compartment primary tasks. Two families are exported with their own evidence:
    `fmA_all` (the pre-specified foundation-model features) and `genes_z` (baseline-normalised genes, the deployable default)."""
    heads = {}
    for res in tasks_done:
        if res.get('status') != 'ok' or not res.get('primary'):
            continue
        task = next(t for t in TASKS if t['id'] == res['id'])
        if len(task['compartments']) > 1:
            continue                       # pooled-compartment heads need batch-wise centring: evaluated, not exported
        c = cohorts[task['cohort']]
        sel = _task_selection(task, c)
        y = c.meta.gvhd.to_numpy()[sel].astype(int)
        groups = c.meta.patient.astype(str).to_numpy()[sel]
        for fam in (PRIMARY_SET, DEPLOY_SET):
            X = features[task['cohort']][fam][sel].astype(np.float64)
            head, C = fit_head(X, y, groups)
            sc, lr = head.steps[0][1], head.steps[1][1]
            anchor = features[task['cohort']]['_anchors'].get(task['compartments'][0]) if fam == PRIMARY_SET else None
            ev = res['feature_sets'][fam]
            heads[f"{task['id']}::{fam}"] = dict(
                family=fam, task=task['id'], coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                scaler_mean=sc.mean_.tolist(), scaler_scale=sc.scale_.tolist(), C=float(C),
                anchor_mean=(anchor.tolist() if anchor is not None else None), compartments=task['compartments'],
                kind=task['kind'], question=task['question'], cohort=task['cohort'], platform=c.platform, scale=c.scale,
                n_train=int(len(y)), n_patients=int(len(set(groups))), cv_auroc=ev['auroc'], cv_auroc_ci95=ev['auroc_ci95'],
                perm_p=ev.get('perm_p'), verdict=verdict(res, fam),
                warning="Research use only. Fitted on one platform/compartment/protocol; not a validated clinical device.")
    atomic_torch_save({'format': 'hct_clinical_heads_v4', 'heads': heads}, path)
    return heads


def predict_clinical(expression, task_id, kind='log', family=DEPLOY_SET, model_path=None, heads_path=None):
    """Score new samples with a trained head.

    expression: DataFrame, rows = samples, columns = gene SYMBOLS. kind='log' means already on a log scale (log2 intensity,
    log ratio or log1p-CPM); kind='counts' applies log1p CPM (library size = column sum of the genes you provide, so give
    ALL genes). Refuses samples that measure <CFG['min_vocab_coverage'] of the vocabulary. family='genes_z' (default) scores
    baseline-normalised genes; 'fmA_all' scores the foundation-model features. A head was fitted on one platform and cell
    compartment; applying it elsewhere is outside its validity -- the evidence for each head is attached to the output.
    """
    model, baseline, snap = load_foundation_model(model_path or ROOT / 'model' / 'tcell_baseline_fm.pt')
    hs = torch_load(heads_path or ROOT / 'model' / 'clinical_heads.pt')
    key = f'{task_id}::{family}'
    if key not in hs['heads']:
        raise KeyError(f'unknown task/family {key!r}; available: {sorted(hs["heads"])}')
    h = hs['heads'][key]
    expr = expression.copy()
    expr.columns = [str(c).upper() for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    if kind == 'counts':
        lib = expr.sum(1).to_numpy(float)
        expr = pd.DataFrame(np.log1p(expr.to_numpy(float) / lib[:, None] * 1e6), index=expr.index, columns=expr.columns)
    elif kind != 'log':
        raise ValueError("kind must be 'log' or 'counts'")
    x = expr.reindex(columns=snap['genes']).to_numpy(dtype=np.float32)
    obs = np.isfinite(x)
    cover = obs.mean(1)
    if (cover < CFG['min_vocab_coverage']).any():
        raise ValueError(f'samples measure only {cover.min():.0%} of the {len(snap["genes"])}-gene vocabulary '
                         f'(minimum {CFG["min_vocab_coverage"]:.0%}); refusing to score')
    z, obs = to_model_space(x, obs, baseline)
    za = z
    if h.get('anchor_mean') is not None:                  # calibrate to the training cohort's platform (stored, label-free)
        za = np.where(obs, np.clip(z - np.array(h['anchor_mean'], dtype=np.float32), -CFG['z_clip'], CFG['z_clip']), 0.0).astype(np.float32)
    e = embed_profiles(model, za, obs)
    dev = deviation_scores(model, za, obs)
    X = np.hstack([e['pooled'], comp_features(e, dev)]).astype(np.float64) if family == PRIMARY_SET else z.astype(np.float64)
    Xs = (X - np.array(h['scaler_mean'])) / np.array(h['scaler_scale'])
    logit = Xs @ np.array(h['coef']) + h['intercept']
    out = pd.DataFrame({'risk_probability': 1 / (1 + np.exp(-logit)), 'logit': logit, 'vocabulary_coverage': cover,
                        'baseline_deviation': deviation_scores(model, z, obs)}, index=expression.index)
    p = _softmax(embed_profiles(model, z, obs)['lineage'])
    for k, name in enumerate(LINEAGE_CLASSES):
        out[f'lineage_{name}'] = p[:, k]
    out.attrs['evidence'] = {k: h[k] for k in ('family', 'question', 'kind', 'platform', 'compartments', 'n_train', 'n_patients',
                                              'cv_auroc', 'cv_auroc_ci95', 'perm_p', 'verdict', 'warning')}
    return out


def write_report(path, pre, transfer, tasks, loco):
    L = ['# HCT T-cell foundation model: run report', '',
         f'profile `{PROFILE}`; clinical data were never used for pretraining. All numbers are out-of-fold / held-out.', '',
         '## Pretraining on held-out healthy donors', '']
    r = pre.get('recon', {})
    L += [f"* masked-gene reconstruction MSE: model {r.get('model', float('nan')):.3f} | zero-predictor {r.get('zero_predictor', float('nan')):.3f} | PCA-ridge {r.get('pca_ridge', float('nan')):.3f}"]
    for k in ('fine', 'lineage'):
        if k in pre:
            d = pre[k]
            L += [f"* {k} composition JS-divergence (lower=better): model {d['js_model']:.3f} | PCA-ridge {d['js_pca_ridge']:.3f} | prior {d['js_prior']:.3f}; top-1 acc model {d['top1_accuracy_model']:.2f} vs PCA-ridge {d['top1_accuracy_pca_ridge']:.2f}"]
    if 'activation' in pre:
        L += [f"* activation R2: model {pre['activation']['r2_model']:.2f} | PCA-ridge {pre['activation']['r2_pca_ridge']:.2f}"]
    L += ['', '## Does the T-cell baseline recognise sorted bulk populations on other platforms? (one-vs-rest AUROC)', '',
          '| cohort | population | n | model | marker genes |', '|---|---|---|---|---|']
    for cid, res in transfer.items():
        for cls, d in res.items():
            if cls.startswith('_'):
                continue
            mg = d['auroc_marker_genes']
            L.append(f"| {cid} | {cls} | {d['n']} | {d['auroc_model']:.2f} | {'n/a' if mg is None else f'{mg:.2f}'} |")
    ok = [t for t in tasks if t.get('status') == 'ok']
    sig = lambda t, fam: t['feature_sets'][fam].get('perm_p', 1) < 0.05 and t['feature_sets'][fam]['auroc_ci95'][0] > 0.5
    L += ['', '## Bottom line (computed from the table below)', '',
          f"* tasks evaluated: {len(ok)}; above chance by permutation test AND bootstrap CI -- "
          f"all-platform-genes baseline: **{sum(sig(t, COMPARATOR_SET) for t in ok)}**, baseline-normalised genes (`genes_z`): **{sum(sig(t, DEPLOY_SET) for t in ok)}**, "
          f"pre-specified foundation-model features (`fmA_all`): **{sum(sig(t, PRIMARY_SET) for t in ok)}**.",
          f"* foundation-model features beat the untrained-encoder control (`scratchA`) in {sum(t['feature_sets'][PRIMARY_SET]['auroc'] > t['feature_sets']['scratchA_embed']['auroc'] for t in ok)} of {len(ok)} tasks, "
          f"and `genes_z` in {sum(t['feature_sets'][PRIMARY_SET]['auroc'] > t['feature_sets'][DEPLOY_SET]['auroc'] for t in ok)} of {len(ok)}.",
          '* a feature family that is not above chance here is NOT evidence that the model is useless at larger n -- only that it is not shown to help at this scale.']
    L += ['', '## Clinical tasks (patient-grouped repeated CV; AUROC [95% bootstrap CI])', '',
          'Primary (pre-specified) = `fmA_all`: anchored foundation-model features. Comparator = `genes_native`: all platform genes. '
          '`genes_z` = vocabulary genes after the T-cell-baseline normalisation (the deployable default); `genes_vocab` = same genes, native scale; '
          '`scratchA` = identical encoder WITHOUT pretraining; `hybrid` = genes_z + fmA_all; `fm_raw` = foundation model without anchoring; '
          '`resid` = gene-by-gene deviation from the healthy-T-cell expectation (exploratory, added after iteration-2 results; `resid0` = untrained encoder).', '',
          '| task | n (pts) | pos | fmA_all | genes_z | genes_native | hybrid | genes_vocab | scratchA | fm_raw | resid | resid0 | perm p: fmA / genes_z / native | confounder-only | verdict genes_z | verdict fmA_all |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t in tasks:
        if t.get('status') != 'ok':
            L.append(f"| {t['id']} | {t['n_samples']} ({t['n_patients']}) | {t['n_positive_samples']} | - | - | - | - | - | - | - | - | - | - | - | {t.get('status')} | |")
            continue
        fs = t['feature_sets']
        f = lambda n: f"{fs[n]['auroc']:.2f} [{fs[n]['auroc_ci95'][0]:.2f},{fs[n]['auroc_ci95'][1]:.2f}]"
        conf = ', '.join(f'{k} {v:.2f}' for k, v in t['confounder_only_auroc'].items()) or '-'
        pp = ' / '.join(f"{fs[n].get('perm_p', float('nan')):.3f}" for n in (PRIMARY_SET, DEPLOY_SET, COMPARATOR_SET))
        L.append(f"| {t['id']} | {t['n_samples']} ({t['n_patients']}) | {t['n_positive_samples']} | {f(PRIMARY_SET)} | {f(DEPLOY_SET)} | {f(COMPARATOR_SET)} | "
                 f"{fs['hybrid_genes_fmA']['auroc']:.2f} | {fs['genes_vocab']['auroc']:.2f} | {fs['scratchA_embed']['auroc']:.2f} | {fs['fm_all']['auroc']:.2f} | "
                 f"{fs['fmA_resid']['auroc']:.2f} | {fs['scratchA_resid']['auroc']:.2f} | {pp} | {conf} | {verdict(t, DEPLOY_SET)} | {verdict(t)} |")
    L += ['', '## Cross-study transfer (train on other studies, test on the held-out study)', '',
          '| held-out study | n | fmA_all | resid | hybrid | genes_z | scratchA |', '|---|---|---|---|---|---|---|']
    for held, d in loco.items():
        g = lambda n: (f"{d[n]['auroc']:.2f} [{d[n]['ci95'][0]:.2f},{d[n]['ci95'][1]:.2f}]" if n in d else '-')
        L.append(f"| {held} | {d[next(iter(d))]['n_test']} | {g('fmA_all')} | {g('fmA_resid')} | {g('hybrid_genes_fmA')} | {g('genes_z')} | {g('scratchA_embed')} |")
    Path(path).write_text('\n'.join(L) + '\n')


# ===========================================================================
# SELF-TESTS (offline, seconds): each guards a failure mode that mattered in earlier versions
# ===========================================================================
def _expect_raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    raise AssertionError('expected an exception')


class _StubCorpus:
    """Synthetic stand-in used ONLY by the self-tests to exercise the training loop; never used for real training."""
    V = 48

    def __init__(self, crash_after=None):
        self.src = [dict(id='stub', donor=np.zeros(10))]
        self.calls, self.crash_after = 0, crash_after

    def draw(self, split, rng, B):
        self.calls += 1
        if self.crash_after is not None and self.calls > self.crash_after:
            raise RuntimeError('simulated crash in the data worker')
        x = rng.normal(5.0, 1.0, size=(B, self.V)).astype(np.float32) + rng.normal(0, 1, size=(B, 1)).astype(np.float32)
        fine = rng.dirichlet(np.ones(K_FINE), size=B).astype(np.float32)
        lin = rng.dirichlet(np.ones(K_LIN), size=B).astype(np.float32)
        return dict(x=x, m=np.full(B, 50, np.float32), fine=fine, lineage=lin, act=rng.random(B).astype(np.float32),
                    valid=np.ones((B, 3), bool), source=np.zeros(B, np.int64))


def selftest():
    import tempfile
    log('self-tests ...')
    # 1 label / symbol parsing -------------------------------------------------------------------------------
    assert map_cl_label('CD8-positive, alpha-beta memory T cell, CD45RO-positive') == (None, 'CD8')      # 'CD45RO' is not CD4
    assert map_cl_label('naive thymus-derived CD4-positive, alpha-beta T cell') == ('CD4_naive', 'CD4_conventional')
    assert map_cl_label('CD14-positive monocyte') == (None, None)
    assert decode_gene_symbol('NM_1 // GENEA // d /// NM_2 // GENEB // d', 'gene_assignment') is None      # ambiguous -> excluded
    assert decode_gene_symbol('cd3e', 'Symbol') == 'CD3E' and decode_gene_symbol('CD3E /// CD3D', 'Gene Symbol') is None
    # 2 GEO parser keeps EVERY characteristics row ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        txt = ['!Sample_title\t"a"\t"b"', '!Sample_geo_accession\t"GSM1"\t"GSM2"', '!Sample_characteristics_ch1\t"patient_id: P1"\t"patient_id: P2"',
               '!Sample_characteristics_ch1\t"group: GVHD"\t"group: No GVHD"', '!series_matrix_table_begin', '"ID_REF"\t"GSM1"\t"GSM2"']
        txt += [f'"g{i}"\t{i}.5\t{i}.1' for i in range(120)] + ['!series_matrix_table_end']
        p = Path(td) / 'x.txt.gz'
        with gzip.open(p, 'wt') as f:
            f.write('\n'.join(txt) + '\n')
        meta, expr = parse_series_matrix(p)
        assert list(meta['patient_id']) == ['P1', 'P2'] and list(meta['group']) == ['GVHD', 'No GVHD'] and expr.shape == (120, 2)
    # 3 normalisation ------------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 300))
    ref = np.sort(rng.normal(size=300))
    ref_q = np.interp(np.linspace(0, 1, GRID), (np.arange(300) + 0.5) / 300, ref)
    obs = np.ones(x.shape, bool)
    assert np.allclose(quantile_normalize(x, obs, ref_q), quantile_normalize(np.exp(x) * 3 + 1, obs, ref_q), atol=1e-5)   # monotone-invariant
    obs2 = obs.copy(); obs2[:, ::4] = False
    q = quantile_normalize(x, obs2, ref_q)
    assert np.isnan(q[~obs2]).all() and np.isfinite(q[obs2]).all()
    # 4 no leakage of hidden values (by construction, but verified) --------------------------------------------
    hp = dict(n_genes=40, d_model=32, heads=4, layers=1, latents=8, dropout=0.0)
    m = build_model(40, hp).eval()
    z = torch.randn(5, 40); st = torch.zeros(5, 40, dtype=torch.long); st[:, ::3] = STATE_MASKED; st[:, 1::7] = STATE_MISSING
    lc = torch.full((5,), 2.0)
    with torch.no_grad():
        a = m(z, st, lc)
        z2 = z.clone(); z2[st != 0] += 100.0
        b = m(z2, st, lc)
    worst = {k: float((a[k] - b[k]).abs().max()) for k in a}
    # Inputs downstream of the embedding are bit-identical (hidden values are zeroed first); the residual ~1e-6 comes from
    # non-deterministic multi-thread reduction order. A genuine leak of a +100 shift would move outputs by O(1).
    assert all(v < 1e-4 for v in worst.values()), f'a hidden value influenced the output: {worst}'
    with torch.no_grad():
        z3 = z.clone(); z3[st == 0] += 0.5                               # control: changing OBSERVED values must be detected
        assert max(float((a[k] - m(z3, st, lc)[k]).abs().max()) for k in a) > 1e-3, 'leak test has no power'
    # 5 head/CV machinery: no leakage through patient groups; real signal is found ----------------------------
    rng = np.random.default_rng(1)
    n_pat, per = 24, 3
    pid = np.repeat(np.arange(n_pat), per)
    ylab = np.repeat(rng.permutation([0, 1] * (n_pat // 2)), per)
    leak = np.eye(n_pat)[pid]                                   # a feature that only identifies the PATIENT
    aucs = [auroc(ylab, cv_oof(leak + 1e-3 * rng.normal(size=leak.shape), ylab, pid.astype(str), r)) for r in range(4)]
    assert abs(np.mean(aucs) - 0.5) < 0.2, f'patient-identity feature leaked across folds (AUROC {np.mean(aucs):.2f})'
    Xs = rng.normal(size=(60, 40)); ys = (np.arange(60) % 2).astype(int); Xs[:, :4] += 1.5 * ys[:, None]
    sig = np.mean([auroc(ys, cv_oof(Xs, ys, np.arange(60).astype(str), r)) for r in range(3)])
    assert sig > 0.85, f'failed to find an obvious signal (AUROC {sig:.2f})'
    null = np.mean([auroc(ys, cv_oof(rng.normal(size=(60, 40)), ys, np.arange(60).astype(str), r)) for r in range(6)])
    assert abs(null - 0.5) < 0.12, f'pure-noise features gave AUROC {null:.2f}'
    # 6 integer patient ids are handled (previously silently produced empty splits) -----------------------------
    gm = pd.Series([1, 2, 3, 4]).astype(str).to_numpy()
    assert set(gm) == {'1', '2', '3', '4'}
    # 7 downloader integrity -----------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'src.bin'; src.write_bytes(os.urandom(5000))
        good = sha256_file(src)
        old_data = globals()['DATA']; globals()['DATA'] = Path(td) / 'data'
        try:
            p = fetch(src.as_uri(), globals()['DATA'] / 'a.bin', expected_sha256=good)
            assert p.read_bytes() == src.read_bytes()
            _expect_raises(lambda: fetch(src.as_uri(), globals()['DATA'] / 'b.bin', expected_sha256='0' * 64), RuntimeError)
        finally:
            globals()['DATA'] = old_data
    # 8 exact resumption of pretraining after a crash ------------------------------------------------------------
    saved = {k: CFG[k] for k in ('d_model', 'heads', 'layers', 'latents', 'dropout', 'pretrain_steps', 'batch_size', 'eval_every',
                                 'ckpt_every', 'warmup_steps', 'patience_evals', 'val_pseudobulks', 'lr')}
    CFG.update(d_model=32, heads=4, layers=1, latents=8, dropout=0.1, pretrain_steps=14, batch_size=8, eval_every=7, ckpt_every=3,
               warmup_steps=2, patience_evals=50, val_pseudobulks=16, lr=1e-3)
    old_work = WORK
    base = dict(ref_q=ref_q, mu=np.zeros(_StubCorpus.V), sd=np.ones(_StubCorpus.V), raw_sd=np.ones(_StubCorpus.V))
    base['ref_q'] = np.interp(np.linspace(0, 1, GRID), (np.arange(_StubCorpus.V) + 0.5) / _StubCorpus.V, np.sort(np.random.default_rng(2).normal(5, 1, _StubCorpus.V)))
    try:
        with tempfile.TemporaryDirectory() as td:
            globals()['WORK'] = Path(td) / 'a'
            mA, _ = pretrain(_StubCorpus(), base, 'vk')
            globals()['WORK'] = Path(td) / 'b'
            try:
                pretrain(_StubCorpus(crash_after=9), base, 'vk')
                raise AssertionError('crash did not propagate')
            except RuntimeError as e:
                assert 'simulated crash' in str(e)
            mB, _ = pretrain(_StubCorpus(), base, 'vk')
            same = all(torch.equal(a, b) for a, b in zip(mA.state_dict().values(), mB.state_dict().values()))
            assert same, 'resumed training diverged from the uninterrupted run'
    finally:
        globals()['WORK'] = old_work
        CFG.update(saved)
    log('self-tests passed (labels, parsers, normalisation, leakage, grouped CV, downloads, exact resume)')


# ===========================================================================
# CLINICAL STAGE + MAIN
# ===========================================================================
def stage_clinical(art, vocab):
    ensure_dirs()
    model, baseline, snap = load_foundation_model(art)
    cohorts = load_cohorts(vocab)
    torch.manual_seed(CFG['seed'] + 1)
    scratch = build_model(len(vocab['genes']), snap['hp'])            # identical architecture, NO pretraining (ablation)
    feats = {cid: featurize(c, model, baseline, scratch) for cid, c in cohorts.items()}
    transfer = transfer_checks(cohorts, feats)
    save_json(OUT / 'transfer_checks.json', transfer)
    results = []
    for task in TASKS:
        t0 = time.time()
        res = evaluate_task(task, cohorts[task['cohort']], feats[task['cohort']])
        results.append(res)
        if res.get('status') == 'ok':
            fs = res['feature_sets']
            pr = fs[PRIMARY_SET]
            log(f"{task['id']}: n={res['n_samples']} pos={res['n_positive_samples']} | {PRIMARY_SET} {pr['auroc']:.2f} "
                f"[{pr['auroc_ci95'][0]:.2f},{pr['auroc_ci95'][1]:.2f}] | hybrid {fs['hybrid_genes_fmA']['auroc']:.2f} | "
                f"{COMPARATOR_SET} {fs[COMPARATOR_SET]['auroc']:.2f} | genes_z {fs['genes_z']['auroc']:.2f} | scratch {fs['scratchA_embed']['auroc']:.2f} "
                f"| perm p fmA={pr['perm_p']:.3f} genes_z={fs[DEPLOY_SET]['perm_p']:.3f} | genes_z: {verdict(res, DEPLOY_SET)} ({time.time() - t0:.0f}s)")
        else:
            log(f"{task['id']}: {res['status']}")
    loco = loco_transfer(cohorts, feats)
    save_json(OUT / 'clinical_results.json', {'tasks': [{k: v for k, v in r.items() if k != 'oof'} for r in results], 'loco': loco})
    heads = export_heads(results, cohorts, feats, ROOT / 'model' / 'clinical_heads.pt')
    write_report(OUT / 'RUN_REPORT.md', snap['pretrain_report'], transfer, results, loco)
    log(f'clinical stage complete: {len(heads)} heads exported; report -> {OUT / "RUN_REPORT.md"}')
    return results


def main(mode='all'):
    ensure_dirs()
    log(f'HCT T-cell foundation model | profile={PROFILE} | root={ROOT}')
    if mode in ('selftest', 'all'):
        selftest()
        if mode == 'selftest':
            return
    save_json(OUT / 'run_configuration.json', {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()})
    art, vocab = stage_pretrain()
    if mode in ('clinical', 'all'):
        stage_clinical(art, vocab)
    log('DONE. Foundation model: ' + str(art))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
