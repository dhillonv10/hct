"""
HCT-FORMER M4: SINGLE-FILE AUTOMATIC MPS TRAINER
=====================================================================
RELEASE: M4-ONEFILE-1.1 / v3.2 loaders, MPS and latent cross-attention

STRICT DATA POLICY
------------------
1. All TRAINING and VALIDATION observations MUST originate from measured, public
   individual-level molecular datasets. No synthetic subjects, simulated assays,
   randomly generated expression matrices, synthetic fallback, or smoke-data mode.
2. If a download, clinical label, assay value, time axis, or dataset schema fails
   a QC check, the corresponding track STOPS without exporting a false model.
3. Random masking of REAL, measured values (for self-supervision) is allowed;
   random weight initialization and stochastic batching are normal model training,
   not synthetic training data.
4. An optional user-provided HCT-protein matrix is NEVER included by default and
   must consist of actual measured, patient-linked proteomics, with verified labels.

WHAT EXACTLY ARE WE TRAINING? (TWO DISTINCT TRACKS)
-------------------------------------------------
MODEL A / multi-study real HCT gene expression + GSE73809 GVHD labels:
  Stage A1: MASKED EXPRESSION RECONSTRUCTION (SELF-SUPERVISED)
    Inputs: GPL-annotated gene-symbol expression from several human HCT studies.
    Randomly hide 20% of OBSERVED training gene values; predict those same
    measured values. Train loss = mean squared error on MASKED OBSERVED genes.
    Goal: learn probe-expression structure without needing outcome labels.
  Stage A2: CONTEMPORANEOUS GVHD CASE-CONTROL CLASSIFICATION (SUPERVISED)
    Inputs: same samples, real GVHD-present vs GVHD-absent sample annotations.
    Label: 1 = GVHD sample; 0 = non-GVHD sample.
    Train loss = class-weighted binary cross entropy on GVHD logits.
    Goal: predict contemporaneous GVHD association, NOT upcoming onset/severity.
    Validation: validation BCE controls early stopping; the test split is used ONCE;
                report ROC-AUC, balanced accuracy, Brier; compare logistic regression.
    Caveat: GEO sample IDs, not independently confirmed distinct patient IDs.

MODEL B / PROTEOMIC TRANSFORMER (Gisby COVID/dialysis Olink plasma + serum):
  Stage B1: MASKED PROTEIN RECONSTRUCTION (SELF-SUPERVISED)
    Inputs: measured public plasma and serum Olink NPX; both biofluids contribute to SSL.
    Randomly hide 20% of OBSERVED protein values, then reconstruct them.
    Train loss = MSE only over hidden, measured proteins.
    Goal: learn transferable covariation patterns between circulating proteins.
  Stage B2: NEXT-MEASURED-VISIT PROTEIN FORECASTING (SELF-SUPERVISED REGRESSION)
    Inputs: up to four previous *plasma* patient visits, their day offsets, and forecast horizon.
    The serum CSV has NO timestamps: serum contributes to masked-value SSL only.
    Predict the NEXT recorded protein panel in standardized assay space.
    Train loss = MSE on proteins OBSERVED at the target visit.
    Temporal attention sees only visits prior to the target visit.
    Validation: held-out PATIENTS, validation MSE for checkpoint selection;
                held-out test MSE vs last-observation-carried-forward.
    Caveat: NON-HCT PROTEOMICS; NOT a clinical GVHD predictor.

MAIN SCIENTIFIC GOAL
--------------------
Prove a reproducible real-data multi-stage pipeline and export two genuinely
trained checkpoints. A validated, longitudinal GVHD proteomics predictor is a
FUTURE objective and requires paired HCT plasma omics + GVHD onset/grade labels.
The present tracks are separate encoders; no patient-level cross-cohort pairing
is assumed or fabricated. Neither loads official ProLM or Qwen weights.

USAGE (no flags or secondary files required):
  python3 HCTFormer_M4_ALL_IN_ONE.py

Requirements install automatically into a private Python environment IF missing.
Run settings appear under EDITABLE TRAINING SETTINGS below, within THIS file.
Data, logs, and trained weights are saved in ~/HCTFormer_M4_Run/.
Everything needed to DOWNLOAD and TRAIN is implemented in this single file;
training still creates downloaded data files, output checkpoints and reports.

"""
# SINGLE-FILE ENVIRONMENT BOOTSTRAP (runs BEFORE optional heavy imports).
# The user does not need setup_mac.sh, requirements.txt, shell flags, or notebooks.
import os as _os_boot, sys as _sys_boot, platform as _platform_boot
_os_boot.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

def _bootstrap_mac_dependencies():
    import importlib.util, subprocess, venv
    from pathlib import Path
    if _platform_boot.system() != 'Darwin' or _platform_boot.machine() != 'arm64':
        raise RuntimeError('This single-file edition requires a native arm64 Python on Apple Silicon macOS. '
                           'Install Apple Silicon Python (not Rosetta) and rerun: python3 HCTFormer_M4_ALL_IN_ONE.py')
    if _sys_boot.version_info < (3, 11) or _sys_boot.version_info >= (3, 14):
        raise RuntimeError('Use Apple Silicon Python 3.11–3.13 for this release.')
    required = {
        'torch': 'torch>=2.4',
        'numpy': 'numpy>=1.26,<3',
        'pandas': 'pandas>=2.1,<3',
        'scipy': 'scipy>=1.11,<2',
        'sklearn': 'scikit-learn>=1.4,<2',
        'matplotlib': 'matplotlib>=3.8,<4',
        'anndata': 'anndata>=0.10,<0.13',
        'h5py': 'h5py>=3.10',
    }
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        print('[setup] Required Python packages already available; using current environment.', flush=True)
        return
    venv_root = Path.home() / '.cache' / 'hctformer_m4_singlefile_env'
    python = venv_root / 'bin' / 'python'
    if not python.exists():
        print(f'[setup] Creating private Python environment: {venv_root}', flush=True)
        venv.create(venv_root, with_pip=True)
    probe = subprocess.run(
        [str(python), '-c', 'import torch,numpy,pandas,scipy,sklearn,matplotlib,anndata,h5py'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if probe.returncode:
        print('[setup] Installing Python dependencies (one-time; internet required).', flush=True)
        subprocess.check_call([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([str(python), '-m', 'pip', 'install', *required.values()])
    print('[setup] Switching automatically to private environment.', flush=True)
    _os_boot.execv(str(python), [str(python), str(Path(__file__).resolve())])

if __name__ == '__main__':
    _bootstrap_mac_dependencies()

import os, re, io, csv, gzip, json, time, math, hashlib, random, copy, warnings, traceback, sys, platform, gc
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, brier_score_loss, mean_squared_error
import matplotlib.pyplot as plt

# ===========================================================================
# EDITABLE TRAINING SETTINGS: ALL CONFIGURATION LIVES HERE; NO CLI PARAMETERS
# ===========================================================================
CFG = {
    'seed': 20261021,
    'run_mode': 'REAL_PUBLIC_DATA_ONLY',
    'synthetic_training_allowed': False,
    'gene_features': 1024,
    'protein_features': 512,
    'dim': 192,
    'heads': 4,
    'layers': 4,
    'dropout': 0.12,
    'latent_tokens': 48,
    # Randomly sample REAL observations per SSL epoch for faster convergence.
    # 0 trains on all available observations each SSL epoch.
    'ssl_steps_per_epoch': 350,
    'progress_every': 25,
    'step_ckpt_every': 100,
    'num_workers': 0,
    'epochs_ssl': 35,
    'epochs_task': 35,
    'batch_size': 32,
    'lr_ssl': 2e-4,
    'lr_task': 1e-4,
    'mask_fraction': 0.20,
    'patience': 10,
    'max_delta_days': 150,
    'max_history_visits': 4,
    'download_retries': 4,
    # No external patient data are required or silently ingested.
    'external_gvhd_proteins_csv': '',
    'min_external_hct_samples': 65,
    'min_external_hct_studies': 2,
    'require_serum': True,
    'use_hct_single_cell': True,
    'require_hct_single_cell': True,
    'max_hct_single_cells': 50000,
    'require_external_hct': True,
}
# Reuse the data cache of the prior Mac bundle if one already exists.
ROOT = Path.home() / 'HCTFormer_M4_Run'
ROOT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'data'
DATA.mkdir(parents=True, exist_ok=True)
# MPS-only for training. NEVER silently downgrade to CPU.
if getattr(torch.backends, 'mps', None) is None or not torch.backends.mps.is_available():
    raise RuntimeError('PyTorch MPS is not available. Ensure this is an arm64 Mac '
                       'and install an Apple Silicon-compatible PyTorch build.')
DEVICE = torch.device('mps')

# All objectives are saved alongside each run, not merely explained in prose.
TRAINING_OBJECTIVES = {
  'policy': {
    'training_data': 'Exclusively real, measured public individual-level data',
    'synthetic_examples_permitted': False,
    'download_failure': 'Abort affected track, no replacement data',
    'masking': 'Randomly mask observed REAL measurements; no synthetic samples',
    'clinical_claim': 'Research methods demonstration; not a validated GVHD risk tool',
  },
  'model_A_HCT_transcriptomic': {
    'dataset': 'GSE73809 24 labeled HCT microarrays; external HCT arrays (GSE103569, GSE56495, GSE7510) and optional GSE295359 real T cells for SELF-SUPERVISED pretraining ONLY',
    'pretraining': {
      'objective': 'Recover randomly hidden real gene measurements (shared GPL-mapped features)',
      'inputs': 'GSE73809 training-only + external HCT studies with platform probe-to-gene mapping, train-split standardized; missing genes masked',
      'target': 'Same sample’s observed expression at hidden probe positions',
      'loss': 'Masked-observed mean squared error (MSE)',
      'mask_rate': CFG['mask_fraction'],
    },
    'fine_tuning': {
      'objective': 'Classify contemporaneous GVHD vs no GVHD in HCT samples',
      'target': 'Binary GEO sample-status label (GVHD=1, no GVHD=0)',
      'loss': 'Positive-class-weighted BCEWithLogitsLoss',
      'model_selection': 'Lowest validation binary cross-entropy; no test selection',
      'heldout_metrics': ['ROC AUC','balanced accuracy','Brier score'],
      'baseline': 'Train-only logistic regression',
      'NOT_predicted': 'Future GVHD, onset day, organ, steroid response or severity',
      'split_limitation': 'Unique patient IDs not independently verified; split by sample',
    },
  },
  'model_B_proteomic_temporal': {
    'dataset': 'Gisby public Olink plasma (serial) + serum (untimed SSL only); NOT HCT',
    'pretraining': {
      'objective': 'Recover randomly hidden real log2 NPX measurements',
      'target': 'Observed log2 NPX protein values at masked positions',
      'loss': 'Masked-observed mean squared error (MSE)',
      'mask_rate': CFG['mask_fraction'],
    },
    'temporal_training': {
      'objective': 'Forecast the next recorded protein profile from chronological past visits',
      'inputs': 'Up to N prior plasma visits from one patient, day offsets and horizon; serum metadata has no collection times, so serum excluded from forecasting',
      'target': 'Protein NPX vector at next MEASURED visit (not fixed day +14)',
      'loss': 'MSE over proteins actually measured at forecast visit',
      'split_unit': 'Individual_ID across both cohorts; serial plasma patient groups form test/validation splits, serum-only patients SSL-train only',
      'model_selection': 'Lowest validation target-protein MSE',
      'heldout_metrics': ['Test observed-protein MSE','Persistence-baseline MSE'],
      'baseline': 'Last observed protein profile (persistence)',
      'NOT_predicted': 'Acute GVHD, GVHD severity, or HCT treatment response',
    },
  },
}

def log(message):
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {message}"
    print(line, flush=True)
    with (ROOT / 'training_log.txt').open('a') as f: f.write(line + '\n')

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f: json.dump(obj, f, indent=2, default=str, allow_nan=False)
    os.replace(tmp, path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'mps' and hasattr(torch.mps,'manual_seed'):
        torch.mps.manual_seed(seed)


def fetch(urls, dest, min_bytes=1000):
    """Download atomically, retry/fallback, DO NOT fabricate on failure."""
    dest = Path(dest)
    # A cached file is usable ONLY when this downloader recorded its exact
    # approved origin URL, content SHA256, and byte count. An anonymous CSV or
    # an old simulated file copied into the cache is NEVER trusted.
    sidecar = dest.with_name(dest.name + '.source.json')
    if dest.exists() and sidecar.exists() and dest.stat().st_size >= min_bytes:
        try:
            claim = json.loads(sidecar.read_text())
            if (claim['url'] in urls and claim['sha256'] == sha256(dest)
                    and claim['bytes'] == dest.stat().st_size
                    and claim.get('source_type') == 'public_measured_source'):
                log(f"Verified public-data cache: {dest.name} ({dest.stat().st_size:,} bytes)")
                return dest
            log(f'Ignoring unverified or altered cache: {dest.name}')
        except (OSError, ValueError, KeyError, TypeError):
            log(f'Ignoring malformed data-provenance record: {dest.name}')
    elif dest.exists():
        log(f'Ignoring cache without matching public-source record: {dest.name}')
    err = []
    for url in urls:
        for attempt in range(CFG['download_retries']):
            tmp = dest.with_suffix(dest.suffix + '.partial')
            try:
                req = Request(url, headers={'User-Agent':'HCTFormer research pipeline/1.0', 'Accept':'*/*'})
                with urlopen(req, timeout=100) as res, open(tmp, 'wb') as f:
                    # Visible download progress; avoid silent multi-GB transfers in Colab.
                    total = int(res.headers.get('Content-Length', 0) or 0)
                    nbytes = 0
                    next_report = 64 * 1024 * 1024
                    while True:
                        chunk = res.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        nbytes += len(chunk)
                        if nbytes >= next_report:
                            if total:
                                log(f'Downloading {dest.name}: {nbytes / 1e9:.2f} / {total / 1e9:.2f} GB')
                            else:
                                log(f'Downloading {dest.name}: {nbytes / 1e9:.2f} GB')
                            next_report += 64 * 1024 * 1024
                if tmp.stat().st_size < min_bytes:
                    raise RuntimeError(f"File implausibly small ({tmp.stat().st_size} bytes)")
                os.replace(tmp, dest)
                save_json(sidecar,{'url':url,'sha256':sha256(dest),
                                   'bytes':dest.stat().st_size,
                                   'source_type':'public_measured_source',
                                   'downloaded_utc':datetime.now(timezone.utc).isoformat()})
                log(f"Downloaded verified public data: {dest.name} ({dest.stat().st_size:,} bytes)")
                return dest
            except Exception as ex:
                err.append(f'{url} : {str(ex)[:140]}')
                tmp.unlink(missing_ok=True)
                log(f"Download retry {attempt+1}: {type(ex).__name__}: {str(ex)[:120]}")
                time.sleep(min(2**attempt, 8))
    raise RuntimeError('All verified-source download routes failed. No synthetic fallback in REAL mode.\n'+'\n'.join(err[-5:]))

class SampleSet(Dataset):
    def __init__(self, x, observed, indices):
        self.x=torch.from_numpy(x[indices].astype('float32'))
        self.obs=torch.from_numpy(observed[indices].astype('bool'))
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.obs[i]

class HistorySet(Dataset):
    """Chronological history of 1..K visits, with left padding, target next visit."""
    def __init__(self, x, observed, histories):
        self.x=torch.from_numpy(x.astype('float32'))
        self.obs=torch.from_numpy(observed.astype('bool'))
        self.histories=list(histories)
        self.k=CFG['max_history_visits']
    def __len__(self):return len(self.histories)
    def __getitem__(self,i):
        indices, relative_days, target, delta=self.histories[i]
        f=self.x.shape[1]
        h=torch.zeros((self.k,f),dtype=torch.float32)
        o=torch.zeros((self.k,f),dtype=torch.bool)
        days=torch.zeros(self.k,dtype=torch.float32)
        pad=torch.ones(self.k,dtype=torch.bool)
        length=len(indices); assert 1<=length<=self.k
        h[-length:]=self.x[indices];o[-length:]=self.obs[indices]
        days[-length:]=torch.tensor(relative_days,dtype=torch.float32)
        pad[-length:]=False
        last=int(indices[-1])
        return h,o,days,pad,self.x[last],self.x[target],self.obs[target],torch.tensor(delta,dtype=torch.float32)

class LabelSet(Dataset):
    def __init__(self,x,observed,y,indices):
        self.x=torch.from_numpy(x[indices].astype('float32'))
        self.obs=torch.from_numpy(observed[indices].astype('bool'))
        self.y=torch.from_numpy(np.asarray(y)[indices].astype('float32'))
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.obs[i],self.y[i]

class OmicsTransformer(nn.Module):
    """MPS-efficient cross-attention bottleneck over ALL measured feature tokens.

    A fixed vocabulary of feature-ID/value tokens is read by L learned latent
    queries, followed by self-attention over just L tokens rather than F tokens.
    Reconstruction cross-attends F gene/protein queries to L latents. This changes
    the architecture from v3.2; the two scientific training objectives and data
    acquisition protocols are retained. Old v3.2 checkpoints are NOT compatible.

    Complexity of cross-attention ~O(F * L) rather than O(F^2) for each layer.
    """
    def __init__(self, n_features, dim=192, heads=4, layers=4,
                 dropout=0.12, latent_tokens=48):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.latent_tokens = latent_tokens
        self.ids = nn.Embedding(n_features, dim)
        self.value = nn.Linear(1, dim)
        self.unknown = nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.mask_token = nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.latents = nn.Parameter(torch.randn(1,latent_tokens,dim)*0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.cross_in = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_out = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim,heads,dim*3,dropout,
                                           batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        temporal_layer = nn.TransformerEncoderLayer(dim,heads,dim*2,dropout,
                           batch_first=True,activation='gelu',norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer,1,enable_nested_tensor=False)
        self.temporal_time = nn.Linear(1,dim)
        self.temporal_norm = nn.LayerNorm(dim)
        self.reconstruction = nn.Linear(dim,1)
        self.classification = nn.Linear(dim,1)
        self.forecast_net = nn.Sequential(nn.Linear(dim+1,dim*2),nn.GELU(),
                                         nn.Dropout(dropout),nn.Linear(dim*2,n_features))
        nn.init.zeros_(self.forecast_net[-1].weight)
        nn.init.zeros_(self.forecast_net[-1].bias)

    def forward(self,x,observed,mask=None):
        b,f=x.shape
        assert f == self.n_features, f'Expected {self.n_features} features; received {f}'
        if mask is None:mask=torch.zeros_like(observed,dtype=torch.bool)
        ids=self.ids(torch.arange(f,device=x.device))[None,:,:].expand(b,-1,-1)
        token=ids+self.value(x.unsqueeze(-1))
        token=torch.where((~observed).unsqueeze(-1),self.unknown,token)
        token=torch.where(mask.unsqueeze(-1),self.mask_token,token)
        token=self.input_norm(token)
        lat=self.latents.expand(b,-1,-1)
        lat=lat+self.cross_in(lat,token,token,need_weights=False)[0]
        lat=self.norm(self.encoder(lat))
        pooled=lat.mean(dim=1)
        query=self.input_norm(ids)
        recon_tokens=query+self.cross_out(query,lat,lat,need_weights=False)[0]
        pred=self.reconstruction(recon_tokens).squeeze(-1)
        return pooled,pred

    def encode_history(self, history, observed, days, padded):
        b,t,f=history.shape
        z,_=self.forward(history.reshape(b*t,f),observed.reshape(b*t,f))
        z=z.reshape(b,t,self.dim)
        time_embedding=torch.asinh(days/14.0).unsqueeze(-1)
        sequence=z+self.temporal_time(time_embedding)
        causal=torch.triu(torch.ones(t,t,device=history.device,dtype=torch.bool),diagonal=1)
        contextual=self.temporal_encoder(sequence,mask=causal,src_key_padding_mask=padded)
        return self.temporal_norm(contextual[:,-1,:])

    def predict_future(self,latent,x,gap):
        scaled=torch.log1p(gap.clamp(min=0,max=365)).unsqueeze(-1)/math.log1p(365)
        return x+self.forecast_net(torch.cat([latent,scaled],dim=-1))


def mkmodel(f):
    return OmicsTransformer(f,CFG['dim'],CFG['heads'],CFG['layers'],
                             CFG['dropout'],CFG['latent_tokens']).to(DEVICE)

def masked_batch(x, observed, frac, generator=None):
    noise=torch.rand(x.shape,device=x.device,generator=generator)
    m=(noise < frac)&observed
    # Make sure at least one truly measured value is masked per sample.
    for i in range(x.size(0)):
        if not bool(m[i].any()) and bool(observed[i].any()):
            m[i,observed[i].nonzero(as_tuple=True)[0][0]]=True
    return m

def masked_loss(model,x,obs,gen=None):
    mask=masked_batch(x,obs,CFG['mask_fraction'],gen)
    _,rec=model(x,obs,mask)
    return ((rec-x).square()*mask.float()).sum()/mask.float().sum().clamp_min(1),mask

def optimizer_for(model,mode):
    return torch.optim.AdamW(model.parameters(),lr=CFG['lr_ssl'] if mode=='ssl' else CFG['lr_task'],weight_decay=1e-3)

def checkpoint_write(path, payload):
    p=Path(path);p.parent.mkdir(parents=True, exist_ok=True)
    temp=p.with_name(p.name+'.partial')
    torch.save(payload,temp);os.replace(temp,p)

def chk_read(path):
    try: return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(path,map_location='cpu')

def _cpu_tree(obj):
    """Put checkpoint tensors on CPU to make archives portable Mac→Linux HPC."""
    if torch.is_tensor(obj):return obj.detach().cpu().clone()
    if isinstance(obj,dict):return {k:_cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list):return [_cpu_tree(v) for v in obj]
    if isinstance(obj,tuple):return tuple(_cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def _optimizer_to_device(opt):
    for state in opt.state.values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.to(DEVICE)


def _memory_note():
    if DEVICE.type=='mps' and hasattr(torch.mps,'current_allocated_memory'):
        return f' | MPS allocated={torch.mps.current_allocated_memory()/2**30:.2f} GiB'
    return ''


def _epoch_indices(n, epoch, phase):
    """Real-data indices only; small SSL epochs rotate stochastic subjects/cells."""
    rng=np.random.default_rng(CFG['seed']+3181*epoch+(11 if phase=='ssl' else 211))
    order=rng.permutation(n)
    if phase=='ssl' and CFG['ssl_steps_per_epoch']>0:
        take=min(n,CFG['ssl_steps_per_epoch']*CFG['batch_size'])
        return order[:take].tolist()
    return order.tolist()


def fit(model,train_ds,val_ds,phase,work,steps,identity,features,scaler,extra=None):
    """Train on real observations with MPS, visible batches, exact resumable batch position.

    After a hard stop the latest checkpoint contains the next batch index and
    matching optimizer state. Shuffling is recreated deterministically per epoch.
    Masking/dropout RNG states are captured at step checkpoints where supported.
    """
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    latest=work/(phase+'_latest.pt');best=work/(phase+'_best.pt')
    epoch_start_file=work/(phase+'_epoch_start.pt')
    opt=optimizer_for(model,'ssl' if phase=='ssl' else 'task')
    epochs=CFG['epochs_ssl'] if phase=='ssl' else CFG['epochs_task']
    start=0;stale=0;best_score=float('inf');history=[]
    batch_size=min(CFG['batch_size'],max(1,len(train_ds)))
    partial=None
    if latest.exists():
        snap=chk_read(latest)
        if snap.get('identity')!=identity:
            raise RuntimeError(f'{phase}: incompatible data/model fingerprint in {latest}. '
                'Use a new output directory (change ROOT at top of THIS file) for a different model configuration.')
        model.load_state_dict(snap['model']);opt.load_state_dict(snap['optimizer'])
        _optimizer_to_device(opt)
        start=snap['epoch']+1
        stale=snap['stale'];best_score=snap['best_score'];history=snap['history']
        batch_size=snap.get('batch_size',batch_size)
        partial=snap.get('partial_epoch')
        if partial is not None:
            start=partial['epoch']; batch_size=partial['batch_size']
        log(f'{work.name}/{phase}: resuming at epoch {start+1}, '
            f'batch {partial["next_batch"] if partial else 0}, batch_size={batch_size}')
    if start>=epochs or stale>=CFG['patience']:
        if best.exists():model.load_state_dict(chk_read(best)['model'])
        return history

    for epoch in range(start,epochs):
        if partial is None or partial.get('epoch')!=epoch:
            checkpoint_write(epoch_start_file,{'model':_cpu_tree(model.state_dict()),
                'optimizer':_cpu_tree(opt.state_dict()),'identity':identity,'epoch':epoch})
        elif not epoch_start_file.exists():
            raise RuntimeError('Mid-epoch resume needs epoch-start rollback checkpoint; file missing.')
        finished=False
        for attempt in range(4):
            retry_state=_cpu_tree(model.state_dict())
            retry_optimizer=_cpu_tree(opt.state_dict())
            order=_epoch_indices(len(train_ds),epoch,phase)
            # If an OOM requires halving batch size, use entire shuffled epoch.
            use_partial=partial is not None and partial.get('epoch')==epoch and attempt==0
            next_batch=int(partial['next_batch']) if use_partial else 0
            old_total=float(partial['loss_weighted']) if use_partial else 0.
            old_n=int(partial['num_seen']) if use_partial else 0
            if use_partial:
                if partial.get('order_hash')!=stable_hash(order):
                    raise RuntimeError('Cannot resume mid-epoch: sampled real-cell order changed.')
            else:seed_all(CFG['seed']+epoch)
            sampler=Subset(train_ds,order)
            loader=DataLoader(sampler,batch_size=batch_size,shuffle=False,
                              num_workers=CFG['num_workers'],pin_memory=False)
            if use_partial and partial.get('rng_cpu') is not None:
                torch.set_rng_state(partial['rng_cpu'])
                if DEVICE.type=='mps' and partial.get('rng_mps') is not None:
                    torch.mps.set_rng_state(partial['rng_mps'])
            model.train()
            total=torch.tensor(old_total,device=DEVICE,dtype=torch.float32)
            n=old_n
            n_batches=len(loader)
            epoch_started=time.monotonic()
            log(f'{work.name}/{phase}: epoch {epoch+1}/{epochs} '
                f'| {len(order):,} REAL samples/visits | {n_batches:,} batches '
                f'| batch={batch_size} | device={DEVICE}{_memory_note()}')
            try:
                for bi,batch in enumerate(loader):
                    if bi<next_batch:continue
                    batch=[v.to(DEVICE) for v in batch]
                    opt.zero_grad(set_to_none=True)
                    loss=steps(model,batch,True)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                    opt.step()
                    bs=len(batch[0]);total+=loss.detach()*bs;n+=bs
                    del loss,batch
                    if ((bi+1)%CFG['progress_every']==0 or bi+1==n_batches):
                        elapsed=max(1,time.monotonic()-epoch_started)
                        speed=(bi+1-next_batch)/elapsed
                        log(f'  {work.name}/{phase} E{epoch+1}: batch {bi+1:,}/{n_batches:,} '
                            f'| loss~{float(total)/max(1,n):.4f} '
                            f'| {speed:.1f} batch/s{_memory_note()}')
                    if CFG['step_ckpt_every']>0 and (bi+1)%CFG['step_ckpt_every']==0 and bi+1<n_batches:
                        snapshot={'model':_cpu_tree(model.state_dict()),
                            'optimizer':_cpu_tree(opt.state_dict()),'identity':identity,
                            'epoch':epoch-1,'best_score':best_score,'stale':stale,
                            'history':history,'batch_size':batch_size,
                            'partial_epoch':{'epoch':epoch,'next_batch':bi+1,
                                             'batch_size':batch_size,'loss_weighted':float(total),
                                             'num_seen':n,'order_hash':stable_hash(order),
                                             'rng_cpu':torch.get_rng_state(),
                                             'rng_mps':torch.mps.get_rng_state() if DEVICE.type=='mps' else None}}
                        checkpoint_write(latest,snapshot)
                        del snapshot
                finished=True
                break
            except (RuntimeError, FloatingPointError) as ex:
                string=str(ex).lower()
                is_oom=any(v in string for v in ['out of memory','mps backend out','not enough memory',
                        'mps allocator','failed to allocate'])
                is_bad=isinstance(ex,FloatingPointError) or 'nonfinite' in string
                if not(is_oom or is_bad):raise
                log(f'{work.name}/{phase}: recoverable {type(ex).__name__}: {str(ex)[:200]}')
                if use_partial:
                    epoch_seed=chk_read(epoch_start_file)
                    if epoch_seed.get('identity')!=identity or epoch_seed.get('epoch')!=epoch:
                        raise RuntimeError('Epoch-start rollback checkpoint fingerprint mismatch.')
                    model.load_state_dict(epoch_seed['model']);opt.load_state_dict(epoch_seed['optimizer'])
                    log('Recovery after partial resume: rolling back to complete epoch start.')
                else:
                    model.load_state_dict(retry_state);opt.load_state_dict(retry_optimizer)
                _optimizer_to_device(opt)
                partial=None
                gc.collect()
                if DEVICE.type=='mps':torch.mps.empty_cache()
                if is_oom and batch_size>1:
                    batch_size=max(1,batch_size//2)
                else:
                    for group in opt.param_groups:group['lr']*=0.5
                log(f'Retrying epoch {epoch+1} from start, batch={batch_size}, lr={opt.param_groups[0]["lr"]:.2e}')
                if attempt==3:raise RuntimeError('Four recovery attempts failed; stop training.') from ex
        if not finished:raise RuntimeError('Epoch did not complete')
        partial=None
        model.eval();val_loss=0.;vn=0
        with torch.no_grad():
            for bi,batch in enumerate(DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)):
                batch=[v.to(DEVICE) for v in batch]
                if phase=='ssl':
                    # Repeatable (non-synthetic) validation masks per batch.
                    torch.manual_seed(CFG['seed']+7183+bi)
                    if DEVICE.type=='mps':torch.mps.manual_seed(CFG['seed']+7183+bi)
                v=steps(model,batch,False)
                if not bool(torch.isfinite(v)):raise RuntimeError('Nonfinite validation loss; stopping.')
                val_loss+=float(v)*len(batch[0]);vn+=len(batch[0])
        val_loss/=max(1,vn)
        train_loss=float(total)/max(1,n)
        history.append({'epoch':epoch+1,'train_loss':train_loss,'val_loss':val_loss,
                        'lr':opt.param_groups[0]['lr'],'batch':batch_size,
                        'real_training_observations_seen':n})
        improved=val_loss<best_score-1e-5
        if improved:
            best_score=val_loss;stale=0
            checkpoint_write(best,{'model':_cpu_tree(model.state_dict()),
                'identity':identity,'feature_names':features,'preprocessing':scaler,
                'epoch':epoch,'val_loss':val_loss,'phase':phase,
                'model_config':model_config(model),'provenance':extra or {}})
        else:stale+=1
        checkpoint_write(latest,{'model':_cpu_tree(model.state_dict()),
            'optimizer':_cpu_tree(opt.state_dict()),'identity':identity,
            'epoch':epoch,'best_score':best_score,'stale':stale,
            'history':history,'batch_size':batch_size,'partial_epoch':None})
        save_json(work/(phase+'_history.json'),history)
        log(f'{work.name}/{phase} epoch {epoch+1}/{epochs}: train={train_loss:.5f}, '
            f'validation={val_loss:.5f}, best={best_score:.5f} '
            f'| patience={stale}/{CFG["patience"]}')
        if stale>=CFG['patience']:
            log(f'{work.name}/{phase}: early stopped, best val={best_score:.5f}')
            break
    if not best.exists():raise RuntimeError(f'Best checkpoint missing: {best}')
    model.load_state_dict(chk_read(best)['model'])
    return history

def model_config(model):
    return {k:CFG[k] for k in ('dim','heads','layers','dropout','latent_tokens')} | {'n_features':model.n_features}

def preprocessor(raw,train_i,top_k=512,min_observed=0.40):
    """Training-fold ONLY selection, imputation and standardization."""
    raw=np.asarray(raw,dtype=np.float32)
    obs=np.isfinite(raw)
    cover=obs[train_i].mean(axis=0)
    eligible=np.where(cover >= min_observed)[0]
    if len(eligible)<8:raise RuntimeError('Insufficient measured features for training')
    # Variance rankings learned on TRAIN only, excluding missingness.
    variance=np.nanvar(np.where(obs[train_i][:,eligible],raw[train_i][:,eligible],np.nan),axis=0)
    variance=np.where(np.isfinite(variance),variance,0)
    ranking=np.argsort(-variance,kind='stable')[:min(top_k,len(eligible))]
    select=np.sort(eligible[ranking])
    arr=raw[:,select];mask=obs[:,select]
    train=arr[train_i]
    med=np.nanmedian(train,axis=0)
    med=np.where(np.isfinite(med),med,0)
    filled=np.where(mask,arr,med)
    mean=np.mean(filled[train_i],axis=0)
    std=np.std(filled[train_i],axis=0)
    std=np.maximum(std,1e-3)
    x=np.clip((filled-mean)/std,-8,8).astype('float32')
    return x,mask,select,{'median':med.tolist(),'mean':mean.tolist(),'std':std.tolist(),
                          'clip':8,'selection_method':'training_variance'}

def fig_curves(work,title):
    plt.figure(figsize=(8,4))
    for phase in ('ssl','task'):
        p=Path(work)/(phase+'_history.json')
        if not p.exists():continue
        h=json.loads(p.read_text())
        if not h:continue
        plt.plot([v['epoch'] for v in h],[v['val_loss'] for v in h],label=f'{phase} validation loss')
        plt.plot([v['epoch'] for v in h],[v['train_loss'] for v in h],linestyle='--',label=f'{phase} training loss')
    plt.xlabel('Epoch');plt.ylabel('Loss');plt.title(title);plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(Path(work)/'learning_curves.png',dpi=160);plt.close()

def parse_geo_series_matrix(path):
    """GEO Series Matrix loader; extracts required metadata and expression, fails closed."""
    metadata={};expression_lines=[];inside=False
    with gzip.open(path,'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_'):
                fields=next(csv.reader([line.rstrip('\r\n')],delimiter='\t',quotechar='"'))
                metadata[fields[0]]=fields[1:]
            elif line.startswith('!series_matrix_table_begin'):
                inside=True
            elif line.startswith('!series_matrix_table_end'):
                break
            elif inside and not line.startswith('!'):
                expression_lines.append(line)
    if len(expression_lines)<100:
        raise RuntimeError(f'{path} has no substantial expression matrix; inspected {len(expression_lines)} rows')
    expr=pd.read_csv(io.StringIO(''.join(expression_lines)),sep='\t',low_memory=False)
    if expr.shape[1]<3:raise RuntimeError('Unexpected GEO table columns')
    accessions=metadata.get('!Sample_geo_accession')
    titles=metadata.get('!Sample_title')
    if not accessions or not titles or len(accessions)!=len(titles):
        raise RuntimeError('Missing GEO sample accessions or clinical labels')
    columns=[str(c).strip().strip('"') for c in expr.columns[1:]]
    if columns!=accessions:
        raise RuntimeError(f'Sample order mismatch between GEO metadata and expression matrix: {columns[:3]} vs {accessions[:3]}')
    y=[]
    for title in titles:
        s=title.lower().replace('_',' ')
        if re.search(r'no[ ._-]*gvhd|without[ ._-]*gvhd',s):y.append(0)
        elif re.search(r'gvhd',s):y.append(1)
        else: raise RuntimeError(f'Unclear GVHD sample label: {title!r}; no guessing allowed')
    # GEO probes are assay-specific identifiers, NOT automatically gene symbols.
    feats=expr.iloc[:,0].astype(str).tolist()
    val=expr.iloc[:,1:].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32).T
    if len(feats)!=len(set(feats)):
        raise RuntimeError('Duplicate probe identifiers require a deterministic aggregation strategy')
    if len(feats) < 1000 or val.shape[0] != 24:
        raise RuntimeError('GSE73809 data dimensionality/sample-count mismatch; abort.')
    if len(y)!=val.shape[0] or len(np.unique(y))!=2:
        raise RuntimeError('GEO sample count/labels failed validation')
    return val,feats,np.asarray(y,dtype=np.int32),accessions,titles

GEO_URLS=[
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73809/matrix/GSE73809_series_matrix.txt.gz',
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73809/matrix/GSE73809_series_matrix.txt.gz?tool=hctformer',
]
PROTEOMICS_URLS={
    'npx':'https://raw.githubusercontent.com/jackgisby/longitudinal_olink_proteomics/main/data/plasma_npx_level.csv',
    'meta':'https://raw.githubusercontent.com/jackgisby/longitudinal_olink_proteomics/main/data/plasma_sample_level.csv',
}

# Expanded public-only HCT expression manifest. Study-specific phenotypes are NEVER
# promoted to a common GVHD training label: these studies serve SSL ONLY.
HCT_SSL_GEO = {
  'GSE103569': {'context':'human aHCT sorted CD8 subsets; repeated sorted samples per patient',
                'min_samples':40},
  'GSE56495': {'context':'human cGVHD whole blood, multiple visits; healthy controls excluded',
               'min_samples':30},
  'GSE7510': {'context':'human allo-HCT engraftment peripheral blood; 8 distinct recipients',
              'min_samples':6},
}

def geo_url(gse):
    assert re.fullmatch(r'GSE\d+', gse)
    return f'https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/{gse}_series_matrix.txt.gz'

def load_geo_unlabeled(gse):
    path=fetch([geo_url(gse)],DATA/f'{gse}_series_matrix.txt.gz',min_bytes=1000)
    sample_meta={}; series_meta={}; table=[]; inside=False
    with gzip.open(path,'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_'):
                fields=next(csv.reader([line.rstrip('\r\n')],delimiter='\t',quotechar='"'))
                sample_meta[fields[0]]=fields[1:]
            elif line.startswith('!Series_'):
                fields=next(csv.reader([line.rstrip('\r\n')],delimiter='\t',quotechar='"'))
                series_meta[fields[0]]=fields[1:]
            elif line.startswith('!series_matrix_table_begin'):inside=True
            elif line.startswith('!series_matrix_table_end'):break
            elif inside and not line.startswith('!'):table.append(line)
    if len(table)<1000:raise RuntimeError(f'{gse}: series matrix has no substantial measured values (rows={len(table)})')
    expr=pd.read_csv(io.StringIO(''.join(table)),sep='\t',low_memory=False)
    ids=sample_meta.get('!Sample_geo_accession',[])
    titles=sample_meta.get('!Sample_title',[])
    platforms=sample_meta.get('!Sample_platform_id',[])
    if not ids or len(ids)!=len(titles) or len(ids)!=len(platforms):
        raise RuntimeError(f'{gse}: missing required metadata with consistent lengths')
    if len(set(platforms))!=1:raise RuntimeError(f'{gse}: multiple platforms in one matrix require separate mapping')
    if list(expr.columns[1:])!=ids:raise RuntimeError(f'{gse}: sample header order does not match GEO IDs')
    vals=expr.iloc[:,1:].apply(pd.to_numeric,errors='coerce').to_numpy(dtype='float32')
    probe=expr.iloc[:,0].astype(str).values
    if len(set(probe))!=len(probe):
        raise RuntimeError(f'{gse}: duplicate assay IDs need source-specific aggregation')
    if not np.isfinite(vals).any():raise RuntimeError(f'{gse}: no valid quantitative values')
    if np.nanquantile(vals,.99)>50:
        if np.nanmin(vals)<0:raise RuntimeError(f'{gse}: ambiguous raw expression scale')
        vals=np.log2(1+np.maximum(vals,0))
    return pd.DataFrame(vals.T,index=ids,columns=probe),ids,titles,platforms[0],sha256(path)

def gpl_soft_url(gpl):
    return f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl[:-3]}nnn/{gpl}/soft/{gpl}_family.soft.gz'

def gpl_annot_url(gpl):
    return f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl[:-3]}nnn/{gpl}/annot/{gpl}.annot.gz'

def gpl_table_url(gpl):
    # Official NCBI GEO accession-view API; serves the original GPL data table.
    return f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&view=data&form=text'

def decode_geo_gene_symbol(raw, column_name):
    """Convert an official GPL gene-symbol field, including HTA2 gene_assignment.

    HTA-2.0 gene_assignment starts with e.g.
    'NR_046018 // DDX11L1 // description // ... /// other transcript'.
    The gene symbol is the SECOND ' // '-delimited field; NEVER the RefSeq ID.
    Ambiguous/unmapped entries are excluded, not guessed.
    """
    if pd.isna(raw):
        return None
    field = str(raw).strip()
    if not field or field in ('---', 'NA', 'NULL', 'None', 'nan'):
        return None
    if re.sub('[^a-z]','',column_name.lower()) == 'geneassignment':
        first = re.split(r'\s*///\s*', field, maxsplit=1)[0]
        parts = re.split(r'\s*//\s*', first)
        if len(parts) < 2:
            return None
        field = parts[1].strip()
    else:
        field = re.split(r'\s*///\s*|\s*//\s*|;', field, maxsplit=1)[0].strip()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9._-]{1,29}', field):
        return None
    if field.upper() in {'NA','NULL','NONE','---','UNKNOWN'}:
        return None
    return field.upper()

def gpl_feature_map(gpl, requested_probe_ids=None):
    """GEO probe→gene mapping; streamed from official GPL table, without fake symbols.

    GPL17586 has gene_assignment and both ID / probeset_id. Probe identifier
    column is selected by intersection with actual expression feature IDs.
    A previously cached multi-GB .soft.gz is processed ONE LINE AT A TIME.
    """
    requested = (set(map(str, requested_probe_ids)) if requested_probe_ids is not None else None)
    cache = DATA / f'{gpl}_probes_to_genes.json'
    if cache.exists():
        try:
            previous = json.loads(cache.read_text())
            existing = previous.get('mapping', {}) if previous.get('gpl') == gpl else {}
            matched = len(set(existing).intersection(requested)) if requested is not None else len(existing)
            required = max(100, min(2000, int(len(requested) * 0.15))) if requested else 1000
            if matched >= required:
                log(f'{gpl}: valid cached probe-to-gene crosswalk ({matched:,} matches)')
                return existing
        except (ValueError, OSError, KeyError, TypeError):
            pass

    # Official GPL17586 /annot/ URL returns 404 and is deliberately skipped.
    # Prefer a cached SOFT file, if already downloaded; never read entire file into RAM.
    routes = [('table', gpl_table_url(gpl))]
    if gpl != 'GPL17586':
        routes.append(('annot', gpl_annot_url(gpl)))
    # Official table may fail or truncate; full GEO SOFT is the verified fallback.
    # A user may also supply a cached .soft.gz with valid *.source.json provenance.
    routes.append(('soft', gpl_soft_url(gpl)))
    errors = []
    for kind, url in routes:
        try:
            ext = '.txt' if kind == 'table' else '.gz'
            path = fetch([url], DATA / f'{gpl}.{kind}{ext}', min_bytes=1000)
            columns = None
            maps_by_col = {}
            n_table_rows = 0
            # Gene assignments can be long; increase parser field cap safely.
            csv.field_size_limit(16 * 1024 * 1024)
            # SOFT can exceed multiple gigabytes. Strict constant-memory scan
            # relative to the file size; only collect actual matching feature IDs.
            opener = open if kind == 'table' else gzip.open
            with opener(path, 'rt', encoding='utf-8', errors='replace', newline='') as handle:
                in_table = kind != 'soft'
                for cells in csv.reader(handle, delimiter='\t', quotechar='"'):
                    if not cells:
                        continue
                    first = cells[0].strip()
                    if kind == 'soft' and first.lower().startswith('!platform_table_begin'):
                        in_table = True
                        continue
                    if kind == 'soft' and first.lower().startswith('!platform_table_end'):
                        break
                    if not in_table or first.startswith(('!', '^', '#')):
                        continue
                    if columns is None:
                        columns = [str(c).strip().lstrip('#').strip() for c in cells]
                        normalized = {re.sub('[^a-z]', '', c.lower()): i for i, c in enumerate(columns)}
                        symbol_priority = ['genesymbol', 'genesymbols', 'symbol', 'ilmngene', 'geneassignment']
                        symbol_idx = next((normalized[x] for x in symbol_priority if x in normalized), None)
                        id_fields = {'id', 'id_ref', 'probe_id', 'probe name', 'probeid',
                                     'probeset_id', 'transcript_cluster_id'}
                        id_indices = [i for i, c in enumerate(columns) if c.lower() in id_fields]
                        if symbol_idx is None or not id_indices:
                            raise RuntimeError(f'{gpl}: no valid ID or gene symbol columns; {columns[:20]}')
                        gene_col = columns[symbol_idx]
                        maps_by_col = {columns[i]: {} for i in id_indices}
                        continue
                    n_table_rows += 1
                    if symbol_idx >= len(cells):
                        continue
                    symbol = decode_geo_gene_symbol(cells[symbol_idx], gene_col)
                    if not symbol:
                        continue
                    for i in id_indices:
                        if i < len(cells):
                            probe = str(cells[i]).strip()
                            if probe and (requested is None or probe in requested):
                                maps_by_col[columns[i]][probe] = symbol
            if not maps_by_col:
                raise RuntimeError(f'{gpl}: empty/invalid {kind} GPL table')
            chosen_col, mapping = max(maps_by_col.items(), key=lambda pair: len(pair[1]))
            if requested:
                required = max(100, min(2000, int(len(requested) * 0.15)))
            else:
                required = 2000
            if len(mapping) < required:
                raise RuntimeError(f'{gpl}: mapped only {len(mapping)} of {len(requested) if requested else "all"} probes via {chosen_col}; table rows={n_table_rows}')
            save_json(cache, {'gpl': gpl, 'mapping': mapping, 'source_url': url,
                              'sha256_source': sha256(path), 'n_mappings': len(mapping),
                              'gene_column': gene_col, 'probe_column': chosen_col,
                              'requested_probes': len(requested) if requested else None,
                              'table_rows_scanned': n_table_rows})
            log(f'{gpl}: mapped {len(mapping):,} measured probes using {chosen_col} → {gene_col} ({kind})')
            return mapping
        except Exception as exc:
            errors.append(f'{kind}: {str(exc)[:400]}')
            log(f'{gpl} {kind} mapping failed: {type(exc).__name__}: {str(exc)[:250]}')
    raise RuntimeError(f'{gpl}: no verified probe-to-gene crosswalk. '+ '; '.join(errors))

def geo_genes(gse):
    df,ids,titles,platform,digest=load_geo_unlabeled(gse)
    annotation=gpl_feature_map(platform, requested_probe_ids=df.columns)
    probe_to_gene=pd.Series(annotation)
    common=df.columns.intersection(probe_to_gene.index)
    if len(common)<2000:raise RuntimeError(f'{gse}: <2000 probes mapped to gene symbols for {platform}')
    tmp=df[common].T.copy()
    tmp.index=probe_to_gene.loc[common].values
    genes=tmp.groupby(level=0,sort=True).median(numeric_only=True).T
    if genes.shape[1]<1500:raise RuntimeError(f'{gse}: too few unique mapped genes')
    log(f'{gse}: {len(ids)} measured samples, {len(genes.columns)} gene symbols, platform {platform}')
    return genes,ids,titles,platform,digest

def ssl_patient(gse,title,sample_id):
    if gse=='GSE103569':
        m=re.search(r'Patient\s*No\.?\s*(\d+)',title,re.I)
        return f'{gse}:patient:{m.group(1)}' if m else None
    if gse=='GSE56495':
        if re.search('healthy control',title,re.I):return None
        m=re.match(r'([A-Za-z]{2,5}[- ]?\d{1,3})[-_ ]',title)
        return f'{gse}:patient:{m.group(1).upper()}' if m else None
    if gse=='GSE7510':
        # GEO design: 8 distinct recipients; each GEO sample is independent.
        return f'{gse}:individual:{sample_id}'
    return None

def real_hct_single_cell_ssl(features, output_folder):
    """Real HCT T-cell samples from Song et al Zenodo GSE295359.

    Used ONLY as UNLABELED SSL training. No cell is given a patient-level clinical
    label. Cells are correlated and do not count as independent subjects.
    """
    src='https://zenodo.org/records/15264188/files/total_t_cells.h5ad?download=1'
    path=fetch([src],DATA/'GSE295359_total_t_cells.h5ad',min_bytes=100_000_000)
    try:
        import anndata as ad
        import scipy.sparse as sparse
    except ImportError as e:
        raise RuntimeError('Install anndata and scipy to enable real HCT single-cell ingestion') from e
    data=ad.read_h5ad(path,backed='r')
    try:
        choices=[('var_names',np.array(data.var_names.astype(str)))]
        for c in data.var.columns:
            if any(p in c.lower() for p in ('gene_symbol','gene_name','symbol','feature_name')):
                choices.append((str(c),data.var[c].fillna('').astype(str).to_numpy()))
        # Gene symbols must match the feature vocabulary learned from public GPL tables.
        gene_set=set(features)
        best=max(choices,key=lambda t:len(gene_set.intersection({v.upper() for v in t[1]})))
        names=np.char.upper(best[1].astype(str))
        lookup={}
        for j,name in enumerate(names):
            if name in gene_set and name not in lookup:lookup[name]=j
        matched=[(k,lookup[f]) for k,f in enumerate(features) if f in lookup]
        if len(matched)<200:raise RuntimeError(f'Single-cell gene mapping has only {len(matched)} matched genes; stop this optional branch')
        max_cells=max(1,CFG['max_hct_single_cells'])
        if len(data)<1:raise RuntimeError('Empty real AnnData object')
        rng=np.random.default_rng(CFG['seed'])
        ids=np.sort(rng.choice(np.arange(len(data)),size=min(len(data),max_cells),replace=False))
        # Load only small blocks of genuine observed cells. This prevents a temporary
        # dense (50,000 x all genes) allocation that may exhaust Colab system RAM.
        matched_gene_indices = [j for _, j in matched]
        blocks = []
        block_size = 256
        for start in range(0, len(ids), block_size):
            selected = ids[start:start+block_size]
            block = data.X[selected, :]
            if sparse.issparse(block):
                block = block[:, matched_gene_indices].toarray()
            else:
                block = np.asarray(block)[:, matched_gene_indices]
            blocks.append(np.asarray(block, dtype='float32'))
            if start % (block_size * 40) == 0:
                log(f'Reading real HCT T-cell measurements: {min(start + block_size, len(ids)):,}/{len(ids):,} cells')
        mat = np.concatenate(blocks, axis=0)
        if not np.isfinite(mat).all():raise RuntimeError('Single-cell table includes nonfinite values')
        if mat.min()<0:raise RuntimeError('Unexpected negative single-cell measurements: cannot choose preprocessing')
        if np.quantile(mat,.99)>15:mat=np.log1p(mat)
        full=np.zeros((len(ids),len(features)),dtype='float32')
        obs=np.zeros(full.shape,dtype=bool)
        positions=[k for k,j in matched]
        # Only non-zero real observed transcripts contribute to masking objective;
        # zero entries are true observed non-detection, NOT imputed positive counts.
        means=mat.mean(axis=0);std=np.maximum(mat.std(axis=0),1e-3)
        scaled=np.clip((mat-means)/std,-8,8)
        full[:,positions]=scaled
        obs[:,positions]=mat>0
        have=obs.any(axis=1)
        full=full[have];obs=obs[have]
        if len(full)<1000:raise RuntimeError('Too few observed single cells after filtering')
        audit={'study':'GSE295359','source':'Song et al. human allo-HCT T-cell scRNA-seq',
            'url':src,'sha256':sha256(path),'ann_data_obs':len(data),
            'selected_cells':len(full),'matched_genes':len(matched),
            'random_cell_ids_hash':hashlib.sha256(ids.tobytes()).hexdigest(),
            'feature_vocabulary_hash':stable_hash(features),
            'use':'UNLABELED self-supervised masked real expressed gene reconstruction only',
            'independent_clinical_patients':'10 total in paper, cells NOT independent patients',
            'synthetic_data_used':False}
        save_json(Path(output_folder)/'single_cell_data_audit.json',audit)
        log(f'Single cell SSL: {len(full)} real HCT T cells with {len(matched)} matched genes')
        return full,obs,audit
    finally:
        # close H5 file before notebook restart or Colab disconnect
        if getattr(data,'file',None) is not None:data.file.close()

def geo_expanded_workflow():
    log('MODEL A V3: external real HCT cohorts -> gene-mapped SSL, then GSE73809 TRAIN/VAL/TEST supervised classification')
    work=ROOT/'model_A_HCT_transcriptomic';work.mkdir(exist_ok=True)
    raw,genes,y,ids,titles,provenance=get_genes()
    # The original 24-subject labeled dataset cannot be pooled with external clinical labels.
    labeled_geo,ids2,titles2,gpl,digest=geo_genes('GSE73809')
    if ids!=ids2 or len(y)!=24:raise RuntimeError('GSE73809 provenance mismatch')
    idx=np.arange(len(y))
    trva,te=train_test_split(idx,test_size=.25,stratify=y,random_state=CFG['seed'])
    tr,va=train_test_split(trva,test_size=.25,stratify=y[trva],random_state=CFG['seed']+1)
    ext=[]; failures=[]; ext_audit=[]
    for gse,detail in HCT_SSL_GEO.items():
        try:
            df,accessions,descriptions,plat,digest=geo_genes(gse)
            people=[ssl_patient(gse,t,s) for s,t in zip(accessions,descriptions)]
            keep=[i for i,p in enumerate(people) if p is not None]
            if len(keep)<detail['min_samples']:
                raise RuntimeError(f'Only {len(keep)} valid patient-identifiable measured samples; min={detail["min_samples"]}')
            df=df.iloc[keep];people=np.array([people[i] for i in keep]);accessions=[accessions[i] for i in keep]
            u=np.unique(people)
            rng=np.random.default_rng(CFG['seed']);rng.shuffle(u)
            vcount=max(1,round(.2*len(u)));vpeople=set(u[:vcount])
            isval=np.isin(people,list(vpeople))
            if isval.all() or (~isval).all():raise RuntimeError('External patient split failed')
            ext.append((gse,df,people,isval))
            ext_audit.append({'gse':gse,'platform':plat,'n_samples':len(df),'n_people':len(u),
                              'train_samples':int((~isval).sum()),'validation_samples':int(isval.sum()),
                              'gene_symbols':df.shape[1],'source_sha256':digest,
                              'role':'UNLABELED SSL ONLY','clinical_labels_used':False,
                              'source_url':f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}'})
        except Exception as e:
            failures.append({'gse':gse,'reason':str(e)[:900]})
            log(f'External HCT study NOT included (reason recorded): {gse} {str(e)[:300]}')
    save_json(work/'external_study_ingestion.json',{'accepted':ext_audit,'excluded':failures})
    n_ext_train=sum(int((~v).sum()) for _,_,_,v in ext)
    if CFG['require_external_hct'] and (n_ext_train<CFG['min_external_hct_samples'] or len(ext)<CFG['min_external_hct_studies']):
        raise RuntimeError(f'EXPANSION INCOMPLETE: external HCT SSL train={n_ext_train} from {len(ext)} studies; '
            f'need >= {CFG["min_external_hct_samples"]} and >= {CFG["min_external_hct_studies"]}. '
            'Review external_study_ingestion.json; NO downgrade to legacy 24-sample-only training.')
    if len(ext)==0:raise RuntimeError('No external HCT studies validated; stop')
    # Only choose genes actually assayed in labeled cohort and >=2 external datasets.
    support=pd.Series(0,index=labeled_geo.columns,dtype='int32')
    for _,df,_,_ in ext:
        support.loc[df.columns.intersection(support.index)]+=1
    common=support[support>=min(2,len(ext))].index
    if len(common)<min(256,CFG['gene_features']):raise RuntimeError(f'Only {len(common)} shared gene symbols; cannot transfer encoder')
    # Gene selection uses external TRAIN samples ONLY (never test labels or test expression).
    variance=pd.Series(0.0,index=common)
    for _,df,_,val_mask in ext:
        overlap=common.intersection(df.columns)
        if len(overlap):
            candidate=df.loc[~val_mask,overlap].var(axis=0).fillna(0)
            variance.loc[overlap]+=candidate.clip(upper=candidate.quantile(.99))
    features=variance.sort_values(ascending=False).head(min(CFG['gene_features'],len(common))).index.tolist()
    if len(features)<256:raise RuntimeError('Too few harmonized genes')
    def normalize(df,trainmask):
        raw=df.reindex(columns=features).to_numpy(dtype='float32')
        seen=np.isfinite(raw)
        if (~np.asarray(trainmask)).all():raise RuntimeError('Normalization needs train samples')
        ref=raw[np.asarray(trainmask)]
        # Avoid warnings for all-missing, leave explicit missingness mask.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            mu=np.nanmean(ref,axis=0);sd=np.nanstd(ref,axis=0)
        mu=np.nan_to_num(mu,nan=0.0);sd=np.nan_to_num(sd,nan=1.0)
        sd=np.maximum(sd,1e-3)
        scaled=np.clip((np.where(seen,raw,mu)-mu)/sd,-8,8).astype('float32')
        return scaled,seen,{'median':mu.tolist(),'mean':mu.tolist(),'std':sd.tolist(),'features':features}
    xlab,olab,target_scaler=normalize(labeled_geo,np.isin(np.arange(len(y)),tr))
    ext_train_x=[xlab[tr]];ext_train_o=[olab[tr]]
    ext_val_x=[];ext_val_o=[]
    external_saved=[]
    for gse,df,people,isval in ext:
        xx,oo,_=normalize(df,~isval)
        ext_train_x.append(xx[~isval]);ext_train_o.append(oo[~isval])
        ext_val_x.append(xx[isval]);ext_val_o.append(oo[isval])
        external_saved.append({'gse':gse,'n_train':int((~isval).sum()),'n_val':int(isval.sum()),
                               'fraction_observed':float(oo.mean())})
    sc_audit=None;sc_failure=None
    if CFG['use_hct_single_cell']:
        try:
            sc_x,sc_o,sc_audit=real_hct_single_cell_ssl(features,work)
            ext_train_x.append(sc_x);ext_train_o.append(sc_o)
        except Exception as e:
            sc_failure=str(e)[:1000]
            log(f'OPTIONAL real HCT single-cell training not included: {sc_failure[:350]}')
            if CFG['require_hct_single_cell']:
                raise RuntimeError('Required GSE295359 single-cell ingestion failed QC; no synthetic substitution') from e
    save_json(work/'single_cell_ingestion_status.json',{'accepted':sc_audit,'excluded_reason':sc_failure,
            'required':CFG['require_hct_single_cell'],'enabled':CFG['use_hct_single_cell']})
    ssl_train_x=np.concatenate(ext_train_x).astype('float32')
    ssl_train_o=np.concatenate(ext_train_o).astype('bool')
    ssl_val_x=np.concatenate(ext_val_x).astype('float32')
    ssl_val_o=np.concatenate(ext_val_o).astype('bool')
    save_json(work/'data_audit.json',{'labeled_dataset':provenance,
        'split_ids':{'train':[ids[i] for i in tr],'validation':[ids[i] for i in va],'test':[ids[i] for i in te]},
        'split_unit_labeled':'GEO SAMPLE (no patient IDs confirmed)',
        'split_unit_external':'patient identifier parsed from original GEO sample title',
        'external_studies':ext_audit,'ssl_split_by_study':external_saved,
        'single_cell_ssl':sc_audit,'single_cell_ingestion_failure':sc_failure,
        'num_ssl_train_samples':len(ssl_train_x),'num_ssl_validation_samples':len(ssl_val_x),
        'selected_features':len(features),'gene_mapping':'GEO GPL gene-symbol annotation, >=2 external cohorts',
        'supervised_target':'GSE73809 contemporaneous GVHD status ONLY',
        'task':'HCT GVHD contemporaneous case/control association (not future prediction)',
        'no_synthetic_training_data':True})
    identity=stable_hash({'model':'gene_v3','cfg':model_shape_settings(),
        'ext':ext_audit,'feat':features,'sc_audit':sc_audit,
        'ssl_n':int(len(ssl_train_x)),'labeled_sha256':digest,'splits':[tr.tolist(),va.tolist(),te.tolist()]})
    model=mkmodel(len(features))
    def ssl_fn(m,b,training):return masked_loss(m,b[0],b[1])[0]
    fit(model,SampleSet(ssl_train_x,ssl_train_o,np.arange(len(ssl_train_x))),
        SampleSet(ssl_val_x,ssl_val_o,np.arange(len(ssl_val_x))),
        'ssl',work,ssl_fn,identity,features,target_scaler,{'objective':'masked REAL shared-gene expression'})
    pos_weight=(len(tr)-sum(y[tr]))/max(1,sum(y[tr]))
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=DEVICE,dtype=torch.float32))
    def task_fn(m,b,training):
        z,_=m(b[0],b[1]);return bce(m.classification(z).flatten(),b[2])
    fit(model,LabelSet(xlab,olab,y,tr),LabelSet(xlab,olab,y,va),'task',work,task_fn,
        identity,features,target_scaler,{'objective':'labeled GSE73809 GVHD case/control'})
    model.eval()
    with torch.no_grad():
        z,_=model(torch.from_numpy(xlab[te]).to(DEVICE),torch.from_numpy(olab[te]).to(DEVICE))
        proba=torch.sigmoid(model.classification(z).flatten()).cpu().numpy()
    baseline=LogisticRegression(max_iter=2000,C=0.1,class_weight='balanced',random_state=CFG['seed'])
    baseline.fit(xlab[tr],y[tr]);pbase=baseline.predict_proba(xlab[te])[:,1]
    def score(p):return {'n_test':len(te),'auroc':float(roc_auc_score(y[te],p)),
        'balanced_accuracy':float(balanced_accuracy_score(y[te],p>=.5)),
        'brier':float(brier_score_loss(y[te],p))}
    results={'transformer':score(proba),'logistic_baseline':score(pbase),
        'training_ssl_samples':int(len(ssl_train_x)),'training_ssl_external_samples':int(n_ext_train),
        'training_ssl_real_single_cells':int(sc_audit['selected_cells']) if sc_audit else 0,
        'num_external_cohorts':len(ext),'test_samples':int(len(te)),
        'clinical_interpretation':'contemporaneous GVHD association ONLY; still 6 labeled test samples',
        'no_test_tuning':True}
    save_json(work/'results.json',results)
    pd.DataFrame({'sample_id':np.array(ids)[te],'label':y[te],
        'transformer_probability':proba,'logistic_probability':pbase}).to_csv(work/'heldout_predictions.csv',index=False)
    checkpoint_write(work/'hct_gvhd_model.pt',{'model_state_dict':model.cpu().state_dict(),
        'model_config':model_config(model),'feature_names':features,'preprocessing':target_scaler,
        'metrics':results,'data_audit':json.loads((work/'data_audit.json').read_text()),
        'training_status':'trained','synthetic_training_data_used':False,
        'objectives':TRAINING_OBJECTIVES['model_A_HCT_transcriptomic']})
    fig_curves(work,'Model A V3: external HCT SSL + GVHD classification')
    (work/'MODEL_CARD.md').write_text('# Model A — HCT pooled gene self-supervision\n\n'
        'Only real measured data. Gene IDs aligned by GEO platform annotations.\n\n'
        'External HCT cohorts used for masked-gene SSL ONLY. GSE73809 is the ONLY '
        'source for supervised case/control prediction, with six test samples and no verified patient IDs.\n\n'
        'Do not interpret as a prospective GVHD predictor.\n')
    log(f'MODEL A V3 complete: {results}')
    return results

# Backward-compatible loader for the original confirmed GSE73809 labels.
def get_genes():
    dest=fetch(GEO_URLS,DATA/'GSE73809_series_matrix.txt.gz',min_bytes=2000)
    x,genes,y,accessions,titles=parse_geo_series_matrix(dest)
    if len(y)!=24 or int(np.sum(y))!=11 or int(np.sum(y==0))!=13:
        raise RuntimeError('GSE73809 accession-level audit failed: expected 24 samples (11 GVHD, 13 controls).')
    provenance={'source':'GSE73809','type':'REAL_MEASURED_HUMAN_GEO',
                'url':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73809',
                'sha256':sha256(dest),'n_samples':len(accessions),
                'gvhd_positive':int(np.sum(y)), 'gvhd_negative':int(np.sum(y==0))}
    return x,genes,y,accessions,titles,provenance


def model_shape_settings():
    return {k:CFG[k] for k in ['dim','heads','layers','dropout','latent_tokens','seed','max_history_visits','run_mode','mask_fraction']}

def get_protein_tables():
    """Real Olink data. Plasma: chronological forecasting + SSL. Serum: SSL ONLY.

    Serum sample metadata has no collection time; no chronology is constructed
    from SampleID, row order, plate number, or subject status.
    """
    root = 'https://raw.githubusercontent.com/jackgisby/longitudinal_olink_proteomics/main/data/'
    biofluids = ['plasma', 'serum']
    longs, metas, provenance, failures = [], [], [], []
    for cohort in biofluids:
        try:
            npx = fetch([root + cohort + '_npx_level.csv'], DATA / f'gisby_{cohort}_npx_level.csv', min_bytes=100000)
            info = fetch([root + cohort + '_sample_level.csv'], DATA / f'gisby_{cohort}_sample_level.csv', min_bytes=1000)
            long = pd.read_csv(npx, low_memory=False)
            meta = pd.read_csv(info, low_memory=False)
            needed_long = {'SampleID', 'Individual_ID', 'Assay', 'NPX'}
            needed_meta = {'SampleID', 'Individual_ID'}
            if not needed_long.issubset(long.columns) or not needed_meta.issubset(meta.columns):
                raise RuntimeError(f'{cohort}: unexpected data schema, long={list(long.columns)}, meta={list(meta.columns)}')
            if meta['SampleID'].isna().any() or meta['Individual_ID'].isna().any():
                raise RuntimeError(f'{cohort}: missing patient or sample identifiers')
            if (meta.groupby('SampleID')['Individual_ID'].nunique() > 1).any():
                raise RuntimeError(f'{cohort}: a sample maps to multiple patients')
            if meta['SampleID'].duplicated().any():
                raise RuntimeError(f'{cohort}: duplicated sample metadata; audit required')
            if long['SampleID'].isna().any() or long['Individual_ID'].isna().any():
                raise RuntimeError(f'{cohort}: missing patient/sample IDs in molecular rows')
            long = long.copy()
            meta = meta.copy()
            long['SampleID'] = cohort + ':' + long['SampleID'].astype(str)
            meta['SampleID'] = cohort + ':' + meta['SampleID'].astype(str)
            long['Individual_ID'] = long['Individual_ID'].astype(str)
            meta['Individual_ID'] = meta['Individual_ID'].astype(str)
            if (long.groupby('SampleID')['Individual_ID'].nunique() > 1).any():
                raise RuntimeError(f'{cohort}: inconsistent molecular sample-to-person mapping')
            sampled = meta.set_index('SampleID')['Individual_ID']
            matched = long['SampleID'].map(sampled)
            if matched.isna().any() or (matched.to_numpy() != long['Individual_ID'].to_numpy()).any():
                raise RuntimeError(f'{cohort}: molecular sample IDs and metadata persons disagree')
            long['Feature'] = (long['Panel'].astype(str) + '|' + long['Assay'].astype(str)
                               if 'Panel' in long.columns else long['Assay'].astype(str))
            long['NPX'] = pd.to_numeric(long['NPX'], errors='coerce')
            meta['source_cohort'] = cohort
            tcol = next((c for c in ('Time_From_First_Symptoms', 'Time_From_First_Swab')
                         if c in meta.columns), None)
            if cohort == 'plasma' and tcol is None:
                raise RuntimeError('Plasma has no verified chronological time field; cannot forecast')
            if tcol is not None:
                meta['Time_Aligned'] = pd.to_numeric(meta[tcol], errors='coerce')
                temporal_use = 'timed longitudinal plasma'
            else:
                # This is a true MISSING time coordinate, not a manufactured one.
                meta['Time_Aligned'] = np.nan
                temporal_use = 'untimed serum: masked-protein SSL ONLY'
            log(f'{cohort}: {len(meta)} real samples; {meta.Individual_ID.nunique()} people; {temporal_use}')
            longs.append(long[['SampleID', 'Individual_ID', 'Feature', 'NPX']])
            metas.append(meta)
            provenance.append({'source': 'Gisby longitudinal Olink ' + cohort,
                'biofluid': cohort, 'sha256_npx': sha256(npx),
                'sha256_sample_metadata': sha256(info),
                'source_url': 'https://github.com/jackgisby/longitudinal_olink_proteomics',
                'n_samples_metadata': len(meta), 'time_axis': tcol,
                'training_roles': ['masked-protein SSL', 'next-visit forecasting'] if tcol is not None else ['masked-protein SSL only']})
        except Exception as exc:
            failures.append({'biofluid': cohort, 'reason': str(exc)[:700]})
            log(f'Protein subcohort {cohort} failed QC: {str(exc)[:260]}')
            if cohort == 'plasma' or CFG['require_serum']:
                save_json(ROOT / 'protein_ingestion_failures.json', failures)
                raise RuntimeError(f'Required real {cohort} data failed QC; no synthetic fallback') from exc
    long = pd.concat(longs, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    if meta.SampleID.duplicated().any():
        raise RuntimeError('Duplicate biofluid-prefixed sample ID')
    wide = long.pivot_table(index='SampleID', columns='Feature', values='NPX', aggfunc='mean')
    eligible = meta[meta.SampleID.isin(wide.index)].copy()
    if eligible.empty:
        raise RuntimeError('No real matched Olink sample metadata and measurements')
    # Serum rows intentionally retain NaN time, and enter SSL only.
    eligible = eligible.sort_values(['Individual_ID', 'source_cohort', 'Time_Aligned', 'SampleID'],
                                    na_position='last').reset_index(drop=True)
    plasma_dated = eligible[(eligible.source_cohort == 'plasma') & eligible.Time_Aligned.notna()]
    n_dated = (plasma_dated.groupby('Individual_ID')['Time_Aligned'].nunique() >= 2).sum()
    if n_dated < 10:
        raise RuntimeError(f'Only {n_dated} persons with >=2 distinct measured plasma timepoints')
    raw = wide.reindex(eligible.SampleID).to_numpy(dtype='float32')
    manifest = {'sources': provenance, 'n_samples_total_ssl': len(eligible),
                'n_unique_people_total_ssl': int(eligible.Individual_ID.nunique()),
                'biofluid_sample_counts_ssl': eligible.source_cohort.value_counts().to_dict(),
                'protein_union_features': len(wide.columns),
                'people_with_serial_plasma': int(n_dated),
                'serum_chronology': 'ABSENT: serum used only for masked-value SSL',
                'restriction': 'Patient-grouped partitions; forecasts use real dated plasma visits only',
                'not_HCT': True, 'synthetic_training_data_used': False}
    save_json(ROOT / 'protein_download_audit.json', manifest)
    log(f'Proteomics: SSL samples {manifest["biofluid_sample_counts_ssl"]}; {n_dated} plasma-serial patients; {len(wide.columns)} union proteins')
    return raw, list(wide.columns), eligible, 'Time_Aligned', manifest

def patient_split(serial_people):
    people=np.asarray(sorted(set(serial_people)))
    if len(people)<10:raise RuntimeError('Too few individuals for patient-independent split')
    rng=np.random.default_rng(CFG['seed']);people=rng.permutation(people)
    nte=max(2,round(len(people)*.20));nva=max(2,round(len(people)*.20))
    return set(people[nte+nva:]),set(people[nte:nte+nva]),set(people[:nte])

def make_pairs(eligible, tcol, allowed):
    histories=[]
    for grouping,grp in eligible.groupby(['Individual_ID','source_cohort'],sort=True) if 'source_cohort' in eligible.columns else eligible.groupby('Individual_ID',sort=True):
        patient=grouping[0] if isinstance(grouping,tuple) else grouping
        if patient not in allowed:continue
        grp=grp[grp[tcol].notna()].sort_values(tcol)
        if grp.empty:continue
        # No repeated/dropped timestamps: at a particular day, retain first actual visit.
        grp=grp.drop_duplicates(subset=[tcol],keep='first')
        indices=grp.index.to_numpy();times=grp[tcol].to_numpy(dtype='float32')
        for j in range(1,len(indices)):
            gap=float(times[j]-times[j-1])
            if not (0<gap<=CFG['max_delta_days']):continue
            first=max(0,j-CFG['max_history_visits'])
            historic=indices[first:j].tolist()
            relative=(times[first:j]-times[j-1]).tolist()
            histories.append((historic,relative,int(indices[j]),gap))
    return histories

def protein_workflow():
    log('MODEL B objective: masked REAL protein reconstruction (MSE), then next-measured-visit forecasting (observed protein MSE); NOT GVHD prediction')
    work=ROOT/'model_B_proteomic_temporal';work.mkdir(exist_ok=True)
    raw,proteins,meta,tcol,provenance=get_protein_tables()
    # Only persons with genuine repeated plasma dates are eligible for task splits.
    timed_plasma=meta[(meta.source_cohort=='plasma') & meta[tcol].notna()]
    serial_counts=timed_plasma.groupby('Individual_ID')[tcol].nunique()
    serial_people=set(serial_counts[serial_counts>=2].index)
    if len(serial_people)<10:
        raise RuntimeError('Insufficient real plasma longitudinal patients for three-way split')
    tra_grp,val_grp,tes_grp=patient_split(sorted(serial_people))
    # Untimed serum-only and plasma-singleton patients train the masked-protein
    # encoder only; they NEVER enter validation/test or future-state forecasting.
    aux_ssl_people=set(meta.Individual_ID.astype(str)) - serial_people
    ssl_train_people=tra_grp | aux_ssl_people
    ids=meta['Individual_ID'].astype(str).values
    tr=np.where(np.isin(ids,list(ssl_train_people)))[0]
    va=np.where(np.isin(ids,list(val_grp)))[0]
    te=np.where(np.isin(ids,list(tes_grp)))[0]
    assert set(ids[tr]).isdisjoint(ids[va]) and set(ids[tr]).isdisjoint(ids[te]) and set(ids[va]).isdisjoint(ids[te])
    X,obs,sel,scaler=preprocessor(raw,tr,top_k=CFG['protein_features'],min_observed=.30)
    feat=[proteins[i] for i in sel]
    trainpairs=make_pairs(meta,tcol,tra_grp);valpairs=make_pairs(meta,tcol,val_grp)
    testpairs=make_pairs(meta,tcol,tes_grp)
    if min(map(len,(trainpairs,valpairs,testpairs)))<2:
        raise RuntimeError(f'Not enough longitudinal pairs in train/validation/test: {[len(a) for a in (trainpairs,valpairs,testpairs)]}')
    manifest={'data':provenance,'modality':'Timed Olink plasma + untimed Olink serum log2 NPX with explicit missingness',
              'source_disease':'COVID-19 in dialysis patients, NOT allo-HCT (timed plasma; untimed serum SSL only)',
              'time_axis':tcol,'target':'next observed plasma protein profile; serum for masked SSL only',
              'patients':{'train_serial':len(tra_grp),'validation_serial':len(val_grp),'test_serial':len(tes_grp),
                          'train_ssl_only_additional':len(aux_ssl_people)},
              'visit_counts':{'train':len(tr),'validation':len(va),'test':len(te)},
              'adjacent_plasma_pairs':{'train':len(trainpairs),'validation':len(valpairs),'test':len(testpairs)},
              'serum_usage':'Masked-protein SSL only; serum has no measurement dates',
              'temporal_attention':'causal, PLASMA only, max history visits='+str(CFG['max_history_visits']),
              'proteins_selected':len(feat),'heldout_by':'Individual_ID',
              'objectives': TRAINING_OBJECTIVES['model_B_proteomic_temporal'],
              'synthetic_training_data_used': False}
    save_json(work/'data_audit.json',manifest)
    identity=stable_hash({'model':'protein','cfg':model_shape_settings(),'manifest':manifest,'features':feat})
    model=mkmodel(len(feat))
    def ssl_fn(m,b,training):return masked_loss(m,b[0],b[1])[0]
    fit(model,SampleSet(X,obs,tr),SampleSet(X,obs,va),'ssl',work,ssl_fn,identity,feat,scaler,manifest)
    def task_fn(m,b,training):
        hist,seen,days,padded,last,y,valid,gap=b
        z=m.encode_history(hist,seen,days,padded)
        forecast=m.predict_future(z,last,gap)
        return ((forecast-y).square()*valid.float()).sum()/valid.float().sum().clamp_min(1)
    fit(model,HistorySet(X,obs,trainpairs),HistorySet(X,obs,valpairs),'task',work,task_fn,identity,feat,scaler,manifest)
    model.eval()
    forecasts=[];actual=[];persistence=[];maskrows=[];predrows=[]
    with torch.no_grad():
        test_loader=DataLoader(HistorySet(X,obs,testpairs),batch_size=1,shuffle=False)
        for (indices,relative,b,gap),batch in zip(testpairs,test_loader):
            a=indices[-1]
            hist,seen,days,padded,last,y,valid,delta=[q.to(DEVICE) for q in batch]
            contextual=model.encode_history(hist,seen,days,padded)
            pred=model.predict_future(contextual,last,delta).cpu().numpy()[0]
            forecasts.append(pred);actual.append(X[b]);persistence.append(X[a]);maskrows.append(obs[b])
            predrows.append({'individual_id':str(meta.loc[a,'Individual_ID']),
                'source_cohort':str(meta.loc[a,'source_cohort']),
                'baseline_sample_id':str(meta.loc[a,'SampleID']),
                'followup_sample_id':str(meta.loc[b,'SampleID']),
                'history_visits':len(indices),'delta_days':gap,
                'rmse_transformer':float(np.sqrt(np.mean((pred[obs[b]]-X[b,obs[b]])**2))),
                'rmse_persistence':float(np.sqrt(np.mean((X[a,obs[b]]-X[b,obs[b]])**2)))})
    pred=np.array(forecasts);truth=np.array(actual);base=np.array(persistence);msk=np.array(maskrows)
    def mse(a,b):return float(np.square(a-b)[msk].mean())
    scores={'transformer_test_mse':mse(pred,truth),'persistence_test_mse':mse(base,truth),
            'n_test_pairs':len(testpairs),'n_test_patients':len(tes_grp),
            'model_beats_persistence':mse(pred,truth)<mse(base,truth),
            'biofluid_test_pairs':pd.Series([r['source_cohort'] for r in predrows]).value_counts().to_dict(),
            'label':'Measured PLASMA temporal forecasting; serum used only for SSL; NOT GVHD prediction'}
    # Patient-level uncertainty preserves true clustering of adjacent visits.
    patient_rows=pd.DataFrame(predrows)
    patient_rows['mse_gain']=patient_rows.rmse_persistence**2-patient_rows.rmse_transformer**2
    by_patient=patient_rows.groupby('individual_id').mse_gain.mean().values
    rng=np.random.default_rng(CFG['seed']);boots=[float(np.mean(rng.choice(by_patient,size=len(by_patient),replace=True))) for _ in range(2000)]
    scores['patient_average_mse_gain']=float(np.mean(by_patient))
    scores['patient_bootstrap_95ci_mse_gain']=[float(v) for v in np.quantile(boots,[.025,.975])]
    save_json(work/'results.json',scores)
    if not scores['model_beats_persistence']:log('SCIENTIFIC WARNING: Model B failed to improve on persistence in the held-out cohort.')
    pd.DataFrame(predrows).to_csv(work/'heldout_pair_predictions.csv',index=False)
    checkpoint_write(work/'proteomic_temporal_model.pt',{'model_state_dict':model.cpu().state_dict(),
        'model_config':model_config(model),'feature_names':feat,'preprocessing':scaler,
        'data_audit':manifest,'metrics':scores,'training_status':'trained',
        'objectives':TRAINING_OBJECTIVES['model_B_proteomic_temporal'],
        'synthetic_training_data_used':False})
    fig_curves(work,'Model B: longitudinal plasma proteomics')
    (work/'MODEL_CARD.md').write_text('''# Model B — temporal plasma and serum proteomics\n\n**REAL measured data:** Gisby Olink COVID/dialysis plasma and serum subcohorts (NOT HCT). Both enter SSL; only dated plasma enters temporal forecasts.\n\n**Objective B1:** reconstruct masked observed proteins. Loss: masked MSE.\n\n**Objective B2:** forecast the next measured PLASMA visit from prior visits. Loss: MSE on observed target proteins. Serum has no collection-time metadata and is SSL-only.\n\n**Selection:** best validation loss; held-out patient MSE versus persistence.\n\n**Limitations:** Not a GVHD predictor; next observed visit is not a fixed horizon; no paired HCT clinical outcomes.\n\n**Synthetic observations used for training:** None.\n\nSee `TRAINING_OBJECTIVES.json` at run root and local `data_audit.json`, `results.json`.\n''')
    log(f'MODEL B completed: {scores}')
    optional_gvhd_protein_finetune(work,feat,scaler,identity)
    return scores

def optional_gvhd_protein_finetune(work,ref_features,source_scaler,source_identity):
    """Optional separately provided patient-labeled HCT measured-protein CSV.

    Required patient_id, sample_id, gvhd columns; features must EXACTLY match
    Olink tokens e.g 'Inflammation|CXCL9'. No RNA-protein substitution.
    This branch is not enabled for the default public-only run.
    """
    file=CFG['external_gvhd_proteins_csv']
    if not file:
        log('No verified patient-labeled GVHD protein matrix configured; skipping GVHD protein classification.')
        return
    df=pd.read_csv(file)
    req={'patient_id','sample_id','gvhd'}
    if not req.issubset(df.columns):raise RuntimeError(f'External GVHD protein data missing required columns {req-set(df.columns)}')
    if df.sample_id.duplicated().any():raise RuntimeError('Duplicate HCT samples')
    if not df.gvhd.isin([0,1]).all():raise RuntimeError('GVHD labels must be 0 or 1')
    overlap=[f for f in ref_features if f in df.columns]
    if len(overlap)<20:raise RuntimeError(f'Insufficient protein-name matched features ({len(overlap)}); no cross-platform guessing')
    # Maintain pretrained feature order, mask absent proteins explicitly.
    patients=np.array(df.patient_id.astype(str))
    patient_labels=df.groupby('patient_id').gvhd.nunique()
    if (patient_labels>1).any():raise RuntimeError('Inconsistent patient labels; define landmark-specific task first')
    uniq=df.groupby('patient_id').gvhd.first()
    if len(uniq)<20 or min(uniq.value_counts())<5:
        raise RuntimeError('Too few individually labeled patients for HCT proteomic fine-tuning')
    utrain,utest=train_test_split(uniq.index.to_numpy(),test_size=.2,stratify=uniq.values,random_state=CFG['seed'])
    utrain,uval=train_test_split(utrain,test_size=.25,stratify=uniq.loc[utrain].values,random_state=CFG['seed']+1)
    tr=np.where(np.isin(patients,utrain))[0];va=np.where(np.isin(patients,uval))[0];te=np.where(np.isin(patients,utest))[0]
    raw=np.full((len(df),len(ref_features)),np.nan,dtype='float32')
    for j,f in enumerate(ref_features):
        if f in df.columns:raw[:,j]=pd.to_numeric(df[f],errors='coerce').values
    # Use HCT training-fold restandardization, retain original protein token IDs.
    observed=np.isfinite(raw)
    med=np.nanmedian(raw[tr],axis=0);med=np.nan_to_num(med,nan=0)
    mean=np.mean(np.where(observed[tr],raw[tr],med),axis=0)
    std=np.maximum(np.std(np.where(observed[tr],raw[tr],med),axis=0),1e-3)
    X=np.clip((np.where(observed,raw,med)-mean)/std,-8,8).astype('float32')
    model=mkmodel(len(ref_features))
    model.load_state_dict(chk_read(work/'proteomic_temporal_model.pt')['model_state_dict'])
    y=df.gvhd.to_numpy(dtype='float32')
    bce=nn.BCEWithLogitsLoss()
    def fn(m,b,training):
        z,_=m(b[0],b[1]);return bce(m.classification(z).flatten(),b[2])
    out=Path(work)/'external_HCT_proteomics_GVHD'
    iden=stable_hash({'source':'user HCT proteomics', 'file_sha256':sha256(file),
                      'patients':[sorted(map(str,x)) for x in (utrain,uval,utest)],'ref_identity':source_identity})
    pre={'median':med.tolist(),'mean':mean.tolist(),'std':std.tolist(),'features':ref_features}
    fit(model,LabelSet(X,observed,y,tr),LabelSet(X,observed,y,va),'task',out,fn,iden,ref_features,pre,
        {'source_file_sha256':sha256(file),'split_unit':'patient_id','n_patients':len(uniq)})
    model.eval()
    with torch.no_grad():
        z,_=model(torch.from_numpy(X[te]).to(DEVICE),torch.from_numpy(observed[te]).to(DEVICE))
        pred=torch.sigmoid(model.classification(z).flatten()).cpu().numpy()
    score={'test_roc_auc':float(roc_auc_score(y[te],pred)) if len(np.unique(y[te]))==2 else None,
           'n_heldout_patients':len(utest),'task':'GVHD case-control, clinical onset not verified'}
    save_json(out/'results.json',score)
    checkpoint_write(out/'hct_proteomics_gvhd_model.pt',{'model_state_dict':model.cpu().state_dict(),
        'model_config':model_config(model),'feature_names':ref_features,'preprocessing':pre,
        'scores':score,'source':'user-supplied verified HCT proteomics'})
    log(f'External GVHD proteomic fine tuning complete: {score}')

def read_training_artifact(path):
    """Inference loader; enforces exact feature mapping at deployment."""
    snap=chk_read(path)
    conf=snap['model_config']
    model=OmicsTransformer(conf['n_features'],conf['dim'],conf['heads'],conf['layers'],conf['dropout'],conf['latent_tokens'])
    model.load_state_dict(snap['model_state_dict']);model.eval()
    return model,snap

def self_checks():
    """Check configuration without generating any molecular observations or labels."""
    if CFG['run_mode'] != 'REAL_PUBLIC_DATA_ONLY' or CFG['synthetic_training_allowed']:
        raise RuntimeError('Real-public-data-only invariant violated.')
    if 'streamed from official GPL' not in (gpl_feature_map.__doc__ or ''):
        raise RuntimeError('Old GEO annotation loader still active. Restart runtime and run v3.2 from top.')
    if 'Serum: SSL ONLY' not in (get_protein_tables.__doc__ or ''):
        raise RuntimeError('Old serum loader still active. Restart runtime and run v3.2 from top.')
    if CFG['dim'] % CFG['heads']:
        raise ValueError('Model dimension must be divisible by attention head count.')
    if not (0 < CFG['mask_fraction'] < 1):
        raise ValueError('Self-supervised mask fraction must be between zero and one.')
    if CFG['max_history_visits'] < 1:
        raise ValueError('Need at least one real prior visit for temporal forecasting.')
    log('HCT-FORMER M4-MPS VERIFIED: cross-attention latent encoder; real-data only; streaming GEO mapping; serum SSL-only.')
    log('CONFIGURATION CHECK PASSED: only measured public datasets enter training.')
    log(f'MPS DEVICE={DEVICE} | architecture=latent cross-attention with {CFG["latent_tokens"]} latent tokens')


def predict_samples_from_saved_model(checkpoint, external_features_df, task='classification', delta_days=14):
    """Run inference on an external *sample x exactly named feature* pandas DataFrame.
    Only appropriate when same assay/platform and label/task definitions are used.
    """
    model,snap=read_training_artifact(checkpoint)
    names=snap['feature_names'];pre=snap['preprocessing']
    table=external_features_df.reindex(columns=names)
    raw=table.apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
    mask=np.isfinite(raw)
    med=np.asarray(pre['median'],dtype=np.float32)
    mean=np.asarray(pre['mean'],dtype=np.float32)
    std=np.asarray(pre['std'],dtype=np.float32)
    x=np.clip((np.where(mask,raw,med)-mean)/std,-8,8).astype(np.float32)
    model=model.to(DEVICE)
    with torch.no_grad():
        xt=torch.from_numpy(x).to(DEVICE)
        obs=torch.from_numpy(mask).to(DEVICE)
        lat,_=model(xt,obs)
        if task=='classification':
            audit=snap.get('data_audit',{})
            if 'GVHD' not in str(audit.get('task','')) and 'HCT' not in str(snap.get('source','')):
                raise ValueError('This checkpoint has no GVHD-trained classification head; use forecast.')
            ans=torch.sigmoid(model.classification(lat).flatten()).cpu().numpy()
            return pd.DataFrame({'sample':table.index.astype(str),'gvhd_association_score':ans})
        if task=='forecast':
            if 'proteomic' not in str(checkpoint).lower():
                raise ValueError('This checkpoint has no future-state supervised training.')
            days=torch.tensor([delta_days]*len(x),device=DEVICE,dtype=torch.float32)
            temporal=model.encode_history(xt.unsqueeze(1),obs.unsqueeze(1),torch.zeros((len(x),1),device=DEVICE),torch.zeros((len(x),1),device=DEVICE,dtype=torch.bool))
            future=model.predict_future(temporal,xt,days).cpu().numpy()
            return pd.DataFrame(future*std+mean,columns=names,index=table.index)
        raise ValueError('task must be classification or forecast')

def main():
    self_checks()  # Real-only gate BEFORE any downloads, model initialization or training.
    seed_all(CFG['seed'])
    log(f"Start {CFG['run_mode']} | device={DEVICE}, torch={torch.__version__}, root={ROOT}")
    save_json(ROOT/'run_configuration.json',CFG|{'device':str(DEVICE),'root':str(ROOT)})
    save_json(ROOT/'TRAINING_OBJECTIVES.json', TRAINING_OBJECTIVES)
    manifest=[
      {'id':'GSE295359 T-cells','modality':'HCT single-cell RNA-seq','model':'A SSL optional','access':'Zenodo total_t_cells.h5ad (1.8 GB)','endpoint':'no GVHD outcomes used','note':'up to configured number of real HCT T cells; 10 patients, not independent outcomes'},
      {'id':'GSE103569','modality':'HCT sorted CD8 gene expression','model':'A SSL only','access':'GEO matrix + GPL mapping','endpoint':'none trained','note':'60 samples, non-independent patient-level repeat sorted subsets'},
      {'id':'GSE56495','modality':'HCT cGVHD whole blood','model':'A SSL only','access':'GEO matrix + GPL mapping','endpoint':'none trained','note':'54 arrays, multiple samples from some patients; healthy controls excluded'},
      {'id':'GSE7510','modality':'HCT engraftment blood','model':'A optional SSL','access':'GEO matrix + GPL mapping','endpoint':'none trained','note':'8 HCT recipients'},
      {'id':'Gisby serum','modality':'Olink serum NPX','model':'B','access':'public GitHub CSV','endpoint':'masked-protein reconstruction only (no serum time axis)','note':'non-HCT; no GVHD labels'},
      {'id':'GSE73809','modality':'HCT T-cell microarray','model':'A','access':'GEO series matrix',
       'endpoint':'GVHD contemporaneous status','clinical_limit':'No onset timing or validated unique-patient IDs'},
      {'id':'Gisby-2021-Olink','modality':'plasma/serum proteomics','model':'B','access':'public GitHub CSV',
       'endpoint':'next plasma visit; serum contributes to SSL only','clinical_limit':'Non-HCT, dialysis/COVID cohort'},
      {'id':'PXD031349','modality':'HCT weekly serum 2D gel','model':'future','access':'PRIDE raw data',
       'endpoint':'not incorporated','clinical_limit':'5 individuals; raw 2D-gel data require distinct processing and harmonization'},
      {'id':'GSE182610/GSE182612','modality':'HCT PTCy transcriptomes','model':'future','access':'GEO',
       'endpoint':'not incorporated','clinical_limit':'Patient-outcome linkage and format audit required'},
    ]
    save_json(ROOT/'data_manifest.json',manifest)
    outcome={}
    # Fail explicitly if either stage fails, but retain first model and report failure.
    for title,fn in [('A_HCT_GVHD',geo_expanded_workflow),('B_protein_temporal',protein_workflow)]:
        try:outcome[title]={'status':'complete','metrics':fn()}
        except Exception as e:
            log(f'FAILED {title}: {e}')
            outcome[title]={'status':'failed','error':str(e),'traceback':traceback.format_exc()[-5000:]}
        save_json(ROOT/'RUN_SUMMARY.json',outcome)
    if all(v['status']=='complete' for v in outcome.values()):
        log(f"COMPLETE: two {CFG['run_mode']} model tracks trained. Inspect each model results.json.")
    else:
        raise RuntimeError('One or more tracks failed; read RUN_SUMMARY.json. No false success declaration.')

def doctor():
    print('Platform:', platform.platform(), 'machine:',platform.machine())
    print('Python:',sys.version.split()[0], '| PyTorch:',torch.__version__)
    print('MPS compiled:',torch.backends.mps.is_built(), 'available:',torch.backends.mps.is_available())
    a=torch.ones((128,128),device=DEVICE,requires_grad=True)
    loss=(a@a).square().mean()
    loss.backward()
    torch.mps.synchronize()
    print('MPS forward/backward: PASS | model device:', DEVICE)
    if hasattr(torch.mps,'current_allocated_memory'):
        print('MPS current allocated:',round(torch.mps.current_allocated_memory()/1024**2,1),'MiB')
    print('Root:',ROOT)

if __name__ == '__main__':
    # No command-line options and no separate launch script.
    doctor()
    main()
    print('\nFinished. Results and trained checkpoints:', ROOT, flush=True)
    print('Model A:', ROOT / 'model_A_HCT_transcriptomic' / 'hct_gvhd_model.pt', flush=True)
    print('Model B:', ROOT / 'model_B_proteomic_temporal' / 'proteomic_temporal_model.pt', flush=True)

