# Training EpiTrajectory-L on a MacBook (Apple-silicon GPU / MPS only)

`scripts/train_mps.py` is the single entry point. It is MPS-only: it refuses to start without the Apple GPU.
It was developed in a Linux cloud container, where the whole pipeline (data, model, trainer, safeguards, resume, crash
recovery) was tested on CPU. **The MPS-specific numerics could not be tested there**, so the run begins with a pre-flight that
checks them on your machine in minutes. A multi-day run only starts if every pre-flight check passes.

## 0. Offline path (recommended): use the data bundle instead of sections 1-2
```bash
tar -xzf epimod_bundle.tar.gz && cd epimod_bundle           # -> data/, code.tar, MANIFEST.json  (~0.45 GB; ~1.2 GB unpacked)
mkdir code && tar -xf code.tar -C code
python3 -c "import platform; print(platform.machine())"      # must print arm64
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./code/epimod pytest                          # the ONLY step that needs the internet (once)
python code/epimod/scripts/make_bundle.py verify .           # must print: ... ALL OK
export EPIMOD_DATA=$PWD/data
cd code/epimod && pytest -q && python scripts/train_mps.py --name base1 --preset base
```
The bundle holds the compact training data (ComputAge, public GEO series, the label-free GEO series, the twin cohort), the parsed
benchmark prompts, sample metadata, the probe annotation and a code snapshot. Nothing is downloaded at run time.

## 1. Setup (once; online path)
```bash
git clone https://github.com/dhillonv10/hct && cd hct && git checkout claude/gifted-albattani-1hvuiq
python3 -c "import platform; print(platform.machine())"      # must print arm64 (native Python, not Rosetta)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./epimod pytest                               # torch >= 2.2 ships with MPS support
cd epimod && pytest -q                                       # ~1 min; must be all green
export EPIMOD_DATA=$HOME/epimod_data                         # ~20 GB free is enough
```
Mac settings for a long run: keep it **plugged in**, lid **open** (a closed lid sleeps the machine even under `caffeinate` unless
an external display is attached), and turn on *System Settings > Battery > Options > Prevent automatic sleeping when the display is off*.

## 2. Build the data (resumable)
```bash
python scripts/bootstrap.py --workers 6
```
Downloads ComputAge Bench and the public GEO methylation series (keeping only a compact 9,098-CpG panel), the label-free GEO series
for masked-reconstruction pretraining, and the twin cohort (raw IDATs, ~10 GB). Run it again after any interruption. The last stage
is a leakage audit: GSM and series overlap with the benchmark prompts must both be 0. Leukaemia / AML / MDS / myeloma series are
held out of every training set so disease analyses on such cohorts stay clean.

## 3. Train
```bash
python scripts/train_mps.py --name base1 --preset base       # ~3M parameters: hours; do this first
python scripts/train_mps.py --name large1 --preset large     # ~25M parameters: only if base looks healthy
```
1. **Pre-flight** (~5-15 min, first start only): the GPU must reproduce the CPU's loss and gradients for every task on real
   batches; overfitting one fixed batch must cut the loss below 60%; a real mini-run must not diverge, skip an injected NaN, resume from its checkpoint, export a model
   that scores the same on the CPU, and agree with a CPU recomputation of the validation score. It prints `[PASS]` / `[FAIL]` lines and the
   measured speed, from which you can plan the run length. On any `[FAIL]` nothing is started: send me the lines.
2. **Training** under a supervisor: a crash restarts the worker from its last checkpoint (it gives up after 3 restarts without
   progress, or 30 in total). `caffeinate` keeps the Mac awake while it runs.
3. **It stops by itself when validation stops improving.** Validation is on whole held-out series. The learning rate is held after a
   warm-up; if the best validation score improves by less than 0.5% over 6 evaluations (6,000 steps) the LR is halved and the best
   weights restored; after 3 such halvings the run is declared converged. There is no step budget and no fixed schedule, so it
   runs exactly as long as it keeps improving. Slow, steady progress is not mistaken for a plateau.

Safeguards during the run: non-finite or spiking steps are skipped; a run of bad steps rolls back to the last good checkpoint with a
halved LR; checkpoints are written atomically and one generation is kept; every 25 evaluations the validation score is recomputed
on the CPU and compared (a mismatch above 10% stops the run); `status.json` is updated at every evaluation.

## 4. Watch and control
```bash
cat runs/base1/status.json            # step, best score, lr, steps/s, bad steps, elapsed hours
tail -f runs/base1/train.log
touch runs/base1/STOP                 # clean stop at the next step; rerun the same command to resume later
```
The best model so far is always in `runs/base1/model.pt`. When it ends, `runs/base1/summary.json` says why
(`converged`, `stop-file`, `max-hours`, or `unstable` which needs a human).

## 5. Evaluate (score the benchmark ONCE per final model)
```bash
python scripts/lb_baselines.py preds/baselines.json && python scripts/lb_baselines_ext.py preds/baselines_ext.json
python scripts/lb_predict.py runs/large1/model.pt preds/large1.npz
python scripts/hier_impute_eval.py runs/large1/model.pt impute.json                     # methylome completion vs mean / ridge
python scripts/traj_train.py runs/large1/model.pt runs/large1/traj --epochs 8             # dynamics on the twin cohort
python scripts/traj_train.py runs/large1/model.pt runs/large1/traj --epochs 8 --perm      # negative control
python scripts/disease_signal.py --beta your_betas.csv --meta your_meta.csv --age-col age --group-col status --control Control
```
(`lb_report.py` builds the comparison table; see `results/LB_RESULTS.md` section 10 for how the baseline files are merged.)

## 6. Rules to keep
Choose models on the validation series only; score the benchmark once per final model; report paired bootstrap intervals; do not
tune on the held-out AML cohorts.
