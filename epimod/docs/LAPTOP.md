# Running EpiTrajectory-L / EpiSet on your own machine

Everything in this repository was developed in a cloud container (4 CPU cores, no GPU, 2-hour job cap, ephemeral disk). The
code is device-aware (`--device auto|cpu|mps|cuda`), so a laptop GPU lets the same code run bigger models for much longer.
The GPU paths (MPS / CUDA) are **untested** — run the smoke check below first.

## 1. Setup
```bash
git clone https://github.com/dhillonv10/hct && cd hct && git checkout claude/gifted-albattani-1hvuiq
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./epimod pytest        # the default torch wheel includes MPS on Apple silicon and CUDA on Linux/Windows
cd epimod && pytest -q                # 50+ tests, a minute on CPU
export EPIMOD_DATA=$HOME/epimod_data  # ~20 GB free is enough
```

## 2. Build the data (resumable; run it again after any interruption)
```bash
python scripts/bootstrap.py --workers 4            # hours; ComputAge + GEO + label-free GEO + twin cohort (raw IDATs, ~10 GB)
python scripts/bootstrap.py --only geo-unlabeled   # re-run one stage
```
The GEO ingestion is polite to NCBI (one request every 0.4 s for the listing). The kept data is a few GB; downloads are deleted.
Read the `audit` output: GSM / series overlap with the benchmark prompts must be 0 and no LB-0190 profile may match.

## 3. Smoke check (20 steps, ~minutes) — do this before any long run
```bash
python scripts/lb_run.py --model hier --name smoke --out runs --tasks extremes,fixed,clock,pair,mask \
       --steps 20 --bs-fixed 8 --unlabeled --device auto --no-lb-eval
```
If `--device auto` fails on MPS/CUDA, report the traceback; `--device cpu` always works.

## 4. Pretraining presets (all resumable: rerun the same command and it continues from `runs/<name>/ckpt.pt`)
```bash
# base: what was run in the cloud (1.8M parameters)
python scripts/lb_run.py --model hier --name hier_base --out runs --tasks extremes,fixed,clock,pair,mask \
       --steps 3000 --bs-fixed 8 --age-balance 1.0 --unlabeled --device auto --no-lb-eval

# scale-up for a GPU (~15-25M parameters; adjust --bs-mult to your memory)
python scripts/lb_run.py --model hier --name hier_large --out runs --tasks extremes,fixed,clock,pair,mask \
       --d 192 --heads 6 --d-emb 96 --local-layers 3 --global-layers 6 --steps 30000 --bs-mult 2 \
       --bs-fixed 8 --age-balance 1.0 --unlabeled --device auto --no-lb-eval
# keep a Mac awake:  caffeinate -i python scripts/lb_run.py ...
```
Validation is on whole held-out series (never samples); the best checkpoint by validation score is what is saved to `model.pt`.

## 5. Evaluation (needs the baselines once)
```bash
python scripts/lb_baselines.py preds/baselines.json; python scripts/lb_baselines_ext.py preds/baselines_ext.json
python scripts/lb_predict.py runs/hier_large/model.pt preds/hier_large.npz          # scores the benchmark prompts ONCE per model
python scripts/lb_report.py report.md "simple baselines=preds/baselines_all.npz" "EpiTrajectory-L=preds/hier_large.npz" --ref "simple baselines"
python scripts/hier_impute_eval.py runs/hier_large/model.pt impute.json                # methylome completion vs mean / ridge
python scripts/traj_train.py runs/hier_large/model.pt runs/hier_large/traj --epochs 8  # dynamics on the twin cohort
python scripts/traj_train.py runs/hier_large/model.pt runs/hier_large/traj --epochs 8 --perm   # negative control
python scripts/disease_signal.py --beta your_betas.csv --meta your_meta.csv --age-col age --group-col status --control Control   # AML etc.
```
(`preds/baselines_all.npz` is the merge of the two baseline `*_preds.npz` files; see `results/LB_RESULTS.md` §10.)

## 6. Rules of the evaluation (keep them if you change things)
Choose models on the validation series only; score the benchmark once per final model; report paired bootstrap intervals; the
leukaemia / AML / MDS series are held out of training so disease analyses on such cohorts stay clean.
