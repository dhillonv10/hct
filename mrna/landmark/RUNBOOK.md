# Runbook — what to run, in order, and how to read it

## Setup

```bash
python3 -m venv ~/venv-landmark && source ~/venv-landmark/bin/activate
pip install torch numpy scikit-learn
```

## Step 1 — sanity (3 min)

```bash
python3 train_landmark.py --bundle hl_bundle2_human.npz --selftest --multitask --aux-drop 1.0 --stack-gbm
```
Numbers are meaningless (240 genes, 2 epochs). You are checking that it runs and prints `nn / gbm / stack`.

## Step 2 — the decisive run (~45 min)

```bash
python3 train_landmark.py --bundle hl_bundle2_human.npz --quick --multitask \
    --aux-drop 1.0 --stack-gbm --out seqonly_quick.json
```

Three folds, 25 epochs, lmax 8192. Read **`nn`** — the sequence-only network, no engineered features:

| `nn` | meaning | what to do |
|---|---|---|
| **≥ 0.60** | beats Saluki's own sequence-only ablation (0.608). The landmark encoding works. | run step 3 |
| **0.45 – 0.60** | partially working | send me the per-epoch curves; likely a schedule fix |
| **< 0.45** | the architecture is not delivering | **stop** — do not spend 35 h on the sweep |

`stack` should beat 0.7023 regardless, because it can lean on the GBM. That alone is **not** evidence the
architecture works — only `nn` is.

Also send me the first fold's per-epoch lines. Where validation peaks tells me whether the learning-rate
schedule needs changing (last time it peaked at epoch 8 of 25, which means OneCycle never annealed).

## Step 3 — the family (~35 h, resumable)

```bash
python3 run_family.py --bundle hl_bundle2_human.npz --mouse-bundle hl_bundle2_mouse.npz
```

Interrupt it freely — finished runs are skipped on restart. `--dry-run` prints the plan.

| phase | what | time |
|---|---|--:|
| A | 5 sizes × 5 folds — the scaling curve | ~15 h |
| B | best size × 10 folds × 2 seeds — the headline | ~11 h |
| C | mouse, 10 folds | ~6 h |

Pre-registered expectation for A: **the curve saturates early.** With ~14k scorable genes capacity is not
the binding constraint. The driver reports "best" and "cheapest within 0.005 of best" and hands the
**cheapest** to B and C — if they differ, the small model is the honest release.

## Targets

| | r |
|---|--:|
| Saluki human (full) | **0.7726** |
| Saluki mouse | **0.7320** |
| Saluki sequence-only ablation | **0.608** |
| BEeM — their engineered baseline | 0.7681 |
| BC3MS | 0.6732 |
| our 609 features + GBM | 0.7023 |

## Flags worth knowing

| flag | effect |
|---|---|
| `--aux-drop 1.0` | network sees sequence only. **Always use this**; the shared head was a shortcut. |
| `--stack-gbm` | trains the GBM in-fold and fuses late. Needs sklearn. |
| `--multitask` | 54 (human) / 27 (mouse) auxiliary decay-dataset heads — 25× the supervision |
| `--size` | tiny / small / base / large / xl |
| `--bench` | times 20 real steps on your hardware and extrapolates |
| `--lmax` | 12288 truncates 2.2% of transcripts; 8192 truncates 9.2% and runs ~1.5× faster |
| `--seeds` | averages predictions over N seeds per fold |

## If something breaks

MPS op errors: `PYTORCH_ENABLE_MPS_FALLBACK=1` is already set; failing that add `--device cpu` and send me
the traceback. Out of memory: lower `--bs`. Anything else: send the last ~30 lines of output.

## The caveat that goes next to any number

Labels, gene set and target are the Saluki authors' own (PC1 of their 39-dataset decay compendium).
Transcript sequences are rebuilt from GENCODE V50 / VM37 with **our** representative-transcript rule, and
the CV folds are **ours**. Same task and labels, different sequence reconstruction — not a byte-identical
replication.
