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

## Step 2 — just run the family (~33 h, resumable)

```bash
python3 run_family.py --bundle hl_bundle2_human.npz --mouse-bundle hl_bundle2_mouse.npz
```

Interrupt freely — finished runs are skipped on restart. `--dry-run` prints the plan.

| phase | what | time |
|---|---|--:|
| **0** | **read-out ablation at `base`, 3 folds — the deciding phase** | **~2.6 h** |
| A | 5 sizes × 5 folds — the scaling curve | ~13 h |
| B | best size × 10 folds × 2 seeds — the headline | ~10 h |
| C | mouse, 10 folds | ~5 h |

Phase 0 reports first, so you get the answer that matters inside three hours and the driver prints an
explicit **stop** instruction if it fails.

### What phase 0 is testing, and why it is the deciding phase

Structured pooling alone gave `nn = 0.631` twice — before and after the LR fix and the structural scalars —
against Saluki's **0.7443** for the same class of input (sequence + splice). The diagnosis: collapsing ~768
positions into 20 pooled cells destroys the long-range *arrangement*, so the network cannot use junction
positions and performs like a model with no splice track at all. Saluki downsamples to a short sequence and
runs a **GRU** over it. Recurrent integration in order is the one mechanism it has that this model lacked —
structured pooling was validated on 25–100 nt UTRs and then applied to 12 kb transcripts without rechecking
that the bottleneck still made sense.

| read-out | what it is |
|---|---|
| `pool` | structured pooling only — reproduces the 0.631 result as the control |
| `gru` | recurrence over the downsampled sequence, replacing pooling |
| `both` | concatenated |

**Falsifiable expectation:** if the read-out was the bottleneck, `gru` and `both` land at `nn` **0.70–0.75**
(we have strictly more input than Saluki — their 6 tracks plus explicit landmark distances — and now the
same integration mechanism). If they land near 0.65, the bottleneck is elsewhere, phases A–C are not worth
~30 h, and the driver says so in those words.

Read **`nn`**, not `stack`. `stack` will beat 0.70 regardless because it can lean on the GBM; only `nn`
says whether the sequence model works.

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
