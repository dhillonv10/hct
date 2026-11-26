# Landmark-anchored mRNA half-life model family — run on an M4 Mac

## Setup

```bash
python3 -m venv ~/venv-landmark && source ~/venv-landmark/bin/activate
pip install torch numpy scikit-learn    # pip torch on macOS arm64 ships MPS support
```

Nothing else. No genome, no GENCODE, no ViennaRNA — sequences, exon structure, all 609 engineered features
and every label are baked into the two bundles.

## Run in this order

```bash
# 1. ~3 min sanity check.
python3 train_landmark.py --bundle hl_bundle2_human.npz --selftest --multitask

# 2. The decisive 45-minute run: sequence-only network + GBM, fused late.
python3 train_landmark.py --bundle hl_bundle2_human.npz --quick --multitask \
    --aux-drop 1.0 --stack-gbm

# 3. The whole family, if step 2 looks right. Resumable.
python3 run_family.py --bundle hl_bundle2_human.npz --mouse-bundle hl_bundle2_mouse.npz
```

`sklearn` is now required as well (`pip install torch numpy scikit-learn`).

## Why the design changed after the first run

The first version concatenated the 609 engineered features into the network's head. The first real run showed
exactly what that does:

```
ep   1/25  loss 0.7205  val r 0.6741  *
...
ep  25/25  loss 0.4051  val r 0.6677
best val r 0.6751 at epoch 8
```

Val r is reached in **one epoch** and never improves while training loss halves. The head had a shortcut —
the features alone get ~0.67 — so the convolutional trunk never received a useful gradient and learned
nothing. The hybrid scored **0.693, below the 0.7023 that gradient boosting gets on those features alone**.
The claim that the features were a "floor the network can only add to" was wrong: the floor is an *MLP* on
the features (~0.67), not a *GBM* on them (0.702).

So the network now runs on **sequence only** (`--aux-drop 1.0`) and is fused **late** with a GBM trained
inside the same fold on the same training rows (`--stack-gbm`), with the combination weights fitted on the
inner validation split. Every run reports three numbers per fold:

| component | what it is | compare against |
|---|---|--:|
| `nn` | the landmark network, sequence only | Saluki sequence-only **0.608** |
| `gbm` | the 609 features alone | our **0.7023** |
| `stack` | the two combined | Saluki full **0.7726** |

This removes the shortcut, makes the floor real, and folds the old "phase D" mechanism control into every
single run — `nn` above 0.608 means the landmark encoding extracts something Saluki's convolutions do not.

## The family

| size | params |
|---|--:|
| tiny | 0.24M |
| small | 0.77M |
| base | 2.05M |
| large | 4.91M |
| xl | 11.43M |

For scale, RiNALMo-giga is 651M. `tiny` is 1/2700th of it.

## Benchmarks and targets

| | mean r over folds |
|---|--:|
| **Saluki human (the number to beat)** | **0.7726** |
| **Saluki mouse** | **0.7320** |
| Saluki sequence-only ablation (phase D reference) | 0.608 |
| BEeM — the authors' own engineered baseline | 0.7681 |
| BC3MS | 0.6732 |
| our 609 features + gradient boosting | 0.7023 |

All printed automatically with the sign of the difference, and the right reference is chosen per species and
per arm.

## Data in the bundles

| | human | mouse |
|---|--:|--:|
| transcripts | 16,082 | 16,792 |
| scorable on PC1 | 13,817 | 14,406 |
| train-only (no PC1, ≥3 individual datasets) | 2,265 | 2,386 |
| auxiliary decay datasets | 54 | 27 |
| observed label cells | **482,082** | **205,730** |

The authors' PC1 target compresses 54 independent decay experiments into one number per gene. Training the
54 heads gives the trunk 25× the supervision for the same sequences; PC1 alone is still what gets reported.
The train-only genes are marked by a `scored` mask that the fold construction respects, so evaluation can
never touch them.

## If the full run is too long

`--bench` tells you before you commit. Then: raise `--bs` (32 or 64 if memory allows), drop to
`--lmax 8192` (truncates 9.7% of transcripts vs 2.2% at 12288), or pass `--sweep-folds 3`. Fewer folds only
makes the mean noisier; it is not a shortcut around the comparison. On 4 CPU cores, `base` is 680 ms/step at
bs=16/lmax=12288; MPS should cut that several-fold.

If MPS errors on an op the script already sets `PYTORCH_ENABLE_MPS_FALLBACK=1`; failing that add
`--device cpu` and send me the traceback.

## Honest caveat to keep attached to any number

Labels, gene set and target are the authors' own (PC1 of their 39-dataset decay compendium). Transcript
sequences are rebuilt from GENCODE V50 / VM37 with **our** representative-transcript rule, and the CV folds
are **ours**. Same task, same labels, different sequence reconstruction — not a byte-identical replication.

Honest estimate of beating 0.7726: roughly 45–55%, and phase D decides whether a win is architectural.

## Files

* `train_landmark.py` — model, training, evaluation. Self-contained.
* `run_family.py` — the four-phase driver. Resumable.
* `build_data.py` — how the bundles were made (needs the genome; you do not need to run it).
* `hl_bundle2_human.npz`, `hl_bundle2_mouse.npz` — everything above.
* `ARCHITECTURE.md` — the design and the evidence behind it.
