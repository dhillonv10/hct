# Landmark-anchored mRNA half-life model family — run on an M4 Mac

## Setup

```bash
python3 -m venv ~/venv-landmark && source ~/venv-landmark/bin/activate
pip install torch numpy          # pip torch on macOS arm64 ships MPS support
```

Nothing else. No genome, no GENCODE, no ViennaRNA — sequences, exon structure, all 609 engineered features
and every label are baked into the two bundles.

## Run in this order

```bash
# 1. ~3 min. Exercises every path: multi-task heads, both species, the sequence-only control.
python3 train_landmark.py --bundle hl_bundle2_human.npz --selftest --multitask
python3 train_landmark.py --bundle hl_bundle2_mouse.npz --selftest --multitask

# 2. ~1 min. Times 20 real steps on YOUR hardware and extrapolates the whole family.
python3 train_landmark.py --bundle hl_bundle2_human.npz --bench --bs 16 --size base

# 3. The whole family. Resumable — re-run after an interruption and finished phases are skipped.
python3 run_family.py --bundle hl_bundle2_human.npz --mouse-bundle hl_bundle2_mouse.npz
```

`run_family.py --dry-run` prints the plan without running anything.

## What the four phases do, and why in that order

| phase | what | why it comes here |
|---|---|---|
| **A** | 5 sizes, human, 5 folds | the scaling curve. Decides which size phases B–D use. |
| **B** | best size, human, 10 folds, 2 seeds | the headline number vs Saluki 0.7726 |
| **C** | same size, mouse, 10 folds | second published benchmark, Saluki 0.7320 |
| **D** | `--aux-drop 1.0`, features off | **the control that can kill the headline** |

**Phase D is the important one.** The head normally sees the 609 engineered features, which reach r=0.702
alone — so a good phase-B number is uninterpretable without knowing whether the sequence model contributed.
`--aux-drop 1.0` switches the features off entirely. Saluki's own sequence-only ablation is **0.608**:

* phase D **above 0.608** → the landmark encoding extracts something their convolutions miss. Real result.
* phase D **at or below 0.608** → any phase-B win came from the features, not the architecture, and the
  write-up has to say so.

**Pre-registered expectation for phase A: the curve saturates early.** With ~14k scorable genes, capacity is
not the binding constraint. The driver prints both "best" and "cheapest within 0.005 of best" and hands the
**cheapest** to later phases. If those differ, the small model is the honest release — and "mRNA half-life
prediction saturates at N parameters" is a result nobody has published.

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
