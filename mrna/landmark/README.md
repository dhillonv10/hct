# Landmark-anchored mRNA half-life model — run on an M4 Mac

## What you need

Python 3.10+ and two packages. No ViennaRNA, no genome, no GENCODE — the 609 engineered features and all
transcript structure are already baked into the bundle.

```bash
python3 -m venv ~/venv-landmark && source ~/venv-landmark/bin/activate
pip install torch numpy          # pip torch on macOS arm64 ships MPS support
```

## Run it in this order

```bash
# 1. ~2 min. Exercises every code path on 240 transcripts. Numbers are meaningless by design.
python3 train_landmark.py --bundle hl_bundle_human.npz --selftest

# 2. ~1 min. Times 20 real training steps on YOUR hardware and extrapolates the whole run.
python3 train_landmark.py --bundle hl_bundle_human.npz --bench --bs 16

# 3. First real signal: 3 folds, shorter transcripts, 25 epochs.
python3 train_landmark.py --bundle hl_bundle_human.npz --quick

# 4. The real run, once --bench says the wall time is acceptable.
python3 train_landmark.py --bundle hl_bundle_human.npz --seeds 2
```

On 4 CPU cores a full run is ~59 h; MPS should cut that several-fold. If `--bench` still says too long:
raise `--bs` (32 or 64 if memory allows), use `--lmax 8192` (truncates 9.7% of transcripts vs 2.2% at
12288), or `--folds 5`. Fewer folds only makes the mean noisier — it is not a shortcut around the comparison.

If MPS errors on an op, the script already sets `PYTORCH_ENABLE_MPS_FALLBACK=1`; failing that, add
`--device cpu` and send me the traceback.

## What you are comparing against

| | mean r over folds |
|---|--:|
| **Saluki (published, the number to beat)** | **0.7726** |
| BEeM — the authors' own engineered baseline | 0.7681 |
| our 609 features + gradient boosting | 0.7023 |

Printed automatically at the end of every run, with the sign of the difference.

## Honest notes

* **Floor, not ceiling.** The head sees the 609 engineered features, which alone reach 0.702. The network
  can therefore only add. `--aux-drop 0.3` zeroes the whole feature block on 30% of training steps so the
  convolutional path is forced to carry weight instead of being ignored.
* **The comparison is not byte-identical.** Labels, gene set and target are the authors' own (PC1 of their
  39-dataset human decay compendium). Transcript sequences are rebuilt from GENCODE V50 + hg38 with our
  representative-transcript rule, and the CV folds are ours. Same task, same labels, different sequence
  reconstruction. Say that next to any number.
* **It may not work.** Honest estimate of beating 0.7726: roughly 45–55%. The landmark encoding is a real
  advantage for NMD specifically, but Saluki is well-tuned and had the full sequence datapack.

## Files

* `train_landmark.py` — model and training. Self-contained.
* `hl_bundle_human.npz` — 13,817 transcripts: sequence, CDS bounds, exon junctions, 609 features, labels.
* `ARCHITECTURE.md` — what the model does and why, including the evidence it is built on.
