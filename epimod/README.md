# EpiMod

A small, CPU-trainable **transformer for DNA-methylation age** that is *genome-aware*,
*module-decomposed* and *probabilistic*, trained on public Illumina-array data
([ComputAge Bench](https://huggingface.co/datasets/computage/computage_bench)) and scored on
**studies it never saw** with the ComputAge scoring protocol.

It was written as a methylation-native alternative to LLM-based aging models such as Insilico's
Longevity-LLM (Qwen3-14B; [bioRxiv 2026.03.28.714980](https://www.biorxiv.org/content/10.64898/2026.03.28.714980v1)).
See [`docs/DESIGN.md`](docs/DESIGN.md) for what was and was not verifiable about that paper.
This directory is self-contained and does not touch the `HCTFormer_*.py` files at the repo root.

> Results, ablations and caveats: [`results/RESULTS.md`](results/RESULTS.md).

## What is different

| | |
|---|---|
| **Genome-aware CpG tokens** | each CpG is embedded from chromosome, multi-scale genomic position, island/shelf, gene region, regulatory class, enhancer/DHS and probe design (+ an id embedding that is randomly hidden in training) |
| **Epigenetic modules** | K sample-independent soft CpG sets feed a small transformer, so every unit is an inspectable CpG set and 10⁴ CpGs run on a CPU |
| **Exact additive decomposition** | age = bias + Σ per-module contributions |
| **Predictive distribution** | mean and sd of age, with post-hoc calibration on held-out studies |
| **Real-profile augmentation** | masking, borrowed real missingness patterns, logit-space distortion, noise — nothing synthetic is generated |
| **Cross-fitted acceleration head** | optional biological-age stage scored only on held-out studies / disease classes |

Each is switchable (`ModelConfig` / `TrainConfig`), and `scripts/run_ablations.sh` measures them.

## Quickstart

```bash
pip install -e ./epimod
pip install computage --no-deps          # coefficient files for the published-clock baselines
export EPIMOD_DATA=/big/disk/epimod      # ~25 GB peak, ~1 GB kept

epimod prepare                           # download ComputAge Bench, pick a 16k-CpG panel, cache arrays
epimod train --out runs/full --train epochs=30
epimod evaluate --out runs/eval --run full=runs/full
epimod analyze --run runs/full --out runs/analysis
bash scripts/run_ablations.sh            # the full ablation suite
pytest                                   # 27+ unit tests
```

Using a trained model:

```python
from epimod.train.loop import load_checkpoint
model = load_checkpoint("runs/full/model.pt")          # needs the CpG ids in model.cpg_ids
out = model.predict(beta)                              # beta: [samples, len(model.cpg_ids)], NaN = missing
out["age"], out["age_lo"], out["age_hi"]               # years, calibrated 90% interval
out["module_contrib_t"]                                # per-module additive contributions
```

## Layout

```
src/epimod/
  data/      download, ComputAge access, probe annotation, streaming panel statistics, extraction, store
  model/     tokens (CpG context encoder), modules (module pooling), epimod (full model), linear_prior
  train/     loop (masked multi-task training, EMA, calibration), augment, accel (acceleration head)
  eval/      computage_protocol, baselines (published clocks), elasticnet, interpret
  runner.py  analysis.py  cli.py
tests/       unit tests (toy fixtures only for mechanics; never used for training or results)
scripts/     run_ablations.sh
```

## Data policy

Only measured, public, individual-level methylation profiles are used. Augmentation perturbs or
hides parts of *real* profiles; no subject, array or label is simulated. The benchmark studies are
never used to choose CpGs, normalise, stop training or fit the model (the optional acceleration head
is trained only inside a cross-fitting loop and is reported separately).
