# EpiMod

A small, CPU-trainable **transformer for DNA-methylation age** that is *genome-aware*,
*module-decomposed* and *probabilistic*, trained on public Illumina-array data
([ComputAge Bench](https://huggingface.co/datasets/computage/computage_bench)) and scored on
**studies it never saw** with the ComputAge scoring protocol.

It was written as a methylation-native alternative to LLM-based aging models such as Insilico's
Longevity-LLM (Qwen3-14B; [bioRxiv 2026.03.28.714980](https://www.biorxiv.org/content/10.64898/2026.03.28.714980v1)).
See [`docs/DESIGN.md`](docs/DESIGN.md) for what was and was not verifiable about that paper.
This directory is self-contained and does not touch the `HCTFormer_*.py` files at the repo root.

## Results in brief

On the 65 held-out ComputAge Bench studies (details, caveats and ablations in
[`results/RESULTS.md`](results/RESULTS.md)):

* **Age accuracy:** median error on healthy controls **3.93 y** (3.77 y with cell-type-balanced
  sampling) vs 3.68 y for an ElasticNet on the same CpGs/studies, and 4.15 / 5.34 / 7.5–8.0 y for
  Horvath V2 / Horvath V1 / PhenoAge. On whole blood: 2.85 y vs 2.76 y (ElasticNet) vs 3.3–5.5 y
  (published clocks). Competitive; **not better than a linear baseline**.
* **Accelerated aging:** no better than the linear baselines; PhenoAge V2 is far ahead.
* **Weak spots:** purified cell types, saliva, 27K arrays (none are well represented in training);
  90% intervals cover only ~76–86% on new studies.
* **What the design demonstrably adds:** prediction from CpGs the model never saw (r = 0.75, MedAE
  8.5 y, destroyed by shuffling the genomic annotations); exact module-level decomposition of the
  prediction; modest accuracy gains from the linear warm start, module transformer, probabilistic
  loss and augmentation.
* **Not supported by the ablations:** genomic context improving accuracy on seen CpGs; masked-CpG
  pre-training; the acceleration head.
* **Versus Insilico's Longevity-LLM:** their 4.34-year MAE (abstract) is not comparable — evaluation
  set and metric unknown; I could only read the abstract.

## Quickstart

```bash
pip install -e ./epimod
pip install computage --no-deps          # coefficient files for the published-clock baselines
export EPIMOD_DATA=/big/disk/epimod      # ~25 GB peak, ~1 GB kept

epimod prepare                           # download ComputAge Bench, pick a 16k-CpG panel, cache arrays
epimod train --out runs/full --train epochs=30
epimod evaluate --out runs/eval --run full=runs/full
epimod analyze --run runs/full --out runs/analysis
epimod train --out runs/ctx_A --model cpg_split=A use_id_embedding=False linear_skip=False --train epochs=15
epimod transfer --run runs/ctx_A --out runs/transfer      # prediction from CpGs never seen in training
bash scripts/run_ablations.sh            # the full ablation suite
pytest                                   # 30 unit tests
```

Using a trained model:

```python
from epimod.train.loop import load_checkpoint
model = load_checkpoint("checkpoints/epimod_full.pt")          # needs the CpG ids in model.cpg_ids
out = model.predict(beta)                              # beta: [samples, len(model.cpg_ids)], NaN = missing
out["age"], out["age_lo"], out["age_hi"]               # years; 90% interval (calibrated on held-out training studies; overconfident on new ones)
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
