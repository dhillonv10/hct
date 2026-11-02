# `distill/` — compact RNA students distilled from RiNALMo

| file | what it is |
|---|---|
| `RESULTS.md` | what the first Colab run measured: the teacher reproduces exactly, a 2.6M CNN lands within 0.037 R2 of it, distillation adds nothing |
| `PILOT_PROTOCOL.md` | the pre-registered pilot (mean ribosome load): question, splits, arms, controls and the five gates |
| `CLINICAL_PROTOCOL.md` | the pre-registered clinical benchmark (ClinVar variants in non-coding RNA genes) and its gates |
| `clinical_colab.ipynb` | the notebook that runs the clinical benchmark on Colab |
| `COLAB_PLAN.md` | the wider project: prior art, how it differs from BIO-Distiller, staged Colab plan, pivots if a gate fails |
| `PLAN.md` | the first draft of the plan; partly superseded by `COLAB_PLAN.md`, kept for the prior-art and licence notes |
| `pilot_colab.ipynb` | the notebook that runs the pilot on Colab |
| `rnakd/` | the package the notebook calls |

## The pilot in one paragraph

Mean ribosome load (Sample et al. 2019) is measured on a library of **random synthetic 5'UTRs** and on **real human 5'UTRs** from
the same assay, so a distribution shift is built into the data and RiNALMo publishes a number on both (giga 0.925 random /
0.860 human). We cache a 650M teacher once, train a ladder of compact CNN students with and without distillation, and ask a
question the prior art has not: does the student keep the teacher's advantage **when the test distribution moves**, or only
in distribution? Controls (permuted teacher targets, untrained teacher) decide whether any gain is transferred knowledge or
regularisation.

## Two benchmarks

**`rnakd.pilot` -- mean ribosome load.** Trained on random synthetic 5'UTRs, tested on real human ones, so a
distribution shift is built in and RiNALMo publishes a number on both. Supervised regression; the teacher is a
released fine-tuned checkpoint.

**`rnakd.clinical` -- ClinVar variants in non-coding RNA genes.** The only clinical endpoint a sequence model can be
put on directly. Zero-shot masked log-odds, no labels and no fine-tuning, so a 650M model and a ~1M student are
scored by exactly the same rule. The students are distilled from the teacher on human non-coding transcripts with the
panel's genes removed, against an honest baseline arm that trains the same small model on the same sequences with no
teacher at all.

## Run it

```bash
python -m rnakd.clinical selftest                    # clinical path, synthetic, no network, no GPU
python -m rnakd.clinical panel    --work W           # ClinVar + GENCODE -> the labelled panel
python -m rnakd.clinical score    --work W --ckpt-lm rinalmo_<size>_pretrained.pt --cfg <size>
python -m rnakd.clinical transfer --work W --ckpt-lm rinalmo_giga_pretrained.pt --cfg giga
python -m rnakd.clinical distil   --work W --presets s1,s5 --seeds 0,1,2
python -m rnakd.clinical report   --work W

python -m rnakd.pilot selftest                       # whole path on synthetic data, no network, no GPU, ~5 min
python -m rnakd.pilot dumpkeys --ckpt-ft <file.pt>   # look at a checkpoint before trusting it
python -m rnakd.pilot data      --work W
python -m rnakd.pilot teacher   --work W --ckpt-ft rinalmo_giga_mrl_ft.pt     # GATE P1 + one cached teacher pass
python -m rnakd.pilot baselines --work W             # k-mer ridge and frozen-teacher probes (CPU, seconds)
python -m rnakd.pilot train     --work W --grid main|controls|frac|all
python -m rnakd.pilot report    --work W
```

Every trained run appends a row to `W/runs.csv` and a run already in it is skipped, so re-issuing a command after a disconnect
resumes. `train` never loads the teacher: it reads the cache written by `teacher`.

## What the selftest should look like

`selftest` runs on synthetic data against a **randomly initialised** teacher, so it checks plumbing, not science. Expect
gate P1 to report NOT MET (a random teacher cannot reproduce a published R2) and expect the `kd_logit` arm to collapse to a
large negative R2: with alpha = 1 the student is pulled as hard toward the teacher's predictions as toward the labels, and a
random teacher's predictions are noise. Both are the arms behaving correctly. What the selftest proves is that every stage
runs, the ledger resumes, and the report applies the gates.

## Design notes worth knowing before reading results

* **The teacher is re-implemented, not imported.** `rinalmo/model/attention.py` imports `flash_attn` unguarded, so the official
  package cannot be imported without it and its flash path is CUDA + fp16/bf16 only. `rnakd/teacher.py` is the same module tree
  in plain PyTorch, with parameter names kept identical so `load_state_dict(strict=True)` is the correctness test, and the
  published R2 is the end-to-end check (gate P1). The same approach was validated once before on a released 650M RiNALMo
  fine-tune (`compare/oligoai_score.py`).
* **`Random7600` is their validation set.** RiNALMo's code selects checkpoints on it, so the published 0.925 is not a clean
  held-out number. `rnakd/mrl.py` reproduces their construction and adds `iid_test`, the next block of random-library UTRs
  chosen by the same rule and never used for selection.
* **The projection head used for feature distillation is discarded after training** and is not counted in a student's parameter
  total; the count printed in `runs.csv` is the released model.
* **`ctrl_shuffled` and `ctrl_untrained` exist to be believed.** A distillation gain that these controls also produce is a
  regularisation effect; the protocol says so in advance and the report prints the verdict.

## Licences of what this touches

RiNALMo code Apache-2.0, weights CC BY 4.0 (Zenodo 15043668) — not the `multimolecule` re-host, whose cards declare AGPL-3.0 and
whose licence FAQ covers weights derived from the ones it hosts. MRL data from GEO GSE114002 (check the series' own terms before
redistribution; we download, we do not redistribute). BIO-Distiller's repository has no licence file, so nothing is copied from
it; its loss definitions are re-implemented from the paper and cited.
