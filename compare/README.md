# Compact models vs much larger models (run on a GPU, e.g. Colab Pro A100/L4)

Open `compare_colab.ipynb` in Colab, upload `hct_compare.zip`, run the cells top to bottom. Send back `aso_compare_report.txt`, `mito_report.txt`, `mito_report_constraint.txt`.

## Comparison A -- ASO knockdown: compact LightGBM vs OligoAI (RiNALMo-giga fine-tune, ~650M parameters)
* S1 (OligoAI's own protocol): same training patents, same 38 test patents (235 screens). Fair to both; gene overlap with training is shared by both.
* S2 (our 81 locked gene groups, 599 screens): OligoAI has seen most of these genes -> favours OligoAI. S2b = the 14 locked groups OligoAI never saw (47 screens; small, wide CI).
* Metric: mean within-screen Spearman (screens with >= 20 ASOs), bootstrap CI over patents (S1) / gene groups (S2).

## Comparison B -- mtDNA variant effect: conservation + annotation scorer vs Nucleotide Transformer v1 500M (486M parameters)
* ClinVar P/LP (113) vs B/LB (1,597) and gnomAD constraint (6,000 variants, matched within substitution x flanking bases), same controls and VERDICT rule as `mito_lm.py`.

## Decision rules (fixed before any real result was seen)
* "Near parity or better": our point estimate is within 0.02 of the larger model (Spearman for A, AUROC for B) or above it, in S1 (A) / the ClinVar and constraint tests (B), AND not more than 0.03 behind on S2b (A).
* "Behind": more than 0.03 below on the primary test. Then the compact models are not a competitive replacement and the deliverable is the benchmarks, controls and audits.
* In between: report as inconclusive; no SOTA claim.
* A large model that merely adds nothing on top of our scorer (VERDICT: NOT demonstrated) is not a loss for the compact model; it is evidence that size is not what matters here.

## What was and was not tested
Tested: OligoAI re-implementation (random tiny checkpoint through the safe-load path, parameter count 651.4M for the real configuration, padding invariance); ASO evaluation (shuffled labels + random stand-in predictions);
Nucleotide Transformer scoring (4 variants on CPU; tokenizer layout verified); external-score merging (random stand-in scores).
Real OligoAI checkpoint: loads strictly; after fixing one bug (residual taken from the normalised input) predictions on 96 of OligoAI's own training rows correlate with labels (Pearson 0.63), and the --check gate (200 training rows, aborts below 0.3) passes. The first Colab run was stopped by that gate with the buggy version (Pearson 0.14), as designed.
NOT tested: the full scoring run on a GPU, and OligoAI's agreement with the authors' own reported numbers.

## Found while reading OligoAI's released code (not run, not yet confirmed against the paper)
* Chemistry vocabulary key 'cET' vs data token 'CET': every cEt position is encoded as padding. Reproduced to match its predictions; measured to have no effect on them (Pearson 0.545 either way on 96 cEt-containing rows), so not a meaningful flaw.
* Train/val/test split is by patent (44% of test rows share a gene group with training).
