# HCT T-cell foundation model: run report

profile `smoke`; clinical data were never used for pretraining. All numbers are out-of-fold / held-out.

## Pretraining on held-out healthy donors

* masked-gene reconstruction MSE: model 0.689 | zero-predictor 1.034 | PCA-ridge 0.910
* fine composition JS-divergence (lower=better): model 0.157 | PCA-ridge 0.135 | prior 0.275; top-1 acc model 0.58 vs PCA-ridge 0.67
* lineage composition JS-divergence (lower=better): model 0.111 | PCA-ridge 0.063 | prior 0.182; top-1 acc model 0.55 vs PCA-ridge 0.74
* activation R2: model 0.81 | PCA-ridge 0.84

Context pretraining (held-out donors of inflamed / alloimmune sources; 442 profiles, classes {'healthy': 353, 'inflamed': 65, 'alloimmune': 24}):
* masked-gene reconstruction MSE: model 0.752 | zero-predictor 1.148
* context head balanced accuracy: model 0.49 | PCA-logistic 0.49 | majority prior 0.33; cross-entropy model 0.661 vs prior 0.625

## Does the T-cell baseline recognise sorted bulk populations on other platforms? (one-vs-rest AUROC)

| cohort | population | n | model | marker genes |
|---|---|---|---|---|
| GSE4624 | CD4_conventional | 49 | 0.47 | 0.58 |
| GSE4624 | CD8 | 47 | 0.55 | n/a |
| PTCY | CD4_conventional | 33 | 0.82 | 0.91 |
| PTCY | NK | 13 | 1.00 | 1.00 |
| PTCY | CD8 | 35 | 0.25 | 0.96 |
| PTCY | Treg | 20 | 0.83 | 0.95 |
| GSE295359 | CD4_conventional | 25 | 0.97 | 1.00 |
| GSE295359 | CD8 | 25 | 0.04 | 1.00 |

## Bottom line (computed from the table below)

* tasks evaluated: 8; above chance by permutation test AND bootstrap CI -- all-platform-genes baseline: **5**, baseline-normalised genes (`genes_z`): **5**, pre-specified foundation-model features (`fmA_all`): **0**, immune-context features (`fmA_ctx`): **0**.
* foundation-model features beat the untrained-encoder control (`scratchA`) in 4 of 8 tasks, and `genes_z` in 0 of 8.
* a feature family that is not above chance here is NOT evidence that the model is useless at larger n -- only that it is not shown to help at this scale.

## Pre-stated gate for the context-pretraining step (does a GVHD-aligned encoder earn its place?)

* Gate A: an encoder family is above chance (permutation p<0.05 AND CI lower bound >0.5) on >= 3 of 8 tasks.
* Gate B: its mean leave-one-study-out AUROC exceeds `genes_z` by >= 0.03 AND exceeds its untrained-encoder control.

* mean leave-one-study-out AUROC over 4 held-out studies: `genes_z` 0.62, `fmA_all` 0.38, `fmA_ctx` 0.51, `scratchA_embed` 0.66, `scratchA_ctx` 0.47
* `fmA_all`: Gate A fail (0/8 tasks above chance); Gate B fail (LOSO 0.38 vs genes_z 0.62 and scratchA_embed 0.66)
* `fmA_ctx`: Gate A fail (0/8 tasks above chance); Gate B fail (LOSO 0.51 vs genes_z 0.62 and scratchA_ctx 0.47)
* **overall: NOT PASSED -- no encoder family clears both gates; the shipped model should be the baseline-normalised linear one.**
* two encoder families x 8 tasks were examined: a single task-level pass would not survive a multiplicity correction.

## Clinical tasks (patient-grouped repeated CV; AUROC [95% bootstrap CI])

Primary (pre-specified) = `fmA_all`: anchored foundation-model features. Comparator = `genes_native`: all platform genes. `genes_z` = vocabulary genes after the T-cell-baseline normalisation (the deployable default); `genes_vocab` = same genes, native scale; `scratchA` = identical encoder WITHOUT pretraining; `hybrid` = genes_z + fmA_all; `fm_raw` = foundation model without anchoring; `resid` = gene-by-gene deviation from the healthy-T-cell expectation (exploratory, added after iteration-2 results; `resid0` = untrained encoder).

| task | n (pts) | pos | fmA_all | fmA_ctx | genes_z | genes_native | hybrid | genes_vocab | scratchA | scratchA_ctx | fm_raw | resid | resid0 | perm p: fmA / ctx / genes_z / native | confounder-only | verdict genes_z | verdict fmA_all | verdict fmA_ctx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1_donor_CD8_to_recipient_aGVHD | 47 (47) | 18 | 0.54 [0.40,0.75] | 0.58 [0.39,0.74] | 0.75 [0.61,0.88] | 0.79 [0.70,0.93] | 0.75 | 0.61 | 0.60 | 0.62 | 0.50 | 0.76 | 0.63 | 0.328 / 0.279 / 0.016 / 0.016 | array_order 0.68 | above chance (not shown to beat the all-genes baseline) | NOT distinguishable from chance | NOT distinguishable from chance |
| K1_donor_CD4_to_recipient_aGVHD | 49 (49) | 19 | 0.50 [0.34,0.69] | 0.43 [0.25,0.57] | 0.59 [0.44,0.76] | 0.61 [0.45,0.76] | 0.58 | 0.53 | 0.54 | 0.55 | 0.55 | 0.60 | 0.53 | 0.443 / 0.623 / 0.197 / 0.197 | array_order 0.60 | NOT distinguishable from chance | NOT distinguishable from chance | NOT distinguishable from chance |
| K2_PTCy_CD8_GVHD | 35 (35) | 14 | 0.47 [0.27,0.68] | 0.45 [0.27,0.67] | 0.82 [0.71,0.96] | 0.76 [0.63,0.93] | 0.82 | 0.80 | 0.79 | 0.33 | 0.56 | 0.78 | 0.82 | 0.656 / 0.639 / 0.016 / 0.033 | batch 0.45 | above chance (not shown to beat the all-genes baseline) | NOT distinguishable from chance | NOT distinguishable from chance |
| K2_PTCy_CD4Tconv_GVHD | 33 (33) | 15 | 0.63 [0.44,0.82] | 0.56 [0.37,0.79] | 0.80 [0.65,0.99] | 0.90 [0.81,1.00] | 0.81 | 0.81 | 0.53 | 0.45 | 0.68 | 0.87 | 0.77 | 0.164 / 0.230 / 0.033 / 0.016 | batch 0.43 | above chance (not shown to beat the all-genes baseline) | NOT distinguishable from chance | NOT distinguishable from chance |
| K2_PTCy_Treg_GVHD | 20 (20) | 9 | 0.58 [0.28,0.78] | 0.55 [0.35,0.88] | 0.77 [0.52,0.97] | 0.88 [0.70,1.00] | 0.77 | 0.80 | 0.57 | 0.52 | 0.72 | 0.82 | 0.73 | 0.295 / 0.410 / 0.049 / 0.016 | batch 0.45 | above chance (not shown to beat the all-genes baseline) | NOT distinguishable from chance | NOT distinguishable from chance |
| K3_PBMC_T_aGVHD | 24 (24) | 11 | 0.67 [0.43,0.89] | 0.51 [0.25,0.77] | 0.81 [0.57,0.97] | 0.82 [0.65,0.99] | 0.79 | 0.81 | 0.57 | 0.67 | 0.63 | 0.77 | 0.81 | 0.213 / 0.426 / 0.033 / 0.033 | batch 0.26, site 0.51, day_post_hct 0.36 | above chance (not shown to beat the all-genes baseline) | NOT distinguishable from chance | NOT distinguishable from chance |
| K4_CD8_subsets_aGVHD | 60 (20) | 45 | 0.37 [0.17,0.57] | 0.37 [0.15,0.50] | 0.52 [0.23,0.79] | 0.57 [0.32,0.85] | 0.50 | 0.49 | 0.28 | 0.35 | 0.51 | 0.52 | 0.53 | 0.820 / 0.754 / 0.426 / 0.328 | - | NOT distinguishable from chance | NOT distinguishable from chance | NOT distinguishable from chance |
| K5_scRNA_pseudobulk_aGVHD | 75 (10) | 39 | 0.37 [0.18,0.61] | 0.45 [0.30,0.67] | 0.54 [0.38,0.69] | 0.51 [0.26,0.78] | 0.54 | 0.51 | 0.81 | 0.60 | 0.30 | 0.55 | 0.54 | 0.754 / 0.508 / 0.492 / 0.541 | timepoint 0.38 | NOT distinguishable from chance | NOT distinguishable from chance | NOT distinguishable from chance |
| K5x_graft_T_to_aGVHD | 9 (9) | 5 | - | - | - | - | - | - | - | - | - | - | - | - | - | skipped: too few samples/positives | | |

## Cross-study transfer (train on other studies, test on the held-out study)

| held-out study | n | fmA_all | fmA_ctx | resid | hybrid | genes_z | scratchA | scratchA_ctx |
|---|---|---|---|---|---|---|---|---|
| GSE103569 | 60 | 0.37 [0.12,0.66] | 0.59 [0.37,0.78] | 0.73 [0.55,0.89] | 0.70 [0.52,0.87] | 0.71 [0.52,0.87] | 0.59 [0.38,0.83] | 0.47 [0.20,0.76] |
| GSE295359 | 48 | 0.40 [0.14,0.72] | 0.52 [0.20,0.93] | 0.56 [0.24,0.85] | 0.54 [0.20,0.84] | 0.55 [0.21,0.84] | 0.73 [0.49,0.92] | 0.28 [0.08,0.47] |
| GSE73809 | 24 | 0.38 [0.16,0.59] | 0.51 [0.28,0.74] | 0.57 [0.33,0.81] | 0.55 [0.30,0.79] | 0.57 [0.32,0.80] | 0.68 [0.45,0.91] | 0.64 [0.38,0.86] |
| PTCY | 88 | 0.39 [0.28,0.50] | 0.42 [0.31,0.54] | 0.57 [0.45,0.69] | 0.63 [0.51,0.74] | 0.64 [0.52,0.75] | 0.62 [0.50,0.73] | 0.48 [0.36,0.60] |
