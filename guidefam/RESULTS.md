# guidefam — measured results (sandbox, CPU only)

Read `PROTOCOL.md` first for what is held out and how everything is scored. Numbers are copied from the scripts' output; the raw outputs are in `results/`.

**One-line status.** Cas13d on-target: a rank-average of three models beats our re-implementation of TIGER's network by +0.026 within-gene Spearman (16 of 16 genes; replicated on 4 locked genes). siRNA: parity with OligoFormer on its own inter-dataset protocol, no improvement. Eight other ideas were tested and did not help (section 4).

## 1. Cas13d on-target (TIGER HEK293T screen, 9,985 perfect-match guides, 16 genes)

Leave-one-gene-out; within-gene Spearman between measured and predicted activity, block-bootstrap 95 % CI.

| model | all 16 genes | 12 development genes | 4 LOCKED genes |
|---|---|---|---|
| DeepCas13 (published predictions) | 0.324 [0.299, 0.341] | 0.323 | 0.326 |
| random forest (published predictions, leave-gene-out) | 0.553 [0.530, 0.569] | 0.556 | 0.545 |
| "Konermann" 30-mer model (published predictions) | 0.599 [0.575, 0.613] | 0.602 | 0.590 |
| Tiger2D (our PyTorch re-implementation, section 6) | 0.612 [0.589, 0.630] | 0.617 | 0.598 [0.553, 0.635] |
| our LightGBM on guide + flank + TIGER scalar features | 0.600 [0.575, 0.617] | 0.607 | 0.579 |
| **GBM + Tiger2D** (rank-average, no third-party tool) | 0.623 [0.599, 0.640] | 0.630 | 0.604 |
| **GBM + Konermann** | 0.631 [0.607, 0.646] | 0.636 | 0.617 |
| **GBM + Tiger2D + Konermann ("ens3")** | **0.638 [0.613, 0.654]** | 0.643 | **0.622 [0.581, 0.658]** |

Paired differences (block bootstrap), vs Tiger2D: ens3 **+0.026 [+0.020, +0.032]** all genes, +0.026 development, **+0.025 [+0.013, +0.036] locked**; GBM + Tiger2D +0.011 [+0.006, +0.017] all, +0.013 development, +0.006 [-0.003, +0.014] locked (not significant there). ens3 beats Tiger2D in **16 of 16 genes** (4 of 4 locked), mean per-gene gain +0.026 (sd 0.014); it beats the best of its three members in 15 of 16.

Replicate noise ceiling: a perfect predictor would correlate ~0.89 with the observed mean, so a correlation of ~0.64 means the best model here captures roughly half of the explainable variance.

How ens3 was chosen: the 12 development genes only. The two-model GBM + Konermann combination was fixed first (0.636 there); ens3 was the best of five pre-specified combinations (0.643). The locked genes then confirmed it.

### External screens (models trained on all 16 HEK293T genes)

| model | flow-cytometry: CD46/CD55/TFRC (different assay), within-gene Spearman | HAP1 titration, 1,082 genes (different cell line), within-gene Spearman |
|---|---|---|
| DeepCas13 | 0.393 [0.355, 0.432] | — |
| Konermann * | 0.597 [0.563, 0.628] | — |
| Tiger2D | 0.636 [0.601, 0.669] | 0.163 [0.146, 0.180] |
| GBM | 0.626 [0.591, 0.661] | 0.168 [0.151, 0.182] |
| **GBM + Tiger2D** | **0.656 [0.619, 0.689]** | 0.195 [0.179, 0.212] |
| GBM + Konermann * | 0.652 | — |
| ens3 * | 0.665 [0.631, 0.698] | — |

\* developed on the CD46/CD55/TFRC flow screens, so **not an independent test** for the models that include it. The clean external result is GBM + Tiger2D: +0.020 [+0.012, +0.028] over Tiger2D, better in all three genes (CD46 0.727 -> 0.742, CD55 0.533 -> 0.559, TFRC 0.648 -> 0.665).

HAP1 transfer is poor for every model (0.16-0.20) although its labels are reliable (within-gene replicate Spearman 0.67, ceiling ~0.93). 51 % of its perfect-match guides are below -2 LFC against 13 % in HEK293T, consistent with those guides having been selected for high predicted activity (which would remove the predictable range); we cannot confirm the selection from the files. The +0.03 there is not a headline. (Pooled AUROC is meaningless for rank-averaged scores and is not reported for them.)

## 2. siRNA (OligoFormer tables; train Huesken, test Mixset / Takayuki)

Protocol of OligoFormer's `test.py`: binary label = normalised efficacy > 0.7; AUC / PRC / PCC. Design frozen on Huesken contiguous-chunk CV before these sets were read: rank-average of ridge (alpha 100) and LightGBM on siRNA sequence, OligoFormer's 24 thermodynamic descriptors, rule scores, and 8-nt target flanks.

Huesken -> Mixset (472 siRNAs, 58 % effective):

| model | AUC | PRC | PCC |
|---|---|---|---|
| Reynolds rules | 0.712 | 0.747 | 0.405 |
| Ui-Tei rules | 0.742 | 0.752 | 0.469 |
| **ours (frozen ridge + GBM ensemble)** | 0.813 [0.778, 0.853] | 0.855 | 0.574 [0.516, 0.648] |
| ridge without flank features (not the frozen choice) | 0.824 | 0.862 | 0.585 |
| OligoFormer (published, same protocol) | 0.816 | 0.814 | 0.588 |
| DSIR / OligoWalk / Monopoli-RF (published) | 0.748 / 0.708 / 0.744 | 0.764 / 0.700 / 0.739 | 0.503 / 0.559 / 0.463 |

**Parity, not improvement**: both OligoFormer values lie inside our intervals. The flank features that helped on Huesken's own CV (PCC 0.607 -> 0.630) did not transfer (ridge 0.585 without, 0.563 with); recorded, not tuned after the fact.

Huesken -> Takayuki (EGFP, 702 siRNAs, 27 % effective): Reynolds rules AUC 0.763 / PCC 0.509; our ensemble 0.746 / 0.515; GBM 0.762 / 0.547; ridge 0.713 / 0.449. Trained models do not beat a simple rule here. No OligoFormer number exists for this pair in its paper.

Other siRNA analyses (`sirna_leakage_lodo.py`): shuffled 5-fold on Huesken PCC 0.668 / AUC 0.846 vs contiguous 0.644 / 0.836 (the shuffled split flatters the same model by ~0.02 PCC); on their own shuffled recipe OligoFormer reports 0.711 / 0.862, i.e. it is ahead of our simple model on that protocol. Leave-one-dataset-out (adding Takayuki or Mix to training): Mix 0.581 vs 0.574, Takayuki 0.519 vs 0.515 PCC — no meaningful gain. TabDPT as the learner: best single learner on Huesken dev (PCC 0.638 vs 0.629 ridge), adding it to the ensemble +0.003 — within noise, not adopted.

## 3. Protocol audits (details in `PROTOCOL.md`)

* TIGER's recipe of mapping predictions back to LFC with the held-out gene's own median/scale inflates pooled Pearson from 0.632 to 0.717 on identical predictions; a model that is only each gene's true median scores 0.279.
* Early stopping on the evaluation fold: see section 5.
* OligoFormer's intra-dataset CV is a shuffled split (above).

## 4. Things that did NOT help (12 Cas13d development genes unless stated; base GBM 0.607)

| change | within-gene Spearman | vs base [CI] |
|---|---|---|
| rank-Gauss target | 0.609 | +0.002 [-0.003, +0.008] |
| add HAP1 screen to training | 0.596 | **-0.011** [-0.016, -0.005] |
| add flow screen | 0.606 | -0.001 [-0.006, +0.004] |
| add both | 0.602 | -0.005 [-0.010, +0.001] |
| flanks 6 / 12 / 20 nt | 0.601 / 0.600 / 0.601 | **-0.006 / -0.007 / -0.006** |
| own RNAplfold-style accessibility profiles (3 scales, per-nt) | 0.603-0.606 | -0.001 to -0.003 (CIs include 0) |
| RNA-FM embedding of the 23-mer (PCA 8 / 32) | 0.602 / 0.601 | -0.005 / -0.006 |
| smoothing predictions over neighbouring guides | 0.606 -> 0.576 as strength rises | worse at every setting |
| LambdaRank objective | 0.527 / 0.532 | much worse |
| 5-seed GBM bagging | 0.606 | -0.001 |

Local accessibility and RNA-FM embeddings also did nothing for siRNA on Huesken. The residuals still carry short-range structure (neighbouring guides' residuals correlate 0.42 at 1 nt), so something local is unexplained, but none of these features captured it.

## 5. Early stopping on the evaluation fold  (`audit_peek.py`)

RESULT_PEEK_PLACEHOLDER

## 6. Caveats that limit every claim above

* **Tiger2D is our re-implementation** from the architecture, loss, optimiser and early-stopping settings in the authors' repository (PyTorch, Keras-style initialisation, early stopping on an inner split of the training genes). It is not their code, weights or published numbers, and we did not verify it against the paper's figures. "Beats TIGER" in this document means "beats this re-implementation under this protocol".
* The ensembles that include the Konermann scores rely on a third-party tool whose training data we cannot check; they need that tool's output for each transcript (a web portal). The tool-free combination gains less (+0.011 all genes, +0.006 on 4 locked genes — not significant on its own — +0.020 on the clean flow screen).
* Four locked genes is a small test set; block-bootstrap CIs resample regions within genes, not genes. The per-gene sign test (16/16) and the flow screen are the stronger evidence.
* Single seed throughout; no hyper-parameter tuning (defaults) except where stated.
* siRNA: Mixset's binary column does not follow a clean threshold; we used the `test.py` convention. We did not re-run OligoFormer, so its numbers are the paper's.
* Held-out-gene scores of 0.6 are roughly two thirds of what replicate noise would allow; this is a modest-signal prediction problem, not a solved one.
