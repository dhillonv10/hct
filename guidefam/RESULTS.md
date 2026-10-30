# guidefam — measured results (sandbox, CPU only)

Read `PROTOCOL.md` first for what is held out and how everything is scored. Numbers are copied from the scripts' output; the raw outputs are in `results/`.

**One-line status.** Cas13d on-target: a rank-average of three models beats our re-implementation of TIGER's network by +0.026 within-gene Spearman (16 of 16 genes; replicated on 4 locked genes), and on the two screens TIGER never saw, models trained only on HEK293T beat the authors' *released* TIGER weights (flow-cytometry: GBM + Tiger2D 0.656 vs 0.599, paired +0.057 [+0.041, +0.073]). siRNA: parity with the two best published results (OligoFormer, DeepSilencer) on the Huesken -> Mixset protocol, no improvement. Mismatch tolerance (a second Cas13d task, section 9): the released TIGER is the *better* single model (ours -0.011 within-gene on the external HAP1 screen); averaging the two gains +0.020. Eight other ideas were tested and did not help (section 4).

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

### What a guide designer gets: top-k picks per gene (leave-one-gene-out, 16 genes)

Ordering each gene's guides and taking the top 10 (TIGER's own default list length), fraction of picked guides that are active (threshold from non-targeting guides) and mean activity in within-gene z units; random picks are 0.42 active and 0.20 z.

| model | active in top 10 | mean z of top 10 |
|---|---|---|
| Konermann tool | 0.938 | 2.20 |
| GBM | 0.906 | 2.15 |
| Tiger2D (ours) | 0.963 | 2.40 |
| GBM + Tiger2D | 0.950 | 2.40 |
| ens3 | **0.994** | **2.54** |
| oracle (true top 10) | 1.000 | 3.50 |

ens3 minus Tiger2D in mean z of the top 10: +0.141 [+0.039, +0.263] (bootstrap over genes). Every model is close to saturation at the top (the best 10 of ~600 are nearly all active), so read this as supporting the ensemble result, not as a separate claim.

### External screens (models trained on all 16 HEK293T genes)

| model | flow-cytometry: CD46/CD55/TFRC (different assay), within-gene Spearman | HAP1 titration, 1,082 genes (different cell line), within-gene Spearman |
|---|---|---|
| DeepCas13 | 0.393 [0.355, 0.432] | — |
| Konermann * | 0.597 [0.563, 0.628] | — |
| **TIGER as released** (authors' weights, sequence only; section 1b) | 0.599 | 0.111 |
| Tiger2D (ours) | 0.636 [0.601, 0.669] | 0.163 [0.146, 0.180] |
| GBM | 0.626 [0.591, 0.661] | 0.168 [0.151, 0.182] |
| **GBM + Tiger2D** | **0.656 [0.619, 0.689]** | 0.195 [0.179, 0.212] |
| GBM + Konermann * | 0.652 | — |
| ens3 * | 0.665 [0.631, 0.698] | — |

\* developed on the CD46/CD55/TFRC flow screens, so **not an independent test** for the models that include it. The clean external result is GBM + Tiger2D: +0.020 [+0.012, +0.028] over Tiger2D, better in all three genes (CD46 0.727 -> 0.742, CD55 0.533 -> 0.559, TFRC 0.648 -> 0.665).

HAP1 transfer is poor for every model (0.16-0.20) although its labels are reliable (within-gene replicate Spearman 0.67, ceiling ~0.93). 51 % of its perfect-match guides are below -2 LFC against 13 % in HEK293T, consistent with those guides having been selected for high predicted activity (which would remove the predictable range); we cannot confirm the selection from the files. The +0.03 there is not a headline. (Pooled AUROC is meaningless for rank-averaged scores and is not reported for them.)

### 1b. Against the authors' released TIGER weights (`official_tiger.py`, `export_official_tiger.py`)

The Hugging Face release (`Knowles-Lab/tiger`) is a sequence-only network (26-nt target window + guide, no scalar features) trained by the authors on HEK293T perfect-match **and** mismatch/indel guides. Its weights were read from the SavedModel's checkpoint index (nothing executed; the graph's ops are standard numeric ones plus Keras's save/restore plumbing, which inference never calls) and run as a PyTorch forward pass. Known-answer check: on the HEK293T guides it trained on it scores 0.723 within-gene (a mis-ported layout would give ~0); that figure is in-sample and is not a result. On the two screens it never trained on (we use the network's raw LFC estimate; the tool's later calibration and sigmoid scoring were not loaded):

| external screen | released TIGER | Tiger2D (ours) | GBM (ours) | GBM + Tiger2D | paired gain of GBM + Tiger2D |
|---|---|---|---|---|---|
| flow-cytometry (3 genes, 5,635 guides) | 0.599 | 0.636 | 0.626 | **0.656** | **+0.057 [+0.041, +0.073]** |
| HAP1 titration (1,082 genes, 8,653 guides) | 0.111 | 0.163 | 0.168 | **0.195** | **+0.084 [+0.066, +0.101]** |

Even our individual models beat it (flow: Tiger2D +0.037 [+0.021, +0.053], GBM +0.028 [+0.009, +0.047]; HAP1: +0.052 and +0.057). Combining the released model with ours adds nothing over GBM + Tiger2D (flow 0.655, HAP1 0.193). Read this carefully: it is the *released, deployed* tool against models that also use TIGER's scalar descriptors (location, junction distance, accessibility), which the paper's full model uses but the released tool does not; so it supports "a user choosing between tools would do better with ours on these screens", not "our architecture beats theirs". The HAP1 numbers are low for every model for the reason given above. Licence: NYGC academic / non-commercial; the weights are not redistributed.

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

**Parity, not improvement**: both OligoFormer values lie inside our intervals. The same protocol is reported by [DeepSilencer (arXiv 2503.04200, Table 3)](https://arxiv.org/pdf/2503.04200): OligoFormer 0.815 AUC / F1 0.769, DeepSilencer (siRNA + thermodynamic features) **0.820 AUC / F1 0.775** — so the best published figure on this protocol is 0.820 against our 0.813 [0.778, 0.853]. DeepSilencer's repository contains only a README and data (no code, no licence), so it could not be run or combined with. With 472 test siRNAs the interval is about +-0.04 AUC: differences of 0.005-0.01 between OligoFormer, DeepSilencer and us cannot be resolved on this set. Showing a real siRNA improvement needs a larger evaluation (more held-out datasets), not a different statistic. The flank features that helped on Huesken's own CV (PCC 0.607 -> 0.630) did not transfer (ridge 0.585 without, 0.563 with); recorded, not tuned after the fact.

Huesken -> Takayuki (EGFP, 702 siRNAs, 27 % effective): Reynolds rules AUC 0.763 / PCC 0.509; our ensemble 0.746 / 0.515; GBM 0.762 / 0.547; ridge 0.713 / 0.449. Trained models do not beat a simple rule here. No OligoFormer number exists for this pair in its paper.

Other siRNA analyses (`sirna_leakage_lodo.py`): shuffled 5-fold on Huesken PCC 0.668 / AUC 0.846 vs contiguous 0.644 / 0.836 (the shuffled split flatters the same model by ~0.02 PCC); on their own shuffled recipe OligoFormer reports 0.711 / 0.862, i.e. it is ahead of our simple model on that protocol. Leave-one-dataset-out (adding Takayuki or Mix to training): Mix 0.581 vs 0.574, Takayuki 0.519 vs 0.515 PCC — no meaningful gain. TabDPT as the learner: best single learner on Huesken dev (PCC 0.638 vs 0.629 ridge), adding it to the ensemble +0.003 — within noise, not adopted.

## 3. Protocol audits (details in `PROTOCOL.md`)

* TIGER's recipe of mapping predictions back to LFC with the held-out gene's own median/scale inflates pooled Pearson from 0.632 to 0.717 on identical predictions; a model that is only each gene's true median scores 0.279.
* Early stopping on the evaluation fold: measured, no detectable effect (section 5).
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
| replicate-reliability sample weights (two strengths) | 0.600 / 0.596 | **-0.007 / -0.011** |
| Huber objective | 0.604 | -0.003 [-0.005, +0.000] |
| nearest-neighbour duplex stability profile (Turner stacking) | 0.605 | -0.002 [-0.006, +0.003] |
| Pol III terminator / homopolymer / 3-mer motif features | 0.612 | +0.005 [-0.000, +0.010] (borderline; not adopted) |
| using the mismatch/indel variants to denoise labels | not possible | variants exist for only ~100 sites in each of 6 genes (~600 of 9,985 sites, ~181 variants each) |

Local accessibility and RNA-FM embeddings also did nothing for siRNA on Huesken. The residuals still carry short-range structure (neighbouring guides' residuals correlate 0.42 at 1 nt), so something local is unexplained, but none of these features captured it.

## 5. Early stopping on the evaluation fold  (`audit_peek.py`)

The authors' cross-validation script passes the held-out fold as the validation set and restores its best-`val_loss` epoch. In principle that selects one hyper-parameter (the stopping epoch) on test data. We measured it on the re-implementation: 3 held-out genes (CCT7, SF3A2, AQR) x 2 seeds, same data and seed with (a) early stopping on an inner split of the training genes and (b) on the held-out gene itself.

| gene | clean | stopping on the held-out gene |
|---|---|---|
| CCT7 | 0.694 | 0.695 |
| SF3A2 | 0.389 | 0.395 |
| AQR | 0.637 | 0.635 |
| mean | 0.573 | 0.575 (**+0.002**) |

Individual pairs ranged from -0.012 to +0.013, i.e. seed-to-seed noise is as large as any effect. **No detectable inflation of within-gene Spearman**; this is a methodological leak that does not matter much here (6 pairs, so small effects cannot be excluded). It is reported so that it is not repeated as a criticism of TIGER without evidence. It also shows that single-seed Tiger2D scores carry about +-0.01 of seed noise, which is why seed-bagging is in the M4 stage.

## 6. Caveats that limit every claim above

* **Tiger2D is our re-implementation** from the architecture, loss, optimiser and early-stopping settings in the authors' repository (PyTorch, Keras-style initialisation, early stopping on an inner split of the training genes). It is not their code or weights, and we did not verify it against the paper's figures. Where this document says "beats TIGER" it means either the re-implementation under leave-one-gene-out (section 1) or the authors' released weights on two external screens (section 1b); the latter is a sequence-only deployed tool, the former uses scalar descriptors.
* The ensembles that include the Konermann scores rely on a third-party tool whose training data we cannot check; they need that tool's output for each transcript (a web portal). The tool-free combination gains less (+0.011 all genes, +0.006 on 4 locked genes — not significant on its own — +0.020 on the clean flow screen).
* Four locked genes is a small test set; block-bootstrap CIs resample regions within genes, not genes. The per-gene sign test (16/16) and the flow screen are the stronger evidence.
* Single seed throughout; no hyper-parameter tuning (defaults) except where stated.
* siRNA: Mixset's binary column does not follow a clean threshold; we used the `test.py` convention. We did not re-run OligoFormer, so its numbers are the paper's.
* Held-out-gene scores of 0.6 are roughly two thirds of what replicate noise would allow; this is a modest-signal prediction problem, not a solved one.

## 7. Licences you need to respect

* **TIGER** (code, data, predictions, released weights): NYGC *academic and non-commercial research only*; for-profit entities and any commercial use must contact NYGC first. We neither redistribute nor modify it; `run_all.sh` fetches it on your machine.
* **OligoFormer**: *proprietary* ("All Rights Reserved", Tsinghua): use "subject to payment", no modification, no redistribution, no use to develop competing products. We therefore did **not** run, port or retrain its network, and this package contains none of its code. We do read its published data tables (Huesken / Mixset / Takayuki and its thermodynamic descriptors) from its repository; whether that is permitted for your use is for you (and its licensor, ott@tsinghua.edu.cn) to settle. The original Huesken and other source datasets are published and could replace those files. The combination script `sirna_plus_oligoformer.py` only reads a predictions file you produce yourself.
* Reading its public training script showed that, for the `--datasets Hu Mix` path, the "validation" set is the *test* dataset and the checkpoint is saved when validation loss and AUC improve on it. Whether the paper's Table 2 was produced that way we cannot tell, and we did not measure how large the effect would be; it is a reason to prefer comparisons that do not depend on that number.

## 8. Three "novel angle" tests: small model, biology-aware transformer, transfer from a non-biology model

Cas13d, leave-one-gene-out, 12 development genes, within-gene Spearman (`figures/params_vs_accuracy.png`, numbers in `figures/params_vs_accuracy.csv`, parameter counts from `results/efficiency_table.csv`, taken from the objects and checkpoint headers):

| model | parameters | accuracy | note |
|---|---|---|---|
| ridge on guide + flank sequence | 130 | 0.522 | |
| gradient-boosted trees (ours) | 17,200 | 0.607 | fits in ~5 s on 4 cores |
| MiniPairFormer **with** RNA-structure attention bias | 25,165 | 0.585 | 2-layer, d = 32 transformer; attention biased by ViennaRNA base-pair probabilities, unpaired probability as token input |
| MiniPairFormer **without** the structure bias | 25,165 | 0.587 | the ablation: the bias does not help |
| 2-D CNN (TIGER-style, our re-implementation) | 236,289 | 0.617 | the released TIGER has 234,753 |
| TabDPT (pretrained tabular foundation model, not trained on biology) | 63,040,624 | 0.590 | open weights; used as the learner on the same tabular features |
| RNA-FM embedding of the 23-mer + ridge | 99,523,469 | 0.487 | frozen encoder |
| rank-average of GBM + CNN + published 30-mer tool | — | **0.643** | the only thing that moves it |

* **Smaller model, same performance** — supported: 17,200 parameters get 98 % of the CNN's accuracy (0.607 / 0.617; on all 16 genes 0.600 vs 0.612, a paired difference of -0.012 [-0.022, -0.002]), 14x fewer parameters, and a 100 M-parameter RNA language-model embedding is *worse* (0.487). It is not "better than TIGER with fewer parameters": the released TIGER itself is small.
* **A transformer designed for biology** — tested, no gain: a 25k-parameter pair-aware transformer reaches 0.585-0.587; the RNA-structure attention bias adds nothing (-0.0015, noise); adding it to the three-model ensemble gives 0.640 vs 0.643. This agrees with the accessibility-profile result (section 4): local structure is not the missing information.
* **Transfer from a non-biology model** — TabDPT on Cas13d: 0.590, no ensemble gain (0.640 vs 0.643). On siRNA (**post-hoc**: Mix and Takayuki had already been read), the same learner on the same features without flanks reaches Huesken -> Mixset AUC 0.825 [0.788, 0.858], PCC 0.599 [0.534, 0.663], nominally above OligoFormer (0.816 / 0.588) and DeepSilencer (0.820 AUC) but far inside the +-0.035 noise of 472 test siRNAs; with flank features 0.809 / 0.578; on Takayuki 0.760 / 0.537. This is consistent with the GSK report that a tabular foundation model matches specialised siRNA predictors ([arXiv 2510.02476](https://arxiv.org/html/2510.02476v2)); it is not evidence of a win.

What this supports: across 146 to 63 million parameters, three architecture families and a non-biology foundation model, held-out-gene accuracy saturates near 0.6 at ~10^4 parameters. It does **not** by itself show the cause (missing information versus label noise; the replicate ceiling is ~0.89), and it does not exclude an architecture we did not try. A claim of "a new architecture beats the state of the art" is not supported by any of these runs.

## 9. Round 3: is the plateau noise, which features are missing, and a second task (mismatch tolerance)

### 9.1 The plateau is not label noise (for the sites where this can be tested)  (`noise_vs_information.py`; output in `results/diagnostics/`)

About 600 perfect-match sites (~100 in each of AQR, EIF3B, NUP133, SF3B1, SNRNP200, WDR75) were also measured with ~181 mismatch/indel variants of the *same* site, whose measurement noise is independent of the perfect-match replicates. The first principal component of the 23 per-position single-mismatch LFCs (gene-standardised) explains 51-59 % of that panel's variance and correlates **0.82-0.91 (mean 0.85)** with the perfect-match label, so it is a usable independent estimate of site activity. On those 599 sites (within-gene Spearman): prediction vs perfect-match label GBM 0.630, Tiger2D 0.712, published 30-mer tool 0.661, ensemble 0.741; prediction vs the independent estimate 0.624 / 0.687 / 0.710 / 0.751. The decisive test: after removing what the model predicts, the perfect-match label and the independent estimate are still strongly correlated, **partial r = +0.75 [0.70, 0.79] controlling for the GBM and +0.66 [0.59, 0.70] controlling for the ensemble** (site bootstrap). What the models miss is therefore reproducible site-level biology, not measurement noise; the "replicate-noise ceiling" of ~0.89 is a loose bound, not a sign that we are done.

**Caveat that limits this.** The titration sites are not a random sample: they are enriched for active guides (65.6 % active vs 34 % for the other guides in those genes; mean activity percentile 0.68 vs 0.48). The result is established for these selected sites and may not hold across all guides. It also cannot tell "information the inputs do not contain" from "a better model of the same inputs could capture it", because the variants share the guide and target context.

### 9.2 Screening features against the denoised residual, and a correction that did not work  (`feature_discovery.py`, `residual_correction.py`)

Against the variant-denoised residual (after the ensemble), 307 candidate features were screened (Stouffer across genes, Benjamini-Hochberg). Six pass q < 0.05, **all target-accessibility measures** (minimum unpaired probability of the target in a 300-nt window, q = 0.001; unpaired probability at target positions 5-6, q = 0.015; and three related window summaries), with positive sign (more accessible than the ensemble predicts -> more active than predicted). Using only the four development titration genes the screen again selects accessibility features (plus two guide-position indicators). A linear correction fitted on those sites and applied to everything else **did not transfer** and made held-out predictions worse (locked titration genes -0.001 to -0.046; other development genes -0.017 to -0.052; external flow screen -0.015 to -0.062, depending on the fitting variant; even the fit genes got worse). The most likely reason is the selection effect above: relationships learned on active-enriched sites do not hold for the full guide set. So accessibility is a *candidate* for the missing signal, supported by a significant association on selected sites, but we have no working way to exploit it; this is a lead, not a result.

### 9.3 Mismatch tolerance (single-mismatch variants)  (`mismatch.py`, `mismatch_task.py`; log in `results/diagnostics/mismatch_task_log.txt`)

Predict the LFC of a single-mismatch variant of a guide from the variant guide's sequence, the mismatch position and base change, and the parent site's predicted activity (gradient-boosted trees). Trained on the single-mismatch rows of the HEK293T titration genes; evaluated by within-gene Spearman over a gene's variants and within-site Spearman among the variants of one site (the "which mismatch hurts most here" signal). Design fixed on leave-one-gene-out among the four development titration genes.

| model | dev genes (LOGO) within-gene / within-site | locked SF3B1 + WDR75 | **external HAP1 titration** (43,263 variants, 1,082 genes) within-gene / within-site |
|---|---|---|---|
| position-only lookup | 0.334 / 0.403 | 0.258 / 0.330 | 0.461 / 0.517 |
| position + parent's predicted activity | 0.515 / 0.403 | 0.473 / 0.330 | 0.409 / 0.517 |
| GBM, variant features + parent's predicted activity (ours) | 0.542 / 0.462 | 0.509 / 0.381 | 0.529 / 0.594 |
| GBM without the parent's predicted activity | 0.507 / 0.454 | 0.470 / 0.390 | 0.531 / 0.589 |
| TIGER as released (118k perfect-match + mismatch training rows) | — | — | **0.540 / 0.608** |
| GBM + released TIGER (rank-average) | — | — | 0.560 / 0.615 |

Paired on HAP1 (block bootstrap over sites): ours minus released TIGER **-0.011 [-0.019, -0.004]** within-gene and **-0.014 [-0.017, -0.012]** within-site, i.e. **the released TIGER is the better single model on this task**, plausibly because it saw 118k rows of every mismatch/indel type while ours saw 13.7k single-mismatch rows. Averaging the two gives **+0.020 [+0.013, +0.021]** within-gene and +0.007 [+0.005, +0.009] within-site over the released TIGER alone: a small complementary-model gain of the same kind as on-target, not a win over it.
