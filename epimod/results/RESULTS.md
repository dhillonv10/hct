# EpiMod — results

All numbers are from this repository's code on **studies never used for training, panel selection,
normalisation, or model selection** (the 65 ComputAge Bench benchmark studies; 10,404 samples after
removing 6 duplicate registrations). Scoring is my re-implementation of the ComputAge protocol
(ages 18–90, one-sided tests, Benjamini–Hochberg across tests per model, the curated list of
41 AA2 + 24 AA1 tests); it was written from the `computage` package source, not run through it, and
it is not the benchmark's official leaderboard code. Published clocks are run with the coefficients
shipped in the `computage` package and SeSAMe-median imputation.

Single training/validation split; two initialisation seeds for the main model. Raw tables:
[`tables_full.md`](tables_full.md) and the CSV/JSON files next to this file.

## Short version

1. **Age accuracy: competitive, not better than a linear baseline.** Median absolute error on
   benchmark healthy controls is **3.93 y** (3.77 y with cell-type-balanced sampling), against
   **3.68 y for an ElasticNet** on the same CpGs and studies, and 4.15 / 5.34 / 7.5–8.0 y for
   Horvath V2 / Horvath V1 / PhenoAge. On whole-blood samples: 2.85 y (EpiMod) vs 2.76 y (ElasticNet)
   vs 3.3–5.5 y (published clocks).
2. **Detecting accelerated aging: no better than the linear baselines.** Mean case-vs-control
   age acceleration across the 41 AA2 tests is +0.33 y (EpiMod), +0.20 (ridge), −0.55 (ElasticNet);
   PhenoAge V2 is +2.34 y and is far ahead (19 of 41 AA2 hits vs 5). Among the models trained
   here, a better fit to chronological age goes with *less* disease signal.
3. **Where it fails:** purified cell types, saliva and the 27K array — all rare or absent in training.
4. **What the design buys, with evidence:** (a) age can be predicted from CpGs the model never saw
   (§4); (b) the age prediction decomposes exactly over modules, which gives age-adjusted
   module-level attributions (§6); (c) the learned linear warm start, module transformer,
   probabilistic loss and augmentation each improve age accuracy (§3).
5. **What did not pan out:** the genomic-context embedding does not improve accuracy on seen CpGs;
   masked-CpG pre-training has no measurable effect; the acceleration head did not help;
   interval calibration does not carry over to new studies (76% coverage at nominal 90%).

**Versus Insilico's Longevity-LLM.** Their abstract reports a 4.34-year MAE for epigenetic age.
I could not read the paper's methods, so I do not know the evaluation set, tissue, or whether it
is a mean or median error; it is **not comparable** to the numbers here and I draw no conclusion
from it. See [`../docs/DESIGN.md`](../docs/DESIGN.md).

## 1. Head-to-head on the held-out benchmark

| model | MedAE | MedAE 18–90 | MedE (bias) | AA2 hits /41 | AA1 hits /24 | score | mean AA2 effect (y) |
|---|---|---|---|---|---|---|---|
| PhenoAge V2 (published) | 7.53 | 7.50 | −2.55 | **19** | 9 | **28.0** | +2.34 |
| PhenoAge V1 (published) | 8.00 | 7.99 | −4.19 | 9 | 7 | 16.0 | +1.46 |
| Horvath V1 (published) | 5.34 | 5.33 | −0.04 | 3 | 12 | 15.0 | −0.01 |
| Horvath V2 (published) | 4.15 | 4.22 | +1.15 | 5 | 12 | 13.7 | −0.17 |
| Hannum (published) | 8.40 | 8.38 | +7.63 | 1 | 20 | 2.8 | +0.08 |
| Lin (published) | 7.46 | 7.51 | +2.19 | 5 | 9 | 11.4 | +0.15 |
| Ridge, same CpGs/studies | 4.12 | 4.15 | −0.81 | 5 | 9 | 14.0 | +0.20 |
| ElasticNet, same CpGs/studies | **3.68** | **3.72** | +1.51 | 3 | 14 | 11.3 | −0.55 |
| **EpiMod (seed 0)** | 3.93 | 3.96 | +2.03 | 5 | 15 | 12.3 | +0.33 |
| EpiMod (seed 1) | 3.93 | 3.98 | +2.21 | 5 | 16 | 12.0 | +0.27 |
| EpiMod + cell-type-balanced sampling (post hoc) | 3.77 | 3.80 | +1.55 | 5 | 13 | 12.7 | +0.28 |

How to read this:

* **AA1 hits are inflated by bias.** AA1 only asks whether predicted age exceeds true age in cases.
  A clock that reads everyone older passes (Hannum: bias +7.6 y, 20 AA1 hits). The protocol's score
  discounts AA1 hits by `1 − MedE/MedAE`; compare scores, not raw AA1 hits.
* **Hit counts are fragile; effect sizes are steadier.** An earlier 6-epoch checkpoint got 14 AA2
  hits and the fully trained model gets 5, yet their per-test effect sizes correlate at Spearman
  0.98 — many tests sit near the BH threshold, and the mean effect shrank by about a third with
  training (+0.52 → +0.33 y). Seed 0 and seed 1 agree closely (MedAE 3.928 vs 3.933; 5 vs 5 AA2
  hits), so the 14→5 change is not explained by seed noise. Treat differences of a few hits as noise.
* Excluding the two benchmark studies that are sibling series of training studies (GSE72774,
  GSE72776; same project as GSE72773/GSE72777, no shared samples) changes nothing material
  (`benchmark/summary_no_siblings.csv`; EpiMod MedAE 4.14, 5/39 AA2 hits).

## 2. Where the error comes from

Healthy-control MedAE (years), benchmark:

| subgroup | n | EpiMod | + cell-type balance | ElasticNet | Ridge | Horvath V2 | Horvath V1 | PhenoAge V2 |
|---|---|---|---|---|---|---|---|---|
| whole-blood-like (whole blood, PBMC, PBL, buffy coat) | 1,747 | 2.85 | 2.91 | **2.76** | 3.85 | 3.28 | 5.35 | 5.51 |
| purified / other cell types | 1,083 | 7.06 | 5.79 | 6.73 | **4.81** | 6.49 | 5.32 | 14.17 |
| – CD14+ monocytes | 440 | 8.68 | 6.21 | 8.70 | **4.09** | 9.38 | 4.81 | 13.64 |
| – CD4+ T cells | 173 | 5.06 | 5.15 | 6.16 | **3.82** | 7.89 | 4.31 | 10.63 |
| tissue = saliva | 131 | 16.3 | – | 12.4 | 8.7 | **2.6** | 10.0 | 32.3 |
| platform = 27K | 103 | 12.1 | – | 11.5 | 12.3 | 10.9 | **3.9** | 9.9 |

* Error is dominated by **study- and cell-type-level offsets**, not by noise within a study: the
  two largest healthy-control sets (GSE56046 monocytes, GSE56581 CD4+ T cells) are off by +9 to +11 y
  for both EpiMod and ElasticNet.
* The training data are blood only (mostly whole blood/PBMC); purified cells have ~370 training
  samples. The flexible models (EpiMod, ElasticNet) extrapolate worse than the heavily regularised
  ridge. Cell-type-balanced sampling helps partway (overall 3.93→3.77, monocytes 8.7→6.2) but does
  not close the gap to ridge.
* **The cell-type finding was made by looking at benchmark errors**, so this fix is post hoc and
  exploratory; it was run once.
* Do not use this model for saliva, buccal or 27K data.

## 3. Ablations (matched 15-epoch budget; `full_seed1` is the noise floor)

| variant | MedAE | Δ vs full | mean AA2 effect (y) | 90%-interval coverage (HC) |
|---|---|---|---|---|
| **full** | 3.93 | – | +0.33 | 0.756 |
| full, other init seed | 3.93 | +0.005 | +0.27 | 0.766 |
| no linear ridge-warm-started route | 4.42 | **+0.49** | +0.68 | 0.707 |
| no learned CpG-id embedding (context-only tokens) | 4.27 | **+0.34** | +0.38 | 0.746 |
| no transformer over modules (purely additive) | 4.10 | **+0.17** | +0.33 | 0.746 |
| point estimate instead of distribution | 4.06 | **+0.13** | +0.43 | n/a (no σ) |
| no augmentation | 4.06 | **+0.13** | −0.10 | 0.794 |
| no masked-CpG reconstruction loss | 3.98 | +0.05 | +0.41 | 0.779 |
| **no genomic-context features** | **3.84** | **−0.09** | +0.27 | 0.752 |
| *post hoc:* cell-type-balanced sampling | 3.77 | −0.16 | +0.28 | 0.811 |

* Components that help age accuracy beyond the noise floor: the warm-started linear route (the
  cold-start model needs far longer to converge), learned CpG ids, the module transformer, the
  probabilistic loss, augmentation.
* **Genomic context does not help on CpGs the model has seen** (it is slightly worse, −0.09 y).
  Its value shows up elsewhere (§4).
* **Masked-CpG reconstruction does nothing measurable** for age here (Δ +0.05, near noise).
* The variant with the *worst* chronological accuracy (no linear route) has the *largest* disease
  signal (+0.68 y) and the best protocol score of the learned models (14.8, 10 AA2 hits), and
  `no_aug` has the lowest (−0.10 y). This is the same trade-off as in §1, not a feature.
* These are single runs on one split; Δ below ~0.05–0.1 y should not be interpreted.

## 4. Prediction from CpGs the model never saw

A context-only model (no id embeddings, no linear route) is trained on a random half **A** of the
12 k panel CpGs, then its weights are re-targeted (`EpiMod.rebind`) to the other half **B**. No
weight of the model has ever been tied to a half-B CpG. Scored on benchmark healthy controls:

| predictor | MedAE (all HC) | MedAE (whole blood) | r |
|---|---|---|---|
| constant (training median age) | 15.00 | 14.20 | – |
| seen CpGs (half A, n = 5,958) | 4.85 | 3.81 | 0.88 |
| **unseen CpGs (half B, n = 6,042), true genomic context** | **8.50** | **9.07** | **0.75** |
| unseen, chromosome + position shuffled among CpGs | 11.21 | 10.79 | 0.53 |
| unseen, island/gene/regulatory annotations shuffled | 11.32 | 11.98 | 0.32 |
| unseen, all context shuffled | 11.24 | 11.85 | 0.31 |

From genomic context alone, with no CpG-specific parameters, the model recovers much of the age
signal (r = 0.75), and shuffling either the annotation categories or the coordinates removes most
of it. It is nowhere near the accuracy of a model trained on those CpGs (3.9 y), so this is a
capability for new probe sets, not a replacement for training. Caveat: halves A and B are
interleaved at random, so most half-B CpGs have a half-A neighbour; I have not tested transfer to
a different chromosome or a different array.

## 5. Uncertainty

The model predicts a mean and sd of transformed age; the sd is rescaled once on held-out *training*
studies so 90% intervals are exact there. On the benchmark healthy controls the 90% intervals cover:

| subset | EpiMod | + cell-type balance |
|---|---|---|
| all HC | 0.756 | 0.811 |
| blood, 450K/EPIC (the training distribution) | 0.802 | 0.861 |
| EPIC | 0.844 | – |
| 27K | 0.311 | – |
| saliva | 0.191 | – |

Before calibration the raw σ was about 30% too wide on the held-out training studies (scale
0.755). After it, intervals are close to nominal there but **overconfident on new studies**
(80–86% on in-distribution blood, 76–81% overall), and badly so out of distribution. Do not read
them as guarantees; per-study recalibration would be needed.

## 6. What the modules are, and module-level attribution

* 24 modules, each a soft assignment over the 12 k CpGs. They are **structured but diffuse**:
  effective size 1.6 k–8.6 k CpGs (median 4.8 k), the top 1% of CpGs carry a median 9% of a
  module's mass (max 20%). Enrichments are for CpG-island context (shores/shelves) and cell-type-
  specific regulatory regions. The most concentrated modules are led by developmental
  transcription-factor loci (HOXD, GATA4, PAX6, SOX1, ZIC1), the classic age-methylated
  polycomb-target class. `interpretation/modules.csv`.
* Age-adjusted, study-adjusted module shifts in cases vs controls (OLS of module contribution on
  study + age + case; `interpretation/`): HIV +3.5 y (spread across 16 of 24 modules at p<0.001),
  Werner syndrome +4.5 y, TB +2.4, COPD +1.4, others near 0, RA −0.65. These match the sign and rough
  size of the protocol's own AA2 effects; the decomposition shows the acceleration is **distributed
  across modules rather than carried by one**. P-values are descriptive (samples within a study are
  not independent).

## 7. Acceleration head (negative result)

A small head on frozen module tokens trained to place cases above age-matched same-study controls,
scored only on held-out studies or disease classes (`train/accel.py`):

| | MedAE | AA2 hits | mean AA2 effect (y) |
|---|---|---|---|
| EpiMod, age only | 3.93 | 5 | +0.33 |
| + head, study-held-out folds | 4.08 | 7 | +0.35 |
| + head, disease-class-held-out folds | 4.15 | 7 | +0.32 |

No change in mean acceleration; slightly worse chronological error. The +2 hits are within the
threshold-fragility noted above. I tried two versions (unweighted, then pair-weighted per test) and
stopped there to avoid tuning against the same test studies. Not recommended as is.

## 8. Limitations and open items

* One validation split and two seeds for the main model; ablations are single runs.
* Training set is blood-only (46 studies, 7.4 k samples); the model was not retrained on all 46
  studies (6 held out for early stopping), and ElasticNet/ridge baselines use the same 40.
* Panel selection used the ages of all 46 training studies, so *validation-study* numbers during
  training are mildly optimistic; benchmark numbers are unaffected.
* Possible lab/cohort similarity between sibling series (see §1) and unknown overlap of individuals
  under different sample ids cannot be ruled out from metadata.
* Not tested: other tissues, non-human data, longitudinal/mortality outcomes, EPICv2-only probes,
  more than 12 k CpGs, larger or longer-trained models.
* The assignment-entropy option (`w_assign_entropy`) for sparser modules is evaluated in the
  section below if the run completed; modules in the headline model are diffuse.
