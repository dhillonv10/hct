# Compact RNA model family: plan, claims we can and cannot make, go/no-go gates

Status: plan only, nothing below "Next step" has been run. "Measured" = produced by code in this directory (`RESULTS.md`).
"Search only" = read in a web-search snippet this session, not verified against the paper, data or licence.

## 1. What "the family" is

One recipe applied to many RNA tasks, not one big model. Three tiers, all small enough to train on an M4 in hours:

| tier | members | size |
|---|---|---|
| T0 thermo-GBM | nearest-neighbour thermodynamics (ViennaRNA), local accessibility, k-mers, position one-hots -> LightGBM / ridge | ~10^2 - 2x10^4 parameters |
| T1 mini-nets | Tiger2D-style CNN, MiniPairFormer (structure-biased attention) | 2.5x10^4 - 2.4x10^5 |
| T2 rank ensemble | within-unit rank average over heterogeneous T0/T1 members (and published tools where the licence allows) | sum of members |

## 2. What we have measured (Cas13d on-target, siRNA; leak-free)

* Parameters 130 -> 63,000,000 (ridge ... TabDPT, RNA-FM embeddings) all land at within-gene Spearman 0.52 - 0.62. Beyond ~10^4 parameters size does not help.
* The only thing that moved the number is heterogeneous ensembling: 3-member ensemble 0.638 vs best single 0.612 (paired +0.026 [0.020, 0.032]; locked genes +0.025 [0.013, 0.036]).
* Against the released TIGER on two external on-target screens: flow-cytometry 0.656 vs 0.599 (+0.057 [0.041, 0.073]); HAP1 0.195 vs 0.111 (+0.084 [0.066, 0.101]). Against our own re-trained Tiger2D the margin is the +0.026 above.
* siRNA (Huesken -> Mixset): ours AUC 0.813 [0.778, 0.853] vs OligoFormer's published 0.816 and DeepSilencer's 0.820. Parity, and the test set is too small (+-0.04 AUC) to resolve differences of that size.
* Mismatch tolerance: the released TIGER is better than ours.
* Evaluation audits that hold regardless of model: pooled-Pearson inflation from per-gene de-normalisation, shuffled-CV inflation, checkpoint-on-test (read, not measured).

## 3. Claims

Defensible now:
1. Under gene-held-out evaluation, models with ~10^4 - 10^5 parameters match 10^7 - 10^8-parameter models on guide efficacy, and a heterogeneous ensemble beats the best single model by ~0.03.
2. That ensemble beats the released TIGER on two independent Cas13d screens.
3. Published guide-efficacy evaluations contain measurable leaks (6.1, 6.3) or unmeasured ones (6.4).

Not defensible now: "beats SOTA" on siRNA, mismatch tolerance, or any task not yet run. Do not write it until a locked-set result says so.

## 4. Task shortlist and gate status

A task joins the family only if all four gates pass: (G1) data downloadable and licence permits benchmarking; (G2) a published SOTA that we can run or whose predictions we can read; (G3) a leak-free held-out unit exists (gene, family, chromosome); (G4) headroom, see section 5.

| task | G1 data | G2 SOTA | G3 unit | expectation (my judgement, not a result) |
|---|---|---|---|---|
| Cas13d on-target | done | TIGER, DeepCas13 (done) | gene (done) | achieved: beats released TIGER, parity with retrained |
| siRNA | done (OligoFormer tables) | OligoFormer is proprietary, not run; DeepSilencer repo has no code | target chunk | parity; needs a larger test set before any claim |
| ASO (RNase H gapmers) | VERIFIED. Zenodo 10.5281/zenodo.20794660, CC-BY-4.0, one 7.0 MB gz CSV, md5 matches. 178,617 rows after de-duplication (the paper says 188,521; this is the frozen 2025-08-21 table), 155k unique sequences, 2,022 screens (patent tables; one gene + cell line + dose each, ~78 ASOs), 327 gene groups | OligoAI = RiNALMo-giga fine-tune (read from its repo: `--lm_config giga`; ~650M parameters per the RiNALMo paper, not counted by me). Code Apache-2.0. Weights: one 3.08 GB `.ckpt` on Hugging Face, licence NOT stated on the card, and it is a pickle referencing `rinalmo.utils.scaler.StandardScaler`, so loading it executes code (untrusted: allow-list loader only, M4). The model card gives no metrics and no split. The released training code scores mean within-screen Spearman (our metric) but SPLITS BY PATENT (80/10/10, seed 42): re-created on this table, 5 of 38 test gene groups (43.9% of test rows) also occur in training patents (`scripts/aso_oligoai_split_audit.py`). The paper's own test set (search snippet: 299 screens vs 235 here) is NOT verified | gene groups (aliases merged, see `aso.py`) | best candidate: same feature stack applies, large n. A screen-level split would leak (8.6k gene-sequence pairs sit in 2+ screens) |
| mRNA half-life | Zenodo 6326409 (18.8 GB; licence field not checked) | Saluki, code reported Apache-2.0 (search only; check LICENSE) | gene / chromosome (split not verified) | Saluki is already compact; "close to" is plausible, "beats" is unlikely |
| 5' UTR mean ribosome load | public GEO libraries | Optimus (small CNN); UTR-LM (search only: +<=5% Spearman over it) | random-sequence libraries, low leakage risk | near saturation; low novelty |
| BEACON suite | code Apache-2.0; data on Google Drive (manual download, no data licence in the README; per-source licences in the paper's Table 22 are mixed, e.g. OpenVaccine and StructImpute "non-commercial") | 17 models x 13 tasks published (section 4a) | mostly random or unstated splits; no family-held-out split | cheapest breadth; "SOTA" is a table, not one tool |

### 4a. BEACON, as read by a subagent from arXiv 2406.10391v2 and its GitHub repo (not re-checked by me; verify against the paper before quoting)

13 tasks, ~967k sequences. Pretrained LMs beat the paper's literature SOTA on 8 of 13 (SSP, SSI, SPL, APA, ncRNA, Modif, MRL, PRS); the literature row is still ahead on CMP, DMP, VDP, CRI-On, CRI-Off.

Three tiny supervised nets (CNN / ResNet / LSTM) are the useful comparison for our claim. From the paper's Table 3 means, gap between the best of the three and the best of all 17 models:

| task (metric) | best of CNN/ResNet/LSTM | best of all 17 | gap |
|---|---|---|---|
| Modif (AUC) | 94.83 | 95.14 | 0.3 |
| MRL (R2) | 83.94 | 84.98 | 1.0 |
| CMP (P@L) | 59.59 | 61.20 | 1.6 |
| PRS (R2) | 55.45 | 57.72 | 2.3 |
| VDP (MCRMSE, lower better) | 0.329 | 0.313 | 0.016 |
| CRI-Off (Spearman) | 11.50 | 11.50 | 0 |
| SSI (R2) | 37.74 | 42.36 | 4.6 |
| CRI-On (Spearman) | 29.69 | 34.92 | 5.2 |
| APA (R2) | 67.03 | 72.71 | 5.7 |
| ncRNA (ACC) | 88.78 | 96.81 | 8.0 |
| SSP (F1) | 58.61 | 68.50 | 9.9 |
| DMP (R2) | 44.77 | 56.82 | 12.0 |
| SPL (ACC@K) | 36.66 | 50.55 | 13.9 |

Reading: tiny nets are within ~2 points of the best foundation model on six of thirteen tasks, and a tiny net is already best on CRI-Off. On the other seven the large pretrained models are clearly ahead, and a compact model is not going to close 5 - 14 points there. Those seven stay out of the family.

Limits of this table as a SOTA reference: no homology-held-out split is described for any task; the split is unstated for CMP, DMP, Modif, CRI-On and CRI-Off; the paper's own counts disagree with its Table 1 in several places; the "first place" claim for SpliceBERT-MS1024 is not supported by Table 3. CRI-On / CRI-Off are SpCas9 knockout (DeepCRISPR) data, not RNA-targeting guides, and the literature SOTA there (44.1 / 12.6) is above every model in the table, so a gap there would need information (epigenetic features) that sequence-only models do not have.

## 5. Protocol and kill criteria (fixed before running)

* Per task: freeze a dev/locked split before the first model is fit (as for Cas13d: 12 dev genes, 4 locked). Design on dev only; the locked set and external sets are scored once.
* G4 headroom test, dev only, one day of M4 compute: run T0 and T1 and the best published tool. Proceed if the best of T0-T2 is within 0.02 of SOTA on the primary metric; stop on that task otherwise.
* Primary metric per task: within-unit Spearman (guide tasks), Spearman / Pearson as the task's literature uses (mRNA tasks), always with a block or unit bootstrap CI.
* Family-level kill: if after three tasks the ensemble is below SOTA by more than 0.03 on two of them, drop "family beats SOTA". The remaining paper is then (a) parameter-efficiency plus (b) the evaluation audits, which is already supported by section 2.
* Never combine with, run, or port a proprietary comparator (OligoFormer). Respect TIGER's non-commercial licence; do not redistribute weights or per-guide predictions.

## 6. Next step

1. Verify ASO Atlas bulk download and licence (the bioRxiv full text was rate-limited, HTTP 429, on both attempts; retry or have the data-availability statement pasted).
2. If G1-G3 pass, export ASO Atlas with gene ids, freeze a dev/locked split, run the G4 headroom test.
3. BEACON breadth track (user downloads the Drive folders on the M4; the sandbox cannot be assumed to reach Google Drive): fit T0 / T1 / T2 on the six near-parity tasks (Modif, MRL, CMP, PRS, VDP, CRI-Off), dev-first on each task's own validation split, test split scored once. The realistic target there is "within noise of the best foundation model at 10^-3 of the size", and "beats" only if the ensemble gain seen on Cas13d (~+0.03 Spearman) carries over. Before quoting any BEACON number, re-check Table 3 against the paper.
4. Saluki: read the repo LICENSE and the split definition before deciding whether it enters.

## 7. ASO dev results (measured; development genes only, 246 gene groups, 1,360 screens; the 81 locked groups are untouched)

Within-screen Spearman, mean over screens with n >= 20, gene groups held out (5-fold), 95% CI by bootstrap over gene groups. Logs: `results/aso/aso_dev_log.txt`, `aso_dev2_log.txt`; table: `aso_dev_table.csv`.

| model | rho_screen [CI] |
|---|---|
| LightGBM, sequence + chemistry + experiment | 0.352 [0.317, 0.378] |
| + thermodynamics (duplex energy, ASO self-fold) | 0.375 [0.338, 0.401]  (paired +0.023 [+0.015, +0.029]) |
| + target-site accessibility (ViennaRNA partition function on the supplied context) | 0.388 [0.352, 0.417]  (paired +0.013 [+0.004, +0.024]) |
| same, within-screen rank-Gauss target | 0.399 [0.360, 0.427]  (not tested paired against the raw target) |
| small CNN, 110,789 parameters | 0.346 [0.308, 0.375]  (vs LightGBM -0.042 [-0.074, -0.014]) |
| rank ensemble LightGBM + CNN | 0.398 [0.362, 0.423]  (vs LightGBM +0.010 [-0.004, +0.021], not distinguishable from 0) |

Split-type test, same model (LightGBM, all features), difference from the gene-group split: patents held out -0.000 [-0.009, +0.006]; screens held out +0.062 [+0.049, +0.077]; random rows held out +0.071 [+0.058, +0.087]. So the patent split used by OligoAI's released code does not inflate this model, while screen-level and random splits do by 0.06 - 0.07. Whether a ~650M-parameter model benefits more from shared genes than a boosted model is untested.

What this does and does not show:
* each physics-motivated feature group helps, with intervals above zero;
* a small neural net does not beat the boosted model on this task, and ensembling them gains +0.010 here (not significant), less than on Cas13d (+0.026);
* NO comparison with OligoAI exists yet, so there is no claim about SOTA. That needs OligoAI's predictions on the same rows (M4 job; weights licence unstated; checkpoint is a pickle, load with a restricted-class loader).

## 8. Head-to-head results against much larger models (pre-registered rules in compare/README.md; locked groups scored once)

| arm | compact model | larger model | result | rule |
|---|---|---|---|---|
| ASO, S1 (OligoAI's own protocol, 38 test patents) | LightGBM, 0.451 [0.341, 0.522] | OligoAI ~650M, 0.481 [0.349, 0.571] | -0.030 [-0.060, +0.017] | in the "inconclusive" band (0.02-0.03 behind) |
| ASO, S2 (81 locked gene groups; favours OligoAI, which trained on these genes) | 0.417 | 0.487 | -0.069 [-0.130, -0.018] | biased toward OligoAI |
| ASO, S2b (14 groups OligoAI never saw; 47 screens) | 0.388 | 0.420 | -0.033 [-0.063, +0.040] | misses the 0.03 condition by 0.003; CI includes 0 |
| mtDNA, ClinVar (113 P/LP) | 0.902 AUROC | Nucleotide Transformer 486M, 0.380 | NT adds 0 to the compact stack | better |
| mtDNA, population constraint (matched) | 0.831 | 0.487 | NT adds 0 | better |

Reading: the compact ASO model retains about 94% of OligoAI's within-screen Spearman on its own protocol (0.451 / 0.481) with ~10^-4 of the parameters, and picks equally good top-1 ASOs on S1 (1.48 vs 1.48);
it is not at parity by our own rule, and on gene-held-out screens it is behind by 0.03-0.07. The mtDNA comparator is a general nuclear-trained DNA model and a weak one; no mitochondria-specific tool has been compared.

### 8b. Post-hoc: compact model as an add-on to OligoAI (fixed 50/50 within-screen rank average, weight not tuned)

| setting | OligoAI alone | 50/50 with ours | difference |
|---|---|---|---|
| S1 (OligoAI's protocol, 232 screens) | 0.481 | 0.510 | +0.029 [+0.006, +0.052] |
| S2 (locked groups, favours OligoAI) | 0.487 | 0.503 | +0.017 [-0.015, +0.043] |
| S2b (genes OligoAI never saw) | 0.420 | 0.440 | +0.020 [-0.000, +0.059] |

The two models' within-screen rankings agree at Pearson 0.57-0.64, so they are complementary. Only S1 has an interval above zero; this is a post-hoc analysis and needs confirmation on genes neither model has seen. See compare/results/aso_ensemble_posthoc.txt.
