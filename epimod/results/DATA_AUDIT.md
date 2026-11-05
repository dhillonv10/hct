# Data audit for the EpiClock training corpus

Goal: train on the cleanest data available; anything not clean is counted, labelled with the reason, and not used.
All numbers below come from audits run on the real data (EWAS Data Hub metadata: 218,448 samples; the bundle: 45,280 profiles; a 10,000-profile sample of the 226,555-profile MethylGPT set).
`epiclock.py` redoes every step on first run and writes the exact counts for the full MethylGPT set to `data_audit.md` next to the results.

## 1. What "the most data" means here

Every human DNA-methylation array profile we can obtain in processed form AND attach a health annotation to:

| source | profiles | note |
|---|---|---|
| bundle (ComputAge Bench training studies 7,418; ComputAge benchmark 10,404; public GEO labelled 11,807; public GEO unlabelled 15,651) | 45,280 | 9,098-CpG panel |
| MethylGPT / EWAS Data Hub / ClockBase | 226,555 | 49,156 CpGs, 3,709 of them in our panel; no labels in the file |
| overlap between the two (same sample id) | about 53,000 | |
| **union** | **about 218,000 unique profiles** | the ceiling; everything below is a subset |

EWAS Data Hub's curated metadata (218,448 rows = 212,455 unique sample ids: sample type, disease, tissue, age, ~370 other fields) is the master list for health status. ClockBase publishes no downloadable annotations.

## 2. Disease-state breakdown of the universe (EWAS Data Hub metadata, 212,455 unique sample ids)

| tier | samples | with an age | used? |
|---|---|---|---|
| CANCER (tumour, adjacent normal, TCGA/TARGET/CPTAC-type projects, any cancer word) | 81,540 (38.4%) | 37,910 | **never** |
| clean healthy (curated control, none of ~150 condition fields filled, not a cell culture) | 61,461 (28.9%) | 37,467 | yes, after the checks in section 5 |
| non-cancer disease | 35,117 (16.5%) | 18,082 | no |
| control carrying a condition field (treatment, stage, time point, genotype, ...) | 29,357 (13.8%) | 11,661 | no (first candidate to add back if more data is wanted) |
| HIV cohort, no cancer | 3,346 (1.6%) | 3,161 | allowed, but these cohorts are the benchmark's own test data, so almost none reach training |
| cell culture / line | 1,622 (0.8%) | 476 | no |

Most frequent non-cancer disease labels: COVID-19 2,240; Alzheimer's 2,239; Parkinson's 1,943; schizophrenia 1,691; dementia 1,227; rheumatoid arthritis 1,090; asthma 949; stroke 943; Crohn's 911; SLE 801; first-episode psychosis 720; endometriosis 702; ulcerative colitis 691; depressive disorder 620; multiple sclerosis 572; idiopathic pulmonary fibrosis 510; obesity 468; autism 341; epilepsy 307.

EWAS's own "control" label is NOT "healthy": 29,357 controls carry a condition field, 4,823 controls carry a disease label, and "control" includes HIV+ patients, childhood-cancer survivors, treated samples.

## 3. The bundle, by source (EWAS tier of each profile)

| | ComputAge training (7,418) | labelled GEO (11,807) | unlabelled GEO (15,651) |
|---|---|---|---|
| clean healthy | 4,581 | 3,272 | 1,910 |
| control carrying a condition field | 743 | 1,113 | 1,358 |
| cancer | 5 | 799 | 1,078 |
| non-cancer disease | 0 | 1,960 | 1,200 |
| cell culture | 0 | 21 | 57 |
| **no annotation (not in the EWAS table)** | 2,089 | 4,642 | 10,048 |

The ComputAge benchmark (10,404 profiles) is the TEST set: 2,830 healthy controls are scored; the disease cohorts (HIV 3,102, RA 947, AS 930, PD 853, MS 498, IBD 402, ...) are used only to test disease signal. None of it is trained on.

## 4. MethylGPT set (226,555 profiles), from a 10,000-profile sample

| what we know about the profile | share | ~profiles |
|---|---|---|
| cancer | 24.1% | 54,600 |
| already in the bundle (or a benchmark / held-out-series sample) | 23.6% | 53,600 |
| clean healthy (EWAS tier) | 19.5% | 44,200 |
| not in the EWAS table (no health annotation) | 15.6% | 35,400 |
| non-cancer disease | 8.6% | 19,500 |
| control carrying a condition field | 7.9% | 17,900 |
| cell culture / line | 0.6% | 1,300 |
| HIV cohort | 0.05% | 110 |

The 44,200 EWAS-clean profiles shrink to about 36,700 after the GEO-description, cancer-themed-study, benchmark-study and copy-of-test-sample checks (2.4% of the survivors were copies of benchmark samples).

## 5. The checks a profile must pass to be trained on, and what each removes

1. EWAS tier is clean healthy or HIV cohort (ComputAge training studies are healthy controls by curation: EWAS cancer/disease/culture still removes them).
2. GEO's own sample description (title, source, characteristics) does not name a disease or cancer (independent second opinion; conservative, some false positives). Disease words are not applied to ComputAge's curated controls, whose descriptions name the disease of the study they were drawn from.
3. The study is not cancer-themed (title / summary / design). Found necessary: EWAS "clean" included cancer patients, e.g. EPIC-Italy cancer cases (GSE51032), esophageal cancer saliva (GSE232332), pancreatic cancer (GSE74071), Wilms tumour normals (GSE59157), osteosarcoma (GSE193000).
4. The study is not one of the benchmark datasets.
5. The profile is not a re-deposited copy of a benchmark sample (found by comparing methylation profiles, RMS difference < 0.04 over co-measured CpGs; the distribution has no values between 0.005 and 0.04 for the bundle, i.e. a clean gap).

Result (bundle exact; MethylGPT extrapolated from the sample):

| source | start | removed by the checks | **clean** | of which with an age |
|---|---|---|---|---|
| ComputAge training | 7,418 | 5 cancer; 159 cancer-themed studies; **257 exact copies of benchmark test samples** | **6,997** | 6,997 |
| bundle labelled GEO | 11,807 | 799 cancer; 1,981 disease/culture; 1,113 control+condition; 4,642 no annotation; 597 GEO text; 348 cancer-themed | **2,327** | 2,327 |
| bundle unlabelled GEO | 15,651 | 1,078 cancer; 1,257 disease/culture; 1,358 control+condition; 10,048 no annotation; 154 GEO text; 247 cancer-themed | **1,509** | 0 |
| MethylGPT, not already in the bundle (about 173,000) | 173,000 | about 55,000 cancer; 19,500 disease; 18,000 control+condition; 35,400 no annotation; 7,500 GEO text / cancer-themed / benchmark study; 900 copies of benchmark samples | **about 36,700** | about 22,000 |
| **total training corpus** | | | **about 47,500** | **about 31,000 labelled, about 16,000 label-free** |

For comparison, the unscreened bundle had 16,849 labelled training samples and a 32,500-profile pre-training pool.

## 6. Findings that change earlier results

- 257 ComputAge "training" samples (study GSE72773) are exact copies of benchmark healthy controls (GSE111629: 237, GSE72776: 20), and 45 labelled GEO rows are exact copies of benchmark samples. All earlier test numbers (including ridge 3.57 and the xl model's 3.22) were measured with these copies in training. The clean corpus removes them.
- 5 cancer patients (colon, prostate, breast, stomach, leukaemia; GSE130748) sit inside ComputAge's "healthy control" training set.
- 2.4% of the otherwise-clean MethylGPT profiles (39 of 1,657 in the sample) are re-deposited copies of benchmark samples under different ids (e.g. the PD cohort GSE111629); removed by profile comparison.
- Age labels in EWAS Data Hub agree exactly (|diff| <= 0.5 y) with the bundle's ages for all 12,230 overlapping samples.

## 7. Known limits

- "Clean" means: curated as control, no condition field, and not contradicted by GEO's own description. Curated labels can hide unlabelled illness; this cannot be excluded.
- MethylGPT profiles measure 3,709 of the 9,098 panel CpGs (41%), but 86% of ridge's top-500 age CpGs and 94-96% of the Horvath V1 / Lin clock CpGs.
- About 16% of MethylGPT profiles and 64% of the bundle's unlabelled GEO rows have no health annotation; by the rule "no annotation, no data" they are not used.
- About 4% of the 2,830 benchmark healthy controls that EWAS knows are marked "disease tissue" there; the benchmark's own definition is kept for the test set.
- Test-set exposure of the bundle was audited only against the ComputAge benchmark; other public cohorts that are not in either source are not covered.
