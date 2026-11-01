# Distilled compact RNA encoders: v2 project plan (Colab-first), differentiated from BIO-Distiller

Status: PLAN ONLY, nothing run. Written 2026-11-01 BEFORE any code or result. Supersedes sections 1 and 4-7 of `PLAN.md` where they conflict (prior art, licences, design choices in `PLAN.md` still stand). The decision rules in section 6 are fixed at commit time.
Evidence labels as in `PLAN.md`: **[read]**, **[figure]**, **[search]**, **[ours]**, **[judgement]**.

## 0. Why v1 was not enough, and where we could lose to BIO-Distiller

BIO-Distiller (EPFL, ICLR 2026 MLGenX workshop; PDF read in full) distils RNA-FM / RNA-MSM / UTR-LM into a **task-specific 3-layer CNN** on six BEACON tasks, uses a **teacher fine-tuned on each task**, feature-level "eucosine" loss, ASHA-tuned (100 configs) students and baselines, and reports +~2% R2 on regression tasks and +10 MCC points on non-coding RNA function **[read]**. Code is public but **has no licence file** (repo root has none) **[read: repo listing]**.

On THEIR ground (IID BEACON splits, per-task students, task-fine-tuned teacher) a generic backbone distilled once will most likely lose to per-task distillation **[judgement]**. So v2 does not compete there. We lose to BIO-Distiller if our only claim is "distillation improves compact RNA models". We must claim things they did not test.

## 1. What is different (the four differentiators, each testable)

1. **Out-of-distribution retention.** They report in-distribution test sets of BEACON; their tuned CNN matches or beats the foundation models on 4 of 6 tasks **[read]**. That conclusion may depend on the split: a CNN can exploit family/sequence similarity between train and test that a foundation model need not. We evaluate under **family-held-out and species-held-out** protocols where RiNALMo's published numbers show the strongest size dependence (inter-family secondary structure mean F1 0.43 for 33M vs 0.71 for 650M; splice site on unseen species) **[read: RiNALMo arXiv v2 Tables S3, S5]**. Question: does the student keep the teacher's OOD advantage, or only its IID advantage?
2. **Amortisation.** One task-agnostic distillation + N cheap fine-tunes vs N per-task distillations (their recipe, same student architecture). We report accuracy vs total compute and the crossover N.
3. **Size ladder and teacher.** Students from 0.25M to 20M parameters from RiNALMo-giga (650M), not a single CNN from ~100M teachers.
4. **Leakage control and release.** Distillation corpus filtered against every evaluation sequence; code, weights and the exact filter lists released with a licence file.

What we will NOT claim: beating per-task KD on IID tasks; "SOTA"; "foundation model".

## 2. Research questions

* **RQ1 (headline).** On tasks with an IID and an OOD test set, does a task-agnostic distilled student retain more of the teacher's advantage over a tuned from-scratch twin under OOD than a from-scratch student does? (Is the OOD teacher advantage there to inherit at all? Gate G1.)
* **RQ2.** Backbone-once vs per-task KD: accuracy per total GPU-hour over N tasks.
* **RQ3.** Which recipe ingredients matter (loss, matched layers, target compression, corpus size/clustering, teacher size) and how small can the student be.

## 3. Arms (same student architecture family at each size)

| arm | what | purpose |
|---|---|---|
| T | RiNALMo teacher: official fine-tuned checkpoints (inference only) where they exist, else published numbers | upper reference |
| S0 | from scratch, fine-tuned per task, tuned (small budget of ASHA-style search) | B in the retention formula; BIO-Distiller's "tuned baseline" analogue |
| S1 | MLM-only backbone pretrained on the SAME corpus, then fine-tuned | separates "more pretraining" from "distillation" |
| S2 | OUR arm: task-agnostic embedding-distilled backbone (cosine + MSE, 2 layers, PCA-compressed cached targets), then fine-tuned | RQ1/RQ2 |
| S3 | BIO-Distiller recipe re-implemented on the SAME student architecture: per-task fine-tuned teacher, eucosine on penultimate features, plus their 3-layer CNN as S3b if budget allows | RQ2 and the direct comparison with prior art |
| S4 | contrastive + MLM backbone (Rfam-family positives, Orthrus-style idea) | tests whether non-distillation tricks beat distillation (the mRNA figure hints they might **[figure]**) |

## 4. Tasks and splits

OOD primary (published protocols with published teacher numbers, **[read]** in RiNALMo arXiv v2):
* **Splice site (GS_1)**: IID (multi-species held-out sequences) and OOD (unseen species Fish/Fly/Plant/Worm).
* **Inter-family secondary structure** (Szikszai et al., 3,865 RNAs, 9 families, leave-one-family-out) as the **stretch** task: it needs a pair head and is where I expect tiny students to lose most **[judgement]**.
* **bpRNA TS0** secondary structure as IID counterpart of the stretch task.

Head-to-head with BIO-Distiller (their six tasks, BEACON data): NCR, APA, MRL, PRS OFF/ON/ONF. What the BEACON paper says about those splits (read by a subagent; main text verbatim, appendix only as a summary, so appendix items are **unverified**): no task has a family- or species-held-out test set; the main text never states how any split was made or addresses leakage; NCR's test set is class-balanced with a CD-HIT redundancy filter (threshold reported as "20%", probably misreported); APA holds out 3 of 12 libraries and has a separate "unseen libraries" set in appendix A.6; MRL ships a separate set of 7,600 real human 5'UTRs; PRS tests on the highest QC level. The paper's Table 1 counts disagree with its own text and appendix in several places. So their tasks are mostly IID-like, and **their conclusion that tuned CNNs match foundation models was not tested under family- or homology-held-out splits**.
Consequences for us: (a) the like-for-like IID comparison uses BEACON's own splits; (b) we add OOD pairs that already exist in the data: **MRL Random7600 vs Human7600** (RiNALMo published R2 0.925 vs 0.860 for 650M, 0.884 vs 0.827 for 33M), APA unseen libraries (if the files are in the BEACON download), and a **homology-aware re-split** of NCR (cluster by identity, split by cluster; the BEACON filter threshold is unverified) **[judgement on feasibility]**.

Secondary frozen-probe tasks (cheap, ours): LncBench `loc_clf`, `rbp_K562`, `rbp_HepG2`, `coding_clf` (**[ours]**, chromosome/homology-aware).

## 5. Colab execution plan (everything runs on Colab; M4 only for debugging and the single headline reproduction)

Rules for every notebook: one notebook + one zip uploaded as before; all state on Drive (`/content/drive/MyDrive/rna_distill/`); every stage idempotent and resumable (checkpoint every N minutes, skip finished stages); bf16; plain PyTorch attention (no flash-attn dependency; our RiNALMo re-implementation `compare/oligoai_score.py` already runs without it **[ours]**); fixed seeds; each notebook has a printed "stage ledger" so a disconnect costs minutes, not hours.

| nb | does | needs | est. A100 time | stop/go |
|---|---|---|---|---|
| N0 | install; fetch from Zenodo record 15043668 (v2, CC BY 4.0, 42.4 GB, 18 files): `rinalmo_giga_pretrained.pt` (2.6 GB) + the fine-tuned giga files we need (splice donor and acceptor, MRL, ncRNA class; the ten secondary-structure files only for the stretch task, ~2.6 GB each) **[read]**; choose a route: **(a)** official repo in a pinned environment (README pins python 3.11, torch 2.1 + CUDA 11.8, flash-attn 2.3.2 **[read]**; compiling or finding a matching wheel on Colab is the risk) or **(b)** our plain-PyTorch encoder (`compare/oligoai_score.py`, no flash-attn) plus re-implemented task heads read from the Apache-2.0 source (the OligoAI precedent worked: strict key match, Pearson 0.633 **[ours]**); throughput at 128/256/512 nt | the `rinalmo/` package source (pretrained.py and the four datamodules were NOT read) | ~0.5-1 h | stop if neither route reproduces a published number within 0.01 on one small eval (e.g. MRL Random7600 R2 0.925) |
| N1 | **Gate G1**: teacher (official fine-tuned) vs S0 (tuned scratch, 3 sizes) on splice site IID/OOD (and TS0 / inter-family if the pair-head code is cheap); plus the published ladder for context | GS_1 download (git.unistra.fr / Zenodo 7995778), CT splits (github marcellszi/dl-rna release) | ~3-5 h | **G1: pivot to Plan B if the teacher's advantage over S0 does not grow from IID to OOD on any task (section 7)** |
| N2 | corpus + leak filter + cached targets: stream RNAcentral active FASTA (11 GB compressed, CC0, release 28), length 16-512, cluster (MMseqs2 or MinHash), filter against all eval sequences, run the teacher once, store PCA-compressed per-token targets for 2 layers + pooled vector on Drive | disk (~11 GB download, 10-30 GB targets), MMseqs2 binary availability on Colab (unverified) | ~3-5 h (teacher pass ~0.5-1 h **[judgement]**) | stop if targets exceed Drive/disk or teacher throughput is <20 seq/s at 256 nt |
| N3 | **Pilot at 5M**: S0, S1, S2 x 3 seeds, pretrain + fine-tune on splice IID/OOD | N1 task list, N2 cache | ~6-10 h | **G2: stop if S2 does not beat S1 (MLM-only) by more than the seed spread on the OOD split** |
| N4 | full ladder (0.25M, 1M, 5M, 20M) x arms S0-S3 (+S4 if G2 passed) x 3 seeds, all included tasks, plus ablations for RQ3 | G2 passed | ~20-30 h | rule in section 6 |
| N5 | analysis, tables, plots, model cards, upload to Hugging Face, draft report | - | ~1 h | - |

**Budget.** N0-N3 (the part before any large spend): roughly 13-21 A100 hours; full plan roughly 35-55 hours **[judgement, nothing measured]**. At the reported A100 burn of ~5-13 compute units/hour (varies by source; read your own Resources tab) that is ~65-270 units through G2 and ~175-715 for the full plan. Pro has 100 units/month; Pro+ ~500-600 **[search, unverified against Google]**. So: G1 and G2 are designed to be reachable on one Pro+ month; the full ladder may need a second month or a reduced grid (drop S4, 2 seeds on the 0.25M and 1M sizes).

## 6. Pre-registered decision rules (fixed now; do not edit after results)

Definitions per task t and split s in {IID, OOD}: B = mean score of S0 over >= 3 seeds; T = teacher score (official fine-tuned checkpoint, our evaluation; published number only as a cross-check); S = student arm mean over >= 3 seeds; "spread" = max seed-to-seed standard deviation among arms on that task/split. Retention R(s) = (S - B) / (T - B).
* **Task inclusion (decided in N1, before any distilled student exists):** t is included iff T - B > 2 x spread on the OOD split.
* **G1 (pivot gate):** G1 passes iff at least one included task exists AND (T - B)_OOD > (T - B)_IID by more than the spread on at least one task. Else go to Plan B.
* **G2 (stop gate, 5M pilot):** passes iff S2 > S1 by more than the spread on the OOD split of at least one included task, and S2 > S0 on the same.
* **Headline (RQ1) met** iff, at 20M, R_OOD(S2) >= 0.75 on at least half of included tasks AND S2 > S1 OOD by more than the spread on those tasks AND R_OOD(S2) >= R_OOD(S0-like control: S1) on all of them. The 0.75 and the margins are my judgement of "meaningful", chosen before data.
* **Ladder claim** for a smaller size needs R_OOD >= 0.60 on at least half of included tasks and S2 > S1 OOD.
* **RQ2 claim** only with total GPU-hours measured for both S2 (one distillation + N fine-tunes) and S3 (N distillations incl. N teacher fine-tunes counted); report crossover N; no claim on accuracy superiority over S3 unless S2 exceeds S3 by more than the spread.
* **Stop/negative:** R_OOD(S2) < 0.5 on a majority of included tasks at 20M, or S2 <= S1 on every included task. Reported as a negative result ("distillation does not transfer OOD advantage at these sizes") and released as such.
* No use of "SOTA", "foundation model", "beats RiNALMo" unless a student exceeds T with the seed spread reported.

## 7. If we fail: pivots (decided by gates, not by mood)

* **Plan B (triggered by G1 failing; cheap, positive-or-negative, our strength).** "Do random splits hide the foundation-model advantage on RNA benchmarks?" Reuse N1 machinery: on the BEACON tasks BIO-Distiller used (and splice/structure tasks), compare tuned from-scratch CNNs with RiNALMo/RNA-FM fine-tuned (official checkpoints where they exist) under the original split AND a homology-aware split; report how the FM-vs-CNN ranking changes. Whatever the direction, it is a measurement that bears directly on BIO-Distiller's headline premise (tuned CNNs match FMs). Cost: ~10-15 A100 hours **[judgement]**. It is a benchmark-audit paper, in the same vein as our earlier leakage audits **[ours]**.
* **Plan C (G2 fails but G1 passes).** Release the ladder and S0/S1 models as a resource with the OOD tables and say distillation did not help beyond MLM-only training. Smaller contribution; honest.
* **Plan D (stretch, only if A or B finishes early).** Distil structure: add a pair/contact-map target from RiNALMo (inter-family structure is where size mattered most) and test whether a tiny student keeps cross-family structure generalisation. High risk, high interest **[judgement]**.

## 8. Facts resolved while writing, and what is still open

Resolved **[read]**:
* RiNALMo weights: Zenodo 15043668 (v2, 2025) is a superset of 10725749 (v1, 2024); both list licence "Creative Commons Attribution 4.0 International". v2 files: pretrained micro 134 MB / mega 592 MB / giga 2.6 GB; giga fine-tuned splice (donor, acceptor), MRL, ncRNA class (0-noise and 200-noise), and ten secondary-structure files (bpRNA plus nine ArchiveII per-family files, apparently one per held-out family, naming only). Code is Apache-2.0.
* RiNALMo official usage: README pins flash-attn 2.3.2 and CUDA 11.8; none of the four training scripts imports flash_attn directly, but the `rinalmo/` package was not read, so whether it hard-requires it is unknown. The README documents fine-tune/eval commands for secondary structure, MRL and splice only; the ncRNA script is undocumented. Splice scripts have no `--test_only`; eval-only means omitting `--data_dir` (a reading of the code, not run). Dataset download is behind `--prepare_data` in four datamodules not yet read.
* RNAcentral: CC0 from release 20 onward; current release 28 (2026-10-07), 61,026,481 unique sequences; active FASTA ~11 GB compressed at `ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz`; no confirmed per-sequence RNA-type table. Rfam: CC0, release 15.1, 4,227 families, `family.txt.gz` 370 KB (columns unverified).
* BIO-Distiller code: public, no licence file, PyTorch Lightning + Hydra + Ray Tune, `multimolecule==0.0.8` for teachers (the AGPL port, see `PLAN.md` section 3), old pins (torch<=2.3, numpy<2); split handling not visible in what was read.
* BEACON: see section 4.

Still open:
* MMseqs2 on Colab (static binary availability); GS_1 size and licence; the Szikszai/ct-splits download route; whether `multimolecule` is needed anywhere in our own pipeline (it should not be).
* Whether BIO-Distiller's datamodule uses BEACON's own splits.
* The BEACON data licence (none stated) and whether the Drive download includes the APA unseen-library and MRL Human7600 files.
* Everything about throughput, memory and units: unmeasured.
