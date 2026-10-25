# lncRNA foundation model: plan, benchmark, and what the pilot showed

> Status legend used throughout: **[measured]** = I ran it in this repo and the number is in `results/`;
> **[verified]** = checked against the live source during this session (file listing, API, README);
> **[design]** = implemented or specified but not run at scale; **[assumption]** = my estimate, flagged so you can check it.

> **Update (LncBench v2).** §10 at the end of this file describes the two-layer benchmark (molecular + AML clinical) that
> supersedes the v1 benchmark in §5/§6, with new results. The verdicts in §0 still stand; §10.9 says what v2 adds.

## 0. Bottom line (read this first)

**Verdict: worth continuing, not yet proven, and the evidence says "start with transfer, run the from-scratch model alongside it".**
Everything below was measured on a 4-core CPU sandbox, so these are *pilot-scale* results (details and caveats in §5-§9).

1. **There is real but modest sequence signal in lncRNA.** On whole transcripts with chromosome-held-out evaluation, plain k-mer features
   predict nuclear-vs-cytosolic localization (Spearman 0.31, AUROC 0.71 for nuclear vs cytoplasmic terciles) and tissue specificity
   (Spearman 0.42), versus 0.07 / 0.56 / 0.17 for length + GC alone. That is the ceiling any sequence-only model is fighting for.
2. **A small from-scratch model learns something real but does not yet beat k-mers.** A 0.89M-parameter model trained for 49 CPU-minutes
   (~77 Mnt, about one pass over the unique gene-level corpus) beats its own *untrained twin* on 5 of 6 tasks (+0.01 to +0.05, paired 95% CIs exclude 0).
   Probe quality rose from step 500 to ~2000 and then flattened (so did validation MLM loss, after ~1,250 steps), but on whole transcripts it **ties** the k-mer baseline (localization Δρ +0.020, CI -0.011 to +0.048) and is
   worse on the coding-vs-lncRNA sanity task. Adding exon-junction + isoform-contrastive objectives helped modestly over plain masked LM.
3. **Transfer (Option B) is the strongest cheap lever, with caveats.** Frozen RiNALMo-micro (33M params, pretrained on ncRNA) significantly beats k-mers on
   localization (Δρ +0.10), testis-vs-rest and coding-vs-lncRNA at matched input (first 400 nt, n=800). Continual pretraining is implemented and smoke-tested but **not run at scale**.
   Evo-1 never saw a eukaryote; Evo 2 needs an NVIDIA GPU (1B also Hopper/FP8); Caduceus needs a port off `mamba_ssm`.
4. **Negatives that matter.** (a) Ortholog retrieval: nothing learned beats the alignment-free shared-10-mer baseline (300 queries vs a 3,000-gene mouse pool: hit@10 0.375 on the 56 named pairs vs 0.21 for the best learned model; at full scale, 36,103 genes, shared 10-mers still give 0.41).
   (b) Zero-shot ClinVar variant scoring *with my 0.89M pilot* is at chance (within-gene AUROC 0.55, CI includes 0.5), and ClinVar cannot support a lncRNA-wide variant benchmark at all (§6). **Update, §12:** with frozen RiNALMo-micro the same kind of score reaches 0.72 [0.61,0.83] within-gene on 13 small-ncRNA genes (mito tRNAs 0.84), and 0.84 [0.79,0.89] on the 7 nuclear genes when combined with CADD; still no usable ground truth for lncRNAs (only TERC).
   (c) Tying a patient model to lncRNA *sequence embeddings* gives no survival benefit, even with 60 training patients or RiNALMo embeddings.
5. **The clinical hook that works today is expression, not sequence.** lncRNA expression alone predicts TCGA survival about as well as mRNA
   (C-index: LGG 0.84 / KIRC 0.69-0.71 / LUAD 0.60-0.62) and adds +0.02 (LUAD) to +0.03 (KIRC) over age + stage; those gains are the size of the fold-to-fold standard error (~0.015) and were not tested for significance (§6).
6. **Do next (in this order):** (i) `selfcheck` + `speedtest` on the M4; (ii) frozen RiNALMo vs k-mer on *whole transcripts, all genes* (I could only afford 400 nt, n=800 here);
   (iii) one overnight `small` run with all objectives, judged on whole transcripts; (iv) continual pretraining of RiNALMo only if (ii) shows it wins; (v) a single rented-GPU session for Evo 2 7B embeddings if you want the only large model with eukaryotic training data.
   Decision criteria are in §8. Don't commit to a multi-day `base` run until (ii)/(iii) clear their gates.

---

## 1. What a lncRNA model could actually do

| # | Capability | What the model has to supply | Public data that exists | Status in this repo | Path to clinical outcome |
|---|---|---|---|---|---|
| 1 | **Functional triage** of the ~36k GENCODE lncRNA genes (>99% have no characterised function): subcellular localization, tissue restriction, RBP/m6A sites | A sequence embedding that predicts gene-level properties | ENCODE nuclear/cytosolic RNA-seq (15 cell lines), GTEx v8 (54 tissues); eCLIP/m6A are next | **[measured]** localization + tissue tasks | Prioritise which lncRNAs to follow up as biomarkers / targets |
| 2 | **Cross-species functional mapping**: find the mouse counterpart of a human lncRNA when sequence alignment fails | Embedding similarity that tracks function, not just alignable sequence | GENCODE human v50 + mouse M39 synteny | **[measured]** ortholog retrieval | Choose a mouse model for a lncRNA of clinical interest |
| 3 | **Expression biomarkers / prognosis**: lncRNA signatures for survival, subtype, tumour vs normal | Gene embeddings as priors when patients are few; imputation of lncRNAs missing on a platform | TCGA (GDC open access); GEO / ICGC / CPTAC for external validation | **[measured]** TCGA survival; sequence-tied prior is a **null so far** | Prognostic/diagnostic panels; the only track with real patient outcomes today |
| 4 | **Variant interpretation** inside lncRNAs / small ncRNAs | Zero-shot variant effect from sequence likelihood | ClinVar (thin, see §6), GWAS (noisy), MPRA/saturation mutagenesis (not yet public for lncRNA at scale) | **[measured]** feasibility + within-gene harness | Reclassifying VUS in ncRNA disease genes (RMRP, RNU4ATAC, RNU4-2, TERC, ...) |
| 5 | **Therapeutic design**: ASO/gapmer site accessibility, CRISPRi target choice | Accessibility/structure-aware embeddings | ASO tiling screens (scattered), not wired in | **[design]** only | Gapmer drug design for nuclear-retained lncRNAs |
| 6 | **Liquid-biopsy markers** (PCA3-like) | Stability + tissue-origin priors | exoRBase etc., not wired in | **[design]** only | Diagnostics |

What it will **not** do: lncRNA function is largely set by genomic context, expression program and cell state, not by transcript sequence alone. Sequence-only embeddings are a *prior*, and the pilot below says how strong that prior is.

---

## 2. Why lncRNA is hard, and the trick used for each difficulty

| Difficulty | Evidence from the data pulled in this session | Trick | Where |
|---|---|---|---|
| **Small corpus** | GENCODE v50 human: 191,063 lncRNA transcripts / 35,885 genes / 195.8 Mb, but only ~5.3 isoforms per gene so unique content is ~36 Mb; mouse M39: 155,878 tx / 36,103 genes / 143.7 Mb. **[verified]** | Small models (1-40M params), gene-balanced sampling (a random isoform per sampled gene), add mouse, held-out *chromosomes* instead of random splits | `pretrain.py: Corpus`, `gencode.py` |
| **Little conservation** | ~60% of known same-name human/mouse lncRNA orthologs are *not* retrieved by shared 10-mers **[measured]** | Evaluate with synteny-defined positional orthologs; isoform-level contrastive objective so the embedding is invariant to splicing; (design) cross-species positives | `orthology.py`, `retrieval.py`, `pretrain.py: nt_xent` |
| **Repeat-rich, low-complexity** | high-entropy: masked-LM accuracy plateaus near 0.40 | Span masking (not single-nt) so repeats cannot be copied from a neighbour; held-out chromosomes as the leakage control; (design) down-weight TE-derived windows | `pretrain.py: span_mask` |
| **Few labels, weak labels** | ENCODE localization n=2,921 genes; GTEx n=8,608 expressed lncRNAs | Label-free pretraining; labels used only for chromosome-grouped CV with bootstrap CIs; an untrained copy of the same network is always reported | `evalkit.py`, `bench.py` |
| **Length (median 0.8 kb, p99 4 kb, up to 350 kb)** | **[measured]** | Single-nucleotide input (no tokenizer to destroy repeat boundaries) but a stride-4 conv stem so attention runs at 1/4 length; windows + length-weighted pooling | `model.py` |
| **Compute on a laptop** | counted training FLOPs/nt: tiny 1.4M, small 10.4M, base 58.9M **[measured]** | Presets sized from counted FLOPs; pure PyTorch (CPU/MPS/CUDA), resumable, `speedtest.py` measures *your* machine | `speedtest.py` |
| **Don't pretrain what already exists** | RNA-native models trained on millions of ncRNA exist (RiNALMo, RNA-FM) | "Option B": frozen features first, continual pretraining second | `embed.py`, `cpt.py`, §4 |

---

## 3. Training objectives

Implemented in `lncfm/pretrain.py`; none uses a benchmark label.

1. **Span-masked language modelling** (15% of nt, spans of mean length 3, BERT 80/10/10 corruption). The base objective.
2. **Exon-junction tagging.** Per nucleotide, predict whether an exon-exon junction follows it (labels from the GENCODE GTF). Makes the encoder splice-aware, which matters because the mature transcript is what is localised, bound and degraded.
3. **Isoform contrastive (NT-Xent).** Two views of the same *gene* (different isoform, or two crops) are positives; other genes in the batch are negatives. Gives an embedding that is stable under alternative splicing and indel-scale noise.

Designed, not run (kept out of the pilot to keep ablations clean):

4. Cross-species positional-ortholog contrastive pairs, **using only human train-chromosome genes** (otherwise the retrieval task leaks).
5. Structure auxiliary targets (windowed MFE / pairing probability from ViennaRNA).
6. A supervised multitask stage on ENCODE localization + GTEx tissue vectors, *train chromosomes only*, as a second stage.

Ablation switches: `--lambda-j 0 --lambda-c 0` gives plain span-MLM on identical batches.

---

## 4. Option B (transfer / continual pretraining): verified facts and decision

Candidate base models, checked against the live HuggingFace repos / READMEs this session:

| Model | Params | Trained on | Runs on a Mac? | Notes |
|---|---|---|---|---|
| **Evo-1** (131k) | 6.45B in checkpoint (README says 7B) | OpenGenome: **prokaryotic** genomes only | No practical path (7B, custom StripedHyena kernels) | Never saw a eukaryote, so poor prior for human lncRNA |
| **Evo 2** 1B / 7B | 1B / 7B | OpenGenome2, 8.8T tokens "from all domains of life" (README) | **No.** README: 1B *requires* FP8 + Transformer Engine on a Hopper GPU; 7B requires CUDA + flash-attn | The only big candidate whose training data spans eukaryotes (the README does not itemise species, so "has seen human lncRNA loci" is likely but unverified); needs a rented NVIDIA GPU |
| **Caduceus-ps/ph** 131k | 7.7M | human reference genome | **Not as shipped**: `modeling_caduceus.py` imports `mamba_ssm` (CUDA/Triton) | Would need a pure-PyTorch Mamba port; also RC-equivariant, which is a mismatch for single-stranded RNA |
| **RiNALMo-micro / mega** | 33.5M / 148M | ncRNA; model-card dataset tags: RNAcentral, Rfam, nucleotide | **Yes, tested.** Loads via the `multimolecule` PyPI package, no remote code | AGPL-3.0 licence |
| **RNA-FM** | 99.5M | ncRNA | Yes via `multimolecule` (not run here) | AGPL-3.0 |
| **NT-v2-50M** | 55.9M | multi-species genomes | Needs `trust_remote_code` and an older transformers; failed to load here | CC-BY-NC-SA |
| **HyenaDNA-small-32k** | 4.1M | human reference genome | Needs `trust_remote_code`; **not run** (I did not execute third-party repo code in this sandbox) | opt-in only: `LNCFM_TRUST_REMOTE_CODE=1` |

Decision rule used (and why): continual pretraining on ~36 Mb of unique lncRNA cannot add much knowledge to a model that already saw millions of ncRNAs, so Option B is only worth the engineering if the **frozen** pretrained features already beat the k-mer baseline and the untrained control on this benchmark. That is exactly the measurement in §5.

**Measured, frozen features** (matched protocol: every model sees the same 800 transcripts, clipped to the first 400 nt; paired bootstrap; `results/paired_comparisons.md`) **[measured]**:

| contrast (A minus B) | loc_reg (rho) | loc_clf (AUROC) | tau_reg (rho) | testis_clf | brain_clf | coding_clf |
|---|---|---|---|---|---|---|
| RiNALMo-micro minus k-mer | **+0.102** [+0.050,+0.153] | **+0.052** [+0.024,+0.079] | +0.023 [-0.011,+0.058] | **+0.048** [+0.011,+0.088] | -0.007 [-0.059,+0.049] | **+0.107** [+0.079,+0.136] |
| RiNALMo-micro minus my 0.89M pilot | +0.040 [-0.014,+0.094] | +0.030 [+0.001,+0.057] | +0.013 [-0.018,+0.044] | -0.001 [-0.041,+0.038] | -0.028 [-0.079,+0.017] | **+0.145** [+0.117,+0.178] |

Reading: a frozen, off-the-shelf RNA model already beats k-mers where sequence matters most (localization) and on lncRNA-vs-coding, at zero training cost. My
37x smaller pilot is within noise of it on four of six tasks, but loses badly on lncRNA-vs-coding (RiNALMo saw millions of RNAs, mine saw ~77 Mnt of lncRNA).
On ortholog retrieval frozen RiNALMo is *not* better than a k-mer spectrum (hit@10 on the 56 named pairs: 0.214 vs 0.196; shared 10-mers: 0.375).

**What is not measured:** RiNALMo on whole transcripts and all genes (about 0.5 s per 400-nt sequence on 2 CPU threads here, so hours for the full benchmark; trivial on a GPU and feasible overnight on the M4);
continual pretraining at any meaningful scale (`lncfm/cpt.py` runs; a 25-step smoke test moved held-out MLM loss 1.87 -> 1.64, but that is mostly the model learning to put mass on A/C/G/U only,
so compare MLM losses on the four nucleotide logits or, better, use the probes). Whether CPT beats the frozen model is the open question; run it with the commands in §8.

**Decision rule (written after the matched-protocol results above, but before any whole-transcript RiNALMo or CPT results exist):** use the frozen pretrained model as the default feature extractor. Spend compute on continual pretraining only if, on whole transcripts and all
genes, CPT beats the *frozen* model with a paired CI excluding 0 on at least two of {loc_reg, loc_clf, tau_reg, testis_clf}. Otherwise keep the frozen features and put the effort into the clinical and
multi-modal tracks.

---

## 5. LncBench

**Tasks** (all labels public; builders in `lncfm/tasks.py`, `orthology.py`, `variants.py`):

| id | what is predicted | label source | n (full) | metric |
|---|---|---|---|---|
| `loc_reg` | mean log2(cytosol/nucleus) over cell lines | ENCODE polyA+ fractionation, 15 human cell lines, gene "expressed" if cytosol+nucleus TPM > 2 in >= 2 lines. Cell-line split-half Spearman = 0.80, so the label is a stable gene property | 2,921 genes | Spearman |
| `loc_clf` | most-nuclear vs most-cytoplasmic tercile of the same | same | 1,948 | AUROC |
| `tau_reg` | tissue-specificity index tau over 54 GTEx tissues | GTEx v8 median TPM, lncRNAs with max >= 1 TPM (median tau 0.86) | 8,608 | Spearman |
| `testis_clf` / `brain_clf` | top tissue is testis (42.9% positive) / a brain region (15.9%) | same | 8,608 | AUROC |
| `coding_clf` | lncRNA vs protein-coding transcript | GENCODE (sanity check only) | 8,000, balanced | AUROC |
| `retrieval` | human lncRNA -> mouse positional ortholog | synteny from GENCODE annotations only; subsets all (17,971 queries) / strict 1:1 (2,823) / named same-symbol (56) | vs 36,103 mouse genes (pilot protocol: 300 queries vs 3,000) | MRR, hit@k |
| `variants` | pathogenic vs benign within a gene | ClinVar GRCh38 SNVs in small ncRNA disease genes (§6) | 233 SNVs in 8 genes (6 usable) | within-gene AUROC |

**Protocol (identical for every model):** one representative isoform per gene; frozen embeddings; 5-fold *leave-chromosome-group-out* CV (fold sizes range 15-31% of genes);
probe = logistic regression / ridge with strength chosen on an inner 20% split of the training folds; pooled out-of-fold predictions; bootstrap 95% CIs; **paired** bootstrap for every
model-vs-model claim (`scripts/compare.py`). Always report the untrained twin of any network you train.

**Whole transcripts, all genes** (the real bar; tiny model only, because it is the only one cheap enough to run unrestricted here) **[measured]**:

| model | loc_reg | loc_clf | tau_reg | testis_clf | brain_clf | coding_clf |
|---|---|---|---|---|---|---|
| length + GC only | 0.070 | 0.564 | 0.169 | 0.558 | 0.580 | 0.858 |
| **k-mer spectrum (k=1-5)** | 0.308 [0.27,0.35] | 0.707 [0.68,0.73] | 0.419 [0.40,0.44] | 0.667 [0.66,0.68] | 0.650 [0.63,0.67] | 0.935 [0.93,0.94] |
| untrained twin (0.89M) | 0.274 | 0.693 | 0.371 | 0.640 | 0.622 | 0.874 |
| pilot, MLM only | 0.290 | 0.717 | 0.390 | 0.666 | 0.638 | 0.839 |
| **pilot, all objectives** | 0.328 [0.29,0.36] | 0.721 [0.70,0.74] | 0.407 [0.39,0.43] | 0.668 [0.66,0.68] | 0.636 [0.62,0.65] | 0.871 [0.86,0.88] |

Paired, all-objectives pilot minus k-mer: loc_reg +0.020 [-0.011,+0.048], loc_clf +0.014 [-0.005,+0.033], tau -0.012 [-0.026,+0.002], testis +0.001, brain -0.015 [-0.026,-0.002], coding -0.064 [-0.072,-0.056].
Minus untrained twin: loc_reg +0.053, loc_clf +0.028, tau +0.036, testis +0.028, brain +0.013 (all CIs exclude 0), coding -0.002 (`results/paired_comparisons_fulllength.md`).
Note how strong the *untrained* network is (it sits within 0.03-0.06 of k-mers on every task): random convolutional features already encode k-mer statistics, so gains over "untrained" are the honest unit of progress.

**Matched protocol** (first 400 nt, n=800; the only setting where external models were affordable) **[measured]**: k-mer 0.270 / 0.697 / 0.360 / 0.576 / 0.586 / 0.812; untrained twin
0.155 / 0.648 / 0.255 / 0.591 / 0.605 / 0.765; pilot all-objectives 0.331 / 0.719 / 0.370 / 0.624 / 0.607 / 0.774; pilot MLM-only 0.288 / 0.730 / 0.322 / 0.584 / 0.564 / 0.748;
RiNALMo-micro 0.372 / 0.749 / 0.383 / 0.623 / 0.579 / 0.919 (columns as above; CIs in `results/cv_matched400.json`). Absolute numbers here are *lower* than the whole-transcript table because of less
data and shorter input; do not compare across tables.

**Learning curve** (matched protocol; probe noise is about +-0.03, so only the trend is meaningful) **[measured]**:

| step | all-obj loc_reg | all-obj tau_reg | all-obj testis | MLM-only loc_reg | MLM-only tau_reg | MLM-only testis |
|---|---|---|---|---|---|---|
| 500 | 0.202 | 0.323 | 0.599 | 0.210 | 0.310 | 0.600 |
| 1000 | 0.270 | 0.327 | 0.578 | 0.282 | 0.325 | 0.599 |
| 2000 | 0.325 | 0.363 | 0.596 | 0.277 | 0.324 | 0.599 |
| 3000 | 0.331 | 0.370 | 0.624 | 0.288 | 0.322 | 0.584 |
| untrained | 0.155 | 0.255 | 0.591 | 0.155 | 0.255 | 0.591 |

At step 1000 the extra objectives looked useless (and slightly harmful on loc_clf); by step 3000 they are ahead of MLM-only on most tasks. The lesson is not to judge auxiliary objectives early.
Final validation MLM loss: 1.260 (MLM-only) and 1.265 (all objectives) against ln 4 = 1.386 for chance: lncRNA sequence is high-entropy.

**Ortholog retrieval** (300 human queries vs 3,000 mouse genes, first 400 nt; hit@10; random = 0.009) **[measured]**:

| model | all (n=300) | strict 1:1 (n=54) | named (n=56) |
|---|---|---|---|
| length + GC | 0.007 | 0.000 | 0.018 |
| untrained twin | 0.027 | 0.037 | 0.071 |
| pilot MLM-only | 0.037 | 0.056 | 0.089 |
| pilot all-objectives | 0.030 | 0.037 | 0.125 |
| k-mer spectrum | 0.063 | 0.056 | 0.196 |
| RiNALMo-micro | 0.060 | 0.000 | 0.214 |
| **shared 10-mers (alignment-free)** | **0.123** | **0.074** | **0.375** |

At full scale (17,971 queries vs 36,103 genes, whole transcripts) shared 10-mers give hit@10 = 0.052 (all), 0.072 (strict), 0.411 (named), so ~59% of known same-symbol orthologs are invisible
to alignment-style similarity. That is the opportunity; no model here has captured it. Synteny labels are noisy (median 2 partners per human gene), and "named" pairs are biased toward well-studied genes.

**Variants** (within-gene AUROC, 192 SNVs in 6 genes with >= 3 variants per class; zero-shot masked-LM log-likelihood ratio) **[measured]**: untrained twin 0.531 [0.42,0.63],
pilot all-objectives 0.545 [0.43,0.62], pilot MLM-only 0.558 [0.46,0.64]. All indistinguishable from chance. The harness is there for bigger models (`python -m lncfm.variant_eval`).

---

## 6. Clinical track

### 6a. Expression -> survival (TCGA, GDC open access) **[measured]**

10-fold CV (2 repeats x 5 folds, event-stratified); every selection, PCA and scaler is fit inside the training fold; ridge-Cox with strength chosen by inner CV.
Numbers are C-index +- standard error across folds. `clin` = age (+ stage with a missing-indicator where >= 50% of patients have it; LGG has no stage in this extract, so it is age only).
"lncRNA" / "mRNA" = the 2,000 most variable genes of that class in the training fold.

| cohort (patients, deaths) | clin | lncRNA ridge | mRNA ridge | lncRNA PCA(64) | clin + lncRNA PCA | sequence-tied: random | k-mer | pilot all-obj |
|---|---|---|---|---|---|---|---|---|
| LGG (511, 125) | 0.740 +-0.013 | 0.839 +-0.008 | 0.839 +-0.011 | 0.825 +-0.011 | **0.848** +-0.010 | 0.773 | 0.766 | 0.797 |
| KIRC (529, 173) | 0.722 +-0.017 | 0.689 +-0.017 | 0.712 +-0.010 | 0.711 +-0.014 | **0.754** +-0.014 | 0.682 | 0.679 | 0.690 |
| LUAD (504, 182) | 0.646 +-0.019 | 0.603 +-0.014 | 0.602 +-0.021 | 0.623 +-0.019 | **0.664** +-0.018 | 0.602 | 0.578 | 0.605 |

What this says: (1) lncRNA expression carries about as much prognostic information as mRNA expression in all three cohorts. (2) Added to a clinical baseline it gives +0.032 (KIRC) and +0.018 (LUAD),
about the size of one standard error; I did not test significance, and there is no external validation cohort. (3) In LGG the lncRNA signal is large (0.84 vs 0.74) but largely proxies IDH/1p19q molecular
subtype, which is *not* in my clinical baseline; do not read it as an independent biomarker. (4) The sequence-tied model (patient = expression-weighted sum of 64-d gene embeddings, so 64 Cox parameters
instead of ~2,000) never beats the free-coefficient ridge model, and k-mer embeddings are no better than random ones.

**Does the sequence prior help when patients are few?** (`results/survival_low_n.txt`, 15 folds, training set subsampled) **[measured]**:

| cohort, patients used for fitting | free ridge | tied: random | tied: pilot | clinical |
|---|---|---|---|---|
| LGG 60 | 0.782 | 0.761 | 0.768 | 0.709 |
| LGG 120 | 0.798 | 0.783 | 0.782 | 0.741 |
| LGG 240 | 0.830 | 0.794 | 0.798 | 0.741 |
| KIRC 60 | 0.638 | 0.618 | 0.611 | 0.707 |
| KIRC 120 | 0.662 | 0.661 | 0.643 | 0.720 |
| KIRC 240 | 0.686 | 0.678 | 0.683 | 0.715 |

No. The tied model is never better than free ridge, and pilot embeddings are indistinguishable from random ones (differences within +-0.02, standard errors 0.01-0.02). Repeating with the strongest embedding available (frozen
RiNALMo-micro, first 400 nt, restricted to the 1,713 most variable lncRNAs, `results/survival_rinalmo_embeddings.txt`) also shows nothing: LGG 0.803 vs 0.799 random; KIRC 0.683 vs 0.698 random.
In KIRC with <= 240 patients the clinical baseline beats every lncRNA model. **Conclusion: at this scale, gene-sequence embeddings are not a useful prior for survival models.** The idea is not dead (a model trained
on function-relevant labels might encode something sequence-only models do not), but it is not where the evidence points today.

### 6b. Variant interpretation: what ClinVar can and cannot support **[measured]**

- Naively intersecting ClinVar with GENCODE lncRNA exons gives 5,290 pathogenic/likely-pathogenic SNVs, which is an artifact: the top "lncRNA" genes are TTN-AS1, MFF-DT, CELF6-AS1, PRPF31-AS1, SCN1A-AS1, antisense/divergent
  genes whose exons overlap protein-coding exons, so the labels are really TTN/PRPF31/SCN1A variants. A model could score well by recognising codons.
- Excluding any variant within 20 bp of a protein-coding exon leaves **50 P/LP SNVs (TERC alone supplies 30 of them) against 2,649 benign SNVs (dominated by KCNQ1OT1, ~700 variants counting indels; H19, XIST and MEG3 add ~45-50 each)**. A pooled AUROC would measure gene identity.
- The only defensible design is within-gene discrimination in genes with both classes. Small ncRNA disease genes provide it: RMRP 29 P / 9 B, RNU4ATAC 28/9, RNU4-2 21/8, RNU2-2 19/6, SNORD118 15/30, RNU7-1 5/13 (RNU12 7/2 and TERC 30/2 are too lopsided).
  233/233 genomic positions mapped to transcript coordinates with a matching reference base (coordinate logic verified). 192 variants in 6 genes are usable. That is a sanity check, not a benchmark.
- What would make variants a real track: saturation-mutagenesis / MPRA data on lncRNAs, curated lncRNA GWAS fine-mapping, or gnomAD-style constraint for lncRNA exons. None is wired in.

### 6c. What it would take to reach a clinical outcome

1. External validation cohorts for lncRNA signatures (GEO / ICGC / CPTAC) and a clinical baseline that includes the real prognostic variables (IDH, 1p/19q, grade, stage, performance status).
2. Cross-platform robustness (RNA-seq vs arrays vs targeted panels): lncRNAs are low-abundance and platform-sensitive, so this, not sequence priors, is likely the binding constraint.
3. A defined clinical question (e.g. add-on to stage in ccRCC) with a pre-specified endpoint, not a pan-cancer leaderboard.

---

## 7. Compute and hardware

**Counted training cost** (PyTorch `FlopCounterMode`, forward + backward, per nucleotide) **[measured]**:

| preset | params | MFLOP / nt | FLOPs for 1e9 nt |
|---|---|---|---|
| tiny | 0.89M | 1.41 | 1.4e15 |
| small | 6.67M | 10.4 | 1.0e16 |
| base | 38.8M | 58.9 | 5.9e16 |

**Corpus size.** Human + mouse lncRNA, all isoforms: 195.8 + 143.7 = 339.5 Mnt; unique at gene level only ~75 Mnt
(36k genes per species at ~1 kb). The 3,000-step pilot sees ~77 Mnt, i.e. **about one pass over the unique content**, so it is
far from converged and says little about the ceiling. 1e9 nt is ~13 passes over the gene-level content. **[measured / arithmetic]**

**Wall-clock on your M4 is not something I could measure** (the sandbox is a 4-core CPU box: ~32k nt/s per single-thread
process on the tiny model, i.e. ~45 GFLOP/s). Run `python -m lncfm.speedtest` and read "achieved TFLOP/s". The projection is then plain arithmetic:

| effective TFLOP/s on the M4 **[assumption, measure it]** | small, 1e9 nt | base, 1e9 nt |
|---|---|---|
| 0.3 | 9.6 h | 55 h |
| 1.0 | 2.9 h | 16 h |
| 3.0 | 1.0 h | 5.5 h |

**CUDA / Colab H100.** An H100 would train even `base` on 1e9 nt in minutes to tens of minutes (assuming 100-300 effective TFLOP/s for
a model this small; **[assumption]**). So a rented GPU is *not* needed for the from-scratch arm. It is needed for Evo 2 7B (CUDA +
flash-attn) or Evo 2 1B (Hopper + FP8) experiments. Check current Colab availability and pricing yourself; I did not.

**Where time actually goes locally:** the benchmark with external models (RiNALMo-micro ~0.5 s per 400-nt sequence on 2 CPU
threads here), TCGA downloads (~25 min per cohort through the GDC API), and ENCODE (57 files).

---

## 8. Run order

Exact commands are in `README.md`. The order, with a stop/go criterion per stage so no stage's compute is spent before the previous one earns it:

| stage | what | cost on the M4 (guess; measure with `speedtest`) | go if |
|---|---|---|---|
| 0 | `python -m lncfm.selfcheck` | seconds | all checks pass; **on MPS the device-parity line must pass** or stop and debug |
| 1 | downloads + `lncfm.bench --providers lengc kmer` (whole transcripts, all genes) | ~30-60 min, mostly network | you reproduce my k-mer numbers (loc_reg ~0.31, tau ~0.42, within CI) |
| 2 | frozen RiNALMo-micro (and RNA-FM, mega if you want) on the *same* whole-transcript benchmark, `--save-oof`, then `scripts/compare.py` | hours | it beats k-mers with a paired CI excluding 0 on >= 2 of {loc_reg, loc_clf, tau_reg, testis_clf}. If yes: it is your default feature extractor. |
| 3 | `small` (6.7M) pretraining, all objectives, >= 3e8 nt, checkpoints every 10k steps; plus the MLM-only twin if you want the ablation again | ~3-10 h at 0.3-1 TFLOP/s | whole-transcript paired CI vs k-mer excludes 0 on loc_reg **and** tau_reg, and it beats its untrained twin. If it only ties k-mers again, stop scaling and go to stage 4/5 |
| 4 | continual pretraining of RiNALMo (`lncfm.cpt`), then bench the saved `local:` model | hours | beats the frozen model per the decision rule in §4 |
| 5 | (optional, rented NVIDIA GPU) Evo 2 7B embeddings on LncBench | a few GPU-hours + setup | beats stage 2 features. This is the only large model with eukaryotic training data. |
| 6 | clinical: add external cohorts + real clinical baselines; stop using sequence-tied priors | days of data wrangling | a pre-specified endpoint and an external cohort exist |
| 7 | only now: `base` (39M) for a long run, `--include-heldout` for the production model | 1-3 days | stage 2 or 3 passed; otherwise it is wasted electricity |

Do not skip stage 2: it is the cheapest experiment and the one most likely to change the plan.

---

## 9. Threats to validity (read before trusting any number)

1. **Pilot scale.** 0.89M parameters, ~77 Mnt seen (about one pass); input clipped to 400 nt in the matched protocol (up to ~2 kb in the whole-transcript runs). A weak pilot result does not falsify the full-size
   plan; a good one does not prove it. The learning curve (§5) is the only evidence about scaling.
2. **Small n, many comparisons.** CV tasks use n=800 subsamples to match what external models can afford on CPU. I use paired bootstrap
   on identical items, but ~6 tasks x several contrasts are uncorrected: treat p between 0.01 and 0.05 as suggestive.
3. **Leakage controls are partial.** Chromosome-grouped folds stop neighbouring/duplicated loci straddling train and test; they do
   *not* stop paralogs or the same transposable-element family on different chromosomes. A near-duplicate-removed analysis was **not run**.
   Pretraining excludes the val/test chromosomes of human lncRNAs; mouse is fully included (no labels from mouse are used).
4. **Label caveats.** Localization comes from polyA+ RNA-seq only, gene-level labels are attached to one representative isoform, and
   expression level is a possible confounder that I did not regress out. GTEx tissue labels are dominated by testis (43% of expressed lncRNAs).
   Ortholog labels are synteny-defined (noisy); the 56 "named" pairs are the cleanest but are biased towards well-studied genes.
5. **Clinical track.** TCGA only, no external validation cohort. The clinical baseline is age + stage where stage exists (LGG: age only; gender
   came back empty from my GDC query and is not used), so it is weaker than a real clinical model: in LGG the lncRNA signal
   largely proxies IDH/1p19q molecular subtype, which I did not include as a baseline. C-indices are 10-fold CV means with SEs across folds.
6. **Untested paths.** MPS and CUDA code paths (parity check provided in `selfcheck`), continual pretraining beyond a 25-step smoke test,
   HyenaDNA / NT-v2 / Caduceus / Evo 2 (blocked or incompatible, see §4), and every Colab/H100 claim.
7. **Bugs found and fixed during this work** (so you know the checks exist): the conv blocks' GroupNorm made embeddings depend on batch
   padding (cosine 0.10 between the same sequence alone vs padded for a trained model; now 2e-7, regression-tested by `selfcheck`);
   TCGA censoring times ignored follow-up records (death rate inflated to 52% in KIRC; now 33%); the LUAD clinical baseline dropped stage.
8. **Licences.** RiNALMo and RNA-FM are AGPL-3.0, NT-v2 is CC-BY-NC-SA. Check these before any clinical or commercial use.
9. **Release pins.** GENCODE v50 / M39, GTEx v8 (GENCODE v26 IDs), ENCODE GRCh38 + GENCODE V29 quantifications, TCGA STAR counts (GENCODE v36).
   Genes whose Ensembl IDs do not exist in v50 are silently dropped when joining.


---

## 10. LncBench v2: two-layer benchmark (molecular + AML), measured

Code: `lncfm/lncbench.py` (runner), `splits.py`, `eclip.py`, `aml/`, `finetune.py`, `cnn.py`. Results: `results/v2/`.
Everything below was run in the CPU sandbox on **800 genes per gene-level task, first 400 nt, identical items for every model**
(RBP tasks at full size: 1,738 K562 / 1,396 HepG2 expressed lncRNAs). The numbers are real but small-sample; section 10.10 lists what that does and does not support.

### 10.1 What is in v2

| layer | task | label (all public lab or clinical data) | metric |
|---|---|---|---|
| 1 | `loc_reg`, `loc_clf` | ENCODE nuclear/cytosolic RNA-seq, 15 cell lines | Spearman / AUROC |
| 1 | `tau_reg`, `testis_clf`, `brain_clf` | GTEx v8 tissue specificity | Spearman / AUROC |
| 1 | `rbp_K562` (43 RBPs), `rbp_HepG2` (35 RBPs) | ENCODE eCLIP IDR peaks overlapping the transcript's exons; negatives = expressed lncRNAs without peaks | macro AUROC **stratified by expression x length** |
| 1 sanity | `coding_clf` | GENCODE | AUROC (see 10.3: solved by length alone) |
| 2b | `aml_vs_normal_reg`, `_resid_reg`, `_resid_clf` | AML (TARGET, n=300) vs normal marrow (n=300), label reliability (split-half) 0.97-0.98; `_resid_` removes what mean expression + length explain | Spearman / AUROC |
| 2b | `aml_npm1_reg` | NPM1-mutant vs wild-type, BeatAML + TCGA-LAML combined; cross-cohort reliability 0.58; top hits are the known HOX-antisense lncRNAs (HOXB-AS3, HOXA-AS2/3, HOXB-AS2, BAALC-AS1) | Spearman |
| 2a | `npm1`, `flt3_itd`, `eln_adverse`, `induction_cr`, `os` | BeatAML (adult, initial diagnosis, n = 273-345) -> external TCGA-LAML (adult, 151) and TARGET (paediatric, 300) | AUROC / C-index |

**Splits and controls.** Dev = leave-chromosome-group-out 5-fold over non-test chromosomes (fold sizes balanced to 470-559 on localization). **Locked test = chr8/13/18/21**, scored only with `--final`.
Homology: pairs of lncRNAs sharing >= 40% of the shorter transcript's 10-mers are flagged; the threshold comes from the measured edge distribution (25% flags 77% of genes, 40% flags 14%, 50% flags 7%, i.e. 25% was catching Alu/L1 sharing, not paralogy).
Only ~1% of the *expressed* genes used in these tasks are flagged, so pruning barely moves any number; it is reported anyway (`dev_all` vs `dev_pruned`).
Every comparison includes a **sequence-free baseline** (`expr_len`: expression and/or length only), length+GC, k-mer spectra, a one-hot CNN, and my pilot model; paired bootstrap for model-vs-model claims.

**Data facts checked while building** (live APIs): ENCODE has 252 eCLIP experiments for 168 RBPs (K562 145, HepG2 105); BeatAML has 653 patients with open RNA-seq in the GDC, 632 of which link to cBioPortal clinical records
(ELN-2017, NPM1, FLT3-ITD, induction response, survival); TCGA-LAML 151; TARGET 2,162 (899 in the curated study with outcomes and censoring). BeatAML drug-response data (Vizome) is not reachable from here.

### 10.2 Layer 1: dev results on homology-pruned items (higher is better; CIs in `results/v2/report_dev_pruned.md`)

| model | loc_reg | loc_clf | tau_reg | testis_clf | brain_clf | rbp_K562 | rbp_HepG2 | coding_clf |
|---|---|---|---|---|---|---|---|---|
| expression/length only | 0.110 | 0.554 | 0.090 | 0.513 | 0.521 | 0.470 | 0.459 | **0.892** |
| length + GC | 0.126 | 0.602 | 0.119 | 0.479 | 0.522 | 0.491 | 0.505 | 0.736 |
| k-mer spectrum | 0.283 | 0.756 | 0.306 | 0.575 | 0.590 | 0.565 | 0.579 | 0.816 |
| one-hot CNN (trained on ~700 genes) | 0.187 | 0.659 | 0.097 | 0.498 | 0.554 | 0.534 | 0.535 | 0.805 |
| my 0.89M pilot (frozen) | 0.203 | 0.779 | 0.315 | 0.573 | 0.606 | 0.587 | 0.598 | 0.789 |
| **RiNALMo-micro (frozen)** | **0.315** | **0.796** | 0.292 | **0.615** | 0.522 | **0.619** | **0.616** | **0.921** |

Paired differences on identical items (`results/v2/paired_comparisons.md`):

| contrast | loc_reg | loc_clf | tau_reg | testis_clf | brain_clf | coding_clf | rbp_K562 (over 43 RBPs) | rbp_HepG2 (over 35 RBPs) |
|---|---|---|---|---|---|---|---|---|
| RiNALMo - k-mer | +0.034 [-0.007,+0.078] | **+0.040** [+0.018,+0.063] | -0.012 [-0.062,+0.033] | **+0.042** [+0.001,+0.084] | **-0.065** [-0.133,-0.002] | **+0.105** [+0.074,+0.136] | **+0.055** [+0.033,+0.078], 34/43 RBPs | **+0.038** [+0.021,+0.055], 26/35 |
| pilot - k-mer | **-0.079** [-0.145,-0.009] | +0.023 [-0.004,+0.051] | +0.007 [-0.036,+0.049] | -0.004 [-0.054,+0.043] | +0.017 [-0.036,+0.066] | -0.027 [-0.056,0.000] | +0.023 [-0.001,+0.046] | **+0.020** [+0.003,+0.038] |
| RiNALMo - pilot | **+0.112** [+0.055,+0.170] | +0.017 [-0.007,+0.044] | -0.019 [-0.065,+0.027] | +0.046 [-0.002,+0.093] | **-0.082** [-0.150,-0.016] | **+0.132** [+0.101,+0.161] | **+0.032** [+0.011,+0.054] | **+0.018** [+0.003,+0.035] |

What this shows:
1. **The benchmark separates models, which is its job.** RBP binding and localization spread the models out with paired intervals that exclude 0; tissue specificity and the AML tasks (10.4) do not.
2. **RBP binding has real sequence signal beyond its confounds.** Within expression x length strata, the sequence-free baseline is at chance (0.47), length+GC is at chance (0.49-0.50), and k-mers reach 0.57-0.58 (k-mer minus length+GC = +0.074 on both cell lines, 33/43 and 33/35 RBPs). The expression-only baseline's *unstratified* macro-AUROC is 0.559 (K562) and 0.549 (HepG2) and falls to 0.472 / 0.468 once stratified, which is exactly the confound the stratification removes; the sequence models barely move (k-mer 0.567 -> 0.565 on K562).
3. **Frozen RiNALMo-micro is the best single model on RBP (+0.04 to +0.055 over k-mers), localization classification and the testis task**, ties on regression localization and tissue specificity, and is **worse on brain-specificity** (-0.065). It is not a uniform win.
4. **My 0.89M pilot is roughly k-mer level**: small gains on RBP, a loss on localization regression at 400 nt (it tied k-mers on whole transcripts in v1, so input length matters for it). It is far below RiNALMo on RBP, localization regression and the sanity task.
5. **A CNN trained from scratch on ~700 genes is weak** (0.10-0.19 where k-mers get 0.28-0.31). That is expected at this size and is why the CNN row must be re-run on the full data on your machine (10.9).
6. **`coding_clf` is not a test.** Transcript length alone gives AUROC 0.892; it is a pipeline check.

### 10.3 Layer 2b: which lncRNAs are leukaemia-associated (gene level, `results/v2/report_dev_pruned.md`)

| model | aml_vs_normal_reg (raw) | aml_vs_normal_resid_reg | aml_vs_normal_resid_clf | aml_npm1_reg |
|---|---|---|---|---|
| expression/length only | **0.274** | -0.077 | 0.568 | 0.028 |
| length + GC | 0.050 | 0.091 | 0.538 | -0.071 |
| k-mer | 0.239 | 0.235 | 0.634 | -0.021 |
| CNN | 0.138 | 0.148 | 0.559 | -0.009 |
| my pilot | 0.276 | 0.235 | 0.595 | -0.098 |
| RiNALMo-micro | 0.256 | 0.254 | 0.624 | -0.015 |

- The **raw fold-change label is confounded**: expression alone gives rho 0.274, as good as any sequence model, which is why the expression-corrected `_resid_` labels exist. On those the sequence-free baseline falls to ~0 and sequence models reach rho 0.24-0.25 / AUROC 0.60-0.63.
  The sequence models are tied within noise on the regression label (RiNALMo - k-mer: +0.018 [-0.025,+0.063]; pilot - k-mer: -0.002 [-0.053,+0.046]); on the tercile classification the pilot is slightly lower (-0.038 [-0.079,+0.001]). So: **sequence carries modest information about which lncRNAs are leukaemia-up-regulated beyond expression, and a k-mer spectrum already captures it.**
- **No model predicts which lncRNAs are NPM1-associated** (all rho within -0.10..0.03), even though the labels are reliable (0.58 cross-cohort). The HOX-antisense lncRNAs behind that signal are regulated by genomic context, not by their transcript sequence. This is a useful "limits of sequence" task, not a task a better sequence model should be expected to win.

### 10.4 Layer 2a: patient-level AML (`results/v2/aml_patient.txt`)

BeatAML (adult, initial diagnosis) 5-fold CV x3; external cohorts are scored with the model fit on all of BeatAML, genes z-scored within each cohort, top-1,000 genes by raw variance. Mean +- SE across folds; external = value [95% CI over patients].

| task (n) | age only | lncRNA | mRNA | lncRNA + age |
|---|---|---|---|---|
| `npm1` (345; 109 mutated) CV AUROC | 0.562 | **0.987** | 0.987 | 0.986 |
| ...external TCGA-LAML (151; 38 mutated) | 0.475 | **0.966** [0.92,1.00] | 0.966 | 0.966 |
| `flt3_itd` (345; 92) CV AUROC | 0.437 | 0.888 | 0.908 | 0.887 |
| `eln_adverse` vs favourable (252) CV AUROC | 0.649 | 0.989 | 0.988 | 0.989 |
| `induction_cr` (273; 199 CR) CV AUROC | 0.572 | 0.665 | 0.701 | 0.665 |
| ...external TARGET (paediatric, 297) | 0.480 | 0.630 [0.56,0.70] | 0.653 | 0.630 |
| `os` (342; 161 deaths) CV C-index | 0.699 (age + ELN: 0.710) | 0.717 | 0.715 | **0.730** (+ELN 0.729) |
| ...external TCGA-LAML OS (130) | **0.669** [0.60,0.73] | 0.608 [0.54,0.67] | 0.635 | 0.619 |
| ...external TARGET OS (300) | 0.545 [0.48,0.60] | **0.619** [0.57,0.68] | 0.638 | 0.621 |

Reading it honestly:
- **lncRNA expression carries essentially the same information as mRNA expression** for every task (differences <= 0.04), and reads out genotype very well: NPM1 0.987 in BeatAML and **0.966 in an independent adult cohort**. ELN adverse-vs-favourable at 0.989 is largely the same fact (ELN is built from the mutations the transcriptome reveals), so it shows readout of the genotype, not new prognostic information.
- **Prognosis is the weak part.** Within BeatAML, lncRNA + age is 0.730 against 0.710 for age + ELN (+0.020; SE ~0.01, not tested for significance). On external cohorts lncRNA expression does *not* beat age in adults (TCGA 0.608-0.619 vs 0.669) but does in children (TARGET 0.62 vs 0.545). Treatment era, cohort, and transplant differ across these sets and are not modelled.
- The sequence-embedding "tied" feature set exists (`--emb`) but was **not run in v2**; in the solid-tumour experiments of §6 it was a null.
- These are n = 150-350 problems; treat the third decimal as noise.

### 10.5 The Marcucci / Garzon lncRNA paper (GSE63646)
I could not find an NEJM paper by Marcucci on lncRNAs; the match is Garzon et al., PNAS 2014 (PMID 25512507; Marcucci co-author). Its public GEO series GSE63646 is the **RNA-seq component only**: 71 samples, 8,702 lncRNAs by symbol, age for 48, ELN group for 37, **no survival or response data**.
The 148-patient prognostic cohort is not in that series. So it cannot be a prognosis benchmark. It can be a small independent expression set to cross-check the direction of the 2b labels, and its published 48-lncRNA signature a literature case study. Neither is wired in.

### 10.6 Adaptation ladder smoke test (`results/v2/ladder_smoke.txt`)
RiNALMo-micro, `loc_reg`, 500 genes, 256 nt, fold 0 only (95 test genes), 3 epochs, CPU: frozen 0.429, LoRA 0.428, top-2 layers 0.449, full 0.423; paired differences vs frozen all have 95% intervals of about +-0.13. **This validates the code, not the ranking.**
It also measured cost on CPU (2 contended threads, batch 8 x 256 nt): LoRA 184,801 trainable parameters at 6.6 s/step, top-2 layers 5.5M at 4.2 s/step, full 33.5M at 4.8 s/step. **LoRA was not faster than full fine-tuning**: it saves optimizer memory, not backward-pass compute. On a 33M model that memory is not the constraint, so LoRA is only worth it if it generalises better, which the real run on 2,500+ genes and 5 folds will show.

### 10.7 Bugs found and fixed while building v2 (the checks that now exist)
- AML expression matrix and sample table were in **different row orders**, so every patient label was effectively shuffled: NPM1 prediction looked like chance and NPM1-associated lncRNAs looked unreproducible (cross-cohort rho 0.10). After aligning by ID: AUROC 0.987 and rho 0.58. `load_cohort` now asserts alignment.
- Top-variable-gene selection ran on z-scored data (every gene has variance 1), i.e. random. Now uses raw-expression variance.
- eCLIP peak files are gzip-compressed under a `.bed` name; homology edges at 25% containment flagged 77% of genes (now 40%, 14%); two quadratic gene-matching loops; a test-scope bookkeeping bug; a TARGET cohort whose sampled patients mostly lacked outcome data.
- RBP labels are confounded by expression **and transcript length** (at full-length input, length+GC alone reached macro-AUROC 0.64 before stratifying on both; in the 400-nt tables length is hidden from the sequence models, which is why length+GC reads ~0.50 there). The benchmark now reports expression x length stratified AUROC and flags tasks with too few positives as underpowered.

### 10.8 What the locked test says
The locked test was scored once per model (`results/v2/report_test_pruned.md`). With ~100 genes per gene-level task (80 for AML, 90/78 for RBP, ~1-9 scoreable RBPs) its intervals are +-0.15, so it can only catch gross overfitting, and it does not contradict the dev results: e.g. k-mer 0.430 [0.26,0.56] vs dev 0.283 on `loc_reg`, RiNALMo 0.785 vs dev 0.796 on `loc_clf`. On the RBP tasks the test set holds a median of ~2 positives per RBP, so those rows are marked underpowered and should be ignored; use the dev macro-average.
(Honesty note: smoke runs of the *baselines* during development printed locked-test numbers; no model, threshold or design choice was made from them.)

### 10.9 What v2 adds to the model question
- **Build on RiNALMo, and aim the adaptation at RBP binding and localization.** That is where a frozen RiNALMo already beats k-mers with paired intervals excluding zero and where the benchmark has headroom (RBP 0.62 vs a chance level of 0.47). Tissue specificity, brain, and the AML gene-level tasks are not where an adapted model will show gains (k-mers already tie; some are label-limited).
- **Open question the M4 run answers:** does adapting RiNALMo (LoRA / top layers / full / continual pretraining) beat the frozen probe on RBP and localization on the full data? The smoke test could not tell.
- **Clinical layer:** lncRNA expression is informative for AML genotype and weakly for survival, and sequence models predict leukaemia-associated lncRNAs to a modest, k-mer-saturated degree. The FM's clinical value, if any, will have to come from a task where sequence knowledge matters (e.g. variant effect, binding), not from patient-level expression.

### 10.10 Limits of this sandbox run (read before trusting any number)
- 800 genes per gene-level task, first 400 nt, CPU only, 4 cores. CIs are wide; the paired intervals above are the reliable statements, point estimates are not.
- The pilot is a 0.89M model trained for one pass; it is a reference, not a candidate. The CNN was trained on ~700 genes and is under-powered.
- AML cohorts here: BeatAML primary specimens only (346 of 653 patients); TARGET 300 patients (all with outcomes); TCGA-LAML 151. The M4 script downloads more. OS time origin and treatment heterogeneity are not modelled; external survival uses age as the only shared covariate.
- Clipping to 400 nt hides transcript length from every sequence model, while the sequence-free `expr_len` baseline sees the true length; that is deliberate for matched input but it changes what length can explain (see the RBP note in 10.7).
- Cross-cohort normalisation is a per-cohort gene z-score; no batch-correction method was compared.
- MPS/CUDA paths and the full ladder were not run. Locked-test numbers are too noisy to rank models.

### 10.11 Run it on your M4
```bash
export LNCFM_DATA=$HOME/lncfm_data LNCFM_RUNS=$HOME/lncfm_runs
scripts/m4_run_all.sh check     # selfcheck (CPU-vs-MPS parity) + measured speed: do this first
scripts/m4_run_all.sh data      # downloads; BeatAML/TARGET RNA-seq are the slow part (set --target-aml / --beat higher for the full cohorts)
scripts/m4_run_all.sh quick     # reproduce the tables above (800 genes, 400 nt)  ~1-2 h
scripts/m4_run_all.sh full      # full-size tasks, 1000-nt input, locked test, CNN on all data
scripts/m4_run_all.sh ladder    # frozen / LoRA / top-2 / full on loc_reg, tau_reg, rbp_K562, 5 folds
scripts/m4_run_all.sh aml       # patient layer incl. sequence-tied embeddings (--emb random kmer rinalmo-micro)
python -m lncfm.lncbench report $LNCFM_RUNS/v2/full_*.json --scope dev_pruned
python -m lncfm.lncbench compare "$LNCFM_RUNS/v2/full_rinalmo.json:rinalmo" "$LNCFM_RUNS/v2/full_baselines.json:kmer"
```
Go / no-go for building on RiNALMo: on the full data, an adapted model must beat the **frozen** RiNALMo probe on `rbp_K562` and `rbp_HepG2` (paired CI over RBPs excluding 0) and on `loc_clf`; otherwise keep the frozen features and spend the effort on the clinical layer and new labels (m6A, half-life, in-cell structure).

---

## 11. Yield upgrades, the AML genotype panel, and whether to release a model family

Everything in this section was run in the same CPU sandbox as section 10 (600-1,700 genes per task, first 400 nt, 200 bootstrap resamples). Files: `results/v3/` and `results/v2/aml_genotype.*`.

### 11.1 What was added to make the full run more informative
| addition | where | verified |
|---|---|---|
| `rbp_union`: K562 + HepG2 eCLIP pooled (RBPs assayed in both; label = peak in any line where the gene is expressed) | `lncbench.py`, `eclip.build_union_table` | 2,196 genes, 38 RBPs, median 32 positives per RBP (K562 alone: 1,738 genes, 43 RBPs) |
| AML **genotype panel**: 14 driver genes, lncRNA vs mRNA vs age, BeatAML CV + external TCGA-LAML | `aml/genotype.py`, stage `genotype` | scored, section 11.3 |
| frozen-feature sweep: layer subset x pooling x multi-window tiling x k-mer stacking | `embed.py`, `lncbench.parse_variant`, stage `sweep` | scored, section 11.2 |
| more foundation models: RNA-FM (99.5M), RiNALMo-mega (148M) | stage `fms` | **load and embed only** (0.30 and 0.66 s/sequence on CPU; all-layer pass equals the plain pass, cosine 1.0000). Not benchmarked. |
| Benjamini-Hochberg column in `compare` | `lncbench.compare` | run on existing paired tables |
| ortholog retrieval stage | stage `retrieval` | v1 code, unchanged |

`rbp_union` scored (RiNALMo-micro, 400 nt, `results/v3/rbp_union.txt`): macro-AUROC stratified by expression and length **0.630 vs 0.586 for k-mers**; the sequence-free `expr_len` baseline falls to 0.501 under that stratification (its raw 0.633 is entirely expression + length). Paired CIs were not computed for this run; the K562/HepG2 versions (section 10.2) are the ones with paired intervals.

### 11.2 The feature sweep did **not** produce a better recipe
RiNALMo-micro, 600 genes per gene-level task and the full K562 RBP task, 400 nt, dev scope. Gene-task intervals are about +-0.04 to +-0.08 and overlap across every row; the K562 RBP interval is about +-0.03 (e.g. last layer 0.620 [0.586,0.653]), so none of the differences between feature recipes is established.

| task | last layer, mean | mid layer, mean | mid, mean+max | layers 4,8,last mean+max | k-mer + mid mean+max (stacked) | k-mer |
|---|---|---|---|---|---|---|
| loc_reg (rho) | 0.383 | 0.391 | 0.393 | 0.391 | 0.379 | 0.336 |
| loc_clf (AUROC) | **0.761** | 0.748 | 0.747 | 0.752 | 0.745 | 0.709 |
| tau_reg (rho) | 0.329 | **0.371** | **0.374** | 0.339 | 0.357 | 0.323 |
| testis_clf (AUROC) | **0.633** | 0.589 | 0.592 | 0.606 | 0.621 | 0.617 |
| rbp_K562 (expr+len-strat. macro-AUROC) | 0.620 | 0.582 | 0.593 | **0.635** | 0.599 | 0.565 |

- No configuration wins across tasks. The middle layer helps tau_reg (+0.04) and hurts testis and RBP; the last layer is best on testis and loc_clf; the three-layer mean+max is best on RBP (+0.015 over last layer, inside the noise).
- Stacking k-mer features on top of RiNALMo never beat RiNALMo alone: whatever the k-mers know, the language model already has.
- **Recommendation: keep last-layer mean pooling as the default.** If you want layer choice, treat it as a hyperparameter selected inside the training folds on the full run, never from the table above (that would be choosing on the evaluation data). The `sweep` stage exists to measure this properly on the full data; I would not expect it to change the conclusions.

### 11.3 AML genotype panel (`results/v2/aml_genotype.txt`)
Per-gene 5-fold CV AUROC for "is gene X mutated" from expression of the 1,000 most variable lncRNAs, the 1,000 most variable mRNAs, or age, BeatAML (n=346) with external TCGA-LAML (n=151, PanCancer Atlas mutation calls) where there are >= 8 mutated patients.

| | age | lncRNA | mRNA |
|---|---|---|---|
| 14 genes, BeatAML CV, mean AUROC | 0.633 | **0.871** | 0.877 |
| 11 genes, TCGA-LAML external, mean AUROC | n/a | **0.841** | 0.871 |
| lncRNA minus mRNA, CV (per-gene range) | | -0.006 (-0.034 to +0.027) | |

Highlights (CV / external): NPM1 0.984 / 0.967; TP53 0.975 / 0.996; SRSF2 0.969 (no external events); IDH2 0.929 / 0.882; RUNX1 0.920 / 0.951; CEBPA 0.889 / 0.871. Weakest: NRAS 0.687 / 0.642, ASXL1 0.795.

How to read it: the lncRNA transcriptome reads out the AML molecular subtype about as well as the mRNA transcriptome, replicated externally. It does **not** say a lncRNA is special, and it is not evidence that these lncRNAs are functional. Driver mutations co-occur (NPM1 with DNMT3A and FLT3; SRSF2, RUNX1 and ASXL1 define a secondary-type AML signature), so a high score for one gene partly reflects its co-mutated partners, and the panel measures subtype recognition, not gene-specific biology. As a benchmark item it is robust and cheap, but a sequence model can be scored on it only through the sequence-tied embedding route (`--emb`), which was a null in the solid-tumour experiments.

### 11.4 Data that would add real yield but is **not** built (not verified, so not claimed)
- **m6A, RNA half-life, translation / ribosome-association labels** for lncRNAs: I have not found and checked a public, versioned source with lncRNA-level labels that I would trust, so no task was written.
- **In-cell structure**: ENCODE lists 6 icSHAPE experiments; not downloaded or parsed.
- **Task-specific published predictors** (for RBP binding and for subcellular localization) as extra baselines: not run. A reviewer will ask for at least one; they have to be run by hand and compared on the same folds.
- **Token-level probing** (can the model find exon junctions / splice-aware positions from sequence alone?): the exon-junction head exists in pretraining, but no probe benchmark was written.
- **Seeds**: not needed for the probes (outer folds are fixed, probes are deterministic); needed for any trained-from-scratch or fine-tuned arm.

### 11.5 Should we train and release a small family? A gated plan
Short answer: **yes as a plan, not as a commitment, and only members that clear a pre-stated gate get released.** Three facts decide this:
1. The 0.89M from-scratch pilot was about equal to k-mers (section 10), so "a bigger one will be good" is a hypothesis, not a result.
2. Human lncRNA is small data for a language model: ~195.8 Mb of transcript sequence, 191k transcripts of which far fewer are independent after de-duplicating isoforms and repeats (roughly 75 Mb unique per species by my count). A 39M model on that will see each nucleotide many times.
3. A frozen RiNALMo-micro already beats k-mers on RBP binding and localization with paired CIs excluding zero. A new model has to beat **that**, not k-mers.

| member | what it is | cost (estimates from counted FLOPs; run `speedtest`) | release gate |
|---|---|---|---|
| **A: RiNALMo-lnc** | RiNALMo-micro (33.5M) after continual pretraining on lncRNA (`cpt.py`, span-MLM) and/or LoRA / top-2 layers, with a lightweight multi-task head | hours to ~1 day on the M4 | beats the **frozen** probe on `rbp_K562`, `rbp_HepG2` and `loc_clf` (paired CI excludes 0) and is not worse elsewhere. Inherits AGPL-3.0, so the weights must be released under it. |
| **B: LncFM-small** | 6.7M, from scratch: conv stem + RoPE transformer, span-MLM + exon-junction + isoform contrastive | 1e9 nt of training ~ 9.6 h at 0.3 TFLOP/s, 2.9 h at 1.0 TFLOP/s | >= k-mers on full-length inputs with a paired CI **and** within noise of frozen RiNALMo on >= 2 tasks. Else it is a negative result, not a release. |
| **C: LncFM-base** | 39M, same recipe | ~55 h at 0.3 TFLOP/s, ~16 h at 1.0 | only trained if small beats tiny (0.89M) by a clear margin on held-out-chromosome MLM loss **and** on the benchmark. If tiny -> small shows no gain, do not train base. |

Order of work (cheapest first, each step can stop the next): adaptation ladder -> continual pretraining (A) -> tiny vs small on the same batches (B) -> base (C).

What the family would accomplish if the gates are met:
- A controlled **scaling and objective study** for lncRNA (0.9M -> 6.7M -> 39M, same data, same batches, with and without junction / contrastive heads). Nobody in this niche has published one; it is reportable whichever way it comes out, including "scale does not help on this much data".
- **Small CPU-friendly models** others can run on a laptop, with the benchmark as the credibility anchor.
- A set of **fixed reference points** on LncBench so later models are compared against something other than k-mers.

What it would not accomplish: it would not be a clinical tool, and "foundation model" is an overclaim unless transfer to several unrelated tasks is shown. I would call the artefacts "lncRNA language models" in the first release and let the numbers justify more.

Release hygiene (must be decided before training, not after):
- Pretraining excludes human chr8/13/18/21 (locked test) and chr9/16/22 (validation) by default. A released model keeps those exclusions and says so; otherwise the benchmark can no longer be used on it.
- Model card with: data release pins (GENCODE v50, M39), exclusions, license, the exact benchmark table with CIs, and the homology caveat (a held-out chromosome is not a homology-held-out gene; the `leak` flag and pruned scope exist for this).
- Maintenance: Hugging Face-compatible weights plus the `multimolecule`-style loading path, tests on CPU/MPS, and a pinned benchmark version.

---

## 12. Variant-effect scoring in ncRNA genes

Question: can a sequence model rank which single-nucleotide variants in a non-coding RNA gene are pathogenic? This replaces the v1 attempt (§6, pilot model, AUROC 0.55 on 8 genes). Code: `lncfm/ncvar.py` (data), `gnomad.py` (extra benign set), `varscore.py` (scorers, evaluation, VCF scorer, saturation map). Results: `results/v4/`.

### 12.1 The data, and what is wrong with "ClinVar lncRNA variants"
ClinVar (file dated 2026-10-19) has 1,537,469 SNVs classed pathogenic / likely pathogenic (P/LP) or benign / likely benign (B/LB). Keeping only those that fall in exons of a non-coding gene, that ClinVar attributes to *that* gene, and that carry no coding consequence (missense, nonsense, splice site, ...) leaves **4,678 SNV-gene pairs: 220 P/LP and 4,458 B/LB in 583 genes**:

| gene type | SNVs | P/LP | comment |
|---|---|---|---|
| lncRNA | 3,670 | 31 | **30 of the 31 are TERC**; 17 more P/LP records were dropped as belonging to an overlapping coding gene (see below) |
| Mt_tRNA | 363 | 52 | the richest source: 22 mitochondrial tRNA genes |
| snRNA | 120 | 81 | RNU4ATAC, RNU4-2, RNU2-2, RNU12, RNU7-1 |
| ribozyme | 38 | 29 | RMRP |
| snoRNA | 52 | 15 | SNORD118 |
| miRNA / Mt_rRNA | 365 / 70 | 9 / 3 | too few per gene |

Three things that matter for any claim:
1. **There is essentially no lncRNA ground truth.** TERC (30 P/LP, 2 B/LB) is the only lncRNA with a real disease burden, and with 2 benign variants it cannot support a within-gene test (it enters only with the "any gnomAD-observed variant" background, which was not scored here). Before the attribution filter there were 48 lncRNA P/LP records; 17 of them belong to the neighbouring coding gene (LIX1L-AS1 is RBM8A / TAR syndrome, HNRNPK-AS1 is HNRNPK / Au-Kline, DSP-AS1, SCN1A-AS1, ...). Everything below is therefore about **small structured ncRNAs (snRNA, ribozyme, snoRNA, mitochondrial tRNA), not lncRNAs**, and nothing here shows the score works on lncRNAs.
2. **Why the set is larger and cleaner than v1's.** v1 scored a hand-picked list of 8 genes, which could not include the 22 mitochondrial tRNA genes (the largest source; ClinVar names the contig `MT`, GENCODE `chrM`). While building the new mapping, two traps appeared and were fixed: excluding everything within 20 bp of a protein-coding exon removed RNU2-2 (it sits inside the 5' UTR exon of WDR74), so ClinVar's own molecular-consequence field is used instead; and TERC variants were being assigned to an antisense lncRNA on the opposite strand at the same locus, so ClinVar's gene attribution is now required. The final table checks the reference base of every variant against the transcript (0 mismatches in 4,678), and the transcript-offset mapping is tested to invert the genomic one on both strands (`varscore check`).
3. **Benign examples are scarce**, so common population variants are added as presumed benign: gnomAD v4 (nuclear) or v3.1 (mitochondrial) variants with allele frequency >= 1e-4 that ClinVar has not classified (279 variants). Every analysis is repeated with ClinVar-only benign variants and with a stricter frequency (1e-3).

One rule was added after looking: for lncRNA genes only, a pathogenic record that ClinVar also attributes to a protein-coding gene is dropped unless the disease name mentions the lncRNA (17 records; removes LIX1L-AS1 and HNRNPK-AS1 from the panel). A first version applied to every gene type wrongly removed RNU4-2 and most of SNORD118, whose disease names do not contain the gene symbol, so it was restricted. Effect on the headline: 0.712 [0.606,0.818] on 15 genes before, 0.721 [0.611,0.833] on 13 genes after (`results/v4/first_run_before_lncrna_rule_15genes.txt`).

### 12.2 Evaluation (fixed before the numbers below were seen, apart from the rule above)
Within-gene AUROC: for each gene with >= 3 pathogenic and >= 3 benign variants, how well does the score separate them inside that gene; mean over genes weighted by min(n_path, n_benign). Pooled AUROC is not used because gene identity alone separates the classes. 95% CIs resample genes and then variants within gene. **Panel: 13 genes, 585 variants (158 P/LP): 7 nuclear genes (RNU2-2, SNORD118, RNU4ATAC, RNU12, RNU4-2, RMRP, RNU7-1) and 6 mitochondrial tRNA genes (MT-TK, -TL1, -TW, -TS1, -TV, -TN).** All scores are oriented "higher = more damaging" and never flipped afterwards. Primary score: RiNALMo-micro masked log-odds (log p(ref) - log p(alt) with the variant position masked), frozen, zero-shot. Everything else is a comparison or an exploratory add-on.

### 12.3 Results (`results/v4/final_main.txt`)
Within-gene AUROC [95% CI]. Benign = ClinVar B/LB + gnomAD common.

| score | all 13 genes | 7 nuclear genes | 6 mito tRNA genes |
|---|---|---|---|
| **RiNALMo-micro, masked log-odds (primary)** | **0.721** [0.611,0.833] | 0.687 [0.565,0.805] | **0.841** [0.668,0.946] |
| same, position unmasked | 0.707 | 0.674 | 0.824 |
| embedding distance (+-8 nt) | 0.571 | 0.573 | 0.565 |
| ViennaRNA change in ensemble free energy | 0.523 | 0.470 | 0.711 |
| transversion vs transition (no model) | 0.587 [0.505,0.669] | 0.629 [0.540,0.715] | 0.438 |
| CADD v1.7 (public API; nuclear only) | n/a | 0.684 [0.554,0.810] | n/a |
| combiner: LM + structure + baselines | 0.748 [0.662,0.824] | 0.739 [0.643,0.827] | 0.777 |
| **combiner + CADD** (leave-one-gene-out) | **0.826** [0.768,0.876] | **0.843** [0.792,0.892] | 0.765 [0.634,0.885] |

Robustness of the primary score to how the benign set is built (all genes; nuclear in brackets): common gnomAD AF >= 1e-4 **0.721** (0.687); AF >= 1e-3 0.757 (0.721); ClinVar B/LB only, 12 genes, 0.738 (0.675); ClinVar >= 1 star only 0.733 (0.700). CADD on the nuclear genes in the same four designs: 0.684, 0.779, 0.825, 0.688.

Per gene (primary LM score / CADD / transversion): RNU4-2 **0.885** / 0.571 / 0.615; RNU2-2 **0.837** / 0.520 / 0.717; SNORD118 0.644 / 0.726 / 0.737; RNU12 0.658 / 0.413 / 0.652; RNU4ATAC 0.590 / **0.748** / 0.627; RNU7-1 0.393 / **0.880** / 0.483; RMRP 0.579 / **0.906** / 0.495; mitochondrial: MT-TL1 **0.972**, MT-TN 0.944, MT-TS1 0.843, MT-TV 0.795, MT-TW 0.650, MT-TK 0.648 (no CADD, transversion 0.39-0.63).

### 12.3b Other pretrained models on the same variants (`results/v4/final_fms.txt`)
Frozen, zero-shot masked log-odds, same 585 variants:

| model | all 13 genes | 7 nuclear | 6 mito tRNA |
|---|---|---|---|
| RiNALMo-micro (33M) | 0.721 [0.611,0.833] | 0.687 [0.565,0.805] | 0.841 [0.668,0.946] |
| RiNALMo-mega (148M) | 0.763 [0.664,0.849] | 0.735 [0.589,0.840] | 0.862 [0.743,0.945] |
| RNA-FM (99M) | **0.780** [0.707,0.853] | **0.772** [0.689,0.841] | 0.806 [0.600,0.923] |
| CADD | n/a | 0.684 [0.554,0.810] | n/a |
| combiner: 3 models + structure + baselines + CADD | 0.830 [0.767,0.887] | 0.839 [0.769,0.898] | 0.795 [0.660,0.905] |

Paired differences against RiNALMo-micro: mega +0.042 [-0.064,+0.164] (all genes); RNA-FM +0.059 [-0.027,+0.139] (all), +0.085 [+0.007,+0.178] (nuclear). Per gene (micro / mega / RNA-FM / CADD): RNU4-2 0.89 / 0.81 / 0.88 / 0.57; RNU2-2 0.84 / 0.85 / 0.85 / 0.52; RMRP 0.58 / 0.58 / 0.75 / 0.91; RNU7-1 0.39 / 0.24 / 0.77 / 0.88; RNU4ATAC 0.59 / 0.88 / 0.72 / 0.75; SNORD118 0.64 / 0.65 / 0.69 / 0.73.
Reading: RNA-FM is the best single score on the nuclear genes, but the one nominally significant contrast (RNA-FM vs micro, nuclear) is uncorrected for comparing three models on 7 genes, so call it suggestive. Going from 33M to 148M parameters (RiNALMo micro -> mega) bought +0.04, indistinguishable from 0. The models disagree sharply gene by gene (RNU7-1: 0.24 to 0.77; RNU4ATAC: 0.59 to 0.88), which is mostly a warning that a gene with 5-28 pathogenic variants gives a noisy AUROC. Pooling all three models into the combiner did not help (0.839 vs 0.843 for the micro-only combiner), so the models share most of their information. RNA-FM is AGPL-3.0 like RiNALMo.

### 12.4 Reading it honestly
- **The signal is real but concentrated, and small-sample.** The zero-shot score is clearly above the no-model baselines only in the mitochondrial tRNAs (0.84 vs 0.44 for transversion, paired CI excludes 0). In the 7 nuclear genes it is 0.69, which is statistically indistinguishable from CADD (0.68) and from the substitution-type baseline (0.63). The error bars are +-0.12 because there are only 7 nuclear and 6 mitochondrial genes, and 6 of the 13 genes have <= 5 pathogenic variants.
- **The language model and CADD fail on different genes.** CADD wins RMRP (0.91 vs 0.58), RNU7-1 (0.88 vs 0.39) and RNU4ATAC; the language model wins RNU4-2 (0.89 vs 0.57) and RNU2-2 (0.84 vs 0.52). Combining them gives 0.84 on the nuclear genes, +0.156 [+0.047,+0.272] over the language model alone. The combiner is evaluated leave-one-gene-out, so it is a genuine held-out-gene prediction. But the decision to add CADD to it came after an exploratory comparison of seven feature sets on these same 7 genes (the reported model is the all-features one, not the best of the seven: `LM + CADD + transversion` scored 0.831 against 0.828), and 7 genes is a small basis. Treat 0.84 as encouraging, not established.
- **The combiner hurts the mitochondrial genes** (0.77 vs 0.84 for the raw LM score): its weights are driven by the nuclear genes. For chrM use the raw masked-LM score.
- **RiNALMo has probably seen these genes.** It was pretrained on RNAcentral / Rfam-style ncRNA sequence, which contains snRNAs, RMRP and mitochondrial tRNAs and their relatives. The signal may be "this position is conserved across the family" learned from those sequences. That is a legitimate source of signal, but it is why nothing here transfers automatically to a lncRNA, which has no family alignment to remember.
- **Structure and embedding-distance scores are null in the nuclear genes** (ViennaRNA free-energy change 0.47, embedding distance 0.57). Folding a +-75 nt window is a crude proxy for the real structure of a snRNP-bound RNA.
- **The labels measure "looks like a clinically reported pathogenic site", not effect size.** P/LP variants are found where clinicians sequence and where the RNA is functional; benign / common variants are scattered elsewhere. Within-gene AUROC cannot separate "damaging" from "in the functional domain". CADD is trained on human-derived alleles, which correlates with population frequency, so it is favoured on the gnomAD-common benign set; it also leads on the ClinVar-only set (0.825 vs 0.675 nuclear, interval overlaps), so that is not the whole story.
- **Not covered: insertions and deletions.** The best-known recurrent pathogenic RNU4-2 variant is a one-base insertion (n.64_65insT, to my knowledge of the literature), which this SNV-only pipeline cannot score. Splicing, regulatory and non-exonic variants are also out of scope. Research use only; this is not a clinical classifier.

### 12.5 What is usable now
1. **`varscore score`**: give it a GRCh38 VCF or a four-column table; it maps SNVs onto the exons of every annotated non-coding gene (strand-aware), and returns each score plus the combined score and its percentile among your variants in that gene. Best configuration by evidence: nuclear genes -> `combined + CADD` (0.84; `--cadd`, `combiner_micro_cadd.json`, or `combiner_all_cadd.json` for three models) or, offline, the RNA-FM masked score alone (0.77); mitochondrial tRNA genes -> the raw masked-LM score of RiNALMo (0.84-0.86); lncRNAs -> no validated use.
2. **`varscore saturate --gene X`**: scores all 3 x length possible substitutions of one ncRNA and writes a per-position intolerance map (genomic coordinates included), e.g. for RMRP, RNU4-2, TERC or an MT-tRNA.
3. **`varscore check`**: regression tests (batch/padding invariance 3e-6, score orientation, window bookkeeping, reference-base agreement).

### 12.6 Where the long training run fits
`scripts/m4_run_all.sh cpt` continually pretrains RiNALMo-micro (or another model: `CPT_MODEL=rnafm scripts/m4_run_all.sh cpt`, ~3x slower) on lncRNA, protein-coding replay, and **small ncRNAs (snRNA, snoRNA, tRNA, ...) with all 13 panel genes held out**, so `cpt_eval` is a clean held-out-gene test: does adaptation to ncRNA improve the zero-shot score? The run is resumable (checkpoint every 500 steps; re-run the same command after an interruption). Decision rule, set now: the CPT model counts as better only if its masked-LM within-gene AUROC exceeds the frozen model's with a paired CI (printed by `varscore eval` as "delta vs ...") that excludes 0 on the same variants, **and** it does not lose on `rbp_K562`, `rbp_HepG2`, `loc_clf`. With 13 genes that is a high bar; a smaller true effect than ~+0.1 will not be detectable, so a "no difference" outcome is the likely one and should be reported as such.
Next in line, if this task is worth pushing: score indels with a windowed pseudo-log-likelihood ratio; add a second labelled source (e.g. a saturation-mutagenesis or MPRA study of an ncRNA, if one is public and can be verified); and test whether training on multi-species alignments (the GPN-MSA / PhyloGPN idea) beats any single-sequence model, which is the natural reason to build a dedicated ncRNA family.
