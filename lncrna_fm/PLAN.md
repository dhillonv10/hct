# lncRNA foundation model: plan, benchmark, and what the pilot showed

> Status legend used throughout: **[measured]** = I ran it in this repo and the number is in `results/`;
> **[verified]** = checked against the live source during this session (file listing, API, README);
> **[design]** = implemented or specified but not run at scale; **[assumption]** = my estimate, flagged so you can check it.

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
   (b) Zero-shot ClinVar variant scoring is at chance (within-gene AUROC 0.55, CI includes 0.5), and ClinVar cannot support a lncRNA-wide variant benchmark at all (§6).
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
