# lncRNA foundation model: plan, benchmark, and what the pilot showed

> Status legend used throughout: **[measured]** = I ran it in this repo and the number is in `results/`;
> **[verified]** = checked against the live source during this session (file listing, API, README);
> **[design]** = implemented or specified but not run at scale; **[assumption]** = my estimate, flagged so you can check it.

RESULTS_PLACEHOLDER

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

OPTION_B_PLACEHOLDER

---

## 5. LncBench

LNCBENCH_PLACEHOLDER

---

## 6. Clinical track

CLINICAL_PLACEHOLDER

---

## 7. Compute and hardware

COMPUTE_PLACEHOLDER

---

## 8. Run order

RUN_PLACEHOLDER

---

## 9. Threats to validity (read before trusting any number)

THREATS_PLACEHOLDER
