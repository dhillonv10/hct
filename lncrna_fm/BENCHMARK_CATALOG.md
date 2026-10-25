# Benchmark catalog: RNA tasks from the literature, and how each translates to lncRNA

Purpose: stop inventing tasks and start from tasks the RNA field already uses, then say honestly which ones transfer to lncRNA, which need a new lncRNA-native version, and which should not be built.

Evidence tags: **[V]** = checked by web search while writing this (link given); **[M]** = from my memory, *not* verified, check the paper before citing; **[ours]** = built in this repo.

## 0. What the search found (the gap this project can fill)
- **BEACON** (13 tasks, NeurIPS 2024) is the general RNA language-model benchmark: structure (secondary structure, contact map, distance map, structural-score imputation), function (splice site, alternative polyadenylation, ncRNA family classification, RNA modification), engineering (mean ribosome loading, vaccine degradation, programmable switches, CRISPR on/off-target). Its headline finding: simple ResNet / LSTM baselines beat the pretrained RNA language models on several tasks, and language models set a new best on 8 of 13 [V: [arXiv 2406.10391](https://arxiv.org/abs/2406.10391)]. **Lesson for us: every task needs a trained non-language-model baseline, not only k-mers.**
- **RiNALMo** is evaluated on secondary structure, multi-species splice sites, mean ribosome load, and generalization to unseen RNA families [V: [arXiv 2403.00043](https://arxiv.org/abs/2403.00043)]. None of this is lncRNA.
- I found **no lncRNA-specific benchmark for RNA language models**. Published lncRNA predictors (localization, lncRNA-protein interaction) each use their own dataset, split and class definition, so their reported numbers cannot be ranked against each other [V: localization, [Loc4Lnc](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12806047/); interaction, [RNA-protein interaction survey](https://arxiv.org/pdf/2410.00077)].
- Consequence: for lncRNA there is no leaderboard to anchor on. The anchor has to be (a) general-RNA tasks that prove our *harness* reproduces the field's numbers, plus (b) lncRNA tasks built to the field's *best-practice formulation* with the field's known pitfalls controlled.

## 1. A task counts only if it passes this checklist
1. The label comes from an experiment or a clinical record, not from a prediction tool.
2. There is a published formulation of the task (not ours), or a clear reason it is new.
3. A trained non-language-model baseline (CNN/ResNet-style) and a sequence-free baseline are run next to every language model.
4. The known confound for that task is controlled by design (expression, length, GC, negative-sampling choice, family leakage).
5. Held-out chromosomes **and** homology flags (already in `splits.py`).
6. A positive control (a planted signal the harness must find) and a negative control (shuffled labels must give 0.5) run in the same code path.
7. At least one published task-specific predictor is run on the same split, or the task is labelled "no external comparator yet".

## 2. Catalog

### A. Harness controls (these decide whether the numbers mean anything)
| ID | Control | Status |
|---|---|---|
| A1 | Planted-motif positive control: windows with an inserted motif must be detected (AUROC near 1) through the same feature -> probe -> metric path | **built** (`lncfm.controls`) |
| A2 | Shuffled-label negative control: AUROC must be 0.5 +- CI | **built** |
| A3 | Untrained-twin and sequence-free baselines in every table | built [ours] |
| A4 | Batch/padding invariance of embeddings; offset/strand mapping round trip | built (`selfcheck`, `varscore check`) |
| A5 | Anchor to a published number: reproduce a published RNA-LM result with our probe (e.g. ncRNA family classification on nRC13, 8,840 sequences / 13 classes [V: [nRC](https://link.springer.com/doi/10.1186/s13040-017-0148-2), [data](https://ieee-dataport.org/documents/nrc13)]; published accuracies come from different subsets and splits, so match one paper's split exactly) | **not built**; needs one paper's exact split |

### B. Transcript/gene-level properties
| ID | Task | Literature anchor | lncRNA translation | Status | Main validity risk |
|---|---|---|---|---|---|
| B1 | Subcellular localization (nucleus vs cytoplasm, 5-class) | lncLocator, iLoc-lncRNA, DeepLncLoc, GraphLncLoc, Loc4Lnc on RNALocate-derived labels [V] | ours uses ENCODE nuclear/cytosolic fractionation (quantitative, 15 cell lines) [ours] | built; **add** an RNALocate 5-class external set and run >=1 published predictor | published numbers not comparable across papers; fractionation ratio partly follows expression |
| B2 | Coding vs non-coding | CPAT / CPC2 / LncFinder [M] | `coding_clf` [ours] | built; **control only** (length alone gives AUROC 0.89) | trivial |
| B3 | Ribosome-engaged lncRNAs / micropeptide potential | misannotated-lncRNA detection with training dynamics [V: [Nabi et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9825752/)]; no standard benchmark found | label = Ribo-seq-supported translated ORF in a lncRNA (public Ribo-seq/ORF catalogues; not yet chosen) | not built | Ribo-seq evidence is cell-type dependent and not proof of function |
| B4 | RNA half-life / stability | Saluki, mRNA half-life from sequence, 39 human + 27 mouse datasets compiled [V: [Agarwal & Kelley 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9684954)] | BRIC-seq in HeLa measured 1,418 ncRNAs [V: [Tani et al. 2012](https://genome.cshlp.org/content/22/5/947)]; also the lncRNA rows of the human datasets in the Saluki compendium | **new; not built** (needs the supplementary table) | decay estimates are noisy and method-dependent (the Saluki paper says so); 1.4k labelled ncRNAs |
| B5 | Tissue specificity (GTEx tau, testis, brain) | none for sequence models | `tau_reg`, `testis_clf`, `brain_clf` [ours] | built; **demote to exploratory** | no literature tradition; frozen RiNALMo loses on brain |
| B6 | Orthology / conservation retrieval | positional (synteny) orthologs [M] | `lncfm.retrieval` [ours] | built; negative result (shared 10-mers win); exploratory | few named ortholog pairs |
| B7 | lncRNA-disease association | LncRNADisease, Lnc2Cancer [M] | none | **do not build** | literature-curated, popularity-biased: well-studied lncRNAs have more links |
| B8 | GENCODE biotype / lncRNA subclass | trivial | none | **do not build** | genomic context, not sequence |

### C. Position-level (a window or a nucleotide), the formulation the RNA field prefers
| ID | Task | Literature anchor | lncRNA translation | Status | Main validity risk |
|---|---|---|---|---|---|
| C1 | **RBP binding sites (eCLIP peaks)** | systematic benchmark of 37 methods on CLIP-seq; result depends on how negatives are drawn; sites bound by other RBPs are a bias-aware negative [V: [Brief. Bioinform. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10516373/)] | **peak windows inside lncRNA exons vs windows from the same transcripts: unbound, and bound by other RBPs; K562 -> HepG2 transfer** | **built now** (`rbp_peak`) | local coverage and accessibility still differ between peak and non-peak windows |
| C2 | RNA modification sites (m6A etc.) | MultiRM (12 modifications), DeepM6ASeq on miCLIP [V]; BEACON "Modif" task | evaluate on lncRNA-only sites (needs site coordinates from miCLIP / m6A-Atlas-type catalogues) | **new; not built** | site catalogues mix methods and cell types; no lncRNA-specific benchmark found |
| C3 | Splice sites | RiNALMo multi-species splice-site task [V]; SpliceAI [M] | donor/acceptor vs GT/AG decoys inside lncRNA genes (needs the genome FASTA ~0.9 GB) | **new; not built** | lncRNA splice sites are weaker and less conserved, which is the interesting part; decoy choice matters |
| C4 | Alternative polyadenylation | BEACON APA task [V] | lncRNA 3' end / poly(A) site usage from 3'-end sequencing | new; not built | 3' ends of lncRNAs poorly annotated |
| C5 | Probing reactivity (SHAPE/DMS/icSHAPE) | BEACON structural-score imputation [V] | in-cell probing of lncRNAs (ENCODE lists icSHAPE experiments) | new; not built | coverage of lncRNAs is low; reactivity is expression dependent |
| C6 | Secondary structure | bpRNA, ArchiveII; RiNALMo, BEACON SSP [V] | no gold lncRNA structures at scale | **anchor only** (general RNA, shows the harness works) | not about lncRNA |
| C7 | **Nuclear-enrichment sequences (MPRA)** | Shukla et al. 2018: 11,969 110-nt oligos tiling 38 human lncRNAs, 109 nuclear-enriched regions from 29 lncRNAs [V: [EMBO J](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5852646/)] | native lncRNA sequence -> function labels | **new; not built**: need the processed table (GEO accession not yet located) | 38 lncRNAs only; tiles overlap, so split by lncRNA |

### D. Variant effect
| ID | Task | Anchor | lncRNA translation | Status | Risk |
|---|---|---|---|---|---|
| D1 | ClinVar P/LP vs B/LB, within gene, vs CADD | standard design in variant-effect benchmarks [M] | **only ~1 lncRNA disease gene exists (TERC)** | built (`varscore`) on small ncRNAs | n = 13 genes; see PLAN section 12 |
| D2 | Population-constraint test: is predicted damage depleted among variants observed in gnomAD, relative to all possible SNVs of the same trinucleotide context? | the logic behind genome-wide constraint maps [V: [Gnocchi](https://www.biorxiv.org/content/10.1101/2022.03.20.485034.full.pdf)] | works for **every** lncRNA exon, no clinical labels needed | **new; not built** | small effect sizes; needs a mutation-rate model |
| D3 | Regulatory-variant MPRA | Kircher et al. 2019, saturation mutagenesis of 20 promoters/enhancers [V] | lncRNA *promoters* only, not the RNA body | not built | not RNA function |
| D4 | lncRNA saturation mutagenesis | none found [V: search returned none] | n/a | **no data exists** | -- |

### E. Interaction tasks
| ID | Task | Anchor | Verdict |
|---|---|---|---|
| E1 | lncRNA-protein interaction (RPI369/488/1807/2241, NPInter) | many papers [V] | **do not build as a headline**: surveys describe weak deduplication, inconsistent negatives, structure-derived positives dominated by ribosomal complexes; random-split accuracies of 0.95-0.99 are consistent with leakage [V: [survey](https://arxiv.org/pdf/2410.00077)]. If built, split by RNA and protein sequence identity. |
| E2 | lncRNA-miRNA interaction (starBase/ENCORI) [M] | pair-level, negatives artificial | not built; same caveats as E1 |

### F. lncRNA-native functional labels (gene level)
| ID | Task | Anchor | Verdict |
|---|---|---|---|
| F1 | CRISPRi growth hits | Liu et al. 2017: 16,401 lncRNA loci, 499 hits, 89% hit in only one of 7 cell lines [V: [Science 2017](https://doi.org/10.1126/science.aah7111)] | exploratory: the perturbation is at the promoter, so a hit is a property of the *locus*, not necessarily of the transcript sequence; cell-line-specific labels |
| F2 | lncRNA expression-derived clinical labels (AML, TCGA) | many [M] | keep as a separate **expression-based** clinical track, not a sequence benchmark |

### G. General-RNA anchors (not lncRNA; used only to show the harness reproduces known behaviour)
Secondary structure (C6), ncRNA family classification (A5), mean ribosome loading [V: RiNALMo], splice sites in non-lncRNA genes, APA, modification sites (MultiRM-style). Run them once, compare to the paper the split comes from, then stop.

## 3. Re-grade of what is already in LncBench
| current task | verdict | why |
|---|---|---|
| `rbp_K562/HepG2/union` (gene-level "does the lncRNA have any peak for RBP X") | keep as supporting; **replace as headline by C1 `rbp_peak`** | gene-level labels are dominated by expression and length even after stratification; the field's formulation is peak-level with a stated negative-sampling rule |
| `loc_clf`, `loc_reg` | keep (headline) + add B1 external set and one published predictor | |
| `tau_reg`, `testis_clf`, `brain_clf` | demote to exploratory | no sequence-model tradition; mixed results |
| `coding_clf` | control only | length alone solves it |
| AML gene-level tasks | demote / drop | labels depend on cohort and normalisation |
| AML patient-level + genotype panel | keep as an **expression-based clinical track** | robust, replicated externally, but it is not about sequence |
| variant panel (`varscore`) | keep as supporting; add D2 | 13 genes; honest about it |
| orthology retrieval | exploratory | negative result |

## 4. Variant data beyond ClinVar (answer to "any other places to look?")
- **LOVD** has curated germline TERC variants with pathogenicity classes [V: [LOVD TERC](https://databases.lovd.nl/shared/view/TERC)], useful to add TERC pathogenic variants but not benign ones, and the entries can contain errors.
- **LincSNP / lncRNASNP v3** list disease-associated (GWAS, LD) SNPs and eQTLs that fall in lncRNAs [V: [LincSNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4038069/)]. These are *associations*: a GWAS tag SNP in a lncRNA is rarely the causal variant, and eQTLs act on expression. Not a damaging-vs-neutral label.
- **gnomAD / Gnocchi** gives population constraint in 1 kb windows [V], too coarse for nucleotides but the basis of task D2.
- **MPRA / saturation mutagenesis:** none found for lncRNA bodies [V]. Kircher 2019 covers promoters and enhancers.
- **Somatic cancer mutations** in lncRNAs (e.g. NEAT1, MALAT1) exist in pan-cancer data [M]; recurrent-mutation claims there have been disputed [M], so I would not use them as labels.
- Conclusion: **there is no lncRNA-body variant gold standard to find.** The valid options are (1) small ncRNAs with ClinVar labels (done), (2) the population-constraint test D2 on lncRNA exons, (3) a nuclear-enrichment MPRA (C7) where position-level function is measured on lncRNA sequence.

## 5. Build queue
| order | item | why first | data in hand? |
|---|---|---|---|
| 1 | A1/A2 controls | proves the harness can detect a real signal and does not invent one | yes |
| 2 | C1 `rbp_peak` | the RNA field's standard formulation, with its main pitfall (negatives) controlled | yes (eCLIP peaks, lncRNA exons) |
| 3 | B1 external localization set + a published predictor | gives a comparator | RNALocate download, not checked |
| 4 | D2 population-constraint on lncRNA exons | the only variant test that covers real lncRNAs | gnomAD API (works) |
| 5 | C7 nuclear-enrichment MPRA | lncRNA-native, position-level | processed table not located |
| 6 | B4 half-life | experimentally grounded, mRNA precedent | supplementary tables |
| 7 | C3 splice decoys, C2 m6A | standard RNA tasks, new for lncRNA | genome FASTA / site catalogues |
| 8 | A5 anchor to nRC13 | ties the harness to a published number | one paper's exact split |
