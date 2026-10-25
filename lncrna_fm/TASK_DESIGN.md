# LncBench task design: principles, and a defence card for every task

There is no established lncRNA benchmark for RNA language models (see `BENCHMARK_CATALOG.md`), so these tasks have to be designed, not borrowed. Biology statements marked † are standard background from memory that I did not verify by search; check them before citing. This document says *how* they are designed and why each one can be defended to a sceptical reviewer. The long-term goal is a clinical benchmark; this stage only fixes tasks that are real and grounded in biology. Tasks are tagged **built** (code + results in this repo) or **designed** (specified here, not implemented).

## 1. Principles (from ML benchmark practice, applied to biology)

| # | Principle | What it means for an lncRNA task |
|---|---|---|
| P1 | **Construct validity** | Name the biological property being measured, and say what else the label could be measuring. |
| P2 | **Mechanism** | Name a plausible way that *sequence* determines the label. A task with no mechanism is a correlation test, not a sequence benchmark. |
| P3 | **Label provenance and noise ceiling** | Labels come from an experiment or a clinical record, never from another predictor. Measure label reliability (replicates / cell lines) and report it as the ceiling. |
| P4 | **Leakage and contamination** | Split by the unit across which we claim to generalise (chromosome, gene, lncRNA, family) and flag homologs. Also disclose *pretraining exposure*: public RNA models were trained on RNAcentral, which contains GENCODE lncRNAs, so the language models may have seen our test **sequences** (never the labels). |
| P5 | **Confounds controlled by design** | Match negatives, stratify, or residualise *before* modelling. Do not rely on a model not noticing. |
| P6 | **Baseline ladder** | Every table has: sequence-free covariates -> composition (length/GC) -> k-mer spectrum -> a trained CNN -> a published predictor (when one exists) -> language-model probe. A language model must beat the rung below it to claim anything. |
| P7 | **Controls through the same code path** | Planted-signal (must be found) and shuffled-label (must give chance) controls run in the real pipeline (`lncfm.controls`). |
| P8 | **Right unit of analysis, honest statistics** | Bootstrap or permute over independent units (RBPs, genes, lncRNAs), not over correlated rows. One primary metric per task, declared in advance. Report intervals. |
| P9 | **Floor and ceiling, no trivial solutions** | A task that length or expression alone solves is a control, not a benchmark. |
| P10 | **Locked test and a change log** | Test chromosomes are scored once. Every post-hoc change is written down (PLAN.md does this). |
| P11 | **Datasheet** | Pin data versions (GENCODE v50, ClinVar release date, gnomAD v4), licences, and known failure modes next to the task. |
| P12 | **Decision relevance** | Say what decision would improve if the score were good. This is the bridge to the clinical benchmark. |

## 2. Defence cards

Each card: question, mechanism, label, formulation, split and controls, baselines, primary metric, what counts as a pass, the objections a reviewer will raise (with answers), and the clinical bridge.

### T1. Where on an lncRNA does an RNA-binding protein bind?  (built: `lncfm.rbppeak`)
- **Mechanism.** RBPs recognise short motifs and local structure (U-rich for ELAVL1†, CU-rich for PTBP1†, AU-rich for KHSRP†). lncRNAs act largely as scaffolds for such proteins, so binding-site sequence is a central lncRNA function.
- **Label.** ENCODE eCLIP IDR peaks (K562, HepG2): an experimental, nucleotide-resolution measurement.
- **Formulation.** 101-nt window centred on a peak (positive) vs windows that are not (negative), inside expressed lncRNA exons; one binary problem per RBP.
- **Negatives (the part that decides validity).** (a) *Same transcript, same relative position, >=100 nt from any peak of that RBP*: expression, length and position are matched by construction. (b) *Peaks of other RBPs*: removes "this region is generally accessible and protein-bound" and asks only for RBP specificity. Both are reported because the choice changes the answer.
- **Split.** Chromosome groups; at most 12 positive windows per gene per RBP (NEAT1 and MALAT1 would otherwise dominate); RBPs with <30 windows in a class are dropped.
- **Baselines.** length+GC, k-mer, per-RBP CNN, frozen language model.
- **Primary metric.** Macro-AUROC over RBPs, bootstrap over RBPs; paired differences between models over the same RBPs.
- **Pass.** A model beats k-mers with a paired interval excluding 0 on *both* negative types.
- **Objections.** *"Peaks are partly coverage artefacts."* Yes: local read depth differs between peak and non-peak windows. Mitigation is the same-transcript design; the composition-only baseline sits at chance on (a) (0.51), which is the evidence that composition is not leaking through. *"Why not the gene-level label?"* Because it is dominated by expression and length even after stratification; it is kept as a supporting task only.
- **Current result.** k-mer 0.58, frozen RiNALMo-micro 0.61, composition 0.51 on (a); RiNALMo vs k-mer +0.024 [-0.002,+0.051]. Small signal, real, and RiNALMo does not clearly beat k-mers.
- **Clinical bridge.** Which disease variants or ASO sites in a lncRNA disrupt a protein-binding site.

### T2. Where in the cell does an lncRNA live?  (built: `loc_clf`, `loc_reg`)
- **Mechanism.** Nuclear retention and export are driven by sequence elements: Alu/SINE-derived elements† and C-rich motifs (reported by the nuclear-enrichment MPRA in T3) promote nuclear enrichment, splicing and U1 snRNP binding† promote export, m6A readers† direct fate. Localization sets what an lncRNA can do (chromatin vs cytoplasm).
- **Label.** log2(cytosol/nucleus) from ENCODE fractionation RNA-seq in 15 cell lines, averaged over the lines where the lncRNA is expressed; classification = most-nuclear vs most-cytoplasmic tercile.
- **Label quality (measured here).** Split-half correlation across cell lines is 0.86 (Spearman-Brown reliability of the full mean about 0.93, lncRNAs expressed in >=4 lines); correlation with expression level is only +0.07. So the label is a stable, reproducible gene property that is not an expression proxy. Models currently reach rho 0.3-0.4, so there is wide headroom below the ~0.9 ceiling.
- **Split / baselines.** Chromosome-group folds, homology flags; expression+length, length+GC, k-mer, CNN, language model.
- **Primary metric.** AUROC (classification), Spearman rho (regression), CIs by bootstrap over genes.
- **Objections.** *"Fractionation ratios depend on cell line."* Averaged over lines; reliability reported. *"Published predictors use RNALocate."* Those curated labels mix methods and cell types and are not comparable across papers; an RNALocate-based external set and one published predictor are queued (`BENCHMARK_CATALOG.md` B1). 
- **Clinical bridge.** Nuclear vs cytoplasmic lncRNAs are different drug targets: ASOs that recruit RNase H act in the nucleus, siRNAs in the cytoplasm.

### T3. Which stretches of an lncRNA are sufficient for nuclear retention?  (designed; NOT buildable from the public GEO files, see below)
- **Mechanism.** Sufficiency experiment: a 110-nt lncRNA fragment appended to a cytoplasmic reporter either retains it in the nucleus or does not. This is the cleanest lncRNA-native sequence -> function measurement I found.
- **Source.** GEO GSE98828 (Shukla et al. 2018): 11,969 oligos tiling 38 human lncRNAs (tile step 10 nt, up to 2,291 tiles per lncRNA), nuclear and total counts in 6 samples each; the paper reports 109 enriched regions from 29 lncRNAs.
- **What I checked, and why it is not built.** (1) GEO carries counts and region calls but **not the oligo sequences or the lncRNA name mapping** (oligo ids are RefSeq/XLOC ids; the region table uses gene symbols); both are in a supplementary table that PMC and Europe PMC serve only behind a browser challenge, which I did not work around. (2) With the six nuclear and six total columns pooled in the obvious ways, the nuclear/total log-ratio is **not reproducible**: split-half correlation across oligos is about 0 (-0.01), and stays about 0 after averaging neighbouring tiles over windows of up to 41 tiles. Replicate pairing is not documented in the GEO table, and the authors' regions come from their own statistical model, so a naive label would be noise. P3 (label reliability) therefore fails for the simple version of this task. 
- **To build it.** Obtain the supplementary sequence/mapping table, reproduce the published region calls with the authors' model, check reproducibility on the called regions, then run the split-by-lncRNA task. Until then T3 is a design, not a result.
- **Clinical bridge.** Predictable localization for therapeutic lncRNAs and oligo design.

### T4. Does predicted damage track natural selection?  (built: `lncfm.constraint`)
- **Mechanism.** Purifying selection removes variants that damage function, so damaging variants are seen less often in the population than mutation alone predicts. No clinical label is needed, so it applies to every lncRNA.
- **Label.** gnomAD v4 observed/unobserved status of every possible SNV in the exons of an lncRNA, against a trinucleotide-context mutation model calibrated per gene.
- **Formulation.** Cluster-robust score test: within-gene damage vs the residual (observed - expected); O/E by damage decile.
- **Controls.** Shuffled-damage control (z ~ 0); a **positive-control set chosen before any result** (abundant functional ncRNAs: RNU*, RN7SK, RMRP, TERC, vault, Y RNAs, ...).
- **Primary metric.** beta per SD of damage with gene-clustered SE.
- **Objections.** *"Weak constraint in lncRNAs means no power."* Right; that is why the positive-control set is run first, and a null on lncRNAs with a positive on the control set is itself a finding. *"gnomAD coverage differs by site."* Per-gene calibration absorbs gene-level coverage; within-gene coverage remains a limitation.
- **Clinical bridge.** Prioritising variants in lncRNA exons without any disease labels.

### T5. Is a variant in a disease ncRNA pathogenic?  (built: `lncfm.varscore`, audited: `lncfm.varaudit`)
- **Mechanism.** Mutations in snRNAs, RMRP, tRNAs and TERC disrupt structure and protein binding and cause named diseases (e.g. RNU4ATAC, RMRP, mitochondrial tRNA disease, telomere disorders).
- **Label.** ClinVar P/LP vs B/LB, with gnomAD common variants as extra benign examples; strict mapping to the right transcript, strand and gene.
- **Formulation.** Within-gene ranking (gene identity alone separates the classes), weighted over genes; hierarchical bootstrap.
- **Controls.** Permutation test within gene; substitution-class stratification; position-vs-allele decomposition; frequency analysis; CADD as the external comparator.
- **Honest scope.** Small structured ncRNAs only; **no lncRNA ground truth exists** (TERC aside). See PLAN section 12 and the audit in section 14.
- **Clinical bridge.** This is the first clinical-grade task: ncRNA variant interpretation.

### T6-T9. Designed, not yet implemented
| ID | Question | Mechanism | Label (public) | Formulation / split | Main risk |
|---|---|---|---|---|---|
| T6 | How stable is the lncRNA? | AU-rich elements†, m6A†, miRNA sites†, exon-junction density† | BRIC-seq half-lives for 1,418 ncRNAs in HeLa (Tani 2012) and lncRNA rows of the compendium assembled for Saluki | regression on log half-life, chromosome folds; covariates: expression, length | decay estimates are noisy and method-dependent; report cross-dataset reliability first |
| T7 | Where are the splice sites and 3' ends of an lncRNA, and how strong? | lncRNAs are spliced and 3'-processed less efficiently than mRNAs† (weaker 5'/3' splice signals, early termination), which is thought to contribute to nuclear retention and instability† | GENCODE introns; polyA-site catalogues | donor/acceptor vs GT/AG decoys within the same gene; strength regression from RNA-seq junction usage | needs the genome FASTA; decoy design matters |
| T8 | Is the lncRNA translated? | uORF/sORF with Kozak context, codon usage | Ribo-seq-supported ORFs | classification of lncRNA with vs without translated ORF; ORF-level windows | Ribo-seq evidence is cell-type dependent and is not proof of function |
| T9 | Which adenosines are m6A? | DRACH motif† + context; nuclear lncRNAs (XIST, MALAT1, NEAT1) are reported m6A-rich† | miCLIP / m6A-site catalogues | site-level windows, lncRNA subset; negatives = unmodified DRACH sites in the same transcripts | catalogues mix methods and cell types |

### T10. Expression-based clinical track (built, sequence-agnostic)
AML genotype and risk from lncRNA expression (BeatAML -> TCGA-LAML / TARGET). It tests whether lncRNA *expression* carries clinical information (it does, equally to mRNA), not whether a sequence model does. It stays as a separate track and as the template for the clinical benchmark: cohort-held-out external validation, a clinical-baseline ladder (age, ELN, mRNA), and no claim beyond what the external cohort supports.

## 3. From tasks to a clinical benchmark (the stated end goal)
A clinical task is admitted only when its biological parent task already passes P1-P12.

| clinical question | parent tasks | label needed | status |
|---|---|---|---|
| Is this variant in a disease ncRNA pathogenic? | T5, T4 | ClinVar / ClinGen-curated | first candidate; small |
| Will a variant in an lncRNA exon disrupt protein binding or processing? | T1, T7, T9 | none directly; validated by T1/T7 held-out sites | needs T7 |
| Where will an lncRNA-targeting oligonucleotide act, and does the site look accessible? | T2, T3, T1 | oligo efficacy data (not located) | future |
| Does lncRNA expression stratify patients beyond clinical covariates? | T10 | outcome cohorts | partly built |

## 4. How to defend any task in one paragraph (template)
"We ask whether *sequence* predicts **X**. X is measured by **experiment E**, whose replicates agree at **R**, so the ceiling is **R**. Sequence plausibly determines X through **mechanism M**. Negatives / splits are built so that **confound C** cannot explain a positive result, and we show that a baseline that sees only C sits at chance. We compare against **rungs of the baseline ladder** and report **paired intervals over independent units**. A planted-signal control shows the pipeline can find a real effect and a shuffled-label control shows it does not invent one. We state what the task does **not** show."

## 5. What this benchmark does not claim
No task here shows that a model understands lncRNA function. T1 and T2 show small but real sequence signal; T5 shows signal in small structured ncRNAs, not lncRNAs; T10 shows expression, not sequence, carries clinical information. The case for building a dedicated lncRNA model rests on tasks that the stricter versions (T1, T3, T4) have not yet established.
