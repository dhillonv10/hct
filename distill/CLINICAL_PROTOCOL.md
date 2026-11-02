# Clinical benchmark protocol: ClinVar variants in non-coding RNA genes

**Pre-registered. Written 2026-11-02, before the benchmark was run on real data.** Code: `rnakd/clinvar.py`,
`rnakd/lm.py`, `rnakd/clinical.py`. Notebook: `clinical_colab.ipynb`.

## 1. Why this task exists in this project

Everything else we have measured is a research endpoint. This is the only **clinical** one a sequence model can be
put on directly: ClinVar's pathogenic / benign calls on single-nucleotide variants inside non-coding RNA genes. Both
a 650M language model and a ~1M student score the same string by the same rule, with no labels and no fine-tuning:

```
s = log p(ref | context, variant position masked) - log p(alt | same)      higher = more damaging
```

The patient-level AML tasks elsewhere in this repository (BeatAML -> TCGA-LAML / TARGET) read expression matrices,
not sequence, so a teacher and a student cannot be compared on them without an arbitrary bridge. They are a
different paper.

## 2. The panel

Built from the ClinVar GRCh38 VCF and GENCODE, with filters ported from `lncrna_fm/lncfm/ncvar.py`:

* single-nucleotide variants classified Pathogenic / Likely pathogenic (label 1) or Benign / Likely benign (label 0);
* inside an exon of a non-coding gene, **attributed by ClinVar to that gene** (without this, TERC variants land on the
  antisense lncRNA at the same locus);
* carrying **no coding molecular consequence** (missense, nonsense, splice donor/acceptor, ...), because the label
  would then belong to an overlapping protein-coding gene. UTR and intronic consequences are kept: RNU2-2 sits inside
  the 5' UTR exon of WDR74, and excluding it by position deletes a genuine disease gene;
* for lncRNA genes only, a pathogenic record co-attributed to a protein-coding gene is dropped unless the disease name
  names the lncRNA. Restricted to lncRNA because applying it to snRNA/snoRNA wrongly removed RNU4-2 and SNORD118;
* one transcript per gene (the longest), de-duplicated by (gene, offset, alt);
* genes with at least 3 pathogenic and 3 benign variants.

A hard correctness test runs inside the builder: the reference base of every variant must match the transcript
sequence at the mapped spliced offset, strand-corrected. Any mismatch is dropped and the count is printed; a large
count means the genomic-to-transcript mapping is broken.

Expected membership, from the earlier run in `lncrna_fm`: RNU4-2, RNU2-2, RNU4ATAC, RNU12, RNU7-1, RMRP, SNORD118 and
a set of mitochondrial tRNAs including MT-TL1 (MELAS), MT-TK, MT-TL1, MT-TN, MT-TS1, MT-TV, MT-TW.

## 3. Metric

**Within-gene AUROC**, weighted by min(pathogenic, benign) per gene. Pooled AUROC is not used and will not be
reported: gene identity alone separates the classes in ClinVar, so a pooled number measures which genes are famous.
95% intervals come from a hierarchical bootstrap that resamples genes and then variants inside each resampled gene.

## 4. Arms

| arm | what it is |
|---|---|
| `transversion`, `gc_loss` | no-model baselines. A language model that cannot beat these is not doing biology |
| `lm_micro`, `lm_mega`, `lm_giga` | RiNALMo 33M / 150M / 650M pretrained, zero-shot, identical rows and procedure |
| `s1 / s5` x `mlm` | a ~1M or ~5M transformer trained on the transfer set with the **true base** as target: train your own small language model, no teacher |
| `s1 / s5` x `kd` | the same model, same sequences, same masked positions, with the **teacher's distribution** as target |
| `s1 / s5` x `mlm_kd` | both targets |

The transfer set is human non-coding transcripts in 256-nt windows **with every panel gene removed**, so the student
is never trained on a sequence it is tested on. The masked positions are fixed and shared across arms, so `mlm`,
`kd` and `mlm_kd` differ only in the target. That is what makes "did distillation buy anything" a controlled
comparison rather than a vibe.

## 5. Gates (fixed; the report prints each)

* **C1, wiring.** Every teacher recovers a masked nucleotide above chance (accuracy > 0.30 against 0.25). This is the
  gate that catches a mis-wired mask head before any plausible-looking AUROC is believed. The head is
  `linear1 -> GELU -> layer_norm -> linear2`, read from `rinalmo/model/modules.py`.
* **C1b, internal reproduction.** RiNALMo-micro scored **0.72** within-gene AUROC on a 13-gene panel in
  `lncrna_fm/PLAN.md` section 12. That run used the `multimolecule` re-host rather than the official Zenodo weights
  and a slightly different benign set, so the target here is 0.72 +- 0.10. Outside that band, investigate before
  reading anything else.
* **C2, does scale buy clinical accuracy?** Report `lm_giga - lm_micro` with its interval. **No threshold is set**:
  a flat ladder (650M no better than 33M on a clinical endpoint) is as interesting as a steep one and will be
  reported either way.
* **C3, does distillation beat training your own small model?** `kd` minus `mlm` at matched size, called real only
  above the seed spread (the largest standard deviation across seeds within an arm).
* **C4, retention.** Best student AUROC as a fraction of the best teacher's, with the parameter ratio.
* **C5, sanity.** The best teacher must beat both no-model baselines. If it does not, the panel is not measuring what
  we think and nothing else in the report stands.

**Headline claim permitted only if** C1 passes, C5 passes, and a student reaches at least 90% of the best teacher's
within-gene AUROC. Anything weaker is reported as the measurement it is.

## 6. What this benchmark does not show (read before quoting any number)

* **The labels measure "looks like a clinically reported pathogenic site", not effect size.** Pathogenic variants are
  found where clinicians sequence and where the RNA is functional; benign ones are scattered elsewhere. Within-gene
  AUROC cannot separate "damaging" from "in the functional domain".
* **This is the easier of the two possible backgrounds.** The benign set is ClinVar B/LB. Our earlier audit
  (`lncrna_fm/PLAN.md` section 14) found that against *rare* gnomAD-observed variants, which is the situation a
  clinician actually faces, the zero-shot language-model score fell to 0.52-0.58 while CADD held at 0.71. Nothing here
  contradicts that, and no clinical-use claim follows from this panel.
* **The teacher has probably seen these families.** RiNALMo was pretrained on RNAcentral-scale non-coding RNA, which
  contains snRNAs, RMRP and mitochondrial tRNAs and their relatives. Part of any signal is "this position is conserved
  across the family", learned from those sequences. That is legitimate signal and it is also why nothing here
  transfers automatically to a lncRNA with no family alignment to remember.
* **The panel is small**, on the order of a dozen genes and a few hundred variants, several with fewer than six
  pathogenic records. Intervals are wide and per-gene numbers are noisy.
* **CADD is not included.** The earlier work scored it through a public API; it is the comparator a reviewer will ask
  for on the nuclear genes, and adding it is the first extension.
* Research use only. This is not a clinical classifier.
