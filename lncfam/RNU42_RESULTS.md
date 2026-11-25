# Clinical labels overstate RNA language models by ~0.3 AUROC on the one gene where we can check

Run 2026-11-25. Data: Nature 2026;654:429-436 (CC BY 4.0), saturation genome editing of RNU4-2.
Scripts: `rnu42_sge.py`, `export_figdata.py`. Raw: `/home/user/lncfam_data/rnu42/`.

## The setup

RNU4-2 is the ReNU syndrome gene. It appears in two benchmarks we can run:

* **Clinical labels.** Our 12-gene ClinVar panel: 21 pathogenic and 8 benign SNVs in RNU4-2.
* **Functional labels.** The Nature saturation-editing screen: **435 SNVs**, essentially every possible
  substitution, each with a measured growth function score. No clinical judgement enters the label.

The same three RiNALMo checkpoints are scored on both, by the same rule (masked log-odds, zero-shot, no training).

## Correctness gate, passed before anything new was computed

The paper ships CADD and ViennaRNA ddG per variant, so the harness can be checked against its own published
figures first:

| | ours | published | |
|---|--:|--:|---|
| CADD v1.7 | 0.649 | 0.65 | match |
| ViennaRNA ddG | 0.718 | 0.72 | match |
| SGE assay | 0.934 | 0.93 | match |

Counts match the paper's figure legend (12 ReNU vs 346 population SNVs), and all 435 HGVS reference bases agree
with the transcript extracted from hg38 at chr12:120,291,753-120,291,903 on the minus strand.

## The result

**On clinical labels, scale looks like it helps. On functional labels, it does nothing.**

| | 33M | 148M | 651M |
|---|--:|--:|--:|
| ClinVar, RNU4-2 only (21 P/LP vs 8 B/LB) | **0.96** | 0.84 | 0.86 |
| ClinVar, all 12 genes | 0.738 | 0.792 | **0.836** |
| **Functional, depleted vs not (106 vs 329)** | **0.611** | 0.598 | 0.603 |
| Functional, Spearman vs effect size (435) | 0.179 | 0.167 | 0.189 |

Across the 12-gene ClinVar panel the ladder is clean and monotone: 0.738 -> 0.792 -> 0.836, +0.1 AUROC for 20x
the parameters. On functionally measured variants **in one of those same genes**, the same models give
0.611 -> 0.598 -> 0.603. Flat. Twenty times the parameters buys nothing.

## And a thermodynamic calculation beats all of them

Best-powered contrast, the assay's own q<0.01 hit calls (106 depleted vs 329 not):

| score | AUC | 95% CI |
|---|--:|---|
| CADD v1.7 | 0.564 | [0.503, 0.622] |
| **ViennaRNA ddG** | **0.629** | [0.564, 0.689] |
| RiNALMo-micro (33M) | 0.611 | [0.552, 0.671] |
| RiNALMo-mega (148M) | 0.598 | [0.538, 0.652] |
| RiNALMo-giga (651M) | 0.603 | [0.544, 0.661] |
| SGE assay (ceiling) | 0.934 | |

The Spearman against the continuous effect size over all 435 variants gives the same ordering: ddG 0.237,
giga 0.189, micro 0.179, mega 0.167, CADD 0.088. A folding free-energy calculation beats a 651M-parameter
language model on both.

## Why the paper's own contrast cannot settle it

On the 12-ReNU-vs-346-population comparison the paper reports, our models give micro **0.750**, mega **0.447**,
giga **0.539** — a 0.3 spread across three checkpoints of the same model family. With 12 positives the intervals
are +-0.15 and span chance. Quoting micro's 0.750 as "a language model beats CADD's 0.65 and ddG's 0.72" would
be true, cherry-picked, and unreplicated by its two larger siblings. That is why metric C exists.

## The likely explanation, and why it matters

ClinVar pathogenicity calls rest partly on conservation evidence. A masked language model's score is, in large
part, learned conservation. So on clinical labels the model is being scored against something correlated with
its own input, and it looks strong. Give it a label that was never informed by conservation — a measured growth
effect — and the apparent advantage largely goes.

This is the caveat we wrote into `distill/CLINICAL_PROTOCOL.md` before running anything ("the labels measure
'looks like a clinically reported pathogenic site', not effect size"). This is the first time we have been able
to put a number on it: **~0.96 against ClinVar, ~0.61 against function, same gene, same model.**

## What this does not show

* The assay is a growth screen in HAP1 cells; it is a functional proxy, not the disease phenotype, and the paper
  says it "does not inform directly on underlying mechanisms".
* One gene. RNU4-2 is a structured spliceosomal snRNA; folding energy should do well here, and ddG does. Whether
  it beats language models on other ncRNA classes is untested.
* The RNU4-2 ClinVar slice is 29 variants, so 0.96 is itself imprecise. The direction is supported by all three
  models and by the 12-gene panel, not by that one number.
* We did not test RNA-FM, ERNIE-RNA, or mitochondria-specific tools.

## Figure data

`fig_data/` holds four tidy CSVs, enough for every planned figure without recomputation:
`variants_scored.csv` (per-variant, all scores), `metrics.csv` (every AUC and Spearman with bootstrap
intervals), `roc_curves.csv` (ROC points per score per contrast), `by_position.csv` (per-position aggregates for
the positional track).
