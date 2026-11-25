# T2 — homology leakage in a lncRNA benchmark: precondition MET (2026-11-25)

Full result: `results/t2_homology.txt`. Script: `homology_audit.py`.

## What was wrong with the committed result

LncBench reports every metric on all items and on a homology-pruned subset, and the committed v2 run shows the
two agreeing to within 0.003 everywhere. That reads as a clean null. It is an artifact of subsampling:

* the run scored an **800-gene subsample** of 36,476 GENCODE v50 lncRNA genes (`items=800 ... homology-flagged 0.4%`);
* pruning removed **3 items of 700**;
* subsampling alone predicts that. With ~2,500 homolog pairs among the full gene set, the chance both members of
  a pair survive a 700-gene sample is (700/36476)^2, giving one or two surviving pairs, so two to four flagged genes.

The comparison could not have detected leakage in either direction.

## What the full-size audit found

All 34,866 lncRNA genes with recoverable spliced sequence (median 858 nt), folds replicated exactly from
`lncfm/splits.py` (chr8/13/18/21 locked; remaining chromosomes greedily packed into 5 dev folds by gene count):

| 10-mer containment | pairs | genes with a homolog | genes with a **cross-fold** homolog |
|---|--:|--:|--:|
| 0.10 | 931,502 | 34,804 (99.8%) | 31,192 (89.5%) |
| 0.25 | 97,035 | 33,545 (96.2%) | 28,400 (81.5%) |
| **0.40 (what LncBench uses)** | 26,548 | 20,227 (58.0%) | **16,174 (46.4%)** |
| 0.50 | 7,779 | 6,505 (18.7%) | 4,571 (13.1%) |
| 0.70 | 1,266 | 1,330 (3.8%) | 404 (1.2%) |

**At the benchmark's own threshold, honest pruning would remove 46.4% of the gene set. The run removed 0.4%.**

## An unresolved discrepancy, stated rather than buried

`lncfm/splits.py` carries the comment *"25% flags 77% of genes (generic Alu/L1 sharing); 40% flags 14%; 50%
flags 7% (measured on GENCODE v50)"*. We measure 96.2% / 58.0% / 18.7% at those thresholds — about four times
higher at 0.40. The most likely cause is a different gene set or representative-transcript rule (`rep_genes()`
builds from `gencode.build_table("human", "lnc")`, which may filter transcripts in ways this script does not).
That data is not in this container, so the discrepancy is not resolved here.

It does not change the conclusion. Whether the true rate is 14% or 58%, the committed run flagged **0.4%**, so it
under-removed by one to two orders of magnitude either way.

## Caveat on what "homology" means here

10-mer containment at 40% catches paralogs *and* shared repeat families; lncRNAs are repeat-rich, so part of
this is Alu/L1 content rather than common ancestry. For benchmark purposes that distinction matters less than it
sounds: a test gene sharing 40% of its 10-mers with a training gene is exploitable by a k-mer model whatever the
reason for the sharing. But the number should be described as "shared k-mer content", not "paralogy", and an
alignment-based measure would both drop the repeat-driven pairs and add diverged homologs this misses.

## Status against the kill criterion

T2's criterion is about the **metric drop**: "if the drop is under 0.03 AUROC, the leakage story is not ours to
tell." That is not yet measured. What is now established is the precondition — there is a large cross-fold
homologous fraction available to remove, so the comparison is worth running.

Next: a full-size labelled task, all items vs pruned, to measure the actual drop.
