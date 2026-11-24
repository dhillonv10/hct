# lncRNA model family: five-day verification sprint (Tue 2026-11-24 -> Sun 2026-11-29)

Purpose: decide whether to build a compact lncRNA model family, by testing the three assumptions it rests on.
Each task is self-contained, ends in a committed artifact, and has a kill criterion fixed **before** it runs.
If a task's kill criterion fires, the arm it tests is dropped and the sprint moves on; it does not get re-argued.

The failure mode this sprint exists to avoid: every direction in this project so far was proposed before it was
checked, and each one died to a fact that a day of work would have surfaced. So nothing is built this week.

## The family these tasks would justify

| arm | member | label source | tested by |
|---|---|---|---|
| DNA / locus | lncRNA essentiality, cell-type specific | CRiNCL CRISPRi (16,401 loci x 7 lines) + Cas13 screens | T1 |
| RNA / transcript | subcellular localization | ENCODE fractionation | T2 |
| RNA / transcript | RBP binding profile | ENCODE eCLIP (150+ RBPs) | T3 |
| RNA | coding potential | GENCODE | sanity tier, not tested here |

Novelty, as far as searching showed on 2026-11-24 (not exhaustive; re-check before any claim):
* no lncRNA-specific foundation model with released weights exists;
* Miller et al., NAR Genomics & Bioinformatics 2024, found lncRNA localization predictors fall to ~60% accuracy on
  unfiltered data and flagged test-set leakage -- a named problem with no published corrective;
* DeepLocRNA (2024) validates the recipe (multi-task eCLIP pretraining -> localization) but is all-RNA, not lncRNA;
* no ML model has been trained on CRiNCL to predict which lncRNA loci are functional.

## T1 (Tue) -- is lncRNA essentiality predictable from sequence at all?

The member that would make this novel rather than incremental, so it is tested first.

Steps: obtain CRiNCL per-locus hit calls across the 7 cell lines; map loci to GENCODE genes; build three feature
families and score each under a chromosome-held-out split:
  A sequence only      k-mer spectrum of the transcript (+ length)
  B locus context      distance to nearest gene, neighbour essentiality, neighbour expression, TSS density, bidirectional/antisense/intergenic class
  C expression         the locus's own expression in that cell line
Report AUROC per family and for A+B+C, with CIs over held-out chromosomes, plus a shuffled-label control.

**Kill criterion.** The DNA arm is dropped if sequence (A) does not beat 0.55 AUROC, OR if A adds nothing over
B+C (paired CI on the difference includes 0). Either means the screen measures genomic neighbourhood, not the
lncRNA, which is the confound the CRISPRi literature itself warns about.

## T2 (Wed) -- does homology pruning actually move published predictors?

The whole benchmark claim rests on this. Take one published lncRNA localization predictor, score it on its own
split and on our homology-pruned split, same transcripts where they overlap.

**Kill criterion.** If the drop is under 0.03 AUROC, the leakage story is not ours to tell and the benchmark
becomes a minor section rather than a contribution.

## T3 (Thu) -- does an eCLIP-trained compact backbone transfer?

Train a small encoder on eCLIP binding, then probe it for localization, against a k-mer baseline and against the
frozen RiNALMo probe we already measured (+0.055 over k-mer on rbp_K562).

**Kill criterion.** If the eCLIP-trained backbone does not beat k-mers on localization by more than the seed
spread, the shared-backbone design is dropped and the family becomes independent per-task models.

## T4 (Fri) -- assemble

One results table across T1-T3. Decide go / no-go per arm against the criteria above. No new experiments.

## T5 (Sat-Sun) -- write the verdict

A short report: what was tested, what each number was, which arms survive, and what the six-week build would be
if any did. Committed whether the answer is yes or no.

## Rules for the week

* One task at a time, in order. A task is finished when its artifact is committed.
* Every number comes from a script in this directory, re-runnable, with its output committed.
* Kill criteria are not revised after seeing results.
* No model is trained for release this week.
