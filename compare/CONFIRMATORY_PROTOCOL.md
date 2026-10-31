# Confirmatory protocol: 5-fold comparison on ASO Atlas 2.0 with the authors' out-of-fold OligoAI predictions
Written and committed BEFORE any model of ours was fitted on these folds or scored against these predictions.

## Data (all CC BY 4.0, https://github.com/barneyhill/aso-atlas-2 , DOI 10.57967/hf/8687)
* Table: `data/oligostack/processed/oligoai_train.csv.gz` (180,026 rows, 174 patents; sequence, chemistry tokens, target context, dose, cell line).
* Folds: `aso-atlas-2-release/folds/oligoai_folds.csv.gz` (helm_annotation -> test fold 0-4; patent-grouped, identical compounds deduplicated across folds).
* OligoAI predictions: `data/results/oligoai_fold{0..4}_predictions.parquet`, produced by the dataset's authors with OligoAI re-trained for each fold; each row is out-of-fold.

## Our model (frozen)
LightGBM with the parameters in `compare/aso_compare.py`; features = `guidefam.aso.features` (sequence, gapmer chemistry, experiment, thermodynamics, target-site accessibility);
target = within-screen rank-Gauss computed on training rows only. For fold k it is trained on all rows whose compound is not in fold k and predicts fold k. No tuning on these folds.

## Combination (frozen)
MIX = mean of the within-screen rank percentiles of ours and OligoAI, weight 0.5 (the weight was specified before any result in `compare/results/aso_ensemble_posthoc.txt` and is not tuned here).

## Metric
Within-screen Spearman; screen = (custom_id, cell_line, dosage) with >= 20 ASOs; mean over screens; 95% CI by bootstrap over patents (2,000 draws).

## Primary outcome and decision rule
Primary: MIX - OligoAI, pooled over the five folds. Success = lower 95% bound > 0. If not met, the claim "the compact model adds signal to OligoAI" is withdrawn.
Secondary (descriptive): ours - OligoAI; in how many of the 5 folds MIX > OligoAI; per-table Spearman (OligoAI's own definition, `custom_id`); median top-1 / mean target.
Strata (descriptive): (A) rows from the 102 patents absent from the Aug-2025 ASO Atlas table; (B) rows whose ASO sequence is absent from the Aug-2025 table.

## Caveats stated in advance
* This is an independent REPLICATION, not fresh data: the rows overlap with the table on which the combination was first noticed; what differs is the split protocol (patent-grouped, compound-deduplicated), OligoAI re-trained by its authors per fold, and 102 additional patents.
* 94.8% of the rows from new patents carry an ASO sequence that is already in the Aug-2025 table, so "new patent" does not mean "new compound"; stratum (B) addresses that.
* Genuinely new genes have almost no usable rows (16 rows in 2 genes have a target context), so a new-gene test is not possible with this release.
* The authors' predictions are used as released; we did not re-run OligoAI.
