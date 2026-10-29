# Evaluation protocol (what is held out, what is scored, what was audited)

Everything here is checked by code in this directory; numbers quoted come from the scripts named next to them.

## 1. Units of generalisation

| task | held-out unit | why |
|---|---|---|
| Cas13d on-target (TIGER HEK293T screen) | **gene** (leave-one-gene-out, 16 genes) | guides tile each transcript at ~1-nt steps; neighbouring guides share 22/23 nt and have correlated activity (activity correlation r = 0.61 at 1 nt, 0.39 at 2-3 nt, 0.21 at 4-7 nt, 0.10 at 8-15 nt, < 0.05 beyond 16 nt; measured on the 12 development genes), so any split that puts a gene's guides on both sides leaks |
| siRNA (OligoFormer tables) | **target mRNA** (contiguous chunks of the gene-ordered file, ~target-held-out) | the tables give no gene ids; rows are ordered along each mRNA, so contiguous chunks keep nearly every gene whole. OligoFormer's own intra-dataset CV is a *shuffled* 5-fold split |
| external | whole other screens / datasets | flow-cytometry (CD46/CD55/TFRC) and HAP1 (1,082 genes) for Cas13d; Mixset and Takayuki for siRNA |

Normalisation (`cv.activity_target`) is computed from the **training rows only**; the held-out gene's labels never enter its own target, features or scaling.

## 2. Metrics

* **within-gene Spearman** (headline): mean over genes of the rank correlation between measured and predicted activity. It answers "which guides do I order", and is invariant to any per-gene offset or scale, so it cannot be inflated by between-gene differences in knockdown depth.
* AUROC of active vs inactive (threshold = 1st percentile of a normal fitted to the non-targeting LFCs, TIGER's definition).
* siRNA: PCC, AUC, PRC as OligoFormer reports them (see §4 for the two label conventions).
* CIs: block bootstrap over 50-nt target regions (Cas13d) or contiguous 25-siRNA / overlap-linked blocks (siRNA); paired bootstrap for differences.

Replicate noise ceiling (HEK293T): single-replicate within-gene Spearman between replicates = 0.57, so the mean of three has reliability ~0.80 and a perfect predictor would correlate ~0.89 with the observed mean.

## 3. Development / locked discipline

* Cas13d: four HEK293T genes (CDC5L, POLA1, SF3B1, WDR75; every 4th gene alphabetically) and both external screens were **not used for any design decision**. The untuned baseline ladder (default settings) had been run on all 16 genes before this rule was fixed; no choice was derived from it.
* siRNA: design decisions on Huesken only; Mixset and Takayuki were first read by `final_sirna.py`. After that, `sirna_leakage_lodo.py` re-used them for training-set composition and split analyses only, with the design unchanged.
* Consequence worth stating: the Cas13d "GBM + Konermann" combination was chosen on the 12 development genes; its test is the 4 locked genes (flow-cytometry is *not* independent for it, see §5).

## 4. siRNA label conventions (found by reading the code, they differ by path)

OligoFormer's `data/*.csv` carry two targets: `label` (normalised efficacy) and `y` (binary).

* `test.py` (inter-dataset: train one set, test another) binarises `label > 0.7`. Used for Huesken -> Mixset/Takayuki. Mixset: 58 % positive; Takayuki: 27 %.
* `train.py` (intra-dataset k-fold) uses the file's own `y`. For Huesken `y = 1` iff the *un-normalised* efficacy >= 0.7 (**~50 % positive**, a threshold of 0.522 on the normalised column). `label > 0.7` would give 10 % positive and a different AUC/PRC; PRC is not comparable across the two.

`sirna.binary(d, "inter"|"intra")` encodes this. (Mixset's `y` column does not follow a clean threshold on either `label` column; we use the `test.py` convention for it.)

## 5. Published-tool predictions: what they can and cannot support

`predictions (other models)/` in the TIGER repo holds DeepCas13, a 30-mer model filed as "Konermann", and a random forest (`Gene_Holdout` column = leave-gene-out on this screen). We map them to our guides exactly (all 9,985 PM guides found; orientation checked, `comparators.py`). Caveats:

* We cannot verify what those tools were trained on. Konermann-rules and the random forest were developed on the CD46/CD55/TFRC flow-cytometry screens, so that set is **not an external test for them** (it is for models trained on HEK293T).
* The "Konermann" folder's original paper attribution is inferred from the TIGER README ("Wei et al. 2021", "Cheng et al. 2023"); treat the name as a label for those prediction files.

## 6. Audits of the published evaluation recipes

Two of the three audits below found a measurable effect (6.1, 6.3); one found none (6.2).

### 6.1 TIGER de-normalises with the held-out gene's own labels  (`scripts/audit_denorm.py`)

TIGER trains on per-gene-normalised LFC, then maps predictions back with the gene's median and (q90 - q10) *computed from that gene's observed LFCs, including for a held-out gene*, and reports pooled correlations. For the same leave-one-gene-out predictions (our GBM, all 16 genes):

| mapping back to LFC | pooled Pearson | pooled Spearman | within-gene Spearman |
|---|---|---|---|
| TIGER recipe (held-out gene's own median/scale) | 0.717 | 0.627 | 0.598 |
| one constant median/scale for every gene (what a new gene gets) | 0.632 | 0.593 | 0.598 |
| *no model*: each gene's true median alone | 0.279 | 0.238 | - |

So about 0.09 of pooled Pearson comes from label information about the test gene. Within-gene metrics are unaffected. (An earlier version of this audit mis-aligned rows and reported ~0 for the constant mapping; decomposing the numbers exposed it, and the fix is the `reindex(x.index)` in `audit_denorm.py`.)

### 6.2 TIGER early-stops on the evaluation fold  (`scripts/audit_peek.py`)

`predictor_validation.py` passes the held-out fold as the validation set and `PerformanceCallback` restores the best-`val_loss` epoch (patience 100, up to 2000 epochs). Measured on our re-implementation (3 genes x 2 seeds): clean 0.573 vs stopping-on-the-held-out-gene 0.575, difference +0.002 with seed noise of about +-0.01, so **no detectable effect** on within-gene Spearman (`RESULTS.md` section 5). A leak in principle, negligible in practice here.

### 6.3 OligoFormer's intra-dataset CV is a shuffled split  (`scripts/sirna_leakage_lodo.py`)

Same frozen model on Huesken: shuffled 5-fold (their recipe) PCC 0.668 / AUC 0.846; contiguous 5-fold 0.644 / 0.836; contiguous 10-fold 0.646 / 0.835. A shuffled split flatters the model by ~0.02 PCC. Whether it flatters OligoFormer more than a simple model cannot be said without re-training it.
