# EpiSet / LongevityBench DNA-methylation tasks — results

Scope: the five DNA-methylation tasks of Insilico's LongevityBench
(`insilicomedicine/longebench`, eval-only): LB-0010 (age from the 100 lowest + 100 highest methylated sites), LB-0002
(age decade from the same input), LB-0006 (which of two donors is older, from the 160 most different sites),
LB-0190 (age from a fixed 353-CpG panel) and LB-0195 (age from a published clock's own sites, 9 clocks).
Paper numbers are Table S6 of the Cell paper as supplied; they are **not** produced by this code.

## Short version

1. **The new transformer (EpiSet) does not beat simple baselines on any task beyond bootstrap noise.**
   It is tied with ridge / logistic regression on LB-0010, LB-0002 and LB-0195, and worse on LB-0006
   (−0.028 concordance, 95% CI [−0.048, −0.008]) and LB-0190 (+3.3 y MAE).
2. **The simple baselines beat the paper's best LLM on three of five tasks** (LB-0006, LB-0190, LB-0195), using only
   the training data below and no language model. On LB-0195 they do not reach the paper's own elastic net (8.93).
3. **Nothing here beats the paper's LLM on LB-0010** (age MAE 12.96 vs 15.8–16.3 for everything I trained); LB-0002 is
   a statistical tie (0.213 [0.18, 0.24] vs the paper's 0.203).
4. **Validation on held-out ComputAge studies over-states LongevityBench performance.** Extremes-task MAE is 13.4–14.2 on my
   validation studies and 16.0–16.3 on the benchmark; fixed-panel MAE for EpiSet goes from 3.0–3.2 (validation) to 7.4
   (benchmark), for ridge from 3.0 to 4.1. A cohort effect is the best-supported explanation (§4); I have not shown it is the only one.
5. **Mixture-of-experts, composition-centering and an explicit linear route did not help** (§5).

## 1. Results (500 items per pairwise / extremes task, 1,438 for LB-0190, 57,944 for LB-0195)

95% bootstrap intervals resample items; "even" is the half of the items (even prompt index) whose CpG identities were
never used to derive candidate pools (§3).

| task | metric | paper best LLM | ridge/logit baselines | EpiSet extremes-only | EpiSet multi-task |
|---|---|---|---|---|---|
| LB-0010 | age MAE ↓ | 12.96 (ridge 15.90) | 15.811 [14.798, 16.826] (even 16.532) | 16.034 [15.076, 17.055] (even 16.952) | 16.313 [15.309, 17.404] (even 17.130) |
| LB-0002 | decade bal-acc (11 cl.) ↑ | 0.203 (NB 0.13) | 0.164 [0.121, 0.223] (even 0.130) | 0.207 [0.137, 0.255] (even 0.190) | 0.213 [0.181, 0.241] (even 0.219) |
| LB-0006 | pair concordance ↑ | 0.868 (NB 0.61) | 0.958 [0.940, 0.974] (even 0.956) | – | 0.930 [0.906, 0.952] (even 0.932) |
| LB-0190 | age MAE ↓ | 5.82 (ridge 4.30) | 4.114 [3.902, 4.319] (even 4.236) | – | 7.397 [7.179, 7.622] (even 7.549) |
| LB-0195 | age MAE ↓ | 13.17 (EN 8.93) | 11.199 [11.115, 11.278] (even 11.144) | – | 11.180 [11.107, 11.252] (even 11.130) |

Table columns: *ridge/logit baselines* = ridge (LB-0190, LB-0195), bag-of-CpGs ridge with values (LB-0010, LB-0002), pairwise logistic regression (LB-0006); *EpiSet extremes-only* = age-balanced extremes model; *EpiSet multi-task* = dense, extremes + fixed + clock + pair views. The linear-route variant was stopped before finishing and was never scored on LongevityBench (§5).

**§1a. LB-0002 decade head vs age-posterior decade** (balanced accuracy over 11 classes, all items / even): extremes-only 0.207 / 0.190 (head), 0.200 / 0.191 (posterior); multi-task 0.213 / 0.219 (head), 0.137 / 0.157 (posterior).

The paper's best LLM numbers are from the paper's own, different training data (§6).
LB-0002 is balanced accuracy averaged over all 11 decade classes (chance 0.09); EpiSet's number is its decade
classification head; the decade implied by its age posterior is lower for the multi-task model (§1a).

## 2. What was trained and what was selected

* **EpiSet** (`src/epimod/lb/episet.py`): a set transformer over (CpG, β) tokens. A token is
  id-embedding + CpG-specific slope × z(β) + a Fourier value MLP + a *view* embedding, so one model reads all four LB
  input formats (lowest/highest lists, a fixed panel, a clock's sites, the two views of a donor pair). Heads: Gaussian
  age (Horvath-transformed), 11-way decade, and an antisymmetric pairwise logit (μ_A − μ_B)/τ. 1.54 M parameters, 4
  layers, d = 96. Variants: optional token-level sparse MoE FFN (4 experts, top-2, load-balancing loss) and an optional
  explicit per-CpG linear route.
* **Training data:** ComputAge Bench samples (train + benchmark splits, de-duplicated, QC'd, 16,199 samples from 102
  series) **excluding every one of the 42 series that appear in any LongevityBench DNAm prompt**; 13,539 samples train and 2,660 samples
  (9 whole *series*, 16% of the samples) validate. All LB-format inputs are generated from real
  profiles (extreme lists inside candidate pools, most-different-site lists, fixed-panel and clock columns), with 2-decimal
  rounding, token dropout and a 0–0.04 tail-value shift.
* **Selection:** early stopping / checkpointing and all hyper-parameter choices (baseline ridge strengths, sampling
  scheme, dense vs MoE) used the validation series only. Rule fixed before the MoE run finished: take MoE only if its
  best validation score beat dense by > 0.01; it did not (0.3870 vs 0.3872).
* **LongevityBench evaluations performed (all logged in `lb_eval_log.jsonl` in the run directory):** EpiSet extremes-only
  (age-balanced sampling; chosen over natural sampling on validation, 0.887 vs 0.920) and EpiSet multi-task dense.
  Not scored on LongevityBench: natural-age extremes run, the MoE run (stopped at step 1250 of 2000 by the background
  time limit) and the linear-route run (stopped at step 1500 of 2000). Baselines: ridge / HGB (fixed panel), per-clock ridge and
  elastic net, pairwise logistic regression at three strengths (chosen on validation), bag-of-CpGs ridge with and without
  values (extremes; 10 settings compared on validation).

## 3. Protocol notes and leakage handling

* **Series overlap:** all 42 GEO series in any LB DNAm prompt are removed from training (`lb_series`). GSM-level overlap and
  value matching audits were run earlier and were clean.
* **Candidate pools** for the extremes and pair tasks are not given by the benchmark; I estimated them from the CpG
  *identities* (no values, no labels) in the **odd-numbered** prompts, and report the **even** half as the cleaner
  number. Pools of 1,004 (low) / 833 (high) / 4,772 (pair) CpGs. Even and odd halves agree to within the intervals above.
* **Input-format fidelity:** the real LB lists are less extreme than lists generated from my profiles (rank-1 low value
  0.02–0.03 vs 0.00–0.02; rank-100 high 0.96 vs 0.97) and two of 16 series (GSE20067, GSE37008; 129 of 500 items) have
  a very different CpG composition. I compensated only with the tail-value shift augmentation; I did not tune on LB labels.
* HGB varies by ±0.1 y between runs (3.90 / 3.96 / 4.03 on LB-0190); ridge is deterministic and is the reported baseline.

## 4. Why validation does not carry over: a cohort effect

On LB-0010 the extremes model predicts 56.1 y on average for items whose mean age is 42.9 y (bias +13.2 y, correlation
0.67, prediction SD 17.0 vs age SD 18.7). The bias is positive in all 15 series with the most items (+2 to +29 y) and largest for the
young: +17 y for ages 0–20, +22 y for 20–30, −3 y for 70–80. On LB-0190 the multi-task model is highly correlated with
age (r = 0.95) but compressed (prediction SD 16.6 vs 21.2): +9 y for under-30s, −8 to −10 y for over-70s. Training data
are 6% under 20 and 22% in their 20s–30s, so this is not just thin tails. The pattern fits cohort-specific signatures
acting as age proxies: many training series span a narrow age range, a high-capacity model can use series-specific
batch signatures instead of age, and ridge cannot. This is a hypothesis consistent with the diagnostics
(`scripts/lb_diag.py`, `scripts/lb_inputs_check.py`), not a demonstrated mechanism.

## 5. Things that were tried and did not help

* **Token-level sparse MoE** (4 experts, top-2): validation score tied with dense at every comparable step
  (0.3952 / 0.3946 at step 750, 0.3870 / 0.3872 at 1000), about twice the wall-clock. I would not expect token-level MoE to
  help with 13.5 k samples; it was run because it was asked for, not because the data suggest it.
* **Composition-centering** (`scripts/cc_feasibility.py`, `scripts/cc_linear.py`): subtract the cell-type composition
  (NNLS deconvolution on 6 lineages, 425 reference CpGs available) from the profile before a linear age model.
  With signatures learned from training data only: no gain (purified-cell MedAE 5.77 → 5.67, bias worse). With *oracle*
  signatures (built from all purified samples, test included, so leaky): purified-cell MedAE 5.77 → 4.25 for centering
  plus composition, nearly all from monocytes (6.3 → 3.4); centering alone is worse (6.35); B-cells worse (6.3 → 10.0);
  mixed whole-blood-like samples slightly worse (4.42 → 4.59). I did not build the differentiable-deconvolution architecture.
* **Explicit linear route** (per-CpG weights × z(β), per-view gain): learns much faster early (step-250 validation score
  0.68 vs 0.87 for dense) but is worse on validation (0.413 vs 0.387 at step 1500 of 2000). The run was stopped at step 1500
  by the user and was **not scored on LongevityBench**, so the hypothesis that a linear route fixes the LB-0190 shrinkage is untested.
* **Natural vs age-balanced sampling** for extremes: 0.920 vs 0.887 validation score (single seed, 9 series; small).

## 6. Comparability with the paper

* The paper's training data and any overlap with these series are unknown to me. If its training set contained other
  samples from these same series, its numbers are partly within-cohort and mine are not; my exclusion of all 42 series is
  deliberately stricter and probably costs accuracy (the two clocks with ≤ 150 sites, Hannum and Lin, have 4.2–5.2 y
  validation error but 8.2–8.3 y benchmark error for an elastic net).
* I re-implemented scoring from the task descriptions and the supplied table. Metric details I could not verify:
  LB-0002 balanced accuracy over 11 classes (I report this; over the 9 classes present it is higher), and whether
  the paper reports a decade head or a posterior.
* One training seed per configuration; single validation split of 9 series. Differences smaller than the bootstrap
  intervals above should be read as ties.

## 7. Reproducing

```
EPIMOD_DATA=... python scripts/lb_run.py --name mt_dense --steps 2000 --bs-fixed 8 --age-balance 1.0 --no-lb-eval
python scripts/lb_run.py --name mt_dense --tasks extremes,fixed,clock,pair --eval-only
python scripts/lb_predict.py runs/lb/mt_dense/model.pt preds/mt_dense.npz
python scripts/lb_baselines.py preds/baselines.json; python scripts/lb_baselines_ext.py preds/baselines_ext.json
python scripts/lb_baselines_en.py preds/baselines_en.json
python scripts/lb_report.py report.md "ridge/logit baselines=preds/baselines_all.npz" "EpiSet=preds/mt_dense.npz" --ref "ridge/logit baselines"
```

## Paired differences

Paired bootstrap difference (same resampled items; metric as above):

| task | comparison | mean diff | 95% CI |
|---|---|---|---|
| LB-0010 | EpiSet extremes-only − ridge/logit baselines | +0.223 | [-0.656, +1.161] |
| LB-0010 | EpiSet multi-task − ridge/logit baselines | +0.499 | [-0.421, +1.439] |
| LB-0002 | EpiSet extremes-only − ridge/logit baselines | +0.043 | [-0.021, +0.110] |
| LB-0002 | EpiSet multi-task − ridge/logit baselines | +0.047 | [-0.021, +0.100] |
| LB-0006 | EpiSet multi-task − ridge/logit baselines | -0.028 | [-0.048, -0.008] |
| LB-0190 | EpiSet multi-task − ridge/logit baselines | +3.287 | [+3.056, +3.522] |
| LB-0195 | EpiSet multi-task − ridge/logit baselines | -0.019 | [-0.100, +0.064] |


## 8. Literature check (novelty is **not** established)

I searched for prior work on cell-composition-routed mixture-of-experts clocks, differentiable deconvolution plus
clock, and set-transformers over CpG sets; I found no direct match, but the search was a handful of queries and I did
not read the closest papers in full. Related work I found: AgeMoE (federated MoE for methylation age,
[arXiv 2609.10108](https://arxiv.org/abs/2609.10108)); cell-type-specific clocks, Tong et al. 2024
([Aging 206184](https://www.aging-us.com/article/206184)); DeepMAge saliva deconvolution, Galkin et al. 2021;
MethylGPT ([bioRxiv 2024.10.30.621013](https://www.biorxiv.org/content/10.1101/2024.10.30.621013)); CpGPT
([bioRxiv 2024.10.24.619766](https://www.biorxiv.org/content/10.1101/2024.10.24.619766)). Given the results above, I
would not claim a new architecture advance on the strength of this work.

## 9. What would most likely help

The binding constraint appears to be cohort generalisation and data volume, not model class: (i) more public arrays
(GEO) with many cohorts of narrow age ranges, (ii) training that penalises series-specific shortcuts (study-adversarial or
within-study-centred objectives, group-robust optimisation), (iii) an ensemble or linear-prior hybrid of EpiSet and
ridge, which has not been tested.
