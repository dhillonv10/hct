# EpiSet / LongevityBench DNA-methylation tasks — results

Scope: the five DNA-methylation tasks of Insilico's LongevityBench
(`insilicomedicine/longebench`, eval-only): LB-0010 (age from the 100 lowest + 100 highest methylated sites), LB-0002
(age decade from the same input), LB-0006 (which of two donors is older, from the 160 most different sites),
LB-0190 (age from a fixed 353-CpG panel) and LB-0195 (age from a published clock's own sites, 9 clocks).
Paper numbers are Table S6 of the Cell paper as supplied; they are **not** produced by this code.

## Short version

Sections 1–9 describe the first phase (ComputAge-only training). §10 describes the second phase (ComputAge + a public
GEO corpus) and supersedes the headline numbers below where they differ.

1. **The biggest lever was data, not architecture.** Training on ComputAge plus 14,211 public GEO samples
   (28,341 samples from 262 series after de-duplication, up from 16,199 from 102) cuts the plain bag-of-CpGs ridge's LB-0010
   age MAE from 15.81 to **11.35** [10.64, 12.05] — better than the paper's best LLM (12.96); paired difference
   −4.46 [−5.37, −3.61]. The benchmark cohorts look like a random slice of GEO (27K/450K/EPIC arrays, many tissues and
   conditions), not like ComputAge's curated healthy controls (§10).
2. **With that corpus, simple models beat the paper's best LLM on LB-0010, LB-0006, LB-0190 and LB-0195** and tie on
   LB-0002. They do not reach the paper's own elastic net on LB-0195 (8.93 vs ≥ 11.2).
3. **Mixed data costs the fixed-panel and clock tasks** (ridge LB-0190 4.11 → 4.62, LB-0195 11.20 → 12.04).
   Inferring a latent group per sample and fitting one ridge expert per group recovers it on the same data: LB-0190
   4.54 → **3.95** [3.74, 4.16] (paired −0.59 [−0.72, −0.47]); LB-0195 12.10 → 11.28.
4. **No architecture advance is established.** The dense transformer (EpiSet) trained on the same corpus ties or loses to
   ridge (LB-0010 12.47 vs 11.35, worse by 1.1 [0.5, 1.7]; LB-0006 0.922 vs 0.948; LB-0190 tie; LB-0002 tie) and is better only
   on LB-0195 against the same-data ridge (11.49 vs 12.04), still short of the ComputAge-only ridge (11.20). A
   sample-routed transformer was worse than dense on validation at every matched step through step 2000 and was stopped by the
   job time limit; it was not scored on the benchmark.
5. **Validation on held-out ComputAge studies over-states LongevityBench performance** (ComputAge-only phase: extremes MAE
   13.4–14.2 on validation vs 16.0–16.3 on the benchmark; fixed-panel 3.0–3.2 vs 7.4). A cohort effect is the best-supported
   explanation (§4).
6. **Mixture-of-experts (token-level), composition-centering and an explicit linear route did not help** (§5).

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
(GEO) with many cohorts of narrow age ranges (done in §10 for the series with usable probe matrices; 71 more
screened series have age labels but only raw IDATs), (ii) training that penalises series-specific shortcuts (study-adversarial or
within-study-centred objectives, group-robust optimisation), (iii) an ensemble or linear-prior hybrid of EpiSet and
ridge, which has not been tested.


## 10. Update: training on a public GEO corpus

**Why.** The 42 benchmark series are small GEO association studies (arthritis, MS, alcohol, HIV, twins, children, buccal,
forensic fluids, ovarian cancer) on 27K, 450K and EPIC arrays; only 9 of them are in ComputAge. They look like a random
slice of GEO, so the most direct way to close the gap was a broader GEO training corpus, not a different architecture.

**Corpus** (`src/epimod/data/geo.py`, `scripts/geo_ingest.py`; compact store `lb/lite.py`).
NCBI E-utilities listed 3,842 methylation series on GPL13534/21145/23976/33022/8490; 1,335 passed metadata filters (human,
"Methylation profiling by array", 8–1,500 samples, not excluded); 269 had a parsable age for ≥ 70% of samples; 260 were
attempted, 71 had no probe matrix in the series-matrix file (raw IDATs only), and 189 series with 14,211 samples were
ingested (450K 7,578; 27K 3,834; EPIC 2,799; 19% under age 20; blood, brain, saliva, skin, liver, muscle, buccal …).
Tumour/cell-line samples were dropped by tissue text. Samples are stored as uint8 β×100 on the 9,098 CpGs that appear in any
benchmark prompt, which makes the store 20× smaller and loading seconds. After de-duplication against ComputAge and removal
of the 9 benchmark series that are in ComputAge, the training corpus is 28,341 samples from 262 series.

**Leakage audit** (`scripts/lb_audit.py`): no training sample is a GSM that appears in any prompt; none of the 42 prompt series,
nor any super/sub-series of them, is in the corpus; and none of the 1,438 LB-0190 profiles (which carry no sample or series
ids) matches any training profile (nearest RMS difference 0.037; a rounding-level match would be < 0.006).

**Results** (95% bootstrap intervals over items; point estimates where no interval was computed):

| task | metric | paper best LLM | simple, ComputAge only | simple, + GEO | routed ridge experts, + GEO | EpiSet dense, + GEO |
|---|---|---|---|---|---|---|
| LB-0010 | age MAE ↓ | 12.96 (ridge 15.90) | 15.81 [14.80, 16.83] | **11.35** [10.64, 12.05] | – | 12.47 [11.60, 13.33] |
| LB-0002 | decade bal-acc, 11 classes ↑ | 0.203 | 0.164 [0.121, 0.223] | 0.190 [0.158, 0.223] | – | 0.222 [0.160, 0.284] |
| LB-0006 | pair concordance ↑ | 0.868 | **0.958** [0.940, 0.974] | 0.948 [0.928, 0.966] | – | 0.922 [0.898, 0.944] |
| LB-0190 | age MAE ↓ | 5.82 (ridge 4.30) | 4.11 [3.90, 4.32] | 4.62 [4.40, 4.84] | **3.95** [3.74, 4.16] | 4.64 [4.43, 4.86] |
| LB-0195 | age MAE ↓ | 13.17 (EN 8.93) | **11.20** [11.12, 11.28] | 12.04 [11.96, 12.11] (EN 12.07) | 11.28 | 11.49 [11.42, 11.57] |

"Simple" = bag-of-CpGs ridge with values (LB-0010, LB-0002), pairwise logistic regression (LB-0006), ridge (LB-0190,
LB-0195). Hyper-parameters (ridge strength, C, number of groups) were chosen on held-out validation series; the validation
split changed with the corpus, so validation numbers are not comparable across phases. The LB-0190 "+ GEO" ridge of 4.62 and
the global ridge of 4.54 used in the routing comparison differ only in how the ridge strength was selected.

**Where the LB-0010 gain comes from.** On the two 27K-array cohorts (GSE20067, GSE37008; 129 items) the ridge's MAE falls from
18.8 to 10.7; on the other 371 items from 14.8 to 11.6. The gain is broad but largest where the arrays differ from
ComputAge's.

**Routed ridge experts** (`scripts/lb_routed_ridge.py`, `scripts/lb_routed_clock.py`): a Gaussian mixture on PCA scores of the
sample's own CpG profile (no tissue or platform labels) gives soft group weights; one ridge expert per group; prediction is
the weighted sum. LB-0190 selected 6 groups (alpha 300) on validation and was scored once. LB-0195 used per-clock settings;
8 of 9 clocks improved on the benchmark, Horvath v1 got worse (12.6 → 15.9). Mixtures of linear experts are an old idea;
the claim here is limited to their use for mixed-array methylation age.

**Sample-routed transformer** (`--moe-level sample`, 4 experts, top-2, one routing decision per sample from the pooled
set): validation score 0.819 / 0.742 / 0.618 / 0.607 / 0.604 / 0.595 / 0.571 / 0.581 at steps 250 … 2000 against dense 0.788 /
0.701 / 0.609 / 0.578 / 0.577 / 0.572 / 0.555 / 0.557. It was stopped at step 2000 of 3000 by the maximum background time limit
and not restarted, and not scored on the benchmark. Token-level MoE (§5) also tied dense.

**Benchmark scorings performed in this phase:** EpiSet dense + GEO (once). Baselines and routed ridge experts: one scoring of
the validation-selected setting each. Not scored: the sample-routed transformer, the MoE and linear-route runs of §5.

**Reproduce:** `scripts/lb_make_lite.py`; `scripts/geo_ingest.py discover | screen | ingest`; `scripts/lb_audit.py`;
`scripts/lb_baselines.py`, `lb_baselines_ext.py`, `lb_baselines_en.py`; `scripts/lb_routed_ridge.py`, `lb_routed_clock.py`;
`scripts/lb_run.py --name geo_mt --steps 3000 --bs-fixed 8 --age-balance 1.0`; `scripts/lb_predict.py`; `scripts/lb_report.py`.
