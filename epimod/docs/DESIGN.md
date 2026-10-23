# EpiMod — design notes

EpiMod is a small (~1 M parameter), CPU-trainable transformer for DNA-methylation age. It is built
around four ideas that a text-LLM approach to methylation does not naturally provide, and it is
evaluated on studies it never saw, with the ComputAge Bench protocol.

## 1. What this is measured against (and what I could and could not verify)

**Insilico's paper.** *The End of Aging Clocks: Training Foundation Models to Reason in Aging and
Longevity* (Zhavoronkov, Galkin et al., bioRxiv, 30 Mar 2026, doi:10.64898/2026.03.28.714980).
What I could verify from the abstract (retrieved through the bioRxiv/EuropePMC/CrossRef APIs):

* **Longevity-LLM v0.1** is a **Qwen3-14B** model fine-tuned by supervised and reinforcement
  learning on DNA methylation, proteomics, clinical biomarker and RNA data;
* after reinforcement fine-tuning it reports a **4.34-year MAE in epigenetic age prediction**,
  "surpassing the Horvath multi-tissue clock"; it also ranks highly on *Longevity Bench*;
* the stated motivation is that specialist clocks are "fixed-modality", use "a predetermined
  feature set" and give "limited biological interpretation".

What I could **not** verify: the full text is behind Cloudflare for this environment, so I have not
read how methylation is serialised for the LLM, how many CpGs it sees, what the 4.34-year number is
computed on (dataset, tissue, mean vs median error), or what interpretability it offers.
**Therefore no claim below says Insilico "lacks" a feature; the claims are about what EpiMod does,
tested in ablations.** Their 4.34-year figure is not comparable to anything here until the
evaluation set is known.

Other methylation transformers exist and are cited by the paper (MethylGPT, CpGPT, Precious1GPT /
Precious2GPT). EpiMod does not claim to be the first methylation transformer.

## 2. Architecture

```
beta[B,N] (NaN = missing)
   │ z-score per CpG, missing -> 0
   ▼
CpG tokens e_i = ctx(chrom, multi-scale position, island, gene region, regulatory class,
                     enhancer, DHS, probe design) + id_i (hidden at random during training)
   │
   ▼  sample-independent soft assignment A[K,N]   (each module = a weighted CpG set)
module activations m_k = LN( Σ_i A_ki ( z_i·u_i  +  rbf(beta_i)·W_i ) )     [B,K,d]
linear route        l_k = Σ_i A_ki s_i z_i                                  [B,K]
   │
   ▼  Transformer over the K module tokens (K ≈ 24)
h_k  [B,K,d]
   ├─ age   : mu_t = b + Σ_k ( <w_k,h_k> + l_k )      exact additive decomposition
   ├─ sigma : predictive sd of age (Gaussian NLL)     calibrated uncertainty
   ├─ sex   : auxiliary classification
   └─ recon : masked-CpG reconstruction (self-supervised)
```

Cost is O(B·N·K·d) with no `[B,N,d]` tensor, so 12 k CpGs train on four CPU cores.

### Features, the ablation that tests each, and what was measured

Numbers: held-out benchmark healthy controls, MedAE in years; full model 3.93, seed noise ≈ 0.005
(see `results/RESULTS.md`).

| Feature | Why | Ablation flag | Measured outcome |
|---|---|---|---|
| **Genome-aware CpG tokens** (chromosome, multi-scale position, island/shelf, gene region, regulatory class, enhancer, DHS, probe design) with *id dropout* | share strength between related CpGs; allow moving trained weights to another CpG set (`EpiMod.rebind`) | `use_genomic_context=False`, `use_id_embedding=False` | **No accuracy gain on seen CpGs** (off: 3.84, i.e. 0.09 better). **Does enable prediction from CpGs never seen**: MedAE 8.5 y / r 0.75 vs 15.0 y for a constant, destroyed by shuffling the annotations or the coordinates (r 0.32 / 0.53) |
| **Epigenetic modules**: K sample-independent soft CpG sets feeding a small transformer | inspectable units; 10⁴ CpGs without `N²` attention | `n_modules`, `n_layers=0` | Module transformer helps (+0.17 when removed). Modules are structured (CpG-island/regulatory context, developmental TF loci) but **diffuse** (median effective size 4.8 k of 12 k CpGs) |
| **Exact additive age decomposition** over modules | attribute condition-specific acceleration to modules without post-hoc approximation | structural | Works (tested to 1e-6). Age-adjusted shifts: HIV +3.5 y over 16 of 24 modules, Werner +4.5 y, TB +2.4 y |
| **Predictive distribution** (mean, sd; Gaussian NLL) | per-sample uncertainty | `distributional=False` | Slightly better accuracy (+0.13 when removed). Intervals ~calibrated on held-out training studies after rescaling, **overconfident on new studies** (76–86% for nominal 90% on blood) |
| **Masked multi-task training + real-profile augmentation** | robustness to array/normalisation differences | `aug=False`, `w_ssl=0` | Augmentation helps (+0.13 when removed). **Masked-CpG reconstruction: no measurable effect** (+0.05) |
| **Linear route warm-started from a study-grouped ridge fit** | a wide linear sum under Adam over-shoots ≈√N (see notes) | `linear_skip=False`, `warm_start_linear=False` | Largest effect: +0.49 y when removed |
| **Cross-fitted acceleration head** (`train/accel.py`) | chronological accuracy does not imply that a clock reads older in disease | — | **Did not help** (mean acceleration +0.33 → +0.32/+0.35 y; MedAE worse) |
| *Cell-type-balanced sampling* (added post hoc) | purified cell types dominate the error | `celltype_balance_power` | Overall 3.93 → 3.77; purified cells 7.06 → 5.79 y; whole blood unchanged |

## 3. Data

ComputAge Bench (Kriukov et al., CC BY-SA 4.0): 46 healthy-control **training** studies
(7,419 samples; all blood; 450K / EPIC / EPICv2) and 65 **benchmark** studies (10,410 samples;
blood and saliva; healthy controls plus aging-accelerating conditions). Benchmark studies are
never used for panel selection, normalisation statistics, model selection or training.

* QC: one training sample with age 891 (a unit error) is dropped; probes with common SNPs in or near
  the probe are excluded from the data-driven part of the panel; Y-chromosome probes are excluded.
* Panel (16 k CpGs, priority-ordered; the model uses the first 12 k): published-clock CpGs
  (so published clocks can be scored on the same matrix), then CpGs ranked by *within-study*
  meta-analytic age association (so between-study batch differences cannot masquerade as age
  signal), then 400 sex-informative X-chromosome CpGs, then high-variance autosomal CpGs.
* Validation for early stopping is a held-out set of *training studies*. Panel selection saw those
  studies' ages, so validation numbers are mildly optimistic; **benchmark numbers are not**.

## 4. Notes from development (things that did not work, or that surprised me)

* A learned per-CpG linear route initialised near zero made training diverge (Huber loss in the
  hundreds in epoch 1). Under Adam, a weight vector summed over ~10⁴ CpGs moves the output ≈ √N too
  fast. Fix: closed-form ridge warm start (λ chosen by study-grouped CV) and a 100× smaller learning
  rate for those weights. Without the linear route the model still trains (4.42 y at 15 epochs) but
  is worse.
* An early 6-epoch checkpoint looked much better at detecting accelerated aging (14 AA2 hits) than
  the final model (5). The per-test effects were nearly identical (Spearman 0.98); the count is a
  thresholded statistic and many tests sit at the cutoff. Report effect sizes next to hit counts.
* Benchmark error is dominated by study- and cell-type offsets (purified monocytes / T cells, saliva,
  27K array), not by within-study noise. That was visible only after breaking errors down by cell type.
* Data issues found and handled: a training donor aged "891" (dropped); 6 samples registered under
  two benchmark studies (deduplicated); probes with common SNPs flood a variance-ranked panel (excluded);
  two benchmark studies are sibling series of training studies (reported with and without);
  only ~73% of training samples contain X-chromosome probes (X CpGs allowed at ≥70% coverage).
