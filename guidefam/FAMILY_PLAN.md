# Compact RNA model family: plan, claims we can and cannot make, go/no-go gates

Status: plan only, nothing below "Next step" has been run. "Measured" = produced by code in this directory (`RESULTS.md`).
"Search only" = read in a web-search snippet this session, not verified against the paper, data or licence.

## 1. What "the family" is

One recipe applied to many RNA tasks, not one big model. Three tiers, all small enough to train on an M4 in hours:

| tier | members | size |
|---|---|---|
| T0 thermo-GBM | nearest-neighbour thermodynamics (ViennaRNA), local accessibility, k-mers, position one-hots -> LightGBM / ridge | ~10^2 - 2x10^4 parameters |
| T1 mini-nets | Tiger2D-style CNN, MiniPairFormer (structure-biased attention) | 2.5x10^4 - 2.4x10^5 |
| T2 rank ensemble | within-unit rank average over heterogeneous T0/T1 members (and published tools where the licence allows) | sum of members |

## 2. What we have measured (Cas13d on-target, siRNA; leak-free)

* Parameters 130 -> 63,000,000 (ridge ... TabDPT, RNA-FM embeddings) all land at within-gene Spearman 0.52 - 0.62. Beyond ~10^4 parameters size does not help.
* The only thing that moved the number is heterogeneous ensembling: 3-member ensemble 0.638 vs best single 0.612 (paired +0.026 [0.020, 0.032]; locked genes +0.025 [0.013, 0.036]).
* Against the released TIGER on two external on-target screens: flow-cytometry 0.656 vs 0.599 (+0.057 [0.041, 0.073]); HAP1 0.195 vs 0.111 (+0.084 [0.066, 0.101]). Against our own re-trained Tiger2D the margin is the +0.026 above.
* siRNA (Huesken -> Mixset): ours AUC 0.813 [0.778, 0.853] vs OligoFormer's published 0.816 and DeepSilencer's 0.820. Parity, and the test set is too small (+-0.04 AUC) to resolve differences of that size.
* Mismatch tolerance: the released TIGER is better than ours.
* Evaluation audits that hold regardless of model: pooled-Pearson inflation from per-gene de-normalisation, shuffled-CV inflation, checkpoint-on-test (read, not measured).

## 3. Claims

Defensible now:
1. Under gene-held-out evaluation, models with ~10^4 - 10^5 parameters match 10^7 - 10^8-parameter models on guide efficacy, and a heterogeneous ensemble beats the best single model by ~0.03.
2. That ensemble beats the released TIGER on two independent Cas13d screens.
3. Published guide-efficacy evaluations contain measurable leaks (6.1, 6.3) or unmeasured ones (6.4).

Not defensible now: "beats SOTA" on siRNA, mismatch tolerance, or any task not yet run. Do not write it until a locked-set result says so.

## 4. Task shortlist and gate status

A task joins the family only if all four gates pass: (G1) data downloadable and licence permits benchmarking; (G2) a published SOTA that we can run or whose predictions we can read; (G3) a leak-free held-out unit exists (gene, family, chromosome); (G4) headroom, see section 5.

| task | G1 data | G2 SOTA | G3 unit | expectation (my judgement, not a result) |
|---|---|---|---|---|
| Cas13d on-target | done | TIGER, DeepCas13 (done) | gene (done) | achieved: beats released TIGER, parity with retrained |
| siRNA | done (OligoFormer tables) | OligoFormer is proprietary, not run; DeepSilencer repo has no code | target chunk | parity; needs a larger test set before any claim |
| ASO (RNase H gapmers) | UNVERIFIED. Search only: ASO Atlas, 188,521 ASOs, 334 genes, from USPTO patents, "freely accessible through an online web-tool"; no download link or licence seen | OligoAI (search only: Mol Ther Nucleic Acids 2026, "299 held-out test screens", reports Spearman) | gene possible if gene ids are in the table | best candidate: same feature stack applies, large n. Dies if there is no bulk download or the licence forbids it |
| mRNA half-life | Zenodo 6326409 (18.8 GB; licence field not checked) | Saluki, code reported Apache-2.0 (search only; check LICENSE) | gene / chromosome (split not verified) | Saluki is already compact; "close to" is plausible, "beats" is unlikely |
| 5' UTR mean ribosome load | public GEO libraries | Optimus (small CNN); UTR-LM (search only: +<=5% Spearman over it) | random-sequence libraries, low leakage risk | near saturation; low novelty |
| BEACON suite | public (repo and licence not yet verified) | per-task baselines incl. RNA-FM, UTR-LM (search only: pretrained models beat prior SOTA on 8 of 13 tasks; ResNet/LSTM strong baselines) | task-specific; leakage of its splits not yet checked | cheapest breadth, weakest "SOTA" |

## 5. Protocol and kill criteria (fixed before running)

* Per task: freeze a dev/locked split before the first model is fit (as for Cas13d: 12 dev genes, 4 locked). Design on dev only; the locked set and external sets are scored once.
* G4 headroom test, dev only, one day of M4 compute: run T0 and T1 and the best published tool. Proceed if the best of T0-T2 is within 0.02 of SOTA on the primary metric; stop on that task otherwise.
* Primary metric per task: within-unit Spearman (guide tasks), Spearman / Pearson as the task's literature uses (mRNA tasks), always with a block or unit bootstrap CI.
* Family-level kill: if after three tasks the ensemble is below SOTA by more than 0.03 on two of them, drop "family beats SOTA". The remaining paper is then (a) parameter-efficiency plus (b) the evaluation audits, which is already supported by section 2.
* Never combine with, run, or port a proprietary comparator (OligoFormer). Respect TIGER's non-commercial licence; do not redistribute weights or per-guide predictions.

## 6. Next step

1. Verify ASO Atlas bulk download and licence (the bioRxiv full text was rate-limited, HTTP 429, on both attempts; retry or have the data-availability statement pasted).
2. If G1-G3 pass, export ASO Atlas with gene ids, freeze a dev/locked split, run the G4 headroom test.
3. In parallel, read the Saluki and BEACON repos' licences and splits (cheap), then decide whether either enters the family.
