# Distilled compact RNA backbone family: plan, prior art, pre-registered decision rule

Status: PLAN ONLY. Nothing below has been run. Written 2026-11-01, before any distillation code exists, so the decision rule in section 5 is fixed before results.
**Superseded in part by `COLAB_PLAN.md` (v2, same day):** its research questions, arms, gates and decision rules (section 6 there) replace sections 1, 4 and 5 below; prior art, licences and design choices below still stand.
Evidence labels: **[read]** = text read directly this session; **[figure]** = read off a figure image, not a table; **[search]** = web-search snippet only; **[ours]** = measured by code in this repo; **[judgement]** = my estimate, not a result.

## 1. The claim we would be testing (and the claims we would NOT make)

Question: can a general-purpose RNA encoder with 0.25M-20M parameters, distilled from the strongest open ncRNA language model, keep most of the teacher's benefit over simple baselines, on established benchmarks, with a recipe that trains on a laptop (M4) after a one-off Colab step?

Would claim if the rule in section 5 is met: "a size ladder of compact encoders retains X% of the teacher's gain over a k-mer / small-supervised baseline at 1/30 to 1/2500 of its size, and a distilled 20M student matches or beats the independently pretrained RiNALMo-micro (33.5M)."
Would NOT claim: "foundation model", "beats SOTA", or anything on tasks not in the pre-registered set.

## 2. Prior art (read before we claim novelty)

| work | domain | teacher -> student | what they report | released? | status |
|---|---|---|---|---|---|
| HelixNano-mRNA (arXiv 2604.08574, ICLR 2026 MLGenX tiny-paper track) | mRNA | Evo2-1B -> ~5M hybrid Mamba-attention | embedding matching (cosine 1.0 + MSE 0.1), 2 matched layers (5th, 8th student vs 12th, 25th teacher), logit KD "unstable"; 27M RefSeq mRNAs, 4xA100, 24 h; mRNA-bench [read] | weights on HF (not opened), no code link | **[read]**. Figure: student mean z ~0.97 vs Evo2-1B ~1.4 vs Orthrus-1M ~0.76; per task the student is below the teacher on 8/9 tasks in the bar chart (half-life 0.55 vs 0.64, MRL 0.37 vs 0.43), roughly equal on 2 noisy tasks **[figure]**. The same plot shows an "Orthrus+MLM" point (no distillation, ~1M params) above the teacher **[figure, unlabeled in text]** |
| HAD (arXiv 2505.20836) | human DNA | NTv2-500M -> 1.1M GDN+attention | MSE on visible tokens + masked reconstruction, 15% k-mer group masking; beats teacher on 12/18 NT-benchmark rows, loses all 6 promoter/splice rows (up to -12.3 pts); text overstates "consistently outperforms" [read] | no code, no weights, no compute reported | **[read]**; DNA only |
| BIO-Distiller (ICLR 2026 MLGenX workshop paper; EPFL) | RNA, 6 BEACON tasks (NCR, APA, MRL, PRS OFF/ON/ONF) | RNA-FM, RNA-MSM, UTR-LM (MRL), UTR-LM (TE+EL), also HyenaDNA-tiny and ESM-2 t6-8M -> a **task-specific 3-layer CNN** per task | feature KD (fitnet; "eucosine" = Euclidean + cosine on penultimate layers, with a learned projection) beats logit KD; **fine-tuning the teacher on the task first helps significantly**; best distilled CNN beats the tuned CNN baseline by ~2% R2 on regression and ~10 MCC points on NCR (84.15 -> 94.64); ensembling teachers helps (EnsAvg-RNA best on APA/MRL/ON/ONF). Baseline CNN tuned with 100 ASHA configs **[read, full PDF]** | code: github.com/epfl-lts2/biod-mlgenx (stated) | **[read]**. Differences from our plan: their student is a per-task CNN, not a pretrained general backbone; no size ladder; teachers are the older ~100M models, not RiNALMo-giga; each student distils from a teacher fine-tuned on that task. Their results are the closest prior art for our loss choice and for "tuned supervised baselines are strong" |
| RNAMergeDistill (Japanese Soc. Mol. Biol. 2025 abstract) | RNA LM | multi-teacher | title and authors only [search] | unknown | not read |
| BEACON-B, RNAMamba-14M, BiRNA-BERT, LncRNA-BERT | RNA | none (trained directly) | small models reaching or beating larger ones on some tasks [search] | various | the real competitors at small size |

Reading: distillation of big nucleotide models into tiny students is NOT an empty niche (mRNA, DNA, and a workshop RNA paper exist). What I did not find (not exhaustive): (a) a size ladder distilled from the strongest open **ncRNA** model with code and weights; (b) homology-aware leakage controls on the distillation corpus and the benchmarks; (c) a controlled ablation of which recipe ingredients matter. Those are the only defensible differentiators.

## 3. Design

**Teacher.** RiNALMo-giga (650M), from the **official** lbcb-sci repo and Zenodo weights: code Apache-2.0, weights "CC BY 4.0" per its README **[read]**. Do NOT load through the `multimolecule` port: its cards declare AGPL-3.0 and its licence FAQ says fine-tuned weights "derived from" hosted weights must be conveyed under the AGPL **[read]**. (Our earlier note "inherits AGPL-3.0" in `lncrna_fm/PLAN.md` came from using the port; the official weights do not say that.) Distilling from CC BY 4.0 weights plausibly allows a permissively licensed student with attribution; that is a reading of the text, not legal advice, and the Zenodo record's own licence field was not opened. ERNIE-RNA and RNA-FM have MIT code but no stated weights licence **[read]**, so they are not primary teachers. AIDO.RNA licence not checked.
Our plain-PyTorch RiNALMo re-implementation (`compare/oligoai_score.py`, validated against a real checkpoint: Pearson 0.633 on the fine-tuned OligoAI model **[ours]**) can serve as the loader once the Zenodo pretrained file format is checked.

**Students.** Ladder 0.25M / 1M / 5M / 20M, single-nucleotide tokens. Candidate backbone: our `LncFM` conv stem + RoPE transformer (stride-4 stem keeps attention cheap) **[ours]**; one alternative (BiGRU/Mamba-style) only if time allows. One architecture family across sizes so the ladder is clean.

**Corpus.** RNAcentral sequences (public; size and licence to verify), length 16-512 nt, clustered (MMseqs2 linclust) at ~80% identity, sample ~0.3-0.5M sequences balanced across RNA types. **Leak control (differentiator):** remove every corpus sequence within the clustering threshold of any test or locked sequence in the evaluation sets before targets are computed. The teacher itself has seen RNAcentral-style data; students inherit that, and we say so.

**Targets (the cost problem).** Per-token 1280-d teacher embeddings for 0.4M x ~200 nt would be hundreds of GB. Trick to be tested, not assumed: cache **PCA-compressed per-token targets** (e.g. 64-128 dims fitted on a 20k-sequence sample, from 2 layers) plus the mean-pooled full vector. One Colab pass computes and stores them (~10-30 GB); every student then trains from disk on the M4 with no teacher in the loop.

**Loss and tricks (from the prior art above; each is an ablation arm, not an assumption).**
1. Embedding matching = cosine + small MSE on post-norm hidden states; no logit KD (HelixNano reports logits unstable) **[read]**.
2. Two matched layers (mid and last) vs one **[read: "better with two layer matching", not an extensive analysis]**.
3. HAD-style group masking + masked-token reconstruction as an auxiliary head **[read]**.
4. Contrastive auxiliary: positives = same Rfam family / aligned orthologs (Orthrus-style biology-informed augmentation; the figure suggests contrastive + MLM can beat plain distillation at ~1M **[figure]**).
5. Data efficiency: clustered sampling vs random at equal token count.
6. Teacher-size check: HAD found only the 500M teacher helped, 50M/100M did not **[read]**; we can repeat with RiNALMo micro / mega / giga as teachers.

**Budget and where to run what.** Nothing below is measured for THIS model; the only measured number is an earlier M4 run of a different 4.87M model (~204 windows/s x 512 nt, ~1e5 nt/s **[ours, user's M4]**).
* **Teacher caching: Colab A100.** 650M parameters x 256 nt is ~0.33 TFLOP per sequence forward; at an effective 60-100 TFLOP/s that is ~180-300 sequences/s, so 0.4M sequences take ~25-60 min **[judgement, measure first]**. On the M4 the same job is plausibly a day or more (assumed ~1-2 TFLOP/s effective) **[judgement]**.
* **One student, one run:** M4 is fine: ~3 h per 1e9 tokens at 5M parameters (20M ~3-4x slower) **[judgement from the measured number]**. An A100 should be several times faster **[judgement]**.
* **The grid is the real cost.** Pilot = 3 arms (scratch, MLM-only, distilled) x 3 seeds at 5M = 9 runs; full study = 4 sizes x ~6 arms x 3 seeds ~ 70 runs, plus fine-tuning each on every task. On the M4 that is hundreds of hours; on an A100 it is tens of hours. Colab units: A100 burn is reported anywhere from ~5 to ~13 units/h (check the Resources tab in your own session); Pro has 100 units/month, Pro+ ~500-600, and 24 h background runs need Pro+ with enough units [search: https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/ and others, unverified against Google's page].
* **So:** develop and debug on the M4 with tiny runs; cache targets and run the grids on Colab (resumable, checkpoints to Drive, one notebook plus one zip as before); reproduce the single headline student on the M4 so the "laptop" claim is true.

## 4. Evaluation

Two lessons from the RiNALMo paper (arXiv v2, read) set the protocol:
* **Frozen probes understate the teacher.** Frozen RiNALMo-650M vs fine-tuned: bpRNA TS0 F1 0.652 vs 0.747; splice Fish F1 81.02 vs 97.47; MRL Random7600 R2 0.610 vs 0.925. So fine-tuned evaluation on established tasks is the PRIMARY protocol; frozen probes (our LncBench harness) are secondary and cheap.
* **The published ladder gives us the targets without running the teacher.** Same tasks, splits and metrics, RiNALMo 33M / 150M / 650M (and RNA-FM 100M) side by side:

| task (metric) | 33M | 150M | 650M | RNA-FM |
|---|---|---|---|---|
| bpRNA TS0 (F1) | 0.685 | 0.723 | 0.747 | 0.676 |
| inter-family SS, 9 families (mean F1) | 0.43 | 0.66 | 0.71 | 0.31 |
| splice Fish / Fly / Plant / Worm (F1 x100) | 94.86 / 93.08 / 90.83 / 91.23 | 95.96 / 93.80 / 93.35 / 92.81 | 97.47 / 95.83 / 95.38 / 94.86 | 94.94 / 93.21 / 89.93 / 90.79 |
| MRL Random7600 / Human7600 (R2) | 0.884 / 0.827 | 0.908 / 0.836 | 0.925 / 0.860 | 0.842 / 0.790 |

(arXiv v2 Tables S2-S6, "33M/150M/650M" naming; the Nature Communications 2025 version could not be read, numbers may differ there.)

**Primary tasks (fine-tuned, published splits):**
* **Splice-site (GS_1; train multi-species, test unseen Fish/Fly/Plant/Worm, F1 averaged over donor and acceptor).** Headroom exists: 650M beats 33M by 2.6-4.6 points and the BEACON SPL task showed a 13.9-point FM advantage over tiny nets (`guidefam/FAMILY_PLAN.md` section 4a, from BEACON Table 3, not re-checked). Data: Spliceator GS_1 (git.unistra.fr/nscalzitti/spliceator, Zenodo 7995778) **[read in RiNALMo paper]**.
* **bpRNA TS0 secondary structure (F1 with +-1 nt tolerance).** Needs a pair head (RiNALMo fine-tunes a 2D head; head compute is the cost). Optional in v1; treat as stretch.
* **NoncodingRNAFamily and Modification (BEACON).** Established second source; a new encoder needs its own model class and tokenizer in the BEACON repo (Apache-2.0 code, data on Google Drive, no data licence stated) **[read]**. BEACON's own numbers exclude RiNALMo/ERNIE-RNA, so "SOTA" there is the BEACON table, not RiNALMo.

**MRL (Sample et al. 2019): headroom depends on the baseline, so no pre-labelling.** BEACON's table (not re-checked) has tiny nets within 1.0 of the best model (83.94 vs 84.98 R2), but BIO-Distiller's tuned 3-layer CNN gets R2 74.80 vs 82.94 for the best fine-tuned FM on the same BEACON task **[read]**. MRL therefore goes through the inclusion rule in section 5 like every other task; whichever table it lands in is reported.

**Secondary (frozen probe, LncBench):** `loc_clf`, `rbp_K562`, `rbp_HepG2`, `coding_clf`, where frozen RiNALMo-micro beats k-mers with a paired CI above 0 on dev **[ours, results/v2/paired_comparisons.md]**. Task inclusion for the giga teacher is decided on dev BEFORE any student is trained.

**Comparators at equal size:** (i) a from-scratch supervised model of the SAME architecture and size, fine-tuned with the same recipe (this is B), (ii) an MLM-only twin pretrained on the same sequences (separates distillation from extra pretraining), (iii) RiNALMo 33M published numbers, (iv) RNA-FM, RNAMamba-14M, BEACON-B where the task and numbers exist, (v) k-mer ridge on LncBench. Fine-tuned students are cheap to train: 1M-20M parameters on a splice-site dataset fits an M4 or Colab **[judgement; dataset size unchecked]**.

## 5. Pre-registered decision rule (fixed now; do not edit after results)

Retention R = (S - B) / (T - B) per task and metric, with T = published RiNALMo-650M number (arXiv v2) on the same split and metric, B = from-scratch same-architecture student fine-tuned with the same recipe, S = distilled student fine-tuned with the same recipe. Seeds: >= 3 per student; report the spread. A task is INCLUDED only if T - B > 0 with the seed spread of B below (T - B) / 2, decided from B and the published T BEFORE any distilled student is trained.
* **Headline met** if the 20M distilled student has R >= 0.90 on at least 60% of included primary tasks, AND its score is >= the published RiNALMo-33M number on those tasks.
* **Ladder claim** for a smaller student only if R >= 0.80 on at least 50% of included tasks AND it beats both its from-scratch twin B and its MLM-only twin by more than the seed spread.
* **Distillation-matters claim** only if distilled > MLM-only twin by more than the seed spread on at least half of included tasks.
* **Stop** (report as a negative result; do not release as "near-teacher"): 20M student R < 0.75 on a majority of included tasks, or no student beats its MLM-only twin.
* MRL and any task failing the inclusion rule are reported in a separate "no headroom" table, not averaged into R.
* No claim of beating RiNALMo-650M, and no use of the word "SOTA", unless a student exceeds the published 650M number with the seed spread reported.
* Frozen-probe LncBench results (secondary) use paired bootstrap over genes/RBPs as in `lncrna_fm`; dev first, locked sets scored once.

## 6. Risks (what could make this unpublishable)

1. **Novelty.** HelixNano, HAD and BIO-Distiller show the method exists, and BIO-Distiller (code released) already shows distillation helps tuned CNNs on BEACON tasks. Our edge is a general-purpose ncRNA backbone ladder from RiNALMo-giga + leak controls + ablation + released code, a tiny-paper-level contribution at best **[judgement]**. A second design risk from BIO-Distiller: **a teacher fine-tuned on the task helped significantly**, which a task-agnostic backbone distilled once cannot use; our results may therefore trail per-task distillation.
2. **Retention on tiny students may be poor.** In the mRNA case the 5M student sits between the 1M specialist and the 1B teacher; per-task gaps to the teacher were 0.02-0.10 in score **[figure]**. At <= 1M expect worse. **[judgement]**
3. **Probe protocol vs fine-tuning.** Teacher advantage under frozen probes can differ from fine-tuning; our lncRNA tasks are noisy (intervals about +-0.03 to +-0.08) **[ours]**, so R may be unmeasurable on some tasks.
4. **Inherited contamination.** Teacher saw RNAcentral-like data; students do too. Reported, not removable.
5. **Licence.** Zenodo record licence field not opened (the paper cites record 10725749, the repo README cites record 15043668; check which holds the pretrained 650M and what each says); BEACON data licence unstated; RNAcentral terms to verify.
6. **Headroom may be thin.** On bpRNA and MRL the published 33M-to-650M gap is 0.04-0.06; where a from-scratch compact model already matches the teacher (MRL), there is nothing to distil. Splice and inter-family structure are where size mattered most (inter-family F1 0.43 to 0.71), and structure generalisation across families is exactly what I expect a tiny student to lose **[judgement]**.

## 7. Steps, each with a stop

0. DONE: BIO-Distiller PDF read (see section 2). It does per-task CNN students from older teachers; it does not do a general backbone ladder from RiNALMo-giga. Overlap is partial; the loss choices (feature KD with eucosine, fine-tuned teacher) are borrowed from it and must be cited.
1. Measure RiNALMo-giga forward throughput on the user's Colab GPU at 128/256/512 nt; check the Zenodo checkpoint format and licence field.
2. Build B (from-scratch same-architecture students) on splice-site (GS_1) and the LncBench tasks; apply the inclusion rule of section 5 using the published 650M numbers; fix the included task list BEFORE any distilled student exists. Check GS_1 size and licence.
3. Build corpus, leak filter, compressed targets (Colab, one pass).
4. Train 5M student on the M4 (pilot, ~3 h) with the base recipe; compare to MLM-only twin and untrained twin. Stop if no gap to the MLM twin.
5. Full ladder + ablation arms; locked evaluation once.
6. Write the short report with the section-5 rule applied verbatim.
