# Pilot result: mean ribosome load, A100, 2026-11-02

Run by the user on a Colab A100-SXM4-40GB from `pilot_colab.ipynb` and the committed `rnakd/` package, against the protocol
fixed in `PILOT_PROTOCOL.md`. Numbers below are copied from the notebook's own output. Raw `runs.csv` not yet in the repo.

## 1. The teacher reproduces, exactly (GATE P1 MET)

Our 240-line plain-PyTorch re-implementation, loading the released `rinalmo_giga_mrl_ft.pt` (650,979,682 parameters) with a
strict key match:

| split | n | our R2 [95% CI] | our rho | published | delta |
|---|--:|---|--:|--:|--:|
| `repro_val` (Random7600) | 7,600 | 0.9253 [0.9214, 0.9291] | 0.9471 | 0.925 | **+0.000** |
| `ood_test` (Human7600) | 7,600 | 0.8596 [0.8512, 0.8680] | 0.8732 | 0.860 | **-0.000** |
| `iid_test` (ours) | 7,600 | 0.9841 | 0.9937 | - | teacher trained on these rows; not a teacher reference |

The 0.9841 on `iid_test` is the contamination the protocol predicted and excluded in advance: those rows are inside the
released checkpoint's training set. It is reported only as evidence that the exclusion was necessary.

Teacher throughput, measured: 398 sequences/s at batch 128.

## 2. A 2.6M CNN lands within 0.03-0.04 R2 of a 650M model (the headline measurement)

Best tuned from-scratch student, 3 seeds, no distillation:

| model | params | IID R2 (Random7600) | OOD R2 (Human7600) | seq/s | x smaller | x faster |
|---|--:|--:|--:|--:|--:|--:|
| RiNALMo-giga, fine-tuned | 650,979,682 | 0.9253 | 0.8596 | 398 | 1 | 1 |
| `medium` CNN, from scratch | 2,633,473 | 0.888 +-0.007 | 0.827 +-0.007 | 73,285 | 247 | 184 |
| `small` CNN, from scratch | 661,377 | 0.886 +-0.016 | 0.816 +-0.016 | 74,532 | 984 | 187 |
| `tiny` CNN, from scratch | 125,505 | 0.791 +-0.007 | 0.773 +-0.007 | 94,327 | 5,187 | 237 |
| `nano` CNN, from scratch | 32,033 | 0.796 +-0.009 | 0.775 +-0.009 | 89,324 | 20,322 | 224 |
| k-mer ridge (no learned representation) | 258 | 0.367 | 0.418 | - | - | - |

Gaps to the teacher: **0.037 IID, 0.033 OOD** at 2.6M parameters. The k-mer row matters: at 0.37/0.42 it is far below every
student, so the task does need a learned sequence model and the students are doing real work.

There is a sharp jump between `tiny` (125k, 0.79) and `small` (661k, 0.89); the ladder is not smooth, which suggests the
3-block dilated CNN is capacity-limited below ~600k rather than that 0.79 is a floor for small models.

## 3. Distillation adds nothing here (GATE P3 NOT MET -> NO GO on the study as framed)

Five distillation variants, four sizes, three seeds, against the same tuned baseline. IID R2 / OOD R2 at frac 1.0:

| size | scratch | kd_feat | kd_logit | kd_both | kd_fitnet | kd_feat_enc |
|---|---|---|---|---|---|---|
| nano | 0.796/0.775 | 0.781/0.768 | 0.784/0.769 | 0.792/0.772 | - | - |
| tiny | 0.791/0.773 | 0.796/0.779 | 0.781/0.763 | 0.786/0.776 | 0.795/0.777 | 0.794/0.778 |
| small | 0.886/0.816 | 0.889/0.828 | 0.880/0.810 | 0.881/0.824 | 0.878/0.813 | 0.886/0.821 |
| medium | 0.888/0.827 | 0.887/0.827 | 0.885/0.816 | 0.885/0.828 | - | - |

The largest IID gain over scratch anywhere is +0.005 (`tiny`, kd_feat); the largest OOD gain is +0.012 (`small`, kd_feat).
The seed standard deviation of the scratch cells at frac 1.0 is 0.016, so nothing here clears the noise. This is a
well-powered null, not an inconclusive one: it covers feature KD on the teacher's penultimate layer (the setting BIO-Distiller
reports as its best), feature KD on the pooled encoder, FitNet MSE, response KD, and feature+response together.

**Controls behave (GATE P5 MET):** `ctrl_shuffled` (teacher targets permuted across examples) and `ctrl_untrained` (targets
from a randomly initialised teacher of the same architecture) do not beat scratch either. So the null is a property of the
task and the baseline, not of a broken harness.

## 4. A spread bug found while reading the output, and what it changes

`stage_report` computed the seed-spread yardstick over the scratch cells at **every** train fraction, so the noise of the
5%-label cells (sd 0.051-0.063) was used to judge comparisons made at 100% labels. That is a coding error, not a judgement
call: the protocol's yardstick is meant to describe the noise of the comparison being gated. Fixed to use frac 1.0 only.

| gate | as printed (spread 0.0509) | corrected (spread 0.0160) |
|---|---|---|
| P1 reproduction | MET | MET |
| P2 headroom over best scratch, > 2x spread | NOT MET (needed 0.102) | **MET, marginally** (IID gap 0.0373 and OOD gap 0.0326 vs a 0.032 bar) |
| P3 distillation beats scratch on IID | NOT MET | **NOT MET** (best margin +0.005 vs a 0.016 bar) |
| P5 controls | MET | MET |

The load-bearing conclusion is unchanged, and is now sharper: there **is** a small amount of headroom between the best tuned
compact model and the 650M teacher, and distillation does not capture any of it.

P4 (R_OOD - R_IID) printed +0.046 but should not be read as a result: retention divides by (teacher - scratch), which is
~0.03 here, so the per-cell retention numbers swing between -0.38 and +0.26 on differences that are themselves inside the
noise. With this little headroom the retention statistic is undefined in practice.

## 5. The one live thread: label efficiency

The distillation effect is not flat across training-set size. `small` students, kd_feat vs scratch:

| labels | scratch IID/OOD | kd_feat IID/OOD | IID gain | seed sd |
|---|---|---|--:|--:|
| 5% (~3.3k) | 0.498/0.569 | 0.562/0.610 | **+0.064** | 0.051-0.063 |
| 25% (~16k) | 0.821/0.774 | 0.836/0.779 | **+0.015** | 0.008-0.012 |
| 100% (65k) | 0.886/0.816 | 0.889/0.828 | +0.003 | 0.016 |

Monotone and in the expected direction: the 25% gain (+0.015) is just above its own seed spread (0.012), the 5% gain is
larger but inside it. Three seeds and three fractions cannot settle this. It is cheap to settle, because the teacher cache
already exists on Drive and these runs take 1-17 s each.

## 6. Where this leaves the project

The pre-registered rule says GO only if P1, P3 and P5 are met. **P3 failed, so the distillation study as framed does not
proceed.** What the run did establish, and what it costs nothing more to write up:

1. An independent re-implementation reproduces a published 650M RNA model to within 0.0005 R2 on both of its reported splits.
2. A 2.6M CNN is within 0.037/0.033 R2 of it at 184x the throughput, and a 661k CNN within 0.039/0.044.
3. Distillation from that teacher, in five variants with controls, adds nothing at full labels.
4. The gain appears only as labels become scarce, and that is the only claim worth more compute.

Open comparison, not yet run: BIO-Distiller reports a tuned CNN at R2 0.748 on BEACON's MRL split against a best foundation
model at 0.829, and a distillation gain to 0.772. Our tuned CNN reaches 0.886/0.816 on RiNALMo's splits. If those splits are
comparable (both derive from Sample et al. 2019 and both use 7,600-sequence test sets, but this has NOT been checked), then
our baseline is stronger than their best foundation model, and the natural reading of our null is that **their distillation
gain is a function of baseline strength**. Verifying that needs our student run on BEACON's exact MRL split; until then it is
a hypothesis, not a claim.
