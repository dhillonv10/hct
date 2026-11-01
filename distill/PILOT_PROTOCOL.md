# Pilot protocol: can a compact student inherit a 650M RNA model's advantage, and does it survive a distribution shift?

**Pre-registered. Committed before the pilot was run; the gates in section 5 are not to be edited afterwards.** Written 2026-11-01.
Code: `distill/rnakd/` (`pilot.py` drives four resumable stages). Notebook: `distill/pilot_colab.ipynb`. Background and prior art: `distill/COLAB_PLAN.md`.

## 1. The question the pilot answers

Prior art already shows that distilling an RNA foundation model into a small supervised model helps on in-distribution test sets
(BIO-Distiller, ICLR 2026 MLGenX workshop: a tuned 3-layer CNN gains ~2% R2 on regression tasks and ~10 MCC points on ncRNA
classification). What nobody has tested, as far as I could find, is whether the distilled student keeps the teacher's advantage
**when the test distribution moves**. That matters because the case for a large pretrained model is usually generalisation, and
because BIO-Distiller's own headline ("well-tuned supervised baselines can still outperform foundation models") was measured on
BEACON splits whose construction the BEACON paper never states and which contain no family- or species-held-out test set.

Mean ribosome load is the cheapest task that can answer it, because the shift is built into the data: models are trained on a
library of **random synthetic 5'UTRs** and evaluated on **real human 5'UTRs** from the same assay (Sample et al. 2019, GEO
GSE114002). RiNALMo publishes both numbers, so the teacher is checkable: giga 0.925 random / 0.860 human, 33M 0.884 / 0.827.

## 2. Design

| | |
|---|---|
| Teacher | RiNALMo-giga (650M), official fine-tuned MRL checkpoint from Zenodo 15043668 (weights CC BY 4.0, code Apache-2.0), run through our plain-PyTorch re-implementation (`rnakd/teacher.py`) because the official package hard-imports flash-attn |
| Students | one dilated residual CNN family at five widths, ~30k to ~7M parameters (`rnakd/student.py`) |
| Arms | `scratch` (supervised only), `kd_feat` (eucosine on the teacher's 32-d penultimate head feature: BIO-Distiller's best setting), `kd_feat_enc` (same loss on the 1280-d pooled encoder output, a task-agnostic target), `kd_fitnet` (MSE feature loss), `kd_logit` (response KD), `kd_both`, plus two controls |
| Controls | `ctrl_shuffled` (teacher targets permuted across examples: same statistics, no correspondence) and `ctrl_untrained` (targets from a randomly initialised teacher of the same architecture). Either beating `scratch` means the "gain" is regularisation, not transferred knowledge |
| Seeds | 3 per cell; the yardstick ("spread") for calling a margin real is the largest seed standard deviation among the `scratch` cells, so that an arm which collapses cannot inflate the bar. The maximum over every cell is reported beside it |
| Data efficiency | train fractions 0.05 / 0.25 / 1.0 at one size, since distillation is claimed to matter most when labels are scarce |

## 3. Splits, and which one the teacher is allowed to be scored on

RiNALMo's own code uses **Random7600 as the validation set** (checkpoint selection) and Human7600 as the test set, so the
published 0.925 is not a clean held-out number. `rnakd/mrl.py` reproduces their construction exactly and adds one set of ours:

| split | how it is built | used for |
|---|---|---|
| `train` | the random library, minus everything below | fitting students |
| `val` | random 5% of `train` | student early stopping only |
| `iid_test` | the **next** top-100-reads-per-length block of random-library UTRs, their selection rule applied after removing Random7600 | a clean in-distribution set **for students only** |
| `ood_test` | Human7600: top-100-reads-per-length real human UTRs | the distribution-shift test, **clean for everyone** |
| `repro_val` | Random7600, the authors' validation set | reproducing their published number, and the IID set in the teacher-versus-student comparison |

**The released teacher was fine-tuned on the whole random library except Random7600, so it trained on the rows in `iid_test`.**
No teacher number on `iid_test` is usable and none is reported. The comparison therefore uses:

* **OOD = `ood_test`.** The teacher never trained on human-library rows and reported this as its test set, so it is clean for
  teacher and students alike. Every out-of-distribution number is sound.
* **IID = `repro_val`.** The teacher selected checkpoints on it; our students never touch it. Its teacher score is therefore
  mildly optimistic. That bias inflates the teacher's IID headroom, shrinks R_IID and pushes P4 toward "the out-of-distribution
  advantage transfers better". **A negative P4 is robust to this bias; a positive P4 is reported as suggestive only.** Stating
  the direction in advance is the point of writing this down before the run.
* `iid_test` is reported for students alone, as a check that the Random7600 figures are not selection-inflated.

A teacher we fine-tune ourselves on exactly our splits would remove the caveat entirely; at RiNALMo's recipe (50 epochs over
~75k UTRs with a 650M model) that is several A100-hours, so it belongs to the full study, not the pilot.

Filters before splitting, as in their code: drop missing load, keep the random and human libraries, require total_reads >= 10,
and drop every UTR that occurs more than once (all copies).

## 4. Metrics

R2 (the task's published metric) and Spearman, on each test set, with 95% bootstrap intervals over test sequences.
**Retention** R = (student - scratch) / (teacher - scratch), computed separately for IID and OOD at matched size; the quantity of
interest for the new question is **R_OOD - R_IID**.

## 5. Gates (fixed; the report prints each one)

* **P1, reproduction.** Our evaluation of the released fine-tuned checkpoint is within 0.02 R2 of both published numbers
  (0.925 on `repro_val`, 0.860 on `ood_test`). If not met, no student result is interpreted until the loader is fixed or we
  fine-tune our own teacher and report our number beside the published one. This runs before any student exists.
* **P2, headroom.** The teacher beats the best tuned `scratch` student by more than 2x spread on IID or OOD. If headroom exists
  on neither, there is nothing to distil and the pilot reports that as its finding.
* **P3, replication.** Some distillation arm beats `scratch` on the IID test by more than the spread. This is the published
  effect; failing it means our implementation, not the idea, is at fault.
* **P4, the new question.** Mean R_OOD - R_IID over the distillation arms. Positive means the teacher's out-of-distribution
  advantage transfers at least as well as its in-distribution advantage; negative means distillation buys less exactly where the
  large model is supposed to earn its keep. **Either direction is a result worth reporting**, which is why no threshold is set
  on it here.
* **P5, controls.** Neither control beats `scratch` by more than the spread.

**Go to the full study** (more tasks, the task-agnostic backbone, larger students) **iff P1, P3 and P5 are met and P4 is not
strongly negative.** If P3 fails while P5 passes, the finding is a failed replication of prior art on a task both studies use,
and is reported as such. If P5 fails, the honest reading is that reported distillation gains on this task are a regularisation
effect, and that becomes the paper.

## 6. What the pilot cannot settle

One task, one teacher, one student family. A result here does not generalise to structure prediction, to classification, or to
the task-agnostic backbone in `COLAB_PLAN.md`; it decides whether those are worth the compute. The human-versus-random UTR shift
is a library shift within one assay, not a species or family shift. The teacher has seen RNAcentral-scale ncRNA data, and we do
not control what overlap that implies for human 5'UTRs. Costs are reported as measured wall-clock on the GPU the run used.
