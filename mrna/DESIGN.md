# Can we design an mRNA model that beats all the others?

Straight answer first: **not one model, no.** Three reasons, and they are structural rather than a lack of effort.

1. "All the others" means 79 heterogeneous tasks. GO function prediction, ribosome load, RBP binding and
   localisation need different signal. No single architecture dominates that spread, and nothing in the
   literature does either — the mRNABench leaderboard is won by different models on different task families.
2. Several tasks are near their assay-noise ceiling. MRL on the random library sits at r^2~0.93 against a
   measurement whose own replicate agreement is not much higher. There is no 0.05 to win there.
3. Pretraining a genuinely better backbone is a 100+ GPU-hour job minimum. We have 4 CPU cores here and
   whatever Colab gives.

So the question worth asking is not "can we pretrain something better". It is **where the headroom actually
is**, and there is a specific, testable answer.

## The hypothesis: the read-out is the bottleneck, not the backbone

Every model on mRNABench is evaluated the same way: run the transcript through the backbone, **mean-pool the
token embeddings into one vector**, fit a linear probe. Orthrus, NT, DNABERT, HyenaDNA, RiNALMo, mRNA-FM —
all of them, all 79 tasks.

For mRNA that pooling step throws away the three things that determine function:

* **Region.** 5'UTR, CDS and 3'UTR do different jobs. A motif's meaning depends on which one it is in.
* **Frame.** An upstream AUG that is in-frame with the main ORF gives an N-terminal extension; out-of-frame
  it creates a competing ORF and represses translation hard. Same motif, opposite effect, and mean pooling
  cannot tell them apart. This is the documented FramePool mechanism (Karollus et al. 2021).
* **Position.** Cap-proximal structure blocks initiation; the same hairpin 300 nt downstream does not.

The backbones are position-aware internally — RiNALMo has RoPE, Orthrus has the CDS track as input. The
information is *in* the token embeddings. The benchmark's read-out then averages it away.

**So: keep the backbone frozen, replace mean-pooling with a structured pooling basis that preserves region,
frame and position, and fit the same linear probe.** Mean-pooling is a strict special case of that basis, so
with selection on validation it should not lose, and it should win wherever those three things matter.

That is a claim about the whole benchmark, it is cheap to test, and the comparison is airtight because the
backbone and its weights are identical on both sides. Only the read-out differs.

## The structured pooling basis

Per-token reps `H` (L x d) from a frozen backbone, pooled into 16 cells instead of 1:

| cells | what | why |
|---|---|---|
| `mean`, `max`, `std` | whole sequence | `mean` alone is the benchmark's read-out, so the baseline is nested inside |
| `frame0/1/2` | mean over tokens at distance `(L-i) mod 3` from the main AUG | the in-frame vs out-of-frame uAUG distinction |
| `fnear0/1/2` | same, restricted to the last 30 nt | where frame matters most for reinitiation |
| `cap10`, `q1`, `q2`, `q3`, `aug10` | positional buckets: first 10 nt, three thirds, last 10 nt | cap-proximal vs distal |
| `first`, `last` | boundary tokens | the backbone's own summary of each end |

16 x d dims. For RiNALMo-micro (d=480) that is 7680 features against ~10^5 training rows — a regime ridge
handles without trouble.

## Why this could win broadly rather than on one task

Because it is a *protocol* improvement, not a task-specific trick. It applies to every backbone and every
task in the benchmark, it costs one extra pooling pass, and its baseline is nested inside it. The honest
expected result is not "+0.1 everywhere" but "a consistent small-to-moderate gain, concentrated on tasks
where region and frame carry signal, and ~0 on tasks where they do not" — plus an ablation saying which
cells did the work. That is a finding either way.

## What else is on the table, ranked by honest odds

| idea | odds it beats the same-backbone mean-pool probe | odds it beats *published* SOTA |
|---|---|---|
| **structured read-out** (above) | **~80%** — the baseline is nested inside it | low alone: frozen 33M vs fine-tuned 651M |
| **scratch CNN on one-hot**, Optimus-style | n/a, different class | **real** on data-rich tasks; Optimus hits ~0.93 on MRL by itself |
| **explicit biophysics + GBM** | task-dependent | we already saw ViennaRNA beat a 651M LM on RNU4-2, so: plausible on some |
| **stack of CNN + structured read-out** | — | **best shot**, ~40% on MRL OOD |
| multi-backbone ensemble (`ensemble_colab.ipynb`) | — | ~50-60% on a majority of tasks |
| pretrain a better backbone | — | not reachable with this compute |

The thing I will not claim: that any of these sweeps all 79 tasks. The realistic deliverable is a method that
wins *broadly* because it adapts per task, plus a clean answer on where the limit actually sits.

## The first experiment, and why this task

MRL on Sample et al. 2019 (GEO GSE114002), because it is the one place we can test the core hypothesis today
with no new ground truth and no GPU:

* the data is 25-100 nt 5'UTRs, so a frozen 33M backbone embeds the whole set on CPU;
* `distill/rnakd/mrl.py` already reproduces the authors' exact splits, and our harness already reproduced
  RiNALMo's published **0.925 / 0.860** to 4 d.p., so the correctness gate is behind us;
* frame is *known* to matter here, which makes it a fair test of the mechanism rather than a fishing trip;
* `ood_test` (Human7600) is where published models are weakest, so it is where headroom plausibly exists.

Arms, all selected on `val` only, reported on `iid_test` / `ood_test` / `repro_val`:

* **B1** frozen RiNALMo-micro, mean-pool + ridge — *the mRNABench protocol, the thing to beat*
* **B2** explicit sequence features + ridge (no LM at all)
* **B3** scratch CNN on one-hot (the Optimus-style play)
* **R** frozen RiNALMo-micro, **structured pooling** + ridge — the hypothesis
* **S** stack of B3 + R on validation predictions

## Pre-registered reading of the result

* **R > B1 by >=0.01 r^2 on both test sets** -> the read-out claim holds, and it generalises to the whole
  benchmark as a cheap protocol fix. This is the result worth writing up.
* **R ~= B1** -> mean pooling was already sufficient; the structured basis is nested so it cannot lose much,
  and a null here kills the idea cleanly rather than ambiguously.
* **B2 competitive with B1** -> the backbone is not earning its parameters on this task, same pattern as
  ViennaRNA beating a 651M model on RNU4-2.
* **S > 0.925 / 0.860** -> we beat the published number, with the caveat that the published number is a
  fine-tuned 651M model and ours is a stack on a frozen 33M one. That caveat gets stated, not buried.

The ablation over pooling cells is part of the deliverable, not an extra. A read-out that wins without an
identifiable mechanism is not a finding.
