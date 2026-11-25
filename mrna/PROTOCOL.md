# Pre-registration: can an ensemble + biophysical features beat the best single model on mRNABench?

Written 2026-11-25, **before any number is computed**. Nothing below is revised after seeing results;
if a decision turns out wrong, it gets an explicit amendment section with a date.

## Why this and not another foundation model

Everything we tried in the pretrain-a-new-backbone direction died for a structural reason, and the reason
is recorded (`distill/RESULTS.md`, `lncfam/`): distillation of RNA LMs is a null on MRL with controls; a
lncRNA-only backbone has ~75 Mb of sequence and a ~0.05 AUROC ceiling; every other small-RNA class is
either too short to have grammar or too few genes to learn one. Pretraining is not where the headroom is.

Where there *is* plausible headroom: mRNABench evaluates each model **alone**, by linear probe, on 79 tasks.
Nobody has published the combination. Different backbones are trained on different objectives (Orthrus =
contrastive on isoforms, NT/DNABERT = genomic MLM, CodonBERT = codon-level, mRNA-FM = codon MLM,
RiNALMo = nucleotide MLM on ncRNA) and the explicit biophysics (folding energy, codon usage, UTR features)
is not in any of their inputs as a feature. If their errors are decorrelated, a stacked combination wins.
If they are all reading the same signal, it does not, and that is a real finding about the field.

## Correctness gate — must pass before any new number is believed

We will run our own probe code rather than only `LinearProbeBuilder`, because stacking needs out-of-fold
predictions that the built-in builder does not expose. That makes a fairness gate mandatory:

> **G1.** Using mRNABench's own homology splitter indices, our probe must reproduce `LinearProbeBuilder`'s
> metric for the same single model on the same dataset to within **0.01** absolute, on at least 2 datasets
> of different task types (one regression, one multilabel).

If G1 fails, we stop and debug; we do not report any ensemble number computed by an unvalidated harness.
A second, weaker check: our single-model numbers should land near the leaderboard's published values for
that model. A mismatch there is informative but not disqualifying (version/seed differences).

## Fixed design

**Datasets** (chosen now, not after seeing results). Spread of task type, size, and incumbent:

| dataset | type | why it is in |
|---|---|---|
| `rnahl-human` | regression | Saluki's own half-life data — a named, strong, non-LM incumbent |
| `rnahl-mouse` | regression | same task, other species: tests whether a win transfers |
| `mrl-sugimoto` | regression | isoform-resolved ribosome load, the task RiNALMo-class models are tuned on |
| `translation-efficiency-human` | regression | TE, where codon usage should matter and is an explicit feature |
| `prot-loc` | multilabel | localisation, driven by UTR elements |
| `go-mf` | multilabel | function prediction, the broadest label set |
| `eclip-binding-k562` | multilabel | ~80 RBP binding tasks — the bulk of the 79-task count |

**Models to combine.** All that load in the container, from: `orthrus-large-6-track`,
`orthrus-base-4-track`, `nucleotide-transformer 500m-human-ref`, `dnabert2`, `splicebert-human.510nt`,
`rinalmo`, `mrna-fm`, `codonbert`, `hyenadna-medium-160k`. Minimum viable set is **4 models from at least
3 distinct pretraining objectives**; below that the ensemble question is not being asked.

**Feature block** (`bio`), computed from sequence + the track annotations the datasets ship:
lengths (5'UTR / CDS / 3'UTR / total, log), GC, GC3 and per-region GC, ViennaRNA MFE per nt and unpaired
fraction on four capped windows (cap-proximal 70 nt, 5'UTR last 200, first 200 nt CDS, 3'UTR last 300; all
at `max_bp_span=100`), uAUG count, Kozak -3 purine and +4 G, AUG-present-at-CDS-start, ATTTA (ARE) count,
polyA-signal count, 64-dim relative codon frequency, premature-stop count. **99 features**, deterministic,
cached, computed in ~3 min per 10k sequences on 4 cores.

Two decisions taken when the cost was measured, recorded here rather than silently:

* **Ensemble free energy is off by default** (`USE_PF=False`). `fc.pf()` is ~70% of folding time (110 ms vs
  33 ms per 300 nt window, measured) and the only feature it adds is ensemble free energy per nt, which is
  ~0.99 correlated with MFE per nt. The flag turns it on; the cache filename records the setting so the two
  cannot be mixed.
* **CAI is dropped.** It was in the first draft of this protocol. For a linear probe it is redundant: CAI is
  a weighted sum of codon log-frequencies, so the 64-dim codon frequency vector already spans it, and a
  hardcoded reference weight table would have been one more thing to get wrong.

**Probe configurations**, all on identical split indices and identical seeds:
1. `single:<model>` — each model alone (reproduces the leaderboard contrast)
2. `bio` — feature block alone
3. `concat-all` — per-block standardised, hstacked, alpha tuned on val
4. `stack` — level-1 probe per block on train, level-2 ridge/logistic on **validation** predictions, test untouched
5. `stack+bio` — as 4 with the feature block as an extra level-1 block

Level-2 trains on validation predictions only. Test is touched once per configuration per seed.

**Seeds.** 5 seeds, fixed: `2541, 1234, 42, 7, 31337`. Every configuration sees the same 5.

**Metric and comparison.** Whatever mRNABench uses per task type (confirmed in phase 1, not assumed).
Primary contrast is **paired**: per task, `stack+bio` minus the best single model, where "best single" is
chosen **on validation**, never on test. Pooled across tasks by hierarchical bootstrap over (task, seed).

## Kill criteria — decided now

* **K1.** If G1 fails after one debugging pass, stop. No ensemble number gets reported.
* **K2.** If `stack+bio` beats the val-selected best single model on **fewer than 60% of tasks**, the
  across-the-board claim is dead. We say so plainly and report the win rate as the result.
* **K3.** If the pooled paired improvement's 95% interval includes 0, there is no ensemble effect.
  Report it as a null — "nine RNA backbones carry one signal" is publishable and honest.
* **K4.** If `bio` alone is within 0.01 of the best single model on a task, that task's language models are
  not earning their parameters, and that is the headline for that task, not the ensemble.

## What would be an honest positive result

A win rate above 60% with a pooled interval excluding 0, **plus** a decomposition saying where it comes
from: which model pairs are decorrelated, and whether the biophysical block or the model diversity does the
work. "Ensemble is better" with no mechanism is not worth writing up.

## Stated odds, before the run

Beating the best single model on a majority of tasks: ~50-60%. On *every* task: ~15%. The honest expectation
is a modest, uneven win plus a useful decorrelation map — not a clean sweep. If the across-the-board claim is
what matters, the likely answer is no, and K2 is how we find that out in one run instead of talking ourselves
into it over three.
