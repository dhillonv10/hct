# Landmark-anchored encoding

## The evidence this is built on

Two measurements from this project, both pointing the same way:

1. **Mean-ribosome load.** Replacing mean-pooling of frozen RiNALMo embeddings with a 16-cell structured
   basis gained **+0.145 r²** (0.247 → 0.391) with identical frozen weights and an identical probe. The
   per-cell stack then assigned the mean-pooled vector a weight of **0.028** — given the choice, it threw
   the global average away. Each individual *frame* cell beat the whole mean-pooled vector on its own.
2. **Half-life.** Exon-junction *density* dominated 609 engineered features by **6.5×** in permutation
   importance. But collapsing junction positions into that one scalar left the model 0.070 short of Saluki.

Together: what matters is position *relative to biological landmarks*, and collapsing it loses a lot.

## What existing models do

| model | positional information |
|---|---|
| Saluki | CNN+GRU, frame and splice as input channels — must learn distance-to-stop from convolutions |
| Orthrus, NT, DNABERT, RiNALMo | absolute or rotary encoding, relative to sequence start |
| Optimus 5-Prime, FramePool | frame-aware, but 5'UTR only |

None encodes distance to the start codon, stop codon, cap, poly(A) site, or flanking exon junctions.

## The three components

**1. Landmark-relative Fourier encoding.** Per nucleotide *i*, signed distances to six landmarks: cap,
start codon, stop codon, poly(A), previous junction, next junction. Each distance is encoded as sin/cos at
eight log-spaced periods (3, 10, 30, 100, 300, 1000, 3000, 10000 nt). **The 3 nt period makes reading frame
an explicit input** rather than something to infer. 6 × 8 × 2 = 96 channels, plus base one-hot, region
one-hot, junction indicator and a validity mask: **105 input channels**.

The mechanistic payoff: nonsense-mediated decay fires when a junction sits >50 nt downstream of a stop
codon. This model reads that configuration directly off its input.

**2. Landmark-anchored read-out.** Dilated residual conv body (full resolution → /4 → /16, dilations to 32),
then pooling into 20 cells rather than one global average:

* mean and max within 5'UTR, CDS, 3'UTR (6)
* distance bands: ±50 and ±200 nt around start and stop, first 100 nt, last 200 nt (6)
* **mean over junction positions, and over junctions >50 nt past the stop** — the NMD cell (2)
* global mean, global max, attention-pooled (3)
* **frame cells at full resolution**, mean over each (i − start) mod 3 class inside the CDS (3)

This is the structured read-out that gained +0.145, generalised from frame to arbitrary landmarks.

**3. The engineered features, fused LATE.** All 609: exon/junction geometry, codon usage and the Forrest
et al. codon stability coefficients, region lengths and GC/GC3, 3'UTR AREs and DRACH m6A sites and Pumilio
motifs, uORFs by frame, capped-window folding energies. A gradient-boosted model is trained on them inside
each fold, on the same training rows, and combined with the network on the inner validation split.

## What the first real run falsified, and what replaced it

The first design concatenated those 609 features **into the network's head**, on the argument that they set
a floor the network could only add to. The first full run killed that argument:

```
ep   1/25  loss 0.7205  val r 0.6741  *
...
ep  25/25  loss 0.4051  val r 0.6677
best val r 0.6751 at epoch 8
```

Validation r is reached in **one epoch** and never improves, while training loss halves from 0.72 to 0.40.
All three folds behaved the same way. What a head can learn in one epoch is the feature block — so the
convolutional trunk had a shortcut, never received a useful gradient, and contributed essentially nothing.
The hybrid scored **0.6933, below the 0.7023 that gradient boosting reaches on those features with no
network at all**.

The floor argument was simply wrong. The floor a shared head provides is an *MLP* on the features (~0.67),
not a *GBM* on them (0.702); a two-layer MLP with dropout is worse at tabular features than boosting. And
aux-block dropout at 0.3 was far too weak to prevent it — 70% of batches still handed the head the shortcut.

**Late fusion replaces it.** The network trains on sequence only (`--aux-drop 1.0`); the GBM trains on the
features; the two are combined on the inner validation split (`--stack-gbm`). The shortcut is gone, the
floor is real because the GBM is now a component rather than something an MLP must re-derive, and every run
reports all three numbers per fold.

## Reading any result

| component | what it is | reference |
|---|---|--:|
| `nn` | the landmark network, sequence only | Saluki's sequence-only ablation **0.608** |
| `gbm` | the 609 features alone | our **0.7023** |
| `stack` | the two combined | full Saluki **0.7726** |

**`nn` is the number that decides whether this architecture is worth anything.** Above 0.608 it extracts
something Saluki's convolutions do not. Near 0.60 it merely matches them. Below ~0.45 the landmark encoding
is not delivering, and a `stack` that still looks respectable would be the GBM carrying it — which must be
reported that way rather than as an architectural win.

This control used to be a separate phase that could be skipped or forgotten. It is now part of every single
run, which is where it belonged.

## Size and cost

Five sizes: 0.24M / 0.77M / 2.05M / 4.91M / 11.43M parameters. RiNALMo-giga is 651M, so `tiny` is about
1/2700th of it. On an M4 via MPS, `base` runs ~55 ms/step at bs=16, lmax=12288 — roughly 0.85 min/epoch.

## Known open issue

The learning-rate schedule is mismatched to the observed dynamics: validation peaked at epochs 8–12 of a
25-epoch OneCycle, so the selected checkpoint is taken mid-schedule with the learning rate still near its
maximum and never sees the annealing phase. With the shortcut removed the dynamics will differ, so this is
worth re-reading from the new curves before being tuned blind.
