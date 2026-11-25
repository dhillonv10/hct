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

**3. Engineered features at the head.** All 609: exon/junction geometry, codon usage and the Forrest et al.
codon stability coefficients, region lengths and GC/GC3, 3'UTR AREs and DRACH m6A sites and Pumilio motifs,
uORFs by frame, capped-window folding energies. Saluki learns codon effects from scratch; this model is
given them.

Because those features alone reach r=0.702, the head would happily ignore the convolutional path. **Aux-block
dropout** (zeroing the entire feature block on 30% of training steps) forces the sequence path to carry
weight. This sets a floor near 0.70 while leaving the upside to the sequence model.

## Size and cost

2.04M parameters — ~1/300th of RiNALMo-giga. 680 ms/step at bs=16, lmax=12288 on 4 CPU cores.

## What would falsify the idea

If the landmark encoding is doing nothing, the model lands at the aux-feature floor (~0.70) regardless of
how long it trains, and `--aux-drop 1.0` (sequence path only) collapses toward the sequence-only ablation
Saluki reports, 0.608. That is the control to run if the headline is disappointing: **a sequence-only arm
that beats 0.608 means the landmark encoding is contributing something Saluki's convolutions do not get.**
