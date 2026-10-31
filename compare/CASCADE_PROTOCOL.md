# Cascade protocol: compact model as a cheap first stage in front of OligoAI (simulated on out-of-fold predictions)
Written and committed BEFORE the simulation was run. Data: compare/results fold_compare_rows.csv.gz (per-ASO out-of-fold predictions of our compact model and of the authors' OligoAI; see CONFIRMATORY_PROTOCOL.md).

## Question
If the compact model scores every candidate ASO in a screen and the large model (OligoAI, ~650M parameters) is called only on a fraction f of them, how much of the full two-model quality is kept?

## Policies (fixed)
* Screen = (table, cell line, dose) with >= 20 ASOs (the primary-analysis screens).
* Stage 1 ranks all ASOs by the compact score. Stage 2 calls the large model on the top ceil(f * n) ASOs; for those, final score = 0.5 * rank percentile of the compact score + 0.5 * rank percentile of the large-model score, both computed within the called set.
  ASOs not called rank below all called ones, in compact order. f in {0.05, 0.10, 0.25, 0.50, 1.00}.
* Controls: (a) RANDOM routing (the same f, called ASOs chosen at random); (b) compact only (f = 0); (c) large model only on all ASOs (f = 1, no compact); (d) full mixture (f = 1).
* Selection: top k = 5 ASOs of each screen by final score. Quality = mean measured inhibition (percentage points) of the 5 selected ASOs, averaged over screens; also reported as the ratio to the screen mean.
* Uncertainty: bootstrap over patents (2,000 draws). Cost is reported as the fraction of large-model calls, NOT as seconds (no timing was measured).

## Decision rules
* "The cascade works": at f = 0.25 (a) quality is within 1.0 percentage point of the full mixture (f = 1), with the upper 95% bound of the shortfall below 2.0 points, AND (b) quality exceeds the large model alone (called on every ASO) with the 95% interval of the difference above 0.
* If (a) fails, the large model is needed on most candidates and the cascade is not worth building on this task. If (a) holds but (b) fails, it is a pure cost saving with no quality gain.
* Random routing must be worse than compact routing at the same f, otherwise the compact model is not doing any routing.

## Caveats stated in advance
* Within one ASO screen the large model is cheap in absolute terms (hundreds to a few thousand candidates), so savings matter only for large-scale design or much larger models; this experiment tests whether the routing signal exists, not wall-clock savings.
* The compact stage's top-f is a ranking filter, not a calibrated confidence; uncertainty-based routing is not tested here.
