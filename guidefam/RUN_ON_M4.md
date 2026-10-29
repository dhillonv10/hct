# Running guidefam on an Apple-silicon laptop

Everything runs on CPU (PyTorch CPU wheels are fine on M-series). Budget: the baseline ladder and siRNA stages take minutes; Tiger2D leave-one-gene-out takes ~1-2 h; the extension stage ~4-8 h with 8-10 threads.

**What more compute buys, honestly.** These models converge in minutes; longer training of the same models will not move the benchmark. The ceiling is data and label noise (the best model captures roughly half of the explainable variance). Compute is useful for (a) seed ensembles, (b) a proper hyper-parameter search, (c) larger learners such as TabDPT, and (d) running OligoFormer itself so it can be combined with our scores. `m4_extend.py` and `sirna_plus_oligoformer.py` do exactly that, with the same dev-first discipline.

## 0. Environment and data
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r guidefam/requirements.txt
export OMP_NUM_THREADS=8        # use your performance cores
```

## 1. Reproduce the headline results (1-2 h)
```bash
bash guidefam/scripts/run_all.sh ~/guidefam_work
```
Expected (sandbox values, single seed; yours may differ in the third decimal): Cas13d ens3 within-gene Spearman 0.638 all genes / 0.622 locked genes; siRNA Huesken -> Mixset AUC 0.813, PCC 0.574. If your numbers differ by more than ~0.01, check library versions first and tell me.

## 1b. Optional: compare with the authors' released TIGER weights
```bash
pip install tensorflow huggingface_hub          # TensorFlow is used only to read the checkpoint index
python -m guidefam.scripts.export_official_tiger ~/guidefam_work/official_tiger.npz
python -m guidefam.scripts.final_cas13d ... --official-npz ~/guidefam_work/official_tiger.npz
```
This accepts NYGC's academic / non-commercial licence (do not redistribute the .npz, and do not use it commercially). Expect the flow-cytometry "released TIGER" row near 0.60 and the HAP1 row near 0.11.

## 2. Extension stage: try to beat the current best (dev-first)
```bash
python -m guidefam.scripts.m4_extend dev --data ~/guidefam_work/data --pred "~/guidefam_work/tiger/predictions (other models)" \
    --out ~/guidefam_work/m4 --base-oof ~/guidefam_work/final_cas13d/hek_oof.csv --bag 5 --search 12
```
Trains four candidates (tuned GBM, Tiger2D with standardised inputs, 5-seed Tiger2D, TabDPT) leave-one-gene-out and prints scores on the **12 development genes only**, plus the best rank-average combinations. Pick one combination from that table, then — and only then:
```bash
python -m guidefam.scripts.m4_extend report --data ... --pred ... --out ~/guidefam_work/m4 --base-oof ... --use gbm,tiger_bag,konermann
```
That prints the 4 locked genes and the paired difference to Tiger2D, and writes your choice to `m4/chosen.json`. If you look at the locked genes first, say so: they stop being a test.

## 3. siRNA: combine with OligoFormer (the idea that worked for Cas13d) -- only if you hold a licence
OligoFormer is proprietary (payment, no modification, no competing use; see `RESULTS.md` section 7) and nothing from it is run here, so this step is yours to clear with the licensor. If you do: run OligoFormer from its README (`python scripts/main.py --datasets Hu Mix -t`, then `Hu Taka`), save `siRNA,score` CSVs, and
```bash
python -m guidefam.scripts.sirna_plus_oligoformer --ours ~/guidefam_work/final_sirna/preds_Mix.csv --theirs oligoformer_Mix.csv --name Mix
```
It reports each model and the rank-average with CIs and a paired difference. Check which data the OligoFormer checkpoint you run was trained on: using a checkpoint that saw Mixset/Takayuki invalidates the comparison.

## 4. What to claim, and what not to
* Allowed: the Cas13d gain over the Tiger2D re-implementation, with the caveats in `RESULTS.md` section 6; parity with OligoFormer on siRNA; the two protocol audits.
* Not allowed without new evidence: "beats TIGER" (it is a re-implementation), "beats OligoFormer", anything about HAP1, anything that includes the Konermann scores described as independent on the flow-cytometry screen.
* Before submitting anything, re-run `python -m guidefam.tests.test_basic`.
