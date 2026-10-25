#!/usr/bin/env bash
# LncBench v2 on your Mac, in stages. Every stage is resumable (downloads and embeddings are cached).
#   export LNCFM_DATA=$HOME/lncfm_data LNCFM_RUNS=$HOME/lncfm_runs
#   scripts/m4_run_all.sh check        # 1 min: correctness + speed on YOUR machine
#   scripts/m4_run_all.sh data         # downloads (~2-3 GB; TARGET/BeatAML RNA-seq take the longest)
#   scripts/m4_run_all.sh quick        # baselines + frozen RiNALMo on 800 genes/task, 400 nt   (~1-2 h)
#   scripts/m4_run_all.sh full         # whole benchmark, longer inputs, locked test             (several hours)
#   scripts/m4_run_all.sh ladder       # frozen / LoRA / top-k / full fine-tuning of RiNALMo-micro
#   scripts/m4_run_all.sh aml          # patient-level AML layer with external validation
#   scripts/m4_run_all.sh genotype     # which AML driver genotypes can be read from lncRNA expression
#   scripts/m4_run_all.sh variants     # frozen-feature sweep: layers x pooling x tiling x stacking with k-mers (RiNALMo-micro)
#   scripts/m4_run_all.sh fms          # more foundation models: RNA-FM, RiNALMo-mega (best feature recipe from `variants`)
#   scripts/m4_run_all.sh retrieval    # human->mouse positional-ortholog retrieval
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=${LNCFM_RUNS:-runs}/v2; mkdir -p "$OUT"
stage=${1:-check}
case "$stage" in
  check)
    python -m lncfm.selfcheck
    python -m lncfm.speedtest --preset small --batch 16 --max-len 512 --steps 8 --target-nt 1e9 ;;
  data)
    python -m lncfm.download gencode gtex encode variants
    python -m lncfm.eclip download && python -m lncfm.eclip build
    python -m lncfm.aml.data clinical
    python -m lncfm.aml.data download --target-aml 600 --target-normal 400 ;;      # raise for the full cohorts
  quick)
    # gene-level tasks subsampled to 800 genes; the RBP tasks must run at FULL size (at 800 genes they have ~10 positives
    # per RBP and every model collapses to chance)
    python -m lncfm.lncbench run --providers expr_len lengc kmer cnn rinalmo-micro \
        --tasks loc_reg loc_clf tau_reg testis_clf brain_clf coding_clf aml_vs_normal_reg aml_vs_normal_resid_reg aml_vs_normal_resid_clf aml_npm1_reg \
        --trunc 400 --max-n 800 --final --save-oof --out "$OUT/quick_a.json"
    python -m lncfm.lncbench run --providers expr_len lengc kmer cnn rinalmo-micro --tasks rbp_K562 rbp_HepG2 \
        --trunc 400 --final --save-oof --out "$OUT/quick_b.json"
    for s in dev_pruned test_pruned; do python -m lncfm.lncbench report "$OUT"/quick_a.json "$OUT"/quick_b.json --scope $s; done ;;
  full)
    python -m lncfm.lncbench run --providers expr_len lengc kmer cnn --tasks all --final --save-oof --out "$OUT/full_baselines.json"
    python -m lncfm.lncbench run --providers rinalmo-micro --tasks all --trunc 1000 --final --save-oof --out "$OUT/full_rinalmo.json"
    for s in dev_pruned test_pruned; do python -m lncfm.lncbench report "$OUT"/full_*.json --scope $s; done ;;
  ladder)
    for t in loc_reg tau_reg rbp_K562; do
      python -m lncfm.finetune --task $t --modes frozen lora topk full --folds 0 1 2 3 4 --epochs 4 --batch 16 --max-nt 512 \
          --out "$OUT/ft_$t.json"
    done ;;
  aml)
    python -m lncfm.aml.patient --reps 3 --emb random kmer rinalmo-micro --out "$OUT/aml_patient.json" ;;
  genotype)
    python -m lncfm.aml.genotype --reps 3 --out "$OUT/aml_genotype.json" ;;
  variants)
    R=rinalmo-micro
    python -m lncfm.lncbench run --providers kmer "$R?layers=last&pool=mean" "$R?layers=mid&pool=mean" "$R?layers=mid&pool=meanmax" \
        "$R?layers=4,8,last&pool=meanmax" "$R?layers=mid&pool=meanmax&win=3" "concat:kmer+$R?layers=mid&pool=meanmax" \
        --tasks loc_reg loc_clf tau_reg testis_clf brain_clf rbp_K562 rbp_HepG2 rbp_union aml_vs_normal_resid_reg aml_vs_normal_resid_clf \
        --trunc 512 --final --save-oof --out "$OUT/variants.json"
    python -m lncfm.lncbench report "$OUT/variants.json" --scope dev_pruned ;;
  fms)
    BEST=${BEST:-"?layers=mid&pool=meanmax"}          # set to the winner of `variants`
    python -m lncfm.lncbench run --providers "rnafm$BEST" "rinalmo-mega$BEST" "concat:kmer+rinalmo-mega$BEST" \
        --tasks loc_reg loc_clf tau_reg testis_clf brain_clf rbp_K562 rbp_HepG2 rbp_union aml_vs_normal_resid_reg aml_vs_normal_resid_clf \
        --trunc 512 --final --save-oof --out "$OUT/fms.json"
    python -m lncfm.lncbench report "$OUT/fms.json" "$OUT/variants.json" --scope dev_pruned ;;
  retrieval)
    python -m lncfm.bench --retrieval --providers kmer shared10mer rinalmo-micro --max-queries 2000 --pool-size 10000 --trunc 400 \
        --out "$OUT/retrieval.json" ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
