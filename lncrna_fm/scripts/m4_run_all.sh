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
#   scripts/m4_run_all.sh sweep        # frozen-feature sweep: layers x pooling x tiling x stacking with k-mers (RiNALMo-micro)
#   scripts/m4_run_all.sh fms          # more foundation models: RNA-FM, RiNALMo-mega (best feature recipe from `sweep`)
#   scripts/m4_run_all.sh retrieval    # human->mouse positional-ortholog retrieval
#   scripts/m4_run_all.sh varscore     # ncRNA variant-effect scoring: ClinVar + gnomAD, RiNALMo/RNA-FM/structure/CADD, within-gene AUROC
#   scripts/m4_run_all.sh cpt          # LONG, resumable: continual pretraining of RiNALMo-micro on lncRNA + small ncRNAs (CPT_STEPS=20000)
#   scripts/m4_run_all.sh cpt_eval     # does the CPT model beat the frozen one?  variant scoring + RBP/localization probes
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
  sweep)
    R=rinalmo-micro
    python -m lncfm.lncbench run --providers kmer "$R?layers=last&pool=mean" "$R?layers=mid&pool=mean" "$R?layers=mid&pool=meanmax" \
        "$R?layers=4,8,last&pool=meanmax" "$R?layers=mid&pool=meanmax&win=3" "concat:kmer+$R?layers=mid&pool=meanmax" \
        --tasks loc_reg loc_clf tau_reg testis_clf brain_clf rbp_K562 rbp_HepG2 rbp_union aml_vs_normal_resid_reg aml_vs_normal_resid_clf \
        --trunc 512 --final --save-oof --out "$OUT/sweep.json"
    python -m lncfm.lncbench report "$OUT/sweep.json" --scope dev_pruned ;;
  fms)
    BEST=${BEST:-"?layers=mid&pool=meanmax"}          # set to the winner of `sweep`
    python -m lncfm.lncbench run --providers "rnafm$BEST" "rinalmo-mega$BEST" "concat:kmer+rinalmo-mega$BEST" \
        --tasks loc_reg loc_clf tau_reg testis_clf brain_clf rbp_K562 rbp_HepG2 rbp_union aml_vs_normal_resid_reg aml_vs_normal_resid_clf \
        --trunc 512 --final --save-oof --out "$OUT/fms.json"
    python -m lncfm.lncbench report "$OUT/fms.json" "$OUT/sweep.json" --scope dev_pruned ;;
  retrieval)
    python -m lncfm.bench --retrieval --providers kmer shared10mer rinalmo-micro --max-queries 2000 --pool-size 10000 --trunc 400 \
        --out "$OUT/retrieval.json" ;;
  varscore)
    python -m lncfm.gnomad                                 # builds + prints the evaluation table (needs internet: gnomAD API)
    python -m lncfm.varscore eval --models rinalmo-micro --combine \
        --save-combiner "$OUT/combiner_micro.json" --out "$OUT/varscore_micro_nocadd.json"      # offline combiner for `varscore score`
    python -m lncfm.varscore eval --models rinalmo-micro --cadd --combine \
        --save-combiner "$OUT/combiner_micro_cadd.json" --out "$OUT/varscore_micro.json"        # + CADD (internet); best nuclear-gene scorer
    python -m lncfm.varscore eval --models rinalmo-micro rinalmo-mega rnafm --no-emb --cadd --combine \
        --save-combiner "$OUT/combiner_all_cadd.json" --out "$OUT/varscore_main.json"          # --no-emb: embedding-distance scores were null
    # sensitivity: do the conclusions depend on how the benign set is built?
    python -m lncfm.varscore eval --models rinalmo-micro --bg clinvar --out "$OUT/varscore_clinvar_only.json"
    python -m lncfm.varscore eval --models rinalmo-micro --af-min 0.001 --out "$OUT/varscore_af1e-3.json"
    python -m lncfm.varscore eval --models rinalmo-micro --min-stars 1 --out "$OUT/varscore_stars1.json"
    python -m lncfm.varscore eval --models rinalmo-micro --bg any --out "$OUT/varscore_any.json" ;;   # adds TERC (background = any observed variant)
  cpt)
    # Resumable: if interrupted, run the same command again. Watch $LNCFM_RUNS/cpt_<model>_nc/log.jsonl (val_mlm must go DOWN).
    M=${CPT_MODEL:-rinalmo-micro}
    python -m lncfm.cpt --model "$M" --steps "${CPT_STEPS:-20000}" --batch 16 --max-nt 512 --lr 3e-5 --replay 0.2 \
        --ncrna 0.15 --save-every 500 --out "${LNCFM_RUNS:-runs}/cpt_${M}_nc" ;;
  cpt_eval)
    M=${CPT_MODEL:-rinalmo-micro}
    C="local:${LNCFM_RUNS:-runs}/cpt_${M}_nc/hf"
    python -m lncfm.varscore eval --models "$M" "$C" --out "$OUT/varscore_cpt_$M.json"
    python -m lncfm.lncbench run --providers "$M" "$C" --tasks loc_clf loc_reg tau_reg testis_clf rbp_K562 rbp_HepG2 rbp_union \
        --trunc 1000 --final --save-oof --out "$OUT/cpt_probe_$M.json"
    python -m lncfm.lncbench report "$OUT/cpt_probe_$M.json" --scope dev_pruned ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
