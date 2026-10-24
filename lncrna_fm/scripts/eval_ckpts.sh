#!/usr/bin/env bash
# Score pretrained checkpoints on LncBench with the SAME protocol used for every baseline.
# usage: scripts/eval_ckpts.sh <tag> <ckpt> [<ckpt> ...]      (tag names the output files)
set -euo pipefail
TAG=$1; shift
OUT=${LNCFM_RUNS:-runs}/eval; mkdir -p "$OUT"
PROV=(); for c in "$@"; do PROV+=("lncfm:$c"); done
python -u -m lncfm.bench --providers "${PROV[@]}" --tasks all --max-n 800 --trunc 400 --n-boot 300 --out "$OUT/${TAG}_cv.json"
python -u -m lncfm.bench --retrieval --providers "${PROV[@]}" --max-queries 300 --pool-size 3000 --trunc 400 --out "$OUT/${TAG}_retrieval.json"
