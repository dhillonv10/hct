#!/usr/bin/env bash
# Frozen-feature arm of "Option B": external pretrained model vs k-mer vs our model, SAME inputs.
# usage: scripts/run_option_b_frozen.sh rinalmo-micro 400 800
set -euo pipefail
MODEL=${1:-rinalmo-micro}; TRUNC=${2:-400}; MAXN=${3:-800}
OUT=${LNCFM_RUNS:-runs}/option_b; mkdir -p "$OUT"
python -u -m lncfm.bench --providers "$MODEL" --tasks all --max-n "$MAXN" --trunc "$TRUNC" --n-boot 300 --out "$OUT/${MODEL}_cv.json"
python -u -m lncfm.bench --retrieval --providers "$MODEL" --max-queries 300 --pool-size 3000 --trunc "$TRUNC" --out "$OUT/${MODEL}_retrieval.json"
