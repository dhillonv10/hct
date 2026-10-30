#!/usr/bin/env bash
# ASO dev stage (development genes only): GBM ablation + split-leakage test, then Tier-1 net + rank ensemble.
# Resumable: finished models are cached under $CACHE (oof_*.npy, net_*_fold*.npy); re-running skips them.
# Usage (from the repo root):  bash guidefam/scripts/run_aso_dev.sh [CSV] [CACHE] [OUT]
set -euo pipefail
CSV=${1:-data_untrusted/aso_zenodo_20794660/aso.csv.gz}
CACHE=${2:-data_untrusted/cache}
OUT=${3:-guidefam/results/aso}
mkdir -p "$OUT" "$CACHE"
if [ ! -f "$CACHE/aso_access.npy" ]; then
  python3 -m guidefam.scripts.aso_access --csv "$CSV" --out "$CACHE"
fi
python3 -m guidefam.scripts.aso_dev  --csv "$CSV" --cache "$CACHE" --out "$OUT" 2>&1 | tee "$OUT/aso_dev.log"
python3 -m guidefam.scripts.aso_dev2 --csv "$CSV" --cache "$CACHE" --out "$OUT" 2>&1 | grep -v Warning | tee "$OUT/aso_dev2.log"
