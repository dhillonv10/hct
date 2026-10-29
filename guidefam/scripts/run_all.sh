#!/usr/bin/env bash
# End-to-end reproduction on a laptop (CPU only). Run from the repository root:  bash guidefam/scripts/run_all.sh WORKDIR
#   ~15 min without Tiger2D; the leave-one-gene-out Tiger2D baseline adds ~1-2 h on 4 cores (set SKIP_TIGER=1 to skip it).
set -euo pipefail
W=${1:?usage: run_all.sh WORKDIR}
mkdir -p "$W" && cd "$(dirname "$0")/../.." 

# 0. third-party data/code (public GitHub repos; nothing is executed from them)
[ -d "$W/tiger" ]       || git clone --depth 1 https://github.com/daklab/tiger.git "$W/tiger"
[ -d "$W/OligoFormer" ] || git clone --depth 1 https://github.com/lulab/OligoFormer.git "$W/OligoFormer"
PRED="$W/tiger/predictions (other models)"

# 1. TIGER ships pandas-1.x pickles. They are read through an exact allowlist (guidefam/safepickle.py) and written back out as CSV.
python -c "from guidefam import data; data.export_tiger('$W/tiger/data-processed', '$W/data')"

# 2. unit tests for the invariants that would silently invalidate results
python -m guidefam.tests.test_basic

# 3. published Cas13d tools scored on the HEK293T guides, then the baseline ladder (GBM / ridge; Tiger2D optional)
python -m guidefam.scripts.eval_published --data "$W/data" --pred "$PRED"
python -m guidefam.scripts.ladder_cas13d --data "$W/data" --pred "$PRED" --out "$W/results" --models scalars_gbm,seq_ridge,gbm_core,gbm_full
if [ -z "${SKIP_TIGER:-}" ]; then
  OMP_NUM_THREADS=3 python -m guidefam.scripts.ladder_cas13d --data "$W/data" --pred "$PRED" --out "$W/results_tiger" --models tiger2d_port
  python -m guidefam.scripts.final_cas13d --data "$W/data" --pred "$PRED" --tiger-oof "$W/results_tiger/oof_hek293t_logo.csv" --out "$W/final_cas13d"
fi

# 4. protocol audits
python -m guidefam.scripts.audit_denorm --data "$W/data" --oof "$W/results/oof_hek293t_logo.csv"

# 5. siRNA: development, then the one-shot Huesken -> Mixset/Takayuki evaluation, then leakage + leave-one-dataset-out
python -m guidefam.scripts.sirna_dev --oligo "$W/OligoFormer/data"
python -m guidefam.scripts.final_sirna --oligo "$W/OligoFormer/data" --out "$W/final_sirna"
python -m guidefam.scripts.sirna_leakage_lodo --oligo "$W/OligoFormer/data"
