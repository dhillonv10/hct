#!/usr/bin/env bash
# Train the full model and one-feature-off ablations with an identical budget, then score them all
# (plus ElasticNet, ridge and the published clocks) on the held-out ComputAge benchmark studies.
#
#   EPIMOD_DATA=/path/to/data RUNS=./runs EPOCHS=15 bash scripts/run_ablations.sh
set -euo pipefail
RUNS=${RUNS:-./runs}
EPOCHS=${EPOCHS:-15}
THREADS=${THREADS:-4}
PY=${PY:-python}

run() {  # run NAME [model overrides...] -- [train overrides...]
  local name=$1; shift
  local model=() train=("epochs=${EPOCHS}" "threads=${THREADS}")
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do model+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  train+=("$@")
  if [ -f "${RUNS}/${name}/model.pt" ]; then echo "[skip] ${name}"; return; fi
  echo "[train] ${name}: model=${model[*]:-} train=${train[*]}"
  ${PY} -u -m epimod.cli train --out "${RUNS}/${name}" ${model:+--model "${model[@]}"} --train "${train[@]}" \
    2>&1 | grep --line-buffered -v Warning | tee "${RUNS}/${name}.log" | tail -n 3
}

mkdir -p "${RUNS}"
run full
run full_seed1      --  seed=1                      # same studies, different init: a noise floor for the ablations
run no_context      use_genomic_context=False
run no_id           use_id_embedding=False
run no_module_mixing n_layers=0
run point_estimate  distributional=False
run no_aug          --  aug=False
run no_ssl          --  w_ssl=0.0
run no_linear_route linear_skip=False

${PY} -m epimod.cli evaluate --out "${RUNS}/eval" --run \
  full="${RUNS}/full" full_seed1="${RUNS}/full_seed1" no_context="${RUNS}/no_context" no_id="${RUNS}/no_id" \
  no_module_mixing="${RUNS}/no_module_mixing" point_estimate="${RUNS}/point_estimate" \
  no_aug="${RUNS}/no_aug" no_ssl="${RUNS}/no_ssl" no_linear_route="${RUNS}/no_linear_route"
