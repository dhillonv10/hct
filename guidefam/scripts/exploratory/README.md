# Exploratory scripts (provenance only)

These produced individual numbers quoted in `RESULTS.md` / `PROTOCOL.md` (negative results, spatial autocorrelation, per-gene breakdown, noise ceilings,
data-format checks). They were written in a scratch directory: **paths at the top of each file are sandbox-specific** (`S = ".../scratchpad"`) and need editing
to your layout (data in `WORK/data`, results in the folders under `guidefam/results/`). They are not part of the tested pipeline (`run_all.sh`).

| script | produces |
|---|---|
| `resid_spatial.py` | activity / residual correlation vs distance between guides (PROTOCOL section 1) |
| `smooth_test.py` | neighbour smoothing of predictions (worse at every strength) |
| `dev_rank.py` | LambdaRank objective and 5-seed GBM bagging (no gain) |
| `stack_pub.py`, `dev_stack.py`, `dev_combo.py` | combining GBM / Tiger2D / published scores on the 12 development genes |
| `per_gene.py`, `flow_paired.py` | per-gene breakdown and the paired comparison on the flow screen |
| `hap_ceiling.py` | within-gene replicate noise ceilings for HEK293T, flow, HAP1 |
| `denorm_debug.py` | decomposition that exposed the index-alignment bug in the first denormalisation audit |
| `quick_dev.py` | point estimates for every column of `results/cas13d_dev/dev_oof.csv` |
| `sirna_dev_tabdpt.py` | TabDPT as an siRNA learner (neutral) |
| `win_check.py`, `sirna_peek.py`, `profile_tiger.py`, `map_comparators.py` | data-format checks (window layout, orientation, group structure, guide mapping) |
