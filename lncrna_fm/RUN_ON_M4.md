# Run LncBench v2 on your M4

## 0. Get the code (pick one)
* Unzip `lncrna_fm.zip` (this folder), or
* `git clone -b claude/optimistic-albattani-fwn08z https://github.com/dhillonv10/hct.git && cd hct/lncrna_fm`

## 1. Environment (once)
```bash
cd lncrna_fm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # if anything breaks: pip install -r requirements-tested.txt
export LNCFM_DATA=$HOME/lncfm_data LNCFM_RUNS=$HOME/lncfm_runs      # put these two lines in ~/.zshrc
```

## 2. Stages, in this order (each is resumable; re-run it after any interruption)
| command | what it does | roughly |
|---|---|---|
| `scripts/m4_run_all.sh check` | self-test incl. MPS-vs-CPU parity + measures your speed | 1-2 min |
| `scripts/m4_run_all.sh data` | downloads GENCODE, GTEx, ENCODE, BeatAML/TCGA-LAML/TARGET | 2-3 GB, 1-3 h (mostly network) |
| `scripts/m4_run_all.sh quick` | baselines + frozen RiNALMo, 800 genes, 400 nt. Reproduces PLAN section 10 | 1-2 h |
| `scripts/m4_run_all.sh full` | whole benchmark, 1000-nt inputs, locked test | several hours |
| `scripts/m4_run_all.sh ladder` | frozen vs LoRA vs top-2 vs full fine-tune of RiNALMo | hours |
| `scripts/m4_run_all.sh aml` | patient-level AML tasks, external validation | < 1 h |
| `scripts/m4_run_all.sh genotype` | 14-gene AML genotype panel | minutes |
| `scripts/m4_run_all.sh variants` / `fms` / `retrieval` | optional: feature sweep, RNA-FM + RiNALMo-mega, ortholog retrieval | hours |

Do `check` first. If it reports an MPS/CPU mismatch or a very low speed, stop and send me the output.
The model decision (section 11.5 of PLAN.md) hinges on the `ladder` stage: does adapting RiNALMo beat the frozen probe on
`rbp_K562`, `rbp_HepG2`, `loc_clf`?

## 3. Look at results
```bash
python -m lncfm.lncbench report $LNCFM_RUNS/v2/full_*.json --scope dev_pruned
python -m lncfm.lncbench compare "$LNCFM_RUNS/v2/full_rinalmo.json:rinalmo" "$LNCFM_RUNS/v2/full_baselines.json:kmer"
```
Result JSON/text files land in `$LNCFM_RUNS/v2`. Send me that folder (without `emb/`) and I will read it with you.

## Notes
* The locked test set (chr8/13/18/21) is only scored with `--final`; do not tune on it.
* RiNALMo / RNA-FM weights download from Hugging Face on first use (AGPL-3.0).
* Not tested in the sandbox: the MPS path, the full ladder, and the full-size data. `check` tells you quickly if MPS is sane.
