# lncfm: a local-first lncRNA foundation-model toolkit

**Start here for the benchmark: LncBench v2** (two layers, leakage-controlled, built to run on a Mac).

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export LNCFM_DATA=$HOME/lncfm_data LNCFM_RUNS=$HOME/lncfm_runs
scripts/m4_run_all.sh check      # selfcheck (incl. MPS vs CPU parity) + speed on your machine
scripts/m4_run_all.sh data       # public data: GENCODE, GTEx, ENCODE (fractionation + eCLIP), BeatAML / TCGA-LAML / TARGET
scripts/m4_run_all.sh quick      # baselines + frozen RiNALMo on 800 genes/task, 400 nt
scripts/m4_run_all.sh full       # everything, whole benchmark + locked test
scripts/m4_run_all.sh ladder     # frozen / LoRA / top-k / full fine-tuning of RiNALMo
scripts/m4_run_all.sh aml        # patient-level AML tasks with external validation
```

| layer | tasks | label source | what it asks |
|---|---|---|---|
| 1 molecular | `loc_reg/clf`, `tau_reg`, `testis_clf`, `brain_clf` | ENCODE fractionation, GTEx | where is it, in which tissue |
| 1 molecular | `rbp_K562`, `rbp_HepG2` (multi-label) | ENCODE eCLIP IDR peaks | which RNA-binding proteins bind it (expression- and length-controlled) |
| 1 sanity | `coding_clf` | GENCODE | easy tier only |
| 2b AML, gene level | `aml_vs_normal_*`, `aml_npm1_reg` | TARGET marrow vs normal; BeatAML + TCGA-LAML | which lncRNAs are leukaemia / NPM1-associated |
| 2a AML, patient level | `npm1`, `flt3_itd`, `eln_adverse`, `induction_cr`, `os` | BeatAML (dev) -> TCGA-LAML, TARGET (external) | does lncRNA expression carry genotype / risk / outcome |

Protocol: dev = chromosome-group 5-fold CV; locked test = chr8/13/18/21 (scored only with `--final`); every metric is
reported on all items and on the homology-pruned subset; sequence-free baselines (`expr_len`) and an untrained twin must
be in every comparison; paired bootstrap for model-vs-model claims (`lncbench report`, `scripts/compare.py`).

Measured results, caveats and the go/no-go criteria are in `PLAN.md` section 10; raw outputs are in `results/v2/`.

Everything below this line is the v1 toolkit (pretraining, Option B, TCGA solid tumours) and still works.

---

Everything here was written to run on a laptop (CPU / Apple MPS / CUDA, pure PyTorch, no custom kernels) and to be
**benchmarked against honest baselines before you spend days training**. Read `PLAN.md` first: it has the reasoning,
the measured pilot results, and what I could *not* test from the cloud sandbox I developed this in.

```
lncfm/
  download.py     resumable, gzip-verified downloads (GENCODE human v50 + mouse M39, GTEx v8, ENCODE fractions, ClinVar)
  gencode.py      FASTA + GTF -> transcript table (exon structure, canonical isoform, chromosome folds)
  tasks.py        LncBench task builders (localization, tissue specificity, coding sanity)
  orthology.py    synteny-defined human<->mouse positional lncRNA orthologs (no external orthology table)
  retrieval.py    zero-shot ortholog retrieval + alignment-free shared-10-mer baseline
  kmer.py         k-mer spectrum features (the baseline every model must beat)
  evalkit.py      identical probes for every model + bootstrap CIs, chromosome-grouped CV
  bench.py        LncBench runner:  python -m lncfm.bench --providers kmer lncfm:<ckpt> rinalmo-micro
  model.py        LncFM: conv stem -> stride-4 -> RoPE transformer -> upsample; presets tiny/small/base
  pretrain.py     span-MLM + exon-junction tagging + isoform contrastive; resumable
  embed.py        embedding providers (our model, random-init control, external HF models)
  cpt.py          "Option B": continual pretraining of an existing masked-LM on lncRNA
  speedtest.py    measure YOUR machine's throughput and project wall-clock; counts real FLOPs/nt
  variants.py, variant_eval.py   within-gene ClinVar ncRNA variant scoring (zero-shot MLM log-likelihood ratio)
  clinical/tcga.py, clinical/survival.py   TCGA lncRNA expression + survival benchmark
scripts/          run_option_b_frozen.sh, eval_ckpts.sh
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # plain `pip install torch` gives MPS on Apple silicon
export LNCFM_DATA=$HOME/lncfm_data       # ~1.5 GB of downloads + ~2 GB processed (+ ~120 MB per TCGA cohort)
export LNCFM_RUNS=$HOME/lncfm_runs
```

## Run order (each stage has a stop/go criterion in PLAN.md section 8)

```bash
# 1. data (GENCODE/GTEx/ENCODE ~1.5 GB; ENCODE alone is 57 files)
python -m lncfm.download gencode gtex encode variants
python -m lncfm.clinical.tcga download TCGA-LGG TCGA-KIRC TCGA-LUAD      # ~25 min per cohort through the GDC API

# 2. GATE 1: is there sequence signal at all?  (minutes)
python -m lncfm.bench --providers lengc kmer --tasks all
python -m lncfm.bench --retrieval --providers kmer shared10mer --max-queries 2000 --pool-size 10000

# 3. measure your machine, then pick a preset
python -m lncfm.speedtest --preset small --batch 32 --max-len 1024 --target-nt 1e9
python -m lncfm.speedtest --preset base  --batch 16 --max-len 1024 --target-nt 1e9

# 4. pretrain (resumable; re-run the same command after any interruption)
python -m lncfm.pretrain --preset small --steps 60000 --batch 32 --max-len 1024 \
       --snap-every 10000 --out $LNCFM_RUNS/small_full
python -m lncfm.pretrain --preset small --steps 60000 --batch 32 --max-len 1024 \
       --lambda-j 0 --lambda-c 0 --snap-every 10000 --out $LNCFM_RUNS/small_mlm    # ablation, same batches

# 5. GATE 2: does pretraining beat the untrained network and k-mers?
scripts/eval_ckpts.sh small $LNCFM_RUNS/small_full/ckpt_final.pt random:small $LNCFM_RUNS/small_mlm/ckpt_final.pt
python -m lncfm.variant_eval $LNCFM_RUNS/small_full/ckpt_final.pt random:small

# 6. Option B: frozen external model on the same protocol, then continual pretraining
scripts/run_option_b_frozen.sh rinalmo-micro 1000 2000
python -m lncfm.cpt --model rinalmo-micro --steps 5000 --freeze-bottom 6 --out $LNCFM_RUNS/cpt_rinalmo
python -m lncfm.bench --providers local:$LNCFM_RUNS/cpt_rinalmo/hf rinalmo-micro --tasks all --max-n 800 --trunc 400

# 7. Clinical track
python -m lncfm.clinical.survival --projects TCGA-LGG TCGA-KIRC TCGA-LUAD --emb random kmer lncfm:$LNCFM_RUNS/small_full/ckpt_final.pt
```

## Things that will bite you

* `--include-heldout` is **off** by default: pretraining excludes the validation/test chromosomes (chr8/13/18/21 and
  chr9/16/22) of the human lncRNAs so the pilot comparison is leak-free. Turn it on only for a final production model.
* External models that ship custom Python (NT-v2, HyenaDNA) are blocked unless you read their repo and
  `export LNCFM_TRUST_REMOTE_CODE=1`. RiNALMo / RNA-FM load through the `multimolecule` package and need no flag.
* Evo 2 and Caduceus are not wired in: Evo 2 needs an NVIDIA GPU (1B additionally needs a Hopper card for FP8);
  Caduceus imports `mamba_ssm`. See PLAN.md section 4.
* The MPS and CUDA code paths are written but **untested** (the development sandbox was CPU-only). Run
  `python -m lncfm.speedtest` first; if MPS is much slower than expected, check the fallback warnings it prints.
