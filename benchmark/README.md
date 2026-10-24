# GvHD-Bench (proof-of-concept) — data, scripts, cached results

The notebook that uses everything here is `notebooks/GvHD_FM_benchmark_blueprint.ipynb`
([open in Colab](https://colab.research.google.com/github/dhillonv10/hct/blob/claude/magical-johnson-9ff3l8/notebooks/GvHD_FM_benchmark_blueprint.ipynb)).

| Path | What it is |
|---|---|
| `data/hsct_bundle.npz`, `hsct_samples.csv` | log-scale expression (3,072 pipeline-vocabulary genes ∪ Hallmark members) and metadata for 9 public GEO studies, 838 samples after QC |
| `data/qc_removed.csv` | libraries removed at build time (4 empty skin libraries in GSE263557) |
| `data/geneformer_embeddings.npz` | frozen Geneformer 10M / 104M / 316M embeddings (+ random-weight twins) for the 4 whole-transcriptome RNA-seq studies |
| `data/hallmark50.json`, `vocab_genes.json`, `geo_census.json`, `bundle_info.json` | gene sets, vocabulary, GEO census, per-study provenance |
| `results/*.json` | cached output of the three slow notebook cells (permutation nulls, Geneformer within-study arms, label-efficiency curves); regenerate with `RECOMPUTE=True` |
| `build_bundle.py` | script **as run** that produced the bundle from the project's cached GEO downloads |
| `embed_geneformer.py`, `merge_embeddings.py`, `rest_pickle.py` | Geneformer embedding (rank-value encoding, 2nd-to-last layer, mean over gene tokens) and merge; dictionaries are read through a numpy-scalar-only unpickler |
| `evidence/` | logs of earlier experiments cited in Part C.3 of the notebook (not re-executed there) |

Accessions: GSE73809, GSE103569, GSE4624, GSE182610 / GSE182612 / GSE182678 (PtCy), GSE295359, GSE215068, GSE168116, GSE263557, GSE268164.
