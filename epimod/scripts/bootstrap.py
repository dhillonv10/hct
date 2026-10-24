"""Build every dataset from public sources on a fresh machine (resumable: finished work is skipped).

    export EPIMOD_DATA=$HOME/epimod_data          # ~20 GB free is enough; most downloads are deleted after processing
    python scripts/bootstrap.py --workers 4 [--only parsed computage ...] [--skip twins]

Stages, in order:
  parsed          LongevityBench DNAm prompts (Hugging Face)  -> lb/parsed.pkl
  computage       ComputAge Bench studies (HF) streamed once   -> lb/store_lite/{train,benchmark}   (compact 9,098-CpG panel)
  geo-discover    NCBI E-utilities: human array-methylation series on 27K/450K/EPIC/EPICv2
  geo-holdout     series about haematological malignancy (AML, MDS, leukaemia ...) held out of every training set
  geo-screen      read each series header: age / sex / tissue availability
  geo-labeled     series with an age label                     -> lb/store_lite/geo
  geo-unlabeled   series without an age label (masked-reconstruction pretraining only) -> lb/store_lite/geo_unlabeled
  twins           E-MTAB-7309 older twins, repeated blood, raw IDATs -> betas in Python (needs ~10 GB of downloads)
  audit           leakage audit of the training corpus against the benchmark prompts"""
import argparse, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=3); ap.add_argument("--only", nargs="*"); ap.add_argument("--skip", nargs="*", default=[])
ap.add_argument("--max-gb-labeled", type=float, default=60); ap.add_argument("--max-gb-unlabeled", type=float, default=120)
a = ap.parse_args()
here = Path(__file__).resolve().parent
PY = sys.executable


def run(*cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def code(src):
    return [PY, "-c", src]


STAGES = {
    "parsed": lambda: run(*code("from epimod.lb.fetch import build_parsed; print(build_parsed())")),
    "computage": lambda: run(*code(f"from epimod.lb.tasks import load_parsed; from epimod.lb.lite import extract_lite; extract_lite(load_parsed(), workers={a.workers})")),
    "geo-discover": lambda: run(PY, here / "geo_ingest.py", "discover"),
    "geo-holdout": lambda: run(*code("from epimod.data.geo import heme_holdout; h = heme_holdout(); print(len(h), 'series held out')")),
    "geo-screen": lambda: run(PY, here / "geo_ingest.py", "screen", "--threads", "8"),
    "geo-labeled": lambda: run(PY, here / "geo_ingest.py", "ingest", "--workers", a.workers, "--max-gb", a.max_gb_labeled),
    "geo-unlabeled": lambda: run(PY, here / "geo_ingest.py", "ingest", "--unlabeled", "--workers", a.workers, "--max-gb", a.max_gb_unlabeled),
    "twins": lambda: run(PY, here / "emtab_ingest.py"),
    "audit": lambda: run(PY, here / "lb_audit.py"),
}
for name, fn in STAGES.items():
    if (a.only and name not in a.only) or name in a.skip:
        continue
    t0 = time.time(); print(f"\n===== {name} =====", flush=True); fn(); print(f"===== {name} done in {(time.time() - t0) / 60:.1f} min =====", flush=True)
