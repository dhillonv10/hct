"""GEO corpus builder.   python scripts/geo_ingest.py discover | screen [--threads 8] | ingest [--workers 3] [--max-gb 80]

Excluded by construction: the 42 series that appear in any LongevityBench DNAm prompt, series that are super/sub-series
of them, series already in ComputAge, and every sample whose GSM appears in a benchmark prompt or in ComputAge."""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd

from epimod.data import geo
from epimod.lb.lite import load_lite_universe
from epimod.lb.store import lb_meta, lb_series
from epimod.lb.tasks import load_parsed
from epimod.paths import subdir

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["discover", "screen", "ingest"])
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--max-gb", type=float, default=80.0); ap.add_argument("--max-series", type=int, default=100000)
ap.add_argument("--unlabeled", action="store_true", help="ingest the screened series WITHOUT a usable age label (self-supervised pretraining data)")
ap.add_argument("--min-samples", type=int, default=8); ap.add_argument("--max-samples", type=int, default=1500)
a = ap.parse_args()
root = subdir("geo")

parsed = load_parsed(); lb_ser = lb_series(parsed)
computage = lb_meta(None)
excluded_series = set(lb_ser) | set(computage.DatasetID.unique())

if a.cmd == "discover":
    df = geo.discover(); print(len(df), "series;", df.gpl.str.split(";").explode().value_counts().head(6).to_dict())

elif a.cmd == "screen":
    df = pd.read_parquet(root / "series.parquet")
    df = df[df.taxon.str.contains("Homo sapiens", na=False) & df.gdstype.str.contains("Methylation profiling by array", na=False)
            & df.n_samples.between(a.min_samples, a.max_samples) & ~df.gse.isin(excluded_series)]
    print(len(df), "candidate series after metadata filters", flush=True)
    (root / "screen").mkdir(exist_ok=True)
    t0, n_ok = time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(geo.screen_one, g, root / "screen") for g in df.gse]
        for i, f in enumerate(as_completed(futs), 1):
            n_ok += bool(f.result()["ok"])
            if i % 100 == 0:
                print(f"{i}/{len(futs)} screened, {n_ok} usable, {round(time.time() - t0)}s", flush=True)
    print(f"done: {n_ok} usable of {len(futs)}")

else:
    recs = [json.loads(f.read_text()) for f in (root / "screen").glob("GSE*.json")]
    hold = set((root / "holdout.txt").read_text().split()) if (root / "holdout.txt").exists() else set()
    excluded_series |= hold
    ok = [r for r in recs if (not r["ok"] if a.unlabeled else r["ok"]) and "file" in r and r["gse"] not in excluded_series and not (set(r.get("related", [])) & excluded_series)]
    ok.sort(key=lambda r: r["bytes"])
    chosen, tot = [], 0.0
    for r in ok:
        if tot + r["bytes"] / 2**30 > a.max_gb or len(chosen) >= a.max_series:
            break
        chosen.append(r); tot += r["bytes"] / 2**30
    print(f"{len(ok)} usable series; ingesting {len(chosen)} ({tot:.1f} GB compressed)", flush=True)
    lite = load_lite_universe()
    skip = set(computage.index)
    for t in parsed.values():
        skip |= set(t.gsm) | set(t.gsm_b)
    store_dir = subdir("lb", "store_lite", "geo_unlabeled" if a.unlabeled else "geo")
    log = open(root / "ingest.jsonl", "a")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(geo.ingest_one, r["gse"], r["file"], r["gpl"], lite, skip, root, store_dir, not a.unlabeled, "meta_unlabeled" if a.unlabeled else "meta"): r["gse"] for r in chosen}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
            except Exception as e:
                res = {"gse": futs[f], "status": f"error: {str(e)[:150]}"}
            log.write(json.dumps(res) + "\n"); log.flush()
            if i % 10 == 0:
                m = geo.geo_unlabeled_meta() if a.unlabeled else geo.geo_meta()
                print(f"{i}/{len(futs)} series processed | {len(m)} samples from {m.DatasetID.nunique() if len(m) else 0} series | {round(time.time() - t0)}s", flush=True)
