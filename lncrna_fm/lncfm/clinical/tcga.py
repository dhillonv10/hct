"""TCGA (GDC open access) lncRNA expression + survival, for the clinical track.

    python -m lncfm.clinical.tcga download TCGA-LGG TCGA-KIRC TCGA-LUAD --max-files 400

Writes, per project, $LNCFM_DATA/tcga/<project>/{expr_tpm.pkl, clinical.pkl}:
  expr_tpm.pkl : DataFrame (patients x genes) of log2(TPM+1), columns = ENSG (no version)
  gene_meta.pkl: gene_id -> (gene_name, gene_type)           (shared across projects)
  clinical.pkl : DataFrame indexed by patient with os_time (days), os_event, age, stage, gender
One primary-tumour sample per patient (first by file order).
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time

import numpy as np
import pandas as pd
import requests

from .. import paths

GDC = "https://api.gdc.cancer.gov"


def _post_json(endpoint, payload, retries=5):
    for i in range(retries):
        try:
            r = requests.post(GDC + endpoint, json=payload, timeout=300)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {i+1}: {e!r}")
            time.sleep(2 ** (i + 1))
    raise RuntimeError(endpoint)


def list_files(project: str, max_files: int | None):
    flt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]}
    r = requests.get(GDC + "/files", params={
        "filters": json.dumps(flt), "format": "JSON", "size": 3000,
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type,cases.samples.submitter_id"}, timeout=120)
    r.raise_for_status()
    rows, seen = [], set()
    for h in r.json()["data"]["hits"]:
        case = h["cases"][0]["submitter_id"]
        if case in seen:
            continue
        seen.add(case)
        rows.append((h["file_id"], case))
    return rows[:max_files] if max_files else rows


def download_expression(project: str, max_files: int | None = None, batch: int = 40):
    out = paths.ensure(paths.DATA / "tcga" / project)
    if (out / "expr_tpm.pkl").exists():
        return
    files = list_files(project, max_files)
    print(f"{project}: {len(files)} patients")
    id2case = dict(files)
    cols, meta = {}, None
    for b in range(0, len(files), batch):
        ids = [f for f, _ in files[b:b + batch]]
        r = _post_json("/data", {"ids": ids})
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".tsv"):
                    continue
                fid = m.name.split("/")[0]
                df = pd.read_csv(tf.extractfile(m), sep="\t", comment="#")
                df = df[df.gene_id.str.startswith("ENSG")]
                if meta is None:
                    meta = df.set_index("gene_id")[["gene_name", "gene_type"]]
                cols[id2case[fid]] = df.set_index("gene_id")["tpm_unstranded"]
        print(f"  {min(b+batch, len(files))}/{len(files)}", flush=True)
    X = pd.DataFrame(cols).T
    X.columns = X.columns.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    X = np.log2(X.loc[:, ~X.columns.duplicated()] + 1).astype(np.float32)
    meta.index = meta.index.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    meta = meta[~meta.index.duplicated()]
    X.to_pickle(out / "expr_tpm.pkl")
    meta.to_pickle(paths.DATA / "tcga" / "gene_meta.pkl")


def download_clinical(project: str):
    out = paths.ensure(paths.DATA / "tcga" / project)
    if (out / "clinical.pkl").exists():
        return
    flt = {"op": "in", "content": {"field": "project.project_id", "value": [project]}}
    r = requests.get(GDC + "/cases", params={
        "filters": json.dumps(flt), "format": "JSON", "size": 3000,
        "fields": "submitter_id,demographic.vital_status,demographic.days_to_death,demographic.gender,"
                  "diagnoses.days_to_last_follow_up,diagnoses.age_at_diagnosis,diagnoses.ajcc_pathologic_stage,"
                  "follow_ups.days_to_follow_up"}, timeout=120)
    r.raise_for_status()
    rows = []
    for h in r.json()["data"]["hits"]:
        dem, dx = h.get("demographic", {}) or {}, (h.get("diagnoses") or [{}])[0]
        dead = dem.get("vital_status") == "Dead"
        # censoring time = latest recorded contact (diagnosis-level or any follow-up visit), as in TCGA-CDR
        contacts = [dx.get("days_to_last_follow_up")] + [f.get("days_to_follow_up") for f in (h.get("follow_ups") or [])]
        contacts = [c for c in contacts if c is not None]
        t = dem.get("days_to_death") if dead else (max(contacts) if contacts else None)
        age = dx.get("age_at_diagnosis")
        rows.append({"patient": h["submitter_id"], "os_time": t, "os_event": int(dead),
                     "age": age / 365.25 if age else np.nan, "stage": dx.get("ajcc_pathologic_stage"),
                     "gender": dem.get("gender")})
    df = pd.DataFrame(rows).set_index("patient")
    df = df[df.os_time.notna() & (df.os_time > 0)]
    df.to_pickle(out / "clinical.pkl")


def load(project: str):
    d = paths.DATA / "tcga" / project
    X, C = pd.read_pickle(d / "expr_tpm.pkl"), pd.read_pickle(d / "clinical.pkl")
    ids = X.index.intersection(C.index)
    return X.loc[ids], C.loc[ids]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download"])
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args(argv)
    for p in a.projects:
        download_clinical(p)
        download_expression(p, a.max_files)


if __name__ == "__main__":
    main()
