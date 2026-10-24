"""AML cohorts: RNA-seq from the GDC (open access) + curated clinical/genetic data from cBioPortal.

    python -m lncfm.aml.data download                       # sandbox-sized defaults
    python -m lncfm.aml.data download --target-aml 1700 --target-normal 400 --beat 400   # full-ish (hours)

Cohorts
  beataml   BEATAML1.0-COHORT   adult, primary (initial-diagnosis) bone marrow / blood RNA-seq   -> development
  tcga_laml TCGA-LAML           adult, primary peripheral blood                                   -> external validation
  target    TARGET-AML          paediatric AML bone marrow (+ normal marrow for AML-vs-normal DE) -> shifted external

Everything is stored under $LNCFM_DATA/aml/<cohort>/{expr.pkl,samples.pkl}; expr = log2(TPM+1), samples x ENSG.
Clinical tables come from cBioPortal studies aml_ohsu_2022 / aml_tcga_gdc / aml_target_gdc.
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
CBIO = "https://www.cbioportal.org/api"

COHORTS = {
    "beataml": dict(project="BEATAML1.0-COHORT",
                    types=["Primary Blood Derived Cancer - Bone Marrow", "Primary Blood Derived Cancer - Peripheral Blood"]),
    "tcga_laml": dict(project="TCGA-LAML", types=["Primary Blood Derived Cancer - Peripheral Blood"]),
    "target_aml": dict(project="TARGET-AML", types=["Primary Blood Derived Cancer - Bone Marrow"]),
    "target_normal": dict(project="TARGET-AML", types=["Bone Marrow Normal"]),
}


def _post(endpoint, payload, retries=5):
    for i in range(retries):
        try:
            r = requests.post(GDC + endpoint, json=payload, timeout=300)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {i+1}: {e!r}", flush=True)
            time.sleep(2 ** (i + 1))
    raise RuntimeError(endpoint)


def list_files(project, sample_types, max_files=None, seed=0, allow_cases=None):
    flt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": sample_types}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
    r = requests.get(GDC + "/files", params={"filters": json.dumps(flt), "format": "JSON", "size": 5000,
                     "fields": "file_id,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"}, timeout=180)
    r.raise_for_status()
    rows, seen = [], set()
    for h in r.json()["data"]["hits"]:
        c = h["cases"][0]
        case = c["submitter_id"]
        if case in seen or (allow_cases is not None and case not in allow_cases):   # one specimen per patient
            continue
        seen.add(case)
        rows.append((h["file_id"], case, c["samples"][0]["submitter_id"], c["samples"][0]["sample_type"]))
    if max_files and len(rows) > max_files:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), max_files, replace=False))]
    return rows


def download_cohort(name, max_files=None, batch=40, allow_cases=None):
    spec = COHORTS[name]
    out = paths.ensure(paths.DATA / "aml" / name)
    if (out / "expr.pkl").exists():
        print(name, "already downloaded")
        return
    files = list_files(spec["project"], spec["types"], max_files, allow_cases=allow_cases)
    print(f"{name}: {len(files)} specimens", flush=True)
    meta = {f[0]: f for f in files}
    cols, gmeta = {}, None
    for b in range(0, len(files), batch):
        ids = [f[0] for f in files[b:b + batch]]
        for attempt in range(6):                      # the GDC sometimes returns a truncated archive: retry the batch
            try:
                r = _post("/data", {"ids": ids})
                got = {}
                with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        if not m.name.endswith(".tsv"):
                            continue
                        fid = m.name.split("/")[0]
                        df = pd.read_csv(tf.extractfile(m), sep="\t", comment="#")
                        df = df[df.gene_id.str.startswith("ENSG")]
                        got[fid] = df
                break
            except (EOFError, tarfile.TarError, OSError, pd.errors.ParserError) as e:
                print(f"  batch {b} attempt {attempt+1} failed: {e!r}", flush=True)
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"batch {b} failed repeatedly")
        for fid, df in got.items():
            if gmeta is None:
                gmeta = df.set_index("gene_id")[["gene_name", "gene_type"]]
            cols[meta[fid][2]] = df.set_index("gene_id")["tpm_unstranded"]
        print(f"  {name} {min(b+batch, len(files))}/{len(files)}", flush=True)
    X = pd.DataFrame(cols).T
    X.columns = X.columns.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    X = np.log2(X.loc[:, ~X.columns.duplicated()] + 1).astype(np.float32)
    gmeta.index = gmeta.index.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)
    gmeta = gmeta[~gmeta.index.duplicated()]
    S = pd.DataFrame([{"sample_id": f[2], "case_id": f[1], "sample_type": f[3]} for f in files]).set_index("sample_id")
    S.to_pickle(out / "samples.pkl")
    gmeta.to_pickle(paths.ensure(paths.DATA / "aml") / "gene_meta.pkl")
    X.to_pickle(out / "expr.pkl")


# --------------------------------------------------------------------------- cBioPortal clinical
def _cbio(path, **params):
    r = requests.get(CBIO + path, params=params, headers={"accept": "application/json"}, timeout=180)
    r.raise_for_status()
    return r.json()


def _pivot(rows):
    d = pd.DataFrame(rows)
    return d.pivot_table(index="patientId" if "sampleId" not in d else "sampleId", columns="clinicalAttributeId",
                         values="value", aggfunc="first")


def beataml_clinical() -> pd.DataFrame:
    """One row per patient, initial-diagnosis specimen attributes. Index = GDC case id (e.g. '2000')."""
    f = paths.ensure(paths.DATA / "aml") / "beataml_clinical.pkl"
    if f.exists():
        return pd.read_pickle(f)
    sid = "aml_ohsu_2022"
    pat = pd.DataFrame(_cbio(f"/studies/{sid}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY", pageSize=100000))
    smp = pd.DataFrame(_cbio(f"/studies/{sid}/clinical-data", clinicalDataType="SAMPLE", projection="SUMMARY", pageSize=100000))
    P = pat.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    S = smp.pivot_table(index=["patientId", "sampleId"], columns="clinicalAttributeId", values="value", aggfunc="first").reset_index()
    S = S[S.DISEASE_STAGE_AT_SPECIMEN_COLLECTION == "Initial Diagnosis"].drop_duplicates("patientId").set_index("patientId")
    D = P.join(S, how="inner", lsuffix="_p", rsuffix="")
    D.index = D.index.str.replace(sid + "_", "", regex=False)
    out = pd.DataFrame(index=D.index)
    os_m = pd.to_numeric(D.get("OS_MONTHS"), errors="coerce")
    out["os_days"] = os_m * 30.4375
    out["os_event"] = D["OS_STATUS"].astype(str).str.startswith("1").astype(float).where(D["OS_STATUS"].notna())
    out["age"] = pd.to_numeric(D.get("AGE_AT_DIAGNOSIS"), errors="coerce")
    out["eln2017"] = D.get("ELN2017")
    out["npm1"] = D.get("NPM1").map({"Positive": 1, "Negative": 0})
    out["flt3_itd"] = D.get("FLT3_ITD").map({"Positive": 1, "Negative": 0})
    resp = D.get("RESPONSE_TO_INDUCTION_TX").astype(str)
    out["induction_cr"] = np.where(resp.str.startswith("Complete"), 1.0, np.where(resp == "Refractory", 0.0, np.nan))
    out["fab"] = D.get("FAB_BLAST_MORPHOLOGY")
    out["specimen"] = D.get("SPECIMEN_TYPE")
    out.to_pickle(f)
    return out


def tcga_laml_clinical() -> pd.DataFrame:
    """Index = TCGA case id (e.g. 'TCGA-AB-2802'). OS + mutation calls from the PanCancer Atlas study (complete indel
    calls; the GDC-derived study in cBioPortal has only ~1/6 of NPM1 mutations), age/risk from the NEJM 2013 study."""
    f = paths.ensure(paths.DATA / "aml") / "tcga_laml_clinical.pkl"
    if f.exists():
        return pd.read_pickle(f)
    sid, pub = "laml_tcga_pan_can_atlas_2018", "laml_tcga_pub"
    P = pd.DataFrame(_cbio(f"/studies/{sid}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY", pageSize=100000)
                     ).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    Q = pd.DataFrame(_cbio(f"/studies/{pub}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY", pageSize=100000)
                     ).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    out = pd.DataFrame(index=P.index)
    out["os_days"] = pd.to_numeric(P.get("OS_MONTHS"), errors="coerce") * 30.4375
    out["os_event"] = P["OS_STATUS"].astype(str).str.startswith("1").astype(float).where(P["OS_STATUS"].notna())
    out["age"] = pd.to_numeric(Q.reindex(out.index).get("AGE"), errors="coerce")
    out["risk_cyto"] = Q.reindex(out.index).get("RISK_CYTO")
    r = requests.post(f"{CBIO}/molecular-profiles/{sid}_mutations/mutations/fetch", params={"projection": "SUMMARY"},
                      json={"sampleListId": f"{sid}_all"}, headers={"accept": "application/json", "Content-Type": "application/json"}, timeout=180)
    r.raise_for_status()
    m = pd.DataFrame(r.json())
    m = m[~m.mutationType.astype(str).str.contains("Silent|Intron|UTR|Flank", regex=True)]
    for col, ent in (("npm1", 4869), ("flt3_mut", 2322), ("tp53", 7157)):
        hit = set(m[m.entrezGeneId == ent].patientId)
        out[col] = [1.0 if i in hit else 0.0 for i in out.index]
    out.to_pickle(f)
    return out


def target_clinical() -> pd.DataFrame:
    """Paediatric AML (TARGET 2018 publication study, 899 patients, proper censoring). Index = TARGET case id."""
    f = paths.ensure(paths.DATA / "aml") / "target_clinical.pkl"
    if f.exists():
        return pd.read_pickle(f)
    sid = "aml_target_2018_pub"
    P = pd.DataFrame(_cbio(f"/studies/{sid}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY", pageSize=100000)
                     ).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    out = pd.DataFrame(index=P.index)
    out["os_days"] = pd.to_numeric(P.get("OS_DAYS"), errors="coerce")
    out["os_event"] = P["OS_STATUS"].astype(str).str.startswith("1").astype(float).where(P["OS_STATUS"].notna())
    out["age"] = pd.to_numeric(P.get("AGE"), errors="coerce")
    cr = P.get("CR_STATUS_AT_END_OF_COURSE_1").astype(str)
    out["induction_cr"] = np.where(cr == "CR", 1.0, np.where(cr.isin(["Not in CR", "Death"]), 0.0, np.nan))
    out.to_pickle(f)
    return out


def load_cohort(name):
    """(expression, sample table) with rows in the SAME order. The raw expression matrix is in download order, the
    sample table in query order; an earlier version assumed they matched and silently shuffled every patient label."""
    d = paths.DATA / "aml" / name
    X, S = pd.read_pickle(d / "expr.pkl"), pd.read_pickle(d / "samples.pkl")
    X = X.loc[S.index]
    assert X.index.equals(S.index), "expression rows and sample metadata are misaligned"
    return X, S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "clinical"])
    ap.add_argument("--beat", type=int, default=None, help="max BeatAML specimens (default: all primary)")
    ap.add_argument("--tcga", type=int, default=None)
    ap.add_argument("--target-aml", type=int, default=300)
    ap.add_argument("--target-normal", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=None, help="subset of cohorts, e.g. beataml tcga_laml")
    a = ap.parse_args(argv)
    if a.cmd == "clinical":
        for fn in (beataml_clinical, tcga_laml_clinical, target_clinical):
            d = fn(); print(fn.__name__, d.shape)
        return
    todo = {"beataml": a.beat, "tcga_laml": a.tcga, "target_aml": a.target_aml, "target_normal": a.target_normal}
    for k, n in todo.items():
        if a.only and k not in a.only:
            continue
        allow = set(target_clinical().dropna(subset=["os_days"]).index) if k == "target_aml" else None   # only patients with outcome data
        download_cohort(k, n, allow_cases=allow)


if __name__ == "__main__":
    main()
