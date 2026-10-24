"""Public GEO DNA-methylation arrays: discovery, header screening, streaming ingestion into the compact store.

Why: the LongevityBench cohorts are small GEO association studies on 27K/450K/EPIC arrays, not the curated
healthy-control collection in ComputAge. More, and more varied, cohorts is the most direct way to narrow that gap.

Pipeline (all steps resumable):
  discover  -> geo/series.parquet            candidate series from NCBI E-utilities
  screen    -> geo/screen/<GSE>.json         age / sex availability from the first MBs of each series matrix
  ingest    -> lb/store_lite/geo/<GSE>.npz   uint8 beta*100 restricted to the compact CpG universe (255 = missing)
               geo/meta/<GSE>.parquet        per-sample metadata (written last: its presence marks completion)
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import subdir
from .download import download

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/"
PLATFORMS = {"GPL13534": "450K", "GPL21145": "EPIC", "GPL23976": "EPIC", "GPL33022": "EPICv2", "GPL8490": "27K"}
MISSING = 255

_lock = threading.Lock()
_last_call = [0.0]


def _get(url: str, headers: dict | None = None, timeout: int = 90, retries: int = 4, eutils: bool = False) -> bytes:
    err: Exception | None = None
    for attempt in range(retries):
        if eutils:                       # NCBI allows ~3 requests/s without an API key
            with _lock:
                time.sleep(max(0.0, 0.40 - (time.time() - _last_call[0]))); _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "epimod/0.1", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (404, 403):
                break
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"GET {url}: {err}")


# ----------------------------------------------------------------------------------------- discovery
def discover(out: Path | None = None) -> pd.DataFrame:
    out = out or subdir("geo") / "series.parquet"
    rows = []
    for gpl in PLATFORMS:
        r = json.loads(_get(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(
            {"db": "gds", "term": f"{gpl}[ACCN] AND gse[ETYP]", "retmax": 10000, "retmode": "json"}), eutils=True))
        ids = r["esearchresult"]["idlist"]
        print(f"{gpl}: {len(ids)} series", flush=True)
        for i in range(0, len(ids), 150):
            s = json.loads(_get(EUTILS + "esummary.fcgi?" + urllib.parse.urlencode(
                {"db": "gds", "id": ",".join(ids[i:i + 150]), "retmode": "json"}), eutils=True))["result"]
            for uid in s.get("uids", []):
                d = s[uid]
                if not str(d.get("accession", "")).startswith("GSE"):
                    continue
                rows.append({"gse": d["accession"], "title": d.get("title", ""), "summary": d.get("summary", "")[:400],
                             "gpl": ";".join("GPL" + g for g in str(d.get("gpl", "")).split(";") if g), "taxon": d.get("taxon", ""),
                             "n_samples": int(d.get("n_samples") or 0), "gdstype": d.get("gdstype", "")})
    df = pd.DataFrame(rows).drop_duplicates("gse").reset_index(drop=True)
    df.to_parquet(out)
    return df


def series_dir(gse: str) -> str:
    return f"{FTP}{gse[:-3]}nnn/{gse}/matrix/"


def matrix_files(gse: str) -> list[tuple[str, int]]:
    """(file name, bytes) of the series-matrix files of a series."""
    html = _get(series_dir(gse)).decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'href="([^"]+_series_matrix\.txt\.gz)">[^<]*</a>\s+\S+\s+\S+\s+(\d+(?:\.\d+)?)([KMG]?)', html):
        out.append((m.group(1), int(float(m.group(2)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(3)])))
    return out


def pick_file(files: list[tuple[str, int]]) -> tuple[str, int, str] | None:
    """Prefer the file of a methylation platform we handle (multi-platform series have one file per platform)."""
    for name, size in files:
        for gpl in PLATFORMS:
            if gpl in name:
                return name, size, gpl
    if len(files) == 1:
        return files[0][0], files[0][1], ""
    return None


# ----------------------------------------------------------------------------------------- header parsing
def read_header(gse: str, fname: str, nbytes: int = 4_000_000) -> dict:
    for n in (nbytes, 4 * nbytes):
        raw = _get(series_dir(gse) + fname, headers={"Range": f"bytes=0-{n - 1}"})
        text = zlib.decompressobj(31).decompress(raw).decode("utf-8", "replace")      # truncated gzip streams decode up to the cut
        if "!series_matrix_table_begin" in text or len(raw) < n:
            break
    return parse_header(text)


def parse_header(text: str) -> dict:
    h: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        if line.startswith("!series_matrix_table_begin"):
            break
        if line.startswith("!"):
            k, _, rest = line[1:].partition("\t")
            h.setdefault(k, []).append([v.strip().strip('"') for v in rest.split("\t")])
    return h


AGE_KEY = re.compile(r"^\s*(donor[ _]?|subject[ _]?|patient[ _]?|chronological[ _]?|chronologic[ _]?)?age(?![a-z])", re.I)
BAD_KEY = re.compile(r"gestation|onset|diagnos|menarche|menopaus|mother|matern|father|patern|sibling|pregnan|week|parent|difference|"
                     r"accel|epigenetic|dnam|clock|predict|estimat|bone|methyl|sample age|storage|grade|stage|bmi|height", re.I)
SEX_KEY = re.compile(r"^\s*(gender|sex)\s*$", re.I)
TUMOR = re.compile(r"tumou?r|carcinoma|neoplas|malignan|metasta|glioma|lymphoma|sarcoma|melanoma|blastoma|cell line|ipsc|organoid", re.I)


def parse_age(v: str) -> float:
    v = v.strip().lower()
    if not v or re.search(r"\d\s*-\s*\d|n/?a|unknown|not ", v):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", v)
    if not m:
        return np.nan
    x = float(m.group())
    if "month" in v:
        x /= 12
    elif "day" in v:
        x /= 365.25
    elif "week" in v:
        x /= 52.18
    return x if 0 <= x <= 105 else np.nan


def sample_table(h: dict) -> pd.DataFrame:
    """One row per sample: gsm, age, sex, tissue, tumor flag."""
    gsm = h["Sample_geo_accession"][0]
    n = len(gsm)
    title = h.get("Sample_title", [[""] * n])[0]
    source = h.get("Sample_source_name_ch1", [[""] * n])[0]
    chars = h.get("Sample_characteristics_ch1", [])
    age, sex, tissue_txt = [np.nan] * n, [""] * n, [""] * n
    for j in range(n):
        tissue_bits = [title[j] if j < len(title) else "", source[j] if j < len(source) else ""]
        for row in chars:
            if j >= len(row) or ":" not in row[j]:
                continue
            k, _, v = row[j].partition(":")
            if np.isnan(age[j]) and AGE_KEY.search(k) and not BAD_KEY.search(k):
                age[j] = parse_age(v)
            elif SEX_KEY.search(k):
                s = v.strip().lower()
                sex[j] = "M" if s.startswith("m") else "F" if s.startswith("f") else ""
            elif re.search(r"tissue|cell type|sample type|source", k, re.I):
                tissue_bits.append(v)
        tissue_txt[j] = " | ".join(b for b in tissue_bits if b)
    df = pd.DataFrame({"gsm": gsm, "age": age, "sex": sex, "tissue": tissue_txt})
    df["tumor"] = df.tissue.str.contains(TUMOR)
    return df


def related_series(h: dict) -> set[str]:
    return set(re.findall(r"GSE\d+", " ".join(" ".join(r) for r in h.get("Series_relation", []))))


def screen_one(gse: str, out_dir: Path) -> dict:
    f = out_dir / f"{gse}.json"
    if f.exists():
        return json.loads(f.read_text())
    rec: dict = {"gse": gse, "ok": False}
    try:
        pick = pick_file(matrix_files(gse))
        if pick is None:
            rec["why"] = "no usable matrix file"
        else:
            name, size, gpl = pick
            h = read_header(gse, name)
            if "Sample_geo_accession" not in h:
                rec["why"] = "no sample header"
            else:
                t = sample_table(h)
                ok = t.age.notna() & ~t.tumor
                rec.update({"file": name, "bytes": size, "gpl": gpl, "n": len(t), "n_age": int(ok.sum()), "n_tumor": int(t.tumor.sum()),
                            "age_min": float(t.age.min()) if t.age.notna().any() else None, "age_max": float(t.age.max()) if t.age.notna().any() else None,
                            "related": sorted(related_series(h)), "ok": bool(ok.sum() >= 8 and ok.mean() >= 0.7)})
                if not rec["ok"]:
                    rec["why"] = "age missing for most samples"
    except Exception as e:                                   # network / parsing problems are recorded, not fatal
        rec["why"] = f"error: {str(e)[:120]}"
    f.write_text(json.dumps(rec))
    return rec


# ----------------------------------------------------------------------------------------- ingestion
def ingest_one(gse: str, fname: str, gpl: str, lite: pd.Index, skip_gsm: set[str], root: Path, store_dir: Path,
               require_age: bool = True, meta_name: str = "meta") -> dict:
    """``require_age=False`` keeps samples without an age label (Age = NaN): for self-supervised pretraining only."""
    meta_f = root / meta_name / f"{gse}.parquet"
    if meta_f.exists():
        return {"gse": gse, "status": "done-before"}
    t0 = time.time()
    path = download(series_dir(gse) + fname, root / "tmp" / fname, quiet=True)
    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            head, n_head = [], 0
            for line in fh:
                n_head += 1
                if line.startswith("!series_matrix_table_begin"):
                    break
                head.append(line)
        h = parse_header("".join(head))
        tab = sample_table(h)
        gpl = gpl or next((p for p in h.get("Sample_platform_id", [[""]])[0] if p in PLATFORMS), "")
        lite_set = set(lite)
        parts = []
        for chunk in pd.read_csv(path, sep="\t", skiprows=n_head, index_col=0, chunksize=20000, comment="!", low_memory=False):
            sub = chunk[chunk.index.isin(lite_set)]
            if len(sub):
                parts.append(sub.astype(np.float32))
        if not parts:
            return {"gse": gse, "status": "no-probes"}
        X = pd.concat(parts); X = X[~X.index.duplicated()].reindex(lite)
        X.columns = [str(c).strip('"') for c in X.columns]
        v = X.to_numpy(np.float32)
        q = np.nanpercentile(v, [1, 99]) if np.isfinite(v).any() else (np.nan, np.nan)
        if q[0] < -0.5 or q[1] > 1.5:                          # M-values -> beta
            if q[0] < -0.5 and q[1] < 25:
                v = 1.0 / (1.0 + np.exp(-v * np.log(2)))
            elif q[0] >= 0 and q[1] <= 100:                    # percent
                v = v / 100.0
            else:
                return {"gse": gse, "status": f"unrecognised-scale {q}"}
        v = np.clip(v, 0.0, 1.0)
        tab = tab.set_index("gsm")
        pos = {c: i for i, c in enumerate(X.columns)}
        keep = [g for g in X.columns if g in tab.index and (np.isfinite(tab.at[g, "age"]) or not require_age) and not tab.at[g, "tumor"]
                and g not in skip_gsm and np.isfinite(v[:, pos[g]]).mean() > 0.02]
        if len(keep) < 8:
            return {"gse": gse, "status": f"too-few-samples {len(keep)}"}
        cols = [pos[g] for g in keep]
        prof = np.where(np.isfinite(v[:, cols]), np.rint(v[:, cols] * 100), MISSING).astype(np.uint8).T      # [S, L]
        store_dir.mkdir(parents=True, exist_ok=True)
        tmp = store_dir / f"{gse}.tmp.npz"
        np.savez(tmp, samples=np.array(keep), prof=prof)
        tmp.replace(store_dir / f"{gse}.npz")
        label = PLATFORMS.get(gpl, "")
        meta = pd.DataFrame({"DatasetID": gse, "PlatformID": gpl, "Tissue": tab.loc[keep, "tissue"].str.slice(0, 80).to_numpy(),
                             "CellType": "", "Gender": tab.loc[keep, "sex"].replace("", "U").to_numpy(), "Age": tab.loc[keep, "age"].to_numpy(float),
                             "Condition": "NA", "Class": "", "Platform": label, "split": "geo"}, index=pd.Index(keep, name="SampleID"))
        (root / meta_name).mkdir(parents=True, exist_ok=True)
        meta.to_parquet(meta_f)
        return {"gse": gse, "status": "ok", "n": len(keep), "coverage": float((prof != MISSING).mean()), "sec": round(time.time() - t0)}
    finally:
        Path(path).unlink(missing_ok=True)


HEME = re.compile(r"leuk[a]?emi|myeloid|myelodys|myeloma|myeloprolif|lymphoblast|myeloblast|blast crisis|hematologic|haematologic|CAR[- ]T", re.I)
HEME_CASE = re.compile(r"\b(AML|MDS|CML|CLL|MPN|ALL|T-ALL|B-ALL)\b")


def heme_holdout(write: bool = True) -> list[str]:
    """Series about haematological malignancy (AML, MDS, leukaemia ...), held out of every training corpus so that
    disease / age-acceleration analyses on such cohorts stay clean.  Matched on series title and summary."""
    s = pd.read_parquet(subdir("geo") / "series.parquet")
    text = (s.title.fillna("") + " " + s.summary.fillna(""))
    hit = s[text.str.contains(HEME) | text.str.contains(HEME_CASE)].gse.tolist()
    if write:
        (subdir("geo") / "holdout.txt").write_text("\n".join(sorted(hit)) + "\n")
    return sorted(hit)


def geo_unlabeled_meta() -> pd.DataFrame:
    """Samples ingested without an age label (self-supervised pretraining only), minus held-out series."""
    files = sorted((subdir("geo") / "meta_unlabeled").glob("GSE*.parquet"))
    m = pd.concat([pd.read_parquet(f) for f in files]) if files else pd.DataFrame()
    hold = subdir("geo") / "holdout.txt"
    if len(m) and hold.exists():
        m = m[~m.DatasetID.isin(set(hold.read_text().split()))]
    return m


def geo_meta() -> pd.DataFrame:
    """Metadata of every ingested sample, minus the held-out series in geo/holdout.txt (empty frame if nothing ingested)."""
    files = sorted((subdir("geo") / "meta").glob("GSE*.parquet"))
    m = pd.concat([pd.read_parquet(f) for f in files]) if files else pd.DataFrame()
    hold = subdir("geo") / "holdout.txt"
    if len(m) and hold.exists():
        m = m[~m.DatasetID.isin(set(hold.read_text().split()))]
    return m
