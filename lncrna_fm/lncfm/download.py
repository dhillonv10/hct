"""Resumable, verified downloads of every public dataset the project uses.

    python -m lncfm.download gencode gtex encode          # benchmark data
    python -m lncfm.download encode --max-lines 4          # quick subset

Everything lands under $LNCFM_DATA (default lncrna_fm/data). Files are written to
`<name>.part` and renamed only after size/gzip verification, so an interrupted run
can simply be re-run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import requests

from . import paths

HEADERS_JSON = {"accept": "application/json"}


def _gz_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 24):
                pass
        return True
    except (OSError, EOFError):
        return False


def fetch(url: str, dest: Path, retries: int = 5, verify_gz: bool | None = None) -> Path:
    """Download url -> dest with resume + retries. Returns dest."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verify_gz is None:
        verify_gz = dest.suffix == ".gz"
    if dest.exists() and dest.stat().st_size > 0 and (not verify_gz or _gz_ok(dest)):
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            break
        except (requests.RequestException, OSError) as e:  # noqa: PERF203
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{retries}] {url}: {e!r}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {url}")
    if verify_gz and not _gz_ok(part):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"corrupt gzip after download: {url}")
    shutil.move(part, dest)
    return dest


# --------------------------------------------------------------------------- GENCODE
def gencode_files(species: str):
    if species == "human":
        rel, base, tag = paths.GENCODE_HUMAN_RELEASE, paths.GENCODE_HUMAN_URL, "v" + paths.GENCODE_HUMAN_RELEASE
        out = paths.human
    else:
        rel, base, tag = paths.GENCODE_MOUSE_RELEASE, paths.GENCODE_MOUSE_URL, "v" + paths.GENCODE_MOUSE_RELEASE
        out = paths.mouse
        tag = "vM" + rel[1:]
    names = [
        f"gencode.{tag}.lncRNA_transcripts.fa.gz",
        f"gencode.{tag}.pc_transcripts.fa.gz",
        f"gencode.{tag}.annotation.gtf.gz",
    ]
    return [(f"{base}/{n}", out(n)) for n in names]


def get_gencode():
    for sp in ("human", "mouse"):
        for url, dest in gencode_files(sp):
            print(f"gencode[{sp}] {dest.name}")
            fetch(url, dest)


def get_all_transcripts():
    """All GENCODE human transcripts (needed for small ncRNA disease genes: RMRP, RNU4ATAC, ...)."""
    n = f"gencode.v{paths.GENCODE_HUMAN_RELEASE}.transcripts.fa.gz"
    print("gencode", n)
    fetch(f"{paths.GENCODE_HUMAN_URL}/{n}", paths.human(n))


def get_clinvar():
    print("clinvar vcf")
    fetch("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz", paths.DATA / "clinvar" / "clinvar.vcf.gz")


# --------------------------------------------------------------------------- GTEx
def get_gtex():
    dest = paths.DATA / "gtex" / "gene_median_tpm.gct.gz"
    print("gtex", dest.name)
    fetch(paths.GTEX_MEDIAN_URL, dest)


# --------------------------------------------------------------------------- ENCODE
ENCODE = "https://www.encodeproject.org"


def encode_fraction_experiments(assay: str = "polyA plus RNA-seq"):
    """Human cell-line experiments with a nucleus or cytosol fraction."""
    out = {}
    for frac in ("nucleus", "cytosol"):
        r = requests.get(
            ENCODE + "/search/",
            params={
                "type": "Experiment", "status": "released", "limit": "all", "format": "json",
                "assay_title": assay,
                "replicates.library.biosample.subcellular_fraction_term_name": frac,
            },
            headers=HEADERS_JSON, timeout=60,
        )
        r.raise_for_status()
        for e in r.json()["@graph"]:
            cell = e["biosample_ontology"]["term_name"]
            if cell in ("BLaER1", "macrophage", "ML-DmD17-c3"):  # time-courses / non-human
                continue
            out.setdefault(cell, {})[frac] = e["@id"]
    return {c: v for c, v in out.items() if set(v) == {"nucleus", "cytosol"}}


def get_encode(max_lines: int | None = None, annotation: str = "V29"):
    """Download GRCh38 gene-quantification TSVs for nucleus/cytosol polyA+ RNA-seq."""
    ddir = paths.ensure(paths.DATA / "encode_fractions")
    exps = encode_fraction_experiments()
    cells = sorted(exps)[:max_lines] if max_lines else sorted(exps)
    manifest = []
    for cell in cells:
        for frac, eid in exps[cell].items():
            j = requests.get(ENCODE + eid + "?format=json", headers=HEADERS_JSON, timeout=60).json()
            files = [
                f for f in j["files"]
                if f.get("output_type") == "gene quantifications" and f.get("file_format") == "tsv"
                and f.get("assembly") == "GRCh38" and f.get("genome_annotation") == annotation
                and f.get("status") == "released"
            ]
            for f in files:
                dest = ddir / f"{cell.replace(' ', '_')}__{frac}__{f['accession']}.tsv"
                print(f"encode {cell:>12s} {frac:>7s} {f['accession']}")
                fetch(ENCODE + f["href"], dest, verify_gz=False)
                manifest.append({"cell": cell, "fraction": frac, "experiment": j["accession"],
                                 "file": dest.name, "replicates": f.get("biological_replicates")})
    (ddir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"encode: {len(manifest)} files, {len(cells)} cell lines")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["gencode", "gtex", "encode", "variants", "all"])
    ap.add_argument("--max-lines", type=int, default=None, help="encode: limit number of cell lines")
    a = ap.parse_args(argv)
    todo = set(a.what)
    if "all" in todo:
        todo = {"gencode", "gtex", "encode", "variants"}
    if "gencode" in todo:
        get_gencode()
    if "gtex" in todo:
        get_gtex()
    if "encode" in todo:
        get_encode(a.max_lines)
    if "variants" in todo:
        get_all_transcripts()
        get_clinvar()


if __name__ == "__main__":
    main()
