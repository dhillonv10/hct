"""Pack everything needed to run OFFLINE into one archive (and verify it on the other side).

    python scripts/make_bundle.py pack   epimod_bundle.tar.gz          # on the machine that built the data
    python scripts/make_bundle.py verify epimod_bundle/                # after extracting it

Contents: data/ (compact stores, parsed benchmark prompts, sample metadata, probe annotation, GEO tables and holdout list, twin
cohort), code.tar (git snapshot of the repository), MANIFEST.json (sha256 of every file, commit, counts).
Not included: raw downloads, temporary files, the legacy 200k-CpG store and mmap caches (all rebuildable or unnecessary)."""
import hashlib, io, json, subprocess, sys, tarfile, time
from pathlib import Path

from epimod.paths import data_root

KEEP = ["lb/store_lite", "lb/parsed.pkl", "lb/lite_universe.parquet", "lb/longitudinal.npz", "lb/longitudinal_meta.csv",
        "geo/meta", "geo/meta_unlabeled", "geo/series.parquet", "geo/screen", "geo/holdout.txt", "meta", "annot"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files(root: Path):
    for k in KEEP:
        p = root / k
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.is_file() and not x.name.endswith((".tmp", ".part", ".tmp.npz")))


def pack(out: str):
    root = data_root(); repo = Path(__file__).resolve().parents[2]
    code = subprocess.run(["git", "archive", "--format=tar", "HEAD", "epimod"], cwd=repo, capture_output=True, check=True).stdout
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    man = {"commit": commit, "created": time.strftime("%F %T"), "files": {}}
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for p in files(root):
            rel = "data/" + str(p.relative_to(root)); man["files"][rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
            tar.add(p, arcname="epimod_bundle/" + rel)
        ti = tarfile.TarInfo("epimod_bundle/code.tar"); ti.size = len(code); tar.addfile(ti, io.BytesIO(code))
        man["code_sha256"] = hashlib.sha256(code).hexdigest()
        mb = json.dumps(man, indent=1).encode(); ti = tarfile.TarInfo("epimod_bundle/MANIFEST.json"); ti.size = len(mb); tar.addfile(ti, io.BytesIO(mb))
    print(f"{out}: {len(man['files'])} data files, {Path(out).stat().st_size / 2**20:.0f} MB, commit {commit[:10]}")


def verify(d: str):
    d = Path(d); man = json.loads((d / "MANIFEST.json").read_text()); bad = [r for r, m in man["files"].items() if not (d / r).exists() or sha(d / r) != m["sha256"]]
    code_ok = hashlib.sha256((d / "code.tar").read_bytes()).hexdigest() == man["code_sha256"]
    print(f"{len(man['files'])} files checked: {'ALL OK' if not bad and code_ok else f'{len(bad)} BAD, code ok={code_ok}'}; commit {man['commit'][:10]}")
    sys.exit(0 if not bad and code_ok else 1)


if __name__ == "__main__":
    {"pack": pack, "verify": verify}[sys.argv[1]](sys.argv[2])
