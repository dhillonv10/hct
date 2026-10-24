"""E-MTAB-7309 (older Swedish twins, repeated whole-blood 450K arrays): download the raw IDAT zips, compute raw betas on the compact
CpG universe in Python, and write the longitudinal table  lb/longitudinal.npz + lb/longitudinal_meta.csv .
Resumable: one npz per zip under longitudinal/parts."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from epimod.data.download import download
from epimod.data.idat import probe_table, zip_betas
from epimod.lb.lite import load_lite_universe
from epimod.paths import subdir

ACC = "E-MTAB-7309"
SDRF = f"https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/309/{ACC}/Files/{ACC}.sdrf.txt"
root = subdir("longitudinal"); parts = subdir("longitudinal", "parts"); tmp = subdir("longitudinal", "tmp")
sdrf = pd.read_csv(download(SDRF, root / "sdrf.txt"), sep="\t")
tab = probe_table(subdir("annot") / "GPL13534_full.txt", load_lite_universe())
links = sorted(set(sdrf["Comment [ArrayExpress FTP file]"].dropna()), key=lambda s: int(s.split(".raw.")[1].split(".")[0]))
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(links)


def work(link):
    name = Path(link).name; out = parts / (name + ".npz")
    if out.exists():
        return name, "cached"
    url = link.replace("ftp://ftp.ebi.ac.uk/", "https://ftp.ebi.ac.uk/")
    z = download(url, tmp / name)
    t0 = time.time(); b = zip_betas(z, tab)
    np.savez(out, names=np.array(list(b)), X=np.stack(list(b.values())))
    Path(z).unlink(missing_ok=True)
    return name, f"{len(b)} arrays in {time.time() - t0:.0f}s"


with ThreadPoolExecutor(3) as ex:
    for name, msg in ex.map(work, links[:limit]):
        print(name, msg, flush=True)

names, X = [], []
for f in sorted(parts.glob("*.npz")):
    z = np.load(f); names += z["names"].tolist(); X.append(z["X"])
X = np.vstack(X); sid = pd.Series(range(len(names)), index=names)
g = sdrf[sdrf["Array Data File"].astype(str).str.endswith("_Grn.idat")].copy()
g["sample_id"] = g["Array Data File"].str.replace("_Grn.idat", "", regex=False)
g["subject_id"] = g["Characteristics[individual]"].astype(str); g["family_id"] = g["Characteristics[twin pair]"].astype(str)
g["age"] = pd.to_numeric(g["Characteristics[age]"], errors="coerce"); g["sex"] = g["Characteristics[sex]"].str[0].str.upper()
g = g[g.sample_id.isin(sid.index)].dropna(subset=["age"]).drop_duplicates("sample_id")
g["time_years"] = g.age - g.groupby("subject_id").age.transform("min")
g[["sample_id", "subject_id", "family_id", "age", "time_years", "sex"]].to_csv(root.parent / "lb" / "longitudinal_meta.csv", index=False)
np.savez(root.parent / "lb" / "longitudinal.npz", X=X[sid.loc[g.sample_id].to_numpy()], sample_ids=g.sample_id.to_numpy().astype(str))
print("longitudinal:", len(g), "arrays |", g.subject_id.nunique(), "individuals |", (g.groupby("subject_id").size() >= 2).sum(), "with >= 2 visits")
