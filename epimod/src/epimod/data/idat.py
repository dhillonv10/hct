"""Minimal Illumina 450K IDAT reader and raw beta computation (no R / minfi needed).

IDAT v3 layout: 'IDAT', int64 version, int32 n_fields, then (uint16 code, int64 offset) per field;
field 1000 = number of probes, 102 = int32 bead-type addresses (sorted), 104 = uint16 mean intensities.
beta = M / (M + U + 100) with Infinium I probes read from their own colour channel (A = unmethylated bead,
B = methylated bead) and Infinium II probes from AddressA (green = methylated, red = unmethylated). No background or
dye-bias correction is applied: this is used for within-person changes, where a constant preprocessing cancels."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd


def read_idat(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    assert buf[:4] == b"IDAT", "not an IDAT file"
    assert int(np.frombuffer(buf[4:12], "<i8")[0]) == 3, "unsupported IDAT version"
    n_fields = int(np.frombuffer(buf[12:16], "<i4")[0])
    fields = {}
    for k in range(n_fields):
        o = 16 + 10 * k
        fields[int(np.frombuffer(buf[o:o + 2], "<u2")[0])] = int(np.frombuffer(buf[o + 2:o + 10], "<i8")[0])
    n = int(np.frombuffer(buf[fields[1000]:fields[1000] + 4], "<i4")[0])
    ids = np.frombuffer(buf[fields[102]:fields[102] + 4 * n], "<i4")
    mean = np.frombuffer(buf[fields[104]:fields[104] + 2 * n], "<u2")
    return ids, mean


@dataclass
class ProbeTable:
    cpg: pd.Index
    addr_a: np.ndarray
    addr_b: np.ndarray       # 0 for Infinium II
    is_ii: np.ndarray
    red: np.ndarray          # Infinium I colour channel (True = red)


def probe_table(gpl_path, cpgs: pd.Index) -> ProbeTable:
    """From the cached GPL13534 platform table (has AddressA_ID / AddressB_ID / design / colour)."""
    with open(gpl_path) as fh:
        for i, line in enumerate(fh):
            if line.startswith("ID\tName\tAddressA_ID"):
                skip = i
                break
    t = pd.read_csv(gpl_path, sep="\t", skiprows=skip, usecols=["ID", "AddressA_ID", "AddressB_ID", "Infinium_Design_Type", "Color_Channel"],
                    dtype={"ID": str}, low_memory=False).drop_duplicates("ID").set_index("ID").reindex(cpgs)
    assert t.AddressA_ID.notna().all(), "lite CpGs missing from the platform table"
    return ProbeTable(cpgs, t.AddressA_ID.to_numpy(np.int64), t.AddressB_ID.fillna(0).to_numpy(np.int64),
                      (t.Infinium_Design_Type == "II").to_numpy(), (t.Color_Channel == "Red").to_numpy())


def _look(ids, mean, addr):
    pos = np.clip(np.searchsorted(ids, addr), 0, len(ids) - 1)
    return np.where(ids[pos] == addr, mean[pos].astype(np.float32), np.nan)


def beta_from_idats(grn: bytes, red: bytes, tab: ProbeTable) -> np.ndarray:
    gi, gm = read_idat(grn); ri, rm = read_idat(red)
    gA, rA, gB, rB = _look(gi, gm, tab.addr_a), _look(ri, rm, tab.addr_a), _look(gi, gm, tab.addr_b), _look(ri, rm, tab.addr_b)
    M = np.where(tab.is_ii, gA, np.where(tab.red, rB, gB)); U = np.where(tab.is_ii, rA, np.where(tab.red, rA, gA))
    return np.clip(M / (M + U + 100.0), 0.0, 1.0).astype(np.float32)


def zip_betas(zpath, tab: ProbeTable) -> dict[str, np.ndarray]:
    """{array basename: beta over tab.cpg} for every Grn/Red pair in an ArrayExpress raw zip."""
    out = {}
    with zipfile.ZipFile(zpath) as z:
        names = {n.split("/")[-1]: n for n in z.namelist() if n.endswith(".idat")}
        for n in sorted(names):
            if n.endswith("_Grn.idat") and n[:-9] + "_Red.idat" in names:
                out[n[:-9]] = beta_from_idats(z.read(names[n]), z.read(names[n[:-9] + "_Red.idat"]), tab)
    return out
