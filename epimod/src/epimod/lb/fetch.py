"""Download the LongevityBench DNAm prompts from Hugging Face and parse them into ``lb/parsed.pkl`` (the five DNAm tasks).

``benchmark`` config: LB-0002, LB-0006, LB-0010.   ``extra`` config: LB-0190 (fixed 353-CpG panel), LB-0195 (clock replica)."""
from __future__ import annotations

import json
import pickle
import urllib.request
from pathlib import Path

import pandas as pd

from ..data.download import download
from ..paths import subdir
from .parse import HF_BASE, parse_clock, parse_pairwise, parse_raw, parse_single

API = "https://huggingface.co/api/datasets/insilicomedicine/longebench/tree/main/{cfg}"


def _config_frame(cfg: str, cache: Path) -> pd.DataFrame:
    with urllib.request.urlopen(urllib.request.Request(API.format(cfg=cfg), headers={"User-Agent": "epimod/0.1"}), timeout=60) as r:
        files = sorted(x["path"] for x in json.load(r) if x["path"].endswith(".parquet"))
    return pd.concat([pd.read_parquet(download(f"{HF_BASE}/{p}", cache / f"{cfg}_{Path(p).name}")) for p in files], ignore_index=True)


def build_parsed(out: Path | None = None) -> Path:
    out = out or subdir("lb") / "parsed.pkl"
    cache = subdir("lb", "hf")
    bench, extra = _config_frame("benchmark", cache), _config_frame("extra", cache)
    tasks = {"LB-0010": parse_single(bench, "LB-0010"), "LB-0002": parse_single(bench, "LB-0002"), "LB-0006": parse_pairwise(bench, "LB-0006"),
             "LB-0190": parse_raw(extra, "LB-0190"), "LB-0195": parse_clock(extra, "LB-0195")}
    pickle.dump(tasks, open(out, "wb"))
    return out
