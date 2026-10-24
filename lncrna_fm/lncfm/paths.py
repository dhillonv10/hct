"""Central place for filesystem locations and release pins."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LNCFM_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("LNCFM_RUNS", ROOT / "runs"))

# Release pins. Bump these deliberately; benchmark labels are tied to them.
GENCODE_HUMAN_RELEASE = os.environ.get("GENCODE_HUMAN_RELEASE", "50")
GENCODE_MOUSE_RELEASE = os.environ.get("GENCODE_MOUSE_RELEASE", "M39")

GENCODE_HUMAN_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_" + GENCODE_HUMAN_RELEASE
)
GENCODE_MOUSE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_" + GENCODE_MOUSE_RELEASE
)
GTEX_MEDIAN_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)


def human(name: str) -> Path:
    return DATA / "gencode_human" / name


def mouse(name: str) -> Path:
    return DATA / "gencode_mouse" / name


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
