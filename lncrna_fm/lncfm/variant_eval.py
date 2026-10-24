"""Score the within-gene ncRNA variant set with one or more LncFM checkpoints (zero-shot MLM LLR).

    python -m lncfm.variant_eval runs/small_v1/ckpt_final.pt random:small
"""
import argparse

import numpy as np
import torch

from . import variants
from .model import LncFM, LncFMConfig, build, pick_device


def load(spec: str):
    if spec.startswith("random:"):
        torch.manual_seed(0)
        return build(spec.split(":", 1)[1], max_len=1024)
    ck = torch.load(spec, map_location="cpu")
    m = LncFM(LncFMConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args(argv)
    df = variants.build_variant_table().reset_index(drop=True)
    dev = pick_device("cpu")
    print(f"{len(df)} SNVs in {df.gene.nunique()} genes; {int(df.label.sum())} pathogenic / {int((1-df.label).sum())} benign")
    rng = np.random.default_rng(0)
    for spec in a.models:
        m = load(spec).to(dev)
        s = variants.mlm_llr(m, dev, df, window=512)
        auc, per = variants.within_gene_auroc(df, s)
        boots = []
        for _ in range(a.boot):   # bootstrap variants within gene
            idx = np.concatenate([rng.choice(np.where(df.gene.values == g)[0], (df.gene == g).sum()) for g in df.gene.unique()])
            b = df.iloc[idx].reset_index(drop=True)
            v, _ = variants.within_gene_auroc(b, s[idx])
            if v == v:
                boots.append(v)
        print(f"{spec:<44s} within-gene AUROC {auc:.3f} [{np.percentile(boots,2.5):.3f},{np.percentile(boots,97.5):.3f}]  "
              f"(genes used: {len(per)})")
        print(per.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
