"""Frozen RNA language-model embeddings (RNA-FM via the `multimolecule` port; weights loaded from safetensors, never from a pickle).

`embed(seqs, ...)` returns the mean over nucleotide positions of the final hidden layer (special tokens excluded), cached on disk by content hash.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def _load(path: str):
    import torch
    from multimolecule import RnaFmModel, RnaTokenizer
    tok = RnaTokenizer.from_pretrained(path)
    model = RnaFmModel.from_pretrained(path, use_safetensors=True).eval()
    return torch, tok, model


def embed(seqs, path: str, cache_dir: str | None = None, batch: int = 32, max_len: int = 400, layers=(-1,)) -> np.ndarray:
    seqs = [s.upper().replace("T", "U") for s in seqs]
    key = hashlib.sha1(("|".join(seqs) + str(layers) + str(max_len)).encode()).hexdigest()[:16]
    f = os.path.join(cache_dir, f"rnafm_{key}.npy") if cache_dir else None
    if f and os.path.exists(f):
        return np.load(f)
    torch, tok, model = _load(path)
    order = np.argsort([len(s) for s in seqs])
    out = [None] * len(seqs)
    with torch.no_grad():
        for i in range(0, len(order), batch):
            ix = order[i:i + batch]
            chunk = [seqs[j][:max_len] for j in ix]
            enc = tok(chunk, return_tensors="pt", padding=True)
            hs = model(**enc, output_hidden_states=True).hidden_states
            h = torch.cat([hs[l] for l in layers], dim=-1)
            lens = enc["attention_mask"].sum(1)
            for r, j in enumerate(ix):
                n = int(lens[r])
                out[j] = h[r, 1:n - 1].mean(0).numpy()               # drop <cls> and <eos>
    E = np.stack(out).astype(np.float32)
    if f:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(f, E)
    return E
