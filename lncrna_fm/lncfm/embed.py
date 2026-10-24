"""Embedding providers: turn RNA sequences into fixed-size vectors for the benchmark.

  LncFMProvider(ckpt)    -- our model (or a random-init control: ckpt='random:<preset>')
  HFProvider(name)       -- external pretrained models used as the 'Option B' arm
"""
from __future__ import annotations

import numpy as np
import torch

from .model import LncFM, LncFMConfig, build, encode, pick_device, PAD


class LncFMProvider:
    def __init__(self, ckpt: str, device: str = "auto", max_len: int = 512, max_windows: int = 4, feat: str = "last+mid",
                 batch: int = 32):
        self.device = pick_device(device)
        if ckpt.startswith("random:"):             # untrained control with identical architecture
            torch.manual_seed(0)
            self.model = build(ckpt.split(":", 1)[1], max_len=max(max_len, 1024))
        else:
            ck = torch.load(ckpt, map_location="cpu")
            self.model = LncFM(LncFMConfig(**ck["cfg"]))
            self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.max_len, self.max_windows, self.feat, self.batch = max_len, max_windows, feat, batch

    @torch.no_grad()
    def _pool(self, ids: torch.Tensor):
        out = self.model(ids, return_hidden=True)
        w = out["tok_ok"].unsqueeze(-1).float()
        mid = out["hiddens"][len(out["hiddens"]) // 2]
        mid = (mid * w).sum(1) / w.sum(1).clamp(min=1)
        if self.feat == "last":
            return out["pooled"]
        if self.feat == "mid":
            return mid
        return torch.cat([out["pooled"], mid], dim=1)

    @torch.no_grad()
    def __call__(self, seqs):
        # windows of <= max_len nt (first `max_windows` windows), length-weighted mean of window embeddings
        items = []
        for i, s in enumerate(seqs):
            ids = encode(s)
            for w in range(min(self.max_windows, max(1, -(-len(ids) // self.max_len)))):
                items.append((i, ids[w * self.max_len:(w + 1) * self.max_len]))
        order = np.argsort([len(x[1]) for x in items])
        d = None
        acc, wsum = None, np.zeros(len(seqs))
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            L = max(len(items[j][1]) for j in idx)
            L = -(-L // self.model.cfg.stride) * self.model.cfg.stride
            X = np.zeros((len(idx), L), np.int64)
            for r, j in enumerate(idx):
                X[r, :len(items[j][1])] = items[j][1]
            z = self._pool(torch.from_numpy(X).to(self.device)).float().cpu().numpy()
            if acc is None:
                acc = np.zeros((len(seqs), z.shape[1]), np.float64)
            for r, j in enumerate(idx):
                i, ids = items[j]
                acc[i] += z[r] * len(ids)
                wsum[i] += len(ids)
        return (acc / wsum[:, None]).astype(np.float32)


# Models whose repos ship custom Python that executes on load. Off by default; you opt in on
# YOUR machine with  export LNCFM_TRUST_REMOTE_CODE=1  after reading the repo's modeling code.
REMOTE_CODE = {"nt-v2-50m", "hyenadna-small"}

HF_MODELS = {   # name -> (hub id, alphabet)
    "rinalmo-micro": ("multimolecule/rinalmo-micro", "rna"),
    "rinalmo-mega": ("multimolecule/rinalmo-mega", "rna"),
    "rnafm": ("multimolecule/rnafm", "rna"),
    "nt-v2-50m": ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", "dna"),   # needs transformers<~4.45
    "hyenadna-small": ("LongSafari/hyenadna-small-32k-seqlen-hf", "dna"),              # 4M, trained on hg38
}


class HFProvider:
    """Frozen external model; mean-pooled last hidden state over the first `max_nt` nucleotides."""

    def __init__(self, name: str, device: str = "auto", max_nt: int = 1000, batch: int = 8):
        import os
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        if name.startswith("local:"):                 # a directory saved by lncfm.cpt (no remote code)
            hub, self.alpha, trust = name.split(":", 1)[1], "rna", False
            import multimolecule  # noqa: F401
        else:
            hub, self.alpha = HF_MODELS[name]
            trust = name in REMOTE_CODE
        if trust and os.environ.get("LNCFM_TRUST_REMOTE_CODE") != "1":
            raise RuntimeError(
                f"{name} ({hub}) runs custom code from its model repo. Read it first, then re-run with "
                "LNCFM_TRUST_REMOTE_CODE=1 to opt in.")
        if hub.startswith("multimolecule/"):
            import multimolecule  # noqa: F401  (registers the architectures; normal PyPI package)
        self.tok = AutoTokenizer.from_pretrained(hub, trust_remote_code=trust)
        cfg = AutoConfig.from_pretrained(hub, trust_remote_code=trust)
        if trust and not hasattr(cfg, "rope_theta"):   # NT-v2 remote code vs newer transformers
            cfg.rope_theta = 10000.0
        self.model = AutoModel.from_pretrained(hub, config=cfg, trust_remote_code=trust).eval()
        self.device = pick_device(device)
        self.model.to(self.device)
        import inspect
        self._fwd_args = set(inspect.signature(self.model.forward).parameters)
        self.max_nt, self.batch = max_nt, batch

    @torch.no_grad()
    def __call__(self, seqs):
        out = []
        order = np.argsort([len(s) for s in seqs])
        res = [None] * len(seqs)
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            txt = [seqs[i][: self.max_nt] for i in idx]
            txt = [t.replace("U", "T") if self.alpha == "dna" else t.replace("T", "U") for t in txt]
            enc = self.tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            ids = enc["input_ids"].to(self.device)
            am = enc["attention_mask"] if "attention_mask" in enc else (enc["input_ids"] != self.tok.pad_token_id).long()
            am = am.to(self.device)
            kw = {"attention_mask": am} if "attention_mask" in self._fwd_args else {}
            h = self.model(input_ids=ids, **kw).last_hidden_state
            m = am.unsqueeze(-1).float()
            z = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
            for r, i in enumerate(idx):
                res[i] = z[r]
        return np.stack(res)
