"""Embedding providers: turn RNA sequences into fixed-size vectors for the benchmark.

  LncFMProvider(ckpt)    -- our model (or a random-init control: ckpt='random:<preset>')
  HFProvider(name)       -- external pretrained models used as the 'Option B' arm
"""
from __future__ import annotations

import os

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


def hf_dims(name: str):
    """(n_hidden_states, hidden_size) of a model, from its config only (cheap)."""
    from transformers import AutoConfig
    if name.startswith("local:"):
        import multimolecule  # noqa: F401
        cfg = AutoConfig.from_pretrained(name.split(":", 1)[1])
    else:
        hub = HF_MODELS[name][0]
        if hub.startswith("multimolecule/"):
            import multimolecule  # noqa: F401
        cfg = AutoConfig.from_pretrained(hub, trust_remote_code=name in REMOTE_CODE and os.environ.get("LNCFM_TRUST_REMOTE_CODE") == "1")
    return cfg.num_hidden_layers + 1, cfg.hidden_size          # + embedding output


class HFProvider:
    """Frozen external model.

    default            mean-pooled last hidden state over the first `max_nt` nucleotides  -> (n, d)
    all_layers=True    per-sequence array (n_hidden_states * 2 * d): for EVERY layer, masked mean- and max-pool;
                       slice it later (see bench.slice_layers) so one forward pass serves every layer/pool variant
    windows=k          long transcripts: embed up to k consecutive windows of `max_nt` nt and aggregate
                       (mean of means weighted by length, element-wise max of maxes) instead of truncating
    """

    def __init__(self, name: str, device: str = "auto", max_nt: int = 1000, batch: int = 8,
                 all_layers: bool = False, windows: int = 1):
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
        self.max_nt, self.batch, self.all_layers, self.windows = max_nt, batch, all_layers, windows

    def _chunks(self, seqs):
        items = []
        for i, s in enumerate(seqs):
            n = max(1, min(self.windows, -(-len(s) // self.max_nt)))
            for w in range(n):
                items.append((i, s[w * self.max_nt:(w + 1) * self.max_nt]))
        return items

    @torch.no_grad()
    def __call__(self, seqs):
        items = self._chunks(seqs)
        order = np.argsort([len(c) for _, c in items])
        feats = [None] * len(items)
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            txt = [items[j][1] for j in idx]
            txt = [t.replace("U", "T") if self.alpha == "dna" else t.replace("T", "U") for t in txt]
            enc = self.tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_nt + 8)
            ids = enc["input_ids"].to(self.device)
            am = enc["attention_mask"] if "attention_mask" in enc else (enc["input_ids"] != self.tok.pad_token_id).long()
            am = am.to(self.device)
            kw = {"attention_mask": am} if "attention_mask" in self._fwd_args else {}
            if self.all_layers:
                hs = self.model(input_ids=ids, output_hidden_states=True, **kw).hidden_states
                H = torch.stack(hs, 1)                                      # (B, L+1, T, d)
            else:
                H = self.model(input_ids=ids, **kw).last_hidden_state.unsqueeze(1)   # (B, 1, T, d)
            m = am[:, None, :, None].float()
            mean = (H * m).sum(2) / m.sum(2).clamp(min=1)                    # (B, L+1, d)
            mx = (H - 1e4 * (1 - m)).max(2).values
            n_tok = am.sum(1).float().cpu().numpy()
            out = torch.stack([mean, mx], 2) if self.all_layers else mean[:, 0]
            out = out.float().cpu().numpy()
            for r, j in enumerate(idx):
                feats[j] = (out[r], n_tok[r])
        # aggregate windows per sequence
        res, k = [], 0
        for i in range(len(seqs)):
            parts = [feats[j] for j, (si, _) in enumerate(items) if si == i] if self.windows > 1 else [feats[i]]
            w = np.array([p[1] for p in parts]); w = w / w.sum()
            if self.all_layers:
                mean = sum(wi * p[0][:, 0] for wi, p in zip(w, parts))
                mx = np.max([p[0][:, 1] for p in parts], axis=0)
                res.append(np.stack([mean, mx], 1).reshape(-1).astype(np.float16))
            else:
                res.append(sum(wi * p[0] for wi, p in zip(w, parts)).astype(np.float32))
        return np.stack(res)
