#!/usr/bin/env python3
"""oligoai_score.py -- score ASO rows with the released OligoAI checkpoint WITHOUT running the OligoAI repository's code.

Verified against the real checkpoint (OligoAI_11_09_25.ckpt): parameter names match strictly, and on 96 rows from OligoAI's own training patents the predictions
correlate with the labels (Pearson 0.63, MAE 17.9; the authors' checkpoint name suggests a validation MAE of about 19). An earlier version that added the attention
residual to the un-normalised input (the released block adds it to the NORMALISED input) gave Pearson 0.14 and was stopped by the --check gate.

Why a re-implementation: (1) the repository's attention module imports `flash_attn` (CUDA-only, pinned to an old build) at import time, and
(2) its inference path uses Lightning's `load_from_checkpoint`, which un-pickles the file without restrictions. Here the forward pass is re-written in
plain PyTorch from the repository's source (Apache-2.0: rinalmo/model/*.py, train_aso.py, rinalmo/data/*), parameter names are kept identical to
the checkpoint's, and the checkpoint is read with torch.load(weights_only=True) with ONE class allowed (the scaler). Any naming mismatch fails loudly.

Faithfully reproduced quirks of the released pipeline (they matter for matching its predictions):
  * the chemistry vocabulary has the key 'cET' but the data's sugar token is 'CET', so every cEt position is encoded as padding (index 0)
    (measured on 96 training rows that all carry cEt wings: Pearson 0.545 with this encoding and 0.545 with cEt as a real token -> no measurable effect);
  * TokenDropout is applied at inference too (embeddings are rescaled by 1 - 0.15*0.8 even when nothing is masked);
  * unknown nucleotides in the target context ('<unk>') are replaced by '<mask>'; missing dosage is imputed with the table's median.

Usage:
  python oligoai_score.py --csv aso.csv.gz --ckpt OligoAI_11_09_25.ckpt --ids ids.csv --out oligoai_pred.csv [--check]
  ids.csv has one column 'raw_id' = row position in the ORIGINAL table (before any de-duplication). Output: raw_id, oligoai_pred (% inhibition).
  --check scores 200 rows from OligoAI's own training patents first and aborts if the predictions do not track the labels (guards against a silent
  mismatch). --selftest builds a tiny random checkpoint and runs the whole path (no download needed).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------- tokenizer / vocabularies
SPECIAL = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
STANDARD = ["A", "C", "G", "T", "I", "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N", "-"]
IDX = {t: i for i, t in enumerate(SPECIAL + STANDARD)}
CLS, PAD, EOS, UNK, MASK = (IDX[t] for t in SPECIAL)
ALPHABET_SIZE = len(IDX)                                                       # 22
CHEM = {"<pad>": 0, "DNA": 1, "MOE": 2, "cET": 3}                               # case-sensitive, exactly as in the released code
BACKBONE = {"<pad>": 0, "PO": 1, "PS": 2}
TRANSFECTION = {"Electroporation": 0, "Gymnosis": 1, "Other": 2, "Lipofection": 3}
LM_CONFIGS = {"giga": dict(embed=1280, blocks=33, heads=20), "mega": dict(embed=640, blocks=30, heads=20),
              "micro": dict(embed=480, blocks=12, heads=20), "nano": dict(embed=320, blocks=6, heads=20),
              "selftest": dict(embed=96, blocks=2, heads=4)}


def encode(seq: str, pad_to: int):
    seq = seq.upper().replace("U", "T")
    e = [CLS] + [IDX.get(c, UNK) for c in seq] + [EOS]
    return e + [PAD] * (pad_to - len(e)) if len(e) < pad_to else e


def mod_tokens(tokens, vocab, max_len):
    t = [0] + [vocab.get(x, 0) for x in tokens]                                  # position 0 aligns with <cls>
    t = t + [0] * (max_len - len(t))
    return t[:max_len]


def list_tokens(s):
    return re.findall(r"'([^']+)'", s) if isinstance(s, str) else []              # lists are stored as Python-repr strings; regex, never eval


# ---------------------------------------------------------------------------------------------------------------- model (names match the checkpoint)
class SwiGLU(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=True)
        self.linear_gate = nn.Linear(size_in, size_out, bias=True)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h = self.linear(x)
        return h * torch.sigmoid(self.beta * h) * self.linear_gate(x)


class Attn(nn.Module):
    """Same parameters as FlashMultiHeadSelfAttention: fused Wqkv laid out as (three, heads, head_dim), rotary on q and k, no biases."""

    def __init__(self, embed, heads):
        super().__init__()
        self.h, self.d = heads, embed // heads
        self.Wqkv = nn.Linear(embed, 3 * embed, bias=False)
        self.out_proj = nn.Linear(embed, embed, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.d, 2).float() / self.d))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x, keep):                                                  # keep: (B, S) True = real token
        B, S, _ = x.shape
        q, k, v = self.Wqkv(x).view(B, S, 3, self.h, self.d).permute(2, 0, 3, 1, 4)       # each (B, h, S, d)
        t = torch.arange(S, device=x.device, dtype=torch.float32)
        fr = torch.outer(t, self.inv_freq.to(x.device))
        cos, sin = torch.cat([fr, fr], -1).cos()[None, None].to(q.dtype), torch.cat([fr, fr], -1).sin()[None, None].to(q.dtype)

        def rot(z):
            z1, z2 = z.chunk(2, -1)
            return z * cos + torch.cat([-z2, z1], -1) * sin
        q, k = rot(q), rot(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])      # boolean mask: True = attend
        return self.out_proj(o.transpose(1, 2).reshape(B, S, self.h * self.d))


class Block(nn.Module):
    def __init__(self, embed, heads):
        super().__init__()
        self.mh_attn = Attn(embed, heads)
        self.attn_layer_norm = nn.LayerNorm(embed)
        hid = int(2 / 3 * 4 * embed)
        self.transition = nn.Sequential(SwiGLU(embed, hid), nn.Dropout(0.0), nn.Linear(hid, embed, bias=True))
        self.out_layer_norm = nn.LayerNorm(embed)

    def forward(self, x, keep):
        # Exactly as the released TransformerBlock: `x = self.attn_layer_norm(x)` OVERWRITES x, so the attention residual is taken from the
        # NORMALISED input (not the raw one); the second residual is taken before out_layer_norm.
        x = self.attn_layer_norm(x)
        x = x + self.mh_attn(x, keep)
        return x + self.transition(self.out_layer_norm(x))


class Transformer(nn.Module):
    def __init__(self, embed, blocks, heads):
        super().__init__()
        self.blocks = nn.ModuleList([Block(embed, heads) for _ in range(blocks)])
        self.final_layer_norm = nn.LayerNorm(embed)

    def forward(self, x, keep):
        for b in self.blocks:
            x = b(x, keep)
        return self.final_layer_norm(x)


class MaskHead(nn.Module):                                                      # unused at inference; kept so the checkpoint loads strictly
    def __init__(self, embed, alphabet):
        super().__init__()
        self.linear1, self.layer_norm, self.linear2 = nn.Linear(embed, embed), nn.LayerNorm(embed), nn.Linear(embed, alphabet)


class RiNALMo(nn.Module):
    def __init__(self, embed, blocks, heads):
        super().__init__()
        self.embedding = nn.Embedding(ALPHABET_SIZE, embed, padding_idx=PAD)
        self.transformer = Transformer(embed, blocks, heads)
        self.lm_mask_head = MaskHead(embed, ALPHABET_SIZE)
        self.mask_ratio_train = 0.15 * 0.8

    def forward(self, tokens):
        pad = tokens.eq(PAD)
        x = self.embedding(tokens)
        src_lens = (~pad).sum(-1)                                                # TokenDropout, active at inference in the released model
        x = torch.where((tokens == MASK).unsqueeze(-1), torch.zeros_like(x), x)
        obs = (tokens == MASK).sum(-1) / src_lens
        x = x * (1 - self.mask_ratio_train) / (1 - obs[..., None, None])
        return self.transformer(x, ~pad)


class Pool(nn.Module):
    def __init__(self, embed, proj):
        super().__init__()
        self.projection = nn.Linear(embed, proj) if proj < embed else nn.Identity()

    def forward(self, r, pad):
        p = self.projection(r).masked_fill(pad.unsqueeze(-1), 0.0)
        return p.sum(1) / (~pad).sum(1)[:, None]


class Head(nn.Module):
    def __init__(self, embed, tm_dim, hidden, layers, proj):
        super().__init__()
        self.aso_pooler, self.context_pooler = Pool(embed, proj), Pool(embed, proj)
        out = proj if proj < embed else embed
        mlp_in = 2 * out + tm_dim
        mlp = []
        for i in range(layers):
            mlp.append(nn.Linear(mlp_in if i == 0 else hidden, 1 if i == layers - 1 else hidden))
            if i < layers - 1:
                mlp += [nn.ReLU(), nn.Dropout(0.0)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, a, c, ap, cp, md):
        return self.mlp(torch.cat([self.aso_pooler(a, ap), self.context_pooler(c, cp), md], -1)).squeeze(-1)


class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_mean", torch.tensor([0.0]))
        self.register_buffer("_std", torch.tensor([1.0]))


class OligoAI(nn.Module):
    def __init__(self, hp):
        super().__init__()
        c = LM_CONFIGS[hp["lm_config"]]
        E = c["embed"]
        self.lm = RiNALMo(E, c["blocks"], c["heads"])
        self.chem_embedder = nn.Embedding(hp["chem_vocab_size"], hp["chem_embed_dim"], padding_idx=0)
        self.backbone_embedder = nn.Embedding(hp["backbone_vocab_size"], hp["backbone_embed_dim"], padding_idx=0)
        self.transfection_method_embedder = nn.Embedding(hp["transfection_method_vocab_size"], hp["transfection_method_embed_dim"])
        self.aso_feature_combiner = nn.Sequential(nn.Linear(E + hp["chem_embed_dim"] + hp["backbone_embed_dim"], 128), nn.ReLU(), nn.Dropout(0.0),
                                                  nn.Linear(128, E), nn.ReLU())
        self.pred_head = Head(E, hp["transfection_method_embed_dim"], hp["hidden_dim"], hp["num_layers"], hp.get("pooling_projection_dim", 64))
        self.scaler = Scaler()
        self.pad_idx = PAD

    def forward(self, aso, chem, bb, ctx, dosage, tm):
        a = self.lm(aso)
        c = self.lm(ctx)
        comb = self.aso_feature_combiner(torch.cat([a, self.chem_embedder(chem), self.backbone_embedder(bb)], -1))
        ap, cp = aso.eq(PAD), ctx.eq(PAD)
        comb = comb * (~ap).unsqueeze(-1).to(comb.dtype)
        c = c * (~cp).unsqueeze(-1).to(c.dtype)
        md = self.transfection_method_embedder(tm) * torch.log1p(dosage).unsqueeze(-1)
        return self.pred_head(comb, c, ap, cp, md)

    def predict(self, *batch):
        return self.forward(*batch) * self.scaler._std + self.scaler._mean


# ---------------------------------------------------------------------------------------------------------------- safe checkpoint loading
def install_scaler_stub():
    """Register a stand-in for rinalmo.utils.scaler.StandardScaler so the checkpoint's pickled scaler can be rebuilt WITHOUT importing the repository."""
    for name in ("rinalmo", "rinalmo.utils", "rinalmo.utils.scaler"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class StandardScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_mean", torch.tensor([0.0]))
            self.register_buffer("_std", torch.tensor([1.0]))
            self._seen_samples = []
            self._need_update = False
    StandardScaler.__module__ = "rinalmo.utils.scaler"
    StandardScaler.__qualname__ = "StandardScaler"
    sys.modules["rinalmo.utils.scaler"].StandardScaler = StandardScaler
    return StandardScaler


def load_checkpoint(path):
    Stub = install_scaler_stub()
    try:
        torch.serialization.add_safe_globals([Stub, set])
    except AttributeError:
        sys.exit("This script needs torch >= 2.4 (torch.serialization.add_safe_globals) to read the checkpoint safely. Upgrade torch.")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        sys.exit(f"Could not read the checkpoint with weights_only=True: {e}\nNot falling back to an unrestricted load. Send me this message.")
    hp, sd = dict(ck["hyper_parameters"]), ck["state_dict"]
    model = OligoAI(hp)
    res = model.load_state_dict(sd, strict=False)
    ignorable = ("r2_metric", "mae_metric", "rmse_metric", "loss.", "inv_freq")
    bad_unexp = [k for k in res.unexpected_keys if not k.startswith(ignorable) and not k.endswith("inv_freq")]
    bad_miss = [k for k in res.missing_keys if not k.endswith("inv_freq")]
    if bad_unexp or bad_miss:
        sys.exit(f"Checkpoint / model parameter mismatch.\n  missing ({len(bad_miss)}): {bad_miss[:8]}\n  unexpected ({len(bad_unexp)}): {bad_unexp[:8]}\n"
                 "Nothing was scored. Send me this message.")
    n = sum(p.numel() for p in model.parameters())
    return model.eval(), hp, n


# ---------------------------------------------------------------------------------------------------------------- data
def oligoai_split(raw: pd.DataFrame, seed=42, tr=0.8, va=0.1):
    """OligoAI's released train/val/test assignment by PATENT (rinalmo/data/downstream/aso/dataset.py): np.random.seed(42), permutation of unique patents."""
    pid = raw.custom_id.map(lambda c: c.split("/")[-1].split("_table_")[0])
    np.random.seed(seed)
    perm = np.random.permutation(pid.unique())
    n = len(perm)
    a, b = int(tr * n), int(va * n)
    part = {p: "train" for p in perm[:a]} | {p: "val" for p in perm[a:a + b]} | {p: "test" for p in perm[a + b:]}
    return pid.map(part).values, pid.values


def make_batches(raw: pd.DataFrame, ids, full: pd.DataFrame, bs):
    max_aso = full.aso_sequence_5_to_3.str.len().max() + 2
    max_ctx = full.rna_context.fillna("").str.len().max() + 2
    med_dose = full.dosage.median()
    sub = raw.iloc[ids]
    for i in range(0, len(sub), bs):
        c = sub.iloc[i:i + bs]
        aso = [encode(s.replace("T", "U"), max_aso) for s in c.aso_sequence_5_to_3]
        chem = [mod_tokens(list_tokens(s), CHEM, max_aso) for s in c.sugar_mods]
        bb = [mod_tokens(list_tokens(s), BACKBONE, max_aso) for s in c.backbone_mods]
        ctx = np.array([encode(s, max_ctx) for s in c.rna_context.fillna("")])
        ctx[ctx == UNK] = MASK
        dose = c.dosage.fillna(med_dose).values.astype(np.float32)
        tm = [TRANSFECTION.get(x, 0) for x in c.transfection_method]
        yield (torch.tensor(aso), torch.tensor(chem), torch.tensor(bb), torch.tensor(ctx), torch.tensor(dose), torch.tensor(tm))


@torch.inference_mode()
def score(model, raw, ids, full, bs, device, amp):
    out = []
    t0 = time.time()
    for k, b in enumerate(make_batches(raw, ids, full, bs)):
        b = [x.to(device) for x in b]
        if amp and device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.predict(*b)
        else:
            p = model.predict(*b)
        out.append(p.float().cpu().numpy())
        if (k + 1) % 20 == 0:
            done = (k + 1) * bs
            print(f"  {done}/{len(ids)} rows, {done / (time.time() - t0):.1f} rows/s", flush=True)
    return np.concatenate(out)


def selftest():
    """Random tiny checkpoint -> full safe-load path -> scoring. Checks plumbing and parameter naming, not accuracy."""
    import tempfile
    Stub = install_scaler_stub()
    hp = dict(lm_config="selftest", chem_vocab_size=4, chem_embed_dim=16, backbone_vocab_size=3, backbone_embed_dim=8, transfection_method_vocab_size=4,
              transfection_method_embed_dim=8, hidden_dim=32, num_layers=3, dropout=0.3, lr=5e-5, scaler=Stub(), pooling_projection_dim=64)
    m = OligoAI({k: v for k, v in hp.items() if k != "scaler"})
    sd = {k: v for k, v in m.state_dict().items()}
    sd["scaler._mean"], sd["scaler._std"] = torch.tensor([45.0]), torch.tensor([25.0])
    with tempfile.TemporaryDirectory() as d:
        p = d + "/t.ckpt"
        torch.save({"state_dict": sd, "hyper_parameters": hp, "pytorch-lightning_version": "2.1.3"}, p)
        model, hp2, n = load_checkpoint(p)
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(6)]
    raw = pd.DataFrame({"aso_sequence_5_to_3": seqs, "rna_context": ["".join(rng.choice(list("ACGU"), 120)) for _ in seqs],
                        "sugar_mods": ["['CET', 'CET', 'CET', 'CET', 'CET'] + ['DNA']*10"] * 6, "backbone_mods": ["['PS', 'PO', 'PS', '<PAD>']"] * 6,
                        "dosage": [4000.0, np.nan, 1000.0, 500.0, 2000.0, 100.0], "transfection_method": ["Gymnosis"] * 6, "inhibition_percent": rng.uniform(0, 100, 6)})
    p = score(model, raw, np.arange(6), raw, 4, "cpu", False)
    assert p.shape == (6,) and np.isfinite(p).all(), p
    # padding invariance: scoring a row alone vs inside a batch with different padding must agree
    q = score(model, raw, np.array([2]), raw, 1, "cpu", False)
    assert abs(q[0] - p[2]) < 1e-3, (q, p[2])
    print(f"selftest OK: {n} parameters, predictions {np.round(p, 2).tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv")
    ap.add_argument("--ckpt")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", action="store_true", help="CUDA only: bfloat16 autocast (faster, slightly different numerics)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    raw = pd.read_csv(a.csv, low_memory=False)
    raw = raw.dropna(subset=["inhibition_percent"]).reset_index(drop=True)
    model, hp, n = load_checkpoint(a.ckpt)
    model.to(device)
    print(f"loaded {n / 1e6:.0f}M parameters ({hp['lm_config']}), scaler mean {float(model.scaler._mean):.2f} std {float(model.scaler._std):.2f}, device {device}")
    if a.check:
        split, _ = oligoai_split(raw)
        tr = np.flatnonzero(split == "train")
        pick = np.random.default_rng(0).choice(tr, 200, replace=False)
        p = score(model, raw, pick, raw, a.batch, device, a.bf16)
        r = np.corrcoef(p, raw.inhibition_percent.values[pick])[0, 1]
        print(f"integrity check on 200 rows from OligoAI's TRAINING patents: Pearson {r:.3f}, MAE {np.abs(p - raw.inhibition_percent.values[pick]).mean():.1f}")
        if r < 0.3:
            sys.exit("Predictions do not track the labels on rows the model was trained on: the re-implementation or the checkpoint mapping is wrong. Stopping.")
    ids = pd.read_csv(a.ids).raw_id.values.astype(int)
    pred = score(model, raw, ids, raw, a.batch, device, a.bf16)
    pd.DataFrame({"raw_id": ids, "oligoai_pred": pred}).to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(ids)} rows)")


if __name__ == "__main__":
    main()
