"""RiNALMo teacher in plain PyTorch: encoder + the MRL head, re-implemented from the repository's Apache-2.0 source.

Why not the official package: `rinalmo/model/attention.py` imports `flash_attn` unguarded and `modules.py` imports it transitively, so
`import rinalmo.pretrained` fails without flash-attn, and the flash path is CUDA + fp16/bf16 only. The encoder below is the same module tree
copied from `compare/oligoai_score.py` (validated against a real 650M RiNALMo fine-tune: strict key match, Pearson 0.633 on held-in rows).

Parameter names are kept identical to the released checkpoints so `load_state_dict(strict=True)` is the correctness test. The head names are
a best guess read from `rinalmo/model/downstream.py`; `dump_keys()` prints what a checkpoint actually holds so a mismatch is a two-minute fix,
not a failed run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIAL = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
STANDARD = ["A", "C", "G", "T", "I", "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N", "-"]
IDX = {t: i for i, t in enumerate(SPECIAL + STANDARD)}
CLS, PAD, EOS, UNK, MASK = (IDX[t] for t in SPECIAL)
ALPHABET_SIZE = len(IDX)                                                        # 22
LM_CONFIGS = {"giga": dict(embed=1280, blocks=33, heads=20), "mega": dict(embed=640, blocks=30, heads=20),
              "micro": dict(embed=480, blocks=12, heads=20), "selftest": dict(embed=64, blocks=2, heads=4)}


def encode(seq: str, pad_to: int):
    """`<cls> seq <eos>` then pad. U->T, unknown characters -> <unk>, exactly as rinalmo/data/alphabet.py."""
    seq = seq.upper().replace("U", "T")
    e = [CLS] + [IDX.get(c, UNK) for c in seq] + [EOS]
    return e + [PAD] * (pad_to - len(e)) if len(e) < pad_to else e


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

    def forward(self, x, keep):
        B, S, _ = x.shape
        q, k, v = self.Wqkv(x).view(B, S, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        t = torch.arange(S, device=x.device, dtype=torch.float32)
        fr = torch.outer(t, self.inv_freq.to(x.device))
        cos, sin = torch.cat([fr, fr], -1).cos()[None, None].to(q.dtype), torch.cat([fr, fr], -1).sin()[None, None].to(q.dtype)

        def rot(z):
            z1, z2 = z.chunk(2, -1)
            return z * cos + torch.cat([-z2, z1], -1) * sin
        q, k = rot(q), rot(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
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
        # As the released TransformerBlock: `x = self.attn_layer_norm(x)` OVERWRITES x, so the attention residual is taken from the
        # NORMALISED input. Getting this wrong cost Pearson 0.63 -> 0.14 on the OligoAI check.
        x = self.attn_layer_norm(x)
        x = x + self.mh_attn(x, keep)
        return x + self.transition(self.out_layer_norm(x))


class Transformer(nn.Module):
    def __init__(self, embed, blocks, heads):
        super().__init__()
        self.blocks = nn.ModuleList([Block(embed, heads) for _ in range(blocks)])
        self.final_layer_norm = nn.LayerNorm(embed)

    def forward(self, x, keep, want=()):
        hid = {}
        for i, b in enumerate(self.blocks):
            x = b(x, keep)
            if i in want:
                hid[i] = x
        return self.final_layer_norm(x), hid


class MaskHead(nn.Module):
    """rinalmo/model/modules.py MaskedLanguageModelHead, verbatim order: linear1 -> gelu -> layer_norm -> linear2."""

    def __init__(self, embed, alphabet):
        super().__init__()
        self.linear1, self.layer_norm, self.linear2 = nn.Linear(embed, embed), nn.LayerNorm(embed), nn.Linear(embed, alphabet)

    def forward(self, x):
        return self.linear2(self.layer_norm(F.gelu(self.linear1(x))))


class RiNALMo(nn.Module):
    def __init__(self, embed, blocks, heads):
        super().__init__()
        self.embedding = nn.Embedding(ALPHABET_SIZE, embed, padding_idx=PAD)
        self.transformer = Transformer(embed, blocks, heads)
        self.lm_mask_head = MaskHead(embed, ALPHABET_SIZE)
        self.mask_ratio_train = 0.15 * 0.8

    def forward(self, tokens, want=()):
        pad = tokens.eq(PAD)
        x = self.embedding(tokens)
        src_lens = (~pad).sum(-1)
        x = torch.where((tokens == MASK).unsqueeze(-1), torch.zeros_like(x), x)
        obs = (tokens == MASK).sum(-1) / src_lens                               # TokenDropout: active at inference in the released model
        x = x * (1 - self.mask_ratio_train) / (1 - obs[..., None, None])
        return self.transformer(x, ~pad, want)

    def logits(self, tokens):
        """Per-position scores over the 22-token alphabet, the quantity a zero-shot variant score is built from."""
        rep, _ = self.forward(tokens)
        return self.lm_mask_head(rep)


# ------------------------------------------------------------------------------------------------------------------------ MRL head
class ResNet1DBlock(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(c, c, k, padding="same", bias=False), nn.InstanceNorm1d(c), nn.ELU(inplace=True),
            nn.Conv1d(c, c, k, padding="same", bias=False), nn.InstanceNorm1d(c), nn.ELU(inplace=True))

    def forward(self, x):
        return x + self.conv_net(x)                                             # no norm/activation after the add, as in the source


class ResNet1D(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.blocks = nn.ModuleList([ResNet1DBlock(c) for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class MRLHead(nn.Module):
    """Linear(1280->32) -> 6 ResNet1D blocks -> masked mean -> Dropout(0.2) -> Linear(32->1)."""

    def __init__(self, c_in=1280, embed=32, blocks=6, dropout=0.2):
        super().__init__()
        self.linear_in = nn.Linear(c_in, embed)
        self.resnet = ResNet1D(embed, blocks)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(embed, 1)

    def forward(self, x, pad_mask=None, return_feat=False):
        x = self.linear_in(x)
        x = self.resnet(x.transpose(1, 2)).transpose(1, 2)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            feat = x.sum(-2) / (~pad_mask).sum(-1)[:, None]
        else:
            feat = x.mean(-2)
        out = self.linear_out(self.dropout(feat)).squeeze(-1)
        return (out, feat) if return_feat else out


class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1))
        self.register_buffer("_std", torch.ones(1))

    def inverse(self, y):
        return y * self._std + self._mean


class TeacherMRL(nn.Module):
    """The fine-tuned wrapper: attribute names `lm`, `pred_head`, `scaler` give the checkpoint's key prefixes."""

    def __init__(self, cfg="giga"):
        super().__init__()
        c = LM_CONFIGS[cfg]
        self.lm = RiNALMo(**c)
        self.pred_head = MRLHead(c_in=c["embed"])
        self.scaler = Scaler()

    def forward(self, tokens, return_feat=False):
        rep, _ = self.lm(tokens)
        pad = tokens.eq(PAD)
        rep = rep.masked_fill(pad.unsqueeze(-1), 0.0)
        out = self.pred_head(rep, pad, return_feat=return_feat)
        if return_feat:
            y, feat = out
            pooled = rep.sum(-2) / (~pad).sum(-1)[:, None]                      # encoder-level pooled vector, for feature KD
            return self.scaler.inverse(y).clamp(min=0.0), feat, pooled
        return self.scaler.inverse(out).clamp(min=0.0)


# ------------------------------------------------------------------------------------------------------------------------ loading
def read_ckpt(path):
    """Restricted load. Lightning files wrap the weights in 'state_dict'; the released .pt files are bare."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return obj


def dump_keys(path, n=40):
    """Print the key set of a checkpoint grouped by prefix. Run this FIRST on every new file: it costs seconds and turns a
    silent architecture mismatch into a known fact before any GPU time is spent."""
    sd = read_ckpt(path)
    groups = {}
    for k, v in sd.items():
        groups.setdefault(k.split(".")[0], []).append((k, tuple(v.shape) if hasattr(v, "shape") else None))
    print(f"{path}: {len(sd)} tensors, top-level prefixes {sorted(groups)}")
    for g, items in sorted(groups.items()):
        print(f"  [{g}] {len(items)} tensors")
        for k, s in items[:n]:
            print(f"      {k:60s} {s}")
        if len(items) > n:
            print(f"      ... {len(items) - n} more")
    return sd


def load_strict(model, path, prefix=""):
    """Load and fail loudly. `prefix` strips a leading namespace (e.g. the pretrained file is bare while the wrapper expects `lm.`)."""
    sd = read_ckpt(path)
    if prefix:
        sd = {prefix + k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if "inv_freq" not in k]
    if miss or res.unexpected_keys:
        raise SystemExit(
            f"Checkpoint does not match the re-implementation.\n  missing ({len(miss)}): {miss[:12]}\n"
            f"  unexpected ({len(res.unexpected_keys)}): {res.unexpected_keys[:12]}\n"
            f"Run `dump_keys('{path}')` and fix the module names before spending GPU time.")
    return model
