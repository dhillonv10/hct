"""Zero-shot variant scoring from a masked language model, and a compact student distilled from one.

The score every row of the clinical table gets is the masked log-odds
        s = log p(ref | context, position masked) - log p(alt | context, position masked)
oriented so that a HIGHER score means MORE damaging. It needs no labels and no fine-tuning, so the 650M teacher,
its 33M and 150M siblings and a 5M student are all scored by exactly the same procedure on exactly the same rows.

The student is a small transformer trained on human non-coding transcripts that the panel genes are removed from,
in three arms that differ only in the target at the masked positions:
    mlm     the true base                      (train your own small LM: the honest baseline)
    kd      the teacher's 22-way distribution  (distillation)
    mlm_kd  both
Because all three see the same sequences and the same masked positions, "does distilling a 650M RNA language model
buy anything a 5M language model cannot learn by itself" is a controlled comparison rather than a vibe.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .teacher import ALPHABET_SIZE, CLS, EOS, IDX, MASK, PAD, RiNALMo

NT = [IDX[c] for c in "ACGT"]
STUDENT_PRESETS = {                                                             # (d_model, layers, heads)
    "s1":  (128, 4, 4),
    "s5":  (256, 6, 8),
    "s20": (448, 8, 8),
}


MAX_WIDTH = 1024


def encode_window(seq: str, offset: int, width: int):
    """A window of at most `width` nucleotides around `offset`, and the offset's index inside it.

    Long transcripts are windowed so every model sees the same context budget; short ones are untouched."""
    width = min(width, MAX_WIDTH)
    seq = seq.upper().replace("U", "T")
    if len(seq) <= width:
        return seq, offset
    half = width // 2
    lo = max(0, min(offset - half, len(seq) - width))
    return seq[lo:lo + width], offset - lo


def tokenise(seqs, pad_to=None):
    pad_to = pad_to or max(len(s) for s in seqs) + 2
    out = np.full((len(seqs), pad_to), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        ids = [CLS] + [IDX.get(c, IDX["<unk>"]) for c in s] + [EOS]
        out[i, :len(ids)] = ids[:pad_to]
    return out


@torch.no_grad()
def masked_log_odds(model, rows, device, width=512, bs=16, amp=True, progress=True):
    """Score a table with columns seq, offset, ref, alt. One masked forward pass per variant."""
    model.eval().to(device)
    wins = [encode_window(s, int(o), width) for s, o in zip(rows.seq, rows.offset)]
    order = np.argsort([len(w[0]) for w in wins])                               # batch similar lengths together
    out = np.zeros(len(rows), dtype=np.float32)
    refs = [IDX[r] for r in rows.ref]
    alts = [IDX[a] for a in rows.alt]
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        tok = torch.from_numpy(tokenise([wins[k][0] for k in j])).to(device)
        at = torch.tensor([wins[k][1] + 1 for k in j], device=device)           # +1 for <cls>
        tok[torch.arange(len(j), device=device), at] = MASK
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            lg = model.logits(tok) if hasattr(model, "logits") else model(tok)
        lp = F.log_softmax(lg.float()[torch.arange(len(j), device=device), at], dim=-1).cpu().numpy()
        for n, k in enumerate(j):
            out[k] = lp[n, refs[k]] - lp[n, alts[k]]
        if progress and i % (bs * 25) == 0:
            print(f"    {i + len(j)}/{len(order)}", flush=True)
    return out


@torch.no_grad()
def recovery_accuracy(model, seqs, device, n=256, width=512, bs=16, seed=0):
    """Mask one random nucleotide per sequence and ask whether the model puts the true base top among A/C/G/T.

    This is the gate that catches a wrong mask-head wiring before any result is believed: chance is 0.25, and a
    working 650M RNA language model should be far above it."""
    model.eval().to(device)
    rs = np.random.default_rng(seed)
    pick = rs.integers(0, len(seqs), n)                                         # with replacement: a 13-gene panel
    hit = tot = 0                                                               # still gives a stable estimate
    for i in range(0, len(pick), bs):
        chunk = [seqs[k].upper().replace("U", "T")[:width] for k in pick[i:i + bs]]
        pos = [int(rs.integers(0, len(c))) for c in chunk]
        tok = torch.from_numpy(tokenise(chunk)).to(device)
        at = torch.tensor([p + 1 for p in pos], device=device)
        truth = torch.tensor([IDX.get(c[p], IDX["<unk>"]) for c, p in zip(chunk, pos)], device=device)
        tok[torch.arange(len(chunk), device=device), at] = MASK
        lg = model.logits(tok) if hasattr(model, "logits") else model(tok)
        sel = lg.float()[torch.arange(len(chunk), device=device), at][:, NT]
        pred = torch.tensor(NT, device=device)[sel.argmax(-1)]
        hit += int((pred == truth).sum())
        tot += len(chunk)
    return hit / max(tot, 1)


# ------------------------------------------------------------------------------------------------------------ student
class Student(nn.Module):
    """A small pre-LN transformer masked LM over single-nucleotide tokens, with learned positional embeddings."""

    def __init__(self, preset="s5", max_len=1026):          # max_len must cover the longest window ever scored
        super().__init__()
        d, n, h = STUDENT_PRESETS[preset]
        self.preset = preset
        self.embed = nn.Embedding(ALPHABET_SIZE, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, ALPHABET_SIZE)

    def logits(self, tokens):
        p = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.pos(p)[None]
        x = self.enc(x, src_key_padding_mask=tokens.eq(PAD))
        return self.head(self.norm(x))

    def forward(self, tokens):
        return self.logits(tokens)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def kd_loss(student_logits, teacher_logprobs, true_ids, arm, temperature=1.0):
    """Targets at the masked positions. `mlm` uses the real base, `kd` the teacher's distribution, `mlm_kd` both.

    `teacher_logprobs` may be None for `mlm`, which is what lets an arm mask fresh positions every step instead of
    being tied to the positions the teacher was cached at."""
    if arm == "mlm":
        return F.cross_entropy(student_logits, true_ids)
    if teacher_logprobs is None:
        raise ValueError(f"arm {arm!r} needs teacher targets")
    lp = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(lp, teacher_logprobs.exp(), reduction="batchmean") * temperature ** 2
    if arm == "kd":
        return kl
    if arm == "mlm_kd":
        return F.cross_entropy(student_logits, true_ids) + kl
    raise ValueError(arm)
