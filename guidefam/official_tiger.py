"""The authors' released TIGER network (Hugging Face space Knowles-Lab/tiger), run as a plain PyTorch forward pass.

Weights are read from the SavedModel's checkpoint index (no graph is executed) and stored as .npz by scripts/export_official_tiger.py.
Licence: the release is under NYGC's academic / non-commercial licence, so the weights are NOT redistributed here -- export them yourself.

Architecture (from tiger.py + the checkpoint shapes): input = 26-nt target window (3 nt of 5' flank + the 23-mer; unknown = all-zero) and the guide written as the
complement of the 23-mer, left-padded with 3 empty positions; the two one-hot matrices are stacked as channels of a (4 x 26) image ->
Conv2D(32, 4x4, same) ReLU -> Conv2D(32, 4x4, same) ReLU -> MaxPool(1x2) -> flatten (height, width, channel order, as Keras) ->
Dense128 sigmoid -> Dense32 sigmoid -> Dense1 (linear LFC estimate; lower = more active).  Sequence only: no scalar features.
The released tool then multiplies by a per-mismatch-count calibration slope and applies a monotone sigmoid; neither changes a ranking, so we use the raw estimate.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import features as F

CTX5, GUIDE_LEN = 3, 23


class OfficialTiger(nn.Module):
    def __init__(self, w: dict):
        super().__init__()
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        self.c1w, self.c1b = nn.Parameter(t(w["conv1_kernel"]).permute(3, 2, 0, 1).contiguous()), nn.Parameter(t(w["conv1_bias"]))
        self.c2w, self.c2b = nn.Parameter(t(w["conv2_kernel"]).permute(3, 2, 0, 1).contiguous()), nn.Parameter(t(w["conv2_bias"]))
        self.d1w, self.d1b = nn.Parameter(t(w["dense1_kernel"]).T.contiguous()), nn.Parameter(t(w["dense1_bias"]))
        self.d2w, self.d2b = nn.Parameter(t(w["dense2_kernel"]).T.contiguous()), nn.Parameter(t(w["dense2_bias"]))
        self.d3w, self.d3b = nn.Parameter(t(w["dense3_kernel"]).T.contiguous()), nn.Parameter(t(w["dense3_bias"]))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, target, guide):                       # (B, 26, 4) each
        x = torch.stack([target.transpose(1, 2), guide.transpose(1, 2)], dim=1)          # (B, 2, 4, 26)
        x = torch.relu(nn.functional.conv2d(x, self.c1w, self.c1b, padding="same"))
        x = torch.relu(nn.functional.conv2d(x, self.c2w, self.c2b, padding="same"))
        x = nn.functional.max_pool2d(x, (1, 2))                                           # (B, 32, 4, 13)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)                                 # Keras flatten order: height, width, channel
        x = torch.sigmoid(x @ self.d1w.T + self.d1b)
        x = torch.sigmoid(x @ self.d2w.T + self.d2b)
        return (x @ self.d3w.T + self.d3b).squeeze(-1)


def load(npz_path: str) -> OfficialTiger:
    z = np.load(npz_path)
    return OfficialTiger({k: z[k] for k in z.files})


def encode(d) -> tuple:
    """Official model inputs from a frame with `5p_context`, `target_seq` (23 nt, sense) -- missing flank is padded with all-zero positions."""
    tgt, gde = [], []
    for c5, t in zip(d["5p_context"].astype(str), d["target_seq"].astype(str)):
        left = c5[-CTX5:].rjust(CTX5, "N")
        tgt.append(left + t)
        comp = t.upper().translate(str.maketrans("ACGT", "TGCA"))
        gde.append("N" * CTX5 + comp)
    return torch.from_numpy(F.onehot(tgt, CTX5 + GUIDE_LEN)), torch.from_numpy(F.onehot(gde, CTX5 + GUIDE_LEN))


def predict(model: OfficialTiger, d, batch: int = 2048) -> np.ndarray:
    """Activity score, higher = more active (the negative of the network's LFC estimate)."""
    t, g = encode(d)
    out = []
    with torch.no_grad():
        for i in range(0, len(t), batch):
            out.append(model(t[i:i + batch], g[i:i + batch]).numpy())
    return -np.concatenate(out)
