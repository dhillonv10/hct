"""Distillation losses, written to match the prior art we are comparing against.

`eucosine` and `fitnet` are BIO-Distiller's feature losses (their kd_losses.py: fitnet is a plain MSE on the projected student
features; eucosine L2-normalises both sides, then adds the mean Euclidean distance and 1 - mean cosine similarity). `logit` is
their response loss, an MSE between teacher and student outputs. Their best configurations use eucosine with the summed
aggregation L = L_task + L_KD, which is the default here.

The projection is a single learned linear map from the student width to the teacher width, as in FitNets; it is discarded
after training, so it does not count toward the released student's parameters (stated, not hidden).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, d_student, d_teacher):
        super().__init__()
        self.p = nn.Identity() if d_student == d_teacher else nn.Linear(d_student, d_teacher)

    def forward(self, x):
        return self.p(x)


def feature_loss(s_proj, t, kind="eucosine"):
    if kind == "fitnet":
        return F.mse_loss(s_proj, t)
    if kind == "eucosine":
        a, b = F.normalize(s_proj, dim=-1), F.normalize(t, dim=-1)
        return torch.norm(a - b, p=2, dim=-1).mean() + (1.0 - F.cosine_similarity(a, b, dim=-1).mean())
    raise ValueError(kind)


def logit_loss(s, t):
    return F.mse_loss(s, t)


ARMS = {
    # name           feature loss   feature target        logit KD   what it isolates
    "scratch":       (None,         None,                 False),    # supervised only: the baseline B in the retention formula
    "kd_feat":       ("eucosine",   "head",               False),    # BIO-Distiller's best setting, on the teacher's penultimate layer
    "kd_feat_enc":   ("eucosine",   "enc",                False),    # same loss on the encoder's pooled output (task-agnostic target)
    "kd_fitnet":     ("fitnet",     "head",               False),    # their weaker feature loss, kept as the published contrast
    "kd_logit":      (None,         None,                 True),     # response KD alone
    "kd_both":       ("eucosine",   "head",               True),     # feature + response
    "ctrl_shuffled": ("eucosine",   "head_shuffled",      False),    # CONTROL: teacher targets permuted across examples
    "ctrl_untrained":("eucosine",   "head_untrained",     False),    # CONTROL: targets from a randomly initialised teacher
}
