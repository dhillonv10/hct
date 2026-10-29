"""Tiger2D (Wessels/Stirn et al. 2023) re-implemented in PyTorch from the architecture in the authors' repo -- not their code or weights.

Spec taken from daklab/tiger (models.py, layers.py, callbacks.py): input = one-hot of 5' flank | 23-nt target | 3' flank laid out as a
(4 x L) image, two Conv2D(32, 4x4, same, ReLU), MaxPool(1x2), flatten, dropout .25, concat scalar descriptors, Dense128 sigmoid, dropout .1,
Dense32 sigmoid, dropout .1, Dense1 linear; Adam 1e-3, log-cosh loss, batch 2048, up to 2000 epochs, early stopping patience 100.

The one deliberate difference: early stopping watches an inner validation split carved out of the TRAINING genes. The authors' cross-validation
script passes the held-out fold as the validation set and restores the best epoch on it.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from . import features as F


class Tiger2D(nn.Module):
    def __init__(self, length: int, n_scalar: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), padding="same"), nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), padding="same"), nn.ReLU(),
            nn.MaxPool2d((1, 2), ceil_mode=True), nn.Flatten(), nn.Dropout(0.25))
        flat = 32 * 4 * math.ceil(length / 2)
        self.head = nn.Sequential(nn.Linear(flat + n_scalar, 128), nn.Sigmoid(), nn.Dropout(0.1),
                                  nn.Linear(128, 32), nn.Sigmoid(), nn.Dropout(0.1), nn.Linear(32, 1))
        for m in self.modules():                       # Keras defaults: Glorot-uniform weights, zero biases
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq, scal):                    # seq: (B, L, 4)
        z = self.conv(seq.transpose(1, 2).unsqueeze(1))
        return self.head(torch.cat([z, scal], dim=1)).squeeze(-1)


def logcosh(pred, y):
    x = pred - y
    return (x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)).mean()


class Tiger2DRegressor:
    """fit(train_df, y) / predict(df). `y` is the per-gene normalised target; rows of `train_df` need a `block` column for the inner split."""

    def __init__(self, flank: int = 1, scalars: bool = True, max_epochs: int = 2000, patience: int = 100, batch_size: int = 2048, seed: int = 0,
                 lr: float = 1e-3, cols=None, standardize: bool = False):
        self.flank, self.use_scalars, self.max_epochs, self.patience, self.bs, self.seed, self.lr = flank, scalars, max_epochs, patience, batch_size, seed, lr
        self.fixed_cols, self.standardize = cols, standardize        # fixed_cols: restrict to scalars present in every dataset the model will see

    def _inputs(self, d):
        s = torch.from_numpy(F.target_window(d, self.flank))
        if not self.use_scalars:
            return s, torch.zeros(len(d), 0)
        x = F.scalar_frame(d).reindex(columns=self.cols)
        x = (x.fillna(self.mu) if self.mu is not None else x.fillna(0.0))
        if self.standardize:
            x = (x - self.mu) / self.sd
        x = x.values.astype(np.float32)
        return s, torch.from_numpy(x)

    def fit(self, d, y, val_frac: float = 0.1, peek=None):
        """peek=(df, y): early-stop on THESE rows instead of an inner split (the authors' cross-validation passes the held-out fold here;
        use only to measure how much that flatters the score)."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.cols = (list(self.fixed_cols) if self.fixed_cols is not None else list(F.scalar_frame(d.iloc[:2]).columns)) if self.use_scalars else []
        sf = F.scalar_frame(d).reindex(columns=self.cols) if self.use_scalars else None
        self.mu = sf.mean() if self.use_scalars else None
        self.sd = sf.std().replace(0, 1.0) if self.use_scalars else None
        s, x = self._inputs(d)
        y = torch.from_numpy(np.asarray(y, np.float32))
        blocks = d["block"].values
        ub = np.unique(blocks)
        vb = set(rng.choice(ub, max(1, int(len(ub) * val_frac)), replace=False))
        vm = np.array([b in vb for b in blocks])
        tr, va = np.where(~vm)[0], np.where(vm)[0]
        if peek is not None:                                                  # train on everything, stop on the evaluation fold
            ps, px = self._inputs(peek[0])
            s, x = torch.cat([s, ps]), torch.cat([x, px])
            y = torch.cat([y, torch.from_numpy(np.asarray(peek[1], np.float32))])
            tr, va = np.arange(len(d)), np.arange(len(d), len(d) + len(peek[0]))
        self.net = Tiger2D(s.shape[1], x.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad()
                loss = logcosh(self.net(s[b], x[b]), y[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(logcosh(self.net(s[va], x[va]), y[va]))
            if vl < best:
                best, wait = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_ = ep + 1
        return self

    def predict(self, d):
        s, x = self._inputs(d)
        self.net.eval()
        with torch.no_grad():
            return self.net(s, x).numpy()
