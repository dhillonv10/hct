"""Serializable configuration dataclasses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    n_cpgs: int = 12000         # first N CpGs of the priority-ordered panel (clock, age-assoc, X, variance)
    d_model: int = 64
    n_modules: int = 24         # K learned epigenetic modules
    n_layers: int = 3           # transformer layers over the K module tokens (0 = purely additive modules)
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    n_basis: int = 6            # RBF basis functions for the per-CpG nonlinear value path
    d_nl: int = 16
    n_pos_scales: int = 14      # multi-scale genomic position features (period 1 kb ... 100 Mb)
    d_key: int = 32
    linear_skip: bool = True           # per-module linear (clock-like) term pooled through the module assignment
    use_genomic_context: bool = True   # ablation switch: False -> learned CpG-id embeddings only
    use_id_embedding: bool = True      # False -> tokens are built from genomic context alone
    id_dropout: float = 0.25           # train-time prob. that a CpG's id embedding is hidden
    distributional: bool = True        # predict (mu, sigma) instead of a point estimate
    ssl_decoder: bool = True
    cpg_split: str = ""         # "A" / "B": train on a random half of the panel (for the unseen-CpG transfer test)
    cpg_split_seed: int = 123

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.02
    warmup_steps: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    study_balance_power: float = 0.5   # sample weight ~ n_study ** -power
    celltype_balance_power: float = 0.0  # >0 also up-weights rare annotated cell types (weight ~ n_celltype ** -power)
    warm_start_linear: bool = True     # initialise the linear route from a study-grouped ridge fit
    skip_lr_mult: float = 0.01         # lr multiplier for the per-CpG linear weights (they sum over ~1e4 CpGs)
    # loss weights
    w_nll: float = 1.0
    w_huber: float = 1.0
    w_sex: float = 0.2
    w_ssl: float = 0.5
    w_assign_entropy: float = 0.0   # >0 sharpens modules into smaller CpG sets (normalised entropy penalty)
    # augmentation (all applied to *real* measured profiles; nothing synthetic is generated)
    aug: bool = True
    mask_max: float = 0.3        # per-sample masked fraction ~ U(0, mask_max); masked CpGs feed the SSL loss
    noise_max: float = 0.02
    distort_prob: float = 0.5    # monotone logit-space distortion mimicking normalisation differences
    template_miss_prob: float = 0.3  # copy another sample's real missingness pattern (platform-style)
    val_study_frac: float = 0.15
    split_seed: int = 0          # which training studies are held out for validation (independent of init seed)
    seed: int = 0
    threads: int = 4
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
