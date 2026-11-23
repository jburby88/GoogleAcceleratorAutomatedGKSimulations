"""Strongly typed configuration and benchmark schemas for the model registry.

All experiments in this codebase use the out-of-distribution stellarator-grouped
train/validation/test split (`DataSplitConfig`), in which zero flux tubes in
validation or test share a 3D MHD `equilibrium_file` or a positive `QUASR_ID`
boundary family with the training set. Random per-flux-tube splits are
intentionally excluded because 99.09% of randomly held-out flux tubes share a
parent 3D MHD equilibrium with radial/field-line neighbors in the training set,
and 100.00% of QUASR test flux tubes share a `QUASR_ID` boundary shape across
vacuum and finite-beta equilibria (99.35% canonical stellarator overlap overall).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

ActivationName = Literal["relu", "gelu", "silu"]
NormType = Literal["layernorm", "batchnorm"]
TargetMode = Literal["log1p", "log_q"]
DatasetFilterMode = Literal["positive_q", "unstable_only"]


@dataclasses.dataclass(frozen=True)
class DataSplitConfig:
    """Configuration for preprocessing and the unseen-stellarator grouped split.

    Attributes:
      split_seed: Deterministic PRNG seed for stratified stellarator-group
        partitioning across configuration strata.
      train_frac: Fraction of canonical stellarator groups assigned to training.
      val_frac: Fraction of canonical stellarator groups assigned to validation.
      test_frac: Fraction of canonical stellarator groups assigned to held-out
        testing.
      filter_mode: Sample filter (`'positive_q'` keeps `Q > 0`, `N = 98,932`;
        `'unstable_only'` keeps `Q > 0.1`, `N = 67,782`).
      target_mode: Regression target transform (`'log1p'` for Eq. 3.1 `ln(1 + Q)`,
        or `'log_q'` for `ln(max(Q, 0.1))`).
      include_local_shear: Whether to append spectral local shear
        `S = d/dz(gds21 / gds22)` as an 8th periodic field-line channel.
    """

    split_seed: int = 42
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10
    filter_mode: DatasetFilterMode = "positive_q"
    target_mode: TargetMode = "log1p"
    include_local_shear: bool = False

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0; got {total:.6f} "
                f"({self.train_frac}, {self.val_frac}, {self.test_frac})."
            )
        if self.train_frac <= 0.0 or self.val_frac <= 0.0 or self.test_frac <= 0.0:
            raise ValueError("All split fractions (train, val, test) must be > 0.")


@dataclasses.dataclass(frozen=True)
class ArchConfig:
    """Architecture hyperparameters for `StellaratorPeriodicCNN`.

    Attributes:
      conv_channels: Output channels for each of the 5 periodic `Conv1D + MaxPool(2)`
        stages (`96 -> 48 -> 24 -> 12 -> 6 -> 3`).
      kernel_sizes: Kernel widths along `z` for each of the 5 periodic conv stages.
      fc_dims: Hidden widths of the post-pooling fully connected stages.
      activation: Pointwise nonlinearity (`'relu'`, `'gelu'`, or `'silu'`).
      norm_type: Feature normalization (`'layernorm'` default; `'batchnorm'` kept
        only for reproducing Figure 7 / diagnosing heavy-tail variance spikes).
      use_residual: Adds zero-initialized Pre-LN residual branches in conv and FC
        stages.
      use_film: Modulates normalized conv features at all 5 spatial scales via
        identity-initialized (`1 + gamma`, `beta`) Feature-wise Linear Modulation
        from `(a/LT, a/Ln)`.
      use_multistat_pool: Pools the 3 coarse spatial bins via `[mean, max, std]`
        instead of `mean` alone (preserving exact 32-coset `Z_96` invariance).
      drop_path_rate: Peak Stochastic Depth rate linearly scheduled across
        residual blocks (`0.0` disables DropPath). Applied strictly to residual
        branches, never before `LayerNorm` or the final linear readout head.
      film_hidden_dim: Hidden width of the scalar-gradient FiLM embedding MLP.
    """

    conv_channels: tuple[int, ...] = (40, 56, 64, 80, 80)
    kernel_sizes: tuple[int, ...] = (11, 9, 7, 5, 3)
    fc_dims: tuple[int, ...] = (64, 48)
    activation: ActivationName = "silu"
    norm_type: NormType = "layernorm"
    use_residual: bool = True
    use_film: bool = True
    use_multistat_pool: bool = True
    drop_path_rate: float = 0.0
    film_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if len(self.conv_channels) != 5 or len(self.kernel_sizes) != 5:
            raise ValueError(
                "Expected exactly 5 conv stages (96 -> 48 -> 24 -> 12 -> 6 -> 3); "
                f"got {len(self.conv_channels)} channels and "
                f"{len(self.kernel_sizes)} kernel sizes."
            )
        if not (0.0 <= self.drop_path_rate < 1.0):
            raise ValueError(f"drop_path_rate must be in [0.0, 1.0); got {self.drop_path_rate}.")
        if self.drop_path_rate > 0.0 and not self.use_residual:
            raise ValueError("drop_path_rate > 0 requires use_residual=True.")


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Training schedule and decoupled weight-decay hyperparameters.

    Note:
      Training always minimizes pure L2 MSE loss so predictions estimate the
      unbiased conditional mean scored by `R^2` (avoiding Huber conditional-median
      bias). When `mask_kernels_only=True`, decoupled AdamW weight decay is
      restricted to multidimensional `Conv`/`Dense` weight tensors (`ndim >= 2`),
      leaving 1D biases and `LayerNorm` scales unpenalized.
    """

    num_epochs: int = 110
    batch_size: int = 128
    init_lr: float = 1e-4
    peak_lr: float = 2.5e-3
    end_lr: float = 1e-5
    warmup_epochs: int = 5
    grad_clip_norm: float = 2.0
    weight_decay: float = 1e-2
    mask_kernels_only: bool = True


@dataclasses.dataclass(frozen=True)
class AugmentationConfig:
    """Symmetry-preserving training augmentations and evaluation coset averaging.

    Attributes:
      use_z96_shift: Applies independent uniform random `Z_96` cyclic shifts along
        `z` to every sample in each training step.
      use_d96_parity: Applies exact Dihedral `D_96 = Z_96 x| Z_2` gyrokinetic
        reflection (`z -> -z` with sign flip on pseudoscalar radial-shear channels
        `gbdrift0` and `gds21`) with probability `0.5` during training.
      input_noise_std: Standard deviation of zero-mean Gaussian perturbation added
        to standardized `z` channels during training (acts as a Tikhonov Jacobian
        smoothness penalty `sigma^2 ||grad_z y_hat||_2^2`).
      eval_num_shifts: Number of `Z_96` cyclic shifts averaged at inference (`32`
        gives exact `Z_96` invariance across the 5 `MaxPool(2)` downsamplings).
      eval_d96_reflection: Whether to also average predictions across `z` and its
        exact `D_96` reflection at inference (`64` total coset evaluations when
        combined with `eval_num_shifts=32`).
      eval_chunk_size: Batch chunk size for memory-bounded TPU/GPU evaluation.
    """

    use_z96_shift: bool = True
    use_d96_parity: bool = False
    input_noise_std: float = 0.0
    eval_num_shifts: int = 32
    eval_d96_reflection: bool = False
    eval_chunk_size: int = 512


@dataclasses.dataclass(frozen=True)
class BenchmarkRecord:
    """Verified reference metrics on the unseen-stellarator grouped test split."""

    indiv_z96_test_r2: float
    ens8_test_r2: float
    ens8_val_r2: float
    val_over_train_mse_ratio: float
    n_params: int
    per_class_test_r2: dict[str, float] = dataclasses.field(default_factory=dict)
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Complete reproducible specification for an ensemble surrogate experiment."""

    name: str
    description: str
    arch: ArchConfig = dataclasses.field(default_factory=ArchConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)
    data_split: DataSplitConfig = dataclasses.field(default_factory=DataSplitConfig)
    num_seeds: int = 8
    base_seed: int = 4000
    reference_benchmark: BenchmarkRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the full experiment configuration to a plain dictionary."""
        return dataclasses.asdict(self)
