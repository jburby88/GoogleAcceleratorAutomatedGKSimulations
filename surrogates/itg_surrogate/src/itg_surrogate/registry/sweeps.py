"""Checked-in ablation sweeps for architecture, regularization, and norms.

All sweeps share `CANONICAL_OOD_SPLIT` (zero shared `equilibrium_files` or
`QUASR_IDs` across train/validation/test) so every experiment is directly
comparable against `LEADING_EXPERIMENT` (`resfilm_multipool_80_strong_wd`).

Documented Anti-Patterns (Intentionally Excluded from Active Sweeps):
  1. Pre-LayerNorm / Pre-Readout Feature Dropout (`nn.Dropout` immediately
     before `nn.LayerNorm` or the final linear regression `Dense(1)` head):
     Multiplicative `1 / (1 - p)` scaling during training vs. `1` at inference
     shifts activation variance entering `LayerNorm` and induces ridge-like
     attenuation on continuous regression outputs (`R^2` dropped `0.99091 ->
     0.98914` in controlled ablations). Use residual-branch `drop_path_rate`
     instead, which leaves the normalized skip highway untouched.
  2. Huber Training Loss (`optax.huber_loss`) When Scoring by `R^2`:
     Linear tail penalties pull predictions toward the conditional median on
     skewed gyrokinetic turbulence residuals, biasing the `L_2` conditional mean
     and degrading `R^2`. Always train with pure MSE when `R^2` / `L_2` accuracy
     is the target metric.
"""

from __future__ import annotations

import dataclasses

from itg_surrogate.registry.leading import (
    CANONICAL_OOD_SPLIT,
    LEADING_AUGMENTATION,
    LEADING_EXPERIMENT,
    LEADING_OOD_COMBO_EXPERIMENT,
)
from itg_surrogate.registry.specs import (
    ArchConfig,
    AugmentationConfig,
    BenchmarkRecord,
    ExperimentConfig,
    OptimizerConfig,
)


def _default_unreg_optimizer(weight_decay: float = 1e-4) -> OptimizerConfig:
    return OptimizerConfig(
        num_epochs=110,
        batch_size=128,
        init_lr=1e-4,
        peak_lr=2.5e-3,
        end_lr=1e-5,
        warmup_epochs=5,
        grad_clip_norm=2.0,
        weight_decay=weight_decay,
        mask_kernels_only=True,
    )


# -----------------------------------------------------------------------------
# Sweep 1: Controlled Architecture Ladder (Baseline wd=1e-4 on Unseen Equilibria)
# -----------------------------------------------------------------------------
ARCHITECTURE_LADDER_SWEEP: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        name="table1_gelu_32",
        description=(
            "Compact 32-channel baseline matching the [4, 32] channel cap in "
            "Landreman et al. (2025) Table 1, stabilized with LayerNorm."
        ),
        arch=ArchConfig(
            conv_channels=(32, 32, 32, 32, 32),
            kernel_sizes=(15, 11, 9, 5, 3),
            fc_dims=(32, 32),
            activation="gelu",
            norm_type="layernorm",
            use_residual=False,
            use_film=False,
            use_multistat_pool=False,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4000,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98618,
            ens8_test_r2=0.98889,
            ens8_val_r2=0.98734,
            val_over_train_mse_ratio=3.79,
            n_params=34_849,
        ),
    ),
    ExperimentConfig(
        name="pyramid_relu_64",
        description="5-stage ReLU pyramid CNN widening channels from 32 to 64.",
        arch=ArchConfig(
            conv_channels=(32, 48, 64, 64, 64),
            kernel_sizes=(13, 9, 7, 5, 3),
            fc_dims=(64, 32),
            activation="relu",
            norm_type="layernorm",
            use_residual=False,
            use_film=False,
            use_multistat_pool=False,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4050,
    ),
    ExperimentConfig(
        name="pyramid_gelu_64",
        description="5-stage GELU pyramid CNN widening channels from 32 to 64.",
        arch=ArchConfig(
            conv_channels=(32, 48, 64, 64, 64),
            kernel_sizes=(15, 11, 7, 5, 3),
            fc_dims=(64, 48),
            activation="gelu",
            norm_type="layernorm",
            use_residual=False,
            use_film=False,
            use_multistat_pool=False,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4080,
    ),
    ExperimentConfig(
        name="pyramid_silu_80",
        description=("Unconditioned 5-stage SiLU pyramid CNN (40 -> 80 channels, GAP)."),
        arch=ArchConfig(
            conv_channels=(40, 56, 64, 80, 80),
            kernel_sizes=(11, 9, 7, 5, 3),
            fc_dims=(64, 48),
            activation="silu",
            norm_type="layernorm",
            use_residual=False,
            use_film=False,
            use_multistat_pool=False,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4100,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98776,
            ens8_test_r2=0.98970,
            ens8_val_r2=0.98831,
            val_over_train_mse_ratio=5.43,
            n_params=102_793,
        ),
    ),
    ExperimentConfig(
        name="pyramid_multistat_pool_80",
        description=(
            "Isolates [mean, max, std] spatial pooling over the 3 coarse bins "
            "(without FiLM or residual blocks)."
        ),
        arch=ArchConfig(
            conv_channels=(40, 56, 64, 80, 80),
            kernel_sizes=(11, 9, 7, 5, 3),
            fc_dims=(64, 48),
            activation="silu",
            norm_type="layernorm",
            use_residual=False,
            use_film=False,
            use_multistat_pool=True,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4200,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98779,
            ens8_test_r2=0.98964,
            ens8_val_r2=0.98842,
            val_over_train_mse_ratio=6.00,
            n_params=113_033,
        ),
    ),
    ExperimentConfig(
        name="pyramid_preln_residual_80",
        description=(
            "Isolates zero-initialized Pre-LN periodic residual blocks in all "
            "5 conv stages and 2 FC stages (without FiLM or multi-stat pool)."
        ),
        arch=ArchConfig(
            conv_channels=(40, 56, 64, 80, 80),
            kernel_sizes=(11, 9, 7, 5, 3),
            fc_dims=(64, 48),
            activation="silu",
            norm_type="layernorm",
            use_residual=True,
            use_film=False,
            use_multistat_pool=False,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4300,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98793,
            ens8_test_r2=0.99000,
            ens8_val_r2=0.98878,
            val_over_train_mse_ratio=6.76,
            n_params=277_449,
        ),
    ),
    ExperimentConfig(
        name="pyramid_early_film_80",
        description=(
            "Isolates Early FiLM conditioning of all 5 conv stages on "
            "(a/LT, a/Ln) without residual blocks or multi-stat pooling."
        ),
        arch=ArchConfig(
            conv_channels=(40, 56, 64, 80, 80),
            kernel_sizes=(11, 9, 7, 5, 3),
            fc_dims=(64, 48),
            activation="silu",
            norm_type="layernorm",
            use_residual=False,
            use_film=True,
            use_multistat_pool=False,
        ),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4400,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98802,
            ens8_test_r2=0.99042,
            ens8_val_r2=0.98889,
            val_over_train_mse_ratio=8.90,
            n_params=144_713,
        ),
    ),
    ExperimentConfig(
        name="resfilm_multipool_80_unreg",
        description=(
            "Combined ResFiLM-MultiPool-80 architecture at light baseline weight decay (wd=1e-4)."
        ),
        arch=LEADING_EXPERIMENT.arch,
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4500,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98823,
            ens8_test_r2=0.99047,
            ens8_val_r2=0.98880,
            val_over_train_mse_ratio=11.13,
            n_params=329_609,
        ),
    ),
)

# -----------------------------------------------------------------------------
# Sweep 2: Clean L2-Unbiased Regularization Sweep on ResFiLM-MultiPool-80
# -----------------------------------------------------------------------------
OOD_REGULARIZATION_SWEEP: tuple[ExperimentConfig, ...] = (
    ARCHITECTURE_LADDER_SWEEP[-1],  # resfilm_multipool_80_unreg (wd=1e-4)
    LEADING_EXPERIMENT,  # resfilm_multipool_80_strong_wd (wd=1e-2)
    ExperimentConfig(
        name="resfilm_multipool_80_droppath",
        description=(
            "Isolates linearly depth-scheduled Stochastic Depth "
            "(drop_path_rate=0.14) across the 7 zero-init residual branches."
        ),
        arch=dataclasses.replace(LEADING_EXPERIMENT.arch, drop_path_rate=0.14),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4700,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98838,
            ens8_test_r2=0.99056,
            ens8_val_r2=0.98891,
            val_over_train_mse_ratio=9.32,
            n_params=329_609,
        ),
    ),
    ExperimentConfig(
        name="resfilm_multipool_80_d96_parity",
        description=(
            "Isolates exact 192-element Dihedral D_96 gyrokinetic parity "
            "augmentation (z -> -z with gbdrift0, gds21 sign flip) + 64-coset "
            "D_96 inference averaging."
        ),
        arch=LEADING_EXPERIMENT.arch,
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=AugmentationConfig(
            use_z96_shift=True,
            use_d96_parity=True,
            input_noise_std=0.0,
            eval_num_shifts=32,
            eval_d96_reflection=True,
            eval_chunk_size=512,
        ),
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=4800,
        reference_benchmark=BenchmarkRecord(
            indiv_z96_test_r2=0.98839,
            ens8_test_r2=0.99055,
            ens8_val_r2=0.98896,
            val_over_train_mse_ratio=10.44,
            n_params=329_609,
        ),
    ),
    LEADING_OOD_COMBO_EXPERIMENT,
)

# -----------------------------------------------------------------------------
# Sweep 3: Logarithmic Kernel Weight-Decay Scan on ResFiLM-MultiPool-80
# -----------------------------------------------------------------------------
WEIGHT_DECAY_SCAN_SWEEP: tuple[ExperimentConfig, ...] = tuple(
    ExperimentConfig(
        name=f"resfilm_multipool_80_wd_{label}",
        description=(
            f"Decoupled AdamW kernel weight decay scan (wd={wd:.0e}, "
            "masked to ndim >= 2) on ResFiLM-MultiPool-80."
        ),
        arch=LEADING_EXPERIMENT.arch,
        optimizer=_default_unreg_optimizer(wd),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=5000 + 100 * idx,
    )
    for idx, (label, wd) in enumerate(
        (
            ("1e4", 1e-4),
            ("1e3", 1e-3),
            ("3e3", 3e-3),
            ("1e2", 1e-2),
            ("3e2", 3e-2),
            ("1e1", 1e-1),
        )
    )
)

# -----------------------------------------------------------------------------
# Sweep 4: LayerNorm vs. BatchNorm Heavy-Tail Stability Diagnostic
# -----------------------------------------------------------------------------
NORMALIZATION_DIAGNOSTIC_SWEEP: tuple[ExperimentConfig, ...] = (
    ARCHITECTURE_LADDER_SWEEP[0],  # table1_gelu_32 (layernorm)
    ExperimentConfig(
        name="table1_gelu_32_batchnorm",
        description=(
            "Exact Figure 7 BatchNorm control on Table1-GELU-32, illustrating "
            "running-variance spikes from >250-700 sigma gds2/gds22 outliers."
        ),
        arch=dataclasses.replace(ARCHITECTURE_LADDER_SWEEP[0].arch, norm_type="batchnorm"),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=6000,
    ),
    ARCHITECTURE_LADDER_SWEEP[3],  # pyramid_silu_80 (layernorm)
    ExperimentConfig(
        name="pyramid_silu_80_batchnorm",
        description="BatchNorm control on Pyramid-SiLU-80.",
        arch=dataclasses.replace(ARCHITECTURE_LADDER_SWEEP[3].arch, norm_type="batchnorm"),
        optimizer=_default_unreg_optimizer(1e-4),
        augmentation=LEADING_AUGMENTATION,
        data_split=CANONICAL_OOD_SPLIT,
        base_seed=6100,
    ),
)

SWEEP_REGISTRY: dict[str, tuple[ExperimentConfig, ...]] = {
    "architecture_ladder": ARCHITECTURE_LADDER_SWEEP,
    "ood_regularization": OOD_REGULARIZATION_SWEEP,
    "weight_decay_scan": WEIGHT_DECAY_SCAN_SWEEP,
    "normalization_diagnostic": NORMALIZATION_DIAGNOSTIC_SWEEP,
}


def _build_experiment_index() -> dict[str, ExperimentConfig]:
    index: dict[str, ExperimentConfig] = {
        LEADING_EXPERIMENT.name: LEADING_EXPERIMENT,
        LEADING_OOD_COMBO_EXPERIMENT.name: LEADING_OOD_COMBO_EXPERIMENT,
    }
    for sweep_exps in SWEEP_REGISTRY.values():
        for exp in sweep_exps:
            index[exp.name] = exp
    return index


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = _build_experiment_index()


def list_sweeps() -> list[str]:
    """Returns sorted names of all checked-in ablation sweeps."""
    return sorted(SWEEP_REGISTRY.keys())


def get_sweep(name: str) -> tuple[ExperimentConfig, ...]:
    """Looks up a named ablation sweep in `SWEEP_REGISTRY`."""
    if name not in SWEEP_REGISTRY:
        available = ", ".join(list_sweeps())
        raise KeyError(f"Unknown sweep {name!r}. Available sweeps: {available}")
    return SWEEP_REGISTRY[name]


def list_experiments() -> list[str]:
    """Returns sorted names of all checked-in experiment configurations."""
    return sorted(EXPERIMENT_REGISTRY.keys())


def get_experiment(name: str) -> ExperimentConfig:
    """Looks up a single named experiment configuration (`'leading'` alias supported)."""
    if name == "leading":
        return LEADING_EXPERIMENT
    if name not in EXPERIMENT_REGISTRY:
        available = ", ".join(list_experiments())
        raise KeyError(f"Unknown experiment {name!r}. Available experiments: {available}")
    return EXPERIMENT_REGISTRY[name]
