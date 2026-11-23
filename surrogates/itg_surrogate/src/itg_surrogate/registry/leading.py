"""Checked-in leading baseline-to-beat model configurations and OOD benchmarks.

Import `LEADING_EXPERIMENT` whenever comparing a new architecture, regularizer,
or physics feature against the strongest single-family surrogate on unseen
stellarators (`0.00%` overlap in `equilibrium_files` and `QUASR_IDs`).
"""

from __future__ import annotations

from itg_surrogate.registry.specs import (
    ArchConfig,
    AugmentationConfig,
    BenchmarkRecord,
    DataSplitConfig,
    ExperimentConfig,
    OptimizerConfig,
)

CANONICAL_OOD_SPLIT = DataSplitConfig(
    split_seed=42,
    train_frac=0.80,
    val_frac=0.10,
    test_frac=0.10,
    filter_mode="positive_q",
    target_mode="log1p",
    include_local_shear=False,
)

# -----------------------------------------------------------------------------
# Primary Baseline to Beat (Single Architecture Family, 8 Sharded Seeds)
# -----------------------------------------------------------------------------
LEADING_ARCH = ArchConfig(
    conv_channels=(40, 56, 64, 80, 80),
    kernel_sizes=(11, 9, 7, 5, 3),
    fc_dims=(64, 48),
    activation="silu",
    norm_type="layernorm",
    use_residual=True,
    use_film=True,
    use_multistat_pool=True,
    drop_path_rate=0.0,
    film_hidden_dim=64,
)

LEADING_OPTIMIZER = OptimizerConfig(
    num_epochs=110,
    batch_size=128,
    init_lr=1e-4,
    peak_lr=2.5e-3,
    end_lr=1e-5,
    warmup_epochs=5,
    grad_clip_norm=2.0,
    weight_decay=1e-2,
    mask_kernels_only=True,
)

LEADING_AUGMENTATION = AugmentationConfig(
    use_z96_shift=True,
    use_d96_parity=False,
    input_noise_std=0.0,
    eval_num_shifts=32,
    eval_d96_reflection=False,
    eval_chunk_size=512,
)

LEADING_EXPERIMENT = ExperimentConfig(
    name="resfilm_multipool_80_strong_wd",
    description=(
        "Leading single-family surrogate on unseen stellarators: 5-stage "
        "periodic Pre-LN ResNet-1D with identity-initialized Early FiLM "
        "(a/LT, a/Ln) modulation, [mean, max, std] spatial pooling, per-sample "
        "LayerNorm, Z_96 shift invariance, and strong decoupled weight decay "
        "(wd=1e-2 masked to Conv/Dense kernels)."
    ),
    arch=LEADING_ARCH,
    optimizer=LEADING_OPTIMIZER,
    augmentation=LEADING_AUGMENTATION,
    data_split=CANONICAL_OOD_SPLIT,
    num_seeds=8,
    base_seed=4600,
    reference_benchmark=BenchmarkRecord(
        indiv_z96_test_r2=0.98858,
        ens8_test_r2=0.99070,
        ens8_val_r2=0.98912,
        val_over_train_mse_ratio=10.92,
        n_params=329_609,
        per_class_test_r2={
            "Random boundary Fourier modes": 0.98788,
            "Rotating ellipses, centered on a circle": 0.99504,
            "Rotating ellipses, centered on a curve with torsion": 0.99413,
            "QUASR configurations, vacuum fields": 0.99267,
            "QUASR configurations with added pressure": 0.99215,
        },
        notes=(
            "Evaluated on N_test=9,920 flux tubes from 1,962 held-out "
            "stellarator families (0.00% shared equilibrium_files or QUASR_IDs "
            "with the 15,694 training stellarator families)."
        ),
    ),
)

# -----------------------------------------------------------------------------
# Multi-Regularizer Runner-Up (Lowest Val/Train Overfitting Ratio: 9.30x)
# -----------------------------------------------------------------------------
LEADING_OOD_COMBO_EXPERIMENT = ExperimentConfig(
    name="resfilm_multipool_80_ood_combo",
    description=(
        "Complementary regularized ResFiLM-MultiPool-80 combining kernel "
        "weight decay (wd=5e-3), depth-scheduled Stochastic Depth "
        "(drop_path=0.10), exact 192-element Dihedral D_96 gyrokinetic parity "
        "augmentation, and Gaussian input Jacobian smoothing (sigma=0.015)."
    ),
    arch=ArchConfig(
        conv_channels=(40, 56, 64, 80, 80),
        kernel_sizes=(11, 9, 7, 5, 3),
        fc_dims=(64, 48),
        activation="silu",
        norm_type="layernorm",
        use_residual=True,
        use_film=True,
        use_multistat_pool=True,
        drop_path_rate=0.10,
        film_hidden_dim=64,
    ),
    optimizer=OptimizerConfig(
        num_epochs=110,
        batch_size=128,
        init_lr=1e-4,
        peak_lr=2.5e-3,
        end_lr=1e-5,
        warmup_epochs=5,
        grad_clip_norm=2.0,
        weight_decay=5e-3,
        mask_kernels_only=True,
    ),
    augmentation=AugmentationConfig(
        use_z96_shift=True,
        use_d96_parity=True,
        input_noise_std=0.015,
        eval_num_shifts=32,
        eval_d96_reflection=True,
        eval_chunk_size=512,
    ),
    data_split=CANONICAL_OOD_SPLIT,
    num_seeds=8,
    base_seed=4900,
    reference_benchmark=BenchmarkRecord(
        indiv_z96_test_r2=0.98858,
        ens8_test_r2=0.99065,
        ens8_val_r2=0.98895,
        val_over_train_mse_ratio=9.30,
        n_params=329_609,
        notes=(
            "Achieves the lowest Val/Train MSE ratio (9.30x vs 11.13x unreg) "
            "and provides strong ensemble diversity when combined with "
            "LEADING_EXPERIMENT."
        ),
    ),
)

# Reference numbers for multi-family validation-selected top-K* ensemble and
# tabular baselines on the identical canonical unseen-stellarator split.
REFERENCE_OOD_BASELINES: dict[str, dict[str, float]] = {
    "xgboost_3_features": {
        "test_r2": 0.93856,
        "unstable_test_r2": 0.90004,
    },
    "xgboost_12_paper_features": {
        "test_r2": 0.96891,
        "unstable_test_r2": 0.95020,
    },
    "table1_gelu_32_ens8": {
        "test_r2": 0.98889,
    },
    "resfilm_strong_wd_ens8": {
        "test_r2": 0.99070,
    },
    "regularized_grand_ensemble_top10_val": {
        "test_r2": 0.99073,
        "unstable_test_r2": 0.98473,
        "test_rmse": 0.1386,
    },
}
