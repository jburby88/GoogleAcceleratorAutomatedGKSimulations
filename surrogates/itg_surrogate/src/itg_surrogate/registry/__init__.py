"""Checked-in model configurations, leading baselines, and ablation sweeps."""

from itg_surrogate.registry.leading import (
    CANONICAL_OOD_SPLIT,
    LEADING_ARCH,
    LEADING_AUGMENTATION,
    LEADING_EXPERIMENT,
    LEADING_OOD_COMBO_EXPERIMENT,
    LEADING_OPTIMIZER,
    REFERENCE_OOD_BASELINES,
)
from itg_surrogate.registry.specs import (
    ArchConfig,
    AugmentationConfig,
    BenchmarkRecord,
    DataSplitConfig,
    ExperimentConfig,
    OptimizerConfig,
)
from itg_surrogate.registry.sweeps import (
    ARCHITECTURE_LADDER_SWEEP,
    EXPERIMENT_REGISTRY,
    NORMALIZATION_DIAGNOSTIC_SWEEP,
    OOD_REGULARIZATION_SWEEP,
    SWEEP_REGISTRY,
    WEIGHT_DECAY_SCAN_SWEEP,
    get_experiment,
    get_sweep,
    list_experiments,
    list_sweeps,
)

__all__ = [
    "ARCHITECTURE_LADDER_SWEEP",
    "ArchConfig",
    "AugmentationConfig",
    "BenchmarkRecord",
    "CANONICAL_OOD_SPLIT",
    "DataSplitConfig",
    "EXPERIMENT_REGISTRY",
    "ExperimentConfig",
    "LEADING_ARCH",
    "LEADING_AUGMENTATION",
    "LEADING_EXPERIMENT",
    "LEADING_OOD_COMBO_EXPERIMENT",
    "LEADING_OPTIMIZER",
    "NORMALIZATION_DIAGNOSTIC_SWEEP",
    "OOD_REGULARIZATION_SWEEP",
    "OptimizerConfig",
    "REFERENCE_OOD_BASELINES",
    "SWEEP_REGISTRY",
    "WEIGHT_DECAY_SCAN_SWEEP",
    "get_experiment",
    "get_sweep",
    "list_experiments",
    "list_sweeps",
]
