"""GSPMD sharding, evaluation, metrics, and ensemble training loops."""

from itg_surrogate.training.evaluation import eval_batched_single, eval_ensemble_sharded
from itg_surrogate.training.metrics import (
    ValidationSelectedEnsembleResult,
    compute_per_class_r2,
    compute_r2,
    compute_rmse,
    select_validation_topk_ensemble,
)
from itg_surrogate.training.sharding import (
    EnsembleMeshContext,
    configure_jax_float32,
    create_ensemble_mesh,
)
from itg_surrogate.training.trainer import (
    ExperimentRunResult,
    SweepRunSummary,
    build_optimizer,
    run_sweep,
    train_experiment,
    wd_kernel_mask,
)

__all__ = [
    "EnsembleMeshContext",
    "ExperimentRunResult",
    "SweepRunSummary",
    "ValidationSelectedEnsembleResult",
    "build_optimizer",
    "compute_per_class_r2",
    "compute_r2",
    "compute_rmse",
    "configure_jax_float32",
    "create_ensemble_mesh",
    "eval_batched_single",
    "eval_ensemble_sharded",
    "run_sweep",
    "select_validation_topk_ensemble",
    "train_experiment",
    "wd_kernel_mask",
]
