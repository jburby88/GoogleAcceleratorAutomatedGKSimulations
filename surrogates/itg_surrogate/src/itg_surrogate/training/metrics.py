"""Regression metrics, per-class breakdowns, and validation top-K* selection."""

from __future__ import annotations

import dataclasses

import numpy as np

from itg_surrogate.data.dataset import invert_target_to_q
from itg_surrogate.registry.specs import TargetMode


@dataclasses.dataclass(frozen=True)
class ValidationSelectedEnsembleResult:
    """Metrics for a validation-ranked top-K* ensemble (zero test peeking)."""

    best_k: int
    total_candidates: int
    selected_indices: np.ndarray
    val_r2: float
    test_r2: float
    unstable_test_r2: float
    unstable_lnq_r2: float
    test_rmse: float
    test_mean_prediction: np.ndarray
    test_std_prediction: np.ndarray
    per_class_test_r2: dict[str, float]


def compute_r2(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Computes coefficient of determination `R^2`."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Computes root-mean-square error."""
    return float(np.sqrt(np.mean((targets - predictions) ** 2)))


def compute_per_class_r2(
    targets: np.ndarray,
    predictions: np.ndarray,
    equilibrium_class: np.ndarray,
    equilibrium_descriptions: tuple[str, ...],
) -> dict[str, float]:
    """Computes `R^2` separately within each stellarator equilibrium family."""
    breakdown: dict[str, float] = {}
    for c_id, desc in enumerate(equilibrium_descriptions):
        mask = equilibrium_class == c_id
        if np.any(mask):
            breakdown[desc] = compute_r2(targets[mask], predictions[mask])
    return breakdown


def select_validation_topk_ensemble(
    val_member_preds: np.ndarray,
    test_member_preds: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    q_test_true: np.ndarray,
    unstable_test_mask: np.ndarray,
    test_equilibrium_class: np.ndarray,
    equilibrium_descriptions: tuple[str, ...],
    target_mode: TargetMode = "log1p",
) -> ValidationSelectedEnsembleResult:
    """Orders candidate models by validation MSE and selects optimal `K*` on val."""
    num_members = val_member_preds.shape[0]
    val_mses = np.array(
        [float(np.mean((y_val - val_member_preds[i]) ** 2)) for i in range(num_members)],
        dtype=np.float64,
    )
    order = np.argsort(val_mses)
    ordered_val = val_member_preds[order]
    ordered_test = test_member_preds[order]

    val_r2_curve = [
        compute_r2(y_val, np.mean(ordered_val[:k], axis=0)) for k in range(1, num_members + 1)
    ]
    best_k = int(np.argmax(val_r2_curve)) + 1
    selected_indices = order[:best_k]

    ens_mean = np.mean(ordered_test[:best_k], axis=0)
    ens_std = np.std(ordered_test[:best_k], axis=0)

    q_pred = invert_target_to_q(ens_mean, target_mode)
    if np.any(unstable_test_mask):
        unstable_r2 = compute_r2(y_test[unstable_test_mask], ens_mean[unstable_test_mask])
        ln_q_true = np.log(q_test_true[unstable_test_mask])
        ln_q_pred = np.log(np.maximum(q_pred[unstable_test_mask], 1e-4))
        unstable_lnq_r2 = compute_r2(ln_q_true, ln_q_pred)
    else:
        unstable_r2 = float("nan")
        unstable_lnq_r2 = float("nan")

    per_cls = compute_per_class_r2(
        y_test, ens_mean, test_equilibrium_class, equilibrium_descriptions
    )
    return ValidationSelectedEnsembleResult(
        best_k=best_k,
        total_candidates=num_members,
        selected_indices=selected_indices,
        val_r2=float(val_r2_curve[best_k - 1]),
        test_r2=compute_r2(y_test, ens_mean),
        unstable_test_r2=unstable_r2,
        unstable_lnq_r2=unstable_lnq_r2,
        test_rmse=compute_rmse(y_test, ens_mean),
        test_mean_prediction=ens_mean.astype(np.float32),
        test_std_prediction=ens_std.astype(np.float32),
        per_class_test_r2=per_cls,
    )
