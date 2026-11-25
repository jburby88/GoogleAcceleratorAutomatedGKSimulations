"""Tabular XGBoost baselines evaluated on the unseen-stellarator split."""

from __future__ import annotations

import dataclasses

import numpy as np

from itg_surrogate.data.dataset import PreparedStellaratorDataset
from itg_surrogate.data.features import compute_3_tabular_features, compute_paper_12_features
from itg_surrogate.training.metrics import compute_per_class_r2, compute_r2


@dataclasses.dataclass(frozen=True)
class TabularBaselineResult:
    """Validation and held-out unseen-stellarator test metrics for XGBoost."""

    name: str
    val_r2: float
    test_r2: float
    unstable_test_r2: float
    per_class_test_r2: dict[str, float]
    test_predictions: np.ndarray


def run_xgboost_baselines(
    dataset: PreparedStellaratorDataset,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.15,
    random_state: int = 42,
) -> list[TabularBaselineResult]:
    """Fits and scores the 3-feature and 12-feature Section 5 XGBoost baselines."""
    try:
        import xgboost as xgb  # pylint: disable=g-import-not-at-top
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for tabular baselines. "
            'Install via `pip install -e ".[baselines]"`.'
        ) from exc

    lt = dataset.scalars_raw[:, 0]
    ln = dataset.scalars_raw[:, 1]
    x_3, _ = compute_3_tabular_features(dataset.raw_feature_tensor, lt, ln)
    x_12, _ = compute_paper_12_features(dataset.raw_feature_tensor, lt, ln)

    tr = dataset.split.train_idx
    va = dataset.split.val_idx
    te = dataset.split.test_idx
    y_tr = dataset.targets[tr]
    y_va = dataset.targets[va]
    y_te = dataset.targets[te]
    uns_te = dataset.unstable_mask[te]
    cls_te = dataset.equilibrium_class[te]

    results: list[TabularBaselineResult] = []
    for name, x_mat in (
        ("xgboost_3_features", x_3),
        ("xgboost_12_paper_features", x_12),
    ):
        reg = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
        )
        reg.fit(x_mat[tr], y_tr)
        va_pred = reg.predict(x_mat[va])
        te_pred = reg.predict(x_mat[te])

        per_cls = compute_per_class_r2(y_te, te_pred, cls_te, dataset.equilibrium_descriptions)
        unstable_r2 = compute_r2(y_te[uns_te], te_pred[uns_te]) if np.any(uns_te) else float("nan")

        results.append(
            TabularBaselineResult(
                name=name,
                val_r2=compute_r2(y_va, va_pred),
                test_r2=compute_r2(y_te, te_pred),
                unstable_test_r2=unstable_r2,
                per_class_test_r2=per_cls,
                test_predictions=te_pred.astype(np.float32),
            )
        )
    return results
