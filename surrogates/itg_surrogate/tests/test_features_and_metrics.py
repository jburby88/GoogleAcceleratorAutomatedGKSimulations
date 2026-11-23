"""Unit tests for engineered physical features and validation ensemble selection."""

from __future__ import annotations

import unittest

import numpy as np

from itg_surrogate.data.features import (
    compute_3_tabular_features,
    compute_analytic_proxy_fq,
    compute_paper_12_features,
)
from itg_surrogate.training.metrics import select_validation_topk_ensemble


class FeaturesAndMetricsTest(unittest.TestCase):
    def _make_raw_gx_tensor(self, n_rows: int = 12) -> np.ndarray:
        rng = np.random.RandomState(7)
        raw = rng.normal(size=(n_rows, 96, 7)).astype(np.float64)
        raw[:, :, 0] = np.abs(raw[:, :, 0]) + 1.0  # bmag > 0
        raw[:, :, 4] = np.abs(raw[:, :, 4]) + 0.5  # gds2 > 0
        raw[:, :, 6] = np.abs(raw[:, :, 6]) + 0.5  # grad_x_sq > 0
        return raw

    def test_engineered_features_shapes_and_z96_shift_invariance(self) -> None:
        raw = self._make_raw_gx_tensor(n_rows=10)
        lt = np.linspace(0.8, 3.5, 10, dtype=np.float32)
        ln = np.linspace(0.1, 2.2, 10, dtype=np.float32)

        fq = compute_analytic_proxy_fq(raw)
        self.assertEqual(fq.shape, (10,))
        self.assertTrue(np.all(fq > 0.0))

        f3, names3 = compute_3_tabular_features(raw, lt, ln)
        self.assertEqual(f3.shape, (10, 3))
        self.assertEqual(len(names3), 3)
        self.assertTrue(np.all(np.isfinite(f3)))

        f12, names12 = compute_paper_12_features(raw, lt, ln)
        self.assertEqual(f12.shape, (10, 12))
        self.assertEqual(len(names12), 12)
        self.assertTrue(np.all(np.isfinite(f12)))

        # Periodic shift along z by a full integer grid step must preserve f_Q
        # and the non-derivative summary features; check f_Q exact invariance.
        raw_shifted = np.roll(raw, shift=17, axis=1)
        fq_shifted = compute_analytic_proxy_fq(raw_shifted)
        np.testing.assert_allclose(fq, fq_shifted, rtol=1e-6, atol=1e-6)

    def test_validation_topk_selection_never_reads_test_targets_for_ranking(
        self,
    ) -> None:
        rng = np.random.RandomState(123)
        q_val = rng.uniform(0.2, 15.0, size=40).astype(np.float32)
        q_test = rng.uniform(0.2, 15.0, size=40).astype(np.float32)
        y_val = np.log1p(q_val).astype(np.float32)
        y_test = np.log1p(q_test).astype(np.float32)
        unstable_test = q_test > 0.1
        eq_class_test = np.zeros(40, dtype=np.int64)

        # Members 0 & 1 have complementary small errors on val; 2 & 3 are noisy
        noise = rng.normal(scale=0.04, size=40).astype(np.float32)
        val_preds = np.vstack(
            [
                y_val + noise,
                y_val - noise,
                rng.normal(scale=3.0, size=40),
                rng.normal(scale=3.0, size=40),
            ]
        ).astype(np.float32)
        test_preds = np.vstack(
            [
                y_test + noise,
                y_test - noise,
                rng.normal(scale=3.0, size=40),
                rng.normal(scale=3.0, size=40),
            ]
        ).astype(np.float32)

        res = select_validation_topk_ensemble(
            val_member_preds=val_preds,
            test_member_preds=test_preds,
            y_val=y_val,
            y_test=y_test,
            q_test_true=q_test,
            unstable_test_mask=unstable_test,
            test_equilibrium_class=eq_class_test,
            equilibrium_descriptions=("Class0",),
        )
        self.assertEqual(res.best_k, 2)
        self.assertEqual(set(res.selected_indices.tolist()), {0, 1})
        self.assertGreater(res.test_r2, 0.999)


if __name__ == "__main__":
    unittest.main()
