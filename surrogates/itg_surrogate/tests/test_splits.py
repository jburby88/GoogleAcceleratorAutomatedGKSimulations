"""Unit tests for canonical stellarator-grouped splitting and zero leakage."""

from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np

from itg_surrogate.data.dataset import load_stellarator_h5
from itg_surrogate.data.splits import (
    StellaratorSplitIndices,
    build_canonical_stellarator_groups,
    make_stellarator_grouped_split,
    verify_zero_stellarator_leakage,
)
from itg_surrogate.registry.specs import DataSplitConfig


class StellaratorGroupedSplitTest(unittest.TestCase):
    def _make_synthetic_metadata(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds synthetic 5-class stellarator metadata including shared QUASR IDs."""
        eq_files: list[bytes] = []
        quasr_ids: list[int] = []
        eq_class: list[int] = []

        # Classes 0, 1, 2: non-QUASR (QUASR_ID = -1), 20 equilibria x 4 radial tubes
        for c_id in (0, 1, 2):
            for eq_i in range(20):
                fname = f"class{c_id}_eq_{eq_i:04d}.h5".encode()
                for _ in range(4):
                    eq_files.append(fname)
                    quasr_ids.append(-1)
                    eq_class.append(c_id)

        # Classes 3 (vacuum) & 4 (finite-beta) sharing the SAME 20 QUASR_IDs!
        for q_idx in range(20):
            qid = 100_000 + q_idx
            vac_file = f"quasr_vac_{q_idx:04d}.h5".encode()
            for _ in range(3):
                eq_files.append(vac_file)
                quasr_ids.append(qid)
                eq_class.append(3)
            for beta_i in range(2):
                fb_file = f"quasr_fb_{q_idx:04d}_b{beta_i}.h5".encode()
                for _ in range(3):
                    eq_files.append(fb_file)
                    quasr_ids.append(qid)
                    eq_class.append(4)

        return (
            np.array(eq_files, dtype=object),
            np.array(quasr_ids, dtype=np.int64),
            np.array(eq_class, dtype=np.int64),
        )

    def test_quasr_vacuum_and_finite_beta_stay_in_same_split(self) -> None:
        eq_files, quasr_ids, eq_class = self._make_synthetic_metadata()
        cfg = DataSplitConfig(split_seed=42, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        split = make_stellarator_grouped_split(eq_files, quasr_ids, eq_class, cfg)

        # Partition covers every row exactly once
        all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
        self.assertEqual(len(all_idx), len(eq_files))
        self.assertEqual(len(np.unique(all_idx)), len(eq_files))

        # Total unique canonical groups = 3 * 20 + 20 = 80 -> 64 / 8 / 8
        self.assertEqual(split.n_train_groups, 64)
        self.assertEqual(split.n_val_groups, 8)
        self.assertEqual(split.n_test_groups, 8)

        # Both Class 3 (QUASR vacuum) and Class 4 (QUASR finite-beta) appear in test
        self.assertTrue(np.any(eq_class[split.test_idx] == 3))
        self.assertTrue(np.any(eq_class[split.test_idx] == 4))

        # Explicitly verify zero equilibrium_file or QUASR_ID overlap
        verify_zero_stellarator_leakage(split, eq_files, quasr_ids)

    def test_leakage_verifier_rejects_naive_equilibrium_only_quasr_split(
        self,
    ) -> None:
        eq_files, quasr_ids, _ = self._make_synthetic_metadata()
        groups = build_canonical_stellarator_groups(eq_files, quasr_ids)

        # Deliberately put Class 3 vacuum rows of qid=100_000 in train and its
        # Class 4 finite-beta rows (different equilibrium_file, SAME QUASR_ID!) in test.
        vac_rows = np.where(eq_files == b"quasr_vac_0000.h5")[0]
        fb_rows = np.where(eq_files == b"quasr_fb_0000_b0.h5")[0]
        other_rows = np.where(quasr_ids == -1)[0][:10]

        leaking_split = StellaratorSplitIndices(
            train_idx=vac_rows,
            val_idx=other_rows,
            test_idx=fb_rows,
            canonical_group_ids=groups,
            n_train_groups=1,
            n_val_groups=3,
            n_test_groups=1,
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            verify_zero_stellarator_leakage(leaking_split, eq_files, quasr_ids)

    def test_end_to_end_hdf5_loader_normalizes_strictly_on_train(self) -> None:
        eq_files, quasr_ids, eq_class = self._make_synthetic_metadata()
        n_rows = len(eq_files)
        rng = np.random.RandomState(0)
        raw_tensor = rng.normal(size=(n_rows, 96, 7)).astype(np.float64)
        raw_tensor[:, :, 0] = np.abs(raw_tensor[:, :, 0]) + 1.0  # B > 0
        raw_tensor[:, :, 4] = np.abs(raw_tensor[:, :, 4]) + 0.5  # gds2 > 0
        raw_tensor[:, :, 6] = np.abs(raw_tensor[:, :, 6]) + 0.5  # gds22 > 0

        # Inject huge outlier in test-only candidates to confirm train mean/std
        # equals exact statistics on split.train_idx.
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset("raw_feature_tensor", data=raw_tensor)
                f.create_dataset("equilibrium_files", data=np.asarray(eq_files, dtype="S32"))
                f.create_dataset("QUASR_IDs", data=quasr_ids)
                f.create_dataset("equilibrium_class", data=eq_class)
                f.create_dataset(
                    "equilibrium_class_descriptions",
                    data=np.array([b"Random", b"Circle", b"Torsion", b"QUASR vac", b"QUASR fb"]),
                )
                vg = f.create_group("varied_gradient_simulations")
                vg.create_dataset("a_over_LT", data=rng.uniform(0.5, 4.0, size=n_rows))
                vg.create_dataset("a_over_Ln", data=rng.uniform(0.0, 3.0, size=n_rows))
                vg.create_dataset("Q_avgs", data=rng.uniform(0.05, 25.0, size=n_rows))
                fg = f.create_group("fixed_gradient_simulations")
                fg.create_dataset("Q_avgs", data=rng.uniform(0.1, 20.0, size=n_rows))

            ds = load_stellarator_h5(tmp.name)
            tr_norm = ds.z_norm[ds.split.train_idx]
            np.testing.assert_allclose(tr_norm.mean(axis=(0, 1)), np.zeros(7), atol=1e-5)
            np.testing.assert_allclose(tr_norm.std(axis=(0, 1)), np.ones(7), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
