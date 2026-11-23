"""Unit tests for Z_96 shift invariance, D_96 parity, and GSPMD training."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from itg_surrogate.data.dataset import PreparedStellaratorDataset
from itg_surrogate.data.splits import make_stellarator_grouped_split
from itg_surrogate.models.cnn import StellaratorPeriodicCNN
from itg_surrogate.models.symmetry import D96ParityTransform
from itg_surrogate.registry import LEADING_ARCH
from itg_surrogate.registry.specs import (
    ArchConfig,
    AugmentationConfig,
    DataSplitConfig,
    ExperimentConfig,
    OptimizerConfig,
)
from itg_surrogate.training.evaluation import eval_batched_single
from itg_surrogate.training.sharding import configure_jax_float32
from itg_surrogate.training.trainer import train_experiment, wd_kernel_mask


class SymmetryAndCNNTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        configure_jax_float32()

    def test_exact_32_coset_z96_shift_invariance(self) -> None:
        model = StellaratorPeriodicCNN(config=LEADING_ARCH)
        rng = jax.random.PRNGKey(123)
        z_in = jax.random.normal(rng, (4, 96, 7), dtype=jnp.float32)
        s_in = jax.random.normal(jax.random.PRNGKey(456), (4, 2), dtype=jnp.float32)

        variables = model.init({"params": rng, "droppath": rng}, z_in, s_in, train=False)
        params = variables["params"]

        p_orig = eval_batched_single(model, params, {}, z_in, s_in, chunk_size=4, num_shifts=32)
        # Shift by an arbitrary element 57 in Z_96; 32-coset average must match!
        p_shifted = eval_batched_single(
            model,
            params,
            {},
            jnp.roll(z_in, shift=57, axis=1),
            s_in,
            chunk_size=4,
            num_shifts=32,
        )
        np.testing.assert_allclose(np.asarray(p_orig), np.asarray(p_shifted), rtol=1e-5, atol=1e-5)

    def test_d96_parity_transform_is_exact_involution_in_normalized_space(
        self,
    ) -> None:
        channel_names = (
            "B",
            "gbdrift",
            "cvdrift",
            "gbdrift0",
            "gds2",
            "gds21",
            "gds22",
        )
        rng = np.random.RandomState(7)
        z_raw = rng.normal(loc=1.3, scale=2.1, size=(6, 96, 7)).astype(np.float32)
        z_mean = z_raw.mean(axis=(0, 1), keepdims=True)
        z_std = z_raw.std(axis=(0, 1), keepdims=True) + 1e-8
        z_norm = (z_raw - z_mean) / z_std

        d96 = D96ParityTransform.from_channel_stats(96, channel_names, z_mean, z_std)

        # Involution check: T(T(z_norm)) == z_norm
        refl_once = d96.reflect_numpy(z_norm)
        refl_twice = d96.reflect_numpy(refl_once)
        np.testing.assert_allclose(refl_twice, z_norm, rtol=1e-5, atol=1e-5)

        # Physical equivalence check: reflecting in raw space then normalizing == reflect_numpy
        rev_idx = (-np.arange(96)) % 96
        raw_signs = np.array([1, 1, 1, -1, 1, -1, 1], dtype=np.float32)[None, None, :]
        expected_norm = (z_raw[:, rev_idx, :] * raw_signs - z_mean) / z_std
        np.testing.assert_allclose(refl_once, expected_norm, rtol=1e-5, atol=1e-5)

    def test_wd_kernel_mask_excludes_layernorm_scales_and_biases(self) -> None:
        model = StellaratorPeriodicCNN(config=LEADING_ARCH)
        variables = model.init(
            {"params": jax.random.PRNGKey(0), "droppath": jax.random.PRNGKey(1)},
            jnp.ones((1, 96, 7), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.float32),
            train=True,
        )
        mask = wd_kernel_mask(variables["params"])
        self.assertTrue(mask["conv_0"]["kernel"])
        self.assertFalse(mask["conv_0"]["bias"])
        self.assertFalse(mask["norm_conv_0"]["scale"])
        self.assertFalse(mask["norm_conv_0"]["bias"])
        self.assertTrue(mask["head"]["kernel"])
        self.assertFalse(mask["head"]["bias"])

    def test_sharded_train_and_eval_smoke_run(self) -> None:
        rng = np.random.RandomState(11)
        n_groups = 20
        rows_per_grp = 3
        n_rows = n_groups * rows_per_grp
        eq_files = np.array(
            [f"eq_{g:03d}.h5" for g in range(n_groups) for _ in range(rows_per_grp)],
            dtype=object,
        )
        quasr_ids = np.full(n_rows, -1, dtype=np.int64)
        eq_class = np.zeros(n_rows, dtype=np.int64)
        split_cfg = DataSplitConfig(split_seed=0)
        split = make_stellarator_grouped_split(eq_files, quasr_ids, eq_class, split_cfg)

        z_norm = rng.normal(size=(n_rows, 96, 7)).astype(np.float32)
        s_norm = rng.normal(size=(n_rows, 2)).astype(np.float32)
        targets = (0.8 * s_norm[:, 0] - 0.3 * s_norm[:, 1] + 1.5).astype(np.float32)
        q_varied = np.expm1(np.maximum(targets, 0.05)).astype(np.float32)

        ds = PreparedStellaratorDataset(
            z_norm=z_norm,
            s_norm=s_norm,
            z_raw=z_norm,
            scalars_raw=s_norm,
            raw_feature_tensor=z_norm,
            targets=targets,
            q_varied=q_varied,
            q_fixed=q_varied,
            unstable_mask=q_varied > 0.1,
            equilibrium_class=eq_class,
            equilibrium_descriptions=("Random boundary Fourier modes",),
            channel_names=(
                "B",
                "gbdrift",
                "cvdrift",
                "gbdrift0",
                "gds2",
                "gds21",
                "gds22",
            ),
            z_mean=np.zeros((1, 1, 7), dtype=np.float32),
            z_std=np.ones((1, 1, 7), dtype=np.float32),
            s_mean=np.zeros((1, 2), dtype=np.float32),
            s_std=np.ones((1, 2), dtype=np.float32),
            split=split,
            config=split_cfg,
        )

        exp = ExperimentConfig(
            name="smoke_resfilm",
            description="Fast smoke test of GSPMD training + D96 + DropPath.",
            arch=ArchConfig(
                conv_channels=(8, 8, 8, 8, 8),
                kernel_sizes=(5, 5, 3, 3, 3),
                fc_dims=(8, 8),
                activation="silu",
                norm_type="layernorm",
                use_residual=True,
                use_film=True,
                use_multistat_pool=True,
                drop_path_rate=0.10,
                film_hidden_dim=8,
            ),
            optimizer=OptimizerConfig(
                num_epochs=2,
                batch_size=16,
                weight_decay=1e-2,
                mask_kernels_only=True,
            ),
            augmentation=AugmentationConfig(
                use_z96_shift=True,
                use_d96_parity=True,
                input_noise_std=0.01,
                eval_num_shifts=4,
                eval_d96_reflection=True,
                eval_chunk_size=16,
            ),
            num_seeds=4,
            base_seed=99,
        )

        result = train_experiment(exp, ds, verbose=False)
        self.assertEqual(result.test_predictions.shape, (4, len(split.test_idx)))
        self.assertTrue(np.all(np.isfinite(result.test_predictions)))


if __name__ == "__main__":
    unittest.main()
