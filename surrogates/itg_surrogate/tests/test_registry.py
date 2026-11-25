"""Unit tests for the model registry, leading baseline, and ablation sweeps."""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from itg_surrogate import cli
from itg_surrogate.registry import (
    CANONICAL_OOD_SPLIT,
    EXPERIMENT_REGISTRY,
    LEADING_EXPERIMENT,
    get_experiment,
    get_sweep,
    list_experiments,
    list_sweeps,
)
from itg_surrogate.registry.specs import (
    ArchConfig,
    AugmentationConfig,
    DataSplitConfig,
    ExperimentConfig,
    OptimizerConfig,
)


class RegistryTest(unittest.TestCase):
    def test_leading_experiment_is_registered_and_benchmarked(self) -> None:
        self.assertEqual(get_experiment("leading"), LEADING_EXPERIMENT)
        self.assertEqual(get_experiment(LEADING_EXPERIMENT.name), LEADING_EXPERIMENT)
        self.assertIsNotNone(LEADING_EXPERIMENT.reference_benchmark)
        assert LEADING_EXPERIMENT.reference_benchmark is not None
        self.assertGreater(LEADING_EXPERIMENT.reference_benchmark.ens8_test_r2, 0.99)
        self.assertEqual(LEADING_EXPERIMENT.data_split, CANONICAL_OOD_SPLIT)
        self.assertTrue(LEADING_EXPERIMENT.optimizer.mask_kernels_only)
        self.assertEqual(LEADING_EXPERIMENT.optimizer.weight_decay, 1e-2)

    def test_all_sweeps_and_experiments_are_consistent(self) -> None:
        sweep_names = list_sweeps()
        self.assertIn("architecture_ladder", sweep_names)
        self.assertIn("ood_regularization", sweep_names)
        self.assertIn("weight_decay_scan", sweep_names)
        self.assertIn("normalization_diagnostic", sweep_names)

        for s_name in sweep_names:
            exps = get_sweep(s_name)
            self.assertGreaterEqual(len(exps), 4)
            seen_in_sweep: set[str] = set()
            for exp in exps:
                self.assertNotIn(exp.name, seen_in_sweep)
                seen_in_sweep.add(exp.name)
                self.assertIn(exp.name, EXPERIMENT_REGISTRY)
                self.assertEqual(exp.data_split, CANONICAL_OOD_SPLIT)
                serialized = json.dumps(exp.to_dict())
                self.assertIn(exp.name, serialized)

        self.assertEqual(len(list_experiments()), len(EXPERIMENT_REGISTRY))

    def test_spec_validation_catches_invalid_configs(self) -> None:
        with self.assertRaises(ValueError):
            DataSplitConfig(train_frac=0.8, val_frac=0.2, test_frac=0.1)
        with self.assertRaises(ValueError):
            ArchConfig(conv_channels=(32, 32, 32, 32))
        with self.assertRaises(ValueError):
            ArchConfig(use_residual=False, drop_path_rate=0.1)
        with self.assertRaises(ValueError):
            OptimizerConfig(num_epochs=0)
        with self.assertRaises(ValueError):
            OptimizerConfig(batch_size=0)
        with self.assertRaises(ValueError):
            AugmentationConfig(eval_num_shifts=0)
        with self.assertRaises(ValueError):
            ExperimentConfig(name="   ", description="empty name")
        with self.assertRaises(ValueError):
            ExperimentConfig(name="zero_seeds", description="invalid", num_seeds=0)

    def test_cli_list_and_show_commands(self) -> None:
        buf_list = io.StringIO()
        with mock.patch("sys.stdout", buf_list):
            cli.main(["list"])
        out_list = buf_list.getvalue()
        self.assertIn(LEADING_EXPERIMENT.name, out_list)
        self.assertIn("ood_regularization", out_list)

        buf_show = io.StringIO()
        with mock.patch("sys.stdout", buf_show):
            cli.main(["show", "leading"])
        parsed = json.loads(buf_show.getvalue())
        self.assertEqual(parsed["name"], LEADING_EXPERIMENT.name)


if __name__ == "__main__":
    unittest.main()
