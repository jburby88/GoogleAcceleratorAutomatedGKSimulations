"""Tests for CLI `--output-json` serialization and GKE Cloud TPU YAML manifests."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import h5py
import numpy as np
import yaml

from itg_surrogate.cli import _sanitize_json_floats
from itg_surrogate.cli import main as cli_main

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
GKE_DIR = PACKAGE_DIR / "gke"

_HERMETIC_UNSET_KEYS = (
    "TPU_MACHINE_TYPE",
    "TPU_ACCELERATOR",
    "TPU_TOPOLOGY",
    "TPU_CHIPS",
    "PARALLEL_SWEEPS",
    "K8S_NAMESPACE",
    "KSA_NAME",
    "IMAGE_TAG",
)


class GkeManifestAndCliJsonTest(unittest.TestCase):
    def test_makefile_renders_valid_tpu_sweep_and_indexed_manifests(self) -> None:
        if not GKE_DIR.is_dir():
            self.skipTest("GKE directory not bundled in minimal runtime container")

        clean_env = {k: v for k, v in os.environ.items() if k not in _HERMETIC_UNSET_KEYS}

        # Syntax-check entrypoint_sweep.sh
        subprocess.run(
            ["bash", "-n", str(GKE_DIR / "entrypoint_sweep.sh")],
            check=True,
            env=clean_env,
        )

        # Render single-sweep Job manifest hermetically (ignoring any local gcp.env)
        single_out = subprocess.check_output(
            [
                "make",
                "-C",
                str(GKE_DIR),
                "--no-print-directory",
                "render",
                "ENV_FILE=/dev/null",
                "PROJECT_ID=test-proj",
                "GCS_BUCKET=test-bucket",
                "SWEEP=ood_regularization",
                "RUN_ID=20261105-010000",
            ],
            text=True,
            env=clean_env,
        )
        self.assertNotIn("${", single_out)
        doc = yaml.safe_load(single_out)
        self.assertEqual(doc["apiVersion"], "batch/v1")
        self.assertEqual(doc["kind"], "Job")
        pod_meta = doc["spec"]["template"]["metadata"]
        self.assertEqual(pod_meta["annotations"]["gke-gcsfuse/volumes"], "true")

        pod_spec = doc["spec"]["template"]["spec"]
        self.assertEqual(pod_spec["restartPolicy"], "Never")
        self.assertEqual(
            pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"],
            "tpu-v6e-slice",
        )
        self.assertEqual(
            pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-topology"],
            "2x4",
        )
        container = pod_spec["containers"][0]
        self.assertEqual(container["resources"]["limits"]["google.com/tpu"], "8")
        self.assertEqual(container["resources"]["requests"]["google.com/tpu"], "8")

        # Render 4-way Indexed Job manifest for all sweeps
        matrix_out = subprocess.check_output(
            [
                "make",
                "-C",
                str(GKE_DIR),
                "--no-print-directory",
                "render-all-sweeps",
                "ENV_FILE=/dev/null",
                "PROJECT_ID=test-proj",
                "GCS_BUCKET=test-bucket",
                "RUN_ID=20261105-010000",
            ],
            text=True,
            env=clean_env,
        )
        self.assertNotIn("${", matrix_out)
        matrix_doc = yaml.safe_load(matrix_out)
        self.assertEqual(matrix_doc["spec"]["completionMode"], "Indexed")
        self.assertEqual(matrix_doc["spec"]["completions"], 4)
        self.assertEqual(matrix_doc["spec"]["parallelism"], 4)
        self.assertEqual(matrix_doc["spec"]["backoffLimit"], 8)
        self.assertEqual(matrix_doc["spec"]["template"]["spec"]["restartPolicy"], "Never")

    def test_sanitize_json_floats_produces_strict_rfc8259_json(self) -> None:
        raw = {"ok": 0.99, "bad": float("nan"), "nested": [float("inf"), -1.5]}
        sanitized = _sanitize_json_floats(raw)
        encoded = json.dumps(sanitized, allow_nan=False)
        self.assertEqual(json.loads(encoded), {"ok": 0.99, "bad": None, "nested": [None, -1.5]})

    def test_cli_train_writes_structured_output_json(self) -> None:
        rng = np.random.RandomState(0)
        eq_files: list[bytes] = []
        quasr_ids: list[int] = []
        eq_class: list[int] = []
        for c_id in range(5):
            for eq_i in range(10):
                fname = f"cls{c_id}_eq{eq_i:02d}.h5".encode()
                qid = (500 + eq_i) if c_id in (3, 4) else -1
                for _ in range(2):
                    eq_files.append(fname)
                    quasr_ids.append(qid)
                    eq_class.append(c_id)

        n_rows = len(eq_files)
        raw_tensor = rng.normal(size=(n_rows, 96, 7)).astype(np.float64)
        raw_tensor[:, :, 0] = np.abs(raw_tensor[:, :, 0]) + 1.0
        raw_tensor[:, :, 4] = np.abs(raw_tensor[:, :, 4]) + 0.5
        raw_tensor[:, :, 6] = np.abs(raw_tensor[:, :, 6]) + 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = pathlib.Path(tmpdir) / "synthetic.h5"
            json_path = pathlib.Path(tmpdir) / "results" / "summary.json"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("raw_feature_tensor", data=raw_tensor)
                f.create_dataset("equilibrium_files", data=np.asarray(eq_files, dtype="S32"))
                f.create_dataset("QUASR_IDs", data=np.asarray(quasr_ids, dtype=np.int64))
                f.create_dataset("equilibrium_class", data=np.asarray(eq_class, dtype=np.int64))
                f.create_dataset(
                    "equilibrium_class_descriptions",
                    data=np.array([b"C0", b"C1", b"C2", b"C3", b"C4"]),
                )
                vg = f.create_group("varied_gradient_simulations")
                vg.create_dataset("a_over_LT", data=rng.uniform(0.8, 3.5, size=n_rows))
                vg.create_dataset("a_over_Ln", data=rng.uniform(0.1, 2.5, size=n_rows))
                vg.create_dataset("Q_avgs", data=rng.uniform(0.2, 15.0, size=n_rows))
                fg = f.create_group("fixed_gradient_simulations")
                fg.create_dataset("Q_avgs", data=rng.uniform(0.2, 15.0, size=n_rows))

            cli_main(
                [
                    "train",
                    "--h5-path",
                    str(h5_path),
                    "--experiment",
                    "table1_gelu_32",
                    "--epochs",
                    "1",
                    "--num-seeds",
                    "2",
                    "--output-json",
                    str(json_path),
                ]
            )

            self.assertTrue(json_path.is_file())
            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["mode"], "experiment")
            self.assertEqual(payload["target_name"], "table1_gelu_32")
            self.assertIn("result", payload)
            self.assertEqual(payload["result"]["experiment"]["name"], "table1_gelu_32")
            self.assertTrue(np.isfinite(payload["result"]["ens_test_r2"]))


if __name__ == "__main__":
    unittest.main()
