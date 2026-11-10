"""Unit tests for GCP Cloud Batch / GKE job specs, status mapping and VRAM routing.

Dockerfile and ComputeClass checks live in test_container_and_k8s_specs.py; Terraform checks
belong to platform/infra (fix-infra), not to this package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tacc_exec.patchers.gx_toml import read_gx_parameters

from gcp_exec.batch_backend import CloudBatchBackend, map_batch_status_to_canonical
from gcp_exec.gke_backend import GKEComputeClassBackend, map_k8s_job_to_canonical
from gcp_exec.gpu_selector import (
    ObtainabilityExhaustedError,
    estimate_gx_vram_gb,
    select_gpu_tier,
    select_obtainable_target,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER_SA = "gx-batch-worker-sa@ut-test.iam.gserviceaccount.com"


def test_batch_status_maps_timeout_preempted_and_spot_oom_priority() -> None:
    # 1. Exit code 50005 (maxRunDuration exceeded) -> TIMEOUT
    timeout_st = map_batch_status_to_canonical(
        {
            "name": "projects/p/locations/us-central1/jobs/j-timeout",
            "labels": {"gpu_tier": "l4"},
            "status": {
                "state": "FAILED",
                "runDuration": "28800s",
                "statusEvents": [
                    {
                        "description": "Task failed because maxRunDuration was exceeded",
                        "taskExecution": {"exitCode": 50005},
                    }
                ],
            },
        }
    )
    assert timeout_st.state == "TIMEOUT"
    assert timeout_st.elapsed_seconds == pytest.approx(28800.0)

    # 2. Exit code 50002 (Spot VM host loss / reclamation) -> PREEMPTED
    preempt_50002 = map_batch_status_to_canonical(
        {
            "name": "projects/p/locations/us-central1/jobs/j-preempt",
            "labels": {"gpu_tier": "l4"},
            "status": {
                "state": "FAILED",
                "statusEvents": [
                    {
                        "description": "VM instance terminated by Compute Engine",
                        "taskExecution": {"exitCode": 50002},
                    }
                ],
            },
        }
    )
    assert preempt_50002.state == "PREEMPTED"

    # 3. OOM on a Spot VM (events mention "spot" AND exitCode 137 / "oom") -> OUT_OF_MEMORY (never PREEMPTED!)
    spot_oom = map_batch_status_to_canonical(
        {
            "name": "projects/p/locations/us-central1/jobs/j-spot-oom",
            "labels": {"gpu_tier": "l4"},
            "status": {
                "state": "FAILED",
                "statusEvents": [
                    {"description": "Scheduled on SPOT provisioning model"},
                    {
                        "description": "Container killed on spot instance",
                        "taskExecution": {"exitCode": 137},
                    },
                ],
            },
        }
    )
    assert spot_oom.state == "OUT_OF_MEMORY"


def test_build_job_spec_runs_gx_task_without_shell_and_bounded_waterfall(tmp_path: Path) -> None:
    backend = CloudBatchBackend(
        project_id="ut-test",
        region="us-central1",
        gcs_bucket="ut-test-gk-dataset",
        gcs_mount_path="/mnt/gk_bucket",
        gx_container_image="us-central1-docker.pkg.dev/ut-test/aurora-platform/gx-runner:v1",
        worker_service_account_email=WORKER_SA,
    )
    run_dir = tmp_path / "run_with'quote"
    run_dir.mkdir()

    _, gpu_spec, job_body = backend.build_job_spec(
        run_dir,
        parameters={"nx": 64, "ny": 32, "nz": 48},
    )
    assert gpu_spec.tier_name == "l4"
    runnable = job_body["taskGroups"][0]["taskSpec"]["runnables"][0]
    assert runnable["environment"]["variables"]["HDF5_USE_FILE_LOCKING"] == "FALSE"
    container = runnable["container"]
    # No inline bash: the in-job task runner owns staging, checkpoint sync and preemption.
    assert container["entrypoint"] == "gx-task"
    assert "/bin/bash" not in json.dumps(job_body)
    # The run directory is a separate argv element, so a quote cannot break out of a shell.
    assert "/mnt/gk_bucket/queue/running/run_with'quote" in container["commands"]

    # Verify CPU-only fallback mode (for GCP projects with 0 GPU quota) uses e2-standard-4 with 0 GPUs
    backend_cpu = CloudBatchBackend(
        project_id="ut-test",
        region="us-central1",
        gcs_bucket="ut-test-gk-dataset",
        gx_container_image="us-central1-docker.pkg.dev/ut-test/aurora-platform/gx-runner:v1",
        worker_service_account_email=WORKER_SA,
        cpu_only=True,
        environment="sim",
    )
    _, cpu_spec, cpu_body = backend_cpu.build_job_spec(run_dir)
    assert cpu_spec.tier_name == "cpu"
    assert cpu_spec.machine_type == "e2-standard-4"
    inst = cpu_body["allocationPolicy"]["instances"][0]
    assert inst["installGpuDrivers"] is False
    assert "accelerators" not in inst["policy"]

    # Bounded multi-region + multi-SKU waterfall: L4 SPOT -> T4 SPOT, STANDARD only on opt-in
    regions = ["us-central1", "us-south1"]
    small = {"nx": 64, "ny": 32, "nz": 48}
    steps = [
        select_obtainable_target(small, rotation_index=i, candidate_regions=regions)
        for i in range(4)
    ]
    assert [(r, s.tier_name, p) for r, s, p in steps] == [
        ("us-central1", "l4", "SPOT"),
        ("us-south1", "l4", "SPOT"),
        ("us-central1", "t4", "SPOT"),
        ("us-south1", "t4", "SPOT"),
    ]
    with pytest.raises(ObtainabilityExhaustedError):
        select_obtainable_target(small, rotation_index=4, candidate_regions=regions)
    r4, s4, p4 = select_obtainable_target(
        small, rotation_index=4, candidate_regions=regions, allow_on_demand=True
    )
    assert (r4, s4.tier_name, p4) == ("us-central1", "l4", "STANDARD")


def test_gke_compute_class_backend_waterfall_and_status(tmp_path: Path) -> None:
    """Verify GKEComputeClassBackend binds Jobs to cloud.google.com/compute-class."""
    run_dir = tmp_path / "gx_gke_001"
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text("[Dimensions]\nntheta = 48\n", encoding="utf-8")

    gke = GKEComputeClassBackend(
        gcs_bucket="ut-austin-gk-dataset",
        gx_container_image="us-central1-docker.pkg.dev/ut-proj/containers/gx-runner:latest",
    )
    job_name, cclass, manifest = gke.build_job_spec(
        run_dir, parameters={"nx": 64, "ny": 32, "nz": 48}
    )
    # Deterministic per submission (no attempt counter, GCP-02), valid RFC 1123 label.
    assert re.match(r"^gx-gx-gke-001-r0-[0-9a-f]{10}$", job_name)
    assert job_name == gke.build_job_spec(run_dir, parameters={"nx": 64, "ny": 32, "nz": 48})[0]
    assert cclass == "gx-gpu-obtainable"
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"]["cloud.google.com/compute-class"] == "gx-gpu-obtainable"
    assert pod_spec["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == "1"

    # Medium-VRAM (13.5 < VRAM <= 20 GB) switches to gx-gpu-l4-or-a100 (excludes 16 GB T4)
    _, med_class, _ = gke.build_job_spec(
        run_dir,
        parameters={"nx": 160, "ny": 80, "nz": 64, "nhermite": 12, "nlaguerre": 6},
    )
    assert med_class == "gx-gpu-l4-or-a100"

    # High-VRAM (> 20 GB) automatically switches to gx-gpu-highmem ComputeClass
    _, hi_class, hi_manifest = gke.build_job_spec(
        run_dir,
        parameters={"nx": 256, "ny": 128, "nz": 96, "nhermite": 16, "nlaguerre": 8},
    )
    assert hi_class == "gx-gpu-highmem"
    assert (
        hi_manifest["spec"]["template"]["spec"]["nodeSelector"]["cloud.google.com/compute-class"]
        == "gx-gpu-highmem"
    )

    # K8s status mapping (Complete, Pod-level OOMKilled 137, Pod DisruptionTarget Preempted)
    done_status = map_k8s_job_to_canonical(
        {
            "metadata": {"name": job_name, "labels": {"gpu_tier": "l4"}},
            "status": {"succeeded": 1},
        }
    )
    assert done_status.state == "COMPLETED"
    assert done_status.gpu_tier == "l4"

    oom_status = map_k8s_job_to_canonical(
        {
            "metadata": {"name": job_name, "labels": {"gpu_tier": "l4"}},
            "status": {
                "failed": 1,
                "conditions": [
                    {"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}
                ],
            },
        },
        pods_payload={
            "items": [
                {
                    "status": {
                        "containerStatuses": [
                            {"state": {"terminated": {"exitCode": 137, "reason": "OOMKilled"}}}
                        ]
                    }
                }
            ]
        },
    )
    assert oom_status.state == "OUT_OF_MEMORY"
    assert oom_status.exit_code == "137:0"

    preempt_status = map_k8s_job_to_canonical(
        {
            "metadata": {"name": job_name, "labels": {"gpu_tier": "l4"}},
            "status": {"failed": 1},
        },
        pods_payload={
            "items": [
                {
                    "status": {
                        "conditions": [
                            {
                                "type": "DisruptionTarget",
                                "status": "True",
                                "reason": "PreemptionByKubelet",
                            }
                        ],
                        "containerStatuses": [
                            {"state": {"terminated": {"exitCode": 137, "reason": "Error"}}}
                        ],
                    }
                }
            ]
        },
    )
    assert preempt_status.state == "PREEMPTED"


def test_read_gx_parameters_extracts_velocity_and_species_for_vram_routing(tmp_path: Path) -> None:
    gx_in = tmp_path / "gx.in"
    gx_in.write_text(
        "[Dimensions]\n"
        "nx = 256\n"
        "ny = 128\n"
        "ntheta = 128\n"
        "nhermite = 32\n"
        "nlaguerre = 16\n\n"
        "[[Species]]\n"
        "z = 1.0\n"
        "mass = 1.0\n\n"
        "[[Species]]\n"
        "z = -1.0\n"
        "mass = 0.000272\n\n"
        "[[Species]]\n"
        "z = 6.0\n"
        "mass = 12.0\n",
        encoding="utf-8",
    )
    params = read_gx_parameters(gx_in)
    assert params["nhermite"] == 32
    assert params["nlaguerre"] == 16
    assert params["nspecies"] == 3
    assert estimate_gx_vram_gb(params) > 20.0
    assert select_gpu_tier(params).tier_name == "a100"


def test_gx_diagnostics_extracts_metrics(tmp_path: Path) -> None:
    from gcp_exec.gx_diagnostics import parse_gx_log

    log_path = tmp_path / "gx.log"
    log_path.write_text(
        "run_name: Step     10: Time =    0.10000  dt = 1.000e-03   Phi**2 = 1.234e-05   Q_i = 1.234e-03   Q_e = 1.234e-03   Gam_i = 0.000e+00\n"
        "run_name: Step     20: Time =    0.20000  dt = 5.000e-04   Phi**2 = 1.500e-05   Q_i = 2.000e-03   Q_e = 1.500e-03   Gam_i = 5.000e-04\n",
        encoding="utf-8",
    )

    times, dts, fluxes, scalar_metrics, has_nan = parse_gx_log(log_path)

    assert times == [0.1, 0.2]
    assert dts == [0.001, 0.0005]
    assert fluxes["Q_i"] == 0.002
    assert fluxes["Q_e"] == 0.0015
    assert fluxes["Gam_i"] == 0.0005
    assert scalar_metrics["phi2_total"] == 0.000015
    assert has_nan == 0.0
