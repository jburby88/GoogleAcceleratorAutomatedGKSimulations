"""Tests for the `gcp-gx-babysitter` entrypoint (QA-03): config validation, prod refusals,
backend wiring, and an offline dry-run tick."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gcp_exec.batch_backend import CloudBatchBackend
from gcp_exec.gke_backend import GKEComputeClassBackend
from gcp_exec.runner import ConfigError, build_backend, main, validate_config

IMAGE = "us-central1-docker.pkg.dev/p/aurora-platform/gx-runner@sha256:" + "b" * 64


def _cfg(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "backend": "gcp_batch",
        "environment": "prod",
        "aurora_steer_url": "https://aurora-steer.example.run.app",
        "gcp_project_id": "ut-test",
        "gcp_region": "us-central1",
        "gcs_bucket": "ut-test-gk-dataset",
        "gx_container_image": IMAGE,
        "worker_service_account_email": "gx-batch-worker-sa@ut-test.iam.gserviceaccount.com",
        "queue_dir": str(tmp_path / "queue"),
        "state_dir": str(tmp_path / "state"),
    }
    cfg.update(kw)
    return cfg


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GX_SIMULATE_RUN", "GCP_BATCH_CPU_ONLY", "GCP_BATCH_CANDIDATE_REGIONS"):
        monkeypatch.delenv(var, raising=False)


def test_valid_prod_config_builds_idempotent_batch_backend(tmp_path: Path) -> None:
    cfg = validate_config(_cfg(tmp_path))
    backend, name = build_backend(cfg, tmp_path / "state")
    assert isinstance(backend, CloudBatchBackend) and name == "gcp_batch"
    assert backend.capabilities.idempotent_submit is True
    assert backend.allow_on_demand is False


@pytest.mark.parametrize(
    "override",
    [
        {"cpu_only": True},
        {"cpu_only": "false"},  # strings are not booleans
        {"gx_container_image": "us-central1-docker.pkg.dev/p/r/gx-runner:latest"},
        {"gx_container_image": "us-central1-docker.pkg.dev/p/r/gx-runner"},
        {"worker_service_account_email": ""},
        {"backend": "slurm"},
        {"max_pending_s": -1},
        {"aurora_steer_url": ""},
        {"network": "projects/ut-test/global/networks/ut-test-vpc"},  # missing subnetwork
        {"subnetwork": "projects/ut-test/regions/us-central1/subnetworks/ut-test-subnet"},
        {
            "network": "default",
            "subnetwork": "projects/ut-test/regions/us-central1/subnetworks/ut-test-subnet",
        },
        {
            "network": "projects/ut-test/global/networks/ut-test-vpc",
            "subnetwork": "projects/ut-test/regions/us-central1/subnetworks/ut-test-subnet",
            "no_external_ip": "true",
        },
        {
            "network": "projects/ut-test/global/networks/ut-test-vpc",
            "subnetwork": "projects/ut-test/regions/us-central1/subnetworks/ut-test-subnet",
            "candidate_regions": ["us-central1", "us-east4"],
        },
        {"candidate_regions": []},
    ],
)
def test_unsafe_configs_are_refused(tmp_path: Path, override: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        validate_config(_cfg(tmp_path, **override))


def test_prod_refuses_simulation_and_cpu_only_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GX_SIMULATE_RUN", "1")
    with pytest.raises(ConfigError, match="GX_SIMULATE_RUN"):
        validate_config(_cfg(tmp_path))
    monkeypatch.delenv("GX_SIMULATE_RUN")
    monkeypatch.setenv("GCP_BATCH_CPU_ONLY", "1")
    with pytest.raises(ConfigError, match="cpu_only"):
        validate_config(_cfg(tmp_path))
    # Outside prod a CPU-only plumbing test is allowed.
    validate_config(_cfg(tmp_path, environment="sim", cpu_only=True))


def test_gke_backend_wiring(tmp_path: Path) -> None:
    cfg = validate_config(
        _cfg(
            tmp_path,
            backend="gcp_k8s",
            allow_on_demand=True,
            k8s_api_url="https://10.0.0.1",
            k8s_auth="google",
            compute_class_medium_vram="mid-class",
        )
    )
    backend, name = build_backend(cfg, tmp_path / "state")
    assert isinstance(backend, GKEComputeClassBackend) and name == "gcp_k8s"
    assert backend.compute_class_medium_vram == "mid-class"
    assert backend.k8s_auth == "google" and backend.allow_on_demand is True


def test_main_dry_run_tick_is_offline_and_bad_config_exits_2(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_cfg(tmp_path)), encoding="utf-8")
    assert main(["--config", str(good), "--once", "--dry-run"]) == 0
    assert (tmp_path / "queue" / "pending").is_dir()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_cfg(tmp_path, cpu_only=True)), encoding="utf-8")
    assert main(["--config", str(bad), "--once", "--dry-run"]) == 2


def test_example_config_is_valid_and_mirrors_the_executor_allowlist(tmp_path: Path) -> None:
    from tacc_exec.config import load_config

    gcp_dir = Path(__file__).resolve().parents[1]
    cfg = validate_config(load_config(gcp_dir / "config.batch.example.json"))
    hpc_cfg = json.loads(
        (gcp_dir.parent / "hpc" / "config.example.json").read_text(encoding="utf-8")
    )
    assert cfg["allowed_parameters"] == hpc_cfg["allowed_parameters"]
    for key in ("dt_collapse_threshold", "transient_discard_fraction", "max_flux_drift"):
        assert cfg[key] == hpc_cfg[key]
    assert cfg["allow_on_demand"] is False and cfg["cpu_only"] is False
    assert cfg["candidate_regions"] == ["us-central1"]
    assert cfg["network"] == "projects/ut-austin-bsitter/global/networks/ut-austin-bsitter-vpc"
    assert (
        cfg["subnetwork"]
        == "projects/ut-austin-bsitter/regions/us-central1/subnetworks/ut-austin-bsitter-subnet"
    )
    assert cfg["no_external_ip"] is True
    backend, _ = build_backend(
        {**cfg, "gx_container_image": IMAGE},
        tmp_path / "state",
    )
    assert isinstance(backend, CloudBatchBackend)
    assert backend.network == cfg["network"]
    assert backend.subnetwork == cfg["subnetwork"]
    assert backend.no_external_ip is True
