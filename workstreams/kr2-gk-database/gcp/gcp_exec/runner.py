"""CLI entrypoint (`gcp-gx-babysitter`) wiring the GCP backends into `QueueDriver` + `SimulationWatchdog`."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from hpc_core import (
    AuroraSteerClient,
    DriverAlreadyRunningError,
    SchedulerBackend,
    StateStore,
    TelemetryEmitter,
    acquire_single_instance_lock,
)
from hpc_core.state import DEFAULT_LOCK_STALE_AFTER_S, StateCorruptError, StateSchemaError
from tacc_exec.config import ConfigError as ExecutorConfigError
from tacc_exec.config import load_config
from tacc_exec.driver import QueueDriver
from tacc_exec.watchdog import configure_cli_logging

from gcp_exec.batch_backend import CloudBatchBackend, validate_batch_network_config
from gcp_exec.gke_backend import GKEComputeClassBackend

_TRUTHY = ("1", "true", "yes", "on")
_GKE_MODES = ("gcp_k8s", "gke", "gke_compute_class")
_BATCH_MODES = ("gcp_batch", "batch")


logger = logging.getLogger(__name__)


class ConfigError(ExecutorConfigError):
    """Invalid or unsafe babysitter configuration (the process exits before any mutation)."""


def _require(cfg: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ConfigError(f"config is missing required key {' or '.join(keys)!r}")


def _bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    raise ConfigError(f"config key {key!r} must be a JSON boolean (got {value!r})")


def validate_config(cfg: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Fail closed on configurations that could burn GPU money or poison the dataset."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a JSON object")
    environment = str(cfg.get("environment", "prod")).strip().lower()
    backend = str(cfg.get("backend", "gcp_batch")).strip().lower()
    if backend not in _GKE_MODES + _BATCH_MODES:
        raise ConfigError(f"unknown backend {backend!r}")
    _require(cfg, "gcs_bucket")
    _require(cfg, "gx_container_image")
    if not dry_run:
        _require(cfg, "aurora_steer_url")
    if environment == "prod":
        if (
            _bool(cfg, "cpu_only", False)
            or os.environ.get("GCP_BATCH_CPU_ONLY", "").strip().lower() in _TRUTHY
        ):
            raise ConfigError(
                "cpu_only/GCP_BATCH_CPU_ONLY is refused in environment=prod: GX is CUDA-only, so "
                "CPU-only jobs can only crash or produce non-physical output"
            )
        if os.environ.get("GX_SIMULATE_RUN", "").strip().lower() in _TRUTHY:
            raise ConfigError("GX_SIMULATE_RUN is refused in environment=prod")
        image = str(cfg["gx_container_image"])
        last_segment = image.rsplit("/", 1)[-1]
        if "@sha256:" not in image and (
            ":" not in last_segment or last_segment.endswith(":latest")
        ):
            raise ConfigError(
                "gx_container_image must be pinned in prod (immutable tag or @sha256 digest), "
                f"not {image!r}: the dataset records which GX binary produced each run"
            )
    if backend in _BATCH_MODES:
        _require(cfg, "gcp_project_id", "project_id")
        _require(cfg, "worker_service_account_email")
        no_ext_ip = _bool(cfg, "no_external_ip", True)
        cand = cfg.get("candidate_regions")
        if cand is not None and (
            not isinstance(cand, list)
            or not cand
            or not all(isinstance(r, str) and r.strip() for r in cand)
        ):
            raise ConfigError(
                "config key 'candidate_regions' must be a non-empty list of region strings"
            )
        region = str(cfg.get("gcp_region") or cfg.get("region", "us-central1")).strip()
        try:
            validate_batch_network_config(
                network=cfg.get("network"),
                subnetwork=cfg.get("subnetwork"),
                no_external_ip=no_ext_ip,
                region=region,
                candidate_regions=[r.strip() for r in cand] if cand else None,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    for key in ("max_run_duration_s", "max_pending_s"):
        if key in cfg and not (isinstance(cfg[key], (int, float)) and cfg[key] > 0):
            raise ConfigError(f"config key {key!r} must be a positive number")
    return cfg


def build_backend(cfg: dict[str, Any], state_dir: Path) -> tuple[SchedulerBackend, str]:
    """Construct the configured backend; raises ConfigError/ValueError on unsafe settings."""
    backend_mode = str(cfg.get("backend", "gcp_batch")).strip().lower()
    environment = str(cfg.get("environment", "prod")).strip().lower()
    common: dict[str, Any] = {
        "gcs_bucket": str(cfg["gcs_bucket"]),
        "gx_container_image": str(cfg["gx_container_image"]),
        "gcs_mount_path": str(cfg.get("gcs_mount_path", "/mnt/gk_bucket")),
        "cpu_only": _bool(cfg, "cpu_only", False),
        "max_run_duration_s": int(cfg.get("max_run_duration_s", 28800)),
        "environment": environment,
        "allow_on_demand": _bool(cfg, "allow_on_demand", False),
        "max_pending_s": float(cfg.get("max_pending_s", 1800.0)),
        "state_dir": state_dir,
    }
    if backend_mode in _GKE_MODES:
        gke = GKEComputeClassBackend(
            namespace=str(cfg.get("k8s_namespace", "default")),
            compute_class_default=str(cfg.get("compute_class_default", "gx-gpu-obtainable")),
            compute_class_medium_vram=str(
                cfg.get("compute_class_medium_vram", "gx-gpu-l4-or-a100")
            ),
            compute_class_highmem=str(cfg.get("compute_class_highmem", "gx-gpu-highmem")),
            compute_class_cpu=str(cfg.get("compute_class_cpu", "gx-cpu")),
            ksa_name=str(cfg.get("ksa_name", "gx-runner-ksa")),
            k8s_api_url=str(cfg.get("k8s_api_url", "https://kubernetes.default.svc")),
            k8s_auth=str(cfg.get("k8s_auth", "in_cluster")),
            k8s_ca_cert_path=cfg.get("k8s_ca_cert_path"),
            gcs_mount_options=cfg.get("gcs_mount_options_k8s"),
            ttl_seconds_after_finished=int(cfg.get("ttl_seconds_after_finished", 7 * 24 * 3600)),
            **common,
        )
        return gke, "gcp_k8s"
    batch = CloudBatchBackend(
        project_id=str(cfg.get("gcp_project_id") or cfg["project_id"]),
        region=str(cfg.get("gcp_region") or cfg.get("region", "us-central1")),
        worker_service_account_email=cfg.get("worker_service_account_email"),
        spot_provisioning=_bool(cfg, "spot_provisioning", True),
        candidate_regions=cfg.get("candidate_regions"),
        allow_a100_fallback=_bool(cfg, "allow_a100_fallback", False),
        gcs_mount_options=cfg.get("gcs_mount_options_batch"),
        network=cfg.get("network"),
        subnetwork=cfg.get("subnetwork"),
        no_external_ip=_bool(cfg, "no_external_ip", True),
        **common,
    )
    return batch, "gcp_batch"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `gcp-gx-babysitter` (one Cloud Run job execution per tick)."""
    parser = argparse.ArgumentParser(
        description="Serverless babysitter for containerized GX runs on GCP Cloud Batch / GKE."
    )
    parser.add_argument("--config", required=True, help="Path to config.batch.json")
    parser.add_argument(
        "--once", action="store_true", help="Run one poll + submit + harvest tick and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Observe jobs without mutating anything"
    )
    args = parser.parse_args(argv)
    configure_cli_logging()

    try:
        # Shared executor validation (absolute paths, allowlist == patcher maps, environment),
        # then the GCP-specific spend/provenance gates.
        cfg = validate_config(load_config(args.config), dry_run=args.dry_run)
        state_dir = Path(cfg.get("state_dir", "./state")).resolve()
        backend, default_backend_name = build_backend(cfg, state_dir)
    except ValueError as exc:  # ConfigError and backend constructor refusals
        logger.error("gcp-gx-babysitter: refusing to start: %s", exc)
        return 2
    lock_path = state_dir / "gcp_batch_driver.lock"
    stale_after = float(cfg.get("lock_stale_after_seconds", DEFAULT_LOCK_STALE_AFTER_S))

    try:
        # flock + heartbeated lease file: also excludes a concurrent Cloud Run execution that
        # mounts the same bucket from another instance.
        with acquire_single_instance_lock(lock_path, stale_after_s=stale_after) as lock:
            store = StateStore(
                state_dir / "state.json", state_dir / "audit.jsonl", dry_run=args.dry_run
            )
            client = None if args.dry_run else AuroraSteerClient(base_url=cfg["aurora_steer_url"])
            telemetry = TelemetryEmitter(
                state_dir / "outbox.jsonl",
                client=client,
                default_workstream=str(cfg.get("workstream", "kr2-gk-database")),
                default_campaign_id=str(cfg.get("campaign_id", "pilot_50")),
                default_environment=str(cfg.get("environment", "prod")),
                default_backend=default_backend_name,
            )
            driver = QueueDriver(
                backend=backend,
                state_store=store,
                steer_client=client,
                telemetry=telemetry,
                config=cfg,
                dry_run=args.dry_run,
                heartbeat=lock.refresh,
            )
            if args.once:
                driver.tick()
            else:
                driver.run_forever()
    except DriverAlreadyRunningError as exc:
        logger.info("Another babysitter execution is active; skipping this tick (%s)", exc)
        return 0
    except (StateCorruptError, StateSchemaError) as exc:
        logger.error("Refusing to run: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
