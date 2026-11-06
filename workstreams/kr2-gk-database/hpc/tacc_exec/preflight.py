"""Pre-campaign connectivity & configuration preflight check (`tacc-preflight`)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any

import requests
from hpc_core import AuroraSteerClient, TelemetryEmitter, validate_aurora_steer_url

from tacc_exec.config import ConfigError, load_config, validate_allowed_parameters

REQUIRED_ALLOWED_PARAMS = {
    "nx",
    "ny",
    "nz",
    "dt_max",
    "cfl_fraction",
    "hyperdiffusion_coeff",
}
SLURM_BINARIES = ("sbatch", "squeue", "sacct", "scancel")


def _under(path: Path, root: str | None) -> bool:
    if not root:
        return False
    try:
        path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _purge_risk(path: Path) -> bool:
    """True if `path` is on a purged scratch filesystem (TACC $SCRATCH: files unused ~10 days)."""
    return _under(path, os.environ.get("SCRATCH")) or "scratch" in {
        part.lower() for part in path.resolve().parts
    }


def collect_warnings(config: dict[str, Any]) -> list[str]:
    """Non-fatal operational risks an operator must see before a campaign (HPC-04, HPC-07)."""
    warnings: list[str] = []
    dataset_dir = config.get("dataset_output_dir")
    if not dataset_dir:
        warnings.append("dataset_output_dir is not set: completed runs will not be harvested.")
    elif _purge_risk(Path(dataset_dir)):
        warnings.append(
            f"dataset_output_dir {dataset_dir} is on a purged scratch filesystem; TACC deletes "
            "files not accessed for ~10 days. Use $WORK (or sync to GCS) for the dataset."
        )
    for key in ("state_dir", "queue_dir"):
        value = config.get(key)
        if value and _purge_risk(Path(value)):
            warnings.append(
                f"{key} {value} is on a purged scratch filesystem; state or queued inputs can be "
                "deleted while jobs wait. Prefer $WORK."
            )
    if str(config.get("environment", "prod")) != "prod":
        warnings.append(
            f"environment={config.get('environment')!r}: harvested runs are never dataset-eligible."
        )
    if str(config.get("backend", "tacc_slurm")) == "tacc_slurm":
        missing = [b for b in SLURM_BINARIES if shutil.which(b) is None]
        if missing:
            warnings.append(
                f"Slurm commands not on PATH: {missing} (cron has a minimal PATH; set PATH in the "
                "crontab or the wrapper)."
            )
    expected = os.environ.get("TACC_EXPECTED_LOGIN_HOST", "").strip()
    if expected:
        this_host = socket.gethostname().split(".")[0]
        if this_host != expected.split(".")[0]:
            warnings.append(
                f"This host is {this_host!r} but TACC_EXPECTED_LOGIN_HOST={expected!r}: install "
                "and inspect the crontab on the pinned login node only."
            )
    return warnings


def run_preflight(config: dict[str, Any], check_remote: bool = True) -> dict[str, Any]:
    """Validate config, queue folders, `/healthz`, and `environment='preflight'` telemetry."""
    from hpc_core.client import resolve_aurora_steer_url

    url = validate_aurora_steer_url(resolve_aurora_steer_url(config))
    allowed = config.get("allowed_parameters", {})
    missing = REQUIRED_ALLOWED_PARAMS - set(allowed.keys())
    if missing:
        raise ValueError(f"Config is missing required allowed_parameters: {sorted(missing)}")
    validate_allowed_parameters(allowed)

    queue_dir = Path(config.get("queue_dir", "./queue")).resolve()
    for sub in ("pending", "running", "completed", "needs_human"):
        (queue_dir / sub).mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "config_valid": True,
        "queue_dirs_ready": True,
        "aurora_steer_url": url,
        "remote_checked": check_remote,
        "warnings": collect_warnings(config),
    }

    if check_remote:
        resp = requests.get(f"{url}/health", timeout=10.0)
        if resp.status_code == 404:
            resp = requests.get(f"{url}/healthz", timeout=10.0)
        resp.raise_for_status()
        result["healthz"] = resp.json()

        client = AuroraSteerClient(base_url=url)
        state_dir = Path(config.get("state_dir", "./state")).resolve()
        emitter = TelemetryEmitter(
            state_dir / "preflight_outbox.jsonl",
            client=client,
            default_workstream=str(config.get("workstream", "kr2-gk-database")),
            default_environment="preflight",
        )
        emitter.emit(
            event_type="submitted",
            run_id="preflight_probe",
            solver="GX",
            environment="preflight",
            payload={"note": "preflight connectivity check"},
        )
        flushed = emitter.flush(force=True)
        result["preflight_events_flushed"] = flushed

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify config, queue folders, and aurora-steer connectivity."
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--offline", action="store_true", help="Skip remote HTTPS probe")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(json.dumps({"config_valid": False, "error": str(exc)}, indent=2))
        return 2
    summary = run_preflight(cfg, check_remote=not args.offline)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
