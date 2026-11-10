"""Resumable queue driver (`pending/` -> `running/` -> `completed/` | `needs_human/`).

Ingestion is crash-safe and never destructive:
- A pending folder is fully validated (safe run_id, parseable solver input, run_meta.json ids)
  before anything moves; invalid folders are quarantined under `needs_human/` with a unique
  name and an ISSUE.md.
- A run_id that is active (RUNNING/SUBMITTING) or already completed, or whose `running/` or
  `completed/` folder exists, is never re-ingested. (The original code `rmtree`'d the live
  `running/<run_id>` folder of an active job in that case.)
- The SUBMITTING intent is persisted before the folder moves and before the scheduler is
  called (`SimulationWatchdog.begin_submission`); `reconcile()` finishes interrupted moves.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hpc_core import (
    AuroraSteerClient,
    DriverAlreadyRunningError,
    RunRecord,
    SchedulerBackend,
    SlurmBackend,
    StateStore,
    TelemetryEmitter,
    acquire_single_instance_lock,
    atomic_write_text,
    is_safe_id,
)
from hpc_core.state import DEFAULT_LOCK_STALE_AFTER_S, StateCorruptError, StateSchemaError

from tacc_exec.config import ConfigError, load_config
from tacc_exec.diagnostics_adapter import (
    consume_stale_diagnostics_metrics,
    detect_solver_and_input_file,
)
from tacc_exec.patchers import read_gene_parameters, read_gx_parameters
from tacc_exec.watchdog import SimulationWatchdog, configure_cli_logging

logger = logging.getLogger(__name__)

SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_CAMPAIGN_ID_LEN = 64  # aurora-steer DiagnosticSummary.campaign_id limit
VALID_FIDELITIES = frozenset({"low", "high"})


def read_run_meta(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read and validate `run_meta.json` provenance; raise ValueError on unusable values.

    Nothing is inferred from folder names: an undeclared equilibrium_id stays empty rather
    than silently defaulting to a real equilibrium (which would corrupt leak-free ML splits).
    """
    meta: dict[str, Any] = {}
    meta_file = run_dir / "run_meta.json"
    if meta_file.is_file():
        try:
            raw = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"run_meta.json is unreadable: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("run_meta.json must contain a JSON object")
        meta = raw

    campaign_id = meta.get("campaign_id", config.get("campaign_id", "pilot_50"))
    if not is_safe_id(campaign_id) or len(campaign_id) > MAX_CAMPAIGN_ID_LEN:
        raise ValueError(
            f"campaign_id {campaign_id!r} must match [A-Za-z0-9][A-Za-z0-9_.-]* "
            f"(<= {MAX_CAMPAIGN_ID_LEN} chars)"
        )
    equilibrium_id = meta.get("equilibrium_id") or meta.get("geometry_id") or ""
    if equilibrium_id and not is_safe_id(equilibrium_id):
        raise ValueError(f"equilibrium_id {equilibrium_id!r} is not a safe identifier")
    fidelity = str(meta.get("fidelity", config.get("default_fidelity", "low"))).lower()
    if fidelity not in VALID_FIDELITIES:
        raise ValueError(f"fidelity {fidelity!r} must be one of {sorted(VALID_FIDELITIES)}")

    def _optional_text(key: str, limit: int) -> str | None:
        value = meta.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{key} must be a non-empty string of at most {limit} characters")
        return value.strip()

    return {
        "campaign_id": campaign_id,
        "equilibrium_id": equilibrium_id,
        "fidelity": fidelity,
        "code_version": _optional_text("code_version", 128),
        "equilibrium_origin": _optional_text("equilibrium_origin", 256),
    }


class QueueDriver:
    """Manages simulation queue lifecycle across TACC Slurm or GCP Cloud Batch."""

    def __init__(
        self,
        *,
        backend: SchedulerBackend,
        state_store: StateStore,
        steer_client: AuroraSteerClient | None,
        telemetry: TelemetryEmitter,
        config: dict[str, Any],
        dry_run: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        self.backend = backend
        self.state = state_store
        self.client = steer_client
        self.telemetry = telemetry
        self.config = config
        self.dry_run = bool(dry_run)
        self.max_concurrent_jobs = int(config.get("max_concurrent_jobs", 8))
        self.poll_interval_seconds = float(config.get("poll_interval_seconds", 60.0))
        # A prod run without a declared equilibrium can never enter the dataset: refuse it
        # before it burns allocation.
        self.require_equilibrium_id = bool(
            config.get("require_equilibrium_id", str(config.get("environment", "prod")) == "prod")
        )
        self._stop_requested = False

        self.watchdog = SimulationWatchdog(
            backend=backend,
            state_store=state_store,
            steer_client=steer_client,
            telemetry=telemetry,
            config=config,
            dry_run=dry_run,
            heartbeat=heartbeat,
        )

    def request_stop(self, *_args: Any) -> None:
        """Signal handler for graceful shutdown after current tick."""
        self._stop_requested = True

    def _active_running_count(self) -> int:
        return sum(1 for r in self.state.runs.values() if r.status in ("RUNNING", "SUBMITTING"))

    def _quarantine_invalid_pending_folder(self, folder: Path, reason: str) -> None:
        """Move malformed, unsafe or conflicting `pending/` entries to `needs_human/` (no deletes)."""
        safe_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", folder.name)[:80] or "invalid_run"
        dest = self.watchdog.needs_human_dir / f"quarantined_{safe_slug}"
        if dest.exists() or dest.is_symlink():
            dest = self.watchdog._unique_sibling(dest)
        shutil.move(str(folder), str(dest))
        atomic_write_text(
            dest / "ISSUE.md",
            "# Quarantined Pending Folder\n\n"
            f"- **Original Name:** `{folder.name}`\n- **Reason:** {reason}\n",
        )
        self.state.append_audit_event(
            "pending_folder_quarantined",
            safe_slug,
            {"original": folder.name, "reason": reason[:500], "dest": dest.name},
        )

    def _ingestion_conflict(self, run_id: str) -> str | None:
        existing = self.state.get_run(run_id)
        if existing is not None and existing.status in ("RUNNING", "SUBMITTING"):
            return (
                f"run_id {run_id!r} is already {existing.status} (job {existing.slurm_job_id}); "
                "a second folder with the same id is never submitted"
            )
        if existing is not None and existing.status == "COMPLETED":
            return f"run_id {run_id!r} already COMPLETED; dataset entries are immutable, use a new run_id"
        if (self.watchdog.running_dir / run_id).exists():
            return f"running/{run_id} already exists"
        if (self.watchdog.completed_dir / run_id).exists():
            return f"completed/{run_id} already exists; use a new run_id"
        return None

    def _ingest_pending_folder(self, folder: Path) -> bool:
        run_id = folder.name
        if not SAFE_RUN_ID_RE.match(run_id):
            if not self.dry_run:
                self._quarantine_invalid_pending_folder(
                    folder, f"Unsafe run directory name {run_id!r}"
                )
            return False
        conflict = self._ingestion_conflict(run_id)
        if conflict:
            if not self.dry_run:
                self._quarantine_invalid_pending_folder(folder, conflict)
            return False

        solver, input_path = detect_solver_and_input_file(folder)
        initial_params = (
            read_gx_parameters(input_path) if solver == "GX" else read_gene_parameters(input_path)
        )
        meta = read_run_meta(folder, self.config)
        if self.require_equilibrium_id and not meta["equilibrium_id"]:
            if not self.dry_run:
                self._quarantine_invalid_pending_folder(
                    folder,
                    "run_meta.json must declare equilibrium_id (required in prod: the run "
                    "could never enter the dataset)",
                )
            return False

        if self.dry_run:
            self.state.append_audit_event(
                "dry_run_would_submit", run_id, {"solver": solver, "params": initial_params}
            )
            return False

        dest_running = self.watchdog.running_dir / run_id
        existing = self.state.get_run(run_id)
        record = (
            existing
            if existing is not None
            else RunRecord(run_id=run_id, solver=solver, run_dir=str(dest_running))
        )
        record.solver = solver
        record.run_dir = str(dest_running)
        record.slurm_job_id = None
        record.retry_count = 0
        record.campaign_id = meta["campaign_id"]
        record.workstream = str(self.config.get("workstream", "kr2-gk-database"))
        record.backend = self.backend.capabilities.backend_name
        record.gpu_tier = self.backend.capabilities.default_gpu_tier
        record.fidelity = meta["fidelity"]
        record.geometry_id = meta["equilibrium_id"]
        record.equilibrium_id = meta["equilibrium_id"]
        record.code_version = meta["code_version"]
        record.equilibrium_origin = meta["equilibrium_origin"]
        record.active_failure_event_id = None
        record.cancel_requested = None
        record.consecutive_unknown_polls = 0
        record.consecutive_poll_errors = 0
        record.triage_deferrals = 0
        record.has_reported_running = False

        # Durable intent first, then the move, then the scheduler call.
        self.watchdog.begin_submission(record, kind="initial")
        try:
            os.rename(folder, dest_running)
        except OSError:
            shutil.move(str(folder), str(dest_running))
        consume_stale_diagnostics_metrics(dest_running)
        job_id = self.watchdog.execute_submission(record, dest_running)
        if job_id is None:
            return False

        self.state.append_audit_event(
            "run_submitted",
            run_id,
            {"job_id": job_id, "solver": solver, "backend": record.backend},
        )
        self.telemetry.emit(
            event_type="submitted",
            run_id=run_id,
            solver=solver,
            retry_count=0,
            walltime_resubmits=record.walltime_resubmits,
            workstream=record.workstream,
            backend=record.backend,
            campaign_id=record.campaign_id,
            fidelity=record.fidelity,
            payload={
                "slurm_job_id": job_id,
                "geometry_id": record.equilibrium_id,
                "current_parameters": initial_params,
            },
        )
        return True

    def _submit_pending_runs(self) -> int:
        submitted = 0
        try:
            entries = sorted(self.watchdog.pending_dir.iterdir())
        except OSError as exc:
            logger.error("Cannot list %s: %s", self.watchdog.pending_dir, exc)
            return 0
        for folder in entries:
            if folder.name.startswith("."):
                continue
            if self._active_running_count() >= self.max_concurrent_jobs:
                break
            if folder.is_symlink():
                if not self.dry_run:
                    self._quarantine_invalid_pending_folder(
                        folder, "Symlinked run folders are not accepted (path safety)"
                    )
                continue
            if not folder.is_dir():
                continue
            self.watchdog._heartbeat()
            try:
                if self._ingest_pending_folder(folder):
                    submitted += 1
            except DriverAlreadyRunningError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad folder must not block the queue
                logger.warning("Rejecting pending folder %s: %s", folder.name, exc)
                if not self.dry_run and folder.exists():
                    try:
                        self._quarantine_invalid_pending_folder(folder, str(exc)[:500])
                    except Exception:  # noqa: BLE001
                        logger.exception("Could not quarantine %s", folder)
        return submitted

    def tick(self) -> dict[str, int]:
        """Run one reconcile + poll + submit + telemetry flush cycle with per-run fault isolation."""
        polled = self.watchdog.poll_all()
        submitted = self._submit_pending_runs()
        flushed = 0 if self.dry_run else self.telemetry.flush()
        return {"polled": polled, "submitted": submitted, "flushed": flushed}

    def run_forever(self) -> None:
        """Continuously run `tick()` until `SIGINT`/`SIGTERM` is received."""
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        while not self._stop_requested:
            try:
                self.tick()
            except DriverAlreadyRunningError:
                logger.error("Lost the executor lock; stopping to avoid concurrent drivers")
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Top-level QueueDriver tick error (continuing): %s", exc)
            if self._stop_requested:
                break
            time.sleep(self.poll_interval_seconds)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `tacc-driver` on TACC login nodes."""
    parser = argparse.ArgumentParser(
        description="Run TACC Slurm simulation babysitter queue driver."
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--once", action="store_true", help="Execute a single tick and exit")
    parser.add_argument(
        "--dry-run", action="store_true", help="Observe queue without submitting/mutating"
    )
    args = parser.parse_args(argv)
    configure_cli_logging()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    state_dir = Path(cfg.get("state_dir", "./state")).resolve()
    lock_path = state_dir / "driver.lock"
    stale_after = float(cfg.get("lock_stale_after_seconds", DEFAULT_LOCK_STALE_AFTER_S))

    try:
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
                default_backend=str(cfg.get("backend", "tacc_slurm")),
            )
            driver = QueueDriver(
                backend=SlurmBackend(dry_run=args.dry_run),
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
        logger.info("Another executor instance is active; skipping this tick (%s)", exc)
        return 0
    except (StateCorruptError, StateSchemaError) as exc:
        logger.error("Refusing to run: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
