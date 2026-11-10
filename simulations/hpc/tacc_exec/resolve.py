"""Human escalation resolution CLI (`tacc-resolve`): returns fixed runs to `pending/`.

Runs under the same executor lock as the driver (waits for a running tick), so the state file
and the telemetry outbox are never written concurrently, and refuses to re-queue a run whose
previous scheduler job may still be active (two jobs in one run directory corrupt outputs).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from hpc_core import (
    POSSIBLY_ACTIVE_STATES,
    AuroraSteerClient,
    SchedulerBackend,
    SlurmBackend,
    StateStore,
    TelemetryEmitter,
    acquire_single_instance_lock,
)
from hpc_core.state import DEFAULT_LOCK_STALE_AFTER_S

from tacc_exec.config import load_config
from tacc_exec.diagnostics_adapter import consume_stale_diagnostics_metrics
from tacc_exec.driver import SAFE_RUN_ID_RE

logger = logging.getLogger(__name__)


class ResolveRefusedError(RuntimeError):
    """Raised when re-queueing would be unsafe (active run or possibly-active job)."""


def resolve_escalated_run(
    *,
    config: dict[str, Any],
    run_id: str,
    minutes_spent: float = 15.0,
    note: str = "Physicist resolved input parameters and re-queued",
    operator: str | None = None,
    flush_remote: bool = False,
    backend: SchedulerBackend | None = None,
    force: bool = False,
    lock_wait_s: float = 120.0,
) -> Path:
    """Move `queue/needs_human/<run_id>` back to `queue/pending/<run_id>` and log `human_resolved`."""
    if not SAFE_RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"Invalid run_id {run_id!r}")
    queue_dir = Path(config.get("queue_dir", "./queue")).resolve()
    state_dir = Path(config.get("state_dir", "./state")).resolve()
    stale_after = float(config.get("lock_stale_after_seconds", DEFAULT_LOCK_STALE_AFTER_S))

    with acquire_single_instance_lock(
        state_dir / "driver.lock", wait_s=lock_wait_s, stale_after_s=stale_after
    ):
        src = queue_dir / "needs_human" / run_id
        if not src.is_dir():
            raise FileNotFoundError(f"Escalated run directory not found: {src}")
        dest = queue_dir / "pending" / run_id
        if dest.exists():
            raise FileExistsError(f"Destination already exists in pending/: {dest}")

        store = StateStore(state_dir / "state.json", state_dir / "audit.jsonl")
        rec = store.get_run(run_id)
        if rec is not None and rec.status in ("RUNNING", "SUBMITTING"):
            raise ResolveRefusedError(
                f"Run {run_id} is {rec.status} in state; refusing to re-queue"
            )
        if rec is not None and rec.slurm_job_id and backend is not None and not force:
            job = backend.query_job(rec.slurm_job_id)
            if job.state in POSSIBLY_ACTIVE_STATES and job.state != "UNKNOWN":
                raise ResolveRefusedError(
                    f"Previous job {rec.slurm_job_id} of run {run_id} is still {job.state}; "
                    "cancel it first (or pass --force after verifying it cannot write)"
                )

        issue_md = src / "ISSUE.md"
        if issue_md.is_file():
            resolved = src / "ISSUE.resolved.md"
            idx = 1
            while resolved.exists():
                resolved = src / f"ISSUE.resolved.{idx}.md"
                idx += 1
            os.replace(issue_md, resolved)

        consume_stale_diagnostics_metrics(src)
        shutil.move(str(src), str(dest))

        solver = rec.solver if rec else ("GX" if (dest / "gx.in").is_file() else "GENE")
        if rec is not None:
            rec.status = "PENDING"
            rec.run_dir = str(dest)
            rec.retry_count = 0
            rec.walltime_resubmits = 0
            rec.active_failure_event_id = None
            rec.pending_submission = None
            rec.cancel_requested = None
            rec.human_resolutions += 1
            store.upsert_run(rec)

        store.append_audit_event(
            "human_resolved",
            run_id,
            {"minutes": minutes_spent, "note": note[:500], "forced": bool(force)},
        )

        client = AuroraSteerClient(base_url=config["aurora_steer_url"]) if flush_remote else None
        emitter = TelemetryEmitter(
            state_dir / "outbox.jsonl",
            client=client,
            default_workstream=str(config.get("workstream", "kr2-gk-database")),
            default_campaign_id=str(config.get("campaign_id", "pilot_50")),
            default_environment=str(config.get("environment", "prod")),
            default_backend=str(config.get("backend", "tacc_slurm")),
        )
        emitter.emit(
            event_type="human_resolved",
            run_id=run_id,
            solver=solver,
            operator=operator or os.getenv("USER"),
            payload={"minutes": float(minutes_spent), "note": note[:200]},
        )
        if flush_remote:
            emitter.flush(force=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move a human-fixed run from needs_human/ back to pending/."
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--run-id", required=True, help="Run ID under queue/needs_human/")
    parser.add_argument("--minutes", type=float, default=15.0, help="Human troubleshooting minutes")
    parser.add_argument("--note", default="Resolved manually", help="Short resolution note")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-queue even if the previous scheduler job still looks active",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    backend = SlurmBackend() if str(cfg.get("backend", "tacc_slurm")) == "tacc_slurm" else None
    dest = resolve_escalated_run(
        config=cfg,
        run_id=args.run_id,
        minutes_spent=args.minutes,
        note=args.note,
        backend=backend,
        force=args.force,
    )
    print(f"Re-queued {args.run_id} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
