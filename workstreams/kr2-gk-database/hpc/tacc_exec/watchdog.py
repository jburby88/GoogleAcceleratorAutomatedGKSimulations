"""Backend-agnostic simulation watchdog (works identically on TACC Slurm & GCP Cloud Batch).

Run state machine (persisted as `hpc_core.state.RunRecord.status`):

    PENDING --(driver: intent persisted, folder moved)--> SUBMITTING --(job accepted)--> RUNNING
    RUNNING --(continuation / patch / tier upgrade: intent persisted)--> SUBMITTING --> RUNNING
    RUNNING --(job COMPLETED, converged, harvested)--> COMPLETED
    PENDING|SUBMITTING|RUNNING --(guardrail, breaker, unknown outcome, external cancel ...)-->
        NEEDS_HUMAN

Crash-safety and spend-safety rules enforced here:
- Every scheduler submission is preceded by a durable SUBMITTING intent with a unique
  `submission_id`; `reconcile()` resolves interrupted submissions on the next tick (adopt the
  job or resubmit idempotently when the backend supports it, otherwise escalate). A crash can
  therefore never produce a job the state file does not know about, nor a blind resubmission.
- Nothing deletes a run directory. Name collisions are escalated, never `rmtree`'d.
- A run directory is never patched, moved or resubmitted while its job may still be active
  (`POSSIBLY_ACTIVE_STATES`): mid-run failures cancel and wait for a terminal state first, and
  escalations cancel a possibly-active job before moving its folder.
- Counters that bound automated actions (unknown polls, poll errors, triage deferrals,
  retries, walltime continuations) are persisted, so cron `--once` restarts never reset them.
- aurora-steer down or returning garbage never leads to a resubmission: transient errors are
  retried on later ticks (bounded), anything else escalates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpc_core import (
    ACTIVE_STATES,
    INFRA_STATES,
    POSSIBLY_ACTIVE_STATES,
    TRANSIENT_STATES,
    AuroraSteerClient,
    DriverAlreadyRunningError,
    RunRecord,
    SchedulerBackend,
    SlurmBackend,
    StateStore,
    TelemetryEmitter,
    TriageClientError,
    acquire_single_instance_lock,
    atomic_write_text,
    harvest_completed_run_to_dataset,
    send_escalation_notification,
    sha256_file,
)
from hpc_core.client import TriageTransientError
from hpc_core.manifest import DatasetEntryExistsError
from hpc_core.slurm import SlurmQueryError
from hpc_core.state import DEFAULT_LOCK_STALE_AFTER_S, StateCorruptError, StateSchemaError

from tacc_exec.config import WARM_RESTART_ALLOWED_PARAMS
from tacc_exec.diagnostics_adapter import (
    DiagnosticReport,
    DiagnosticsSettings,
    consume_stale_diagnostics_metrics,
    detect_solver_and_input_file,
    find_checkpoint_files,
    inspect_run_directory,
)
from tacc_exec.patchers import patch_gene_parameters, patch_gx_toml

logger = logging.getLogger(__name__)

GRID_PARAMS: frozenset[str] = frozenset({"nx", "ny", "nz"})
VALID_DIRECTIVE_ACTIONS: frozenset[str] = frozenset(
    {"warm_restart", "cold_restart", "resubmit_walltime", "escalate_to_human"}
)
# Infrastructure terminations resumed from checkpoint with zero LLM calls (bounded).
CONTINUATION_STATES: frozenset[str] = frozenset({"TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL"})
TRIAGE_FAILURE_TYPES: frozenset[str] = frozenset(
    {"STALL", "TIMESTEP_COLLAPSE", "SPECTRAL_PILEUP", "DIVERGENCE", "CRASH", "UNKNOWN"}
)
GCP_BACKENDS: frozenset[str] = frozenset({"gcp_batch", "gcp_k8s"})
TRIAGE_PAYLOAD_LIMIT_BYTES = 2040  # strictly under the client's 2048-byte refusal threshold
CANCEL_POLL_INTERVAL_S = 3.0


class ClientGuardrailError(ValueError):
    """Raised when client-side reject-only guardrails reject an ActionDirective."""


def validate_directive_client_side(
    directive: dict[str, Any],
    allowed_parameters: dict[str, dict[str, Any]],
) -> dict[str, int | float]:
    """Client-side reject-only verification before touching any solver file."""
    action = directive.get("action")
    mods = directive.get("parameter_modifications", {})
    restart_flag = directive.get("restart_from_checkpoint", False)

    if action not in VALID_DIRECTIVE_ACTIONS:
        raise ClientGuardrailError(f"Unknown action {action!r}")
    if not isinstance(restart_flag, bool):
        raise ClientGuardrailError("restart_from_checkpoint must be a JSON boolean")
    if not isinstance(mods, dict):
        raise ClientGuardrailError("parameter_modifications must be a dict")

    if action in ("escalate_to_human", "resubmit_walltime") and mods:
        raise ClientGuardrailError(f"action={action!r} cannot include parameter_modifications")

    if action == "resubmit_walltime":
        # Walltime continuations are handled locally with 0 LLM calls; aurora-steer never
        # issues this action from /triage (manifest.yaml action policy), so reject it.
        raise ClientGuardrailError("resubmit_walltime is not accepted from /triage")

    if action == "warm_restart" and not restart_flag:
        raise ClientGuardrailError("warm_restart requires restart_from_checkpoint=True")

    if action == "cold_restart" and restart_flag:
        raise ClientGuardrailError("cold_restart requires restart_from_checkpoint=False")

    if action in ("warm_restart", "cold_restart") and not mods:
        raise ClientGuardrailError(f"action={action!r} requires parameter_modifications")

    validated: dict[str, int | float] = {}
    for name, raw_val in mods.items():
        if name not in allowed_parameters:
            raise ClientGuardrailError(f"Parameter {name!r} is not in client allowlist")
        spec = allowed_parameters[name]
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise ClientGuardrailError(f"Parameter {name!r} has non-numeric value {raw_val!r}")
        if not math.isfinite(float(raw_val)):
            raise ClientGuardrailError(f"Parameter {name!r} is non-finite ({raw_val!r})")

        vtype = str(spec.get("type", "float"))
        if vtype == "int":
            if isinstance(raw_val, float) and not raw_val.is_integer():
                raise ClientGuardrailError(
                    f"Integer parameter {name!r} received fractional float {raw_val}"
                )
            val_num: int | float = int(raw_val)
        else:
            val_num = float(raw_val)

        try:
            min_v = float(spec["min"])
            max_v = float(spec["max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientGuardrailError(f"Allowlist entry for {name!r} has no valid bounds") from exc
        if not (math.isfinite(min_v) and math.isfinite(max_v)):
            raise ClientGuardrailError(f"Allowlist entry for {name!r} has non-finite bounds")
        if float(val_num) < min_v or float(val_num) > max_v:
            raise ClientGuardrailError(
                f"Parameter {name!r}={val_num} outside [{spec['min']}, {spec['max']}]"
            )
        validated[name] = val_num

    if action == "warm_restart" or restart_flag:
        disallowed = sorted(
            name
            for name in validated
            if name not in WARM_RESTART_ALLOWED_PARAMS
            or (allowed_parameters.get(name) or {}).get("warm_restart_allowed") is not True
        )
        if disallowed:
            prefix = (
                "Grid resolution changes (nx, ny, nz) are incompatible with binary checkpoints; "
                if GRID_PARAMS & set(disallowed)
                else ""
            )
            raise ClientGuardrailError(
                f"{prefix}parameters {disallowed} may only change with action='cold_restart' "
                "and restart_from_checkpoint=False (warm restarts may change only parameters "
                "with warm_restart_allowed=true: "
                f"{sorted(WARM_RESTART_ALLOWED_PARAMS)})"
            )

    return validated


def _fit_triage_payload_under_2kb(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministically trim `log_tail`, `previous_directives` and metrics to stay < 2048 B.

    Whole entries are dropped (never byte-sliced), so the result is always valid JSON. If even
    the minimal payload is too large, the client refuses to send it and the run escalates.
    """
    candidate = dict(payload)
    candidate["log_tail"] = list(candidate.get("log_tail", []))[-25:]
    candidate["previous_directives"] = [
        {"action": d.get("action"), "mods": d.get("parameter_modifications", {})}
        for d in list(candidate.get("previous_directives", []))[-3:]
    ]

    while len(json.dumps(candidate, allow_nan=False).encode("utf-8")) >= TRIAGE_PAYLOAD_LIMIT_BYTES:
        if candidate["log_tail"]:
            candidate["log_tail"].pop(0)
        elif candidate["previous_directives"]:
            candidate["previous_directives"].pop(0)
        elif len(candidate.get("scalar_metrics", {})) > 3:
            first_key = next(iter(candidate["scalar_metrics"]))
            candidate["scalar_metrics"] = {
                k: v for k, v in candidate["scalar_metrics"].items() if k != first_key
            }
        else:
            break
    return candidate


def _write_issue_markdown(
    target_dir: Path,
    record: RunRecord,
    failure_type: str,
    reason: str,
    diag: DiagnosticReport | None,
) -> str:
    """Write human-readable `ISSUE.md` summarizing failure context and tried patches."""
    lines = [
        f"# Escalation Handoff: `{record.run_id}`",
        "",
        f"- **Workstream:** `{record.workstream}`",
        f"- **Campaign:** `{record.campaign_id}`",
        f"- **Backend:** `{record.backend}` (`gpu_tier={record.gpu_tier}`)",
        f"- **Solver / Fidelity:** `{record.solver}` (`{record.fidelity}`)",
        f"- **Equilibrium ID:** `{record.equilibrium_id or '(not declared in run_meta.json)'}`",
        f"- **Failure Type:** `{failure_type}`",
        f"- **Retry Count:** `{record.retry_count}` (Walltime/Spot continuations: `{record.walltime_resubmits}`)",
        f"- **Last scheduler job:** `{record.slurm_job_id}` (last state `{record.last_scheduler_state}`)",
        f"- **Escalation Reason:** {reason}",
        "",
        "## Scheduler Submissions",
    ]
    if record.restart_history:
        for entry in record.restart_history[-10:]:
            lines.append(
                f"- `{entry.get('submission_id')}` -> job `{entry.get('job_id')}` "
                f"({entry.get('kind')}, restart_from_checkpoint={entry.get('restart_from_checkpoint')})"
            )
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Prior Directives Tried"])
    if record.directives_history:
        for idx, d in enumerate(record.directives_history, start=1):
            lines.append(
                f"{idx}. `{d.get('action')}` — mods={json.dumps(d.get('parameter_modifications', {}))} "
                f"— {d.get('reason', '')}"
            )
    else:
        lines.append("- None (escalated on initial failure or guardrail check).")

    if diag is not None:
        lines.extend(
            [
                "",
                "## Latest Numerical Parameters & Scalar Metrics",
                f"- **Parameters:** `{json.dumps(diag.current_parameters, sort_keys=True)}`",
                f"- **Metrics:** `{json.dumps(diag.scalar_metrics, sort_keys=True)}`",
                "",
                "## Log Tail",
                "```text",
                *(diag.log_tail[-15:] if diag.log_tail else ["(no solver log lines available)"]),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## How to Resolve & Re-Queue",
            "1. Check the scheduler: no job of this run may still be queued or running "
            f"(e.g. `squeue --me --name=bsit-{record.run_id}-{record.submit_nonce or '<nonce>'}-*`).",
            f"2. Inspect and edit `{target_dir / ('gx.in' if record.solver == 'GX' else 'parameters')}`.",
            f"3. Run `tacc-resolve --config config.json --run-id {record.run_id} --minutes <mins>`.",
            "",
        ]
    )
    text = "\n".join(lines)
    atomic_write_text(target_dir / "ISSUE.md", text)
    return text


class SimulationWatchdog:
    """Polls active simulations on Slurm or GCP Cloud Batch, steers failures, and harvests datasets."""

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
        now_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.state = state_store
        self.client = steer_client
        self.telemetry = telemetry
        self.config = config
        self.dry_run = bool(dry_run)
        self.heartbeat = heartbeat
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

        queue_root = Path(config.get("queue_dir", "./queue")).resolve()
        self.pending_dir = queue_root / "pending"
        self.running_dir = queue_root / "running"
        self.completed_dir = queue_root / "completed"
        self.needs_human_dir = queue_root / "needs_human"
        for d in (self.pending_dir, self.running_dir, self.completed_dir, self.needs_human_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.max_retries = int(config.get("max_retries", 2))
        self.max_walltime_resubmits = int(config.get("max_walltime_resubmits", 5))
        self.max_unknown_polls = int(config.get("max_unknown_polls", 10))
        self.max_poll_errors = max(1, int(config.get("max_poll_errors", 3)))
        self.max_triage_deferrals = max(0, int(config.get("max_triage_deferrals", 3)))
        self.max_pending_s = float(config.get("max_pending_hours", 168.0)) * 3600.0
        self.cancel_wait_s = max(0.0, float(config.get("cancel_wait_seconds", 30.0)))
        self.cancel_timeout_s = float(config.get("cancel_timeout_seconds", 1800.0))
        self.environment = str(config.get("environment", "prod"))
        self.allowed_parameters: dict[str, dict[str, Any]] = dict(
            config.get("allowed_parameters", {})
        )
        self.diagnostics_cmd: list[str] | None = config.get("diagnostic_script_cmd") or config.get(
            "diagnostics_cmd"
        )
        self.sim_time_target_default = float(config.get("sim_time_target_default", 200.0))
        self.settings = DiagnosticsSettings.from_config(config)
        raw_ds = config.get("dataset_output_dir")
        self.dataset_output_dir = Path(raw_ds).resolve() if raw_ds else None

    # ------------------------------------------------------------------ small helpers
    def _now(self) -> float:
        return float(self.now_fn())

    def _heartbeat(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat()

    def _is_gcp(self) -> bool:
        return self.backend.capabilities.backend_name in GCP_BACKENDS

    def _inspect(self, run_dir: Path) -> DiagnosticReport:
        return inspect_run_directory(
            run_dir=run_dir,
            diagnostics_cmd=self.diagnostics_cmd,
            sim_time_target_default=self.sim_time_target_default,
            settings=self.settings,
        )

    def _emit(self, event_type: str, record: RunRecord, payload: dict[str, Any], **kw: Any) -> None:
        self.telemetry.emit(
            event_type=event_type,
            run_id=record.run_id,
            solver=record.solver,
            retry_count=record.retry_count,
            walltime_resubmits=record.walltime_resubmits,
            workstream=record.workstream,
            backend=record.backend,
            campaign_id=record.campaign_id,
            fidelity=record.fidelity,
            payload=payload,
            **kw,
        )

    @staticmethod
    def _input_sha256(run_dir: Path) -> str | None:
        try:
            _solver, input_path = detect_solver_and_input_file(run_dir)
            return sha256_file(input_path)
        except OSError:
            return None

    @staticmethod
    def _unique_sibling(path: Path) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        candidate = path.with_name(f"{path.name}.{stamp}")
        idx = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.name}.{stamp}.{idx}")
            idx += 1
        return candidate

    # ------------------------------------------------------------------ submission protocol
    def begin_submission(
        self,
        record: RunRecord,
        *,
        kind: str,
        force_high_memory: bool = False,
        restart_from_checkpoint: bool | None = None,
        scheduler_state: str | None = None,
    ) -> str:
        """Persist a SUBMITTING intent (unique submission_id) BEFORE any scheduler call."""
        if self.dry_run:
            raise RuntimeError("begin_submission must never be called in dry-run mode")
        if not record.submit_nonce:
            record.submit_nonce = uuid.uuid4().hex[:8]
        record.submit_seq += 1
        submission_id = f"bsit-{record.run_id}-{record.submit_nonce}-{record.submit_seq}"
        record.pending_submission = {
            "submission_id": submission_id,
            "kind": kind,
            "force_high_memory": bool(force_high_memory),
            "restart_from_checkpoint": restart_from_checkpoint,
            "scheduler_state": scheduler_state,
            "previous_job_id": record.slurm_job_id,
            "started_at_epoch": self._now(),
        }
        record.status = "SUBMITTING"
        self.state.upsert_run(record)
        self.state.append_audit_event(
            "submission_intent", record.run_id, {"submission_id": submission_id, "kind": kind}
        )
        return submission_id

    def _call_submit(self, record: RunRecord, run_dir: Path, submission_id: str) -> str:
        ps = record.pending_submission or {}
        kwargs: dict[str, Any] = {
            "solver": record.solver,
            "parameters": self._current_parameters(run_dir),
            "retry_count": record.retry_count,
            "force_high_memory": bool(ps.get("force_high_memory", False)),
        }
        if self.backend.capabilities.idempotent_submit:
            kwargs["submission_id"] = submission_id
        return self.backend.submit(run_dir, **kwargs)

    @staticmethod
    def _current_parameters(run_dir: Path) -> dict[str, int | float]:
        from tacc_exec.patchers import read_gene_parameters, read_gx_parameters

        try:
            solver, input_path = detect_solver_and_input_file(run_dir)
            if solver == "GX":
                return read_gx_parameters(input_path)
            return read_gene_parameters(input_path)
        except Exception:  # noqa: BLE001 - parameters are informational for the backend
            return {}

    def _find_submission(self, submission_id: str) -> str | None:
        if not self.backend.capabilities.idempotent_submit:
            return None
        finder = getattr(self.backend, "find_submission", None)
        if finder is None:
            return None
        return finder(submission_id)

    def execute_submission(self, record: RunRecord, run_dir: Path) -> str | None:
        """Call the scheduler for the persisted intent; commit RUNNING or escalate (fail safe)."""
        ps = record.pending_submission or {}
        submission_id = str(ps.get("submission_id") or "")
        if record.status != "SUBMITTING" or not submission_id:
            raise RuntimeError(f"Run {record.run_id} has no persisted submission intent")
        self._heartbeat()
        input_sha = self._input_sha256(run_dir)
        try:
            job_id = self._call_submit(record, run_dir, submission_id)
        except Exception as exc:  # noqa: BLE001 - any scheduler failure must escalate
            if self.backend.capabilities.idempotent_submit and getattr(
                exc, "outcome_unknown", True
            ):
                # The job may still have been created (e.g. slurmctld timed out after
                # queueing it). Stay SUBMITTING with the same submission_id: the next tick's
                # reconcile adopts it or resubmits idempotently. Bounded by max_poll_errors.
                record.consecutive_poll_errors += 1
                self.state.upsert_run(record)
                self.state.append_audit_event(
                    "submission_outcome_unknown",
                    record.run_id,
                    {"submission_id": submission_id, "error": str(exc)[:300]},
                )
                if record.consecutive_poll_errors < self.max_poll_errors:
                    return None
            found: str | None = None
            try:
                found = self._find_submission(submission_id)
            except Exception:  # noqa: BLE001
                found = None
            if found:
                record.slurm_job_id = found
            self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="CRASH",
                reason=(
                    f"Scheduler submit failed for submission {submission_id} "
                    f"({ps.get('kind')}): {str(exc)[:300]}"
                ),
                diag=None,
                job_may_be_active=bool(found),
            )
            return None
        self._commit_submission(record, job_id, input_sha)
        return job_id

    def _commit_submission(self, record: RunRecord, job_id: str, input_sha: str | None) -> None:
        ps = record.pending_submission or {}
        now = self._now()
        record.restart_history.append(
            {
                "seq": record.submit_seq,
                "submission_id": ps.get("submission_id"),
                "kind": ps.get("kind"),
                "job_id": job_id,
                "previous_job_id": ps.get("previous_job_id"),
                "scheduler_state": ps.get("scheduler_state"),
                "restart_from_checkpoint": ps.get("restart_from_checkpoint"),
                "force_high_memory": ps.get("force_high_memory"),
                "retry_count": record.retry_count,
                "walltime_resubmits": record.walltime_resubmits,
                "input_sha256": input_sha,
                "submitted_at_utc": datetime.fromtimestamp(now, tz=UTC).isoformat(),
            }
        )
        record.slurm_job_id = job_id
        record.status = "RUNNING"
        record.submitted_at_epoch = now
        if not record.first_submitted_at_epoch:
            record.first_submitted_at_epoch = now
        record.input_sha256_submitted = input_sha
        record.pending_submission = None
        record.cancel_requested = None
        record.consecutive_unknown_polls = 0
        record.consecutive_poll_errors = 0
        record.triage_deferrals = 0
        record.last_scheduler_state = None
        self.state.upsert_run(record)
        self.state.append_audit_event(
            "submission_committed",
            record.run_id,
            {"job_id": job_id, "submission_id": ps.get("submission_id"), "kind": ps.get("kind")},
        )

    def recover_submission(self, record: RunRecord) -> str:
        """Resolve a SUBMITTING intent left by a crash or timeout (never a blind resubmit)."""
        if self.dry_run:
            self.state.append_audit_event(
                "dry_run_submission_recovery_needed",
                record.run_id,
                {"intent": record.pending_submission},
            )
            return record.status
        ps = record.pending_submission or {}
        submission_id = str(ps.get("submission_id") or "")
        run_dir = Path(record.run_dir)
        if not run_dir.is_dir():
            pending_copy = self.pending_dir / record.run_id
            if ps.get("kind") == "initial" and pending_copy.is_dir() and not run_dir.exists():
                # Crash between persisting the intent and moving the folder: finish the move.
                shutil.move(str(pending_copy), str(run_dir))
            else:
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="CRASH",
                    reason=f"Submission intent {submission_id or '?'} found but run folder is missing.",
                    diag=None,
                )
        if not submission_id:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="CRASH",
                reason="SUBMITTING record without a submission_id (state invariant violated).",
                diag=None,
            )
        if not self.backend.capabilities.idempotent_submit or not hasattr(
            self.backend, "find_submission"
        ):
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="CRASH",
                reason=(
                    f"Submission {submission_id} was interrupted and backend "
                    f"{self.backend.capabilities.backend_name!r} cannot confirm whether a job was "
                    f"created. Check the scheduler for a job named {submission_id} before "
                    "re-queueing."
                ),
                diag=None,
            )
        try:
            found = self._find_submission(submission_id)
        except Exception as exc:  # noqa: BLE001
            record.consecutive_poll_errors += 1
            self.state.upsert_run(record)
            if record.consecutive_poll_errors >= self.max_poll_errors:
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="CRASH",
                    reason=f"Cannot look up interrupted submission {submission_id}: {exc}",
                    diag=None,
                )
            return record.status
        if found:
            self._commit_submission(record, found, self._input_sha256(run_dir))
            self.state.append_audit_event(
                "submission_adopted",
                record.run_id,
                {"submission_id": submission_id, "job_id": found},
            )
            return record.status
        self.execute_submission(record, run_dir)
        return record.status

    # ------------------------------------------------------------------ reconcile
    def reconcile(self) -> None:
        """Repair state after a crash: interrupted submissions, moved/vanished/untracked folders."""
        for record in list(self.state.runs.values()):
            self._heartbeat()
            try:
                if record.status == "SUBMITTING":
                    self.recover_submission(record)
                elif record.status == "RUNNING" and not Path(record.run_dir).exists():
                    self._reconcile_missing_run_dir(record)
            except DriverAlreadyRunningError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad record must not block the queue
                logger.exception("Reconcile failed for run %s: %s", record.run_id, exc)
                self.escalate_safely(record, f"Reconcile failed: {exc}", job_may_be_active=True)
        self._quarantine_untracked_running_folders()

    def _reconcile_missing_run_dir(self, record: RunRecord) -> None:
        human_copy = self.needs_human_dir / record.run_id
        done_copy = self.completed_dir / record.run_id
        if human_copy.is_dir():
            # Crash after the escalation move but before the state write: finish escalating.
            record.status = "NEEDS_HUMAN"
            record.run_dir = str(human_copy)
            record.escalation_count += 1
            record.pending_submission = None
            record.cancel_requested = None
            self.state.upsert_run(record)
            self.state.append_audit_event("escalation_reconciled", record.run_id, {})
        elif done_copy.is_dir():
            # Crash after the completion move: continue; _complete_run is idempotent.
            record.run_dir = str(done_copy)
            self.state.upsert_run(record)
            self.state.append_audit_event("completion_move_reconciled", record.run_id, {})
        else:
            self._escalate_run(
                record=record,
                run_dir=Path(record.run_dir),
                failure_type="CRASH",
                reason="Run directory vanished while the run was RUNNING.",
                diag=None,
                job_may_be_active=True,
            )

    def _quarantine_untracked_running_folders(self) -> None:
        active = {
            rid for rid, r in self.state.runs.items() if r.status in ("RUNNING", "SUBMITTING")
        }
        for folder in sorted(self.running_dir.iterdir()):
            if folder.name.startswith(".") or folder.name in active or not folder.is_dir():
                continue
            record = self.state.get_run(folder.name)
            if self.dry_run:
                self.state.append_audit_event(
                    "dry_run_untracked_running_folder", folder.name, {"path": str(folder)}
                )
                continue
            if record is not None:
                record.run_dir = str(folder)
                self._escalate_run(
                    record=record,
                    run_dir=folder,
                    failure_type="CRASH",
                    reason=(
                        f"Folder found in running/ while the state says {record.status}; "
                        "a job may exist for it."
                    ),
                    diag=None,
                    job_may_be_active=bool(record.slurm_job_id),
                )
                continue
            dest = self._unique_sibling(self.needs_human_dir / f"untracked_{folder.name}")
            shutil.move(str(folder), str(dest))
            atomic_write_text(
                dest / "ISSUE.md",
                "# Untracked folder found in running/\n\n"
                f"- **Original path:** `{folder}`\n"
                "- No state record exists for it, so the babysitter cannot know whether a "
                "scheduler job was submitted. Check the scheduler before re-queueing it.\n",
            )
            self.state.append_audit_event(
                "untracked_running_folder_quarantined", folder.name, {"dest": str(dest)}
            )

    # ------------------------------------------------------------------ polling
    def poll_all(self) -> int:
        """Reconcile, then poll every RUNNING run with per-run fault isolation."""
        self.reconcile()
        polled = 0
        for record in list(self.state.runs.values()):
            if record.status != "RUNNING":
                continue
            self._heartbeat()
            try:
                self.poll_run(record)
                polled += 1
                if record.status == "RUNNING" and record.consecutive_poll_errors:
                    record.consecutive_poll_errors = 0
                    self.state.upsert_run(record)
            except DriverAlreadyRunningError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate failures per run
                logger.exception("Unhandled error polling run %s: %s", record.run_id, exc)
                record.consecutive_poll_errors += 1
                try:
                    self.state.upsert_run(record)
                    self.state.append_audit_event(
                        "poll_run_exception",
                        record.run_id,
                        {"error": str(exc)[:300], "count": record.consecutive_poll_errors},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Could not persist poll error for run %s", record.run_id)
                if record.consecutive_poll_errors >= self.max_poll_errors:
                    self.escalate_safely(
                        record,
                        f"Poll failed {record.consecutive_poll_errors} consecutive times: {exc}",
                        job_may_be_active=True,
                    )
        return polled

    def poll_run(self, record: RunRecord) -> str:
        """Inspect a single `RunRecord` and advance its state machine safely."""
        if record.status == "SUBMITTING":
            return self.recover_submission(record)
        if record.status != "RUNNING":
            return record.status
        run_dir = Path(record.run_dir)
        if not record.slurm_job_id:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="CRASH",
                reason="RUNNING record has no scheduler job id (state invariant violated).",
                diag=None,
            )

        try:
            job = self.backend.query_job(record.slurm_job_id)
        except SlurmQueryError as exc:
            # Scheduler outage (squeue and sacct both failed, e.g. slurmctld maintenance):
            # nothing is known about the job, so do nothing destructive and do not count it
            # against this run; the next tick retries.
            logger.warning("Scheduler unavailable for run %s: %s", record.run_id, exc)
            self.state.append_audit_event(
                "scheduler_query_deferred", record.run_id, {"error": str(exc)[:300]}
            )
            return record.status
        record.last_scheduler_state = job.state
        if job.gpu_tier:
            record.gpu_tier = job.gpu_tier

        # 1. Scheduler UNKNOWN -> bounded patience, then escalate (cancelling the job first).
        if job.state in INFRA_STATES:
            record.consecutive_unknown_polls += 1
            self.state.upsert_run(record)
            if record.consecutive_unknown_polls >= self.max_unknown_polls:
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="UNKNOWN",
                    reason=(
                        f"Scheduler returned UNKNOWN for {record.consecutive_unknown_polls} "
                        f"consecutive polls (job_id={record.slurm_job_id})."
                    ),
                    diag=None,
                    job_may_be_active=True,
                )
            return record.status
        record.consecutive_unknown_polls = 0

        # 2. Queued / winding-down states -> wait, unless the job can never start.
        if job.state in TRANSIENT_STATES:
            if job.blocked:
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="UNKNOWN",
                    reason=(
                        f"Scheduler reports job {record.slurm_job_id} cannot run without human "
                        f"action ({job.state}: {job.reason or 'no reason given'})."
                    ),
                    diag=None,
                    job_may_be_active=True,
                )
            if (
                job.state == "PENDING"
                and record.cancel_requested is None
                and self.max_pending_s > 0
                and record.submitted_at_epoch > 0
                and self._now() - record.submitted_at_epoch > self.max_pending_s
            ):
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="UNKNOWN",
                    reason=(
                        f"Job {record.slurm_job_id} has been pending for more than "
                        f"{self.max_pending_s / 3600.0:.0f} h ({job.reason or 'no reason given'})."
                    ),
                    diag=None,
                    job_may_be_active=True,
                )
            if record.cancel_requested is not None:
                return self._await_cancellation(record, run_dir)
            self.state.upsert_run(record)
            return record.status

        # 3. Active job.
        if job.state in ACTIVE_STATES:
            if record.cancel_requested is not None:
                return self._await_cancellation(record, run_dir)
            diag = self._inspect(run_dir)
            if not record.has_reported_running:
                record.has_reported_running = True
                self._emit(
                    "running",
                    record,
                    {
                        "slurm_job_id": record.slurm_job_id,
                        "gpu_tier": record.gpu_tier,
                        "geometry_id": record.equilibrium_id,
                        "current_parameters": diag.current_parameters,
                    },
                )
            if diag.failure_type is not None:
                if self.dry_run:
                    self.state.append_audit_event(
                        "dry_run_midrun_failure_observed",
                        record.run_id,
                        {"failure_type": diag.failure_type, "job_id": record.slurm_job_id},
                    )
                    return record.status
                return self._cancel_then_handle_failure(record, run_dir, diag)
            self._emit(
                "progress",
                record,
                {
                    "sim_time_reached": diag.sim_time_reached,
                    "metrics": diag.scalar_metrics,
                    "current_parameters": diag.current_parameters,
                },
            )
            self.state.upsert_run(record)
            return record.status

        # 4. Terminal states from here on: the job no longer runs.
        diag = self._inspect(run_dir)

        if record.cancel_requested is not None:
            # The job we cancelled for a mid-run failure has terminated: handle that failure.
            stored = str(record.cancel_requested.get("failure_type") or "")
            diag.failure_type = stored or diag.failure_type or "CRASH"
            return self._dispatch_failure(
                record,
                run_dir,
                diag,
                early_cancelled=True,
                elapsed_s=job.elapsed_seconds,
                scheduler_state=job.state,
            )

        if job.state == "CANCELLED":
            if self.dry_run:
                self.state.append_audit_event(
                    "dry_run_external_cancel_observed", record.run_id, {"reason": job.reason}
                )
                return record.status
            who = "an administrator/the system" if job.cancelled_externally else "a user or admin"
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="CRASH",
                reason=(
                    f"Job {record.slurm_job_id} was cancelled outside the babysitter by {who} "
                    f"({job.reason or 'no detail'}); not resubmitting automatically."
                ),
                diag=diag,
            )

        if job.state in CONTINUATION_STATES:
            return self._handle_walltime_or_spot_continuation(
                record=record, run_dir=run_dir, diag=diag, scheduler_state=job.state
            )

        if (
            (job.state == "OUT_OF_MEMORY" or diag.failure_type == "OUT_OF_MEMORY")
            and self._is_gcp()
            and record.gpu_tier != "a100"  # TODO(david): confirm A100 is the highest available tier
        ):
            if self.dry_run:
                self.state.append_audit_event(
                    "dry_run_oom_upgrade_l4_to_a100", record.run_id, {"from_tier": record.gpu_tier}
                )
                return record.status
            return self._handle_gcp_oom_tier_upgrade(record=record, run_dir=run_dir, diag=diag)

        # 5. Scheduler COMPLETED -> verify diagnostics, convergence and finite fluxes.
        if job.state == "COMPLETED" and diag.failure_type is None:
            if diag.diagnostics_error:
                if self.dry_run:
                    return record.status
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="UNKNOWN",
                    reason=f"Diagnostics unusable, refusing to harvest: {diag.diagnostics_error}",
                    diag=diag,
                )
            if diag.simulated:
                # Synthetic output (GX_SIMULATE_RUN): never triage/resubmit it. The harvester
                # records it with dataset_eligible=false (reason "simulated").
                return self._complete_run(
                    record=record, run_dir=run_dir, diag=diag, elapsed_s=job.elapsed_seconds
                )
            q_es = diag.fluxes.get("heat_flux_es")
            has_valid_flux = (
                q_es is not None
                and not isinstance(q_es, bool)
                and isinstance(q_es, (int, float))
                and math.isfinite(float(q_es))
            )
            time_ok = not (
                diag.sim_time_target > 0
                and diag.sim_time_reached
                < self.settings.convergence_time_fraction * diag.sim_time_target
            )
            if not diag.converged or not has_valid_flux or not time_ok:
                diag.failure_type = "STALL"
                if self.dry_run:
                    self.state.append_audit_event(
                        "dry_run_terminal_failure_observed",
                        record.run_id,
                        {"failure_type": diag.failure_type, "scheduler_state": job.state},
                    )
                    return record.status
                return self._handle_physics_or_solver_failure(
                    record=record,
                    run_dir=run_dir,
                    diag=diag,
                    early_cancelled=False,
                    elapsed_s=job.elapsed_seconds,
                    scheduler_state=job.state,
                    failure_type="STALL",
                )
            return self._complete_run(
                record=record, run_dir=run_dir, diag=diag, elapsed_s=job.elapsed_seconds
            )

        # 6. Terminal FAILED / DEADLINE / OUT_OF_MEMORY (TACC) / post-run failure -> triage.
        if diag.failure_type is None:
            diag.failure_type = "CRASH"
        if self.dry_run:
            self.state.append_audit_event(
                "dry_run_terminal_failure_observed",
                record.run_id,
                {"failure_type": diag.failure_type, "scheduler_state": job.state},
            )
            return record.status
        return self._handle_physics_or_solver_failure(
            record=record,
            run_dir=run_dir,
            diag=diag,
            early_cancelled=False,
            elapsed_s=job.elapsed_seconds,
            scheduler_state=job.state,
        )

    # ------------------------------------------------------------------ cancellation
    def _cancel_then_handle_failure(
        self, record: RunRecord, run_dir: Path, diag: DiagnosticReport
    ) -> str:
        """Cancel the job for a mid-run failure; act only once it is no longer active."""
        job_id = str(record.slurm_job_id)
        record.cancel_requested = {
            "failure_type": diag.failure_type,
            "job_id": job_id,
            "requested_at_epoch": self._now(),
        }
        self.state.upsert_run(record)
        try:
            self.backend.cancel(job_id)
        except Exception as exc:  # noqa: BLE001 - the job may already have ended
            logger.warning("Cancel of job %s for run %s failed: %s", job_id, record.run_id, exc)
        self.state.append_audit_event(
            "cancel_requested", record.run_id, {"job_id": job_id, "failure_type": diag.failure_type}
        )
        deadline = time.monotonic() + self.cancel_wait_s
        while True:
            try:
                state = self.backend.query_job(job_id).state
            except Exception:  # noqa: BLE001
                state = None
            if state is not None and state not in POSSIBLY_ACTIVE_STATES:
                break
            if time.monotonic() >= deadline:
                # Still winding down: never patch/resubmit under a live job. Next tick resumes.
                return record.status
            self.sleep_fn(CANCEL_POLL_INTERVAL_S)
        record.last_scheduler_state = state
        return self._dispatch_failure(
            record, run_dir, diag, early_cancelled=True, elapsed_s=0.0, scheduler_state="RUNNING"
        )

    def _await_cancellation(self, record: RunRecord, run_dir: Path) -> str:
        requested = float((record.cancel_requested or {}).get("requested_at_epoch", 0.0))
        if requested and self._now() - requested > self.cancel_timeout_s:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=str((record.cancel_requested or {}).get("failure_type") or "CRASH"),
                reason=(
                    f"Job {record.slurm_job_id} is still active {self.cancel_timeout_s:.0f}s after "
                    "the babysitter cancelled it."
                ),
                diag=None,
                job_may_be_active=True,
            )
        if record.slurm_job_id:
            try:
                self.backend.cancel(record.slurm_job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Re-cancel of job %s failed: %s", record.slurm_job_id, exc)
        self.state.upsert_run(record)
        return record.status

    def _dispatch_failure(
        self,
        record: RunRecord,
        run_dir: Path,
        diag: DiagnosticReport,
        *,
        early_cancelled: bool,
        elapsed_s: float,
        scheduler_state: str,
    ) -> str:
        if self.dry_run:
            self.state.append_audit_event(
                "dry_run_failure_after_cancel_observed",
                record.run_id,
                {"failure_type": diag.failure_type, "scheduler_state": scheduler_state},
            )
            return record.status
        if diag.failure_type == "OUT_OF_MEMORY" and self._is_gcp() and record.gpu_tier != "a100":
            return self._handle_gcp_oom_tier_upgrade(record=record, run_dir=run_dir, diag=diag)
        return self._handle_physics_or_solver_failure(
            record=record,
            run_dir=run_dir,
            diag=diag,
            early_cancelled=early_cancelled,
            elapsed_s=elapsed_s,
            scheduler_state=scheduler_state,
        )

    # ------------------------------------------------------------------ automated recovery
    def _patch_restart_flag(self, run_dir: Path, restart: bool) -> None:
        solver, input_path = detect_solver_and_input_file(run_dir)
        if solver == "GX":
            patch_gx_toml(input_path, modifications={}, restart_from_checkpoint=restart)
        else:
            patch_gene_parameters(input_path, modifications={}, restart_from_checkpoint=restart)

    def _handle_walltime_or_spot_continuation(
        self,
        *,
        record: RunRecord,
        run_dir: Path,
        diag: DiagnosticReport,
        scheduler_state: str,
    ) -> str:
        """Resume after TIMEOUT / Spot PREEMPTED / NODE_FAIL / BOOT_FAIL with ZERO LLM calls."""
        if self.dry_run:
            self.state.append_audit_event(
                "dry_run_walltime_or_spot_continuation",
                record.run_id,
                {"scheduler_state": scheduler_state},
            )
            return record.status

        if diag.failure_type is not None:
            return self._handle_physics_or_solver_failure(
                record=record,
                run_dir=run_dir,
                diag=diag,
                early_cancelled=False,
                elapsed_s=0.0,
                scheduler_state=scheduler_state,
            )

        if record.walltime_resubmits >= self.max_walltime_resubmits:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="STALL",
                reason=(
                    f"Exceeded max_walltime_resubmits ({self.max_walltime_resubmits}) on "
                    f"{scheduler_state}."
                ),
                diag=diag,
            )

        has_checkpoint = bool(find_checkpoint_files(run_dir, record.solver))
        if not has_checkpoint and scheduler_state == "TIMEOUT":
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type="STALL",
                reason=(
                    "Walltime TIMEOUT before any checkpoint was written; a cold re-run would time "
                    "out at the same point. Increase walltime or checkpoint frequency."
                ),
                diag=diag,
            )

        self._patch_restart_flag(run_dir, has_checkpoint)
        consume_stale_diagnostics_metrics(run_dir)
        record.walltime_resubmits += 1
        kind = {
            "TIMEOUT": "walltime_continuation",
            "PREEMPTED": "spot_preemption_continuation",
        }.get(scheduler_state, "infrastructure_continuation")
        self.begin_submission(
            record,
            kind=kind,
            force_high_memory=(record.gpu_tier == "a100"),
            restart_from_checkpoint=has_checkpoint,
            scheduler_state=scheduler_state,
        )
        new_job_id = self.execute_submission(record, run_dir)
        if new_job_id is None:
            return record.status
        self.state.append_audit_event(
            "walltime_or_spot_resubmitted",
            record.run_id,
            {
                "scheduler_state": scheduler_state,
                "new_job_id": new_job_id,
                "walltime_resubmits": record.walltime_resubmits,
                "restart_from_checkpoint": has_checkpoint,
                "llm_called": False,
            },
        )
        self._emit(
            "walltime_resubmitted",
            record,
            {
                "slurm_job_id": new_job_id,
                "slurm_state": scheduler_state,
                "walltime_resubmits": record.walltime_resubmits,
                "geometry_id": record.equilibrium_id,
            },
        )
        return record.status

    def _handle_gcp_oom_tier_upgrade(
        self,
        *,
        record: RunRecord,
        run_dir: Path,
        diag: DiagnosticReport,
    ) -> str:
        """Upgrade GCP Cloud Batch GX run from 24 GB Spot L4 to 40 GB Spot A100 with 0 LLM calls."""
        has_ckpt = bool(find_checkpoint_files(run_dir, record.solver))
        solver, input_path = detect_solver_and_input_file(run_dir)
        if solver == "GX":
            patch_gx_toml(input_path, modifications={}, restart_from_checkpoint=has_ckpt)
        consume_stale_diagnostics_metrics(run_dir)

        record.gpu_tier = "a100"
        self.begin_submission(
            record,
            kind="oom_tier_upgrade",
            force_high_memory=True,
            restart_from_checkpoint=has_ckpt,
            scheduler_state="OUT_OF_MEMORY",
        )
        new_job_id = self.execute_submission(record, run_dir)
        if new_job_id is None:
            return record.status
        self.state.append_audit_event(
            "gcp_oom_upgraded_l4_to_a100",
            record.run_id,
            {"new_job_id": new_job_id, "restart_from_checkpoint": has_ckpt, "llm_called": False},
        )
        self._emit(
            "resubmitted",
            record,
            {
                "slurm_job_id": new_job_id,
                "gpu_tier": "a100",
                "note": "Auto-upgraded Spot L4 -> Spot A100 on OOM (0 LLM calls)",
                "geometry_id": record.equilibrium_id,
            },
        )
        return record.status

    def _handle_physics_or_solver_failure(
        self,
        *,
        record: RunRecord,
        run_dir: Path,
        diag: DiagnosticReport,
        early_cancelled: bool,
        elapsed_s: float,
        scheduler_state: str,
        failure_type: str | None = None,
    ) -> str:
        if failure_type is not None:
            diag.failure_type = failure_type
        failure_type = diag.failure_type or "UNKNOWN"
        if not record.active_failure_event_id:
            record.active_failure_event_id = str(uuid.uuid4())
            self.state.upsert_run(record)
            self._emit(
                "failed",
                record,
                {
                    "failure_type": failure_type,
                    "slurm_state": scheduler_state,
                    "early_cancelled": early_cancelled,
                    "walltime_elapsed_s": round(elapsed_s, 2),
                    "metrics": diag.scalar_metrics,
                    "current_parameters": diag.current_parameters,
                    "geometry_id": record.equilibrium_id,
                    "log_tail": "\n".join(diag.log_tail[-10:]),
                },
                event_id=record.active_failure_event_id,
            )

        # Client-side 3-strike circuit breaker: 0 LLM calls when retry_count >= max_retries
        if record.retry_count >= self.max_retries:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=failure_type,
                reason=(
                    f"3-strike circuit breaker reached locally (retry_count={record.retry_count} "
                    f">= max_retries={self.max_retries}); escalating with 0 extra LLM calls."
                ),
                diag=diag,
            )

        if self.client is None:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=failure_type,
                reason="AuroraSteerClient is not configured; escalating safely.",
                diag=diag,
            )

        triage_failure_type = failure_type if failure_type in TRIAGE_FAILURE_TYPES else "CRASH"
        summary_payload = _fit_triage_payload_under_2kb(
            {
                "schema_version": "1.0",
                "workstream": record.workstream,
                "backend": record.backend,
                "skill_name": str(self.config.get("skill_name", "sim-babysitter")),
                "run_id": record.run_id,
                "campaign_id": record.campaign_id,
                "solver": record.solver,
                "fidelity": record.fidelity,
                "environment": self.environment,
                "failure_type": triage_failure_type,
                "retry_count": record.retry_count,
                "walltime_resubmits": record.walltime_resubmits,
                "sim_time_reached": diag.sim_time_reached,
                "sim_time_target": diag.sim_time_target,
                "current_parameters": diag.current_parameters,
                "scalar_metrics": diag.scalar_metrics,
                "log_tail": diag.log_tail,
                "previous_directives": record.directives_history,
            }
        )

        idem_key = f"{record.run_id}:{record.active_failure_event_id}"
        try:
            directive = self.client.triage(summary_payload, idempotency_key=idem_key)
        except TriageTransientError as exc:
            record.triage_deferrals += 1
            self.state.upsert_run(record)
            self.state.append_audit_event(
                "triage_deferred",
                record.run_id,
                {"error": str(exc)[:300], "deferrals": record.triage_deferrals},
            )
            if record.triage_deferrals > self.max_triage_deferrals:
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type=failure_type,
                    reason=(
                        f"aurora-steer unreachable for {record.triage_deferrals} consecutive "
                        f"ticks ({str(exc)[:200]}); escalating without resubmitting."
                    ),
                    diag=diag,
                )
            # Job is terminal and nothing was changed: retry the same idempotency key next tick.
            return record.status
        except TriageClientError as exc:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=failure_type,
                reason=f"Triage request failed ({str(exc)[:300]}); escalating safely without blind retry.",
                diag=diag,
            )
        record.triage_deferrals = 0

        record.directives_history.append(directive)
        self._emit(
            "directive_received",
            record,
            {
                "action": directive.get("action"),
                "reason": directive.get("reason", ""),
                "failure_type": failure_type,
                "geometry_id": record.equilibrium_id,
            },
        )

        try:
            validated_mods = validate_directive_client_side(directive, self.allowed_parameters)
        except ClientGuardrailError as exc:
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=failure_type,
                reason=f"Client guardrail rejected directive: {exc}",
                diag=diag,
            )

        action = directive.get("action")
        if action == "escalate_to_human":
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=failure_type,
                reason=str(directive.get("reason", "Escalated by aurora-steer"))[:300],
                diag=diag,
            )

        restart_ckpt = directive.get("restart_from_checkpoint", False) is True
        downgraded = False
        if restart_ckpt and not find_checkpoint_files(run_dir, record.solver):
            # Only timestep parameters can reach here (PHYS-03), so starting over from t=0 with
            # them is always valid; resuming without a checkpoint would just crash the job.
            restart_ckpt = False
            downgraded = True
        before_params = dict(diag.current_parameters)
        before_sha = self._input_sha256(run_dir)
        try:
            solver, input_path = detect_solver_and_input_file(run_dir)
            if solver == "GX":
                after_params = patch_gx_toml(
                    input_path, modifications=validated_mods, restart_from_checkpoint=restart_ckpt
                )
            else:
                after_params = patch_gene_parameters(
                    input_path, modifications=validated_mods, restart_from_checkpoint=restart_ckpt
                )
        except Exception as exc:  # noqa: BLE001 - patchers are atomic; the input is unchanged
            return self._escalate_run(
                record=record,
                run_dir=run_dir,
                failure_type=failure_type,
                reason=f"Applying directive {validated_mods} failed: {str(exc)[:300]}",
                diag=diag,
            )

        record.parameter_patches.append(
            {
                "strike": record.retry_count + 1,
                "action": action,
                "failure_type": failure_type,
                "modifications": validated_mods,
                "before": before_params,
                "after": after_params,
                "restart_from_checkpoint": restart_ckpt,
                "warm_restart_downgraded_no_checkpoint": downgraded,
                "directive_reason": str(directive.get("reason", ""))[:300],
                "okf_concepts_used": list(directive.get("okf_concepts_used", []) or [])[:10],
                "input_sha256_before": before_sha,
                "input_sha256_after": self._input_sha256(run_dir),
                "applied_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        self._emit(
            "patched",
            record,
            {
                "action": action,
                "current_parameters": after_params,
                "geometry_id": record.equilibrium_id,
            },
        )

        consume_stale_diagnostics_metrics(run_dir)
        record.retry_count += 1
        record.active_failure_event_id = None

        kind = {
            "warm_restart": "patch_warm_restart",
            "cold_restart": "patch_cold_restart",
        }.get(str(action), "resubmit")
        self.begin_submission(
            record,
            kind=kind,
            force_high_memory=(record.gpu_tier == "a100"),
            restart_from_checkpoint=restart_ckpt,
            scheduler_state=scheduler_state,
        )
        new_job_id = self.execute_submission(record, run_dir)
        if new_job_id is None:
            return record.status
        self.state.append_audit_event(
            "patched_and_resubmitted",
            record.run_id,
            {
                "action": action,
                "modifications": validated_mods,
                "new_job_id": new_job_id,
                "retry_count": record.retry_count,
                "restart_from_checkpoint": restart_ckpt,
            },
        )
        self._emit(
            "resubmitted",
            record,
            {
                "slurm_job_id": new_job_id,
                "action": action,
                "current_parameters": after_params,
                "geometry_id": record.equilibrium_id,
            },
        )
        return record.status

    # ------------------------------------------------------------------ terminal transitions
    def _complete_run(
        self,
        *,
        record: RunRecord,
        run_dir: Path,
        diag: DiagnosticReport,
        elapsed_s: float,
    ) -> str:
        if self.dry_run:
            self.state.append_audit_event("dry_run_completed_observed", record.run_id, {})
            return record.status

        dest_dir = self.completed_dir / record.run_id
        if run_dir.resolve() != dest_dir.resolve():
            if dest_dir.exists():
                return self._escalate_run(
                    record=record,
                    run_dir=run_dir,
                    failure_type="CRASH",
                    reason=f"{dest_dir} already exists; refusing to overwrite a completed run.",
                    diag=diag,
                )
            shutil.move(str(run_dir), str(dest_dir))
            record.run_dir = str(dest_dir)
            self.state.upsert_run(record)  # location is durable before the harvest

        record.active_failure_event_id = None
        if self.dataset_output_dir is not None:
            record.status = "COMPLETED"  # the harvester checks the final status
            try:
                manifest_path = harvest_completed_run_to_dataset(
                    record=record,
                    run_dir=dest_dir,
                    dataset_root=self.dataset_output_dir,
                    final_parameters=diag.current_parameters,
                    final_metrics={**diag.scalar_metrics, **diag.fluxes},
                    walltime_elapsed_s=elapsed_s,
                    gcs_uri_prefix=self.config.get("dataset_gcs_uri_prefix"),
                    converged=diag.converged,
                    convergence_method=diag.convergence_method,
                    simulated=diag.simulated,
                    environment=self.environment,
                    flux_units=diag.flux_units,
                    code_version=diag.code_version or record.code_version,
                )
            except DatasetEntryExistsError as exc:
                record.status = "RUNNING"
                return self._escalate_run(
                    record=record,
                    run_dir=dest_dir,
                    failure_type="CRASH",
                    reason=f"Dataset entry already exists for this run_id: {exc}",
                    diag=diag,
                )
            except Exception:
                record.status = "RUNNING"  # retried next tick; bounded by max_poll_errors
                raise
            record.dataset_manifest_path = str(manifest_path)

        record.status = "COMPLETED"
        record.run_dir = str(dest_dir)
        self.state.upsert_run(record)
        self.state.append_audit_event(
            "run_completed",
            record.run_id,
            {
                "retry_count": record.retry_count,
                "walltime_resubmits": record.walltime_resubmits,
                "dataset_manifest_path": record.dataset_manifest_path,
                "convergence_method": diag.convergence_method,
            },
        )
        self._emit(
            "completed",
            record,
            {
                "metrics": {**diag.scalar_metrics, **diag.fluxes},
                "current_parameters": diag.current_parameters,
                "geometry_id": record.equilibrium_id,
                "gpu_tier": record.gpu_tier,
                "walltime_elapsed_s": round(elapsed_s, 2),
            },
        )
        return record.status

    def escalate_safely(self, record: RunRecord, reason: str, *, job_may_be_active: bool) -> None:
        """Escalate without ever raising; if even that fails, stop polling the run."""
        try:
            self._escalate_run(
                record=record,
                run_dir=Path(record.run_dir),
                failure_type="CRASH",
                reason=reason,
                diag=None,
                job_may_be_active=job_may_be_active,
            )
        except Exception as exc:  # noqa: BLE001
            logger.critical(
                "Escalation of run %s failed (%s); marking NEEDS_HUMAN in place", record.run_id, exc
            )
            record.status = "NEEDS_HUMAN"
            record.pending_submission = None
            try:
                self.state.upsert_run(record)
            except Exception:  # noqa: BLE001
                logger.critical("Could not persist NEEDS_HUMAN for run %s", record.run_id)

    def _escalate_run(
        self,
        *,
        record: RunRecord,
        run_dir: Path,
        failure_type: str,
        reason: str,
        diag: DiagnosticReport | None,
        job_may_be_active: bool = False,
    ) -> str:
        if self.dry_run:
            self.state.append_audit_event(
                "dry_run_would_escalate",
                record.run_id,
                {"failure_type": failure_type, "reason": reason[:300]},
            )
            return record.status

        notes: list[str] = []
        if job_may_be_active and record.slurm_job_id:
            try:
                self.backend.cancel(record.slurm_job_id)
                notes.append(
                    f"Scheduler job {record.slurm_job_id} was cancelled before escalation."
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    f"WARNING: cancelling job {record.slurm_job_id} failed ({str(exc)[:200]}); it "
                    "may still be running. Check the scheduler before re-queueing."
                )

        dest_dir = self.needs_human_dir / record.run_id
        try:
            if run_dir.exists() and run_dir.resolve() != dest_dir.resolve():
                if dest_dir.exists():
                    dest_dir = self._unique_sibling(dest_dir)
                    notes.append(
                        f"needs_human/{record.run_id} already existed; moved to {dest_dir.name}."
                    )
                shutil.move(str(run_dir), str(dest_dir))
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception("Could not move %s to needs_human/: %s", run_dir, exc)
            notes.append(
                f"WARNING: could not move the run folder ({exc}); it is still at {run_dir}."
            )
            dest_dir = run_dir if run_dir.exists() else dest_dir
            dest_dir.mkdir(parents=True, exist_ok=True)

        full_reason = " ".join([reason, *notes])
        issue_text = ""
        try:
            issue_text = _write_issue_markdown(
                target_dir=dest_dir,
                record=record,
                failure_type=failure_type,
                reason=full_reason,
                diag=diag,
            )
        except OSError as exc:
            logger.exception("Could not write ISSUE.md for run %s: %s", record.run_id, exc)

        record.status = "NEEDS_HUMAN"
        record.run_dir = str(dest_dir)
        record.escalation_count += 1
        record.active_failure_event_id = None
        record.pending_submission = None
        record.cancel_requested = None
        self.state.upsert_run(record)
        self.state.append_audit_event(
            "escalated_to_human",
            record.run_id,
            {"failure_type": failure_type, "reason": full_reason[:500]},
        )
        self._emit(
            "escalated",
            record,
            {
                "failure_type": failure_type,
                "reason": full_reason[:200],
                "tacc_folder_path": f"queue/needs_human/{dest_dir.name}",
                "geometry_id": record.equilibrium_id,
                "directives_tried": [
                    {"action": d.get("action"), "mods": d.get("parameter_modifications", {})}
                    for d in record.directives_history[-3:]
                ],
                "issue_md_text": issue_text[:280],
            },
        )
        send_escalation_notification(
            run_id=record.run_id,
            solver=record.solver,
            reason=full_reason,
            needs_human_dir=dest_dir,
            diagnostic={"failure_type": failure_type, "retry_count": record.retry_count},
            webhook_url=self.config.get("notify_webhook_url"),
        )
        return record.status


def configure_cli_logging() -> None:
    """Timestamped log lines for cron logs (idempotent)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to run a single watchdog pass over active Slurm runs (no submissions)."""
    from tacc_exec.config import ConfigError, load_config

    parser = argparse.ArgumentParser(description="Poll active Slurm simulations once.")
    parser.add_argument("--config", required=True, help="Path to executor config JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="Observe without mutating or calling /triage"
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
            wd = SimulationWatchdog(
                backend=SlurmBackend(dry_run=args.dry_run),
                state_store=store,
                steer_client=client,
                telemetry=telemetry,
                config=cfg,
                dry_run=args.dry_run,
                heartbeat=lock.refresh,
            )
            wd.poll_all()
            if not args.dry_run:
                telemetry.flush()
    except DriverAlreadyRunningError as exc:
        logger.info("Another executor instance is active; skipping this tick (%s)", exc)
        return 0
    except (StateCorruptError, StateSchemaError) as exc:
        logger.error("Refusing to run: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
