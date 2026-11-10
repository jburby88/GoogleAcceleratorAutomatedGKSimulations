"""Slurm CLI wrappers for job submission, monitoring, and cancellation.

Executes sbatch, squeue, sacct, and scancel with timeouts and argument validation,
handling job deduplication and state aggregation across tasks.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpc_core.protocol import (
    ACTIVE_STATES,
    INFRA_STATES,
    KNOWN_JOB_STATES,
    TERMINAL_STATES,
    TRANSIENT_STATES,
    BackendCapabilities,
    JobStatus,
)

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
# `sbatch --parsable` prints "<jobid>" or "<jobid>;<cluster>".
_PARSABLE_LINE_RE = re.compile(r"^\s*(\d+)(?:;[^\s;]+)?\s*$")
# Job ids accepted by query/cancel: numeric, optionally an array task or a het-job component.
_SLURM_JOB_ID_RE = re.compile(r"^\d+(?:_\d+|\+\d+)?$")
_DRY_RUN_JOB_ID_RE = re.compile(r"^dryrun-\d+$")
_SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_LEADING_JOB_ID_RE = re.compile(r"^(\d+)")
_CANCELLED_BY_RE = re.compile(r"^CANCELLED\+?\s+by\s+(\d+)", re.IGNORECASE)

MAX_ERROR_TEXT_CHARS = 300
SUBMISSION_LOOKBACK = "now-30days"

KNOWN_SLURM_STATES: frozenset[str] = KNOWN_JOB_STATES
SlurmJobStatus = JobStatus

_SHORT_STATE_ALIASES: dict[str, str] = {
    "R": "RUNNING",
    "PD": "PENDING",
    "CG": "COMPLETING",
    "CF": "CONFIGURING",
    "S": "SUSPENDED",
    "ST": "STOPPED",
    "RQ": "REQUEUED",
    "RF": "REQUEUED",
    "RH": "PENDING",
    "SE": "PENDING",
    "RD": "PENDING",
    "CD": "COMPLETED",
    "TO": "TIMEOUT",
    "F": "FAILED",
    "CA": "CANCELLED",
    "OOM": "OUT_OF_MEMORY",
    "NF": "NODE_FAIL",
    "PR": "PREEMPTED",
    "BF": "BOOT_FAIL",
    "DL": "DEADLINE",
    "SO": "STAGE_OUT",
    "RS": "RESIZING",
    "SI": "SIGNALING",
    "RV": "REVOKED",
}
_LONG_STATE_ALIASES: dict[str, str] = {
    "REQUEUE_FED": "REQUEUED",
    "REQUEUE_HOLD": "PENDING",
    "SPECIAL_EXIT": "PENDING",
    "RESV_DEL_HOLD": "PENDING",
    "UPDATE_DB": "CONFIGURING",
    "POWER_UP_NODE": "CONFIGURING",
}
# Raw states that keep a job queued but held until a human releases it.
_BLOCKED_RAW_STATES = frozenset({"REQUEUE_HOLD", "RH", "SPECIAL_EXIT", "SE", "RESV_DEL_HOLD", "RD"})

_TRANSIENT_PRIORITY: tuple[str, ...] = (
    "COMPLETING",
    "STAGE_OUT",
    "CONFIGURING",
    "SIGNALING",
    "RESIZING",
    "SUSPENDED",
    "STOPPED",
    "REQUEUED",
    "PENDING",
    "REVOKED",
)
_TERMINAL_SEVERITY: tuple[str, ...] = (
    "CANCELLED",
    "NODE_FAIL",
    "BOOT_FAIL",
    "OUT_OF_MEMORY",
    "FAILED",
    "DEADLINE",
    "TIMEOUT",
    "PREEMPTED",
    "COMPLETED",
)

# Pending reasons (squeue %r, see the squeue man page "JOB REASON CODES" and
# https://slurm.schedmd.com/resource_limits.html) that never clear on their own.
_BLOCKED_REASONS_EXACT = frozenset(
    {
        "InvalidAccount",
        "InvalidQOS",
        "PartitionConfig",
        "PartitionTimeLimit",
        "BadConstraints",
        "DependencyNeverSatisfied",
        "JobLaunchFailure",
    }
)
_BLOCKED_REASON_PATTERNS: tuple[re.Pattern[str], ...] = (
    # The job itself asks for more than a per-job / per-node maximum allows.
    re.compile(r"^(?:Assoc|QOS)Max\w*Per(?:Job|Node)\w*$"),
    # Usage-based (decayed) group limits: allocation minutes / wall time are exhausted.
    re.compile(r"^(?:Assoc|QOS)Grp\w*Minutes\w*$"),
    re.compile(r"^(?:Assoc|QOS)GrpWall\w*$"),
    # The job asks for less than a QOS minimum.
    re.compile(r"^QOSMin\w*$"),
)
_HELD_REASON_RE = re.compile(r"^JobH(?:eld|old)|\bheld\b", re.IGNORECASE)

# sbatch failures where slurmctld may have queued the job although sbatch reported an error.
_AMBIGUOUS_SBATCH_ERRORS = (
    "socket timed out",
    "timed out",
    "unable to contact slurm controller",
    "zero bytes were transmitted",
    "connection refused",
    "communication connection failure",
    "slurm_receive_msg",
)


class SlurmError(RuntimeError):
    """Base exception for Slurm CLI invocation failures."""


class SlurmSubmitError(SlurmError):
    """Raised when sbatch fails, times out, or returns unparseable output.

    `outcome_unknown` is True when a job may nevertheless exist for this submission (sbatch
    timed out or lost its reply, the output was unparseable, or the duplicate check could not
    run). A caller using submission ids must then reconcile with `find_submission()` (or call
    `submit()` again with the same id) instead of assuming that nothing was submitted.
    """

    def __init__(self, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.outcome_unknown = bool(outcome_unknown)


class SlurmCancelError(SlurmError):
    """Raised when scancel fails or times out."""


class SlurmQueryError(SlurmError):
    """Raised when squeue and sacct cannot answer (or an answer is ambiguous)."""


def _current_uid() -> int:
    """Unix uid of this executor (Slurm's "CANCELLED by <uid>" is compared against it)."""
    return os.getuid()


def _truncate(text: str, limit: int = MAX_ERROR_TEXT_CHARS) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _classify_state(raw: str | None) -> tuple[str, bool, int | None]:
    """Return (canonical state, held by state, canceller uid or None) for a squeue/sacct state."""
    text = (raw or "").strip()
    if not text:
        return "UNKNOWN", False, None
    token = text.split()[0].rstrip("+").upper()
    if token.startswith("CANCELLED"):
        match = _CANCELLED_BY_RE.match(text)
        return "CANCELLED", False, int(match.group(1)) if match else None
    canonical = _LONG_STATE_ALIASES.get(token) or _SHORT_STATE_ALIASES.get(token) or token
    if canonical not in KNOWN_JOB_STATES:
        return "UNKNOWN", False, None
    return canonical, token in _BLOCKED_RAW_STATES, None


def normalize_slurm_state(raw: str) -> str:
    """Normalize a raw squeue/sacct state token into a canonical KNOWN_JOB_STATES member."""
    return _classify_state(raw)[0]


def is_hopeless_pending_reason(reason: str | None) -> bool:
    """True if a Slurm pending reason means the job can never start without human action."""
    text = (reason or "").strip().strip("()").strip()
    if not text or text == "None":
        return False
    if _HELD_REASON_RE.search(text):
        return True  # JobHeldUser, JobHeldAdmin, JobHoldMaxRequeue, "launch failed requeued held"
    first = text.split(",", 1)[0].strip()  # e.g. "ReqNodeNotAvail, Reserved for maintenance"
    if first in _BLOCKED_REASONS_EXACT:
        return True
    if "RunMinutes" in first:
        return False  # limits on *running* jobs clear as those jobs finish
    return any(pattern.match(first) for pattern in _BLOCKED_REASON_PATTERNS)


def slurm_job_name_for_submission(submission_id: str) -> str:
    """Deterministic Slurm job name for a submission id: `[A-Za-z0-9_-]`, at most 57 chars.

    The readable prefix helps operators in `squeue`; the sha256 suffix keeps distinct ids
    distinct and keeps the name within the conservative charset that site submit filters
    (e.g. TACC's "Verifying valid jobname" check) accept.
    """
    if not isinstance(submission_id, str) or not _SUBMISSION_ID_RE.fullmatch(submission_id):
        raise ValueError(
            f"Invalid submission_id {submission_id!r}: expected 1-160 characters of "
            "[A-Za-z0-9_.-] starting with a letter or digit"
        )
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", submission_id)[:40]
    digest = hashlib.sha256(submission_id.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}"


def find_batch_script(run_dir: Path) -> Path:
    """Locate the Slurm batch submission script inside a run folder."""
    for name in ("submit.slurm", "job.slurm", "run.slurm"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    slurm_files = sorted(run_dir.glob("*.slurm"))
    if slurm_files:
        return slurm_files[0]
    raise FileNotFoundError(f"No .slurm batch script found in {run_dir}")


def _parse_slurm_elapsed(raw: str) -> float:
    """Convert a Slurm duration ([DD-][HH:]MM:SS[.fff]) to seconds; 0.0 if unparseable."""
    try:
        text = (raw or "").strip()
        if not text:
            return 0.0
        days = 0
        if "-" in text:
            day_part, text = text.split("-", 1)
            days = int(day_part)
        parts = text.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), float(parts[1])
        else:
            return 0.0
        total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (ValueError, OverflowError):
        return 0.0
    return float(total) if math.isfinite(total) and total >= 0 else 0.0


def _parse_sbatch_job_id(stdout: str, stderr: str) -> str | None:
    """Job id from `sbatch --parsable` output (last matching line), else the classic message."""
    for line in reversed((stdout or "").splitlines()):
        match = _PARSABLE_LINE_RE.match(line)
        if match:
            return match.group(1)
    match = _JOB_ID_RE.search(f"{stdout or ''}\n{stderr or ''}")
    return match.group(1) if match else None


def _base_job_id(job_id: str) -> str | None:
    match = _LEADING_JOB_ID_RE.match(job_id.strip())
    return match.group(1) if match else None


def _distinct_base_ids(stdout: str) -> set[str]:
    found: set[str] = set()
    for line in (stdout or "").splitlines():
        base = _base_job_id(line)
        if base:
            found.add(base)
    return found


def _clean_reason(raw: str | None) -> str | None:
    text = (raw or "").strip().strip("()").strip()
    if not text or text == "None":
        return None
    return _truncate(text, 200)


@dataclass(frozen=True)
class _Row:
    state: str
    elapsed: float
    exit_code: str
    reason: str | None
    blocked: bool
    canceller_uid: int | None


@dataclass(frozen=True)
class _CmdResult:
    ok: bool
    stdout: str
    error: str
    missing_binary: bool = False


def _is_pending_like(state: str) -> bool:
    return state not in ACTIVE_STATES and state not in TERMINAL_STATES


def _parse_squeue_rows(stdout: str) -> list[_Row]:
    """Rows of `squeue -h -o '%T|%M|%r'` (state | time used | reason)."""
    rows: list[_Row] = []
    for line in (stdout or "").splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        state, held, canceller = _classify_state(parts[0])
        reason = _clean_reason(parts[2]) if len(parts) > 2 else None
        blocked = _is_pending_like(state) and (held or is_hopeless_pending_reason(reason))
        rows.append(
            _Row(
                state=state,
                elapsed=_parse_slurm_elapsed(parts[1]) if len(parts) > 1 else 0.0,
                exit_code="",
                reason=reason,
                blocked=blocked,
                canceller_uid=canceller,
            )
        )
    return rows


def _parse_sacct_rows(stdout: str, job_id: str) -> list[_Row]:
    """Rows of `sacct -n -X -P -o JobID,State,Elapsed,ExitCode` that belong to `job_id`."""
    wanted = _base_job_id(job_id)
    rows: list[_Row] = []
    for line in (stdout or "").splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 2:
            continue
        row_job_id = fields[0].strip()
        if "." in row_job_id:
            continue  # job steps (.batch, .extern, .0) never decide the job state
        if wanted is not None and _base_job_id(row_job_id) != wanted:
            continue
        state, held, canceller = _classify_state(fields[1])
        reason = f"cancelled by uid={canceller}" if canceller is not None else None
        rows.append(
            _Row(
                state=state,
                elapsed=_parse_slurm_elapsed(fields[2]) if len(fields) > 2 else 0.0,
                exit_code=fields[3].strip() if len(fields) > 3 else "",
                reason=reason,
                blocked=held and _is_pending_like(state),
                canceller_uid=canceller,
            )
        )
    return rows


def _rank(order: tuple[str, ...], state: str) -> int:
    return order.index(state) if state in order else len(order)


def _aggregate(job_id: str, rows: list[_Row]) -> JobStatus:
    """Reduce squeue/sacct rows of one job to a single JobStatus (rules in the module doc)."""
    states = [row.state for row in rows]
    if any(state in ACTIVE_STATES for state in states):
        agg = next(state for state in states if state in ACTIVE_STATES)
    elif any(state in TRANSIENT_STATES for state in states):
        agg = min(
            (state for state in states if state in TRANSIENT_STATES),
            key=lambda s: _rank(_TRANSIENT_PRIORITY, s),
        )
    elif any(state in INFRA_STATES for state in states):
        agg = "UNKNOWN"
    else:
        agg = min(states, key=lambda s: _rank(_TERMINAL_SEVERITY, s))

    my_uid = _current_uid()
    external = [
        row
        for row in rows
        if row.state == "CANCELLED"
        and row.canceller_uid is not None
        and row.canceller_uid != my_uid
    ]
    blocked_rows = [row for row in rows if row.blocked and _is_pending_like(row.state)]
    blocked = _is_pending_like(agg) and agg not in INFRA_STATES and bool(blocked_rows)

    representative = next(row for row in rows if row.state == agg)
    if agg == "CANCELLED" and external:
        representative = external[0]
    elif blocked:
        representative = blocked_rows[0]
    reason = representative.reason
    if agg == "CANCELLED" and reason is None:
        reason = "cancelled (canceller not reported)"

    return JobStatus(
        job_id=job_id,
        state=agg,
        elapsed_seconds=max(row.elapsed for row in rows),
        exit_code=representative.exit_code,
        reason=reason,
        blocked=blocked,
        cancelled_externally=agg == "CANCELLED" and bool(external),
    )


class SlurmBackend:
    """Subprocess wrapper around TACC Slurm utilities implementing SchedulerBackend."""

    capabilities = BackendCapabilities(
        backend_name="tacc_slurm",
        supports_gene=True,
        supports_gx=True,
        default_gpu_tier="tacc_gh200_or_cpu",
        supports_spot_continuation=False,
        idempotent_submit=True,
    )

    # TODO(tan): confirm any required TACC cluster flags or environment modules
    def __init__(self, dry_run: bool = False, timeout_s: int = 30) -> None:
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self._dry_run_counter = 900000
        self._dry_run_submissions: dict[str, str] = {}

    # -- helpers ---------------------------------------------------------------------------
    def _run(self, argv: list[str], *, cwd: Path | None = None) -> _CmdResult:
        """Run a read-only Slurm query command; never raises for command failures."""
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            return _CmdResult(False, "", f"{argv[0]} not found: {exc}", missing_binary=True)
        except subprocess.TimeoutExpired:
            return _CmdResult(False, "", f"{argv[0]} timed out after {self.timeout_s}s")
        except OSError as exc:
            return _CmdResult(False, "", f"{argv[0]} failed to execute: {exc}")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
            return _CmdResult(False, proc.stdout or "", _truncate(detail))
        return _CmdResult(True, proc.stdout or "", "")

    def _validate_job_id(self, job_id: str) -> None:
        if isinstance(job_id, str) and (
            _SLURM_JOB_ID_RE.fullmatch(job_id)
            or (self.dry_run and _DRY_RUN_JOB_ID_RE.fullmatch(job_id))
        ):
            return
        raise ValueError(f"Invalid Slurm job_id: {job_id!r}")

    # -- SchedulerBackend ------------------------------------------------------------------
    def submit(
        self,
        run_dir: Path,
        *,
        solver: str = "GX",
        parameters: dict[str, Any] | None = None,
        retry_count: int = 0,
        force_high_memory: bool = False,
        submission_id: str | None = None,
    ) -> str:
        """Locate the .slurm script in `run_dir` and submit it (idempotent per submission_id)."""
        _ = (solver, parameters, retry_count, force_high_memory)
        script_path = find_batch_script(run_dir)
        if submission_id is None:
            return self.sbatch(script_path, run_dir)

        job_name = slurm_job_name_for_submission(submission_id)
        try:
            existing = self.find_submission(submission_id)
        except SlurmQueryError as exc:
            raise SlurmSubmitError(
                f"Cannot verify that submission {submission_id} has no job yet; not "
                f"submitting: {exc}",
                outcome_unknown=True,
            ) from exc
        if existing is not None:
            logger.warning(
                "Adopting existing Slurm job %s for submission %s (no new sbatch)",
                existing,
                submission_id,
            )
            return existing

        try:
            job_id = self.sbatch(script_path, run_dir, job_name=job_name)
        except SlurmSubmitError as exc:
            if not exc.outcome_unknown:
                raise
            # sbatch may have queued the job although it reported a failure (timeout, lost
            # reply from slurmctld). Look for it before reporting the failure.
            try:
                adopted = self.find_submission(submission_id)
            except SlurmQueryError as query_exc:
                raise SlurmSubmitError(
                    f"{exc}; follow-up lookup failed: {query_exc}", outcome_unknown=True
                ) from exc
            if adopted is not None:
                logger.warning(
                    "sbatch reported a failure but job %s exists for submission %s; adopting it",
                    adopted,
                    submission_id,
                )
                return adopted
            raise
        if self.dry_run:
            self._dry_run_submissions[submission_id] = job_id
        return job_id

    def cancel(self, job_id: str) -> None:
        """SchedulerBackend entrypoint: cancel active Slurm job."""
        self.scancel(job_id)

    def sbatch(self, script_path: Path, cwd: Path, *, job_name: str | None = None) -> str:
        """Submit a batch script via `sbatch --parsable` and return the Slurm job ID."""
        resolved_script = script_path.resolve()
        resolved_cwd = cwd.resolve()
        if not resolved_script.is_file():
            raise FileNotFoundError(f"Slurm batch script not found: {resolved_script}")

        argv = ["sbatch", "--parsable"]
        if job_name is not None:
            argv.append(f"--job-name={job_name}")
        argv.append(str(resolved_script))

        if self.dry_run:
            self._dry_run_counter += 1
            fake_id = f"dryrun-{self._dry_run_counter}"
            logger.info(
                "[DRY-RUN] Would run: %s %s (in %s) -> %s",
                " ".join(argv[:-1]),
                resolved_script.name,
                resolved_cwd,
                fake_id,
            )
            return fake_id

        try:
            proc = subprocess.run(
                argv,
                cwd=str(resolved_cwd),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SlurmSubmitError(
                f"sbatch timed out after {self.timeout_s}s; the job may or may not be queued",
                outcome_unknown=True,
            ) from exc
        except OSError as exc:
            raise SlurmSubmitError(f"Failed to execute sbatch: {exc}") from exc

        job_id = _parse_sbatch_job_id(proc.stdout or "", proc.stderr or "")
        if proc.returncode != 0:
            detail = _truncate((proc.stderr or proc.stdout or "").strip())
            ambiguous = job_id is not None or any(
                marker in detail.lower() for marker in _AMBIGUOUS_SBATCH_ERRORS
            )
            raise SlurmSubmitError(
                f"sbatch failed with exit code {proc.returncode}: {detail}",
                outcome_unknown=ambiguous,
            )
        if job_id is None:
            raise SlurmSubmitError(
                f"Could not parse job ID from sbatch output: {_truncate(proc.stdout or '')!r}",
                outcome_unknown=True,
            )
        return job_id

    def find_submission(self, submission_id: str) -> str | None:
        """Return the job id queued for `submission_id`, or None if Slurm has no such job.

        Raises SlurmQueryError when absence cannot be proven (squeue or sacct failed) or when
        several jobs carry the submission's name (a human must decide).
        """
        job_name = slurm_job_name_for_submission(submission_id)
        if self.dry_run:
            return self._dry_run_submissions.get(submission_id)

        uid = str(_current_uid())
        squeue = self._run(
            ["squeue", f"--user={uid}", f"--name={job_name}", "--states=all", "-h", "-o", "%i"]
        )
        found = _distinct_base_ids(squeue.stdout) if squeue.ok else set()
        if len(found) > 1:
            raise SlurmQueryError(
                f"Several Slurm jobs {sorted(found)} carry submission name {job_name}; "
                "refusing to choose one"
            )
        if found:
            return found.pop()

        sacct = self._run(
            [
                "sacct",
                f"--user={uid}",
                f"--name={job_name}",
                "-X",
                "-n",
                "-P",
                "-o",
                "JobID",
                f"--starttime={SUBMISSION_LOOKBACK}",
            ]
        )
        if not sacct.ok:
            squeue_note = "" if squeue.ok else f"; squeue also failed: {squeue.error}"
            raise SlurmQueryError(f"sacct lookup of {job_name} failed: {sacct.error}{squeue_note}")
        found = _distinct_base_ids(sacct.stdout)
        if len(found) > 1:
            raise SlurmQueryError(
                f"Several Slurm jobs {sorted(found)} carry submission name {job_name}; "
                "refusing to choose one"
            )
        if found:
            return found.pop()
        if not squeue.ok:
            raise SlurmQueryError(
                f"squeue lookup of {job_name} failed ({squeue.error}) and sacct has no record "
                "yet; cannot prove that the job does not exist"
            )
        return None

    def scancel(self, job_id: str) -> None:
        """Cancel a Slurm job via scancel. A job that already finished counts as cancelled."""
        self._validate_job_id(job_id)

        if self.dry_run:
            logger.info("[DRY-RUN] Would run: scancel %s", job_id)
            return

        result = self._run(["scancel", job_id])
        if result.ok:
            return
        lowered = result.error.lower()
        if "already completing or completed" in lowered or "invalid job id" in lowered:
            logger.info("scancel %s: job is no longer active (%s)", job_id, result.error)
            return
        raise SlurmCancelError(f"scancel {job_id} failed: {result.error}")

    def query_job(self, job_id: str) -> JobStatus:
        """Query job status via squeue, falling back to (or confirming with) sacct."""
        self._validate_job_id(job_id)

        if self.dry_run and job_id.startswith("dryrun-"):
            return JobStatus(job_id=job_id, state="RUNNING", elapsed_seconds=120.0)

        squeue = self._run(["squeue", f"--jobs={job_id}", "-h", "-o", "%T|%M|%r"])
        if squeue.missing_binary and self.dry_run:
            return JobStatus(job_id=job_id, state="RUNNING", elapsed_seconds=120.0)
        # Older Slurm exits non-zero with "Invalid job id specified" once a job left memory.
        squeue_ok = squeue.ok or "invalid job id" in squeue.error.lower()
        squeue_rows = _parse_squeue_rows(squeue.stdout) if squeue.ok else []
        if squeue_rows:
            status = _aggregate(job_id, squeue_rows)
            if status.state not in TERMINAL_STATES:
                return status

        # Terminal or not in the queue: sacct is authoritative (exit code, who cancelled).
        sacct = self._run(
            ["sacct", f"--jobs={job_id}", "-n", "-X", "-P", "-o", "JobID,State,Elapsed,ExitCode"]
        )
        if sacct.missing_binary and self.dry_run:
            return JobStatus(job_id=job_id, state="RUNNING", elapsed_seconds=120.0)
        sacct_rows = _parse_sacct_rows(sacct.stdout, job_id) if sacct.ok else []
        if sacct_rows:
            return _aggregate(job_id, sacct_rows)

        if squeue_rows:
            status = _aggregate(job_id, squeue_rows)
            if status.state == "CANCELLED":
                # squeue does not say who cancelled; never let an unattributed cancel look
                # like our own. Wait for the accounting record.
                return JobStatus(
                    job_id=job_id,
                    state="UNKNOWN",
                    elapsed_seconds=status.elapsed_seconds,
                    reason="CANCELLED in squeue; waiting for the sacct record to identify the canceller",
                )
            return status

        if not squeue_ok and not sacct.ok:
            raise SlurmQueryError(
                f"squeue and sacct both failed for job {job_id}: squeue: {squeue.error}; "
                f"sacct: {sacct.error}"
            )
        detail = "" if sacct.ok else f" (sacct failed: {sacct.error})"
        return JobStatus(
            job_id=job_id,
            state="UNKNOWN",
            reason=f"job not found in squeue or sacct (accounting lag or purged record){detail}",
        )


SlurmCLI = SlurmBackend

__all__ = [
    "ACTIVE_STATES",
    "INFRA_STATES",
    "KNOWN_SLURM_STATES",
    "TERMINAL_STATES",
    "TRANSIENT_STATES",
    "SlurmBackend",
    "SlurmCLI",
    "SlurmCancelError",
    "SlurmError",
    "SlurmJobStatus",
    "SlurmQueryError",
    "SlurmSubmitError",
    "find_batch_script",
    "is_hopeless_pending_reason",
    "normalize_slurm_state",
    "slurm_job_name_for_submission",
]
