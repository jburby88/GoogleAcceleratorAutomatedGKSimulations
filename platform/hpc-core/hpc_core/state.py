"""Resumable local state store (atomic JSON), single-instance lock, and JSONL audit logger."""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpc_core.fsutil import append_line_durable, atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)

# v1: original layout (no schema_version key). v2: adds submission intent, restart history,
# persisted poll-error/deferral counters and provenance fields to RunRecord.
STATE_SCHEMA_VERSION = 2

# Default for how long a lease written by another host is honoured without a heartbeat.
# Must exceed the longest possible driver tick; the driver refreshes the lease every tick.
DEFAULT_LOCK_STALE_AFTER_S = 1800.0

_FLOCK_UNSUPPORTED_ERRNOS = frozenset(
    {errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOTSUP}
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


class DriverAlreadyRunningError(RuntimeError):
    """Raised when another daemon instance already holds the exclusive lock (any login node)."""


class StateCorruptError(RuntimeError):
    """Raised when state.json exists but cannot be parsed. The driver must not continue."""


class StateSchemaError(RuntimeError):
    """Raised when state.json was written by a newer, unsupported schema version."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class InstanceLock:
    """Handle for a held single-instance lock: a host-local `flock` plus a shared-FS lease.

    `flock` alone is not enough on TACC: crontabs are per login node and Lustre may be
    mounted with node-local (or no) flock semantics, so two drivers on `login1` and `login2`
    could both "hold" it. The lease file (`<lock>.lease`, created with O_CREAT|O_EXCL on the
    shared filesystem) records host, pid and a heartbeat; another host only takes it over
    after `stale_after_s` without a heartbeat.
    """

    def __init__(self, lock_path: Path, *, stale_after_s: float) -> None:
        self.lock_path = Path(lock_path)
        self.lease_path = self.lock_path.with_name(self.lock_path.name + ".lease")
        self.stale_after_s = float(stale_after_s)
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.token = uuid.uuid4().hex
        self.flock_supported = True
        self._fh: Any = None

    # -- lease helpers -------------------------------------------------------------------
    def _lease_payload(self) -> dict[str, Any]:
        now = time.time()
        return {
            "host": self.host,
            "pid": self.pid,
            "token": self.token,
            "heartbeat_epoch": now,
            "heartbeat_utc": datetime.fromtimestamp(now, tz=UTC).isoformat(),
        }

    def _read_lease(self) -> tuple[dict[str, Any] | None, float | None]:
        try:
            raw = self.lease_path.read_text(encoding="utf-8")
            mtime = self.lease_path.stat().st_mtime
        except FileNotFoundError:
            return None, None
        try:
            data = json.loads(raw)
        except ValueError:
            data = {}
        return (data if isinstance(data, dict) else {}), mtime

    def _lease_is_stale(self, lease: dict[str, Any], mtime: float) -> bool:
        holder_host = str(lease.get("host", ""))
        try:
            holder_pid = int(lease.get("pid", -1))
        except (TypeError, ValueError):
            holder_pid = -1
        if holder_host == self.host:
            # Same host: if we hold the flock, no live driver on this host holds the lease.
            if self.flock_supported:
                return True
            return not _pid_alive(holder_pid)
        try:
            heartbeat = float(lease.get("heartbeat_epoch", 0.0))
        except (TypeError, ValueError):
            heartbeat = 0.0
        last_seen = max(heartbeat, mtime)
        return (time.time() - last_seen) > self.stale_after_s

    def _try_create_lease(self) -> bool:
        try:
            fd = os.open(str(self.lease_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(self._lease_payload(), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True

    def _acquire_lease(self) -> None:
        for _ in range(3):
            if self._try_create_lease():
                return
            lease, mtime = self._read_lease()
            if lease is None or mtime is None:
                continue  # vanished between create and read: retry
            if not self._lease_is_stale(lease, mtime):
                raise DriverAlreadyRunningError(
                    f"Lock lease {self.lease_path} is held by host={lease.get('host')!r} "
                    f"pid={lease.get('pid')!r} (last heartbeat {lease.get('heartbeat_utc')!r})"
                )
            # Stale: move it aside atomically; only one contender can rename this inode.
            aside = self.lease_path.with_name(f"{self.lease_path.name}.stale.{self.token}")
            try:
                os.rename(self.lease_path, aside)
            except FileNotFoundError:
                continue
            logger.warning(
                "Taking over stale driver lease from host=%s pid=%s (moved to %s)",
                lease.get("host"),
                lease.get("pid"),
                aside.name,
            )
        raise DriverAlreadyRunningError(f"Could not acquire lock lease {self.lease_path}")

    # -- public API ----------------------------------------------------------------------
    def acquire(self, *, wait_s: float = 0.0) -> None:
        """Acquire flock + lease, optionally waiting up to `wait_s` seconds for a live holder."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, float(wait_s))
        self._fh = self.lock_path.open("a+", encoding="utf-8")
        try:
            while True:
                try:
                    self._lock_once()
                    return
                except DriverAlreadyRunningError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.5)
        except BaseException:
            self._close_fh()
            raise

    def _lock_once(self) -> None:
        if self.flock_supported:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DriverAlreadyRunningError(
                    f"Another executor instance on this host holds lock {self.lock_path}"
                ) from exc
            except OSError as exc:
                if exc.errno not in _FLOCK_UNSUPPORTED_ERRNOS:
                    raise
                self.flock_supported = False
                logger.warning(
                    "flock is not supported for %s (%s); relying on the lease file only",
                    self.lock_path,
                    exc,
                )
        try:
            self._acquire_lease()
        except BaseException:
            self._unflock()
            raise

    def refresh(self) -> None:
        """Heartbeat the lease. Raises DriverAlreadyRunningError if another host took it over."""
        lease, _mtime = self._read_lease()
        if lease is None or lease.get("token") != self.token:
            raise DriverAlreadyRunningError(
                f"Lost lock lease {self.lease_path} (now held by "
                f"{(lease or {}).get('host')!r}); stopping to avoid concurrent drivers"
            )
        atomic_write_text(self.lease_path, json.dumps(self._lease_payload(), sort_keys=True) + "\n")

    def release(self) -> None:
        """Remove our lease (only if it is still ours) and drop the flock."""
        try:
            lease, _mtime = self._read_lease()
            if lease is not None and lease.get("token") == self.token:
                with contextlib.suppress(FileNotFoundError):
                    self.lease_path.unlink()
        finally:
            self._unflock()
            self._close_fh()

    def _unflock(self) -> None:
        if self._fh is not None and self.flock_supported:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)

    def _close_fh(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.close()
            self._fh = None


@contextmanager
def acquire_single_instance_lock(
    lock_path: Path,
    *,
    wait_s: float = 0.0,
    stale_after_s: float = DEFAULT_LOCK_STALE_AFTER_S,
) -> Iterator[InstanceLock]:
    """Hold the executor lock for the duration of the block (cron overlap + cross-host safe).

    Raises DriverAlreadyRunningError if a live holder exists (after waiting `wait_s`).
    """
    lock = InstanceLock(Path(lock_path), stale_after_s=stale_after_s)
    lock.acquire(wait_s=wait_s)
    try:
        yield lock
    finally:
        lock.release()


@dataclass
class RunRecord:
    """Persistent state for a single gyrokinetic simulation run across TACC or GCP.

    Status lifecycle: PENDING -> SUBMITTING -> RUNNING -> COMPLETED | NEEDS_HUMAN, with
    RUNNING -> SUBMITTING -> RUNNING on every automated resubmission. SUBMITTING is a durable
    intent: it is persisted *before* the scheduler is called, so a crash can never leave a
    job that the state file does not know about.
    """

    run_id: str
    solver: str  # "GENE" or "GX"
    run_dir: str
    status: str = "PENDING"  # PENDING | SUBMITTING | RUNNING | COMPLETED | NEEDS_HUMAN
    slurm_job_id: str | None = None
    retry_count: int = 0
    walltime_resubmits: int = 0
    submitted_at_epoch: float = 0.0
    last_diagnostic: dict[str, Any] | None = None
    directives_history: list[dict[str, Any]] = field(default_factory=list)
    parameter_patches: list[dict[str, Any]] = field(default_factory=list)
    campaign_id: str = "pilot_50"
    workstream: str = "kr2-gk-database"
    backend: str = "tacc_slurm"  # "tacc_slurm" | "gcp_batch" | "gcp_k8s"
    gpu_tier: str = "tacc_gh200_or_cpu"
    fidelity: str = "low"  # "low" or "high"
    # Empty string = not declared in run_meta.json. Never silently defaulted to a real id.
    geometry_id: str = ""
    equilibrium_id: str = ""
    dataset_manifest_path: str | None = None
    active_failure_event_id: str | None = None
    consecutive_unknown_polls: int = 0
    has_reported_running: bool = False
    last_progress_epoch: float = 0.0
    # --- schema v2 -------------------------------------------------------------------------
    # Submission intent / idempotency. submission_id = f"bsit-{run_id}-{submit_nonce}-{seq}".
    submit_nonce: str = ""
    submit_seq: int = 0
    pending_submission: dict[str, Any] | None = None
    first_submitted_at_epoch: float = 0.0
    # Every scheduler submission of this run, oldest first (provenance + restart history).
    restart_history: list[dict[str, Any]] = field(default_factory=list)
    last_scheduler_state: str | None = None
    # Persisted failure counters (must survive cron --once process restarts).
    consecutive_poll_errors: int = 0
    triage_deferrals: int = 0
    # Set when the watchdog cancelled a job mid-run and is waiting for it to terminate.
    cancel_requested: dict[str, Any] | None = None
    # Human-in-the-loop provenance.
    escalation_count: int = 0
    human_resolutions: int = 0
    # sha256 of the solver input file as submitted most recently (detects unrecorded edits).
    input_sha256_submitted: str | None = None
    # Provenance declared in run_meta.json (None = not declared).
    code_version: str | None = None
    equilibrium_origin: str | None = None


_RUN_RECORD_FIELDS = {f.name: f for f in dataclasses.fields(RunRecord)}
_INT_FIELDS = {
    "retry_count",
    "walltime_resubmits",
    "consecutive_unknown_polls",
    "submit_seq",
    "consecutive_poll_errors",
    "triage_deferrals",
    "escalation_count",
    "human_resolutions",
}
_FLOAT_FIELDS = {
    "submitted_at_epoch",
    "last_progress_epoch",
    "first_submitted_at_epoch",
}
_STR_FIELDS = {
    "run_id",
    "solver",
    "run_dir",
    "status",
    "campaign_id",
    "workstream",
    "backend",
    "gpu_tier",
    "fidelity",
    "geometry_id",
    "equilibrium_id",
    "submit_nonce",
}
_LIST_FIELDS = {"directives_history", "parameter_patches", "restart_history"}


def _record_from_dict(run_id_key: str, rec: dict[str, Any]) -> RunRecord:
    if not isinstance(rec, dict):
        raise StateCorruptError(f"State entry for {run_id_key!r} is not an object")
    kwargs: dict[str, Any] = {}
    for name in ("run_id", "solver", "run_dir"):
        if name not in rec:
            raise StateCorruptError(f"State entry for {run_id_key!r} lacks required {name!r}")
    for name, value in rec.items():
        if name not in _RUN_RECORD_FIELDS:
            logger.warning("Dropping unknown state field %r for run %s", name, run_id_key)
            continue
        if value is None:
            kwargs[name] = None
        elif name in _INT_FIELDS:
            kwargs[name] = int(value)
        elif name in _FLOAT_FIELDS:
            kwargs[name] = float(value)
        elif name in _STR_FIELDS:
            kwargs[name] = str(value)
        elif name in _LIST_FIELDS:
            kwargs[name] = list(value)
        elif name == "has_reported_running":
            kwargs[name] = bool(value)
        else:
            kwargs[name] = value
    # Legacy v1 records: equilibrium_id fell back to geometry_id.
    if not kwargs.get("equilibrium_id") and kwargs.get("geometry_id"):
        kwargs["equilibrium_id"] = kwargs["geometry_id"]
    for name in _INT_FIELDS | _FLOAT_FIELDS | _STR_FIELDS | _LIST_FIELDS:
        if kwargs.get(name, ...) is None:
            kwargs.pop(name)  # None for a non-optional field: fall back to the default
    return RunRecord(**kwargs)


class StateStore:
    """Crash-safe JSON state store and append-only JSONL audit log."""

    def __init__(self, state_file: Path, audit_log_file: Path, dry_run: bool = False) -> None:
        self.state_file = Path(state_file)
        self.audit_log_file = Path(audit_log_file)
        self.dry_run = bool(dry_run)
        self.runs: dict[str, RunRecord] = {}
        self.load()

    def load(self) -> None:
        """Load persisted run state. Fails closed on corruption or an unknown newer schema.

        A corrupted state file is never silently replaced by an empty one: forgetting a
        RUNNING record would orphan its job and could lead to a double submission.
        """
        if not self.state_file.is_file():
            self.runs = {}
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise StateCorruptError(
                f"Cannot parse {self.state_file}: {exc}. Refusing to continue; restore it "
                "from backup or inspect it manually (never delete it while jobs are queued)."
            ) from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("runs", {}), dict):
            raise StateCorruptError(f"{self.state_file} does not contain a 'runs' object")
        version = raw.get("schema_version", 1)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise StateCorruptError(f"{self.state_file} has invalid schema_version {version!r}")
        if version > STATE_SCHEMA_VERSION:
            raise StateSchemaError(
                f"{self.state_file} has schema_version {version}, newer than supported "
                f"{STATE_SCHEMA_VERSION}; upgrade ut-hpc-core instead of downgrading state"
            )
        loaded: dict[str, RunRecord] = {}
        for run_id, rec in raw.get("runs", {}).items():
            try:
                loaded[str(run_id)] = _record_from_dict(str(run_id), rec)
            except (TypeError, ValueError) as exc:
                raise StateCorruptError(f"Invalid state entry for {run_id!r}: {exc}") from exc
        self.runs = loaded

    def _target_state_file(self) -> Path:
        if not self.dry_run:
            return self.state_file
        if str(self.state_file).endswith("state.json"):
            return self.state_file.with_name("state.dryrun.json")
        return self.state_file.with_suffix(".dryrun" + self.state_file.suffix)

    def save(self) -> None:
        """Durably persist all run records (temp + fsync + rename; .dryrun sidecar in dry_run)."""
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "dry_run": self.dry_run,
            "runs": {rid: asdict(rec) for rid, rec in self.runs.items()},
        }
        atomic_write_json(self._target_state_file(), payload, indent=2, sort_keys=False)

    def upsert_run(self, record: RunRecord) -> None:
        """Insert or update a run record and flush to disk."""
        self.runs[record.run_id] = record
        self.save()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Fetch a run record by run_id."""
        return self.runs.get(run_id)

    def append_audit_event(self, event_type: str, run_id: str, details: dict[str, Any]) -> None:
        """Append a structured JSON line to audit.jsonl (or audit.dryrun.jsonl), fsync'd."""
        target_audit = self.audit_log_file
        if self.dry_run:
            if str(self.audit_log_file).endswith("audit.jsonl"):
                target_audit = self.audit_log_file.with_name("audit.dryrun.jsonl")
            else:
                target_audit = self.audit_log_file.with_suffix(
                    ".dryrun" + self.audit_log_file.suffix
                )
        entry = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "run_id": run_id,
            "dry_run": self.dry_run,
            **details,
        }
        append_line_durable(target_audit, json.dumps(entry, sort_keys=True, default=str))
