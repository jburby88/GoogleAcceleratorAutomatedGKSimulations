"""Crash-safe, idempotent job submission shared by the Cloud Batch and GKE backends.

Why this exists (GCP-01 / GCP-02): a scheduler ``submit()`` can time out *after* the server
created the job, and the babysitter (a Cloud Run job killed at its task timeout) can die
between the create call and persisting the job id in ``state.json``.  Random job names
(``uuid4`` suffixes, attempt counters bumped before the API call) turned both cases into
duplicate GPU jobs.

Every submission now has a *submission id*:

* preferred: supplied by the watchdog, which persists it before calling ``submit`` (the
  ``idempotent_submit`` contract in ``hpc_core/protocol.py``);
* fallback (callers that pass none): ``ledger-<run_id>-<ordinal>``, where the ordinal comes
  from the per-run ledger below.

The scheduler job id is a pure function of ``(run_id, submission id)``, the job carries the
labels ``run_key`` and ``submission_key``, and a create that fails with HTTP 409 adopts the
existing job after checking those labels.  Re-sending the same submission therefore never
creates a second job.

Per run directory (on the shared GCS FUSE volume):

* ``.gcp_submission.json`` (the *ledger*) records the current submission: ids, the full
  scheduler ``job_name`` (incl. region), the region/SKU rotation state and a ``status`` of
  ``pending`` (written *before* the create call) or ``created``.
* ``.gcp_active_attempt`` is a fencing token naming the job that may write into the run
  directory.  The in-job task runner (``gcp_exec.task_runner``) stops publishing outputs
  once it names another job, so a cancelled, abandoned or late-starting job can never
  overwrite the files of its successor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

LEDGER_FILENAME = ".gcp_submission.json"
FENCE_FILENAME = ".gcp_active_attempt"
LEDGER_SCHEMA_VERSION = 1

# Batch job ids and Kubernetes Job names (used as the pod label `job-name`) share this shape.
JOB_ID_RE = re.compile(r"^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")
SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
_LABEL_VALUE_RE = re.compile(r"[^a-z0-9_-]+")

RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

T = TypeVar("T")


class SchedulerApiError(RuntimeError):
    """Error from the Batch / Kubernetes REST API, carrying the HTTP status when known.

    ``status_code`` is ``None`` for transport-level failures (DNS, TLS, timeouts). Those are
    retryable: the request may or may not have reached the server, and every create call is
    idempotent (deterministic job id + ``requestId``), so re-sending is safe.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code in RETRYABLE_HTTP_STATUSES


class SubmissionConflictError(RuntimeError):
    """The deterministic job id is taken by a job that does not belong to this submission."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_key(run_id: str) -> str:
    """Stable 12-hex-char, label-safe key for a run id."""
    return _sha(run_id)[:12]


def submission_key(submission_id: str) -> str:
    """Stable 12-hex-char, label-safe key for a submission id."""
    return _sha(submission_id)[:12]


def validate_submission_id(submission_id: str) -> str:
    if not isinstance(submission_id, str) or not SUBMISSION_ID_RE.match(submission_id):
        raise ValueError(
            f"Invalid submission_id {submission_id!r}: expected [A-Za-z0-9_.-], 1-160 chars"
        )
    return submission_id


def label_value(raw: str, max_len: int = 63) -> str:
    """Sanitize a string into a GCE/Kubernetes label value (lowercase, [a-z0-9_-], <= 63)."""
    cleaned = _LABEL_VALUE_RE.sub("-", str(raw).lower()).strip("-_")
    return cleaned[:max_len].rstrip("-_") or "unknown"


def job_id_for_submission(submission_id: str, *, prefix: str = "gx") -> str:
    """Deterministic, RFC-1123 job id (<= 63 chars) for a submission id.

    A pure function of the submission id, so ``find_submission`` can recompute it without the
    run directory. The sanitized id (it embeds the run id) keeps names readable in the console;
    the 10-hex hash makes ids unique per submission and immune to sanitization or truncation
    collisions (``Run_A`` vs ``run-a``).
    """
    validate_submission_id(submission_id)
    slug = re.sub(r"^(bsit|ledger|adhoc)-", "", submission_id.lower())
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    suffix = f"-{_sha(submission_id)[:10]}"
    max_slug = 63 - len(prefix) - 1 - len(suffix)
    slug = slug[:max_slug].strip("-")
    job_id = f"{prefix}-{slug}{suffix}" if slug else f"{prefix}{suffix}"
    if not JOB_ID_RE.match(job_id):  # pragma: no cover - guaranteed by construction
        raise ValueError(f"Generated invalid job id {job_id!r}")
    return job_id


def _id_safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def fallback_submission_id(run_id: str, ordinal: int) -> str:
    """Submission id for callers that pass none (ledger ordinal based)."""
    return f"ledger-{_id_safe(run_id)}-{int(ordinal)}"[:160]


def adhoc_submission_id(run_id: str, retry_count: int) -> str:
    """Submission id for spec-only calls (`build_job_spec` without a submission)."""
    return f"adhoc-{_id_safe(run_id)}-r{int(retry_count)}"[:160]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, fsync, then ``os.replace``.

    On GCS FUSE the object is only created when the temp file is closed successfully, so a
    crash leaves either the old or the new content, never a torn file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class SubmissionRecord:
    """Ledger entry for the current scheduler submission of one run."""

    run_id: str
    submission_id: str
    job_id: str
    job_name: str
    status: str  # "pending" | "created"
    ordinal: int = 0
    rotation_index: int = 0
    region: str | None = None
    gpu_tier: str | None = None
    provisioning_model: str | None = None
    updated_at_utc: str = field(default_factory=utc_now_iso)
    schema_version: int = LEDGER_SCHEMA_VERSION

    @property
    def submission_key(self) -> str:
        return submission_key(self.submission_id)


def ledger_path(run_dir: Path) -> Path:
    return Path(run_dir) / LEDGER_FILENAME


def load_record(run_dir: Path, run_id: str) -> SubmissionRecord | None:
    """Load the ledger. A missing, unreadable or foreign ledger yields ``None``.

    Callers must treat ``None`` as "unknown", not as "nothing was ever submitted": with an
    explicit submission id the backend probes the scheduler before creating anything.
    """
    path = ledger_path(run_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("run_id") != run_id:
            return None
        rec = SubmissionRecord(
            run_id=str(raw["run_id"]),
            submission_id=validate_submission_id(str(raw["submission_id"])),
            job_id=str(raw["job_id"]),
            job_name=str(raw["job_name"]),
            status=str(raw["status"]),
            ordinal=max(0, int(raw.get("ordinal", 0))),
            rotation_index=max(0, int(raw.get("rotation_index", 0))),
            region=raw.get("region"),
            gpu_tier=raw.get("gpu_tier"),
            provisioning_model=raw.get("provisioning_model"),
            updated_at_utc=str(raw.get("updated_at_utc", "")),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if rec.status not in ("pending", "created") or not JOB_ID_RE.match(rec.job_id):
        return None
    return rec


def save_record(run_dir: Path, record: SubmissionRecord) -> None:
    record.updated_at_utc = utc_now_iso()
    payload = {**asdict(record), "submission_key": record.submission_key}
    atomic_write_text(ledger_path(run_dir), json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_fence(run_dir: Path, job_id: str) -> None:
    atomic_write_text(Path(run_dir) / FENCE_FILENAME, job_id + "\n")


def read_fence(run_dir: Path) -> str | None:
    try:
        return (Path(run_dir) / FENCE_FILENAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@dataclass(frozen=True)
class JobView:
    """Backend-neutral view of an existing scheduler job, used for adopt/supersede decisions."""

    name: str
    active: bool  # queued / scheduled / running, and not being cancelled or deleted
    run_key: str | None = None  # value of the `run_key` label
    submission_key: str | None = None  # value of the `submission_key` label
    capacity_failure: bool = False  # never got capacity (stockout / quota / stuck pending)


def call_with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry ``fn`` on retryable :class:`SchedulerApiError` with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except SchedulerApiError as exc:
            if not exc.retryable or attempt == attempts:
                raise
            sleep(base_delay_s * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class SubmissionPlan:
    """What ``submit()`` must do next, decided from the ledger and the previous job."""

    action: str  # "adopt" | "resend" | "new"
    submission_id: str
    ordinal: int
    rotation_index: int
    record: SubmissionRecord | None  # current ledger entry (the previous submission for "new")
    previous: JobView | None  # scheduler view of the ledger's job, when it was looked up


def plan_submission(
    record: SubmissionRecord | None,
    *,
    run_id: str,
    submission_id: str | None,
    get_job: Callable[[str], JobView | None],
    superseded: Iterable[str],
) -> SubmissionPlan:
    """Decide between adopting the ledger's job, re-sending it, or starting a new submission.

    With an explicit ``submission_id`` (watchdog contract) the decision is exact: the same id
    is re-sent (the create call is idempotent); any other id is a new submission.  Without one
    (fallback) a ``pending`` entry is re-sent, and a ``created`` entry whose job is still
    active is adopted unless this process superseded it (cancelled or abandoned it).
    """
    if submission_id is not None:
        validate_submission_id(submission_id)
        if record is not None and record.submission_id == submission_id:
            return SubmissionPlan(
                "resend", submission_id, record.ordinal, record.rotation_index, record, None
            )
        if record is None:
            return SubmissionPlan("new", submission_id, 0, 0, None, None)
        previous = get_job(record.job_name)
        return SubmissionPlan(
            "new",
            submission_id,
            record.ordinal + 1,
            record.rotation_index + (1 if previous and previous.capacity_failure else 0),
            record,
            previous,
        )

    if record is None:
        return SubmissionPlan("new", fallback_submission_id(run_id, 0), 0, 0, None, None)
    if record.status == "pending":
        return SubmissionPlan(
            "resend", record.submission_id, record.ordinal, record.rotation_index, record, None
        )
    previous = get_job(record.job_name)
    superseded_set = set(superseded)
    if (
        previous is not None
        and previous.active
        and record.job_name not in superseded_set
        and previous.submission_key == record.submission_key
    ):
        return SubmissionPlan(
            "adopt", record.submission_id, record.ordinal, record.rotation_index, record, previous
        )
    ordinal = record.ordinal + 1
    return SubmissionPlan(
        "new",
        fallback_submission_id(run_id, ordinal),
        ordinal,
        record.rotation_index + (1 if previous and previous.capacity_failure else 0),
        record,
        previous,
    )


class CancelRegistry:
    """Persistent record of the job names this executor cancelled (``cancelled_externally``).

    The babysitter runs as a fresh process every tick, so an in-memory set cannot tell a
    cancellation we issued from one issued by a human or the platform. Bounded in size.
    """

    MAX_ENTRIES = 2000

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self._names: dict[str, str] = {}
        if self.path and self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._names = {str(k): str(v) for k, v in raw.items()}
            except (OSError, ValueError):
                self._names = {}

    @property
    def persistent(self) -> bool:
        return self.path is not None

    def add(self, job_name: str) -> None:
        self._names[job_name] = utc_now_iso()
        if len(self._names) > self.MAX_ENTRIES:
            for key in sorted(self._names, key=self._names.__getitem__)[: -self.MAX_ENTRIES]:
                del self._names[key]
        if self.path:
            atomic_write_text(self.path, json.dumps(self._names, indent=1, sort_keys=True) + "\n")

    def __contains__(self, job_name: object) -> bool:
        return job_name in self._names


def parse_rfc3339(value: Any) -> datetime | None:
    """Parse RFC 3339 timestamps as returned by the Batch and Kubernetes APIs."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    # Batch returns up to 9 fractional digits; datetime accepts at most 6.
    match = re.match(r"^(.*?\.\d{6})\d*(.*)$", text)
    if match:
        text = match.group(1) + match.group(2)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
