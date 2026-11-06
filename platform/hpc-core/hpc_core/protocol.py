"""Backend-agnostic scheduler protocol shared by TACC Slurm and GCP Cloud Batch / K8s."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ACTIVE_STATES: frozenset[str] = frozenset({"RUNNING"})
TRANSIENT_STATES: frozenset[str] = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "REQUEUED",
        "SUSPENDED",
        "RESIZING",
        "SIGNALING",
        "REVOKED",
        "COMPLETING",
        "STAGE_OUT",
        "STOPPED",
    }
)
TERMINAL_STATES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "TIMEOUT",
        "FAILED",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
INFRA_STATES: frozenset[str] = frozenset({"UNKNOWN"})
KNOWN_JOB_STATES: frozenset[str] = ACTIVE_STATES | TRANSIENT_STATES | TERMINAL_STATES | INFRA_STATES

# A job in one of these states may still be consuming allocation or writing into its run
# directory. Nothing may patch, move or resubmit the run directory while the job is here.
POSSIBLY_ACTIVE_STATES: frozenset[str] = ACTIVE_STATES | TRANSIENT_STATES | INFRA_STATES


@dataclass(frozen=True)
class BackendCapabilities:
    """Static metadata describing an execution backend."""

    backend_name: str  # e.g. "tacc_slurm", "gcp_batch", "gcp_k8s"
    supports_gene: bool = True
    supports_gx: bool = True
    default_gpu_tier: str = "cpu_or_gh200"
    supports_spot_continuation: bool = False
    # Idempotent submission contract (see SchedulerBackend.submit / find_submission).
    # A backend may set this to True only if BOTH hold:
    #   1. submit(..., submission_id=S) creates at most one job per S, even when called again
    #      after a crash or a timeout (the second call returns the existing job's id);
    #   2. find_submission(S) returns the job id created for S, or None if none exists.
    # When False, the driver never re-calls submit() for a submission whose outcome is
    # unknown; it escalates to a human instead (fail safe: no blind resubmission).
    idempotent_submit: bool = False


@dataclass(frozen=True)
class JobStatus:
    """Normalized execution job status across Slurm, GCP Cloud Batch, and K8s Jobs."""

    job_id: str
    state: str  # Canonical member of KNOWN_JOB_STATES
    elapsed_seconds: float = 0.0
    exit_code: str = "0:0"
    gpu_tier: str | None = None
    # Free-text scheduler detail, e.g. the Slurm pending reason ("Priority", "JobHeldUser")
    # or who cancelled the job ("cancelled by uid=0"). Informational; never parsed for policy.
    reason: str | None = None
    # True when the scheduler reports that the job cannot start or continue without human
    # action (held, invalid account/QOS/partition, allocation exhausted, dependency never
    # satisfied). The watchdog escalates instead of waiting forever.
    blocked: bool = False
    # For CANCELLED jobs: True if the cancellation was NOT issued by this executor (user,
    # admin or system). False when unknown. The watchdog never auto-resubmits such runs.
    cancelled_externally: bool = False


class SchedulerBackend(Protocol):
    """Protocol implemented by SlurmBackend (TACC) and CloudBatchBackend / K8sJobBackend (GCP).

    Idempotent submission (only when `capabilities.idempotent_submit` is True):
    the caller passes a `submission_id` that is unique per submission attempt and is persisted
    *before* the call (charset `[A-Za-z0-9_.-]`, at most 160 characters). Calling `submit`
    twice with the same `submission_id` must not create a second job. Backends without this
    capability are always called without `submission_id`.
    """

    capabilities: BackendCapabilities

    def submit(
        self,
        run_dir: Path,
        *,
        solver: str = "GX",
        parameters: dict[str, Any] | None = None,
        retry_count: int = 0,
        force_high_memory: bool = False,
    ) -> str:
        """Submit a simulation job for `run_dir` and return its scheduler job ID.

        Backends with `capabilities.idempotent_submit` also accept a keyword-only
        `submission_id: str | None = None` argument (see the class docstring).
        """
        ...

    def cancel(self, job_id: str) -> None:
        """Cancel an active job by ID."""
        ...

    def query_job(self, job_id: str) -> JobStatus:
        """Return normalized JobStatus for `job_id`."""
        ...
