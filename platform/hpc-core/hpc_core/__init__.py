"""Shared HPC execution core with lightweight dependencies."""

from hpc_core.client import (
    AuroraSteerClient,
    EventsValidationError,
    TriageClientError,
    TriageTransientError,
    validate_aurora_steer_url,
)
from hpc_core.fsutil import (
    SAFE_ID_RE,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    is_safe_id,
    sha256_file,
)
from hpc_core.manifest import DatasetEntryExistsError, harvest_completed_run_to_dataset
from hpc_core.notify import send_escalation_notification
from hpc_core.protocol import (
    ACTIVE_STATES,
    INFRA_STATES,
    KNOWN_JOB_STATES,
    POSSIBLY_ACTIVE_STATES,
    TERMINAL_STATES,
    TRANSIENT_STATES,
    BackendCapabilities,
    JobStatus,
    SchedulerBackend,
)
from hpc_core.slurm import (
    SlurmBackend,
    SlurmCLI,
    normalize_slurm_state,
    slurm_job_name_for_submission,
)
from hpc_core.state import (
    STATE_SCHEMA_VERSION,
    DriverAlreadyRunningError,
    InstanceLock,
    RunRecord,
    StateCorruptError,
    StateSchemaError,
    StateStore,
    acquire_single_instance_lock,
)
from hpc_core.telemetry import (
    TelemetryEmitter,
    hash_operator_handle,
    infer_run_metadata,
)

__all__ = [
    "ACTIVE_STATES",
    "INFRA_STATES",
    "KNOWN_JOB_STATES",
    "POSSIBLY_ACTIVE_STATES",
    "SAFE_ID_RE",
    "STATE_SCHEMA_VERSION",
    "TERMINAL_STATES",
    "TRANSIENT_STATES",
    "AuroraSteerClient",
    "BackendCapabilities",
    "DatasetEntryExistsError",
    "DriverAlreadyRunningError",
    "EventsValidationError",
    "InstanceLock",
    "JobStatus",
    "RunRecord",
    "SchedulerBackend",
    "SlurmBackend",
    "SlurmCLI",
    "StateCorruptError",
    "StateSchemaError",
    "StateStore",
    "TelemetryEmitter",
    "TriageClientError",
    "TriageTransientError",
    "acquire_single_instance_lock",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "harvest_completed_run_to_dataset",
    "hash_operator_handle",
    "infer_run_metadata",
    "is_safe_id",
    "normalize_slurm_state",
    "send_escalation_notification",
    "sha256_file",
    "slurm_job_name_for_submission",
    "validate_aurora_steer_url",
]
