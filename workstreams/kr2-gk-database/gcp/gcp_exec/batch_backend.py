"""GCP Cloud Batch (`batch.googleapis.com/v1`) implementation of `SchedulerBackend`.

Production properties (see docs/review/FIXLOG-GCP.md):

* idempotent submission (GCP-01): deterministic job ids per submission id, `requestId`
  dedup, 409 ALREADY_EXISTS adopts the existing job after checking its labels, retries only
  on retryable errors, and a per-run ledger + fencing token (`gcp_exec.submission`);
* bounded obtainability waterfall (GCP-04): the region/SKU step only advances when the
  previous attempt never got capacity; on-demand capacity and the Spot-A100 fallback are
  explicit opt-ins; past the last step submission is refused;
* least privilege (GCP-05): a user-managed worker service account is mandatory; the
  Compute Engine default service account is rejected;
* the job runs `gx-task` (`gcp_exec.task_runner`) for fenced, validated checkpointing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from hpc_core import BackendCapabilities, JobStatus

from gcp_exec.gpu_selector import GPUTierSpec, resolve_candidate_regions, select_obtainable_target
from gcp_exec.submission import (
    CancelRegistry,
    JobView,
    SchedulerApiError,
    SubmissionConflictError,
    SubmissionRecord,
    adhoc_submission_id,
    call_with_retries,
    job_id_for_submission,
    label_value,
    load_record,
    parse_rfc3339,
    plan_submission,
    read_fence,
    run_key,
    save_record,
    submission_key,
    validate_submission_id,
    write_fence,
)

logger = logging.getLogger(__name__)

BatchTransport = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]

BATCH_API = "https://batch.googleapis.com/v1"
_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
)
# User-managed service accounts only: <name>@<project>.iam.gserviceaccount.com.
# The Compute Engine default SA (<number>-compute@developer.gserviceaccount.com) holds the
# project-wide Editor role by default and is rejected (GCP-05).
_USER_MANAGED_SA_RE = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com$"
)
_NETWORK_URI_RE = re.compile(
    r"^projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/global/networks/[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)
_SUBNETWORK_PINNED_RE = re.compile(
    r"^projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/regions/(?P<region>[a-z]+(?:-[a-z]+)+[0-9]+)/subnetworks/[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)
_SUBNETWORK_TEMPLATE_RE = re.compile(
    r"^projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/regions/\{region\}/subnetworks/[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)
_REQUEST_ID_NAMESPACE = uuid.UUID("6f1c3c2e-5b7e-4d0a-9a51-3b8f0d7e2c41")

# gx-runner runs as UID/GID 10001 (GCP-06). Batch mounts the bucket with gcsfuse as root on the
# host, so the container user needs ownership mapping plus allow_other
# (https://cloud.google.com/batch/docs/create-run-job-storage: all gcsfuse flags are accepted).
DEFAULT_GCS_MOUNT_OPTIONS = (
    "--implicit-dirs",
    "--uid=10001",
    "--gid=10001",
    "--file-mode=664",
    "--dir-mode=775",
    "--o=allow_other",
)

ACTIVE_BATCH_STATES = frozenset({"QUEUED", "SCHEDULED", "RUNNING"})
PENDING_BATCH_STATES = frozenset({"QUEUED", "SCHEDULED"})
# Reserved Batch exit codes (https://cloud.google.com/batch/docs/troubleshooting):
# 50001 Spot/preemptible VM preempted, 50002 VM reporting timeout, 50003 VM rebooted,
# 50004 VM and task unresponsive, 50005 maxRunDuration exceeded, 50006 VM recreated.
INFRA_EXIT_CODES = frozenset({50001, 50002, 50003, 50004, 50006})
MAX_RUN_DURATION_EXIT_CODE = 50005
# gx-task exit codes (gcp_exec.task_runner)
TASK_EXIT_CHECKPOINT_INVALID = 65
TASK_EXIT_SUPERSEDED = 75
TASK_EXIT_REFUSED = 78

_CAPACITY_MARKERS = (
    "zone_resource_pool_exhausted",
    "resource_pool_exhausted",
    "stockout",
    "does not have enough resources",
    "quota_exceeded",
    "quota exceeded",
    "insufficient capacity",
)
_BLOCKED_MARKERS = (
    "permission_denied",
    "permission denied",
    "does not have permission",
    "iam.serviceaccounts.actas",
    "is not authorized",
    "manifest unknown",
    "pull access denied",
    "failed to pull image",
    "repository does not exist",
    "requested access to the resource is denied",
    "invalid_argument",
    "invalid value for field",
    "constraints/compute.",
    "violated constraint",
    "was not found",
)


def _resolve_backend_regions(
    region: str,
    candidate_regions: list[str] | None,
    pinned_subnet_region: str | None,
) -> list[str]:
    if candidate_regions:
        resolved = resolve_candidate_regions(candidate_regions)
    elif os.getenv("GCP_BATCH_CANDIDATE_REGIONS", "").strip():
        resolved = resolve_candidate_regions(None)
    elif pinned_subnet_region is not None:
        resolved = [pinned_subnet_region]
    else:
        resolved = resolve_candidate_regions(
            [region, "us-east4", "us-west1", "us-east1", "us-west4"]
        )
    if pinned_subnet_region is not None and any(r != pinned_subnet_region for r in resolved):
        raise ValueError(
            f"subnetwork is pinned to region {pinned_subnet_region!r}, "
            f"which conflicts with candidate_regions {resolved!r}; use "
            f"candidate_regions=[{pinned_subnet_region!r}] or a '{{region}}' subnetwork template"
        )
    return resolved


def validate_batch_network_config(
    *,
    network: Any,
    subnetwork: Any,
    no_external_ip: Any = True,
    region: str = "us-central1",
    candidate_regions: list[str] | None = None,
) -> tuple[str | None, str | None, bool, str | None]:
    """Validate optional custom VPC / subnet / external-IP settings for Cloud Batch.

    Returns ``(network, subnetwork, no_external_ip, pinned_subnet_region)``.
    """
    if not isinstance(no_external_ip, bool):
        raise ValueError(f"no_external_ip must be a boolean (got {no_external_ip!r})")
    if network is None and subnetwork is None:
        return None, None, no_external_ip, None
    if network is None or subnetwork is None:
        raise ValueError("network and subnetwork must both be set when configuring a custom VPC")
    if not isinstance(network, str) or not isinstance(subnetwork, str):
        raise ValueError("network and subnetwork must be strings")
    net = network.strip()
    subnet = subnetwork.strip()
    if not net or not subnet:
        raise ValueError("network and subnetwork must not be blank")
    if not _NETWORK_URI_RE.match(net):
        raise ValueError(
            f"network {net!r} must match 'projects/<project>/global/networks/<network>'"
        )
    pinned_match = _SUBNETWORK_PINNED_RE.match(subnet)
    template_match = _SUBNETWORK_TEMPLATE_RE.match(subnet)
    if not pinned_match and not template_match:
        raise ValueError(
            f"subnetwork {subnet!r} must match "
            "'projects/<project>/regions/<region>/subnetworks/<subnetwork>' "
            "(or use '{region}' for multi-region templates)"
        )
    pinned_region = pinned_match.group("region") if pinned_match else None
    if pinned_region is not None:
        if region != pinned_region:
            raise ValueError(
                f"subnetwork {subnet!r} is pinned to region {pinned_region!r}, "
                f"which does not match region={region!r}"
            )
        _resolve_backend_regions(region, candidate_regions, pinned_region)
    return net, subnet, no_external_ip, pinned_region


def validate_worker_service_account(email: str | None) -> str:
    """Fail closed unless a user-managed, least-privilege worker SA is configured (GCP-05)."""
    value = (email or "").strip()
    if not value:
        raise ValueError(
            "worker_service_account_email is required: Cloud Batch would otherwise run GX as "
            "the Compute Engine default service account (project Editor). Use the dedicated "
            "least-privilege SA from platform/infra/terraform (gx-batch-worker-sa)."
        )
    if value.endswith("-compute@developer.gserviceaccount.com") or not _USER_MANAGED_SA_RE.match(
        value
    ):
        raise ValueError(
            f"worker_service_account_email {value!r} must be a user-managed service account "
            "(<name>@<project>.iam.gserviceaccount.com); the Compute Engine default SA is refused."
        )
    return value


def _events_text(status_obj: dict[str, Any]) -> tuple[str, list[int]]:
    events = status_obj.get("statusEvents", []) or []
    parts: list[str] = []
    exit_codes: list[int] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        task_exec = ev.get("taskExecution") if isinstance(ev.get("taskExecution"), dict) else {}
        parts.append(f"{ev.get('type', '')} {ev.get('description', '')}")
        if "exitCode" in task_exec:
            try:
                exit_codes.append(int(task_exec["exitCode"]))
            except (TypeError, ValueError):
                pass
    return " ".join(parts).lower(), exit_codes


def map_batch_status_to_canonical(
    batch_job: dict[str, Any],
    *,
    now: datetime | None = None,
    max_pending_s: float | None = None,
    cancel_registry: CancelRegistry | None = None,
) -> JobStatus:
    """Normalize a GCP Cloud Batch `Job` resource dict into `hpc_core.JobStatus`.

    A job stuck in QUEUED/SCHEDULED for longer than `max_pending_s` (Batch keeps retrying VM
    creation indefinitely on a stockout) is reported as PREEMPTED with a capacity reason, so
    the watchdog performs a zero-LLM continuation and the backend rotates region/SKU.
    """
    name = str(batch_job.get("name", batch_job.get("uid", "unknown")))
    status_obj = batch_job.get("status", {}) or {}
    raw_state = str(status_obj.get("state", "STATE_UNSPECIFIED")).upper()
    labels = batch_job.get("labels", {}) or {}
    gpu_tier = labels.get("gpu_tier")
    joined_desc, exit_codes = _events_text(status_obj)
    last_exit = exit_codes[-1] if exit_codes else 0

    elapsed_s = 0.0
    run_duration = str(status_obj.get("runDuration", "")).rstrip("s")
    if run_duration:
        try:
            elapsed_s = max(0.0, float(run_duration))
        except ValueError:
            elapsed_s = 0.0

    reason: str | None = None
    blocked = False
    cancelled_externally = False
    if raw_state in PENDING_BATCH_STATES or raw_state == "ADMISSION_PENDING":
        canonical = "PENDING"
        if any(m in joined_desc for m in _BLOCKED_MARKERS):
            blocked = True
            reason = "scheduling blocked by a configuration/permission error"
        created = parse_rfc3339(batch_job.get("createTime"))
        if max_pending_s and created is not None:
            pending_s = ((now or datetime.now(UTC)) - created).total_seconds()
            if pending_s > max_pending_s and not blocked:
                canonical = "PREEMPTED"
                marker = next((m for m in _CAPACITY_MARKERS if m in joined_desc), "no capacity")
                reason = f"capacity: {raw_state} for {int(pending_s)}s ({marker})"
    elif raw_state == "RUNNING":
        canonical = "RUNNING"
    elif raw_state == "SUCCEEDED":
        canonical = "COMPLETED"
    elif raw_state in ("CANCELLATION_IN_PROGRESS", "DELETION_IN_PROGRESS"):
        # Still possibly active: the VM may keep writing into the run directory until it is
        # gone, so the watchdog must not patch or resubmit yet (POSSIBLY_ACTIVE_STATES).
        canonical = "COMPLETING"
        reason = f"batch state {raw_state}"
    elif raw_state == "CANCELLED":
        canonical = "CANCELLED"
        if cancel_registry is not None and cancel_registry.persistent:
            cancelled_externally = name not in cancel_registry
        reason = "batch state CANCELLED"
    elif raw_state == "FAILED":
        if last_exit == MAX_RUN_DURATION_EXIT_CODE or "maxrunduration" in joined_desc:
            canonical = "TIMEOUT"
            reason = "maxRunDuration exceeded"
        elif last_exit == TASK_EXIT_REFUSED:
            canonical = "FAILED"
            blocked = True
            reason = "gx-task refused to run (GX_SIMULATE_RUN or unverified GX build in prod)"
        elif (
            last_exit in INFRA_EXIT_CODES
            or "preempt" in joined_desc
            or (not exit_codes and "spot" in joined_desc)
        ):
            canonical = "PREEMPTED"
            reason = f"infrastructure interruption (exit {last_exit})"
        elif (
            "out of memory" in joined_desc or re.search(r"\boom\b", joined_desc) or last_exit == 137
        ):
            canonical = "OUT_OF_MEMORY"
            reason = f"out of memory (exit {last_exit})"
        elif any(m in joined_desc for m in _CAPACITY_MARKERS):
            canonical = "PREEMPTED"
            reason = "capacity: " + next(m for m in _CAPACITY_MARKERS if m in joined_desc)
        elif any(m in joined_desc for m in _BLOCKED_MARKERS):
            canonical = "FAILED"
            blocked = True
            reason = "job failed on a configuration/permission error"
        else:
            canonical = "FAILED"
            reason = {
                TASK_EXIT_CHECKPOINT_INVALID: "restart checkpoint missing or invalid (gx-task exit 65)",
                TASK_EXIT_SUPERSEDED: "attempt superseded by a newer submission (gx-task exit 75)",
            }.get(last_exit, f"task failed (exit {last_exit})")
    else:
        canonical = "UNKNOWN"
        reason = f"unrecognized batch state {raw_state}"

    return JobStatus(
        job_id=name,
        state=canonical,
        elapsed_seconds=elapsed_s,
        exit_code=f"{last_exit}:0",
        gpu_tier=gpu_tier,
        reason=reason,
        blocked=blocked,
        cancelled_externally=cancelled_externally,
    )


def _is_capacity_failure(status: JobStatus) -> bool:
    return status.state == "PREEMPTED" and bool(
        status.reason and status.reason.startswith("capacity")
    )


class CloudBatchBackend:
    """Submits, polls, and cancels containerized GX runs on GCP Cloud Batch (Spot L4 / A100)."""

    def __init__(
        self,
        *,
        project_id: str,
        region: str = "us-central1",
        gcs_bucket: str,
        gcs_mount_path: str = "/mnt/gk_bucket",
        gx_container_image: str,
        worker_service_account_email: str | None = None,
        spot_provisioning: bool = True,
        cpu_only: bool = False,
        candidate_regions: list[str] | None = None,
        max_run_duration_s: int = 28800,
        transport: BatchTransport | None = None,
        environment: str = "prod",
        allow_on_demand: bool = False,
        allow_a100_fallback: bool = False,
        max_pending_s: float = 1800.0,
        state_dir: Path | str | None = None,
        gcs_mount_options: list[str] | None = None,
        network: str | None = None,
        subnetwork: str | None = None,
        no_external_ip: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.candidate_regions = list(candidate_regions) if candidate_regions else None
        (
            self.network,
            self.subnetwork,
            self.no_external_ip,
            self._subnet_region,
        ) = validate_batch_network_config(
            network=network,
            subnetwork=subnetwork,
            no_external_ip=no_external_ip,
            region=self.region,
            candidate_regions=self.candidate_regions,
        )
        self.gcs_bucket = gcs_bucket
        self.gcs_mount_path = gcs_mount_path.rstrip("/")
        self.gx_container_image = gx_container_image
        self.worker_service_account_email = validate_worker_service_account(
            worker_service_account_email
        )
        self.spot_provisioning = bool(spot_provisioning)
        self.cpu_only = bool(cpu_only)
        self.max_run_duration_s = int(max_run_duration_s)
        if self.max_run_duration_s <= 0:
            raise ValueError("max_run_duration_s must be > 0")
        self.environment = str(environment).strip().lower()
        self.allow_on_demand = bool(allow_on_demand)
        self.allow_a100_fallback = bool(allow_a100_fallback)
        if not self.spot_provisioning and not self.allow_on_demand:
            raise ValueError("spot_provisioning=false requires allow_on_demand=true (GCP-04)")
        self.max_pending_s = float(max_pending_s)
        self.gcs_mount_options = list(
            DEFAULT_GCS_MOUNT_OPTIONS if gcs_mount_options is None else gcs_mount_options
        )
        self._transport = transport or self._default_gcp_http_transport
        self._sleep = sleep
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token: tuple[str, float] | None = None
        self._job_name_cache: dict[str, str] = {}
        self._superseded: set[str] = set()
        self._cancel_registry = CancelRegistry(
            Path(state_dir) / "gcp_cancelled_jobs.json" if state_dir else None
        )

        self.capabilities = BackendCapabilities(
            backend_name="gcp_batch",
            supports_gene=False,
            supports_gx=True,
            default_gpu_tier="l4",
            supports_spot_continuation=True,
            idempotent_submit=True,
        )

    # ------------------------------------------------------------------ transport
    def _get_access_token(self) -> str:
        """OAuth token from the metadata server (Cloud Run / GCE), else `gcloud` (local dev)."""
        if self._token and self._token[1] > time.monotonic():
            return self._token[0]
        try:
            meta_resp = requests.get(
                _METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"}, timeout=2.0
            )
            if meta_resp.status_code == 200:
                payload = meta_resp.json()
                ttl = max(0.0, float(payload.get("expires_in", 300)) - 60.0)
                self._token = (str(payload["access_token"]), time.monotonic() + ttl)
                return self._token[0]
        except (requests.RequestException, ValueError, KeyError):
            pass
        proc = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return proc.stdout.strip()

    def _default_gcp_http_transport(
        self, method: str, url: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.request(
                method=method, url=url, json=body, headers=headers, timeout=(10.0, 30.0)
            )
        except requests.RequestException as exc:
            raise SchedulerApiError(f"Batch API {method} failed: {exc.__class__.__name__}") from exc
        if resp.status_code >= 400:
            raise SchedulerApiError(
                f"Batch API {method} {url.split('?')[0]} -> HTTP {resp.status_code}: "
                f"{resp.text[:300]}",
                status_code=resp.status_code,
            )
        try:
            return resp.json() if resp.text.strip() else {}
        except ValueError as exc:
            raise SchedulerApiError(f"Batch API {method} returned non-JSON body") from exc

    def _call(self, method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return call_with_retries(lambda: self._transport(method, url, body), sleep=self._sleep)

    # ------------------------------------------------------------------ helpers
    def _regions(self) -> list[str]:
        return _resolve_backend_regions(self.region, self.candidate_regions, self._subnet_region)

    def _job_name(self, region: str, job_id: str) -> str:
        return f"projects/{self.project_id}/locations/{region}/jobs/{job_id}"

    def _container_workdir(self, run_dir: Path) -> str:
        resolved_str = str(run_dir.resolve())
        if resolved_str.startswith(self.gcs_mount_path + "/"):
            return resolved_str
        return f"{self.gcs_mount_path}/queue/running/{run_dir.name}"

    def _get_job(self, job_name: str) -> dict[str, Any] | None:
        try:
            return self._call("GET", f"{BATCH_API}/{job_name}", None)
        except SchedulerApiError as exc:
            if exc.status_code == 404:
                return None
            raise

    def _job_view(self, job_name: str) -> JobView | None:
        job = self._get_job(job_name)
        if job is None:
            return None
        status = map_batch_status_to_canonical(
            job, now=self._clock(), max_pending_s=self.max_pending_s
        )
        raw_state = str((job.get("status") or {}).get("state", "")).upper()
        labels = job.get("labels", {}) or {}
        return JobView(
            name=str(job.get("name", job_name)),
            active=raw_state in ACTIVE_BATCH_STATES,
            run_key=labels.get("run_key"),
            submission_key=labels.get("submission_key"),
            capacity_failure=_is_capacity_failure(status),
        )

    # ------------------------------------------------------------------ job spec
    def build_job_spec(
        self,
        run_dir: Path,
        *,
        parameters: dict[str, Any] | None = None,
        retry_count: int = 0,
        force_high_memory: bool = False,
        attempt_index: int = 0,
        submission_id: str | None = None,
        rotation_index: int | None = None,
    ) -> tuple[str, GPUTierSpec, dict[str, Any]]:
        """Construct the Cloud Batch v1 `Job` body. Returns `(job_id, gpu_spec, body)`.

        The target region is in `body["labels"]["target_region"]`.
        """
        meta_force_a100 = False
        meta_path = run_dir / "run_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_force_a100 = str(meta.get("gpu_tier", "")).lower() == "a100"
            except (OSError, ValueError, AttributeError):
                meta_force_a100 = False

        target_region, gpu_spec, provisioning_model = select_obtainable_target(
            parameters=parameters,
            rotation_index=attempt_index if rotation_index is None else rotation_index,
            candidate_regions=self._regions(),
            force_high_memory=(force_high_memory or meta_force_a100),
            spot_preferred=self.spot_provisioning,
            cpu_only=self.cpu_only,
            allow_on_demand=self.allow_on_demand,
            allow_a100_fallback=self.allow_a100_fallback,
        )
        sub_id = submission_id or adhoc_submission_id(run_dir.name, retry_count)
        job_id = job_id_for_submission(sub_id)
        container_workdir = self._container_workdir(run_dir)

        has_gpu = gpu_spec.accelerator_count > 0 and bool(gpu_spec.accelerator_type)
        instance_policy: dict[str, Any] = {
            "machineType": gpu_spec.machine_type,
            "provisioningModel": provisioning_model,
        }
        if has_gpu:
            instance_policy["accelerators"] = [
                {"type": gpu_spec.accelerator_type, "count": str(gpu_spec.accelerator_count)}
            ]

        labels = {
            "workstream": "kr2-gk-database",
            "solver": "gx",
            "environment": label_value(self.environment),
            "gpu_tier": gpu_spec.tier_name,
            "provisioning": provisioning_model.lower(),
            "target_region": target_region,
            "run_key": run_key(run_dir.name),
            "submission_key": submission_key(sub_id),
            "run_id": label_value(run_dir.name),
        }
        volume: dict[str, Any] = {
            "gcs": {"remotePath": self.gcs_bucket},
            "mountPath": self.gcs_mount_path,
        }
        if self.gcs_mount_options:
            volume["mountOptions"] = list(self.gcs_mount_options)

        job_body: dict[str, Any] = {
            "taskGroups": [
                {
                    "taskCount": "1",
                    "parallelism": "1",
                    "taskSpec": {
                        # Batch-level retries stay off: a retried task would re-run gx.in without
                        # the restart flag and without the watchdog's continuation accounting.
                        # Preemption/stockout continuations are counted and capped by the watchdog.
                        "maxRetryCount": 0,
                        "maxRunDuration": f"{self.max_run_duration_s}s",
                        "volumes": [volume],
                        "runnables": [
                            {
                                "environment": {
                                    "variables": {
                                        "HDF5_USE_FILE_LOCKING": "FALSE",
                                        "HOME": "/tmp",
                                        "GK_ENVIRONMENT": self.environment,
                                    }
                                },
                                "container": {
                                    "imageUri": self.gx_container_image,
                                    "entrypoint": "gx-task",
                                    "commands": [
                                        "--src-dir",
                                        container_workdir,
                                        "--attempt-id",
                                        job_id,
                                        "--environment",
                                        self.environment,
                                    ],
                                },
                            }
                        ],
                    },
                }
            ],
            "allocationPolicy": {
                "instances": [{"installGpuDrivers": has_gpu, "policy": instance_policy}],
                "serviceAccount": {"email": self.worker_service_account_email},
                # Propagated to the VMs, disks and GPUs for per-run cost attribution.
                "labels": dict(labels),
            },
            "labels": labels,
            "logsPolicy": {"destination": "CLOUD_LOGGING"},
        }
        if self.network is not None and self.subnetwork is not None:
            resolved_subnetwork = self.subnetwork.format(region=target_region)
            job_body["allocationPolicy"]["network"] = {
                "networkInterfaces": [
                    {
                        "network": self.network,
                        "subnetwork": resolved_subnetwork,
                        "noExternalIpAddress": self.no_external_ip,
                    }
                ]
            }
        return job_id, gpu_spec, job_body

    # ------------------------------------------------------------------ submission
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
        """Submit (at most once per submission id) a containerized GX job to Cloud Batch."""
        if solver.upper() != "GX":
            raise ValueError(
                f"CloudBatchBackend supports containerized GX only (got solver={solver!r}); "
                "run Fortran GENE on TACC SlurmBackend."
            )
        run_dir = Path(run_dir)
        run_id = run_dir.name
        if submission_id is not None:
            validate_submission_id(submission_id)
        record = load_record(run_dir, run_id)

        if submission_id is not None and record is None and read_fence(run_dir) is not None:
            # The ledger is gone but this directory was submitted before: never guess, look.
            found = self.find_submission(submission_id)
            if found:
                return found

        plan = plan_submission(
            record,
            run_id=run_id,
            submission_id=submission_id,
            get_job=self._job_view,
            superseded=self._superseded,
        )
        if plan.action == "adopt" and plan.record is not None:
            logger.warning("Adopting active Batch job %s for run %s", plan.record.job_name, run_id)
            return plan.record.job_name
        if plan.action == "resend" and plan.record is not None and plan.record.status == "created":
            return plan.record.job_name

        if plan.action == "resend" and plan.record is not None:
            rec = plan.record
            _, _, job_body = self.build_job_spec(
                run_dir,
                parameters=parameters,
                retry_count=retry_count,
                force_high_memory=force_high_memory,
                submission_id=rec.submission_id,
                rotation_index=rec.rotation_index,
            )
            job_body["labels"]["target_region"] = rec.region or self.region
            region = rec.region or self.region
        else:
            # New submission: supersede the previous job if it is somehow still active.
            if plan.previous is not None and plan.previous.active and plan.record is not None:
                self._supersede(plan.record.job_name)
            job_id, gpu_spec, job_body = self.build_job_spec(
                run_dir,
                parameters=parameters,
                retry_count=retry_count,
                force_high_memory=force_high_memory,
                submission_id=plan.submission_id,
                rotation_index=plan.rotation_index,
            )
            region = str(job_body["labels"]["target_region"])
            rec = SubmissionRecord(
                run_id=run_id,
                submission_id=plan.submission_id,
                job_id=job_id,
                job_name=self._job_name(region, job_id),
                status="pending",
                ordinal=plan.ordinal,
                rotation_index=plan.rotation_index,
                region=region,
                gpu_tier=gpu_spec.tier_name,
                provisioning_model=str(job_body["labels"]["provisioning"]).upper(),
            )
            # Write-ahead: the ledger and the fence exist before the job can.
            save_record(run_dir, rec)
            write_fence(run_dir, rec.job_id)

        full_name = self._create(rec, job_body, region)
        rec.job_name = full_name
        rec.status = "created"
        save_record(run_dir, rec)
        self._job_name_cache[rec.job_id] = full_name
        return full_name

    def _create(self, rec: SubmissionRecord, job_body: dict[str, Any], region: str) -> str:
        parent = f"projects/{self.project_id}/locations/{region}"
        request_id = uuid.uuid5(_REQUEST_ID_NAMESPACE, rec.job_name)
        url = f"{BATCH_API}/{parent}/jobs?jobId={rec.job_id}&requestId={request_id}"
        try:
            resp = self._call("POST", url, job_body)
        except SchedulerApiError as exc:
            if exc.status_code != 409:
                raise
            existing = self._get_job(rec.job_name)
            labels = (existing or {}).get("labels", {}) or {}
            if existing is None or labels.get("submission_key") != rec.submission_key:
                raise SubmissionConflictError(
                    f"Batch job id {rec.job_id} exists but does not belong to submission "
                    f"{rec.submission_id!r}"
                ) from exc
            logger.info("Batch job %s already existed for this submission; adopting", rec.job_name)
            return str(existing.get("name", rec.job_name))
        return str(resp.get("name", rec.job_name))

    def find_submission(self, submission_id: str) -> str | None:
        """Return the Batch job name created for `submission_id`, or None.

        The job id is a pure function of the submission id; the region is not, so every
        candidate region is probed and the `submission_key` label must match.
        """
        key = submission_key(validate_submission_id(submission_id))
        job_id = job_id_for_submission(submission_id)
        for region in self._regions():
            name = self._job_name(region, job_id)
            job = self._get_job(name)
            if job is not None and (job.get("labels", {}) or {}).get("submission_key") == key:
                return str(job.get("name", name))
        return None

    def _supersede(self, job_name: str) -> None:
        self._superseded.add(job_name)
        try:
            self.cancel(job_name)
        except SchedulerApiError as exc:
            logger.warning("Could not cancel superseded Batch job %s: %s", job_name, exc)

    # ------------------------------------------------------------------ status / cancel
    def _resolve_job_name(self, job_id: str) -> str:
        if job_id.startswith("projects/"):
            return job_id
        if job_id in self._job_name_cache:
            return self._job_name_cache[job_id]
        return self._job_name(self.region, job_id)

    def cancel(self, job_id: str) -> None:
        """Cancel a queued or running Batch job. A job that no longer exists is a no-op."""
        name = self._resolve_job_name(job_id)
        self._cancel_registry.add(name)
        self._superseded.add(name)
        try:
            self._call("POST", f"{BATCH_API}/{name}:cancel", {})
        except SchedulerApiError as exc:
            if exc.status_code == 404:
                return
            raise

    def query_job(self, job_id: str) -> JobStatus:
        """Query Cloud Batch job status and normalize it to `JobStatus`."""
        name = self._resolve_job_name(job_id)
        try:
            batch_job = self._call("GET", f"{BATCH_API}/{name}", None)
        except SchedulerApiError as exc:
            return JobStatus(job_id=name, state="UNKNOWN", reason=f"Batch API error: {exc}"[:300])
        except Exception as exc:  # transport bugs must not crash the whole tick
            logger.warning("Unexpected error querying %s: %s", name, exc)
            return JobStatus(job_id=name, state="UNKNOWN", reason=exc.__class__.__name__)
        status = map_batch_status_to_canonical(
            batch_job,
            now=self._clock(),
            max_pending_s=self.max_pending_s,
            cancel_registry=self._cancel_registry,
        )
        if _is_capacity_failure(status):
            # Abandon the starved job: the next submit() cancels it and rotates region/SKU.
            self._superseded.add(name)
        return status
