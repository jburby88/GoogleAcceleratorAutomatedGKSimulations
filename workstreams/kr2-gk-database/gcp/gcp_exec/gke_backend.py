"""GKE Custom Compute Class (`ComputeClass` CRD) Kubernetes Job backend for GX simulations.

Uses GKE's `cloud.google.com/compute-class` nodeSelector so the cluster autoscaler walks the
GPU priority list of `k8s/compute-classes.yaml`. The default classes are Spot-only; the
`<class>-ondemand` variants (DWS flex-start / on-demand fallbacks) are selected only with
`allow_on_demand=True` (GCP-04).

Production properties (see docs/review/FIXLOG-GCP.md):

* idempotent submission (GCP-02): the Job name is a pure function of the submission id (no
  attempt counter), so a re-sent create returns 409 AlreadyExists and the existing Job is
  adopted after its `submission_key` label is checked;
* finished Jobs are garbage-collected via `ttlSecondsAfterFinished` (GCP-07), long enough
  that the babysitter always observes the terminal state first;
* the pod runs `gx-task` (`gcp_exec.task_runner`) as a non-root user with fenced, validated
  checkpointing, explicit GPU/compute-class tolerations and a preemption grace period;
* the default transport verifies the cluster CA (in-cluster) and supports Google OAuth for
  a babysitter running outside the cluster (Cloud Run).
"""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpc_core import BackendCapabilities, JobStatus

from gcp_exec.gpu_selector import (
    T4_SAFE_VRAM_CEILING_GB,
    estimate_gx_vram_gb,
    select_gpu_tier,
)
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

K8sTransport = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]

DEFAULT_COMPUTE_CLASS = "gx-gpu-obtainable"
MEDIUM_VRAM_COMPUTE_CLASS = "gx-gpu-l4-or-a100"
HIGHMEM_COMPUTE_CLASS = "gx-gpu-highmem"
CPU_FALLBACK_COMPUTE_CLASS = "gx-cpu"
ON_DEMAND_SUFFIX = "-ondemand"

RUNNER_UID = 10001  # non-root `gx` user of Dockerfile.gx-runner
DEFAULT_TTL_AFTER_FINISHED_S = 7 * 24 * 3600
_IN_CLUSTER_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
_IN_CLUSTER_CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
)
_BLOCKED_WAITING_REASONS = frozenset(
    {"ErrImagePull", "ImagePullBackOff", "InvalidImageName", "CreateContainerConfigError"}
)


def _pods(pods_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(pods_payload, dict):
        return []
    return [p for p in pods_payload.get("items", []) or [] if isinstance(p, dict)]


def map_k8s_job_to_canonical(
    job_payload: dict[str, Any],
    pods_payload: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
    max_pending_s: float | None = None,
) -> JobStatus:
    """Translate a Kubernetes `batch/v1` `Job` (and optional Pod list) into a canonical `JobStatus`."""
    meta = job_payload.get("metadata", {}) or {}
    status = job_payload.get("status", {}) or {}
    labels = meta.get("labels", {}) or {}
    job_name = str(meta.get("name", "unknown"))
    gpu_tier = labels.get("gpu_tier")

    conditions = status.get("conditions", []) or []
    cond_types = {str(c.get("type", "")): c for c in conditions if c.get("status") == "True"}
    reasons_parts = [f"{c.get('reason', '')} {c.get('message', '')}".lower() for c in conditions]

    pod_exit_code: int | None = None
    any_running = False
    blocked_reason: str | None = None
    oldest_unscheduled: datetime | None = None
    for pod in _pods(pods_payload):
        pod_status = pod.get("status", {}) or {}
        phase = str(pod_status.get("phase", ""))
        any_running = any_running or phase == "Running"
        reasons_parts.append(
            f"{pod_status.get('reason', '')} {pod_status.get('message', '')}".lower()
        )
        for pcond in pod_status.get("conditions", []) or []:
            if pcond.get("type") == "DisruptionTarget" and pcond.get("status") == "True":
                reasons_parts.append(
                    f"disruptiontarget preempt {pcond.get('reason', '')} {pcond.get('message', '')}".lower()
                )
            if (
                phase == "Pending"
                and pcond.get("type") == "PodScheduled"
                and pcond.get("status") == "False"
            ):
                created = parse_rfc3339((pod.get("metadata", {}) or {}).get("creationTimestamp"))
                if created and (oldest_unscheduled is None or created < oldest_unscheduled):
                    oldest_unscheduled = created
        for cstat in pod_status.get("containerStatuses", []) or []:
            waiting = (cstat.get("state", {}) or {}).get("waiting") or {}
            if waiting.get("reason") in _BLOCKED_WAITING_REASONS:
                blocked_reason = f"container waiting: {waiting.get('reason')}"
            term = (cstat.get("state", {}) or {}).get("terminated") or (
                cstat.get("lastState", {}) or {}
            ).get("terminated")
            if isinstance(term, dict):
                reasons_parts.append(f"{term.get('reason', '')} {term.get('message', '')}".lower())
                if term.get("exitCode") is not None:
                    try:
                        pod_exit_code = int(term["exitCode"])
                    except (TypeError, ValueError):
                        pass

    reasons_text = " ".join(reasons_parts)
    succeeded = int(status.get("succeeded", 0) or 0)
    failed = int(status.get("failed", 0) or 0)
    active = int(status.get("active", 0) or 0)

    reason: str | None = None
    blocked = False
    exit_code_str = f"{pod_exit_code}:0" if pod_exit_code is not None else "0:0"
    if meta.get("deletionTimestamp") and not (succeeded > 0 or "Complete" in cond_types):
        # Foreground deletion in progress: pods may still be running and writing.
        canonical = "COMPLETING"
        reason = "job deletion in progress"
    elif succeeded > 0 or "Complete" in cond_types:
        canonical = "COMPLETED"
        exit_code_str = "0:0"
    elif "Failed" in cond_types or failed > 0 or (pod_exit_code is not None and pod_exit_code != 0):
        if exit_code_str == "0:0":
            exit_code_str = "1:0"
        if "deadlineexceeded" in reasons_text:
            canonical = "TIMEOUT"
            reason = "activeDeadlineSeconds exceeded"
        elif pod_exit_code == 78:
            canonical = "FAILED"
            blocked = True
            reason = "gx-task refused to run (GX_SIMULATE_RUN or unverified GX build in prod)"
        elif (
            "disruptiontarget" in reasons_text
            or "preempt" in reasons_text
            or "evict" in reasons_text
            or "shutdown" in reasons_text
        ):
            canonical = "PREEMPTED"
            reason = "pod disrupted (Spot preemption / eviction / node shutdown)"
        elif pod_exit_code == 137 or "oomkilled" in reasons_text or "out_of_memory" in reasons_text:
            canonical = "OUT_OF_MEMORY"
            exit_code_str = "137:0"
            reason = "container OOMKilled"
        else:
            canonical = "FAILED"
            reason = f"pod failed (exit {pod_exit_code})"
    elif blocked_reason is not None:
        canonical = "PENDING"
        blocked = True
        reason = blocked_reason
    elif active > 0 and any_running:
        canonical = "RUNNING"
    elif active > 0 and oldest_unscheduled is not None and max_pending_s:
        waited = ((now or datetime.now(UTC)) - oldest_unscheduled).total_seconds()
        if waited > max_pending_s:
            canonical = "PREEMPTED"
            reason = f"capacity: pod unschedulable for {int(waited)}s"
        else:
            canonical = "PENDING"
    elif active > 0 and pods_payload is None:
        canonical = "RUNNING"  # no pod view available: keep the legacy interpretation
    else:
        canonical = "PENDING"

    elapsed = 0.0
    t0 = parse_rfc3339(status.get("startTime"))
    t1 = parse_rfc3339(status.get("completionTime"))
    if t0 is not None and t1 is not None:
        elapsed = max(0.0, (t1 - t0).total_seconds())

    return JobStatus(
        job_id=job_name,
        state=canonical,
        elapsed_seconds=elapsed,
        exit_code=exit_code_str,
        gpu_tier=gpu_tier,
        reason=reason,
        blocked=blocked,
    )


class GKEComputeClassBackend:
    """Kubernetes `batch/v1` `Job` backend targeting GKE Custom Compute Classes (`ComputeClass`)."""

    capabilities = BackendCapabilities(
        backend_name="gcp_k8s",
        supports_gene=False,
        supports_gx=True,
        default_gpu_tier="l4",
        supports_spot_continuation=True,
        idempotent_submit=True,
    )

    def __init__(
        self,
        *,
        gcs_bucket: str,
        gx_container_image: str,
        namespace: str = "default",
        gcs_mount_path: str = "/mnt/gk_bucket",
        compute_class_default: str = DEFAULT_COMPUTE_CLASS,
        compute_class_medium_vram: str = MEDIUM_VRAM_COMPUTE_CLASS,
        compute_class_highmem: str = HIGHMEM_COMPUTE_CLASS,
        compute_class_cpu: str = CPU_FALLBACK_COMPUTE_CLASS,
        ksa_name: str = "gx-runner-ksa",
        cpu_only: bool = False,
        max_run_duration_s: int = 28800,
        k8s_api_url: str = "https://kubernetes.default.svc",
        transport: K8sTransport | None = None,
        environment: str = "prod",
        allow_on_demand: bool = False,
        max_pending_s: float = 1800.0,
        ttl_seconds_after_finished: int = DEFAULT_TTL_AFTER_FINISHED_S,
        k8s_ca_cert_path: str | None = None,
        k8s_auth: str = "in_cluster",
        gcs_mount_options: str | None = None,
        state_dir: Path | str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.gcs_bucket = gcs_bucket
        self.gx_container_image = gx_container_image
        self.namespace = namespace
        self.gcs_mount_path = gcs_mount_path.rstrip("/")
        self.compute_class_default = compute_class_default
        self.compute_class_medium_vram = compute_class_medium_vram
        self.compute_class_highmem = compute_class_highmem
        self.compute_class_cpu = compute_class_cpu
        self.ksa_name = ksa_name
        self.cpu_only = bool(cpu_only)
        self.max_run_duration_s = int(max_run_duration_s)
        if self.max_run_duration_s <= 0:
            raise ValueError("max_run_duration_s must be > 0")
        self.environment = str(environment).strip().lower()
        self.allow_on_demand = bool(allow_on_demand)
        self.max_pending_s = float(max_pending_s)
        self.ttl_seconds_after_finished = int(ttl_seconds_after_finished)
        if self.ttl_seconds_after_finished < 3600:
            raise ValueError(
                "ttl_seconds_after_finished must be >= 3600 so the babysitter observes the "
                "terminal state before the Job is garbage-collected"
            )
        self.k8s_api_url = k8s_api_url.rstrip("/")
        if k8s_auth not in ("in_cluster", "google"):
            raise ValueError("k8s_auth must be 'in_cluster' or 'google'")
        self.k8s_auth = k8s_auth
        self.k8s_ca_cert_path = k8s_ca_cert_path
        self.gcs_mount_options = gcs_mount_options or (
            f"implicit-dirs,uid={RUNNER_UID},gid={RUNNER_UID},file-mode=664,dir-mode=775"
        )
        self._transport = transport or self._default_k8s_http_transport
        self._sleep = sleep
        self._clock = clock or (lambda: datetime.now(UTC))
        self._superseded: set[str] = set()
        self._cancel_registry = CancelRegistry(
            Path(state_dir) / "gcp_cancelled_jobs.json" if state_dir else None
        )
        self._google_token: tuple[str, float] | None = None

    # ------------------------------------------------------------------ class selection
    def _with_tier(self, name: str) -> str:
        return f"{name}{ON_DEMAND_SUFFIX}" if self.allow_on_demand else name

    def select_compute_class(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        force_high_memory: bool = False,
    ) -> tuple[str, str, int]:
        """Return `(compute_class_name, gpu_tier_label, gpu_count)`.

        - `<= 13.5 GB` VRAM -> `gx-gpu-obtainable` (Spot L4 -> Spot T4)
        - `13.5 < VRAM <= 20.0 GB` -> `gx-gpu-l4-or-a100` (excludes the 16 GB T4)
        - `> 20.0 GB` or `force_high_memory` -> `gx-gpu-highmem` (Spot A100)
        With `allow_on_demand=True` the `-ondemand` variant of each class is used.
        """
        gpu_spec = select_gpu_tier(
            parameters=parameters,
            force_high_memory=force_high_memory,
            cpu_only=self.cpu_only,
        )
        if gpu_spec.tier_name == "cpu" or gpu_spec.accelerator_count == 0:
            return self._with_tier(self.compute_class_cpu), "cpu", 0
        if gpu_spec.tier_name == "a100":
            return self._with_tier(self.compute_class_highmem), "a100", gpu_spec.accelerator_count
        if estimate_gx_vram_gb(parameters) > T4_SAFE_VRAM_CEILING_GB:
            return (
                self._with_tier(self.compute_class_medium_vram),
                "l4",
                gpu_spec.accelerator_count,
            )
        return self._with_tier(self.compute_class_default), "l4", gpu_spec.accelerator_count

    def _container_workdir(self, run_dir: Path) -> str:
        resolved_str = str(run_dir.resolve())
        if resolved_str.startswith(self.gcs_mount_path + "/"):
            return resolved_str
        return f"{self.gcs_mount_path}/queue/running/{run_dir.name}"

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
    ) -> tuple[str, str, dict[str, Any]]:
        """Construct a Kubernetes `batch/v1` `Job` manifest bound to a GKE `ComputeClass`.

        `attempt_index` is accepted for backward compatibility and ignored: the Job name
        depends only on the submission id (GCP-02).
        """
        del attempt_index
        meta_path = run_dir / "run_meta.json"
        meta_force_a100 = False
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_force_a100 = str(meta.get("gpu_tier", "")).lower() == "a100"
            except (OSError, ValueError, AttributeError):
                meta_force_a100 = False

        compute_class, tier_label, gpu_count = self.select_compute_class(
            parameters=parameters,
            force_high_memory=(force_high_memory or meta_force_a100),
        )
        sub_id = submission_id or adhoc_submission_id(run_dir.name, retry_count)
        job_name = job_id_for_submission(sub_id)
        container_workdir = self._container_workdir(run_dir)

        resources: dict[str, Any] = {
            "requests": {"cpu": "4", "memory": "16Gi"},
            "limits": {"memory": "24Gi" if tier_label != "a100" else "40Gi"},
        }
        tolerations: list[dict[str, Any]] = [
            # GKE taints auto-created compute-class nodes; GKE usually injects these, but an
            # explicit toleration keeps the pod schedulable if that admission step is absent.
            {
                "key": "cloud.google.com/compute-class",
                "operator": "Equal",
                "value": compute_class,
                "effect": "NoSchedule",
            },
            {"key": "cloud.google.com/gke-spot", "operator": "Exists", "effect": "NoSchedule"},
        ]
        if gpu_count > 0:
            resources["requests"]["nvidia.com/gpu"] = str(gpu_count)
            resources["limits"]["nvidia.com/gpu"] = str(gpu_count)
            tolerations.append(
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
            )

        labels = {
            "workstream": "kr2-gk-database",
            "solver": "gx",
            "environment": label_value(self.environment),
            "gpu_tier": tier_label,
            "compute_class": compute_class,
            "run_key": run_key(run_dir.name),
            "submission_key": submission_key(sub_id),
        }

        job_manifest: dict[str, Any] = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": self.namespace, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": self.max_run_duration_s,
                "ttlSecondsAfterFinished": self.ttl_seconds_after_finished,
                "podFailurePolicy": {
                    "rules": [
                        {
                            "action": "FailJob",
                            "onPodConditions": [{"type": "DisruptionTarget", "status": "True"}],
                        },
                        {
                            "action": "FailJob",
                            "onExitCodes": {
                                "containerName": "gx-runner",
                                "operator": "In",
                                "values": [137],
                            },
                        },
                    ]
                },
                "template": {
                    "metadata": {
                        "annotations": {"gke-gcsfuse/volumes": "true"},
                        "labels": dict(labels),
                    },
                    "spec": {
                        "serviceAccountName": self.ksa_name,
                        "restartPolicy": "Never",
                        # Spot preemption notice is ~30 s; gx-task needs a few seconds to
                        # publish the log and the newest stable checkpoint.
                        "terminationGracePeriodSeconds": 30,
                        "nodeSelector": {"cloud.google.com/compute-class": compute_class},
                        "tolerations": tolerations,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": RUNNER_UID,
                            "runAsGroup": RUNNER_UID,
                            "fsGroup": RUNNER_UID,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "gx-runner",
                                "image": self.gx_container_image,
                                "command": ["gx-task"],
                                "args": [
                                    "--src-dir",
                                    container_workdir,
                                    "--attempt-id",
                                    job_name,
                                    "--environment",
                                    self.environment,
                                ],
                                "env": [
                                    {"name": "HDF5_USE_FILE_LOCKING", "value": "FALSE"},
                                    {"name": "HOME", "value": "/tmp"},
                                    {"name": "GK_ENVIRONMENT", "value": self.environment},
                                ],
                                "resources": resources,
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "volumeMounts": [
                                    {"name": "gk-dataset-gcs", "mountPath": self.gcs_mount_path}
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "gk-dataset-gcs",
                                "csi": {
                                    "driver": "gcsfuse.csi.storage.gke.io",
                                    "volumeAttributes": {
                                        "bucketName": self.gcs_bucket,
                                        "mountOptions": self.gcs_mount_options,
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        }
        return job_name, compute_class, job_manifest

    # ------------------------------------------------------------------ API helpers
    def _jobs_url(self, name: str | None = None) -> str:
        base = f"{self.k8s_api_url}/apis/batch/v1/namespaces/{self.namespace}/jobs"
        return f"{base}/{name}" if name else base

    def _call(self, method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return call_with_retries(lambda: self._transport(method, url, body), sleep=self._sleep)

    def _get_job(self, name: str) -> dict[str, Any] | None:
        try:
            return self._call("GET", self._jobs_url(name), None)
        except SchedulerApiError as exc:
            if exc.status_code == 404:
                return None
            raise

    def _get_pods(self, job_name: str) -> dict[str, Any] | None:
        selector = urllib.parse.quote(f"job-name={job_name}")
        url = f"{self.k8s_api_url}/api/v1/namespaces/{self.namespace}/pods?labelSelector={selector}"
        try:
            return self._call("GET", url, None)
        except SchedulerApiError:
            return None

    @staticmethod
    def _is_active(job: dict[str, Any]) -> bool:
        meta = job.get("metadata", {}) or {}
        if meta.get("deletionTimestamp"):
            return False
        conditions = (job.get("status", {}) or {}).get("conditions", []) or []
        return not any(
            c.get("type") in ("Complete", "Failed") and c.get("status") == "True"
            for c in conditions
        )

    def _job_view(self, name: str) -> JobView | None:
        job = self._get_job(name)
        if job is None:
            return None
        labels = (job.get("metadata", {}) or {}).get("labels", {}) or {}
        status = map_k8s_job_to_canonical(
            job, self._get_pods(name), now=self._clock(), max_pending_s=self.max_pending_s
        )
        return JobView(
            name=name,
            active=self._is_active(job),
            run_key=labels.get("run_key"),
            submission_key=labels.get("submission_key"),
            capacity_failure=bool(status.reason and status.reason.startswith("capacity")),
        )

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
        """Create (at most once per submission id) a GX Job bound to a GKE Compute Class."""
        if solver.upper() != "GX":
            raise ValueError(
                f"GKEComputeClassBackend supports containerized GX only (got solver={solver!r}); "
                "run Fortran GENE on TACC SlurmBackend."
            )
        run_dir = Path(run_dir)
        run_id = run_dir.name
        if submission_id is not None:
            validate_submission_id(submission_id)
        record = load_record(run_dir, run_id)
        if submission_id is not None and record is None and read_fence(run_dir) is not None:
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
            logger.warning("Adopting active Job %s for run %s", plan.record.job_name, run_id)
            return plan.record.job_name
        if plan.action == "resend" and plan.record is not None and plan.record.status == "created":
            return plan.record.job_name

        if plan.action == "new" and plan.previous is not None and plan.previous.active:
            assert plan.record is not None
            self._supersede(plan.record.job_name)
        sub_id = plan.submission_id
        job_name, compute_class, manifest = self.build_job_spec(
            run_dir,
            parameters=parameters,
            retry_count=retry_count,
            force_high_memory=force_high_memory,
            submission_id=sub_id,
        )
        if plan.action == "resend" and plan.record is not None:
            rec = plan.record
        else:
            rec = SubmissionRecord(
                run_id=run_id,
                submission_id=sub_id,
                job_id=job_name,
                job_name=job_name,
                status="pending",
                ordinal=plan.ordinal,
                rotation_index=plan.rotation_index,
                region=None,
                gpu_tier=str(manifest["metadata"]["labels"]["gpu_tier"]),
                provisioning_model=compute_class,
            )
            save_record(run_dir, rec)
            write_fence(run_dir, rec.job_id)

        try:
            resp = self._call("POST", self._jobs_url(), manifest)
            created_name = str((resp.get("metadata", {}) or {}).get("name", job_name))
        except SchedulerApiError as exc:
            if exc.status_code != 409:
                raise
            existing = self._get_job(job_name)
            labels = ((existing or {}).get("metadata", {}) or {}).get("labels", {}) or {}
            if existing is None or labels.get("submission_key") != rec.submission_key:
                raise SubmissionConflictError(
                    f"Job {job_name} exists but does not belong to submission {sub_id!r}"
                ) from exc
            logger.info("Job %s already existed for this submission; adopting", job_name)
            created_name = job_name
        rec.job_name = created_name
        rec.status = "created"
        save_record(run_dir, rec)
        return created_name

    def find_submission(self, submission_id: str) -> str | None:
        """Return the Job name created for `submission_id`, or None."""
        key = submission_key(validate_submission_id(submission_id))
        name = job_id_for_submission(submission_id)
        job = self._get_job(name)
        if job is None:
            return None
        labels = (job.get("metadata", {}) or {}).get("labels", {}) or {}
        return name if labels.get("submission_key") == key else None

    def _supersede(self, job_name: str) -> None:
        self._superseded.add(job_name)
        try:
            self.cancel(job_name)
        except SchedulerApiError as exc:
            logger.warning("Could not delete superseded Job %s: %s", job_name, exc)

    # ------------------------------------------------------------------ status / cancel
    def query_job(self, job_id: str) -> JobStatus:
        """Query the Job and its Pods and map them to a canonical `JobStatus`."""
        try:
            job_payload = self._call("GET", self._jobs_url(job_id), None)
        except SchedulerApiError as exc:
            if exc.status_code == 404 and job_id in self._cancel_registry:
                # Our own Foreground delete has completed: the Job and its pods are gone.
                return JobStatus(job_id=job_id, state="CANCELLED", reason="deleted by executor")
            return JobStatus(job_id=job_id, state="UNKNOWN", reason=f"K8s API error: {exc}"[:300])
        except Exception as exc:  # transport bugs must not crash the whole tick
            logger.warning("Unexpected error querying Job %s: %s", job_id, exc)
            return JobStatus(job_id=job_id, state="UNKNOWN", reason=exc.__class__.__name__)
        status = map_k8s_job_to_canonical(
            job_payload,
            pods_payload=self._get_pods(job_id),
            now=self._clock(),
            max_pending_s=self.max_pending_s,
        )
        if status.reason and status.reason.startswith("capacity"):
            self._superseded.add(job_id)
        return status

    def cancel(self, job_id: str) -> None:
        """Delete the Job and its Pods. 404 is a no-op.

        `propagationPolicy=Foreground` keeps the Job object (with a deletionTimestamp, reported
        as COMPLETING) until its pods are gone, so the watchdog never patches or resubmits the
        run while the old pod can still write into the run directory.
        """
        self._cancel_registry.add(job_id)
        self._superseded.add(job_id)
        url = f"{self._jobs_url(job_id)}?propagationPolicy=Foreground"
        try:
            self._call("DELETE", url, None)
        except SchedulerApiError as exc:
            if exc.status_code == 404:
                return
            raise

    # ------------------------------------------------------------------ transport
    def _bearer_token(self) -> str | None:
        if self.k8s_auth == "in_cluster":
            if _IN_CLUSTER_TOKEN.is_file():
                return _IN_CLUSTER_TOKEN.read_text(encoding="utf-8").strip()
            return None
        if self._google_token and self._google_token[1] > time.monotonic():
            return self._google_token[0]
        req = urllib.request.Request(_METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ttl = max(0.0, float(payload.get("expires_in", 300)) - 60.0)
        self._google_token = (str(payload["access_token"]), time.monotonic() + ttl)
        return self._google_token[0]

    def _ssl_context(self) -> ssl.SSLContext:
        cafile = self.k8s_ca_cert_path
        if cafile is None and self.k8s_auth == "in_cluster" and _IN_CLUSTER_CA.is_file():
            cafile = str(_IN_CLUSTER_CA)
        return ssl.create_default_context(cafile=cafile)

    def _default_k8s_http_transport(
        self, method: str, url: str, body: dict[str, Any] | None
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            token = self._bearer_token()
        except (OSError, ValueError, KeyError) as exc:
            raise SchedulerApiError(f"cannot obtain K8s credentials: {exc}") from exc
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30.0, context=self._ssl_context()) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:300]
            raise SchedulerApiError(
                f"K8s API {method} {url.split('?')[0]} failed ({exc.code}): {err_body}",
                status_code=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SchedulerApiError(f"K8s API {method} transport error: {exc}") from exc
        except ValueError as exc:
            raise SchedulerApiError(f"K8s API {method} returned non-JSON body") from exc
