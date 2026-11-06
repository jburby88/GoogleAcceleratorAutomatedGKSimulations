"""Tests for `GKEComputeClassBackend`: idempotent Job names (GCP-02), TTL (GCP-07), pod spec
hardening, on-demand opt-in classes (GCP-04), status mapping and the default transport."""

from __future__ import annotations

import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gcp_exec.gke_backend import GKEComputeClassBackend, map_k8s_job_to_canonical
from gcp_exec.submission import SchedulerApiError, SubmissionConflictError, read_fence

SMALL = {"nx": 64, "ny": 32, "nz": 48}


def _gke(transport: Any, tmp_path: Path, **kw: Any) -> GKEComputeClassBackend:
    return GKEComputeClassBackend(
        gcs_bucket="ut-austin-gk-dataset",
        gx_container_image="us-central1-docker.pkg.dev/ut-proj/containers/gx-runner:v1",
        transport=transport,
        state_dir=tmp_path / "state",
        sleep=lambda _s: None,
        **kw,
    )


def _run_dir(tmp_path: Path, name: str = "gx_gke_001") -> Path:
    run_dir = tmp_path / "queue" / "running" / name
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text("[Dimensions]\nntheta = 48\n", encoding="utf-8")
    return run_dir


def test_job_manifest_is_hardened_and_garbage_collected(tmp_path: Path) -> None:
    gke = _gke(lambda *_a: {}, tmp_path)
    name, cclass, manifest = gke.build_job_spec(
        _run_dir(tmp_path), parameters=SMALL, submission_id="bsit-gx_gke_001-00000001-1"
    )
    spec = manifest["spec"]
    assert spec["backoffLimit"] == 0
    assert spec["ttlSecondsAfterFinished"] == 7 * 24 * 3600
    pod = spec["template"]["spec"]
    assert pod["terminationGracePeriodSeconds"] == 30
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 10001
    container = pod["containers"][0]
    assert container["command"] == ["gx-task"]
    assert container["args"][container["args"].index("--attempt-id") + 1] == name
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    tol_keys = {t["key"] for t in pod["tolerations"]}
    assert {"nvidia.com/gpu", "cloud.google.com/compute-class"} <= tol_keys
    assert pod["nodeSelector"]["cloud.google.com/compute-class"] == cclass == "gx-gpu-obtainable"
    mount_opts = pod["volumes"][0]["csi"]["volumeAttributes"]["mountOptions"]
    assert "uid=10001" in mount_opts and "gid=10001" in mount_opts
    with pytest.raises(ValueError):
        _gke(lambda *_a: {}, tmp_path, ttl_seconds_after_finished=60)


def test_on_demand_classes_require_opt_in(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    spot_only = _gke(lambda *_a: {}, tmp_path)
    opted = _gke(lambda *_a: {}, tmp_path, allow_on_demand=True)
    assert spot_only.build_job_spec(run_dir, parameters=SMALL)[1] == "gx-gpu-obtainable"
    assert opted.build_job_spec(run_dir, parameters=SMALL)[1] == "gx-gpu-obtainable-ondemand"
    assert (
        opted.build_job_spec(run_dir, parameters=SMALL, force_high_memory=True)[1]
        == "gx-gpu-highmem-ondemand"
    )


def test_same_submission_is_created_once_and_adopted_after_lost_response(
    tmp_path: Path, k8s_server: Any
) -> None:
    k8s_server.fail_next_post_after_create = True
    run_dir = _run_dir(tmp_path)
    name = _gke(k8s_server, tmp_path).submit(
        run_dir, parameters=SMALL, submission_id="bsit-gx_gke_001-00000002-1"
    )
    again = _gke(k8s_server, tmp_path).submit(
        run_dir, parameters=SMALL, submission_id="bsit-gx_gke_001-00000002-1"
    )
    assert name == again and list(k8s_server.jobs) == [name]
    assert read_fence(run_dir) == name
    assert "-a1" not in name  # no attempt counter in the name (GCP-02)
    assert _gke(k8s_server, tmp_path).find_submission("bsit-gx_gke_001-00000002-1") == name
    assert _gke(k8s_server, tmp_path).find_submission("bsit-gx_gke_001-00000002-9") is None


def test_409_for_foreign_job_is_a_conflict(tmp_path: Path, k8s_server: Any) -> None:
    gke = _gke(k8s_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    sid = "bsit-gx_gke_001-00000003-1"
    name, _, manifest = gke.build_job_spec(run_dir, parameters=SMALL, submission_id=sid)
    manifest["metadata"]["labels"]["submission_key"] = "000000000000"
    k8s_server.jobs[name] = manifest
    with pytest.raises(SubmissionConflictError):
        gke.submit(run_dir, parameters=SMALL, submission_id=sid)


def test_new_submission_deletes_still_active_previous_job(tmp_path: Path, k8s_server: Any) -> None:
    gke = _gke(k8s_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    first = gke.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_gke_001-00000004-1")
    second = gke.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_gke_001-00000004-2")
    assert first not in k8s_server.jobs and second in k8s_server.jobs


def test_cancel_404_is_noop_and_other_errors_raise(tmp_path: Path, k8s_server: Any) -> None:
    gke = _gke(k8s_server, tmp_path)
    gke.cancel("gx-does-not-exist")

    def broken(*_a: Any) -> dict[str, Any]:
        raise SchedulerApiError("forbidden", status_code=403)

    with pytest.raises(SchedulerApiError):
        _gke(broken, tmp_path).cancel("gx-any")


def test_status_mapping_running_pending_capacity_and_blocked() -> None:
    now = datetime(2026, 11, 6, 9, 0, tzinfo=UTC)
    job = {"metadata": {"name": "j", "labels": {"gpu_tier": "l4"}}, "status": {"active": 1}}

    def pod(phase: str, **status: Any) -> dict[str, Any]:
        return {
            "metadata": {"creationTimestamp": (now - timedelta(hours=2)).isoformat()},
            "status": {"phase": phase, **status},
        }

    running = map_k8s_job_to_canonical(job, {"items": [pod("Running")]}, now=now)
    assert running.state == "RUNNING"
    unscheduled = pod(
        "Pending",
        conditions=[{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}],
    )
    waiting = map_k8s_job_to_canonical(job, {"items": [unscheduled]}, now=now, max_pending_s=9000)
    assert waiting.state == "PENDING"
    starved = map_k8s_job_to_canonical(job, {"items": [unscheduled]}, now=now, max_pending_s=1800)
    assert starved.state == "PREEMPTED" and starved.reason.startswith("capacity")  # type: ignore[union-attr]
    pull = pod(
        "Pending", containerStatuses=[{"state": {"waiting": {"reason": "ImagePullBackOff"}}}]
    )
    blocked = map_k8s_job_to_canonical(job, {"items": [pull]}, now=now)
    assert blocked.blocked and blocked.state == "PENDING"
    deadline = map_k8s_job_to_canonical(
        {
            "metadata": {"name": "j"},
            "status": {
                "failed": 1,
                "conditions": [{"type": "Failed", "status": "True", "reason": "DeadlineExceeded"}],
            },
        }
    )
    assert deadline.state == "TIMEOUT"


def test_default_transport_uses_cluster_ca_and_raises_typed_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gke = GKEComputeClassBackend(
        gcs_bucket="b", gx_container_image="img:v1", k8s_ca_cert_path=None, k8s_auth="in_cluster"
    )
    assert isinstance(gke._ssl_context(), ssl.SSLContext)
    with pytest.raises(ValueError):
        GKEComputeClassBackend(gcs_bucket="b", gx_container_image="img:v1", k8s_auth="anonymous")

    import urllib.error
    import urllib.request

    def fake_urlopen(req: Any, timeout: float, context: Any) -> Any:
        assert isinstance(context, ssl.SSLContext)
        raise urllib.error.HTTPError(req.full_url, 409, "Conflict", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SchedulerApiError) as exc_info:
        gke._default_k8s_http_transport("POST", "https://kubernetes.default.svc/x", {})
    assert exc_info.value.status_code == 409
