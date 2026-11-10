"""Unit tests for `CloudBatchBackend`: job spec, status mapping, idempotent submission (GCP-01),
bounded obtainability waterfall (GCP-04), mandatory worker service account (GCP-05)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gcp_exec.batch_backend import (
    CloudBatchBackend,
    map_batch_status_to_canonical,
    validate_worker_service_account,
)
from gcp_exec.gpu_selector import ObtainabilityExhaustedError, estimate_gx_vram_gb, select_gpu_tier
from gcp_exec.submission import (
    FENCE_FILENAME,
    LEDGER_FILENAME,
    SchedulerApiError,
    SubmissionConflictError,
    load_record,
    read_fence,
    submission_key,
)

WORKER_SA = "gx-batch-worker-sa@ut-test.iam.gserviceaccount.com"
SMALL = {"nx": 64, "ny": 32, "nz": 48}


def _backend(transport: Any, tmp_path: Path, **kw: Any) -> CloudBatchBackend:
    kw.setdefault("region", "us-central1")
    kw.setdefault("worker_service_account_email", WORKER_SA)
    kw.setdefault("candidate_regions", ["us-central1", "us-east4"])
    return CloudBatchBackend(
        project_id="ut-test",
        gcs_bucket="ut-test-gk-dataset",
        gx_container_image="us-central1-docker.pkg.dev/ut-test/aurora/gx@sha256:" + "a" * 64,
        transport=transport,
        state_dir=tmp_path / "state",
        sleep=lambda _s: None,
        **kw,
    )


def _run_dir(tmp_path: Path, name: str = "gx_w7x_01") -> Path:
    run_dir = tmp_path / "queue" / "running" / name
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text("[Dimensions]\nntheta = 48\n", encoding="utf-8")
    return run_dir


def test_gpu_selector_defaults_to_spot_l4_and_escalates_to_a100_on_high_vram() -> None:
    low_params = {"nx": 64, "ny": 32, "nz": 48}
    assert estimate_gx_vram_gb(low_params) < 20.0
    spec_low = select_gpu_tier(low_params)
    assert (spec_low.tier_name, spec_low.machine_type, spec_low.accelerator_type) == (
        "l4",
        "g2-standard-8",
        "nvidia-l4",
    )
    high = {"nx": 256, "ny": 128, "nz": 128, "nhermite": 32, "nlaguerre": 16, "nspecies": 3}
    assert estimate_gx_vram_gb(high) > 20.0
    spec_high = select_gpu_tier(high)
    assert (spec_high.tier_name, spec_high.machine_type, spec_high.accelerator_type) == (
        "a100",
        "a2-highgpu-1g",
        "nvidia-tesla-a100",
    )


def test_job_spec_runs_gx_task_with_fence_id_labels_and_mandatory_sa(
    tmp_path: Path, batch_server: Any
) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    job_name = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-0a1b2c3d-1")

    (_, url, body) = batch_server.posts()[0]
    assert "jobId=" in url and "requestId=" in url
    job_id = url.split("jobId=")[1].split("&")[0]
    assert job_name.endswith(f"/jobs/{job_id}")
    assert body is not None
    policy = body["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "g2-standard-8"
    assert policy["provisioningModel"] == "SPOT"
    assert policy["accelerators"] == [{"type": "nvidia-l4", "count": "1"}]
    assert body["allocationPolicy"]["instances"][0]["installGpuDrivers"] is True
    assert body["allocationPolicy"]["serviceAccount"] == {"email": WORKER_SA}
    task = body["taskGroups"][0]["taskSpec"]
    assert task["maxRetryCount"] == 0
    assert task["maxRunDuration"] == "28800s"
    assert task["volumes"][0]["gcs"]["remotePath"] == "ut-test-gk-dataset"
    # Non-root UID 10001 in the image (GCP-06): gcsfuse must map ownership and allow_other.
    assert {"--uid=10001", "--gid=10001", "--o=allow_other"} <= set(
        task["volumes"][0]["mountOptions"]
    )
    container = task["runnables"][0]["container"]
    assert container["entrypoint"] == "gx-task"
    cmd = container["commands"]
    assert cmd[cmd.index("--attempt-id") + 1] == job_id
    assert cmd[cmd.index("--src-dir") + 1] == "/mnt/gk_bucket/queue/running/gx_w7x_01"
    assert task["runnables"][0]["environment"]["variables"]["GK_ENVIRONMENT"] == "prod"
    labels = body["labels"]
    assert labels["submission_key"] == submission_key("bsit-gx_w7x_01-0a1b2c3d-1")
    assert labels["run_id"] == "gx_w7x_01"
    assert body["allocationPolicy"]["labels"] == labels  # propagated to VMs for cost attribution
    # Write-ahead ledger + fencing token name exactly this job.
    assert read_fence(run_dir) == job_id
    rec = load_record(run_dir, "gx_w7x_01")
    assert rec is not None and rec.status == "created" and rec.job_name == job_name


def test_worker_service_account_is_mandatory_and_default_compute_sa_refused(
    tmp_path: Path,
) -> None:
    for bad in (None, "", "123456789-compute@developer.gserviceaccount.com", "not-an-email"):
        with pytest.raises(ValueError):
            _backend(lambda *_a: {}, tmp_path, worker_service_account_email=bad)
    assert validate_worker_service_account(WORKER_SA) == WORKER_SA


def test_resubmitting_same_submission_never_creates_second_job(
    tmp_path: Path, batch_server: Any
) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    first = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000001-1")
    # Babysitter killed after the create but before state.json was saved: same id again.
    second = _backend(batch_server, tmp_path).submit(
        run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000001-1"
    )
    assert first == second
    assert len(batch_server.jobs) == 1
    assert len(batch_server.posts()) == 1


def test_lost_create_response_is_retried_and_adopted(tmp_path: Path, batch_server: Any) -> None:
    batch_server.fail_next_post_after_create = True
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    name = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000002-1")
    assert len(batch_server.jobs) == 1 and name in batch_server.jobs
    assert len(batch_server.posts()) == 2  # retried with the same jobId + requestId


def test_crash_after_write_ahead_resends_same_job_id(tmp_path: Path, batch_server: Any) -> None:
    batch_server.fail_posts_with = [None, None, None]  # create never reached the server
    run_dir = _run_dir(tmp_path)
    with pytest.raises(SchedulerApiError):
        _backend(batch_server, tmp_path).submit(
            run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000003-1"
        )
    pending = load_record(run_dir, "gx_w7x_01")
    assert pending is not None and pending.status == "pending"
    name = _backend(batch_server, tmp_path).submit(
        run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000003-1"
    )
    assert name.endswith(pending.job_id)
    assert len(batch_server.jobs) == 1


def test_409_for_foreign_job_is_a_conflict_not_an_adoption(
    tmp_path: Path, batch_server: Any
) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    sid = "bsit-gx_w7x_01-00000004-1"
    _, _, body = backend.build_job_spec(run_dir, parameters=SMALL, submission_id=sid)
    from gcp_exec.submission import job_id_for_submission

    squatter = f"projects/ut-test/locations/us-central1/jobs/{job_id_for_submission(sid)}"
    batch_server.jobs[squatter] = {
        "name": squatter,
        "labels": {"submission_key": "000000000000"},
        "status": {"state": "RUNNING"},
    }
    with pytest.raises(SubmissionConflictError):
        backend.submit(run_dir, parameters=SMALL, submission_id=sid)


def test_new_submission_supersedes_still_active_previous_job(
    tmp_path: Path, batch_server: Any
) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    first = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000005-1")
    batch_server.set_state(first, "RUNNING")
    second = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000005-2")
    assert first != second
    assert batch_server.jobs[first]["status"]["state"] == "CANCELLATION_IN_PROGRESS"
    assert read_fence(run_dir) == second.rsplit("/", 1)[1]


def test_fallback_without_submission_id_adopts_unrecorded_active_job(
    tmp_path: Path, batch_server: Any
) -> None:
    run_dir = _run_dir(tmp_path)
    first = _backend(batch_server, tmp_path).submit(run_dir, parameters=SMALL)
    batch_server.set_state(first, "RUNNING")
    # New process (Cloud Run tick); state.json never learned about `first`.
    adopted = _backend(batch_server, tmp_path).submit(run_dir, parameters=SMALL)
    assert adopted == first and len(batch_server.jobs) == 1
    # After an explicit cancel (mid-run failure path) a new job is created instead.
    backend = _backend(batch_server, tmp_path)
    backend.cancel(first)
    batch_server.set_state(first, "RUNNING")  # cancellation still propagating
    replacement = backend.submit(run_dir, parameters=SMALL)
    assert replacement != first and len(batch_server.jobs) == 2


def test_capacity_starved_job_is_abandoned_and_region_rotates(
    tmp_path: Path, batch_server: Any
) -> None:
    now = datetime(2026, 11, 6, 9, 0, tzinfo=UTC)
    backend = _backend(batch_server, tmp_path, max_pending_s=1800, clock=lambda: now)
    run_dir = _run_dir(tmp_path)
    first = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000006-1")
    assert "/locations/us-central1/" in first
    batch_server.set_state(
        first,
        "SCHEDULED",
        statusEvents=[{"description": "VM creation failed: ZONE_RESOURCE_POOL_EXHAUSTED"}],
    )
    batch_server.jobs[first]["createTime"] = (now - timedelta(hours=1)).isoformat()
    status = backend.query_job(first)
    assert status.state == "PREEMPTED" and status.reason and status.reason.startswith("capacity")
    second = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000006-2")
    assert "/locations/us-east4/" in second  # rotated only because capacity was missing
    assert batch_server.jobs[first]["status"]["state"] == "CANCELLATION_IN_PROGRESS"
    assert load_record(run_dir, "gx_w7x_01").rotation_index == 1  # type: ignore[union-attr]


def test_preemption_continuation_keeps_region_and_spot(tmp_path: Path, batch_server: Any) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    names = []
    for i in range(1, 12):  # far more continuations than regions: never escalates to STANDARD
        name = backend.submit(
            run_dir, parameters=SMALL, submission_id=f"bsit-gx_w7x_01-0000000a-{i}"
        )
        batch_server.set_state(
            name,
            "FAILED",
            statusEvents=[{"description": "preempted", "taskExecution": {"exitCode": 50001}}],
        )
        names.append(name)
    assert all("/locations/us-central1/" in n for n in names)
    for n in names:
        policy = batch_server.jobs[n]["allocationPolicy"]["instances"][0]["policy"]
        assert policy["provisioningModel"] == "SPOT"


def test_on_demand_requires_opt_in_and_waterfall_is_capped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow_on_demand"):
        _backend(lambda *_a: {}, tmp_path, spot_provisioning=False)
    backend = _backend(lambda *_a: {}, tmp_path)
    run_dir = _run_dir(tmp_path)
    provs = set()
    for idx in range(4):  # 2 regions x (L4, T4) on SPOT only
        _, _, body = backend.build_job_spec(run_dir, parameters=SMALL, rotation_index=idx)
        provs.add(body["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"])
    assert provs == {"SPOT"}
    with pytest.raises(ObtainabilityExhaustedError):
        backend.build_job_spec(run_dir, parameters=SMALL, rotation_index=4)
    opted = _backend(lambda *_a: {}, tmp_path, allow_on_demand=True)
    _, _, body = opted.build_job_spec(run_dir, parameters=SMALL, rotation_index=4)
    assert body["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"] == "STANDARD"


def test_find_submission_probes_candidate_regions(tmp_path: Path, batch_server: Any) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    assert backend.find_submission("bsit-gx_w7x_01-00000007-1") is None
    name = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000007-1")
    assert backend.find_submission("bsit-gx_w7x_01-00000007-1") == name
    # Lost ledger but a fence exists: submit() looks before it creates.
    (run_dir / LEDGER_FILENAME).unlink()
    assert (run_dir / FENCE_FILENAME).is_file()
    again = _backend(batch_server, tmp_path).submit(
        run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000007-1"
    )
    assert again == name and len(batch_server.jobs) == 1


def test_cancel_is_idempotent_and_external_cancel_is_flagged(
    tmp_path: Path, batch_server: Any
) -> None:
    backend = _backend(batch_server, tmp_path)
    run_dir = _run_dir(tmp_path)
    ours = backend.submit(run_dir, parameters=SMALL, submission_id="bsit-gx_w7x_01-00000008-1")
    backend.cancel(ours)
    backend.cancel("projects/ut-test/locations/us-central1/jobs/gone")  # 404 is a no-op
    batch_server.set_state(ours, "CANCELLED")
    fresh = _backend(batch_server, tmp_path)  # registry is persisted in state_dir
    assert fresh.query_job(ours).cancelled_externally is False
    other = fresh.submit(
        _run_dir(tmp_path, "gx_other"), parameters=SMALL, submission_id="bsit-gx_other-00000008-1"
    )
    batch_server.set_state(other, "CANCELLED")
    assert fresh.query_job(other).cancelled_externally is True


def test_query_job_maps_api_errors_to_unknown(tmp_path: Path) -> None:
    def broken(*_a: Any) -> dict[str, Any]:
        raise SchedulerApiError("boom", status_code=503)

    st = _backend(broken, tmp_path).query_job("projects/p/locations/us-central1/jobs/x")
    assert st.state == "UNKNOWN"


def test_status_mapping_preemption_oom_timeout_blocked() -> None:
    def job(state: str, *events: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "name": "projects/p/locations/us-central1/jobs/j",
            "labels": {"gpu_tier": "l4"},
            "status": {"state": state, "statusEvents": list(events), **extra},
        }

    st = map_batch_status_to_canonical(
        job(
            "FAILED",
            {
                "description": "Task failed due to SPOT VM preemption",
                "taskExecution": {"exitCode": 50001},
            },
            runDuration="912.5s",
        )
    )
    assert (st.state, st.gpu_tier, st.elapsed_seconds) == ("PREEMPTED", "l4", pytest.approx(912.5))
    assert (
        map_batch_status_to_canonical(
            job("FAILED", {"description": "VM recreated", "taskExecution": {"exitCode": 50006}})
        ).state
        == "PREEMPTED"
    )
    assert (
        map_batch_status_to_canonical(
            job("FAILED", {"description": "Container killed", "taskExecution": {"exitCode": 137}})
        ).state
        == "OUT_OF_MEMORY"
    )
    assert (
        map_batch_status_to_canonical(
            job("FAILED", {"description": "x", "taskExecution": {"exitCode": 50005}})
        ).state
        == "TIMEOUT"
    )
    refused = map_batch_status_to_canonical(
        job("FAILED", {"description": "x", "taskExecution": {"exitCode": 78}})
    )
    assert refused.state == "FAILED" and refused.blocked
    perm = map_batch_status_to_canonical(
        job("FAILED", {"description": "PERMISSION_DENIED: iam.serviceAccounts.actAs"})
    )
    assert perm.blocked
    for desc in (
        "Constraint constraints/compute.vmExternalIpAccess violated for project",
        "Operation violated constraint constraints/compute.skipDefaultNetworkCreation",
        "The resource 'projects/p/global/networks/default' was not found",
    ):
        assert map_batch_status_to_canonical(job("SCHEDULED", {"description": desc})).blocked
        assert map_batch_status_to_canonical(job("FAILED", {"description": desc})).blocked
    assert map_batch_status_to_canonical(job("SCHEDULED")).state == "PENDING"
    assert map_batch_status_to_canonical(job("SUCCEEDED")).state == "COMPLETED"


def test_custom_vpc_network_interfaces_and_pinned_regional_subnet_colocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCP_BATCH_CANDIDATE_REGIONS", raising=False)
    net = "projects/ut-test/global/networks/ut-test-vpc"
    subnet = "projects/ut-test/regions/us-central1/subnetworks/ut-test-subnet"
    backend = CloudBatchBackend(
        project_id="ut-test",
        region="us-central1",
        gcs_bucket="ut-test-gk-dataset",
        gx_container_image="us-central1-docker.pkg.dev/ut-test/aurora/gx@sha256:" + "a" * 64,
        worker_service_account_email=WORKER_SA,
        network=net,
        subnetwork=subnet,
        no_external_ip=True,
        transport=lambda *_a: {},
        state_dir=tmp_path / "state",
    )
    assert backend._regions() == ["us-central1"]
    run_dir = _run_dir(tmp_path)
    _, _, body = backend.build_job_spec(run_dir, parameters=SMALL)
    assert body["allocationPolicy"]["network"] == {
        "networkInterfaces": [
            {
                "network": net,
                "subnetwork": subnet,
                "noExternalIpAddress": True,
            }
        ]
    }
    # Single-region subnet refuses multi-region candidate_regions or mismatched region.
    with pytest.raises(ValueError, match="does not match region"):
        _backend(
            lambda *_a: {},
            tmp_path,
            region="us-east4",
            candidate_regions=["us-east4"],
            network=net,
            subnetwork=subnet,
        )
    with pytest.raises(ValueError, match="conflicts with candidate_regions"):
        _backend(
            lambda *_a: {},
            tmp_path,
            candidate_regions=["us-central1", "us-east4"],
            network=net,
            subnetwork=subnet,
        )
    monkeypatch.setenv("GCP_BATCH_CANDIDATE_REGIONS", "us-central1,us-east4")
    with pytest.raises(ValueError, match="conflicts with candidate_regions"):
        backend._regions()


def test_multi_region_subnetwork_template_and_invalid_vpc_inputs(tmp_path: Path) -> None:
    net = "projects/ut-test/global/networks/ut-test-vpc"
    tmpl = "projects/ut-test/regions/{region}/subnetworks/ut-test-subnet"
    backend = _backend(
        lambda *_a: {},
        tmp_path,
        candidate_regions=["us-central1", "us-east4"],
        network=net,
        subnetwork=tmpl,
        no_external_ip=False,
    )
    run_dir = _run_dir(tmp_path)
    _, _, b0 = backend.build_job_spec(run_dir, parameters=SMALL, rotation_index=0)
    _, _, b1 = backend.build_job_spec(run_dir, parameters=SMALL, rotation_index=1)
    iface0 = b0["allocationPolicy"]["network"]["networkInterfaces"][0]
    iface1 = b1["allocationPolicy"]["network"]["networkInterfaces"][0]
    assert iface0["subnetwork"] == "projects/ut-test/regions/us-central1/subnetworks/ut-test-subnet"
    assert iface0["noExternalIpAddress"] is False
    assert iface1["subnetwork"] == "projects/ut-test/regions/us-east4/subnetworks/ut-test-subnet"

    for bad_kw in (
        {"network": net, "subnetwork": None},
        {"network": None, "subnetwork": tmpl},
        {"network": "", "subnetwork": tmpl},
        {"network": "default", "subnetwork": tmpl},
        {"network": net, "subnetwork": "projects/ut-test/global/subnetworks/bad"},
        {"network": net, "subnetwork": tmpl, "no_external_ip": "true"},
        {"network": net, "subnetwork": tmpl, "no_external_ip": 1},
    ):
        with pytest.raises(ValueError):
            _backend(lambda *_a: {}, tmp_path, **bad_kw)


def test_ledger_survives_unrelated_run_dirs(tmp_path: Path, batch_server: Any) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir / LEDGER_FILENAME).write_text(json.dumps({"run_id": "someone_else"}), encoding="utf-8")
    assert load_record(run_dir, "gx_w7x_01") is None
    name = _backend(batch_server, tmp_path).submit(run_dir, parameters=SMALL)
    assert name in batch_server.jobs


def test_gene_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GX only"):
        _backend(lambda *_a: {}, tmp_path).submit(_run_dir(tmp_path), solver="GENE")
