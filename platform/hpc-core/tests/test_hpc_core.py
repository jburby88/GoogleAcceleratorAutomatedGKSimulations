"""Unit tests for `platform/hpc-core`: flock, poison-event quarantine, Slurm normalization, manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hpc_core.client import (
    AuroraSteerClient,
    EventsValidationError,
    validate_aurora_steer_url,
)
from hpc_core.manifest import harvest_completed_run_to_dataset
from hpc_core.slurm import normalize_slurm_state
from hpc_core.state import (
    DriverAlreadyRunningError,
    RunRecord,
    StateStore,
    acquire_single_instance_lock,
)
from hpc_core.telemetry import TelemetryEmitter


def test_single_instance_flock_blocks_second_holder(tmp_path: Path) -> None:
    lock_file = tmp_path / "state.lock"
    with (
        acquire_single_instance_lock(lock_file),
        pytest.raises(DriverAlreadyRunningError),
        acquire_single_instance_lock(lock_file),
    ):
        pass
    # Lock released cleanly
    with acquire_single_instance_lock(lock_file):
        assert lock_file.is_file()


def test_poison_event_422_quarantined_without_blocking_valid_events(
    tmp_path: Path,
) -> None:
    delivered: list[str] = []

    class FakeClient:
        def post_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
            for ev in events:
                if ev["run_id"] == "POISON_RUN":
                    raise EventsValidationError("Invalid event schema", status_code=422)
            delivered.extend(ev["run_id"] for ev in events)
            return {"status": "ok"}

    outbox = tmp_path / "outbox.jsonl"
    emitter = TelemetryEmitter(outbox_file=outbox, client=FakeClient())
    emitter.emit(event_type="submitted", run_id="GOOD_1", solver="GX")
    emitter.emit(event_type="submitted", run_id="POISON_RUN", solver="GX")
    emitter.emit(event_type="completed", run_id="GOOD_2", solver="GX")

    sent = emitter.flush(force=True)
    assert sent == 2
    assert delivered == ["GOOD_1", "GOOD_2"]
    assert emitter.read_pending_events() == []
    assert emitter.quarantine_file.is_file()
    quarantined_lines = emitter.quarantine_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(quarantined_lines) == 1
    assert json.loads(quarantined_lines[0])["event"]["run_id"] == "POISON_RUN"


def test_slurm_state_normalization_and_https_url_validation() -> None:
    assert normalize_slurm_state("R") == "RUNNING"
    assert normalize_slurm_state("CANCELLED+") == "CANCELLED"
    assert normalize_slurm_state("TO") == "TIMEOUT"
    assert normalize_slurm_state("PR") == "PREEMPTED"
    assert normalize_slurm_state("") == "UNKNOWN"

    assert (
        validate_aurora_steer_url("https://aurora-steer.run.app/") == "https://aurora-steer.run.app"
    )
    assert validate_aurora_steer_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    with pytest.raises(ValueError):
        validate_aurora_steer_url("http://aurora-steer.run.app")

    loopback_client = AuroraSteerClient(base_url="http://127.0.0.1:8080", api_token="tok")
    assert loopback_client._session.trust_env is False


def test_zenodo_dataset_harvester_writes_manifest_and_fluxes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_w7x_01"
    run_dir.mkdir()
    (run_dir / "gx.in").write_text("[Dimensions]\nnx = 64\n", encoding="utf-8")
    (run_dir / "w7x.eik.nc").write_bytes(b"mock-geometry")

    rec = RunRecord(
        run_id="run_w7x_01",
        solver="GX",
        run_dir=str(run_dir),
        status="COMPLETED",
        retry_count=1,
        backend="gcp_batch",
        gpu_tier="nvidia-l4",
        equilibrium_id="w7x_standard",
        parameter_patches=[{"retry": 0, "parameter": "dt_max", "old": 0.02, "new": 0.01}],
    )
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    store.upsert_run(rec)

    dataset_root = tmp_path / "dataset"
    manifest_path = harvest_completed_run_to_dataset(
        rec,
        run_dir,
        dataset_root,
        final_parameters={"nx": 64, "dt_max": 0.01},
        final_metrics={
            "heat_flux_es": 4.25,
            "heat_flux_em": 0.35,
            "particle_flux": 0.12,
            "maxwell_reynolds_stress": -0.084,
            "momentum_flux": -0.062,
            "growth_rate_gamma": 0.31,
        },
        walltime_elapsed_s=420.5,
        gcs_uri_prefix="gs://ut-austin-gk-dataset",
    )

    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["equilibrium_id"] == "w7x_standard"
    assert manifest["hands_free"] is True
    assert manifest["gpu_tier"] == "nvidia-l4"
    assert manifest["gcs_uri"] == "gs://ut-austin-gk-dataset/campaigns/pilot_50/runs/run_w7x_01"
    assert (
        dataset_root / "campaigns" / "pilot_50" / "equilibria" / "w7x_standard" / "w7x.eik.nc"
    ).is_file()

    fluxes_path = manifest_path.parent / "fluxes.json"
    fluxes = json.loads(fluxes_path.read_text(encoding="utf-8"))
    assert fluxes["heat_flux_es"] == pytest.approx(4.25)
    assert fluxes["maxwell_reynolds_stress"] == pytest.approx(-0.084)
    assert fluxes["momentum_flux"] == pytest.approx(-0.062)


def test_zenodo_harvester_marks_human_resolved_runs_not_hands_free(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"

    # Case 1: ISSUE.resolved.md present (e.g., 3-strike local circuit-breaker followed by tacc-resolve)
    run_dir_resolved = tmp_path / "run_resolved"
    run_dir_resolved.mkdir()
    (run_dir_resolved / "gx.in").write_text("[Dimensions]\nnx = 64\n", encoding="utf-8")
    (run_dir_resolved / "ISSUE.resolved.md").write_text("# Resolved\n", encoding="utf-8")
    rec_resolved = RunRecord(
        run_id="run_resolved",
        solver="GX",
        run_dir=str(run_dir_resolved),
        status="COMPLETED",
    )
    m_resolved = json.loads(
        harvest_completed_run_to_dataset(
            rec_resolved,
            run_dir_resolved,
            dataset_root,
            final_parameters={"nx": 64},
            final_metrics={"heat_flux_es": 2.1},
        ).read_text(encoding="utf-8")
    )
    assert m_resolved["hands_free"] is False

    # Case 2: ISSUE.md present in run_dir
    run_dir_issue = tmp_path / "run_issue"
    run_dir_issue.mkdir()
    (run_dir_issue / "ISSUE.md").write_text("# Escalated\n", encoding="utf-8")
    rec_issue = RunRecord(
        run_id="run_issue",
        solver="GX",
        run_dir=str(run_dir_issue),
        status="COMPLETED",
    )
    m_issue = json.loads(
        harvest_completed_run_to_dataset(
            rec_issue,
            run_dir_issue,
            dataset_root,
            final_parameters={"nx": 64},
            final_metrics={"heat_flux_es": 2.1},
        ).read_text(encoding="utf-8")
    )
    assert m_issue["hands_free"] is False

    # Case 3: directive history contains escalate_to_human
    run_dir_dir = tmp_path / "run_dir_esc"
    run_dir_dir.mkdir()
    rec_dir = RunRecord(
        run_id="run_dir_esc",
        solver="GX",
        run_dir=str(run_dir_dir),
        status="COMPLETED",
        directives_history=[{"action": "escalate_to_human", "reason": "Manual review"}],
    )
    m_dir = json.loads(
        harvest_completed_run_to_dataset(
            rec_dir,
            run_dir_dir,
            dataset_root,
            final_parameters={"nx": 64},
            final_metrics={"heat_flux_es": 2.1},
        ).read_text(encoding="utf-8")
    )
    assert m_dir["hands_free"] is False
