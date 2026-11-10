"""Crash-safety, spend-safety and fail-safe tests for the TACC queue driver and watchdog.

Each test pins one production failure mode (see docs/review/FIXLOG-HPC.md for the IDs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hpc_core import (
    BackendCapabilities,
    DriverAlreadyRunningError,
    JobStatus,
    RunRecord,
    StateStore,
    TelemetryEmitter,
    acquire_single_instance_lock,
)
from hpc_core.client import TriageClientError, TriageTransientError

from tacc_exec.config import ConfigError, load_config, normalize_config
from tacc_exec.driver import QueueDriver
from tacc_exec.resolve import ResolveRefusedError, resolve_escalated_run
from tacc_exec.watchdog import (
    ClientGuardrailError,
    _fit_triage_payload_under_2kb,
    validate_directive_client_side,
)

GX_IN = (
    "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n[Time]\ndt = 0.02\ncfl = 0.8\nt_max = 200.0\n"
    "[Dissipation]\nhyper = true\nD_hyper = 0.05\n[Restart]\nrestart = false\n"
)


def _config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    cfg_path = Path(__file__).resolve().parents[1] / "config.example.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update(
        queue_dir=str(tmp_path / "queue"),
        state_dir=str(tmp_path / "state"),
        dataset_output_dir=str(tmp_path / "dataset"),
        aurora_steer_url="http://127.0.0.1:18080",
        cancel_wait_seconds=0,
    )
    cfg.update(overrides)
    return cfg


class FakeBackend:
    """In-memory SchedulerBackend with optional idempotent submission and delayed cancels."""

    def __init__(self, *, idempotent: bool = False, backend_name: str = "tacc_slurm") -> None:
        self.capabilities = BackendCapabilities(
            backend_name=backend_name, default_gpu_tier="gh200", idempotent_submit=idempotent
        )
        self.states: dict[str, JobStatus] = {}
        self.by_submission: dict[str, str] = {}
        self.submit_calls: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.counter = 0
        self.fail_submit: Exception | None = None
        self.cancel_keeps_running = False

    def submit(self, run_dir: Path, **kwargs: Any) -> str:
        self.submit_calls.append({"run_dir": str(run_dir), **kwargs})
        if self.fail_submit is not None:
            raise self.fail_submit
        sid = kwargs.get("submission_id")
        if sid and sid in self.by_submission:
            return self.by_submission[sid]
        self.counter += 1
        jid = f"job-{self.counter}"
        self.states[jid] = JobStatus(job_id=jid, state="RUNNING", elapsed_seconds=60.0)
        if sid:
            self.by_submission[sid] = jid
        return jid

    def find_submission(self, submission_id: str) -> str | None:
        return self.by_submission.get(submission_id)

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)
        if not self.cancel_keeps_running:
            self.states[job_id] = JobStatus(job_id=job_id, state="CANCELLED")

    def query_job(self, job_id: str) -> JobStatus:
        return self.states.get(job_id, JobStatus(job_id=job_id, state="UNKNOWN"))


class FakeSteer:
    def __init__(self, directive: dict[str, Any] | None = None, error: Exception | None = None):
        self.calls = 0
        self.error = error
        self.directive = directive or {
            "action": "warm_restart",
            "reason": "Reduce dt_max and cfl_fraction.",
            "parameter_modifications": {"dt_max": 0.005, "cfl_fraction": 0.5},
            "restart_from_checkpoint": True,
        }

    def triage(self, payload: dict[str, Any], *, idempotency_key: str | None = None) -> dict:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return dict(self.directive)

    def post_events(self, events: list[dict[str, Any]], timeout_s: float = 5.0) -> dict:
        return {"accepted": len(events), "deduplicated": 0}


def _make(
    tmp_path: Path,
    *,
    backend: FakeBackend | None = None,
    steer: FakeSteer | None = None,
    **cfg_overrides: Any,
) -> tuple[QueueDriver, FakeBackend, FakeSteer, StateStore, dict[str, Any]]:
    cfg = _config(tmp_path, **cfg_overrides)
    backend = backend or FakeBackend()
    steer = steer or FakeSteer()
    state_dir = Path(cfg["state_dir"])
    store = StateStore(state_dir / "state.json", state_dir / "audit.jsonl")
    emitter = TelemetryEmitter(state_dir / "outbox.jsonl", client=steer)
    driver = QueueDriver(
        backend=backend, state_store=store, steer_client=steer, telemetry=emitter, config=cfg
    )
    driver.watchdog.sleep_fn = lambda _s: None
    return driver, backend, steer, store, cfg


def _pending_run(cfg: dict[str, Any], run_id: str, *, eq: str = "mrx_eq_0001") -> Path:
    run_dir = Path(cfg["queue_dir"]) / "pending" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text(GX_IN, encoding="utf-8")
    (run_dir / "run_meta.json").write_text(
        json.dumps({"equilibrium_id": eq, "campaign_id": "pilot_50"}), encoding="utf-8"
    )
    return run_dir


def _reload(cfg: dict[str, Any]) -> StateStore:
    state_dir = Path(cfg["state_dir"])
    return StateStore(state_dir / "state.json", state_dir / "audit.jsonl")


# --------------------------------------------------------------------------- ingestion


def test_pending_duplicate_of_active_run_is_quarantined_not_rmtreed(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path)
    _pending_run(cfg, "gx_dup_01")
    driver.tick()
    rec = store.get_run("gx_dup_01")
    assert rec is not None and rec.status == "RUNNING"
    live_dir = Path(rec.run_dir)
    (live_dir / "gx.out.nc").write_bytes(b"precious output")

    _pending_run(cfg, "gx_dup_01")  # same run_id dropped again while the job runs
    driver.tick()

    assert (live_dir / "gx.out.nc").read_bytes() == b"precious output"
    assert len(backend.submit_calls) == 1
    quarantined = list((Path(cfg["queue_dir"]) / "needs_human").glob("quarantined_gx_dup_01*"))
    assert len(quarantined) == 1 and (quarantined[0] / "ISSUE.md").is_file()


def test_completed_run_id_reuse_is_refused(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path)
    store.upsert_run(RunRecord(run_id="gx_done", solver="GX", run_dir="/x", status="COMPLETED"))
    _pending_run(cfg, "gx_done")
    driver.tick()
    assert backend.submit_calls == []
    assert store.get_run("gx_done").status == "COMPLETED"


def test_prod_run_without_equilibrium_id_is_quarantined_before_spending(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path)
    run_dir = Path(cfg["queue_dir"]) / "pending" / "gx_noeq"
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text(GX_IN, encoding="utf-8")
    driver.tick()
    assert backend.submit_calls == []
    assert list((Path(cfg["queue_dir"]) / "needs_human").glob("quarantined_gx_noeq*"))


def test_unsafe_run_meta_ids_are_quarantined(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path)
    _pending_run(cfg, "gx_traversal", eq="../../etc")
    driver.tick()
    assert backend.submit_calls == []


def test_two_quarantines_with_same_slug_never_overwrite_each_other(tmp_path: Path) -> None:
    driver, _backend, _steer, _store, cfg = _make(tmp_path)
    for name in ("bad name!", "bad name?"):
        d = Path(cfg["queue_dir"]) / "pending" / name
        d.mkdir(parents=True)
        (d / "note.txt").write_text(name, encoding="utf-8")
        driver.tick()
    found = sorted((Path(cfg["queue_dir"]) / "needs_human").glob("quarantined_bad_name_*"))
    assert len(found) == 2
    assert {(f / "note.txt").read_text(encoding="utf-8") for f in found} == {
        "bad name!",
        "bad name?",
    }


# --------------------------------------------------------------------------- submission intent


def test_submission_intent_is_persisted_before_the_scheduler_call(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path)
    seen: list[str] = []
    original_submit = backend.submit

    def spying_submit(run_dir: Path, **kwargs: Any) -> str:
        on_disk = json.loads((Path(cfg["state_dir"]) / "state.json").read_text(encoding="utf-8"))
        seen.append(on_disk["runs"]["gx_intent"]["status"])
        return original_submit(run_dir, **kwargs)

    backend.submit = spying_submit  # type: ignore[method-assign]
    _pending_run(cfg, "gx_intent")
    driver.tick()
    assert seen == ["SUBMITTING"]
    rec = _reload(cfg).get_run("gx_intent")
    assert rec.status == "RUNNING" and rec.slurm_job_id == "job-1"
    assert rec.restart_history[0]["kind"] == "initial"
    assert rec.input_sha256_submitted


def _crash_after_intent(cfg: dict[str, Any], backend: FakeBackend, *, job_created: bool) -> str:
    """Simulate a crash between persisting the intent and recording the job id."""
    store = _reload(cfg)
    rec = RunRecord(run_id="gx_crash", solver="GX", run_dir="", equilibrium_id="mrx_eq_1")
    running = Path(cfg["queue_dir"]) / "running" / "gx_crash"
    running.mkdir(parents=True)
    (running / "gx.in").write_text(GX_IN, encoding="utf-8")
    rec.run_dir = str(running)
    rec.submit_nonce = "abcd1234"
    rec.submit_seq = 1
    sid = "bsit-gx_crash-abcd1234-1"
    rec.pending_submission = {"submission_id": sid, "kind": "initial"}
    rec.status = "SUBMITTING"
    store.upsert_run(rec)
    if job_created:
        backend.submit(running, submission_id=sid)
        backend.submit_calls.clear()
    return sid


def test_crash_after_job_creation_is_adopted_not_resubmitted(tmp_path: Path) -> None:
    backend = FakeBackend(idempotent=True)
    cfg = _config(tmp_path)
    _crash_after_intent(cfg, backend, job_created=True)
    driver, backend, _steer, store, _cfg = _make(tmp_path, backend=backend)
    driver.tick()
    rec = store.get_run("gx_crash")
    assert rec.status == "RUNNING" and rec.slurm_job_id == "job-1"
    assert backend.submit_calls == []  # adopted via find_submission, no second job


def test_crash_before_job_creation_resubmits_with_same_submission_id(tmp_path: Path) -> None:
    backend = FakeBackend(idempotent=True)
    cfg = _config(tmp_path)
    sid = _crash_after_intent(cfg, backend, job_created=False)
    driver, backend, _steer, store, _cfg = _make(tmp_path, backend=backend)
    driver.tick()
    assert [c["submission_id"] for c in backend.submit_calls] == [sid]
    assert store.get_run("gx_crash").status == "RUNNING"


def test_interrupted_submission_without_idempotent_backend_escalates(tmp_path: Path) -> None:
    backend = FakeBackend(idempotent=False)
    cfg = _config(tmp_path)
    sid = _crash_after_intent(cfg, backend, job_created=False)
    driver, backend, _steer, store, _cfg = _make(tmp_path, backend=backend)
    driver.tick()
    rec = store.get_run("gx_crash")
    assert rec.status == "NEEDS_HUMAN"
    assert backend.submit_calls == []  # never a blind resubmission
    assert sid in (Path(rec.run_dir) / "ISSUE.md").read_text(encoding="utf-8")


def test_submit_failure_escalates_and_records_submission_id(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path)
    backend.fail_submit = RuntimeError("sbatch: error: Batch job submission failed")
    _pending_run(cfg, "gx_subfail")
    driver.tick()
    rec = store.get_run("gx_subfail")
    assert rec.status == "NEEDS_HUMAN"
    issue = (Path(rec.run_dir) / "ISSUE.md").read_text(encoding="utf-8")
    assert "bsit-gx_subfail-" in issue


class _OutcomeUnknown(RuntimeError):
    outcome_unknown = True


def test_unknown_submit_outcome_stays_submitting_then_recovers(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path, backend=FakeBackend(idempotent=True))
    backend.fail_submit = _OutcomeUnknown("sbatch: Socket timed out on send/recv operation")
    _pending_run(cfg, "gx_slowctld")
    driver.tick()
    rec = store.get_run("gx_slowctld")
    assert rec.status == "SUBMITTING" and rec.consecutive_poll_errors == 1
    backend.fail_submit = None
    driver.tick()  # reconcile: find_submission -> none -> idempotent resubmit, same id
    assert rec.status == "RUNNING"
    sids = {c["submission_id"] for c in backend.submit_calls}
    assert len(sids) == 1


def test_scheduler_outage_is_deferred_not_counted_or_escalated(tmp_path: Path) -> None:
    from hpc_core.slurm import SlurmQueryError

    driver, backend, _steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)

    def outage(_job_id: str) -> JobStatus:
        raise SlurmQueryError("squeue and sacct failed: slurm_load_jobs error")

    backend.query_job = outage  # type: ignore[method-assign]
    for _ in range(5):
        driver.tick()
    assert rec.status == "RUNNING" and rec.consecutive_poll_errors == 0
    assert backend.cancelled == []


# --------------------------------------------------------------------------- reconcile


def test_untracked_running_folder_is_quarantined_not_deleted(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path)
    orphan = Path(cfg["queue_dir"]) / "running" / "gx_orphan"
    orphan.mkdir(parents=True)
    (orphan / "gx.in").write_text(GX_IN, encoding="utf-8")
    driver.tick()
    assert not orphan.exists()
    moved = list((Path(cfg["queue_dir"]) / "needs_human").glob("untracked_gx_orphan*"))
    assert len(moved) == 1 and (moved[0] / "gx.in").is_file()
    assert backend.submit_calls == []


def test_vanished_run_dir_escalates_and_cancels_job(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path)
    _pending_run(cfg, "gx_gone")
    driver.tick()
    rec = store.get_run("gx_gone")
    import shutil

    shutil.rmtree(rec.run_dir)
    driver.tick()
    assert rec.status == "NEEDS_HUMAN"
    assert backend.cancelled == ["job-1"]


def test_poll_errors_are_persisted_across_cron_restarts(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    driver, backend, _steer, store, _ = _make(tmp_path)
    _pending_run(cfg, "gx_err")
    driver.tick()
    rec = store.get_run("gx_err")
    (Path(rec.run_dir) / "gx.in").write_text("not = [valid toml", encoding="utf-8")
    for _ in range(3):  # each tick is a fresh process in cron --once mode
        d2, _b, _s, s2, _c = _make(tmp_path, backend=backend)
        d2.tick()
    assert _reload(cfg).get_run("gx_err").status == "NEEDS_HUMAN"
    assert backend.cancelled == ["job-1"]


# --------------------------------------------------------------------------- scheduler states


def _running_record(driver: QueueDriver, cfg: dict[str, Any], run_id: str = "gx_s") -> RunRecord:
    _pending_run(cfg, run_id)
    driver.tick()
    rec = driver.state.get_run(run_id)
    assert rec is not None and rec.status == "RUNNING"
    return rec


def test_external_cancel_escalates_without_triage(tmp_path: Path) -> None:
    driver, backend, steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    backend.states[rec.slurm_job_id] = JobStatus(
        job_id=rec.slurm_job_id,
        state="CANCELLED",
        reason="cancelled by uid=0",
        cancelled_externally=True,
    )
    driver.tick()
    assert rec.status == "NEEDS_HUMAN" and steer.calls == 0
    assert len(backend.submit_calls) == 1


def test_timeout_without_checkpoint_escalates(tmp_path: Path) -> None:
    driver, backend, steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="TIMEOUT")
    driver.tick()
    assert rec.status == "NEEDS_HUMAN" and len(backend.submit_calls) == 1 and steer.calls == 0


def test_timeout_with_checkpoint_resumes_from_it(tmp_path: Path) -> None:
    driver, backend, steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    (Path(rec.run_dir) / "gx.restart.nc").write_bytes(b"ckpt")
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="TIMEOUT")
    driver.tick()
    assert rec.status == "RUNNING" and rec.walltime_resubmits == 1 and steer.calls == 0
    assert rec.restart_history[-1]["restart_from_checkpoint"] is True
    assert "restart = true" in (Path(rec.run_dir) / "gx.in").read_text(encoding="utf-8")


def test_preemption_without_checkpoint_cold_restarts(tmp_path: Path) -> None:
    driver, backend, steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="NODE_FAIL")
    driver.tick()
    assert rec.status == "RUNNING" and rec.walltime_resubmits == 1 and steer.calls == 0
    assert rec.restart_history[-1]["restart_from_checkpoint"] is False


def test_blocked_pending_job_escalates_and_is_cancelled(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    backend.states[rec.slurm_job_id] = JobStatus(
        job_id=rec.slurm_job_id, state="PENDING", reason="AssocGrpBillingMinutes", blocked=True
    )
    driver.tick()
    assert rec.status == "NEEDS_HUMAN" and backend.cancelled == ["job-1"]


def test_pending_forever_escalates_after_max_pending_hours(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path, max_pending_hours=1)
    rec = _running_record(driver, cfg)
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="PENDING")
    rec.submitted_at_epoch -= 7200
    driver.tick()
    assert rec.status == "NEEDS_HUMAN" and backend.cancelled == ["job-1"]


def test_midrun_failure_never_patches_while_job_still_active(tmp_path: Path) -> None:
    driver, backend, steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    backend.cancel_keeps_running = True
    (Path(rec.run_dir) / "diagnostics_metrics.json").write_text(
        json.dumps({"failure_type": "TIMESTEP_COLLAPSE", "sim_time_reached": 12.0}),
        encoding="utf-8",
    )
    before = (Path(rec.run_dir) / "gx.in").read_text(encoding="utf-8")
    driver.tick()
    assert steer.calls == 0 and len(backend.submit_calls) == 1
    assert (Path(rec.run_dir) / "gx.in").read_text(encoding="utf-8") == before
    assert rec.cancel_requested is not None

    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="CANCELLED")
    driver.tick()  # our own cancel completed: handle the stored failure, not an external cancel
    assert steer.calls == 1 and rec.status == "RUNNING" and rec.retry_count == 1
    assert rec.cancel_requested is None


# --------------------------------------------------------------------------- aurora-steer fail-safe


def test_simulated_completed_run_is_harvested_ineligible_not_retriaged(tmp_path: Path) -> None:
    driver, backend, steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    (Path(rec.run_dir) / "diagnostics_metrics.json").write_text(
        json.dumps(
            {
                "converged": False,
                "simulated": True,
                "sim_time_reached": 200.0,
                "scalar_metrics": {"heat_flux_es": 1.0},
            }
        ),
        encoding="utf-8",
    )
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="COMPLETED")
    driver.tick()
    assert rec.status == "COMPLETED" and steer.calls == 0 and len(backend.submit_calls) == 1
    manifest = json.loads(Path(rec.dataset_manifest_path).read_text(encoding="utf-8"))
    assert manifest["dataset_eligible"] is False and "simulated" in manifest["eligibility_reasons"]


def test_flux_window_stats_overflow_raises_and_min_samples_is_guarded() -> None:
    from tacc_exec.timeseries import flux_window_stats

    with pytest.raises(OverflowError):
        flux_window_stats([1, 2, 3, 4], [1.7e308, 1.7e308, 1.7e308, 1.7e308])
    assert flux_window_stats([1.0, 2.0], [1.0, 2.0], min_samples=1) is not None


def _fail_run(driver: QueueDriver, backend: FakeBackend, rec: RunRecord) -> None:
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="FAILED")
    driver.tick()


def test_transient_triage_errors_defer_then_escalate_without_resubmitting(tmp_path: Path) -> None:
    steer = FakeSteer(error=TriageTransientError("503 Service Unavailable"))
    driver, backend, _s, _store, cfg = _make(tmp_path, steer=steer, max_triage_deferrals=2)
    rec = _running_record(driver, cfg)
    _fail_run(driver, backend, rec)
    assert rec.status == "RUNNING" and rec.triage_deferrals == 1
    driver.tick()
    assert rec.status == "RUNNING" and rec.triage_deferrals == 2
    driver.tick()
    assert rec.status == "NEEDS_HUMAN" and len(backend.submit_calls) == 1


def test_permanent_triage_error_escalates_immediately(tmp_path: Path) -> None:
    steer = FakeSteer(error=TriageClientError("Invalid JSON in aurora-steer response"))
    driver, backend, _s, _store, cfg = _make(tmp_path, steer=steer)
    rec = _running_record(driver, cfg)
    _fail_run(driver, backend, rec)
    assert rec.status == "NEEDS_HUMAN" and len(backend.submit_calls) == 1


def test_warm_restart_without_checkpoint_downgrades_to_cold(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg)
    _fail_run(driver, backend, rec)
    assert rec.status == "RUNNING" and rec.retry_count == 1
    patch = rec.parameter_patches[-1]
    assert patch["restart_from_checkpoint"] is False
    assert patch["warm_restart_downgraded_no_checkpoint"] is True
    assert patch["input_sha256_before"] != patch["input_sha256_after"]


def test_client_guardrail_warm_restart_allowlist_and_actions(tmp_path: Path) -> None:
    allowed = _config(tmp_path)["allowed_parameters"]
    with pytest.raises(ClientGuardrailError, match="cold_restart"):
        validate_directive_client_side(
            {
                "action": "warm_restart",
                "parameter_modifications": {"hyperdiffusion_coeff": 0.2},
                "restart_from_checkpoint": True,
            },
            allowed,
        )
    with pytest.raises(ClientGuardrailError, match="not accepted"):
        validate_directive_client_side(
            {
                "action": "resubmit_walltime",
                "parameter_modifications": {},
                "restart_from_checkpoint": True,
            },
            allowed,
        )
    with pytest.raises(ClientGuardrailError, match="boolean"):
        validate_directive_client_side(
            {
                "action": "warm_restart",
                "parameter_modifications": {"dt_max": 0.01},
                "restart_from_checkpoint": "false",
            },
            allowed,
        )
    without_flag = {k: dict(v) for k, v in allowed.items()}
    without_flag["dt_max"].pop("warm_restart_allowed")
    with pytest.raises(ClientGuardrailError):
        validate_directive_client_side(
            {
                "action": "warm_restart",
                "parameter_modifications": {"dt_max": 0.01},
                "restart_from_checkpoint": True,
            },
            without_flag,
        )
    assert validate_directive_client_side(
        {
            "action": "cold_restart",
            "parameter_modifications": {"hyperdiffusion_coeff": 0.2, "nx": 96},
            "restart_from_checkpoint": False,
        },
        allowed,
    ) == {"hyperdiffusion_coeff": 0.2, "nx": 96}


def test_triage_payload_fit_is_valid_json_under_2kb_for_adversarial_input() -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": "r" * 128,
        "solver": "GX",
        "failure_type": "CRASH",
        "retry_count": 1,
        "sim_time_reached": 1.0,
        "sim_time_target": 2.0,
        "current_parameters": {f"p{i}": float(i) for i in range(9)},
        "scalar_metrics": {f"metric_{i:02d}": 1.0e-7 * i for i in range(40)},
        "log_tail": ["é" * 180 for _ in range(30)],
        "previous_directives": [
            {"action": "warm_restart", "parameter_modifications": {"dt_max": 0.01}}
        ]
        * 8,
    }
    fitted = _fit_triage_payload_under_2kb(payload)
    encoded = json.dumps(fitted, allow_nan=False).encode("utf-8")
    assert len(encoded) < 2048
    assert json.loads(encoded)["run_id"] == "r" * 128


# --------------------------------------------------------------------------- escalation / dry run


def test_escalation_never_deletes_an_existing_needs_human_folder(tmp_path: Path) -> None:
    driver, backend, _steer, _store, cfg = _make(tmp_path, max_retries=0)
    rec = _running_record(driver, cfg)
    stale = Path(cfg["queue_dir"]) / "needs_human" / rec.run_id
    stale.mkdir(parents=True)
    (stale / "human_notes.txt").write_text("keep me", encoding="utf-8")
    _fail_run(driver, backend, rec)
    assert (stale / "human_notes.txt").read_text(encoding="utf-8") == "keep me"
    assert rec.status == "NEEDS_HUMAN" and Path(rec.run_dir) != stale


def test_dry_run_never_moves_folders_even_when_escalation_is_due(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path, max_unknown_polls=1)
    rec = _running_record(driver, cfg)
    backend.states.clear()  # scheduler now answers UNKNOWN
    state_dir = Path(cfg["state_dir"])
    dry = QueueDriver(
        backend=backend,
        state_store=StateStore(state_dir / "state.json", state_dir / "audit.jsonl", dry_run=True),
        steer_client=None,
        telemetry=TelemetryEmitter(state_dir / "outbox.jsonl"),
        config=cfg,
        dry_run=True,
    )
    dry.tick()
    assert Path(rec.run_dir).is_dir()
    assert backend.cancelled == []


# --------------------------------------------------------------------------- config / resolve


def test_config_paths_resolve_against_config_dir_not_cwd(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path, queue_dir="./queue", state_dir="./state", dataset_output_dir="ds")
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded = load_config(cfg_dir / "config.json")
    assert loaded["queue_dir"] == str((cfg_dir / "queue").resolve())
    assert loaded["dataset_output_dir"] == str((cfg_dir / "ds").resolve())

    monkeypatch.delenv("BSIT_UNSET_VAR", raising=False)
    with pytest.raises(ConfigError, match="unset environment variable"):
        normalize_config({**cfg, "queue_dir": "$BSIT_UNSET_VAR/queue"}, cfg_dir)


def test_config_allowlist_drift_is_refused(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    stale = json.loads(json.dumps(cfg))
    stale["allowed_parameters"]["hyperdiffusion_coeff"]["gx_path"] = "Dissipation.hyperviscosity"
    with pytest.raises(ConfigError, match="gx_path"):
        normalize_config(stale, tmp_path)
    warm = json.loads(json.dumps(cfg))
    warm["allowed_parameters"]["nx"]["warm_restart_allowed"] = True
    with pytest.raises(ConfigError, match="warm_restart_allowed"):
        normalize_config(warm, tmp_path)
    nan = json.loads(json.dumps(cfg))
    nan["allowed_parameters"]["nx"]["max"] = float("inf")
    with pytest.raises(ConfigError, match="finite"):
        normalize_config(nan, tmp_path)


def test_resolve_refuses_when_previous_job_is_still_active(tmp_path: Path) -> None:
    driver, backend, _steer, store, cfg = _make(tmp_path)
    rec = _running_record(driver, cfg, "gx_res")
    rec.status = "NEEDS_HUMAN"
    human = Path(cfg["queue_dir"]) / "needs_human" / "gx_res"
    Path(rec.run_dir).rename(human)
    rec.run_dir = str(human)
    store.upsert_run(rec)
    with pytest.raises(ResolveRefusedError, match="still RUNNING"):
        resolve_escalated_run(config=cfg, run_id="gx_res", backend=backend)
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="FAILED")
    dest = resolve_escalated_run(config=cfg, run_id="gx_res", backend=backend)
    assert dest.is_dir()
    assert _reload(cfg).get_run("gx_res").human_resolutions == 1
    with pytest.raises(ValueError):
        resolve_escalated_run(config=cfg, run_id="../running/x")


def test_resolve_waits_for_the_driver_lock(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    lock_path = Path(cfg["state_dir"]) / "driver.lock"
    with acquire_single_instance_lock(lock_path), pytest.raises(DriverAlreadyRunningError):
        resolve_escalated_run(config=cfg, run_id="gx_any", lock_wait_s=0.0)
