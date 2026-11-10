"""Unit tests for `simulations/hpc` (patchers, guardrails, watchdog, driver)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from hpc_core import BackendCapabilities, JobStatus, StateStore, TelemetryEmitter

from tacc_exec.diagnostics_adapter import (
    consume_stale_diagnostics_metrics,
    inspect_run_directory,
)
from tacc_exec.driver import QueueDriver
from tacc_exec.patchers import (
    patch_gene_parameters,
    patch_gx_toml,
    read_gx_parameters,
)
from tacc_exec.preflight import run_preflight
from tacc_exec.resolve import resolve_escalated_run
from tacc_exec.watchdog import (
    ClientGuardrailError,
    SimulationWatchdog,
    validate_directive_client_side,
)

FORBIDDEN_IMPORT_PREFIXES = (
    "google",
    "vertexai",
    "openai",
    "anthropic",
    "langchain",
    "litellm",
    "fastapi",
    "pydantic",
    "streamlit",
    "duckdb",
    "kubernetes",
)


def _load_example_config(tmp_path: Path) -> dict[str, Any]:
    cfg_path = Path(__file__).resolve().parents[1] / "config.example.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["queue_dir"] = str(tmp_path / "queue")
    cfg["state_dir"] = str(tmp_path / "state")
    cfg["dataset_output_dir"] = str(tmp_path / "dataset")
    cfg["aurora_steer_url"] = "http://127.0.0.1:18080"
    return cfg


ALLOWED_ROOT_MODULES = set(sys.stdlib_module_names) | {
    "hpc_core",
    "tacc_exec",
    "requests",
    "f90nml",
    "tomlkit",
}


def _scan_ast_for_violations(code: str) -> list[str]:
    errors = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"__import__", "exec", "eval"}:
                    errors.append(f"Forbidden call {node.func.id}")
                elif node.func.id == "getattr":
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "builtins"
                    ):
                        errors.append("Forbidden getattr(builtins, ...)")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr == "import_module":
                    errors.append("Forbidden call import_module")
        for mod in mods:
            if mod.split(".")[0] not in ALLOWED_ROOT_MODULES:
                errors.append(f"Forbidden import {mod}")
    return errors


def test_zero_ai_or_cloud_imports_in_hpc_packages() -> None:
    hpc_dir = Path(__file__).resolve().parents[1] / "tacc_exec"
    core_dir = Path(__file__).resolve().parents[3] / "platform" / "hpc-core" / "hpc_core"
    for py_file in [*hpc_dir.rglob("*.py"), *core_dir.rglob("*.py")]:
        errors = _scan_ast_for_violations(py_file.read_text(encoding="utf-8"))
        assert not errors, f"Violations in {py_file}: {errors}"


@pytest.mark.parametrize(
    "mutation",
    [
        "import google.cloud",
        "import vertexai",
        "import openai",
        "import langchain",
        "from google.cloud import storage",
        "import pydantic",
        "import streamlit",
        "__import__('google.cloud')",
        "import importlib; importlib.import_module('google.cloud')",
        "exec('import google')",
        "eval('__import__(\"google\")')",
        "import builtins; getattr(builtins, '__import__')('google')",
        "import app.models",
    ],
)
def test_rep3_mutation_classes_rejected(mutation: str) -> None:
    errors = _scan_ast_for_violations(mutation)
    assert errors, f"Mutation not caught: {mutation}"


def test_gene_and_gx_patchers_preserve_physics_and_comments(tmp_path: Path) -> None:
    gene_file = tmp_path / "parameters"
    gene_file.write_text(
        "&box\n  nx0 = 64\n  nky0 = 32\n  nz0 = 48\n/\n"
        "&general\n  dt_max = 0.02\n  courant = 0.8\n  hyp_z = 0.25\n  beta = 0.035\n/\n",
        encoding="utf-8",
    )
    after_gene = patch_gene_parameters(
        gene_file,
        modifications={"dt_max": 0.005, "cfl_fraction": 0.5},
        restart_from_checkpoint=True,
    )
    assert after_gene["dt_max"] == pytest.approx(0.005)
    assert after_gene["cfl_fraction"] == pytest.approx(0.5)
    assert "0.035" in gene_file.read_text(encoding="utf-8")
    assert (tmp_path / "parameters.bak").is_file()

    gx_file = tmp_path / "gx.in"
    gx_file.write_text(
        "# Physics comment preserved!\n"
        "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n\n"
        "[Time]\ndt = 0.02\ncfl = 0.8\n\n"
        "[Dissipation]\nhyperviscosity = 0.1\n\n"
        "[Physics]\nbeta = 0.04\n",
        encoding="utf-8",
    )
    after_gx = patch_gx_toml(
        gx_file,
        modifications={"nx": 96, "nz": 64, "hyperdiffusion_coeff": 0.2},
        restart_from_checkpoint=False,
    )
    assert after_gx["nx"] == 96
    assert after_gx["nz"] == 64
    raw_after = gx_file.read_text(encoding="utf-8")
    assert "# Physics comment preserved!" in raw_after
    assert "beta = 0.04" in raw_after
    assert read_gx_parameters(gx_file)["nz"] == 64


def test_client_guardrails_reject_invalid_directives(tmp_path: Path) -> None:
    cfg = _load_example_config(tmp_path)
    allowed = cfg["allowed_parameters"]

    with pytest.raises(ClientGuardrailError, match="not in client allowlist"):
        validate_directive_client_side(
            {
                "action": "warm_restart",
                "parameter_modifications": {"beta": 0.05},
                "restart_from_checkpoint": True,
            },
            allowed,
        )

    with pytest.raises(ClientGuardrailError, match="outside"):
        validate_directive_client_side(
            {
                "action": "cold_restart",
                "parameter_modifications": {"nx": 1024},
                "restart_from_checkpoint": False,
            },
            allowed,
        )

    # Grid resolution changes (nx, ny, nz) must be rejected on warm_restart / restart_from_checkpoint=True
    with pytest.raises(ClientGuardrailError, match="Grid resolution changes"):
        validate_directive_client_side(
            {
                "action": "warm_restart",
                "parameter_modifications": {"nx": 96},
                "restart_from_checkpoint": True,
            },
            allowed,
        )


class FakeBackend:
    """In-memory SchedulerBackend for testing timeout, Spot preemption, OOM, unknowns, and dry-run."""

    def __init__(self, backend_name: str = "tacc_slurm", default_gpu: str = "gh200") -> None:
        self.capabilities = BackendCapabilities(
            backend_name=backend_name,
            default_gpu_tier=default_gpu,
            supports_spot_continuation=(backend_name == "gcp_batch"),
        )
        self.states: dict[str, JobStatus] = {}
        self.counter = 0
        self.fail_on_resubmit = False
        self.cancelled: list[str] = []

    def submit(
        self,
        run_dir: Path,
        *,
        solver: str = "GX",
        parameters: dict[str, Any] | None = None,
        retry_count: int = 0,
        force_high_memory: bool = False,
    ) -> str:
        if self.fail_on_resubmit and retry_count > 0:
            raise RuntimeError("Simulated scheduler outage on resubmit")
        self.counter += 1
        jid = f"job-{self.counter}"
        tier = "a100" if force_high_memory else self.capabilities.default_gpu_tier
        self.states[jid] = JobStatus(
            job_id=jid, state="RUNNING", elapsed_seconds=120.0, gpu_tier=tier
        )
        return jid

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)
        self.states[job_id] = JobStatus(job_id=job_id, state="CANCELLED")

    def query_job(self, job_id: str) -> JobStatus:
        return self.states.get(job_id, JobStatus(job_id=job_id, state="UNKNOWN"))


class FakeSteerClient:
    def __init__(self) -> None:
        self.calls = 0

    def triage(
        self, diagnostic_summary: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "action": "warm_restart",
            "reason": "Reduce dt_max and cfl_fraction.",
            "parameter_modifications": {"dt_max": 0.005, "cfl_fraction": 0.5},
            "restart_from_checkpoint": True,
            "confidence": "high",
            "okf_concepts_used": ["runbook-stalled"],
        }

    def post_events(self, events: list[dict[str, Any]], timeout_s: float = 5.0) -> dict[str, Any]:
        return {"accepted": len(events), "deduplicated": 0}


def test_watchdog_dry_run_quarantine_spot_preemption_oom_and_resolve(tmp_path: Path) -> None:
    cfg = _load_example_config(tmp_path)
    run_preflight(cfg, check_remote=False)

    backend = FakeBackend(backend_name="gcp_batch", default_gpu="l4")
    steer = FakeSteerClient()
    state_dir = Path(cfg["state_dir"])
    store = StateStore(state_dir / "state.json", state_dir / "audit.jsonl")
    emitter = TelemetryEmitter(
        state_dir / "outbox.jsonl", client=steer, default_backend="gcp_batch"
    )
    driver = QueueDriver(
        backend=backend,
        state_store=store,
        steer_client=steer,
        telemetry=emitter,
        config=cfg,
    )

    # 1. Invalid folder name in pending/ -> quarantined cleanly without crashing
    bad_dir = Path(cfg["queue_dir"]) / "pending" / "bad name with spaces!"
    bad_dir.mkdir(parents=True)
    # 2. Valid GX run in pending/
    gx_dir = Path(cfg["queue_dir"]) / "pending" / "gx_mrx_001"
    gx_dir.mkdir(parents=True)
    (gx_dir / "gx.in").write_text(
        "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n[Time]\ndt = 0.02\ncfl = 0.8\n[Dissipation]\nhyperviscosity = 0.1\n",
        encoding="utf-8",
    )
    (gx_dir / "run_meta.json").write_text(
        json.dumps({"equilibrium_id": "mrx_eq_0001", "campaign_id": "pilot_50"}),
        encoding="utf-8",
    )

    res = driver.tick()
    assert res["submitted"] == 1
    assert any((Path(cfg["queue_dir"]) / "needs_human").glob("quarantined_*"))

    rec = store.get_run("gx_mrx_001")
    assert rec is not None
    first_job = rec.slurm_job_id
    assert first_job == "job-1"

    # 3. GCP Cloud Batch Spot PREEMPTED -> 0 LLM calls, restarts from checkpoint!
    backend.states["job-1"] = JobStatus(job_id="job-1", state="PREEMPTED", gpu_tier="l4")
    driver.tick()
    assert steer.calls == 0
    assert rec.walltime_resubmits == 1
    assert rec.slurm_job_id == "job-2"

    # 4. GCP Cloud Batch Spot L4 OUT_OF_MEMORY -> 0 LLM calls, upgrades gpu_tier l4 -> a100!
    backend.states["job-2"] = JobStatus(job_id="job-2", state="OUT_OF_MEMORY", gpu_tier="l4")
    driver.tick()
    assert steer.calls == 0
    assert rec.gpu_tier == "a100"
    assert rec.slurm_job_id == "job-3"

    # 5. Dry-run makes ZERO /triage calls even when failure metrics are present
    running_gx = Path(rec.run_dir)
    (running_gx / "diagnostics_metrics.json").write_text(
        json.dumps({"failure_type": "TIMESTEP_COLLAPSE", "sim_time_reached": 12.0}),
        encoding="utf-8",
    )
    dry_driver = QueueDriver(
        backend=backend,
        state_store=StateStore(state_dir / "state.json", state_dir / "audit.jsonl", dry_run=True),
        steer_client=steer,
        telemetry=emitter,
        config=cfg,
        dry_run=True,
    )
    dry_driver.tick()
    assert steer.calls == 0

    # 6. Live failure + submit failure after patch -> escalates to needs_human cleanly (`NEW-TAC-2`)
    backend.fail_on_resubmit = True
    driver.tick()
    assert steer.calls == 1
    assert rec.status == "NEEDS_HUMAN"
    assert (Path(cfg["queue_dir"]) / "needs_human" / "gx_mrx_001" / "ISSUE.md").is_file()

    # 7. Human resolves and re-queues -> moves back to pending/
    requeued = resolve_escalated_run(config=cfg, run_id="gx_mrx_001", minutes_spent=10.0)
    assert requeued.is_dir()
    assert (requeued / "ISSUE.resolved.md").is_file()


def test_diagnostics_nan_word_boundary_oom_detection_and_log_rotation(tmp_path: Path) -> None:
    run_dir = tmp_path / "gx_diag_test"
    run_dir.mkdir()
    (run_dir / "gx.in").write_text(
        "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\nnhermite = 32\nnlaguerre = 16\nnspecies = 3\n",
        encoding="utf-8",
    )

    # 1. Words containing "nan" substring ("resonance", "dominant", "determinant", "maintenance")
    #    must NOT trigger false-positive DIVERGENCE
    (run_dir / "gx.log").write_text(
        "Step 10: resonance heating dominant mode determinant ok during maintenance\n",
        encoding="utf-8",
    )
    diag_ok = inspect_run_directory(run_dir)
    assert diag_ok.failure_type is None
    assert diag_ok.current_parameters["nhermite"] == 32
    assert diag_ok.current_parameters["nlaguerre"] == 16
    assert diag_ok.current_parameters["nspecies"] == 3

    # 2. Actual NaN token triggers DIVERGENCE
    (run_dir / "gx.log").write_text("Step 20: heat_flux = -nan\n", encoding="utf-8")
    diag_nan = inspect_run_directory(run_dir)
    assert diag_nan.failure_type == "DIVERGENCE"

    # 3. CUDA / GPU OOM log strings trigger OUT_OF_MEMORY
    (run_dir / "gx.log").write_text(
        "CUDA error: cudaErrorMemoryAllocation: out of memory\n", encoding="utf-8"
    )
    diag_oom = inspect_run_directory(run_dir)
    assert diag_oom.failure_type == "OUT_OF_MEMORY"

    # 4. consume_stale_diagnostics_metrics archives (never overwrites) metrics and solver logs
    (run_dir / "diagnostics_metrics.json").write_text('{"failure_type": "STALL"}', encoding="utf-8")
    (run_dir / "gene.log").write_text("old gene log\n", encoding="utf-8")
    (run_dir / "slurm.out").write_text("old slurm out\n", encoding="utf-8")
    consume_stale_diagnostics_metrics(run_dir)
    assert not (run_dir / "diagnostics_metrics.json").exists()
    assert (run_dir / "diagnostics_metrics.1.json").is_file()
    assert not (run_dir / "gx.log").exists()
    assert (run_dir / "gx.log.1").is_file()
    assert (run_dir / "gene.log.1").is_file()
    assert (run_dir / "slurm.out.1").is_file()


def test_watchdog_cancel_error_resilience_and_unconverged_completion_gate(tmp_path: Path) -> None:
    cfg = _load_example_config(tmp_path)
    cfg["cancel_wait_seconds"] = 0
    backend = FakeBackend(backend_name="tacc_slurm", default_gpu="gh200")
    steer = FakeSteerClient()
    state_dir = Path(cfg["state_dir"])
    store = StateStore(state_dir / "state.json", state_dir / "audit.jsonl")
    emitter = TelemetryEmitter(state_dir / "outbox.jsonl", client=steer)
    wd = SimulationWatchdog(
        backend=backend,
        state_store=store,
        steer_client=steer,
        telemetry=emitter,
        config=cfg,
    )

    run_dir = Path(cfg["queue_dir"]) / "running" / "gx_unconv_01"
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text(
        "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n[Time]\ndt = 0.02\ncfl = 0.8\n",
        encoding="utf-8",
    )
    jid = backend.submit(run_dir, solver="GX")
    from hpc_core import RunRecord

    rec = RunRecord(
        run_id="gx_unconv_01",
        solver="GX",
        run_dir=str(run_dir),
        status="RUNNING",
        slurm_job_id=jid,
    )
    store.upsert_run(rec)

    # 1. If backend.cancel raises SlurmCancelError on mid-run failure, watchdog does not crash,
    #    and it never patches/resubmits while the job is still RUNNING (two jobs, one run dir).
    from hpc_core.slurm import SlurmCancelError

    def raising_cancel(_job_id: str) -> None:
        raise SlurmCancelError("scancel error: Invalid job id specified")

    backend.cancel = raising_cancel  # type: ignore[method-assign]
    (run_dir / "diagnostics_metrics.json").write_text(
        json.dumps({"failure_type": "TIMESTEP_COLLAPSE", "sim_time_reached": 10.0}),
        encoding="utf-8",
    )
    wd.poll_run(rec)
    assert rec.retry_count == 0
    assert steer.calls == 0
    assert rec.cancel_requested is not None

    # Once the job has actually terminated, the stored failure is triaged and patched.
    backend.states[jid] = JobStatus(job_id=jid, state="FAILED")
    wd.poll_run(rec)
    assert rec.retry_count == 1
    assert steer.calls == 1
    assert rec.cancel_requested is None

    # 2. If job.state == COMPLETED but diagnostics_metrics.json shows converged=False / short sim_time,
    #    watchdog routes to _handle_physics_or_solver_failure(failure_type="STALL") instead of harvesting
    assert rec.slurm_job_id is not None
    backend.states[rec.slurm_job_id] = JobStatus(job_id=rec.slurm_job_id, state="COMPLETED")
    (run_dir / "diagnostics_metrics.json").write_text(
        json.dumps({"converged": False, "sim_time_reached": 25.0, "sim_time_target": 200.0}),
        encoding="utf-8",
    )
    wd.poll_run(rec)
    assert rec.status != "COMPLETED"
    assert rec.retry_count == 2
    assert steer.calls == 2


def test_diagnostics_adapter_parses_real_gx_stdout_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "gx_real_log_test"
    run_dir.mkdir()
    (run_dir / "gx.in").write_text(
        "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n",
        encoding="utf-8",
    )
    (run_dir / "gx.log").write_text(
        "run_name: Step     10: Time =    0.10000  dt = 1.000e-03   Phi**2 = 1.234e-05   Q_i = 1.234e-03   Q_e = 1.234e-03   Gam_i = 0.000e+00\n"
        "run_name: Step     20: Time =    0.20000  dt = 1.000e-03   Phi**2 = 1.500e-05   Q_i = 2.000e-03   Q_e = 1.500e-03   Gam_i = 5.000e-04\n",
        encoding="utf-8",
    )
    report = inspect_run_directory(run_dir)
    assert report.sim_time_reached == 0.20000
    assert report.scalar_metrics["heat_flux_es"] == 0.003500
    assert report.scalar_metrics["particle_flux"] == 0.000500
    assert report.scalar_metrics["phi2_total"] == 0.000015
    assert report.scalar_metrics["dt_collapse_ratio"] == 1.0
