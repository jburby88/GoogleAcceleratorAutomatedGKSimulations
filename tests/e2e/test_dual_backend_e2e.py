"""Cross-workstream E2E test verifying TACC Slurm + GCP Cloud Batch babysitting -> Zenodo -> KR1 -> KR3."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import requests
import uvicorn
from app.events import EventStore
from app.main import create_app
from app.okf_loader import LoadedSkill
from app.runner import SteeringEngine
from app.schemas import ActionDirective, DiagnosticSummary
from gcp_exec.batch_backend import CloudBatchBackend
from gcp_exec.submission import SchedulerApiError
from hpc_core import (
    AuroraSteerClient,
    BackendCapabilities,
    JobStatus,
    StateStore,
    TelemetryEmitter,
)
from kr1_surrogate import (
    AugmentedLagrangianState,
    load_campaign_samples,
    split_by_equilibrium,
)
from kr3_design import enqueue_validation_run
from tacc_exec.driver import QueueDriver

REPO_ROOT = Path(__file__).resolve().parents[2]


class _E2ESteeringEngine(SteeringEngine):
    def diagnose(  # type: ignore[override]
        self,
        diagnostic: DiagnosticSummary,
        skill: LoadedSkill,
        *,
        allowed_actions: frozenset[str],
    ) -> ActionDirective:
        _, concept_ids = skill.format_prompt_context(diagnostic.failure_type)
        ft = diagnostic.failure_type
        params = diagnostic.current_parameters
        rules = skill.parameter_rules

        if ft in ("STALL", "TIMESTEP_COLLAPSE"):
            return ActionDirective(
                action="warm_restart",
                reason="Reduced cfl_fraction and dt_max for checkpoint continuation.",
                parameter_modifications={
                    "cfl_fraction": round(
                        max(
                            float(rules["cfl_fraction"].min_value),
                            float(params.get("cfl_fraction", 0.8)) * 0.75,
                        ),
                        4,
                    ),
                    "dt_max": round(
                        max(
                            float(rules["dt_max"].min_value),
                            float(params.get("dt_max", 0.02)) * 0.5,
                        ),
                        6,
                    ),
                },
                restart_from_checkpoint=True,
                confidence="high",
                okf_concepts_used=list(
                    dict.fromkeys([*concept_ids, "runbook-stalled", "allowed-parameters"])
                ),
            )
        if ft == "SPECTRAL_PILEUP":
            return ActionDirective(
                action="cold_restart",
                reason="Refined grid resolution and hyperdiffusion_coeff.",
                parameter_modifications={"nx": 96, "ny": 48, "hyperdiffusion_coeff": 0.25},
                restart_from_checkpoint=False,
                confidence="high",
                okf_concepts_used=list(
                    dict.fromkeys([*concept_ids, "runbook-spectra", "allowed-parameters"])
                ),
            )
        return ActionDirective(
            action="escalate_to_human",
            reason="Escalating to physicist review.",
            parameter_modifications={},
            restart_from_checkpoint=False,
            confidence="high",
            okf_concepts_used=list(
                dict.fromkeys([*concept_ids, "runbook-divergence", "escalation-policy"])
            ),
        )


class _StubSlurmBackend:
    def __init__(self) -> None:
        self.capabilities = BackendCapabilities(
            backend_name="tacc_slurm",
            supports_gene=True,
            supports_gx=True,
            default_gpu_tier="tacc_gh200",
        )
        self._seq = 0
        self.jobs: dict[str, str] = {}

    def submit(
        self,
        run_dir: Path,
        *,
        solver: str = "GENE",
        parameters: dict[str, Any] | None = None,
        retry_count: int = 0,
        force_high_memory: bool = False,
    ) -> str:
        self._seq += 1
        jid = f"slurm-{1000 + self._seq}"
        self.jobs[jid] = "RUNNING"
        return jid

    def cancel(self, job_id: str) -> None:
        self.jobs[job_id] = "CANCELLED"

    def query_job(self, job_id: str) -> JobStatus:
        return JobStatus(
            job_id=job_id, state=self.jobs.get(job_id, "UNKNOWN"), elapsed_seconds=420.0
        )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_dual_backend_e2e_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = "e2e-test-bearer-token"
    # monkeypatch (not os.environ) so the token does not leak into other test modules.
    monkeypatch.setenv("AURORA_STEER_BEARER_TOKEN", token)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    event_store = EventStore(duckdb_path=str(tmp_path / "aurora_events.duckdb"))
    fastapi_app = create_app(steering_engine=_E2ESteeringEngine(), event_store=event_store)
    server = uvicorn.Server(
        uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="error")
    )
    srv_thread = threading.Thread(target=server.run, daemon=True)
    srv_thread.start()

    try:
        sess = requests.Session()
        sess.trust_env = False
        for _ in range(50):
            try:
                if sess.get(f"{base_url}/healthz", timeout=0.5).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.05)

        dataset_dir = tmp_path / "gcs_dataset_bucket"
        base_cfg = json.loads(
            (REPO_ROOT / "simulations" / "hpc" / "config.example.json").read_text(encoding="utf-8")
        )

        # 1. TACC Slurm GENE 3-strike circuit breaker
        tacc_cfg = {
            **base_cfg,
            "backend": "tacc_slurm",
            "campaign_id": "e2e_campaign",
            "aurora_steer_url": base_url,
            "queue_dir": str(tmp_path / "tacc_queue"),
            "state_dir": str(tmp_path / "tacc_state"),
            "dataset_output_dir": str(dataset_dir),
        }
        tacc_slurm = _StubSlurmBackend()
        tacc_store = StateStore(
            Path(tacc_cfg["state_dir"]) / "state.json",
            Path(tacc_cfg["state_dir"]) / "audit.jsonl",
        )
        steer_client = AuroraSteerClient(base_url=base_url, api_token=token)
        tacc_telemetry = TelemetryEmitter(
            Path(tacc_cfg["state_dir"]) / "outbox.jsonl",
            client=steer_client,
            default_campaign_id="e2e_campaign",
            default_backend="tacc_slurm",
        )
        tacc_driver = QueueDriver(
            backend=tacc_slurm,
            state_store=tacc_store,
            steer_client=steer_client,
            telemetry=tacc_telemetry,
            config=tacc_cfg,
        )

        gene_dir = Path(tacc_cfg["queue_dir"]) / "pending" / "gene_tacc_001"
        gene_dir.mkdir(parents=True)
        (gene_dir / "parameters").write_text(
            "&box\n  nx0 = 64\n  nky0 = 32\n  nz0 = 48\n/\n"
            "&general\n  dt_max = 0.02\n  courant = 0.8\n  hyp_z = 0.2\n  beta = 0.035\n/\n"
            "&in_out\n  read_checkpoint = .false.\n/\n",
            encoding="utf-8",
        )
        (gene_dir / "run_meta.json").write_text(
            json.dumps(
                {"campaign_id": "e2e_campaign", "equilibrium_id": "mrx_eq_0007", "fidelity": "high"}
            ),
            encoding="utf-8",
        )

        tacc_driver.tick()
        gene_rec = tacc_store.get_run("gene_tacc_001")
        assert gene_rec is not None

        for strike_idx in range(3):
            (Path(gene_rec.run_dir) / "diagnostics_metrics.json").write_text(
                json.dumps(
                    {
                        "failure_type": "TIMESTEP_COLLAPSE",
                        "sim_time_reached": 15.0 + strike_idx,
                        "scalar_metrics": {"dt_collapse_ratio": 0.002},
                    }
                ),
                encoding="utf-8",
            )
            tacc_driver.tick()

        llm_calls_after_tacc = sess.get(f"{base_url}/healthz").json()["llm_calls_total"]
        assert gene_rec.status == "NEEDS_HUMAN"
        assert (Path(gene_rec.run_dir) / "ISSUE.md").is_file()
        assert llm_calls_after_tacc == 2

        # 2. GCP Cloud Batch GX babysitting
        batch_jobs: dict[str, dict[str, Any]] = {}
        batch_seq = {"n": 0}

        def stub_batch_transport(
            method: str, url: str, body: dict[str, Any] | None
        ) -> dict[str, Any]:
            if method == "POST" and ":cancel" not in url:
                batch_seq["n"] += 1
                name = f"projects/ut-e2e/locations/us-central1/jobs/gx-spot-{batch_seq['n']}"
                tier = (body or {}).get("labels", {}).get("gpu_tier", "l4")
                batch_jobs[name] = {
                    "name": name,
                    "labels": {"gpu_tier": tier},
                    "status": {"state": "RUNNING", "runDuration": "600s", "statusEvents": []},
                }
                return {"name": name}
            if method == "POST" and ":cancel" in url:
                jname = url.split("/v1/")[1].split(":cancel")[0]
                if jname in batch_jobs:
                    batch_jobs[jname]["status"]["state"] = "CANCELLED"
                return {}
            jname = url.split("/v1/")[1]
            if jname not in batch_jobs:
                # Same contract as the real REST transport: unknown job -> HTTP 404.
                raise SchedulerApiError(f"job {jname} not found", status_code=404)
            return batch_jobs[jname]

        gcp_backend = CloudBatchBackend(
            project_id="ut-e2e",
            region="us-central1",
            gcs_bucket="ut-e2e-gk-dataset",
            gx_container_image="us-central1-docker.pkg.dev/ut-e2e/aurora/gx:latest",
            worker_service_account_email="gx-batch-worker-sa@ut-e2e.iam.gserviceaccount.com",
            transport=stub_batch_transport,
        )
        gcp_cfg = {
            **base_cfg,
            "backend": "gcp_batch",
            "campaign_id": "e2e_campaign",
            "aurora_steer_url": base_url,
            "queue_dir": str(tmp_path / "gcp_queue"),
            "state_dir": str(tmp_path / "gcp_state"),
            "dataset_output_dir": str(dataset_dir),
            "dataset_gcs_uri_prefix": "gs://ut-e2e-gk-dataset",
        }
        gcp_store = StateStore(
            Path(gcp_cfg["state_dir"]) / "state.json",
            Path(gcp_cfg["state_dir"]) / "audit.jsonl",
        )
        gcp_telemetry = TelemetryEmitter(
            Path(gcp_cfg["state_dir"]) / "outbox.jsonl",
            client=steer_client,
            default_campaign_id="e2e_campaign",
            default_backend="gcp_batch",
        )
        gcp_driver = QueueDriver(
            backend=gcp_backend,
            state_store=gcp_store,
            steer_client=steer_client,
            telemetry=gcp_telemetry,
            config=gcp_cfg,
        )

        gx_pending = Path(gcp_cfg["queue_dir"]) / "pending" / "gx_batch_mrx_042"
        gx_pending.mkdir(parents=True)
        (gx_pending / "gx.in").write_text(
            "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n\n"
            "[Time]\ndt = 0.02\ncfl = 0.8\n\n"
            "[Dissipation]\nhyper = true\nD_hyper = 0.1\n\n"
            "[Physics]\nbeta = 0.038\n",
            encoding="utf-8",
        )
        (gx_pending / "equilibrium_mrx_0042.eik.nc").write_text("eik-geometry", encoding="utf-8")
        (gx_pending / "run_meta.json").write_text(
            json.dumps(
                {"campaign_id": "e2e_campaign", "equilibrium_id": "mrx_eq_0042", "fidelity": "low"}
            ),
            encoding="utf-8",
        )

        gcp_driver.tick()
        gx_rec = gcp_store.get_run("gx_batch_mrx_042")
        assert gx_rec is not None and gx_rec.slurm_job_id is not None

        batch_jobs[gx_rec.slurm_job_id]["status"] = {
            "state": "FAILED",
            "runDuration": "810s",
            "statusEvents": [
                {"description": "SPOT VM preempted", "taskExecution": {"exitCode": 50001}}
            ],
        }
        gcp_driver.tick()
        assert sess.get(f"{base_url}/healthz").json()["llm_calls_total"] == 2
        assert gx_rec.walltime_resubmits == 1

        (Path(gx_rec.run_dir) / "diagnostics_metrics.json").write_text(
            json.dumps(
                {
                    "failure_type": "SPECTRAL_PILEUP",
                    "sim_time_reached": 64.0,
                    "scalar_metrics": {"high_k_tail_ratio": 0.41},
                }
            ),
            encoding="utf-8",
        )
        gcp_driver.tick()

        assert gx_rec.slurm_job_id is not None
        batch_jobs[gx_rec.slurm_job_id]["status"] = {
            "state": "SUCCEEDED",
            "runDuration": "1450s",
            "statusEvents": [],
        }
        (Path(gx_rec.run_dir) / "diagnostics_metrics.json").write_text(
            json.dumps(
                {
                    "converged": True,
                    "sim_time_reached": 200.0,
                    # provenance_ok=1.0: the GCP runner verified the run's provenance
                    # (required for dataset eligibility of gcp_batch runs, manifest v2).
                    "scalar_metrics": {
                        "heat_flux_es": 3.72,
                        "growth_rate_gamma": 0.24,
                        "provenance_ok": 1.0,
                    },
                    "fluxes": {"heat_flux_es": 3.72, "heat_flux_em": 0.41, "particle_flux": 0.18},
                }
            ),
            encoding="utf-8",
        )
        gcp_driver.tick()
        assert gx_rec.status == "COMPLETED"
        assert sess.get(f"{base_url}/healthz").json()["llm_calls_total"] == 3
        assert gx_rec.dataset_manifest_path and Path(gx_rec.dataset_manifest_path).is_file()

        # 3. KR1 loader & KR3 active learning queue
        samples = load_campaign_samples(dataset_dir, "e2e_campaign")
        splits = split_by_equilibrium(samples, train_frac=0.7, val_frac=0.15)
        assert len(samples) == 1
        assert sum(len(v) for v in splits.values()) == 1

        al = AugmentedLagrangianState(["mrx_force_balance", "bootstrap_current", "gk_heat_flux"])
        al.update_outer_loop(
            {"mrx_force_balance": 0.01, "bootstrap_current": -0.02, "gk_heat_flux": 0.08}
        )
        kr3_queued_dir = enqueue_validation_run(
            kr2_pending_dir=Path(gcp_cfg["queue_dir"]) / "pending",
            run_id="gx_kr3_validation_099",
            equilibrium_id="mrx_eq_0099",
            surrogate_epistemic_std=0.21,
            beta_target=0.041,
        )
        assert kr3_queued_dir.is_dir()

        summary_prod = sess.get(
            f"{base_url}/v1/events/summary?environment=prod&workstream=kr2-gk-database",
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        assert summary_prod["total_events"] >= 10
    finally:
        server.should_exit = True
        srv_thread.join(timeout=3.0)
