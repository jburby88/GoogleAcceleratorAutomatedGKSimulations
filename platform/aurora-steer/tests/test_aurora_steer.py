"""Unit tests for OKF loader, guardrails, schemas, `/triage`, and `/v1/events`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.events import EventStore, translate_bigquery_views_to_duckdb
from app.guardrails import validate_directive
from app.main import create_app
from app.okf_loader import OKFLoadError, load_skill
from app.schemas import ActionDirective, DiagnosticSummary, LifecycleEvent

ALL_ACTIONS = frozenset({"warm_restart", "cold_restart", "resubmit_walltime", "escalate_to_human"})


def test_okf_loader_discovers_sim_babysitter_across_workstreams() -> None:
    skill = load_skill("sim-babysitter")
    assert skill.skill_name == "sim-babysitter"
    assert set(skill.parameter_rules.keys()) == {
        "nx",
        "ny",
        "nz",
        "dt_max",
        "cfl_fraction",
        "hyperdiffusion_coeff",
    }
    assert skill.parameter_rules["dt_max"].min_value == pytest.approx(1e-6)
    assert skill.parameter_rules["dt_max"].max_value == pytest.approx(0.1)
    assert skill.parameter_rules["nz"].gx_path == "Dimensions.ntheta"
    # PHYS-04: real GX key; PHYS-03: only timestep controls are warm-restart safe.
    assert skill.parameter_rules["hyperdiffusion_coeff"].gx_path == "Dissipation.D_hyper"
    assert {n for n, r in skill.parameter_rules.items() if r.warm_restart_allowed} == {
        "dt_max",
        "cfl_fraction",
    }

    _, concept_ids = skill.format_prompt_context("SPECTRAL_PILEUP")
    assert "allowed-parameters" in concept_ids
    assert "escalation-policy" in concept_ids
    assert "runbook-spectra" in concept_ids


def test_okf_action_policy_matches_runbooks() -> None:
    skill = load_skill("sim-babysitter")
    assert skill.allowed_actions_for("STALL") == {"warm_restart", "escalate_to_human"}
    assert skill.allowed_actions_for("TIMESTEP_COLLAPSE") == {"warm_restart", "escalate_to_human"}
    assert skill.allowed_actions_for("SPECTRAL_PILEUP") == {"cold_restart", "escalate_to_human"}
    for ft in ("DIVERGENCE", "CRASH", "UNKNOWN", "NOT_A_FAILURE_TYPE"):
        assert skill.allowed_actions_for(ft) == {"escalate_to_human"}


def test_okf_loader_blocks_unsafe_skill_name() -> None:
    with pytest.raises(OKFLoadError):
        load_skill("../etc/passwd")


def test_okf_loader_blocks_manifest_path_traversal(tmp_path: Path) -> None:
    okf_dir = tmp_path / "bad-skill" / "okf"
    okf_dir.mkdir(parents=True)
    outside = tmp_path / "secret.md"
    outside.write_text("# secret", encoding="utf-8")
    (okf_dir / "manifest.yaml").write_text(
        "version: '1.0'\nconcepts:\n  - id: leak\n    path: ../../secret.md\n",
        encoding="utf-8",
    )
    with pytest.raises(OKFLoadError, match="traversal"):
        load_skill("bad-skill", skills_root=tmp_path)


def test_guardrails_reject_only_never_clamps() -> None:
    skill = load_skill("sim-babysitter")

    def check(directive: ActionDirective) -> object:
        return validate_directive(directive, skill.parameter_rules, allowed_actions=ALL_ACTIONS)

    # 1. Physics parameter forbidden -> reject to escalate_to_human
    res1 = check(
        ActionDirective(
            action="warm_restart",
            reason="Attempt to alter equilibrium beta.",
            parameter_modifications={"beta": 0.04},
            restart_from_checkpoint=True,
            confidence="high",
            okf_concepts_used=["allowed-parameters"],
        )
    )
    assert not res1.accepted
    assert res1.forced_escalation
    assert res1.directive.action == "escalate_to_human"
    assert res1.directive.parameter_modifications == {}

    # 2. Out-of-bounds numerical parameter -> reject (never clamp to 512!)
    res2 = check(
        ActionDirective(
            action="cold_restart",
            reason="Increase nx beyond ceiling.",
            parameter_modifications={"nx": 2048},
            restart_from_checkpoint=False,
            confidence="high",
            okf_concepts_used=["runbook-spectra"],
        )
    )
    assert not res2.accepted
    assert res2.directive.action == "escalate_to_human"
    assert res2.directive.parameter_modifications == {}

    # 3. warm_restart with restart_from_checkpoint=False -> reject
    res3 = check(
        ActionDirective(
            action="warm_restart",
            reason="Warm restart without checkpoint flag.",
            parameter_modifications={"dt_max": 0.01},
            restart_from_checkpoint=False,
            confidence="medium",
            okf_concepts_used=["runbook-stalled"],
        )
    )
    assert not res3.accepted
    assert res3.directive.action == "escalate_to_human"

    # 4. warm_restart (or restart_from_checkpoint=True) with nx/ny/nz -> reject
    for grid_key in ("nx", "ny", "nz"):
        res_grid = check(
            ActionDirective(
                action="warm_restart",
                reason=f"Attempt {grid_key} change on warm restart.",
                parameter_modifications={grid_key: 64},
                restart_from_checkpoint=True,
                confidence="high",
                okf_concepts_used=["runbook-spectra"],
            )
        )
        assert not res_grid.accepted
        assert res_grid.forced_escalation
        assert res_grid.directive.action == "escalate_to_human"
        assert "incompatible with binary checkpoints" in (res_grid.rejection_reason or "")

    # 5. warm_restart or cold_restart with empty parameter_modifications -> reject
    res_empty = check(
        ActionDirective(
            action="warm_restart",
            reason="Warm restart with empty modifications.",
            parameter_modifications={},
            restart_from_checkpoint=True,
            confidence="medium",
            okf_concepts_used=["runbook-stalled"],
        )
    )
    assert not res_empty.accepted
    assert res_empty.directive.action == "escalate_to_human"

    # 6. resubmit_walltime with restart_from_checkpoint=False -> reject
    res_wt = check(
        ActionDirective(
            action="resubmit_walltime",
            reason="Walltime resubmit without checkpoint.",
            parameter_modifications={},
            restart_from_checkpoint=False,
            confidence="high",
            okf_concepts_used=["runbook-stalled"],
        )
    )
    assert not res_wt.accepted
    assert res_wt.directive.action == "escalate_to_human"


def test_schemas_forbid_nan_and_oversized_events() -> None:
    with pytest.raises(ValueError):
        DiagnosticSummary(
            run_id="r1",
            solver="GENE",
            failure_type="DIVERGENCE",
            retry_count=0,
            sim_time_reached=float("nan"),
            sim_time_target=100.0,
        )

    with pytest.raises(ValueError, match="1024"):
        LifecycleEvent(
            event_id="evt-12345678",
            run_id="r1",
            solver="GX",
            event_type="failed",
            ts_utc=datetime.now(UTC),
            payload={"huge": "x" * 1200},
        )


def test_triage_api_auth_size_idempotency_and_3strike(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    # Unauthenticated -> 401
    r_unauth = client.post("/triage", json={})
    assert r_unauth.status_code == 401

    # >= 2048 bytes -> 413
    oversized = {"padding": "a" * 2100}
    r_big = client.post("/triage", content=json.dumps(oversized), headers=auth_headers)
    assert r_big.status_code == 413

    # Valid GX Cloud Batch SPECTRAL_PILEUP -> cold_restart + idempotency hit = 1 LLM call
    payload = {
        "workstream": "kr2-gk-database",
        "backend": "gcp_batch",
        "run_id": "gx_w7x_001",
        "campaign_id": "pilot_50",
        "solver": "GX",
        "fidelity": "low",
        "environment": "prod",
        "failure_type": "SPECTRAL_PILEUP",
        "retry_count": 0,
        "sim_time_reached": 42.5,
        "sim_time_target": 200.0,
        "current_parameters": {"nx": 64, "ny": 32, "nz": 48, "hyperdiffusion_coeff": 0.1},
        "scalar_metrics": {"high_k_tail_ratio": 0.44},
        "log_tail": ["gx: step 1200 high-k tail ratio exceeded threshold"],
    }
    headers_idem = {**auth_headers, "Idempotency-Key": "gx_w7x_001:evt-1"}
    r1 = client.post("/triage", json=payload, headers=headers_idem)
    assert r1.status_code == 200
    assert r1.headers["X-Aurora-Decision"] == "llm"
    d1 = r1.json()
    assert d1["action"] == "cold_restart"
    assert d1["restart_from_checkpoint"] is False
    assert d1["parameter_modifications"]["nx"] == 96

    calls_after_first = client.get("/healthz").json()["llm_calls_total"]
    assert calls_after_first == 1

    # Duplicate delivery with same Idempotency-Key -> cached, 0 extra LLM calls
    r2 = client.post("/triage", json=payload, headers=headers_idem)
    assert r2.status_code == 200
    assert r2.headers["X-Aurora-Decision"] == "cached"
    assert r2.json() == d1
    assert client.get("/healthz").json()["llm_calls_total"] == 1

    # 3-strike circuit breaker (retry_count >= 2) -> 0 extra LLM calls
    payload_strike3 = {**payload, "retry_count": 2}
    r3 = client.post(
        "/triage",
        json=payload_strike3,
        headers={**auth_headers, "Idempotency-Key": "gx_w7x_001:evt-3"},
    )
    assert r3.status_code == 200
    assert r3.json()["action"] == "escalate_to_human"
    assert r3.headers["X-Aurora-Decision"] == "circuit_breaker"
    assert client.get("/healthz").json()["llm_calls_total"] == 1


def test_events_ingestion_dedup_and_summary(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    now = datetime.now(UTC).isoformat()
    batch = {
        "events": [
            {
                "event_id": "evt-00000001",
                "workstream": "kr2-gk-database",
                "backend": "gcp_batch",
                "run_id": "gx_batch_01",
                "campaign_id": "pilot_50",
                "solver": "GX",
                "fidelity": "low",
                "event_type": "submitted",
                "ts_utc": now,
                "environment": "prod",
                "payload": {"geometry_id": "mrx_eq_0042"},
            },
            {
                "event_id": "evt-00000002",
                "workstream": "kr2-gk-database",
                "backend": "gcp_batch",
                "run_id": "gx_batch_01",
                "campaign_id": "pilot_50",
                "solver": "GX",
                "fidelity": "low",
                "event_type": "completed",
                "ts_utc": now,
                "environment": "prod",
                "payload": {"metrics": {"heat_flux_es": 3.14}},
            },
        ]
    }
    resp = client.post("/v1/events", json=batch, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {"accepted": 2, "deduplicated": 0}

    # Re-sending identical event_ids deduplicates cleanly
    resp_dup = client.post("/v1/events", json=batch, headers=auth_headers)
    assert resp_dup.json() == {"accepted": 0, "deduplicated": 2}

    # Verify prod events are ingested and summarized directly
    prod_summary = client.get(
        "/v1/events/summary?environment=prod&workstream=kr2-gk-database",
        headers=auth_headers,
    ).json()
    assert prod_summary["total_events"] == 2
    assert prod_summary["storage_backend"] == "duckdb_local_instance_memory"
    assert len(prod_summary["campaigns"]) == 1
    assert prod_summary["campaigns"][0]["completed_runs"] == 1


def test_k_service_blocks_allow_unauth_local_and_aurora_api_token_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AURORA_STEER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("AURORA_API_TOKEN", raising=False)
    monkeypatch.setenv("ALLOW_UNAUTH_LOCAL", "1")
    monkeypatch.setenv("ENVIRONMENT", "dev")
    monkeypatch.setenv("K_SERVICE", "aurora-steer")

    store = EventStore(duckdb_path=":memory:")
    local_client = TestClient(create_app(event_store=store))

    # K_SERVICE must block ALLOW_UNAUTH_LOCAL=1 (fail closed with 503)
    resp_cloud_run = local_client.get("/v1/events/summary")
    assert resp_cloud_run.status_code == 503

    # AURORA_API_TOKEN fallback works when AURORA_STEER_BEARER_TOKEN is unset
    monkeypatch.setenv("AURORA_API_TOKEN", "fallback-token-abc")
    resp_ok = local_client.get(
        "/v1/events/summary",
        headers={"Authorization": "Bearer fallback-token-abc"},
    )
    assert resp_ok.status_code == 200


def test_translate_bigquery_views_to_duckdb_parameterized_project_id() -> None:
    raw_sql = (
        "CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_run_status` AS "
        "SELECT SAFE_DIVIDE(1, 2) AS r FROM `custom-proj-123.custom_ds.events`;"
    )
    translated = translate_bigquery_views_to_duckdb(raw_sql)
    assert "`" not in translated
    assert "CREATE OR REPLACE VIEW v_run_status AS" in translated
    assert "FROM events;" in translated


def test_mrx_sfincs_solvers_equilibrium_id_and_divergence_after_patch() -> None:
    diag = DiagnosticSummary(
        run_id="mrx_01",
        solver="MRX",
        equilibrium_id="mrx_eq_0099",
        failure_type="STALL",
        retry_count=0,
        sim_time_reached=10.0,
        sim_time_target=100.0,
    )
    assert diag.solver == "MRX"
    assert diag.equilibrium_id == "mrx_eq_0099"

    now = datetime.now(UTC)
    store = EventStore(duckdb_path=":memory:")
    events = [
        LifecycleEvent(
            event_id="evt-div-0001",
            run_id="sfincs_run_1",
            campaign_id="camp_1",
            solver="SFINCS",
            equilibrium_id="mrx_eq_0099",
            event_type="failed",
            ts_utc=now,
            retry_count=0,
            environment="sim",
            payload={"failure_type": "TIMESTEP_COLLAPSE"},
        ),
        LifecycleEvent(
            event_id="evt-div-0002",
            run_id="sfincs_run_1",
            campaign_id="camp_1",
            solver="SFINCS",
            equilibrium_id="mrx_eq_0099",
            event_type="patched",
            ts_utc=now,
            retry_count=1,
            environment="sim",
            payload={"action": "warm_restart"},
        ),
        LifecycleEvent(
            event_id="evt-div-0003",
            run_id="sfincs_run_1",
            campaign_id="camp_1",
            solver="SFINCS",
            equilibrium_id="mrx_eq_0099",
            event_type="failed",
            ts_utc=now,
            retry_count=1,
            environment="sim",
            payload={"failure_type": "DIVERGENCE"},
        ),
    ]
    store.ingest_events(events)
    row = store._conn.execute(
        "SELECT geometry_id, is_divergence_escalated FROM v_run_status WHERE run_id = 'sfincs_run_1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "mrx_eq_0099"
    assert row[1] is True
    # STEER-08: non-prod telemetry is always flagged synthetic.
    flags = store._conn.execute(
        "SELECT DISTINCT is_synthetic FROM events WHERE run_id = 'sfincs_run_1'"
    ).fetchall()
    assert flags == [(True,)]


def test_model_id_default_validation_and_bearer_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import config

    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "ignored-legacy-alias")
    assert config.resolve_model_id() == "gemini-4-argon"
    assert config.DEFAULT_MODEL_ID == "gemini-4-argon"

    for good in (
        "gemini-4-argon",
        "gemini-3.8-flash",
        "publishers/google/models/gemini-4-argon",
        "projects/ut-austin-bsitter/locations/global/publishers/google/models/gemini-3.8-flash",
        "projects/ut-austin-bsitter/locations/us-central1/endpoints/1234567890",
    ):
        monkeypatch.setenv("MODEL_ID", good)
        assert config.resolve_model_id() == good

    for bad in ("models/gemini-3.8-flash", "gemini flash", "gpt-4o", "gemini-", "../x", "GEMINI-3"):
        monkeypatch.setenv("MODEL_ID", bad)
        with pytest.raises(config.ConfigError):
            config.resolve_model_id()

    monkeypatch.setenv("AURORA_STEER_BEARER_TOKEN", "primary-tok, secondary-rotation-tok ")
    assert config.resolve_bearer_tokens() == ["primary-tok", "secondary-rotation-tok"]


def test_tacc_login_node_smoke_test_script(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    import importlib.util
    from pathlib import Path
    from typing import Any

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "tacc_smoke_test.py"
    spec = importlib.util.spec_from_file_location("tacc_smoke_test", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    token = auth_headers["Authorization"].removeprefix("Bearer ").strip()

    def _inproc_transport(
        method: str,
        url: str,
        tok: str | None,
        payload: dict[str, Any] | None,
        timeout_s: float,
    ) -> tuple[int, dict[str, Any]]:
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        hdrs = {"Authorization": f"Bearer {tok}"} if tok else {}
        if method == "GET":
            resp = client.get(path, headers=hdrs)
        else:
            resp = client.post(path, json=payload, headers=hdrs)
        return resp.status_code, resp.json()

    rc = mod.run_smoke_test(
        "https://aurora-steer.example.run.app",
        token=token,
        transport=_inproc_transport,
    )
    assert rc == 0
