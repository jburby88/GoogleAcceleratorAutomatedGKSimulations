"""Shared fixtures for `platform/aurora-steer/tests`."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# QA-05: Starlette 1.x deprecates driving its TestClient with `httpx` in favour of
# `httpx2`, which is not an approved dependency yet. Silence exactly that warning so
# `-W error` runs stay green; switch to httpx2 once it is added to the dev extras.
warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated",
)

from fastapi.testclient import TestClient  # noqa: E402

from app.events import EventStore  # noqa: E402
from app.main import create_app  # noqa: E402
from app.okf_loader import LoadedSkill  # noqa: E402
from app.runner import SteeringEngine  # noqa: E402
from app.schemas import ActionDirective, DiagnosticSummary  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = REPO_ROOT / "workstreams" / "kr2-gk-database" / "skill" / "sim-babysitter"


class FakeSteeringEngine(SteeringEngine):
    """Test-only synchronous SteeringEngine double for unit tests (runbook-faithful)."""

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
        parameter_rules = skill.parameter_rules

        if ft in ("STALL", "TIMESTEP_COLLAPSE"):
            cur_cfl = float(params.get("cfl_fraction", 0.8))
            cur_dt = float(params.get("dt_max", 0.02))
            cfl_rule = parameter_rules["cfl_fraction"]
            dt_rule = parameter_rules["dt_max"]
            new_cfl = round(max(float(cfl_rule.min_value), cur_cfl * 0.75), 4)
            new_dt = round(max(float(dt_rule.min_value), cur_dt * 0.5), 6)
            return ActionDirective(
                action="warm_restart",
                reason=(
                    f"Detected {ft}: reducing cfl_fraction ({cur_cfl} -> {new_cfl}) "
                    f"and dt_max ({cur_dt} -> {new_dt}) for stable checkpoint continuation."
                )[:300],
                parameter_modifications={
                    "cfl_fraction": new_cfl,
                    "dt_max": new_dt,
                },
                restart_from_checkpoint=True,
                confidence="high",
                okf_concepts_used=list(
                    dict.fromkeys([*concept_ids, "runbook-stalled", "allowed-parameters"])
                ),
            )

        if ft == "SPECTRAL_PILEUP":
            cur_nx = int(params.get("nx", 64))
            cur_ny = int(params.get("ny", 32))
            cur_hyp = float(params.get("hyperdiffusion_coeff", 0.15))
            nx_rule = parameter_rules["nx"]
            ny_rule = parameter_rules["ny"]
            hyp_rule = parameter_rules["hyperdiffusion_coeff"]
            next_nx = min(int(nx_rule.max_value), max(int(nx_rule.min_value), int(cur_nx * 1.5)))
            next_ny = min(int(ny_rule.max_value), max(int(ny_rule.min_value), int(cur_ny * 1.5)))
            next_hyp = round(
                min(float(hyp_rule.max_value), max(float(hyp_rule.min_value), cur_hyp * 1.5)), 4
            )
            return ActionDirective(
                action="cold_restart",
                reason=(
                    f"High-k spectral pile-up detected: increasing grid ({cur_nx}x{cur_ny} -> "
                    f"{next_nx}x{next_ny}) and hyperdiffusion_coeff ({cur_hyp} -> {next_hyp}); "
                    "cold restart required after grid change."
                )[:300],
                parameter_modifications={
                    "nx": next_nx,
                    "ny": next_ny,
                    "hyperdiffusion_coeff": next_hyp,
                },
                restart_from_checkpoint=False,
                confidence="high",
                okf_concepts_used=list(
                    dict.fromkeys([*concept_ids, "runbook-spectra", "allowed-parameters"])
                ),
            )

        return ActionDirective(
            action="escalate_to_human",
            reason=(
                f"Failure mode {ft} ({diagnostic.solver}) cannot be safely recovered via numerical "
                "parameters alone; escalating to physicist review per OKF runbook."
            )[:300],
            parameter_modifications={},
            restart_from_checkpoint=False,
            confidence="high",
            okf_concepts_used=list(
                dict.fromkeys([*concept_ids, "runbook-divergence", "escalation-policy"])
            ),
        )


class ScriptedEngine(SteeringEngine):
    """Async engine double returning (or raising) whatever `script()` produces per call."""

    def __init__(self, script: Callable[[DiagnosticSummary, int], Any]) -> None:
        super().__init__()
        self.script = script
        self.calls = 0
        self.allowed_actions_seen: list[frozenset[str]] = []

    async def diagnose(  # type: ignore[override]
        self,
        diagnostic: DiagnosticSummary,
        skill: LoadedSkill,
        *,
        allowed_actions: frozenset[str],
    ) -> ActionDirective:
        self.calls += 1
        self.allowed_actions_seen.append(allowed_actions)
        result = self.script(diagnostic, self.calls)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, BaseException):
            raise result
        return result


def make_payload(**overrides: Any) -> dict[str, Any]:
    """A valid GX `DiagnosticSummary` request body."""
    payload: dict[str, Any] = {
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
        "current_parameters": {
            "nx": 64,
            "ny": 32,
            "nz": 48,
            "dt_max": 0.02,
            "cfl_fraction": 0.8,
            "hyperdiffusion_coeff": 0.1,
        },
        "scalar_metrics": {"high_k_tail_ratio": 0.44},
        "log_tail": ["gx: step 1200 high-k tail ratio exceeded threshold"],
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def auth_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    token = "test-secret-bearer-token-12345"
    monkeypatch.setenv("AURORA_STEER_BEARER_TOKEN", token)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def client(auth_headers: dict[str, str]) -> TestClient:
    store = EventStore(duckdb_path=":memory:")
    app = create_app(steering_engine=FakeSteeringEngine(), event_store=store)
    return TestClient(app)


@pytest.fixture()
def scripted_engine_cls() -> type[ScriptedEngine]:
    return ScriptedEngine


@pytest.fixture()
def payload_factory() -> Callable[..., dict[str, Any]]:
    return make_payload
