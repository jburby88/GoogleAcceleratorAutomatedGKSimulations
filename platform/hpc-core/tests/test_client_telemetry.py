"""Tests for the aurora-steer HTTPS client, escalation notifier, telemetry outbox and smoke test."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import requests
from hpc_core import notify as notify_mod
from hpc_core import telemetry as telemetry_mod
from hpc_core.client import (
    MAX_PAYLOAD_BYTES,
    AuroraSteerClient,
    EventsValidationError,
    TriageClientError,
    TriageTransientError,
    resolve_api_token,
    validate_aurora_steer_url,
)
from hpc_core.notify import send_escalation_notification
from hpc_core.telemetry import TelemetryEmitter, infer_run_metadata

TOKEN = "s3cr3t-token-value-123"
BASE_URL = "https://aurora.example.run.app"
SMOKE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "tacc_smoke_test.py"


# ------------------------------------------------------------------------ fakes/helpers
class FakeResponse:
    def __init__(
        self, status: int = 200, body: Any = b"", headers: dict[str, str] | None = None
    ) -> None:
        self.status_code = status
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.closed = False

    def iter_content(self, chunk_size: int = 1) -> Any:
        for idx in range(0, len(self._body), chunk_size):
            yield self._body[idx : idx + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append((url, kwargs))
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _directive(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "action": "warm_restart",
        "reason": "Reduce dt_max and cfl_fraction.",
        "parameter_modifications": {"dt_max": 0.005, "cfl_fraction": 0.5},
        "restart_from_checkpoint": True,
        "confidence": "high",
        "okf_concepts_used": ["runbook-stalled"],
    }
    base.update(overrides)
    return base


def _summary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": "gx_run_01",
        "solver": "GX",
        "failure_type": "TIMESTEP_COLLAPSE",
        "retry_count": 0,
        "sim_time_reached": 12.0,
        "sim_time_target": 200.0,
        "current_parameters": {"dt_max": 0.01},
        "scalar_metrics": {"dt_collapse_ratio": 0.002},
        "log_tail": ["step 10 ok"],
    }
    base.update(overrides)
    return base


def _ok(body: Any = None, decision: str = "llm") -> FakeResponse:
    return FakeResponse(
        200, _directive() if body is None else body, {"X-Aurora-Decision": decision}
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AURORA_STEER_URL",
        "AURORA_STEER_BEARER_TOKEN",
        "AURORA_API_TOKEN",
        "AURORA_STEER_API_TOKEN",
        "AURORA_STEER_TOKEN",
        "TELEMETRY_OPERATOR_SALT",
        "TACC_OPERATOR_SALT",
        "TELEMETRY_STORE_USERNAMES",
        "ENVIRONMENT",
        "TACC_NOTIFY_WEBHOOK_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _client(*outcomes: Any, **kwargs: Any) -> tuple[AuroraSteerClient, FakeSession]:
    cli = AuroraSteerClient(base_url=BASE_URL, api_token=TOKEN, backoff_base_s=0.0, **kwargs)
    session = FakeSession(*outcomes)
    cli._session = session  # type: ignore[assignment]
    return cli, session


# ------------------------------------------------------------------------ client: request
def test_triage_sends_strict_json_with_bounded_timeouts() -> None:
    cli, session = _client(_ok())
    directive = cli.triage(_summary(), idempotency_key="gx_run_01:evt-1")
    assert directive["action"] == "warm_restart"
    assert cli.last_decision == "llm"
    url, kwargs = session.calls[0]
    assert url == f"{BASE_URL}/v1/skills/sim-babysitter/triage"
    assert json.loads(kwargs["data"]) == _summary()  # valid, complete JSON (HPC-03)
    assert len(kwargs["data"]) < MAX_PAYLOAD_BYTES
    assert kwargs["timeout"] == (10.0, 100.0)
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert kwargs["headers"]["Idempotency-Key"] == "gx_run_01:evt-1"


def test_oversize_payload_is_refused_never_truncated() -> None:
    cli, session = _client()
    big = _summary(log_tail=["x" * 200] * 12)
    with pytest.raises(TriageClientError, match="never truncated") as info:
        cli.triage(big)
    assert not isinstance(info.value, TriageTransientError)
    assert session.calls == []


def test_non_finite_payload_is_refused() -> None:
    cli, session = _client()
    with pytest.raises(TriageClientError, match="strict JSON"):
        cli.triage(_summary(scalar_metrics={"q": float("nan")}))
    assert session.calls == []


def test_invalid_idempotency_key_is_refused() -> None:
    cli, session = _client()
    with pytest.raises(TriageClientError, match="Idempotency-Key"):
        cli.triage(_summary(), idempotency_key="has space")
    assert session.calls == []


def test_token_precedence_and_rotation_list(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_api_token() == ""
    monkeypatch.setenv("AURORA_STEER_API_TOKEN", "third")
    assert resolve_api_token() == "third"
    monkeypatch.setenv("AURORA_API_TOKEN", "second")
    assert resolve_api_token() == "second"
    monkeypatch.setenv("AURORA_STEER_BEARER_TOKEN", " first-a , first-b ")
    assert resolve_api_token() == "first-a"
    assert resolve_api_token("explicit") == "explicit"
    assert AuroraSteerClient(base_url=BASE_URL).api_token == "first-a"


def test_missing_token_is_not_transient() -> None:
    cli = AuroraSteerClient(base_url=BASE_URL)
    with pytest.raises(TriageClientError, match="AURORA_STEER_BEARER_TOKEN") as info:
        cli.triage(_summary())
    assert not isinstance(info.value, TriageTransientError)


def test_url_validation_rejects_credentials_and_plain_http() -> None:
    for bad in (
        "https://user:pw@aurora.example.run.app",
        "https://aurora.example.run.app/?token=x",
        "http://aurora.example.run.app",
        "",
    ):
        with pytest.raises(ValueError):
            validate_aurora_steer_url(bad)
    assert validate_aurora_steer_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"


# ------------------------------------------------------------------------ client: errors
@pytest.mark.parametrize(
    ("outcomes", "calls"),
    [
        ([requests.ConnectionError("refused")] * 3, 3),
        ([FakeResponse(503, b"busy")] * 3, 3),
        ([FakeResponse(429, b"slow down")] * 3, 3),
        ([FakeResponse(500, b"boom")], 1),
        ([requests.exceptions.ConnectTimeout("connect timeout")] * 3, 3),
    ],
)
def test_unreachable_or_overloaded_server_is_transient(outcomes: list[Any], calls: int) -> None:
    cli, session = _client(*outcomes)
    with pytest.raises(TriageTransientError):
        cli.triage(_summary())
    assert len(session.calls) == calls


def test_transient_error_then_success_is_retried_in_call() -> None:
    cli, session = _client(FakeResponse(502, b"bad gateway"), _ok())
    assert cli.triage(_summary())["action"] == "warm_restart"
    assert len(session.calls) == 2


def test_read_timeout_is_transient_and_never_retried() -> None:
    cli, session = _client(requests.ReadTimeout("read timed out"), _ok())
    with pytest.raises(TriageTransientError, match="read timeout"):
        cli.triage(_summary())
    assert len(session.calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 302])
def test_client_side_http_errors_are_not_transient(status: int) -> None:
    cli, _ = _client(FakeResponse(status, b"nope"))
    with pytest.raises(TriageClientError) as info:
        cli.triage(_summary())
    assert not isinstance(info.value, TriageTransientError)


@pytest.mark.parametrize("decision", ["llm_error", "llm_timeout"])
def test_llm_failure_decisions_are_transient(decision: str) -> None:
    escalation = _directive(
        action="escalate_to_human",
        parameter_modifications={},
        restart_from_checkpoint=False,
        reason="LLM unavailable",
    )
    cli, _ = _client(_ok(escalation, decision=decision))
    with pytest.raises(TriageTransientError, match=decision):
        cli.triage(_summary())
    assert cli.last_decision == decision


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"action": "warm_restart", "x": NaN}',
        json.dumps(["warm_restart"]).encode(),
        _directive(action="delete_everything"),
        _directive(reason=5),
        _directive(reason="r" * 301),
        _directive(parameter_modifications=None),
        _directive(parameter_modifications={"dt_max": True}),
        _directive(parameter_modifications={"dt_max": "0.01"}),
        b'{"action":"warm_restart","reason":"ok reason","parameter_modifications":{"dt_max":1e400}}',
        b'{"action":"warm_restart","reason":"ok reason","parameter_modifications":{"nx":'
        + b"9" * 400
        + b"}}",
        _directive(restart_from_checkpoint="yes"),
    ],
)
def test_malformed_or_unsafe_directives_are_rejected(body: Any) -> None:
    cli, _ = _client(_ok(body))
    with pytest.raises(TriageClientError) as info:
        cli.triage(_summary())
    assert not isinstance(info.value, TriageTransientError)


def test_directive_unknown_keys_dropped_and_defaults_applied() -> None:
    body = _directive(extra_key="ignored", confidence="certain")
    del body["restart_from_checkpoint"]
    cli, _ = _client(_ok(body))
    directive = cli.triage(_summary())
    assert "extra_key" not in directive
    assert "confidence" not in directive  # invalid optional value dropped, not trusted
    assert directive["restart_from_checkpoint"] is False


def test_error_text_is_truncated_and_never_contains_the_token() -> None:
    echo = f"Authorization: Bearer {TOKEN} " + "y" * 2000
    cli, _ = _client(FakeResponse(400, echo.encode()))
    with pytest.raises(TriageClientError) as info:
        cli.triage(_summary())
    message = str(info.value)
    assert TOKEN not in message
    assert "[REDACTED]" in message
    assert len(message) < 400


def test_oversize_response_is_rejected() -> None:
    huge = _directive(reason="ok reason", okf_concepts_used=["c" * 100] * 1000)
    cli, _ = _client(_ok(huge))
    with pytest.raises(TriageClientError, match="exceeds"):
        cli.triage(_summary())


# ------------------------------------------------------------------------ client: events
def test_post_events_classifies_failures() -> None:
    cli, session = _client(
        FakeResponse(200, {"accepted": 1, "deduplicated": 0}),
        FakeResponse(200, b"<html>not json</html>"),
        FakeResponse(422, b"bad event"),
        FakeResponse(503, b"busy"),
        FakeResponse(401, b"unauthorized"),
        requests.ConnectionError("down"),
    )
    assert cli.post_events([{"event_id": "e1"}]) == {"accepted": 1, "deduplicated": 0}
    with pytest.raises(TriageClientError, match="Invalid JSON"):
        cli.post_events([{"event_id": "e2"}])
    with pytest.raises(EventsValidationError):
        cli.post_events([{"event_id": "e3"}])
    with pytest.raises(TriageTransientError):
        cli.post_events([{"event_id": "e4"}])
    with pytest.raises(TriageClientError) as info:
        cli.post_events([{"event_id": "e5"}])
    assert not isinstance(info.value, EventsValidationError)  # never quarantine on auth errors
    with pytest.raises(TriageTransientError):
        cli.post_events([{"event_id": "e6"}])
    assert session.calls[0][1]["timeout"] == (5.0, 5.0)


def test_post_events_non_finite_event_is_poison_and_not_sent() -> None:
    cli, session = _client()
    with pytest.raises(EventsValidationError):
        cli.post_events([{"event_id": "e1", "payload": {"q": float("inf")}}])
    assert session.calls == []


# ------------------------------------------------------------------------ notify
# Built at runtime so the repo hygiene scan never sees a webhook-looking literal.
SECRET_WEBHOOK = "https://hooks." + "slack.com/services/" + "T000/B000/SECRETPATH123"


def test_notify_never_logs_the_webhook_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    def boom(url: str, **_kwargs: Any) -> Any:
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")

    monkeypatch.setattr(notify_mod.requests, "post", boom)
    with caplog.at_level(logging.WARNING):
        ok = send_escalation_notification(
            "run_1", "GX", "reason", tmp_path, webhook_url=SECRET_WEBHOOK
        )
    assert ok is False
    assert "SECRETPATH123" not in caplog.text
    assert "ConnectionError" in caplog.text


def test_notify_requires_https_and_escapes_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sent: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> Any:
        sent.append({"url": url, **kwargs})
        return FakeResponse(200)

    monkeypatch.setattr(notify_mod.requests, "post", fake_post)
    assert not send_escalation_notification(
        "run_1", "GX", "r", tmp_path, webhook_url="http://hooks.example.com/x"
    )
    assert sent == []
    assert send_escalation_notification(
        "run_1", "GX", "<!channel> & <@U123>", tmp_path, webhook_url=SECRET_WEBHOOK
    )
    text = sent[0]["json"]["text"]
    assert "<!channel>" not in text and "&lt;!channel&gt; &amp; &lt;@U123&gt;" in text
    assert sent[0]["allow_redirects"] is False


def test_notify_non_2xx_logs_status_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setattr(notify_mod.requests, "post", lambda url, **kw: FakeResponse(404, b"x"))
    with caplog.at_level(logging.WARNING):
        assert not send_escalation_notification(
            "run_1", "GX", "r", tmp_path, webhook_url=SECRET_WEBHOOK
        )
    assert "404" in caplog.text and "SECRETPATH123" not in caplog.text


# ------------------------------------------------------------------------ telemetry
class RecordingClient:
    def __init__(self, on_post: Any = None) -> None:
        self.batches: list[list[dict[str, Any]]] = []
        self.on_post = on_post

    def post_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        self.batches.append(list(events))
        if self.on_post is not None:
            hook, self.on_post = self.on_post, None
            hook()
        return {"accepted": len(events), "deduplicated": 0}


def test_outbox_appends_are_durable_and_rewrites_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    appended: list[Path] = []
    rewritten: list[Path] = []
    real_append = telemetry_mod.append_line_durable
    real_rewrite = telemetry_mod.atomic_write_text

    def spy_append(path: Path, line: str) -> None:
        appended.append(Path(path))
        real_append(path, line)

    def spy_rewrite(path: Path, text: str) -> None:
        rewritten.append(Path(path))
        real_rewrite(path, text)

    monkeypatch.setattr(telemetry_mod, "append_line_durable", spy_append)
    monkeypatch.setattr(telemetry_mod, "atomic_write_text", spy_rewrite)
    outbox = tmp_path / "state" / "outbox.jsonl"
    emitter = TelemetryEmitter(outbox, client=RecordingClient())
    emitter.emit(event_type="submitted", run_id="r1", solver="GX")
    assert appended == [outbox]
    assert emitter.flush(force=True) == 1
    assert rewritten == [outbox]
    assert emitter.read_pending_events() == []


def test_flush_keeps_events_appended_while_a_batch_is_in_flight(tmp_path: Path) -> None:
    outbox = tmp_path / "state" / "outbox.jsonl"
    late = TelemetryEmitter(outbox)  # e.g. tacc-resolve writing to the same outbox
    cli = RecordingClient(
        on_post=lambda: late.emit(event_type="human_resolved", run_id="r2", solver="GX")
    )
    emitter = TelemetryEmitter(outbox, client=cli)
    emitter.emit(event_type="submitted", run_id="r1", solver="GX")
    assert emitter.flush(force=True) == 1
    pending = emitter.read_pending_events()
    assert [ev["run_id"] for ev in pending] == ["r2"]


def test_unparseable_outbox_lines_are_quarantined_not_dropped(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox.jsonl"
    emitter = TelemetryEmitter(outbox, client=RecordingClient())
    emitter.emit(event_type="submitted", run_id="r1", solver="GX")
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write('{"torn": \n')
    assert emitter.flush(force=True) == 1
    quarantined = emitter.quarantine_file.read_text(encoding="utf-8").splitlines()
    assert len(quarantined) == 1 and '{"torn":' in json.loads(quarantined[0])["raw_line"]
    assert outbox.read_text(encoding="utf-8") == ""


def test_operator_salt_is_random_persisted_0600_and_stable(tmp_path: Path) -> None:
    outbox = tmp_path / "state" / "outbox.jsonl"
    first = TelemetryEmitter(outbox).emit(
        event_type="human_resolved", run_id="r1", solver="GX", operator="alice"
    )
    assert first is not None
    salt_file = outbox.with_name(".operator_hash_salt")
    assert stat.S_IMODE(salt_file.stat().st_mode) == 0o600
    public_default = hashlib.sha256(b"ut-austin-bsitter-default-salt:alice").hexdigest()[:16]
    assert first["operator"] != f"op_{public_default}"
    second = TelemetryEmitter(outbox).emit(
        event_type="human_resolved", run_id="r1", solver="GX", operator="alice"
    )
    assert second is not None and second["operator"] == first["operator"]


def test_operator_salt_from_environment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TELEMETRY_OPERATOR_SALT", "env-salt-value-0123456789")
    ev = TelemetryEmitter(tmp_path / "outbox.jsonl").emit(
        event_type="human_resolved", run_id="r1", solver="GX", operator="alice"
    )
    expected = hashlib.sha256(b"env-salt-value-0123456789:alice").hexdigest()[:16]
    assert ev is not None and ev["operator"] == f"op_{expected}"
    assert not (tmp_path / ".operator_hash_salt").exists()


def test_unknown_environment_is_never_recorded_as_prod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")
    emitter = TelemetryEmitter(tmp_path / "outbox.jsonl")
    assert emitter.default_environment == "sim"
    ev = emitter.emit(event_type="submitted", run_id="r1", solver="GX", environment="Prod ")
    assert ev is not None and ev["environment"] == "prod"
    ev = emitter.emit(event_type="submitted", run_id="r1", solver="GX", environment="qa")
    assert ev is not None and ev["environment"] == "sim"


def test_infer_run_metadata_never_invents_ids(tmp_path: Path) -> None:
    cfg = {"campaign_id": "pilot_50", "default_fidelity": "low"}
    guessable = tmp_path / "w7x_highfi_run"
    guessable.mkdir()
    assert infer_run_metadata(guessable, cfg) == ("pilot_50", "low", "")

    declared = tmp_path / "run_a"
    declared.mkdir()
    (declared / "run_meta.json").write_text(
        json.dumps({"campaign_id": "camp_1", "fidelity": "HIGH", "geometry_id": "mrx_eq_0042"})
    )
    assert infer_run_metadata(declared, cfg) == ("camp_1", "high", "mrx_eq_0042")


@pytest.mark.parametrize(
    "meta",
    [
        "{not json",
        "[1, 2]",
        json.dumps({"campaign_id": "../../etc"}),
        json.dumps({"equilibrium_id": "a/b"}),
        json.dumps({"equilibrium_id": 42}),
        json.dumps({"fidelity": "medium"}),
        json.dumps({"campaign_id": "c" * 65}),
    ],
)
def test_infer_run_metadata_fails_closed_on_bad_run_meta(tmp_path: Path, meta: str) -> None:
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(meta)
    with pytest.raises(ValueError):
        infer_run_metadata(run_dir, {"campaign_id": "pilot_50"})


# ------------------------------------------------------------------------ smoke test script
@pytest.fixture
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tacc_smoke_test_under_test", SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SmokeTransport:
    def __init__(self, decision: str | None = "llm", echo_token: bool = False) -> None:
        self.decision = decision
        self.echo_token = echo_token
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, method: str, url: str, tok: str | None, payload: Any, timeout_s: float
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        self.calls.append(
            {"method": method, "url": url, "token": tok, "payload": payload, "timeout": timeout_s}
        )
        if url.endswith(("/health", "/healthz", "/v1/healthz")):
            return 200, {"status": "ok"}, {}
        if url.endswith("/v1/events"):
            return 200, {"accepted": 1, "deduplicated": 0}, {}
        headers = {"x-aurora-decision": self.decision} if self.decision else {}
        body = _directive()
        if self.echo_token:
            body["reason"] = f"echo {tok}"
        return 200, body, headers


def test_smoke_token_precedence(monkeypatch: pytest.MonkeyPatch, smoke: ModuleType) -> None:
    monkeypatch.setenv("AURORA_STEER_TOKEN", "legacy-token")
    assert smoke.resolve_token(None) == "legacy-token"
    monkeypatch.setenv("AURORA_STEER_BEARER_TOKEN", "canonical-token,next")
    assert smoke.resolve_token(None) == "canonical-token"
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        smoke, "run_smoke_test", lambda url, token, skip_triage: captured.update(token=token) or 0
    )
    assert smoke.main(["--url", BASE_URL]) == 0
    assert captured["token"] == "canonical-token"


def test_smoke_uses_client_route_llm_decision_and_long_timeout(
    smoke: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = SmokeTransport()
    assert smoke.run_smoke_test(BASE_URL, TOKEN, transport=transport) == 0
    health, events, triage = transport.calls
    assert health["token"] is None  # token never sent to the unauthenticated probe
    assert events["token"] == TOKEN
    assert triage["url"] == f"{BASE_URL}/v1/skills/sim-babysitter/triage"
    assert triage["timeout"] >= 75.0
    assert TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize("decision", ["llm_error", "guardrail_rejected", "cached", None])
def test_smoke_fails_unless_the_llm_answered(smoke: ModuleType, decision: str | None) -> None:
    assert smoke.run_smoke_test(BASE_URL, TOKEN, transport=SmokeTransport(decision)) == 1


def test_smoke_never_prints_the_token(
    smoke: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = smoke.run_smoke_test(
        BASE_URL, TOKEN, transport=SmokeTransport("llm_error", echo_token=True)
    )
    out = capsys.readouterr()
    assert rc == 1
    assert TOKEN not in out.out and TOKEN not in out.err


def test_smoke_rejects_plain_http_remote_url(smoke: ModuleType) -> None:
    transport = SmokeTransport()
    assert smoke.run_smoke_test("http://aurora.example.run.app", TOKEN, transport=transport) == 2
    assert transport.calls == []


def test_smoke_default_transport_never_forwards_token_on_redirect(
    monkeypatch: pytest.MonkeyPatch, smoke: ModuleType
) -> None:
    captured: dict[str, Any] = {}

    class FakeHTTPResponse:
        status = 200
        headers = requests.structures.CaseInsensitiveDict({"X-Aurora-Decision": "llm"})

        def read(self, _n: int = -1) -> bytes:
            return b'{"ok": true}'

        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float) -> FakeHTTPResponse:
        captured["req"] = req
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)
    status, body, headers = smoke._default_http_json("POST", f"{BASE_URL}/x", TOKEN, {"a": 1}, 9.0)
    req = captured["req"]
    assert (status, body, captured["timeout"]) == (200, {"ok": True}, 9.0)
    assert req.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    assert "Authorization" not in req.headers
    assert headers["X-Aurora-Decision"] == "llm"


def test_smoke_script_runs_on_tacc_system_python() -> None:
    source = SMOKE_SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 8))
    assert "from datetime import UTC" not in source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = {node.module.split(".")[0]}
        else:
            continue
        assert roots <= set(sys.stdlib_module_names), f"non-stdlib import {roots}"


def test_smoke_payloads_match_the_aurora_steer_schemas(smoke: ModuleType) -> None:
    schemas = pytest.importorskip("app.schemas")
    transport = SmokeTransport()
    assert smoke.run_smoke_test(BASE_URL, TOKEN, transport=transport) == 0
    schemas.EventsBatchRequest.model_validate(transport.calls[1]["payload"])
    schemas.DiagnosticSummary.model_validate(transport.calls[2]["payload"])
    raw = json.dumps(transport.calls[2]["payload"]).encode("utf-8")
    assert len(raw) < schemas.MAX_TRIAGE_BYTES
