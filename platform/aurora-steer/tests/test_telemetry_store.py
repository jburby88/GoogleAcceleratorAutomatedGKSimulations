"""Telemetry store and BigQuery write path (STEER-06, STEER-07, STEER-08)."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime
from typing import Any

import pytest

from app import events
from app.events import EventStore, _BoundedIdSet, _stream_rows_to_bigquery
from app.schemas import LifecycleEvent


def _event(event_id: str, **kw: Any) -> LifecycleEvent:
    base: dict[str, Any] = {
        "event_id": event_id,
        "run_id": "gx_run_1",
        "campaign_id": "pilot_50",
        "solver": "GX",
        "event_type": "submitted",
        "ts_utc": datetime.now(UTC),
        "environment": "prod",
    }
    base.update(kw)
    return LifecycleEvent(**base)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture()
def bq_env(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    monkeypatch.setenv("BQ_ENABLED", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ut-austin-test")
    monkeypatch.setattr(events, "_get_bq_token", lambda: "fake-token")
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls: list[Any] = []
    return calls


def _failures() -> int:
    return events.bq_insert_failure_count()


def test_bigquery_insert_errors_with_http_200_are_detected_and_logged(
    bq_env: list[Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_urlopen(req: Any, timeout: float) -> _FakeResponse:
        bq_env.append(req)
        body = {"insertErrors": [{"index": 0, "errors": [{"reason": "invalid"}]}]}
        return _FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    before = _failures()
    assert _stream_rows_to_bigquery("directives", [{"directive_id": "d1"}]) is False
    assert _failures() == before + 1
    assert len(bq_env) == 1  # row-level rejections are not retried
    logged = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(e["severity"] == "ERROR" and "insertErrors" in e["error"] for e in logged)


def test_bigquery_success_and_transient_retry(
    bq_env: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def flaky(req: Any, timeout: float) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, None)
        return _FakeResponse(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    before = _failures()
    assert _stream_rows_to_bigquery("events", [{"event_id": "e1"}]) is True
    assert attempts["n"] == 2
    assert _failures() == before


def test_bigquery_permanent_http_error_is_not_retried(
    bq_env: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def forbidden(req: Any, timeout: float) -> _FakeResponse:
        attempts["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    before = _failures()
    assert _stream_rows_to_bigquery("events", [{"event_id": "e1"}]) is False
    assert attempts["n"] == 1
    assert _failures() == before + 1


def test_bigquery_disabled_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BQ_ENABLED", raising=False)

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("must not call BigQuery when disabled")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert _stream_rows_to_bigquery("events", [{"event_id": "e1"}]) is True


def test_seen_event_ids_are_bounded() -> None:
    ids = _BoundedIdSet(3)
    for i in range(5):
        ids.add(f"evt-{i}")
    assert len(ids) == 3
    assert "evt-0" not in ids and "evt-4" in ids


def test_failed_insert_does_not_mark_events_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    store = EventStore(":memory:")
    real_conn = store._conn

    class FailingConn:
        def __init__(self) -> None:
            self.fail = True

        def executemany(self, *a: Any, **k: Any) -> Any:
            if self.fail:
                self.fail = False
                raise RuntimeError("disk full")
            return real_conn.executemany(*a, **k)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_conn, name)

    store._conn = FailingConn()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        store.ingest_events([_event("evt-retry-0001")])
    # The client's retry of the same batch is accepted, not "deduplicated" away.
    assert store.ingest_events([_event("evt-retry-0001")]) == (1, 0)
    assert real_conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_duplicate_ids_inside_one_batch_and_synthetic_flags() -> None:
    store = EventStore(":memory:")
    accepted, dedup = store.ingest_events(
        [
            _event("evt-a-000001"),
            _event("evt-a-000001"),
            _event("evt-b-000001", environment="sim"),
            _event("evt-c-000001", environment="preflight"),
        ]
    )
    assert (accepted, dedup) == (3, 1)
    rows = dict(
        store._conn.execute(
            "SELECT event_id, is_synthetic FROM events ORDER BY event_id"
        ).fetchall()
    )
    assert rows == {"evt-a-000001": False, "evt-b-000001": True, "evt-c-000001": True}
