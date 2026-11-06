"""Tests for the shared idempotent-submission primitives (`gcp_exec.submission`)."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from gcp_exec.submission import (
    JOB_ID_RE,
    LEDGER_FILENAME,
    CancelRegistry,
    JobView,
    SchedulerApiError,
    SubmissionRecord,
    call_with_retries,
    fallback_submission_id,
    job_id_for_submission,
    label_value,
    load_record,
    parse_rfc3339,
    plan_submission,
    save_record,
    submission_key,
    validate_submission_id,
)


def test_job_ids_are_deterministic_valid_and_collision_resistant() -> None:
    a = job_id_for_submission("bsit-Run_A-0a1b2c3d-1")
    assert a == job_id_for_submission("bsit-Run_A-0a1b2c3d-1")
    assert a != job_id_for_submission("bsit-run-a-0a1b2c3d-1")  # sanitization collision
    assert a != job_id_for_submission("bsit-Run_A-0a1b2c3d-2")
    long_id = "bsit-" + "x" * 140 + "-0a1b2c3d-1"
    for sid in (a, job_id_for_submission(long_id), job_id_for_submission("bsit-9lives-00000000-1")):
        assert JOB_ID_RE.match(sid) and len(sid) <= 63
    with pytest.raises(ValueError):
        validate_submission_id("has space")
    with pytest.raises(ValueError):
        validate_submission_id("x" * 161)
    assert label_value("Run.With:Odd/Chars") == "run-with-odd-chars"


def _rec(**kw: object) -> SubmissionRecord:
    base: dict[str, object] = {
        "run_id": "run_a",
        "submission_id": "bsit-run_a-00000000-1",
        "job_id": "gx-run-a-00000000-1-abcdef0123",
        "job_name": "projects/p/locations/us-central1/jobs/gx-run-a-00000000-1-abcdef0123",
        "status": "created",
    }
    base.update(kw)
    return SubmissionRecord(**base)  # type: ignore[arg-type]


def test_ledger_roundtrip_and_rejects_foreign_or_corrupt(tmp_path: Path) -> None:
    rec = _rec(rotation_index=2, region="us-east4")
    save_record(tmp_path, rec)
    loaded = load_record(tmp_path, "run_a")
    assert loaded is not None and loaded.rotation_index == 2 and loaded.region == "us-east4"
    assert json.loads((tmp_path / LEDGER_FILENAME).read_text())["submission_key"] == submission_key(
        rec.submission_id
    )
    assert load_record(tmp_path, "other_run") is None
    (tmp_path / LEDGER_FILENAME).write_text("{not json", encoding="utf-8")
    assert load_record(tmp_path, "run_a") is None


def test_plan_with_explicit_ids() -> None:
    rec = _rec()
    same = plan_submission(
        rec, run_id="run_a", submission_id=rec.submission_id, get_job=lambda _n: None, superseded=()
    )
    assert same.action == "resend"
    starved = JobView(
        name=rec.job_name, active=True, submission_key=rec.submission_key, capacity_failure=True
    )
    new = plan_submission(
        rec,
        run_id="run_a",
        submission_id="bsit-run_a-00000000-2",
        get_job=lambda _n: starved,
        superseded=(),
    )
    assert (new.action, new.ordinal, new.rotation_index) == ("new", 1, 1)
    first = plan_submission(
        None,
        run_id="run_a",
        submission_id="bsit-run_a-00000000-1",
        get_job=lambda _n: None,
        superseded=(),
    )
    assert (first.action, first.ordinal) == ("new", 0)


def test_plan_fallback_adopts_only_unsuperseded_active_jobs() -> None:
    rec = _rec(submission_id=fallback_submission_id("run_a", 3), ordinal=3)
    active = JobView(name=rec.job_name, active=True, submission_key=rec.submission_key)
    adopt = plan_submission(
        rec, run_id="run_a", submission_id=None, get_job=lambda _n: active, superseded=()
    )
    assert adopt.action == "adopt"
    cancelled = plan_submission(
        rec,
        run_id="run_a",
        submission_id=None,
        get_job=lambda _n: active,
        superseded={rec.job_name},
    )
    assert (cancelled.action, cancelled.submission_id) == (
        "new",
        fallback_submission_id("run_a", 4),
    )
    done = JobView(name=rec.job_name, active=False, submission_key=rec.submission_key)
    assert (
        plan_submission(
            rec, run_id="run_a", submission_id=None, get_job=lambda _n: done, superseded=()
        ).action
        == "new"
    )
    pending = _rec(status="pending")
    assert (
        plan_submission(
            pending, run_id="run_a", submission_id=None, get_job=lambda _n: None, superseded=()
        ).action
        == "resend"
    )


def test_call_with_retries_only_retries_retryable_errors() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise SchedulerApiError("unavailable", status_code=503)
        return "ok"

    assert call_with_retries(flaky, sleep=sleeps.append) == "ok" and sleeps == [1.0, 2.0]

    def forbidden() -> str:
        raise SchedulerApiError("no", status_code=403)

    with pytest.raises(SchedulerApiError):
        call_with_retries(forbidden, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_cancel_registry_persists_and_parse_rfc3339(tmp_path: Path) -> None:
    reg = CancelRegistry(tmp_path / "reg.json")
    reg.add("projects/p/locations/l/jobs/a")
    assert "projects/p/locations/l/jobs/a" in CancelRegistry(tmp_path / "reg.json")
    assert not CancelRegistry(None).persistent
    ts = parse_rfc3339("2026-11-06T08:00:00.123456789Z")
    assert ts is not None and ts.tzinfo is not None and ts.astimezone(UTC).hour == 8
    assert parse_rfc3339("garbage") is None and parse_rfc3339(None) is None
