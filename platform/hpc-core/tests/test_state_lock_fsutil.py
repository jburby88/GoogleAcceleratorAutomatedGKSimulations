"""Tests for crash-safe writes (HPC-02), fail-closed state loading and the cross-host lock."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest
from hpc_core import fsutil
from hpc_core.state import (
    STATE_SCHEMA_VERSION,
    DriverAlreadyRunningError,
    InstanceLock,
    RunRecord,
    StateCorruptError,
    StateSchemaError,
    StateStore,
    acquire_single_instance_lock,
)


def test_atomic_write_fsyncs_data_before_rename(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(fsutil.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        fsutil.os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1]
    )
    target = tmp_path / "state.json"
    fsutil.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert events.index("fsync") < events.index("replace")
    assert "fsync" in events[events.index("replace") :]  # parent directory fsync after rename


def test_atomic_write_failure_leaves_target_and_no_temp_files(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    def boom(_a, _b):
        raise OSError("disk full")

    monkeypatch.setattr(fsutil.os, "replace", boom)
    with pytest.raises(OSError):
        fsutil.atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_atomic_write_json_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fsutil.atomic_write_json(tmp_path / "x.json", {"q": float("nan")})
    assert not (tmp_path / "x.json").exists()


def test_safe_ids_reject_traversal() -> None:
    assert fsutil.is_safe_id("mrx_eq_0001")
    for bad in ("", "..", "../x", "a/b", ".hidden", "-flag", "x" * 129, None, 5):
        assert not fsutil.is_safe_id(bad)


def test_state_store_round_trips_v2_fields(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl")
    rec = RunRecord(run_id="r1", solver="GX", run_dir="/d", status="SUBMITTING")
    rec.pending_submission = {"submission_id": "bsit-r1-abcd-1", "kind": "initial"}
    rec.restart_history.append({"seq": 1, "job_id": "123"})
    rec.consecutive_poll_errors = 2
    store.upsert_run(rec)
    raw = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == STATE_SCHEMA_VERSION
    loaded = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl").get_run("r1")
    assert loaded == rec


def test_legacy_v1_state_loads_with_defaults(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "runs": {
                    "old": {
                        "run_id": "old",
                        "solver": "GENE",
                        "run_dir": "/x",
                        "status": "RUNNING",
                        "geometry_id": "w7x_standard",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rec = StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl").get_run("old")
    assert rec.equilibrium_id == "w7x_standard" and rec.submit_seq == 0
    assert rec.restart_history == []


def test_corrupt_or_newer_state_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"runs": {"r": ', encoding="utf-8")
    with pytest.raises(StateCorruptError):
        StateStore(path, tmp_path / "audit.jsonl")
    path.write_text(json.dumps({"schema_version": 99, "runs": {}}), encoding="utf-8")
    with pytest.raises(StateSchemaError):
        StateStore(path, tmp_path / "audit.jsonl")
    path.write_text(json.dumps({"runs": {"r": {"solver": "GX"}}}), encoding="utf-8")
    with pytest.raises(StateCorruptError):
        StateStore(path, tmp_path / "audit.jsonl")


def test_record_defaults_never_invent_an_equilibrium() -> None:
    rec = RunRecord(run_id="r", solver="GX", run_dir="/d")
    assert rec.equilibrium_id == "" and rec.geometry_id == ""


def _write_foreign_lease(lock_path: Path, *, age_s: float) -> Path:
    lease = lock_path.with_name(lock_path.name + ".lease")
    lease.parent.mkdir(parents=True, exist_ok=True)
    beat = time.time() - age_s
    lease.write_text(
        json.dumps(
            {"host": "login2.other", "pid": 4242, "token": "foreign", "heartbeat_epoch": beat}
        ),
        encoding="utf-8",
    )
    os.utime(lease, (beat, beat))
    return lease


def test_fresh_lease_from_another_login_node_blocks(tmp_path: Path) -> None:
    lock_path = tmp_path / "driver.lock"
    _write_foreign_lease(lock_path, age_s=10)
    with pytest.raises(DriverAlreadyRunningError, match="login2.other"):
        with acquire_single_instance_lock(lock_path, stale_after_s=600):
            pass


def test_stale_lease_from_another_node_is_taken_over(tmp_path: Path) -> None:
    lock_path = tmp_path / "driver.lock"
    lease = _write_foreign_lease(lock_path, age_s=3600)
    with acquire_single_instance_lock(lock_path, stale_after_s=600) as lock:
        data = json.loads(lease.read_text(encoding="utf-8"))
        assert data["host"] == socket.gethostname() and data["token"] == lock.token
        assert list(tmp_path.glob("driver.lock.lease.stale.*"))
    assert not lease.exists()  # released


def test_refresh_detects_takeover_by_another_host(tmp_path: Path) -> None:
    lock_path = tmp_path / "driver.lock"
    with acquire_single_instance_lock(lock_path) as lock:
        lock.refresh()
        _write_foreign_lease(lock_path, age_s=0)
        with pytest.raises(DriverAlreadyRunningError, match="Lost lock lease"):
            lock.refresh()
    # our release must not delete the foreign holder's lease
    assert lock_path.with_name("driver.lock.lease").exists()


def test_lease_left_by_dead_process_on_same_host_is_reclaimed(tmp_path: Path) -> None:
    lock_path = tmp_path / "driver.lock"
    lease = lock_path.with_name("driver.lock.lease")
    lease.write_text(
        json.dumps({"host": socket.gethostname(), "pid": 999999, "token": "dead"}),
        encoding="utf-8",
    )
    with acquire_single_instance_lock(lock_path):
        assert json.loads(lease.read_text(encoding="utf-8"))["token"] != "dead"


def test_flock_unsupported_falls_back_to_lease(tmp_path: Path, monkeypatch) -> None:
    import errno

    from hpc_core import state as state_mod

    def no_flock(_fd, _op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(state_mod.fcntl, "flock", no_flock)
    lock = InstanceLock(tmp_path / "driver.lock", stale_after_s=600)
    lock.acquire()
    try:
        assert lock.flock_supported is False
        with pytest.raises(DriverAlreadyRunningError):
            InstanceLock(tmp_path / "driver.lock", stale_after_s=600).acquire()
    finally:
        lock.release()


def test_lock_wait_times_out(tmp_path: Path) -> None:
    lock_path = tmp_path / "driver.lock"
    with acquire_single_instance_lock(lock_path):
        start = time.monotonic()
        with pytest.raises(DriverAlreadyRunningError):
            with acquire_single_instance_lock(lock_path, wait_s=1.0):
                pass
        assert time.monotonic() - start >= 0.9
