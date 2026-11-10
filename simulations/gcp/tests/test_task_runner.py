"""Tests for the in-job task runner (`gx-task`) and the checkpoint validator (GCP-03, GCP-C01..C04)."""

from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from gcp_exec.checkpoint import netcdf_file_problem
from gcp_exec.submission import FENCE_FILENAME, LEDGER_FILENAME
from gcp_exec.task_runner import (
    EXIT_CHECKPOINT_INVALID,
    EXIT_REFUSED,
    EXIT_SUPERSEDED,
    EXIT_TERMINATED,
    TASK_RESULT_FILENAME,
    TaskConfig,
    TaskRunner,
)

ATTEMPT = "gx-run-a-0123456789"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
UNDEF = 0xFFFFFFFFFFFFFFFF


def hdf5_bytes(total: int, *, eof: int | None = None, version: int = 0, flags: int = 0) -> bytes:
    """Minimal HDF5 superblock (v0 or v2/v3) padded to `total` bytes."""
    sig = b"\x89HDF\r\n\x1a\n"
    end = total if eof is None else eof
    if version == 0:
        head = sig + bytes([0, 0, 0, 0, 0, 8, 8, 0]) + struct.pack("<HHI", 4, 16, 0)
        head += struct.pack("<QQQQ", 0, UNDEF, end, UNDEF)
    else:
        head = sig + bytes([version, 8, 8, flags]) + struct.pack("<QQQQ", 0, UNDEF, end, 48)
        head += b"\0\0\0\0"
    return head + b"\0" * max(0, total - len(head))


FAKE_GX = textwrap.dedent(
    """
    import json, os, struct, sys, time
    UNDEF = 0xFFFFFFFFFFFFFFFF
    def hdf5(total, eof=None):
        sig = b"\\x89HDF\\r\\n\\x1a\\n"
        end = total if eof is None else eof
        head = sig + bytes([0, 0, 0, 0, 0, 8, 8, 0]) + struct.pack("<HHI", 4, 16, 0)
        head += struct.pack("<QQQQ", 0, UNDEF, end, UNDEF)
        return head + b"\\0" * max(0, total - len(head))
    plan = json.loads(open(os.environ["FAKE_GX_PLAN"]).read())
    if plan.get("require_valid_restart"):
        data = open("gx.restart.nc", "rb").read()
        if len(data) < 64 or data[:4] != b"\\x89HDF":
            sys.exit(3)
    for step in plan["steps"]:
        kind = step[0]
        if kind == "print":
            print(step[1], flush=True)
        elif kind == "write":
            with open(step[1], "wb") as fh:
                fh.write(hdf5(step[2], step[3] if len(step) > 3 else None))
        elif kind == "sleep":
            time.sleep(step[1])
    sys.exit(plan.get("rc", 0))
    """
)


def make_run(
    tmp_path: Path,
    *,
    restart: bool = False,
    fence: str | None = ATTEMPT,
    fluxes: bool | None = True,
) -> Path:
    src = tmp_path / "bucket" / "queue" / "running" / "run_a"
    src.mkdir(parents=True)
    diag = f"\n[Diagnostics]\nfluxes = {str(fluxes).lower()}\n" if fluxes is not None else ""
    (src / "gx.in").write_text(
        f"[Dimensions]\nntheta = 32\n\n[Restart]\nrestart = {str(restart).lower()}\n"
        f"save_for_restart = true\n{diag}",
        encoding="utf-8",
    )
    if fence is not None:
        (src / FENCE_FILENAME).write_text(fence + "\n", encoding="utf-8")
    return src


def make_build(tmp_path: Path, *, real: bool = True) -> Path:
    path = tmp_path / "gx_build.json"
    path.write_text(
        json.dumps({"schema_version": 1, "gx_commit": COMMIT, "real_gx": real}), encoding="utf-8"
    )
    return path


def make_cfg(tmp_path: Path, src: Path, plan: dict, **kw: object) -> TaskConfig:
    script = tmp_path / "fake_gx.py"
    script.write_text(FAKE_GX, encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    os.environ["FAKE_GX_PLAN"] = str(plan_path)
    defaults: dict[str, object] = {
        "src_dir": src,
        "attempt_id": ATTEMPT,
        "gx_command": [sys.executable, str(script)],
        "work_root": tmp_path / "work",
        "sync_interval_s": 0.05,
        "output_sync_interval_s": 0.0,
        "stable_seconds": 0.0,
        "poll_s": 0.02,
        "environment": "prod",
        "build_info_path": make_build(tmp_path),
    }
    defaults.update(kw)
    return TaskConfig(**defaults)  # type: ignore[arg-type]


def result(src: Path) -> dict:
    return json.loads((src / TASK_RESULT_FILENAME).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- checkpoint.py


def test_checkpoint_validator_detects_truncated_and_open_files(tmp_path: Path) -> None:
    good = tmp_path / "good.nc"
    good.write_bytes(hdf5_bytes(4096))
    assert netcdf_file_problem(good) is None
    v2 = tmp_path / "v2.nc"
    v2.write_bytes(hdf5_bytes(2048, version=2))
    assert netcdf_file_problem(v2) is None
    truncated = tmp_path / "trunc.nc"
    truncated.write_bytes(hdf5_bytes(1024, eof=8192))
    assert "truncated" in (netcdf_file_problem(truncated) or "")
    open_v3 = tmp_path / "open.nc"
    open_v3.write_bytes(hdf5_bytes(2048, version=3, flags=1))
    assert "open for writing" in (netcdf_file_problem(open_v3) or "")
    undefined = tmp_path / "undef.nc"
    undefined.write_bytes(hdf5_bytes(2048, eof=UNDEF))
    assert netcdf_file_problem(undefined) is not None
    classic = tmp_path / "classic.nc"
    classic.write_bytes(b"CDF\x02" + b"\0" * 60)
    assert netcdf_file_problem(classic) is None
    empty = tmp_path / "empty.nc"
    empty.write_bytes(b"")
    assert netcdf_file_problem(empty) == "empty file"
    garbage = tmp_path / "garbage.nc"
    garbage.write_bytes(b"not a netcdf file at all")
    assert "bad magic" in (netcdf_file_problem(garbage) or "")
    assert netcdf_file_problem(tmp_path / "missing.nc") is not None


def test_checkpoint_validator_requires_g_and_time_for_restart_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    vars_by_name: dict[str, dict[str, object]] = {}

    class _FakeDataset:
        def __init__(self, path: str, _mode: str) -> None:
            self.variables = vars_by_name.get(Path(path).name, {})
            self.groups: dict[str, object] = {}

        def __enter__(self) -> _FakeDataset:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    fake_nc = types.SimpleNamespace(Dataset=_FakeDataset)
    monkeypatch.setitem(sys.modules, "netCDF4", fake_nc)

    restart = tmp_path / "gx.restart.nc"
    restart.write_bytes(hdf5_bytes(4096))
    vars_by_name["gx.restart.nc"] = {"G": object()}  # missing 'time'
    assert "missing required 'G' and 'time'" in (netcdf_file_problem(restart) or "")

    vars_by_name["gx.restart.nc"] = {"G": object(), "time": object()}
    assert netcdf_file_problem(restart) is None

    out = tmp_path / "gx.out.nc"
    out.write_bytes(hdf5_bytes(4096))
    vars_by_name["gx.out.nc"] = {"time": object()}
    assert netcdf_file_problem(out) is None


# --------------------------------------------------------------------------- task runner


def test_happy_path_publishes_validated_outputs_and_provenance(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    plan = {
        "steps": [
            ["print", "Step 1: Time = 0.1 dt = 0.01 Q_i = 1.0"],
            ["write", "gx.restart.nc", 4096],
            ["sleep", 0.3],
            ["write", "gx.out.nc", 8192],
            ["print", "done"],
        ],
        "rc": 0,
    }
    rc = TaskRunner(make_cfg(tmp_path, src, plan)).run()
    assert rc == 0
    assert "Time = 0.1" in (src / "gx.log").read_text(encoding="utf-8")
    assert netcdf_file_problem(src / "gx.restart.nc") is None
    assert netcdf_file_problem(src / "gx.out.nc") is None
    res = result(src)
    assert res["outcome"] == "gx_exited" and res["gx_exit_code"] == 0
    assert res["simulated"] is False and res["gx_build"]["gx_commit"] == COMMIT
    assert not [p for p in src.iterdir() if ".tmp" in p.name]


def test_superseded_attempt_never_writes(tmp_path: Path) -> None:
    src = make_run(tmp_path, fence="gx-newer-attempt")
    rc = TaskRunner(make_cfg(tmp_path, src, {"steps": [["print", "x"]]})).run()
    assert rc == EXIT_SUPERSEDED
    assert not (src / "gx.log").exists() and not (src / TASK_RESULT_FILENAME).exists()


def test_superseded_mid_run_stops_gx_without_writing(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    plan = {"steps": [["print", "start"], ["sleep", 30]]}
    runner = TaskRunner(make_cfg(tmp_path, src, plan, sync_interval_s=0.2))
    import threading

    def flip() -> None:
        time.sleep(0.6)
        (src / FENCE_FILENAME).write_text("gx-newer-attempt\n", encoding="utf-8")

    threading.Thread(target=flip, daemon=True).start()
    t0 = time.monotonic()
    assert runner.run() == EXIT_SUPERSEDED
    assert time.monotonic() - t0 < 10
    assert not (src / TASK_RESULT_FILENAME).exists()


def test_torn_checkpoint_is_never_published(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    old = hdf5_bytes(3000)
    (src / "gx.restart.nc").write_bytes(old)
    plan = {"steps": [["write", "gx.restart.nc", 1024, 9999]], "rc": 0}  # eof beyond size
    assert TaskRunner(make_cfg(tmp_path, src, plan)).run() == 0
    assert (src / "gx.restart.nc").read_bytes() == old  # last good generation kept


def test_new_checkpoint_rotates_previous_generation(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    old = hdf5_bytes(3000)
    (src / "gx.restart.nc").write_bytes(old)
    plan = {"steps": [["write", "gx.restart.nc", 5000]], "rc": 0}
    assert TaskRunner(make_cfg(tmp_path, src, plan)).run() == 0
    assert (src / "gx.restart.nc").stat().st_size == 5000
    assert (src / "gx.prev.restart.nc").read_bytes() == old


def test_restart_without_any_checkpoint_fails_closed(tmp_path: Path) -> None:
    src = make_run(tmp_path, restart=True)
    rc = TaskRunner(make_cfg(tmp_path, src, {"steps": [["print", "should not run"]]})).run()
    assert rc == EXIT_CHECKPOINT_INVALID
    assert not (src / "gx.log").exists()
    assert result(src)["outcome"] == "checkpoint_invalid"


def test_restart_falls_back_to_previous_valid_generation(tmp_path: Path) -> None:
    src = make_run(tmp_path, restart=True)
    (src / "gx.restart.nc").write_bytes(hdf5_bytes(512, eof=4096))  # torn primary
    (src / "gx.prev.restart.nc").write_bytes(hdf5_bytes(4096))
    plan = {"require_valid_restart": True, "steps": [["print", "resumed"]], "rc": 0}
    assert TaskRunner(make_cfg(tmp_path, src, plan)).run() == 0
    res = result(src)
    assert res["restart"]["requested"] is True
    assert res["restart"]["fallback_to_previous"] is True
    assert res["restart"]["used"] == "gx.prev.restart.nc"


def test_restart_with_only_invalid_checkpoints_fails_closed(tmp_path: Path) -> None:
    src = make_run(tmp_path, restart=True)
    (src / "gx.restart.nc").write_bytes(b"garbage")
    (src / "gx.prev.restart.nc").write_bytes(hdf5_bytes(100, eof=5000))
    assert TaskRunner(make_cfg(tmp_path, src, {"steps": []})).run() == EXIT_CHECKPOINT_INVALID
    assert set(result(src)["restart"]["problems"]) == {"gx.restart.nc", "gx.prev.restart.nc"}


def test_gx_simulate_run_is_always_refused(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    cfg = make_cfg(tmp_path, src, {"steps": []}, simulate_requested=True, environment="sim")
    assert TaskRunner(cfg).run() == EXIT_REFUSED
    res = result(src)
    assert res["outcome"] == "refused" and res["simulated"] is True


def test_prod_refuses_unverified_build_and_non_prod_marks_simulated(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    cfg = make_cfg(tmp_path, src, {"steps": []}, build_info_path=tmp_path / "missing.json")
    assert TaskRunner(cfg).run() == EXIT_REFUSED
    src2 = make_run(tmp_path / "b")
    cfg2 = make_cfg(
        tmp_path / "b",
        src2,
        {"steps": [["print", "ok"]]},
        environment="dev",
        build_info_path=make_build(tmp_path, real=False),
    )
    assert TaskRunner(cfg2).run() == 0
    assert result(src2)["simulated"] is True  # never dataset-eligible


@pytest.mark.parametrize("fluxes_flag", [None, False])
def test_missing_or_false_diagnostics_fluxes_is_refused_on_real_run(
    tmp_path: Path, fluxes_flag: bool | None
) -> None:
    src = make_run(tmp_path, fluxes=fluxes_flag)
    cfg = make_cfg(tmp_path, src, {"steps": [["print", "should not run"]]})
    assert TaskRunner(cfg).run() == EXIT_REFUSED
    assert not (src / "gx.log").exists()
    res = result(src)
    assert res["outcome"] == "refused"
    assert "[Diagnostics] fluxes = true" in res["detail"]


def test_staging_skips_babysitter_bookkeeping(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    for name in (
        LEDGER_FILENAME,
        "gx.log.prev",
        "gx.log.1",
        "gx.log",
        "diagnostics_metrics.json",
        "ISSUE.md",
        TASK_RESULT_FILENAME,
    ):
        (src / name).write_text("x", encoding="utf-8")
    (src / "geometry.eik.nc").write_bytes(hdf5_bytes(256))
    runner = TaskRunner(make_cfg(tmp_path, src, {"steps": []}))
    work = runner.stage()
    assert sorted(p.name for p in work.iterdir()) == ["geometry.eik.nc", "gx.in"]
    runner.cleanup()


def test_sigterm_publishes_log_and_checkpoint_then_stops_gx(tmp_path: Path) -> None:
    src = make_run(tmp_path)
    script = tmp_path / "fake_gx_exec"
    script.write_text(f"#!{sys.executable}\n" + FAKE_GX, encoding="utf-8")
    script.chmod(0o755)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {"steps": [["print", "Time = 1.0"], ["write", "gx.restart.nc", 2048], ["sleep", 60]]}
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "FAKE_GX_PLAN": str(plan_path),
        "GX_BUILD_INFO": str(make_build(tmp_path)),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    env.pop("GX_SIMULATE_RUN", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gcp_exec.task_runner",
            "--src-dir",
            str(src),
            "--attempt-id",
            ATTEMPT,
            "--gx-binary",
            str(script),
            "--work-root",
            str(tmp_path / "work"),
            "--sync-interval",
            "0.1",
            "--stable-seconds",
            "0",
            "--environment",
            "prod",
        ],
        env=env,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (src / "gx.restart.nc").exists():
        time.sleep(0.05)
    assert (src / "gx.restart.nc").exists(), "periodic sync never published the checkpoint"
    proc.send_signal(signal.SIGTERM)
    rc = proc.wait(timeout=30)
    assert rc == EXIT_TERMINATED
    res = result(src)
    assert res["outcome"] == "terminated" and res["signal"] == "SIGTERM"
    assert "Time = 1.0" in (src / "gx.log").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GX_SIMULATE_RUN", raising=False)


def test_gx_task_console_script_is_declared_for_the_image() -> None:
    import tomllib

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["scripts"]["gx-task"] == "gcp_exec.task_runner:main"
