"""Slurm wrapper tests (HPC-05, QA-04): states, aggregation, lag, idempotent submission."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hpc_core import slurm as slurm_mod
from hpc_core.protocol import KNOWN_JOB_STATES
from hpc_core.slurm import (
    SlurmBackend,
    SlurmCancelError,
    SlurmQueryError,
    SlurmSubmitError,
    _parse_slurm_elapsed,
    is_hopeless_pending_reason,
    normalize_slurm_state,
    slurm_job_name_for_submission,
)

MY_UID = 5000
SID = "bsit-gx_run_01-deadbeef-1"
Handler = Callable[[list[str]], Any]


def _cp(argv: list[str], stdout: str = "", stderr: str = "", rc: int = 0) -> Any:
    return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)


class FakeSlurm:
    """Replaces subprocess.run; dispatches on argv[0] to queued responses (in call order)."""

    def __init__(self, **responses: list[Any]) -> None:
        self.responses = {cmd: list(items) for cmd, items in responses.items()}
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        assert isinstance(argv, list), "argv must be a list (never a shell string)"
        assert not kwargs.get("shell"), "shell=True is forbidden"
        assert kwargs.get("timeout"), "every Slurm call needs a timeout"
        self.calls.append(list(argv))
        queue = self.responses.get(argv[0])
        assert queue, f"unexpected call {argv}"
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(argv)
        stdout, stderr, rc = item if isinstance(item, tuple) else (item, "", 0)
        return _cp(argv, stdout, stderr, rc)

    def commands(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeSlurm]:
    monkeypatch.setattr(slurm_mod, "_current_uid", lambda: MY_UID)

    def install(**responses: list[Any]) -> FakeSlurm:
        runner = FakeSlurm(**responses)
        monkeypatch.setattr(slurm_mod.subprocess, "run", runner)
        return runner

    return install


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gx_run_01"
    d.mkdir()
    (d / "submit.slurm").write_text("#!/bin/bash\n#SBATCH -J mine\nibrun gx gx.in\n")
    return d


# --------------------------------------------------------------------------- states
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RUNNING", "RUNNING"),
        ("R", "RUNNING"),
        ("PENDING", "PENDING"),
        ("PD", "PENDING"),
        ("CONFIGURING", "CONFIGURING"),
        ("COMPLETING", "COMPLETING"),
        ("SUSPENDED", "SUSPENDED"),
        ("STOPPED", "STOPPED"),
        ("ST", "STOPPED"),
        ("REQUEUED", "REQUEUED"),
        ("REQUEUE_FED", "REQUEUED"),
        ("REQUEUE_HOLD", "PENDING"),
        ("SPECIAL_EXIT", "PENDING"),
        ("RESV_DEL_HOLD", "PENDING"),
        ("UPDATE_DB", "CONFIGURING"),
        ("POWER_UP_NODE", "CONFIGURING"),
        ("RESIZING", "RESIZING"),
        ("SIGNALING", "SIGNALING"),
        ("STAGE_OUT", "STAGE_OUT"),
        ("REVOKED", "REVOKED"),
        ("COMPLETED", "COMPLETED"),
        ("CD", "COMPLETED"),
        ("TIMEOUT", "TIMEOUT"),
        ("TO", "TIMEOUT"),
        ("FAILED", "FAILED"),
        ("CANCELLED", "CANCELLED"),
        ("CANCELLED+", "CANCELLED"),
        ("CANCELLED by 0", "CANCELLED"),
        ("OUT_OF_MEMORY", "OUT_OF_MEMORY"),
        ("OOM", "OUT_OF_MEMORY"),
        ("NODE_FAIL", "NODE_FAIL"),
        ("PREEMPTED", "PREEMPTED"),
        ("PR", "PREEMPTED"),
        ("BOOT_FAIL", "BOOT_FAIL"),
        ("DEADLINE", "DEADLINE"),
        ("", "UNKNOWN"),
        ("SOMETHING_NEW", "UNKNOWN"),
    ],
)
def test_normalize_every_slurm_state(raw: str, expected: str) -> None:
    assert normalize_slurm_state(raw) == expected
    assert normalize_slurm_state(raw) in KNOWN_JOB_STATES


@pytest.mark.parametrize(
    "reason",
    [
        "JobHeldUser",
        "JobHeldAdmin",
        "JobHoldMaxRequeue",
        "launch failed requeued held",
        "InvalidAccount",
        "InvalidQOS",
        "PartitionConfig",
        "PartitionTimeLimit",
        "BadConstraints",
        "DependencyNeverSatisfied",
        "AssocGrpCPUMinutesLimit",
        "AssocGrpBillingMinutes",
        "QOSGrpGRESMinutes",
        "AssocGrpWallLimit",
        "QOSMaxWallDurationPerJobLimit",
        "QOSMaxNodePerJobLimit",
        "AssocMaxCpuPerJobLimit",
        "QOSMinCpuNotSatisfied",
        "(JobHeldUser)",
    ],
)
def test_hopeless_pending_reasons_are_blocked(reason: str) -> None:
    assert is_hopeless_pending_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "",
        "None",
        "Priority",
        "Resources",
        "Dependency",
        "BeginTime",
        "AssocGrpCpuLimit",
        "AssocGrpJobsLimit",
        "AssocGrpCPURunMinutesLimit",
        "QOSMaxJobsPerUserLimit",
        "QOSUsageThreshold",
        "ReqNodeNotAvail, Reserved for maintenance",
        "PartitionDown",
    ],
)
def test_transient_pending_reasons_are_not_blocked(reason: str | None) -> None:
    assert not is_hopeless_pending_reason(reason)


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("", 0.0),
        ("INVALID", 0.0),
        ("UNLIMITED", 0.0),
        ("x-1:02", 0.0),
        ("nan:nan", 0.0),
        ("5:03", 303.0),
        ("1:05:03", 3903.0),
        ("2-01:05:03", 2 * 86400 + 3903.0),
        ("00:00:01.500", 1.5),
    ],
)
def test_elapsed_parse_never_raises(raw: str, seconds: float) -> None:
    assert _parse_slurm_elapsed(raw) == pytest.approx(seconds)


# --------------------------------------------------------------------------- query_job
def test_squeue_running_and_pending_reason(fake: Callable[..., FakeSlurm]) -> None:
    runner = fake(squeue=["RUNNING|1:02:03|None\n", "PENDING|0:00|Priority\n"])
    backend = SlurmBackend()
    running = backend.query_job("123")
    assert (running.state, running.elapsed_seconds, running.blocked) == ("RUNNING", 3723.0, False)
    pending = backend.query_job("123")
    assert (pending.state, pending.reason, pending.blocked) == ("PENDING", "Priority", False)
    assert runner.calls[0] == ["squeue", "--jobs=123", "-h", "-o", "%T|%M|%r"]
    assert runner.commands() == ["squeue", "squeue"]  # non-terminal: sacct not needed


@pytest.mark.parametrize(
    "line",
    [
        "PENDING|0:00|JobHeldUser",
        "PENDING|0:00|AssocGrpBillingMinutes",
        "PENDING|0:00|launch failed requeued held",
        "REQUEUE_HOLD|0:00|None",
        "SPECIAL_EXIT|0:00|None",
        "RESV_DEL_HOLD|0:00|None",
    ],
)
def test_held_or_hopeless_pending_job_is_blocked(fake: Callable[..., FakeSlurm], line: str) -> None:
    fake(squeue=[line + "\n"])
    status = SlurmBackend().query_job("123")
    assert status.state == "PENDING"
    assert status.blocked is True


def test_terminal_in_squeue_is_confirmed_by_sacct(fake: Callable[..., FakeSlurm]) -> None:
    runner = fake(squeue=["COMPLETED|10:00|None\n"], sacct=["123|COMPLETED|00:10:00|0:0\n"])
    status = SlurmBackend().query_job("123")
    assert (status.state, status.exit_code, status.elapsed_seconds) == ("COMPLETED", "0:0", 600.0)
    assert runner.calls[1] == [
        "sacct",
        "--jobs=123",
        "-n",
        "-X",
        "-P",
        "-o",
        "JobID,State,Elapsed,ExitCode",
    ]


def test_cancelled_by_other_uid_is_external(fake: Callable[..., FakeSlurm]) -> None:
    fake(
        squeue=[("", "slurm_load_jobs error: Invalid job id specified", 1)],
        sacct=["123|CANCELLED by 0|00:05:00|0:15\n"],
    )
    status = SlurmBackend().query_job("123")
    assert status.state == "CANCELLED"
    assert status.cancelled_externally is True
    assert status.reason == "cancelled by uid=0"


def test_cancelled_by_own_uid_is_not_external(fake: Callable[..., FakeSlurm]) -> None:
    fake(squeue=[""], sacct=[f"123|CANCELLED by {MY_UID}|00:05:00|0:15\n"])
    status = SlurmBackend().query_job("123")
    assert status.state == "CANCELLED"
    assert status.cancelled_externally is False
    assert status.reason == f"cancelled by uid={MY_UID}"


def test_unattributed_cancel_in_squeue_waits_for_sacct(fake: Callable[..., FakeSlurm]) -> None:
    fake(squeue=["CANCELLED|1:00|None\n"], sacct=[""])
    status = SlurmBackend().query_job("123")
    assert status.state == "UNKNOWN"
    assert "canceller" in (status.reason or "")


def test_other_terminal_in_squeue_without_sacct_record(fake: Callable[..., FakeSlurm]) -> None:
    fake(squeue=["TIMEOUT|2:00:00|TimeLimit\n"], sacct=[""])
    status = SlurmBackend().query_job("123")
    assert status.state == "TIMEOUT"
    assert status.exit_code == ""


def test_sacct_lag_reports_unknown(fake: Callable[..., FakeSlurm]) -> None:
    fake(squeue=[""], sacct=[""])
    status = SlurmBackend().query_job("123")
    assert status.state == "UNKNOWN"
    assert "accounting lag" in (status.reason or "")


def test_squeue_down_sacct_answers(fake: Callable[..., FakeSlurm]) -> None:
    fake(
        squeue=[subprocess.TimeoutExpired(["squeue"], 30)],
        sacct=["123|RUNNING|01:00:00|0:0\n"],
    )
    assert SlurmBackend().query_job("123").state == "RUNNING"


def test_both_channels_failing_raises(fake: Callable[..., FakeSlurm]) -> None:
    fake(
        squeue=[subprocess.TimeoutExpired(["squeue"], 30)],
        sacct=[("", "sacct: error: slurmdbd: Connection refused", 1)],
    )
    with pytest.raises(SlurmQueryError, match="both failed"):
        SlurmBackend().query_job("123")


@pytest.mark.parametrize(
    ("sacct_out", "expected", "external"),
    [
        ("123_1|COMPLETED|00:10:00|0:0\n123_2|TIMEOUT|02:00:00|0:0\n", "TIMEOUT", False),
        ("123_1|COMPLETED|00:10:00|0:0\n123_2|COMPLETED|00:11:00|0:0\n", "COMPLETED", False),
        ("123_1|FAILED|00:01:00|1:0\n123_2|OUT_OF_MEMORY|00:01:00|0:125\n", "OUT_OF_MEMORY", False),
        ("123_1|FAILED|00:01:00|1:0\n123_2|CANCELLED by 0|00:01:00|0:15\n", "CANCELLED", True),
        ("123_1|COMPLETED|00:10:00|0:0\n123_2|PENDING|00:00:00|0:0\n", "PENDING", False),
        ("123+0|COMPLETED|00:10:00|0:0\n123+1|FAILED|00:10:00|2:0\n", "FAILED", False),
        (
            "123|FAILED|00:10:00|1:0\n123.batch|FAILED|00:10:00|1:0\n123.0|COMPLETED|0:1|0:0\n",
            "FAILED",
            False,
        ),
    ],
)
def test_multi_row_sacct_aggregation(
    fake: Callable[..., FakeSlurm], sacct_out: str, expected: str, external: bool
) -> None:
    fake(squeue=[""], sacct=[sacct_out])
    status = SlurmBackend().query_job("123")
    assert status.state == expected
    assert status.cancelled_externally is external


def test_array_with_a_running_task_is_running(fake: Callable[..., FakeSlurm]) -> None:
    fake(squeue=["PENDING|0:00|Resources\nRUNNING|5:00|None\nCOMPLETED|9:00|None\n"])
    status = SlurmBackend().query_job("123")
    assert (status.state, status.elapsed_seconds, status.blocked) == ("RUNNING", 540.0, False)


def test_array_with_held_task_and_no_running_task_is_blocked(
    fake: Callable[..., FakeSlurm],
) -> None:
    fake(squeue=["PENDING|0:00|Priority\nPENDING|0:00|JobHeldAdmin\n"])
    status = SlurmBackend().query_job("123")
    assert (status.state, status.blocked, status.reason) == ("PENDING", True, "JobHeldAdmin")


@pytest.mark.parametrize("bad", ["--me", "-u", "123;rm -rf ~", "abc", "dryrun-5", "", "12 3"])
def test_job_id_validation_blocks_option_injection(
    fake: Callable[..., FakeSlurm], bad: str
) -> None:
    runner = fake()
    backend = SlurmBackend()
    with pytest.raises(ValueError):
        backend.query_job(bad)
    with pytest.raises(ValueError):
        backend.scancel(bad)
    assert runner.calls == []


def test_dry_run_accepts_fake_ids_without_subprocess(fake: Callable[..., FakeSlurm]) -> None:
    runner = fake()
    backend = SlurmBackend(dry_run=True)
    assert backend.query_job("dryrun-900001").state == "RUNNING"
    backend.scancel("dryrun-900001")
    assert runner.calls == []


# --------------------------------------------------------------------------- scancel
def test_scancel_of_finished_job_is_success(fake: Callable[..., FakeSlurm]) -> None:
    fake(
        scancel=[
            (
                "",
                "scancel: error: Kill job error on job id 123: Job/step already completing or completed",
                1,
            ),
            ("", "scancel: error: Kill job error on job id 124: Invalid job id specified", 1),
            ("", "", 0),
        ]
    )
    backend = SlurmBackend()
    backend.scancel("123")
    backend.scancel("124")
    backend.scancel("125")


def test_scancel_other_error_raises(fake: Callable[..., FakeSlurm]) -> None:
    fake(scancel=[("", "scancel: error: Unable to contact slurm controller", 1)])
    with pytest.raises(SlurmCancelError):
        SlurmBackend().scancel("123")


# --------------------------------------------------------------------------- submission
def test_job_name_mapping_is_deterministic_and_site_safe() -> None:
    name = slurm_job_name_for_submission(SID)
    assert name == slurm_job_name_for_submission(SID)
    assert len(name) <= 57
    assert all(ch.isalnum() or ch in "_-" for ch in name)
    longer = "bsit-" + "a" * 120 + "-cafebabe-2"
    other = "bsit-" + "a" * 120 + "-cafebabe-3"
    assert slurm_job_name_for_submission(longer) != slurm_job_name_for_submission(other)
    for bad in ("", "-leading-dash", "a/b", "x" * 161, "white space"):
        with pytest.raises(ValueError):
            slurm_job_name_for_submission(bad)


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("12345\n", "", "12345"),
        ("12345;vista\n", "", "12345"),
        (
            "--> Verifying valid submit host (login1)...OK\n--> Verifying valid jobname...OK\n4242\n",
            "",
            "4242",
        ),
        ("Submitted batch job 777\n", "", "777"),
        ("", "sbatch: Submitted batch job 778\n", "778"),
    ],
)
def test_sbatch_parsable_output_and_regex_fallback(
    fake: Callable[..., FakeSlurm], run_dir: Path, stdout: str, stderr: str, expected: str
) -> None:
    runner = fake(sbatch=[(stdout, stderr, 0)])
    assert SlurmBackend().submit(run_dir, solver="GX") == expected
    argv = runner.calls[0]
    assert argv[:2] == ["sbatch", "--parsable"]
    assert argv[-1].endswith("submit.slurm")
    assert not any(a.startswith("--job-name") for a in argv)


def test_unparseable_sbatch_output_raises(fake: Callable[..., FakeSlurm], run_dir: Path) -> None:
    fake(sbatch=[("something unexpected\n", "", 0)])
    with pytest.raises(SlurmSubmitError) as info:
        SlurmBackend().submit(run_dir)
    assert info.value.outcome_unknown is True


def test_submit_with_id_checks_for_duplicates_then_names_the_job(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    runner = fake(squeue=[""], sacct=[""], sbatch=["5551\n"])
    backend = SlurmBackend()
    assert backend.capabilities.idempotent_submit is True
    assert backend.submit(run_dir, submission_id=SID) == "5551"
    name = slurm_job_name_for_submission(SID)
    assert runner.commands() == ["squeue", "sacct", "sbatch"]
    assert runner.calls[0] == [
        "squeue",
        f"--user={MY_UID}",
        f"--name={name}",
        "--states=all",
        "-h",
        "-o",
        "%i",
    ]
    assert f"--name={name}" in runner.calls[1]
    assert "--starttime=now-30days" in runner.calls[1]
    assert f"--job-name={name}" in runner.calls[2]


def test_submit_adopts_existing_job_without_sbatch(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    runner = fake(squeue=["555\n"])
    assert SlurmBackend().submit(run_dir, submission_id=SID) == "555"
    assert runner.commands() == ["squeue"]


def test_submit_timeout_adopts_job_found_afterwards(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    runner = fake(
        squeue=["", "999\n"],
        sacct=[""],
        sbatch=[subprocess.TimeoutExpired(["sbatch"], 30)],
    )
    assert SlurmBackend().submit(run_dir, submission_id=SID) == "999"
    assert runner.commands() == ["squeue", "sacct", "sbatch", "squeue"]


def test_submit_timeout_without_job_reports_unknown_outcome(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    fake(
        squeue=["", ""],
        sacct=["", ""],
        sbatch=[subprocess.TimeoutExpired(["sbatch"], 30)],
    )
    with pytest.raises(SlurmSubmitError) as info:
        SlurmBackend().submit(run_dir, submission_id=SID)
    assert info.value.outcome_unknown is True


def test_socket_timeout_error_is_reconciled(fake: Callable[..., FakeSlurm], run_dir: Path) -> None:
    fake(
        squeue=["", ""],
        sacct=["", "1001\n"],
        sbatch=[
            (
                "",
                "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation",
                1,
            )
        ],
    )
    assert SlurmBackend().submit(run_dir, submission_id=SID) == "1001"


def test_definitive_sbatch_rejection_is_not_ambiguous(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    runner = fake(
        squeue=[""],
        sacct=[""],
        sbatch=[
            ("", "sbatch: error: Invalid account or account/partition combination specified", 1)
        ],
    )
    with pytest.raises(SlurmSubmitError) as info:
        SlurmBackend().submit(run_dir, submission_id=SID)
    assert info.value.outcome_unknown is False
    assert runner.commands() == ["squeue", "sacct", "sbatch"]  # no follow-up lookup needed


def test_submit_refuses_when_duplicate_check_cannot_run(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    runner = fake(
        squeue=[("", "slurm_load_jobs error: Unable to contact slurm controller", 1)],
        sacct=[("", "sacct: error: Problem talking to the database", 1)],
    )
    with pytest.raises(SlurmSubmitError) as info:
        SlurmBackend().submit(run_dir, submission_id=SID)
    assert info.value.outcome_unknown is True
    assert "sbatch" not in runner.commands()


def test_find_submission_duplicate_jobs_raise(fake: Callable[..., FakeSlurm]) -> None:
    fake(squeue=["100\n101\n"])
    with pytest.raises(SlurmQueryError, match="Several"):
        SlurmBackend().find_submission(SID)


def test_find_submission_falls_back_to_sacct_and_collapses_arrays(
    fake: Callable[..., FakeSlurm],
) -> None:
    fake(squeue=["", "300_1\n300_[2-4]\n"], sacct=["200\n"])
    backend = SlurmBackend()
    assert backend.find_submission(SID) == "200"
    assert backend.find_submission(SID) == "300"


def test_find_submission_cannot_prove_absence_when_squeue_fails(
    fake: Callable[..., FakeSlurm],
) -> None:
    fake(squeue=[subprocess.TimeoutExpired(["squeue"], 30)], sacct=[""])
    with pytest.raises(SlurmQueryError, match="cannot prove"):
        SlurmBackend().find_submission(SID)


def test_dry_run_submission_is_idempotent_and_never_calls_slurm(
    fake: Callable[..., FakeSlurm], run_dir: Path
) -> None:
    runner = fake()
    backend = SlurmBackend(dry_run=True)
    first = backend.submit(run_dir, submission_id=SID)
    assert first.startswith("dryrun-")
    assert backend.submit(run_dir, submission_id=SID) == first
    assert backend.find_submission(SID) == first
    assert backend.find_submission("bsit-other-00000000-1") is None
    assert runner.calls == []
