"""In-job GX task runner (`gx-task`) shared by the Cloud Batch and GKE backends.

Replaces the inline bash scripts that used to stage, run and sync GX. Those scripts had
four defects (GCP-03, GCP-C01..C04 in docs/review/FIXLOG-GCP.md):

* the periodic sync copied ``*.restart.nc`` while GX was rewriting it, and published the torn
  copy as *the* checkpoint;
* ``trap ... TERM`` never fired on Spot preemption, because bash defers traps while a
  foreground child (``gx``) runs, and ``*.out.nc`` was only copied at exit, so it was lost;
* a cancelled or superseded job kept writing ``gx.log`` and checkpoints into the run
  directory after its successor had started (false failures, stale checkpoints);
* the GKE variant wrote ``gx.out``/``gx.err`` instead of ``gx.log`` and staged only ``gx.in``
  plus the restart file, so watchdog diagnostics and geometry files were missing.

What this runner guarantees:

* **Fencing**: it writes into the run directory only while ``.gcp_active_attempt`` names its
  own attempt id (written by the babysitter before the job is created). A superseded job
  stops GX and exits without writing anything.
* **Checkpoint integrity**: a restart file is published only when it has been stable for
  ``stable_seconds`` and passes :mod:`gcp_exec.checkpoint` validation. Publishing goes through
  a temp file plus ``os.replace``, and the previous generation is kept as
  ``<stem>.prev.restart.nc``.
* **Resume validation**: with ``[Restart] restart = true``, the checkpoint must validate. The
  runner falls back to the previous generation, and otherwise fails closed (exit 65). It never
  silently cold-starts a run that asked to restart.
* **Preemption**: GX runs as a child process, SIGTERM/SIGINT are handled, the log and the newest
  stable checkpoint are published, then GX is stopped.
* **Provenance**: ``task_result.json`` records the outcome, exit code, GX build (from
  ``$GX_BUILD_INFO``) and the restart decision. gx-diagnose turns it into
  ``scalar_metrics.provenance_ok``, which the dataset writer requires for GCP runs.
* **No simulated output**: ``GX_SIMULATE_RUN`` is always refused (this image has no simulator),
  and in ``prod`` an unverified GX build is refused (exit 78).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gcp_exec.checkpoint import netcdf_file_problem
from gcp_exec.submission import FENCE_FILENAME, read_fence

EXIT_CHECKPOINT_INVALID = 65
EXIT_OUTPUT_SYNC_FAILED = 74
EXIT_SUPERSEDED = 75
EXIT_REFUSED = 78
EXIT_TERMINATED = 143

TASK_RESULT_FILENAME = "task_result.json"
TASK_RESULT_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,200}$")
_TRUTHY = ("1", "true", "yes", "on")

# Never staged into the job: babysitter bookkeeping, previous logs and diagnostics.
_STAGE_SKIP_NAMES = {TASK_RESULT_FILENAME, "ISSUE.md", "ISSUE.resolved.md"}
_STAGE_SKIP_RE = re.compile(
    r"(^\.)|(\.tmp$)|(\.bak$)|(\.prev$)|(\.log$)|(\.log\.)|(^diagnostics_metrics.*\.json$)"
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def prev_checkpoint_name(name: str) -> str:
    """``gx.restart.nc`` -> ``gx.prev.restart.nc`` (still matches ``*.restart.nc``)."""
    suffix = ".restart.nc"
    if name.endswith(suffix):
        return name[: -len(suffix)] + ".prev" + suffix
    return name + ".prev"


def _safe_filename(raw: Any, default: str) -> str | None:
    """Restart file names must be plain names inside the run directory (no paths)."""
    name = default if raw in (None, "") else str(raw)
    return name if _SAFE_NAME_RE.match(name) and ".." not in name else None


def _fsync_copy(src: Path, dst: Path) -> None:
    with src.open("rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        fout.flush()
        os.fsync(fout.fileno())


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


@dataclass
class TaskConfig:
    src_dir: Path
    attempt_id: str
    input_name: str = "gx.in"
    gx_command: list[str] = field(default_factory=lambda: ["gx"])
    work_root: Path = Path(tempfile.gettempdir())
    sync_interval_s: float = 60.0
    output_sync_interval_s: float = 600.0
    stable_seconds: float = 20.0
    term_budget_s: float = 20.0
    environment: str = "prod"
    simulate_requested: bool = False
    build_info_path: Path | None = None
    enforce_fence: bool = True
    poll_s: float = 1.0


class TaskRunner:
    """Stage -> validate restart -> run GX -> fenced, validated publishing."""

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg
        self.src = Path(cfg.src_dir)
        self.work: Path | None = None
        self._stop_signal: int | None = None
        self._published: dict[str, tuple[int, int]] = {}
        self._last_output_publish: dict[str, float] = {}
        self._staged: dict[str, tuple[int, int]] = {}
        self.result: dict[str, Any] = {
            "schema_version": TASK_RESULT_SCHEMA_VERSION,
            "attempt_id": cfg.attempt_id,
            "environment": cfg.environment,
            "outcome": None,
            "gx_exit_code": None,
            "signal": None,
            "simulated": bool(cfg.simulate_requested),
            "gx_build": None,
            "restart": {"requested": False, "used": None, "fallback_to_previous": False},
            "outputs_published": [],
            "publish_failures": [],
            "detail": None,
            "started_at_utc": utc_now_iso(),
            "finished_at_utc": None,
        }
        self.restart_to_name = "gx.restart.nc"

    # ------------------------------------------------------------------ fencing
    def fence_ok(self) -> bool:
        if not self.cfg.enforce_fence:
            return True
        return read_fence(self.src) == self.cfg.attempt_id

    # ------------------------------------------------------------------ signals
    def _on_signal(self, signum: int, _frame: Any) -> None:
        self._stop_signal = signum

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    # ------------------------------------------------------------------ provenance
    def _load_build_info(self) -> dict[str, Any] | None:
        path = self.cfg.build_info_path
        if path is None:
            return None
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def build_verified(info: dict[str, Any] | None) -> bool:
        return bool(
            info
            and info.get("real_gx") is True
            and isinstance(info.get("gx_commit"), str)
            and _COMMIT_RE.match(info["gx_commit"])
        )

    # ------------------------------------------------------------------ results
    def write_result(self, outcome: str, *, detail: str | None = None) -> None:
        self.result["outcome"] = outcome
        if detail:
            self.result["detail"] = detail[:500]
        self.result["finished_at_utc"] = utc_now_iso()
        if not self.fence_ok():
            return  # never write into a run directory owned by another attempt
        text = json.dumps(self.result, indent=2, sort_keys=True) + "\n"
        tmp = self.src / f".{TASK_RESULT_FILENAME}.tmp-{os.getpid()}"
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.src / TASK_RESULT_FILENAME)
        except OSError as exc:
            print(f"gx-task: cannot write {TASK_RESULT_FILENAME}: {exc}", file=sys.stderr)
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ staging
    def stage(self) -> Path:
        self.cfg.work_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="gx_work_", dir=str(self.cfg.work_root)))

        def _ignore(_dir: str, names: list[str]) -> set[str]:
            return {n for n in names if n in _STAGE_SKIP_NAMES or _STAGE_SKIP_RE.search(n)}

        shutil.copytree(self.src, work, ignore=_ignore, dirs_exist_ok=True)
        for p in work.iterdir():
            key = _stat_key(p)
            if p.is_file() and key:
                self._staged[p.name] = key
        self.work = work
        return work

    def read_input(self) -> dict[str, Any]:
        assert self.work is not None
        with (self.work / self.cfg.input_name).open("rb") as fh:
            return tomllib.load(fh)

    def validate_diagnostics(self, doc: dict[str, Any]) -> str | None:
        """Ensure ``[Diagnostics] fluxes = true`` is set on real GX runs (CLAUDE-P0-2)."""
        if self.result.get("simulated"):
            return None
        section = doc.get("Diagnostics") or doc.get("diagnostics") or {}
        if not isinstance(section, dict) or section.get("fluxes") is not True:
            return (
                f"{self.cfg.input_name} must set `[Diagnostics] fluxes = true` "
                "(GX defaults `fluxes` to false, which suppresses HeatFlux/ParticleFlux output)"
            )
        return None

    def prepare_restart(self, doc: dict[str, Any]) -> str | None:
        """Validate the checkpoint requested by ``[Restart]``. Returns an error or None."""
        assert self.work is not None
        section = doc.get("Restart") or doc.get("restart") or {}
        if not isinstance(section, dict):
            section = {}
        stem = Path(self.cfg.input_name).stem
        from_name = _safe_filename(section.get("restart_from_file"), f"{stem}.restart.nc")
        to_name = _safe_filename(section.get("restart_to_file"), f"{stem}.restart.nc")
        if from_name is None or to_name is None:
            return "unsafe restart_from_file/restart_to_file (must be a plain file name)"
        self.restart_to_name = to_name
        requested = section.get("restart") is True
        self.result["restart"]["requested"] = requested
        if not requested:
            return None

        primary = self.work / from_name
        prev = self.work / prev_checkpoint_name(from_name)
        problems: dict[str, str] = {}
        if primary.exists():
            problem = netcdf_file_problem(primary)
            if problem is None:
                self.result["restart"]["used"] = from_name
                return None
            problems[from_name] = problem
        if prev.exists():
            problem = netcdf_file_problem(prev)
            if problem is None:
                shutil.copyfile(prev, primary)
                self.result["restart"].update(used=prev.name, fallback_to_previous=True)
                return None
            problems[prev.name] = problem
        self.result["restart"]["problems"] = problems
        if not problems:
            return f"restart requested but no checkpoint ({from_name}) exists"
        return f"restart requested but no valid checkpoint: {problems}"

    # ------------------------------------------------------------------ publishing
    def _publish(self, local: Path, dest_name: str, *, rotate_prev: bool = False) -> bool:
        """Copy ``local`` into the run directory atomically; False if it changed mid-copy."""
        before = _stat_key(local)
        if before is None:
            return False
        tmp = self.src / f".{dest_name}.tmp-{os.getpid()}"
        try:
            _fsync_copy(local, tmp)
            if _stat_key(local) != before:
                tmp.unlink(missing_ok=True)
                return False
            if not self.fence_ok():
                tmp.unlink(missing_ok=True)
                return False
            dest = self.src / dest_name
            if rotate_prev and dest.exists():
                os.replace(dest, self.src / prev_checkpoint_name(dest_name))
            os.replace(tmp, dest)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        self._published[dest_name] = before
        if dest_name not in self.result["outputs_published"]:
            self.result["outputs_published"].append(dest_name)
        return True

    def _is_stable(self, path: Path, stable_seconds: float) -> bool:
        try:
            return (time.time() - path.stat().st_mtime) >= stable_seconds
        except OSError:
            return False

    def _changed(self, path: Path) -> bool:
        key = _stat_key(path)
        if key is None:
            return False
        name = path.name
        if self._published.get(name) == key:
            return False
        return self._staged.get(name) != key

    def publish_log(self) -> None:
        assert self.work is not None
        log = self.work / "gx.log"
        if log.exists() and self._changed(log):
            self._publish(log, "gx.log")

    def publish_checkpoint(self, *, stable_seconds: float) -> None:
        assert self.work is not None
        ckpt = self.work / self.restart_to_name
        if not ckpt.exists() or not self._changed(ckpt):
            return
        if not self._is_stable(ckpt, stable_seconds):
            return
        if netcdf_file_problem(ckpt) is not None:
            return  # torn or still being written: keep the last published generation
        self._publish(ckpt, self.restart_to_name, rotate_prev=True)

    def publish_outputs(self, *, stable_seconds: float, force: bool) -> None:
        assert self.work is not None
        skip = {self.restart_to_name, prev_checkpoint_name(self.restart_to_name)}
        now = time.monotonic()
        for path in sorted(self.work.glob("*.nc")):
            name = path.name
            if name in skip or name.endswith(".restart.nc") or not self._changed(path):
                continue
            last = self._last_output_publish.get(name)
            if not force and last is not None and now - last < self.cfg.output_sync_interval_s:
                continue
            if not self._is_stable(path, stable_seconds) or netcdf_file_problem(path):
                continue
            if self._publish(path, name):
                self._last_output_publish[name] = now

    def _try(self, label: str, fn: Any, *args: Any, attempts: int = 3, **kwargs: Any) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                fn(*args, **kwargs)
                return True
            except OSError as exc:
                print(f"gx-task: {label} failed ({attempt}/{attempts}): {exc}", file=sys.stderr)
                time.sleep(min(2.0, 0.5 * attempt))
        if label not in self.result["publish_failures"]:
            self.result["publish_failures"].append(label)
        return False

    # ------------------------------------------------------------------ process control
    @staticmethod
    def _stop_process(proc: subprocess.Popen[bytes], grace_s: float) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=max(0.5, grace_s))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=10)

    # ------------------------------------------------------------------ main
    def run(self) -> int:
        cfg = self.cfg
        if not (self.src / cfg.input_name).is_file():
            print(f"gx-task: missing {cfg.input_name} in {self.src}", file=sys.stderr)
            return EXIT_REFUSED
        if not self.fence_ok():
            print(
                f"gx-task: attempt {cfg.attempt_id} is superseded "
                f"({FENCE_FILENAME}={read_fence(self.src)!r}); exiting without writing",
                file=sys.stderr,
            )
            return EXIT_SUPERSEDED

        build = self._load_build_info()
        self.result["gx_build"] = build
        if not self.build_verified(build):
            self.result["simulated"] = True  # provenance cannot be proven: never dataset-eligible
        if cfg.simulate_requested:
            self.write_result(
                "refused", detail="GX_SIMULATE_RUN is not supported: refusing non-physical output"
            )
            return EXIT_REFUSED
        if cfg.environment == "prod" and not self.build_verified(build):
            self.write_result(
                "refused",
                detail="prod requires a verified GX build ($GX_BUILD_INFO with real_gx and a pinned commit)",
            )
            return EXIT_REFUSED

        try:
            self.stage()
            doc = self.read_input()
        except (OSError, tomllib.TOMLDecodeError) as exc:
            self.write_result("refused", detail=f"staging/input error: {exc}")
            return EXIT_REFUSED

        diag_error = self.validate_diagnostics(doc)
        if diag_error:
            self.write_result("refused", detail=diag_error)
            return EXIT_REFUSED

        restart_error = self.prepare_restart(doc)
        if restart_error:
            self.write_result("checkpoint_invalid", detail=restart_error)
            return EXIT_CHECKPOINT_INVALID

        assert self.work is not None
        env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"}
        with (self.work / "gx.log").open("ab") as log_fh:
            proc = subprocess.Popen(
                [*cfg.gx_command, cfg.input_name],
                cwd=str(self.work),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        last_sync = time.monotonic()
        while proc.poll() is None:
            if self._stop_signal is not None:
                return self._terminate(proc)
            if time.monotonic() - last_sync >= cfg.sync_interval_s:
                last_sync = time.monotonic()
                if not self.fence_ok():
                    self._stop_process(proc, grace_s=5.0)
                    print("gx-task: superseded during run; stopped GX", file=sys.stderr)
                    return EXIT_SUPERSEDED
                self._try("gx.log", self.publish_log)
                self._try("checkpoint", self.publish_checkpoint, stable_seconds=cfg.stable_seconds)
                self._try(
                    "outputs",
                    self.publish_outputs,
                    stable_seconds=cfg.stable_seconds,
                    force=False,
                )
            time.sleep(cfg.poll_s)

        rc = int(proc.returncode)
        self.result["gx_exit_code"] = rc
        if not self.fence_ok():
            return EXIT_SUPERSEDED
        ok = self._try("gx.log", self.publish_log)
        ok &= self._try("checkpoint", self.publish_checkpoint, stable_seconds=0.0)
        ok &= self._try("outputs", self.publish_outputs, stable_seconds=0.0, force=True)
        if rc == 0 and not ok:
            self.write_result("output_sync_failed", detail=str(self.result["publish_failures"]))
            return EXIT_OUTPUT_SYNC_FAILED
        self.write_result("gx_exited")
        return rc

    def _terminate(self, proc: subprocess.Popen[bytes]) -> int:
        """SIGTERM/SIGINT (Spot preemption, cancel): publish what is safe, then stop GX."""
        deadline = time.monotonic() + self.cfg.term_budget_s
        signum = self._stop_signal or signal.SIGTERM
        self.result["signal"] = signal.Signals(signum).name
        if self.fence_ok():
            self._try("gx.log", self.publish_log, attempts=1)
            if time.monotonic() < deadline:
                self._try(
                    "checkpoint",
                    self.publish_checkpoint,
                    stable_seconds=self.cfg.stable_seconds,
                    attempts=1,
                )
            if time.monotonic() < deadline:
                self._try(
                    "outputs",
                    self.publish_outputs,
                    stable_seconds=self.cfg.stable_seconds,
                    force=True,
                    attempts=1,
                )
            self.write_result("terminated")
        self._stop_process(proc, grace_s=max(0.5, deadline - time.monotonic()))
        return EXIT_TERMINATED

    def cleanup(self) -> None:
        if self.work is not None:
            shutil.rmtree(self.work, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gx-task", description="Fenced, checkpoint-safe GX runner for Batch/GKE."
    )
    p.add_argument("--src-dir", required=True, help="Run directory on the GCS FUSE mount")
    p.add_argument("--attempt-id", required=True, help="Scheduler job id of this attempt")
    p.add_argument("--input", default="gx.in")
    p.add_argument("--gx-binary", default=os.environ.get("GX_BINARY", "gx"))
    p.add_argument("--work-root", default=os.environ.get("GX_WORK_ROOT", tempfile.gettempdir()))
    p.add_argument("--sync-interval", type=float, default=60.0)
    p.add_argument("--output-sync-interval", type=float, default=600.0)
    p.add_argument("--stable-seconds", type=float, default=20.0)
    p.add_argument("--term-budget", type=float, default=20.0)
    p.add_argument("--environment", default=os.environ.get("GK_ENVIRONMENT", "prod"))
    p.add_argument(
        "--no-fence",
        action="store_true",
        help="Manual debugging only: do not require the fencing token",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    build_info = os.environ.get("GX_BUILD_INFO", "/opt/gx/gx_build.json")
    cfg = TaskConfig(
        src_dir=Path(args.src_dir),
        attempt_id=args.attempt_id,
        input_name=args.input,
        gx_command=[args.gx_binary],
        work_root=Path(args.work_root),
        sync_interval_s=args.sync_interval,
        output_sync_interval_s=args.output_sync_interval,
        stable_seconds=args.stable_seconds,
        term_budget_s=args.term_budget,
        environment=str(args.environment).strip().lower(),
        simulate_requested=os.environ.get("GX_SIMULATE_RUN", "").strip().lower() in _TRUTHY,
        build_info_path=Path(build_info) if build_info else None,
        enforce_fence=not args.no_fence,
    )
    runner = TaskRunner(cfg)
    runner.install_signal_handlers()
    try:
        return runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
