"""Command-line interface (`gx-jax`) for Stellarator GX-JAX simulations.

Provides a drop-in CLI driver compatible with C++ GX `.in` TOML decks and KR2
runner workflows:
  * Configures JAX 64-bit precision (`jax_enable_x64=True`) and highest matmul
    precision (`jax_default_matmul_precision="highest"`) on startup before device
    array creation.
  * Parses `<input.in>` into `GXConfig`, builds grids, geometry, field solver,
    and gyrokinetic operators.
  * Initializes state via `init_moments` or loads a `.restart.nc` checkpoint
    (`--restart-file` or `restart = true` in `[Restart]`).
  * Runs single-device or 2D `("sp", "m")` TPU/CPU `shard_map` time stepping
    (`sspx2`, `rk3`, `sspx3`, `rk4`, `k10`) honoring both `nstep` and `t_max`
    (`run_gx.cu:L128`).
  * Streams C++ `.out.nc`-compatible diagnostics matching `Diagnostics_GK::loop`
    cadence (`diagnostics.cu:L124-177`, never writing `t=0`), writes atomic
    periodic (`nsave`) and final `.restart.nc` checkpoints (`run_gx.cu:L140,L150`),
    and emits optional KR2 JSON metrics or throughput benchmarks.
  * Multi-host `jax.distributed` runs (`--distributed`, `--coordinator-address`,
    `--num-processes`, `--process-id`; auto-detected on GKE JobSet / TPU slices, Cloud
    Batch and SLURM `srun` steps): every process executes the stepper, diagnostics and
    gathers in strict lockstep, while only process 0 opens or writes `.out.nc`,
    `.restart.nc` and JSON files. Single-host runs never start the runtime.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import math
import os
import time
import zlib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from gx_jax.config import load_config
from gx_jax.diagnostics_nc import (
    DiagnosticsSnapshot,
    DiagnosticsState,
    DiagnosticsWriter,
    compute_diagnostics_step,
    extract_kr2_diagnostics,
    init_diagnostics_state,
    sanitize_json_payload,
)
from gx_jax.distributed import (
    abort_distributed_run,
    add_distributed_cli_args,
    assert_same_across_processes,
    gather_global_tree,
    host_value,
    init_distributed_runtime_from_args,
    is_distributed,
    is_primary_process,
    shutdown_distributed_runtime,
    sync_processes,
    validate_multihost_mesh,
)
from gx_jax.fields import (
    _time_dtype,
    init_moments,
    load_restart_nc,
    save_restart_nc,
)
from gx_jax.geometry_nc import init_geometry
from gx_jax.integrator import (
    _is_truly_fixed_dt,
    init_gyrokinetic_operator,
    init_integrator_state,
    make_chunk_stepper,
)
from gx_jax.shard_tpu import (
    _place_diag_state_on_mesh,
    benchmark_sharded_step,
    configure_tpu_precision,
    create_device_mesh,
    gather_g_state,
    host_precompute_context,
    init_sharded_integrator_state,
    make_sharded_chunk_stepper,
    move_tree_to_default_device,
    sharded_compute_diagnostics,
)


class AsyncDiskIO:
    """Thread-safe single-worker FIFO executor for host NetCDF4 snapshot & restart I/O.

    Routes both :meth:`DiagnosticsWriter.write_snapshot` and
    :func:`_atomic_save_restart_nc` through the same dedicated background thread
    so HDF5/netCDF4 C APIs are never invoked concurrently across threads, while
    overlapping disk I/O with compiled on-device stepping chunks.
    """

    def __init__(self, *, enabled: bool = True, max_pending: int = 8) -> None:
        self._enabled = bool(enabled)
        self._max_pending = max(1, int(max_pending))
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="gx-jax-netcdf-io",
            )
            if self._enabled
            else None
        )
        self._pending: deque[concurrent.futures.Future[Any]] = deque()
        self._submitted_count: int = 0

    @property
    def enabled(self) -> bool:
        """Whether asynchronous background execution is active."""
        return self._enabled

    @property
    def submitted_count(self) -> int:
        """Total number of I/O tasks submitted to this queue."""
        return self._submitted_count

    def _drain_completed(self) -> None:
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future[Any] | None:
        """Enqueue ``fn(*args, **kwargs)`` on the single FIFO worker (or run inline)."""
        if not self._enabled:
            self._submitted_count += 1
            fn(*args, **kwargs)
            return None
        if self._executor is None:
            raise RuntimeError("AsyncDiskIO executor has already been closed.")
        self._drain_completed()
        while len(self._pending) >= self._max_pending:
            self._pending.popleft().result()
        fut = self._executor.submit(fn, *args, **kwargs)
        self._pending.append(fut)
        self._submitted_count += 1
        return fut

    def flush(self) -> None:
        """Block until all queued writes finish, deterministically re-raising any failure."""
        first_exc: BaseException | None = None
        while self._pending:
            fut = self._pending.popleft()
            try:
                fut.result()
            except BaseException as exc:
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc

    def close(self) -> None:
        """Flush pending writes and shut down the background worker thread."""
        try:
            self.flush()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def __enter__(self) -> AsyncDiskIO:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            while self._pending:
                fut = self._pending.popleft()
                try:
                    fut.result()
                except BaseException:
                    pass
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
            return
        self.close()


def parse_mesh_arg(mesh_str: str | None) -> tuple[int, int] | None:
    """Parse `--mesh SPxM` or `--mesh SP,M` into `(sp_shards, m_shards)`."""
    if mesh_str is None or mesh_str.strip() == "" or mesh_str.strip().lower() == "auto":
        return None
    cleaned = mesh_str.strip().lower().replace("x", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --mesh specification {mesh_str!r}; expected 'SPxM' or 'SP,M' (e.g. '2x4')"
        )
    sp_shards = int(parts[0])
    m_shards = int(parts[1])
    if sp_shards < 1 or m_shards < 1:
        raise ValueError(f"Mesh dimensions must be >= 1, got ({sp_shards}, {m_shards})")
    return (sp_shards, m_shards)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the `gx-jax` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gx-jax",
        description="Stellarator GX-JAX Hermite-Laguerre gyrokinetic solver.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to GX input TOML deck (<stem>.in or <stem>.toml).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output NetCDF path (defaults to <input_stem>.out.nc).",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=None,
        help="Override number of timesteps to execute.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override timestep size dt.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Override maximum simulation time t_max.",
    )
    parser.add_argument(
        "--nwrite",
        type=int,
        default=None,
        help="Override diagnostic write cadence (steps between .out.nc frames).",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        choices=["sspx2", "rk3", "sspx3", "rk4", "k10"],
        help="Override time integration scheme.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--nonlinear",
        action="store_true",
        default=False,
        help="Force nonlinear ExB advection on.",
    )
    mode_group.add_argument(
        "--linear",
        action="store_true",
        default=False,
        help="Force linear mode (disable nonlinear ExB advection).",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="2D ('sp', 'm') device mesh shape, e.g. '1x1', '2x2', '1x4', or 'auto'.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp64"],
        help=(
            "Floating-point precision: 'fp32' (complex64, default for TPU MXU throughput) "
            "or 'fp64' (complex128)."
        ),
    )
    parser.add_argument(
        "--restart-file",
        type=Path,
        default=None,
        help="Load initial G state from a C++/JAX .restart.nc file.",
    )
    parser.add_argument(
        "--save-restart",
        type=Path,
        default=None,
        help="Save final G state to a .restart.nc file.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Write KR2-compatible summary diagnostics JSON to this path.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run JIT warmup + step throughput benchmark and print timing metrics.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=5,
        help="Number of timed steps when --benchmark is enabled (default: 5).",
    )
    parser.add_argument(
        "--async-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlap host NetCDF4 snapshot and restart disk writes with compiled "
            "on-device stepping via a single-worker background executor (default: True)."
        ),
    )
    parser.add_argument(
        "--fft-backend",
        type=str,
        choices=("auto", "fft", "mxu"),
        default=None,
        help="Spectral transform backend override ('auto', 'fft', or 'mxu').",
    )
    parser.add_argument(
        "--use-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt into fixed-dt jax.lax.scan fast path across steady-state chunks "
            "(FP-identical, not bit-identical vs default while_loop; default: False)."
        ),
    )
    parser.add_argument(
        "--write-fields",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [Diagnostics] fields = true/false.",
    )
    parser.add_argument(
        "--write-moments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [Diagnostics] moments = true/false.",
    )
    parser.add_argument(
        "--save-for-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [Restart] save_for_restart = true/false.",
    )
    add_distributed_cli_args(parser)
    return parser


def _default_output_path(input_file: Path) -> Path:
    """Resolve `<stem>.out.nc` matching C++ GX CLI convention."""
    name = input_file.name
    if name.endswith(".in"):
        stem = name[:-3]
    elif name.endswith(".toml"):
        stem = name[:-5]
    else:
        stem = input_file.stem
    return input_file.with_name(f"{stem}.out.nc")


def _atomic_save_restart_nc(
    path: Path,
    g_state: jax.Array,
    time_val: float,
    grids: Any,
) -> None:
    """Atomically write `.restart.nc` via a hidden temp file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_restart_nc(tmp_path, g_state, float(time_val), grids)
    os.replace(tmp_path, path)


def _save_restart_task(
    path: Path,
    g_state: jax.Array,
    time_val: float | jax.Array,
    grids: Any,
) -> None:
    """Worker-thread helper converting device scalar time to host float before atomic save."""
    _atomic_save_restart_nc(path, g_state, float(time_val), grids)


def run_cli_simulation(
    input_file: str | Path,
    *,
    output_path: str | Path | None = None,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    mesh_shape: tuple[int, int] | None = None,
    precision: str = "fp32",
    restart_file: str | Path | None = None,
    save_restart_path: str | Path | None = None,
    metrics_json_path: str | Path | None = None,
    benchmark: bool = False,
    benchmark_steps: int = 5,
    async_io: bool = True,
    fft_backend: str | None = None,
    use_scan: bool = False,
    write_fields: bool | None = None,
    write_moments: bool | None = None,
    save_for_restart: bool | None = None,
) -> dict[str, Any]:
    """Execute a complete GX-JAX simulation from a TOML input deck."""
    use_fp64 = precision.lower() != "fp32"
    configure_tpu_precision(enable_x64=True)
    real_dtype = jnp.float64 if use_fp64 else jnp.float32
    complex_dtype = jnp.complex128 if use_fp64 else jnp.complex64

    if fft_backend is not None:
        fft_backend_clean = str(fft_backend).strip().lower()
        if fft_backend_clean not in ("auto", "fft", "mxu"):
            raise ValueError(
                f"Unsupported fft_backend={fft_backend!r}; expected 'auto', 'fft', or 'mxu'."
            )
        fft_backend = fft_backend_clean

    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if force_nonlinear and force_linear:
        raise ValueError("force_nonlinear and force_linear are mutually exclusive")
    out_nc_path = Path(output_path) if output_path is not None else _default_output_path(input_path)

    cfg = load_config(input_path)
    overrides: dict[str, Any] = {}
    if nstep is not None:
        overrides["nstep"] = int(nstep)
    if dt is not None:
        overrides["dt"] = float(dt)
    if t_max is not None:
        overrides["t_max"] = float(t_max)
    if scheme is not None:
        overrides["scheme"] = str(scheme).lower()
    if nwrite is not None:
        overrides["nwrite"] = int(nwrite)
    if force_nonlinear:
        overrides["nonlinear_mode"] = True
    if force_linear:
        overrides["nonlinear_mode"] = False
    if fft_backend is not None and hasattr(cfg, "fft_backend"):
        overrides["fft_backend"] = fft_backend
    if write_fields is not None:
        overrides["write_fields"] = bool(write_fields)
    if write_moments is not None:
        overrides["write_moms"] = bool(write_moments)
    if save_for_restart is not None:
        overrides["save_for_restart"] = bool(save_for_restart)
    if overrides:
        cfg = replace(cfg, **overrides)

    dt_val = float(cfg.dt) if cfg.dt > 0.0 else 0.05
    nstep_max = max(0, int(cfg.nstep))
    t_max_val = float(cfg.t_max)
    nwrite_val = max(1, int(cfg.nwrite))
    nsave_val = max(1, int(cfg.nsave))
    t_dtype = _time_dtype(real_dtype)
    eps_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-11
    eps_t = eps_rel * max(abs(t_max_val), 1.0e-30)

    with host_precompute_context():
        cfg, grids_host, geo_host = init_geometry(cfg, dtype=real_dtype)
        gk_op_host = init_gyrokinetic_operator(cfg, grids_host, geo_host, dtype=real_dtype)

    # Resolve restart output path upfront for periodic nsave + final atomic checkpoints
    restart_out: Path | None = None
    if save_restart_path is not None:
        restart_out = Path(save_restart_path)
    elif cfg.save_for_restart:
        if cfg.restart_to_file:
            restart_out = Path(cfg.restart_to_file)
            if not restart_out.is_absolute():
                restart_out = out_nc_path.parent / restart_out
        else:
            restart_out = out_nc_path.with_name(f"{input_path.stem}.restart.nc")

    # Initial condition or restart load (honoring restart_if_exists & restart_with_perturb)
    restart_src: Path | None = None
    if restart_file is not None:
        restart_src = Path(restart_file)
    elif cfg.restart_from_file:
        candidate = Path(cfg.restart_from_file)
        if not candidate.is_absolute():
            candidate = input_path.parent / candidate
        if cfg.restart_if_exists:
            if candidate.is_file():
                restart_src = candidate
        elif cfg.restart:
            restart_src = candidate

    is_restarting = restart_src is not None
    t0 = 0.0
    with host_precompute_context():
        if restart_src is not None:
            if not restart_src.is_file():
                raise FileNotFoundError(f"Restart file does not exist: {restart_src}")
            g_loaded, t_loaded = load_restart_nc(
                restart_src,
                grids_host,
                scale=float(cfg.restart_scale),
                dtype=complex_dtype,
            )
            t0 = float(t_loaded)
            if cfg.restart_with_perturb:
                g_init_host = (
                    init_moments(cfg, grids_host, geo_host, dtype=complex_dtype) + g_loaded
                )
            else:
                g_init_host = g_loaded
        else:
            g_init_host = init_moments(cfg, grids_host, geo_host, dtype=complex_dtype)

    grids = move_tree_to_default_device(grids_host.astype(real_dtype))
    geo = move_tree_to_default_device(geo_host)
    gk_op = move_tree_to_default_device(gk_op_host)
    solver = gk_op.linear_op.solver
    g_init = move_tree_to_default_device(g_init_host)

    mesh = create_device_mesh(
        mesh_shape=mesh_shape,
        nspecies=cfg.nspecies,
        nhermite=cfg.nhermite,
        enable_x64=True,
    )
    # Multi-host (jax.process_count() > 1): every process runs the stepper, diagnostics
    # and gathers in strict lockstep; only the primary process (jax.process_index() == 0)
    # opens or writes .out.nc / .restart.nc / .json files.
    multi_host = is_distributed()
    is_primary = is_primary_process()
    if multi_host:
        validate_multihost_mesh(mesh)
        assert_same_across_processes(
            (
                zlib.crc32(input_path.read_bytes()),
                (t0, is_restarting, tuple(g_init.shape), str(g_init.dtype)),
                (nstep_max, dt_val, t_max_val, nwrite_val, nsave_val, str(cfg.scheme)),
                (dt, bool(cfg.nonlinear_mode), bool(use_scan), restart_out is not None),
                tuple(int(n) for n in mesh.devices.shape),
            ),
            "run inputs (deck, overrides, restart state, mesh)",
        )
    use_sharded = (
        (int(mesh.shape["sp"]) * int(mesh.shape["m"]) > 1) or (mesh_shape is not None) or multi_host
    )

    benchmark_result: dict[str, Any] | None = None
    if benchmark:
        benchmark_result = benchmark_sharded_step(
            cfg,
            grids,
            geo,
            mesh,
            warmup_steps=2,
            timed_steps=max(1, int(benchmark_steps)),
            dtype=complex_dtype,
            scheme=cfg.scheme,
            gk_op=gk_op,
            g_init=g_init,
        )

    is_fixed_dt = _is_truly_fixed_dt(gk_op, cfg.scheme, dt)
    step_dt_known = float(dt) if dt is not None else dt_val
    has_finite_t_max = math.isfinite(t_max_val) and (t_max is not None or t_max_val < 1.0e19)
    steady_chunk_len = math.gcd(nwrite_val, nsave_val) if restart_out is not None else nwrite_val
    # Default to zero-sync while_loop for bit-exact parity with step_sharded_integrator_state;
    # jax.lax.scan is opt-in via use_scan=True / --use-scan (FP-identical, ~1 ULP XLA diff).
    scan_chunk_len = (
        steady_chunk_len if (use_scan and is_fixed_dt and steady_chunk_len >= 1) else None
    )
    dt_init = float(dt) if dt is not None else (dt_val if cfg.fixed_dt else None)

    t_wall_start = time.perf_counter()
    counter = 0
    last_saved_step = -1
    cur_t_host: float | None = t0

    writer_cm: contextlib.AbstractContextManager[DiagnosticsWriter | None] = (
        DiagnosticsWriter(
            out_nc_path,
            cfg,
            grids_host,
            geo_host,
            append=bool(cfg.append_on_restart and is_restarting),
        )
        if is_primary
        else contextlib.nullcontext(None)
    )
    with writer_cm as writer:
        if use_sharded:
            state = init_sharded_integrator_state(
                g_init,
                gk_op,
                mesh,
                time=t0,
                dt=dt_init,
            )
            diag_state = _place_diag_state_on_mesh(
                init_diagnostics_state(
                    state.G,
                    state.fields,
                    time=state.time,
                    has_old=False,
                ),
                mesh,
            )
            diag_fn = jax.jit(
                lambda g_a, f_a, d_a, t_a, dt_a: sharded_compute_diagnostics(
                    g_a,
                    f_a,
                    d_a,
                    grids,
                    geo,
                    solver,
                    mesh,
                    time=t_a,
                    dt=dt_a,
                    beta=float(cfg.beta),
                )
            )
            stepper = make_sharded_chunk_stepper(
                gk_op,
                mesh,
                dt_override=dt,
                scheme=cfg.scheme,
                t_max=t_max_val,
                fixed_chunk_len=scan_chunk_len,
                diag_nwrite=nwrite_val,
            )
        else:
            state = init_integrator_state(
                g_init,
                gk_op,
                time=t0,
                dt=dt_init,
            )
            diag_state = init_diagnostics_state(
                state.G,
                state.fields,
                time=state.time,
                has_old=False,
            )

            # Keep unsharded diagnostics evaluation eager (matching
            # DiagnosticsWriter.write_step) while offloading NetCDF4 I/O to AsyncDiskIO:
            # TurbulentHeating* cancels near-equal energies / dt, so eager vs jax.jit
            # reordering shifts it by roundoff. The sharded branch above keeps
            # jax.jit(sharded_compute_diagnostics).
            def diag_fn(
                g_a: jax.Array,
                f_a: Any,
                d_a: DiagnosticsState,
                t_a: jax.Array,
                dt_a: jax.Array,
            ) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
                return compute_diagnostics_step(
                    g_a,
                    f_a,
                    d_a,
                    grids,
                    geo,
                    solver,
                    time=t_a,
                    dt=dt_a,
                    beta=float(cfg.beta),
                )

            stepper = make_chunk_stepper(
                gk_op,
                dt_override=dt,
                scheme=cfg.scheme,
                t_max=t_max_val,
                fixed_chunk_len=scan_chunk_len,
                diag_nwrite=nwrite_val,
            )

        with AsyncDiskIO(enabled=async_io) as io_queue:
            while counter < nstep_max and (cur_t_host is None or cur_t_host < t_max_val - eps_t):
                steps_to_write = 1 + ((-counter) % nwrite_val)
                if restart_out is not None:
                    steps_to_save = 1 + ((-counter) % nsave_val)
                    max_steps = min(steps_to_write, steps_to_save, nstep_max - counter)
                else:
                    max_steps = min(steps_to_write, nstep_max - counter)

                if is_fixed_dt:
                    if not has_finite_t_max:
                        safe_no_tmax_stop = True
                    elif cur_t_host is not None:
                        t_after_est = cur_t_host + float(max_steps) * step_dt_known
                        t_margin = max(1.0e-7 * max(abs(t_max_val), 1.0), 10.0 * eps_t)
                        safe_no_tmax_stop = t_after_est < (t_max_val - eps_t - t_margin)
                    else:
                        safe_no_tmax_stop = False
                else:
                    safe_no_tmax_stop = not has_finite_t_max

                use_scan_now = bool(
                    is_fixed_dt
                    and scan_chunk_len is not None
                    and max_steps == scan_chunk_len
                    and safe_no_tmax_stop
                )
                chunk = stepper(
                    state,
                    max_steps,
                    diag_state=diag_state,
                    safe_for_scan=use_scan_now,
                )
                state = chunk.state
                assert chunk.diag_state is not None
                diag_state = chunk.diag_state

                if safe_no_tmax_stop:
                    steps_done = max_steps
                    counter += steps_done
                    if is_fixed_dt and cur_t_host is not None:
                        cur_t_host = cur_t_host + float(steps_done) * step_dt_known
                    else:
                        cur_t_host = None
                    is_write_step = (counter - 1) % nwrite_val == 0
                    is_save_step = restart_out is not None and ((counter - 1) % nsave_val == 0)
                else:
                    steps_done = int(host_value(chunk.steps_done))
                    if steps_done <= 0:
                        cur_t_host = float(host_value(state.time))
                        break
                    counter += steps_done
                    end_t = float(host_value(state.time))
                    cur_t_host = end_t
                    is_write_step = ((counter - 1) % nwrite_val == 0) or (end_t > t_max_val + eps_t)
                    is_save_step = restart_out is not None and ((counter - 1) % nsave_val == 0)

                if is_write_step:
                    snap, diag_state = diag_fn(
                        state.G,
                        state.fields,
                        diag_state,
                        state.time,
                        state.dt,
                    )
                    g_snap = (
                        (gather_g_state(state.G, mesh) if use_sharded else state.G)
                        if cfg.write_moms
                        else None
                    )
                    f_snap = state.fields if (cfg.write_fields or cfg.write_moms) else None
                    if multi_host:
                        # Lockstep host gather on every process (never hand a
                        # non-addressable global array to the writer thread).
                        snap, f_snap = gather_global_tree((snap, f_snap))
                    if writer is not None:
                        io_queue.submit(
                            writer.write_snapshot,
                            snap,
                            G=g_snap,
                            fields=f_snap,
                            solver=solver,
                        )

                if is_save_step:
                    assert restart_out is not None
                    # Every gather / host read runs on all processes, above the
                    # primary-only gate, so no collective can ever run on one rank alone.
                    g_save = gather_g_state(state.G, mesh) if use_sharded else state.G
                    t_save = host_value(state.time)
                    if is_primary:
                        io_queue.submit(
                            _save_restart_task,
                            restart_out,
                            g_save,
                            t_save,
                            grids_host,
                        )
                    last_saved_step = counter

        g_final = gather_g_state(state.G, mesh) if use_sharded else state.G
        fields_final = state.fields
        if multi_host:
            fields_final = jax.tree_util.tree_map(jnp.asarray, gather_global_tree(fields_final))
        t_end = float(host_value(state.time))

    wall_seconds = time.perf_counter() - t_wall_start

    if restart_out is not None and last_saved_step != counter and is_primary:
        _atomic_save_restart_nc(restart_out, g_final, t_end, grids_host)

    has_explicit_t_max = (t_max is not None) or (t_max_val < 1.0e19)
    step_dt_target = float(dt) if dt is not None else dt_val
    nstep_end_target = t0 + float(nstep_max) * step_dt_target
    if t_max is not None:
        sim_time_target = t_max_val
    elif nstep is not None and has_explicit_t_max:
        sim_time_target = min(t_max_val, nstep_end_target)
    elif has_explicit_t_max:
        sim_time_target = t_max_val
    else:
        sim_time_target = nstep_end_target

    kr2_summary: dict[str, Any] | None = None
    if is_primary:
        kr2_summary = extract_kr2_diagnostics(
            out_nc_path,
            metrics_out_path=metrics_json_path,
            sim_time_target=sim_time_target,
        )
    if metrics_json_path is not None and is_primary:
        metrics_path = Path(metrics_json_path)
        payload = json.loads(metrics_path.read_text())
        payload["input_file"] = str(input_path)
        payload["output_nc"] = str(out_nc_path)
        payload["nstep"] = counter
        payload["dt"] = dt_val
        payload["t_final"] = t_end
        payload["sim_time_target"] = sim_time_target
        payload["scheme"] = cfg.scheme
        payload["nonlinear"] = bool(cfg.nonlinear_mode)
        payload["precision"] = precision
        payload["wall_seconds"] = wall_seconds
        if benchmark_result is not None:
            payload["benchmark"] = benchmark_result
        clean_payload = sanitize_json_payload(payload)
        metrics_path.write_text(
            json.dumps(clean_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    # Multi-host: every process leaves the run together, after the primary's disk writes.
    sync_processes("gx-jax run complete")

    return {
        "output_nc": out_nc_path,
        "restart_nc": restart_out,
        "g_final": g_final,
        "fields_final": fields_final,
        "nstep": counter,
        "t_final": t_end,
        "sim_time_target": sim_time_target,
        "wall_seconds": wall_seconds,
        "kr2_summary": kr2_summary,
        "benchmark": benchmark_result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `gx-jax`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.input_file.is_file():
        parser.error(f"input file does not exist: {args.input_file}")
    if args.restart_file is not None and not args.restart_file.is_file():
        parser.error(f"restart file does not exist: {args.restart_file}")
    try:
        mesh_shape = parse_mesh_arg(args.mesh)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        # Multi-host: initialize jax.distributed before any JAX device access
        # (zero-overhead no-op on a single host or with --no-distributed).
        try:
            init_distributed_runtime_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))

        result = run_cli_simulation(
            args.input_file,
            output_path=args.output,
            nstep=args.nstep,
            dt=args.dt,
            t_max=args.t_max,
            nwrite=args.nwrite,
            scheme=args.scheme,
            force_nonlinear=args.nonlinear,
            force_linear=args.linear,
            mesh_shape=mesh_shape,
            precision=args.precision,
            restart_file=args.restart_file,
            save_restart_path=args.save_restart,
            metrics_json_path=args.metrics_json,
            benchmark=args.benchmark,
            benchmark_steps=args.benchmark_steps,
            async_io=args.async_io,
            fft_backend=args.fft_backend,
            use_scan=args.use_scan,
            write_fields=args.write_fields,
            write_moments=args.write_moments,
            save_for_restart=args.save_for_restart,
        )

        if args.benchmark and result["benchmark"] is not None and is_primary_process():
            b = result["benchmark"]
            print(
                f"[gx-jax benchmark] ms/step={b['ms_per_step']:.3f} "
                f"steps/sec={b['steps_per_sec']:.2f} "
                f"devices={int(b['num_devices'])} mesh={b['mesh_shape']}"
            )
    except BaseException as exc:
        # Multi-host only: a process failing alone exits immediately instead of
        # deadlocking at the shutdown barrier (no-op on a single host).
        abort_distributed_run(exc)
        raise
    finally:
        shutdown_distributed_runtime()
    return 0


__all__ = [
    "AsyncDiskIO",
    "build_arg_parser",
    "main",
    "parse_mesh_arg",
    "run_cli_simulation",
]


if __name__ == "__main__":
    raise SystemExit(main())
