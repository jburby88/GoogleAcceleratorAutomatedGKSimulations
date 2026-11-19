"""Multi-Host `jax.distributed` Runtime & Lockstep Collectives.

Every `XLA_FLAGS=--xla_force_host_platform_device_count=...` run and every real
`jax.distributed` runtime lives in an isolated subprocess with an explicit timeout; the
pytest process itself never initializes `jax.distributed`.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec

import gx_jax.cli as cli_mod
import gx_jax.distributed as gxd
import gx_jax.ensemble as ens_mod
from gx_jax.cli import build_arg_parser, run_cli_simulation
from gx_jax.distributed import (
    DEFAULT_COORDINATOR_PORT,
    DistributedConfig,
    abort_distributed_run,
    detect_distributed_env,
    device_put_global,
    gather_global_array,
    gather_global_tree,
    host_value,
    init_distributed_runtime,
    is_distributed,
    is_primary_process,
    normalize_coordinator_address,
    process_count,
    process_index,
    shutdown_distributed_runtime,
    sync_processes,
    validate_multihost_mesh,
)
from gx_jax.ensemble import (
    build_ensemble_arg_parser,
    create_ensemble_mesh,
    gather_ensemble_tree,
    shard_ensemble_tree,
)
from gx_jax.shard_tpu import (
    create_device_mesh,
    gather_g_state,
    make_g_sharding,
    make_replicated_sharding,
    shard_g_state,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Explicit wall-clock budget for every subprocess group (a hang fails the test).
SUBPROCESS_TIMEOUT_S = float(os.environ.get("GX_JAX_TEST_SUBPROCESS_TIMEOUT", "360"))

# Ambient cluster variables that must not leak into hermetic child processes.
_CLUSTER_ENV_PREFIXES = (
    "SLURM_",
    "MEGASCALE_",
    "TPU_WORKER_",
    "BATCH_",
    "JAX_COORDINATOR",
    "JAX_NUM_PROCESSES",
    "JAX_PROCESS_ID",
    "JAX_LOCAL_DEVICE_IDS",
    "JOB_COMPLETION_INDEX",
    "GX_JAX_",
)


# `sbatch --ntasks=4` batch step running ONE process (no `srun`): SLURM exports the task
# count / ids but no SLURM_STEP_NODELIST. Must stay a single-host run (as in Session 7).
_SBATCH_WITHOUT_SRUN_ENV = {
    "SLURM_JOB_ID": "123456",
    "SLURM_NTASKS": "4",
    "SLURM_PROCID": "0",
    "SLURM_LOCALID": "0",
    "SLURM_JOB_NODELIST": "n1",
}
_SINGLE_HOST_TPU_ENV = {"TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost"}


@pytest.fixture(scope="module", autouse=True)
def _enable_x64() -> None:
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _fresh_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the module-level ownership record of `gx_jax.distributed` per test."""
    monkeypatch.setattr(gxd, "_STATE", gxd._RuntimeState())


# =============================================================================
# Helpers: decks, subprocess groups, and output comparison
# =============================================================================


def _write_deck(
    path: Path,
    *,
    fixed_dt: bool,
    nonlinear: bool = True,
    tprim: float = 2.49,
    beta: float = 0.015,
    cfl: float = 0.9,
    nstep: int = 6,
    nwrite: int = 2,
    nsave: int = 3,
    save_for_restart: bool = True,
    write_fields: bool = True,
    write_moms: bool = True,
) -> Path:
    """Write a compact 2-species electromagnetic Miller deck (`nspecies=2`, `nhermite=4`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
[Dimensions]
 ntheta = 8
 nperiod = 1
 ny = 6
 nx = 6
 nhermite = 4
 nlaguerre = 2
 nspecies = 2

[Domain]
 y0 = 10.0
 boundary = "linked"

[Physics]
 beta = {beta}
 nonlinear_mode = {str(nonlinear).lower()}
 fphi = 1.0
 fapar = 1.0
 fbpar = 1.0

[Time]
 dt = 0.002
 dt_max = 0.01
 dt_min = 1.0e-7
 cfl = {cfl}
 scheme = "rk3"
 fixed_dt = {str(fixed_dt).lower()}
 nstep = {nstep}

[Initialization]
 ikpar_init = 0
 init_field = "density"
 init_amp = 1.0e-5

[Geometry]
 geo_option = "miller"
 rhoc = 0.5
 Rmaj = 2.77778
 R_geo = 2.77778
 qinp = 1.4
 shat = 0.8
 shift = 0.0
 akappa = 1.0
 akappri = 0.0
 tri = 0.0
 tripri = 0.0
 betaprim = 0.0

[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.02 ]
 dens = [ 1.0, 1.0 ]
 temp = [ 1.0, 1.0 ]
 tprim = [ {tprim}, {tprim} ]
 fprim = [ 0.8, 0.8 ]
 vnewk = [ 0.01, 0.01 ]
 type = [ "ion", "electron" ]

[Boltzmann]
 add_Boltzmann_species = false

[Dissipation]
 closure_model = "none"
 hypercollisions = true
 nu_hyper_m = 0.1
 p_hyper_m = 6
 nu_hyper_l = 0.1
 p_hyper_l = 6
 hyper = true
 D_hyper = 0.05

[Diagnostics]
 nwrite = {nwrite}
 omega = true
 free_energy = true
 fluxes = true
 fields = {str(write_fields).lower()}
 moments = {str(write_moms).lower()}

[Restart]
 save_for_restart = {str(save_for_restart).lower()}
 nsave = {nsave}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_ensemble_decks(
    deck_dir: Path, *, fixed_dt: bool = False, nonlinear: bool = True
) -> list[Path]:
    """Write B=4 structurally compatible decks sweeping gradients, beta, CFL and `nstep`."""
    sweep = [
        {"tprim": 2.49, "beta": 0.015, "cfl": 0.90, "nstep": 6},
        {"tprim": 3.00, "beta": 0.010, "cfl": 0.80, "nstep": 5},
        {"tprim": 2.00, "beta": 0.020, "cfl": 0.70, "nstep": 6},
        {"tprim": 3.50, "beta": 0.005, "cfl": 0.85, "nstep": 4},
    ]
    return [
        _write_deck(deck_dir / f"member{idx}.in", fixed_dt=fixed_dt, nonlinear=nonlinear, **params)
        for idx, params in enumerate(sweep)
    ]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _child_env(num_devices: int, **extra: str) -> dict[str, str]:
    """Hermetic child environment: worktree `PYTHONPATH`, virtual CPU devices, short timeouts."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(_CLUSTER_ENV_PREFIXES)
    }
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
    env["JAX_PLATFORMS"] = "cpu"
    # Bound every jax.distributed wait well below the subprocess timeout.
    env["GX_JAX_DISTRIBUTED_INIT_TIMEOUT"] = "90"
    env["GX_JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT"] = "30"
    env["GX_JAX_DISTRIBUTED_SHUTDOWN_TIMEOUT"] = "30"
    env.update(extra)
    return env


def _log_tail(log_path: Path, limit: int = 6000) -> str:
    text = log_path.read_text(errors="replace") if log_path.is_file() else "<no log>"
    lines = [ln for ln in text.splitlines() if not ln.startswith("[Gloo]")]
    return "\n".join(lines)[-limit:]


def _run_process_group(
    jobs: dict[str, tuple[list[str], dict[str, str], Path]],
    log_dir: Path,
    *,
    timeout: float = SUBPROCESS_TIMEOUT_S,
    expect_success: bool = True,
) -> dict[str, int]:
    """Run `{name: (cmd, env, cwd)}` concurrently; a hang or non-zero exit fails the test."""
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: dict[str, tuple[subprocess.Popen[str], Any, Path]] = {}
    hung: list[str] = []
    try:
        for name, (cmd, env, cwd) in jobs.items():
            cwd.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{name}.log"
            log_file = log_path.open("w")
            procs[name] = (
                subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
                log_file,
                log_path,
            )
        deadline = time.monotonic() + timeout
        for name, (proc, _, _) in procs.items():
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                hung.append(name)
    finally:
        for proc, log_file, _ in procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
            log_file.close()

    report = "\n".join(
        f"----- {name} (rc={proc.returncode}) -----\n{_log_tail(log_path)}"
        for name, (proc, _, log_path) in procs.items()
    )
    assert not hung, f"Processes {hung} hung past {timeout:.0f}s (deadlock?):\n{report}"
    codes = {name: int(proc.returncode) for name, (proc, _, _) in procs.items()}
    if expect_success:
        assert all(rc == 0 for rc in codes.values()), f"Non-zero exit codes {codes}:\n{report}"
    return codes


def _files_under(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _nc_max_rel_err(cand: Path, ref: Path) -> tuple[float, str, int]:
    """Worst `|cand - ref| / max(1, max|ref|)` over every numeric variable of every group."""
    worst = 0.0
    worst_name = ""
    n_vars = 0

    def _walk(g_c: Any, g_r: Any, prefix: str) -> None:
        nonlocal worst, worst_name, n_vars
        assert set(g_c.variables) == set(g_r.variables), f"{prefix}: variable sets differ"
        assert set(g_c.groups) == set(g_r.groups), f"{prefix}: group sets differ"
        for name, var_r in g_r.variables.items():
            arr_r = np.ma.getdata(var_r[...])
            arr_c = np.ma.getdata(g_c.variables[name][...])
            if arr_r.dtype.kind not in "fiu":
                continue
            arr_r = np.asarray(arr_r, dtype=np.float64)
            arr_c = np.asarray(arr_c, dtype=np.float64)
            assert arr_c.shape == arr_r.shape, f"{prefix}{name}: {arr_c.shape} != {arr_r.shape}"
            n_vars += 1
            if arr_r.size == 0:
                continue
            assert np.all(np.isfinite(arr_r)), f"{prefix}{name}: reference has NaN/Inf"
            assert np.all(np.isfinite(arr_c)), f"{prefix}{name}: candidate has NaN/Inf"
            scale = max(1.0, float(np.max(np.abs(arr_r))))
            err = float(np.max(np.abs(arr_c - arr_r))) / scale
            if err > worst:
                worst, worst_name = err, f"{prefix}{name}"
        for name in g_r.groups:
            _walk(g_c.groups[name], g_r.groups[name], f"{prefix}{name}/")

    with nc.Dataset(cand, "r") as ds_c, nc.Dataset(ref, "r") as ds_r:
        _walk(ds_c, ds_r, "")
    return worst, worst_name, n_vars


def _assert_nc_close(cand: Path, ref: Path, tol: float, label: str) -> None:
    worst, worst_name, n_vars = _nc_max_rel_err(cand, ref)
    assert n_vars > 0, f"{label}: no numeric variables compared"
    assert worst < tol, f"{label}: {worst_name} differs by {worst:.3e} >= {tol:.1e}"


def _assert_nc_bit_identical(cand: Path, ref: Path, label: str) -> None:
    """`np.array_equal` on every variable of every group (dtype and bits)."""

    def _walk(g_c: Any, g_r: Any, prefix: str) -> int:
        assert set(g_c.variables) == set(g_r.variables), f"{label}:{prefix}: variable sets differ"
        assert set(g_c.groups) == set(g_r.groups), f"{label}:{prefix}: group sets differ"
        count = 0
        for name, var_r in g_r.variables.items():
            arr_r = np.ma.getdata(var_r[...])
            arr_c = np.ma.getdata(g_c.variables[name][...])
            assert arr_c.dtype == arr_r.dtype, f"{label}:{prefix}{name}: dtype differs"
            equal_nan = arr_r.dtype.kind in "fc"
            assert np.array_equal(arr_c, arr_r, equal_nan=equal_nan), (
                f"{label}:{prefix}{name} is not bit-identical"
            )
            count += 1
        for name in g_r.groups:
            count += _walk(g_c.groups[name], g_r.groups[name], f"{prefix}{name}/")
        return count

    with nc.Dataset(cand, "r") as ds_c, nc.Dataset(ref, "r") as ds_r:
        assert _walk(ds_c, ds_r, "") > 0


_VOLATILE_JSON_KEYS = frozenset({"wall_seconds", "benchmark"})


def _assert_json_close(cand: Any, ref: Any, tol: float, label: str) -> None:
    """Recursive strict-structure JSON comparison (numbers within `tol`, wall time ignored)."""
    if isinstance(ref, dict):
        assert isinstance(cand, dict) and set(cand) == set(ref), f"{label}: keys differ"
        for key in ref:
            if key in _VOLATILE_JSON_KEYS:
                continue
            _assert_json_close(cand[key], ref[key], tol, f"{label}.{key}")
    elif isinstance(ref, list):
        assert isinstance(cand, list) and len(cand) == len(ref), f"{label}: list length differs"
        for idx, (c_item, r_item) in enumerate(zip(cand, ref, strict=True)):
            _assert_json_close(c_item, r_item, tol, f"{label}[{idx}]")
    elif isinstance(ref, bool) or ref is None or isinstance(ref, str):
        assert cand == ref, f"{label}: {cand!r} != {ref!r}"
    else:
        assert isinstance(cand, int | float), f"{label}: {cand!r} is not numeric"
        assert abs(float(cand) - float(ref)) <= tol * max(1.0, abs(float(ref))), (
            f"{label}: {cand!r} != {ref!r} within {tol:.1e}"
        )


def _load_strict_json(path: Path) -> Any:
    def _reject(token: str) -> Any:
        raise AssertionError(f"Non-RFC-8259 JSON token {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=_reject)


def _assert_strictly_increasing_time(out_nc: Path) -> int:
    with nc.Dataset(out_nc, "r") as ds:
        t = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
    assert t.size >= 1 and np.all(np.diff(t) > 0.0), f"{out_nc}: duplicated/unordered frames {t}"
    return int(t.size)


# Launcher emulating the strict multi-host `jax.Array` contract of older JAX releases
# inside a real multi-process run: `jax.device_put` onto a sharding with non-addressable
# devices, `jax.device_get` and host value fetches (`float()`, `int()`, `np.asarray`) of a
# non-addressable array all raise instead of being silently tolerated.
_STRICT_LAUNCHER = r"""
import runpy
import sys

import jax

_real_device_put = jax.device_put
_real_device_get = jax.device_get


def _strict_device_put(x, device=None, *args, **kwargs):
    for target in jax.tree_util.tree_leaves(device):
        if not getattr(target, "is_fully_addressable", True):
            raise ValueError(
                "STRICT-MULTIHOST: jax.device_put cannot target non-addressable devices"
            )
    return _real_device_put(x, device, *args, **kwargs)


def _strict_device_get(x):
    for leaf in jax.tree_util.tree_leaves(x):
        if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
            raise RuntimeError("STRICT-MULTIHOST: jax.device_get of a non-addressable array")
    return _real_device_get(x)


jax.device_put = _strict_device_put
jax.device_get = _strict_device_get

from jax._src import array as _jax_array

_guarded = []
for _name in ("__float__", "__int__", "__index__", "__bool__", "__complex__", "__array__",
              "item", "tolist"):
    _orig = getattr(_jax_array.ArrayImpl, _name, None)
    if _orig is None:
        continue

    def _make(orig, name):
        def _strict(self, *args, **kwargs):
            if not self.is_fully_addressable:
                raise RuntimeError(
                    f"STRICT-MULTIHOST: {name} on a non-addressable jax.Array"
                )
            return orig(self, *args, **kwargs)

        return _strict

    try:
        setattr(_jax_array.ArrayImpl, _name, _make(_orig, _name))
        _guarded.append(_name)
    except (TypeError, AttributeError):
        pass
print("STRICT-MULTIHOST guards:", ",".join(_guarded), flush=True)

module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


def _module_cmd(module: str, args: list[str], *, strict_launcher: Path | None) -> list[str]:
    if strict_launcher is not None:
        return [sys.executable, str(strict_launcher), module, *args]
    return [sys.executable, "-m", module, *args]


def _distributed_flags(port: int, process_id: int, num_processes: int = 2) -> list[str]:
    return [
        "--distributed",
        "--coordinator-address",
        f"127.0.0.1:{port}",
        "--num-processes",
        str(num_processes),
        "--process-id",
        str(process_id),
    ]


# =============================================================================
# 1. Environment Detection (pure, no JAX runtime)
# =============================================================================


def test_detect_single_host_environments_are_noop(tmp_path: Path) -> None:
    """Single-host CPU / GPU / TPU and `--no-distributed` never request `jax.distributed`."""
    one_host_file = tmp_path / "batch_hosts_single"
    one_host_file.write_text("vm-0\n")
    single_host_envs: list[dict[str, str]] = [
        {},
        # Single-host TPU VM / GKE v6e-1..v6e-8 pod: worker 0 of one host, no coordinator.
        {"TPU_WORKER_ID": "0"},
        {"TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost"},
        {"TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "gke-v6e-8-pod-0.svc"},
        # Kubernetes Indexed Job campaign worker (independent single-host tasks).
        {"JOB_COMPLETION_INDEX": "3", "TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost"},
        # Single-task SLURM allocation.
        {"SLURM_NTASKS": "1", "SLURM_PROCID": "0", "SLURM_JOB_ID": "77"},
        # Multi-task SLURM allocation whose batch script runs one process without `srun`.
        _SBATCH_WITHOUT_SRUN_ENV,
        {**_SBATCH_WITHOUT_SRUN_ENV, **_SINGLE_HOST_TPU_ENV},
        # Cloud Batch array job: many independent tasks, each alone on its VM.
        {
            "BATCH_HOSTS_FILE": str(one_host_file),
            "BATCH_TASK_INDEX": "5",
            "BATCH_TASK_COUNT": "8",
        },
        # Explicit single process.
        {"JAX_COORDINATOR_ADDRESS": "10.0.0.1:1234", "JAX_NUM_PROCESSES": "1"},
    ]
    for env in single_host_envs:
        cfg = detect_distributed_env(env=env)
        assert cfg == DistributedConfig(
            enabled=False,
            source="single-host",
            reason="no multi-host environment detected (single-host fast path)",
        ), env
        assert cfg.initialize_kwargs() == {}

    # --no-distributed wins over any multi-host environment and explicit overrides.
    multi_env = {
        "MEGASCALE_COORDINATOR_ADDRESS": "slice-0-0",
        "TPU_WORKER_HOSTNAMES": "slice-0-0,slice-0-1",
        "TPU_WORKER_ID": "1",
        "SLURM_NTASKS": "4",
    }
    cfg_off = detect_distributed_env(
        False, env=multi_env, coordinator_address="10.0.0.1:1", num_processes=2, process_id=1
    )
    assert cfg_off.enabled is False and cfg_off.source == "disabled"
    assert cfg_off.coordinator_address is None and cfg_off.initialize_kwargs() == {}

    # Explicit `--num-processes 1` keeps the fast path even with a coordinator address.
    cfg_one = detect_distributed_env(
        coordinator_address="10.0.0.1:1", num_processes=1, process_id=0, env={}
    )
    assert cfg_one.enabled is False and cfg_one.source == "single-host"


@pytest.mark.parametrize(
    ("env", "coordinator", "num_processes", "process_id"),
    [
        # MegaScale / JobSet: coordinator host without port -> default :8476.
        (
            {
                "MEGASCALE_COORDINATOR_ADDRESS": "js-slice-0-0.js",
                "MEGASCALE_NUM_PROCESSES": "4",
                "MEGASCALE_PROCESS_ID": "2",
            },
            "js-slice-0-0.js:8476",
            4,
            2,
        ),
        # Plain JAX variables: explicit port is preserved.
        (
            {
                "JAX_COORDINATOR_ADDRESS": "10.0.0.7:1234",
                "JAX_NUM_PROCESSES": "2",
                "JAX_PROCESS_ID": "1",
            },
            "10.0.0.7:1234",
            2,
            1,
        ),
        # Kubernetes Indexed Job: JOB_COMPLETION_INDEX is the process id fallback.
        (
            {
                "JAX_COORDINATOR_ADDRESS": "gx-jax-0.gx-jax",
                "JAX_NUM_PROCESSES": "3",
                "JOB_COMPLETION_INDEX": "2",
            },
            "gx-jax-0.gx-jax:8476",
            3,
            2,
        ),
        # GKE multi-host TPU slice (v6e-16 = 2 hosts): hostnames + worker id only.
        (
            {"TPU_WORKER_HOSTNAMES": "tpu-0.svc,tpu-1.svc", "TPU_WORKER_ID": "1"},
            "tpu-0.svc:8476",
            2,
            1,
        ),
        # MegaScale coordinator + per-host TPU variables (JobSet auto-injection).
        (
            {
                "MEGASCALE_COORDINATOR_ADDRESS": "tpu-0.svc",
                "TPU_WORKER_HOSTNAMES": "tpu-0.svc,tpu-1.svc",
                "TPU_WORKER_ID": "0",
            },
            "tpu-0.svc:8476",
            2,
            0,
        ),
        # Multislice: 2 slices x 2 hosts; worker 1 of slice 1 is global process 3. The
        # MegaScale transport port on the coordinator host is never reused for JAX.
        (
            {
                "MEGASCALE_COORDINATOR_ADDRESS": "s0w0:8080",
                "MEGASCALE_NUM_SLICES": "2",
                "MEGASCALE_SLICE_ID": "1",
                "TPU_WORKER_HOSTNAMES": "s1w0,s1w1",
                "TPU_WORKER_ID": "1",
            },
            "s0w0:8476",
            4,
            3,
        ),
    ],
)
def test_detect_gke_jobset_tpu_megascale(
    env: dict[str, str], coordinator: str, num_processes: int, process_id: int
) -> None:
    for distributed in (None, True):
        cfg = detect_distributed_env(distributed, env=env)
        assert cfg.enabled is True and cfg.source == "gke"
        assert cfg.coordinator_address == coordinator
        assert cfg.num_processes == num_processes
        assert cfg.process_id == process_id
        assert cfg.local_device_ids is None
        assert cfg.initialize_kwargs() == {
            "coordinator_address": coordinator,
            "num_processes": num_processes,
            "process_id": process_id,
        }


def test_detect_gke_port_override_and_malformed_values() -> None:
    env = {"TPU_WORKER_HOSTNAMES": "tpu-0,tpu-1", "TPU_WORKER_ID": "0"}
    assert detect_distributed_env(env=env).coordinator_address == "tpu-0:8476"
    for key in ("GX_JAX_COORDINATOR_PORT", "JAX_COORDINATOR_PORT"):
        cfg = detect_distributed_env(env={**env, key: "9123"})
        assert cfg.coordinator_address == "tpu-0:9123"
    assert detect_distributed_env(env=env, default_port=7000).coordinator_address == "tpu-0:7000"

    with pytest.raises(ValueError, match="TPU_WORKER_ID"):
        detect_distributed_env(env={**env, "TPU_WORKER_ID": "zero"})
    # Worker id outside the host list is rejected instead of hanging at connect time.
    with pytest.raises(ValueError, match="out of range"):
        detect_distributed_env(env={**env, "TPU_WORKER_ID": "2"})
    # A coordinator without a resolvable process count / id fails fast.
    with pytest.raises(ValueError, match="num_processes and process_id"):
        detect_distributed_env(env={"MEGASCALE_COORDINATOR_ADDRESS": "h0"})

    # Single-host Cloud TPU with an explicit --distributed: clean 1-process runtime.
    cfg_one = detect_distributed_env(
        True, env={"TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost"}
    )
    assert cfg_one.enabled and cfg_one.source == "gke"
    assert cfg_one.initialize_kwargs() == {
        "coordinator_address": "localhost:8476",
        "num_processes": 1,
        "process_id": 0,
    }


def test_detect_cloud_batch_hosts_file(tmp_path: Path) -> None:
    hosts_file = tmp_path / "batch_hosts"
    hosts_file.write_text("tpu-vm-a\n\ntpu-vm-b  \ntpu-vm-a\n")  # blank + duplicate lines
    env = {"BATCH_HOSTS_FILE": str(hosts_file), "BATCH_TASK_INDEX": "1", "BATCH_TASK_COUNT": "2"}

    for distributed in (None, True):
        cfg = detect_distributed_env(distributed, env=env)
        assert cfg.enabled is True and cfg.source == "cloud-batch"
        assert cfg.coordinator_address == f"tpu-vm-a:{DEFAULT_COORDINATOR_PORT}"
        assert cfg.num_processes == 2 and cfg.process_id == 1

    cfg_port = detect_distributed_env(env={**env, "GX_JAX_COORDINATOR_PORT": "9001"})
    assert cfg_port.coordinator_address == "tpu-vm-a:9001"
    cfg_p0 = detect_distributed_env(env={**env, "BATCH_TASK_INDEX": "0"})
    assert cfg_p0.process_id == 0 and cfg_p0.num_processes == 2

    # Multi-host job without BATCH_TASK_INDEX cannot resolve the process id.
    with pytest.raises(ValueError, match="process_id"):
        detect_distributed_env(env={"BATCH_HOSTS_FILE": str(hosts_file)})
    # An unreadable hosts file never breaks single-host auto mode; --distributed reports it.
    missing_env = {"BATCH_HOSTS_FILE": str(tmp_path / "missing"), "BATCH_TASK_INDEX": "0"}
    assert detect_distributed_env(env=missing_env).enabled is False
    with pytest.raises(ValueError, match="BATCH_HOSTS_FILE"):
        detect_distributed_env(True, env=missing_env)

    # One VM running several tasks: auto mode stays single-host (independent campaign
    # array tasks), an explicit --distributed with BATCH_TASK_COUNT > 1 connects them.
    single = tmp_path / "batch_hosts_single"
    single.write_text("tpu-vm-a\n")
    env_single = {
        "BATCH_HOSTS_FILE": str(single),
        "BATCH_TASK_INDEX": "1",
        "BATCH_TASK_COUNT": "2",
    }
    assert detect_distributed_env(env=env_single).enabled is False
    cfg_forced = detect_distributed_env(True, env=env_single)
    assert cfg_forced.enabled and cfg_forced.source == "cloud-batch"
    assert cfg_forced.initialize_kwargs() == {
        "coordinator_address": "tpu-vm-a:8476",
        "num_processes": 2,
        "process_id": 1,
    }


def test_detect_slurm_multi_task_step() -> None:
    env = {
        "SLURM_JOB_ID": "123456",
        "SLURM_NTASKS": "4",
        "SLURM_PROCID": "2",
        "SLURM_LOCALID": "0",
        "SLURM_STEP_NODELIST": "gpu[017-018]",
    }
    slurm_port = 123456 % 2**12 + (65535 - 2**12 + 1)
    for distributed in (None, True):
        cfg = detect_distributed_env(distributed, env=env)
        assert cfg.enabled is True and cfg.source == "slurm"
        assert cfg.coordinator_address == f"gpu017:{slurm_port}"
        assert cfg.num_processes == 4 and cfg.process_id == 2
        assert cfg.local_device_ids == (0,)
        # Fully resolved here: nothing is left to JAX's own cluster detection.
        assert cfg.initialize_kwargs() == {
            "coordinator_address": f"gpu017:{slurm_port}",
            "num_processes": 4,
            "process_id": 2,
            "local_device_ids": [0],
        }

    # Node-list formats handled like jax._src.clusters.slurm_cluster.
    for node_list, head in (
        ("node001", "node001"),
        ("node001,host2", "node001"),
        ("node[001-015],host2", "node001"),
        ("node[001,007-015],host2", "node001"),
    ):
        cfg = detect_distributed_env(env={**env, "SLURM_STEP_NODELIST": node_list})
        assert cfg.coordinator_address == f"{head}:{slurm_port}", node_list

    # Port override and explicit CLI device ids win over SLURM_LOCALID.
    cfg_port = detect_distributed_env(env={**env, "JAX_COORDINATOR_PORT": "8600"})
    assert cfg_port.coordinator_address == "gpu017:8600"
    cfg_ids = detect_distributed_env(env=env, local_device_ids="0,1")
    assert cfg_ids.local_device_ids == (0, 1)

    # `sbatch --ntasks=4` running ONE process without `srun` (no SLURM_STEP_NODELIST; a
    # job-level node list proves nothing): auto mode is the single-host no-op ...
    cfg_batch = detect_distributed_env(env=_SBATCH_WITHOUT_SRUN_ENV)
    assert cfg_batch.enabled is False and cfg_batch.source == "single-host"
    assert cfg_batch.initialize_kwargs() == {}
    # ... an explicit --distributed reports a clean, actionable error instead of
    # delegating to (and crashing inside) JAX's SLURM detection ...
    with pytest.raises(ValueError, match="SLURM_STEP_NODELIST is unset.*srun"):
        detect_distributed_env(True, env=_SBATCH_WITHOUT_SRUN_ENV)
    # ... and explicit flags still work, reusing SLURM_NTASKS / SLURM_PROCID.
    cfg_cli = detect_distributed_env(
        True, coordinator_address="n1:9000", env={**_SBATCH_WITHOUT_SRUN_ENV, "SLURM_PROCID": "3"}
    )
    assert cfg_cli.enabled and cfg_cli.source == "cli"
    assert cfg_cli.initialize_kwargs() == {
        "coordinator_address": "n1:9000",
        "num_processes": 4,
        "process_id": 3,
    }

    # SLURM_NTASKS == 1 is a single-host no-op; --distributed forces a 1-process runtime.
    env_one = {**env, "SLURM_NTASKS": "1", "SLURM_PROCID": "0"}
    assert detect_distributed_env(env=env_one).enabled is False
    cfg_one = detect_distributed_env(True, env=env_one)
    assert cfg_one.enabled and cfg_one.num_processes == 1 and cfg_one.local_device_ids is None


def test_detect_explicit_cli_overrides_and_validation() -> None:
    cfg = detect_distributed_env(
        coordinator_address="10.1.2.3", num_processes=2, process_id=1, env={}
    )
    assert cfg.enabled is True and cfg.source == "cli"
    assert cfg.initialize_kwargs() == {
        "coordinator_address": "10.1.2.3:8476",
        "num_processes": 2,
        "process_id": 1,
    }

    # Explicit flags override every environment value, field by field.
    gke_env = {
        "MEGASCALE_COORDINATOR_ADDRESS": "h0",
        "MEGASCALE_NUM_PROCESSES": "4",
        "MEGASCALE_PROCESS_ID": "2",
    }
    cfg_all = detect_distributed_env(
        True,
        coordinator_address="127.0.0.1:5555",
        num_processes=2,
        process_id=0,
        local_device_ids=[1, 0],
        env=gke_env,
    )
    assert cfg_all.source == "cli"
    assert cfg_all.initialize_kwargs() == {
        "coordinator_address": "127.0.0.1:5555",
        "num_processes": 2,
        "process_id": 0,
        "local_device_ids": [1, 0],
    }
    cfg_partial = detect_distributed_env(process_id=3, env=gke_env)
    assert cfg_partial.source == "gke" and cfg_partial.coordinator_address == "h0:8476"
    assert cfg_partial.num_processes == 4 and cfg_partial.process_id == 3
    assert detect_distributed_env(env=gke_env, local_device_ids=2).local_device_ids == (2,)

    # --distributed with nothing resolvable: delegated to JAX native cluster detection.
    cfg_auto = detect_distributed_env(True, env={})
    assert cfg_auto.enabled and cfg_auto.source == "auto" and cfg_auto.initialize_kwargs() == {}

    # Address normalization (default port, explicit port, IPv6).
    assert normalize_coordinator_address("host") == "host:8476"
    assert normalize_coordinator_address("host:") == "host:8476"
    assert normalize_coordinator_address("host:9") == "host:9"
    assert normalize_coordinator_address("::1") == "[::1]:8476"
    assert normalize_coordinator_address("[fe80::1]") == "[fe80::1]:8476"
    assert normalize_coordinator_address("[::1]:9000", 1) == "[::1]:9000"
    assert normalize_coordinator_address("host", 9100) == "host:9100"
    for bad in ("", "host:port", ":8476", "[::1", "[::1]x"):
        with pytest.raises(ValueError, match="coordinator address"):
            normalize_coordinator_address(bad)

    with pytest.raises(ValueError, match="out of range"):
        detect_distributed_env(coordinator_address="h:1", num_processes=2, process_id=2, env={})
    with pytest.raises(ValueError, match="num_processes=0"):
        detect_distributed_env(coordinator_address="h:1", num_processes=0, process_id=0, env={})
    with pytest.raises(ValueError, match="process_id=-1"):
        detect_distributed_env(coordinator_address="h:1", num_processes=2, process_id=-1, env={})
    with pytest.raises(ValueError, match="num_processes and process_id"):
        detect_distributed_env(coordinator_address="h:1", env={})
    with pytest.raises(ValueError, match="--coordinator-address"):
        detect_distributed_env(num_processes=2, process_id=0, env={})
    with pytest.raises(ValueError, match="local device ids"):
        detect_distributed_env(True, local_device_ids="0,x", env={})


@pytest.mark.parametrize("build_parser", [build_arg_parser, build_ensemble_arg_parser])
def test_cli_distributed_flags_on_both_parsers(build_parser: Any) -> None:
    parser = build_parser()
    base = ["deck.in"]
    defaults = parser.parse_args(base)
    assert defaults.distributed is None
    assert defaults.coordinator_address is None
    assert defaults.num_processes is None
    assert defaults.process_id is None
    assert defaults.local_device_ids is None

    assert parser.parse_args([*base, "--distributed"]).distributed is True
    assert parser.parse_args([*base, "--no-distributed"]).distributed is False
    args = parser.parse_args(
        [
            *base,
            "--distributed",
            "--coordinator-address",
            "10.0.0.5:8476",
            "--num-processes",
            "2",
            "--process-id",
            "1",
            "--local-device-ids",
            "0,1",
        ]
    )
    assert args.coordinator_address == "10.0.0.5:8476"
    assert args.num_processes == 2 and args.process_id == 1
    cfg = detect_distributed_env(
        args.distributed,
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=args.local_device_ids,
        env={},
    )
    assert cfg.initialize_kwargs() == {
        "coordinator_address": "10.0.0.5:8476",
        "num_processes": 2,
        "process_id": 1,
        "local_device_ids": [0, 1],
    }
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--num-processes", "two"])


# =============================================================================
# 2. Runtime Lifecycle: no-op fast path, idempotency, ownership
# =============================================================================


class _FakeJaxDistributed:
    """Records `jax.distributed` calls without starting a real coordination service."""

    def __init__(self) -> None:
        self.initialized = False
        self.init_calls: list[dict[str, Any]] = []
        self.shutdown_calls = 0
        self.bind_calls: list[tuple[str | None, str | None]] = []

    def initialize(
        self,
        coordinator_address: str | None = None,
        num_processes: int | None = None,
        process_id: int | None = None,
        local_device_ids: Any = None,
        cluster_detection_method: str | None = None,
        initialization_timeout: int = 300,
        heartbeat_timeout_seconds: int = 100,
        shutdown_timeout_seconds: int = 300,
        coordinator_bind_address: str | None = None,
    ) -> None:
        if coordinator_address is None and cluster_detection_method is None:
            raise ValueError("coordinator_address should be defined.")
        self.init_calls.append(
            {
                "coordinator_address": coordinator_address,
                "num_processes": num_processes,
                "process_id": process_id,
                "local_device_ids": local_device_ids,
                "cluster_detection_method": cluster_detection_method,
                "initialization_timeout": initialization_timeout,
                "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
                "shutdown_timeout_seconds": shutdown_timeout_seconds,
                "coordinator_bind_address": coordinator_bind_address,
                # Precision flags as seen by jax.distributed.initialize itself.
                "x64": bool(jax.config.jax_enable_x64),
                "matmul": jax.config.jax_default_matmul_precision,
                "no_proxy": os.environ.get("no_proxy"),
            }
        )
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.initialized = False


@pytest.fixture
def fake_jax_distributed(monkeypatch: pytest.MonkeyPatch) -> _FakeJaxDistributed:
    fake = _FakeJaxDistributed()
    monkeypatch.setattr(jax.distributed, "initialize", fake.initialize)
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: fake.initialized)
    monkeypatch.setattr(jax.distributed, "shutdown", fake.shutdown)
    monkeypatch.setattr(
        gxd,
        "_bind_cpu_collectives",
        lambda host, iface: fake.bind_calls.append((host, iface)) or True,
    )
    for key in list(os.environ):
        if key.startswith(_CLUSTER_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # `init_distributed_runtime` extends no_proxy in os.environ: register both spellings
    # with monkeypatch so the original values are restored after each test.
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.delenv("no_grpc_proxy", raising=False)
    return fake


def test_single_host_runtime_is_zero_overhead_noop(
    fake_jax_distributed: _FakeJaxDistributed,
) -> None:
    """`distributed=False` / single-host `None` never call into `jax.distributed`."""
    cases: tuple[tuple[bool | None, dict[str, str]], ...] = (
        (False, {"SLURM_NTASKS": "8", "SLURM_STEP_NODELIST": "n[1-8]", "SLURM_PROCID": "1"}),
        (None, {}),
        (None, _SINGLE_HOST_TPU_ENV),
        (None, _SBATCH_WITHOUT_SRUN_ENV),
    )
    for distributed, env in cases:
        info = init_distributed_runtime(distributed, env=env)
        assert info.initialized is False and info.enabled is False
        assert info.initialized_by_gx_jax is False
        assert (info.process_index, info.process_count) == (0, 1)
        assert info.is_primary and not info.is_multi_process
        assert info.local_device_count is None and info.global_device_count is None
        assert info.source == ("disabled" if distributed is False else "single-host")
    assert fake_jax_distributed.init_calls == []
    assert fake_jax_distributed.bind_calls == []
    assert not is_distributed() and is_primary_process()
    assert (process_index(), process_count()) == (0, 1)
    # Nothing to tear down: shutdown is a safe no-op, repeatedly.
    assert shutdown_distributed_runtime() is False
    assert shutdown_distributed_runtime() is False
    assert fake_jax_distributed.shutdown_calls == 0
    assert jax.config.jax_enable_x64 is True
    assert jax.config.jax_default_matmul_precision == "highest"


def test_runtime_init_enforces_precision_idempotency_and_ownership(
    fake_jax_distributed: _FakeJaxDistributed, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_jax_distributed
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:3128")
    monkeypatch.setenv("no_proxy", "example.org")
    monkeypatch.setenv("GX_JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT", "45")

    # Precision flags must be (re-)enforced *before* jax.distributed.initialize runs.
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", None)
    try:
        info = init_distributed_runtime(
            None,
            coordinator_address="127.0.0.1:45678",
            num_processes=2,
            process_id=0,
            initialization_timeout=30,
        )
    finally:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")

    assert len(fake.init_calls) == 1
    call = fake.init_calls[0]
    assert call["x64"] is True and call["matmul"] == "highest"
    assert call["coordinator_address"] == "127.0.0.1:45678"
    assert call["num_processes"] == 2 and call["process_id"] == 0
    assert call["initialization_timeout"] == 30  # explicit argument
    assert call["heartbeat_timeout_seconds"] == 45  # environment knob
    assert call["shutdown_timeout_seconds"] == 300  # JAX default untouched
    # Loopback coordinator: bind locally, bypass HTTP proxies, pin CPU collectives.
    assert call["coordinator_bind_address"] == "127.0.0.1:45678"
    assert call["no_proxy"] == "example.org,127.0.0.1"
    assert os.environ["NO_PROXY"] == "127.0.0.1"
    assert "no_grpc_proxy" not in os.environ
    assert fake.bind_calls == [("127.0.0.1", None)]
    assert info.initialized and info.initialized_by_gx_jax
    assert info.source == "cli" and info.coordinator_address == "127.0.0.1:45678"
    assert info.global_device_count == jax.device_count()

    # Idempotent: an already-initialized runtime is never re-initialized.
    again = init_distributed_runtime(
        True, coordinator_address="127.0.0.1:45678", num_processes=2, process_id=0
    )
    assert len(fake.init_calls) == 1 and again.initialized_by_gx_jax
    assert init_distributed_runtime(False).initialized is True  # live state is reported
    assert len(fake.init_calls) == 1

    # Owned runtime: shut down exactly once.
    assert shutdown_distributed_runtime() is True
    assert fake.shutdown_calls == 1 and fake.initialized is False
    assert shutdown_distributed_runtime() is False
    assert fake.shutdown_calls == 1


def test_runtime_never_shuts_down_externally_initialized_jax(
    fake_jax_distributed: _FakeJaxDistributed,
) -> None:
    fake = fake_jax_distributed
    fake.initialized = True  # the embedding application called jax.distributed.initialize()
    info = init_distributed_runtime(
        True, coordinator_address="10.0.0.1:8476", num_processes=2, process_id=1
    )
    assert info.initialized is True and info.initialized_by_gx_jax is False
    assert fake.init_calls == [] and fake.bind_calls == []
    assert shutdown_distributed_runtime() is False
    assert fake.shutdown_calls == 0 and fake.initialized is True


def test_runtime_remote_coordinator_and_unresolvable_environment(
    fake_jax_distributed: _FakeJaxDistributed, tmp_path: Path
) -> None:
    fake = fake_jax_distributed
    # --distributed with nothing resolvable: JAX native detection fails -> actionable error.
    with pytest.raises(ValueError, match="--coordinator-address"):
        init_distributed_runtime(True, env={})
    # --distributed inside a multi-task SLURM allocation but outside an `srun` step.
    with pytest.raises(ValueError, match="srun"):
        init_distributed_runtime(True, env=_SBATCH_WITHOUT_SRUN_ENV)
    assert fake.init_calls == [] and shutdown_distributed_runtime() is False
    assert not is_distributed() and is_primary_process()

    hosts_file = tmp_path / "hosts"
    hosts_file.write_text("tpu-vm-a\ntpu-vm-b\n")
    info = init_distributed_runtime(
        None, env={"BATCH_HOSTS_FILE": str(hosts_file), "BATCH_TASK_INDEX": "1"}
    )
    assert info.source == "cloud-batch" and info.initialized_by_gx_jax
    call = fake.init_calls[0]
    assert call["coordinator_address"] == "tpu-vm-a:8476"
    assert call["num_processes"] == 2 and call["process_id"] == 1
    # Remote coordinator: JAX's default bind address and collectives are left alone.
    assert call["coordinator_bind_address"] is None
    assert fake.bind_calls == []
    assert shutdown_distributed_runtime() is True


def test_abort_and_sync_are_noops_on_single_host_and_fail_fast_when_owned(
    fake_jax_distributed: _FakeJaxDistributed,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exits: list[int] = []

    class _Exited(BaseException):
        pass

    def _fake_exit(code: int) -> None:
        exits.append(code)
        raise _Exited

    monkeypatch.setattr(os, "_exit", _fake_exit)

    # Single host / no runtime: both helpers return and the caller re-raises normally.
    sync_processes("noop")
    assert abort_distributed_run(RuntimeError("boom")) is None
    # Multi-process but initialized by the embedding application: never hard-exit.
    monkeypatch.setattr(gxd, "_process_counts", lambda: (1, 2))
    assert abort_distributed_run(RuntimeError("boom")) is None
    assert exits == []

    # Multi-process runtime owned by gx_jax: print the traceback and exit immediately
    # (no wait at the jax.distributed shutdown barrier).
    gxd._STATE.owned = True
    assert abort_distributed_run(SystemExit(0)) is None  # clean exit is not a failure
    assert abort_distributed_run(SystemExit(None)) is None
    try:
        raise OSError("disk full on the primary host")
    except OSError as exc:
        with pytest.raises(_Exited):
            abort_distributed_run(exc)
    with pytest.raises(_Exited):
        abort_distributed_run(SystemExit(2))
    with pytest.raises(_Exited):
        abort_distributed_run(KeyboardInterrupt())
    assert exits == [1, 2, 130]
    err = capsys.readouterr().err
    assert "OSError: disk full on the primary host" in err
    assert "process 1 of 2 failed; aborting the multi-host run" in err
    assert fake_jax_distributed.shutdown_calls == 0


# =============================================================================
# 3. Global Array Helpers: single-host identity & multi-host placement rule
# =============================================================================


def _simulate_multihost(monkeypatch: pytest.MonkeyPatch, *, process_id: int) -> None:
    """Make `gx_jax` take its multi-host branches inside this single pytest process.

    `jax.device_put` onto a mesh sharding and `jax.device_get` of a mesh-placed array are
    turned into hard errors, emulating the JAX rule that neither may touch non-addressable
    devices when `jax.process_count() > 1`.
    """
    monkeypatch.setattr(gxd, "_process_counts", lambda: (process_id, 2))
    real_put, real_get = jax.device_put, jax.device_get

    def _strict_put(x: Any, device: Any = None, *args: Any, **kwargs: Any) -> Any:
        if any(isinstance(t, NamedSharding) for t in jax.tree_util.tree_leaves(device)):
            raise ValueError("device_put cannot target non-addressable devices (multi-host)")
        return real_put(x, device, *args, **kwargs)

    def _strict_get(x: Any) -> Any:
        for leaf in jax.tree_util.tree_leaves(x):
            if isinstance(leaf, jax.Array) and isinstance(leaf.sharding, NamedSharding):
                raise RuntimeError("device_get of a non-addressable array (multi-host)")
        return real_get(x)

    monkeypatch.setattr(jax, "device_put", _strict_put)
    monkeypatch.setattr(jax, "device_get", _strict_get)
    # The single in-process device can never span two processes.
    monkeypatch.setattr(cli_mod, "validate_multihost_mesh", lambda mesh: None)
    monkeypatch.setattr(ens_mod, "validate_multihost_mesh", lambda mesh: None)


def test_global_array_helpers_single_host_identity_and_multihost_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = create_device_mesh((1, 1))
    g_sh, rep_sh = make_g_sharding(mesh), make_replicated_sharding(mesh)
    rng = np.random.default_rng(8)
    g_shape = (2, 4, 2, 3, 3, 2)
    g_host = (rng.normal(size=g_shape) + 1j * rng.normal(size=g_shape)).astype(np.complex64)
    tree = {"g": jnp.asarray(g_host), "t": jnp.asarray(0.25, dtype=jnp.float64), "none": None}

    # Single host: exactly jax.device_put / jax.device_get (Session 7 paths).
    assert process_count() == 1 and not is_distributed()
    placed = device_put_global(tree["g"], g_sh)
    ref = jax.device_put(tree["g"], g_sh)
    assert placed.sharding == ref.sharding and placed.dtype == ref.dtype
    assert np.array_equal(np.asarray(placed), np.asarray(ref))
    assert np.array_equal(np.asarray(shard_g_state(tree["g"], mesh)), g_host)
    gathered = gather_g_state(placed, mesh)
    assert isinstance(gathered, jax.Array) and gathered.sharding.is_equivalent_to(rep_sh, 6)
    host_tree = gather_global_tree(tree)
    assert isinstance(host_tree["g"], np.ndarray) and host_tree["none"] is None
    assert np.array_equal(host_tree["g"], g_host) and host_tree["t"] == 0.25
    assert host_value(tree["t"]) is tree["t"]  # untouched device scalar on a single host
    validate_multihost_mesh(mesh)  # no-op on a single host

    # Multi-host branch: never jax.device_put(..., NamedSharding) / jax.device_get.
    _simulate_multihost(monkeypatch, process_id=1)
    assert is_distributed() and not is_primary_process()
    assert (process_index(), process_count()) == (1, 2)
    with pytest.raises(ValueError, match="non-addressable"):
        jax.device_put(tree["g"], g_sh)
    placed_mh = device_put_global(tree, rep_sh)
    assert placed_mh["none"] is None
    for key in ("g", "t"):
        assert isinstance(placed_mh[key].sharding, NamedSharding)
        assert placed_mh[key].dtype == tree[key].dtype
        assert np.array_equal(np.asarray(placed_mh[key]), np.asarray(tree[key]))
    g_mh = shard_g_state(tree["g"], mesh)
    assert g_mh.sharding.is_equivalent_to(g_sh, 6) and g_mh.dtype == jnp.complex64
    gathered_mh = gather_g_state(g_mh, mesh)
    assert isinstance(gathered_mh, jax.Array) and gathered_mh.is_fully_addressable
    assert np.array_equal(np.asarray(gathered_mh), g_host)
    assert np.array_equal(gather_global_array(g_mh), g_host)
    host_mh = gather_global_tree({"g": g_mh, "t": placed_mh["t"], "none": None})
    assert isinstance(host_mh["g"], np.ndarray) and host_mh["none"] is None
    assert isinstance(host_value(placed_mh["t"]), np.ndarray)
    assert float(host_value(placed_mh["t"])) == 0.25

    # Ensemble tree placement / gather follow the same rule.
    ens_mesh = create_ensemble_mesh(2)
    ens_tree = {"batched": jnp.arange(6.0).reshape(2, 3), "shared": jnp.arange(3.0), "tag": "x"}
    ens_placed = shard_ensemble_tree(ens_tree, ens_mesh, batch_size=2)
    assert ens_placed["tag"] == "x"
    assert ens_placed["batched"].sharding.spec == PartitionSpec("batch", None)
    assert ens_placed["shared"].sharding.spec == PartitionSpec()
    ens_back = gather_ensemble_tree(ens_placed, ens_mesh)
    assert np.array_equal(np.asarray(ens_back["batched"]), np.arange(6.0).reshape(2, 3))
    assert np.array_equal(np.asarray(ens_back["shared"]), np.arange(3.0))

    # A mesh that leaves a process without devices is rejected up front.
    with pytest.raises(ValueError, match="spanning every process"):
        validate_multihost_mesh(mesh)


# =============================================================================
# 4. Lockstep Collective Safety: non-primary ranks compute but never touch disk
# =============================================================================


def test_cli_non_primary_rank_runs_lockstep_without_disk_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deck = _write_deck(tmp_path / "decks" / "lockstep.in", fixed_dt=False, nonlinear=False)
    kwargs: dict[str, Any] = {"mesh_shape": (1, 1), "precision": "fp64", "async_io": True}

    primary_dir = tmp_path / "primary"
    res_primary = run_cli_simulation(
        deck,
        output_path=primary_dir / "run.out.nc",
        metrics_json_path=primary_dir / "metrics.json",
        **kwargs,
    )
    assert _files_under(primary_dir) == ["lockstep.restart.nc", "metrics.json", "run.out.nc"]

    # Simulated non-primary rank of a 2-process run: identical lockstep computation
    # (stepper, diagnostics, gathers), zero files opened / truncated / written.
    _simulate_multihost(monkeypatch, process_id=1)
    rank1_dir = tmp_path / "rank1"
    res_rank1 = run_cli_simulation(
        deck,
        output_path=rank1_dir / "run.out.nc",
        metrics_json_path=rank1_dir / "metrics.json",
        **kwargs,
    )
    assert not rank1_dir.exists()
    assert res_rank1["kr2_summary"] is None and res_primary["kr2_summary"] is not None
    assert res_rank1["nstep"] == res_primary["nstep"] == 6
    assert res_rank1["t_final"] == res_primary["t_final"]
    assert res_rank1["g_final"].is_fully_addressable
    assert np.array_equal(np.asarray(res_rank1["g_final"]), np.asarray(res_primary["g_final"]))
    for name in ("phi", "apar", "bpar"):
        assert np.array_equal(
            np.asarray(getattr(res_rank1["fields_final"], name)),
            np.asarray(getattr(res_primary["fields_final"], name)),
        )


# =============================================================================
# 5. Real 2-Process CPU Loopback: `gx-jax --mesh 2x2` (2 processes x 2 devices)
# =============================================================================


@pytest.fixture(scope="module")
def strict_launcher(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("strict") / "strict_multihost_launcher.py"
    path.write_text(_STRICT_LAUNCHER)
    return path


@pytest.mark.parametrize(
    ("precision", "tol", "fixed_dt", "strict"),
    [
        ("fp64", 1.0e-12, True, False),
        ("fp64", 1.0e-12, False, True),
        ("fp32", 1.0e-6, True, True),
        ("fp32", 1.0e-6, False, False),
    ],
    ids=["fp64-fixed_dt", "fp64-adaptive_cfl-strict", "fp32-fixed_dt-strict", "fp32-adaptive_cfl"],
)
def test_two_process_cli_mesh_2x2_matches_single_process_4_devices(
    tmp_path: Path,
    strict_launcher: Path,
    precision: str,
    tol: float,
    fixed_dt: bool,
    strict: bool,
) -> None:
    """2 processes x 2 CPU devices == 1 process x 4 CPU devices for `gx-jax --mesh 2x2`."""
    deck = _write_deck(tmp_path / "decks" / "kbm.in", fixed_dt=fixed_dt)
    # Identical *relative* output arguments on every process, as on a real cluster; each
    # process gets its own working directory so every written file is attributable.
    run_args = [
        str(deck),
        "--mesh",
        "2x2",
        "--precision",
        precision,
        "-o",
        "out/kbm.out.nc",
        "--metrics-json",
        "out/metrics.json",
    ]
    if precision == "fp32" and fixed_dt:
        # Also exercise the sharded throughput benchmark in lockstep on every process.
        run_args += ["--benchmark", "--benchmark-steps", "1"]
    port = _free_loopback_port()
    launcher = strict_launcher if strict else None
    jobs = {
        f"rank{pid}": (
            _module_cmd(
                "gx_jax.cli",
                [*run_args, *_distributed_flags(port, pid)],
                strict_launcher=launcher,
            ),
            _child_env(2),
            tmp_path / f"rank{pid}",
        )
        for pid in (0, 1)
    }
    # Reference: 1 process x 4 devices with auto-detection left on inside a single-host
    # TPU-like environment (must stay a no-op and never wait for a coordinator).
    jobs["reference"] = (
        _module_cmd("gx_jax.cli", run_args, strict_launcher=None),
        _child_env(4, **_SINGLE_HOST_TPU_ENV),
        tmp_path / "reference",
    )
    _run_process_group(jobs, tmp_path / "logs")

    expected = ["out/kbm.out.nc", "out/kbm.restart.nc", "out/metrics.json"]
    # Zero duplicate-file races: only the primary process wrote anything at all, and no
    # temp file of an interrupted / concurrent atomic restart write is left behind.
    assert _files_under(tmp_path / "rank0") == expected
    assert _files_under(tmp_path / "rank1") == []
    assert not (tmp_path / "rank1" / "out").exists()
    assert _files_under(tmp_path / "reference") == expected
    logs = {name: (tmp_path / "logs" / f"{name}.log").read_text() for name in jobs}
    if strict:
        assert all("STRICT-MULTIHOST guards:" in logs[f"rank{pid}"] for pid in (0, 1))
    if "--benchmark" in run_args:
        # Only the primary process reports the benchmark.
        assert "[gx-jax benchmark]" in logs["rank0"] and "[gx-jax benchmark]" in logs["reference"]
        assert "[gx-jax benchmark]" not in logs["rank1"]

    dist_out, ref_out = tmp_path / "rank0" / "out", tmp_path / "reference" / "out"
    n_frames = _assert_strictly_increasing_time(dist_out / "kbm.out.nc")
    assert n_frames == _assert_strictly_increasing_time(ref_out / "kbm.out.nc") == 3
    _assert_nc_close(dist_out / "kbm.out.nc", ref_out / "kbm.out.nc", tol, f"{precision} .out.nc")
    _assert_nc_close(
        dist_out / "kbm.restart.nc", ref_out / "kbm.restart.nc", tol, f"{precision} .restart.nc"
    )
    metrics_dist = _load_strict_json(dist_out / "metrics.json")
    metrics_ref = _load_strict_json(ref_out / "metrics.json")
    assert metrics_dist["nstep"] == metrics_ref["nstep"] == 6
    assert metrics_dist["precision"] == precision
    _assert_json_close(metrics_dist, metrics_ref, tol, "metrics.json")


# =============================================================================
# 6. Real 2-Process CPU Loopback: `gx-jax-ensemble` (B=4 over 4 global devices)
# =============================================================================


@pytest.mark.parametrize(
    ("precision", "tol", "strict"),
    [("fp64", 1.0e-12, False), ("fp32", 1.0e-5, True)],
    ids=["fp64", "fp32-strict"],
)
def test_two_process_ensemble_b4_matches_single_process_4_devices(
    tmp_path: Path, strict_launcher: Path, precision: str, tol: float, strict: bool
) -> None:
    """2 processes x 2 devices == 1 process x 4 devices for a B=4 `'batch'`-mesh ensemble."""
    decks = _write_ensemble_decks(tmp_path / "decks", fixed_dt=False)
    run_args = [*(str(d) for d in decks), "--precision", precision, "-o", "out"]
    port = _free_loopback_port()
    launcher = strict_launcher if strict else None
    jobs = {
        f"rank{pid}": (
            _module_cmd(
                "gx_jax.ensemble",
                [*run_args, *_distributed_flags(port, pid)],
                strict_launcher=launcher,
            ),
            _child_env(2),
            tmp_path / f"rank{pid}",
        )
        for pid in (0, 1)
    }
    jobs["reference"] = (
        _module_cmd("gx_jax.ensemble", run_args, strict_launcher=None),
        _child_env(4),
        tmp_path / "reference",
    )
    _run_process_group(jobs, tmp_path / "logs")

    stems = [d.stem for d in decks]
    suffixes = (".diagnostics_metrics.json", ".out.nc", ".restart.nc")
    expected = sorted(
        [f"out/{stem}{suffix}" for stem in stems for suffix in suffixes]
        + ["out/ensemble_summary.json"]
    )
    assert _files_under(tmp_path / "rank0") == expected
    assert _files_under(tmp_path / "rank1") == []
    assert _files_under(tmp_path / "reference") == expected

    dist_out, ref_out = tmp_path / "rank0" / "out", tmp_path / "reference" / "out"
    expected_steps = [6, 5, 6, 4]
    for stem, n_steps in zip(stems, expected_steps, strict=True):
        _assert_strictly_increasing_time(dist_out / f"{stem}.out.nc")
        _assert_nc_close(
            dist_out / f"{stem}.out.nc", ref_out / f"{stem}.out.nc", tol, f"{stem}.out.nc"
        )
        _assert_nc_close(
            dist_out / f"{stem}.restart.nc",
            ref_out / f"{stem}.restart.nc",
            tol,
            f"{stem}.restart.nc",
        )
        m_dist = _load_strict_json(dist_out / f"{stem}.diagnostics_metrics.json")
        m_ref = _load_strict_json(ref_out / f"{stem}.diagnostics_metrics.json")
        assert m_dist["nstep"] == m_ref["nstep"] == n_steps
        _assert_json_close(m_dist, m_ref, tol, f"{stem}.diagnostics_metrics.json")

    summary_dist = _load_strict_json(dist_out / "ensemble_summary.json")
    summary_ref = _load_strict_json(ref_out / "ensemble_summary.json")
    assert summary_dist["batch_size"] == 4 and summary_dist["num_devices"] == 4
    assert summary_dist["mesh_shape"] == {"batch": 4}
    assert summary_dist["total_steps_sum"] == sum(expected_steps)
    _assert_json_close(summary_dist, summary_ref, tol, "ensemble_summary.json")


# =============================================================================
# 6b. Failure Modes: every process exits promptly, nobody deadlocks in a collective
# =============================================================================


def test_two_process_input_mismatch_fails_fast_on_every_process(tmp_path: Path) -> None:
    """Decks that differ between hosts (non-shared storage) abort all processes at once."""
    for pid, nstep in ((0, 6), (1, 5)):
        _write_deck(tmp_path / f"rank{pid}" / "deck.in", fixed_dt=True, nstep=nstep)
    run_args = ["deck.in", "--mesh", "2x2", "-o", "out/run.out.nc"]
    port = _free_loopback_port()
    jobs = {
        f"rank{pid}": (
            _module_cmd(
                "gx_jax.cli", [*run_args, *_distributed_flags(port, pid)], strict_launcher=None
            ),
            _child_env(2),
            tmp_path / f"rank{pid}",
        )
        for pid in (0, 1)
    }
    t_start = time.monotonic()
    codes = _run_process_group(jobs, tmp_path / "logs", expect_success=False)
    elapsed = time.monotonic() - t_start
    assert codes == {"rank0": 1, "rank1": 1}, codes
    assert elapsed < SUBPROCESS_TIMEOUT_S / 2
    for pid in (0, 1):
        log = (tmp_path / "logs" / f"rank{pid}.log").read_text()
        assert "multi-host consistency check failed" in log
        assert "process(es) [1] disagree with process 0" in log
        # Neither process opened an output file.
        assert _files_under(tmp_path / f"rank{pid}") == ["deck.in"]


def test_two_process_primary_io_failure_aborts_without_deadlock(tmp_path: Path) -> None:
    """A disk failure on the primary alone terminates the whole run promptly."""
    deck = _write_deck(tmp_path / "decks" / "kbm.in", fixed_dt=True)
    # The primary cannot create its output directory: `out` is a regular file there.
    (tmp_path / "rank0").mkdir()
    (tmp_path / "rank0" / "out").write_text("not a directory\n")
    run_args = [str(deck), "--mesh", "2x2", "-o", "out/run.out.nc"]
    port = _free_loopback_port()
    jobs = {
        f"rank{pid}": (
            _module_cmd(
                "gx_jax.cli", [*run_args, *_distributed_flags(port, pid)], strict_launcher=None
            ),
            _child_env(2, GX_JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT="10"),
            tmp_path / f"rank{pid}",
        )
        for pid in (0, 1)
    }
    t_start = time.monotonic()
    codes = _run_process_group(jobs, tmp_path / "logs", expect_success=False)
    elapsed = time.monotonic() - t_start
    # The failing primary exits with its own error code and Python traceback instead of
    # waiting at the shutdown barrier; the peer blocked in a collective is terminated
    # by the coordination service (non-zero exit), well within the timeout.
    assert codes["rank0"] == 1 and codes["rank1"] != 0, codes
    assert elapsed < SUBPROCESS_TIMEOUT_S / 2
    log0 = (tmp_path / "logs" / "rank0.log").read_text()
    assert "Traceback (most recent call last)" in log0
    assert "FileExistsError" in log0 or "NotADirectoryError" in log0
    assert "process 0 of 2 failed; aborting the multi-host run" in log0
    assert _files_under(tmp_path / "rank0") == ["out"]
    assert _files_under(tmp_path / "rank1") == []


# =============================================================================
# 7. Single-Process (`jax.process_count() == 1`) Bit-Identity vs Session 7 Paths
# =============================================================================

_SINGLE_PROCESS_SCRIPT = r"""
import sys
from pathlib import Path

variant, out_root, port = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
decks = [Path(p) for p in sys.argv[4:]]

import jax
import numpy as np

if variant != "session7":
    from jax._src import xla_bridge

    from gx_jax.distributed import (
        init_distributed_runtime,
        is_distributed,
        is_primary_process,
        process_count,
        shutdown_distributed_runtime,
    )

if variant == "plain":
    # --no-distributed and single-host auto-detection (even inside a single-host TPU
    # environment or a multi-task SLURM allocation without `srun`) are zero-overhead
    # no-ops: no runtime, no backend initialization.
    assert not init_distributed_runtime(False).initialized
    single_host_env = {
        "TPU_WORKER_ID": "0",
        "TPU_WORKER_HOSTNAMES": "localhost",
        "SLURM_JOB_ID": "123456",
        "SLURM_NTASKS": "4",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_JOB_NODELIST": "n1",
    }
    info = init_distributed_runtime(None, env=single_host_env)
    assert not info.initialized and info.source == "single-host"
    assert not jax.distributed.is_initialized()
    assert not xla_bridge.backends_are_initialized()
    assert shutdown_distributed_runtime() is False
elif variant == "dist1":
    # A real 1-process jax.distributed runtime: precision is enforced before
    # jax.distributed.initialize and before any backend / jax.devices() access.
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", None)
    seen = []
    real_initialize = jax.distributed.initialize

    def spy_initialize(*args, **kwargs):
        seen.append(
            (
                bool(jax.config.jax_enable_x64),
                jax.config.jax_default_matmul_precision,
                xla_bridge.backends_are_initialized(),
            )
        )
        return real_initialize(*args, **kwargs)

    jax.distributed.initialize = spy_initialize
    address = f"127.0.0.1:{port}"
    info = init_distributed_runtime(
        True, coordinator_address=address, num_processes=1, process_id=0
    )
    assert seen == [(True, "highest", False)], seen
    assert info.initialized and info.initialized_by_gx_jax and info.source == "cli"
    assert (info.process_index, info.process_count) == (0, 1)
    assert info.local_device_count == info.global_device_count == 4
    assert jax.distributed.is_initialized() and jax.process_count() == 1
    assert process_count() == 1 and not is_distributed() and is_primary_process()
    again = init_distributed_runtime(
        True, coordinator_address=address, num_processes=1, process_id=0
    )
    assert len(seen) == 1 and again.initialized_by_gx_jax

    # jax.process_count() == 1: the multi-host placement / gather branches stay unused.
    def _forbidden(*args, **kwargs):
        raise AssertionError("multi-host branch taken with jax.process_count() == 1")

    from jax.experimental import multihost_utils

    jax.make_array_from_callback = _forbidden
    multihost_utils.process_allgather = _forbidden

from gx_jax.cli import run_cli_simulation
from gx_jax.ensemble import run_ensemble

assert jax.device_count() == 4 and jax.process_count() == 1

# (a) 2D ("sp", "m") shard_map on an explicit 2x2 mesh: nonlinear, adaptive CFL, fp64.
res = run_cli_simulation(
    decks[0],
    output_path=out_root / "mesh2x2" / "run.out.nc",
    metrics_json_path=out_root / "mesh2x2" / "metrics.json",
    mesh_shape=(2, 2),
    precision="fp64",
)
np.save(out_root / "mesh2x2" / "g_final.npy", np.asarray(res["g_final"]))
np.save(out_root / "mesh2x2" / "phi_final.npy", np.asarray(res["fields_final"].phi))

# (b) B=4 vmap ensemble over a 4-device ("batch",) mesh, fp32.
res = run_ensemble(decks[1:], output_dir=out_root / "ensemble", precision="fp32", batch_devices=4)
assert res["num_devices"] == 4
np.save(out_root / "ensemble" / "g_final.npy", np.asarray(res["g_final"]))
np.save(out_root / "ensemble" / "t_final.npy", np.asarray(res["t_final"]))

if variant == "dist1":
    assert shutdown_distributed_runtime() is True
    assert not jax.distributed.is_initialized()
    assert shutdown_distributed_runtime() is False
print("SINGLE-PROCESS RUN OK", variant)
"""


def _extract_session7_sources(dest: Path) -> Path | None:
    """Extract the Session 7 `gx_jax` package from git history (None if unavailable)."""
    try:
        sha = subprocess.run(
            ["git", "log", "--format=%H", "-n", "1", "--grep", "^Session 7:"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
        if not sha:
            return None
        archive = subprocess.run(
            ["git", "archive", "--format=tar", sha, "gx_jax"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(dest, filter="data")
    except (OSError, subprocess.SubprocessError, tarfile.TarError):
        return None
    if not (dest / "gx_jax" / "cli.py").is_file() or (dest / "gx_jax" / "distributed.py").exists():
        return None
    return dest


@pytest.fixture(scope="module")
def session7_src(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Pristine Session 7 `gx_jax` sources (the commit before the distributed runtime)."""
    return _extract_session7_sources(tmp_path_factory.mktemp("session7_src"))


def _assert_run_dirs_match(
    cand_root: Path, ref_root: Path, label: str, *, tol: float | None, restart_tol: float | None
) -> int:
    """Compare two run directories file by file.

    `tol=None` demands bit-identity (`np.array_equal` on every array of every file);
    otherwise arrays / NetCDF variables must agree within `tol`. `restart_tol` overrides
    the rule for `.restart.nc` files only.
    """
    files_ref = _files_under(ref_root)
    assert _files_under(cand_root) == files_ref and files_ref, f"{label}: file sets differ"
    for rel in files_ref:
        path_c, path_r = cand_root / rel, ref_root / rel
        file_label = f"{label}: {rel}"
        file_tol = restart_tol if rel.endswith(".restart.nc") else tol
        if rel.endswith(".npy"):
            arr_c, arr_r = np.load(path_c), np.load(path_r)
            assert arr_c.dtype == arr_r.dtype and arr_c.shape == arr_r.shape, file_label
            if file_tol is None:
                assert np.array_equal(arr_c, arr_r), f"{file_label} is not bit-identical"
            else:
                scale = max(1.0, float(np.max(np.abs(arr_r))))
                assert float(np.max(np.abs(arr_c - arr_r))) < file_tol * scale, file_label
        elif rel.endswith(".nc"):
            if file_tol is None:
                _assert_nc_bit_identical(path_c, path_r, file_label)
            else:
                _assert_nc_close(path_c, path_r, file_tol, file_label)
        else:
            # Same physics payload; only wall time and the run's root directory differ.
            text_c = json.dumps(_load_strict_json(path_c)).replace(str(cand_root), "<root>")
            text_r = json.dumps(_load_strict_json(path_r)).replace(str(ref_root), "<root>")
            _assert_json_close(
                json.loads(text_c),
                json.loads(text_r),
                0.0 if file_tol is None else file_tol,
                file_label,
            )
    return len(files_ref)


def test_single_process_bit_identical_to_session7_paths(
    tmp_path: Path, session7_src: Path | None
) -> None:
    """Single-host runs (`jax.process_count() == 1`) are bit-identical to Session 7.

    Three 1-process x 4-device children run the same two workloads (`shard_map` on a
    2x2 mesh, B=4 `vmap` ensemble on a 4-device mesh): `plain` (this tree, no runtime:
    `--no-distributed` / single-host auto-detection), `session7` (the pristine Session 7
    package from git) and `dist1` (this tree under a live 1-process `jax.distributed`
    runtime). The unsharded 1-device path is covered by the next test.
    """
    decks = [
        _write_deck(tmp_path / "decks" / "adaptive.in", fixed_dt=False),
        *_write_ensemble_decks(tmp_path / "decks", fixed_dt=False, nonlinear=False),
    ]
    script = tmp_path / "single_process_run.py"
    script.write_text(_SINGLE_PROCESS_SCRIPT)
    port = _free_loopback_port()

    def _job(variant: str, env: dict[str, str]) -> tuple[list[str], dict[str, str], Path]:
        cmd = [
            sys.executable,
            str(script),
            variant,
            str(tmp_path / variant),
            str(port),
            *(str(d) for d in decks),
        ]
        return cmd, env, tmp_path / f"cwd_{variant}"

    jobs = {"plain": _job("plain", _child_env(4)), "dist1": _job("dist1", _child_env(4))}
    if session7_src is not None:
        env7 = _child_env(4)
        env7["PYTHONPATH"] = str(session7_src)  # Session 7 `gx_jax` instead of this tree
        jobs["session7"] = _job("session7", env7)
    _run_process_group(jobs, tmp_path / "logs")
    for variant in jobs:
        log_text = (tmp_path / "logs" / f"{variant}.log").read_text()
        assert f"SINGLE-PROCESS RUN OK {variant}" in log_text

    # A live 1-process runtime takes exactly the single-host code path (the child also
    # forbids the multi-host branches). Its jitted state stays bit-identical; XLA's
    # CPU all-reduce may associate cross-device sums differently once jax.distributed
    # owns the collectives, so file contents are compared at the FP parity tolerances.
    for case, tol in (("mesh2x2", 1.0e-12), ("ensemble", 1.0e-6)):
        n_files = _assert_run_dirs_match(
            tmp_path / "dist1" / case,
            tmp_path / "plain" / case,
            f"dist1 vs plain [{case}]",
            tol=tol,
            restart_tol=tol,
        )
        assert n_files == {"mesh2x2": 5, "ensemble": 15}[case]
        for name in ("g_final.npy",):
            assert np.array_equal(
                np.load(tmp_path / "dist1" / case / name), np.load(tmp_path / "plain" / case / name)
            ), f"dist1 vs plain [{case}]: {name} is not bit-identical"

    if session7_src is None:
        pytest.skip("Session 7 sources unavailable in git history; compared dist1 vs plain only")
    # No runtime: 100% bit-identical to the Session 7 package on every output. Only the
    # fp64 `.restart.nc` storage precision is owned by `fields.save_restart_nc` (Session 8
    # campaign slice), so that file is pinned through `g_final.npy` instead.
    for case, restart_tol in (("mesh2x2", 1.0e-6), ("ensemble", None)):
        _assert_run_dirs_match(
            tmp_path / "plain" / case,
            tmp_path / "session7" / case,
            f"plain vs session7 [{case}]",
            tol=None,
            restart_tol=restart_tol,
        )


def test_single_device_cli_bit_identical_to_session7(
    tmp_path: Path, session7_src: Path | None
) -> None:
    """Unsharded 1-device `gx-jax` (eager diagnostics, async I/O) matches Session 7 bitwise.

    `--no-distributed` and default auto-detection inside a single-host TPU-like
    environment that is also a multi-task SLURM allocation without `srun` must both
    stay zero-overhead no-ops (no coordinator wait, no crash).
    """
    deck = _write_deck(tmp_path / "decks" / "single.in", fixed_dt=False)
    run_args = [
        str(deck),
        "--precision",
        "fp32",
        "-o",
        "out/single.out.nc",
        "--metrics-json",
        "out/metrics.json",
    ]
    single_host_env = {**_SINGLE_HOST_TPU_ENV, **_SBATCH_WITHOUT_SRUN_ENV}
    jobs = {
        "no_distributed": (
            [sys.executable, "-m", "gx_jax.cli", *run_args, "--no-distributed"],
            _child_env(1),
            tmp_path / "no_distributed",
        ),
        "auto_single_host_env": (
            [sys.executable, "-m", "gx_jax.cli", *run_args],
            _child_env(1, **single_host_env),
            tmp_path / "auto_single_host_env",
        ),
    }
    if session7_src is not None:
        env7 = _child_env(1)
        env7["PYTHONPATH"] = str(session7_src)
        jobs["session7"] = (
            [sys.executable, "-m", "gx_jax.cli", *run_args],
            env7,
            tmp_path / "session7",
        )
    _run_process_group(jobs, tmp_path / "logs")

    expected = ["out/metrics.json", "out/single.out.nc", "out/single.restart.nc"]
    for name in jobs:
        assert _files_under(tmp_path / name) == expected
    _assert_run_dirs_match(
        tmp_path / "auto_single_host_env",
        tmp_path / "no_distributed",
        "auto vs --no-distributed",
        tol=None,
        restart_tol=None,
    )
    if session7_src is None:
        pytest.skip("Session 7 sources unavailable in git history")
    for name in ("no_distributed", "auto_single_host_env"):
        _assert_run_dirs_match(
            tmp_path / name,
            tmp_path / "session7",
            f"{name} vs session7",
            tol=None,
            restart_tol=None,
        )
