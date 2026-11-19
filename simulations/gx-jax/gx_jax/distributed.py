"""Multi-host ``jax.distributed`` runtime bootstrap & lockstep global-array helpers.

Scales the 2D ``("sp", "m")`` ``shard_map`` solver (``gx-jax``) and the 1D
``("batch",)`` ``vmap`` ensemble driver (``gx-jax-ensemble``) from one host to multi-host
TPU / GPU / CPU clusters where every host runs one Python process over its local devices
and ``jax.devices()`` spans all hosts:

  * :func:`detect_distributed_env` is a pure (JAX-free) resolver that merges explicit
    CLI overrides with GKE ``JobSet`` / TPU slice / MegaScale, Google Cloud Batch,
    and SLURM environment variables into a :class:`DistributedConfig`.
  * :func:`init_distributed_runtime` enforces ``jax_enable_x64=True`` and
    ``jax_default_matmul_precision="highest"`` *before* ``jax.distributed.initialize``
    and before any ``jax.devices()`` call, initializes the runtime idempotently, and
    records ownership so :func:`shutdown_distributed_runtime` only tears down what
    ``gx_jax`` itself started.
  * :func:`device_put_global` / :func:`gather_global_array` / :func:`gather_global_tree`
    place and gather global ``jax.Array`` leaves. With ``jax.process_count() == 1`` they
    are exactly ``jax.device_put`` / ``jax.device_get`` (zero overhead, bit-identical to
    the single-host path). With ``jax.process_count() > 1`` placement goes through
    ``jax.make_array_from_callback`` (``jax.device_put`` cannot target non-addressable
    devices) and gathers go through
    ``jax.experimental.multihost_utils.process_allgather(..., tiled=True)``.

Lockstep collective rule: every process of a multi-host run must execute the same
sequence of multi-device computations (steppers, diagnostics, gathers). Only host disk
writes are gated on :func:`is_primary_process` (``jax.process_index() == 0``).
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
import os
import sys
import traceback
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_COORDINATOR_PORT = 8476
"""Default ``jax.distributed`` coordinator port when an address carries no ``:port``."""

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_PROXY_ENV_VARS = (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "grpc_proxy",
    "GRPC_PROXY",
)

# Environment knobs (all optional).
ENV_COORDINATOR_PORT = ("GX_JAX_COORDINATOR_PORT", "JAX_COORDINATOR_PORT")
ENV_INIT_TIMEOUT = "GX_JAX_DISTRIBUTED_INIT_TIMEOUT"
ENV_HEARTBEAT_TIMEOUT = "GX_JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT"
ENV_SHUTDOWN_TIMEOUT = "GX_JAX_DISTRIBUTED_SHUTDOWN_TIMEOUT"
ENV_CPU_COLLECTIVES_HOSTNAME = "GX_JAX_CPU_COLLECTIVES_HOSTNAME"
ENV_CPU_COLLECTIVES_INTERFACE = "GX_JAX_CPU_COLLECTIVES_INTERFACE"


# ---------------------------------------------------------------------------
# 1. Configuration & Runtime Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedConfig:
    """Resolved ``jax.distributed.initialize`` parameters for this process.

    ``enabled=False`` means the runtime must stay untouched (single-host fast path).
    ``source`` is one of ``"disabled"`` (``--no-distributed``), ``"single-host"``,
    ``"cli"``, ``"gke"``, ``"cloud-batch"``, ``"slurm"``, or ``"auto"`` (``--distributed``
    with nothing resolvable here: JAX's native cluster detection, e.g. Cloud TPU
    metadata, fills in every parameter).
    """

    enabled: bool
    source: str = "single-host"
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    reason: str = ""

    def initialize_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``jax.distributed.initialize`` (unset values omitted)."""
        kwargs: dict[str, Any] = {}
        if self.coordinator_address is not None:
            kwargs["coordinator_address"] = self.coordinator_address
        if self.num_processes is not None:
            kwargs["num_processes"] = int(self.num_processes)
        if self.process_id is not None:
            kwargs["process_id"] = int(self.process_id)
        if self.local_device_ids is not None:
            kwargs["local_device_ids"] = [int(v) for v in self.local_device_ids]
        return kwargs


@dataclass(frozen=True)
class DistributedRuntimeInfo:
    """Snapshot of the ``jax.distributed`` runtime after :func:`init_distributed_runtime`."""

    config: DistributedConfig
    initialized: bool
    initialized_by_gx_jax: bool
    process_index: int = 0
    process_count: int = 1
    local_device_count: int | None = None
    global_device_count: int | None = None

    @property
    def enabled(self) -> bool:
        """Whether the ``jax.distributed`` runtime is live for this process."""
        return self.initialized

    @property
    def is_primary(self) -> bool:
        """Whether this process owns host disk writes (``process_index == 0``)."""
        return self.process_index == 0

    @property
    def is_multi_process(self) -> bool:
        """Whether more than one process participates (``process_count > 1``)."""
        return self.process_count > 1

    @property
    def source(self) -> str:
        """Environment that resolved the configuration (see :class:`DistributedConfig`)."""
        return self.config.source

    @property
    def coordinator_address(self) -> str | None:
        """Resolved coordinator ``host:port`` (``None`` when delegated or disabled)."""
        return self.config.coordinator_address


@dataclass
class _RuntimeState:
    owned: bool = False
    config: DistributedConfig | None = None
    counts: tuple[int, int] | None = None  # cached (process_index, process_count)


_STATE = _RuntimeState()


# ---------------------------------------------------------------------------
# 2. Pure Environment Detection (no JAX backend access)
# ---------------------------------------------------------------------------


def _split_host_port(address: str) -> tuple[str, str | None]:
    """Split ``host[:port]`` / ``[ipv6][:port]`` / bare IPv6 into ``(host, port | None)``."""
    addr = str(address).strip()
    if not addr:
        raise ValueError("coordinator address must be a non-empty 'host[:port]' string.")
    if addr.startswith("["):
        end = addr.find("]")
        rest = addr[end + 1 :] if end > 0 else None
        if rest is None or (rest and not (rest.startswith(":") and rest[1:].isdigit())):
            raise ValueError(f"Invalid coordinator address {address!r}.")
        return addr[: end + 1], (rest[1:] if rest else None)
    n_colon = addr.count(":")
    if n_colon == 0:
        return addr, None
    if n_colon == 1:
        host, port = addr.split(":")
        if not host or (port and not port.isdigit()):
            raise ValueError(f"Invalid coordinator address {address!r}.")
        return host, (port or None)
    return f"[{addr}]", None  # bare IPv6 literal


def normalize_coordinator_address(
    address: str, default_port: int = DEFAULT_COORDINATOR_PORT
) -> str:
    """Return ``host:port``, appending ``default_port`` when ``address`` has no port."""
    host, port = _split_host_port(address)
    return f"{host}:{port if port is not None else int(default_port)}"


def _env_int(env: Mapping[str, str], *names: str) -> int | None:
    """Return the first set, non-empty integer among ``names`` (``ValueError`` if malformed)."""
    for name in names:
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"Environment variable {name}={raw!r} is not an integer.") from exc
    return None


def _split_hosts(text: str | None) -> list[str]:
    """Split a comma / whitespace separated host list, dropping blanks and duplicates."""
    if not text:
        return []
    hosts: list[str] = []
    for line in str(text).replace(",", "\n").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry and entry not in hosts:
            hosts.append(entry)
    return hosts


def _normalize_local_device_ids(
    local_device_ids: int | str | Sequence[int] | None,
) -> tuple[int, ...] | None:
    if local_device_ids is None:
        return None
    if isinstance(local_device_ids, str):
        parts = [p.strip() for p in local_device_ids.split(",") if p.strip()]
        if not parts:
            return None
        try:
            ids = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(
                f"Invalid local device ids {local_device_ids!r}; expected e.g. '0,1'."
            ) from exc
    elif isinstance(local_device_ids, int):
        ids = (int(local_device_ids),)
    else:
        ids = tuple(int(v) for v in local_device_ids)
        if not ids:
            return None
    if any(v < 0 for v in ids):
        raise ValueError(f"local device ids must be >= 0, got {ids}.")
    return ids


@dataclass(frozen=True)
class _Candidate:
    source: str
    multi_host: bool
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    reason: str = ""


def _detect_gke(env: Mapping[str, str], port: int) -> _Candidate | None:
    """GKE ``JobSet`` / TPU slice / MegaScale (and plain ``JAX_*``) environment."""
    mega = (env.get("MEGASCALE_COORDINATOR_ADDRESS") or "").strip()
    jax_addr = (env.get("JAX_COORDINATOR_ADDRESS") or "").strip()
    hosts = _split_hosts(env.get("TPU_WORKER_HOSTNAMES"))
    worker_id = _env_int(env, "TPU_WORKER_ID")
    if not (mega or jax_addr or hosts or worker_id is not None):
        return None

    num_slices = max(1, _env_int(env, "MEGASCALE_NUM_SLICES") or 1)
    slice_id = max(0, _env_int(env, "MEGASCALE_SLICE_ID") or 0)
    per_slice = len(hosts) if hosts else None

    nproc = _env_int(env, "MEGASCALE_NUM_PROCESSES", "JAX_NUM_PROCESSES")
    if nproc is None and per_slice is not None:
        nproc = per_slice * num_slices

    pid = _env_int(env, "MEGASCALE_PROCESS_ID", "JAX_PROCESS_ID")
    if pid is None and worker_id is not None:
        pid = worker_id + slice_id * (per_slice or 1)
    if pid is None:
        pid = _env_int(env, "JOB_COMPLETION_INDEX")

    coord: str | None
    if mega:
        # MEGASCALE_COORDINATOR_ADDRESS names the host with process id 0; any port on
        # it belongs to the MegaScale transport, not to the JAX coordination service
        # (mirrors jax._src.clusters.cloud_tpu_cluster.BaseTpuCluster).
        coord = f"{_split_host_port(mega)[0]}:{port}"
    elif jax_addr:
        coord = normalize_coordinator_address(jax_addr, port)
    elif hosts:
        coord = normalize_coordinator_address(hosts[0], port)
    else:
        coord = None

    if nproc is not None:
        multi = nproc > 1
    else:
        multi = bool(mega or jax_addr) or (worker_id is not None and worker_id > 0)
    return _Candidate(
        source="gke",
        multi_host=multi,
        coordinator_address=coord,
        num_processes=nproc,
        process_id=pid,
        reason="GKE JobSet / TPU slice / MegaScale environment",
    )


def _detect_cloud_batch(
    env: Mapping[str, str], port: int, distributed: bool | None
) -> _Candidate | None:
    """Google Cloud Batch / GCE multi-node job (``BATCH_HOSTS_FILE``)."""
    hosts_file = (env.get("BATCH_HOSTS_FILE") or "").strip()
    if not hosts_file:
        return None
    try:
        hosts = _split_hosts(Path(hosts_file).read_text(encoding="utf-8"))
    except OSError as exc:
        if distributed is True:
            raise ValueError(f"Cannot read BATCH_HOSTS_FILE={hosts_file!r}: {exc}") from exc
        return None  # auto mode: an unreadable hosts file never breaks single-host runs
    if not hosts:
        return None
    task_count = _env_int(env, "BATCH_TASK_COUNT")
    task_index = _env_int(env, "BATCH_TASK_INDEX")
    multi = len(hosts) > 1
    explicit_multi_task = distributed is True and task_count is not None and task_count > 1
    nproc = len(hosts)
    if (multi or explicit_multi_task) and task_count is not None and task_count > nproc:
        nproc = task_count  # several tasks packed per VM (taskCountPerNode > 1)
    return _Candidate(
        source="cloud-batch",
        multi_host=multi,
        coordinator_address=normalize_coordinator_address(hosts[0], port),
        num_processes=nproc,
        process_id=task_index if task_index is not None else (None if nproc > 1 else 0),
        reason=f"Cloud Batch hosts file {hosts_file} ({len(hosts)} host(s))",
    )


def _slurm_first_node(node_list: str) -> str:
    """First hostname of a SLURM node list (``n1``, ``n1,h2``, ``n[001-015],h2``, ...)."""
    text = node_list.strip()
    ind = next((i for i, ch in enumerate(text) if ch in ",["), len(text))
    if ind == len(text) or text[ind] == ",":
        return text[:ind]
    prefix, suffix = text[:ind], text[ind + 1 :]
    ind2 = next((i for i, ch in enumerate(suffix) if ch in ",-]"), len(suffix))
    return f"{prefix}{suffix[:ind2]}"


def _detect_slurm(env: Mapping[str, str], explicit_port: int | None) -> _Candidate | None:
    """SLURM multi-task (multi-GPU / multi-node) ``srun`` step: ``SLURM_NTASKS > 1``.

    Only an ``srun`` step (``SLURM_STEP_NODELIST``) really launched ``SLURM_NTASKS``
    processes. A batch script that runs a single process inside a multi-task allocation
    (``sbatch --ntasks=4`` without ``srun``) exports ``SLURM_NTASKS`` / ``SLURM_PROCID`` but
    no step node list and must stay a single-host run, exactly like
    ``jax._src.clusters.slurm_cluster.SlurmCluster.is_env_present``.
    """
    ntasks = _env_int(env, "SLURM_NTASKS", "SLURM_NPROCS", "SLURM_STEP_NUM_TASKS")
    if ntasks is None:
        return None
    pid = _env_int(env, "SLURM_PROCID")
    local_id = _env_int(env, "SLURM_LOCALID")
    node_list = (env.get("SLURM_STEP_NODELIST") or "").strip()
    in_step = bool(node_list)
    coord: str | None = None
    if in_step:
        if explicit_port is not None:
            slurm_port = int(explicit_port)
        else:
            job_id = _env_int(env, "SLURM_JOB_ID")
            # Same ephemeral-range port rule as jax._src.clusters.slurm_cluster so
            # concurrent jobs sharing a head node do not collide.
            slurm_port = (
                (job_id % 2**12) + (65535 - 2**12 + 1)
                if job_id is not None
                else DEFAULT_COORDINATOR_PORT
            )
        coord = f"{_slurm_first_node(node_list)}:{slurm_port}"
    multi = ntasks > 1 and in_step
    return _Candidate(
        source="slurm",
        multi_host=multi,
        coordinator_address=coord,
        num_processes=ntasks,
        process_id=pid,
        # One device per task, like JAX's native SLURM support (override: --local-device-ids).
        local_device_ids=(local_id,) if (local_id is not None and multi) else None,
        reason=(
            f"SLURM srun step with SLURM_NTASKS={ntasks}"
            if in_step
            else f"SLURM allocation with SLURM_NTASKS={ntasks} outside an srun step"
        ),
    )


def detect_distributed_env(
    distributed: bool | None = None,
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | str | Sequence[int] | None = None,
    env: Mapping[str, str] | None = None,
    default_port: int | None = None,
) -> DistributedConfig:
    """Resolve whether (and how) this process must call ``jax.distributed.initialize``.

    Pure function: reads ``env`` (default ``os.environ``) and, for Cloud Batch, the
    ``BATCH_HOSTS_FILE``; never touches JAX devices.

    * ``distributed is False`` (``--no-distributed``): always a no-op.
    * ``distributed is None`` (default): enabled only when a multi-host environment
      is detected (resolved ``num_processes > 1``); single-host CPU / GPU / TPU
      (e.g. ``TPU_WORKER_ID=0`` with one ``TPU_WORKER_HOSTNAMES`` entry and no
      coordinator address) stays a no-op.
    * ``distributed is True`` (``--distributed``): always enabled; anything not
      resolvable here is delegated to JAX's native cluster detection.

    Explicit arguments override environment values field by field. Environments are
    probed in the order GKE ``JobSet`` / TPU slice / MegaScale
    (``MEGASCALE_COORDINATOR_ADDRESS`` > ``JAX_COORDINATOR_ADDRESS`` >
    ``TPU_WORKER_HOSTNAMES[0]``; ``MEGASCALE_NUM_PROCESSES`` > ``JAX_NUM_PROCESSES`` >
    ``len(TPU_WORKER_HOSTNAMES) * MEGASCALE_NUM_SLICES``; ``MEGASCALE_PROCESS_ID`` >
    ``JAX_PROCESS_ID`` > ``TPU_WORKER_ID`` > ``JOB_COMPLETION_INDEX``), Google Cloud
    Batch (``BATCH_HOSTS_FILE`` / ``BATCH_TASK_INDEX`` / ``BATCH_TASK_COUNT``), then
    SLURM (``SLURM_NTASKS > 1`` inside an ``srun`` step, with ``SLURM_PROCID`` /
    ``SLURM_LOCALID``); the first one that signals multiple hosts wins.
    """
    env_map: Mapping[str, str] = os.environ if env is None else env
    if distributed is False:
        return DistributedConfig(
            enabled=False, source="disabled", reason="distributed runtime disabled explicitly"
        )

    if num_processes is not None and int(num_processes) < 1:
        raise ValueError(f"num_processes={num_processes} must be >= 1.")
    if process_id is not None and int(process_id) < 0:
        raise ValueError(f"process_id={process_id} must be >= 0.")
    cli_ids = _normalize_local_device_ids(local_device_ids)
    explicit_port = (
        int(default_port) if default_port is not None else _env_int(env_map, *ENV_COORDINATOR_PORT)
    )
    port = explicit_port if explicit_port is not None else DEFAULT_COORDINATOR_PORT

    candidates = [
        cand
        for cand in (
            _detect_gke(env_map, port),
            _detect_cloud_batch(env_map, port, distributed),
            _detect_slurm(env_map, explicit_port),
        )
        if cand is not None
    ]
    cand = next((c for c in candidates if c.multi_host), candidates[0] if candidates else None)

    cli_coord = (
        normalize_coordinator_address(coordinator_address, port)
        if coordinator_address is not None and str(coordinator_address).strip()
        else None
    )
    coord = cli_coord if cli_coord is not None else (cand.coordinator_address if cand else None)
    nproc = (
        int(num_processes) if num_processes is not None else (cand.num_processes if cand else None)
    )
    pid = int(process_id) if process_id is not None else (cand.process_id if cand else None)
    ids = cli_ids if cli_ids is not None else (cand.local_device_ids if cand else None)

    cli_given = cli_coord is not None or num_processes is not None or process_id is not None
    if cli_coord is not None or (cand is None and cli_given):
        source = "cli"
    elif cand is not None:
        source = cand.source
    else:
        source = "auto"

    # Auto mode (`distributed is None`) only turns the runtime on for explicit overrides
    # or for an environment that itself signals several launched processes.
    if num_processes is not None:
        multi = int(num_processes) > 1
    elif cli_coord is not None:
        multi = nproc is None or nproc > 1
    else:
        multi = bool(cand and cand.multi_host)

    if distributed is None and not multi:
        return DistributedConfig(
            enabled=False,
            source="single-host",
            reason="no multi-host environment detected (single-host fast path)",
        )

    if coord is not None and (nproc is None or pid is None):
        missing = [
            name for name, val in (("num_processes", nproc), ("process_id", pid)) if val is None
        ]
        raise ValueError(
            f"Distributed coordinator {coord} resolved from {source!r}, but "
            f"{' and '.join(missing)} could not be determined; pass "
            "--num-processes / --process-id explicitly."
        )
    if coord is None and nproc is not None and nproc > 1:
        hint = (
            " (SLURM_STEP_NODELIST is unset: launch one process per task with `srun`)"
            if source == "slurm"
            else ""
        )
        raise ValueError(
            f"num_processes={nproc} resolved from {source!r}, but no coordinator "
            f"address could be determined{hint}; pass --coordinator-address explicitly "
            "or --no-distributed for a single-host run."
        )
    if nproc is not None and pid is not None and not (0 <= pid < nproc):
        raise ValueError(
            f"process_id={pid} is out of range for num_processes={nproc} "
            f"(resolved from {source!r})."
        )

    if source == "auto":
        reason = "delegated to jax.distributed native cluster detection"
    elif cand is not None and source == cand.source:
        reason = cand.reason
    else:
        reason = "explicit overrides"
    return DistributedConfig(
        enabled=True,
        source=source,
        coordinator_address=coord,
        num_processes=nproc,
        process_id=pid,
        local_device_ids=ids,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 3. Runtime Lifecycle
# ---------------------------------------------------------------------------


def _enforce_precision() -> None:
    """Enable host float64 precomputation and IEEE FP32 MXU matmul precision."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


def _jax_distributed_is_initialized() -> bool:
    fn = getattr(jax.distributed, "is_initialized", None)
    if fn is not None:
        return bool(fn())
    from jax._src import distributed as _jax_distributed  # pragma: no cover

    return _jax_distributed.global_state.client is not None  # pragma: no cover


def _exempt_host_from_proxies(host: str) -> None:
    """Keep coordinator gRPC traffic off HTTP(S) proxies by extending ``no_proxy``.

    ``jax.distributed`` hangs (``RegisterTask DEADLINE_EXCEEDED``) when
    ``http(s)_proxy`` routes the intra-cluster coordinator channel through a proxy.
    """
    bare = host.strip("[]")
    if not bare or not any(os.environ.get(name) for name in _PROXY_ENV_VARS):
        return
    for key in ("no_proxy", "NO_PROXY", "no_grpc_proxy"):
        current = os.environ.get(key)
        if current is None and key == "no_grpc_proxy":
            continue  # gRPC falls back to no_proxy; do not shadow its other entries
        entries = [e.strip() for e in (current or "").split(",") if e.strip()]
        if bare not in entries:
            entries.append(bare)
            os.environ[key] = ",".join(entries)


def _bind_cpu_collectives(hostname: str | None, interface: str | None) -> bool:
    """Pin the cross-process CPU (Gloo TCP) collectives to ``hostname`` / ``interface``.

    By default Gloo resolves the machine's own hostname, which fails in sandboxes and
    containers whose hostname has no DNS / ``/etc/hosts`` entry. For loopback
    coordinators every process lives on this host, so the transport is pinned to the
    loopback address instead. Best effort: relies on private JAX hooks and silently
    keeps JAX's default when they are unavailable.
    """
    try:
        from jax._src import distributed as _jax_distributed
        from jax._src import xla_bridge
        from jax._src.lib import _jax

        if getattr(jax.config, "jax_cpu_collectives_implementation", "gloo") != "gloo":
            return False
        make_collectives = getattr(_jax, "make_gloo_tcp_collectives", None)
        client = _jax_distributed.global_state.client
        if make_collectives is None or client is None:
            return False
        if xla_bridge.backends_are_initialized():
            return False
        if "collectives" not in inspect.signature(xla_bridge.make_cpu_client).parameters:
            return False
        kwargs: dict[str, Any] = {"distributed_client": client}
        if interface:
            kwargs["interface"] = interface
        elif hostname:
            kwargs["hostname"] = hostname
        collectives = make_collectives(**kwargs)
        xla_bridge.register_backend_factory(
            "cpu",
            functools.partial(xla_bridge.make_cpu_client, collectives=collectives),
            priority=0,
        )
        return True
    except Exception as exc:  # pragma: no cover - depends on private JAX internals
        logger.debug("Could not pin CPU collectives (%s); using JAX defaults.", exc)
        return False


def _static_runtime_info(config: DistributedConfig) -> DistributedRuntimeInfo:
    """Runtime info for the no-op path: never touches the JAX backend."""
    return DistributedRuntimeInfo(
        config=config,
        initialized=False,
        initialized_by_gx_jax=False,
        process_index=0,
        process_count=1,
    )


def _live_runtime_info(config: DistributedConfig) -> DistributedRuntimeInfo:
    return DistributedRuntimeInfo(
        config=config,
        initialized=True,
        initialized_by_gx_jax=_STATE.owned,
        process_index=process_index(),
        process_count=process_count(),
        local_device_count=int(jax.local_device_count()),
        global_device_count=int(jax.device_count()),
    )


def init_distributed_runtime(
    distributed: bool | None = None,
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | str | Sequence[int] | None = None,
    env: Mapping[str, str] | None = None,
    initialization_timeout: int | None = None,
    heartbeat_timeout_seconds: int | None = None,
    shutdown_timeout_seconds: int | None = None,
) -> DistributedRuntimeInfo:
    """Initialize ``jax.distributed`` for multi-host runs (no-op on a single host).

    Must be called before any JAX computation or ``jax.devices()`` access. Enforces
    ``jax_enable_x64=True`` and ``jax_default_matmul_precision="highest"`` first.

    * ``distributed is False``, or ``distributed is None`` on a single host: returns
      immediately without touching the JAX backend (single-host behaviour, bit-identical).
    * Already initialized (by the caller or an earlier call): never re-initializes.
    * Otherwise calls ``jax.distributed.initialize(...)`` with the parameters resolved by
      :func:`detect_distributed_env` and records ownership for
      :func:`shutdown_distributed_runtime`.

    Timeouts default to JAX's (300 s init, 100 s heartbeat, 300 s shutdown) and can be
    overridden per call or through ``GX_JAX_DISTRIBUTED_{INIT,HEARTBEAT,SHUTDOWN}_TIMEOUT``.
    """
    _enforce_precision()
    if distributed is False:
        config = DistributedConfig(
            enabled=False, source="disabled", reason="distributed runtime disabled explicitly"
        )
        if _jax_distributed_is_initialized():
            return _live_runtime_info(config)
        return _static_runtime_info(config)

    env_map: Mapping[str, str] = os.environ if env is None else env
    config = detect_distributed_env(
        distributed,
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
        env=env_map,
    )
    if _jax_distributed_is_initialized():
        # Idempotent: keep whatever runtime (ours or the caller's) is already live.
        if _STATE.config is None:
            _STATE.config = config
        return _live_runtime_info(_STATE.config)
    if not config.enabled:
        return _static_runtime_info(config)

    kwargs = config.initialize_kwargs()
    timeouts = {
        "initialization_timeout": (initialization_timeout, ENV_INIT_TIMEOUT),
        "heartbeat_timeout_seconds": (heartbeat_timeout_seconds, ENV_HEARTBEAT_TIMEOUT),
        "shutdown_timeout_seconds": (shutdown_timeout_seconds, ENV_SHUTDOWN_TIMEOUT),
    }
    for key, (value, env_name) in timeouts.items():
        resolved = int(value) if value is not None else _env_int(env_map, env_name)
        if resolved is not None:
            kwargs[key] = resolved

    host: str | None = None
    if config.coordinator_address is not None:
        host = _split_host_port(config.coordinator_address)[0]
        _exempt_host_from_proxies(host)
        if host.strip("[]") in _LOOPBACK_HOSTS and "JAX_COORDINATOR_BIND_ADDRESS" not in os.environ:
            # Loopback multi-process runs never need to listen on external interfaces.
            kwargs["coordinator_bind_address"] = config.coordinator_address

    # Older JAX releases lack some optional keywords (bind address, timeouts).
    supported = inspect.signature(jax.distributed.initialize).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in supported.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in supported}
    logger.info(
        "Initializing jax.distributed (source=%s, coordinator=%s, num_processes=%s, process_id=%s)",
        config.source,
        config.coordinator_address,
        config.num_processes,
        config.process_id,
    )
    try:
        jax.distributed.initialize(**kwargs)
    except (ValueError, RuntimeError, KeyError) as exc:
        if config.coordinator_address is None and not _jax_distributed_is_initialized():
            # Delegated (JAX-native) cluster detection could not complete the parameters.
            raise ValueError(
                f"jax.distributed could not be initialized from the {config.source!r} "
                "environment: no coordinator could be resolved (GKE "
                "MEGASCALE_COORDINATOR_ADDRESS / JAX_COORDINATOR_ADDRESS / "
                "TPU_WORKER_HOSTNAMES, Cloud Batch BATCH_HOSTS_FILE, an `srun` SLURM step, "
                "or Cloud TPU metadata). Pass --coordinator-address, --num-processes and "
                f"--process-id explicitly, or --no-distributed for a single-host run. ({exc})"
            ) from exc
        raise
    _STATE.owned = True
    _STATE.config = config
    _STATE.counts = None

    cpu_host = os.environ.get(ENV_CPU_COLLECTIVES_HOSTNAME) or None
    cpu_iface = os.environ.get(ENV_CPU_COLLECTIVES_INTERFACE) or None
    if cpu_host is None and cpu_iface is None and host is not None:
        bare = host.strip("[]")
        if bare in _LOOPBACK_HOSTS:
            cpu_host = bare
    if cpu_host is not None or cpu_iface is not None:
        _bind_cpu_collectives(cpu_host, cpu_iface)

    return _live_runtime_info(config)


def shutdown_distributed_runtime() -> bool:
    """Shut down ``jax.distributed`` iff :func:`init_distributed_runtime` started it.

    Safe to call unconditionally (e.g. from ``finally``) and more than once; a runtime
    initialized by the caller is left untouched. Returns whether a shutdown happened.
    """
    owned = _STATE.owned
    _STATE.owned = False
    _STATE.counts = None
    if not owned:
        return False
    _STATE.config = None
    if not _jax_distributed_is_initialized():
        return False
    jax.distributed.shutdown()
    return True


def _owned_process_counts() -> tuple[int, int]:
    """``(process_index, process_count)`` for :func:`abort_distributed_run`; never raises."""
    if _STATE.counts is not None:
        return _STATE.counts
    cfg = _STATE.config
    if cfg is not None and cfg.num_processes is not None and cfg.process_id is not None:
        # The JAX backend itself may have failed to initialize: do not query it again.
        return (int(cfg.process_id), int(cfg.num_processes))
    try:
        return _process_counts()
    except Exception:
        return (0, 1)


def abort_distributed_run(exc: BaseException) -> None:
    """Fail fast when this process of a multi-process run is about to die from ``exc``.

    A process that fails alone (e.g. the primary hits a full disk) must not wait at the
    graceful ``jax.distributed`` shutdown barrier: its peers are blocked in a collective
    waiting for it, so the barrier would only time out (300 s by default) and the
    coordination service would then SIGABRT every process, swallowing the Python
    traceback. Instead the traceback is printed and the process exits immediately; the
    coordination service terminates the remaining processes.

    No-op (returns) on a single host, when ``gx_jax`` does not own the runtime, or for a
    clean ``SystemExit(0)``: the caller then re-raises ``exc`` as usual.
    """
    if not _STATE.owned:
        return
    index, count = _owned_process_counts()
    if count <= 1:
        return
    if isinstance(exc, SystemExit):
        if exc.code is None or exc.code == 0:
            return
        code = exc.code if isinstance(exc.code, int) else 1
        if not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
    elif isinstance(exc, KeyboardInterrupt):
        code = 130
    else:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        code = 1
    print(
        f"[gx-jax] process {index} of {count} failed; aborting the "
        "multi-host run without waiting at the jax.distributed shutdown barrier.",
        file=sys.stderr,
    )
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _process_counts() -> tuple[int, int]:
    """Cached ``(jax.process_index(), jax.process_count())`` while the runtime is live."""
    if not _jax_distributed_is_initialized():
        return (0, 1)
    counts = _STATE.counts
    if counts is None:
        counts = (int(jax.process_index()), int(jax.process_count()))
        _STATE.counts = counts
    return counts


def process_index() -> int:
    """``jax.process_index()`` (``0`` without touching the backend on a single host)."""
    return _process_counts()[0]


def process_count() -> int:
    """``jax.process_count()`` (``1`` without touching the backend on a single host)."""
    return _process_counts()[1]


def is_distributed() -> bool:
    """Whether this run spans several processes (``jax.process_count() > 1``)."""
    return _process_counts()[1] > 1


def is_primary_process() -> bool:
    """Whether this process owns host disk writes (``jax.process_index() == 0``)."""
    return _process_counts()[0] == 0


# ---------------------------------------------------------------------------
# 4. Multi-Host Global Array Placement & Lockstep Gathering
# ---------------------------------------------------------------------------


def _identity(x: Any) -> Any:
    return x


@functools.lru_cache(maxsize=64)
def _reshard_fn(sharding: Any) -> Any:
    return jax.jit(_identity, out_shardings=sharding)


def _is_global_array(x: Any) -> bool:
    """Whether ``x`` is a ``jax.Array`` spanning devices this process cannot address."""
    return isinstance(x, jax.Array) and not x.is_fully_addressable


def _place_leaf_multihost(leaf: Any, sharding: Any) -> Any:
    if _is_global_array(leaf):
        if leaf.sharding.is_equivalent_to(sharding, leaf.ndim):
            return leaf
        # Global -> global resharding is a collective: every process runs it in lockstep.
        return _reshard_fn(sharding)(leaf)
    host = np.asarray(leaf)
    return jax.make_array_from_callback(host.shape, sharding, lambda index: np.asarray(host[index]))


def device_put_global(tree: Any, sharding: Any) -> Any:
    """Place every leaf of ``tree`` onto ``sharding`` (single- or multi-host).

    ``jax.process_count() == 1``: exactly ``jax.device_put(tree, sharding)``.

    ``jax.process_count() > 1``: ``jax.device_put`` cannot target non-addressable
    devices, so host / process-local leaves (identical on every process) become
    global arrays through ``jax.make_array_from_callback`` (each process uploads only
    its addressable shards) and leaves that are already global are re-sharded by a
    jitted identity that every process must execute in lockstep.
    """
    if not is_distributed():
        return jax.device_put(tree, sharding)
    return jax.tree_util.tree_map(lambda leaf: _place_leaf_multihost(leaf, sharding), tree)


def gather_global_array(x: Any) -> np.ndarray:
    """Return a host ``numpy`` copy of ``x`` that is safe on every process.

    Fully addressable inputs (NumPy, process-local or single-host arrays) are copied
    directly. Fully replicated multi-host arrays are read from the local replica
    (no collective). Partitioned multi-host arrays are gathered with
    ``multihost_utils.process_allgather(x, tiled=True)``, which every process must
    call in lockstep. Never pass a non-addressable ``jax.Array`` to ``jax.device_get``,
    ``float()`` or ``int()``.
    """
    if _is_global_array(x):
        if x.sharding.is_fully_replicated:
            return np.asarray(x.addressable_data(0))
        from jax.experimental import multihost_utils

        return np.asarray(multihost_utils.process_allgather(x, tiled=True))
    return np.asarray(x)


def gather_global_tree(tree: Any) -> Any:
    """Host ``numpy`` copy of every leaf of ``tree`` (lockstep across processes).

    ``jax.process_count() == 1``: exactly ``jax.device_get(tree)``.
    """
    if not is_distributed():
        return jax.device_get(tree)
    return jax.tree_util.tree_map(gather_global_array, tree)


def validate_multihost_mesh(mesh: Any) -> None:
    """Require every process of a multi-host run to own at least one ``mesh`` device.

    A process without a device in the mesh cannot take part in the lockstep
    collectives. No-op when ``jax.process_count() == 1``.
    """
    if not is_distributed():
        return
    used = sorted({int(dev.process_index) for dev in np.asarray(mesh.devices, dtype=object).flat})
    n_proc = process_count()
    if len(used) != n_proc:
        raise ValueError(
            f"Multi-host run with {n_proc} processes ({jax.device_count()} global devices) "
            f"needs a device mesh spanning every process, but mesh {dict(mesh.shape)} only "
            f"uses devices of process(es) {used}. Choose a mesh covering all devices "
            "(e.g. --mesh SPxM with SP*M equal to the global device count) or run "
            "single-host with --no-distributed."
        )


def assert_same_across_processes(items: Any, label: str) -> None:
    """Fail fast, on *every* process, if host-side run inputs differ between processes.

    Each process fingerprints ``repr(items)`` (CRC-32) and the fingerprints are exchanged
    with ``multihost_utils.process_allgather`` (a lockstep collective), so a mismatch
    (e.g. a deck or restart file that is only visible on one host) raises the same
    ``ValueError`` on all processes instead of leaving healthy processes blocked in a
    collective, waiting for a peer that already bailed out.
    No-op when ``jax.process_count() == 1``.
    """
    if not is_distributed():
        return
    from jax.experimental import multihost_utils

    digest = zlib.crc32(repr(items).encode("utf-8"))
    local = np.asarray([digest >> 16, digest & 0xFFFF], dtype=np.int32)
    stacked = np.asarray(multihost_utils.process_allgather(local, tiled=False)).reshape(-1, 2)
    if not bool(np.all(stacked == stacked[0])):
        mismatched = sorted({int(i) for i in np.nonzero(np.any(stacked != stacked[0], axis=1))[0]})
        raise ValueError(
            f"gx-jax multi-host consistency check failed for {label}: process(es) "
            f"{mismatched} disagree with process 0. Every process must see identical "
            "inputs (decks and restart files on shared storage, identical flags)."
        )


def sync_processes(name: str) -> None:
    """Lockstep barrier: return only once every process has reached ``name``.

    Called at the end of a run so that non-primary processes wait *inside a collective*
    until the primary process has finished its final host disk writes, instead of
    waiting at the time-limited ``jax.distributed`` shutdown barrier (which would abort
    the whole run, mid-write, after ``shutdown_timeout_seconds``).
    No-op when ``jax.process_count() == 1``.
    """
    if not is_distributed():
        return
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(name)


def host_value(x: Any) -> Any:
    """Return ``x`` unchanged on a single host, else its host ``numpy`` value.

    Lets host control flow (``int(...)`` / ``float(...)``) read replicated scalars of
    a multi-host run without changing the single-host code path.
    """
    if not is_distributed():
        return x
    return gather_global_array(x)


# ---------------------------------------------------------------------------
# 5. Shared CLI Wiring (`gx-jax` & `gx-jax-ensemble`)
# ---------------------------------------------------------------------------


def add_distributed_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--distributed`` / ``--coordinator-address`` / ``--num-processes`` /
    ``--process-id`` / ``--local-device-ids`` multi-host flags to ``parser``."""
    group = parser.add_argument_group("multi-host jax.distributed runtime")
    group.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force (--distributed) or skip (--no-distributed) jax.distributed "
            "initialization. Default: auto-detect multi-host GKE JobSet / TPU slice / "
            "MegaScale, Cloud Batch, and SLURM environments; single-host runs are a no-op."
        ),
    )
    group.add_argument(
        "--coordinator-address",
        type=str,
        default=None,
        help=(
            "jax.distributed coordinator 'host[:port]' of process 0 "
            f"(default port {DEFAULT_COORDINATOR_PORT})."
        ),
    )
    group.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Total number of processes (hosts) in the multi-host run.",
    )
    group.add_argument(
        "--process-id",
        type=int,
        default=None,
        help="Index of this process in [0, num_processes); process 0 owns disk writes.",
    )
    group.add_argument(
        "--local-device-ids",
        type=str,
        default=None,
        help="Comma-separated local device ids visible to this process (default: all).",
    )


def init_distributed_runtime_from_args(args: argparse.Namespace) -> DistributedRuntimeInfo:
    """Call :func:`init_distributed_runtime` with the flags of :func:`add_distributed_cli_args`."""
    return init_distributed_runtime(
        getattr(args, "distributed", None),
        coordinator_address=getattr(args, "coordinator_address", None),
        num_processes=getattr(args, "num_processes", None),
        process_id=getattr(args, "process_id", None),
        local_device_ids=getattr(args, "local_device_ids", None),
    )


__all__ = [
    "DEFAULT_COORDINATOR_PORT",
    "DistributedConfig",
    "DistributedRuntimeInfo",
    "abort_distributed_run",
    "add_distributed_cli_args",
    "assert_same_across_processes",
    "detect_distributed_env",
    "device_put_global",
    "gather_global_array",
    "gather_global_tree",
    "host_value",
    "init_distributed_runtime",
    "init_distributed_runtime_from_args",
    "is_distributed",
    "is_primary_process",
    "normalize_coordinator_address",
    "process_count",
    "process_index",
    "shutdown_distributed_runtime",
    "sync_processes",
    "validate_multihost_mesh",
]
