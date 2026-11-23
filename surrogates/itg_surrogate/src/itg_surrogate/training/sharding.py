"""Modern `jax.sharding` utilities (`Mesh`, `NamedSharding`, `PartitionSpec`).

Strictly avoids legacy `jax.pmap` in favor of GSPMD `NamedSharding` so ensemble
members shard cleanly across TPU chips, multi-GPU nodes, or single-CPU hosts.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclasses.dataclass(frozen=True)
class EnsembleMeshContext:
    """Device mesh and sharding specifications for ensemble parallelism."""

    mesh: Mesh
    ensemble_sharding: NamedSharding
    replicated_sharding: NamedSharding
    num_devices_used: int


def configure_jax_float32() -> None:
    """Enforces full `float32` matmul precision on TPU/GPU for tail accuracy."""
    jax.config.update("jax_default_matmul_precision", "float32")


def create_ensemble_mesh(
    num_members: int,
    devices: Sequence[jax.Device] | None = None,
) -> EnsembleMeshContext:
    """Creates a 1D `'ensemble'` `Mesh` compatible with `num_members`.

    Selects the largest divisor `d <= len(available_devices)` of `num_members` so
    that `PartitionSpec('ensemble')` evenly shards `num_members` models across
    hardware (`4` TPUs, `8` GPUs, or `1` CPU in unit tests) without requiring
    manual padding.
    """
    if num_members < 1:
        raise ValueError(f"num_members must be >= 1; got {num_members}.")
    avail = list(devices) if devices is not None else list(jax.devices())
    if not avail:
        raise RuntimeError("No JAX devices available.")

    chosen_d = 1
    for candidate in range(len(avail), 0, -1):
        if num_members % candidate == 0:
            chosen_d = candidate
            break

    mesh_devices = np.array(avail[:chosen_d])
    mesh = Mesh(mesh_devices, ("ensemble",))
    return EnsembleMeshContext(
        mesh=mesh,
        ensemble_sharding=NamedSharding(mesh, PartitionSpec("ensemble")),
        replicated_sharding=NamedSharding(mesh, PartitionSpec()),
        num_devices_used=chosen_d,
    )
