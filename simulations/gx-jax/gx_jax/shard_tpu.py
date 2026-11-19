"""Multi-device TPU ``jax.sharding.Mesh`` and ``jax.shard_map`` parallelization for GX-JAX.

Ported from the MPI + NCCL species/Hermite domain decomposition in:
- ``gx/src/grids.cu`` (``iproc_s``, ``iproc_m``, ``nprocs_s``, ``nprocs_m``)
- ``gx/src/moments.cu`` (``MomentsG::syncMPI``, ``MomentsG::syncNCCL``)
- ``gx/src/reductions.cu`` (``ncclAllReduce`` across ``ncclComm_s``, ``ncclComm_m``)
- ``gx/src/solver.cu`` (global species/moment reductions for ``(phi, apar, bpar)``)

Key design notes:
1. ``G[nspecies, nhermite, nlaguerre, nz, ky, kx]`` is partitioned across ``("sp", "m")`` via
   ``PartitionSpec("sp", "m", None, None, None, None)``, keeping the trailing 4D axes
   ``(nlaguerre, nz, ky, kx)`` 100% local on each TPU core so 1D parallel ``z`` FFTs,
   2D perpendicular ``(ky, kx)`` FFTs, and Laguerre ``dot_general`` transforms execute
   in-core on ``(8, 128)`` HBM tiles with zero cross-chip FFT all-to-alls.
2. Field solves and collision conservation moments reduce only 3D ``(nz, ky, kx)`` moments
   over ``("sp", "m")`` via ``jax.lax.psum``, working identically for both
   ``nhermite_local >= 2`` and ``nhermite_local == 1`` (where ``m=0`` and ``m=1`` reside on
   different ``"m"`` shards).
3. Hermite ``m +- 1`` and ``m +- 2`` couplings (parallel streaming, magnetic drifts,
   nonlinear ``A_par`` flutter, and top-of-chain ``Beer 4+2`` / ``SmithPar`` closures) use
   non-wrapping ``jax.lax.ppermute`` halo exchanges along ``"m"`` with strict zero masking at
   physical boundaries ``m < 0`` and ``m >= nhermite``.
4. Multi-host meshes (``jax.process_count() > 1``): every mesh placement goes through
   ``gx_jax.distributed.device_put_global`` (``jax.make_array_from_callback``; plain
   ``jax.device_put`` on a single host) and ``gather_g_state`` gathers with
   ``multihost_utils.process_allgather(..., tiled=True)`` in lockstep on every process.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from gx_jax.config import BoltzmannOption, GXConfig
from gx_jax.diagnostics_nc import (
    DiagnosticsSnapshot,
    DiagnosticsState,
    FluxBundle,
    WgSpectra,
    _reduce_species_summand_4d,
    compute_field_squared_spectra,
    compute_omega_kxky,
    compute_phi2_zonal_spectra,
    compute_wapar_spectra,
    compute_wbpar_spectra,
    compute_wphi_spectra,
    fourier_ky_weights,
    init_diagnostics_state,
)
from gx_jax.distributed import device_put_global, gather_global_array, is_distributed
from gx_jax.fields import (
    Fields,
    FieldSolver,
    _complex_and_real_dtypes,
    _safe_divide,
    _time_dtype,
    enforce_reality,
    mask_g,
)
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids
from gx_jax.integrator import (
    ChunkResult,
    GyrokineticOperator,
    IntegratorState,
    StepResult,
    _is_truly_fixed_dt,
    _resolve_initial_exb_state,
    _resolve_t_limit,
    _run_chunk_scan,
    _run_chunk_while,
    _scheme_honors_fixed_dt,
    _update_linear_op_for_exb,
    compute_cfl_dt,
    init_gyrokinetic_operator,
)
from gx_jax.laguerre import jfac, jflr, jflr_b, laguerre_to_grid, laguerre_to_spectral
from gx_jax.linear_rhs import (
    LinearOperator,
    apply_hb_hyper,
    apply_hyperdiff,
)
from gx_jax.nonlinear_exb import (
    ExBShearState,
    ForcingOperator,
    NonlinearOperator,
    _bessel_j0_runtime,
    _j1_over_alpha_runtime,
    _safe_sqrt,
    flow_shear_g_shift,
    flow_shear_shift,
    poisson_bracket_real,
)

_jax_shard_map: Any = getattr(jax, "shard_map", None)
if _jax_shard_map is None:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _jax_shard_map


def _shard_map(
    f: Any, *, mesh: Mesh, in_specs: Any, out_specs: Any, check_rep: bool = False
) -> Any:
    """Compatibility wrapper around ``jax.shard_map`` mapping ``check_rep`` to ``check_vma``."""
    try:
        return _jax_shard_map(
            f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_rep
        )
    except TypeError:  # pragma: no cover
        return _jax_shard_map(
            f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=check_rep
        )


# ---------------------------------------------------------------------------
# 1. Precision Enforcement, Mesh Construction, & Sharding Helpers
# ---------------------------------------------------------------------------


def configure_tpu_precision(enable_x64: bool | None = None) -> None:
    """Enforce ``jax_enable_x64`` and ``jax_default_matmul_precision="highest"``."""
    if enable_x64 is not None:
        jax.config.update("jax_enable_x64", bool(enable_x64))
    elif os.environ.get("JAX_ENABLE_X64") is not None:
        jax.config.update(
            "jax_enable_x64",
            os.environ.get("JAX_ENABLE_X64") in ("1", "true", "True"),
        )
    else:
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


@contextlib.contextmanager
def host_precompute_context():
    """Run IEEE ``float64`` operator/moment precomputation on the local host CPU when available."""
    try:
        cpu_devs = jax.local_devices(backend="cpu")
    except Exception:
        cpu_devs = []
    if cpu_devs:
        with jax.default_device(cpu_devs[0]):
            yield
    else:
        yield


def move_tree_to_default_device(tree: Any) -> Any:
    """Place host-precomputed tree leaves on the local primary default JAX device."""
    local_devs = jax.local_devices()
    cm = jax.default_device(local_devs[0]) if local_devs else contextlib.nullcontext()
    with cm:
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(np.asarray(x), dtype=x.dtype) if isinstance(x, jax.Array) else x,
            tree,
        )


def infer_mesh_shape(
    num_devices: int,
    *,
    nspecies: int | None = None,
    nhermite: int | None = None,
) -> tuple[int, int]:
    """Choose a valid 2D ``(n_sp, n_m)`` mesh shape for ``num_devices``."""
    if num_devices < 1:
        raise ValueError(f"num_devices={num_devices} must be >= 1.")
    if nspecies is None and nhermite is None:
        return (1, num_devices)

    ns = max(1, int(nspecies or 1))
    nm = max(1, int(nhermite or num_devices))
    best = (1, 1)
    for n_sp in range(1, min(ns, num_devices) + 1):
        if ns % n_sp != 0:
            continue
        for n_m in range(1, (num_devices // n_sp) + 1):
            if nm % n_m != 0:
                continue
            if n_sp * n_m > best[0] * best[1] or (
                n_sp * n_m == best[0] * best[1] and n_m > best[1]
            ):
                best = (n_sp, n_m)
    return best


def create_device_mesh(
    mesh_shape: tuple[int, int] | None = None,
    *,
    devices: Sequence[Any] | None = None,
    axis_names: tuple[str, str] = ("sp", "m"),
    nspecies: int | None = None,
    nhermite: int | None = None,
    enable_x64: bool | None = None,
) -> Mesh:
    """Create a 2D ``jax.sharding.Mesh`` with axes ``("sp", "m")``."""
    configure_tpu_precision(enable_x64=enable_x64)
    devs = list(jax.devices() if devices is None else devices)
    if not devs:
        raise RuntimeError("No JAX devices available to construct Mesh.")

    if mesh_shape is None:
        n_sp, n_m = infer_mesh_shape(len(devs), nspecies=nspecies, nhermite=nhermite)
    else:
        n_sp, n_m = int(mesh_shape[0]), int(mesh_shape[1])

    if n_sp < 1 or n_m < 1:
        raise ValueError(f"mesh_shape={(n_sp, n_m)} must have positive dimensions.")
    if n_sp * n_m > len(devs):
        raise ValueError(
            f"Requested mesh_shape={(n_sp, n_m)} requires {n_sp * n_m} devices, "
            f"but only {len(devs)} devices are available."
        )
    if nspecies is not None and nspecies % n_sp != 0:
        raise ValueError(f"nspecies={nspecies} is not divisible by mesh sp axis={n_sp}.")
    if nhermite is not None and nhermite % n_m != 0:
        raise ValueError(f"nhermite={nhermite} is not divisible by mesh m axis={n_m}.")

    dev_grid = np.asarray(devs[: n_sp * n_m], dtype=object).reshape(n_sp, n_m)
    return Mesh(dev_grid, axis_names=axis_names)


def g_partition_spec(axis_names: tuple[str, str] = ("sp", "m")) -> PartitionSpec:
    """Return ``PartitionSpec("sp", "m", None, None, None, None)`` for 6D ``G``."""
    return PartitionSpec(axis_names[0], axis_names[1], None, None, None, None)


def replicated_partition_spec() -> PartitionSpec:
    """Return fully replicated ``PartitionSpec()``."""
    return PartitionSpec()


def make_g_sharding(mesh: Mesh) -> NamedSharding:
    """Return :class:`NamedSharding` partitioning 6D ``G`` across ``mesh`` axes ``("sp", "m")``."""
    return NamedSharding(mesh, g_partition_spec(mesh.axis_names))


def make_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return replicated :class:`NamedSharding` on ``mesh``."""
    return NamedSharding(mesh, replicated_partition_spec())


def shard_g_state(G: jax.Array, mesh: Mesh) -> jax.Array:
    """Place 6D ``G[nspecies, nhermite, nlaguerre, nz, ky, kx]`` onto ``mesh``."""
    n_sp = int(mesh.shape[mesh.axis_names[0]])
    n_m = int(mesh.shape[mesh.axis_names[1]])
    if G.ndim != 6:
        raise ValueError(f"Expected 6D G state, got shape {G.shape}.")
    if G.shape[0] % n_sp != 0:
        raise ValueError(f"G.shape[0]={G.shape[0]} not divisible by mesh sp={n_sp}.")
    if G.shape[1] % n_m != 0:
        raise ValueError(f"G.shape[1]={G.shape[1]} not divisible by mesh m={n_m}.")
    return device_put_global(G, make_g_sharding(mesh))


def gather_g_state(G_sharded: jax.Array, mesh: Mesh | None = None) -> jax.Array:
    """Gather sharded 6D ``G`` into a replicated array.

    Multi-host (``jax.process_count() > 1``): every process must call this in lockstep;
    ``G`` is gathered with ``multihost_utils.process_allgather(..., tiled=True)`` and
    returned as a process-local (fully addressable) array on every process.
    """
    if is_distributed():
        return jnp.asarray(gather_global_array(G_sharded))
    if mesh is not None:
        return jax.device_put(G_sharded, make_replicated_sharding(mesh))
    return jnp.asarray(G_sharded)


# ---------------------------------------------------------------------------
# 2. Intra-Shard Helpers: Halo Exchange, Global Moment Extraction, & Slicing
# ---------------------------------------------------------------------------


def _slice_species_axis(arr: jax.Array, sp_idx: jax.Array, ns_loc: int, axis: int = 0) -> jax.Array:
    """Slice ``ns_loc`` entries along ``axis`` starting at ``sp_idx * ns_loc``."""
    return jax.lax.dynamic_slice_in_dim(arr, sp_idx * ns_loc, ns_loc, axis=axis)


def _slice_hermite_axis(arr: jax.Array, m_idx: jax.Array, nm_loc: int, axis: int = 0) -> jax.Array:
    """Slice ``nm_loc`` entries along ``axis`` starting at ``m_idx * nm_loc``."""
    return jax.lax.dynamic_slice_in_dim(arr, m_idx * nm_loc, nm_loc, axis=axis)


def exchange_hermite_halos(
    arr_local: jax.Array,
    *,
    m_axis: str = "m",
    n_m: int,
) -> tuple[jax.Array, jax.Array]:
    """Exchange width-2 non-wrapping Hermite ghost halos ``(left_2, right_2)`` along ``m_axis``.

    Supports BOTH ``nm_loc >= 2`` (single-hop ``lax.ppermute`` of 2 slices) and
    ``nm_loc == 1`` (distance-1 and distance-2 ``lax.ppermute`` matching ``MomentsG::syncNCCL``
    in ``moments.cu:L3730-3775``). Physical boundaries ``m < 0`` and ``m >= M_glob`` are
    strictly zero-masked (never wrapped periodically).
    """
    ns_loc = arr_local.shape[0]
    nm_loc = arr_local.shape[1]
    tail_shape = arr_local.shape[2:]
    zero_2 = jnp.zeros((ns_loc, 2, *tail_shape), dtype=arr_local.dtype)

    if n_m <= 1:
        return zero_2, zero_2

    m_idx = jax.lax.axis_index(m_axis)
    perm_to_right = [(i, i + 1) for i in range(n_m - 1)]
    perm_to_left = [(i, i - 1) for i in range(1, n_m)]

    if nm_loc >= 2:
        recv_from_left = jax.lax.ppermute(arr_local[:, -2:], m_axis, perm=perm_to_right)
        recv_from_right = jax.lax.ppermute(arr_local[:, :2], m_axis, perm=perm_to_left)
        left_2 = jnp.where(m_idx > 0, recv_from_left, zero_2)
        right_2 = jnp.where(m_idx < (n_m - 1), recv_from_right, zero_2)
        return left_2, right_2

    # Special case nm_loc == 1 (1 Hermite moment per "m" shard):
    zero_1 = jnp.zeros((ns_loc, 1, *tail_shape), dtype=arr_local.dtype)
    recv_l1 = jax.lax.ppermute(arr_local, m_axis, perm=perm_to_right)
    recv_r1 = jax.lax.ppermute(arr_local, m_axis, perm=perm_to_left)
    left_1 = jnp.where(m_idx >= 1, recv_l1, zero_1)
    right_1 = jnp.where(m_idx + 1 < n_m, recv_r1, zero_1)

    if n_m >= 3:
        perm_to_right2 = [(i, i + 2) for i in range(n_m - 2)]
        perm_to_left2 = [(i, i - 2) for i in range(2, n_m)]
        recv_l2 = jax.lax.ppermute(arr_local, m_axis, perm=perm_to_right2)
        recv_r2 = jax.lax.ppermute(arr_local, m_axis, perm=perm_to_left2)
        left_2_hop = jnp.where(m_idx >= 2, recv_l2, zero_1)
        right_2_hop = jnp.where(m_idx + 2 < n_m, recv_r2, zero_1)
    else:
        left_2_hop = zero_1
        right_2_hop = zero_1

    left_2 = jnp.concatenate([left_2_hop, left_1], axis=1)  # [m-2, m-1]
    right_2 = jnp.concatenate([right_1, right_2_hop], axis=1)  # [m+1, m+2]
    return left_2, right_2


def pad_hermite_with_halos(
    arr_local: jax.Array,
    *,
    m_axis: str = "m",
    n_m: int,
) -> jax.Array:
    """Return ``arr_padded`` of shape ``(ns_loc, nm_loc + 4, ...)`` with width-2 ``m`` halos."""
    left_2, right_2 = exchange_hermite_halos(arr_local, m_axis=m_axis, n_m=n_m)
    return jnp.concatenate([left_2, arr_local, right_2], axis=1)


def _extract_global_m_slice(
    arr_local: jax.Array,
    m_glob_vec: jax.Array,
    target_m: int,
    *,
    m_axis: str = "m",
    n_m: int,
) -> jax.Array:
    """Extract global Hermite slice ``arr[:, target_m, ...]`` across ``m_axis`` via ``psum``."""
    mask = (m_glob_vec == int(target_m))[None, :, None, None, None, None]
    loc = jnp.sum(jnp.where(mask, arr_local, jnp.zeros_like(arr_local)), axis=1)
    if n_m > 1:
        loc = jax.lax.psum(loc, m_axis)
    return loc


def _add_at_global_m(
    arr_local: jax.Array,
    m_glob_vec: jax.Array,
    target_m: int,
    delta_5d: jax.Array,
) -> jax.Array:
    """Add 5D ``delta_5d (ns_loc, nl, nz, naky, nakx)`` to ``arr_local`` at global ``target_m``."""
    mask = (m_glob_vec == int(target_m))[None, :, None, None, None, None]
    return arr_local + jnp.where(mask, delta_5d[:, None, ...], jnp.zeros_like(arr_local))


def _add_at_global_ml(
    arr_local: jax.Array,
    m_glob_vec: jax.Array,
    target_m: int,
    target_l: int,
    delta_4d: jax.Array,
) -> jax.Array:
    """Add 4D ``delta_4d (ns_loc, nz, naky, nakx)`` to ``arr_local`` at global ``(target_m,
    target_l)``.
    """
    nl = arr_local.shape[2]
    mask_ml = (m_glob_vec[:, None] == int(target_m)) & (jnp.arange(nl)[None, :] == int(target_l))
    mask6 = mask_ml[None, :, :, None, None, None]
    return arr_local + jnp.where(mask6, delta_4d[:, None, None, ...], jnp.zeros_like(arr_local))


def _zero_at_global_ml(
    arr_local: jax.Array,
    m_glob_vec: jax.Array,
    target_m: int,
    target_l: int,
) -> jax.Array:
    """Zero ``arr_local`` at global ``(target_m, target_l)``."""
    nl = arr_local.shape[2]
    mask_ml = (m_glob_vec[:, None] == int(target_m)) & (jnp.arange(nl)[None, :] == int(target_l))
    mask6 = mask_ml[None, :, :, None, None, None]
    return jnp.where(mask6, jnp.zeros_like(arr_local), arr_local)


# ---------------------------------------------------------------------------
# 3. Sharded Field Solver & Moments Inside shard_map
# ---------------------------------------------------------------------------


def _solve_fields_local(
    G_local: jax.Array,
    solver: FieldSolver,
    *,
    sp_axis: str = "sp",
    m_axis: str = "m",
    n_sp: int,
    n_m: int,
) -> Fields:
    """Solve ``(phi, apar, bpar, phi_avg)`` from sharded ``G_local`` using ``lax.psum``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    field_shape = (solver.nz, solver.naky, solver.nakx)
    zero3 = jnp.zeros(field_shape, dtype=c_dtype)
    zero1 = jnp.zeros((solver.nakx,), dtype=c_dtype)

    if solver.no_fields:
        return Fields(phi=zero3, apar=zero3, bpar=zero3, phi_avg=zero1)

    ns_loc = G_local.shape[0]
    nm_loc = G_local.shape[1]
    sp_idx = jax.lax.axis_index(sp_axis)
    m_idx = jax.lax.axis_index(m_axis)
    m_glob_vec = m_idx * nm_loc + jnp.arange(nm_loc, dtype=jnp.int32)

    jl_loc = _slice_species_axis(solver.jl.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jlb_loc = _slice_species_axis(solver.jlb.astype(r_dtype), sp_idx, ns_loc, axis=0)
    sp_nz_loc = _slice_species_axis(solver.sp_nz.astype(r_dtype), sp_idx, ns_loc, axis=0)
    sp_jpar_loc = _slice_species_axis(solver.sp_jparfac.astype(r_dtype), sp_idx, ns_loc, axis=0)
    sp_jperp_loc = _slice_species_axis(solver.sp_jperpfac.astype(r_dtype), sp_idx, ns_loc, axis=0)

    # Extract local contributions to m = 0 and m = 1 (works for nm_loc == 1 and nm_loc >= 2)
    mask_m0 = (m_glob_vec == 0)[None, :, None, None, None, None]
    mask_m1 = (m_glob_vec == 1)[None, :, None, None, None, None]
    g0_loc = jnp.sum(jnp.where(mask_m0, G_local.astype(c_dtype), jnp.zeros_like(G_local)), axis=1)
    g1_loc = jnp.sum(jnp.where(mask_m1, G_local.astype(c_dtype), jnp.zeros_like(G_local)), axis=1)

    w_nbar = sp_nz_loc[:, None, None, None, None] * jl_loc
    nbar_loc = jnp.einsum("slzyx,slzyx->zyx", w_nbar, g0_loc, precision=jax.lax.Precision.HIGHEST)

    if solver.fapar > 0.0 and solver.nhermite >= 2:
        w_jpar = sp_jpar_loc[:, None, None, None, None] * jl_loc
        jpar_loc = jnp.einsum(
            "slzyx,slzyx->zyx", w_jpar, g1_loc, precision=jax.lax.Precision.HIGHEST
        )
    else:
        jpar_loc = zero3

    if solver.all_kinetic and solver.fbpar > 0.0:
        bmag_inv = solver.bmag_inv.astype(r_dtype)
        bmag_inv2 = (bmag_inv * bmag_inv)[None, None, :, None, None]
        w_jperp = sp_jperp_loc[:, None, None, None, None] * bmag_inv2 * jlb_loc
        jperp_loc = jnp.einsum(
            "slzyx,slzyx->zyx", w_jperp, g0_loc, precision=jax.lax.Precision.HIGHEST
        )
    else:
        jperp_loc = zero3

    # Single global reduction across ("sp", "m")
    reduce_axes: list[str] = []
    if n_sp > 1:
        reduce_axes.append(sp_axis)
    if n_m > 1:
        reduce_axes.append(m_axis)
    if reduce_axes:
        ax_tuple = tuple(reduce_axes) if len(reduce_axes) > 1 else reduce_axes[0]
        nbar, jparbar, jperpbar = jax.lax.psum((nbar_loc, jpar_loc, jperp_loc), ax_tuple)
    else:
        nbar, jparbar, jperpbar = nbar_loc, jpar_loc, jperp_loc

    mask3d = solver.active_mask[None, :, :]
    nbar = jnp.where(mask3d, nbar, zero3)
    jparbar = jnp.where(mask3d, jparbar, zero3)
    jperpbar = jnp.where(mask3d, jperpbar, zero3)

    fphi = jnp.asarray(solver.fphi, dtype=r_dtype)
    fapar = jnp.asarray(solver.fapar, dtype=r_dtype)
    fbpar = jnp.asarray(solver.fbpar, dtype=r_dtype)

    if solver.all_kinetic:
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        if solver.fbpar > 0.0:
            q_bpar = solver.qneut_fac_bpar.astype(r_dtype)
            a_phi = solver.ampere_perp_fac_phi.astype(r_dtype)
            a_bpar = solver.ampere_perp_fac_bpar.astype(r_dtype)
            det = solver.phi_bpar_det.astype(r_dtype)
            num_phi = fphi * (a_bpar * nbar - q_bpar * jperpbar)
            num_bpar = fbpar * (-a_phi * nbar + q_phi * jperpbar)
            phi = _safe_divide(num_phi, det, mask3d)
            bpar = _safe_divide(num_bpar, det, mask3d)
        else:
            phi = _safe_divide(fphi * nbar, q_phi, mask3d) if solver.fphi != 0.0 else zero3
            bpar = zero3

        if solver.fapar > 0.0 and solver.nhermite >= 2:
            a_par = solver.ampere_par_fac.astype(r_dtype)
            apar = _safe_divide(fapar * jparbar, a_par, mask3d)
        else:
            apar = zero3
        phi_avg = zero1

    elif solver.boltzmann_opt == BoltzmannOption.BOLTZMANN_ELECTRONS:
        tau = jnp.asarray(solver.tau_fac, dtype=r_dtype)
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        denom = tau + q_phi
        jacobian = solver.jacobian.astype(r_dtype)
        mask_ky0 = solver.active_mask[0, :]
        tmp_ky0 = _safe_divide(nbar[:, 0, :], denom[:, 0, :], mask_ky0[None, :]) * jacobian[:, None]
        phi_avg_num = jnp.sum(tmp_ky0, axis=0)
        phi_avg_raw = _safe_divide(phi_avg_num, solver.phiavgdenom.astype(r_dtype), mask_ky0)
        if solver.fphi != 0.0:
            rhs_phi = nbar.at[:, 0, :].add(tau * phi_avg_raw[None, :])
            phi = _safe_divide(fphi * rhs_phi, denom, mask3d)
            phi_avg = (fphi * phi_avg_raw).astype(c_dtype)
        else:
            phi = zero3
            phi_avg = zero1
        apar = zero3
        bpar = zero3

    else:
        tau = jnp.asarray(solver.tau_fac, dtype=r_dtype)
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        denom = tau + q_phi
        phi = _safe_divide(fphi * nbar, denom, mask3d) if solver.fphi != 0.0 else zero3
        if solver.fapar > 0.0 and solver.nhermite >= 2:
            a_par = solver.ampere_par_fac.astype(r_dtype)
            apar = _safe_divide(fapar * jparbar, a_par, mask3d)
        else:
            apar = zero3
        bpar = zero3
        phi_avg = zero1

    if solver.phi_ext != 0.0:
        ext = jnp.asarray(solver.phi_ext, dtype=r_dtype)
        phi = jnp.where(mask3d, phi + ext, zero3)

    return Fields(
        phi=phi.astype(c_dtype),
        apar=apar.astype(c_dtype),
        bpar=bpar.astype(c_dtype),
        phi_avg=phi_avg.astype(c_dtype),
    )


def sharded_solve_fields(G_sharded: jax.Array, solver: FieldSolver, mesh: Mesh) -> Fields:
    """Solve ``(phi, apar, bpar, phi_avg)`` from sharded ``G`` via ``jax.shard_map``."""
    sp_axis, m_axis = mesh.axis_names
    n_sp = int(mesh.shape[sp_axis])
    n_m = int(mesh.shape[m_axis])
    g_spec = g_partition_spec(mesh.axis_names)
    rep_spec = replicated_partition_spec()

    def _fn(g_loc: jax.Array) -> Fields:
        return _solve_fields_local(
            g_loc, solver, sp_axis=sp_axis, m_axis=m_axis, n_sp=n_sp, n_m=n_m
        )

    return _shard_map(
        _fn,
        mesh=mesh,
        in_specs=(g_spec,),
        out_specs=Fields(phi=rep_spec, apar=rep_spec, bpar=rep_spec, phi_avg=rep_spec),
        check_rep=False,
    )(G_sharded)


# ---------------------------------------------------------------------------
# 4. Sharded Linear & Nonlinear RHS Kernels Inside shard_map
# ---------------------------------------------------------------------------


def _build_h_from_g_slice(
    g_slice: jax.Array,
    m_indices: jax.Array,
    fields: Fields,
    solver: FieldSolver,
    sp_idx: jax.Array,
) -> jax.Array:
    """Build non-adiabatic ``H`` on a Hermite slice with global Hermite indices ``m_indices``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(g_slice)
    ns_loc = g_slice.shape[0]
    jl_loc = _slice_species_axis(solver.jl.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jlb_loc = _slice_species_axis(solver.jlb.astype(r_dtype), sp_idx, ns_loc, axis=0)
    zt_loc = _slice_species_axis(solver.sp_zt.astype(r_dtype), sp_idx, ns_loc, axis=0)[
        :, None, None, None, None
    ]
    vt_loc = _slice_species_axis(solver.sp_vt.astype(r_dtype), sp_idx, ns_loc, axis=0)[
        :, None, None, None, None
    ]

    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]

    m0_add = zt_loc * jl_loc * phi + jlb_loc * bpar
    m1_add = -(zt_loc * vt_loc) * jl_loc * apar

    out = _add_at_global_m(g_slice.astype(c_dtype), m_indices, 0, m0_add)
    if solver.nhermite >= 2:
        out = _add_at_global_m(out, m_indices, 1, m1_add)
    valid_m = ((m_indices >= 0) & (m_indices < solver.nhermite))[None, :, None, None, None, None]
    return jnp.where(valid_m, out, jnp.zeros_like(out))


def _compute_conservation_moments_local(
    G_local: jax.Array,
    H_local: jax.Array,
    fields: Fields,
    solver: FieldSolver,
    sp_idx: jax.Array,
    m_glob_vec: jax.Array,
    *,
    m_axis: str = "m",
    n_m: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute per-local-species ``(upar_bar, uperp_bar, t_bar)`` via ``psum`` along ``m_axis``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    ns_loc = G_local.shape[0]
    mask4d = solver.active_mask[None, None, :, :]
    zero_sp = jnp.zeros((ns_loc, solver.nz, solver.naky, solver.nakx), dtype=c_dtype)

    jl_loc = _slice_species_axis(solver.jl.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jlb_loc = _slice_species_axis(solver.jlb.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jla_loc = _slice_species_axis(solver.jla.astype(r_dtype), sp_idx, ns_loc, axis=0)
    rho2_loc = _slice_species_axis(solver.sp_rho2.astype(r_dtype), sp_idx, ns_loc, axis=0)

    h0 = _extract_global_m_slice(H_local, m_glob_vec, 0, m_axis=m_axis, n_m=n_m)
    h1 = (
        _extract_global_m_slice(H_local, m_glob_vec, 1, m_axis=m_axis, n_m=n_m)
        if solver.nhermite >= 2
        else jnp.zeros_like(h0)
    )
    h2 = (
        _extract_global_m_slice(H_local, m_glob_vec, 2, m_axis=m_axis, n_m=n_m)
        if solver.nhermite >= 3
        else jnp.zeros_like(h0)
    )

    upar_bar = jnp.einsum("slzyx,slzyx->szyx", jl_loc, h1, precision=jax.lax.Precision.HIGHEST)
    uperp_raw = jnp.einsum("slzyx,slzyx->szyx", jlb_loc, h0, precision=jax.lax.Precision.HIGHEST)
    b_s = rho2_loc[:, None, None, None] * solver.kperp2.astype(r_dtype)[None, :, :, :]
    sqrt_b = jnp.sqrt(jnp.maximum(b_s, 0.0))
    uperp_bar = sqrt_b * uperp_raw

    if solver.nlaguerre > 1:
        inv_sqrt2 = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        two_thirds = jnp.asarray(2.0 / 3.0, dtype=r_dtype)
        t_bar_h0 = jnp.einsum("slzyx,slzyx->szyx", jla_loc, h0, precision=jax.lax.Precision.HIGHEST)
        t_bar_h2 = jnp.einsum("slzyx,slzyx->szyx", jl_loc, h2, precision=jax.lax.Precision.HIGHEST)
        t_bar = two_thirds * (t_bar_h0 + inv_sqrt2 * t_bar_h2)
    else:
        if solver.nhermite < 3:
            t_bar = zero_sp
        else:
            g2 = _extract_global_m_slice(G_local, m_glob_vec, 2, m_axis=m_axis, n_m=n_m)
            sqrt2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
            t_bar = sqrt2 * jl_loc[:, 0] * g2[:, 0]

    upar_bar = jnp.where(mask4d, upar_bar, zero_sp)
    uperp_bar = jnp.where(mask4d, uperp_bar, zero_sp)
    t_bar = jnp.where(mask4d, t_bar, zero_sp)
    return upar_bar, uperp_bar, t_bar


def _linear_rhs_local(
    G_local: jax.Array,
    fields: Fields,
    op: LinearOperator,
    dt: float | jax.Array,
    *,
    grids: Grids | None = None,
    geo: Geometry | None = None,
    sp_axis: str = "sp",
    m_axis: str = "m",
    n_m: int,
) -> jax.Array:
    """Evaluate ``Linear_GK::rhs`` on sharded ``G_local`` using non-wrapping ``m`` halo exchange."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    ns_loc, nm_loc, nl, _, _, _ = G_local.shape
    sp_idx = jax.lax.axis_index(sp_axis)
    m_idx = jax.lax.axis_index(m_axis)
    m_lo = m_idx * nm_loc
    m_glob_vec = m_lo + jnp.arange(nm_loc, dtype=jnp.int32)
    m_pad_glob = (m_lo - 2) + jnp.arange(nm_loc + 4, dtype=jnp.int32)

    mask6 = op.active_mask[None, None, None, None, :, :]
    g_c = jnp.where(mask6, G_local.astype(c_dtype), jnp.zeros_like(G_local, dtype=c_dtype))
    gp = op.grad_par

    # Single width-2 halo exchange on G_local gives both G_pad and H_pad
    g_pad_m = pad_hermite_with_halos(g_c, m_axis=m_axis, n_m=n_m)
    h_pad_m = _build_h_from_g_slice(g_pad_m, m_pad_glob, fields, op.solver, sp_idx)
    h_local = h_pad_m[:, 2 : 2 + nm_loc]

    # Slice per-species and per-Hermite precomputed vectors
    vt_loc = _slice_species_axis(op.sp_vt.astype(r_dtype), sp_idx, ns_loc, axis=0)
    zt_loc = _slice_species_axis(op.sp_zt.astype(r_dtype), sp_idx, ns_loc, axis=0)
    tz_loc = _slice_species_axis(op.sp_tz.astype(r_dtype), sp_idx, ns_loc, axis=0)
    nz_loc = _slice_species_axis(op.sp_nz.astype(r_dtype), sp_idx, ns_loc, axis=0)
    fprim_loc = _slice_species_axis(op.sp_fprim.astype(r_dtype), sp_idx, ns_loc, axis=0)
    tprim_loc = _slice_species_axis(op.sp_tprim.astype(r_dtype), sp_idx, ns_loc, axis=0)
    nu_ss_loc = _slice_species_axis(op.sp_nu_ss.astype(r_dtype), sp_idx, ns_loc, axis=0)
    nuei_loc = _slice_species_axis(op.sp_nuei.astype(r_dtype), sp_idx, ns_loc, axis=0)
    as_i_loc = _slice_species_axis(op.sp_as_i.astype(r_dtype), sp_idx, ns_loc, axis=0)
    vt_i_loc = _slice_species_axis(op.sp_vt_i.astype(r_dtype), sp_idx, ns_loc, axis=0)
    nzvt_i_loc = _slice_species_axis(op.sp_nzvt_i.astype(r_dtype), sp_idx, ns_loc, axis=0)
    has_upar_i_loc = _slice_species_axis(op.sp_has_upar_i, sp_idx, ns_loc, axis=0)

    jl_loc = _slice_species_axis(op.jl.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jlb_loc = _slice_species_axis(op.jlb.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jla_loc = _slice_species_axis(op.jla.astype(r_dtype), sp_idx, ns_loc, axis=0)
    jlba_loc = _slice_species_axis(op.jlba.astype(r_dtype), sp_idx, ns_loc, axis=0)
    b_s_loc = _slice_species_axis(op.b_s.astype(r_dtype), sp_idx, ns_loc, axis=0)
    sqrt_b_loc = _slice_species_axis(op.sqrt_b.astype(r_dtype), sp_idx, ns_loc, axis=0)

    sqrt_m_loc = _slice_hermite_axis(op.sqrt_m.astype(r_dtype), m_idx, nm_loc, axis=0)[
        None, :, None, None, None, None
    ]
    sqrt_mp1_loc = _slice_hermite_axis(op.sqrt_mp1.astype(r_dtype), m_idx, nm_loc, axis=0)[
        None, :, None, None, None, None
    ]
    sqrt_mp1_mp2_loc = _slice_hermite_axis(op.sqrt_mp1_mp2.astype(r_dtype), m_idx, nm_loc, axis=0)[
        None, :, None, None, None, None
    ]
    sqrt_m_mm1_loc = _slice_hermite_axis(op.sqrt_m_mm1.astype(r_dtype), m_idx, nm_loc, axis=0)[
        None, :, None, None, None, None
    ]
    two_m_p1_loc = _slice_hermite_axis(op.two_m_p1.astype(r_dtype), m_idx, nm_loc, axis=0)[
        None, :, None, None, None, None
    ]

    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]

    # 1. Parallel streaming rhs_par + dz(rhs_par)
    if op.nz > 1:
        vt5 = vt_loc[:, None, None, None, None]
        zt5 = zt_loc[:, None, None, None, None]
        gpar = jnp.asarray(op.gradpar, dtype=r_dtype)
        g_mp1 = g_pad_m[:, 3 : 3 + nm_loc]
        g_mm1 = g_pad_m[:, 1 : 1 + nm_loc]
        rhs_par = -(vt5[:, :, None, :, :, :] * gpar) * (sqrt_mp1_loc * g_mp1 + sqrt_m_loc * g_mm1)
        apar_m0 = jl_loc * apar * (zt5 * vt5 * vt5 * gpar)
        rhs_par = _add_at_global_m(rhs_par, m_glob_vec, 0, apar_m0)
        if op.nhermite >= 2:
            m1_term = (-jl_loc * phi * (zt5 * vt5) - jlb_loc * bpar * vt5) * gpar
            rhs_par = _add_at_global_m(rhs_par, m_glob_vec, 1, m1_term)
        if op.nhermite >= 3:
            rhs_par = _add_at_global_m(
                rhs_par,
                m_glob_vec,
                2,
                jnp.asarray(math.sqrt(2.0), dtype=r_dtype) * apar_m0,
            )
        rhs_par = jnp.where(mask6, rhs_par, jnp.zeros_like(rhs_par))
        rhs = gp.dz(rhs_par)
    else:
        rhs = jnp.zeros_like(g_c)

    # 2. Local linear RHS: mirror, curvature/grad-B drifts, collisions, and omega_* drives
    hp = jnp.pad(h_pad_m, ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))
    h_00 = hp[:, 2 : 2 + nm_loc, 1 : 1 + nl]
    h_mp1 = hp[:, 3 : 3 + nm_loc, 1 : 1 + nl]
    h_mm1 = hp[:, 1 : 1 + nm_loc, 1 : 1 + nl]
    h_mp2 = hp[:, 4 : 4 + nm_loc, 1 : 1 + nl]
    h_mm2 = hp[:, 0:nm_loc, 1 : 1 + nl]
    h_lp1 = hp[:, 2 : 2 + nm_loc, 2 : 2 + nl]
    h_lm1 = hp[:, 2 : 2 + nm_loc, 0:nl]
    h_mp1_lm1 = hp[:, 3 : 3 + nm_loc, 0:nl]
    h_mm1_lp1 = hp[:, 1 : 1 + nm_loc, 2 : 2 + nl]

    vt6 = vt_loc[:, None, None, None, None, None]
    tz6 = tz_loc[:, None, None, None, None, None]
    bgrad6 = op.bgrad.astype(r_dtype)[None, None, None, :, None, None]
    cv_r = (tz6 * op.cv_d.astype(r_dtype)[None, None, None, :, :, :]).astype(r_dtype)
    gb_r = (tz6 * op.gb_d.astype(r_dtype)[None, None, None, :, :, :]).astype(r_dtype)
    icv_d = jax.lax.complex(jnp.zeros_like(cv_r), cv_r).astype(c_dtype)
    igb_d = jax.lax.complex(jnp.zeros_like(gb_r), gb_r).astype(c_dtype)

    l_idx = op.l_idx.astype(r_dtype)[None, None, :, None, None, None]
    lp1 = op.lp1.astype(r_dtype)[None, None, :, None, None, None]
    two_l_p1 = op.two_l_p1.astype(r_dtype)[None, None, :, None, None, None]

    loc_out = (
        -vt6
        * bgrad6
        * (
            -sqrt_mp1_loc * (lp1 * h_mp1 + l_idx * h_mp1_lm1)
            + sqrt_m_loc * (l_idx * h_mm1 + lp1 * h_mm1_lp1)
        )
        - icv_d * (sqrt_mp1_mp2_loc * h_mp2 + two_m_p1_loc * h_00 + sqrt_m_mm1_loc * h_mm2)
        - igb_d * (lp1 * h_lp1 + two_l_p1 * h_00 + l_idx * h_lm1)
    )

    if op.collisions:
        nu_tot = (nu_ss_loc + nuei_loc)[:, None, None, None, None, None]
        b_s6 = b_s_loc[:, None, None, :, :, :]
        m_idx6 = m_glob_vec.astype(r_dtype)[None, :, None, None, None, None]
        loc_out = loc_out - nu_tot * (b_s6 + 2.0 * l_idx + m_idx6) * h_00

        nu5 = nu_ss_loc[:, None, None, None, None]
        nuei5 = nuei_loc[:, None, None, None, None]
        if op.coll_conservation:
            upar_bar, uperp_bar, t_bar = _compute_conservation_moments_local(
                g_c,
                h_local,
                fields,
                op.solver,
                sp_idx,
                m_glob_vec,
                m_axis=m_axis,
                n_m=n_m,
            )
            upar_c = upar_bar.astype(c_dtype)[:, None, :, :, :]
            uperp_c = uperp_bar.astype(c_dtype)[:, None, :, :, :]
            t_c = t_bar.astype(c_dtype)[:, None, :, :, :]
            sqrt_b5 = sqrt_b_loc[:, None, :, :, :]
            nu_tot5 = (nu_ss_loc + nuei_loc)[:, None, None, None, None]

            loc_out = _add_at_global_m(
                loc_out,
                m_glob_vec,
                0,
                nu5 * sqrt_b5 * jlb_loc * uperp_c + 2.0 * nu_tot5 * jla_loc * t_c,
            )
            if op.nhermite >= 3:
                loc_out = _add_at_global_m(
                    loc_out,
                    m_glob_vec,
                    2,
                    nu_tot5 * jnp.asarray(math.sqrt(2.0), dtype=r_dtype) * jl_loc * t_c,
                )
        else:
            upar_c = jnp.zeros((ns_loc, 1, op.nz, op.naky, op.nakx), dtype=c_dtype)

        if op.nhermite >= 2 and (op.coll_conservation or op.ei_colls):
            kp2_b2 = (op.kperp2.astype(r_dtype) * (op.bmag.astype(r_dtype)[:, None, None] ** 2))[
                None, None, :, :, :
            ]
            as_i5 = as_i_loc[:, None, None, None, None]
            nz_vt5 = (nz_loc * vt_loc)[:, None, None, None, None]
            nzvt_i5 = nzvt_i_loc[:, None, None, None, None]
            upar_i_raw = (kp2_b2 * apar) / as_i5 - (nz_vt5 / nzvt_i5) * upar_c
            upar_i = jnp.where(
                has_upar_i_loc[:, None, None, None, None],
                upar_i_raw,
                jnp.zeros_like(upar_i_raw),
            )
            vti_ov_vt = (vt_i_loc / vt_loc)[:, None, None, None, None]
            loc_out = _add_at_global_m(
                loc_out,
                m_glob_vec,
                1,
                jl_loc * (nu5 * upar_c + nuei5 * vti_ov_vt * upar_i),
            )

    vt5 = vt_loc[:, None, None, None, None]
    tz5 = tz_loc[:, None, None, None, None]
    fprim5 = fprim_loc[:, None, None, None, None]
    tprim5 = tprim_loc[:, None, None, None, None]
    kyr = op.ky.astype(r_dtype)[None, None, None, :, None]
    iky = jax.lax.complex(jnp.zeros_like(kyr), kyr).astype(c_dtype)

    loc_out = _add_at_global_m(
        loc_out,
        m_glob_vec,
        0,
        iky * phi * (fprim5 * jl_loc + tprim5 * jla_loc)
        + (iky * tz5) * bpar * (fprim5 * jlb_loc + tprim5 * jlba_loc),
    )
    if op.nhermite >= 2:
        loc_out = _add_at_global_m(
            loc_out,
            m_glob_vec,
            1,
            -vt5 * iky * apar * ((fprim5 + tprim5) * jl_loc + tprim5 * jla_loc),
        )
    if op.nhermite >= 3:
        inv_sqrt2 = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        loc_out = _add_at_global_m(
            loc_out,
            m_glob_vec,
            2,
            iky * phi * jl_loc * (tprim5 * inv_sqrt2)
            + (iky * tz5) * bpar * jlb_loc * (tprim5 * inv_sqrt2),
        )
    if op.nhermite >= 4:
        sqrt_3_2 = jnp.asarray(math.sqrt(1.5), dtype=r_dtype)
        loc_out = _add_at_global_m(
            loc_out,
            m_glob_vec,
            3,
            -vt5 * iky * apar * (sqrt_3_2 * tprim5 * jl_loc),
        )

    rhs = rhs + jnp.where(mask6, loc_out, jnp.zeros_like(loc_out))

    # 3. HB_hyper
    if op.hb_hyper:
        if grids is None or geo is None:
            raise ValueError("grids and geo are required when cfg.hb_hyper=True.")
        rhs = rhs + apply_hb_hyper(g_c, fields, grids, geo, op)

    # 4. Closures (beer4+2, smith_perp, smith_par)
    mask5 = op.active_mask[None, None, None, :, :]
    if op.closure_model == "beer4+2":
        vt4 = vt_loc[:, None, None, None]
        tz4 = tz_loc[:, None, None, None]
        gpar = jnp.asarray(op.gradpar, dtype=r_dtype)

        d_par = 2.0 * math.sqrt(math.pi) / (3.0 * math.pi - 8.0)
        beta_par = (32.0 - 9.0 * math.pi) / (3.0 * math.pi - 8.0)
        d_perp = 0.5 * math.sqrt(math.pi)

        rhs = _zero_at_global_ml(rhs, m_glob_vec, 2, 1)
        rhs = _zero_at_global_ml(rhs, m_glob_vec, 3, 1)

        g_m0 = _extract_global_m_slice(g_c, m_glob_vec, 0, m_axis=m_axis, n_m=n_m)
        g_m1 = _extract_global_m_slice(g_c, m_glob_vec, 1, m_axis=m_axis, n_m=n_m)
        g_m2 = _extract_global_m_slice(g_c, m_glob_vec, 2, m_axis=m_axis, n_m=n_m)
        g_m3 = _extract_global_m_slice(g_c, m_glob_vec, 3, m_axis=m_axis, n_m=n_m)

        g02 = g_m2[:, 0]
        g10 = g_m0[:, 1]
        g01 = g_m1[:, 0]
        g03 = g_m3[:, 0]
        g11 = g_m1[:, 1]

        dz_g20 = gp.dz(g02)
        abs_dz_g30 = gp.abs_dz(g03)
        abs_dz_g11 = gp.abs_dz(g11)

        c_bpar = jnp.asarray(-beta_par / math.sqrt(3.0), dtype=r_dtype) * gpar * vt4
        c_dpar = jnp.asarray(-math.sqrt(2.0) * d_par, dtype=r_dtype) * gpar * vt4
        c_dperp = jnp.asarray(-math.sqrt(2.0) * d_perp, dtype=r_dtype) * gpar * vt4

        wd = op.omegad.astype(r_dtype)[None, :, :, :]
        wd_r = (wd * tz4).astype(r_dtype)
        iwd_s = jax.lax.complex(jnp.zeros_like(wd_r), wd_r).astype(c_dtype)
        awd_s = jnp.abs(wd) * tz4

        nu = op.closure_coeff.astype(c_dtype)
        nu_r = jnp.real(nu)
        nu_i = jnp.imag(nu)
        s2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
        s6 = jnp.asarray(math.sqrt(6.0), dtype=r_dtype)
        inv_s6 = jnp.asarray(1.0 / math.sqrt(6.0), dtype=r_dtype)

        t_02 = -s2 * (
            awd_s * (nu_r[1] * s2 * g02 + nu_r[2] * g10)
            + iwd_s * (nu_i[1] * s2 * g02 + nu_i[2] * g10)
        )
        t_10 = -2.0 * (
            awd_s * (nu_r[3] * s2 * g02 + nu_r[4] * g10)
            + iwd_s * (nu_i[3] * s2 * g02 + nu_i[4] * g10)
        )
        t_03 = (
            c_bpar * dz_g20
            + c_dpar * abs_dz_g30
            - inv_s6
            * (
                awd_s * (nu_r[5] * g01 + nu_r[6] * s6 * g03 + nu_r[7] * g11)
                + iwd_s * (nu_i[5] * g01 + nu_i[6] * s6 * g03 + nu_i[7] * g11)
            )
        )
        t_11 = c_dperp * abs_dz_g11 - (
            awd_s * (nu_r[8] * g01 + nu_r[9] * s6 * g03 + nu_r[10] * g11)
            + iwd_s * (nu_i[8] * g01 + nu_i[9] * s6 * g03 + nu_i[10] * g11)
        )

        rhs = _add_at_global_ml(rhs, m_glob_vec, 2, 0, t_02)
        rhs = _add_at_global_ml(rhs, m_glob_vec, 0, 1, t_10)
        rhs = _add_at_global_ml(rhs, m_glob_vec, 3, 0, t_03)
        rhs = _add_at_global_ml(rhs, m_glob_vec, 1, 1, t_11)
        rhs = jnp.where(mask6, rhs, jnp.zeros_like(rhs))

    elif op.closure_model == "smith_perp":
        q = op.smith_perp_q
        ell_last = op.nlaguerre - 1
        aclos_rev = op.closure_coeff.astype(c_dtype)[::-1]
        ar = jnp.real(aclos_rev)[None, None, :, None, None, None]
        ai = jnp.imag(aclos_rev)[None, None, :, None, None, None]

        wd5 = op.omegad.astype(r_dtype)[None, None, :, :, :]
        tz5 = tz_loc[:, None, None, None, None]
        wd5_r = (wd5 * tz5).astype(r_dtype)[:, :, None, :, :, :]
        iwd_s = jax.lax.complex(jnp.zeros_like(wd5_r), wd5_r).astype(c_dtype)
        awd_s = (jnp.abs(wd5) * tz5)[:, :, None, :, :, :]

        g_tail = g_c[:, :, op.nlaguerre - q :]
        clos = jnp.sum((awd_s * ai + iwd_s * ar) * g_tail, axis=2)
        rhs = rhs.at[:, :, ell_last].add(
            -jnp.asarray(float(ell_last + 1), dtype=r_dtype)
            * jnp.where(mask5, clos, jnp.zeros_like(clos))
        )
        rhs = jnp.where(mask6, rhs, jnp.zeros_like(rhs))

    elif op.closure_model == "smith_par":
        q = op.smith_par_q
        m_last = op.nhermite - 1
        acoeff = op.closure_coeff.astype(c_dtype)[::-1]  # ordered m = nm - q .. m_last
        tail_slices = [
            _extract_global_m_slice(g_c, m_glob_vec, m_val, m_axis=m_axis, n_m=n_m)
            for m_val in range(op.nhermite - q, op.nhermite)
        ]
        g_tail = jnp.stack(tail_slices, axis=1)  # (ns_loc, q, nl, nz, naky, nakx)
        dz_tail = gp.dz(g_tail)
        abs_dz_tail = gp.abs_dz(g_tail)
        ar = jnp.real(acoeff)[None, :, None, None, None, None]
        ai = jnp.imag(acoeff)[None, :, None, None, None, None]
        clos = jnp.sum(-ai * abs_dz_tail + ar * dz_tail, axis=1)
        fac = (
            -jnp.asarray(math.sqrt(float(m_last + 1)) * op.gradpar, dtype=r_dtype)
            * vt_loc[:, None, None, None, None]
        )
        rhs = _add_at_global_m(
            rhs, m_glob_vec, m_last, fac * jnp.where(mask5, clos, jnp.zeros_like(clos))
        )
        rhs = jnp.where(mask6, rhs, jnp.zeros_like(rhs))

    # 5. Hypercollisions & hyperviscosity
    if op.hypercollisions_kz and op.nz > 1:
        fac_kz_sp = _slice_species_axis(op.hypercoll_kz_fac.astype(r_dtype), sp_idx, ns_loc, axis=0)
        fac_kz_loc = _slice_hermite_axis(fac_kz_sp, m_idx, nm_loc, axis=1)[
            :, :, None, None, None, None
        ]
        rhs = rhs + gp.abs_dz(-fac_kz_loc * g_c)

    if op.hyperz:
        rhs = gp.hyperz(g_c, op.nu_hyper_z, accumulate=True, target=rhs)

    if op.hypercollisions_const:
        inv_dt = jnp.asarray(1.0, dtype=r_dtype) / jnp.asarray(dt, dtype=r_dtype)
        rate_sp = _slice_species_axis(
            op.hypercoll_const_rate.astype(r_dtype)
            + op.hypercoll_const_lm.astype(r_dtype) * inv_dt,
            sp_idx,
            ns_loc,
            axis=0,
        )
        rate_loc = _slice_hermite_axis(rate_sp, m_idx, nm_loc, axis=1)[:, :, :, None, None, None]
        rhs = rhs + jnp.where(mask6, -rate_loc * g_c, jnp.zeros_like(g_c))

    if op.hyper:
        rhs = rhs + apply_hyperdiff(g_c, op)

    # 6. Parallel end-damping boundary conditions (dampEnds_linked / dampEnds_linkedNTFT)
    rhs = gp.apply_bcs(g_c, rhs, fields, op.solver, dt, H=h_local)
    return jnp.where(mask6, rhs, jnp.zeros_like(rhs))


def _nonlinear_rhs_local(
    G_local: jax.Array,
    fields: Fields,
    nl_op: NonlinearOperator,
    *,
    exb_state: ExBShearState | None = None,
    sp_axis: str = "sp",
    m_axis: str = "m",
    n_m: int,
) -> jax.Array:
    """Evaluate 3/2-dealiased ExB + flutter nonlinear bracket on sharded ``G_local``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    if not nl_op.nonlinear_mode:
        return jnp.zeros_like(G_local, dtype=c_dtype)

    ns_loc, nm_loc, _, _, _, _ = G_local.shape
    sp_idx = jax.lax.axis_index(sp_axis)
    m_idx = jax.lax.axis_index(m_axis)

    gp = nl_op.grad_perp
    lag = nl_op.laguerre
    mask5 = nl_op.active_mask[None, None, None, :, :]
    mask6 = nl_op.active_mask[None, None, None, None, :, :]
    g_c = jnp.where(mask6, G_local.astype(c_dtype), jnp.zeros_like(G_local, dtype=c_dtype))

    kxstar = exb_state.kxstar if exb_state is not None else None
    pf_exb_nxny = exb_state.phasefac_exb_nxny if exb_state is not None else None
    pfm_exb_nxny = exb_state.phasefacminus_exb_nxny if exb_state is not None else None
    use_exb_phase = exb_state is not None and exb_state.exb_shear_phase
    use_phase = nl_op.non_twist or use_exb_phase

    # 1. Local real-space dG/dx and dG/dy -> Laguerre grid (ns_loc, nm_loc, nj, nz, nx, ny)
    if nl_op.non_twist:
        dG_dx = gp.ntft_dx_c2r_nxny(g_c, kxstar=kxstar)
    else:
        dG_dx = gp.dx_c2r_nxny(g_c, kxstar=kxstar)
    if use_phase:
        dG_dx = gp.phase_mult_nxny(
            dG_dx,
            non_twist=nl_op.non_twist,
            exb_shear=use_exb_phase,
            positive_phase=True,
            phasefac_exb_nxny=pf_exb_nxny,
            phasefacminus_exb_nxny=pfm_exb_nxny,
        )
    dg_dx = laguerre_to_grid(dG_dx, lag, axis=2)

    dG_dy = gp.dy_c2r_nxny(g_c)
    if use_phase:
        dG_dy = gp.phase_mult_nxny(
            dG_dy,
            non_twist=nl_op.non_twist,
            exb_shear=use_exb_phase,
            positive_phase=True,
            phasefac_exb_nxny=pf_exb_nxny,
            phasefacminus_exb_nxny=pfm_exb_nxny,
        )
    dg_dy = laguerre_to_grid(dG_dy, lag, axis=2)

    # 2. Gyroaveraged potentials on local species slice (ns_loc, nj, nz, naky, nakx)
    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]
    fphi = jnp.asarray(nl_op.fphi, dtype=r_dtype)
    fapar = jnp.asarray(nl_op.fapar, dtype=r_dtype)
    fbpar = jnp.asarray(nl_op.fbpar, dtype=r_dtype)

    if exb_state is not None and nl_op.g_exb != 0.0:
        kp2 = exb_state.kperp2.astype(r_dtype)[None, None, :, :, :]
        mu_b = nl_op.laguerre.roots.astype(r_dtype)[None, :, None, None, None]
        rho2_loc = _slice_species_axis(nl_op.sp_rho2.astype(r_dtype), sp_idx, ns_loc, axis=0)[
            :, None, None, None, None
        ]
        tz_loc = _slice_species_axis(nl_op.sp_tz.astype(r_dtype), sp_idx, ns_loc, axis=0)[
            :, None, None, None, None
        ]
        two = jnp.asarray(2.0, dtype=r_dtype)
        alpha = _safe_sqrt(two * mu_b * kp2 * rho2_loc)
        j0_g = _bessel_j0_runtime(alpha)
        if nl_op.fbpar > 0.0:
            j1_oa_g = _j1_over_alpha_runtime(alpha)
            tz_j1_g = tz_loc * two * mu_b * j1_oa_g
            j0_phi_bpar = j0_g * (phi * fphi) + tz_j1_g * (bpar * fbpar)
        else:
            j0_phi_bpar = j0_g * (phi * fphi)
    else:
        j0_g = _slice_species_axis(nl_op.j0_grid.astype(r_dtype), sp_idx, ns_loc, axis=0)
        if nl_op.fbpar > 0.0:
            tz_j1_g = _slice_species_axis(
                nl_op.tz_2mub_j1_oa_grid.astype(r_dtype), sp_idx, ns_loc, axis=0
            )
            j0_phi_bpar = j0_g * (phi * fphi) + tz_j1_g * (bpar * fbpar)
        else:
            j0_phi_bpar = j0_g * (phi * fphi)

    j0_phi_bpar = jnp.where(mask5, j0_phi_bpar, jnp.zeros_like(j0_phi_bpar))

    if nl_op.non_twist:
        dj0phi_dx = gp.ntft_dx_c2r_nxny(j0_phi_bpar, kxstar=kxstar)
    else:
        dj0phi_dx = gp.dx_c2r_nxny(j0_phi_bpar, kxstar=kxstar)
    if use_phase:
        dj0phi_dx = gp.phase_mult_nxny(
            dj0phi_dx,
            non_twist=nl_op.non_twist,
            exb_shear=use_exb_phase,
            positive_phase=True,
            phasefac_exb_nxny=pf_exb_nxny,
            phasefacminus_exb_nxny=pfm_exb_nxny,
        )

    dj0phi_dy = gp.dy_c2r_nxny(j0_phi_bpar)
    if use_phase:
        dj0phi_dy = gp.phase_mult_nxny(
            dj0phi_dy,
            non_twist=nl_op.non_twist,
            exb_shear=use_exb_phase,
            positive_phase=True,
            phasefac_exb_nxny=pf_exb_nxny,
            phasefacminus_exb_nxny=pfm_exb_nxny,
        )

    brk_phi = poisson_bracket_real(dg_dx, dg_dy, dj0phi_dx, dj0phi_dy, nl_op.kxfac)
    dG_res = laguerre_to_spectral(brk_phi, lag, axis=2)
    if use_phase:
        dG_res = gp.phase_mult_nxny(
            dG_res,
            non_twist=nl_op.non_twist,
            exb_shear=use_exb_phase,
            positive_phase=False,
            phasefac_exb_nxny=pf_exb_nxny,
            phasefacminus_exb_nxny=pfm_exb_nxny,
        )
    rhs_nl = gp.r2c_nxny(dG_res)

    # 3. Electromagnetic A_par flutter bracket with m +- 1 halo exchange across "m"
    if nl_op.fapar > 0.0:
        j0_apar = jnp.where(mask5, j0_g * (apar * fapar), jnp.zeros_like(j0_phi_bpar))
        if nl_op.non_twist:
            dj0a_dx = gp.ntft_dx_c2r_nxny(j0_apar, kxstar=kxstar)
        else:
            dj0a_dx = gp.dx_c2r_nxny(j0_apar, kxstar=kxstar)
        if use_phase:
            dj0a_dx = gp.phase_mult_nxny(
                dj0a_dx,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=True,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )

        dj0a_dy = gp.dy_c2r_nxny(j0_apar)
        if use_phase:
            dj0a_dy = gp.phase_mult_nxny(
                dj0a_dy,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=True,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )

        brk_apar = poisson_bracket_real(dg_dx, dg_dy, dj0a_dx, dj0a_dy, nl_op.kxfac)
        dG_apar = laguerre_to_spectral(brk_apar, lag, axis=2)
        if use_phase:
            dG_apar = gp.phase_mult_nxny(
                dG_apar,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=False,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )
        nl_apar_loc = gp.r2c_nxny(dG_apar)
        nl_apar_pad = pad_hermite_with_halos(nl_apar_loc, m_axis=m_axis, n_m=n_m)

        vt_loc = _slice_species_axis(nl_op.sp_vt.astype(r_dtype), sp_idx, ns_loc, axis=0)[
            :, None, None, None, None, None
        ]
        sm_loc = _slice_hermite_axis(nl_op.sqrt_m.astype(r_dtype), m_idx, nm_loc, axis=0)[
            None, :, None, None, None, None
        ]
        smp1_loc = _slice_hermite_axis(nl_op.sqrt_mp1.astype(r_dtype), m_idx, nm_loc, axis=0)[
            None, :, None, None, None, None
        ]
        nl_mp1 = nl_apar_pad[:, 3 : 3 + nm_loc]
        nl_mm1 = nl_apar_pad[:, 1 : 1 + nm_loc]
        rhs_nl = rhs_nl - vt_loc * (smp1_loc * nl_mp1 + sm_loc * nl_mm1)

    return jnp.where(mask6, rhs_nl, jnp.zeros_like(rhs_nl))


def sharded_linear_rhs(
    G_sharded: jax.Array,
    fields: Fields,
    op: LinearOperator,
    dt: float | jax.Array,
    mesh: Mesh,
    *,
    grids: Grids | None = None,
    geo: Geometry | None = None,
) -> jax.Array:
    """Evaluate ``linear_rhs`` on sharded ``G`` via ``jax.shard_map``."""
    sp_axis, m_axis = mesh.axis_names
    n_m = int(mesh.shape[m_axis])
    g_spec = g_partition_spec(mesh.axis_names)
    rep_spec = replicated_partition_spec()
    f_spec = Fields(phi=rep_spec, apar=rep_spec, bpar=rep_spec, phi_avg=rep_spec)
    _, r_dtype = _complex_and_real_dtypes(G_sharded)
    dt_arr = jnp.asarray(dt, dtype=_time_dtype(r_dtype))

    def _fn(g_loc: jax.Array, f_in: Fields, dt_in: jax.Array) -> jax.Array:
        return _linear_rhs_local(
            g_loc,
            f_in,
            op,
            dt_in,
            grids=grids,
            geo=geo,
            sp_axis=sp_axis,
            m_axis=m_axis,
            n_m=n_m,
        )

    return _shard_map(
        _fn,
        mesh=mesh,
        in_specs=(g_spec, f_spec, rep_spec),
        out_specs=g_spec,
        check_rep=False,
    )(G_sharded, fields, dt_arr)


def sharded_nonlinear_rhs(
    G_sharded: jax.Array,
    fields: Fields,
    nl_op: NonlinearOperator,
    mesh: Mesh,
    *,
    exb_state: ExBShearState | None = None,
) -> jax.Array:
    """Evaluate ``nonlinear_rhs`` on sharded ``G`` via ``jax.shard_map``."""
    sp_axis, m_axis = mesh.axis_names
    n_m = int(mesh.shape[m_axis])
    g_spec = g_partition_spec(mesh.axis_names)
    rep_spec = replicated_partition_spec()
    f_spec = Fields(phi=rep_spec, apar=rep_spec, bpar=rep_spec, phi_avg=rep_spec)

    if exb_state is None:

        def _fn_no_exb(g_loc: jax.Array, f_in: Fields) -> jax.Array:
            return _nonlinear_rhs_local(
                g_loc, f_in, nl_op, exb_state=None, sp_axis=sp_axis, m_axis=m_axis, n_m=n_m
            )

        return _shard_map(
            _fn_no_exb,
            mesh=mesh,
            in_specs=(g_spec, f_spec),
            out_specs=g_spec,
            check_rep=False,
        )(G_sharded, fields)

    def _fn_exb(g_loc: jax.Array, f_in: Fields, exb_in: ExBShearState) -> jax.Array:
        return _nonlinear_rhs_local(
            g_loc, f_in, nl_op, exb_state=exb_in, sp_axis=sp_axis, m_axis=m_axis, n_m=n_m
        )

    return _shard_map(
        _fn_exb,
        mesh=mesh,
        in_specs=(g_spec, f_spec, rep_spec),
        out_specs=g_spec,
        check_rep=False,
    )(G_sharded, fields, exb_state)


def _gyrokinetic_rhs_local(
    G_local: jax.Array,
    fields: Fields,
    gk_op: GyrokineticOperator,
    dt: float | jax.Array,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias_linear: bool = False,
    apply_kz_dealias_nonlinear: bool = False,
    sp_axis: str = "sp",
    m_axis: str = "m",
    n_m: int,
) -> jax.Array:
    """Evaluate unified ``nonlinear_rhs + linear_rhs`` inside ``shard_map``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    g_c = G_local.astype(c_dtype)
    f_c = fields.astype(c_dtype)
    dt_r = jnp.asarray(dt, dtype=r_dtype)
    dt_safe = jnp.where(dt_r > 0.0, dt_r, jnp.ones_like(dt_r))

    lin_op = _update_linear_op_for_exb(gk_op.linear_op, exb_state, r_dtype)
    gp = lin_op.grad_par

    rhs_lin = _linear_rhs_local(
        g_c,
        f_c,
        lin_op,
        dt_safe,
        grids=gk_op.grids,
        geo=gk_op.geo,
        sp_axis=sp_axis,
        m_axis=m_axis,
        n_m=n_m,
    )
    if apply_kz_dealias_linear:
        rhs_lin = gp.dealias(rhs_lin)

    if gk_op.nonlinear_mode and gk_op.nonlinear_op is not None:
        rhs_nl = _nonlinear_rhs_local(
            g_c,
            f_c,
            gk_op.nonlinear_op,
            exb_state=exb_state,
            sp_axis=sp_axis,
            m_axis=m_axis,
            n_m=n_m,
        )
        if apply_kz_dealias_nonlinear:
            rhs_nl = gp.dealias(rhs_nl)
        total = rhs_nl + rhs_lin
    else:
        total = rhs_lin

    mask6 = lin_op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, total, jnp.zeros_like(total))


def sharded_gyrokinetic_rhs(
    G_sharded: jax.Array,
    fields: Fields,
    gk_op: GyrokineticOperator,
    dt: float | jax.Array,
    mesh: Mesh,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias_linear: bool = False,
    apply_kz_dealias_nonlinear: bool = False,
) -> jax.Array:
    """Evaluate ``gyrokinetic_rhs`` on sharded ``G`` via ``jax.shard_map``."""
    sp_axis, m_axis = mesh.axis_names
    n_m = int(mesh.shape[m_axis])
    g_spec = g_partition_spec(mesh.axis_names)
    rep_spec = replicated_partition_spec()
    f_spec = Fields(phi=rep_spec, apar=rep_spec, bpar=rep_spec, phi_avg=rep_spec)
    _, r_dtype = _complex_and_real_dtypes(G_sharded)
    dt_arr = jnp.asarray(dt, dtype=_time_dtype(r_dtype))

    if exb_state is None:

        def _fn_no_exb(g_loc: jax.Array, f_in: Fields, dt_in: jax.Array) -> jax.Array:
            return _gyrokinetic_rhs_local(
                g_loc,
                f_in,
                gk_op,
                dt_in,
                exb_state=None,
                apply_kz_dealias_linear=apply_kz_dealias_linear,
                apply_kz_dealias_nonlinear=apply_kz_dealias_nonlinear,
                sp_axis=sp_axis,
                m_axis=m_axis,
                n_m=n_m,
            )

        return _shard_map(
            _fn_no_exb,
            mesh=mesh,
            in_specs=(g_spec, f_spec, rep_spec),
            out_specs=g_spec,
            check_rep=False,
        )(G_sharded, fields, dt_arr)

    def _fn_exb(
        g_loc: jax.Array, f_in: Fields, dt_in: jax.Array, exb_in: ExBShearState
    ) -> jax.Array:
        return _gyrokinetic_rhs_local(
            g_loc,
            f_in,
            gk_op,
            dt_in,
            exb_state=exb_in,
            apply_kz_dealias_linear=apply_kz_dealias_linear,
            apply_kz_dealias_nonlinear=apply_kz_dealias_nonlinear,
            sp_axis=sp_axis,
            m_axis=m_axis,
            n_m=n_m,
        )

    return _shard_map(
        _fn_exb,
        mesh=mesh,
        in_specs=(g_spec, f_spec, rep_spec, rep_spec),
        out_specs=g_spec,
        check_rep=False,
    )(G_sharded, fields, dt_arr, exb_state)


# ---------------------------------------------------------------------------
# 5. Sharded Forcing, Time Integrators, & Sharded Diagnostics Inside shard_map
# ---------------------------------------------------------------------------


def _apply_forcing_local(
    G_local: jax.Array,
    forcing_op: ForcingOperator,
    dt: float | jax.Array,
    key: jax.Array | None,
    *,
    step_idx: int | jax.Array = 0,
    nspecies_glob: int,
    nhermite_glob: int,
    sp_axis: str = "sp",
    m_axis: str = "m",
) -> jax.Array:
    """Apply stochastic/impulse forcing on ``G_local`` matching single-device PRNG draws."""
    if not forcing_op.forcing_init:
        return G_local
    if key is None:
        raise ValueError("A JAX PRNGKey is required when forcing_init=True.")

    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    g_c = G_local.astype(c_dtype)
    ns_loc, nm_loc, nl, _, _, _ = g_c.shape
    sp_idx = jax.lax.axis_index(sp_axis)
    m_idx = jax.lax.axis_index(m_axis)
    m_glob_vec = m_idx * nm_loc + jnp.arange(nm_loc, dtype=jnp.int32)

    dt_r = jnp.asarray(dt, dtype=r_dtype)
    k1, k2 = jax.random.split(key)

    # Draw global (nspecies_glob,) vectors and slice by sp_idx so "sp" sharding matches
    # single-device apply_forcing bit-for-bit!
    u1_glob = jnp.clip(
        jax.random.uniform(k1, (nspecies_glob,), dtype=r_dtype),
        jnp.asarray(1.0e-12, dtype=r_dtype),
        jnp.asarray(1.0, dtype=r_dtype),
    )
    phase_glob = jnp.asarray(math.pi, dtype=r_dtype) * (
        jnp.asarray(2.0, dtype=r_dtype) * jax.random.uniform(k2, (nspecies_glob,), dtype=r_dtype)
        - jnp.asarray(1.0, dtype=r_dtype)
    )
    u1 = _slice_species_axis(u1_glob, sp_idx, ns_loc, axis=0)
    phase = _slice_species_axis(phase_glob, sp_idx, ns_loc, axis=0)

    f_amp_r = jnp.asarray(forcing_op.forcing_amp, dtype=r_dtype)
    amp = _safe_sqrt(jnp.abs(f_amp_r * dt_r * jnp.log(u1)))

    amp_r = amp.astype(r_dtype)
    phase_r = phase.astype(r_dtype)
    ftype = forcing_op.forcing_type.strip().lower()
    if ftype == "kz":
        rf = (amp_r * jnp.cos(phase_r)).astype(c_dtype)
    elif ftype == "kzimpulse":
        rf_raw = jax.lax.complex(amp_r * jnp.cos(phase_r), amp_r * jnp.sin(phase_r)).astype(c_dtype)
        rf = jnp.where(jnp.asarray(step_idx) == 0, rf_raw, jnp.zeros((ns_loc,), dtype=c_dtype))
    else:
        rf = jax.lax.complex(amp_r * jnp.cos(phase_r), amp_r * jnp.sin(phase_r)).astype(c_dtype)

    iz = forcing_op.idz_target
    iy = forcing_op.idy_target
    ix = forcing_op.idx_target
    sfield = forcing_op.stir_field
    s2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
    s6 = jnp.asarray(math.sqrt(6.0), dtype=r_dtype)

    def _kick_ml(arr: jax.Array, target_m: int, target_l: int, val_s: jax.Array) -> jax.Array:
        mask_m = (m_glob_vec == target_m)[:, None]  # (nm_loc, 1)
        mask_l = (jnp.arange(nl, dtype=jnp.int32) == target_l)[None, :]  # (1, nl)
        mask_ml = (mask_m & mask_l)[None, :, :]  # (1, nm_loc, nl)
        cur = arr[:, :, :, iz, iy, ix]
        updated = cur + jnp.where(mask_ml, val_s[:, None, None], jnp.zeros_like(cur))
        return arr.at[:, :, :, iz, iy, ix].set(updated)

    if sfield in ("density", "ppar", "pperp"):
        g_c = _kick_ml(g_c, 0, 0, rf)
    if sfield == "upar" and nhermite_glob >= 2:
        g_c = _kick_ml(g_c, 1, 0, rf)
    if sfield in ("tpar", "ppar") and nhermite_glob >= 3:
        g_c = _kick_ml(g_c, 2, 0, s2 * rf)
    if sfield in ("tperp", "pperp") and nl >= 2:
        g_c = _kick_ml(g_c, 0, 1, rf)
    if sfield == "qpar" and nhermite_glob >= 4:
        g_c = _kick_ml(g_c, 3, 0, s6 * rf)
    if sfield == "qperp" and nhermite_glob >= 2 and nl >= 2:
        g_c = _kick_ml(g_c, 1, 1, s2 * rf)
    return g_c


def _solve_stage_fields_local(
    G_local: jax.Array,
    gk_op: GyrokineticOperator,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias: bool = False,
    sp_axis: str = "sp",
    m_axis: str = "m",
    n_sp: int,
    n_m: int,
) -> Fields:
    """Solve stage fields inside ``shard_map`` with optional ``dealias_kz`` on ``phi``."""
    _, r_dtype = _complex_and_real_dtypes(G_local)
    lin_op = _update_linear_op_for_exb(gk_op.linear_op, exb_state, r_dtype)
    f_out = _solve_fields_local(
        G_local, lin_op.solver, sp_axis=sp_axis, m_axis=m_axis, n_sp=n_sp, n_m=n_m
    )
    if apply_kz_dealias and gk_op.dealias_kz:
        phi_d = lin_op.grad_par.dealias(f_out.phi)
        f_out = Fields(phi=phi_d, apar=f_out.apar, bpar=f_out.bpar, phi_avg=f_out.phi_avg)
    return f_out


def _step_gyrokinetic_local(
    G_local: jax.Array,
    fields: Fields,
    gk_op: GyrokineticOperator,
    dt: float | jax.Array | None,
    *,
    scheme: str | None = None,
    exb_state: ExBShearState | None = None,
    enforce_reality_cond: bool = False,
    key: jax.Array | None = None,
    step_idx: int | jax.Array = 0,
    sp_axis: str = "sp",
    m_axis: str = "m",
    n_sp: int,
    n_m: int,
) -> StepResult:
    """Advance sharded ``(G_local, fields)`` by one step inside ``shard_map``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G_local)
    t_dtype = _time_dtype(r_dtype)
    g0 = G_local.astype(c_dtype)
    f0 = fields.astype(c_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    dealias_kz = bool(gk_op.dealias_kz)
    exb_cur = exb_state.astype(r_dtype) if exb_state is not None else None

    if dt is None:
        dt_f64 = compute_cfl_dt(f0, gk_op, exb_state=exb_cur, scheme=sk)
    else:
        dt_f64 = jnp.asarray(dt, dtype=t_dtype)
    dt_r = dt_f64.astype(r_dtype)

    if exb_cur is not None and sk in ("rk3", "sspx2", "sspx3", "k10"):
        exb_cur, f0 = flow_shear_shift(exb_cur, f0, gk_op.grids, gk_op.geo, dt_r)
        g0 = flow_shear_g_shift(g0, exb_cur, gk_op.grids)

    def _rhs(
        g_in: jax.Array,
        f_in: Fields,
        exb_in: ExBShearState | None,
        *,
        d_lin: bool = False,
        d_nl: bool = False,
    ) -> jax.Array:
        return _gyrokinetic_rhs_local(
            g_in,
            f_in,
            gk_op,
            dt_r,
            exb_state=exb_in,
            apply_kz_dealias_linear=d_lin,
            apply_kz_dealias_nonlinear=d_nl,
            sp_axis=sp_axis,
            m_axis=m_axis,
            n_m=n_m,
        )

    def _solve(g_in: jax.Array, exb_in: ExBShearState | None, *, d_kz: bool = False) -> Fields:
        return _solve_stage_fields_local(
            g_in,
            gk_op,
            exb_state=exb_in,
            apply_kz_dealias=d_kz,
            sp_axis=sp_axis,
            m_axis=m_axis,
            n_sp=n_sp,
            n_m=n_m,
        )

    if sk == "rk4":
        half = jnp.asarray(0.5, dtype=r_dtype)
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        two = jnp.asarray(2.0, dtype=r_dtype)

        k1 = _rhs(g0, f0, exb_cur)
        g1 = g0 + (half * dt_r) * k1
        f1 = _solve(g1, exb_cur)

        if exb_cur is not None:
            exb_cur, f1 = flow_shear_shift(exb_cur, f1, gk_op.grids, gk_op.geo, half * dt_r)
            g0 = flow_shear_g_shift(g0, exb_cur, gk_op.grids)
            k1 = flow_shear_g_shift(k1, exb_cur, gk_op.grids)
            g1 = flow_shear_g_shift(g1, exb_cur, gk_op.grids)

        k2 = _rhs(g1, f1, exb_cur)
        g2 = g0 + (half * dt_r) * k2
        f2 = _solve(g2, exb_cur)

        k3 = _rhs(g2, f2, exb_cur)
        g3 = g0 + dt_r * k3
        f3 = _solve(g3, exb_cur)

        if exb_cur is not None:
            exb_cur, f3 = flow_shear_shift(exb_cur, f3, gk_op.grids, gk_op.geo, half * dt_r)
            g0 = flow_shear_g_shift(g0, exb_cur, gk_op.grids)
            k1 = flow_shear_g_shift(k1, exb_cur, gk_op.grids)
            k2 = flow_shear_g_shift(k2, exb_cur, gk_op.grids)
            k3 = flow_shear_g_shift(k3, exb_cur, gk_op.grids)
            g3 = flow_shear_g_shift(g3, exb_cur, gk_op.grids)

        k4 = _rhs(g3, f3, exb_cur)
        g_next = g0 + (sixth * dt_r) * (k1 + two * k2 + two * k3 + k4)
        apply_final_kz_dealias = False

    elif sk == "rk3":
        c13 = jnp.asarray(1.0 / 3.0, dtype=r_dtype)
        c23 = jnp.asarray(2.0 / 3.0, dtype=r_dtype)
        c34 = jnp.asarray(0.75, dtype=r_dtype)
        c14 = jnp.asarray(0.25, dtype=r_dtype)

        k1 = _rhs(g0, f0, exb_cur)
        g_q1 = g0 + (c13 * dt_r) * k1
        f_q1 = _solve(g_q1, exb_cur)

        k2 = _rhs(g_q1, f_q1, exb_cur)
        g_q2 = g0 + (c23 * dt_r) * k2
        f_q2 = _solve(g_q2, exb_cur)

        k3 = _rhs(g_q2, f_q2, exb_cur)
        g_next = g0 + (c34 * dt_r) * k3 + (c14 * dt_r) * k1
        apply_final_kz_dealias = False

    elif sk == "sspx2":
        adt = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        c_d0 = jnp.asarray(math.sqrt(2.0) - 1.0, dtype=r_dtype)

        d0 = (adt * dt_r) * _rhs(g0, f0, exb_cur)
        g1 = g0 + d0
        f1 = _solve(g1, exb_cur)

        d1 = (adt * dt_r) * _rhs(g1, f1, exb_cur)
        g_next = g0 + (c_d0 * d0 + d1)
        apply_final_kz_dealias = False

    elif sk == "sspx3":
        adt_f = (1.0 / 6.0) ** (1.0 / 3.0)
        wgtfac = math.sqrt(9.0 - 2.0 * (6.0 ** (2.0 / 3.0)))
        w1_f = 0.5 * (wgtfac - 1.0)
        w2_f = 0.5 * (6.0 ** (2.0 / 3.0) - 1.0 - wgtfac)
        w3_f = 1.0 / adt_f - 1.0 - w2_f * (w1_f + 1.0)

        adt = jnp.asarray(adt_f, dtype=r_dtype)
        w1 = jnp.asarray(w1_f, dtype=r_dtype)
        w2 = jnp.asarray(w2_f, dtype=r_dtype)
        w3 = jnp.asarray(w3_f, dtype=r_dtype)

        d0 = (adt * dt_r) * _rhs(g0, f0, exb_cur, d_lin=dealias_kz, d_nl=False)
        g1 = g0 + d0
        f1 = _solve(g1, exb_cur, d_kz=True)

        d1 = (adt * dt_r) * _rhs(g1, f1, exb_cur, d_lin=dealias_kz, d_nl=False)
        d2 = w1 * d0 + d1
        g2 = g0 + d2
        f2 = _solve(g2, exb_cur, d_kz=True)

        d_last = (adt * dt_r) * _rhs(g2, f2, exb_cur, d_lin=dealias_kz, d_nl=False)
        g_next = g0 + (w3 * d0 + w2 * d2 + d_last)
        apply_final_kz_dealias = True

    elif sk == "k10":
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        tenth = jnp.asarray(0.1, dtype=r_dtype)

        def _euler_k10(
            carry: tuple[jax.Array, jax.Array, Fields],
        ) -> tuple[jax.Array, jax.Array, Fields]:
            dq, gq, fq = carry
            rq = _rhs(gq, fq, exb_cur, d_lin=dealias_kz, d_nl=False)
            dq_next = dq + (sixth * dt_r) * rq
            gq_next = g0 + dq_next
            fq_next = _solve(gq_next, exb_cur, d_kz=True)
            return dq_next, gq_next, fq_next

        d_acc = jnp.zeros_like(g0)
        g_q1, f_q1 = g0, f0
        for _ in range(5):
            d_acc, g_q1, f_q1 = _euler_k10((d_acc, g_q1, f_q1))
        d5 = d_acc

        d_acc = jnp.asarray(0.4, dtype=r_dtype) * d5
        g_q1 = g0 + d_acc
        f_q1 = _solve(g_q1, exb_cur, d_kz=True)

        for _ in range(4):
            d_acc, g_q1, f_q1 = _euler_k10((d_acc, g_q1, f_q1))
        d9 = d_acc

        r_last = _rhs(g_q1, f_q1, exb_cur, d_lin=dealias_kz, d_nl=dealias_kz)
        g_next = g0 + (
            jnp.asarray(0.36, dtype=r_dtype) * d5
            + jnp.asarray(0.6, dtype=r_dtype) * d9
            + (tenth * dt_r) * r_last
        )
        apply_final_kz_dealias = True

    else:
        raise ValueError(f"Unsupported scheme={sk!r}.")

    if gk_op.forcing_init and gk_op.forcing_op is not None:
        g_next = _apply_forcing_local(
            g_next,
            gk_op.forcing_op,
            dt_r,
            key,
            step_idx=step_idx,
            nspecies_glob=gk_op.grids.nspecies,
            nhermite_glob=gk_op.grids.nhermite,
            sp_axis=sp_axis,
            m_axis=m_axis,
        )

    mask6 = gk_op.grids.active_mask[None, None, None, None, :, :]
    g_next = jnp.where(mask6, g_next, jnp.zeros_like(g_next))
    if enforce_reality_cond:
        kx_cutoff = (gk_op.grids.nx - 1) // 3
        g_next = g_next.at[..., 0, 0].set(jnp.zeros_like(g_next[..., 0, 0]))
        if kx_cutoff > 0:
            pos_slice = g_next[..., 0, 1 : kx_cutoff + 1]
            g_next = g_next.at[..., 0, kx_cutoff + 1 :].set(jnp.conj(pos_slice[..., ::-1]))

    f_next = _solve(g_next, exb_cur, d_kz=apply_final_kz_dealias)
    return StepResult(
        G=g_next.astype(c_dtype),
        fields=f_next.astype(c_dtype),
        dt=dt_f64.astype(t_dtype),
        exb_state=exb_cur,
    )


def step_sharded_gyrokinetic(
    G_sharded: jax.Array,
    fields: Fields,
    gk_op: GyrokineticOperator,
    mesh: Mesh,
    dt: float | jax.Array | None = None,
    *,
    scheme: str | None = None,
    exb_state: ExBShearState | None = None,
    enforce_reality_cond: bool = False,
    key: jax.Array | None = None,
    step_idx: int | jax.Array = 0,
) -> StepResult:
    """Advance sharded ``(G_sharded, fields)`` by one step via ``jax.shard_map``."""
    sp_axis, m_axis = mesh.axis_names
    n_sp = int(mesh.shape[sp_axis])
    n_m = int(mesh.shape[m_axis])
    g_spec = g_partition_spec(mesh.axis_names)
    rep_spec = replicated_partition_spec()
    f_spec = Fields(phi=rep_spec, apar=rep_spec, bpar=rep_spec, phi_avg=rep_spec)
    _, r_dtype = _complex_and_real_dtypes(G_sharded)
    t_dtype = _time_dtype(r_dtype)

    key_use = jax.random.PRNGKey(gk_op.random_seed) if key is None else key
    step_i = jnp.asarray(step_idx, dtype=jnp.int32)
    dt_in = None if dt is None else jnp.asarray(dt, dtype=t_dtype)

    out_specs = StepResult(
        G=g_spec,
        fields=f_spec,
        dt=rep_spec,
        exb_state=rep_spec if exb_state is not None else None,
    )

    def _fn(
        g_loc: jax.Array,
        f_in: Fields,
        dt_arg: jax.Array | None,
        exb_arg: ExBShearState | None,
        k_arg: jax.Array,
        s_arg: jax.Array,
    ) -> StepResult:
        return _step_gyrokinetic_local(
            g_loc,
            f_in,
            gk_op,
            dt_arg,
            scheme=scheme,
            exb_state=exb_arg,
            enforce_reality_cond=enforce_reality_cond,
            key=k_arg,
            step_idx=s_arg,
            sp_axis=sp_axis,
            m_axis=m_axis,
            n_sp=n_sp,
            n_m=n_m,
        )

    return _shard_map(
        _fn,
        mesh=mesh,
        in_specs=(
            g_spec,
            f_spec,
            rep_spec if dt_in is not None else None,
            rep_spec if exb_state is not None else None,
            rep_spec,
            rep_spec,
        ),
        out_specs=out_specs,
        check_rep=False,
    )(G_sharded, fields, dt_in, exb_state, key_use, step_i)


def init_sharded_integrator_state(
    G: jax.Array,
    gk_op: GyrokineticOperator,
    mesh: Mesh,
    *,
    time: float | jax.Array = 0.0,
    dt: float | jax.Array | None = None,
    enforce_reality_cond: bool = False,
    exb_state: ExBShearState | None = None,
    key: jax.Array | None = None,
) -> IntegratorState:
    """Initialize :class:`IntegratorState` with ``G`` sharded across ``mesh``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    t_dtype = _time_dtype(r_dtype)
    g0 = mask_g(G.astype(c_dtype), gk_op.grids)
    if enforce_reality_cond:
        g0 = enforce_reality(g0, gk_op.grids)
    g_sharded = shard_g_state(g0, mesh)
    rep_sharding = make_replicated_sharding(mesh)
    exb0 = _resolve_initial_exb_state(gk_op, exb_state, r_dtype)
    if exb0 is not None:
        exb0 = device_put_global(exb0, rep_sharding)
    lin_op = _update_linear_op_for_exb(gk_op.linear_op, exb0, r_dtype)
    f0 = device_put_global(
        sharded_solve_fields(g_sharded, lin_op.solver, mesh).astype(c_dtype),
        rep_sharding,
    )
    dt0 = (
        compute_cfl_dt(f0, gk_op, exb_state=exb0) if dt is None else jnp.asarray(dt, dtype=t_dtype)
    )
    key0 = jax.random.PRNGKey(gk_op.random_seed) if key is None else key
    return IntegratorState(
        G=g_sharded,
        fields=f0,
        time=device_put_global(jnp.asarray(time, dtype=t_dtype), rep_sharding),
        dt=device_put_global(dt0.astype(t_dtype), rep_sharding),
        step=device_put_global(jnp.asarray(0, dtype=jnp.int32), rep_sharding),
        exb_state=exb0,
        key=device_put_global(key0, rep_sharding),
    )


def step_sharded_integrator_state(
    state: IntegratorState,
    gk_op: GyrokineticOperator,
    mesh: Mesh,
    *,
    dt_override: float | jax.Array | None = None,
    scheme: str | None = None,
    enforce_reality_cond: bool = False,
) -> IntegratorState:
    """Advance sharded :class:`IntegratorState` by one step."""
    c_dtype, r_dtype = _complex_and_real_dtypes(state.G)
    t_dtype = _time_dtype(r_dtype)
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    exb_in = state.exb_state.astype(r_dtype) if state.exb_state is not None else None

    if dt_override is not None:
        dt_step = jnp.asarray(dt_override, dtype=t_dtype)
    elif gk_op.fixed_dt and _scheme_honors_fixed_dt(sk):
        dt_step = state.dt.astype(t_dtype)
    else:
        dt_step = compute_cfl_dt(state.fields, gk_op, exb_state=exb_in, scheme=sk)

    next_key, step_key = jax.random.split(state.key)
    g_next, f_next, dt_used, exb_next = step_sharded_gyrokinetic(
        state.G,
        state.fields,
        gk_op,
        mesh,
        dt=dt_step,
        scheme=sk,
        exb_state=exb_in,
        enforce_reality_cond=enforce_reality_cond,
        key=step_key,
        step_idx=state.step,
    )
    return IntegratorState(
        G=g_next.astype(c_dtype),
        fields=f_next.astype(c_dtype),
        time=(state.time.astype(t_dtype) + dt_used.astype(t_dtype)),
        dt=dt_used.astype(t_dtype),
        step=(state.step + jnp.int32(1)).astype(jnp.int32),
        exb_state=exb_next,
        key=next_key,
    )


# ---------------------------------------------------------------------------
# 6. Sharded Diagnostics Reductions Inside shard_map (Zero 6D All-Gather)
# ---------------------------------------------------------------------------


def sharded_compute_diagnostics(
    G_sharded: jax.Array,
    fields: Fields,
    diag_state: DiagnosticsState,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
    mesh: Mesh,
    *,
    time: float | jax.Array,
    dt: float | jax.Array,
    beta: float = 0.0,
) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
    """Compute a complete :class:`DiagnosticsSnapshot` from sharded ``G`` without 6D gather."""
    sp_axis, m_axis = mesh.axis_names
    n_sp = int(mesh.shape[sp_axis])
    n_m = int(mesh.shape[m_axis])
    g_spec = g_partition_spec(mesh.axis_names)
    rep_spec = replicated_partition_spec()
    f_spec = Fields(phi=rep_spec, apar=rep_spec, bpar=rep_spec, phi_avg=rep_spec)

    t_dtype = _time_dtype(jnp.real(G_sharded).dtype)
    t_arr = jnp.asarray(time, dtype=t_dtype)
    dt_arr = jnp.asarray(dt, dtype=t_dtype)

    # Replicated field diagnostics
    phi2 = compute_field_squared_spectra(fields.phi, grids, geo)
    phi2_zonal = compute_phi2_zonal_spectra(fields.phi, grids, geo)
    apar2 = compute_field_squared_spectra(fields.apar, grids, geo)
    bpar2 = compute_field_squared_spectra(fields.bpar, grids, geo)
    wphi = compute_wphi_spectra(fields.phi, grids, geo, solver, phi_avg=fields.phi_avg)
    wapar = compute_wapar_spectra(fields.apar, grids, geo, grids.nspecies)
    wbpar = compute_wbpar_spectra(fields.bpar, grids, geo, grids.nspecies, beta=beta)
    omega_kxkyt = compute_omega_kxky(fields.phi, diag_state.fields_old.phi, dt_arr, grids)

    # Sharded moment & flux reductions via shard_map
    def _moment_and_flux_reductions(
        g_loc: jax.Array,
        g_old_loc: jax.Array,
        f_cur: Fields,
        f_old: Fields,
        dt_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        real_dtype = jnp.real(g_loc).dtype
        ns_loc, nm_loc, nl, nz, naky, _ = g_loc.shape
        sp_idx = jax.lax.axis_index(sp_axis)
        m_idx = jax.lax.axis_index(m_axis)
        m_glob_vec = m_idx * nm_loc + jnp.arange(nm_loc, dtype=jnp.int32)

        fac = fourier_ky_weights(grids, dtype=real_dtype)  # (naky, nakx)
        vol = geo.vol_fac.astype(real_dtype)  # (nz,)
        nt_loc = _slice_species_axis(solver.sp_nt.astype(real_dtype), sp_idx, ns_loc, axis=0)
        dens_loc = _slice_species_axis(solver.sp_dens.astype(real_dtype), sp_idx, ns_loc, axis=0)
        rho2_loc = _slice_species_axis(solver.sp_rho2.astype(real_dtype), sp_idx, ns_loc, axis=0)
        vt_loc = _slice_species_axis(solver.sp_vt.astype(real_dtype), sp_idx, ns_loc, axis=0)
        tz_loc = _slice_species_axis(solver.sp_tz.astype(real_dtype), sp_idx, ns_loc, axis=0)
        zt_loc = _slice_species_axis(solver.sp_zt.astype(real_dtype), sp_idx, ns_loc, axis=0)
        nz_s_loc = _slice_species_axis(solver.sp_nz.astype(real_dtype), sp_idx, ns_loc, axis=0)

        # 1. Wg spatial (ns_loc, nz, naky, nakx) and moment (ns_loc, nm_loc, nl)
        wg_6d = (
            0.5
            * fac[None, None, None, None, :, :]
            * vol[None, None, None, :, None, None]
            * nt_loc[:, None, None, None, None, None]
            * (jnp.abs(g_loc) ** 2)
        )
        wg_szyx_loc = jnp.sum(wg_6d, axis=(1, 2))
        if n_m > 1:
            wg_szyx_loc = jax.lax.psum(wg_szyx_loc, m_axis)
        wg_lm_loc = jnp.sum(wg_6d, axis=(3, 4, 5))  # (ns_loc, nm_loc, nl)
        if n_m > 1:
            wg_lm_sp = jax.lax.all_gather(wg_lm_loc, m_axis, axis=1, tiled=True)
        else:
            wg_lm_sp = wg_lm_loc

        # 2. Extract global m = 0, 1, 2, 3 slices across m_axis via psum
        g_m0 = _extract_global_m_slice(g_loc, m_glob_vec, 0, m_axis=m_axis, n_m=n_m)
        g_m1 = (
            _extract_global_m_slice(g_loc, m_glob_vec, 1, m_axis=m_axis, n_m=n_m)
            if grids.nhermite >= 2
            else jnp.zeros_like(g_m0)
        )
        g_m2 = (
            _extract_global_m_slice(g_loc, m_glob_vec, 2, m_axis=m_axis, n_m=n_m)
            if grids.nhermite >= 3
            else jnp.zeros_like(g_m0)
        )
        g_m3 = (
            _extract_global_m_slice(g_loc, m_glob_vec, 3, m_axis=m_axis, n_m=n_m)
            if grids.nhermite >= 4
            else jnp.zeros_like(g_m0)
        )

        g_old_m0 = _extract_global_m_slice(g_old_loc, m_glob_vec, 0, m_axis=m_axis, n_m=n_m)
        g_old_m1 = (
            _extract_global_m_slice(g_old_loc, m_glob_vec, 1, m_axis=m_axis, n_m=n_m)
            if grids.nhermite >= 2
            else jnp.zeros_like(g_old_m0)
        )

        ky_3d = grids.ky.astype(real_dtype)[None, :, None]
        iky_3d = jax.lax.complex(jnp.zeros_like(ky_3d), ky_3d)
        vphi_r = iky_3d * f_cur.phi
        va_r = iky_3d * f_cur.apar
        vb_r = iky_3d * f_cur.bpar

        kperp2 = geo.kperp2.astype(real_dtype)[None, None, :, :, :]
        b_s = kperp2 * rho2_loc[:, None, None, None, None]
        l_idx = jnp.arange(nl, dtype=real_dtype)[None, :, None, None, None]

        jf_l = jfac(l_idx, b_s)
        jf_lm1 = jfac(l_idx - 1.0, b_s)
        j0_l = jflr(l_idx, b_s)
        jb_l = jflr_b(l_idx, b_s)

        inv_sqrt2 = jnp.asarray(1.0 / np.sqrt(2.0), dtype=real_dtype)
        sqrt_1p5 = jnp.asarray(np.sqrt(1.5), dtype=real_dtype)

        p_bar = jnp.sum(jf_l * g_m0 + inv_sqrt2 * j0_l * g_m2, axis=1)
        q_bar = jnp.sum(jf_l * g_m1 + j0_l * (sqrt_1p5 * g_m3 + g_m1), axis=1)
        qb_bar = jnp.sum((jf_l + jf_lm1) * g_m0 + inv_sqrt2 * jb_l * g_m2, axis=1)

        n_bar_g = jnp.sum(j0_l * g_m0, axis=1)
        u_bar_g = jnp.sum(j0_l * g_m1, axis=1)
        ub_bar_g = jnp.sum(jb_l * g_m0, axis=1)

        ky_gt0_mask = (jnp.arange(naky) > 0)[None, None, :, None] & grids.active_mask[
            None, None, :, :
        ]
        flx_jac = geo.flux_fac.astype(real_dtype)[None, :, None, None]
        pres = nt_loc[:, None, None, None]
        dens4 = (
            jnp.zeros_like(dens_loc)[:, None, None, None]
            if grids.nspecies == 1
            else dens_loc[:, None, None, None]
        )
        vt4 = vt_loc[:, None, None, None]
        tz4 = tz_loc[:, None, None, None]
        two = jnp.asarray(2.0, dtype=real_dtype)
        zero = jnp.asarray(0.0, dtype=real_dtype)

        q_es = jnp.where(
            ky_gt0_mask, two * flx_jac * pres * jnp.real(jnp.conj(vphi_r)[None, ...] * p_bar), zero
        )
        q_apar = jnp.where(
            ky_gt0_mask,
            two * flx_jac * pres * jnp.real(-vt4 * jnp.conj(va_r)[None, ...] * q_bar),
            zero,
        )
        q_bpar = jnp.where(
            ky_gt0_mask,
            two * flx_jac * pres * jnp.real(tz4 * jnp.conj(vb_r)[None, ...] * qb_bar),
            zero,
        )
        q_stack_loc = jnp.stack([q_es + q_apar + q_bpar, q_es, q_apar, q_bpar], axis=1)

        p_es = jnp.where(
            ky_gt0_mask,
            two * flx_jac * dens4 * jnp.real(jnp.conj(vphi_r)[None, ...] * n_bar_g),
            zero,
        )
        p_apar = jnp.where(
            ky_gt0_mask,
            two * flx_jac * dens4 * jnp.real(-vt4 * jnp.conj(va_r)[None, ...] * u_bar_g),
            zero,
        )
        p_bpar = jnp.where(
            ky_gt0_mask,
            two * flx_jac * dens4 * jnp.real(tz4 * jnp.conj(vb_r)[None, ...] * ub_bar_g),
            zero,
        )
        p_stack_loc = jnp.stack([p_es + p_apar + p_bpar, p_es, p_apar, p_bpar], axis=1)

        # 3. TurbulentHeating
        dt_safe = jnp.where(
            jnp.abs(dt_in.astype(real_dtype)) > 1.0e-30,
            dt_in.astype(real_dtype),
            jnp.asarray(1.0, dtype=real_dtype),
        )
        dphidt = (f_cur.phi - f_old.phi) / dt_safe
        dadt = (f_cur.apar - f_old.apar) / dt_safe
        dbdt = (f_cur.bpar - f_old.bpar) / dt_safe

        zt5 = zt_loc[:, None, None, None, None]
        vt5 = vt_loc[:, None, None, None, None]
        h_l0 = g_m0 + zt5 * j0_l * f_cur.phi[None, None, ...] + jb_l * f_cur.bpar[None, None, ...]
        h_l1 = g_m1 - zt5 * vt5 * j0_l * f_cur.apar[None, None, ...]
        h_l0_old = (
            g_old_m0 + zt5 * j0_l * f_old.phi[None, None, ...] + jb_l * f_old.bpar[None, None, ...]
        )
        h_l1_old = g_old_m1 - zt5 * vt5 * j0_l * f_old.apar[None, None, ...]

        n_bar = jnp.sum(j0_l * h_l0, axis=1)
        u_bar = jnp.sum(j0_l * h_l1, axis=1)
        ub_bar = jnp.sum(jb_l * h_l0, axis=1)
        n_bar_old = jnp.sum(j0_l * h_l0_old, axis=1)
        u_bar_old = jnp.sum(j0_l * h_l1_old, axis=1)
        ub_bar_old = jnp.sum(jb_l * h_l0_old, axis=1)

        dn_bardt = (n_bar - n_bar_old) / dt_safe
        du_bardt = (u_bar - u_bar_old) / dt_safe
        dub_bardt = (ub_bar - ub_bar_old) / dt_safe

        vol_jac = geo.vol_fac.astype(real_dtype)[None, :, None, None]
        prefac = 0.5 * fac[None, None, :, :] * vol_jac * nz_s_loc[:, None, None, None]
        th_es = prefac * jnp.real(
            jnp.conj(dphidt)[None, ...] * n_bar_old - jnp.conj(f_old.phi)[None, ...] * dn_bardt
        )
        th_apar = prefac * jnp.real(
            -vt4
            * (jnp.conj(dadt)[None, ...] * u_bar_old - jnp.conj(f_old.apar)[None, ...] * du_bardt)
        )
        th_bpar = prefac * jnp.real(
            tz4
            * (jnp.conj(dbdt)[None, ...] * ub_bar_old - jnp.conj(f_old.bpar)[None, ...] * dub_bardt)
        )
        th_stack_loc = jnp.stack([th_es + th_apar + th_bpar, th_es, th_apar, th_bpar], axis=1)

        if n_sp > 1:
            wg_szyx = jax.lax.all_gather(wg_szyx_loc, sp_axis, axis=0, tiled=True)
            wg_lmst = jax.lax.all_gather(wg_lm_sp, sp_axis, axis=0, tiled=True)
            q_stack = jax.lax.all_gather(q_stack_loc, sp_axis, axis=0, tiled=True)
            p_stack = jax.lax.all_gather(p_stack_loc, sp_axis, axis=0, tiled=True)
            th_stack = jax.lax.all_gather(th_stack_loc, sp_axis, axis=0, tiled=True)
        else:
            wg_szyx = wg_szyx_loc
            wg_lmst = wg_lm_sp
            q_stack = q_stack_loc
            p_stack = p_stack_loc
            th_stack = th_stack_loc

        return wg_szyx, wg_lmst, q_stack, p_stack, th_stack

    wg_szyx, wg_lmst, q_stack, p_stack, th_stack = _shard_map(
        _moment_and_flux_reductions,
        mesh=mesh,
        in_specs=(g_spec, g_spec, f_spec, f_spec, rep_spec),
        out_specs=(rep_spec, rep_spec, rep_spec, rep_spec, rep_spec),
        check_rep=False,
    )(G_sharded, diag_state.G_old, fields, diag_state.fields_old, dt_arr)

    wg_sp = _reduce_species_summand_4d(wg_szyx)
    mst = jnp.sum(wg_lmst, axis=2)
    lst = jnp.sum(wg_lmst, axis=1)
    lmt = jnp.sum(wg_lmst, axis=0)
    wg = WgSpectra(
        total=wg_sp.total,
        st=wg_sp.st,
        kxst=wg_sp.kxst,
        kyst=wg_sp.kyst,
        kxkyst=wg_sp.kxkyst,
        zst=wg_sp.zst,
        lmst=wg_lmst,
        mst=mst,
        lst=lst,
        kxt=wg_sp.kxt,
        kyt=wg_sp.kyt,
        kxkyt=wg_sp.kxkyt,
        zt=wg_sp.zt,
        mt=jnp.sum(mst, axis=0),
        lt=jnp.sum(lst, axis=0),
        lmt=lmt,
    )

    def _unpack_bundle(stack_5d: jax.Array) -> FluxBundle:
        return FluxBundle(
            total=_reduce_species_summand_4d(stack_5d[:, 0]),
            es=_reduce_species_summand_4d(stack_5d[:, 1]),
            apar=_reduce_species_summand_4d(stack_5d[:, 2]),
            bpar=_reduce_species_summand_4d(stack_5d[:, 3]),
        )

    snapshot = DiagnosticsSnapshot(
        time=t_arr,
        dt=dt_arr,
        phi2=phi2,
        phi2_zonal=phi2_zonal,
        apar2=apar2,
        bpar2=bpar2,
        wg=wg,
        wphi=wphi,
        wapar=wapar,
        wbpar=wbpar,
        heat_flux=_unpack_bundle(q_stack),
        particle_flux=_unpack_bundle(p_stack),
        turbulent_heating=_unpack_bundle(th_stack),
        omega_kxkyt=omega_kxkyt,
    )
    next_diag_state = DiagnosticsState(
        G_old=G_sharded,
        fields_old=fields,
        time_old=t_arr,
        has_old=jnp.asarray(True, dtype=jnp.bool_),
    )
    return snapshot, next_diag_state


# ---------------------------------------------------------------------------
# 7. High-Level Sharded Simulation Driver & Step-Throughput Benchmark
# ---------------------------------------------------------------------------


def _constrain_sharded_state(st: IntegratorState, mesh: Mesh) -> IntegratorState:
    """Attach explicit ``("sp", "m")`` and replicated sharding constraints to state."""
    g_sharding = make_g_sharding(mesh)
    rep_sharding = make_replicated_sharding(mesh)
    return IntegratorState(
        G=jax.lax.with_sharding_constraint(st.G, g_sharding),
        fields=jax.tree_util.tree_map(
            lambda x: jax.lax.with_sharding_constraint(x, rep_sharding), st.fields
        ),
        time=jax.lax.with_sharding_constraint(st.time, rep_sharding),
        dt=jax.lax.with_sharding_constraint(st.dt, rep_sharding),
        step=jax.lax.with_sharding_constraint(st.step, rep_sharding),
        exb_state=(
            jax.tree_util.tree_map(
                lambda x: jax.lax.with_sharding_constraint(x, rep_sharding), st.exb_state
            )
            if st.exb_state is not None
            else None
        ),
        key=jax.lax.with_sharding_constraint(st.key, rep_sharding),
    )


def _constrain_sharded_diag(
    diag_state: DiagnosticsState | None, mesh: Mesh
) -> DiagnosticsState | None:
    """Attach explicit sharding constraints to an optional :class:`DiagnosticsState`."""
    if diag_state is None:
        return None
    g_sharding = make_g_sharding(mesh)
    rep_sharding = make_replicated_sharding(mesh)
    return DiagnosticsState(
        G_old=jax.lax.with_sharding_constraint(diag_state.G_old, g_sharding),
        fields_old=jax.tree_util.tree_map(
            lambda x: jax.lax.with_sharding_constraint(x, rep_sharding), diag_state.fields_old
        ),
        time_old=jax.lax.with_sharding_constraint(diag_state.time_old, rep_sharding),
        has_old=jax.lax.with_sharding_constraint(diag_state.has_old, rep_sharding),
    )


def _constrain_sharded_chunk(res: ChunkResult, mesh: Mesh) -> ChunkResult:
    """Attach explicit sharding constraints to all leaves of a sharded :class:`ChunkResult`."""
    g_sharding = make_g_sharding(mesh)
    rep_sharding = make_replicated_sharding(mesh)
    return ChunkResult(
        state=_constrain_sharded_state(res.state, mesh),
        G_prev=jax.lax.with_sharding_constraint(res.G_prev, g_sharding),
        fields_prev=jax.tree_util.tree_map(
            lambda x: jax.lax.with_sharding_constraint(x, rep_sharding), res.fields_prev
        ),
        time_prev=jax.lax.with_sharding_constraint(res.time_prev, rep_sharding),
        steps_done=jax.lax.with_sharding_constraint(res.steps_done, rep_sharding),
        diag_state=_constrain_sharded_diag(res.diag_state, mesh),
    )


def _place_diag_state_on_mesh(diag_state: DiagnosticsState, mesh: Mesh) -> DiagnosticsState:
    """Place :class:`DiagnosticsState` leaves onto ``mesh`` (6D sharded, scalars replicated)."""
    g_sharding = make_g_sharding(mesh)
    rep_sharding = make_replicated_sharding(mesh)
    return DiagnosticsState(
        G_old=device_put_global(diag_state.G_old, g_sharding),
        fields_old=jax.tree_util.tree_map(
            lambda x: device_put_global(x, rep_sharding), diag_state.fields_old
        ),
        time_old=device_put_global(diag_state.time_old, rep_sharding),
        has_old=device_put_global(diag_state.has_old, rep_sharding),
    )


def advance_sharded_integrator_chunk(
    state: IntegratorState,
    gk_op: GyrokineticOperator,
    mesh: Mesh,
    max_steps: int | jax.Array,
    *,
    dt_override: float | jax.Array | None = None,
    scheme: str | None = None,
    t_max: float | jax.Array | None = None,
    use_scan: bool = False,
    enforce_reality_cond: bool = False,
    diag_state: DiagnosticsState | None = None,
    diag_nwrite: int | jax.Array | None = None,
) -> ChunkResult:
    """Advance sharded ``state`` on ``mesh`` by up to ``max_steps`` without host-device syncs."""
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    t_max_eff = gk_op.t_max if t_max is None else t_max
    rep_sharding = make_replicated_sharding(mesh)
    t_dtype = _time_dtype(jnp.real(state.G).dtype)
    t_limit = jax.lax.with_sharding_constraint(
        _resolve_t_limit(t_max_eff, dtype=t_dtype), rep_sharding
    )
    diag_nw = (
        None
        if diag_nwrite is None
        else jax.lax.with_sharding_constraint(
            jnp.asarray(max(1, int(diag_nwrite)), dtype=jnp.int32), rep_sharding
        )
    )
    st_in = _constrain_sharded_state(state, mesh)
    d_in = _constrain_sharded_diag(diag_state, mesh)
    dt_arg = (
        None
        if dt_override is None
        else jax.lax.with_sharding_constraint(jnp.asarray(dt_override, dtype=t_dtype), rep_sharding)
    )

    def _step_fn(st: IntegratorState) -> IntegratorState:
        return step_sharded_integrator_state(
            st,
            gk_op,
            mesh,
            dt_override=dt_arg,
            scheme=sk,
            enforce_reality_cond=enforce_reality_cond,
        )

    if use_scan:
        if not _is_truly_fixed_dt(gk_op, sk, dt_override):
            raise ValueError(
                f"use_scan=True requires a fixed-dt scheme honoring fixed_dt or dt_override; "
                f"got scheme={sk!r}, fixed_dt={gk_op.fixed_dt}, dt_override={dt_override}."
            )
        if isinstance(max_steps, jax.Array) or int(max_steps) < 1:
            raise ValueError(
                f"use_scan=True requires concrete positive int max_steps, got {max_steps!r}."
            )
        fixed_len = int(max_steps)
        if not isinstance(t_max_eff, jax.Array) and not math.isfinite(float(t_max_eff)):
            res = _run_chunk_scan(
                st_in,
                fixed_len,
                t_limit,
                step_fn=_step_fn,
                diag_state=d_in,
                diag_nwrite=diag_nw,
            )
            return _constrain_sharded_chunk(res, mesh)
        dt_eff = dt_arg if dt_arg is not None else st_in.dt.astype(t_dtype)
        slack_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-12
        slack = jnp.asarray(slack_rel * float(fixed_len), dtype=t_dtype) * jnp.maximum(
            jnp.abs(t_limit), jnp.asarray(1.0, dtype=t_dtype)
        )
        t_last_step_start = (
            st_in.time.astype(t_dtype) + jnp.asarray(fixed_len - 1, dtype=t_dtype) * dt_eff
        )
        all_fit = jax.lax.with_sharding_constraint(
            t_last_step_start < (t_limit - slack), rep_sharding
        )
        res = jax.lax.cond(
            all_fit,
            lambda: _run_chunk_scan(
                st_in,
                fixed_len,
                t_limit,
                step_fn=_step_fn,
                diag_state=d_in,
                diag_nwrite=diag_nw,
            ),
            lambda: _run_chunk_while(
                st_in,
                jax.lax.with_sharding_constraint(
                    jnp.asarray(fixed_len, dtype=jnp.int32), rep_sharding
                ),
                t_limit,
                step_fn=_step_fn,
                diag_state=d_in,
                diag_nwrite=diag_nw,
            ),
        )
        return _constrain_sharded_chunk(res, mesh)

    max_steps_rep = jax.lax.with_sharding_constraint(
        jnp.asarray(max_steps, dtype=jnp.int32), rep_sharding
    )
    res = _run_chunk_while(
        st_in,
        max_steps_rep,
        t_limit,
        step_fn=_step_fn,
        diag_state=d_in,
        diag_nwrite=diag_nw,
    )
    return _constrain_sharded_chunk(res, mesh)


def make_sharded_chunk_stepper(
    gk_op: GyrokineticOperator,
    mesh: Mesh,
    *,
    dt_override: float | jax.Array | None = None,
    scheme: str | None = None,
    t_max: float | jax.Array | None = None,
    fixed_chunk_len: int | None = None,
    enforce_reality_cond: bool = False,
    diag_nwrite: int | None = None,
) -> Callable[..., ChunkResult]:
    """Compile a reusable single-trace sharded on-device chunk stepper on ``mesh``."""
    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    default_t_max = gk_op.t_max if t_max is None else t_max
    is_fixed_dt = _is_truly_fixed_dt(gk_op, sk, dt_override)
    diag_nw_val = None if diag_nwrite is None else max(1, int(diag_nwrite))
    steady_len = (
        int(fixed_chunk_len) if (fixed_chunk_len is not None and fixed_chunk_len >= 1) else None
    )
    rep_sharding = make_replicated_sharding(mesh)
    op_t_dtype = _time_dtype(gk_op.geo.bmag.dtype)
    trace_counts: dict[str, int] = {"while": 0, "scan": 0}

    def _diag_nw_arr() -> jax.Array | None:
        if diag_nw_val is None:
            return None
        return jax.lax.with_sharding_constraint(
            jnp.asarray(diag_nw_val, dtype=jnp.int32), rep_sharding
        )

    def _step_with_dt(st: IntegratorState, dt_in: jax.Array | None) -> IntegratorState:
        return step_sharded_integrator_state(
            st,
            gk_op,
            mesh,
            dt_override=dt_in,
            scheme=sk,
            enforce_reality_cond=enforce_reality_cond,
        )

    @jax.jit
    def _while_jit(
        st: IntegratorState,
        max_steps_arr: jax.Array,
        t_max_arr: jax.Array,
        dt_in: jax.Array | None,
        d_st: DiagnosticsState | None,
    ) -> ChunkResult:
        trace_counts["while"] += 1
        t_dtype = _time_dtype(jnp.real(st.G).dtype)
        st_c = _constrain_sharded_state(st, mesh)
        d_c = _constrain_sharded_diag(d_st, mesh)
        ms_c = jax.lax.with_sharding_constraint(
            jnp.asarray(max_steps_arr, dtype=jnp.int32), rep_sharding
        )
        t_lim = jax.lax.with_sharding_constraint(
            _resolve_t_limit(t_max_arr, dtype=t_dtype), rep_sharding
        )
        dt_c = (
            None
            if dt_in is None
            else jax.lax.with_sharding_constraint(dt_in.astype(t_dtype), rep_sharding)
        )
        res = _run_chunk_while(
            st_c,
            ms_c,
            t_lim,
            step_fn=lambda s: _step_with_dt(s, dt_c),
            diag_state=d_c,
            diag_nwrite=_diag_nw_arr(),
        )
        return _constrain_sharded_chunk(res, mesh)

    _scan_pure_jit: Callable[..., ChunkResult] | None = None
    _scan_cond_jit: Callable[..., ChunkResult] | None = None
    if is_fixed_dt and steady_len is not None:

        @jax.jit
        def _scan_pure_jit_fn(
            st: IntegratorState,
            t_max_arr: jax.Array,
            dt_in: jax.Array | None,
            d_st: DiagnosticsState | None,
        ) -> ChunkResult:
            trace_counts["scan"] += 1
            t_dtype = _time_dtype(jnp.real(st.G).dtype)
            st_c = _constrain_sharded_state(st, mesh)
            d_c = _constrain_sharded_diag(d_st, mesh)
            t_lim = jax.lax.with_sharding_constraint(
                _resolve_t_limit(t_max_arr, dtype=t_dtype), rep_sharding
            )
            dt_c = (
                None
                if dt_in is None
                else jax.lax.with_sharding_constraint(dt_in.astype(t_dtype), rep_sharding)
            )
            res = _run_chunk_scan(
                st_c,
                steady_len,
                t_lim,
                step_fn=lambda s: _step_with_dt(s, dt_c),
                diag_state=d_c,
                diag_nwrite=_diag_nw_arr(),
            )
            return _constrain_sharded_chunk(res, mesh)

        @jax.jit
        def _scan_cond_jit_fn(
            st: IntegratorState,
            t_max_arr: jax.Array,
            dt_in: jax.Array | None,
            d_st: DiagnosticsState | None,
        ) -> ChunkResult:
            trace_counts["scan"] += 1
            t_dtype = _time_dtype(jnp.real(st.G).dtype)
            st_c = _constrain_sharded_state(st, mesh)
            d_c = _constrain_sharded_diag(d_st, mesh)
            t_lim = jax.lax.with_sharding_constraint(
                _resolve_t_limit(t_max_arr, dtype=t_dtype), rep_sharding
            )
            dt_c = (
                None
                if dt_in is None
                else jax.lax.with_sharding_constraint(dt_in.astype(t_dtype), rep_sharding)
            )
            dt_eff = dt_c if dt_c is not None else st_c.dt.astype(t_dtype)
            slack_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-12
            slack = jnp.asarray(slack_rel * float(steady_len), dtype=t_dtype) * jnp.maximum(
                jnp.abs(t_lim), jnp.asarray(1.0, dtype=t_dtype)
            )
            t_last_step_start = (
                st_c.time.astype(t_dtype) + jnp.asarray(steady_len - 1, dtype=t_dtype) * dt_eff
            )
            all_fit = jax.lax.with_sharding_constraint(
                t_last_step_start < (t_lim - slack), rep_sharding
            )
            res = jax.lax.cond(
                all_fit,
                lambda: _run_chunk_scan(
                    st_c,
                    steady_len,
                    t_lim,
                    step_fn=lambda s: _step_with_dt(s, dt_c),
                    diag_state=d_c,
                    diag_nwrite=_diag_nw_arr(),
                ),
                lambda: _run_chunk_while(
                    st_c,
                    jax.lax.with_sharding_constraint(
                        jnp.asarray(steady_len, dtype=jnp.int32), rep_sharding
                    ),
                    t_lim,
                    step_fn=lambda s: _step_with_dt(s, dt_c),
                    diag_state=d_c,
                    diag_nwrite=_diag_nw_arr(),
                ),
            )
            return _constrain_sharded_chunk(res, mesh)

        _scan_pure_jit = _scan_pure_jit_fn
        _scan_cond_jit = _scan_cond_jit_fn

    def _stepper(
        state: IntegratorState,
        max_steps: int | jax.Array,
        *,
        t_max: float | jax.Array | None = None,
        diag_state: DiagnosticsState | None = None,
        safe_for_scan: bool | None = None,
    ) -> ChunkResult:
        t_max_use = default_t_max if t_max is None else t_max
        t_dtype = _time_dtype(jnp.real(state.G).dtype) if state is not None else op_t_dtype
        t_max_arr = device_put_global(jnp.asarray(t_max_use, dtype=t_dtype), rep_sharding)
        dt_arr = (
            None
            if dt_override is None
            else device_put_global(jnp.asarray(dt_override, dtype=t_dtype), rep_sharding)
        )
        is_steady_len = (
            steady_len is not None
            and not isinstance(max_steps, jax.Array)
            and int(max_steps) == steady_len
        )
        if _scan_pure_jit is not None and is_steady_len:
            if safe_for_scan is True or (
                safe_for_scan is None
                and not isinstance(t_max_use, jax.Array)
                and not math.isfinite(float(t_max_use))
            ):
                return _scan_pure_jit(state, t_max_arr, dt_arr, diag_state)
            if safe_for_scan is None and _scan_cond_jit is not None:
                return _scan_cond_jit(state, t_max_arr, dt_arr, diag_state)
        max_steps_rep = device_put_global(jnp.asarray(max_steps, dtype=jnp.int32), rep_sharding)
        return _while_jit(state, max_steps_rep, t_max_arr, dt_arr, diag_state)

    _stepper.trace_counts = trace_counts  # type: ignore[attr-defined]
    _stepper.is_fixed_dt = is_fixed_dt  # type: ignore[attr-defined]
    _stepper.fixed_chunk_len = steady_len  # type: ignore[attr-defined]
    return _stepper


def run_sharded_simulation(
    G_init: jax.Array,
    gk_op: GyrokineticOperator,
    mesh: Mesh,
    *,
    nsteps: int,
    dt: float | jax.Array | None = None,
    scheme: str | None = None,
    nwrite: int = 0,
    enforce_reality_cond: bool = False,
    exb_state: ExBShearState | None = None,
    key: jax.Array | None = None,
    use_scan: bool = False,
) -> tuple[IntegratorState, list[DiagnosticsSnapshot]]:
    """Run ``nsteps`` sharded timesteps on ``mesh`` with compiled fused chunk stepping.

    Defaults to the zero-sync compiled ``jax.lax.while_loop`` path (``use_scan=False``,
    ``trace_count == 1``) for bit-exact parity with ``step_sharded_integrator_state``.
    Pass ``use_scan=True`` (when ``fixed_dt`` or ``dt`` is fixed) to opt into the fixed-length
    ``jax.lax.scan`` fast path (FP-identical, up to ~1 ULP XLA loop scheduling variance).

    Preserves snapshot semantics verbatim: writes at 1-based steps
    ``counter % nwrite == 1``, ``counter == nsteps``, or every step when ``nwrite == 1``,
    using zeroed ``(G_old, fields_old)`` at ``counter == 1`` and the penultimate step's
    ``(G_prev, fields_prev)`` for ``counter > 1``. (Note: ``cli.py`` follows C++
    ``Diagnostics_GK::loop`` directly by saving ``(G(0), fields(0))`` before step 1 and
    only writing terminal steps on ``(c - 1) % nwrite == 0`` or ``time > t_max``.)
    """
    if nsteps < 1:
        raise ValueError(f"nsteps={nsteps} must be >= 1.")

    state = init_sharded_integrator_state(
        G_init,
        gk_op,
        mesh,
        dt=dt,
        enforce_reality_cond=enforce_reality_cond,
        exb_state=exb_state,
        key=key,
    )
    diag_state = _place_diag_state_on_mesh(
        init_diagnostics_state(state.G, state.fields, time=0.0, has_old=False), mesh
    )
    snapshots: list[DiagnosticsSnapshot] = []

    sk = gk_op.scheme if scheme is None else scheme.strip().lower()
    is_fixed_dt = _is_truly_fixed_dt(gk_op, sk, dt)
    if use_scan and is_fixed_dt:
        steady_len: int | None = nsteps if nwrite <= 0 else nwrite
    else:
        steady_len = None

    stepper = make_sharded_chunk_stepper(
        gk_op,
        mesh,
        dt_override=dt,
        scheme=sk,
        t_max=math.inf,
        fixed_chunk_len=steady_len,
        enforce_reality_cond=enforce_reality_cond,
    )

    if nwrite <= 0:
        chunk = stepper(state, nsteps, t_max=math.inf, safe_for_scan=(use_scan and is_fixed_dt))
        return chunk.state, snapshots

    rep_sharding = make_replicated_sharding(mesh)
    true_rep = device_put_global(jnp.asarray(True, dtype=jnp.bool_), rep_sharding)

    write_steps = [c for c in range(1, nsteps + 1) if c % nwrite == 1 or c == nsteps or nwrite == 1]
    cur_step = 0
    for target_step in write_steps:
        steps_to_run = target_step - cur_step
        chunk = stepper(state, steps_to_run, t_max=math.inf)
        state = chunk.state
        cur_step = target_step
        if target_step > 1:
            diag_state = DiagnosticsState(
                G_old=chunk.G_prev,
                fields_old=chunk.fields_prev,
                time_old=chunk.time_prev,
                has_old=true_rep,
            )
        # Call sharded_compute_diagnostics directly (already shard_map-jitted inside)
        # without an outer jax.jit wrapper so TurbulentHeating* cancellation matches
        # step-by-step run_sharded_simulation bit-for-bit.
        snap, diag_state = sharded_compute_diagnostics(
            state.G,
            state.fields,
            diag_state,
            gk_op.grids,
            gk_op.geo,
            gk_op.linear_op.solver,
            mesh,
            time=state.time,
            dt=state.dt,
            beta=gk_op.linear_op.solver.beta,
        )
        snapshots.append(snap)

    return state, snapshots


def benchmark_sharded_step(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    mesh: Mesh,
    *,
    warmup_steps: int = 2,
    timed_steps: int = 5,
    dtype: jnp.dtype = jnp.complex64,
    scheme: str | None = None,
    gk_op: GyrokineticOperator | None = None,
    g_init: jax.Array | None = None,
) -> dict[str, Any]:
    """Benchmark compiled sharded step throughput on ``mesh`` (TPU / CPU)."""
    configure_tpu_precision()
    c_dtype = jnp.dtype(dtype)
    r_dtype = jnp.float64 if c_dtype == jnp.complex128 else jnp.float32
    if gk_op is None:
        with host_precompute_context():
            gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=r_dtype)
        gk_op = move_tree_to_default_device(gk_op)
    g0 = jnp.zeros(grids.g_shape, dtype=c_dtype) if g_init is None else g_init.astype(c_dtype)
    dt_bench = cfg.dt if cfg.fixed_dt else None
    state = init_sharded_integrator_state(g0, gk_op, mesh, dt=dt_bench)

    step_fn = jax.jit(
        lambda st: step_sharded_integrator_state(
            st, gk_op, mesh, dt_override=dt_bench, scheme=scheme
        )
    )
    for _ in range(max(1, warmup_steps)):
        state = step_fn(state)
    jax.block_until_ready(state.G)

    t0 = time.perf_counter()
    for _ in range(max(1, timed_steps)):
        state = step_fn(state)
    jax.block_until_ready(state.G)
    elapsed = max(1.0e-9, time.perf_counter() - t0)

    n_steps = max(1, timed_steps)
    sec_per_step = elapsed / float(n_steps)
    dof = int(np.prod(grids.g_shape))
    return {
        "mesh_shape": tuple(int(v) for v in mesh.devices.shape),
        "num_devices": int(mesh.devices.size),
        "g_shape": grids.g_shape,
        "dtype": str(c_dtype),
        "scheme": gk_op.scheme if scheme is None else scheme,
        "timed_steps": n_steps,
        "elapsed_sec": elapsed,
        "ms_per_step": 1.0e3 * sec_per_step,
        "steps_per_sec": 1.0 / sec_per_step,
        "mdof_per_sec": (float(dof) / sec_per_step) * 1.0e-6,
    }


__all__ = [
    "ChunkResult",
    "advance_sharded_integrator_chunk",
    "benchmark_sharded_step",
    "configure_tpu_precision",
    "create_device_mesh",
    "exchange_hermite_halos",
    "g_partition_spec",
    "gather_g_state",
    "host_precompute_context",
    "infer_mesh_shape",
    "init_sharded_integrator_state",
    "make_g_sharding",
    "make_replicated_sharding",
    "make_sharded_chunk_stepper",
    "move_tree_to_default_device",
    "pad_hermite_with_halos",
    "replicated_partition_spec",
    "run_sharded_simulation",
    "shard_g_state",
    "sharded_compute_diagnostics",
    "sharded_gyrokinetic_rhs",
    "sharded_linear_rhs",
    "sharded_nonlinear_rhs",
    "sharded_solve_fields",
    "step_sharded_gyrokinetic",
    "step_sharded_integrator_state",
]
