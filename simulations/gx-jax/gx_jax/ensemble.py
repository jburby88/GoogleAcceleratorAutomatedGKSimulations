"""Batched `jax.vmap` KR2 ensemble parameter-sweep driver (`gx-jax-ensemble`).

Stacks `B >= 1` structurally compatible GX input decks (`.in` TOML files, `GXConfig`
instances, or pre-initialized `(GXConfig, Grids, Geometry)` specs) along leading batch
axis `0` and advances the entire ensemble inside a single compiled on-device
`jax.lax.while_loop` + `jax.vmap` execution (optionally partitioned across a 1D
`Mesh(devices, ("batch",))` multi-device TPU/CPU mesh).

Key architectural guarantees:
1. All member-varying operator arrays AND scalar physics/time parameters (`beta_b`,
   `gradpar_b`, `fphi_b`, `fapar_b`, `fbpar_b`, `tau_fac_b`, `omega_linear_b`, `dt_b`,
   `dt_min_b`, `dt_max_b`, `t_max_b`, `cfl_b`, `nstep_b`, `nwrite_b`, `nsave_b`,
   `fixed_dt_b`, etc.) are stored as stacked JAX array leaves of leading dimension `B`
   inside `BatchedEnsembleOperators`, while only true structural compile-time constants
   live in PyTree `aux_data`.
2. Mixed Electrostatic (`beta == 0`) + Electromagnetic (`beta > 0`, e.g. KBM) ensembles
   and heterogeneous Tokamak (`s-alpha` / `miller`) + 3D Stellarator (`W7-X` `.eik.nc` /
   VMEC) geometries execute cleanly in a single `jax.vmap` trace with zero `0/0` NaNs and
   exact union-bucket `LinkedParallelGrad` / `NTFTParallelGrad` chain unification when
   `shat` / `jtwist` vary across members.
3. Per-member `active_b = (step_b < nstep_b) & (time_b < t_max_b - eps_t_b)` masking
   freezes completed members (`G`, `fields`, `time`, `dt`, `step`, `exb_state`, and
   penultimate `DiagnosticsState`) via `jnp.where` inside on-device chunks while
   capturing mid-chunk `t_max_b` terminal diagnostic snapshots so each member's
   `.out.nc`, `.restart.nc`, and strict-JSON KR2 `.diagnostics_metrics.json` match
   standalone single-deck execution.
4. Multi-host `jax.distributed` runs (`--distributed`; `jax.process_count() > 1`) place
   the `'batch'` mesh across every host: all processes run the `vmap` chunk stepper,
   diagnostics and `process_allgather` host gathers in strict lockstep, while only
   process 0 opens or writes `.out.nc`, `.restart.nc` and JSON files.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from gx_jax.boundary_parallel import (
    LinkedBucket,
    LinkedParallelGrad,
    NTFTBucket,
    NTFTParallelGrad,
    _hyperz_weight_np,
    _kz_linked_np,
    _kz_linked_ntft_np,
    _kzm_linked_np,
    init_linked_parallel_grad,
    init_ntft_parallel_grad,
)
from gx_jax.config import BoltzmannOption, GXConfig, load_config
from gx_jax.diagnostics_nc import (
    DiagnosticsSnapshot,
    DiagnosticsState,
    DiagnosticsWriter,
    compute_field_squared_spectra,
    compute_heat_flux_spectra,
    compute_omega_kxky,
    compute_particle_flux_spectra,
    compute_phi2_zonal_spectra,
    compute_turbulent_heating_spectra,
    compute_wapar_spectra,
    compute_wbpar_spectra,
    compute_wg_spectra,
    compute_wphi_spectra,
    extract_kr2_diagnostics,
    init_diagnostics_state,
    sanitize_json_payload,
)
from gx_jax.distributed import (
    abort_distributed_run,
    add_distributed_cli_args,
    assert_same_across_processes,
    device_put_global,
    gather_global_array,
    gather_global_tree,
    init_distributed_runtime_from_args,
    is_distributed,
    is_primary_process,
    shutdown_distributed_runtime,
    sync_processes,
    validate_multihost_mesh,
)
from gx_jax.fields import (
    Fields,
    FieldSolver,
    _complex_and_real_dtypes,
    _host_cast,
    _safe_divide,
    _time_dtype,
    enforce_reality,
    init_moments,
    load_restart_nc,
    mask_g,
    real_space_density,
    real_space_par_current,
    real_space_perp_current,
    save_restart_nc,
)
from gx_jax.geometry_nc import Geometry, init_geometry
from gx_jax.grids import Grids
from gx_jax.integrator import (
    IntegratorState,
    StepResult,
    _scheme_honors_fixed_dt,
    get_scheme_cfl_fac,
    init_gyrokinetic_operator,
)
from gx_jax.linear_rhs import LinearOperator, ParallelGrad, linear_rhs
from gx_jax.nonlinear_exb import (
    ExBShearState,
    ForcingOperator,
    NonlinearOperator,
    _c_round_jax,
    _safe_sqrt,
    init_exb_shear,
    init_forcing,
    nonlinear_rhs,
)
from gx_jax.shard_tpu import (
    configure_tpu_precision,
    host_precompute_context,
    move_tree_to_default_device,
)


@dataclass(frozen=True)
class EnsembleMemberSpec:
    """Resolved host-side configuration, grids, and geometry for one ensemble member."""

    cfg: GXConfig
    grids: Grids
    geo: Geometry
    input_path: Path | None = None
    name: str = "gx"


EnsembleInputItem = str | Path | GXConfig | tuple[GXConfig, Grids, Geometry] | EnsembleMemberSpec


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BatchedEnsembleOperators:
    """Vectorized `B`-member gyrokinetic operator PyTree for `jax.vmap` & `'batch'` sharding.

    Every array leaf has leading dimension `B` (axis `0`), while `aux_data` holds strictly
    member-invariant structural constants so `jax.vmap` and `NamedSharding` see a single
    uniform PyTree structure across arbitrary physics & 1D geometry sweeps.
    """

    # Normalized nested operator PyTrees with leading batch axis B on all array leaves
    grids: Grids
    geo: Geometry
    solver: FieldSolver
    linear_op: LinearOperator
    nonlinear_op: NonlinearOperator | None
    forcing_op: ForcingOperator | None
    exb_init: ExBShearState | None

    # Member-varying physics scalars in runtime real dtype, shape (B,) or (B, 3)
    beta_b: jax.Array
    gradpar_b: jax.Array
    vtmax_b: jax.Array
    tzmax_b: jax.Array
    fphi_b: jax.Array
    fapar_b: jax.Array
    fbpar_b: jax.Array
    tau_fac_b: jax.Array
    phi_ext_b: jax.Array
    kxfac_b: jax.Array
    shat_b: jax.Array
    x0_b: jax.Array
    y0_b: jax.Array
    kx_max_b: jax.Array
    ky_max_b: jax.Array
    kz_max_b: jax.Array
    vpar_max_b: jax.Array
    mub_maxj_b: jax.Array
    kpar_local_b: jax.Array
    damp_ends_amp_b: jax.Array
    nu_hyper_z_b: jax.Array
    w_osc_b: jax.Array
    g_exb_b: jax.Array
    forcing_amp_b: jax.Array
    omega_linear_b: jax.Array  # (B, 3)

    # Member-varying time/control arrays in float64 / int32 / bool, shape (B,)
    dt_b: jax.Array  # (B,) float64
    dt_min_b: jax.Array  # (B,) float64
    dt_max_b: jax.Array  # (B,) float64
    t_max_b: jax.Array  # (B,) float64
    eps_t_b: jax.Array  # (B,) float64
    cfl_b: jax.Array  # (B,) float64
    nstep_b: jax.Array  # (B,) int32
    nwrite_b: jax.Array  # (B,) int32
    nsave_b: jax.Array  # (B,) int32
    random_seed_b: jax.Array  # (B,) int32
    forcing_target_zyx_b: jax.Array  # (B, 3) int32
    fixed_dt_b: jax.Array  # (B,) bool

    # True structural constants (PyTree aux_data)
    batch_size: int
    nspecies: int
    nhermite: int
    nlaguerre: int
    nj: int
    nz: int
    naky: int
    nakx: int
    nx: int
    ny: int
    nyc: int
    scheme: str
    cfl_fac: float
    closure_model: str
    smith_par_q: int
    smith_perp_q: int
    all_kinetic: bool
    boltzmann_opt: BoltzmannOption
    nonlinear_mode: bool
    non_twist: bool
    local_limit: bool
    boundary_option_periodic: bool
    dealias_kz: bool
    long_wavelength_gk: bool
    no_fields: bool
    has_any_fphi: bool
    has_any_apar: bool
    has_any_bpar: bool
    has_any_phi_ext: bool
    has_any_collisions: bool
    coll_conservation: bool
    ei_colls: bool
    has_any_hyper: bool
    has_any_hb_hyper: bool
    has_any_hyperz: bool
    has_any_hypercollisions_const: bool
    has_any_hypercollisions_kz: bool
    has_any_exb_shear: bool
    has_any_exb_shear_phase: bool
    has_any_forcing: bool
    forcing_type: str
    stir_field: str
    fft_backend: str

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = (
            self.grids,
            self.geo,
            self.solver,
            self.linear_op,
            self.nonlinear_op,
            self.forcing_op,
            self.exb_init,
            self.beta_b,
            self.gradpar_b,
            self.vtmax_b,
            self.tzmax_b,
            self.fphi_b,
            self.fapar_b,
            self.fbpar_b,
            self.tau_fac_b,
            self.phi_ext_b,
            self.kxfac_b,
            self.shat_b,
            self.x0_b,
            self.y0_b,
            self.kx_max_b,
            self.ky_max_b,
            self.kz_max_b,
            self.vpar_max_b,
            self.mub_maxj_b,
            self.kpar_local_b,
            self.damp_ends_amp_b,
            self.nu_hyper_z_b,
            self.w_osc_b,
            self.g_exb_b,
            self.forcing_amp_b,
            self.omega_linear_b,
            self.dt_b,
            self.dt_min_b,
            self.dt_max_b,
            self.t_max_b,
            self.eps_t_b,
            self.cfl_b,
            self.nstep_b,
            self.nwrite_b,
            self.nsave_b,
            self.random_seed_b,
            self.forcing_target_zyx_b,
            self.fixed_dt_b,
        )
        aux_data = (
            self.batch_size,
            self.nspecies,
            self.nhermite,
            self.nlaguerre,
            self.nj,
            self.nz,
            self.naky,
            self.nakx,
            self.nx,
            self.ny,
            self.nyc,
            self.scheme,
            self.cfl_fac,
            self.closure_model,
            self.smith_par_q,
            self.smith_perp_q,
            self.all_kinetic,
            int(self.boltzmann_opt),
            self.nonlinear_mode,
            self.non_twist,
            self.local_limit,
            self.boundary_option_periodic,
            self.dealias_kz,
            self.long_wavelength_gk,
            self.no_fields,
            self.has_any_fphi,
            self.has_any_apar,
            self.has_any_bpar,
            self.has_any_phi_ext,
            self.has_any_collisions,
            self.coll_conservation,
            self.ei_colls,
            self.has_any_hyper,
            self.has_any_hb_hyper,
            self.has_any_hyperz,
            self.has_any_hypercollisions_const,
            self.has_any_hypercollisions_kz,
            self.has_any_exb_shear,
            self.has_any_exb_shear_phase,
            self.has_any_forcing,
            self.forcing_type,
            self.stir_field,
            self.fft_backend,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]
    ) -> BatchedEnsembleOperators:
        (
            batch_size,
            nspecies,
            nhermite,
            nlaguerre,
            nj,
            nz,
            naky,
            nakx,
            nx,
            ny,
            nyc,
            scheme,
            cfl_fac,
            closure_model,
            smith_par_q,
            smith_perp_q,
            all_kinetic,
            boltzmann_opt_int,
            nonlinear_mode,
            non_twist,
            local_limit,
            boundary_option_periodic,
            dealias_kz,
            long_wavelength_gk,
            no_fields,
            has_any_fphi,
            has_any_apar,
            has_any_bpar,
            has_any_phi_ext,
            has_any_collisions,
            coll_conservation,
            ei_colls,
            has_any_hyper,
            has_any_hb_hyper,
            has_any_hyperz,
            has_any_hypercollisions_const,
            has_any_hypercollisions_kz,
            has_any_exb_shear,
            has_any_exb_shear_phase,
            has_any_forcing,
            forcing_type,
            stir_field,
            fft_backend,
        ) = aux_data
        (
            grids,
            geo,
            solver,
            linear_op,
            nonlinear_op,
            forcing_op,
            exb_init,
            beta_b,
            gradpar_b,
            vtmax_b,
            tzmax_b,
            fphi_b,
            fapar_b,
            fbpar_b,
            tau_fac_b,
            phi_ext_b,
            kxfac_b,
            shat_b,
            x0_b,
            y0_b,
            kx_max_b,
            ky_max_b,
            kz_max_b,
            vpar_max_b,
            mub_maxj_b,
            kpar_local_b,
            damp_ends_amp_b,
            nu_hyper_z_b,
            w_osc_b,
            g_exb_b,
            forcing_amp_b,
            omega_linear_b,
            dt_b,
            dt_min_b,
            dt_max_b,
            t_max_b,
            eps_t_b,
            cfl_b,
            nstep_b,
            nwrite_b,
            nsave_b,
            random_seed_b,
            forcing_target_zyx_b,
            fixed_dt_b,
        ) = children
        return cls(
            grids=grids,
            geo=geo,
            solver=solver,
            linear_op=linear_op,
            nonlinear_op=nonlinear_op,
            forcing_op=forcing_op,
            exb_init=exb_init,
            beta_b=beta_b,
            gradpar_b=gradpar_b,
            vtmax_b=vtmax_b,
            tzmax_b=tzmax_b,
            fphi_b=fphi_b,
            fapar_b=fapar_b,
            fbpar_b=fbpar_b,
            tau_fac_b=tau_fac_b,
            phi_ext_b=phi_ext_b,
            kxfac_b=kxfac_b,
            shat_b=shat_b,
            x0_b=x0_b,
            y0_b=y0_b,
            kx_max_b=kx_max_b,
            ky_max_b=ky_max_b,
            kz_max_b=kz_max_b,
            vpar_max_b=vpar_max_b,
            mub_maxj_b=mub_maxj_b,
            kpar_local_b=kpar_local_b,
            damp_ends_amp_b=damp_ends_amp_b,
            nu_hyper_z_b=nu_hyper_z_b,
            w_osc_b=w_osc_b,
            g_exb_b=g_exb_b,
            forcing_amp_b=forcing_amp_b,
            omega_linear_b=omega_linear_b,
            dt_b=dt_b,
            dt_min_b=dt_min_b,
            dt_max_b=dt_max_b,
            t_max_b=t_max_b,
            eps_t_b=eps_t_b,
            cfl_b=cfl_b,
            nstep_b=nstep_b,
            nwrite_b=nwrite_b,
            nsave_b=nsave_b,
            random_seed_b=random_seed_b,
            forcing_target_zyx_b=forcing_target_zyx_b,
            fixed_dt_b=fixed_dt_b,
            batch_size=batch_size,
            nspecies=nspecies,
            nhermite=nhermite,
            nlaguerre=nlaguerre,
            nj=nj,
            nz=nz,
            naky=naky,
            nakx=nakx,
            nx=nx,
            ny=ny,
            nyc=nyc,
            scheme=scheme,
            cfl_fac=cfl_fac,
            closure_model=closure_model,
            smith_par_q=smith_par_q,
            smith_perp_q=smith_perp_q,
            all_kinetic=all_kinetic,
            boltzmann_opt=BoltzmannOption(boltzmann_opt_int),
            nonlinear_mode=nonlinear_mode,
            non_twist=non_twist,
            local_limit=local_limit,
            boundary_option_periodic=boundary_option_periodic,
            dealias_kz=dealias_kz,
            long_wavelength_gk=long_wavelength_gk,
            no_fields=no_fields,
            has_any_fphi=has_any_fphi,
            has_any_apar=has_any_apar,
            has_any_bpar=has_any_bpar,
            has_any_phi_ext=has_any_phi_ext,
            has_any_collisions=has_any_collisions,
            coll_conservation=coll_conservation,
            ei_colls=ei_colls,
            has_any_hyper=has_any_hyper,
            has_any_hb_hyper=has_any_hb_hyper,
            has_any_hyperz=has_any_hyperz,
            has_any_hypercollisions_const=has_any_hypercollisions_const,
            has_any_hypercollisions_kz=has_any_hypercollisions_kz,
            has_any_exb_shear=has_any_exb_shear,
            has_any_exb_shear_phase=has_any_exb_shear_phase,
            has_any_forcing=has_any_forcing,
            forcing_type=forcing_type,
            stir_field=stir_field,
            fft_backend=fft_backend,
        )

    def astype(self, dtype: jnp.dtype) -> BatchedEnsembleOperators:
        """Cast runtime operator & physics scalar arrays and time bounds."""
        target = jnp.dtype(dtype)
        r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        t_target = _time_dtype(r_target)
        return replace(
            self,
            grids=self.grids.astype(r_target),
            geo=self.geo.astype(r_target),
            solver=self.solver.astype(r_target),
            linear_op=self.linear_op.astype(r_target),
            nonlinear_op=self.nonlinear_op.astype(r_target)
            if self.nonlinear_op is not None
            else None,
            forcing_op=self.forcing_op.astype(r_target) if self.forcing_op is not None else None,
            exb_init=self.exb_init.astype(r_target) if self.exb_init is not None else None,
            beta_b=_host_cast(self.beta_b, r_target),
            gradpar_b=_host_cast(self.gradpar_b, r_target),
            vtmax_b=_host_cast(self.vtmax_b, r_target),
            tzmax_b=_host_cast(self.tzmax_b, r_target),
            fphi_b=_host_cast(self.fphi_b, r_target),
            fapar_b=_host_cast(self.fapar_b, r_target),
            fbpar_b=_host_cast(self.fbpar_b, r_target),
            tau_fac_b=_host_cast(self.tau_fac_b, r_target),
            phi_ext_b=_host_cast(self.phi_ext_b, r_target),
            kxfac_b=_host_cast(self.kxfac_b, r_target),
            shat_b=_host_cast(self.shat_b, r_target),
            x0_b=_host_cast(self.x0_b, r_target),
            y0_b=_host_cast(self.y0_b, r_target),
            kx_max_b=_host_cast(self.kx_max_b, r_target),
            ky_max_b=_host_cast(self.ky_max_b, r_target),
            kz_max_b=_host_cast(self.kz_max_b, r_target),
            vpar_max_b=_host_cast(self.vpar_max_b, r_target),
            mub_maxj_b=_host_cast(self.mub_maxj_b, r_target),
            kpar_local_b=_host_cast(self.kpar_local_b, r_target),
            damp_ends_amp_b=_host_cast(self.damp_ends_amp_b, r_target),
            nu_hyper_z_b=_host_cast(self.nu_hyper_z_b, r_target),
            w_osc_b=_host_cast(self.w_osc_b, r_target),
            g_exb_b=_host_cast(self.g_exb_b, r_target),
            forcing_amp_b=_host_cast(self.forcing_amp_b, r_target),
            omega_linear_b=_host_cast(self.omega_linear_b, r_target),
            dt_b=_host_cast(self.dt_b, t_target),
            dt_min_b=_host_cast(self.dt_min_b, t_target),
            dt_max_b=_host_cast(self.dt_max_b, t_target),
            t_max_b=_host_cast(self.t_max_b, t_target),
            eps_t_b=_host_cast(self.eps_t_b, t_target),
            cfl_b=_host_cast(self.cfl_b, t_target),
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EnsembleChunkCarry:
    """Device loop carry for fused on-device `jax.lax.while_loop` ensemble chunk stepping."""

    state: IntegratorState
    diag_state: DiagnosticsState
    pending_write: jax.Array  # (B,) bool
    steps_in_chunk: jax.Array  # () int32

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (self.state, self.diag_state, self.pending_write, self.steps_in_chunk), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> EnsembleChunkCarry:
        del aux_data
        state, diag_state, pending_write, steps_in_chunk = children
        return cls(
            state=state,
            diag_state=diag_state,
            pending_write=pending_write,
            steps_in_chunk=steps_in_chunk,
        )


def _deck_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".in"):
        return name[:-3]
    if name.endswith(".toml"):
        return name[:-5]
    return path.stem


def _apply_cfg_overrides(
    cfg: GXConfig,
    *,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    fft_backend: str | None = None,
) -> GXConfig:
    if force_nonlinear and force_linear:
        raise ValueError("force_nonlinear and force_linear are mutually exclusive.")
    overrides: dict[str, Any] = {}
    if nstep is not None:
        overrides["nstep"] = int(nstep)
    if dt is not None:
        overrides["dt"] = float(dt)
    if t_max is not None:
        overrides["t_max"] = float(t_max)
    if nwrite is not None:
        overrides["nwrite"] = int(nwrite)
    if scheme is not None:
        overrides["scheme"] = str(scheme).strip().lower()
    if force_nonlinear:
        overrides["nonlinear_mode"] = True
    if force_linear:
        overrides["nonlinear_mode"] = False
    if fft_backend is not None:
        fb = str(fft_backend).strip().lower()
        if fb not in ("auto", "fft", "mxu"):
            raise ValueError(
                f"Invalid fft_backend={fft_backend!r}; expected 'auto', 'fft', or 'mxu'."
            )
        if hasattr(cfg, "fft_backend"):
            overrides["fft_backend"] = fb
    return replace(cfg, **overrides) if overrides else cfg


def _resolve_member_spec(
    item: EnsembleInputItem,
    *,
    idx: int,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    fft_backend: str | None = None,
) -> EnsembleMemberSpec:
    if isinstance(item, EnsembleMemberSpec):
        cfg_mod = _apply_cfg_overrides(
            item.cfg,
            nstep=nstep,
            dt=dt,
            t_max=t_max,
            nwrite=nwrite,
            scheme=scheme,
            force_nonlinear=force_nonlinear,
            force_linear=force_linear,
            fft_backend=fft_backend,
        )
        return EnsembleMemberSpec(
            cfg=cfg_mod,
            grids=item.grids,
            geo=item.geo,
            input_path=item.input_path,
            name=item.name or cfg_mod.run_name or f"member_{idx:03d}",
        )
    if isinstance(item, tuple) and len(item) == 3:
        cfg_in, grids_in, geo_in = item
        if not isinstance(cfg_in, GXConfig):
            raise TypeError(f"Expected GXConfig as first tuple element at index {idx}.")
        cfg_mod = _apply_cfg_overrides(
            cfg_in,
            nstep=nstep,
            dt=dt,
            t_max=t_max,
            nwrite=nwrite,
            scheme=scheme,
            force_nonlinear=force_nonlinear,
            force_linear=force_linear,
            fft_backend=fft_backend,
        )
        name = (
            _deck_stem(cfg_mod.input_path)
            if cfg_mod.input_path is not None
            else (cfg_mod.run_name or f"member_{idx:03d}")
        )
        return EnsembleMemberSpec(
            cfg=cfg_mod,
            grids=grids_in,
            geo=geo_in,
            input_path=cfg_mod.input_path,
            name=name,
        )
    if isinstance(item, GXConfig):
        cfg_pre = _apply_cfg_overrides(
            item,
            nstep=nstep,
            dt=dt,
            t_max=t_max,
            nwrite=nwrite,
            scheme=scheme,
            force_nonlinear=force_nonlinear,
            force_linear=force_linear,
            fft_backend=fft_backend,
        )
        cfg_fin, grids_fin, geo_fin = init_geometry(cfg_pre)
        name = (
            _deck_stem(cfg_fin.input_path)
            if cfg_fin.input_path is not None
            else (cfg_fin.run_name or f"member_{idx:03d}")
        )
        return EnsembleMemberSpec(
            cfg=cfg_fin,
            grids=grids_fin,
            geo=geo_fin,
            input_path=cfg_fin.input_path,
            name=name,
        )
    if isinstance(item, (str, Path)):
        in_path = Path(item)
        if not in_path.is_file():
            raise FileNotFoundError(f"Ensemble input file does not exist: {in_path}")
        cfg_loaded = load_config(in_path)
        cfg_pre = _apply_cfg_overrides(
            cfg_loaded,
            nstep=nstep,
            dt=dt,
            t_max=t_max,
            nwrite=nwrite,
            scheme=scheme,
            force_nonlinear=force_nonlinear,
            force_linear=force_linear,
            fft_backend=fft_backend,
        )
        cfg_fin, grids_fin, geo_fin = init_geometry(cfg_pre)
        return EnsembleMemberSpec(
            cfg=cfg_fin,
            grids=grids_fin,
            geo=geo_fin,
            input_path=in_path,
            name=_deck_stem(in_path),
        )
    raise TypeError(
        f"Unsupported ensemble input at index {idx}: {type(item).__name__}. "
        "Expected Path, str, GXConfig, (GXConfig, Grids, Geometry), or EnsembleMemberSpec."
    )


def validate_ensemble_compatibility(specs: Sequence[EnsembleMemberSpec]) -> None:
    """Validate structural & compile-time compatibility across all `B` ensemble members."""
    if len(specs) < 1:
        raise ValueError("Ensemble requires at least 1 member deck (got 0).")

    ref = specs[0]
    r_cfg, r_grids = ref.cfg, ref.grids
    r_scheme = r_cfg.scheme.strip().lower()
    r_closure = r_cfg.closure_model.strip().lower()
    r_fb = str(getattr(r_cfg, "fft_backend", "auto")).strip().lower()

    if len(r_cfg.species) != r_grids.nspecies:
        raise ValueError(
            f"Member 0 ({ref.name!r}) has len(cfg.species)={len(r_cfg.species)} "
            f"!= nspecies={r_grids.nspecies}."
        )

    for idx, spec in enumerate(specs[1:], start=1):
        c, g = spec.cfg, spec.grids
        if len(c.species) != g.nspecies:
            raise ValueError(
                f"Member {idx} ({spec.name!r}) has len(cfg.species)={len(c.species)} "
                f"!= nspecies={g.nspecies}."
            )
        for dim_name in (
            "nspecies",
            "nhermite",
            "nlaguerre",
            "nz",
            "naky",
            "nakx",
            "nx",
            "ny",
            "nj",
            "nyc",
        ):
            v0 = getattr(r_grids, dim_name)
            vi = getattr(g, dim_name)
            if vi != v0:
                raise ValueError(
                    f"Structural grid dimension mismatch on {dim_name!r}: "
                    f"member 0 ({ref.name!r}) has {dim_name}={v0}, "
                    f"member {idx} ({spec.name!r}) has {dim_name}={vi}."
                )

        if g.z.shape != r_grids.z.shape or spec.geo.bmag.shape != (r_grids.nz,):
            raise ValueError(
                f"Geometry z-grid shape mismatch on member {idx} ({spec.name!r}): "
                f"expected {(r_grids.nz,)}, got {spec.geo.bmag.shape}."
            )

        c_scheme = c.scheme.strip().lower()
        if c_scheme != r_scheme:
            raise ValueError(
                f"Time scheme mismatch: member 0 ({ref.name!r}) uses {r_scheme!r}, "
                f"member {idx} ({spec.name!r}) uses {c_scheme!r}."
            )

        if bool(c.nonlinear_mode) != bool(r_cfg.nonlinear_mode):
            raise ValueError(
                f"nonlinear_mode mismatch: member 0 ({ref.name!r}) has "
                f"nonlinear_mode={r_cfg.nonlinear_mode}, member {idx} ({spec.name!r}) has "
                f"nonlinear_mode={c.nonlinear_mode}."
            )

        c_closure = c.closure_model.strip().lower()
        if c_closure != r_closure:
            raise ValueError(
                f"closure_model mismatch: member 0 ({ref.name!r}) uses {r_closure!r}, "
                f"member {idx} ({spec.name!r}) uses {c_closure!r}."
            )
        if r_closure == "smith_par" and int(c.smith_par_q) != int(r_cfg.smith_par_q):
            raise ValueError(
                f"smith_par_q mismatch: member 0 has {r_cfg.smith_par_q}, "
                f"member {idx} has {c.smith_par_q}."
            )
        if r_closure == "smith_perp" and int(c.smith_perp_q) != int(r_cfg.smith_perp_q):
            raise ValueError(
                f"smith_perp_q mismatch: member 0 has {r_cfg.smith_perp_q}, "
                f"member {idx} has {c.smith_perp_q}."
            )

        if bool(c.all_kinetic) != bool(r_cfg.all_kinetic):
            raise ValueError(
                f"all_kinetic / Boltzmann mismatch: member 0 ({ref.name!r}) has "
                f"all_kinetic={r_cfg.all_kinetic}, member {idx} ({spec.name!r}) has "
                f"all_kinetic={c.all_kinetic}."
            )
        if not r_cfg.all_kinetic and c.boltzmann_opt != r_cfg.boltzmann_opt:
            raise ValueError(
                f"boltzmann_opt mismatch: member 0 ({ref.name!r}) uses {r_cfg.boltzmann_opt!r}, "
                f"member {idx} ({spec.name!r}) uses {c.boltzmann_opt!r}."
            )

        for flag_name in (
            "local_limit",
            "non_twist",
            "boundary_option_periodic",
            "dealias_kz",
            "long_wavelength_gk",
            "no_fields",
        ):
            f0 = bool(getattr(r_cfg, flag_name))
            fi = bool(getattr(c, flag_name))
            if fi != f0:
                raise ValueError(
                    f"Structural boundary/physics flag mismatch on {flag_name!r}: "
                    f"member 0 ({ref.name!r}) has {f0}, member {idx} ({spec.name!r}) has {fi}."
                )

        c_fb = str(getattr(c, "fft_backend", "auto")).strip().lower()
        if c_fb != r_fb:
            raise ValueError(
                f"fft_backend mismatch: member 0 ({ref.name!r}) uses {r_fb!r}, "
                f"member {idx} ({spec.name!r}) uses {c_fb!r}."
            )

    coll_specs = [s for s in specs if s.cfg.collisions]
    if len(coll_specs) > 1:
        c0 = coll_specs[0].cfg
        for s in coll_specs[1:]:
            if bool(s.cfg.coll_conservation) != bool(c0.coll_conservation) or bool(
                s.cfg.ei_colls
            ) != bool(c0.ei_colls):
                raise ValueError(
                    "Collisional members must share coll_conservation and ei_colls flags."
                )

    forc_specs = [s for s in specs if s.cfg.forcing_init]
    if len(forc_specs) > 1:
        f0 = forc_specs[0].cfg
        for s in forc_specs[1:]:
            if (
                s.cfg.forcing_type.strip().lower() != f0.forcing_type.strip().lower()
                or s.cfg.stir_field.strip().lower() != f0.stir_field.strip().lower()
            ):
                raise ValueError("Forced members must share forcing_type and stir_field.")


def _stack_pytrees(trees: Sequence[Any]) -> Any:
    """Stack a non-empty sequence of identical-treedef PyTrees along leading axis `0`."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


def _unify_parallel_grad_operators(
    specs: Sequence[EnsembleMemberSpec],
    raw_gps: Sequence[Any],
    r_dtype: jnp.dtype,
) -> Any:
    """Unify `B` parallel derivative operators into a single stackable PyTree.

    - When all members produce `ParallelGrad`, normalizes scalar `aux_data` and stacks directly.
    - When `non_twist=False` and at least one member uses multi-link `LinkedParallelGrad`
      (e.g. Tokamak + W7-X stellarator with distinct `shat`/`jtwist`), promotes all members
      onto a shared union-of-`n_links` `LinkedParallelGrad` bucket layout (preserving any
      `MXU1DPlan` attachments via `dataclasses.replace`).
    - When `non_twist=True`, unifies `NTFTParallelGrad` across the union of ballooning chain
      lengths `n_links` and pads `n_extra` to `max(mode_count)`.
    """
    gp0 = raw_gps[0]
    if all(isinstance(gp, ParallelGrad) for gp in raw_gps):
        norm_gps = [
            replace(
                gp,
                zp=gp0.zp,
                damp_ends_widthfrac=gp0.damp_ends_widthfrac,
                damp_ends_amp=gp0.damp_ends_amp,
                kpar_local=gp0.kpar_local,
            )
            for gp in raw_gps
        ]
        return _stack_pytrees(norm_gps)

    if not specs[0].cfg.non_twist:
        linked_gps: list[LinkedParallelGrad] = [
            gp
            if isinstance(gp, LinkedParallelGrad)
            else init_linked_parallel_grad(sp.cfg, sp.grids, dtype=r_dtype)
            for sp, gp in zip(specs, raw_gps, strict=True)
        ]
        same_buckets = all(
            gp.n_links == linked_gps[0].n_links and gp.n_chains == linked_gps[0].n_chains
            for gp in linked_gps[1:]
        )
        lg0 = linked_gps[0]
        if same_buckets:
            norm_lgps = [
                replace(
                    gp,
                    zp=lg0.zp,
                    jtwist=lg0.jtwist,
                    damp_ends_widthfrac=lg0.damp_ends_widthfrac,
                    damp_ends_amp=lg0.damp_ends_amp,
                    p_hyper_z=lg0.p_hyper_z,
                )
                for gp in linked_gps
            ]
            return _stack_pytrees(norm_lgps)

        nz = lg0.nz
        naky = lg0.naky
        nakx = lg0.nakx
        union_links = tuple(sorted({lnk for gp in linked_gps for lnk in gp.n_links}))
        max_chns: list[int] = []
        max_pads: list[int] = []
        ref_buckets: list[LinkedBucket] = []
        for lnk in union_links:
            matching = [gp.buckets[gp.n_links.index(lnk)] for gp in linked_gps if lnk in gp.n_links]
            m_chn = max(b.n_chains for b in matching)
            # Ensure n_chains_padded > n_chains when any member has fewer chains so chain_mask runs
            m_pad = max(max(b.n_chains_padded for b in matching), m_chn + 1)
            max_chns.append(m_chn)
            max_pads.append(m_pad)
            ref_buckets.append(matching[0])

        unified_lgps: list[LinkedParallelGrad] = []
        for sp, gp in zip(specs, linked_gps, strict=True):
            member_buckets: list[LinkedBucket] = []
            inv_yx_np = np.full((naky, nakx), -1, dtype=np.int32)
            seg_offset = 0
            for ic_u, lnk in enumerate(union_links):
                m_chn = max_chns[ic_u]
                m_pad = max_pads[ic_u]
                fft_len = nz * lnk
                ref_b = ref_buckets[ic_u]
                if lnk in gp.n_links:
                    orig_b = gp.buckets[gp.n_links.index(lnk)]
                    chn_b = orig_b.n_chains
                    orig_yx = np.asarray(orig_b.yx_idx)[:chn_b]
                    yx_pad = np.zeros((m_pad, lnk), dtype=np.int32)
                    yx_pad[:chn_b, :] = orig_yx
                    c_mask = np.zeros((m_pad,), dtype=bool)
                    c_mask[:chn_b] = True
                    for n_i in range(chn_b):
                        for p_i in range(lnk):
                            yx_val = int(orig_yx[n_i, p_i])
                            inv_yx_np[yx_val // nakx, yx_val % nakx] = seg_offset + n_i * lnk + p_i
                    kz_arr = orig_b.kz
                    kzm_arr = orig_b.kzm
                    hyp_arr = orig_b.hyperz_weight
                else:
                    yx_pad = np.zeros((m_pad, lnk), dtype=np.int32)
                    c_mask = np.zeros((m_pad,), dtype=bool)
                    kz64_np = _kz_linked_np(nz, sp.grids.zp, lnk, dealias_kz=False)
                    kz_arr = jnp.asarray(kz64_np, dtype=r_dtype)
                    kzm_arr = jnp.asarray(_kzm_linked_np(fft_len), dtype=jnp.int32)
                    hyp_arr = jnp.asarray(
                        _hyperz_weight_np(kz64_np, sp.cfg.p_hyper_z), dtype=r_dtype
                    )
                seg_offset += m_chn * lnk
                member_buckets.append(
                    replace(
                        ref_b,
                        yx_idx=jnp.asarray(yx_pad, dtype=jnp.int32),
                        chain_mask=jnp.asarray(c_mask, dtype=jnp.bool_),
                        kz=kz_arr,
                        kzm=kzm_arr,
                        hyperz_weight=hyp_arr,
                        n_links=lnk,
                        n_chains=m_chn,
                        n_chains_padded=m_pad,
                        fft_len=fft_len,
                    )
                )
            unified_lgps.append(
                replace(
                    gp,
                    buckets=tuple(member_buckets),
                    inv_yx_perm=jnp.asarray(inv_yx_np, dtype=jnp.int32),
                    zp=lg0.zp,
                    jtwist=lg0.jtwist,
                    n_classes=len(union_links),
                    n_links=union_links,
                    n_chains=tuple(max_chns),
                    damp_ends_widthfrac=lg0.damp_ends_widthfrac,
                    damp_ends_amp=lg0.damp_ends_amp,
                    p_hyper_z=lg0.p_hyper_z,
                )
            )
        return _stack_pytrees(unified_lgps)

    ntft_gps: list[NTFTParallelGrad] = [
        gp
        if isinstance(gp, NTFTParallelGrad)
        else init_ntft_parallel_grad(sp.cfg, sp.grids, sp.geo, dtype=r_dtype)
        for sp, gp in zip(specs, raw_gps, strict=True)
    ]
    ng0 = ntft_gps[0]
    nz = ng0.nz
    naky = ng0.naky
    nakx = ng0.nakx
    max_modes = max(gp.mode_count for gp in ntft_gps)
    union_links = tuple(sorted({lnk for gp in ntft_gps for lnk in gp.n_links}))
    max_chns = []
    max_pads = []
    ref_ntft_buckets: list[NTFTBucket] = []
    for lnk in union_links:
        matching_n = [gp.buckets[gp.n_links.index(lnk)] for gp in ntft_gps if lnk in gp.n_links]
        m_chn = max(b.n_chains for b in matching_n)
        m_pad = max(max(b.n_chains_padded for b in matching_n), m_chn + 1)
        max_chns.append(m_chn)
        max_pads.append(m_pad)
        ref_ntft_buckets.append(matching_n[0])

    unified_ngps: list[NTFTParallelGrad] = []
    for sp, gp in zip(specs, ntft_gps, strict=True):
        active_mask_np = np.asarray(sp.grids.active_mask, dtype=bool)
        inv_zyx_np = np.zeros((nz, naky, nakx), dtype=np.int32)
        pt_offset = 1
        member_ntft_buckets: list[NTFTBucket] = []
        for ic_u, lnk in enumerate(union_links):
            m_chn = max_chns[ic_u]
            m_pad = max_pads[ic_u]
            ref_nb = ref_ntft_buckets[ic_u]
            if lnk in gp.n_links:
                orig_nb = gp.buckets[gp.n_links.index(lnk)]
                chn_b = orig_nb.n_chains
                orig_zyx = np.asarray(orig_nb.zyx_idx)[:chn_b]
                zyx_pad = np.zeros((m_pad, lnk), dtype=np.int32)
                zyx_pad[:chn_b, :] = orig_zyx
                c_mask = np.zeros((m_pad,), dtype=bool)
                c_mask[:chn_b] = True
                for n_i in range(chn_b):
                    for p_i in range(lnk):
                        zyx_val = int(orig_zyx[n_i, p_i])
                        idz = zyx_val // (naky * nakx)
                        rem = zyx_val % (naky * nakx)
                        idy, idx = rem // nakx, rem % nakx
                        if active_mask_np[idy, idx]:
                            inv_zyx_np[idz, idy, idx] = pt_offset + n_i * lnk + p_i
                kz_arr = orig_nb.kz
                kzm_arr = orig_nb.kzm
                hyp_arr = orig_nb.hyperz_weight
            else:
                zyx_pad = np.zeros((m_pad, lnk), dtype=np.int32)
                c_mask = np.zeros((m_pad,), dtype=bool)
                kz64_np = _kz_linked_ntft_np(nz, sp.grids.zp, lnk, dealias_kz=False)
                kz_arr = jnp.asarray(kz64_np, dtype=r_dtype)
                kzm_arr = jnp.asarray(_kzm_linked_np(lnk), dtype=jnp.int32)
                hyp_arr = jnp.asarray(_hyperz_weight_np(kz64_np, sp.cfg.p_hyper_z), dtype=r_dtype)
            pt_offset += m_chn * lnk
            member_ntft_buckets.append(
                replace(
                    ref_nb,
                    zyx_idx=jnp.asarray(zyx_pad, dtype=jnp.int32),
                    chain_mask=jnp.asarray(c_mask, dtype=jnp.bool_),
                    kz=kz_arr,
                    kzm=kzm_arr,
                    hyperz_weight=hyp_arr,
                    n_links=lnk,
                    n_chains=m_chn,
                    n_chains_padded=m_pad,
                    fft_len=lnk,
                )
            )
        n_extra_pad = np.zeros((max_modes,), dtype=np.int32)
        orig_extra = np.asarray(gp.n_extra, dtype=np.int32)
        n_extra_pad[: orig_extra.size] = orig_extra
        unified_ngps.append(
            replace(
                gp,
                buckets=tuple(member_ntft_buckets),
                inv_zyx_map=jnp.asarray(inv_zyx_np, dtype=jnp.int32),
                n_extra=jnp.asarray(n_extra_pad, dtype=jnp.int32),
                zp=ng0.zp,
                jtwist=ng0.jtwist,
                mode_count=max_modes,
                n_classes=len(union_links),
                n_links=union_links,
                n_chains=tuple(max_chns),
                damp_ends_widthfrac=ng0.damp_ends_widthfrac,
                damp_ends_amp=ng0.damp_ends_amp,
                p_hyper_z=ng0.p_hyper_z,
            )
        )
    return _stack_pytrees(unified_ngps)


def init_batched_ensemble(
    inputs: Sequence[EnsembleInputItem],
    *,
    dtype: jnp.dtype = jnp.float32,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    fft_backend: str | None = None,
) -> tuple[list[EnsembleMemberSpec], BatchedEnsembleOperators]:
    """Build validated `EnsembleMemberSpec`s and stacked `BatchedEnsembleOperators`."""
    configure_tpu_precision()
    if len(inputs) < 1:
        raise ValueError("Ensemble requires at least 1 member deck (got 0).")

    target = jnp.dtype(dtype)
    r_dtype = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
    t_dtype = _time_dtype(r_dtype)

    with host_precompute_context():
        specs = [
            _resolve_member_spec(
                item,
                idx=idx,
                nstep=nstep,
                dt=dt,
                t_max=t_max,
                nwrite=nwrite,
                scheme=scheme,
                force_nonlinear=force_nonlinear,
                force_linear=force_linear,
                fft_backend=fft_backend,
            )
            for idx, item in enumerate(inputs)
        ]
        validate_ensemble_compatibility(specs)

        gk_ops = [
            init_gyrokinetic_operator(sp.cfg, sp.grids, sp.geo, dtype=r_dtype) for sp in specs
        ]

    ref_sp = specs[0]
    ref_cfg, ref_grids, ref_geo = ref_sp.cfg, ref_sp.grids, ref_sp.geo
    ref_gk = gk_ops[0]
    scheme_clean = ref_cfg.scheme.strip().lower()
    cfl_fac = float(get_scheme_cfl_fac(scheme_clean))
    closure_clean = ref_cfg.closure_model.strip().lower()
    fb_resolved = str(getattr(ref_cfg, "fft_backend", fft_backend or "auto")).strip().lower()

    has_any_fphi = any(sp.cfg.fphi != 0.0 for sp in specs)
    has_any_apar = any(sp.cfg.fapar > 0.0 for sp in specs)
    has_any_bpar = any(sp.cfg.fbpar > 0.0 for sp in specs)
    has_any_phi_ext = any(gk.linear_op.solver.phi_ext != 0.0 for gk in gk_ops)
    has_any_collisions = any(bool(sp.cfg.collisions) for sp in specs)
    coll_specs = [sp for sp in specs if sp.cfg.collisions]
    coll_conservation = (
        bool(coll_specs[0].cfg.coll_conservation) if coll_specs else bool(ref_cfg.coll_conservation)
    )
    ei_colls = bool(coll_specs[0].cfg.ei_colls) if coll_specs else bool(ref_cfg.ei_colls)

    has_any_hyper = any(bool(sp.cfg.hyper) for sp in specs)
    has_any_hb_hyper = any(bool(sp.cfg.hb_hyper) for sp in specs)
    has_any_hyperz = any(bool(sp.cfg.hyperz) for sp in specs)
    has_any_hc_const = any(bool(sp.cfg.hypercollisions_const) for sp in specs)
    has_any_hc_kz = any(bool(sp.cfg.hypercollisions_kz) for sp in specs)
    has_any_exb_shear = any(bool(sp.cfg.exb_shear) for sp in specs)
    has_any_exb_phase = any(bool(sp.cfg.exb_shear_phase) for sp in specs)
    has_any_forcing = any(bool(sp.cfg.forcing_init) for sp in specs)

    forc_specs = [sp for sp in specs if sp.cfg.forcing_init]
    forcing_type = (
        forc_specs[0].cfg.forcing_type.strip() if forc_specs else ref_cfg.forcing_type.strip()
    )
    stir_field = (
        forc_specs[0].cfg.stir_field.strip().lower()
        if forc_specs
        else ref_cfg.stir_field.strip().lower()
    )

    # 1. Normalize Grids & Geometry aux_data across members and stack array leaves
    with host_precompute_context():
        norm_grids = [
            replace(
                sp.grids.astype(r_dtype),
                zp=ref_grids.zp,
                jtwist=ref_grids.jtwist,
                x0=ref_grids.x0,
                y0=ref_grids.y0,
                vpar_max=ref_grids.vpar_max,
                muB_max=ref_grids.muB_max,
                muB_maxJ=ref_grids.muB_maxJ,
                kx_max=ref_grids.kx_max,
                ky_max=ref_grids.ky_max,
                kz_max=ref_grids.kz_max,
                kperp_min=ref_grids.kperp_min,
            )
            for sp in specs
        ]
        batched_grids: Grids = _stack_pytrees(norm_grids)

        norm_geos = [
            replace(
                sp.geo.astype(r_dtype),
                gradpar=ref_geo.gradpar,
                drhodpsi=ref_geo.drhodpsi,
                kxfac=ref_geo.kxfac,
                shat=ref_geo.shat,
                rmaj=ref_geo.rmaj,
                qsf=ref_geo.qsf,
                shift=ref_geo.shift,
                aminor=ref_geo.aminor,
                grho_avg=ref_geo.grho_avg,
                surfarea=ref_geo.surfarea,
                b_ref=ref_geo.b_ref,
                theta_scale=ref_geo.theta_scale,
                alpha=ref_geo.alpha,
                zeta_center=ref_geo.zeta_center,
                nfp=ref_geo.nfp,
                nperiod=ref_geo.nperiod,
                zero_shat=ref_geo.zero_shat,
            )
            for sp in specs
        ]
        batched_geo: Geometry = _stack_pytrees(norm_geos)

        # 2. Normalize FieldSolver aux_data across members and stack
        canon_fphi = 1.0 if has_any_fphi else 0.0
        canon_fapar = 1.0 if has_any_apar else 0.0
        canon_fbpar = 1.0 if has_any_bpar else 0.0
        canon_phi_ext = 1.0 if has_any_phi_ext else 0.0
        norm_solvers = [
            replace(
                gk.linear_op.solver,
                tau_fac=1.0,
                beta=1.0 if (has_any_apar or has_any_bpar) else 0.0,
                fphi=canon_fphi,
                fapar=canon_fapar,
                fbpar=canon_fbpar,
                phi_ext=canon_phi_ext,
            )
            for gk in gk_ops
        ]
        batched_solver: FieldSolver = _stack_pytrees(norm_solvers)

        # 3. Unify parallel boundary operators & normalize LinearOperator
        batched_grad_par = _unify_parallel_grad_operators(
            specs, [gk.linear_op.grad_par for gk in gk_ops], r_dtype
        )
        ref_lin = ref_gk.linear_op
        norm_lins_no_gp = []
        for sp, gk, sol_n in zip(specs, gk_ops, norm_solvers, strict=True):
            lin = gk.linear_op
            hd_fac = lin.hyperdiff_fac if sp.cfg.hyper else jnp.zeros_like(lin.hyperdiff_fac)
            hb10 = lin.hb_d10 if sp.cfg.hb_hyper else jnp.zeros_like(lin.hb_d10)
            hb01 = lin.hb_d01 if sp.cfg.hb_hyper else jnp.zeros_like(lin.hb_d01)
            hb11 = lin.hb_d11 if sp.cfg.hb_hyper else jnp.zeros_like(lin.hb_d11)
            hc_rate = (
                lin.hypercoll_const_rate
                if sp.cfg.hypercollisions_const
                else jnp.zeros_like(lin.hypercoll_const_rate)
            )
            hc_lm = (
                lin.hypercoll_const_lm
                if sp.cfg.hypercollisions_const
                else jnp.zeros_like(lin.hypercoll_const_lm)
            )
            hc_kz = (
                lin.hypercoll_kz_fac
                if sp.cfg.hypercollisions_kz
                else jnp.zeros_like(lin.hypercoll_kz_fac)
            )
            norm_lins_no_gp.append(
                replace(
                    lin,
                    solver=sol_n,
                    grad_par=None,
                    hyperdiff_fac=hd_fac,
                    hb_d10=hb10,
                    hb_d01=hb01,
                    hb_d11=hb11,
                    hypercoll_const_rate=hc_rate,
                    hypercoll_const_lm=hc_lm,
                    hypercoll_kz_fac=hc_kz,
                    gradpar=ref_lin.gradpar,
                    vtmax=ref_lin.vtmax,
                    collisions=has_any_collisions,
                    coll_conservation=coll_conservation,
                    ei_colls=ei_colls,
                    hyper=has_any_hyper,
                    hb_hyper=has_any_hb_hyper,
                    d_hb=ref_lin.d_hb,
                    p_hb=ref_lin.p_hb,
                    w_osc=ref_lin.w_osc,
                    hyperz=has_any_hyperz,
                    nu_hyper_z=ref_lin.nu_hyper_z,
                    hypercollisions_const=has_any_hc_const,
                    hypercollisions_kz=has_any_hc_kz,
                    nu_hyper_lm=ref_lin.nu_hyper_lm,
                )
            )
        stacked_lin_no_gp: LinearOperator = _stack_pytrees(norm_lins_no_gp)
        batched_lin = replace(stacked_lin_no_gp, grad_par=batched_grad_par)

        # 4. Normalize NonlinearOperator (if nonlinear_mode=True)
        batched_nl: NonlinearOperator | None = None
        if ref_cfg.nonlinear_mode:
            nl_list = [gk.nonlinear_op for gk in gk_ops]
            assert all(nl is not None for nl in nl_list)
            nl0 = nl_list[0]
            assert nl0 is not None
            gp_perp0 = nl0.grad_perp
            norm_nls = []
            for nl in nl_list:
                assert nl is not None
                gp_p = replace(
                    nl.grad_perp,
                    x0=gp_perp0.x0,
                    y0=gp_perp0.y0,
                    kxfac=gp_perp0.kxfac,
                    exb_shear=has_any_exb_shear,
                    exb_shear_phase=has_any_exb_phase,
                )
                norm_nls.append(
                    replace(
                        nl,
                        grad_perp=gp_p,
                        kxfac=nl0.kxfac,
                        fphi=canon_fphi,
                        fapar=canon_fapar,
                        fbpar=canon_fbpar,
                        vpar_max=nl0.vpar_max,
                        muB_maxJ=nl0.muB_maxJ,
                        vtmax=nl0.vtmax,
                        tzmax=nl0.tzmax,
                        kx_max=nl0.kx_max,
                        ky_max=nl0.ky_max,
                        g_exb=1.0 if has_any_exb_shear else 0.0,
                        exb_shear=has_any_exb_shear,
                        exb_shear_phase=has_any_exb_phase,
                    )
                )
            batched_nl = _stack_pytrees(norm_nls)

        # 5. Optional ForcingOperator & ExBShearState
        batched_forc: ForcingOperator | None = None
        forc_targets_list: list[tuple[int, int, int]] = []
        if has_any_forcing:
            raw_forcs = [
                gk.forcing_op
                if gk.forcing_op is not None
                else init_forcing(sp.cfg, sp.grids, dtype=r_dtype)
                for sp, gk in zip(specs, gk_ops, strict=True)
            ]
            f0_op = raw_forcs[0]
            norm_forcs = []
            for f_op in raw_forcs:
                forc_targets_list.append((f_op.idz_target, f_op.idy_target, f_op.idx_target))
                norm_forcs.append(
                    replace(
                        f_op,
                        forcing_init=True,
                        forcing_type=forcing_type,
                        stir_field=stir_field,
                        forcing_amp=1.0,
                        pos_forcing_amp=f0_op.pos_forcing_amp,
                        neg_forcing_amp=f0_op.neg_forcing_amp,
                        forcing_index=f0_op.forcing_index,
                        idz_target=f0_op.idz_target,
                        idy_target=f0_op.idy_target,
                        idx_target=f0_op.idx_target,
                        forcing_kz=f0_op.forcing_kz,
                        forcing_k2min=f0_op.forcing_k2min,
                        forcing_k2max=f0_op.forcing_k2max,
                    )
                )
            batched_forc = _stack_pytrees(norm_forcs)
        else:
            forc_targets_list = [(0, 0, 0) for _ in specs]

        batched_exb: ExBShearState | None = None
        if has_any_exb_shear:
            raw_exbs = [
                gk.exb_init
                if gk.exb_init is not None
                else init_exb_shear(sp.cfg, sp.grids, sp.geo, dtype=r_dtype)
                for sp, gk in zip(specs, gk_ops, strict=True)
            ]
            e0 = raw_exbs[0]
            norm_exbs = [
                replace(
                    exb,
                    g_exb=1.0,
                    x0=e0.x0,
                    shat=e0.shat,
                    non_twist=bool(ref_cfg.non_twist),
                    exb_shear_phase=has_any_exb_phase,
                )
                for exb in raw_exbs
            ]
            batched_exb = _stack_pytrees(norm_exbs)

        # 6. Build stacked per-member scalar arrays
        t_max_list = [float(sp.cfg.t_max) for sp in specs]
        eps_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-11
        eps_t_list = [eps_rel * max(abs(tm), 1.0e-30) for tm in t_max_list]
        dt_list = [float(sp.cfg.dt) if sp.cfg.dt > 0.0 else 0.05 for sp in specs]

        ops = BatchedEnsembleOperators(
            grids=batched_grids,
            geo=batched_geo,
            solver=batched_solver,
            linear_op=batched_lin,
            nonlinear_op=batched_nl,
            forcing_op=batched_forc,
            exb_init=batched_exb,
            beta_b=jnp.asarray([float(sp.cfg.beta) for sp in specs], dtype=r_dtype),
            gradpar_b=jnp.asarray([float(sp.geo.gradpar) for sp in specs], dtype=r_dtype),
            vtmax_b=jnp.asarray([float(sp.cfg.vtmax) for sp in specs], dtype=r_dtype),
            tzmax_b=jnp.asarray([float(sp.cfg.tzmax) for sp in specs], dtype=r_dtype),
            fphi_b=jnp.asarray([float(sp.cfg.fphi) for sp in specs], dtype=r_dtype),
            fapar_b=jnp.asarray([float(sp.cfg.fapar) for sp in specs], dtype=r_dtype),
            fbpar_b=jnp.asarray([float(sp.cfg.fbpar) for sp in specs], dtype=r_dtype),
            tau_fac_b=jnp.asarray([float(sp.cfg.tau_fac) for sp in specs], dtype=r_dtype),
            phi_ext_b=jnp.asarray(
                [float(gk.linear_op.solver.phi_ext) for gk in gk_ops], dtype=r_dtype
            ),
            kxfac_b=jnp.asarray([float(sp.geo.kxfac) for sp in specs], dtype=r_dtype),
            shat_b=jnp.asarray([float(sp.geo.shat) for sp in specs], dtype=r_dtype),
            x0_b=jnp.asarray([float(sp.grids.x0) for sp in specs], dtype=r_dtype),
            y0_b=jnp.asarray([float(sp.grids.y0) for sp in specs], dtype=r_dtype),
            kx_max_b=jnp.asarray([float(sp.grids.kx_max) for sp in specs], dtype=r_dtype),
            ky_max_b=jnp.asarray([float(sp.grids.ky_max) for sp in specs], dtype=r_dtype),
            kz_max_b=jnp.asarray([float(sp.grids.kz_max) for sp in specs], dtype=r_dtype),
            vpar_max_b=jnp.asarray([float(sp.grids.vpar_max) for sp in specs], dtype=r_dtype),
            mub_maxj_b=jnp.asarray([float(sp.grids.muB_maxJ) for sp in specs], dtype=r_dtype),
            kpar_local_b=jnp.asarray([1.0 / float(sp.grids.zp) for sp in specs], dtype=r_dtype),
            damp_ends_amp_b=jnp.asarray(
                [float(sp.cfg.damp_ends_amp) for sp in specs], dtype=r_dtype
            ),
            nu_hyper_z_b=jnp.asarray(
                [float(sp.cfg.nu_hyper_z) if sp.cfg.hyperz else 0.0 for sp in specs],
                dtype=r_dtype,
            ),
            w_osc_b=jnp.asarray([float(sp.cfg.w_osc) for sp in specs], dtype=r_dtype),
            g_exb_b=jnp.asarray(
                [float(sp.cfg.g_exb) if sp.cfg.exb_shear else 0.0 for sp in specs],
                dtype=r_dtype,
            ),
            forcing_amp_b=jnp.asarray(
                [float(sp.cfg.forcing_amp) if sp.cfg.forcing_init else 0.0 for sp in specs],
                dtype=r_dtype,
            ),
            omega_linear_b=jnp.asarray([gk.omega_linear for gk in gk_ops], dtype=r_dtype),
            dt_b=jnp.asarray(dt_list, dtype=t_dtype),
            dt_min_b=jnp.asarray([float(sp.cfg.dt_min) for sp in specs], dtype=t_dtype),
            dt_max_b=jnp.asarray([float(sp.cfg.dt_max) for sp in specs], dtype=t_dtype),
            t_max_b=jnp.asarray(t_max_list, dtype=t_dtype),
            eps_t_b=jnp.asarray(eps_t_list, dtype=t_dtype),
            cfl_b=jnp.asarray([float(sp.cfg.cfl) for sp in specs], dtype=t_dtype),
            nstep_b=jnp.asarray([max(0, int(sp.cfg.nstep)) for sp in specs], dtype=jnp.int32),
            nwrite_b=jnp.asarray([max(1, int(sp.cfg.nwrite)) for sp in specs], dtype=jnp.int32),
            nsave_b=jnp.asarray([max(1, int(sp.cfg.nsave)) for sp in specs], dtype=jnp.int32),
            random_seed_b=jnp.asarray([int(sp.cfg.random_seed) for sp in specs], dtype=jnp.int32),
            forcing_target_zyx_b=jnp.asarray(forc_targets_list, dtype=jnp.int32),
            fixed_dt_b=jnp.asarray([bool(sp.cfg.fixed_dt) for sp in specs], dtype=jnp.bool_),
            batch_size=len(specs),
            nspecies=int(ref_grids.nspecies),
            nhermite=int(ref_grids.nhermite),
            nlaguerre=int(ref_grids.nlaguerre),
            nj=int(ref_grids.nj),
            nz=int(ref_grids.nz),
            naky=int(ref_grids.naky),
            nakx=int(ref_grids.nakx),
            nx=int(ref_grids.nx),
            ny=int(ref_grids.ny),
            nyc=int(ref_grids.nyc),
            scheme=scheme_clean,
            cfl_fac=cfl_fac,
            closure_model=closure_clean,
            smith_par_q=int(ref_cfg.smith_par_q),
            smith_perp_q=int(ref_cfg.smith_perp_q),
            all_kinetic=bool(ref_cfg.all_kinetic),
            boltzmann_opt=ref_cfg.boltzmann_opt,
            nonlinear_mode=bool(ref_cfg.nonlinear_mode),
            non_twist=bool(ref_cfg.non_twist),
            local_limit=bool(ref_cfg.local_limit),
            boundary_option_periodic=bool(ref_cfg.boundary_option_periodic),
            dealias_kz=bool(ref_cfg.dealias_kz),
            long_wavelength_gk=bool(ref_cfg.long_wavelength_gk),
            no_fields=bool(ref_cfg.no_fields),
            has_any_fphi=has_any_fphi,
            has_any_apar=has_any_apar,
            has_any_bpar=has_any_bpar,
            has_any_phi_ext=has_any_phi_ext,
            has_any_collisions=has_any_collisions,
            coll_conservation=coll_conservation,
            ei_colls=ei_colls,
            has_any_hyper=has_any_hyper,
            has_any_hb_hyper=has_any_hb_hyper,
            has_any_hyperz=has_any_hyperz,
            has_any_hypercollisions_const=has_any_hc_const,
            has_any_hypercollisions_kz=has_any_hc_kz,
            has_any_exb_shear=has_any_exb_shear,
            has_any_exb_shear_phase=has_any_exb_phase,
            has_any_forcing=has_any_forcing,
            forcing_type=forcing_type,
            stir_field=stir_field,
            fft_backend=fb_resolved,
        )
    return specs, move_tree_to_default_device(ops)


# ---------------------------------------------------------------------------
# Multi-Device 1D `("batch",)` Mesh Construction & Sharding Utilities
# ---------------------------------------------------------------------------


def create_ensemble_mesh(
    batch_size: int,
    *,
    num_devices: int | None = None,
    devices: Sequence[Any] | None = None,
) -> Mesh:
    """Create a 1D `jax.sharding.Mesh` with axis `('batch',)` for `batch_size` members."""
    configure_tpu_precision()
    if batch_size < 1:
        raise ValueError(f"batch_size={batch_size} must be >= 1.")
    devs = list(jax.devices() if devices is None else devices)
    if not devs:
        raise RuntimeError("No JAX devices available to construct ensemble Mesh.")

    if num_devices is not None:
        n_dev = int(num_devices)
        if n_dev < 1:
            raise ValueError(f"num_devices={n_dev} must be >= 1.")
        if n_dev > len(devs):
            raise ValueError(
                f"Requested num_devices={n_dev}, but only {len(devs)} devices are available."
            )
        if batch_size % n_dev != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by num_devices={n_dev} "
                "for 1D 'batch' mesh sharding."
            )
    else:
        n_dev = 1
        for cand in range(min(len(devs), batch_size), 0, -1):
            if batch_size % cand == 0:
                n_dev = cand
                break

    dev_arr = np.asarray(devs[:n_dev], dtype=object)
    return Mesh(dev_arr, axis_names=("batch",))


def _leaf_sharding_for_batch(leaf: Any, mesh: Mesh, batch_size: int) -> NamedSharding:
    if isinstance(leaf, jax.Array) and leaf.ndim >= 1 and leaf.shape[0] == batch_size:
        spec = PartitionSpec("batch", *([None] * (leaf.ndim - 1)))
        return NamedSharding(mesh, spec)
    return NamedSharding(mesh, PartitionSpec())


def _tree_sharding_for_batch(tree: Any, mesh: Mesh, batch_size: int) -> Any:
    return jax.tree_util.tree_map(lambda x: _leaf_sharding_for_batch(x, mesh, batch_size), tree)


def shard_ensemble_tree(tree: Any, mesh: Mesh, *, batch_size: int) -> Any:
    """Place every `(B, ...)` leaf of `tree` onto `mesh` along `'batch'`.

    Multi-host (`jax.process_count() > 1`): leaves become global arrays through
    `jax.make_array_from_callback` (`jax.device_put` cannot target non-addressable devices).
    """
    return jax.tree_util.tree_map(
        lambda x: (
            device_put_global(x, _leaf_sharding_for_batch(x, mesh, batch_size))
            if isinstance(x, jax.Array)
            else x
        ),
        tree,
    )


def gather_ensemble_tree(tree: Any, mesh: Mesh | None = None) -> Any:
    """Gather sharded `(B, ...)` leaves into replicated arrays.

    Multi-host (`jax.process_count() > 1`): every process must call this in lockstep;
    leaves are gathered with `multihost_utils.process_allgather(..., tiled=True)` and
    returned as process-local (fully addressable) arrays on every process.
    """
    if is_distributed():
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(gather_global_array(x)) if isinstance(x, jax.Array) else x,
            tree,
        )
    if mesh is None:
        return tree
    rep = NamedSharding(mesh, PartitionSpec())
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, rep) if isinstance(x, jax.Array) else x,
        tree,
    )


# ---------------------------------------------------------------------------
# Pure-JAX Single-Member Physics Kernels (Vectorized via `jax.vmap`)
# ---------------------------------------------------------------------------


def _solve_fields_member(
    G: jax.Array,
    op_m: BatchedEnsembleOperators,
    *,
    solver_override: FieldSolver | None = None,
) -> Fields:
    """Pure-JAX per-member field solve safe for mixed ES (`beta=0`) + EM (`beta>0`) batches."""
    solver = op_m.solver if solver_override is None else solver_override
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    field_shape = (op_m.nz, op_m.naky, op_m.nakx)
    zero3 = jnp.zeros(field_shape, dtype=c_dtype)
    zero1 = jnp.zeros((op_m.nakx,), dtype=c_dtype)

    if op_m.no_fields:
        return Fields(phi=zero3, apar=zero3, bpar=zero3, phi_avg=zero1)

    mask3d = solver.active_mask[None, :, :]
    fphi = op_m.fphi_b.astype(r_dtype)
    fapar = op_m.fapar_b.astype(r_dtype)
    fbpar = op_m.fbpar_b.astype(r_dtype)
    tau = op_m.tau_fac_b.astype(r_dtype)

    nbar = real_space_density(G, solver)

    if op_m.all_kinetic:
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        phi_es = (
            jnp.where(fphi != 0.0, _safe_divide(fphi * nbar, q_phi, mask3d), zero3)
            if op_m.has_any_fphi
            else zero3
        )
        if op_m.has_any_bpar:
            jperpbar = real_space_perp_current(G, solver)
            q_bpar = solver.qneut_fac_bpar.astype(r_dtype)
            a_phi = solver.ampere_perp_fac_phi.astype(r_dtype)
            a_bpar = solver.ampere_perp_fac_bpar.astype(r_dtype)
            det = solver.phi_bpar_det.astype(r_dtype)
            num_phi = fphi * (a_bpar * nbar - q_bpar * jperpbar)
            num_bpar = fbpar * (-a_phi * nbar + q_phi * jperpbar)
            phi_em = _safe_divide(num_phi, det, mask3d)
            bpar_em = _safe_divide(num_bpar, det, mask3d)
            use_bpar = fbpar > 0.0
            phi = jnp.where(use_bpar, phi_em, phi_es)
            bpar = jnp.where(use_bpar, bpar_em, zero3)
        else:
            phi = phi_es
            bpar = zero3

        if op_m.has_any_apar and op_m.nhermite >= 2:
            jparbar = real_space_par_current(G, solver)
            a_par = solver.ampere_par_fac.astype(r_dtype)
            apar_em = _safe_divide(fapar * jparbar, a_par, mask3d)
            apar = jnp.where(fapar > 0.0, apar_em, zero3)
        else:
            apar = zero3
        phi_avg = zero1

    elif op_m.boltzmann_opt == BoltzmannOption.BOLTZMANN_ELECTRONS:
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        denom = tau + q_phi
        jacobian = solver.jacobian.astype(r_dtype)
        mask_ky0 = solver.active_mask[0, :]
        tmp_ky0 = _safe_divide(nbar[:, 0, :], denom[:, 0, :], mask_ky0[None, :]) * jacobian[:, None]
        phi_avg_num = jnp.sum(tmp_ky0, axis=0)
        phi_avg_raw = _safe_divide(phi_avg_num, solver.phiavgdenom.astype(r_dtype), mask_ky0)
        if op_m.has_any_fphi:
            rhs_phi = nbar.at[:, 0, :].add(tau * phi_avg_raw[None, :])
            phi_b = _safe_divide(fphi * rhs_phi, denom, mask3d)
            phi = jnp.where(fphi != 0.0, phi_b, zero3)
            phi_avg = jnp.where(fphi != 0.0, (fphi * phi_avg_raw).astype(c_dtype), zero1)
        else:
            phi = zero3
            phi_avg = zero1
        apar = zero3
        bpar = zero3

    else:
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        denom = tau + q_phi
        phi = (
            jnp.where(fphi != 0.0, _safe_divide(fphi * nbar, denom, mask3d), zero3)
            if op_m.has_any_fphi
            else zero3
        )
        if op_m.has_any_apar and op_m.nhermite >= 2:
            jparbar = real_space_par_current(G, solver)
            a_par = solver.ampere_par_fac.astype(r_dtype)
            apar = jnp.where(fapar > 0.0, _safe_divide(fapar * jparbar, a_par, mask3d), zero3)
        else:
            apar = zero3
        bpar = zero3
        phi_avg = zero1

    if op_m.has_any_phi_ext:
        ext = op_m.phi_ext_b.astype(r_dtype)
        phi = jnp.where(
            ext != 0.0,
            jnp.where(mask3d, phi + ext, jnp.zeros_like(phi)),
            phi,
        )

    return Fields(
        phi=phi.astype(c_dtype),
        apar=apar.astype(c_dtype),
        bpar=bpar.astype(c_dtype),
        phi_avg=phi_avg.astype(c_dtype),
    )


def _shift_active_kx_member(
    arr: jax.Array,
    exb_state: ExBShearState,
    grids: Grids,
    g_exb_b: jax.Array,
) -> jax.Array:
    nakx = grids.nakx
    if nakx <= 1:
        return arr
    half = nakx // 2
    z1 = jnp.zeros((*arr.shape[:-1], 1), dtype=arr.dtype)
    shifted_pos = jnp.concatenate(
        [arr[..., 1 : half + 1], z1, arr[..., half + 2 :], arr[..., 0:1]],
        axis=-1,
    )
    shifted_neg = jnp.concatenate(
        [arr[..., -1:], arr[..., :half], z1, arr[..., half + 1 : -1]],
        axis=-1,
    )
    shifted = jnp.where(g_exb_b > 0.0, shifted_pos, shifted_neg)
    jumped_row = (
        (g_exb_b != 0.0)
        & (jnp.arange(grids.naky, dtype=jnp.int32) > 0)
        & (exb_state.kxbar_ikx_new[:, 0] != exb_state.kxbar_ikx_old[:, 0])
    )
    out = jnp.where(jumped_row[:, None], shifted, arr)
    return jnp.where(grids.active_mask, out, jnp.zeros_like(out))


def _flow_shear_shift_member(
    exb_state: ExBShearState,
    fields: Fields,
    op_m: BatchedEnsembleOperators,
    dt_r: jax.Array,
) -> tuple[ExBShearState, Fields]:
    r_dtype = exb_state.kxstar.dtype
    c_dtype = exb_state.phasefac_exb.dtype
    grids = op_m.grids
    geo = op_m.geo
    g_exb_r = op_m.g_exb_b.astype(r_dtype)
    x0_r = op_m.x0_b.astype(r_dtype)
    shat_r = op_m.shat_b.astype(r_dtype)
    dkx = jnp.asarray(1.0, dtype=r_dtype) / x0_r

    ky_r = exb_state.ky.astype(r_dtype)[:, None]
    idy_pos = (jnp.arange(op_m.naky, dtype=jnp.int32) > 0)[:, None]

    kxbar_old = jnp.where(
        idy_pos,
        _c_round_jax(exb_state.kxstar / dkx).astype(jnp.int32),
        exb_state.kxbar_ikx_old,
    )
    kxstar_raw = jnp.where(
        idy_pos,
        exb_state.kxstar - ky_r * g_exb_r * dt_r,
        exb_state.kxstar,
    )
    kxbar_new = jnp.where(
        idy_pos,
        _c_round_jax(kxstar_raw / dkx).astype(jnp.int32),
        exb_state.kxbar_ikx_new,
    )
    sign_exb = jnp.where(
        g_exb_r > 0.0,
        jnp.asarray(1.0, dtype=r_dtype),
        jnp.asarray(-1.0, dtype=r_dtype),
    )
    jumped = (g_exb_r != 0.0) & idy_pos & (kxbar_new != kxbar_old)
    kxstar_next = jnp.where(jumped, kxstar_raw + sign_exb * dkx, kxstar_raw)

    if op_m.has_any_exb_shear_phase:
        kxbar_cur = _c_round_jax(kxstar_next[:, :1] / dkx)
        dkx_exb = kxstar_next[:, :1] - kxbar_cur * dkx
        phase_act = jnp.where(
            (g_exb_r != 0.0) & (jnp.arange(op_m.naky) > 0)[:, None],
            dkx_exb * exb_state.x.astype(r_dtype)[None, :],
            jnp.zeros((op_m.naky, op_m.nx), dtype=r_dtype),
        )
        ph = phase_act.astype(r_dtype)
        cos_ph = jnp.cos(ph)
        sin_ph = jnp.sin(ph)
        pf_act = jax.lax.complex(cos_ph, sin_ph).astype(c_dtype)
        pfm_act = jax.lax.complex(cos_ph, -sin_ph).astype(c_dtype)
        pad_y = op_m.nyc - op_m.naky
        if pad_y > 0:
            ones_pad = jnp.ones((pad_y, op_m.nx), dtype=c_dtype)
            pf_exb = jnp.concatenate([pf_act, ones_pad], axis=0)
            pfm_exb = jnp.concatenate([pfm_act, ones_pad], axis=0)
        else:
            pf_exb = pf_act
            pfm_exb = pfm_act
        pf_exb_nxny = jnp.swapaxes(pf_exb, -2, -1)
        pfm_exb_nxny = jnp.swapaxes(pfm_exb, -2, -1)
    else:
        pf_exb = exb_state.phasefac_exb
        pfm_exb = exb_state.phasefacminus_exb
        pf_exb_nxny = exb_state.phasefac_exb_nxny
        pfm_exb_nxny = exb_state.phasefacminus_exb_nxny

    kxs3 = kxstar_next[None, :, :]
    ky3 = exb_state.ky.astype(r_dtype)[None, :, None]
    gds2_3 = geo.gds2.astype(r_dtype)[:, None, None]
    gds21_3 = geo.gds21.astype(r_dtype)[:, None, None]
    gds22_3 = geo.gds22.astype(r_dtype)[:, None, None]
    bmag_inv_3 = geo.bmag_inv.astype(r_dtype)[:, None, None]
    cv_3 = geo.cvdrift.astype(r_dtype)[:, None, None]
    gb_3 = geo.gbdrift.astype(r_dtype)[:, None, None]
    cv0_3 = geo.cvdrift0.astype(r_dtype)[:, None, None]
    gb0_3 = geo.gbdrift0.astype(r_dtype)[:, None, None]
    mask3 = grids.active_mask[None, :, :]
    idy_pos3 = (jnp.arange(op_m.naky) > 0)[None, :, None] & mask3
    safe_shat = jnp.where(shat_r != 0.0, shat_r, jnp.ones_like(shat_r))
    shat_inv = jnp.where(shat_r != 0.0, 1.0 / safe_shat, jnp.ones_like(shat_r))

    if op_m.non_twist and geo.delta_kx.size > 0:
        ft_3 = geo.ftwist.astype(r_dtype)[:, None, None]
        dkx_3 = geo.delta_kx.astype(r_dtype)[:, :, None]
        m0_ov_x0 = (geo.m0.astype(r_dtype) / x0_r)[:, :, None]
        kp2_new = (
            (ky3**2)
            * (gds2_3 - 2.0 * ft_3 * gds21_3 * shat_inv + (ft_3**2) * gds22_3 * shat_inv**2)
            + ((kxs3 + dkx_3) ** 2) * gds22_3 * (shat_inv**2)
        ) * (bmag_inv_3**2)
        cv_d_new = ky3 * cv_3 + (kxs3 + m0_ov_x0) * shat_inv * cv0_3
        gb_d_new = ky3 * gb_3 + (kxs3 + m0_ov_x0) * shat_inv * gb0_3
        omegad_new = cv_d_new + gb_d_new
        kperp2 = jnp.where(idy_pos3, kp2_new, exb_state.kperp2)
        cv_d = jnp.where(idy_pos3, cv_d_new, exb_state.cv_d)
        gb_d = jnp.where(idy_pos3, gb_d_new, exb_state.gb_d)
        omegad = jnp.where(idy_pos3, omegad_new, exb_state.omegad)
    else:
        kxs_eff = kxs3 * shat_inv
        kp2_new = (ky3 * (ky3 * gds2_3 + 2.0 * kxs_eff * gds21_3) + (kxs_eff**2) * gds22_3) * (
            bmag_inv_3**2
        )
        cv_d_new = ky3 * cv_3 + kxs_eff * cv0_3
        gb_d_new = ky3 * gb_3 + kxs_eff * gb0_3
        omegad_new = cv_d_new + gb_d_new
        kperp2 = jnp.where(mask3, kp2_new, jnp.zeros_like(kp2_new))
        cv_d = jnp.where(mask3, cv_d_new, jnp.zeros_like(cv_d_new))
        gb_d = jnp.where(mask3, gb_d_new, jnp.zeros_like(gb_d_new))
        omegad = jnp.where(mask3, omegad_new, jnp.zeros_like(omegad_new))

    shifted_exb = replace(
        exb_state,
        kxstar=kxstar_next,
        kxbar_ikx_new=kxbar_new,
        kxbar_ikx_old=kxbar_old,
        phasefac_exb=pf_exb,
        phasefacminus_exb=pfm_exb,
        phasefac_exb_nxny=pf_exb_nxny,
        phasefacminus_exb_nxny=pfm_exb_nxny,
        kperp2=kperp2,
        omegad=omegad,
        cv_d=cv_d,
        gb_d=gb_d,
    )
    next_exb = jax.tree_util.tree_map(
        lambda s_v, o_v: jnp.where(g_exb_r != 0.0, s_v, o_v),
        shifted_exb,
        exb_state,
    )
    fields_next = Fields(
        phi=_shift_active_kx_member(fields.phi, next_exb, grids, g_exb_r),
        apar=_shift_active_kx_member(fields.apar, next_exb, grids, g_exb_r),
        bpar=_shift_active_kx_member(fields.bpar, next_exb, grids, g_exb_r),
        phi_avg=fields.phi_avg,
    )
    return next_exb, fields_next


def _prepare_member_linear_op(
    op_m: BatchedEnsembleOperators,
    exb_state: ExBShearState | None,
    r_dtype: jnp.dtype,
) -> LinearOperator:
    lin = op_m.linear_op
    gp = lin.grad_par
    if isinstance(gp, ParallelGrad):
        gp_m: Any = replace(
            gp,
            damp_ends_amp=op_m.damp_ends_amp_b.astype(r_dtype),
            kpar_local=op_m.kpar_local_b.astype(r_dtype),
        )
    else:
        gp_m = replace(gp, damp_ends_amp=op_m.damp_ends_amp_b.astype(r_dtype))

    lin_m = replace(
        lin,
        grad_par=gp_m,
        gradpar=op_m.gradpar_b.astype(r_dtype),
        nu_hyper_z=op_m.nu_hyper_z_b.astype(r_dtype),
        w_osc=op_m.w_osc_b.astype(r_dtype),
    )
    if exb_state is None or not op_m.has_any_exb_shear:
        return lin_m

    kp2 = exb_state.kperp2.astype(r_dtype)
    omegad = exb_state.omegad.astype(r_dtype)
    cv_d = exb_state.cv_d.astype(r_dtype)
    gb_d = exb_state.gb_d.astype(r_dtype)

    nl = op_m.nlaguerre
    rho2 = lin_m.solver.sp_rho2.astype(r_dtype)[:, None, None, None]
    b_s = kp2[None, :, :, :] * rho2
    sqrt_b = _safe_sqrt(b_s)
    neg_hb_s = jnp.asarray(-0.5, dtype=r_dtype) * b_s
    j_cur = jnp.exp(neg_hb_s)
    jl_list = [j_cur]
    for l_idx in range(1, nl + 1):
        j_cur = j_cur * (neg_hb_s / jnp.asarray(float(l_idx), dtype=r_dtype))
        jl_list.append(j_cur)
    jl_all = jnp.stack(jl_list, axis=1)

    jl = jl_all[:, :nl]
    jl_p1 = jl_all[:, 1 : nl + 1]
    jl_m1 = jnp.concatenate([jnp.zeros_like(jl[:, :1]), jl[:, :-1]], axis=1)
    l_vec = jnp.arange(nl, dtype=r_dtype)[None, :, None, None, None]
    two = jnp.asarray(2.0, dtype=r_dtype)
    one = jnp.asarray(1.0, dtype=r_dtype)

    jla = l_vec * jl_m1 + (two * l_vec) * jl + (l_vec + one) * jl_p1
    jlb_all = jl_all + jnp.concatenate([jnp.zeros_like(jl_all[:, :1]), jl_all[:, :-1]], axis=1)
    jlb = jlb_all[:, :nl]
    jlb_p1 = jlb_all[:, 1 : nl + 1]
    jlb_m1 = jnp.concatenate([jnp.zeros_like(jlb[:, :1]), jlb[:, :-1]], axis=1)
    jlba = l_vec * jlb_m1 + (two * l_vec) * jlb + (l_vec + one) * jlb_p1

    has_sh = op_m.g_exb_b != 0.0
    jl_out = jnp.where(has_sh, jl, lin_m.jl)
    jlb_out = jnp.where(has_sh, jlb, lin_m.jlb)
    jla_out = jnp.where(has_sh, jla, lin_m.jla)
    jlba_out = jnp.where(has_sh, jlba, lin_m.jlba)
    bs_out = jnp.where(has_sh, b_s, lin_m.b_s)
    sb_out = jnp.where(has_sh, sqrt_b, lin_m.sqrt_b)

    solver_shifted = replace(lin_m.solver, jl=jl_out, jlb=jlb_out, jla=jla_out, kperp2=kp2)
    return replace(
        lin_m,
        solver=solver_shifted,
        jl=jl_out,
        jlb=jlb_out,
        jla=jla_out,
        jlba=jlba_out,
        b_s=bs_out,
        sqrt_b=sb_out,
        kperp2=kp2,
        omegad=omegad,
        cv_d=cv_d,
        gb_d=gb_d,
    )


def _apply_forcing_member(
    G: jax.Array,
    op_m: BatchedEnsembleOperators,
    dt_r: jax.Array,
    key: jax.Array,
    step_idx: jax.Array,
) -> jax.Array:
    if not op_m.has_any_forcing or op_m.forcing_op is None:
        return G
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    ns, nm, nl = op_m.nspecies, op_m.nhermite, op_m.nlaguerre
    k1, k2 = jax.random.split(key)
    u1 = jnp.clip(jax.random.uniform(k1, (ns,), dtype=r_dtype), 1.0e-12, 1.0)
    phase = jnp.asarray(math.pi, dtype=r_dtype) * (
        2.0 * jax.random.uniform(k2, (ns,), dtype=r_dtype) - 1.0
    )
    f_amp_r = op_m.forcing_amp_b.astype(r_dtype)
    amp = _safe_sqrt(jnp.abs(f_amp_r * dt_r * jnp.log(u1)))

    amp_r = amp.astype(r_dtype)
    phase_r = phase.astype(r_dtype)
    ftype = op_m.forcing_type.strip().lower()
    if ftype == "kz":
        rf = (amp_r * jnp.cos(phase_r)).astype(c_dtype)
    elif ftype == "kzimpulse":
        rf_raw = jax.lax.complex(amp_r * jnp.cos(phase_r), amp_r * jnp.sin(phase_r)).astype(c_dtype)
        rf = jnp.where(step_idx == 0, rf_raw, jnp.zeros((ns,), dtype=c_dtype))
    else:
        rf = jax.lax.complex(amp_r * jnp.cos(phase_r), amp_r * jnp.sin(phase_r)).astype(c_dtype)
    rf = jnp.where(f_amp_r != 0.0, rf, jnp.zeros_like(rf))

    iz = op_m.forcing_target_zyx_b[0]
    iy = op_m.forcing_target_zyx_b[1]
    ix = op_m.forcing_target_zyx_b[2]
    sfield = op_m.stir_field
    s2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
    s6 = jnp.asarray(math.sqrt(6.0), dtype=r_dtype)

    if sfield in ("density", "ppar", "pperp"):
        g_c = g_c.at[:, 0, 0, iz, iy, ix].add(rf)
    if sfield == "upar" and nm >= 2:
        g_c = g_c.at[:, 1, 0, iz, iy, ix].add(rf)
    if sfield in ("tpar", "ppar") and nm >= 3:
        g_c = g_c.at[:, 2, 0, iz, iy, ix].add(s2 * rf)
    if sfield in ("tperp", "pperp") and nl >= 2:
        g_c = g_c.at[:, 0, 1, iz, iy, ix].add(rf)
    if sfield == "qpar" and nm >= 4:
        g_c = g_c.at[:, 3, 0, iz, iy, ix].add(s6 * rf)
    if sfield == "qperp" and nm >= 2 and nl >= 2:
        g_c = g_c.at[:, 1, 1, iz, iy, ix].add(s2 * rf)
    return g_c


def _cfl_dt_member(
    fields: Fields,
    op_m: BatchedEnsembleOperators,
    exb_state: ExBShearState | None,
) -> jax.Array:
    """Compute per-member adaptive CFL or fixed `dt` in `float64`.

    NOTE: Mirrors `gx_jax.integrator.compute_cfl_dt` using array-valued `BatchedEnsembleOperators`
    leaves; keep CFL scaling, nonlinear `vmax_x`/`vmax_y` maxima, and `k10`/clamp rules in sync
    with `gx_jax/integrator.py`.
    """
    sk = op_m.scheme
    _, r_dtype = _complex_and_real_dtypes(fields.phi)
    w_lin = op_m.omega_linear_b.astype(r_dtype)
    w0, w1, w2 = w_lin[0], w_lin[1], w_lin[2]

    if op_m.nonlinear_mode and op_m.nonlinear_op is not None:
        nl_op = op_m.nonlinear_op
        gp = nl_op.grad_perp
        kxstar = exb_state.kxstar if exb_state is not None else None
        vpar_max = (op_m.vpar_max_b * op_m.vtmax_b).astype(r_dtype)
        mub_max = (op_m.mub_maxj_b * op_m.tzmax_b).astype(r_dtype)

        vy_grid = jnp.abs(gp.dx_c2r_nxny(fields.phi, kxstar=kxstar))
        if op_m.has_any_apar:
            vy_ap = vpar_max * jnp.abs(gp.dx_c2r_nxny(fields.apar, kxstar=kxstar))
            vy_grid = vy_grid + jnp.where(op_m.fapar_b > 0.0, vy_ap, jnp.zeros_like(vy_ap))
        if op_m.has_any_bpar:
            vy_bp = mub_max * jnp.abs(gp.dx_c2r_nxny(fields.bpar, kxstar=kxstar))
            vy_grid = vy_grid + jnp.where(op_m.fbpar_b > 0.0, vy_bp, jnp.zeros_like(vy_bp))
        vmax_y = jnp.max(vy_grid)

        vx_grid = jnp.abs(gp.dy_c2r_nxny(fields.phi))
        if op_m.has_any_apar:
            vx_ap = vpar_max * jnp.abs(gp.dy_c2r_nxny(fields.apar))
            vx_grid = vx_grid + jnp.where(op_m.fapar_b > 0.0, vx_ap, jnp.zeros_like(vx_ap))
        if op_m.has_any_bpar:
            vx_bp = mub_max * jnp.abs(gp.dy_c2r_nxny(fields.bpar))
            vx_grid = vx_grid + jnp.where(op_m.fbpar_b > 0.0, vx_bp, jnp.zeros_like(vx_bp))
        vmax_x = jnp.max(vx_grid)

        scale = jnp.asarray(0.5, dtype=r_dtype) * jnp.abs(op_m.kxfac_b.astype(r_dtype))
        w0 = jnp.maximum(w0, scale * op_m.kx_max_b.astype(r_dtype) * vmax_x)
        w1 = jnp.maximum(w1, scale * op_m.ky_max_b.astype(r_dtype) * vmax_y)

    t_dtype = _time_dtype(r_dtype)
    wmax = w0.astype(t_dtype) + w1.astype(t_dtype) + w2.astype(t_dtype)
    pos = wmax > 0.0
    safe_wmax = jnp.where(pos, wmax, jnp.ones_like(wmax))
    cfl_fac = jnp.asarray(op_m.cfl_fac, dtype=t_dtype)
    dt_guess = (cfl_fac * op_m.cfl_b.astype(t_dtype)) / safe_wmax
    dt_min = op_m.dt_min_b.astype(t_dtype)
    dt_max = op_m.dt_max_b.astype(t_dtype)
    if sk == "k10":
        dt_clamped = jnp.minimum(dt_guess, dt_max)
    else:
        dt_clamped = jnp.minimum(jnp.maximum(dt_guess, dt_min), dt_max)
    dt_cfl = jnp.where(pos, dt_clamped, dt_max).astype(t_dtype)

    if _scheme_honors_fixed_dt(sk):
        return jnp.where(op_m.fixed_dt_b, op_m.dt_b.astype(t_dtype), dt_cfl)
    return dt_cfl


def _gyrokinetic_rhs_member(
    G: jax.Array,
    fields: Fields,
    op_m: BatchedEnsembleOperators,
    dt_r: jax.Array,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias_linear: bool = False,
    apply_kz_dealias_nonlinear: bool = False,
) -> jax.Array:
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    f_c = fields.astype(c_dtype)
    dt_safe = jnp.where(dt_r > 0.0, dt_r, jnp.ones_like(dt_r))

    lin_op = _prepare_member_linear_op(op_m, exb_state, r_dtype)
    gp = lin_op.grad_par
    rhs_lin = linear_rhs(g_c, f_c, lin_op, dt_safe, grids=op_m.grids, geo=op_m.geo)
    if apply_kz_dealias_linear:
        rhs_lin = gp.dealias(rhs_lin)

    if op_m.nonlinear_mode and op_m.nonlinear_op is not None:
        nl_m = replace(op_m.nonlinear_op, kxfac=op_m.kxfac_b.astype(r_dtype))
        f_scaled = Fields(
            phi=f_c.phi * op_m.fphi_b.astype(r_dtype),
            apar=f_c.apar * op_m.fapar_b.astype(r_dtype),
            bpar=f_c.bpar * op_m.fbpar_b.astype(r_dtype),
            phi_avg=f_c.phi_avg,
        )
        rhs_nl = nonlinear_rhs(g_c, f_scaled, nl_m, exb_state=exb_state)
        if apply_kz_dealias_nonlinear:
            rhs_nl = gp.dealias(rhs_nl)
        total = rhs_nl + rhs_lin
    else:
        total = rhs_lin

    mask6 = lin_op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, total, jnp.zeros_like(total))


def _solve_stage_fields_member(
    G: jax.Array,
    op_m: BatchedEnsembleOperators,
    *,
    exb_state: ExBShearState | None = None,
    apply_kz_dealias: bool = False,
) -> Fields:
    _, r_dtype = _complex_and_real_dtypes(G)
    lin_op = _prepare_member_linear_op(op_m, exb_state, r_dtype)
    f_out = _solve_fields_member(G, op_m, solver_override=lin_op.solver)
    if apply_kz_dealias and op_m.dealias_kz:
        phi_d = lin_op.grad_par.dealias(f_out.phi)
        f_out = Fields(phi=phi_d, apar=f_out.apar, bpar=f_out.bpar, phi_avg=f_out.phi_avg)
    return f_out


def _step_gyrokinetic_member(
    G: jax.Array,
    fields: Fields,
    op_m: BatchedEnsembleOperators,
    dt_f64: jax.Array,
    *,
    exb_state: ExBShearState | None = None,
    enforce_reality_cond: bool = False,
    key: jax.Array,
    step_idx: jax.Array,
) -> StepResult:
    """Advance a single ensemble member by one step (`rk3`, `rk4`, `sspx2`, `sspx3`, or `k10`).

    NOTE: Mirrors `gx_jax.integrator.step_gyrokinetic` on array-leaf `BatchedEnsembleOperators`
    so `jax.vmap` traces once across varying scalar parameters; keep stage weights, ExB shifts,
    dealiasing, and forcing order in sync with `gx_jax/integrator.py`.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g0 = G.astype(c_dtype)
    f0 = fields.astype(c_dtype)
    sk = op_m.scheme
    dealias_kz = op_m.dealias_kz
    exb_cur = exb_state.astype(r_dtype) if exb_state is not None else None
    dt_r = dt_f64.astype(r_dtype)

    if exb_cur is not None and sk in ("rk3", "sspx2", "sspx3", "k10"):
        exb_cur, f0 = _flow_shear_shift_member(exb_cur, f0, op_m, dt_r)
        g0 = _shift_active_kx_member(g0, exb_cur, op_m.grids, op_m.g_exb_b.astype(r_dtype))

    def _rhs(
        g_in: jax.Array,
        f_in: Fields,
        exb_in: ExBShearState | None,
        *,
        d_lin: bool = False,
        d_nl: bool = False,
    ) -> jax.Array:
        return _gyrokinetic_rhs_member(
            g_in,
            f_in,
            op_m,
            dt_r,
            exb_state=exb_in,
            apply_kz_dealias_linear=d_lin,
            apply_kz_dealias_nonlinear=d_nl,
        )

    if sk == "rk4":
        half = jnp.asarray(0.5, dtype=r_dtype)
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        two = jnp.asarray(2.0, dtype=r_dtype)

        k1 = _rhs(g0, f0, exb_cur)
        g1 = g0 + (half * dt_r) * k1
        f1 = _solve_stage_fields_member(g1, op_m, exb_state=exb_cur)

        if exb_cur is not None:
            g_exb_r = op_m.g_exb_b.astype(r_dtype)
            exb_cur, f1 = _flow_shear_shift_member(exb_cur, f1, op_m, half * dt_r)
            g0 = _shift_active_kx_member(g0, exb_cur, op_m.grids, g_exb_r)
            k1 = _shift_active_kx_member(k1, exb_cur, op_m.grids, g_exb_r)
            g1 = _shift_active_kx_member(g1, exb_cur, op_m.grids, g_exb_r)

        k2 = _rhs(g1, f1, exb_cur)
        g2 = g0 + (half * dt_r) * k2
        f2 = _solve_stage_fields_member(g2, op_m, exb_state=exb_cur)

        k3 = _rhs(g2, f2, exb_cur)
        g3 = g0 + dt_r * k3
        f3 = _solve_stage_fields_member(g3, op_m, exb_state=exb_cur)

        if exb_cur is not None:
            g_exb_r = op_m.g_exb_b.astype(r_dtype)
            exb_cur, f3 = _flow_shear_shift_member(exb_cur, f3, op_m, half * dt_r)
            g0 = _shift_active_kx_member(g0, exb_cur, op_m.grids, g_exb_r)
            k1 = _shift_active_kx_member(k1, exb_cur, op_m.grids, g_exb_r)
            k2 = _shift_active_kx_member(k2, exb_cur, op_m.grids, g_exb_r)
            k3 = _shift_active_kx_member(k3, exb_cur, op_m.grids, g_exb_r)
            g3 = _shift_active_kx_member(g3, exb_cur, op_m.grids, g_exb_r)

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
        f_q1 = _solve_stage_fields_member(g_q1, op_m, exb_state=exb_cur)

        k2 = _rhs(g_q1, f_q1, exb_cur)
        g_q2 = g0 + (c23 * dt_r) * k2
        f_q2 = _solve_stage_fields_member(g_q2, op_m, exb_state=exb_cur)

        k3 = _rhs(g_q2, f_q2, exb_cur)
        g_next = g0 + (c34 * dt_r) * k3 + (c14 * dt_r) * k1
        apply_final_kz_dealias = False

    elif sk == "sspx2":
        adt = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        c_d0 = jnp.asarray(math.sqrt(2.0) - 1.0, dtype=r_dtype)

        d0 = (adt * dt_r) * _rhs(g0, f0, exb_cur)
        g1 = g0 + d0
        f1 = _solve_stage_fields_member(g1, op_m, exb_state=exb_cur)

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
        f1 = _solve_stage_fields_member(g1, op_m, exb_state=exb_cur, apply_kz_dealias=True)

        d1 = (adt * dt_r) * _rhs(g1, f1, exb_cur, d_lin=dealias_kz, d_nl=False)
        d2 = w1 * d0 + d1
        g2 = g0 + d2
        f2 = _solve_stage_fields_member(g2, op_m, exb_state=exb_cur, apply_kz_dealias=True)

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
            fq_next = _solve_stage_fields_member(
                gq_next, op_m, exb_state=exb_cur, apply_kz_dealias=True
            )
            return dq_next, gq_next, fq_next

        d_acc = jnp.zeros_like(g0)
        g_q1, f_q1 = g0, f0
        for _ in range(5):
            d_acc, g_q1, f_q1 = _euler_k10((d_acc, g_q1, f_q1))
        d5 = d_acc

        d_acc = jnp.asarray(0.4, dtype=r_dtype) * d5
        g_q1 = g0 + d_acc
        f_q1 = _solve_stage_fields_member(g_q1, op_m, exb_state=exb_cur, apply_kz_dealias=True)

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

    if op_m.has_any_forcing and op_m.forcing_op is not None:
        g_next = _apply_forcing_member(g_next, op_m, dt_r, key, step_idx)

    g_next = mask_g(g_next, op_m.grids)
    if enforce_reality_cond:
        g_next = enforce_reality(g_next, op_m.grids)

    f_next = _solve_stage_fields_member(
        g_next, op_m, exb_state=exb_cur, apply_kz_dealias=apply_final_kz_dealias
    )
    t_dtype = _time_dtype(r_dtype)
    return StepResult(
        G=g_next.astype(c_dtype),
        fields=f_next.astype(c_dtype),
        dt=dt_f64.astype(t_dtype),
        exb_state=exb_cur,
    )


def _step_integrator_state_member(
    state: IntegratorState,
    op_m: BatchedEnsembleOperators,
    *,
    dt_override: float | None = None,
    enforce_reality_cond: bool = False,
) -> IntegratorState:
    c_dtype, r_dtype = _complex_and_real_dtypes(state.G)
    t_dtype = _time_dtype(r_dtype)
    exb_in = state.exb_state.astype(r_dtype) if state.exb_state is not None else None

    if dt_override is not None:
        dt_step = jnp.asarray(dt_override, dtype=t_dtype)
    elif _scheme_honors_fixed_dt(op_m.scheme):
        dt_cfl = _cfl_dt_member(state.fields, op_m, exb_in)
        dt_step = jnp.where(op_m.fixed_dt_b, state.dt.astype(t_dtype), dt_cfl)
    else:
        dt_step = _cfl_dt_member(state.fields, op_m, exb_in)

    next_key, step_key = jax.random.split(state.key)
    g_next, f_next, dt_used, exb_next = _step_gyrokinetic_member(
        state.G,
        state.fields,
        op_m,
        dt_step,
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
# Public Batched `jax.vmap` Kernels & Fused On-Device `while_loop` Chunk Driver
# ---------------------------------------------------------------------------


def batched_solve_fields(
    g_batch: jax.Array,
    batch_ops: BatchedEnsembleOperators,
    *,
    exb_state: ExBShearState | None = None,
) -> Fields:
    """Solve quasineutrality & Ampère's laws across batch axis `0` via `jax.vmap`."""
    exb_axes = 0 if exb_state is not None else None
    return jax.vmap(
        lambda g_m, op_m, exb_m: _solve_stage_fields_member(
            g_m, op_m, exb_state=exb_m, apply_kz_dealias=False
        ),
        in_axes=(0, 0, exb_axes),
    )(g_batch, batch_ops, exb_state)


def batched_gyrokinetic_rhs(
    g_batch: jax.Array,
    fields_batch: Fields,
    batch_ops: BatchedEnsembleOperators,
    dt_batch: jax.Array | float,
    *,
    exb_state: ExBShearState | None = None,
) -> jax.Array:
    """Evaluate unified linear + nonlinear RHS across batch axis `0` via `jax.vmap`."""
    _, r_dtype = _complex_and_real_dtypes(g_batch)
    dt_vec = jnp.broadcast_to(jnp.asarray(dt_batch, dtype=r_dtype), (batch_ops.batch_size,))
    exb_axes = 0 if exb_state is not None else None
    return jax.vmap(
        lambda g_m, f_m, op_m, dt_m, exb_m: _gyrokinetic_rhs_member(
            g_m, f_m, op_m, dt_m, exb_state=exb_m
        ),
        in_axes=(0, 0, 0, 0, exb_axes),
    )(g_batch, fields_batch, batch_ops, dt_vec, exb_state)


def init_batched_integrator_state(
    g_batch: jax.Array,
    batch_ops: BatchedEnsembleOperators,
    *,
    time: jax.Array | Sequence[float] | float = 0.0,
    dt_override: float | None = None,
    enforce_reality_cond: bool = False,
    keys: jax.Array | None = None,
    mesh: Mesh | None = None,
) -> IntegratorState:
    """Create initial `B`-batched `IntegratorState` (optionally sharded on `mesh`)."""
    b_sz = batch_ops.batch_size
    c_dtype, r_dtype = _complex_and_real_dtypes(g_batch)
    t_dtype = _time_dtype(r_dtype)
    t_vec = jnp.broadcast_to(jnp.asarray(time, dtype=t_dtype), (b_sz,))
    exb_init = batch_ops.exb_init.astype(r_dtype) if batch_ops.exb_init is not None else None
    keys_init = jax.vmap(jax.random.PRNGKey)(batch_ops.random_seed_b) if keys is None else keys

    def _init_one(
        g_m: jax.Array,
        op_m: BatchedEnsembleOperators,
        t_m: jax.Array,
        exb_m: ExBShearState | None,
        k_m: jax.Array,
    ) -> IntegratorState:
        g0 = mask_g(g_m.astype(c_dtype), op_m.grids)
        if enforce_reality_cond:
            g0 = enforce_reality(g0, op_m.grids)
        f0 = _solve_stage_fields_member(g0, op_m, exb_state=exb_m, apply_kz_dealias=False)
        if dt_override is not None:
            dt0 = jnp.asarray(dt_override, dtype=t_dtype)
        else:
            dt_cfl0 = _cfl_dt_member(f0, op_m, exb_m)
            dt0 = jnp.where(op_m.fixed_dt_b, op_m.dt_b.astype(t_dtype), dt_cfl0)
        return IntegratorState(
            G=g0.astype(c_dtype),
            fields=f0.astype(c_dtype),
            time=t_m.astype(t_dtype),
            dt=dt0.astype(t_dtype),
            step=jnp.asarray(0, dtype=jnp.int32),
            exb_state=exb_m,
            key=k_m,
        )

    exb_ax = 0 if exb_init is not None else None
    state = jax.vmap(_init_one, in_axes=(0, 0, 0, exb_ax, 0))(
        g_batch, batch_ops, t_vec, exb_init, keys_init
    )
    return shard_ensemble_tree(state, mesh, batch_size=b_sz) if mesh is not None else state


def batched_step_integrator_state(
    state: IntegratorState,
    batch_ops: BatchedEnsembleOperators,
    *,
    dt_override: float | None = None,
    enforce_reality_cond: bool = False,
) -> IntegratorState:
    """Advance active members of batched `state` by 1 step via `jax.vmap`, freezing inactive."""

    def _step_masked(st_m: IntegratorState, op_m: BatchedEnsembleOperators) -> IntegratorState:
        is_active = (st_m.step < op_m.nstep_b) & (st_m.time < (op_m.t_max_b - op_m.eps_t_b))
        st_cand = _step_integrator_state_member(
            st_m,
            op_m,
            dt_override=dt_override,
            enforce_reality_cond=enforce_reality_cond,
        )
        return jax.tree_util.tree_map(
            lambda new_v, old_v: jnp.where(is_active, new_v, old_v),
            st_cand,
            st_m,
        )

    return jax.vmap(_step_masked, in_axes=(0, 0))(state, batch_ops)


def advance_ensemble_chunk(
    state: IntegratorState,
    diag_state: DiagnosticsState,
    batch_ops: BatchedEnsembleOperators,
    max_chunk_steps: int | jax.Array,
    *,
    dt_override: float | None = None,
    enforce_reality_cond: bool = False,
) -> EnsembleChunkCarry:
    """Advance up to `max_chunk_steps` inside one compiled `jax.lax.while_loop` + `jax.vmap`.

    Per-member masking freezes `(G, fields, time, dt, step, exb_state, key)` AND penultimate
    `DiagnosticsState` `(G_old, fields_old, time_old, has_old)` via `jnp.where` as soon as a
    member reaches `nstep_b`, `t_max_b - eps_t_b`, or triggers a `pending_write` snapshot.
    """
    max_steps_i32 = jnp.asarray(max_chunk_steps, dtype=jnp.int32)
    init_carry = EnsembleChunkCarry(
        state=state,
        diag_state=diag_state,
        pending_write=jnp.zeros((batch_ops.batch_size,), dtype=jnp.bool_),
        steps_in_chunk=jnp.asarray(0, dtype=jnp.int32),
    )

    def _cond(carry: EnsembleChunkCarry) -> jax.Array:
        active_b = (
            (carry.state.step < batch_ops.nstep_b)
            & (carry.state.time < (batch_ops.t_max_b - batch_ops.eps_t_b))
            & (~carry.pending_write)
        )
        return (carry.steps_in_chunk < max_steps_i32) & jnp.any(active_b)

    def _member_body(
        st_m: IntegratorState,
        dg_m: DiagnosticsState,
        pw_m: jax.Array,
        op_m: BatchedEnsembleOperators,
    ) -> tuple[IntegratorState, DiagnosticsState, jax.Array]:
        c0 = st_m.step
        cur_t = st_m.time
        cur_dt = st_m.dt
        is_active = (c0 < op_m.nstep_b) & (cur_t < (op_m.t_max_b - op_m.eps_t_b)) & (~pw_m)

        st_cand = _step_integrator_state_member(
            st_m,
            op_m,
            dt_override=dt_override,
            enforce_reality_cond=enforce_reality_cond,
        )
        c1 = st_cand.step
        end_t = st_cand.time

        overshoot_write = end_t > (op_m.t_max_b + op_m.eps_t_b)
        should_save_old = (jnp.mod(c0, op_m.nwrite_b) == 0) | (
            (cur_t + cur_dt) > (op_m.t_max_b - op_m.eps_t_b)
        )
        dg_saved = DiagnosticsState(
            G_old=st_m.G,
            fields_old=st_m.fields,
            time_old=st_m.time,
            has_old=jnp.asarray(True, dtype=jnp.bool_),
        )
        dg_pre = jax.tree_util.tree_map(
            lambda s_v, o_v: jnp.where(should_save_old, s_v, o_v),
            dg_saved,
            dg_m,
        )

        cadence_write = (c1 >= 1) & (jnp.mod(c1 - 1, op_m.nwrite_b) == 0)
        will_write = cadence_write | overshoot_write

        st_out = jax.tree_util.tree_map(
            lambda new_v, old_v: jnp.where(is_active, new_v, old_v),
            st_cand,
            st_m,
        )
        dg_out = jax.tree_util.tree_map(
            lambda new_d, old_d: jnp.where(is_active, new_d, old_d),
            dg_pre,
            dg_m,
        )
        pw_out = jnp.where(is_active, will_write, pw_m)
        return st_out, dg_out, pw_out

    def _body(carry: EnsembleChunkCarry) -> EnsembleChunkCarry:
        st_next, dg_next, pw_next = jax.vmap(_member_body, in_axes=(0, 0, 0, 0))(
            carry.state,
            carry.diag_state,
            carry.pending_write,
            batch_ops,
        )
        return EnsembleChunkCarry(
            state=st_next,
            diag_state=dg_next,
            pending_write=pw_next,
            steps_in_chunk=(carry.steps_in_chunk + jnp.int32(1)).astype(jnp.int32),
        )

    return jax.lax.while_loop(_cond, _body, init_carry)


def _diagnostics_step_member(
    G: jax.Array,
    fields: Fields,
    diag_state: DiagnosticsState,
    op_m: BatchedEnsembleOperators,
    time_val: jax.Array,
    dt_val: jax.Array,
) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
    t_dtype = _time_dtype(jnp.real(G).dtype)
    t_arr = jnp.asarray(time_val, dtype=t_dtype)
    dt_arr = jnp.asarray(dt_val, dtype=t_dtype)
    grids = op_m.grids
    geo = op_m.geo
    solver = op_m.solver

    phi2 = compute_field_squared_spectra(fields.phi, grids, geo)
    phi2_zonal = compute_phi2_zonal_spectra(fields.phi, grids, geo)
    apar2 = compute_field_squared_spectra(fields.apar, grids, geo)
    bpar2 = compute_field_squared_spectra(fields.bpar, grids, geo)

    wg = compute_wg_spectra(G, grids, geo, solver)
    wphi = compute_wphi_spectra(
        fields.phi, grids, geo, solver, phi_avg=fields.phi_avg, tau_fac=op_m.tau_fac_b
    )
    wapar = compute_wapar_spectra(fields.apar, grids, geo, op_m.nspecies)
    wbpar = compute_wbpar_spectra(fields.bpar, grids, geo, op_m.nspecies, beta=op_m.beta_b)

    heat_flux = compute_heat_flux_spectra(G, fields, grids, geo, solver)
    particle_flux = compute_particle_flux_spectra(G, fields, grids, geo, solver)
    turbulent_heating = compute_turbulent_heating_spectra(
        G,
        fields,
        diag_state.G_old,
        diag_state.fields_old,
        dt_arr,
        grids,
        geo,
        solver,
    )
    omega_kxkyt = compute_omega_kxky(fields.phi, diag_state.fields_old.phi, dt_arr, grids)

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
        heat_flux=heat_flux,
        particle_flux=particle_flux,
        turbulent_heating=turbulent_heating,
        omega_kxkyt=omega_kxkyt,
    )
    next_state = DiagnosticsState(
        G_old=G,
        fields_old=fields,
        time_old=t_arr,
        has_old=jnp.asarray(True, dtype=jnp.bool_),
    )
    return snapshot, next_state


def batched_compute_diagnostics_step(
    g_batch: jax.Array,
    fields_batch: Fields,
    diag_state: DiagnosticsState,
    batch_ops: BatchedEnsembleOperators,
    *,
    time: jax.Array,
    dt: jax.Array,
) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
    """Compute full spectral `DiagnosticsSnapshot` across batch axis `0` via `jax.vmap`."""
    t_dtype = _time_dtype(jnp.real(g_batch).dtype)
    return jax.vmap(_diagnostics_step_member, in_axes=(0, 0, 0, 0, 0, 0))(
        g_batch,
        fields_batch,
        diag_state,
        batch_ops,
        jnp.asarray(time, dtype=t_dtype),
        jnp.asarray(dt, dtype=t_dtype),
    )


def make_ensemble_chunk_stepper(
    batch_ops: BatchedEnsembleOperators,
    *,
    mesh: Mesh | None = None,
    dt_override: float | None = None,
    enforce_reality_cond: bool = False,
) -> Callable[[IntegratorState, DiagnosticsState, jax.Array], EnsembleChunkCarry]:
    """Compile a single-trace (`trace_count == 1`) chunk stepper (optionally `'batch'`-sharded)."""
    b_sz = batch_ops.batch_size
    ops_placed = (
        shard_ensemble_tree(batch_ops, mesh, batch_size=b_sz) if mesh is not None else batch_ops
    )

    if is_distributed():
        # Multi-host: `jax.jit` cannot close over arrays that span non-addressable
        # devices, so the `'batch'`-sharded operators are an explicit jit argument.
        def _run_with_ops(
            st: IntegratorState,
            dg: DiagnosticsState,
            max_steps: jax.Array,
            ops: BatchedEnsembleOperators,
        ) -> EnsembleChunkCarry:
            return advance_ensemble_chunk(
                st,
                dg,
                ops,
                max_steps,
                dt_override=dt_override,
                enforce_reality_cond=enforce_reality_cond,
            )

        run_with_ops_jit = jax.jit(_run_with_ops)

        def _run_multihost(
            st: IntegratorState, dg: DiagnosticsState, max_steps: jax.Array
        ) -> EnsembleChunkCarry:
            return run_with_ops_jit(st, dg, max_steps, ops_placed)

        return _run_multihost

    def _run(st: IntegratorState, dg: DiagnosticsState, max_steps: jax.Array) -> EnsembleChunkCarry:
        return advance_ensemble_chunk(
            st,
            dg,
            ops_placed,
            max_steps,
            dt_override=dt_override,
            enforce_reality_cond=enforce_reality_cond,
        )

    return jax.jit(_run)


# ---------------------------------------------------------------------------
# End-to-End KR2 Ensemble Campaign Driver & `gx-jax-ensemble` CLI
# ---------------------------------------------------------------------------


def _atomic_save_member_restart_nc(
    path: Path,
    g_state: Any,
    time_val: float,
    grids: Grids,
    member_idx: int = 0,
) -> None:
    """Atomically write `<stem>.restart.nc` via a hidden temp file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{member_idx}.tmp")
    save_restart_nc(tmp_path, jnp.asarray(g_state), float(time_val), grids)
    os.replace(tmp_path, path)


def _index_batch_host_tree(tree: Any, idx: int) -> Any:
    """Extract member `idx` from a host-transferred `B`-batched PyTree."""
    return jax.tree_util.tree_map(lambda x: x[idx] if x is not None else None, tree)


def _resolve_ensemble_output_paths(
    specs: Sequence[EnsembleMemberSpec],
    *,
    output_dir: Path | None,
    summary_json_path: Path | None,
    save_restart: bool | None,
) -> tuple[list[Path], list[Path | None], list[Path], Path]:
    """Resolve collision-free `.out.nc`, `.restart.nc`, `.diagnostics_metrics.json`, & summary."""
    raw_out_paths: list[Path] = []
    stems: list[str] = []
    for idx, sp in enumerate(specs):
        stem = sp.name or sp.cfg.run_name or f"member_{idx:03d}"
        stems.append(stem)
        if output_dir is not None:
            raw_out_paths.append(output_dir / f"{stem}.out.nc")
        elif sp.input_path is not None:
            raw_out_paths.append(sp.input_path.with_name(f"{stem}.out.nc"))
        else:
            raise ValueError(
                f"output_dir is required when member {idx} ({stem!r}) has no input_path."
            )

    seen_counts: dict[Path, int] = {}
    for p in raw_out_paths:
        seen_counts[p] = seen_counts.get(p, 0) + 1

    out_nc_paths: list[Path] = []
    restart_paths: list[Path | None] = []
    metrics_paths: list[Path] = []

    for idx, (sp, stem, raw_p) in enumerate(zip(specs, stems, raw_out_paths, strict=True)):
        eff_stem = f"member_{idx:03d}_{stem}" if seen_counts[raw_p] > 1 else stem
        out_p = raw_p.with_name(f"{eff_stem}.out.nc")
        out_nc_paths.append(out_p)
        metrics_paths.append(out_p.with_name(f"{eff_stem}.diagnostics_metrics.json"))

        want_restart = sp.cfg.save_for_restart if save_restart is None else bool(save_restart)
        if not want_restart:
            restart_paths.append(None)
        elif output_dir is not None or seen_counts[raw_p] > 1:
            restart_paths.append(out_p.with_name(f"{eff_stem}.restart.nc"))
        elif sp.cfg.restart_to_file:
            cand = Path(sp.cfg.restart_to_file)
            if not cand.is_absolute():
                cand = out_p.parent / cand
            restart_paths.append(cand)
        else:
            restart_paths.append(out_p.with_name(f"{eff_stem}.restart.nc"))

    if summary_json_path is not None:
        summary_p = summary_json_path
    elif output_dir is not None:
        summary_p = output_dir / "ensemble_summary.json"
    else:
        summary_p = out_nc_paths[0].parent / "ensemble_summary.json"

    return out_nc_paths, restart_paths, metrics_paths, summary_p


def _resolve_member_initial_condition(
    sp: EnsembleMemberSpec,
    *,
    complex_dtype: jnp.dtype,
) -> tuple[jax.Array, float, bool]:
    """Initialize member `G` via `init_moments` or load `.restart.nc` matching `cli.py`."""
    cfg, grids, geo = sp.cfg, sp.grids, sp.geo
    restart_src: Path | None = None
    if cfg.restart_from_file:
        cand = Path(cfg.restart_from_file)
        if not cand.is_absolute() and sp.input_path is not None:
            cand = sp.input_path.parent / cand
        if cfg.restart_if_exists:
            if cand.is_file():
                restart_src = cand
        elif cfg.restart:
            restart_src = cand

    if restart_src is not None:
        if not restart_src.is_file():
            raise FileNotFoundError(f"Restart file does not exist: {restart_src}")
        g_loaded, t_loaded = load_restart_nc(
            restart_src,
            grids,
            scale=float(cfg.restart_scale),
            dtype=complex_dtype,
        )
        if cfg.restart_with_perturb:
            g_init = init_moments(cfg, grids, geo, dtype=complex_dtype) + g_loaded
        else:
            g_init = g_loaded
        return g_init, float(t_loaded), True

    return init_moments(cfg, grids, geo, dtype=complex_dtype), 0.0, False


def run_ensemble(
    inputs: Sequence[EnsembleInputItem],
    *,
    output_dir: str | Path | None = None,
    summary_json_path: str | Path | None = None,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    precision: str = "fp32",
    fft_backend: str | None = None,
    batch_devices: int | None = None,
    mesh: Mesh | None = None,
    save_restart: bool | None = None,
    g_init_batch: jax.Array | Sequence[jax.Array] | None = None,
) -> dict[str, Any]:
    """Execute a `B`-member gyrokinetic parameter-sweep campaign via `jax.vmap`."""
    configure_tpu_precision()
    prec_clean = precision.strip().lower()
    if prec_clean not in ("fp32", "fp64"):
        raise ValueError(f"Invalid precision={precision!r}; expected 'fp32' or 'fp64'.")
    use_fp64 = prec_clean == "fp64"
    real_dtype = jnp.float64 if use_fp64 else jnp.float32
    complex_dtype = jnp.complex128 if use_fp64 else jnp.complex64

    specs, batch_ops = init_batched_ensemble(
        inputs,
        dtype=real_dtype,
        nstep=nstep,
        dt=dt,
        t_max=t_max,
        nwrite=nwrite,
        scheme=scheme,
        force_nonlinear=force_nonlinear,
        force_linear=force_linear,
        fft_backend=fft_backend,
    )
    b_sz = batch_ops.batch_size

    out_dir_path = Path(output_dir) if output_dir is not None else None
    sum_path_in = Path(summary_json_path) if summary_json_path is not None else None
    out_nc_paths, restart_paths, metrics_paths, resolved_summary_path = (
        _resolve_ensemble_output_paths(
            specs,
            output_dir=out_dir_path,
            summary_json_path=sum_path_in,
            save_restart=save_restart,
        )
    )

    if mesh is None:
        mesh = create_ensemble_mesh(b_sz, num_devices=batch_devices)
    elif mesh.axis_names != ("batch",):
        raise ValueError(f"Expected 1D Mesh with axis_names=('batch',), got {mesh.axis_names}.")
    elif b_sz % int(mesh.shape["batch"]) != 0:
        raise ValueError(
            f"batch_size={b_sz} is not divisible by mesh batch size {int(mesh.shape['batch'])}."
        )

    # Multi-host (jax.process_count() > 1): every process runs the vmap chunk stepper,
    # diagnostics and gathers in strict lockstep; only the primary process
    # (jax.process_index() == 0) opens or writes .out.nc / .restart.nc / .json files.
    multi_host = is_distributed()
    is_primary = is_primary_process()
    if multi_host:
        validate_multihost_mesh(mesh)

    if g_init_batch is not None:
        if isinstance(g_init_batch, jax.Array):
            if g_init_batch.shape[0] != b_sz:
                raise ValueError(
                    f"Expected g_init_batch.shape[0] == {b_sz}, got {g_init_batch.shape}."
                )
            g_stacked = g_init_batch.astype(complex_dtype)
        else:
            if len(g_init_batch) != b_sz:
                raise ValueError(
                    f"Expected {b_sz} arrays in g_init_batch, got {len(g_init_batch)}."
                )
            g_stacked = jnp.stack(
                [jnp.asarray(g_i, dtype=complex_dtype) for g_i in g_init_batch], axis=0
            )
        t0_list = [0.0] * b_sz
        restarting_list = [False] * b_sz
    else:
        g_list: list[jax.Array] = []
        t0_list = []
        restarting_list = []
        with host_precompute_context():
            for sp in specs:
                g_i, t0_i, is_rest_i = _resolve_member_initial_condition(
                    sp, complex_dtype=complex_dtype
                )
                g_list.append(g_i)
                t0_list.append(t0_i)
                restarting_list.append(is_rest_i)
            g_stacked = jnp.stack(g_list, axis=0)
        g_stacked = move_tree_to_default_device(g_stacked)

    if multi_host:
        assert_same_across_processes(
            (
                tuple(
                    zlib.crc32(sp.input_path.read_bytes()) if sp.input_path is not None else sp.name
                    for sp in specs
                ),
                (tuple(t0_list), tuple(restarting_list)),
                (tuple(g_stacked.shape), str(g_stacked.dtype), prec_clean),
                (nstep, dt, t_max, nwrite, scheme, force_nonlinear, force_linear, fft_backend),
                tuple(int(n) for n in mesh.devices.shape),
            ),
            "ensemble inputs (decks, overrides, restart state, mesh)",
        )
    ops_sh = shard_ensemble_tree(batch_ops, mesh, batch_size=b_sz)
    g_init_sh = shard_ensemble_tree(g_stacked, mesh, batch_size=b_sz)

    t_wall_start = time.perf_counter()

    state = init_batched_integrator_state(
        g_init_sh,
        ops_sh,
        time=t0_list,
        dt_override=dt,
        mesh=mesh,
    )

    @jax.jit
    def _init_batched_diag(st: IntegratorState) -> DiagnosticsState:
        return jax.vmap(
            lambda g_m, f_m, t_m: init_diagnostics_state(g_m, f_m, time=t_m, has_old=False),
            in_axes=(0, 0, 0),
        )(st.G, st.fields, st.time)

    diag_state = _init_batched_diag(state)
    chunk_stepper = make_ensemble_chunk_stepper(ops_sh, mesh=mesh, dt_override=dt)

    @jax.jit
    def _diag_and_update(
        g_b: jax.Array,
        f_b: Fields,
        dg_b: DiagnosticsState,
        pw_b: jax.Array,
        t_b: jax.Array,
        dt_b: jax.Array,
    ) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
        snaps, dg_next = batched_compute_diagnostics_step(g_b, f_b, dg_b, ops_sh, time=t_b, dt=dt_b)
        dg_merged = jax.tree_util.tree_map(
            lambda n_v, o_v: jnp.where(pw_b.reshape((b_sz,) + (1,) * (n_v.ndim - 1)), n_v, o_v),
            dg_next,
            dg_b,
        )
        return snaps, dg_merged

    if multi_host:
        # Multi-host: pass the `'batch'`-sharded operators explicitly (`jax.jit` cannot
        # close over arrays that span non-addressable devices).
        @jax.jit
        def _diag_and_update_with_ops(
            g_b: jax.Array,
            f_b: Fields,
            dg_b: DiagnosticsState,
            pw_b: jax.Array,
            t_b: jax.Array,
            dt_b: jax.Array,
            ops_b: BatchedEnsembleOperators,
        ) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
            snaps, dg_next = batched_compute_diagnostics_step(
                g_b, f_b, dg_b, ops_b, time=t_b, dt=dt_b
            )
            dg_merged = jax.tree_util.tree_map(
                lambda n_v, o_v: jnp.where(pw_b.reshape((b_sz,) + (1,) * (n_v.ndim - 1)), n_v, o_v),
                dg_next,
                dg_b,
            )
            return snaps, dg_merged

        def _diag_and_update(  # noqa: F811
            g_b: jax.Array,
            f_b: Fields,
            dg_b: DiagnosticsState,
            pw_b: jax.Array,
            t_b: jax.Array,
            dt_b: jax.Array,
        ) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
            return _diag_and_update_with_ops(g_b, f_b, dg_b, pw_b, t_b, dt_b, ops_sh)

    any_write_moms = any(bool(sp.cfg.write_moms) for sp in specs)
    any_write_fields = any(bool(sp.cfg.write_fields) for sp in specs)
    host_solvers: list[FieldSolver | None]
    if any_write_moms:
        solver_host_tree = gather_global_tree(batch_ops.solver)
        host_solvers = [_index_batch_host_tree(solver_host_tree, b) for b in range(b_sz)]
    else:
        host_solvers = [None] * b_sz

    nstep_lim = np.asarray([max(0, int(sp.cfg.nstep)) for sp in specs], dtype=np.int64)
    t_max_lim = np.asarray([float(sp.cfg.t_max) for sp in specs], dtype=np.float64)
    t_dtype = _time_dtype(real_dtype)
    eps_rel = 1.0e-6 if t_dtype == jnp.float32 else 1.0e-11
    eps_t_lim = np.asarray([eps_rel * max(abs(tm), 1.0e-30) for tm in t_max_lim], dtype=np.float64)
    nwrite_lim = np.asarray([max(1, int(sp.cfg.nwrite)) for sp in specs], dtype=np.int64)
    nsave_lim = np.asarray([max(1, int(sp.cfg.nsave)) for sp in specs], dtype=np.int64)

    step_np, time_np = gather_global_tree((state.step, state.time))
    step_np = np.asarray(step_np, dtype=np.int64)
    time_np = np.asarray(time_np, dtype=np.float64)
    last_saved_step = [-1] * b_sz

    with contextlib.ExitStack() as stack:
        writers: list[DiagnosticsWriter | None] = [
            stack.enter_context(
                DiagnosticsWriter(
                    out_nc_paths[b],
                    specs[b].cfg,
                    specs[b].grids,
                    specs[b].geo,
                    append=bool(specs[b].cfg.append_on_restart and restarting_list[b]),
                )
                if is_primary
                else contextlib.nullcontext(None)
            )
            for b in range(b_sz)
        ]

        while True:
            active_host = (step_np < nstep_lim) & (time_np < (t_max_lim - eps_t_lim))
            if not bool(np.any(active_host)):
                break

            chunk_candidates: list[int] = []
            for b in range(b_sz):
                if not bool(active_host[b]):
                    continue
                cur_s = int(step_np[b])
                rem_nstep = int(nstep_lim[b]) - cur_s
                rem_write = (
                    1 if cur_s == 0 else int(nwrite_lim[b]) - ((cur_s - 1) % int(nwrite_lim[b]))
                )
                cand = min(rem_nstep, rem_write)
                if restart_paths[b] is not None:
                    rem_save = (
                        1 if cur_s == 0 else int(nsave_lim[b]) - ((cur_s - 1) % int(nsave_lim[b]))
                    )
                    cand = min(cand, rem_save)
                chunk_candidates.append(max(1, cand))
            max_chunk_steps = min(chunk_candidates)

            carry_out = chunk_stepper(
                state,
                diag_state,
                jnp.asarray(max_chunk_steps, dtype=jnp.int32),
            )
            state = carry_out.state
            diag_state = carry_out.diag_state

            prev_step_np = step_np
            step_raw, time_raw, pw_raw = gather_global_tree(
                (state.step, state.time, carry_out.pending_write)
            )
            step_np = np.asarray(step_raw, dtype=np.int64)
            time_np = np.asarray(time_raw, dtype=np.float64)
            pw_np = np.asarray(pw_raw, dtype=bool)

            need_periodic_save = [
                (
                    restart_paths[b] is not None
                    and step_np[b] > prev_step_np[b]
                    and (int(step_np[b]) - 1) % int(nsave_lim[b]) == 0
                )
                for b in range(b_sz)
            ]

            need_g_host = (bool(np.any(pw_np)) and any_write_moms) or any(need_periodic_save)
            need_f_host = bool(np.any(pw_np)) and (any_write_fields or any_write_moms)

            # Lockstep host gathers on every process (multi-host: process_allgather).
            g_chunk_host = gather_global_tree(state.G) if need_g_host else None
            f_chunk_host = gather_global_tree(state.fields) if need_f_host else None

            if bool(np.any(pw_np)):
                snap_batch, diag_state = _diag_and_update(
                    state.G,
                    state.fields,
                    diag_state,
                    carry_out.pending_write,
                    state.time,
                    state.dt,
                )
                snap_host = gather_global_tree(snap_batch)
                for b in range(b_sz):
                    writer_b = writers[b]
                    if writer_b is None or not bool(pw_np[b]):
                        continue
                    snap_b = _index_batch_host_tree(snap_host, b)
                    g_b = (
                        jnp.asarray(g_chunk_host[b])
                        if (specs[b].cfg.write_moms and g_chunk_host is not None)
                        else None
                    )
                    f_b = (
                        Fields(
                            phi=jnp.asarray(f_chunk_host.phi[b]),
                            apar=jnp.asarray(f_chunk_host.apar[b]),
                            bpar=jnp.asarray(f_chunk_host.bpar[b]),
                            phi_avg=jnp.asarray(f_chunk_host.phi_avg[b]),
                        )
                        if (
                            (specs[b].cfg.write_fields or specs[b].cfg.write_moms)
                            and f_chunk_host is not None
                        )
                        else None
                    )
                    writer_b.write_snapshot(snap_b, G=g_b, fields=f_b, solver=host_solvers[b])

            for b in range(b_sz):
                if need_periodic_save[b] and restart_paths[b] is not None:
                    assert g_chunk_host is not None
                    if is_primary:
                        _atomic_save_member_restart_nc(
                            restart_paths[b],
                            g_chunk_host[b],
                            float(time_np[b]),
                            specs[b].grids,
                            member_idx=b,
                        )
                    last_saved_step[b] = int(step_np[b])

    g_final_host, fields_final_host = gather_global_tree((state.G, state.fields))
    wall_seconds = time.perf_counter() - t_wall_start

    for b in range(b_sz):
        r_path = restart_paths[b]
        if is_primary and r_path is not None and last_saved_step[b] != int(step_np[b]):
            _atomic_save_member_restart_nc(
                r_path,
                g_final_host[b],
                float(time_np[b]),
                specs[b].grids,
                member_idx=b,
            )

    member_results: list[dict[str, Any]] = []
    member_json_summaries: list[dict[str, Any]] = []

    for b, sp in enumerate(specs):
        cfg_b = sp.cfg
        dt_nom_b = float(cfg_b.dt) if cfg_b.dt > 0.0 else 0.05
        t_max_val_b = float(cfg_b.t_max)
        nstep_max_b = max(0, int(cfg_b.nstep))
        has_explicit_tmax = (t_max is not None) or (t_max_val_b < 1.0e19)
        step_dt_target = float(dt) if dt is not None else dt_nom_b
        nstep_end_target = t0_list[b] + float(nstep_max_b) * step_dt_target
        if t_max is not None:
            sim_time_target = t_max_val_b
        elif nstep is not None and has_explicit_tmax:
            sim_time_target = min(t_max_val_b, nstep_end_target)
        elif has_explicit_tmax:
            sim_time_target = t_max_val_b
        else:
            sim_time_target = nstep_end_target

        m_path = metrics_paths[b]
        kr2_rep: dict[str, Any] | None = None
        if is_primary:
            kr2_rep = extract_kr2_diagnostics(
                out_nc_paths[b],
                metrics_out_path=m_path,
                sim_time_target=sim_time_target,
            )
            payload = json.loads(m_path.read_text())
            payload["member_index"] = b
            payload["member_name"] = sp.name
            payload["input_file"] = str(sp.input_path) if sp.input_path is not None else sp.name
            payload["output_nc"] = str(out_nc_paths[b])
            payload["restart_nc"] = str(restart_paths[b]) if restart_paths[b] is not None else None
            payload["nstep"] = int(step_np[b])
            payload["dt"] = dt_nom_b
            payload["t_final"] = float(time_np[b])
            payload["sim_time_target"] = float(sim_time_target)
            payload["scheme"] = cfg_b.scheme
            payload["nonlinear"] = bool(cfg_b.nonlinear_mode)
            payload["precision"] = prec_clean
            payload["beta"] = float(cfg_b.beta)
            payload["wall_seconds"] = float(wall_seconds)
            clean_payload = sanitize_json_payload(payload)
            m_path.write_text(
                json.dumps(clean_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            member_json_summaries.append(clean_payload)

        member_results.append(
            {
                "member_index": b,
                "name": sp.name,
                "output_nc": out_nc_paths[b],
                "restart_nc": restart_paths[b],
                "metrics_json": m_path,
                "nstep": int(step_np[b]),
                "t_final": float(time_np[b]),
                "sim_time_target": float(sim_time_target),
                "kr2_summary": kr2_rep,
            }
        )

    # Non-primary processes of a multi-host run never touch shared disk: their
    # per-member `kr2_summary` and the combined `ensemble_summary` stay `None`.
    ensemble_summary: Any = None
    if is_primary:
        ensemble_summary = sanitize_json_payload(
            {
                "batch_size": b_sz,
                "precision": prec_clean,
                "scheme": batch_ops.scheme,
                "nonlinear": bool(batch_ops.nonlinear_mode),
                "fft_backend": batch_ops.fft_backend,
                "num_devices": int(mesh.devices.size),
                "mesh_shape": {"batch": int(mesh.shape["batch"])},
                "wall_seconds": float(wall_seconds),
                "all_converged": bool(
                    all(m.get("converged", False) for m in member_json_summaries)
                ),
                "any_nan_or_inf": bool(
                    any(m.get("has_nan_or_inf", False) for m in member_json_summaries)
                ),
                "total_steps_sum": int(np.sum(step_np)),
                "max_steps_member": int(np.max(step_np)),
                "members": member_json_summaries,
            }
        )
        resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_summary_path.write_text(
            json.dumps(ensemble_summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    # Multi-host: every process leaves the run together, after the primary's disk writes.
    sync_processes("gx-jax-ensemble run complete")

    return {
        "batch_size": b_sz,
        "summary_json": resolved_summary_path,
        "ensemble_summary": ensemble_summary,
        "members": member_results,
        "g_final": jnp.asarray(g_final_host, dtype=complex_dtype),
        "fields_final": Fields(
            phi=jnp.asarray(fields_final_host.phi, dtype=complex_dtype),
            apar=jnp.asarray(fields_final_host.apar, dtype=complex_dtype),
            bpar=jnp.asarray(fields_final_host.bpar, dtype=complex_dtype),
            phi_avg=jnp.asarray(fields_final_host.phi_avg, dtype=complex_dtype),
        ),
        "nstep": np.asarray(step_np, dtype=np.int64),
        "t_final": np.asarray(time_np, dtype=np.float64),
        "wall_seconds": float(wall_seconds),
        "num_devices": int(mesh.devices.size),
    }


def build_ensemble_arg_parser() -> argparse.ArgumentParser:
    """Construct the `gx-jax-ensemble` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gx-jax-ensemble",
        description="Stellarator GX-JAX batched `jax.vmap` KR2 ensemble campaign runner.",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="One or more GX input TOML decks (<stem>.in or <stem>.toml) to batch.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for per-member .out.nc, .restart.nc, & JSON metrics.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Output path for combined ensemble_summary.json.",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=None,
        help="Override maximum number of timesteps across all ensemble members.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override fixed timestep size dt across all ensemble members.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Override maximum simulation time t_max across all ensemble members.",
    )
    parser.add_argument(
        "--nwrite",
        type=int,
        default=None,
        help="Override diagnostic write cadence nwrite across all ensemble members.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        choices=["sspx2", "rk3", "sspx3", "rk4", "k10"],
        help="Override time integration scheme across all ensemble members.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--nonlinear",
        action="store_true",
        default=False,
        help="Force nonlinear ExB advection on across all ensemble members.",
    )
    mode_group.add_argument(
        "--linear",
        action="store_true",
        default=False,
        help="Force linear mode across all ensemble members.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp64"],
        help="Floating-point precision: 'fp32' (complex64, default) or 'fp64' (complex128).",
    )
    parser.add_argument(
        "--fft-backend",
        type=str,
        default=None,
        choices=["auto", "fft", "mxu"],
        help="Spectral transform backend override ('auto', 'fft', or 'mxu').",
    )
    parser.add_argument(
        "--batch-devices",
        "--devices",
        dest="batch_devices",
        type=int,
        default=None,
        help="Number of devices along the 1D ('batch',) mesh axis (default: auto).",
    )
    parser.add_argument(
        "--save-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable writing per-member .restart.nc checkpoints.",
    )
    add_distributed_cli_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `gx-jax-ensemble`."""
    parser = build_ensemble_arg_parser()
    args = parser.parse_args(argv)
    for in_path in args.input_files:
        if not in_path.is_file():
            parser.error(f"input file does not exist: {in_path}")

    try:
        try:
            # Multi-host: initialize jax.distributed before any JAX device access
            # (zero-overhead no-op on a single host or with --no-distributed).
            init_distributed_runtime_from_args(args)
            run_ensemble(
                args.input_files,
                output_dir=args.output_dir,
                summary_json_path=args.summary_json,
                nstep=args.nstep,
                dt=args.dt,
                t_max=args.t_max,
                nwrite=args.nwrite,
                scheme=args.scheme,
                force_nonlinear=args.nonlinear,
                force_linear=args.linear,
                precision=args.precision,
                fft_backend=args.fft_backend,
                batch_devices=args.batch_devices,
                save_restart=args.save_restart,
            )
        except ValueError as exc:
            parser.error(str(exc))
    except BaseException as exc:
        # Multi-host only: a process failing alone exits immediately instead of
        # deadlocking at the shutdown barrier (no-op on a single host).
        abort_distributed_run(exc)
        raise
    finally:
        shutdown_distributed_runtime()
    return 0


ensemble_main = main


__all__ = [
    "BatchedEnsembleOperators",
    "EnsembleChunkCarry",
    "EnsembleMemberSpec",
    "advance_ensemble_chunk",
    "batched_compute_diagnostics_step",
    "batched_gyrokinetic_rhs",
    "batched_solve_fields",
    "batched_step_integrator_state",
    "build_ensemble_arg_parser",
    "create_ensemble_mesh",
    "ensemble_main",
    "gather_ensemble_tree",
    "init_batched_ensemble",
    "init_batched_integrator_state",
    "main",
    "make_ensemble_chunk_stepper",
    "run_ensemble",
    "shard_ensemble_tree",
    "validate_ensemble_compatibility",
]


if __name__ == "__main__":
    raise SystemExit(main())
