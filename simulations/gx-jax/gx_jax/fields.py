"""Quasineutrality & Ampère field solver, moments, and state utilities for GX-JAX.

Ported strictly from:
- ``gx/include/{fields.h,solver.h,moments.h}``
- ``gx/src/{fields.cu,solver.cu,moments.cu}``
- ``gx/src/device_funcs.cu`` (``sum_solverFacs``, ``calc_phiavgdenom``,
  ``real_space_density``, ``real_space_par_current``, ``real_space_perp_current``,
  ``qneut``, ``ampere_apar``, ``qneut_and_ampere_perp``, ``qneutAdiab_part1``,
  ``qneutAdiab_part2``, ``qneutAdiab``, ``add_source``, ``calc_n_bar``,
  ``calc_upar_bar``, ``calc_uperp_bar``, ``calc_T_bar``, ``conservation_terms``,
  ``fieldlineaverage``, ``maskG``, ``reality_kernel``)

Numerical & layout notes:
- State ``G`` has Option A layout ``(nspecies, nhermite, nlaguerre, nz, naky, nakx)``.
- Fields ``(phi, apar, bpar)`` have active dealiased layout ``(nz, naky, nakx)``, with
  ``(iky=0, ikx=0)`` masked out via ``grids.active_mask`` of shape ``(naky, nakx)``.
- All static quasineutrality / Ampère operators (including cancellation-free
  ``1 - sum_l J_l(b)^2 = -expm1(-b) - sum_{l>=1} J_l(b)^2``, the coupled ``(phi, bpar)``
  2x2 determinant, and the Jacobian-weighted ``ky = 0`` zonal denominator ``phiavgdenom``)
  are precomputed in IEEE ``float64`` before runtime ``complex64`` solves.
- All species/Laguerre contractions pass ``precision=jax.lax.Precision.HIGHEST`` so TPU MXU
  never truncates inner products to ``bfloat16``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from gx_jax.config import BoltzmannOption, GXConfig
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids, truncate_full_to_kykx
from gx_jax.laguerre import jflr, jflr_a, jflr_b


def _complex_and_real_dtypes(arr: jax.Array) -> tuple[jnp.dtype, jnp.dtype]:
    """Return matching ``(complex_dtype, real_dtype)`` for ``arr``."""
    if jnp.issubdtype(arr.dtype, jnp.complexfloating):
        c_dtype = jnp.dtype(arr.dtype)
    elif arr.dtype == jnp.float64:
        c_dtype = jnp.dtype(jnp.complex128)
    else:
        c_dtype = jnp.dtype(jnp.complex64)
    r_dtype = jnp.dtype(jnp.real(jnp.empty((), dtype=c_dtype)).dtype)
    return c_dtype, r_dtype


_TPU_BACKEND_PRESENT: bool | None = None


def _is_tpu_primary_backend() -> bool:
    """Return True if a TPU backend is initialized (unaffected by jax.default_device)."""
    global _TPU_BACKEND_PRESENT
    if _TPU_BACKEND_PRESENT is None:
        try:
            _TPU_BACKEND_PRESENT = len(jax.devices("tpu")) > 0
        except Exception:
            _TPU_BACKEND_PRESENT = False
    return _TPU_BACKEND_PRESENT


def _time_dtype(r_dtype: jnp.dtype | type = jnp.float32) -> jnp.dtype:
    """Return scalar ``time``/``dt`` carry dtype (``float32`` on TPU, else ``float64``)."""
    rd = jnp.dtype(r_dtype)
    if rd == jnp.float32 and (
        os.environ.get("GX_JAX_FP32_TIME") == "1" or _is_tpu_primary_backend()
    ):
        return jnp.dtype(jnp.float32)
    return jnp.dtype(jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32)


def _host_cast(arr: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Cast concrete host arrays via NumPy so TPU never compiles ``f64``/``c128`` CVTs."""
    target = jnp.dtype(dtype)
    if arr.dtype == target:
        return arr
    if isinstance(arr, jax.core.Tracer):
        return arr.astype(target)
    return jnp.asarray(np.asarray(arr), dtype=target)


def _safe_divide(numer: jax.Array, denom: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute ``numer / denom`` on ``mask`` without creating ``0 / 0 = NaN`` in AD/XLA."""
    valid = mask & (denom != 0)
    safe_denom = jnp.where(valid, denom, jnp.ones_like(denom))
    return jnp.where(valid, numer / safe_denom, jnp.zeros_like(numer))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fields:
    """Electromagnetic potentials on the active spectral grid ``(nz, naky, nakx)``.

    Ported from ``gx/include/fields.h`` and ``src/fields.cu``.
    ``phi_avg`` stores the Jacobian-weighted flux-surface-averaged zonal potential
    ``<phi>(kx)`` on ``ky = 0`` (shape ``(nakx,)``; non-zero only for adiabatic electrons).
    """

    phi: jax.Array  # (nz, naky, nakx) complex64 / complex128
    apar: jax.Array  # (nz, naky, nakx) complex64 / complex128
    bpar: jax.Array  # (nz, naky, nakx) complex64 / complex128
    phi_avg: jax.Array  # (nakx,) complex64 / complex128

    def tree_flatten(self):
        return (self.phi, self.apar, self.bpar, self.phi_avg), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Fields:
        del aux_data
        phi, apar, bpar, phi_avg = children
        return cls(phi=phi, apar=apar, bpar=bpar, phi_avg=phi_avg)

    def astype(self, dtype: jnp.dtype) -> Fields:
        """Cast all field arrays to complex ``dtype`` (e.g. ``jnp.complex64``)."""
        c_dtype = jnp.dtype(dtype)
        return Fields(
            phi=self.phi.astype(c_dtype),
            apar=self.apar.astype(c_dtype),
            bpar=self.bpar.astype(c_dtype),
            phi_avg=self.phi_avg.astype(c_dtype),
        )


def init_fields(grids: Grids, dtype: jnp.dtype = jnp.complex64) -> Fields:
    """Allocate zero-initialized electromagnetic ``Fields`` on ``grids``."""
    c_dtype = jnp.dtype(dtype)
    z3 = jnp.zeros(grids.field_shape, dtype=c_dtype)
    z1 = jnp.zeros((grids.nakx,), dtype=c_dtype)
    return Fields(phi=z3, apar=z3, bpar=z3, phi_avg=z1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FieldSolver:
    """Precomputed static quasineutrality & Ampère operator matrices (``Solver_GK``).

    Ported from ``gx/include/solver.h``, ``src/solver.cu``, and
    ``src/device_funcs.cu`` (``sum_solverFacs``, ``calc_phiavgdenom``).
    """

    # Static 3D quasineutrality & Ampere matrices on (nz, naky, nakx)
    qneut_fac_phi: jax.Array
    qneut_fac_bpar: jax.Array
    ampere_par_fac: jax.Array
    ampere_perp_fac_phi: jax.Array
    ampere_perp_fac_bpar: jax.Array
    phi_bpar_det: jax.Array

    # Static 1D zonal-flow denominator on ky=0, shape (nakx,)
    phiavgdenom: jax.Array

    # Precomputed species-Laguerre FLR weights on (nspecies, nlaguerre, nz, naky, nakx)
    jl: jax.Array
    jlb: jax.Array
    jla: jax.Array

    # Geometry & mask arrays
    kperp2: jax.Array  # (nz, naky, nakx)
    bmag: jax.Array  # (nz,)
    bmag_inv: jax.Array  # (nz,)
    jacobian: jax.Array  # (nz,)
    vol_fac: jax.Array  # (nz,)
    active_mask: jax.Array  # (naky, nakx) bool

    # Per-species 1D parameter vectors of shape (nspecies,)
    sp_z: jax.Array
    sp_mass: jax.Array
    sp_dens: jax.Array
    sp_temp: jax.Array
    sp_rho2: jax.Array
    sp_tz: jax.Array
    sp_zt: jax.Array
    sp_vt: jax.Array
    sp_nt: jax.Array
    sp_nz: jax.Array
    sp_jparfac: jax.Array
    sp_jperpfac: jax.Array

    # Static grid & physics flags (PyTree aux_data for compile-time specialization)
    nspecies: int
    nhermite: int
    nlaguerre: int
    nz: int
    naky: int
    nakx: int
    all_kinetic: bool
    boltzmann_opt: BoltzmannOption
    tau_fac: float
    beta: float
    fphi: float
    fapar: float
    fbpar: float
    long_wavelength_gk: bool
    no_fields: bool
    phi_ext: float

    def tree_flatten(self):
        children = (
            self.qneut_fac_phi,
            self.qneut_fac_bpar,
            self.ampere_par_fac,
            self.ampere_perp_fac_phi,
            self.ampere_perp_fac_bpar,
            self.phi_bpar_det,
            self.phiavgdenom,
            self.jl,
            self.jlb,
            self.jla,
            self.kperp2,
            self.bmag,
            self.bmag_inv,
            self.jacobian,
            self.vol_fac,
            self.active_mask,
            self.sp_z,
            self.sp_mass,
            self.sp_dens,
            self.sp_temp,
            self.sp_rho2,
            self.sp_tz,
            self.sp_zt,
            self.sp_vt,
            self.sp_nt,
            self.sp_nz,
            self.sp_jparfac,
            self.sp_jperpfac,
        )
        aux_data = (
            self.nspecies,
            self.nhermite,
            self.nlaguerre,
            self.nz,
            self.naky,
            self.nakx,
            self.all_kinetic,
            int(self.boltzmann_opt),
            self.tau_fac,
            self.beta,
            self.fphi,
            self.fapar,
            self.fbpar,
            self.long_wavelength_gk,
            self.no_fields,
            self.phi_ext,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> FieldSolver:
        (
            nspecies,
            nhermite,
            nlaguerre,
            nz,
            naky,
            nakx,
            all_kinetic,
            boltzmann_opt_int,
            tau_fac,
            beta,
            fphi,
            fapar,
            fbpar,
            long_wavelength_gk,
            no_fields,
            phi_ext,
        ) = aux_data
        (
            qneut_fac_phi,
            qneut_fac_bpar,
            ampere_par_fac,
            ampere_perp_fac_phi,
            ampere_perp_fac_bpar,
            phi_bpar_det,
            phiavgdenom,
            jl,
            jlb,
            jla,
            kperp2,
            bmag,
            bmag_inv,
            jacobian,
            vol_fac,
            active_mask,
            sp_z,
            sp_mass,
            sp_dens,
            sp_temp,
            sp_rho2,
            sp_tz,
            sp_zt,
            sp_vt,
            sp_nt,
            sp_nz,
            sp_jparfac,
            sp_jperpfac,
        ) = children
        return cls(
            qneut_fac_phi=qneut_fac_phi,
            qneut_fac_bpar=qneut_fac_bpar,
            ampere_par_fac=ampere_par_fac,
            ampere_perp_fac_phi=ampere_perp_fac_phi,
            ampere_perp_fac_bpar=ampere_perp_fac_bpar,
            phi_bpar_det=phi_bpar_det,
            phiavgdenom=phiavgdenom,
            jl=jl,
            jlb=jlb,
            jla=jla,
            kperp2=kperp2,
            bmag=bmag,
            bmag_inv=bmag_inv,
            jacobian=jacobian,
            vol_fac=vol_fac,
            active_mask=active_mask,
            sp_z=sp_z,
            sp_mass=sp_mass,
            sp_dens=sp_dens,
            sp_temp=sp_temp,
            sp_rho2=sp_rho2,
            sp_tz=sp_tz,
            sp_zt=sp_zt,
            sp_vt=sp_vt,
            sp_nt=sp_nt,
            sp_nz=sp_nz,
            sp_jparfac=sp_jparfac,
            sp_jperpfac=sp_jperpfac,
            nspecies=nspecies,
            nhermite=nhermite,
            nlaguerre=nlaguerre,
            nz=nz,
            naky=naky,
            nakx=nakx,
            all_kinetic=all_kinetic,
            boltzmann_opt=BoltzmannOption(boltzmann_opt_int),
            tau_fac=tau_fac,
            beta=beta,
            fphi=fphi,
            fapar=fapar,
            fbpar=fbpar,
            long_wavelength_gk=long_wavelength_gk,
            no_fields=no_fields,
            phi_ext=phi_ext,
        )

    def astype(self, dtype: jnp.dtype) -> FieldSolver:
        """Cast all real operator arrays to ``dtype`` (e.g. ``jnp.float32`` for runtime)."""
        r_dtype = jnp.dtype(dtype)
        return FieldSolver(
            qneut_fac_phi=_host_cast(self.qneut_fac_phi, r_dtype),
            qneut_fac_bpar=_host_cast(self.qneut_fac_bpar, r_dtype),
            ampere_par_fac=_host_cast(self.ampere_par_fac, r_dtype),
            ampere_perp_fac_phi=_host_cast(self.ampere_perp_fac_phi, r_dtype),
            ampere_perp_fac_bpar=_host_cast(self.ampere_perp_fac_bpar, r_dtype),
            phi_bpar_det=_host_cast(self.phi_bpar_det, r_dtype),
            phiavgdenom=_host_cast(self.phiavgdenom, r_dtype),
            jl=_host_cast(self.jl, r_dtype),
            jlb=_host_cast(self.jlb, r_dtype),
            jla=_host_cast(self.jla, r_dtype),
            kperp2=_host_cast(self.kperp2, r_dtype),
            bmag=_host_cast(self.bmag, r_dtype),
            bmag_inv=_host_cast(self.bmag_inv, r_dtype),
            jacobian=_host_cast(self.jacobian, r_dtype),
            vol_fac=_host_cast(self.vol_fac, r_dtype),
            active_mask=self.active_mask,
            sp_z=_host_cast(self.sp_z, r_dtype),
            sp_mass=_host_cast(self.sp_mass, r_dtype),
            sp_dens=_host_cast(self.sp_dens, r_dtype),
            sp_temp=_host_cast(self.sp_temp, r_dtype),
            sp_rho2=_host_cast(self.sp_rho2, r_dtype),
            sp_tz=_host_cast(self.sp_tz, r_dtype),
            sp_zt=_host_cast(self.sp_zt, r_dtype),
            sp_vt=_host_cast(self.sp_vt, r_dtype),
            sp_nt=_host_cast(self.sp_nt, r_dtype),
            sp_nz=_host_cast(self.sp_nz, r_dtype),
            sp_jparfac=_host_cast(self.sp_jparfac, r_dtype),
            sp_jperpfac=_host_cast(self.sp_jperpfac, r_dtype),
            nspecies=self.nspecies,
            nhermite=self.nhermite,
            nlaguerre=self.nlaguerre,
            nz=self.nz,
            naky=self.naky,
            nakx=self.nakx,
            all_kinetic=self.all_kinetic,
            boltzmann_opt=self.boltzmann_opt,
            tau_fac=self.tau_fac,
            beta=self.beta,
            fphi=self.fphi,
            fapar=self.fapar,
            fbpar=self.fbpar,
            long_wavelength_gk=self.long_wavelength_gk,
            no_fields=self.no_fields,
            phi_ext=self.phi_ext,
        )

    def solve(self, G: jax.Array) -> Fields:
        """Convenience wrapper calling :func:`solve_fields`."""
        return solve_fields(G, self)


def init_field_solver(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    *,
    dtype: jnp.dtype = jnp.float64,
    no_fields: bool = False,
    phi_ext: float = 0.0,
) -> FieldSolver:
    """Precompute static quasineutrality & Ampère matrices in IEEE ``float64``.

    Strictly ports ``Solver_GK::Solver_GK`` (``solver.cu:L8-84``), ``sum_solverFacs``
    (``device_funcs.cu:L2188-2248``), and ``calc_phiavgdenom`` (``L2307-2324``).

    Cancellation-free FLR polarization factor:
        ``1 - sum_{l=0}^{nl-1} J_l(b)^2 = -expm1(-b) - sum_{l=1}^{nl-1} J_l(b)^2``
    Since ``J_0(b)^2 == exp(-b)`` and ``sum_{l>=1} J_l(b)^2 <= 0.31 * (-expm1(-b))``
    for all ``b > 0``, this identity avoids catastrophic cancellation as ``k_perp * rho_s -> 0``
    while remaining algebraically identical to ``1 - sum_{l=0}^{nl-1} J_l(b)^2``.
    """
    ns = grids.nspecies
    if len(cfg.species) != ns:
        raise ValueError(f"Expected {ns} kinetic species in cfg.species, got {len(cfg.species)}.")
    nm = grids.nhermite
    nl = grids.nlaguerre
    nz = grids.nz
    naky = grids.naky
    nakx = grids.nakx

    kperp2_64 = jnp.asarray(geo.kperp2, dtype=jnp.float64)
    bmag64 = jnp.asarray(geo.bmag, dtype=jnp.float64)
    bmag_inv64 = jnp.asarray(geo.bmag_inv, dtype=jnp.float64)
    jacobian64 = jnp.asarray(geo.jacobian, dtype=jnp.float64)
    vol_fac64 = jnp.asarray(geo.vol_fac, dtype=jnp.float64)
    active_mask = jnp.asarray(grids.active_mask, dtype=jnp.bool_)
    mask3d = active_mask[None, :, :]

    ell_vec = jnp.arange(nl, dtype=jnp.int32)[:, None, None, None]
    beta64 = float(cfg.beta)
    bmag2_3d = (bmag64 * bmag64)[:, None, None]
    bmag_inv2_3d = (bmag_inv64 * bmag_inv64)[:, None, None]

    qneut_phi64 = jnp.zeros((nz, naky, nakx), dtype=jnp.float64)
    qneut_bpar64 = jnp.zeros((nz, naky, nakx), dtype=jnp.float64)
    ampere_par64 = kperp2_64 * bmag2_3d
    ampere_perp_phi64 = jnp.zeros((nz, naky, nakx), dtype=jnp.float64)
    ampere_perp_bpar64 = jnp.ones((nz, naky, nakx), dtype=jnp.float64)

    jl_species: list[jax.Array] = []
    jlb_species: list[jax.Array] = []
    jla_species: list[jax.Array] = []

    for sp in cfg.species:
        # Guiding-center source/moment argument uses sp.rho2 (0 when long_wavelength_gk=True;
        # see config.py:L72-77 and device_funcs.cu:L2099, L2115, L2133).
        b_runtime = kperp2_64 * float(sp.rho2)
        jl_rt = jflr(ell_vec, b_runtime)
        jlb_rt = jflr_b(ell_vec, b_runtime)
        jla_rt = jflr_a(ell_vec, b_runtime)
        jl_species.append(jl_rt)
        jlb_species.append(jlb_rt)
        jla_species.append(jla_rt)

        # Solver matrix argument uses sp.rho2_long_wavelength_GK when long_wavelength_gk=True
        # (device_funcs.cu:L2215-2216).
        rho2_sol = float(sp.rho2_long_wavelength_GK if cfg.long_wavelength_gk else sp.rho2)
        b_sol = kperp2_64 * rho2_sol
        jl_sol = jflr(ell_vec, b_sol)
        jlb_sol = jflr_b(ell_vec, b_sol)

        g0_s = jnp.sum(jl_sol * jl_sol, axis=0)
        g01_s = jnp.sum(jl_sol * jlb_sol, axis=0)
        g11_s = jnp.sum(jlb_sol * jlb_sol, axis=0)

        if nl > 1:
            higher_jl2 = jnp.sum(jl_sol[1:] * jl_sol[1:], axis=0)
        else:
            higher_jl2 = jnp.zeros_like(b_sol)
        one_minus_g0_s = -jnp.expm1(-b_sol) - higher_jl2

        nz_zt = float(sp.nz) * float(sp.zt)
        if cfg.long_wavelength_gk:
            qneut_phi64 = qneut_phi64 + nz_zt * b_sol
        else:
            qneut_phi64 = qneut_phi64 + nz_zt * one_minus_g0_s

        qneut_bpar64 = qneut_bpar64 - float(sp.nz) * g01_s
        ampere_par64 = (
            ampere_par64 + (float(sp.nz) * float(sp.z) / float(sp.mass) * 0.5 * beta64) * g0_s
        )
        ampere_perp_phi64 = ampere_perp_phi64 + (float(sp.nz) * 0.5 * beta64) * bmag_inv2_3d * g01_s
        ampere_perp_bpar64 = (
            ampere_perp_bpar64 + (float(sp.nt) * 0.5 * beta64) * bmag_inv2_3d * g11_s
        )

    # Mask (iky=0, ikx=0) to 0.0 matching device_funcs.cu:L2210 unmasked(idx, idy)
    qneut_phi64 = jnp.where(mask3d, qneut_phi64, 0.0)
    qneut_bpar64 = jnp.where(mask3d, qneut_bpar64, 0.0)
    ampere_par64 = jnp.where(mask3d, ampere_par64, 0.0)
    ampere_perp_phi64 = jnp.where(mask3d, ampere_perp_phi64, 0.0)
    ampere_perp_bpar64 = jnp.where(mask3d, ampere_perp_bpar64, 0.0)

    # Coupled 2x2 (phi, bpar) block determinant (device_funcs.cu:L2182)
    phi_bpar_det64 = qneut_phi64 * ampere_perp_bpar64 - qneut_bpar64 * ampere_perp_phi64
    phi_bpar_det64 = jnp.where(mask3d, phi_bpar_det64, 0.0)

    # Jacobian-weighted zonal denominator on ky=0 for adiabatic electrons (solver.cu:L74-83,
    # device_funcs.cu:L2307-2324)
    if (
        not cfg.all_kinetic
        and cfg.boltzmann_opt == BoltzmannOption.BOLTZMANN_ELECTRONS
        and not no_fields
    ):
        mask_ky0 = active_mask[0, :]
        q0_64 = qneut_phi64[:, 0, :]
        denom0_64 = float(cfg.tau_fac) + q0_64
        ratio0_64 = _safe_divide(q0_64, denom0_64, mask_ky0[None, :])
        phiavgdenom64 = jnp.where(
            mask_ky0,
            jnp.sum(jacobian64[:, None] * ratio0_64, axis=0),
            0.0,
        )
    else:
        phiavgdenom64 = jnp.zeros((nakx,), dtype=jnp.float64)

    jl64 = jnp.stack(jl_species, axis=0)
    jlb64 = jnp.stack(jlb_species, axis=0)
    jla64 = jnp.stack(jla_species, axis=0)

    sp_z = jnp.asarray([sp.z for sp in cfg.species], dtype=jnp.float64)
    sp_mass = jnp.asarray([sp.mass for sp in cfg.species], dtype=jnp.float64)
    sp_dens = jnp.asarray([sp.dens for sp in cfg.species], dtype=jnp.float64)
    sp_temp = jnp.asarray([sp.temp for sp in cfg.species], dtype=jnp.float64)
    sp_rho2 = jnp.asarray([sp.rho2 for sp in cfg.species], dtype=jnp.float64)
    sp_tz = jnp.asarray([sp.tz for sp in cfg.species], dtype=jnp.float64)
    sp_zt = jnp.asarray([sp.zt for sp in cfg.species], dtype=jnp.float64)
    sp_vt = jnp.asarray([sp.vt for sp in cfg.species], dtype=jnp.float64)
    sp_nt = jnp.asarray([sp.nt for sp in cfg.species], dtype=jnp.float64)
    sp_nz = jnp.asarray([sp.nz for sp in cfg.species], dtype=jnp.float64)
    sp_jparfac = jnp.asarray([sp.jparfac for sp in cfg.species], dtype=jnp.float64)
    sp_jperpfac = jnp.asarray([sp.jperpfac for sp in cfg.species], dtype=jnp.float64)

    solver64 = FieldSolver(
        qneut_fac_phi=qneut_phi64,
        qneut_fac_bpar=qneut_bpar64,
        ampere_par_fac=ampere_par64,
        ampere_perp_fac_phi=ampere_perp_phi64,
        ampere_perp_fac_bpar=ampere_perp_bpar64,
        phi_bpar_det=phi_bpar_det64,
        phiavgdenom=phiavgdenom64,
        jl=jl64,
        jlb=jlb64,
        jla=jla64,
        kperp2=kperp2_64,
        bmag=bmag64,
        bmag_inv=bmag_inv64,
        jacobian=jacobian64,
        vol_fac=vol_fac64,
        active_mask=active_mask,
        sp_z=sp_z,
        sp_mass=sp_mass,
        sp_dens=sp_dens,
        sp_temp=sp_temp,
        sp_rho2=sp_rho2,
        sp_tz=sp_tz,
        sp_zt=sp_zt,
        sp_vt=sp_vt,
        sp_nt=sp_nt,
        sp_nz=sp_nz,
        sp_jparfac=sp_jparfac,
        sp_jperpfac=sp_jperpfac,
        nspecies=ns,
        nhermite=nm,
        nlaguerre=nl,
        nz=nz,
        naky=naky,
        nakx=nakx,
        all_kinetic=bool(cfg.all_kinetic),
        boltzmann_opt=cfg.boltzmann_opt,
        tau_fac=float(cfg.tau_fac),
        beta=float(cfg.beta),
        fphi=float(cfg.fphi),
        fapar=float(cfg.fapar),
        fbpar=float(cfg.fbpar),
        long_wavelength_gk=bool(cfg.long_wavelength_gk),
        no_fields=bool(no_fields),
        phi_ext=float(phi_ext),
    )
    out_dtype = jnp.dtype(dtype)
    return solver64 if out_dtype == jnp.float64 else solver64.astype(out_dtype)


def real_space_density(G: jax.Array, solver: FieldSolver) -> jax.Array:
    """Compute guiding-center charge density ``nbar`` on ``(nz, naky, nakx)``.

    Ported from ``real_space_density`` (``device_funcs.cu:L2092-2107``):
        ``nbar = sum_{s, l} (n_s Z_s) * J_l(b_s) * G[s, 0, l]``
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g0_spec = G.astype(c_dtype)[:, 0]
    weights = solver.sp_nz.astype(r_dtype)[:, None, None, None, None] * solver.jl.astype(r_dtype)
    nbar = jnp.einsum(
        "slzyx,slzyx->zyx",
        weights,
        g0_spec,
        precision=jax.lax.Precision.HIGHEST,
    )
    mask3d = solver.active_mask[None, :, :]
    return jnp.where(mask3d, nbar, jnp.zeros_like(nbar))


def real_space_par_current(G: jax.Array, solver: FieldSolver) -> jax.Array:
    """Compute parallel current source ``jparbar`` on ``(nz, naky, nakx)``.

    Ported from ``real_space_par_current`` (``device_funcs.cu:L2109-2124``):
        ``jparbar = sum_{s, l} (beta / 2 * n_s Z_s v_{ts}) * J_l(b_s) * G[s, 1, l]``
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    if G.shape[1] < 2:
        return jnp.zeros((solver.nz, solver.naky, solver.nakx), dtype=c_dtype)
    g1_spec = G.astype(c_dtype)[:, 1]
    weights = solver.sp_jparfac.astype(r_dtype)[:, None, None, None, None] * solver.jl.astype(
        r_dtype
    )
    jparbar = jnp.einsum(
        "slzyx,slzyx->zyx",
        weights,
        g1_spec,
        precision=jax.lax.Precision.HIGHEST,
    )
    mask3d = solver.active_mask[None, :, :]
    return jnp.where(mask3d, jparbar, jnp.zeros_like(jparbar))


def real_space_perp_current(G: jax.Array, solver: FieldSolver) -> jax.Array:
    """Compute perpendicular diamagnetic current source ``jperpbar`` on ``(nz, naky, nakx)``.

    Ported from ``real_space_perp_current`` (``device_funcs.cu:L2126-2142``):
        ``jperpbar = sum_{s, l} (-beta / 2 * n_s T_s / B^2) * J_{lB}(b_s) * G[s, 0, l]``
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g0_spec = G.astype(c_dtype)[:, 0]
    bmag_inv = solver.bmag_inv.astype(r_dtype)
    bmag_inv2 = (bmag_inv * bmag_inv)[None, None, :, None, None]
    weights = (
        solver.sp_jperpfac.astype(r_dtype)[:, None, None, None, None]
        * bmag_inv2
        * solver.jlb.astype(r_dtype)
    )
    jperpbar = jnp.einsum(
        "slzyx,slzyx->zyx",
        weights,
        g0_spec,
        precision=jax.lax.Precision.HIGHEST,
    )
    mask3d = solver.active_mask[None, :, :]
    return jnp.where(mask3d, jperpbar, jnp.zeros_like(jperpbar))


def solve_fields(G: jax.Array, solver: FieldSolver) -> Fields:
    """Solve gyrokinetic quasineutrality & Ampère's laws for ``(phi, apar, bpar, phi_avg)``.

    Strictly ports ``Solver_GK::fieldSolve`` (``solver.cu:L100-158``) across all physics regimes:
    1. ``all_kinetic=True``:
       - Electrostatic (``fbpar == 0``): ``phi = fphi * nbar / qneut_fac_phi``, ``bpar = 0``.
       - Electromagnetic (``fbpar > 0``): coupled 2x2 ``(phi, bpar)`` block solve via
         precomputed determinant ``phi_bpar_det``.
       - Parallel Ampère (``fapar > 0``): ``apar = fapar * jparbar / ampere_par_fac``.
    2. ``all_kinetic=False, boltzmann_opt == BOLTZMANN_ELECTRONS``:
       - Jacobian-weighted flux-surface-averaged potential ``<phi>(kx)`` subtracted strictly
         on zonal modes ``ky = 0, kx != 0`` using precomputed ``phiavgdenom``.
    3. ``all_kinetic=False, boltzmann_opt == BOLTZMANN_IONS``:
       - Adiabatic ion response ``tau_fac * phi`` without zonal subtraction, plus optional
         finite-``beta`` ``apar`` solve when ``fapar > 0``.
    """
    expected_shape = (
        solver.nspecies,
        solver.nhermite,
        solver.nlaguerre,
        solver.nz,
        solver.naky,
        solver.nakx,
    )
    if G.shape != expected_shape:
        raise ValueError(f"Expected G shape {expected_shape}, got {G.shape}.")

    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    field_shape = (solver.nz, solver.naky, solver.nakx)
    zero3 = jnp.zeros(field_shape, dtype=c_dtype)
    zero1 = jnp.zeros((solver.nakx,), dtype=c_dtype)

    if solver.no_fields:
        return Fields(phi=zero3, apar=zero3, bpar=zero3, phi_avg=zero1)

    mask3d = solver.active_mask[None, :, :]
    fphi = jnp.asarray(solver.fphi, dtype=r_dtype)
    fapar = jnp.asarray(solver.fapar, dtype=r_dtype)
    fbpar = jnp.asarray(solver.fbpar, dtype=r_dtype)

    nbar = real_space_density(G, solver)

    if solver.all_kinetic:
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        if solver.fbpar > 0.0:
            jperpbar = real_space_perp_current(G, solver)
            q_bpar = solver.qneut_fac_bpar.astype(r_dtype)
            a_phi = solver.ampere_perp_fac_phi.astype(r_dtype)
            a_bpar = solver.ampere_perp_fac_bpar.astype(r_dtype)
            det = solver.phi_bpar_det.astype(r_dtype)

            num_phi = fphi * (a_bpar * nbar - q_bpar * jperpbar)
            num_bpar = fbpar * (-a_phi * nbar + q_phi * jperpbar)
            phi = _safe_divide(num_phi, det, mask3d)
            bpar = _safe_divide(num_bpar, det, mask3d)
        else:
            if solver.fphi != 0.0:
                phi = _safe_divide(fphi * nbar, q_phi, mask3d)
            else:
                phi = zero3
            bpar = zero3

        if solver.fapar > 0.0 and solver.nhermite >= 2:
            jparbar = real_space_par_current(G, solver)
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

        # qneutAdiab_part1 & part2 on ky = 0 (device_funcs.cu:L2251-2305)
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

    elif solver.boltzmann_opt == BoltzmannOption.BOLTZMANN_IONS:
        tau = jnp.asarray(solver.tau_fac, dtype=r_dtype)
        q_phi = solver.qneut_fac_phi.astype(r_dtype)
        denom = tau + q_phi
        if solver.fphi != 0.0:
            phi = _safe_divide(fphi * nbar, denom, mask3d)
        else:
            phi = zero3

        if solver.fapar > 0.0 and solver.nhermite >= 2:
            jparbar = real_space_par_current(G, solver)
            a_par = solver.ampere_par_fac.astype(r_dtype)
            apar = _safe_divide(fapar * jparbar, a_par, mask3d)
        else:
            apar = zero3
        bpar = zero3
        phi_avg = zero1

    else:
        raise ValueError(
            f"Unsupported boltzmann_opt={solver.boltzmann_opt!r} with all_kinetic=False"
        )

    if solver.phi_ext != 0.0:
        ext = jnp.asarray(solver.phi_ext, dtype=r_dtype)
        phi = jnp.where(mask3d, phi + ext, jnp.zeros_like(phi))

    return Fields(
        phi=phi.astype(c_dtype),
        apar=apar.astype(c_dtype),
        bpar=bpar.astype(c_dtype),
        phi_avg=phi_avg.astype(c_dtype),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Moments:
    """Fluid Hermite-Laguerre moments of ``G`` and gyroaveraged particle moments of ``H``.

    Ported from ``moments.cu:L19-44``, ``diagnostic_classes.cu:L1047-1112``, and
    ``device_funcs.cu:L1953-2034, L2092-2142``.

    All per-species arrays have shape ``(nspecies, nz, naky, nakx)``; summed guiding-center
    source arrays ``(nbar, jparbar, jperpbar)`` have shape ``(nz, naky, nakx)``.
    """

    # Named fluid Hermite-Laguerre moments of G
    dens: jax.Array  # G[:, m=0, l=0]
    upar: jax.Array  # G[:, m=1, l=0]
    tpar: jax.Array  # sqrt(2) * G[:, m=2, l=0]
    qpar: jax.Array  # sqrt(6) * G[:, m=3, l=0]
    tperp: jax.Array  # G[:, m=0, l=1]
    qperp: jax.Array  # G[:, m=1, l=1]

    # Gyroaveraged non-adiabatic (H) moments per species
    n_bar: jax.Array
    upar_bar: jax.Array
    uperp_bar: jax.Array
    t_bar: jax.Array

    # Species-summed guiding-center charge/current sources
    nbar: jax.Array
    jparbar: jax.Array
    jperpbar: jax.Array

    def tree_flatten(self):
        children = (
            self.dens,
            self.upar,
            self.tpar,
            self.qpar,
            self.tperp,
            self.qperp,
            self.n_bar,
            self.upar_bar,
            self.uperp_bar,
            self.t_bar,
            self.nbar,
            self.jparbar,
            self.jperpbar,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Moments:
        del aux_data
        (
            dens,
            upar,
            tpar,
            qpar,
            tperp,
            qperp,
            n_bar,
            upar_bar,
            uperp_bar,
            t_bar,
            nbar,
            jparbar,
            jperpbar,
        ) = children
        return cls(
            dens=dens,
            upar=upar,
            tpar=tpar,
            qpar=qpar,
            tperp=tperp,
            qperp=qperp,
            n_bar=n_bar,
            upar_bar=upar_bar,
            uperp_bar=uperp_bar,
            t_bar=t_bar,
            nbar=nbar,
            jparbar=jparbar,
            jperpbar=jperpbar,
        )

    def astype(self, dtype: jnp.dtype) -> Moments:
        """Cast all moment arrays to complex ``dtype``."""
        c_dtype = jnp.dtype(dtype)
        return jax.tree_util.tree_map(lambda x: x.astype(c_dtype), self)


def compute_moments(
    G: jax.Array,
    fields: Fields,
    solver: FieldSolver,
    *,
    zero_ky0: bool = False,
) -> Moments:
    """Compute fluid moments of ``G`` and gyroaveraged ``H`` particle/source moments.

    Ported from ``moments.cu:L19-44`` and ``device_funcs.cu:L1953-2034, L2092-2142``.
    When ``zero_ky0=True``, ``n_bar``, ``upar_bar``, ``uperp_bar``, and ``t_bar`` zero out
    the ``iky = 0`` slice to match the ``idy > 0`` flux-diagnostic guard in ``calc_*_bar``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    ns, nm, nl, nz, naky, nakx = g_c.shape
    mask4d = solver.active_mask[None, None, :, :]
    zero_sp = jnp.zeros((ns, nz, naky, nakx), dtype=c_dtype)

    sqrt2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
    sqrt6 = jnp.asarray(math.sqrt(6.0), dtype=r_dtype)
    inv_sqrt2 = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
    two_thirds = jnp.asarray(2.0 / 3.0, dtype=r_dtype)

    dens = jnp.where(mask4d, g_c[:, 0, 0], zero_sp)
    upar = jnp.where(mask4d, g_c[:, 1, 0], zero_sp) if nm >= 2 else zero_sp
    tpar = jnp.where(mask4d, sqrt2 * g_c[:, 2, 0], zero_sp) if nm >= 3 else zero_sp
    qpar = jnp.where(mask4d, sqrt6 * g_c[:, 3, 0], zero_sp) if nm >= 4 else zero_sp
    tperp = jnp.where(mask4d, g_c[:, 0, 1], zero_sp) if nl >= 2 else zero_sp
    qperp = jnp.where(mask4d, g_c[:, 1, 1], zero_sp) if (nm >= 2 and nl >= 2) else zero_sp

    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]

    jl = solver.jl.astype(r_dtype)
    jlb = solver.jlb.astype(r_dtype)
    jla = solver.jla.astype(r_dtype)
    zt = solver.sp_zt.astype(r_dtype)[:, None, None, None, None]
    vt = solver.sp_vt.astype(r_dtype)[:, None, None, None, None]

    h0 = g_c[:, 0] + zt * jl * phi + jlb * bpar
    g1_slice = g_c[:, 1] if nm >= 2 else jnp.zeros((ns, nl, nz, naky, nakx), dtype=c_dtype)
    h1 = g1_slice - (zt * vt) * jl * apar
    h2 = g_c[:, 2] if nm >= 3 else jnp.zeros((ns, nl, nz, naky, nakx), dtype=c_dtype)

    n_bar = jnp.einsum("slzyx,slzyx->szyx", jl, h0, precision=jax.lax.Precision.HIGHEST)
    upar_bar = jnp.einsum("slzyx,slzyx->szyx", jl, h1, precision=jax.lax.Precision.HIGHEST)
    uperp_raw = jnp.einsum("slzyx,slzyx->szyx", jlb, h0, precision=jax.lax.Precision.HIGHEST)

    b_s = (
        solver.sp_rho2.astype(r_dtype)[:, None, None, None]
        * solver.kperp2.astype(r_dtype)[None, :, :, :]
    )
    sqrt_b = jnp.sqrt(jnp.maximum(b_s, 0.0))
    uperp_bar = sqrt_b * uperp_raw

    t_bar_h0 = jnp.einsum("slzyx,slzyx->szyx", jla, h0, precision=jax.lax.Precision.HIGHEST)
    t_bar_h2 = jnp.einsum("slzyx,slzyx->szyx", jl, h2, precision=jax.lax.Precision.HIGHEST)
    t_bar = two_thirds * (t_bar_h0 + inv_sqrt2 * t_bar_h2)

    hbar_mask = mask4d & (jnp.arange(naky)[None, None, :, None] > 0) if zero_ky0 else mask4d
    n_bar = jnp.where(hbar_mask, n_bar, zero_sp)
    upar_bar = jnp.where(hbar_mask, upar_bar, zero_sp)
    uperp_bar = jnp.where(hbar_mask, uperp_bar, zero_sp)
    t_bar = jnp.where(hbar_mask, t_bar, zero_sp)

    nbar = real_space_density(g_c, solver)
    jparbar = real_space_par_current(g_c, solver)
    jperpbar = real_space_perp_current(g_c, solver)

    return Moments(
        dens=dens,
        upar=upar,
        tpar=tpar,
        qpar=qpar,
        tperp=tperp,
        qperp=qperp,
        n_bar=n_bar,
        upar_bar=upar_bar,
        uperp_bar=uperp_bar,
        t_bar=t_bar,
        nbar=nbar,
        jparbar=jparbar,
        jperpbar=jperpbar,
    )


def compute_conservation_moments(
    G: jax.Array,
    fields: Fields,
    solver: FieldSolver,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ``(upar_bar, uperp_bar, t_bar)`` matching ``conservation_terms``.

    Ported from ``device_funcs.cu:L3464-3522`` (including the ``nlaguerre == 1`` energy
    conservation branch ``t_bar = sqrt(2) * J_0(b_s) * G[:, 2, 0]`` on ``L3509-3510``).
    """
    moms = compute_moments(G, fields, solver, zero_ky0=False)
    if solver.nlaguerre > 1:
        return moms.upar_bar, moms.uperp_bar, moms.t_bar

    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    ns, nm, _, nz, naky, nakx = g_c.shape
    zero_sp = jnp.zeros((ns, nz, naky, nakx), dtype=c_dtype)
    if nm < 3:
        return moms.upar_bar, moms.uperp_bar, zero_sp
    sqrt2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
    j0 = solver.jl[:, 0].astype(r_dtype)
    t_bar_nl1 = jnp.where(solver.active_mask[None, None, :, :], sqrt2 * j0 * g_c[:, 2, 0], zero_sp)
    return moms.upar_bar, moms.uperp_bar, t_bar_nl1


def field_line_average(
    f: jax.Array,
    grids: Grids,
    geo: Geometry,
) -> tuple[jax.Array, jax.Array]:
    """Compute Jacobian-weighted zonal flux-surface average ``(favg, df)``.

    Ported from ``fieldlineaverage`` (``device_funcs.cu:L1379-1407``):
    - ``favg`` has shape ``(..., nakx)`` with
      ``favg[..., ikx] = sum_z f[..., z, 0, ikx] * vol_fac[z]`` for ``ikx != 0`` and ``0`` at
      ``ikx == 0``.
    - ``df`` has shape ``(..., nz, naky, nakx)`` with
      ``df[..., :, 0, ikx] = f[..., :, 0, ikx] - favg[..., None, ikx]`` on ``ikx != 0``,
      ``df = f`` on ``iky > 0``, and ``0`` at ``(iky=0, ikx=0)``.
    """
    if f.shape[-3:] != grids.field_shape:
        raise ValueError(f"Expected trailing shape {grids.field_shape}, got {f.shape}.")
    c_dtype, r_dtype = _complex_and_real_dtypes(f)
    f_c = f.astype(c_dtype)
    vol_fac = geo.vol_fac.astype(r_dtype)
    mask_ky0 = grids.active_mask[0, :]
    mask3d = grids.active_mask[None, :, :]

    favg_raw = jnp.einsum(
        "...zx,z->...x",
        f_c[..., :, 0, :],
        vol_fac,
        precision=jax.lax.Precision.HIGHEST,
    )
    favg = jnp.where(mask_ky0, favg_raw, jnp.zeros_like(favg_raw))
    df_ky0 = jnp.where(
        mask_ky0[None, :],
        f_c[..., :, 0, :] - favg[..., None, :],
        jnp.zeros_like(f_c[..., :, 0, :]),
    )
    df = f_c.at[..., :, 0, :].set(df_ky0)
    df = jnp.where(mask3d, df, jnp.zeros_like(df))
    return favg, df


def mask_g(G: jax.Array, grids: Grids) -> jax.Array:
    """Zero masked ``(iky=0, ikx=0)`` mode on ``G`` (``device_funcs.cu:maskG``, ``L633-648``)."""
    return jnp.where(grids.active_mask, G, jnp.zeros_like(G))


def enforce_reality(G: jax.Array, grids: Grids) -> jax.Array:
    """Enforce Hermitian symmetry ``G(..., ky=0, -kx) = conj(G(..., ky=0, +kx))``.

    Ported from ``MomentsG::reality`` / ``reality_kernel`` (``moments.cu:L313-334``,
    ``device_funcs.cu:L586-605``). Under Option A active ``kx`` ordering
    ``[0, +1, ..., +K, -K, ..., -1] / x0`` (``K = (nx - 1) // 3``), ``+1..+K`` sit at
    indices ``1..K`` and ``-K..-1`` sit at indices ``K+1..2K`` in reverse order.
    """
    if G.shape[-2:] != (grids.naky, grids.nakx):
        raise ValueError(f"Expected trailing shape {(grids.naky, grids.nakx)}, got {G.shape[-2:]}.")
    kx_cutoff = (grids.nx - 1) // 3
    out = G.at[..., 0, 0].set(jnp.zeros_like(G[..., 0, 0]))
    if kx_cutoff > 0:
        pos_slice = out[..., 0, 1 : kx_cutoff + 1]
        out = out.at[..., 0, kx_cutoff + 1 :].set(jnp.conj(pos_slice[..., ::-1]))
    return out


def _map_single_ikx_to_active(ikx: int, grids: Grids) -> int:
    """Map C++ ``ikx_single`` index onto Option A ``0..nakx-1`` index."""
    kx_cutoff = (grids.nx - 1) // 3
    if 0 <= ikx <= kx_cutoff:
        return ikx
    if grids.nx - kx_cutoff <= ikx < grids.nx:
        return grids.nakx - (grids.nx - ikx)
    if -kx_cutoff <= ikx < 0:
        return grids.nakx + ikx
    raise ValueError(
        f"ikx_single={ikx} is outside active dealiased kx range for nx={grids.nx} "
        f"(cutoff={kx_cutoff})."
    )


def init_moments(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    *,
    dtype: jnp.dtype = jnp.complex64,
    key: jax.Array | None = None,
    densfac: float = 1.0,
    uparfac: float = 1.0,
    tparfac: float = 1.0,
    tperpfac: float = 1.0,
    qparfac: float = 1.0,
    qperpfac: float = 1.0,
) -> jax.Array:
    """Initialize 6D distribution moments ``G`` matching ``MomentsG::initialConditions``.

    Ported from ``moments.cu:L91-266`` and ``run_gx.cu:L54-60``:
    - ``cfg.init_single``: single ``(iky_single, ikx_single)`` mode (plus conjugate ``-kx``
      when ``iky_single == 0`` for reality) modulated by ``cos(ikpar_init * z / zp)``.
    - ``cfg.gaussian_init``: ballooning Gaussian envelope in ``z - theta0`` on ``ky > 0``,
      where ``theta0 = kx_pos / (shat * ky)`` uses positive ``kx`` for both ``+-kx``
      (``moments.cu:L148``).
    - Default random-phase ``kx, ky > 0`` perturbation using ``key`` (or ``PRNGKey(random_seed)``)
      with ``+kx`` amplitude ``ra + i rb`` and ``-kx`` amplitude ``rb + i ra`` (``L183-187``).
    - ``cfg.init_electrons_only``: populates only electron species (``sp.type_id == 1``).
    """
    c_dtype = jnp.dtype(dtype)
    r_dtype = jnp.dtype(jnp.real(jnp.empty((), dtype=c_dtype)).dtype)

    ns = grids.nspecies
    if len(cfg.species) != ns:
        raise ValueError(f"Expected {ns} kinetic species in cfg.species, got {len(cfg.species)}.")
    nm = grids.nhermite
    nl = grids.nlaguerre
    nz = grids.nz
    naky = grids.naky
    nakx = grids.nakx
    kx_cutoff = (grids.nx - 1) // 3

    z64 = np.asarray(grids.z, dtype=np.float64)
    zp_f = float(grids.zp)
    if cfg.ikpar_init < 0:
        ikpar = float(-cfg.ikpar_init)
        z_prof = np.cos(ikpar * z64 / zp_f) + np.cos((ikpar + 1.0) * z64 / zp_f)
    else:
        z_prof = np.cos(float(cfg.ikpar_init) * z64 / zp_f)

    init_3d_np = np.zeros((nz, naky, nakx), dtype=np.complex128)

    if cfg.init_single:
        iky = cfg.iky_single
        if not (0 <= iky < naky):
            raise ValueError(f"iky_single={iky} out of range [0, {naky}).")
        ikx_act = _map_single_ikx_to_active(cfg.ikx_single, grids)
        mode_val = (float(cfg.init_amp) + 0.0j) * z_prof
        init_3d_np[:, iky, ikx_act] = mode_val
        if iky == 0 and 0 < ikx_act <= kx_cutoff:
            init_3d_np[:, 0, nakx - ikx_act] = mode_val

    elif cfg.gaussian_init:
        if naky > 1:
            kx_pos = np.asarray(grids.kx[: kx_cutoff + 1], dtype=np.float64)  # (K+1,)
            ky_pos = np.asarray(grids.ky[1:], dtype=np.float64)  # (naky-1,)
            shat_val = float(geo.shat)
            if abs(shat_val) > 0.0:
                theta0 = kx_pos[None, :] / (shat_val * ky_pos[:, None])  # (naky-1, K+1)
            else:
                theta0 = np.zeros((naky - 1, kx_cutoff + 1), dtype=np.float64)
            dz_th = z64[:, None, None] - theta0[None, :, :]  # (nz, naky-1, K+1)
            env = float(cfg.gauss_env_const_coeff) + float(cfg.gauss_env_sin_coeff) * np.sin(dz_th)
            gauss = env * float(cfg.init_amp) * np.exp(-((dz_th / float(cfg.gaussian_width)) ** 2))
            pos_block = (1.0 + 1.0j) * gauss
            init_3d_np[:, 1:, : kx_cutoff + 1] = pos_block
            if kx_cutoff > 0:
                # moments.cu:L145-156 assigns identical values at nx - ikx (active nakx - ikx)
                init_3d_np[:, 1:, kx_cutoff + 1 :] = pos_block[:, :, 1:][:, :, ::-1]

    else:
        if naky > 1:
            prng_key = jax.random.PRNGKey(cfg.random_seed) if key is None else key
            k1, k2 = jax.random.split(prng_key)
            if not cfg.random_init:
                ra = np.asarray(
                    jax.random.uniform(
                        k1,
                        shape=(naky - 1, kx_cutoff + 1),
                        dtype=jnp.float64,
                        minval=-0.5 * float(cfg.init_amp),
                        maxval=0.5 * float(cfg.init_amp),
                    ),
                    dtype=np.float64,
                )
                rb = np.asarray(
                    jax.random.uniform(
                        k2,
                        shape=(naky - 1, kx_cutoff + 1),
                        dtype=jnp.float64,
                        minval=-0.5 * float(cfg.init_amp),
                        maxval=0.5 * float(cfg.init_amp),
                    ),
                    dtype=np.float64,
                )
                pos_c = (ra + 1.0j * rb)[None, :, :] * z_prof[:, None, None]
                init_3d_np[:, 1:, : kx_cutoff + 1] = pos_c
                if kx_cutoff > 0:
                    neg_c = (rb[:, 1:] + 1.0j * ra[:, 1:])[:, ::-1][None, :, :] * z_prof[
                        :, None, None
                    ]
                    init_3d_np[:, 1:, kx_cutoff + 1 :] = neg_c
            else:
                kz_cut = (nz - 1) // 3
                if kz_cut >= 1:
                    jj = np.arange(1, kz_cut + 1, dtype=np.float64)
                    ka_k, pa_k, kb_k, pb_k = jax.random.split(prng_key, 4)
                    n_kx_modes = 2 * kx_cutoff + 1
                    ka = np.asarray(
                        jax.random.uniform(
                            ka_k,
                            shape=(naky - 1, n_kx_modes, kz_cut),
                            dtype=jnp.float64,
                            minval=0.0,
                            maxval=float(cfg.init_amp),
                        ),
                        dtype=np.float64,
                    )
                    pa = np.asarray(
                        jax.random.uniform(
                            pa_k,
                            shape=(naky - 1, n_kx_modes, kz_cut),
                            dtype=jnp.float64,
                            minval=-0.5 * math.pi,
                            maxval=0.5 * math.pi,
                        ),
                        dtype=np.float64,
                    )
                    kb = np.asarray(
                        jax.random.uniform(
                            kb_k,
                            shape=(naky - 1, n_kx_modes, kz_cut),
                            dtype=jnp.float64,
                            minval=0.0,
                            maxval=float(cfg.init_amp),
                        ),
                        dtype=np.float64,
                    )
                    pb = np.asarray(
                        jax.random.uniform(
                            pb_k,
                            shape=(naky - 1, n_kx_modes, kz_cut),
                            dtype=jnp.float64,
                            minval=-0.5 * math.pi,
                            maxval=0.5 * math.pi,
                        ),
                        dtype=np.float64,
                    )
                    re_part = np.sum(
                        ka[None, :, :, :] * np.sin(z64[:, None, None, None] * jj + pa[None, ...]),
                        axis=-1,
                    )
                    im_part = np.sum(
                        kb[None, :, :, :] * np.sin(z64[:, None, None, None] * jj + pb[None, ...]),
                        axis=-1,
                    )
                    init_3d_np[:, 1:, :] = re_part + 1.0j * im_part

    np_c_dtype = np.complex128 if c_dtype == jnp.complex128 else np.complex64
    init_3d_dev = jnp.asarray(init_3d_np.astype(np_c_dtype), dtype=c_dtype)
    init_3d_cast = jnp.where(
        grids.active_mask[None, :, :],
        init_3d_dev,
        jnp.zeros((nz, naky, nakx), dtype=c_dtype),
    )

    G_single = jnp.zeros((nm, nl, nz, naky, nakx), dtype=c_dtype)
    field_key = cfg.init_field.strip().lower()
    if field_key == "density":
        G_single = G_single.at[0, 0].set(init_3d_cast)
    elif field_key == "upar":
        if nm >= 2:
            G_single = G_single.at[1, 0].set(init_3d_cast)
    elif field_key == "tpar":
        if nm >= 3:
            G_single = G_single.at[2, 0].set(init_3d_cast)
    elif field_key == "tperp":
        if nl >= 2:
            G_single = G_single.at[0, 1].set(init_3d_cast)
    elif field_key == "qpar":
        if nm >= 4:
            G_single = G_single.at[3, 0].set(init_3d_cast)
    elif field_key == "qperp":
        if nm >= 2 and nl >= 2:
            G_single = G_single.at[1, 1].set(init_3d_cast)
    elif field_key == "all":
        G_single = G_single.at[0, 0].set(jnp.asarray(densfac, dtype=r_dtype) * init_3d_cast)
        if nm >= 2:
            G_single = G_single.at[1, 0].set(jnp.asarray(uparfac, dtype=r_dtype) * init_3d_cast)
        if nm >= 3:
            G_single = G_single.at[2, 0].set(
                jnp.asarray(tparfac / math.sqrt(2.0), dtype=r_dtype) * init_3d_cast
            )
        if nl >= 2:
            G_single = G_single.at[0, 1].set(jnp.asarray(tperpfac, dtype=r_dtype) * init_3d_cast)
        if nm >= 4:
            G_single = G_single.at[3, 0].set(
                jnp.asarray(qparfac / math.sqrt(6.0), dtype=r_dtype) * init_3d_cast
            )
        if nm >= 2 and nl >= 2:
            G_single = G_single.at[1, 1].set(jnp.asarray(qperpfac, dtype=r_dtype) * init_3d_cast)
    else:
        raise ValueError(f"Unsupported init_field={cfg.init_field!r}.")

    sp_blocks = []
    for sp in cfg.species:
        if cfg.init_electrons_only and sp.type_id != 1:
            sp_blocks.append(jnp.zeros_like(G_single))
        else:
            sp_blocks.append(G_single)
    return mask_g(jnp.stack(sp_blocks, axis=0), grids)


def load_restart_nc(
    path: str | Path,
    grids: Grids,
    *,
    dtype: jnp.dtype = jnp.complex64,
    scale: float = 1.0,
) -> tuple[jax.Array, float]:
    """Load GX ``.restart.nc`` state ``(G, time)`` into Option A layout.

    Ported from ``MomentsG::restart_read`` (``moments.cu:L570-778``) and
    ``Diagnostics::restart_write`` (``diagnostics.cu:L354-407``).
    GX ``.restart.nc`` stores ``G`` with dimensions ``(Nspecies, Nm, Nl, Nz, Nkx, Nky, ri)``
    where ``Nkx`` already uses FFT-active order ``[0, 1, ..., +K, -K, ..., -1]``; swapping
    ``(Nkx, Nky)`` -> ``(naky, nakx)`` and masking ``(0, 0)`` yields canonical ``G``.
    """
    nc_path = Path(path)
    with netCDF4.Dataset(nc_path, "r") as ds:
        raw_g = np.asarray(ds.variables["G"][...], dtype=np.float64)
        t_val = float(np.asarray(ds.variables["time"][...], dtype=np.float64).reshape(-1)[0])

    expected_nc_shape = (
        grids.nspecies,
        grids.nhermite,
        grids.nlaguerre,
        grids.nz,
        grids.nakx,
        grids.naky,
        2,
    )
    legacy_full_nc_shape = (
        grids.nspecies,
        grids.nhermite,
        grids.nlaguerre,
        grids.nz,
        grids.nx,
        grids.nyc,
        2,
    )
    c_dtype = jnp.dtype(dtype)
    if raw_g.shape == expected_nc_shape:
        g_complex = float(scale) * (raw_g[..., 0] + 1.0j * raw_g[..., 1])
        g_opt_a = np.swapaxes(g_complex, -2, -1)
        return mask_g(jnp.asarray(g_opt_a, dtype=c_dtype), grids), t_val

    if raw_g.shape == legacy_full_nc_shape:
        g_complex_full = float(scale) * (raw_g[..., 0] + 1.0j * raw_g[..., 1])
        g_nyc_nx = np.swapaxes(g_complex_full, -2, -1)
        g_act = truncate_full_to_kykx(jnp.asarray(g_nyc_nx, dtype=c_dtype), grids)
        return mask_g(g_act, grids), t_val

    raise ValueError(
        f"Restart file {nc_path} has G shape {raw_g.shape}, expected {expected_nc_shape} "
        f"or legacy full shape {legacy_full_nc_shape}."
    )


def save_restart_nc(
    path: str | Path,
    G: jax.Array,
    time: float,
    grids: Grids,
) -> None:
    """Write Option A ``G`` and ``time`` to a GX-compatible ``.restart.nc`` NetCDF-4 file.

    Ported from ``Diagnostics::restart_write`` (``diagnostics.cu:L354-407``) and
    ``MomentsG::restart_write`` (``moments.cu:L494-568``).
    """
    if G.shape != grids.g_shape:
        raise ValueError(f"Expected G shape {grids.g_shape}, got {G.shape}.")

    is_fp64 = np.dtype(G.dtype) == np.dtype(np.complex128)
    real_np_dtype = np.float64 if is_fp64 else np.float32
    nc_g_dtype = "f8" if is_fp64 else "f4"

    g_masked = np.asarray(mask_g(G, grids))
    g_nkx_nky = np.swapaxes(g_masked, -2, -1)
    g_out = np.empty((*g_nkx_nky.shape, 2), dtype=real_np_dtype)
    g_out[..., 0] = np.real(g_nkx_nky).astype(real_np_dtype)
    g_out[..., 1] = np.imag(g_nkx_nky).astype(real_np_dtype)

    nc_path = Path(path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("Nspecies", grids.nspecies)
        ds.createDimension("ri", 2)
        ds.createDimension("Nz", grids.nz)
        ds.createDimension("Nkx", grids.nakx)
        ds.createDimension("Nky", grids.naky)
        ds.createDimension("Nl", grids.nlaguerre)
        ds.createDimension("Nm", grids.nhermite)

        var_g = ds.createVariable(
            "G",
            nc_g_dtype,
            ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"),
        )
        var_time = ds.createVariable("time", "f8", ())
        var_g[...] = g_out
        var_time[...] = float(time)
