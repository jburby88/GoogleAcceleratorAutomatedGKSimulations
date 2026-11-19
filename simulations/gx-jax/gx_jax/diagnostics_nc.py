"""NetCDF4 ``.out.nc`` diagnostics writer and pure-JAX spectral diagnostic calculators.

Ported from:
- ``gx/src/diagnostics.cu``
- ``gx/src/diagnostic_classes.cu``
- ``gx/src/spectra_calc.cu``
- ``gx/src/reductions.cu``
- ``gx/src/ncdf.cu``
- ``gx/src/device_funcs.cu`` (``Phi2_summand``, ``Wg_summand``,
  ``Wphi_summand``, ``Wapar_summand``, ``heat_flux_summand``, ``particle_flux_summand``,
  ``turbulent_heating_summand``, ``growthRates``)

Key design notes:
1. Pure-JAX ``@jax.jit``-compatible spectral reductions for field energies (``Phi2``,
   ``Phi2_zonal``, ``Apar2``, ``Bpar2``), free energies (``Wg``, ``Wphi`` with exact
   Boltzmann electron/ion contribution, ``Wapar``, ``Wbpar``), fluxes (``HeatFlux*``,
   ``ParticleFlux*``), collisional turbulent heating (``TurbulentHeating*``), and
   instantaneous growth rates / frequencies (``omega_kxkyt``).
2. All radial heat and particle fluxes (``HeatFlux*``, ``ParticleFlux*``) are strictly divided
   by ``grho_avg = <|\nabla x|> = sum(J |\nabla x|) / sum(J)`` via
   ``geo.flux_fac = geo.vol_fac / geo.grho_avg`` (``geometry.cu:L955-960``), while collisional
   turbulent heating (``TurbulentHeating*``) uses
   ``geo.vol_fac`` matching ``TurbulentHeatingDiagnostic`` (``diagnostic_classes.cu:L522-524``).
3. Fourier reality weights ``fac(ky) = 1.0`` for ``iky == 0`` and ``2.0`` for ``iky > 0``
   with ``0.0`` at the masked ``(iky=0, ikx=0)`` origin, and monotonic negative-to-positive
   ``kx_out`` / ``kz_out`` index reordering matching ``SpectraCalc_*::dealias_and_reorder``.
4. Exact C++ ``.out.nc`` NetCDF4 group hierarchy (``Grids``, ``Geometry``, ``Diagnostics``,
   ``Inputs`` and its subgroups) and ``.restart.nc`` round-trip compatibility.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from gx_jax.config import BoltzmannOption, GXConfig
from gx_jax.fields import (
    Fields,
    FieldSolver,
    _time_dtype,
    load_restart_nc,
    save_restart_nc,
)
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids
from gx_jax.laguerre import g0 as laguerre_g0
from gx_jax.laguerre import jfac, jflr, jflr_b

# ---------------------------------------------------------------------------
# 1. Index Reordering & Fourier Reality Weight Helpers (@jax.jit safe)
# ---------------------------------------------------------------------------


def reorder_kx_to_out(arr: jax.Array, axis: int = -1) -> jax.Array:
    """Reorder Option A ``kx`` ``[0, +1..+K, -K..-1]`` to monotonic ``[-K..-1, 0, +1..+K]``.

    Matches ``SpectraCalc_*::dealias_and_reorder`` (``spectra_calc.cu:L2585-2602``)
    and ``grids.kx_out`` where ``NK = nakx // 2``.
    """
    nakx = arr.shape[axis]
    nk = nakx // 2
    if nk == 0:
        return arr
    return jnp.roll(arr, shift=nk, axis=axis)


def reorder_kx_from_out(arr: jax.Array, axis: int = -1) -> jax.Array:
    """Invert :func:`reorder_kx_to_out` from monotonic ``kx_out`` back to Option A order."""
    nakx = arr.shape[axis]
    nk = nakx // 2
    if nk == 0:
        return arr
    return jnp.roll(arr, shift=-nk, axis=axis)


def reorder_kz_to_out(arr: jax.Array, axis: int = -1) -> jax.Array:
    """Reorder FFT ``kz`` ``[0, +1..+Nz/2, -(Nz/2-1)..-1]`` to monotonic ``kz_out`` order."""
    nz = arr.shape[axis]
    nkz = (nz - 1) // 2
    if nkz <= 0:
        return arr
    return jnp.roll(arr, shift=nkz, axis=axis)


def fourier_ky_weights(grids: Grids, dtype: jnp.dtype = jnp.float64) -> jax.Array:
    """Return 2D Fourier reality weights ``fac(ky, kx)`` of shape ``(naky, nakx)``.

    Equals ``1.0`` on ``iky == 0``, ``2.0`` on ``iky > 0``, and ``0.0`` at masked modes
    (specifically ``(iky=0, ikx=0)`` and any dealiased modes).
    """
    fac_ky = jnp.where(
        jnp.arange(grids.naky) == 0,
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(2.0, dtype=dtype),
    )[:, None]
    zero = jnp.asarray(0.0, dtype=dtype)
    return jnp.where(grids.active_mask, fac_ky, zero)


def reorder_complex_field_to_nc(field: jax.Array) -> jax.Array:
    """Convert ``field[nz, naky, nakx]`` to C++ ``FieldsDiagnostic`` layout ``(naky, nakx_out, nz,
    2)``.
    """
    f_kx_out = reorder_kx_to_out(field, axis=-1)
    # Transpose (nz, naky, nakx) -> (naky, nakx, nz)
    f_trans = jnp.transpose(f_kx_out, (1, 2, 0))
    return jnp.stack([jnp.real(f_trans), jnp.imag(f_trans)], axis=-1)


def reorder_complex_moment_to_nc(mom: jax.Array) -> jax.Array:
    """Convert ``mom[nspecies, nz, naky, nakx]`` to ``(nspecies, naky, nakx_out, nz, 2)``."""
    m_kx_out = reorder_kx_to_out(mom, axis=-1)
    m_trans = jnp.transpose(m_kx_out, (0, 2, 3, 1))
    return jnp.stack([jnp.real(m_trans), jnp.imag(m_trans)], axis=-1)


# ---------------------------------------------------------------------------
# 2. PyTree Containers for Spectral Diagnostics
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class FieldSpectra:
    """Spectral reductions of a squared field quantity (``Phi2``, ``Apar2``, ``Bpar2``)."""

    total: jax.Array  # () -> _t
    kx: jax.Array  # (nakx,) in kx_out order -> _kxt
    ky: jax.Array  # (naky,) -> _kyt
    kxky: jax.Array  # (naky, nakx) in kx_out order -> _kxkyt
    z: jax.Array  # (nz,) -> _zt
    kxkyz: jax.Array  # (nz, naky, nakx) in kx_out order -> _kxkyzt

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return (self.total, self.kx, self.ky, self.kxky, self.z, self.kxkyz), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[jax.Array, ...]) -> FieldSpectra:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ZonalFieldSpectra:
    """Zonal (``iky == 0``) spectral reductions (``Phi2_zonal_t``, ``_kxt``, ``_zt``)."""

    total: jax.Array  # () -> Phi2_zonal_t
    kx: jax.Array  # (nakx,) in kx_out order -> Phi2_zonal_kxt
    z: jax.Array  # (nz,) -> Phi2_zonal_zt

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return (self.total, self.kx, self.z), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[jax.Array, ...]) -> ZonalFieldSpectra:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class SpeciesSpectra:
    """Per-species and species-summed spectral reductions (``Wphi``, ``Wapar``, fluxes)."""

    total: jax.Array  # () -> _t
    st: jax.Array  # (nspecies,) -> _st
    kxst: jax.Array  # (nspecies, nakx) in kx_out order -> _kxst
    kyst: jax.Array  # (nspecies, naky) -> _kyst
    kxkyst: jax.Array  # (nspecies, naky, nakx) in kx_out order -> _kxkyst
    zst: jax.Array  # (nspecies, nz) -> _zst
    kxkyzst: jax.Array  # (nspecies, nz, naky, nakx) in kx_out order -> _kxkyzst
    kxt: jax.Array  # (nakx,) in kx_out order -> _kxt
    kyt: jax.Array  # (naky,) -> _kyt
    kxkyt: jax.Array  # (naky, nakx) in kx_out order -> _kxkyt
    zt: jax.Array  # (nz,) -> _zt

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return (
            self.total,
            self.st,
            self.kxst,
            self.kyst,
            self.kxkyst,
            self.zst,
            self.kxkyzst,
            self.kxt,
            self.kyt,
            self.kxkyt,
            self.zt,
        ), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[jax.Array, ...]) -> SpeciesSpectra:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class WgSpectra:
    """Hermite-Laguerre free energy ``Wg`` spectral reductions."""

    total: jax.Array  # () -> Wg_t
    st: jax.Array  # (nspecies,) -> Wg_st
    kxst: jax.Array  # (nspecies, nakx) in kx_out order -> Wg_kxst
    kyst: jax.Array  # (nspecies, naky) -> Wg_kyst
    kxkyst: jax.Array  # (nspecies, naky, nakx) in kx_out order -> Wg_kxkyst
    zst: jax.Array  # (nspecies, nz) -> Wg_zst
    lmst: jax.Array  # (nspecies, nhermite, nlaguerre) -> Wg_lmst
    mst: jax.Array  # (nspecies, nhermite) -> Wg_mst
    lst: jax.Array  # (nspecies, nlaguerre) -> Wg_lst
    kxt: jax.Array  # (nakx,) in kx_out order -> Wg_kxt
    kyt: jax.Array  # (naky,) -> Wg_kyt
    kxkyt: jax.Array  # (naky, nakx) in kx_out order -> Wg_kxkyt
    zt: jax.Array  # (nz,) -> Wg_zt
    mt: jax.Array  # (nhermite,) -> Wg_mt
    lt: jax.Array  # (nlaguerre,) -> Wg_lt
    lmt: jax.Array  # (nhermite, nlaguerre) -> Wg_lmt

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return (
            self.total,
            self.st,
            self.kxst,
            self.kyst,
            self.kxkyst,
            self.zst,
            self.lmst,
            self.mst,
            self.lst,
            self.kxt,
            self.kyt,
            self.kxkyt,
            self.zt,
            self.mt,
            self.lt,
            self.lmt,
        ), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[jax.Array, ...]) -> WgSpectra:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class WphiSpectra:
    """Electrostatic free energy ``Wphi`` spectra including exact Boltzmann species terms."""

    total: jax.Array  # () -> Wphi_t (kinetic polarization + Boltzmann species)
    kinetic_total: jax.Array  # () -> sum_s Wphi_st
    boltzmann_total: jax.Array  # () -> adiabatic Boltzmann species contribution
    st: jax.Array  # (nspecies,) -> Wphi_st (exact C++ Wphi_summand per kinetic species)
    kxst: jax.Array  # (nspecies, nakx) in kx_out order -> Wphi_kxst
    kyst: jax.Array  # (nspecies, naky) -> Wphi_kyst
    kxkyst: jax.Array  # (nspecies, naky, nakx) in kx_out order -> Wphi_kxkyst
    zst: jax.Array  # (nspecies, nz) -> Wphi_zst
    kxt: jax.Array  # (nakx,) in kx_out order -> Wphi_kxt (including Boltzmann term)
    kyt: jax.Array  # (naky,) -> Wphi_kyt (including Boltzmann term)
    kxkyt: jax.Array  # (naky, nakx) in kx_out order -> Wphi_kxkyt (including Boltzmann term)
    zt: jax.Array  # (nz,) -> Wphi_zt (including Boltzmann term)
    boltzmann_kxkyt: jax.Array  # (naky, nakx) in kx_out order

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return (
            self.total,
            self.kinetic_total,
            self.boltzmann_total,
            self.st,
            self.kxst,
            self.kyst,
            self.kxkyst,
            self.zst,
            self.kxt,
            self.kyt,
            self.kxkyt,
            self.zt,
            self.boltzmann_kxkyt,
        ), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[jax.Array, ...]) -> WphiSpectra:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class FluxBundle:
    """Total and component (ES, Apar, Bpar) spectral reductions for fluxes or heating."""

    total: SpeciesSpectra
    es: SpeciesSpectra
    apar: SpeciesSpectra
    bpar: SpeciesSpectra

    def tree_flatten(self) -> tuple[tuple[SpeciesSpectra, ...], None]:
        return (self.total, self.es, self.apar, self.bpar), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[SpeciesSpectra, ...]) -> FluxBundle:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DiagnosticsSnapshot:
    """Complete single-step diagnostic snapshot computed by :func:`compute_diagnostics_step`."""

    time: jax.Array
    dt: jax.Array
    phi2: FieldSpectra
    phi2_zonal: ZonalFieldSpectra
    apar2: FieldSpectra
    bpar2: FieldSpectra
    wg: WgSpectra
    wphi: WphiSpectra
    wapar: SpeciesSpectra
    wbpar: SpeciesSpectra
    heat_flux: FluxBundle
    particle_flux: FluxBundle
    turbulent_heating: FluxBundle
    omega_kxkyt: jax.Array  # (naky, nakx_out, 2) [omega, gamma]

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (
            self.time,
            self.dt,
            self.phi2,
            self.phi2_zonal,
            self.apar2,
            self.bpar2,
            self.wg,
            self.wphi,
            self.wapar,
            self.wbpar,
            self.heat_flux,
            self.particle_flux,
            self.turbulent_heating,
            self.omega_kxkyt,
        ), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Any, ...]) -> DiagnosticsSnapshot:
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DiagnosticsState:
    """Carry state tracking previous ``G`` and ``Fields`` for ``omega_kxkyt`` and heating."""

    G_old: jax.Array
    fields_old: Fields
    time_old: jax.Array
    has_old: jax.Array  # bool scalar

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (self.G_old, self.fields_old, self.time_old, self.has_old), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Any, ...]) -> DiagnosticsState:
        del aux
        return cls(*children)


def init_diagnostics_state(
    G: jax.Array,
    fields: Fields,
    time: float | jax.Array = 0.0,
    *,
    has_old: bool = False,
) -> DiagnosticsState:
    """Initialize :class:`DiagnosticsState` carry.

    In C++ ``Diagnostics_GK`` (``diagnostics.cu:L918-923``), ``G_old`` and ``fields_old``
    are zero-initialized prior to step 1 unless initialized from an existing state.
    When ``has_old=True``, ``G`` and ``fields`` are stored immediately as the reference
    state for the next step's ``omega_kxkyt`` and ``TurbulentHeating`` finite differences.
    """
    if has_old:
        g_prev = G
        f_prev = fields
    else:
        g_prev = jnp.zeros_like(G)
        f_prev = Fields(
            phi=jnp.zeros_like(fields.phi),
            apar=jnp.zeros_like(fields.apar),
            bpar=jnp.zeros_like(fields.bpar),
            phi_avg=jnp.zeros_like(fields.phi_avg),
        )
    t_dtype = _time_dtype(jnp.real(G).dtype)
    return DiagnosticsState(
        G_old=g_prev,
        fields_old=f_prev,
        time_old=jnp.asarray(time, dtype=t_dtype),
        has_old=jnp.asarray(has_old, dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# 3. Pure-JAX Spectral Reduction Kernels (@jax.jit safe)
# ---------------------------------------------------------------------------


def _reduce_field_summand_3d(summand_zyx: jax.Array) -> FieldSpectra:
    """Reduce a 3D summand ``(nz, naky, nakx)`` in Option A layout to :class:`FieldSpectra`."""
    kxkyz = reorder_kx_to_out(summand_zyx, axis=-1)
    kxky = jnp.sum(kxkyz, axis=0)
    kx = jnp.sum(kxky, axis=0)
    ky = jnp.sum(kxky, axis=1)
    z = jnp.sum(kxkyz, axis=(1, 2))
    total = jnp.sum(kx)
    return FieldSpectra(total=total, kx=kx, ky=ky, kxky=kxky, z=z, kxkyz=kxkyz)


def _reduce_species_summand_4d(summand_szyx: jax.Array) -> SpeciesSpectra:
    """Reduce a 4D summand ``(nspecies, nz, naky, nakx)`` to :class:`SpeciesSpectra`."""
    kxkyzst = reorder_kx_to_out(summand_szyx, axis=-1)
    kxkyst = jnp.sum(kxkyzst, axis=1)  # (s, ky, kx_out)
    kxst = jnp.sum(kxkyst, axis=1)  # (s, kx_out)
    kyst = jnp.sum(kxkyst, axis=2)  # (s, ky)
    zst = jnp.sum(kxkyzst, axis=(2, 3))  # (s, z)
    st = jnp.sum(kxst, axis=1)  # (s,)
    kxt = jnp.sum(kxst, axis=0)  # (kx_out,)
    kyt = jnp.sum(kyst, axis=0)  # (ky,)
    kxkyt = jnp.sum(kxkyst, axis=0)  # (ky, kx_out)
    zt = jnp.sum(zst, axis=0)  # (z,)
    total = jnp.sum(st)
    return SpeciesSpectra(
        total=total,
        st=st,
        kxst=kxst,
        kyst=kyst,
        kxkyst=kxkyst,
        zst=zst,
        kxkyzst=kxkyzst,
        kxt=kxt,
        kyt=kyt,
        kxkyt=kxkyt,
        zt=zt,
    )


def compute_field_squared_spectra(
    field: jax.Array,
    grids: Grids,
    geo: Geometry,
) -> FieldSpectra:
    """Compute ``Phi2`` / ``Apar2`` / ``Bpar2`` spectra matching ``Phi2_summand``
    (``device_funcs.cu:L1727-1744``).

    Summand: ``0.5 * fac(ky, kx) * vol_fac(z) * |field(z, ky, kx)|^2``.
    """
    real_dtype = jnp.real(field).dtype
    fac = fourier_ky_weights(grids, dtype=real_dtype)[None, :, :]
    vol = geo.vol_fac.astype(real_dtype)[:, None, None]
    summand = 0.5 * fac * vol * (jnp.abs(field) ** 2)
    return _reduce_field_summand_3d(summand)


def compute_phi2_zonal_spectra(
    phi: jax.Array,
    grids: Grids,
    geo: Geometry,
) -> ZonalFieldSpectra:
    """Compute ``Phi2_zonal`` spectra matching ``Phi2_zonal_summand``
    (``device_funcs.cu:L1746-1766``).

    Summand on ``iky == 0``: ``0.5 * vol_fac(z) * |phi(z, 0, kx)|^2`` for active ``kx != 0``.
    """
    real_dtype = jnp.real(phi).dtype
    mask_zonal = grids.active_mask[0, :]  # (nakx,)
    vol = geo.vol_fac.astype(real_dtype)[:, None]  # (nz, 1)
    phi_zonal = jnp.where(mask_zonal[None, :], phi[:, 0, :], jnp.asarray(0.0, dtype=phi.dtype))
    summand_zx = 0.5 * vol * (jnp.abs(phi_zonal) ** 2)  # (nz, nakx)
    summand_zx_out = reorder_kx_to_out(summand_zx, axis=-1)
    kx = jnp.sum(summand_zx_out, axis=0)
    z = jnp.sum(summand_zx_out, axis=1)
    total = jnp.sum(kx)
    return ZonalFieldSpectra(total=total, kx=kx, z=z)


def compute_wg_spectra(
    G: jax.Array,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
) -> WgSpectra:
    """Compute Hermite-Laguerre free energy ``Wg`` spectra matching ``Wg_summand``
    (``device_funcs.cu:L1691-1711``).

    Summand: ``0.5 * fac(ky, kx) * vol_fac(z) * nt_s * |G(s, m, l, z, ky, kx)|^2``.
    """
    real_dtype = jnp.real(G).dtype
    fac = fourier_ky_weights(grids, dtype=real_dtype)[None, None, None, None, :, :]
    vol = geo.vol_fac.astype(real_dtype)[None, None, None, :, None, None]
    nt = solver.sp_nt.astype(real_dtype)[:, None, None, None, None, None]
    wg_6d = 0.5 * fac * vol * nt * (jnp.abs(G) ** 2)

    # Reduce over (m, l) first for spatial/spectral reductions, and over (z, ky, kx) for moment
    # reductions
    wg_szyx = jnp.sum(wg_6d, axis=(1, 2))  # (s, z, ky, kx)
    sp_spec = _reduce_species_summand_4d(wg_szyx)

    lmst = jnp.sum(wg_6d, axis=(3, 4, 5))  # (s, m, l)
    mst = jnp.sum(lmst, axis=2)  # (s, m)
    lst = jnp.sum(lmst, axis=1)  # (s, l)
    lmt = jnp.sum(lmst, axis=0)  # (m, l)
    mt = jnp.sum(mst, axis=0)  # (m,)
    lt = jnp.sum(lst, axis=0)  # (l,)

    return WgSpectra(
        total=sp_spec.total,
        st=sp_spec.st,
        kxst=sp_spec.kxst,
        kyst=sp_spec.kyst,
        kxkyst=sp_spec.kxkyst,
        zst=sp_spec.zst,
        lmst=lmst,
        mst=mst,
        lst=lst,
        kxt=sp_spec.kxt,
        kyt=sp_spec.kyt,
        kxkyt=sp_spec.kxkyt,
        zt=sp_spec.zt,
        mt=mt,
        lt=lt,
        lmt=lmt,
    )


def compute_wphi_spectra(
    phi: jax.Array,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
    *,
    phi_avg: jax.Array | None = None,
    add_boltzmann_species: bool | None = None,
    boltzmann_opt: BoltzmannOption | int | None = None,
    tau_fac: float | None = None,
) -> WphiSpectra:
    """Compute electrostatic free energy ``Wphi`` spectra (``device_funcs.cu:L1768-1790``).

    - Per-species arrays (``st``, ``kxst``, ``kyst``, ``kxkyst``, ``zst``) match C++
      ``WphiDiagnostic``: ``0.5 * fac(ky, kx) * vol_fac(z) * (1 - g0(b_s)) * |phi|^2``.
    - Total arrays (``total``, ``kxt``, ``kyt``, ``kxkyt``, ``zt``) sum the kinetic species
      polarization energy ``sum_s (z_s^2 n_s / T_s) * Wphi_s`` plus the exact Boltzmann
      electron/ion contribution:
      * ``BOLTZMANN_ELECTRONS``: ``0.5 * fac * vol_fac * tau_fac * |phi - delta_{ky,0} <phi>|^2``
      * ``BOLTZMANN_IONS``: ``0.5 * fac * vol_fac * tau_fac * |phi|^2``
    """
    real_dtype = jnp.real(phi).dtype
    fac = fourier_ky_weights(grids, dtype=real_dtype)[None, :, :]  # (1, naky, nakx)
    vol = geo.vol_fac.astype(real_dtype)[:, None, None]  # (nz, 1, 1)
    kperp2 = geo.kperp2.astype(real_dtype)  # (nz, naky, nakx)
    rho2 = solver.sp_rho2.astype(real_dtype)[:, None, None, None]  # (s, 1, 1, 1)
    b_s = kperp2[None, ...] * rho2
    one_minus_g0 = 1.0 - laguerre_g0(b_s)

    phi2 = jnp.abs(phi) ** 2  # (nz, naky, nakx)
    summand_szyx = 0.5 * fac[None, ...] * vol[None, ...] * one_minus_g0 * phi2[None, ...]
    kin_spec = _reduce_species_summand_4d(summand_szyx)

    # Physical species weighting zn_s = z_s^2 n_s / T_s for total electrostatic free energy
    zn_s = (solver.sp_zt * solver.sp_nz).astype(real_dtype)[:, None, None, None]
    phys_kin_zyx = jnp.sum(zn_s * summand_szyx, axis=0)  # (nz, naky, nakx)

    use_boltz = (
        not solver.all_kinetic if add_boltzmann_species is None else bool(add_boltzmann_species)
    )
    b_opt = solver.boltzmann_opt if boltzmann_opt is None else BoltzmannOption(int(boltzmann_opt))
    tfac = jnp.asarray(solver.tau_fac if tau_fac is None else tau_fac, dtype=real_dtype)

    if use_boltz:
        if b_opt == BoltzmannOption.BOLTZMANN_ELECTRONS:
            if phi_avg is None:
                # Field-line average <phi> on iky == 0
                phi_avg_x = jnp.sum(vol[:, 0, :] * phi[:, 0, :], axis=0)
            else:
                phi_avg_x = phi_avg
            is_ky0 = (jnp.arange(grids.naky) == 0)[None, :, None]
            phi_eff = jnp.where(is_ky0, phi - phi_avg_x[None, None, :], phi)
            boltz_zyx = 0.5 * fac * vol * tfac * (jnp.abs(phi_eff) ** 2)
        else:
            boltz_zyx = 0.5 * fac * vol * tfac * phi2
    else:
        boltz_zyx = jnp.zeros_like(phys_kin_zyx)

    boltz_spec = _reduce_field_summand_3d(boltz_zyx)
    tot_spec = _reduce_field_summand_3d(phys_kin_zyx + boltz_zyx)

    return WphiSpectra(
        total=tot_spec.total,
        kinetic_total=kin_spec.total,
        boltzmann_total=boltz_spec.total,
        st=kin_spec.st,
        kxst=kin_spec.kxst,
        kyst=kin_spec.kyst,
        kxkyst=kin_spec.kxkyst,
        zst=kin_spec.zst,
        kxt=tot_spec.kx,
        kyt=tot_spec.ky,
        kxkyt=tot_spec.kxky,
        zt=tot_spec.z,
        boltzmann_kxkyt=boltz_spec.kxky,
    )


def compute_wapar_spectra(
    apar: jax.Array,
    grids: Grids,
    geo: Geometry,
    nspecies: int,
) -> SpeciesSpectra:
    """Compute magnetic flutter free energy ``Wapar`` spectra matching ``Wapar_summand``
    (``device_funcs.cu:L1792-1815``).

    Per-species summand: ``0.5 * fac(ky, kx) * vol_fac(z) * kperp2(z, ky, kx) * bmag(z)^2 *
    |apar|^2``.
    Note: C++ writes the same ``Wapar`` field energy into each species slot ``is = 0..nspecies-1``;
    the physical species-independent magnetic energy is ``Wapar_st[0]`` (stored in
    ``kxt/kyt/kxkyt/zt/total``).
    """
    real_dtype = jnp.real(apar).dtype
    fac = fourier_ky_weights(grids, dtype=real_dtype)[None, :, :]
    vol = geo.vol_fac.astype(real_dtype)[:, None, None]
    kp2_b2 = geo.kperp2.astype(real_dtype) * (geo.bmag.astype(real_dtype)[:, None, None] ** 2)
    summand_zyx = 0.5 * fac * vol * kp2_b2 * (jnp.abs(apar) ** 2)
    field_spec = _reduce_field_summand_3d(summand_zyx)
    summand_szyx = jnp.broadcast_to(summand_zyx[None, ...], (nspecies, *summand_zyx.shape))
    sp_spec = _reduce_species_summand_4d(summand_szyx)
    return SpeciesSpectra(
        total=field_spec.total,
        st=sp_spec.st,
        kxst=sp_spec.kxst,
        kyst=sp_spec.kyst,
        kxkyst=sp_spec.kxkyst,
        zst=sp_spec.zst,
        kxkyzst=sp_spec.kxkyzst,
        kxt=field_spec.kx,
        kyt=field_spec.ky,
        kxkyt=field_spec.kxky,
        zt=field_spec.z,
    )


def compute_wbpar_spectra(
    bpar: jax.Array,
    grids: Grids,
    geo: Geometry,
    nspecies: int,
    beta: float = 0.0,
) -> SpeciesSpectra:
    """Compute compressional magnetic free energy ``Wbpar`` spectra.

    Summand: ``0.5 * fac(ky, kx) * vol_fac(z) * |bpar|^2 / beta`` when ``beta > 0``
    (and ``0.0`` when ``beta <= 0`` and ``bpar == 0``, or ``1.0`` weight if ``beta <= 0``).
    """
    real_dtype = jnp.real(bpar).dtype
    fac = fourier_ky_weights(grids, dtype=real_dtype)[None, :, :]
    vol = geo.vol_fac.astype(real_dtype)[:, None, None]
    inv_beta = jnp.where(
        jnp.asarray(beta, dtype=real_dtype) > 0.0,
        1.0
        / jnp.maximum(jnp.asarray(beta, dtype=real_dtype), jnp.asarray(1.0e-30, dtype=real_dtype)),
        jnp.asarray(1.0, dtype=real_dtype),
    )
    summand_zyx = 0.5 * fac * vol * inv_beta * (jnp.abs(bpar) ** 2)
    field_spec = _reduce_field_summand_3d(summand_zyx)
    summand_szyx = jnp.broadcast_to(summand_zyx[None, ...], (nspecies, *summand_zyx.shape))
    sp_spec = _reduce_species_summand_4d(summand_szyx)
    return SpeciesSpectra(
        total=field_spec.total,
        st=sp_spec.st,
        kxst=sp_spec.kxst,
        kyst=sp_spec.kyst,
        kxkyst=sp_spec.kxkyst,
        zst=sp_spec.zst,
        kxkyzst=sp_spec.kxkyzst,
        kxt=field_spec.kx,
        kyt=field_spec.ky,
        kxkyt=field_spec.kxky,
        zt=field_spec.z,
    )


def _get_hermite_slice(G: jax.Array, m_idx: int) -> jax.Array:
    """Safely extract ``G[:, m_idx]`` of shape ``(s, l, z, ky, kx)`` or zeros if ``nhermite <=
    m_idx``.
    """
    if G.shape[1] > m_idx:
        return G[:, m_idx]
    return jnp.zeros((G.shape[0], G.shape[2], G.shape[3], G.shape[4], G.shape[5]), dtype=G.dtype)


def compute_heat_flux_spectra(
    G: jax.Array,
    fields: Fields,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
) -> FluxBundle:
    """Compute gyroBohm turbulent heat flux ``HeatFlux*`` spectra (``device_funcs.cu:L1872-2014``).

    Strictly divides by ``grho_avg = <|\nabla x|>`` via ``geo.flux_fac = geo.vol_fac /
    geo.grho_avg``
    per ``geometry.cu:L955-960``.
    """
    real_dtype = jnp.real(G).dtype
    ns, _, nl, nz, naky, nakx = G.shape
    ky_3d = grids.ky.astype(real_dtype)[None, :, None]  # (1, naky, 1)
    iky_3d = jax.lax.complex(jnp.zeros_like(ky_3d), ky_3d)
    vphi_r = iky_3d * fields.phi  # (nz, naky, nakx)
    va_r = iky_3d * fields.apar
    vb_r = iky_3d * fields.bpar

    kperp2 = geo.kperp2.astype(real_dtype)[None, None, :, :, :]  # (1, 1, nz, naky, nakx)
    rho2 = solver.sp_rho2.astype(real_dtype)[:, None, None, None, None]  # (s, 1, 1, 1, 1)
    b_s = kperp2 * rho2  # (s, 1, nz, naky, nakx)
    l_idx = jnp.arange(nl, dtype=real_dtype)[None, :, None, None, None]  # (1, l, 1, 1, 1)

    jf_l = jfac(l_idx, b_s)  # (s, l, nz, naky, nakx)
    jf_lm1 = jfac(l_idx - 1.0, b_s)
    j0_l = jflr(l_idx, b_s)
    jb_l = jflr_b(l_idx, b_s)

    g_m0 = _get_hermite_slice(G, 0)
    g_m1 = _get_hermite_slice(G, 1)
    g_m2 = _get_hermite_slice(G, 2)
    g_m3 = _get_hermite_slice(G, 3)

    inv_sqrt2 = jnp.asarray(1.0 / np.sqrt(2.0), dtype=real_dtype)
    sqrt_1p5 = jnp.asarray(np.sqrt(1.5), dtype=real_dtype)

    p_bar = jnp.sum(jf_l * g_m0 + inv_sqrt2 * j0_l * g_m2, axis=1)  # (s, nz, naky, nakx)
    q_bar = jnp.sum(jf_l * g_m1 + j0_l * (sqrt_1p5 * g_m3 + g_m1), axis=1)
    qb_bar = jnp.sum((jf_l + jf_lm1) * g_m0 + inv_sqrt2 * jb_l * g_m2, axis=1)

    # Active ky > 0 mask (on ky == 0, vPhi_r = i * 0 * phi == 0 anyway, and unmasked enforced)
    ky_gt0_mask = (jnp.arange(naky) > 0)[None, None, :, None] & grids.active_mask[None, None, :, :]
    flx_jac = geo.flux_fac.astype(real_dtype)[None, :, None, None]  # (1, nz, 1, 1)
    pres = solver.sp_nt.astype(real_dtype)[:, None, None, None]  # (s, 1, 1, 1)
    vt = solver.sp_vt.astype(real_dtype)[:, None, None, None]
    tz = solver.sp_tz.astype(real_dtype)[:, None, None, None]
    two = jnp.asarray(2.0, dtype=real_dtype)
    zero = jnp.asarray(0.0, dtype=real_dtype)

    q_es = jnp.where(
        ky_gt0_mask,
        two * flx_jac * pres * jnp.real(jnp.conj(vphi_r)[None, ...] * p_bar),
        zero,
    )
    q_apar = jnp.where(
        ky_gt0_mask,
        two * flx_jac * pres * jnp.real(-vt * jnp.conj(va_r)[None, ...] * q_bar),
        zero,
    )
    q_bpar = jnp.where(
        ky_gt0_mask,
        two * flx_jac * pres * jnp.real(tz * jnp.conj(vb_r)[None, ...] * qb_bar),
        zero,
    )
    q_tot = q_es + q_apar + q_bpar

    return FluxBundle(
        total=_reduce_species_summand_4d(q_tot),
        es=_reduce_species_summand_4d(q_es),
        apar=_reduce_species_summand_4d(q_apar),
        bpar=_reduce_species_summand_4d(q_bpar),
    )


def compute_particle_flux_spectra(
    G: jax.Array,
    fields: Fields,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
    *,
    zero_single_species: bool = True,
) -> FluxBundle:
    """Compute gyroBohm turbulent particle flux ``ParticleFlux*`` spectra
    (``device_funcs.cu:L2016-2170``).

    Strictly divides by ``grho_avg = <|\nabla x|>`` via ``geo.flux_fac = geo.vol_fac /
    geo.grho_avg``
    per ``geometry.cu:L955-960``.
    """
    real_dtype = jnp.real(G).dtype
    ns, _, nl, _, naky, _ = G.shape
    ky_3d = grids.ky.astype(real_dtype)[None, :, None]
    iky_3d = jax.lax.complex(jnp.zeros_like(ky_3d), ky_3d)
    vphi_r = iky_3d * fields.phi
    va_r = iky_3d * fields.apar
    vb_r = iky_3d * fields.bpar

    kperp2 = geo.kperp2.astype(real_dtype)[None, None, :, :, :]
    rho2 = solver.sp_rho2.astype(real_dtype)[:, None, None, None, None]
    b_s = kperp2 * rho2
    l_idx = jnp.arange(nl, dtype=real_dtype)[None, :, None, None, None]

    j0_l = jflr(l_idx, b_s)
    jb_l = jflr_b(l_idx, b_s)

    g_m0 = _get_hermite_slice(G, 0)
    g_m1 = _get_hermite_slice(G, 1)

    n_bar = jnp.sum(j0_l * g_m0, axis=1)  # (s, nz, naky, nakx)
    u_bar = jnp.sum(j0_l * g_m1, axis=1)
    ub_bar = jnp.sum(jb_l * g_m0, axis=1)

    ky_gt0_mask = (jnp.arange(naky) > 0)[None, None, :, None] & grids.active_mask[None, None, :, :]
    flx_jac = geo.flux_fac.astype(real_dtype)[None, :, None, None]
    if zero_single_species and ns == 1:
        dens = jnp.zeros((ns, 1, 1, 1), dtype=real_dtype)
    else:
        dens = solver.sp_dens.astype(real_dtype)[:, None, None, None]
    vt = solver.sp_vt.astype(real_dtype)[:, None, None, None]
    tz = solver.sp_tz.astype(real_dtype)[:, None, None, None]
    two = jnp.asarray(2.0, dtype=real_dtype)
    zero = jnp.asarray(0.0, dtype=real_dtype)

    p_es = jnp.where(
        ky_gt0_mask,
        two * flx_jac * dens * jnp.real(jnp.conj(vphi_r)[None, ...] * n_bar),
        zero,
    )
    p_apar = jnp.where(
        ky_gt0_mask,
        two * flx_jac * dens * jnp.real(-vt * jnp.conj(va_r)[None, ...] * u_bar),
        zero,
    )
    p_bpar = jnp.where(
        ky_gt0_mask,
        two * flx_jac * dens * jnp.real(tz * jnp.conj(vb_r)[None, ...] * ub_bar),
        zero,
    )
    p_tot = p_es + p_apar + p_bpar

    return FluxBundle(
        total=_reduce_species_summand_4d(p_tot),
        es=_reduce_species_summand_4d(p_es),
        apar=_reduce_species_summand_4d(p_apar),
        bpar=_reduce_species_summand_4d(p_bpar),
    )


def compute_turbulent_heating_spectra(
    G: jax.Array,
    fields: Fields,
    G_old: jax.Array,
    fields_old: Fields,
    dt: float | jax.Array,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
    *,
    use_flux_fac: bool = False,
) -> FluxBundle:
    """Compute turbulent heating ``TurbulentHeating*`` spectra (``device_funcs.cu:L1890-1951``).

    Uses ``geo.vol_fac`` by default (``use_flux_fac=False``) to match C++
    ``TurbulentHeatingDiagnostic`` (``diagnostic_classes.cu:L522-524``); pass
    ``use_flux_fac=True`` only if flux-surface-gradient normalization ``geo.flux_fac`` is desired.
    Returns a :class:`FluxBundle` containing ``total``, ``es``, ``apar``, and ``bpar`` components.
    """
    real_dtype = jnp.real(G).dtype
    _, _, nl, _, _, _ = G.shape
    dt_safe = jnp.where(
        jnp.abs(jnp.asarray(dt, dtype=real_dtype)) > 1.0e-30,
        jnp.asarray(dt, dtype=real_dtype),
        jnp.asarray(1.0, dtype=real_dtype),
    )

    dphidt = (fields.phi - fields_old.phi) / dt_safe
    dadt = (fields.apar - fields_old.apar) / dt_safe
    dbdt = (fields.bpar - fields_old.bpar) / dt_safe

    kperp2 = geo.kperp2.astype(real_dtype)[None, None, :, :, :]
    rho2 = solver.sp_rho2.astype(real_dtype)[:, None, None, None, None]
    b_s = kperp2 * rho2
    l_idx = jnp.arange(nl, dtype=real_dtype)[None, :, None, None, None]

    j0_l = jflr(l_idx, b_s)
    jb_l = jflr_b(l_idx, b_s)

    zt = solver.sp_zt.astype(real_dtype)[:, None, None, None, None]
    vt = solver.sp_vt.astype(real_dtype)[:, None, None, None, None]

    g_m0 = _get_hermite_slice(G, 0)
    g_m1 = _get_hermite_slice(G, 1)
    g_m0_old = _get_hermite_slice(G_old, 0)
    g_m1_old = _get_hermite_slice(G_old, 1)

    h_l0 = g_m0 + zt * j0_l * fields.phi[None, None, ...] + jb_l * fields.bpar[None, None, ...]
    h_l1 = g_m1 - zt * vt * j0_l * fields.apar[None, None, ...]
    h_l0_old = (
        g_m0_old
        + zt * j0_l * fields_old.phi[None, None, ...]
        + jb_l * fields_old.bpar[None, None, ...]
    )
    h_l1_old = g_m1_old - zt * vt * j0_l * fields_old.apar[None, None, ...]

    n_bar = jnp.sum(j0_l * h_l0, axis=1)
    u_bar = jnp.sum(j0_l * h_l1, axis=1)
    ub_bar = jnp.sum(jb_l * h_l0, axis=1)

    n_bar_old = jnp.sum(j0_l * h_l0_old, axis=1)
    u_bar_old = jnp.sum(j0_l * h_l1_old, axis=1)
    ub_bar_old = jnp.sum(jb_l * h_l0_old, axis=1)

    dn_bardt = (n_bar - n_bar_old) / dt_safe
    du_bardt = (u_bar - u_bar_old) / dt_safe
    dub_bardt = (ub_bar - ub_bar_old) / dt_safe

    fac = fourier_ky_weights(grids, dtype=real_dtype)[None, None, :, :]
    jac_w = (geo.flux_fac if use_flux_fac else geo.vol_fac).astype(real_dtype)[None, :, None, None]
    nz_s = solver.sp_nz.astype(real_dtype)[:, None, None, None]
    vt_s = solver.sp_vt.astype(real_dtype)[:, None, None, None]
    tz_s = solver.sp_tz.astype(real_dtype)[:, None, None, None]
    prefac = 0.5 * fac * jac_w * nz_s

    es_term = prefac * jnp.real(
        jnp.conj(dphidt)[None, ...] * n_bar_old - jnp.conj(fields_old.phi)[None, ...] * dn_bardt
    )
    apar_term = prefac * jnp.real(
        -vt_s
        * (jnp.conj(dadt)[None, ...] * u_bar_old - jnp.conj(fields_old.apar)[None, ...] * du_bardt)
    )
    bpar_term = prefac * jnp.real(
        tz_s
        * (
            jnp.conj(dbdt)[None, ...] * ub_bar_old
            - jnp.conj(fields_old.bpar)[None, ...] * dub_bardt
        )
    )
    tot_term = es_term + apar_term + bpar_term

    return FluxBundle(
        total=_reduce_species_summand_4d(tot_term),
        es=_reduce_species_summand_4d(es_term),
        apar=_reduce_species_summand_4d(apar_term),
        bpar=_reduce_species_summand_4d(bpar_term),
    )


def compute_omega_kxky(
    phi: jax.Array,
    phi_old: jax.Array,
    dt: float | jax.Array,
    grids: Grids,
) -> jax.Array:
    """Compute instantaneous ``omega_kxkyt`` of shape ``(naky, nakx_out, 2)``
    (``device_funcs.cu:L863-889``).

    Samples ``phi`` at ``iz = (nz // 2) + 1`` when ``nz > 1`` (or ``0`` when ``nz == 1``),
    computes ``omega_complex = (i / dt) * log(phi / phi_old) = (-arg(r)/dt) + i (log|r|/dt)``,
    and reorders ``kx`` to monotonic ``kx_out`` order ``[-K..+K]`` with trailing ``ri=2``
    storing ``[omega, gamma]``.
    """
    real_dtype = jnp.real(phi).dtype
    iz = (grids.nz // 2 + 1) if grids.nz > 1 else 0
    p_now = phi[iz]  # (naky, nakx)
    p_prev = phi_old[iz]

    dt_safe = jnp.where(
        jnp.abs(jnp.asarray(dt, dtype=real_dtype)) > 1.0e-30,
        jnp.asarray(dt, dtype=real_dtype),
        jnp.asarray(1.0, dtype=real_dtype),
    )
    valid = (
        grids.active_mask
        & (jnp.real(p_now) != 0.0)
        & (jnp.imag(p_now) != 0.0)
        & ((jnp.real(p_prev) != 0.0) | (jnp.imag(p_prev) != 0.0))
    )
    p_prev_safe = jnp.where(valid, p_prev, jnp.asarray(1.0 + 0.0j, dtype=phi.dtype))
    p_now_safe = jnp.where(valid, p_now, jnp.asarray(1.0 + 0.0j, dtype=phi.dtype))
    ratio = p_now_safe / p_prev_safe

    omega_real = jnp.where(
        valid,
        -jnp.arctan2(jnp.imag(ratio), jnp.real(ratio)) / dt_safe,
        jnp.asarray(0.0, dtype=real_dtype),
    )
    gamma_imag = jnp.where(
        valid,
        jnp.log(jnp.maximum(jnp.abs(ratio), jnp.asarray(1.0e-37, dtype=real_dtype))) / dt_safe,
        jnp.asarray(0.0, dtype=real_dtype),
    )
    omega_ri = jnp.stack([omega_real, gamma_imag], axis=-1)  # (naky, nakx, 2)
    return reorder_kx_to_out(omega_ri, axis=1)


def compute_moments_diagnostic_arrays(
    G: jax.Array,
    fields: Fields,
    geo: Geometry,
    solver: FieldSolver,
) -> dict[str, jax.Array]:
    """Compute ``Density``, ``Upar``, ``Tpar``, ``Tperp``, and gyroaveraged particle moments.

    Returns arrays in C++ ``MomentsDiagnostic`` NetCDF layout ``(nspecies, naky, nakx_out, nz, 2)``.
    """
    real_dtype = jnp.real(G).dtype
    _, _, nl, _, _, _ = G.shape
    g_m0_l0 = _get_hermite_slice(G, 0)[:, 0]  # (s, z, ky, kx)
    g_m1_l0 = _get_hermite_slice(G, 1)[:, 0]
    g_m2_l0 = _get_hermite_slice(G, 2)[:, 0]
    sqrt2 = jnp.asarray(np.sqrt(2.0), dtype=real_dtype)

    out: dict[str, jax.Array] = {
        "Density": reorder_complex_moment_to_nc(g_m0_l0),
        "Upar": reorder_complex_moment_to_nc(g_m1_l0),
        "Tpar": reorder_complex_moment_to_nc(sqrt2 * g_m2_l0),
    }
    if nl > 1:
        g_m0_l1 = _get_hermite_slice(G, 0)[:, 1]
        out["Tperp"] = reorder_complex_moment_to_nc(g_m0_l1)
    return out


def compute_diagnostics_step(
    G: jax.Array,
    fields: Fields,
    diag_state: DiagnosticsState,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver,
    *,
    time: float | jax.Array,
    dt: float | jax.Array,
    beta: float = 0.0,
) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
    """Compute a full :class:`DiagnosticsSnapshot` and update :class:`DiagnosticsState`."""
    t_dtype = _time_dtype(jnp.real(G).dtype)
    t_arr = jnp.asarray(time, dtype=t_dtype)
    dt_arr = jnp.asarray(dt, dtype=t_dtype)

    phi2 = compute_field_squared_spectra(fields.phi, grids, geo)
    phi2_zonal = compute_phi2_zonal_spectra(fields.phi, grids, geo)
    apar2 = compute_field_squared_spectra(fields.apar, grids, geo)
    bpar2 = compute_field_squared_spectra(fields.bpar, grids, geo)

    wg = compute_wg_spectra(G, grids, geo, solver)
    wphi = compute_wphi_spectra(fields.phi, grids, geo, solver, phi_avg=fields.phi_avg)
    wapar = compute_wapar_spectra(fields.apar, grids, geo, G.shape[0])
    wbpar = compute_wbpar_spectra(fields.bpar, grids, geo, G.shape[0], beta=beta)

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


# ---------------------------------------------------------------------------
# 4. C++ .out.nc-Compatible NetCDF4 Writer (DiagnosticsWriter)
# ---------------------------------------------------------------------------


_SCHEME_TO_DUM: dict[str, int] = {
    "k10": 0,
    "rk4": 1,
    "rk3": 2,
    "sspx3": 5,
    "sspx2": 6,
}

_BOUNDARY_TO_DUM: dict[str, int] = {
    "periodic": 0,
    "linked": 1,
    "fix aspect": 2,
    "ntft": 3,
}

_GEO_TO_IGEO: dict[str, int] = {
    "s-alpha": 0,
    "miller": 1,
    "nc": 2,
    "vmec": 3,
    "desc": 4,
    "slab": 0,
}


class DiagnosticsWriter:
    """NetCDF4 ``.out.nc`` writer matching C++ GX ``NetCDF`` + ``Diagnostics_GK`` schema."""

    def __init__(
        self,
        path: str | Path,
        cfg: GXConfig,
        grids: Grids,
        geo: Geometry,
        *,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.grids = grids
        self.geo = geo
        self.time_index: int = 0
        self._closed: bool = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_append = bool(append and self.path.is_file())
        mode = "a" if is_append else "w"
        self.ds = netCDF4.Dataset(str(self.path), mode=mode, format="NETCDF4")
        try:
            if is_append:
                self.time_index = len(self.ds.dimensions["time"])
                self.grids_grp = self.ds.groups["Grids"]
                self.geo_grp = self.ds.groups["Geometry"]
                self.diag_grp = self.ds.groups["Diagnostics"]
                self.inputs_grp = self.ds.groups["Inputs"]
            else:
                self._init_root_and_groups()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> DiagnosticsWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the underlying NetCDF4 dataset."""
        if not self._closed and getattr(self, "ds", None) is not None:
            try:
                if self.ds.isopen():
                    self.ds.sync()
                    self.ds.close()
            finally:
                self._closed = True

    def sync(self) -> None:
        """Synchronize buffered NetCDF4 records to disk."""
        if not self._closed and self.ds.isopen():
            self.ds.sync()

    def _put_scalar(
        self,
        grp: netCDF4.Dataset | netCDF4.Group,
        name: str,
        dtype: str,
        val: float,
        *,
        value_attr: str | None = None,
    ) -> None:
        if name in grp.variables:
            v = grp.variables[name]
        else:
            v = grp.createVariable(name, dtype, ())
        v[...] = np.asarray(val, dtype=dtype)
        if value_attr is not None:
            v.value = str(value_attr)

    def _def_diag_var(
        self,
        name: str,
        dims: tuple[str, ...],
        description: str = "",
        dtype: str = "f4",
    ) -> netCDF4.Variable:
        if name in self.diag_grp.variables:
            return self.diag_grp.variables[name]
        v = self.diag_grp.createVariable(name, dtype, dims)
        if description:
            v.description = description
        return v

    def _init_root_and_groups(self) -> None:
        cfg = self.cfg
        grids = self.grids
        geo = self.geo

        # 1. Root attributes, dimensions, and scalar parameters (ncdf.h:L58-70,
        # parameters.cu:L1498-1518)
        self.ds.Title = "GX simulation data"
        self.ds.createDimension("ri", 2)
        self.ds.createDimension("x", int(grids.nx))
        self.ds.createDimension("y", int(grids.ny))
        self.ds.createDimension("theta", int(grids.nz))
        self.ds.createDimension("kx", int(grids.nakx))
        self.ds.createDimension("ky", int(grids.naky))
        self.ds.createDimension("kz", int(grids.nz))
        self.ds.createDimension("m", int(grids.nhermite))
        self.ds.createDimension("l", int(grids.nlaguerre))
        self.ds.createDimension("s", int(grids.nspecies))
        self.ds.createDimension("time", None)

        self._put_scalar(self.ds, "ny", "i4", int(grids.ny))
        self._put_scalar(self.ds, "nx", "i4", int(grids.nx))
        self._put_scalar(self.ds, "ntheta", "i4", int(grids.nz))
        self._put_scalar(self.ds, "nhermite", "i4", int(grids.nhermite))
        self._put_scalar(self.ds, "nlaguerre", "i4", int(grids.nlaguerre))
        self._put_scalar(self.ds, "nspecies", "i4", int(grids.nspecies))
        self._put_scalar(self.ds, "nperiod", "i4", int(cfg.nperiod))
        self._put_scalar(self.ds, "debug", "i4", 0)
        self._put_scalar(self.ds, "code_info", "i4", 0)

        # 2. Grids group (ncdf.h:L97-116)
        self.grids_grp = self.ds.createGroup("Grids")
        self.grids_grp.createVariable("time", "f8", ("time",))
        v_kx = self.grids_grp.createVariable("kx", "f4", ("kx",))
        v_ky = self.grids_grp.createVariable("ky", "f4", ("ky",))
        v_kz = self.grids_grp.createVariable("kz", "f4", ("kz",))
        v_x = self.grids_grp.createVariable("x", "f4", ("x",))
        v_y = self.grids_grp.createVariable("y", "f4", ("y",))
        v_th = self.grids_grp.createVariable("theta", "f4", ("theta",))
        v_m = self.grids_grp.createVariable("m", "i4", ("m",))
        v_l = self.grids_grp.createVariable("l", "i4", ("l",))

        v_kx[:] = np.asarray(grids.kx_out[: grids.nakx], dtype=np.float32)
        v_ky[:] = np.asarray(grids.ky, dtype=np.float32)
        v_kz[:] = np.asarray(grids.kz_out, dtype=np.float32)
        v_x[:] = np.asarray(grids.x, dtype=np.float32)
        v_y[:] = np.asarray(grids.y, dtype=np.float32)
        v_th[:] = np.asarray(grids.z, dtype=np.float32)
        v_m[:] = np.arange(grids.nhermite, dtype=np.int32)
        v_l[:] = np.arange(grids.nlaguerre, dtype=np.int32)

        # 3. Geometry group (ncdf.h:L135-191)
        self.geo_grp = self.ds.createGroup("Geometry")
        for arr_name in (
            "bmag",
            "bgrad",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
            "gds2",
            "gds21",
            "gds22",
            "grho",
            "jacobian",
        ):
            v_geo = self.geo_grp.createVariable(arr_name, "f4", ("theta",))
            v_geo[:] = np.asarray(getattr(geo, arr_name), dtype=np.float32)
        v_dv = self.geo_grp.createVariable("dv", "f4", ("theta",))
        v_dv[:] = np.asarray(geo.vol_fac, dtype=np.float32)

        self._put_scalar(self.geo_grp, "gradpar", "f4", float(geo.gradpar))
        self._put_scalar(self.geo_grp, "nperiod", "i4", int(geo.nperiod))
        self._put_scalar(self.geo_grp, "q", "f4", float(geo.qsf))
        self._put_scalar(self.geo_grp, "shat", "f4", float(geo.shat))
        self._put_scalar(self.geo_grp, "shift", "f4", float(geo.shift))
        self._put_scalar(self.geo_grp, "rmaj", "f4", float(geo.rmaj))
        self._put_scalar(self.geo_grp, "aminor", "f4", float(geo.aminor))
        self._put_scalar(self.geo_grp, "kxfac", "f4", float(geo.kxfac))
        self._put_scalar(self.geo_grp, "drhodpsi", "f4", float(geo.drhodpsi))
        self._put_scalar(self.geo_grp, "theta_scale", "f4", float(geo.theta_scale))
        self._put_scalar(self.geo_grp, "nfp", "i4", int(geo.nfp))
        self._put_scalar(self.geo_grp, "alpha", "f4", float(geo.alpha))
        self._put_scalar(self.geo_grp, "zeta_center", "f4", float(geo.zeta_center))
        self._put_scalar(self.geo_grp, "grho_avg", "f4", float(geo.grho_avg))
        self._put_scalar(self.geo_grp, "beta", "f4", float(cfg.beta))
        self._put_scalar(self.geo_grp, "zero_shat", "i4", int(geo.zero_shat))

        # 4. Inputs group & subgroups (parameters.cu:L1468-1745)
        self.inputs_grp = self.ds.createGroup("Inputs")
        igeo = _GEO_TO_IGEO.get(cfg.geo_option.strip().lower(), 0)
        self._put_scalar(self.inputs_grp, "igeo", "i4", igeo)
        self._put_scalar(self.inputs_grp, "slab", "i4", int(cfg.slab))
        self._put_scalar(self.inputs_grp, "const_curv", "i4", int(cfg.const_curv))
        self._put_scalar(self.inputs_grp, "geofile_dum", "i4", 0, value_attr=str(cfg.geofilename))
        self._put_scalar(self.inputs_grp, "drhodpsi", "f4", float(geo.drhodpsi))
        self._put_scalar(self.inputs_grp, "kxfac", "f4", float(geo.kxfac))
        self._put_scalar(self.inputs_grp, "Rmaj", "f4", float(geo.rmaj))
        self._put_scalar(self.inputs_grp, "shift", "f4", float(geo.shift))
        self._put_scalar(self.inputs_grp, "eps", "f4", float(cfg.eps))
        self._put_scalar(self.inputs_grp, "q", "f4", float(geo.qsf))
        self._put_scalar(self.inputs_grp, "shat", "f4", float(geo.shat))
        self._put_scalar(self.inputs_grp, "kappa", "f4", float(cfg.akappa))
        self._put_scalar(self.inputs_grp, "kappa_prime", "f4", float(cfg.akappri))
        self._put_scalar(self.inputs_grp, "tri", "f4", float(cfg.tri))
        self._put_scalar(self.inputs_grp, "tri_prime", "f4", float(cfg.tripri))
        self._put_scalar(self.inputs_grp, "beta", "f4", float(cfg.beta))
        self._put_scalar(self.inputs_grp, "zero_shat", "i4", int(geo.zero_shat))
        self._put_scalar(self.inputs_grp, "B_ref", "f4", 1.0)
        self._put_scalar(self.inputs_grp, "a_ref", "f4", 1.0)
        self._put_scalar(self.inputs_grp, "grhoavg", "f4", float(geo.grho_avg))
        self._put_scalar(self.inputs_grp, "surfarea", "f4", 1.0)

        dom_grp = self.inputs_grp.createGroup("Domain")
        self._put_scalar(dom_grp, "y0", "f4", float(grids.y0))
        self._put_scalar(dom_grp, "x0", "f4", float(grids.x0))
        self._put_scalar(dom_grp, "z0", "f4", float(cfg.z0 if cfg.z0 > 0 else 1.0))
        self._put_scalar(dom_grp, "zp", "i4", int(grids.zp))
        self._put_scalar(dom_grp, "jtwist", "i4", int(grids.jtwist))
        b_dum = _BOUNDARY_TO_DUM.get(cfg.boundary.strip().lower(), 1)
        self._put_scalar(dom_grp, "boundary_dum", "i4", b_dum, value_attr=str(cfg.boundary))
        self._put_scalar(dom_grp, "long_wavelength_GK", "i4", int(cfg.long_wavelength_gk))

        time_grp = self.inputs_grp.createGroup("Time")
        self._put_scalar(time_grp, "dt", "f4", float(cfg.dt))
        self._put_scalar(time_grp, "t_max", "f4", float(cfg.t_max))
        self._put_scalar(time_grp, "nstep", "i4", int(cfg.nstep))
        self._put_scalar(time_grp, "nwrite", "i4", int(cfg.nwrite))
        self._put_scalar(time_grp, "navg", "i4", int(cfg.navg))
        self._put_scalar(time_grp, "nsave", "i4", int(cfg.nsave))

        for empty_sub in ("KS", "Vlasov_Poisson", "KREHM", "collisionalETG", "Reservoir"):
            self.inputs_grp.createGroup(empty_sub)

        rest_grp = self.inputs_grp.createGroup("Restart")
        self._put_scalar(rest_grp, "scale", "f4", float(cfg.restart_scale))
        self._put_scalar(rest_grp, "restart", "i4", int(cfg.restart))
        self._put_scalar(rest_grp, "save_for_restart", "i4", int(cfg.save_for_restart))
        self._put_scalar(
            rest_grp, "restart_from_file_dum", "i4", 0, value_attr=str(cfg.restart_from_file)
        )
        self._put_scalar(
            rest_grp, "restart_to_file_dum", "i4", 0, value_attr=str(cfg.restart_to_file)
        )

        ctrl_grp = self.inputs_grp.createGroup("Controls")
        s_dum = _SCHEME_TO_DUM.get(cfg.scheme.strip().lower(), 5)
        self._put_scalar(ctrl_grp, "scheme_dum", "i4", s_dum, value_attr=str(cfg.scheme))
        self._put_scalar(ctrl_grp, "stages", "i4", 10 if cfg.scheme == "k10" else 3)
        self._put_scalar(ctrl_grp, "dealias_kz", "i4", int(cfg.dealias_kz))
        self._put_scalar(ctrl_grp, "nonlinear_mode", "i4", int(cfg.nonlinear_mode))
        self._put_scalar(ctrl_grp, "closure_model_dum", "i4", 0, value_attr=str(cfg.closure_model))
        self._put_scalar(ctrl_grp, "smith_par_q", "i4", int(cfg.smith_par_q))
        self._put_scalar(ctrl_grp, "smith_perp_q", "i4", int(cfg.smith_perp_q))
        self._put_scalar(ctrl_grp, "cfl", "f4", float(cfg.cfl))
        self._put_scalar(ctrl_grp, "fphi", "f4", float(cfg.fphi))
        self._put_scalar(ctrl_grp, "fapar", "f4", float(cfg.fapar))
        self._put_scalar(ctrl_grp, "fbpar", "f4", float(cfg.fbpar))
        self._put_scalar(ctrl_grp, "init_amp", "f4", float(cfg.init_amp))
        self._put_scalar(ctrl_grp, "collisions", "i4", int(cfg.collisions))
        self._put_scalar(ctrl_grp, "init_field_dum", "i4", 0, value_attr=str(cfg.init_field))
        self._put_scalar(ctrl_grp, "init_electrons_only", "i4", int(cfg.init_electrons_only))
        self._put_scalar(ctrl_grp, "ikpar_init", "i4", int(cfg.ikpar_init))
        self._put_scalar(ctrl_grp, "random_init", "i4", int(cfg.random_init))

        diss_grp = ctrl_grp.createGroup("Numerical_Diss")
        self._put_scalar(
            diss_grp,
            "hypercollisions",
            "i4",
            int(cfg.hypercollisions_const or cfg.hypercollisions_kz),
        )
        self._put_scalar(diss_grp, "nu_hyper_m", "f4", float(cfg.nu_hyper_m))
        self._put_scalar(diss_grp, "nu_hyper_l", "f4", float(cfg.nu_hyper_l))
        self._put_scalar(diss_grp, "p_hyper_m", "i4", int(cfg.p_hyper_m))
        self._put_scalar(diss_grp, "p_hyper_l", "i4", int(cfg.p_hyper_l))
        self._put_scalar(diss_grp, "hyper", "i4", int(cfg.hyper))
        self._put_scalar(diss_grp, "D_hyper", "f4", float(cfg.d_hyper))
        self._put_scalar(diss_grp, "p_hyper", "i4", int(cfg.p_hyper))

        force_grp = ctrl_grp.createGroup("Forcing")
        self._put_scalar(force_grp, "forcing_init", "i4", int(cfg.forcing_init))
        self._put_scalar(force_grp, "forcing_type_dum", "i4", 0, value_attr=str(cfg.forcing_type))
        self._put_scalar(force_grp, "no_fields", "i4", int(cfg.no_fields))
        self._put_scalar(force_grp, "forcing_amp", "f4", float(cfg.forcing_amp))
        self._put_scalar(force_grp, "forcing_index", "i4", int(cfg.forcing_index))
        self._put_scalar(force_grp, "stir_field_dum", "i4", 0, value_attr=str(cfg.stir_field))

        exp_grp = self.inputs_grp.createGroup("Expert")
        self._put_scalar(exp_grp, "i_share", "i4", 8)
        self._put_scalar(exp_grp, "nreal", "i4", 1)
        self._put_scalar(exp_grp, "init_single", "i4", int(cfg.init_single))
        self._put_scalar(exp_grp, "ikx_single", "i4", int(cfg.ikx_single))
        self._put_scalar(exp_grp, "iky_single", "i4", int(cfg.iky_single))
        self._put_scalar(exp_grp, "ikx_fixed", "i4", -1)
        self._put_scalar(exp_grp, "iky_fixed", "i4", -1)
        self._put_scalar(exp_grp, "eqfix", "i4", 0)
        self._put_scalar(exp_grp, "secondary", "i4", 0)
        self._put_scalar(exp_grp, "phi_ext", "f4", 0.0)
        self._put_scalar(exp_grp, "t0", "f4", 0.0)
        self._put_scalar(exp_grp, "tf", "f4", 0.0)
        self._put_scalar(exp_grp, "tprim0", "f4", 0.0)
        self._put_scalar(exp_grp, "tprimf", "f4", 0.0)
        self._put_scalar(exp_grp, "source_dum", "i4", 0, value_attr="default")
        self._put_scalar(exp_grp, "hegna", "i4", 0)

        in_diag_grp = self.inputs_grp.createGroup("Diagnostics")
        self._put_scalar(in_diag_grp, "fixed_amp", "i4", 0)
        self._put_scalar(in_diag_grp, "omega", "i4", int(cfg.write_omega))
        self._put_scalar(in_diag_grp, "fluxes", "i4", int(cfg.write_fluxes))
        self._put_scalar(in_diag_grp, "fields", "i4", int(cfg.write_fields))
        self._put_scalar(in_diag_grp, "eigenfunctions", "i4", 0)
        self._put_scalar(in_diag_grp, "moms", "i4", int(cfg.write_moms))
        self._put_scalar(in_diag_grp, "free_energy", "i4", int(cfg.write_free_energy))
        self._put_scalar(in_diag_grp, "phi", "i4", 1)

        res_grp = self.inputs_grp.createGroup("Resize")
        self._put_scalar(res_grp, "domain_change", "i4", 0)
        for mname in ("x0_mult", "y0_mult", "z0_mult", "nx_mult", "ny_mult", "nz_mult"):
            self._put_scalar(res_grp, mname, "i4", 1)
        for aname in ("nl_add", "nm_add", "ns_add"):
            self._put_scalar(res_grp, aname, "i4", 0)

        sp_grp = self.inputs_grp.createGroup("Species")
        v_sptype = sp_grp.createVariable("species_type", "i4", ("s",))
        v_spz = sp_grp.createVariable("z", "f4", ("s",))
        v_spm = sp_grp.createVariable("m", "f4", ("s",))
        v_spn0 = sp_grp.createVariable("n0", "f4", ("s",))
        v_spnp = sp_grp.createVariable("n0_prime", "f4", ("s",))
        v_spup = sp_grp.createVariable("u0_prime", "f4", ("s",))
        v_spt0 = sp_grp.createVariable("T0", "f4", ("s",))
        v_sptp = sp_grp.createVariable("T0_prime", "f4", ("s",))
        v_spnu = sp_grp.createVariable("nu", "f4", ("s",))

        v_sptype[:] = np.asarray([s.type_id for s in cfg.species[: grids.nspecies]], dtype=np.int32)
        v_spz[:] = np.asarray([s.z for s in cfg.species[: grids.nspecies]], dtype=np.float32)
        v_spm[:] = np.asarray([s.mass for s in cfg.species[: grids.nspecies]], dtype=np.float32)
        v_spn0[:] = np.asarray([s.dens for s in cfg.species[: grids.nspecies]], dtype=np.float32)
        v_spnp[:] = np.asarray([s.fprim for s in cfg.species[: grids.nspecies]], dtype=np.float32)
        v_spup[:] = np.zeros(grids.nspecies, dtype=np.float32)
        v_spt0[:] = np.asarray([s.temp for s in cfg.species[: grids.nspecies]], dtype=np.float32)
        v_sptp[:] = np.asarray([s.tprim for s in cfg.species[: grids.nspecies]], dtype=np.float32)
        v_spnu[:] = np.asarray([s.nu_ss for s in cfg.species[: grids.nspecies]], dtype=np.float32)

        boltz_grp = sp_grp.createGroup("Boltzmann")
        z_ion_val = float(cfg.species[0].z) if cfg.species else 1.0
        self._put_scalar(boltz_grp, "Z_ion", "f4", z_ion_val)
        self._put_scalar(boltz_grp, "tau_fac", "f4", float(cfg.tau_fac))
        self._put_scalar(boltz_grp, "add_Boltzmann_species", "i4", int(cfg.add_boltzmann_species))
        self._put_scalar(boltz_grp, "all_kinetic", "i4", int(cfg.all_kinetic))
        self._put_scalar(
            boltz_grp,
            "Boltzmann_type_dum",
            "i4",
            int(cfg.boltzmann_opt),
            value_attr=str(cfg.boltzmann_type),
        )

        # 5. Diagnostics group variable definitions (diagnostics.cu:L925-978)
        self.diag_grp = self.ds.createGroup("Diagnostics")
        for fstem in ("Phi2", "Apar2", "Bpar2"):
            self._def_diag_var(f"{fstem}_t", ("time",))
            self._def_diag_var(f"{fstem}_kxt", ("time", "kx"))
            self._def_diag_var(f"{fstem}_kyt", ("time", "ky"))
            self._def_diag_var(f"{fstem}_kxkyt", ("time", "ky", "kx"))
            self._def_diag_var(f"{fstem}_zt", ("time", "theta"))

        self._def_diag_var("Phi2_zonal_t", ("time",))
        self._def_diag_var("Phi2_zonal_kxt", ("time", "kx"))
        self._def_diag_var("Phi2_zonal_zt", ("time", "theta"))

        if cfg.write_free_energy:
            for wstem in ("Wg", "Wphi", "Wapar", "Wbpar"):
                self._def_diag_var(f"{wstem}_st", ("time", "s"))
                self._def_diag_var(f"{wstem}_kxst", ("time", "s", "kx"))
                self._def_diag_var(f"{wstem}_kyst", ("time", "s", "ky"))
                self._def_diag_var(f"{wstem}_kxkyst", ("time", "s", "ky", "kx"))
                self._def_diag_var(f"{wstem}_zst", ("time", "s", "theta"))
                self._def_diag_var(f"{wstem}_t", ("time",))
                self._def_diag_var(f"{wstem}_kxt", ("time", "kx"))
                self._def_diag_var(f"{wstem}_kyt", ("time", "ky"))
                self._def_diag_var(f"{wstem}_kxkyt", ("time", "ky", "kx"))
                self._def_diag_var(f"{wstem}_zt", ("time", "theta"))
            self._def_diag_var("Wg_lmst", ("time", "s", "m", "l"))
            self._def_diag_var("Wg_mst", ("time", "s", "m"))
            self._def_diag_var("Wg_lst", ("time", "s", "l"))
            self._def_diag_var("Wg_mt", ("time", "m"))
            self._def_diag_var("Wg_lt", ("time", "l"))
            self._def_diag_var("Wg_lmt", ("time", "m", "l"))
            self._def_diag_var("Wphi_boltzmann_t", ("time",))

        if cfg.write_fluxes:
            for flx_stem, desc in (
                ("HeatFlux", "Turbulent heat flux in gyroBohm units"),
                ("HeatFluxES", "Electrostatic component of turbulent heat flux in gyroBohm units"),
                (
                    "HeatFluxApar",
                    "Electromagnetic (A_parallel) component of turbulent heat flux in "
                    "gyroBohm units",
                ),
                (
                    "HeatFluxBpar",
                    "Electromagnetic (dB_parallel) component of turbulent heat flux in "
                    "gyroBohm units",
                ),
                ("ParticleFlux", "Turbulent particle flux in gyroBohm units"),
                (
                    "ParticleFluxES",
                    "Electrostatic component of turbulent particle flux in gyroBohm units",
                ),
                (
                    "ParticleFluxApar",
                    "Electromagnetic (A_parallel) component of turbulent particle flux in "
                    "gyroBohm units",
                ),
                (
                    "ParticleFluxBpar",
                    "Electromagnetic (dB_parallel) component of turbulent particle flux in "
                    "gyroBohm units",
                ),
                ("TurbulentHeating", "Turbulent heating from collisions in gyroBohm units"),
                (
                    "TurbulentHeatingES",
                    "Electrostatic component of turbulent heating in gyroBohm units",
                ),
                (
                    "TurbulentHeatingApar",
                    "Electromagnetic (A_parallel) component of turbulent heating in gyroBohm units",
                ),
                (
                    "TurbulentHeatingBpar",
                    "Electromagnetic (dB_parallel) component of turbulent heating in "
                    "gyroBohm units",
                ),
            ):
                self._def_diag_var(f"{flx_stem}_st", ("time", "s"), desc)
                self._def_diag_var(f"{flx_stem}_kxst", ("time", "s", "kx"), desc)
                self._def_diag_var(f"{flx_stem}_kyst", ("time", "s", "ky"), desc)
                self._def_diag_var(f"{flx_stem}_kxkyst", ("time", "s", "ky", "kx"), desc)
                self._def_diag_var(f"{flx_stem}_zst", ("time", "s", "theta"), desc)
                self._def_diag_var(f"{flx_stem}_t", ("time",), desc)
                self._def_diag_var(f"{flx_stem}_kxt", ("time", "kx"), desc)
                self._def_diag_var(f"{flx_stem}_kyt", ("time", "ky"), desc)
                self._def_diag_var(f"{flx_stem}_kxkyt", ("time", "ky", "kx"), desc)
                self._def_diag_var(f"{flx_stem}_zt", ("time", "theta"), desc)
            self._def_diag_var(
                "HeatFlux_kxkyzst",
                ("time", "s", "theta", "ky", "kx"),
                "Turbulent heat flux in gyroBohm units",
            )
            self._def_diag_var(
                "ParticleFlux_kxkyzst",
                ("time", "s", "theta", "ky", "kx"),
                "Turbulent particle flux in gyroBohm units",
            )

        if cfg.write_omega:
            self._def_diag_var("omega_kxkyt", ("time", "ky", "kx", "ri"))

        if cfg.write_fields:
            for fname in ("Phi", "Apar", "Bpar"):
                self._def_diag_var(fname, ("time", "ky", "kx", "theta", "ri"))

        if cfg.write_moms:
            for mname in ("Density", "Upar", "Tpar"):
                self._def_diag_var(mname, ("time", "s", "ky", "kx", "theta", "ri"))
            if grids.nlaguerre > 1:
                self._def_diag_var("Tperp", ("time", "s", "ky", "kx", "theta", "ri"))

    def _write_field_spectra(self, idx: int, stem: str, spec: FieldSpectra) -> None:
        d = self.diag_grp.variables
        d[f"{stem}_t"][idx] = np.float32(spec.total)
        d[f"{stem}_kxt"][idx, :] = np.asarray(spec.kx, dtype=np.float32)
        d[f"{stem}_kyt"][idx, :] = np.asarray(spec.ky, dtype=np.float32)
        d[f"{stem}_kxkyt"][idx, :, :] = np.asarray(spec.kxky, dtype=np.float32)
        d[f"{stem}_zt"][idx, :] = np.asarray(spec.z, dtype=np.float32)

    def _write_species_spectra(
        self,
        idx: int,
        stem: str,
        spec: SpeciesSpectra,
        *,
        write_kxkyzst: bool = False,
    ) -> None:
        d = self.diag_grp.variables
        d[f"{stem}_st"][idx, :] = np.asarray(spec.st, dtype=np.float32)
        d[f"{stem}_kxst"][idx, :, :] = np.asarray(spec.kxst, dtype=np.float32)
        d[f"{stem}_kyst"][idx, :, :] = np.asarray(spec.kyst, dtype=np.float32)
        d[f"{stem}_kxkyst"][idx, :, :, :] = np.asarray(spec.kxkyst, dtype=np.float32)
        d[f"{stem}_zst"][idx, :, :] = np.asarray(spec.zst, dtype=np.float32)
        if f"{stem}_t" in d:
            d[f"{stem}_t"][idx] = np.float32(spec.total)
        if f"{stem}_kxt" in d:
            d[f"{stem}_kxt"][idx, :] = np.asarray(spec.kxt, dtype=np.float32)
        if f"{stem}_kyt" in d:
            d[f"{stem}_kyt"][idx, :] = np.asarray(spec.kyt, dtype=np.float32)
        if f"{stem}_kxkyt" in d:
            d[f"{stem}_kxkyt"][idx, :, :] = np.asarray(spec.kxkyt, dtype=np.float32)
        if f"{stem}_zt" in d:
            d[f"{stem}_zt"][idx, :] = np.asarray(spec.zt, dtype=np.float32)
        if write_kxkyzst and f"{stem}_kxkyzst" in d:
            d[f"{stem}_kxkyzst"][idx, :, :, :, :] = np.asarray(spec.kxkyzst, dtype=np.float32)

    def write_snapshot(
        self,
        snapshot: DiagnosticsSnapshot,
        *,
        G: jax.Array | None = None,
        fields: Fields | None = None,
        solver: FieldSolver | None = None,
    ) -> int:
        """Write one :class:`DiagnosticsSnapshot` time slice to the NetCDF4 dataset."""
        idx = self.time_index
        self.grids_grp.variables["time"][idx] = float(snapshot.time)
        d = self.diag_grp.variables

        self._write_field_spectra(idx, "Phi2", snapshot.phi2)
        self._write_field_spectra(idx, "Apar2", snapshot.apar2)
        self._write_field_spectra(idx, "Bpar2", snapshot.bpar2)

        d["Phi2_zonal_t"][idx] = np.float32(snapshot.phi2_zonal.total)
        d["Phi2_zonal_kxt"][idx, :] = np.asarray(snapshot.phi2_zonal.kx, dtype=np.float32)
        d["Phi2_zonal_zt"][idx, :] = np.asarray(snapshot.phi2_zonal.z, dtype=np.float32)

        if self.cfg.write_free_energy:
            wg = snapshot.wg
            d["Wg_st"][idx, :] = np.asarray(wg.st, dtype=np.float32)
            d["Wg_kxst"][idx, :, :] = np.asarray(wg.kxst, dtype=np.float32)
            d["Wg_kyst"][idx, :, :] = np.asarray(wg.kyst, dtype=np.float32)
            d["Wg_kxkyst"][idx, :, :, :] = np.asarray(wg.kxkyst, dtype=np.float32)
            d["Wg_zst"][idx, :, :] = np.asarray(wg.zst, dtype=np.float32)
            d["Wg_lmst"][idx, :, :, :] = np.asarray(wg.lmst, dtype=np.float32)
            d["Wg_mst"][idx, :, :] = np.asarray(wg.mst, dtype=np.float32)
            d["Wg_lst"][idx, :, :] = np.asarray(wg.lst, dtype=np.float32)
            d["Wg_t"][idx] = np.float32(wg.total)
            d["Wg_kxt"][idx, :] = np.asarray(wg.kxt, dtype=np.float32)
            d["Wg_kyt"][idx, :] = np.asarray(wg.kyt, dtype=np.float32)
            d["Wg_kxkyt"][idx, :, :] = np.asarray(wg.kxkyt, dtype=np.float32)
            d["Wg_zt"][idx, :] = np.asarray(wg.zt, dtype=np.float32)
            d["Wg_mt"][idx, :] = np.asarray(wg.mt, dtype=np.float32)
            d["Wg_lt"][idx, :] = np.asarray(wg.lt, dtype=np.float32)
            d["Wg_lmt"][idx, :, :] = np.asarray(wg.lmt, dtype=np.float32)

            wphi = snapshot.wphi
            d["Wphi_st"][idx, :] = np.asarray(wphi.st, dtype=np.float32)
            d["Wphi_kxst"][idx, :, :] = np.asarray(wphi.kxst, dtype=np.float32)
            d["Wphi_kyst"][idx, :, :] = np.asarray(wphi.kyst, dtype=np.float32)
            d["Wphi_kxkyst"][idx, :, :, :] = np.asarray(wphi.kxkyst, dtype=np.float32)
            d["Wphi_zst"][idx, :, :] = np.asarray(wphi.zst, dtype=np.float32)
            d["Wphi_t"][idx] = np.float32(wphi.total)
            d["Wphi_kxt"][idx, :] = np.asarray(wphi.kxt, dtype=np.float32)
            d["Wphi_kyt"][idx, :] = np.asarray(wphi.kyt, dtype=np.float32)
            d["Wphi_kxkyt"][idx, :, :] = np.asarray(wphi.kxkyt, dtype=np.float32)
            d["Wphi_zt"][idx, :] = np.asarray(wphi.zt, dtype=np.float32)
            d["Wphi_boltzmann_t"][idx] = np.float32(wphi.boltzmann_total)

            self._write_species_spectra(idx, "Wapar", snapshot.wapar)
            self._write_species_spectra(idx, "Wbpar", snapshot.wbpar)

        if self.cfg.write_fluxes:
            self._write_species_spectra(
                idx, "HeatFlux", snapshot.heat_flux.total, write_kxkyzst=True
            )
            self._write_species_spectra(idx, "HeatFluxES", snapshot.heat_flux.es)
            self._write_species_spectra(idx, "HeatFluxApar", snapshot.heat_flux.apar)
            self._write_species_spectra(idx, "HeatFluxBpar", snapshot.heat_flux.bpar)

            self._write_species_spectra(
                idx, "ParticleFlux", snapshot.particle_flux.total, write_kxkyzst=True
            )
            self._write_species_spectra(idx, "ParticleFluxES", snapshot.particle_flux.es)
            self._write_species_spectra(idx, "ParticleFluxApar", snapshot.particle_flux.apar)
            self._write_species_spectra(idx, "ParticleFluxBpar", snapshot.particle_flux.bpar)

            self._write_species_spectra(idx, "TurbulentHeating", snapshot.turbulent_heating.total)
            self._write_species_spectra(idx, "TurbulentHeatingES", snapshot.turbulent_heating.es)
            self._write_species_spectra(
                idx, "TurbulentHeatingApar", snapshot.turbulent_heating.apar
            )
            self._write_species_spectra(
                idx, "TurbulentHeatingBpar", snapshot.turbulent_heating.bpar
            )

        if self.cfg.write_omega:
            d["omega_kxkyt"][idx, :, :, :] = np.asarray(snapshot.omega_kxkyt, dtype=np.float32)

        if self.cfg.write_fields and fields is not None:
            d["Phi"][idx, :, :, :, :] = np.asarray(
                reorder_complex_field_to_nc(fields.phi), dtype=np.float32
            )
            d["Apar"][idx, :, :, :, :] = np.asarray(
                reorder_complex_field_to_nc(fields.apar), dtype=np.float32
            )
            d["Bpar"][idx, :, :, :, :] = np.asarray(
                reorder_complex_field_to_nc(fields.bpar), dtype=np.float32
            )

        if self.cfg.write_moms and G is not None and fields is not None and solver is not None:
            moms = compute_moments_diagnostic_arrays(G, fields, self.geo, solver)
            for mname, marr in moms.items():
                if mname in d:
                    d[mname][idx, :, :, :, :, :] = np.asarray(marr, dtype=np.float32)

        self.time_index = idx + 1
        self.sync()
        return idx

    def write_step(
        self,
        G: jax.Array,
        fields: Fields,
        diag_state: DiagnosticsState,
        solver: FieldSolver,
        *,
        time: float | jax.Array,
        dt: float | jax.Array,
    ) -> tuple[DiagnosticsSnapshot, DiagnosticsState]:
        """Compute and write one diagnostic time step."""
        snapshot, next_state = compute_diagnostics_step(
            G,
            fields,
            diag_state,
            self.grids,
            self.geo,
            solver,
            time=time,
            dt=dt,
            beta=self.cfg.beta,
        )
        self.write_snapshot(snapshot, G=G, fields=fields, solver=solver)
        return snapshot, next_state


# ---------------------------------------------------------------------------
# 5. KR2 Diagnostics Adapter Compatibility Helper
# ---------------------------------------------------------------------------


def sanitize_json_payload(obj: Any) -> Any:
    """Recursively map non-finite floats (NaN/Inf) to ``None`` for strict RFC-8259 JSON."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return fv if math.isfinite(fv) else None
    if isinstance(obj, dict):
        return {str(k): sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_payload(v) for v in obj]
    return obj


def extract_kr2_diagnostics(
    out_nc_path: str | Path,
    metrics_out_path: str | Path | None = None,
    *,
    sim_time_target: float | None = None,
    tail_fraction: float = 0.5,
    hermite_tail_bins: int = 4,
    laguerre_tail_bins: int = 2,
) -> dict[str, Any]:
    """Extract KR2 babysitter / database metrics from a GX ``.out.nc`` or legacy ``.nc`` file.

    Matches the schema expected by
    ``ut-austin/workstreams/kr2-gk-database/hpc/tacc_exec/diagnostics_adapter.py``
    and C++
    ``post_processing/{heat_flux,particle_flux,turbulent_heating,hermite_laguerre_spectra}.py``.
    """
    nc_path = Path(out_nc_path)
    inferred_target: float | None = None
    with netCDF4.Dataset(str(nc_path), "r") as ds:
        if "Inputs" in ds.groups and "Time" in ds.groups["Inputs"].groups:
            t_in_grp = ds.groups["Inputs"].groups["Time"].variables
            t_max_nc = float(t_in_grp["t_max"][()]) if "t_max" in t_in_grp else 1.0e20
            nstep_nc = int(t_in_grp["nstep"][()]) if "nstep" in t_in_grp else 2_000_000_000
            dt_nc = float(t_in_grp["dt"][()]) if "dt" in t_in_grp else 0.0
            if math.isfinite(t_max_nc) and 0.0 < t_max_nc < 1.0e19:
                inferred_target = t_max_nc
            elif 0 < nstep_nc < 1_000_000_000 and dt_nc > 0.0:
                inferred_target = float(nstep_nc) * dt_nc
        if "Grids" in ds.groups and "time" in ds.groups["Grids"].variables:
            t_arr = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
            kx_arr = np.asarray(ds.groups["Grids"].variables["kx"][:], dtype=np.float64)
            ky_arr = np.asarray(ds.groups["Grids"].variables["ky"][:], dtype=np.float64)
            kz_arr = np.asarray(ds.groups["Grids"].variables["kz"][:], dtype=np.float64)
            theta_arr = np.asarray(ds.groups["Grids"].variables["theta"][:], dtype=np.float64)
        else:
            t_var = "time" if "time" in ds.variables else "t"
            t_arr = np.asarray(ds.variables[t_var][:], dtype=np.float64)
            kx_arr = (
                np.asarray(ds.variables["kx"][:], dtype=np.float64)
                if "kx" in ds.variables
                else np.zeros((1,))
            )
            ky_arr = (
                np.asarray(ds.variables["ky"][:], dtype=np.float64)
                if "ky" in ds.variables
                else np.zeros((1,))
            )
            kz_arr = (
                np.asarray(ds.variables["kz"][:], dtype=np.float64)
                if "kz" in ds.variables
                else np.zeros((1,))
            )
            theta_arr = (
                np.asarray(ds.groups["Geometry"].variables["theta"][:], dtype=np.float64)
                if ("Geometry" in ds.groups and "theta" in ds.groups["Geometry"].variables)
                else np.zeros((1,))
            )

        nt = len(t_arr)
        it_start = max(0, int(nt * (1.0 - tail_fraction)))

        geo_grp = ds.groups["Geometry"] if "Geometry" in ds.groups else ds
        jac = np.asarray(geo_grp.variables["jacobian"][:], dtype=np.float64)
        grho = np.asarray(geo_grp.variables["grho"][:], dtype=np.float64)
        grho_avg = float(np.sum(jac * grho) / np.sum(jac))

        sp_types = (
            np.asarray(
                ds.groups["Inputs"].groups["Species"].variables["species_type"][:], dtype=np.int32
            )
            if ("Inputs" in ds.groups and "Species" in ds.groups["Inputs"].groups)
            else np.array([0], dtype=np.int32)
        )

        if "Diagnostics" in ds.groups:
            d = ds.groups["Diagnostics"]
            phi2_t = np.asarray(d.variables["Phi2_t"][:], dtype=np.float64)
            phi2_kyt = (
                np.asarray(d.variables["Phi2_kyt"][:], dtype=np.float64)
                if "Phi2_kyt" in d.variables
                else None
            )
            q_st = (
                np.asarray(d.variables["HeatFlux_st"][:], dtype=np.float64)
                if "HeatFlux_st" in d.variables
                else np.zeros((nt, len(sp_types)))
            )
            q_es_st = (
                np.asarray(d.variables["HeatFluxES_st"][:], dtype=np.float64)
                if "HeatFluxES_st" in d.variables
                else q_st
            )
            q_apar_st = (
                np.asarray(d.variables["HeatFluxApar_st"][:], dtype=np.float64)
                if "HeatFluxApar_st" in d.variables
                else np.zeros_like(q_st)
            )
            q_bpar_st = (
                np.asarray(d.variables["HeatFluxBpar_st"][:], dtype=np.float64)
                if "HeatFluxBpar_st" in d.variables
                else np.zeros_like(q_st)
            )
            p_st = (
                np.asarray(d.variables["ParticleFlux_st"][:], dtype=np.float64)
                if "ParticleFlux_st" in d.variables
                else np.zeros((nt, len(sp_types)))
            )
            th_st = (
                np.asarray(d.variables["TurbulentHeating_st"][:], dtype=np.float64)
                if "TurbulentHeating_st" in d.variables
                else np.zeros((nt, len(sp_types)))
            )
            wg_st = (
                np.asarray(d.variables["Wg_st"][:], dtype=np.float64)
                if "Wg_st" in d.variables
                else np.zeros((nt, len(sp_types)))
            )
            wphi_st = (
                np.asarray(d.variables["Wphi_st"][:], dtype=np.float64)
                if "Wphi_st" in d.variables
                else np.zeros((nt, len(sp_types)))
            )
            wg_lmst = (
                np.asarray(d.variables["Wg_lmst"][:], dtype=np.float64)
                if "Wg_lmst" in d.variables
                else None
            )
            omega_kxkyt = (
                np.asarray(d.variables["omega_kxkyt"][:], dtype=np.float64)
                if "omega_kxkyt" in d.variables
                else None
            )
        else:
            # Legacy .nc schema (Spectra / Fluxes / Special groups)
            sp_g = ds.groups["Spectra"]
            fl_g = ds.groups["Fluxes"]
            wg_st = np.asarray(sp_g.variables["Wst"][:], dtype=np.float64)
            wphi_st = np.asarray(sp_g.variables["Pst"][:], dtype=np.float64)
            phi2_t = np.sum(wphi_st, axis=-1)
            phi2_kyt = None
            q_st = (
                np.asarray(fl_g.variables["qflux"][:], dtype=np.float64)
                if "qflux" in fl_g.variables
                else np.zeros_like(wg_st)
            )
            q_es_st = q_st
            q_apar_st = np.zeros_like(q_st)
            q_bpar_st = np.zeros_like(q_st)
            p_st = (
                np.asarray(fl_g.variables["pflux"][:], dtype=np.float64)
                if "pflux" in fl_g.variables
                else np.zeros_like(wg_st)
            )
            th_st = np.zeros_like(wg_st)
            wg_lmst = (
                np.asarray(sp_g.variables["Wlmst"][:], dtype=np.float64)
                if "Wlmst" in sp_g.variables
                else None
            )
            omega_kxkyt = (
                np.asarray(ds.groups["Special"].variables["omega_v_time"][:], dtype=np.float64)
                if ("Special" in ds.groups and "omega_v_time" in ds.groups["Special"].variables)
                else None
            )

    ion_mask = sp_types == 0
    elec_mask = sp_types == 1
    q_i_series = np.sum(q_st[:, ion_mask], axis=-1) if np.any(ion_mask) else np.zeros((nt,))
    q_e_series = np.sum(q_st[:, elec_mask], axis=-1) if np.any(elec_mask) else np.zeros((nt,))
    gam_i_series = np.sum(p_st[:, ion_mask], axis=-1) if np.any(ion_mask) else np.zeros((nt,))
    gam_e_series = np.sum(p_st[:, elec_mask], axis=-1) if np.any(elec_mask) else np.zeros((nt,))

    tail_m_frac = 0.0
    tail_l_frac = 0.0
    if wg_lmst is not None and wg_lmst.size > 0:
        wg_tail_avg = np.mean(wg_lmst[it_start:], axis=0)  # (s, m, l)
        total_wg = float(np.sum(wg_tail_avg))
        if total_wg > 1.0e-30:
            nm = wg_tail_avg.shape[1]
            nl = wg_tail_avg.shape[2]
            m_bins = min(max(1, hermite_tail_bins), nm)
            l_bins = min(max(1, laguerre_tail_bins), nl)
            tail_m_frac = float(np.sum(wg_tail_avg[:, -m_bins:, :]) / total_wg)
            tail_l_frac = float(np.sum(wg_tail_avg[:, :, -l_bins:]) / total_wg)

    gamma_max = 0.0
    omega_at_max = 0.0
    if omega_kxkyt is not None and omega_kxkyt.size > 0:
        om_tail = np.mean(omega_kxkyt[it_start:], axis=0)  # (ky, kx, 2)
        gam_grid = om_tail[..., 1]
        om_grid = om_tail[..., 0]
        flat_idx = int(np.argmax(gam_grid))
        gamma_max = float(gam_grid.ravel()[flat_idx])
        omega_at_max = float(om_grid.ravel()[flat_idx])

    high_k_tail_ratio = 0.0
    if phi2_kyt is not None and phi2_kyt.size > 0 and nt > 0:
        naky = phi2_kyt.shape[1]
        if naky >= 4:
            tail_bins = max(1, naky // 5)
            phi2_kyt_tail_avg = np.mean(phi2_kyt[it_start:], axis=0)
            tot_phi2 = np.sum(phi2_kyt_tail_avg)
            if tot_phi2 > 1e-30:
                high_k_tail_ratio = float(np.sum(phi2_kyt_tail_avg[-tail_bins:]) / tot_phi2)

    q_tot_tail = float(np.mean(np.sum(q_st[it_start:], axis=-1))) if nt > 0 else 0.0
    q_es_tail = float(np.mean(np.sum(q_es_st[it_start:], axis=-1))) if nt > 0 else 0.0
    q_em_tail = (
        float(np.mean(np.sum((q_apar_st + q_bpar_st)[it_start:], axis=-1))) if nt > 0 else 0.0
    )
    p_tot_tail = float(np.mean(np.sum(p_st[it_start:], axis=-1))) if nt > 0 else 0.0

    dt_collapse_ratio = 1.0
    if nt >= 4 and np.all(np.isfinite(t_arr)):
        dt_frames = np.diff(t_arr)
        full = dt_frames[:-1]
        if full[0] > 1.0e-30 and np.all(dt_frames >= 0.0):
            dt_collapse_ratio = float(full[-1] / full[0])

    sim_time_reached = float(t_arr[-1]) if (nt > 0 and math.isfinite(float(t_arr[-1]))) else 0.0
    if sim_time_target is not None and math.isfinite(float(sim_time_target)):
        resolved_target = float(sim_time_target)
    elif inferred_target is not None and math.isfinite(inferred_target):
        resolved_target = float(inferred_target)
    else:
        resolved_target = sim_time_reached

    has_nan = bool(
        not (
            np.all(np.isfinite(t_arr))
            and np.all(np.isfinite(phi2_t))
            and np.all(np.isfinite(q_st))
            and np.all(np.isfinite(q_es_st))
            and np.all(np.isfinite(q_apar_st))
            and np.all(np.isfinite(q_bpar_st))
            and np.all(np.isfinite(p_st))
            and np.all(np.isfinite(th_st))
            and np.all(np.isfinite(wg_st))
            and np.all(np.isfinite(wphi_st))
        )
    )

    raw_report: dict[str, Any] = {
        "n_time": int(nt),
        "t_start": float(t_arr[0]) if nt > 0 else 0.0,
        "t_end": float(t_arr[-1]) if nt > 0 else 0.0,
        "grho_avg": grho_avg,
        "kx_monotonic": bool(np.all(np.diff(kx_arr) > 0)) if len(kx_arr) > 1 else True,
        "kz_monotonic": bool(np.all(np.diff(kz_arr) > 0)) if len(kz_arr) > 1 else True,
        "theta_len": len(theta_arr),
        "ky_len": len(ky_arr),
        "kx_len": len(kx_arr),
        "phi2_end": float(phi2_t[-1]) if nt > 0 else 0.0,
        "phi2_tail_mean": float(np.mean(phi2_t[it_start:])) if nt > 0 else 0.0,
        "Q_i": float(np.mean(q_i_series[it_start:])) if nt > 0 else 0.0,
        "Q_e": float(np.mean(q_e_series[it_start:])) if nt > 0 else 0.0,
        "Q_total": q_tot_tail,
        "Gamma_i": float(np.mean(gam_i_series[it_start:])) if nt > 0 else 0.0,
        "Gamma_e": float(np.mean(gam_e_series[it_start:])) if nt > 0 else 0.0,
        "TurbulentHeating_total": float(np.mean(np.sum(th_st[it_start:], axis=-1)))
        if nt > 0
        else 0.0,
        "Wg_total": float(np.mean(np.sum(wg_st[it_start:], axis=-1))) if nt > 0 else 0.0,
        "Wphi_total": float(np.mean(np.sum(wphi_st[it_start:], axis=-1))) if nt > 0 else 0.0,
        "gamma_max": gamma_max,
        "omega_at_gamma_max": omega_at_max,
        "tail_energy_fraction_m": tail_m_frac,
        "tail_energy_fraction_l": tail_l_frac,
        "has_nan_or_inf": has_nan,
        "converged": bool(nt > 0 and not has_nan),
        "failure_type": "DIVERGENCE" if has_nan else (None if nt > 0 else "STALL"),
        "sim_time_reached": sim_time_reached,
        "sim_time_target": resolved_target,
        "scalar_metrics": {
            "has_nan": has_nan,
            "high_k_tail_ratio": high_k_tail_ratio,
            "dt_collapse_ratio": dt_collapse_ratio,
            "heat_flux_es": q_es_tail,
            "heat_flux_em": q_em_tail,
            "particle_flux": p_tot_tail,
        },
        "fluxes": {
            "heat_flux_es": q_es_tail,
            "heat_flux_em": q_em_tail,
            "particle_flux": p_tot_tail,
        },
    }
    report: dict[str, Any] = sanitize_json_payload(raw_report)
    if metrics_out_path is not None:
        out_p = Path(metrics_out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


__all__ = [
    "DiagnosticsSnapshot",
    "DiagnosticsState",
    "DiagnosticsWriter",
    "FieldSpectra",
    "FluxBundle",
    "SpeciesSpectra",
    "WgSpectra",
    "WphiSpectra",
    "ZonalFieldSpectra",
    "compute_diagnostics_step",
    "compute_field_squared_spectra",
    "compute_heat_flux_spectra",
    "compute_moments_diagnostic_arrays",
    "compute_omega_kxky",
    "compute_particle_flux_spectra",
    "compute_phi2_zonal_spectra",
    "compute_turbulent_heating_spectra",
    "compute_wapar_spectra",
    "compute_wbpar_spectra",
    "compute_wg_spectra",
    "compute_wphi_spectra",
    "extract_kr2_diagnostics",
    "fourier_ky_weights",
    "init_diagnostics_state",
    "load_restart_nc",
    "reorder_complex_field_to_nc",
    "reorder_complex_moment_to_nc",
    "reorder_kx_from_out",
    "reorder_kx_to_out",
    "reorder_kz_to_out",
    "sanitize_json_payload",
    "save_restart_nc",
]
