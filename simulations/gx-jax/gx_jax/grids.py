"""Spectral and real-space grids for GX-JAX.

Ported from ``gx/include/grids.h``, ``gx/src/grids.cu``,
and ``gx/src/device_funcs.cu`` (``kInit``, ``get_ikx``, ``unmasked``).

Array-axis convention:
- Runtime distribution function state has shape ``(nspecies, nhermite, nlaguerre, nz, naky, nakx)``
  with active dealiased spectral axes ``(nz, ky, kx)`` trailing.
- ``ky`` has length ``naky = 1 + (ny - 1) // 3``.
- ``kx`` has length ``nakx = 1 + 2 * ((nx - 1) // 3)`` in FFT-active order
  ``[0, 1, ..., kxmax_idx, -kxmax_idx, ..., -1] / x0`` (matching ``.restart.nc``),
  while ``kx_out`` is shifted monotonic ``[-kxmax_idx, ..., +kxmax_idx] / x0``
  (matching ``Grids/kx`` in ``.out.nc``).
- Full padded FFT arrays ``(nyc, nx)`` and pad/truncate helpers are carried on ``Grids``
  for the 3/2-dealiased nonlinear ``irfft2`` / ``rfft2`` transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax.config import GXConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Grids:
    """Immutable spectral and real-space grid container registered as a JAX PyTree."""

    # Static integer dimensions
    nx: int
    ny: int
    nz: int
    nspecies: int
    nhermite: int
    nlaguerre: int
    nj: int
    nyc: int
    naky: int
    nakx: int
    zp: int
    jtwist: int

    # Static domain/CFL floats
    x0: float
    y0: float
    vpar_max: float
    muB_max: float
    muB_maxJ: float
    kx_max: float
    ky_max: float
    kz_max: float
    kperp_min: float

    # Active dealiased spectral arrays (Option A)
    kx: jax.Array  # (nakx,) float64, FFT-active order [0..+K, -K..-1]/x0
    ky: jax.Array  # (naky,) float64, [0..naky-1]/y0
    kz: jax.Array  # (nz,) float64, [0..nz//2, -(nz - nz//2 - 1)..-1]/zp
    kzm: jax.Array  # (nz,) int32, parallel 2/3 dealiasing mask
    kzp: jax.Array  # (nz,) float64, kz masked by kzm if dealias_kz else kz
    kx_out: jax.Array  # (nakx,) float64, monotonic [-K..+K]/x0 for .out.nc
    kz_out: jax.Array  # (nz,) float64, shifted monotonic kz for .out.nc
    active_mask: jax.Array  # (naky, nakx) bool, False only at (iky=0, ikx=0)

    # Full padded 3/2 FFT spectral arrays for nonlinear pseudo-spectral transforms
    kx_full: jax.Array  # (nx,) float64
    ky_full: jax.Array  # (nyc,) float64
    active_kx_idx: jax.Array  # (nakx,) int32 indices mapping active kx -> 0..nx-1
    unmasked_full: jax.Array  # (nyc, nx) bool matching device_funcs.cu:unmasked

    # Real-space coordinate grids
    x: jax.Array  # (nx,) float64
    y: jax.Array  # (ny,) float64
    z: jax.Array  # (nz,) float64

    @property
    def g_shape(self) -> tuple[int, int, int, int, int, int]:
        """Canonical TPU-friendly state shape ``G[nspecies, nhermite, nlaguerre, nz, ky, kx]``."""
        return (self.nspecies, self.nhermite, self.nlaguerre, self.nz, self.naky, self.nakx)

    @property
    def field_shape(self) -> tuple[int, int, int]:
        """Canonical active spectral field shape ``(nz, naky, nakx)``."""
        return (self.nz, self.naky, self.nakx)

    def tree_flatten(self):
        children = (
            self.kx,
            self.ky,
            self.kz,
            self.kzm,
            self.kzp,
            self.kx_out,
            self.kz_out,
            self.active_mask,
            self.kx_full,
            self.ky_full,
            self.active_kx_idx,
            self.unmasked_full,
            self.x,
            self.y,
            self.z,
        )
        aux_data = (
            self.nx,
            self.ny,
            self.nz,
            self.nspecies,
            self.nhermite,
            self.nlaguerre,
            self.nj,
            self.nyc,
            self.naky,
            self.nakx,
            self.zp,
            self.jtwist,
            self.x0,
            self.y0,
            self.vpar_max,
            self.muB_max,
            self.muB_maxJ,
            self.kx_max,
            self.ky_max,
            self.kz_max,
            self.kperp_min,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Grids:
        (
            nx,
            ny,
            nz,
            nspecies,
            nhermite,
            nlaguerre,
            nj,
            nyc,
            naky,
            nakx,
            zp,
            jtwist,
            x0,
            y0,
            vpar_max,
            muB_max,
            muB_maxJ,
            kx_max,
            ky_max,
            kz_max,
            kperp_min,
        ) = aux_data
        (
            kx,
            ky,
            kz,
            kzm,
            kzp,
            kx_out,
            kz_out,
            active_mask,
            kx_full,
            ky_full,
            active_kx_idx,
            unmasked_full,
            x,
            y,
            z,
        ) = children
        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            nspecies=nspecies,
            nhermite=nhermite,
            nlaguerre=nlaguerre,
            nj=nj,
            nyc=nyc,
            naky=naky,
            nakx=nakx,
            zp=zp,
            jtwist=jtwist,
            x0=x0,
            y0=y0,
            vpar_max=vpar_max,
            muB_max=muB_max,
            muB_maxJ=muB_maxJ,
            kx_max=kx_max,
            ky_max=ky_max,
            kz_max=kz_max,
            kperp_min=kperp_min,
            kx=kx,
            ky=ky,
            kz=kz,
            kzm=kzm,
            kzp=kzp,
            kx_out=kx_out,
            kz_out=kz_out,
            active_mask=active_mask,
            kx_full=kx_full,
            ky_full=ky_full,
            active_kx_idx=active_kx_idx,
            unmasked_full=unmasked_full,
            x=x,
            y=y,
            z=z,
        )

    def astype(self, dtype: jnp.dtype) -> Grids:
        """Cast floating-point coordinate/wavenumber arrays to ``dtype`` (e.g. ``float32``)."""
        r_dtype = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32

        def _cast(leaf):
            if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                if leaf.dtype == r_dtype:
                    return leaf
                if isinstance(leaf, jax.core.Tracer):
                    return leaf.astype(r_dtype)
                return jnp.asarray(np.asarray(leaf), dtype=r_dtype)
            return leaf

        return jax.tree_util.tree_map(_cast, self)


def _laguerre_vmax(nl: int, nj: int) -> tuple[float, float]:
    """Compute ``(vmax, vmax_J)`` as the ``(L-1)``-th and ``(J-1)``-th roots of ``L_J``."""
    idx = jnp.arange(nj, dtype=jnp.float64)
    diag = 2.0 * idx + 1.0
    off = jnp.arange(1, nj, dtype=jnp.float64)
    jacobi = jnp.diag(diag) + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    evals = jnp.linalg.eigvalsh(jacobi)
    return float(evals[nl - 1]), float(evals[nj - 1])


def init_grids(cfg: GXConfig) -> Grids:
    """Initialize spectral and real-space grids matching ``Grids::init_ks_and_coords``."""
    if cfg.x0 <= 0.0:
        raise ValueError(
            f"cfg.x0={cfg.x0} must be positive before calling init_grids "
            "(call resolve_jtwist_x0 or init_geometry first)."
        )

    nx = cfg.nx
    ny = cfg.ny
    nz = cfg.nz
    nspecies = cfg.nspecies
    nhermite = cfg.nhermite
    nlaguerre = cfg.nlaguerre
    nj = max(1, (3 * nlaguerre) // 2 - 1)

    nyc = 1 + ny // 2
    naky = 1 + (ny - 1) // 3
    nakx = 1 + 2 * ((nx - 1) // 3)
    zp = cfg.zp
    x0 = float(cfg.x0)
    y0 = float(cfg.y0)

    # Full spectral arrays matching device_funcs.cu:kInit (L2048-2083)
    idy_full = jnp.arange(nyc, dtype=jnp.int32)
    ky_full = idy_full.astype(jnp.float64) / y0

    idx_full = jnp.arange(nx, dtype=jnp.int32)
    ikx_full = jnp.where(idx_full < (nx // 2 + 1), idx_full, idx_full - nx)
    kx_full = ikx_full.astype(jnp.float64) / x0

    idz = jnp.arange(nz, dtype=jnp.int32)
    ikz = jnp.where(idz < (nz // 2 + 1), idz, idz - nz)
    kz = ikz.astype(jnp.float64) / float(zp)

    kz_cutoff = (nz - 1) // 3
    kzm = jnp.where((idz <= kz_cutoff) | ((idz - nz) > -(1 + kz_cutoff)), 1, 0).astype(jnp.int32)
    kzp = jnp.where(kzm == 1, kz, 0.0) if cfg.dealias_kz else kz

    # Full 2D dealiasing mask on (nyc, nx) matching device_funcs.cu:unmasked (L607-617)
    kx_cutoff = (nx - 1) // 3
    ky_cutoff = (ny - 1) // 3
    not_origin_full = ~((idy_full[:, None] == 0) & (idx_full[None, :] == 0))
    ky_ok = idy_full[:, None] <= ky_cutoff
    kx_ok = (ikx_full[None, :] <= kx_cutoff) & (ikx_full[None, :] >= -kx_cutoff)
    unmasked_full = not_origin_full & ky_ok & kx_ok

    # Active spectral indices & arrays (Option A: shape (naky,) and (nakx,))
    ky = ky_full[:naky]
    if kx_cutoff == 0:
        active_kx_idx = jnp.zeros((1,), dtype=jnp.int32)
    else:
        pos_idx = jnp.arange(0, kx_cutoff + 1, dtype=jnp.int32)
        neg_idx = jnp.arange(nx - kx_cutoff, nx, dtype=jnp.int32)
        active_kx_idx = jnp.concatenate([pos_idx, neg_idx], axis=0)
    kx = kx_full[active_kx_idx]

    active_mask = ~(
        (jnp.arange(naky, dtype=jnp.int32)[:, None] == 0)
        & (jnp.arange(nakx, dtype=jnp.int32)[None, :] == 0)
    )

    # Monotonic diagnostic output grids matching grids.cu:L249-272
    if nx < 4:
        kx_out = kx_full[:nx]
    else:
        half = nakx // 2
        neg_part = kx_full[nx - jnp.arange(half, 0, -1, dtype=jnp.int32)]
        pos_part = kx_full[jnp.arange(1, half + 1, dtype=jnp.int32)]
        kx_out = jnp.concatenate([neg_part, jnp.zeros((1,), dtype=jnp.float64), pos_part], axis=0)

    if nz > 1:
        kz_out = kz[(jnp.arange(nz, dtype=jnp.int32) + nz // 2 + 1) % nz]
    else:
        kz_out = kz

    # Real-space grids matching grids.cu:L275-288
    x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * math.pi * x0 / nx)
    y = jnp.arange(ny, dtype=jnp.float64) * (2.0 * math.pi * y0 / ny)
    z = 2.0 * math.pi * float(zp) * (jnp.arange(nz, dtype=jnp.float64) - (nz // 2)) / nz

    # Velocity & wavenumber bounds matching grids.cu:L290-299
    vpar_max = 2.0 * math.sqrt(float(nhermite))
    muB_max, muB_maxJ = _laguerre_vmax(nlaguerre, nj)
    kx_max = float(kx_full[kx_cutoff])
    ky_max = float(ky_full[ky_cutoff])
    kz_max = float(kz[nz // 2])
    if nx > 1 and nyc > 1:
        kperp_min = min(float(kx_full[1]), float(ky_full[1]))
    elif nyc > 1:
        kperp_min = float(ky_full[1])
    elif nx > 1:
        kperp_min = float(kx_full[1])
    else:
        kperp_min = 0.0

    return Grids(
        nx=nx,
        ny=ny,
        nz=nz,
        nspecies=nspecies,
        nhermite=nhermite,
        nlaguerre=nlaguerre,
        nj=nj,
        nyc=nyc,
        naky=naky,
        nakx=nakx,
        zp=zp,
        jtwist=cfg.jtwist,
        x0=x0,
        y0=y0,
        vpar_max=vpar_max,
        muB_max=muB_max,
        muB_maxJ=muB_maxJ,
        kx_max=kx_max,
        ky_max=ky_max,
        kz_max=kz_max,
        kperp_min=kperp_min,
        kx=kx,
        ky=ky,
        kz=kz,
        kzm=kzm,
        kzp=kzp,
        kx_out=kx_out,
        kz_out=kz_out,
        active_mask=active_mask,
        kx_full=kx_full,
        ky_full=ky_full,
        active_kx_idx=active_kx_idx,
        unmasked_full=unmasked_full,
        x=x,
        y=y,
        z=z,
    )


def pad_kykx_to_full(arr: jax.Array, grids: Grids) -> jax.Array:
    """Zero-pad trailing active ``(..., naky, nakx)`` axes onto full ``(..., nyc, nx)`` FFT grid."""
    if arr.shape[-2:] != (grids.naky, grids.nakx):
        raise ValueError(
            f"Expected trailing shape {(grids.naky, grids.nakx)}, got {arr.shape[-2:]}."
        )
    masked_arr = jnp.where(grids.active_mask, arr, jnp.zeros_like(arr))
    out = jnp.zeros((*arr.shape[:-2], grids.nyc, grids.nx), dtype=arr.dtype)
    out = out.at[..., : grids.naky, grids.active_kx_idx].set(masked_arr)
    return out


def truncate_full_to_kykx(arr_full: jax.Array, grids: Grids) -> jax.Array:
    """Extract active ``(..., naky, nakx)`` modes from full ``(..., nyc, nx)`` FFT grid."""
    if arr_full.shape[-2:] != (grids.nyc, grids.nx):
        raise ValueError(
            f"Expected trailing shape {(grids.nyc, grids.nx)}, got {arr_full.shape[-2:]}."
        )
    sliced = arr_full[..., : grids.naky, :][..., grids.active_kx_idx]
    return jnp.where(grids.active_mask, sliced, jnp.zeros_like(sliced))
