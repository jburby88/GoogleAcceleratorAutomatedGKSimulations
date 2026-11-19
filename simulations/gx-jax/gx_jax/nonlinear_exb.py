"""Nonlinear 3/2-dealiased ExB, electromagnetic flutter/compressional brackets, and flow shear.

Ported strictly from:
- ``gx/include/{nonlinear.h,grad_perp.h,exb.h,forcing.h}``
- ``gx/src/{nonlinear.cu,grad_perp.cu,exb.cu,forcing.cu,geometry.cu}``
- ``gx/src/device_funcs.cu`` (``J0fToGrid``, ``J0phiAndBparToGrid``,
  ``iKxJ0ftoGrid``, ``iKxgtoGrid``, ``iKxgsingletoGrid``, ``iKxphitoGrid``, ``iky_kernel``,
  ``ikx_kernel``, ``ikxstar_kernel``, ``mask_and_scale_kernel``, ``bracket``, ``nl_flutter``,
  ``init_iKx``, ``init_phasefac_ntft``, ``init_kxstar_kxbar_phasefac``,
  ``kxstar_phase_shift``, ``geo_shift``, ``geo_shift_ntft``, ``iKx_shift_ntft``,
  ``field_shift``, ``g_shift``, ``stirring_kernel``, ``kz_stirring_kernel``)

Numerical & XLA/TPU layout notes:
- Runtime distribution state ``G`` has Option A layout
  ``(nspecies, nhermite, nlaguerre, nz, naky, nakx)`` with trailing spectral axes ``(nz, ky, kx)``.
- Inside ``nonlinear_rhs`` and ``compute_nonlinear_max_frequency``, only the small active spectral
  subgrid ``(..., naky, nakx)`` is swapped to ``(..., nakx, naky)``, padded via scatter-free slice
  concatenation + ``jnp.pad`` to ``(..., nx, nyc)``, and transformed via ``jnp.fft.irfftn`` /
  ``jnp.fft.rfftn`` along trailing ``axes=(-2, -1)`` with ``s=(nx, ny)`` and ``norm="forward"``.
  This keeps all full real-space intermediates in ``(..., nz, nx, ny)`` order (with 1D ``y``
  ``phase_mult`` along contiguous ``axis=-1``), eliminating all ``stablehlo.transpose`` ops on
  full ``(nx, ny)`` / ``(ny, nx)`` real tensors.
- Public ``GradPerp`` methods (``c2r``, ``r2c``, ``dx_c2r``, ``dy_c2r``, ``phase_mult``, ``qvar``)
  preserve C++ GX's ``(..., ny, nx)`` real-space convention so C++ unit tests apply directly.
- Gauss-Laguerre velocity transforms ``laguerre_to_grid`` and ``laguerre_to_spectral`` use
  batched ``jax.lax.dot_general`` with ``precision=jax.lax.Precision.HIGHEST`` so TPU MXU never
  truncates to ``bfloat16`` and never emits intermediate tensor transposes.
- Static gyroaveraging Bessel weights on the ``mu * B`` quadrature grid (``J_0(alpha)`` and
  ``tz * 2 * muB * J_1(alpha) / alpha``) are precomputed in IEEE ``float64`` before optional
  runtime ``float32`` / ``complex64`` execution; when time-dependent ``ExB`` shear is active
  (``g_exb != 0``), runtime Bessel evaluation executes strictly in the active real dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gx_jax.config import GXConfig
from gx_jax.fields import Fields, _complex_and_real_dtypes
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids
from gx_jax.laguerre import (
    LaguerreTransform,
    bessel_j0,
    init_laguerre,
    j1_over_alpha,
    laguerre_to_grid,
    laguerre_to_spectral,
)
from gx_jax.mxu_fft import (
    MXU2DPlan,
    init_mxu_2d_plan,
    mxu_irfft,
    mxu_irfft2,
    mxu_irfftn,
    mxu_rfft,
    mxu_rfft2,
    mxu_rfftn,
    resolve_fft_backend,
)


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """Evaluate ``sqrt(max(x, 0))`` with finite zero cotangent at ``x <= 0`` under ``jax.grad``."""
    pos = x > 0.0
    safe_x = jnp.where(pos, x, jnp.ones_like(x))
    return jnp.where(pos, jnp.sqrt(safe_x), jnp.zeros_like(x))


def _c_round_jax(x: jax.Array) -> jax.Array:
    """Round half away from zero on JAX arrays, matching C99/C++ ``round``."""
    return jnp.copysign(jnp.floor(jnp.abs(x) + 0.5), x)


def _hankel_asymptotic_runtime(nu: int, ax: jax.Array, n_pairs: int = 8) -> jax.Array:
    """Evaluate Hankel asymptotic expansion for ``J_nu(ax)`` strictly in ``ax.dtype``."""
    r_dtype = ax.dtype
    mu = jnp.asarray(4.0 * float(nu * nu), dtype=r_dtype)
    inv_8x = jnp.asarray(1.0, dtype=r_dtype) / (jnp.asarray(8.0, dtype=r_dtype) * ax)
    t = jnp.ones_like(ax, dtype=r_dtype)
    p_sum = jnp.ones_like(ax, dtype=r_dtype)
    q_sum = jnp.zeros_like(ax, dtype=r_dtype)
    for m in range(n_pairs):
        k1 = 2 * m + 1
        c1 = jnp.asarray(float((2 * k1 - 1) ** 2), dtype=r_dtype)
        d1 = jnp.asarray(float(k1), dtype=r_dtype)
        t = t * ((mu - c1) / d1) * inv_8x
        q_sum = q_sum - t if (m % 2 == 1) else q_sum + t

        k2 = 2 * m + 2
        c2 = jnp.asarray(float((2 * k2 - 1) ** 2), dtype=r_dtype)
        d2 = jnp.asarray(float(k2), dtype=r_dtype)
        t = t * ((mu - c2) / d2) * inv_8x
        p_sum = p_sum - t if ((m + 1) % 2 == 1) else p_sum + t

    chi = ax - jnp.asarray((0.5 * float(nu) + 0.25) * math.pi, dtype=r_dtype)
    amp = jnp.sqrt(jnp.asarray(2.0 / math.pi, dtype=r_dtype) / ax)
    return amp * (p_sum * jnp.cos(chi) - q_sum * jnp.sin(chi))


def _bessel_j0_runtime(x: jax.Array) -> jax.Array:
    """Evaluate ``J_0(x)`` strictly in ``x.dtype`` (zero ``float64`` ops when ``x`` is ``f32``)."""
    r_dtype = jnp.result_type(x, jnp.float32)
    ax = jnp.abs(jnp.asarray(x, dtype=r_dtype))
    z = jnp.asarray(-0.25, dtype=r_dtype) * (ax * ax)
    term = jnp.ones_like(ax, dtype=r_dtype)
    s_small = term
    n_terms = 18 if r_dtype == jnp.float64 else 11
    for k in range(1, n_terms):
        term = term * z / jnp.asarray(float(k * k), dtype=r_dtype)
        s_small = s_small + term

    n_quad = 32
    theta = (jnp.arange(n_quad, dtype=r_dtype) + jnp.asarray(0.5, dtype=r_dtype)) * jnp.asarray(
        math.pi / float(n_quad), dtype=r_dtype
    )
    sin_th = jnp.sin(theta)
    s_mid = jnp.mean(jnp.cos(ax[..., None] * sin_th), axis=-1)

    c4 = jnp.asarray(4.0, dtype=r_dtype)
    c25 = jnp.asarray(25.0, dtype=r_dtype)
    x_safe = jnp.where(ax > c25, ax, c25)
    s_large = _hankel_asymptotic_runtime(0, x_safe, n_pairs=10 if r_dtype == jnp.float64 else 6)
    return jnp.where(ax <= c4, s_small, jnp.where(ax <= c25, s_mid, s_large))


def _j1_over_alpha_runtime(alpha: jax.Array) -> jax.Array:
    """Evaluate ``J_1(alpha) / alpha`` strictly in ``alpha.dtype`` (zero ``f64`` ops in ``f32``)."""
    r_dtype = jnp.result_type(alpha, jnp.float32)
    ax = jnp.abs(jnp.asarray(alpha, dtype=r_dtype))
    z = jnp.asarray(-0.25, dtype=r_dtype) * (ax * ax)
    term = jnp.full_like(ax, 0.5, dtype=r_dtype)
    s_over_x = term
    n_terms = 18 if r_dtype == jnp.float64 else 11
    for k in range(1, n_terms):
        term = term * z / jnp.asarray(float(k * (k + 1)), dtype=r_dtype)
        s_over_x = s_over_x + term

    n_quad = 32
    theta = (jnp.arange(n_quad, dtype=r_dtype) + jnp.asarray(0.5, dtype=r_dtype)) * jnp.asarray(
        math.pi / float(n_quad), dtype=r_dtype
    )
    sin_th = jnp.sin(theta)
    s_mid = jnp.mean(sin_th * jnp.sin(ax[..., None] * sin_th), axis=-1)

    c4 = jnp.asarray(4.0, dtype=r_dtype)
    c25 = jnp.asarray(25.0, dtype=r_dtype)
    x_safe = jnp.where(ax > c25, ax, c25)
    s_large = _hankel_asymptotic_runtime(1, x_safe, n_pairs=10 if r_dtype == jnp.float64 else 6)

    ax_div = jnp.where(ax > c4, ax, jnp.ones_like(ax))
    j1_tail = jnp.where(ax <= c25, s_mid, s_large) / ax_div
    return jnp.where(ax <= c4, s_over_x, j1_tail)


def _pad_active_to_full_scatter_free(
    arr: jax.Array,
    *,
    nx: int,
    nyc: int,
    naky: int,
    nakx: int,
    active_mask: jax.Array,
) -> jax.Array:
    """Zero-pad trailing ``(..., naky, nakx)`` -> ``(..., nyc, nx)`` without XLA scatter ops."""
    masked = jnp.where(active_mask, arr, jnp.zeros_like(arr))
    if nakx == 1:
        kx_padded = jnp.pad(
            masked,
            (*((0, 0) for _ in range(masked.ndim - 1)), (0, nx - 1)),
        )
    else:
        k_pos = (nakx + 1) // 2
        mid_len = nx - nakx
        if mid_len > 0:
            zeros_mid = jnp.zeros((*masked.shape[:-1], mid_len), dtype=masked.dtype)
            kx_padded = jnp.concatenate(
                [masked[..., :k_pos], zeros_mid, masked[..., k_pos:]], axis=-1
            )
        else:
            kx_padded = masked

    pad_y = nyc - naky
    if pad_y > 0:
        return jnp.pad(
            kx_padded,
            (*((0, 0) for _ in range(kx_padded.ndim - 2)), (0, pad_y), (0, 0)),
        )
    return kx_padded


def _truncate_full_to_active_scatter_free(
    arr_full: jax.Array,
    *,
    nx: int,
    naky: int,
    nakx: int,
    active_mask: jax.Array,
) -> jax.Array:
    """Extract active ``(..., naky, nakx)`` modes from ``(..., nyc, nx)`` without gather/scatter."""
    y_part = arr_full[..., :naky, :]
    if nakx == 1:
        sliced = y_part[..., :1]
    else:
        k_pos = (nakx + 1) // 2
        k_neg = nakx - k_pos
        sliced = jnp.concatenate([y_part[..., :k_pos], y_part[..., nx - k_neg :]], axis=-1)
    return jnp.where(active_mask, sliced, jnp.zeros_like(sliced))


def _swap_last_two_axes_3d(arr: jax.Array) -> jax.Array:
    """Swap trailing two axes via 3D reshape so XLA never emits a 6D ``stablehlo.transpose``."""
    if arr.ndim <= 3:
        return jnp.swapaxes(arr, -2, -1)
    lead_shape = arr.shape[:-2]
    d_m2, d_m1 = arr.shape[-2], arr.shape[-1]
    b_lead = math.prod(lead_shape)
    if b_lead == 0 or d_m2 == 0 or d_m1 == 0:
        return jnp.zeros((*lead_shape, d_m1, d_m2), dtype=arr.dtype)
    swapped3 = jnp.swapaxes(arr.reshape(b_lead, d_m2, d_m1), 1, 2)
    return swapped3.reshape(*lead_shape, d_m1, d_m2)


def _pad_active_to_nxnyc_scatter_free(
    arr: jax.Array,
    *,
    nx: int,
    nyc: int,
    naky: int,
    nakx: int,
    active_mask: jax.Array,
) -> jax.Array:
    """Transpose active ``(..., naky, nakx) -> (..., nakx, naky)`` and pad to ``(..., nx, nyc)``."""
    masked = jnp.where(active_mask, arr, jnp.zeros_like(arr))
    swapped = _swap_last_two_axes_3d(masked)  # (..., nakx, naky) via 3D swapaxes
    if nakx == 1:
        kx_padded = jnp.pad(
            swapped,
            (*((0, 0) for _ in range(swapped.ndim - 2)), (0, nx - 1), (0, 0)),
        )
    else:
        k_pos = (nakx + 1) // 2
        mid_len = nx - nakx
        if mid_len > 0:
            zeros_mid = jnp.zeros(
                (*swapped.shape[:-2], mid_len, swapped.shape[-1]), dtype=swapped.dtype
            )
            kx_padded = jnp.concatenate(
                [swapped[..., :k_pos, :], zeros_mid, swapped[..., k_pos:, :]], axis=-2
            )
        else:
            kx_padded = swapped

    pad_y = nyc - naky
    if pad_y > 0:
        return jnp.pad(
            kx_padded,
            (*((0, 0) for _ in range(kx_padded.ndim - 1)), (0, pad_y)),
        )
    return kx_padded


def _truncate_nxnyc_to_active_scatter_free(
    arr_nxnyc: jax.Array,
    *,
    nx: int,
    naky: int,
    nakx: int,
    active_mask: jax.Array,
) -> jax.Array:
    """Truncate ``(..., nx, nyc)`` -> ``(..., nakx, naky)`` and swap to ``(..., naky, nakx)``."""
    y_part = arr_nxnyc[..., :naky]
    if nakx == 1:
        sliced_kxky = y_part[..., :1, :]
    else:
        k_pos = (nakx + 1) // 2
        k_neg = nakx - k_pos
        sliced_kxky = jnp.concatenate(
            [y_part[..., :k_pos, :], y_part[..., nx - k_neg :, :]], axis=-2
        )
    sliced = _swap_last_two_axes_3d(sliced_kxky)
    return jnp.where(active_mask, sliced, jnp.zeros_like(sliced))


# ---------------------------------------------------------------------------
# 1. 3/2-Dealiased Perpendicular Spectral <-> Real Transform Operator (GradPerp)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GradPerp:
    """2D ``3/2``-dealiased perpendicular spectral/real transform operator (``GradPerp``).

    Ported strictly from ``gx/include/grad_perp.h``,
    ``gx/src/grad_perp.cu``, and ``gx/src/device_funcs.cu``
    (``ikx_kernel``, ``ikxstar_kernel``, ``iky_kernel``, ``mask_and_scale_kernel``,
    ``iKxgtoGrid``, ``iKxJ0ftoGrid``).

    Provides both:
    - Public ``(..., ny, nx)`` transforms (``c2r``, ``r2c``, ``dx_c2r``, ``dy_c2r``,
      ``ntft_dx_c2r``, ``phase_mult``, ``qvar``) matching C++ ``GradPerp`` unit tests.
    - Internal XLA transpose-free ``(..., nx, ny)`` transforms (``c2r_nxny``, ``r2c_nxny``,
      ``dx_c2r_nxny``, ``dy_c2r_nxny``, ``ntft_dx_c2r_nxny``, ``phase_mult_nxny``) used by
      ``nonlinear_rhs`` and ``compute_nonlinear_max_frequency``.
    """

    kx: jax.Array  # (nakx,) active kx
    ky: jax.Array  # (naky,) active ky
    kx_full: jax.Array  # (nx,) full FFT kx
    ky_full: jax.Array  # (nyc,) full RFFT ky
    active_mask: jax.Array  # (naky, nakx) bool
    unmasked_full: jax.Array  # (nyc, nx) bool
    x: jax.Array  # (nx,) real-space x
    delta_kx: jax.Array  # (nz, naky) or (0, 0)
    phasefac_ntft: jax.Array  # (nz, naky, nx) or (0, 0, 0) complex
    phasefacminus_ntft: jax.Array  # (nz, naky, nx) or (0, 0, 0) complex
    phasefac_ntft_full: jax.Array  # (nz, nyc, nx) or (0, 0, 0) complex (1+0j on naky:nyc)
    phasefacminus_ntft_full: jax.Array  # (nz, nyc, nx) or (0, 0, 0) complex
    phasefac_ntft_nxny: jax.Array  # (nz, nx, nyc) or (0, 0, 0) complex
    phasefacminus_ntft_nxny: jax.Array  # (nz, nx, nyc) or (0, 0, 0) complex

    nx: int
    ny: int
    nz: int
    nyc: int
    naky: int
    nakx: int
    x0: float
    y0: float
    kxfac: float
    non_twist: bool
    exb_shear: bool
    exb_shear_phase: bool
    mxu_plan_2d: MXU2DPlan | None = None
    fft_backend: str = "fft"

    def tree_flatten(self):
        children = (
            self.kx,
            self.ky,
            self.kx_full,
            self.ky_full,
            self.active_mask,
            self.unmasked_full,
            self.x,
            self.delta_kx,
            self.phasefac_ntft,
            self.phasefacminus_ntft,
            self.phasefac_ntft_full,
            self.phasefacminus_ntft_full,
            self.phasefac_ntft_nxny,
            self.phasefacminus_ntft_nxny,
            self.mxu_plan_2d,
        )
        aux_data = (
            self.nx,
            self.ny,
            self.nz,
            self.nyc,
            self.naky,
            self.nakx,
            self.x0,
            self.y0,
            self.kxfac,
            self.non_twist,
            self.exb_shear,
            self.exb_shear_phase,
            self.fft_backend,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> GradPerp:
        (
            nx,
            ny,
            nz,
            nyc,
            naky,
            nakx,
            x0,
            y0,
            kxfac,
            non_twist,
            exb_shear,
            exb_shear_phase,
            fft_backend,
        ) = aux_data
        (
            kx,
            ky,
            kx_full,
            ky_full,
            active_mask,
            unmasked_full,
            x,
            delta_kx,
            phasefac_ntft,
            phasefacminus_ntft,
            phasefac_ntft_full,
            phasefacminus_ntft_full,
            phasefac_ntft_nxny,
            phasefacminus_ntft_nxny,
            mxu_plan_2d,
        ) = children
        return cls(
            kx=kx,
            ky=ky,
            kx_full=kx_full,
            ky_full=ky_full,
            active_mask=active_mask,
            unmasked_full=unmasked_full,
            x=x,
            delta_kx=delta_kx,
            phasefac_ntft=phasefac_ntft,
            phasefacminus_ntft=phasefacminus_ntft,
            phasefac_ntft_full=phasefac_ntft_full,
            phasefacminus_ntft_full=phasefacminus_ntft_full,
            phasefac_ntft_nxny=phasefac_ntft_nxny,
            phasefacminus_ntft_nxny=phasefacminus_ntft_nxny,
            nx=nx,
            ny=ny,
            nz=nz,
            nyc=nyc,
            naky=naky,
            nakx=nakx,
            x0=x0,
            y0=y0,
            kxfac=kxfac,
            non_twist=non_twist,
            exb_shear=exb_shear,
            exb_shear_phase=exb_shear_phase,
            mxu_plan_2d=mxu_plan_2d,
            fft_backend=fft_backend,
        )

    def astype(self, dtype: jnp.dtype) -> GradPerp:
        """Cast floating/complex arrays on ``GradPerp`` (including MXU plans) to ``dtype``."""
        target = jnp.dtype(dtype)
        r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        c_target = jnp.complex128 if r_target == jnp.float64 else jnp.complex64

        def _cast(leaf):
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                if isinstance(leaf, jax.core.Tracer):
                    return leaf.astype(c_target)
                return jnp.asarray(np.asarray(leaf), dtype=c_target)
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                if isinstance(leaf, jax.core.Tracer):
                    return leaf.astype(r_target)
                return jnp.asarray(np.asarray(leaf), dtype=r_target)
            return leaf

        return jax.tree_util.tree_map(_cast, self)

    def pad_to_full(self, arr: jax.Array) -> jax.Array:
        """Zero-pad active ``(..., naky, nakx)`` onto full ``(..., nyc, nx)`` FFT grid."""
        if arr.shape[-2:] == (self.nyc, self.nx):
            return arr
        if arr.shape[-2:] != (self.naky, self.nakx):
            raise ValueError(
                f"Expected trailing shape {(self.naky, self.nakx)} or {(self.nyc, self.nx)}, "
                f"got {arr.shape[-2:]}."
            )
        return _pad_active_to_full_scatter_free(
            arr,
            nx=self.nx,
            nyc=self.nyc,
            naky=self.naky,
            nakx=self.nakx,
            active_mask=self.active_mask,
        )

    def truncate_to_active(self, arr_full: jax.Array) -> jax.Array:
        """Extract active ``(..., naky, nakx)`` modes from full ``(..., nyc, nx)`` FFT grid."""
        if arr_full.shape[-2:] == (self.naky, self.nakx):
            return jnp.where(self.active_mask, arr_full, jnp.zeros_like(arr_full))
        if arr_full.shape[-2:] != (self.nyc, self.nx):
            raise ValueError(
                f"Expected trailing shape {(self.nyc, self.nx)}, got {arr_full.shape[-2:]}."
            )
        return _truncate_full_to_active_scatter_free(
            arr_full,
            nx=self.nx,
            naky=self.naky,
            nakx=self.nakx,
            active_mask=self.active_mask,
        )

    def pad_active_to_nxnyc(self, arr: jax.Array) -> jax.Array:
        """Swap ``(..., naky, nakx)`` -> ``(..., nakx, naky)`` and pad to ``(..., nx, nyc)``."""
        return _pad_active_to_nxnyc_scatter_free(
            arr,
            nx=self.nx,
            nyc=self.nyc,
            naky=self.naky,
            nakx=self.nakx,
            active_mask=self.active_mask,
        )

    def truncate_nxnyc_to_active(self, arr_nxnyc: jax.Array) -> jax.Array:
        """Truncate ``(..., nx, nyc)`` -> ``(..., nakx, naky) -> (..., naky, nakx)``."""
        return _truncate_nxnyc_to_active_scatter_free(
            arr_nxnyc,
            nx=self.nx,
            naky=self.naky,
            nakx=self.nakx,
            active_mask=self.active_mask,
        )

    def c2r_nxny(self, arr_act: jax.Array) -> jax.Array:
        """Transpose-free 2D spectral ``(..., naky, nakx)`` -> real ``(..., nx, ny)`` transform."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr_act)
        arr_nxnyc = self.pad_active_to_nxnyc(arr_act.astype(c_dtype))
        if self.fft_backend == "mxu":
            return mxu_irfftn(
                arr_nxnyc,
                s=(self.nx, self.ny),
                axes=(-2, -1),
                norm="forward",
                plan=self.mxu_plan_2d,
            ).astype(r_dtype)
        return jnp.fft.irfftn(
            arr_nxnyc,
            s=(self.nx, self.ny),
            axes=(-2, -1),
            norm="forward",
        ).astype(r_dtype)

    def r2c_nxny(self, arr_real_nxny: jax.Array) -> jax.Array:
        """Transpose-free 2D real ``(..., nx, ny)`` -> active spectral ``(..., naky, nakx)``."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr_real_nxny)
        if self.fft_backend == "mxu":
            spec_nxnyc = mxu_rfftn(
                arr_real_nxny.astype(r_dtype),
                s=(self.nx, self.ny),
                axes=(-2, -1),
                norm="forward",
                plan=self.mxu_plan_2d,
            ).astype(c_dtype)
        else:
            spec_nxnyc = jnp.fft.rfftn(
                arr_real_nxny.astype(r_dtype),
                s=(self.nx, self.ny),
                axes=(-2, -1),
                norm="forward",
            ).astype(c_dtype)
        return self.truncate_nxnyc_to_active(spec_nxnyc)

    def dx_c2r_nxny(
        self,
        arr_act: jax.Array,
        *,
        kxstar: jax.Array | None = None,
    ) -> jax.Array:
        """Compute real-space ``d/dx`` on ``(..., nx, ny)`` without full-array transposes."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr_act)
        arr_c = arr_act.astype(c_dtype)
        if kxstar is not None:
            kxs = (
                kxstar.astype(r_dtype)
                if kxstar.shape[-2:] == (self.naky, self.nakx)
                else self.truncate_to_active(kxstar).astype(r_dtype)
            )
            ikx = jax.lax.complex(jnp.zeros_like(kxs), kxs).astype(c_dtype)
        else:
            kxr = self.kx.astype(r_dtype)[None, :]
            ikx = jax.lax.complex(jnp.zeros_like(kxr), kxr).astype(c_dtype)
        return self.c2r_nxny(ikx * arr_c)

    def ntft_dx_c2r_nxny(
        self,
        arr_act: jax.Array,
        *,
        kxstar: jax.Array | None = None,
    ) -> jax.Array:
        """Compute NTFT spectral ``i * (kx + deltaKx)`` + ``c2r_nxny`` on ``(..., nx, ny)``."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr_act)
        arr_c = self.truncate_to_active(arr_act.astype(c_dtype))
        dkx = (
            self.delta_kx.astype(r_dtype)[:, :, None]
            if self.delta_kx.size > 0
            else jnp.zeros((self.nz, self.naky, 1), dtype=r_dtype)
        )
        base_kx = (
            kxstar.astype(r_dtype)[None, :, :]
            if kxstar is not None and kxstar.shape[-2:] == (self.naky, self.nakx)
            else self.kx.astype(r_dtype)[None, None, :]
        )
        kx_ntft = (base_kx + dkx).astype(r_dtype)
        ikx_ntft = jax.lax.complex(jnp.zeros_like(kx_ntft), kx_ntft).astype(c_dtype)
        return self.c2r_nxny(ikx_ntft * arr_c)

    def dy_c2r_nxny(self, arr_act: jax.Array) -> jax.Array:
        """Compute real-space ``d/dy`` on ``(..., nx, ny)`` without full-array transposes."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr_act)
        kyr = self.ky.astype(r_dtype)[:, None]
        iky = jax.lax.complex(jnp.zeros_like(kyr), kyr).astype(c_dtype)
        return self.c2r_nxny(iky * arr_act.astype(c_dtype))

    def phase_mult_nxny(
        self,
        arr_real_nxny: jax.Array,
        *,
        non_twist: bool | None = None,
        exb_shear: bool | None = None,
        positive_phase: bool = True,
        phasefac_exb_nxny: jax.Array | None = None,
        phasefacminus_exb_nxny: jax.Array | None = None,
    ) -> jax.Array:
        """Apply NTFT / ExB real-space phase factors on ``(..., nz, nx, ny)`` along ``axis=-1``."""
        use_ntft = self.non_twist if non_twist is None else bool(non_twist)
        use_exb = self.exb_shear_phase if exb_shear is None else bool(exb_shear)
        if not use_ntft and not use_exb:
            return arr_real_nxny

        c_dtype, r_dtype = _complex_and_real_dtypes(arr_real_nxny)
        plan_y = self.mxu_plan_2d.plan_y_real if self.mxu_plan_2d is not None else None
        if self.fft_backend == "mxu":
            yk = mxu_rfft(arr_real_nxny.astype(r_dtype), n=self.ny, axis=-1, plan=plan_y).astype(
                c_dtype
            )
        else:
            yk = jnp.fft.rfft(arr_real_nxny.astype(r_dtype), axis=-1).astype(c_dtype)

        if use_ntft and self.phasefac_ntft_nxny.size > 0:
            pf_ntft = (
                self.phasefac_ntft_nxny.astype(c_dtype)
                if positive_phase
                else self.phasefacminus_ntft_nxny.astype(c_dtype)
            )
            yk = yk * pf_ntft

        if use_exb and phasefac_exb_nxny is not None:
            pf_exb = (
                phasefac_exb_nxny.astype(c_dtype)
                if positive_phase
                else (
                    phasefacminus_exb_nxny.astype(c_dtype)
                    if phasefacminus_exb_nxny is not None
                    else jnp.conj(phasefac_exb_nxny.astype(c_dtype))
                )
            )
            yk = yk * pf_exb

        if self.fft_backend == "mxu":
            return mxu_irfft(yk, n=self.ny, axis=-1, plan=plan_y).astype(r_dtype)
        return jnp.fft.irfft(yk, n=self.ny, axis=-1).astype(r_dtype)

    def c2r(self, arr: jax.Array) -> jax.Array:
        """Unnormalized 2D spectral -> real transform matching ``GradPerp::C2R`` (``L242-246``).

        Maps trailing ``(..., naky, nakx)`` or ``(..., nyc, nx)`` -> ``(..., ny, nx)`` real.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_full = self.pad_to_full(arr.astype(c_dtype))
        if self.fft_backend == "mxu":
            return mxu_irfft2(
                arr_full,
                s=(self.nx, self.ny),
                axes=(-1, -2),
                norm="forward",
                plan=self.mxu_plan_2d,
            ).astype(r_dtype)
        return jnp.fft.irfft2(
            arr_full,
            s=(self.nx, self.ny),
            axes=(-1, -2),
            norm="forward",
        ).astype(r_dtype)

    def dx_c2r(
        self,
        arr: jax.Array,
        *,
        kxstar: jax.Array | None = None,
    ) -> jax.Array:
        """Compute real-space ``d/dx`` via spectral ``i * kx`` + 2D C2R (``GradPerp::dxC2R``).

        Ported from ``grad_perp.cu:L146-159`` and ``device_funcs.cu:L1030-1056``.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if arr_c.shape[-2:] == (self.naky, self.nakx):
            if kxstar is not None:
                kxs = (
                    kxstar.astype(r_dtype)
                    if kxstar.shape[-2:] == (self.naky, self.nakx)
                    else self.truncate_to_active(kxstar).astype(r_dtype)
                )
                ikx = jax.lax.complex(jnp.zeros_like(kxs), kxs).astype(c_dtype)
            else:
                kxr = self.kx.astype(r_dtype)[None, :]
                ikx = jax.lax.complex(jnp.zeros_like(kxr), kxr).astype(c_dtype)
            return self.c2r(ikx * arr_c)

        if arr_c.shape[-2:] == (self.nyc, self.nx):
            if kxstar is not None:
                kxs_full = (
                    kxstar.astype(r_dtype)
                    if kxstar.shape[-2:] == (self.nyc, self.nx)
                    else self.pad_to_full(kxstar).astype(r_dtype)
                )
                ikx_full = jax.lax.complex(jnp.zeros_like(kxs_full), kxs_full).astype(c_dtype)
            else:
                kxr_full = self.kx_full.astype(r_dtype)[None, :]
                ikx_full = jax.lax.complex(jnp.zeros_like(kxr_full), kxr_full).astype(c_dtype)
            return self.c2r(ikx_full * arr_c)

        raise ValueError(
            f"Expected trailing shape {(self.naky, self.nakx)} or {(self.nyc, self.nx)}, "
            f"got {arr.shape[-2:]}."
        )

    def ntft_dx_c2r(
        self,
        arr: jax.Array,
        *,
        kxstar: jax.Array | None = None,
    ) -> jax.Array:
        """Compute NTFT spectral ``i * (kx + deltaKx)`` + 2D C2R prior to ``phase_mult``.

        Ported from ``nonlinear.cu:L216-218, L235-237, L275-277`` and ``init_iKx``
        (``device_funcs.cu:L717-729``).
        Note: C++ ``Nonlinear_GK::nlps`` has three bugs in the ``nonTwist`` branch that are
        deliberately fixed here:
        1. ``iKxgtoGrid`` / ``iKxJ0ftoGrid`` (``device_funcs.cu:L944, L962``) divide by ``ny``
           before a 2D ``C2R`` whose inverse ``R2C`` already divides by ``nx * ny``.
        2. ``GBX_ntft`` (``nonlinear.cu:L110, L116-120``) launches with 3rd grid dimension ``1``
           instead of ``Nm``, leaving ``dg_dx = 0`` for all Hermite moments ``m >= 1``.
        3. ``iKxJ0ftoGrid(J0phi, J0phi, ...)`` (``nonlinear.cu:L236, L276``) overwrites ``J0phi``
           and ``J0apar`` in-place before ``dyC2R`` (``L244, L283``) reads them.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_act = self.truncate_to_active(arr.astype(c_dtype))
        dkx = (
            self.delta_kx.astype(r_dtype)[:, :, None]
            if self.delta_kx.size > 0
            else jnp.zeros((self.nz, self.naky, 1), dtype=r_dtype)
        )
        base_kx = (
            kxstar.astype(r_dtype)[None, :, :]
            if kxstar is not None and kxstar.shape[-2:] == (self.naky, self.nakx)
            else self.kx.astype(r_dtype)[None, None, :]
        )
        kx_ntft = (base_kx + dkx).astype(r_dtype)
        ikx_ntft = jax.lax.complex(jnp.zeros_like(kx_ntft), kx_ntft).astype(c_dtype)
        return self.c2r(ikx_ntft * arr_act)

    def dy_c2r(self, arr: jax.Array) -> jax.Array:
        """Compute real-space ``d/dy`` via spectral ``i * ky`` + 2D C2R (``GradPerp::dyC2R``).

        Ported from ``grad_perp.cu:L231-240`` and ``device_funcs.cu:L1016-1028``.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if arr_c.shape[-2:] == (self.naky, self.nakx):
            kyr = self.ky.astype(r_dtype)[:, None]
            iky = jax.lax.complex(jnp.zeros_like(kyr), kyr).astype(c_dtype)
            return self.c2r(iky * arr_c)
        if arr_c.shape[-2:] == (self.nyc, self.nx):
            kyr_full = self.ky_full.astype(r_dtype)[:, None]
            iky_full = jax.lax.complex(jnp.zeros_like(kyr_full), kyr_full).astype(c_dtype)
            return self.c2r(iky_full * arr_c)
        raise ValueError(
            f"Expected trailing shape {(self.naky, self.nakx)} or {(self.nyc, self.nx)}, "
            f"got {arr.shape[-2:]}."
        )

    def phase_mult(
        self,
        arr_real: jax.Array,
        *,
        non_twist: bool | None = None,
        exb_shear: bool | None = None,
        positive_phase: bool = True,
        phasefac_exb: jax.Array | None = None,
        phasefacminus_exb: jax.Array | None = None,
    ) -> jax.Array:
        """Multiply real-space ``(..., nz, ny, nx)`` array by NTFT and/or ExB phase factors.

        Ported strictly from ``GradPerp::phase_mult`` (``grad_perp.cu:L161-199``):
        1. 1D ``rfft`` in ``y`` (``axis=-2``) -> hybrid ``(..., nz, nyc, nx)`` space.
        2. Multiply by ``phasefac_ntft`` (``exp(+i * deltaKx(z, ky) * x)``) when
           ``positive_phase=True`` or ``phasefacminus_ntft`` (``exp(-i * deltaKx(z, ky) * x)``)
           when ``positive_phase=False``, and/or ``phasefac_exb``.
        3. 1D ``irfft`` in ``y`` (``axis=-2``, ``n=ny``) -> real ``(..., nz, ny, nx)`` space.
        """
        use_ntft = self.non_twist if non_twist is None else bool(non_twist)
        use_exb = self.exb_shear_phase if exb_shear is None else bool(exb_shear)
        if not use_ntft and not use_exb:
            return arr_real

        c_dtype, r_dtype = _complex_and_real_dtypes(arr_real)
        plan_y = self.mxu_plan_2d.plan_y_real if self.mxu_plan_2d is not None else None
        if self.fft_backend == "mxu":
            yk = mxu_rfft(arr_real.astype(r_dtype), n=self.ny, axis=-2, plan=plan_y).astype(c_dtype)
        else:
            yk = jnp.fft.rfft(arr_real.astype(r_dtype), axis=-2).astype(c_dtype)

        if use_ntft and self.phasefac_ntft_full.size > 0:
            pf_ntft = (
                self.phasefac_ntft_full.astype(c_dtype)
                if positive_phase
                else self.phasefacminus_ntft_full.astype(c_dtype)
            )
            yk = yk * pf_ntft

        if use_exb and phasefac_exb is not None:
            pf_exb = (
                phasefac_exb.astype(c_dtype)
                if positive_phase
                else (
                    phasefacminus_exb.astype(c_dtype)
                    if phasefacminus_exb is not None
                    else jnp.conj(phasefac_exb.astype(c_dtype))
                )
            )
            if pf_exb.shape[-2:] == (self.naky, self.nx) and self.nyc > self.naky:
                ones_pad = jnp.ones(
                    (*pf_exb.shape[:-2], self.nyc - self.naky, self.nx), dtype=c_dtype
                )
                pf_exb = jnp.concatenate([pf_exb, ones_pad], axis=-2)
            yk = yk * pf_exb

        if self.fft_backend == "mxu":
            return mxu_irfft(yk, n=self.ny, axis=-2, plan=plan_y).astype(r_dtype)
        return jnp.fft.irfft(yk, n=self.ny, axis=-2).astype(r_dtype)

    def r2c(
        self,
        arr_real: jax.Array,
        *,
        accumulate: bool = False,
        target: jax.Array | None = None,
        full_output: bool = False,
    ) -> jax.Array:
        """Normalized & dealiased 2D real -> spectral transform matching ``GradPerp::R2C``.

        Ported strictly from ``grad_perp.cu:L249-262`` and ``mask_and_scale_kernel``
        (``device_funcs.cu:L1058-1076``): applies ``1 / (nx * ny)`` normalization
        (``norm="forward"``) and masks out dealiased modes and ``(iky=0, ikx=0)``.
        """
        c_dtype, r_dtype = _complex_and_real_dtypes(arr_real)
        if self.fft_backend == "mxu":
            spec_full = mxu_rfft2(
                arr_real.astype(r_dtype),
                s=(self.nx, self.ny),
                axes=(-1, -2),
                norm="forward",
                plan=self.mxu_plan_2d,
            ).astype(c_dtype)
        else:
            spec_full = jnp.fft.rfft2(
                arr_real.astype(r_dtype),
                s=(self.nx, self.ny),
                axes=(-1, -2),
                norm="forward",
            ).astype(c_dtype)

        if full_output:
            res = jnp.where(self.unmasked_full, spec_full, jnp.zeros_like(spec_full))
        else:
            res = self.truncate_to_active(spec_full)

        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def qvar(self, arr: jax.Array, n: int | None = None) -> jax.Array:
        """Port ``GradPerp::qvar`` (``grad_perp.cu:L129-144``): round-trip C2R -> R2C slice."""
        real_grid = self.c2r(arr)
        is_full = arr.shape[-2:] == (self.nyc, self.nx)
        spec_out = self.r2c(real_grid, accumulate=False, full_output=is_full)
        flat = spec_out.reshape(-1)
        return flat if n is None else flat[:n]


def init_grad_perp(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry | None = None,
    *,
    dtype: jnp.dtype = jnp.float64,
    fft_backend: str | None = None,
) -> GradPerp:
    """Initialize :class:`GradPerp` in IEEE ``float64`` before optional cast to ``dtype``."""
    r_dtype = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32
    fb_req = cfg.fft_backend if fft_backend is None else fft_backend
    resolved_fb = resolve_fft_backend(
        fb_req,
        nx=int(grids.nx),
        ny=int(grids.ny),
        nz=max(1, int(grids.nz)),
    )
    mxu_plan64 = (
        init_mxu_2d_plan(int(grids.nx), int(grids.ny), dtype=jnp.float64)
        if resolved_fb == "mxu"
        else None
    )
    kx64 = jnp.asarray(grids.kx, dtype=jnp.float64)
    ky64 = jnp.asarray(grids.ky, dtype=jnp.float64)
    kx_full64 = jnp.asarray(grids.kx_full, dtype=jnp.float64)
    ky_full64 = jnp.asarray(grids.ky_full, dtype=jnp.float64)
    x64 = jnp.asarray(grids.x, dtype=jnp.float64)
    kxfac = float(geo.kxfac) if geo is not None else float(cfg.kxfac)

    if cfg.non_twist and geo is not None and geo.delta_kx.size > 0:
        dkx64 = jnp.asarray(geo.delta_kx, dtype=jnp.float64)
        c_pf_dtype = jnp.complex64 if r_dtype == jnp.float32 else jnp.complex128
        np_c_dtype = np.complex64 if r_dtype == jnp.float32 else np.complex128
        if geo.phasefac_ntft.size > 0:
            pf_pos_np = np.asarray(geo.phasefac_ntft, dtype=np.complex128)
            pf_neg_np = np.asarray(geo.phasefacminus_ntft, dtype=np.complex128)
        else:
            phase_np = (
                np.asarray(dkx64, dtype=np.float64)[:, :, None]
                * np.asarray(x64, dtype=np.float64)[None, None, :]
            )
            pf_pos_np = np.exp(1.0j * phase_np)
            pf_neg_np = np.exp(-1.0j * phase_np)

        pad_y = grids.nyc - grids.naky
        if pad_y > 0:
            ones_y_np = np.ones((grids.nz, pad_y, grids.nx), dtype=np.complex128)
            pf_pos_full_np = np.concatenate([pf_pos_np, ones_y_np], axis=1)
            pf_neg_full_np = np.concatenate([pf_neg_np, ones_y_np], axis=1)
        else:
            pf_pos_full_np = pf_pos_np
            pf_neg_full_np = pf_neg_np
        pf_pos = jnp.asarray(pf_pos_np.astype(np_c_dtype), dtype=c_pf_dtype)
        pf_neg = jnp.asarray(pf_neg_np.astype(np_c_dtype), dtype=c_pf_dtype)
        pf_pos_full = jnp.asarray(pf_pos_full_np.astype(np_c_dtype), dtype=c_pf_dtype)
        pf_neg_full = jnp.asarray(pf_neg_full_np.astype(np_c_dtype), dtype=c_pf_dtype)
        pf_pos_nxny = jnp.swapaxes(pf_pos_full, -2, -1)
        pf_neg_nxny = jnp.swapaxes(pf_neg_full, -2, -1)
    else:
        c_pf_dtype = jnp.complex64 if r_dtype == jnp.float32 else jnp.complex128
        dkx64 = jnp.zeros((0, 0), dtype=jnp.float64)
        pf_pos = jnp.zeros((0, 0, 0), dtype=c_pf_dtype)
        pf_neg = jnp.zeros((0, 0, 0), dtype=c_pf_dtype)
        pf_pos_full = jnp.zeros((0, 0, 0), dtype=c_pf_dtype)
        pf_neg_full = jnp.zeros((0, 0, 0), dtype=c_pf_dtype)
        pf_pos_nxny = jnp.zeros((0, 0, 0), dtype=c_pf_dtype)
        pf_neg_nxny = jnp.zeros((0, 0, 0), dtype=c_pf_dtype)

    gp64 = GradPerp(
        kx=kx64,
        ky=ky64,
        kx_full=kx_full64,
        ky_full=ky_full64,
        active_mask=grids.active_mask,
        unmasked_full=grids.unmasked_full,
        x=x64,
        delta_kx=dkx64,
        phasefac_ntft=pf_pos,
        phasefacminus_ntft=pf_neg,
        phasefac_ntft_full=pf_pos_full,
        phasefacminus_ntft_full=pf_neg_full,
        phasefac_ntft_nxny=pf_pos_nxny,
        phasefacminus_ntft_nxny=pf_neg_nxny,
        nx=int(grids.nx),
        ny=int(grids.ny),
        nz=int(grids.nz),
        nyc=int(grids.nyc),
        naky=int(grids.naky),
        nakx=int(grids.nakx),
        x0=float(grids.x0),
        y0=float(grids.y0),
        kxfac=kxfac,
        non_twist=bool(cfg.non_twist),
        exb_shear=bool(cfg.exb_shear),
        exb_shear_phase=bool(cfg.exb_shear_phase),
        mxu_plan_2d=mxu_plan64,
        fft_backend=resolved_fb,
    )
    return gp64 if r_dtype == jnp.float64 else gp64.astype(r_dtype)


# ---------------------------------------------------------------------------
# 2. Equilibrium ExB Flow Shear (ExB_GK in exb.cu & device_funcs.cu:L3526-3751)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ExBShearState:
    """Time-dependent equilibrium ``ExB`` flow-shear state (``ExB_GK`` in ``exb.cu``).

    Ported from ``gx/src/exb.cu`` and ``device_funcs.cu:L3526-3751``, with three
    C++ bugs deliberately fixed:
    1. ``exb.cu:L72`` calls ``field_shift(fieldsTmp->phi, f->phi)`` without copying ``fieldsTmp``
       back to ``f->phi``; :func:`flow_shear_shift` returns the shifted ``Fields`` directly.
    2. ``ts_k10.cu:L103-105`` shifts ``G_q1`` and then immediately overwrites ``G_q1`` with
       unshifted ``G``; ``step_gyrokinetic`` shifts ``G`` before stage 1.
    3. ``kxstar_phase_shift`` (``device_funcs.cu:L3664-3673``) uses the pre-jump ``kxbar_new``
       when computing ``delta_kx_exb``; :func:`flow_shear_shift` recomputes ``round(kxstar/dkx)``
       from the post-jump ``kxstar_next``.
    """

    kxstar: jax.Array  # (naky, nakx) float64/float32
    kxbar_ikx_new: jax.Array  # (naky, nakx) int32
    kxbar_ikx_old: jax.Array  # (naky, nakx) int32
    phasefac_exb: jax.Array  # (nyc, nx) complex
    phasefacminus_exb: jax.Array  # (nyc, nx) complex
    phasefac_exb_nxny: jax.Array  # (nx, nyc) complex
    phasefacminus_exb_nxny: jax.Array  # (nx, nyc) complex
    kperp2: jax.Array  # (nz, naky, nakx) float64/float32
    omegad: jax.Array  # (nz, naky, nakx) float64/float32
    cv_d: jax.Array  # (nz, naky, nakx) float64/float32
    gb_d: jax.Array  # (nz, naky, nakx) float64/float32
    ky: jax.Array  # (naky,) float64/float32
    x: jax.Array  # (nx,) float64/float32

    g_exb: float
    x0: float
    shat: float
    non_twist: bool
    exb_shear_phase: bool

    def tree_flatten(self):
        children = (
            self.kxstar,
            self.kxbar_ikx_new,
            self.kxbar_ikx_old,
            self.phasefac_exb,
            self.phasefacminus_exb,
            self.phasefac_exb_nxny,
            self.phasefacminus_exb_nxny,
            self.kperp2,
            self.omegad,
            self.cv_d,
            self.gb_d,
            self.ky,
            self.x,
        )
        aux_data = (
            self.g_exb,
            self.x0,
            self.shat,
            self.non_twist,
            self.exb_shear_phase,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> ExBShearState:
        g_exb, x0, shat, non_twist, exb_shear_phase = aux_data
        (
            kxstar,
            kxbar_ikx_new,
            kxbar_ikx_old,
            phasefac_exb,
            phasefacminus_exb,
            phasefac_exb_nxny,
            phasefacminus_exb_nxny,
            kperp2,
            omegad,
            cv_d,
            gb_d,
            ky,
            x,
        ) = children
        return cls(
            kxstar=kxstar,
            kxbar_ikx_new=kxbar_ikx_new,
            kxbar_ikx_old=kxbar_ikx_old,
            phasefac_exb=phasefac_exb,
            phasefacminus_exb=phasefacminus_exb,
            phasefac_exb_nxny=phasefac_exb_nxny,
            phasefacminus_exb_nxny=phasefacminus_exb_nxny,
            kperp2=kperp2,
            omegad=omegad,
            cv_d=cv_d,
            gb_d=gb_d,
            ky=ky,
            x=x,
            g_exb=g_exb,
            x0=x0,
            shat=shat,
            non_twist=non_twist,
            exb_shear_phase=exb_shear_phase,
        )

    def astype(self, dtype: jnp.dtype) -> ExBShearState:
        """Cast floating/complex leaves to runtime ``dtype``."""
        target = jnp.dtype(dtype)
        r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        c_target = jnp.complex128 if r_target == jnp.float64 else jnp.complex64

        def _cast(leaf):
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                return (
                    leaf.astype(c_target)
                    if isinstance(leaf, jax.core.Tracer)
                    else jnp.asarray(np.asarray(leaf), dtype=c_target)
                )
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                return (
                    leaf.astype(r_target)
                    if isinstance(leaf, jax.core.Tracer)
                    else jnp.asarray(np.asarray(leaf), dtype=r_target)
                )
            return leaf

        return jax.tree_util.tree_map(_cast, self)


def init_exb_shear(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> ExBShearState:
    """Initialize :class:`ExBShearState` (``init_kxstar_kxbar_phasefac``, ``L3526-3541``)."""
    r_dtype = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32
    c_dtype = jnp.complex128 if r_dtype == jnp.float64 else jnp.complex64

    kx64 = jnp.asarray(grids.kx, dtype=jnp.float64)
    kxstar = jnp.broadcast_to(kx64[None, :], (grids.naky, grids.nakx)).astype(r_dtype)
    ikx_act = _c_round_jax(kx64 * float(grids.x0)).astype(jnp.int32)
    kxbar = jnp.broadcast_to(ikx_act[None, :], (grids.naky, grids.nakx))
    pf = jnp.ones((grids.nyc, grids.nx), dtype=c_dtype)
    pf_nxny = jnp.ones((grids.nx, grids.nyc), dtype=c_dtype)

    return ExBShearState(
        kxstar=kxstar,
        kxbar_ikx_new=kxbar,
        kxbar_ikx_old=kxbar,
        phasefac_exb=pf,
        phasefacminus_exb=pf,
        phasefac_exb_nxny=pf_nxny,
        phasefacminus_exb_nxny=pf_nxny,
        kperp2=geo.kperp2.astype(r_dtype),
        omegad=geo.omegad.astype(r_dtype),
        cv_d=geo.cv_d.astype(r_dtype),
        gb_d=geo.gb_d.astype(r_dtype),
        ky=grids.ky.astype(r_dtype),
        x=grids.x.astype(r_dtype),
        g_exb=float(cfg.g_exb),
        x0=float(grids.x0),
        shat=float(geo.shat),
        non_twist=bool(cfg.non_twist),
        exb_shear_phase=bool(cfg.exb_shear_phase),
    )


def _shift_active_kx_array(
    arr: jax.Array,
    exb_state: ExBShearState,
    grids: Grids,
) -> jax.Array:
    """Apply discrete ``kx`` remapping via gather-free slice concatenation (``L3684-3751``)."""
    nakx = grids.nakx
    if nakx <= 1 or exb_state.g_exb == 0.0:
        return arr

    half = nakx // 2
    z1 = jnp.zeros((*arr.shape[:-1], 1), dtype=arr.dtype)
    if exb_state.g_exb > 0.0:
        # new[ikx_signed] = old[ikx_signed + 1] for ikx_signed < +half, 0 at +half
        shifted = jnp.concatenate(
            [arr[..., 1 : half + 1], z1, arr[..., half + 2 :], arr[..., 0:1]],
            axis=-1,
        )
    else:
        # new[ikx_signed] = old[ikx_signed - 1] for ikx_signed > -half, 0 at -half
        shifted = jnp.concatenate(
            [arr[..., -1:], arr[..., :half], z1, arr[..., half + 1 : -1]],
            axis=-1,
        )

    jumped_row = (jnp.arange(grids.naky, dtype=jnp.int32) > 0) & (
        exb_state.kxbar_ikx_new[:, 0] != exb_state.kxbar_ikx_old[:, 0]
    )
    out = jnp.where(jumped_row[:, None], shifted, arr)
    return jnp.where(grids.active_mask, out, jnp.zeros_like(out))


def flow_shear_g_shift(
    G: jax.Array,
    exb_state: ExBShearState,
    grids: Grids,
) -> jax.Array:
    """Shift distribution ``G`` when ``kxstar`` crosses a ``kx`` grid point (``exb.cu:L75-80``)."""
    return _shift_active_kx_array(G, exb_state, grids)


def flow_shear_shift(
    exb_state: ExBShearState,
    fields: Fields,
    grids: Grids,
    geo: Geometry,
    dt: float | jax.Array,
) -> tuple[ExBShearState, Fields]:
    """Update ``kxstar``, geometry operators, phase factors, and fields (``exb.cu:L43-73``)."""
    r_dtype = exb_state.kxstar.dtype
    c_dtype = exb_state.phasefac_exb.dtype
    dkx = jnp.asarray(1.0 / float(exb_state.x0), dtype=r_dtype)
    dt_r = jnp.asarray(dt, dtype=r_dtype)
    g_exb_r = jnp.asarray(exb_state.g_exb, dtype=r_dtype)

    ky_r = exb_state.ky.astype(r_dtype)[:, None]
    idy_pos = (jnp.arange(grids.naky, dtype=jnp.int32) > 0)[:, None]

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

    sign_exb = jnp.asarray(1.0 if exb_state.g_exb > 0.0 else -1.0, dtype=r_dtype)
    jumped = idy_pos & (kxbar_new != kxbar_old)
    kxstar_next = jnp.where(jumped, kxstar_raw + sign_exb * dkx, kxstar_raw)

    if exb_state.exb_shear_phase:
        # Residual delta_kx_exb(ky) = kxstar - round(kxstar / dkx) * dkx
        kxbar_cur = _c_round_jax(kxstar_next[:, :1] / dkx)
        dkx_exb = kxstar_next[:, :1] - kxbar_cur * dkx  # (naky, 1)
        phase_act = jnp.where(
            (jnp.arange(grids.naky) > 0)[:, None],
            dkx_exb * exb_state.x.astype(r_dtype)[None, :],
            jnp.zeros((grids.naky, grids.nx), dtype=r_dtype),
        )
        ph = phase_act.astype(r_dtype)
        cos_ph = jnp.cos(ph)
        sin_ph = jnp.sin(ph)
        pf_act = jax.lax.complex(cos_ph, sin_ph).astype(c_dtype)
        pfm_act = jax.lax.complex(cos_ph, -sin_ph).astype(c_dtype)
        pad_y = grids.nyc - grids.naky
        if pad_y > 0:
            ones_pad = jnp.ones((pad_y, grids.nx), dtype=c_dtype)
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

    # geo_shift / geo_shift_ntft (device_funcs.cu:L3542-3599)
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
    idy_pos3 = (jnp.arange(grids.naky) > 0)[None, :, None] & mask3

    if exb_state.non_twist and geo.delta_kx.size > 0:
        shat_inv = jnp.asarray(1.0 / float(exb_state.shat), dtype=r_dtype)
        ft_3 = geo.ftwist.astype(r_dtype)[:, None, None]
        dkx_3 = geo.delta_kx.astype(r_dtype)[:, :, None]
        m0_ov_x0 = (geo.m0.astype(r_dtype) / float(exb_state.x0))[:, :, None]
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
        shat_inv = jnp.asarray(
            1.0 if exb_state.shat == 0.0 else (1.0 / float(exb_state.shat)),
            dtype=r_dtype,
        )
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

    next_state = ExBShearState(
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
        ky=exb_state.ky.astype(r_dtype),
        x=exb_state.x.astype(r_dtype),
        g_exb=exb_state.g_exb,
        x0=exb_state.x0,
        shat=exb_state.shat,
        non_twist=exb_state.non_twist,
        exb_shear_phase=exb_state.exb_shear_phase,
    )

    phi_shifted = _shift_active_kx_array(fields.phi, next_state, grids)
    apar_shifted = _shift_active_kx_array(fields.apar, next_state, grids)
    bpar_shifted = _shift_active_kx_array(fields.bpar, next_state, grids)
    fields_next = Fields(
        phi=phi_shifted,
        apar=apar_shifted,
        bpar=bpar_shifted,
        phi_avg=fields.phi_avg,
    )
    return next_state, fields_next


# ---------------------------------------------------------------------------
# 3. Stochastic / Impulse Forcing (Forcing in forcing.cu & device_funcs.cu:L1365-1376)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ForcingOperator:
    """External moment forcing operator (``KzForcing``, ``KzForcingImpulse``, ``genForcing``).

    Ported from ``gx/src/forcing.cu`` and ``device_funcs.cu:L1365-1376``.
    """

    cos_kz_profile: jax.Array  # (nz,) float64/float32
    forcing_init: bool
    forcing_type: str
    stir_field: str
    forcing_amp: float
    pos_forcing_amp: float
    neg_forcing_amp: float
    forcing_index: int
    idz_target: int
    idy_target: int
    idx_target: int
    forcing_kz: int
    forcing_k2min: int
    forcing_k2max: int

    def tree_flatten(self):
        children = (self.cos_kz_profile,)
        aux_data = (
            self.forcing_init,
            self.forcing_type,
            self.stir_field,
            self.forcing_amp,
            self.pos_forcing_amp,
            self.neg_forcing_amp,
            self.forcing_index,
            self.idz_target,
            self.idy_target,
            self.idx_target,
            self.forcing_kz,
            self.forcing_k2min,
            self.forcing_k2max,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> ForcingOperator:
        (
            forcing_init,
            forcing_type,
            stir_field,
            forcing_amp,
            pos_forcing_amp,
            neg_forcing_amp,
            forcing_index,
            idz_target,
            idy_target,
            idx_target,
            forcing_kz,
            forcing_k2min,
            forcing_k2max,
        ) = aux_data
        (cos_kz_profile,) = children
        return cls(
            cos_kz_profile=cos_kz_profile,
            forcing_init=forcing_init,
            forcing_type=forcing_type,
            stir_field=stir_field,
            forcing_amp=forcing_amp,
            pos_forcing_amp=pos_forcing_amp,
            neg_forcing_amp=neg_forcing_amp,
            forcing_index=forcing_index,
            idz_target=idz_target,
            idy_target=idy_target,
            idx_target=idx_target,
            forcing_kz=forcing_kz,
            forcing_k2min=forcing_k2min,
            forcing_k2max=forcing_k2max,
        )

    def astype(self, dtype: jnp.dtype) -> ForcingOperator:
        r_target = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32
        cos_cast = (
            self.cos_kz_profile.astype(r_target)
            if isinstance(self.cos_kz_profile, jax.core.Tracer)
            else jnp.asarray(np.asarray(self.cos_kz_profile), dtype=r_target)
        )
        return ForcingOperator(
            cos_kz_profile=cos_cast,
            forcing_init=self.forcing_init,
            forcing_type=self.forcing_type,
            stir_field=self.stir_field,
            forcing_amp=self.forcing_amp,
            pos_forcing_amp=self.pos_forcing_amp,
            neg_forcing_amp=self.neg_forcing_amp,
            forcing_index=self.forcing_index,
            idz_target=self.idz_target,
            idy_target=self.idy_target,
            idx_target=self.idx_target,
            forcing_kz=self.forcing_kz,
            forcing_k2min=self.forcing_k2min,
            forcing_k2max=self.forcing_k2max,
        )


def init_forcing(
    cfg: GXConfig,
    grids: Grids,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> ForcingOperator:
    """Initialize :class:`ForcingOperator` matching ``forcing.cu``."""
    r_dtype = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32
    ftype = str(cfg.forcing_type).strip()
    ftype_low = ftype.lower()
    if ftype_low == "heliinj":
        raise NotImplementedError("HeliInj forcing is not yet implemented.")
    if ftype_low not in ("kz", "kzimpulse", "general"):
        raise ValueError(
            f"Unsupported forcing_type={cfg.forcing_type!r}. "
            "Supported: ('Kz', 'KzImpulse', 'general')."
        )

    f_idx = int(cfg.forcing_index)
    total_full = grids.nyc * grids.nx * grids.nz
    if f_idx < 0 or f_idx >= total_full:
        if cfg.forcing_init:
            raise ValueError(f"forcing_index={f_idx} is out of range [0, {total_full - 1}].")
        idz, idy_act, idx_act = 0, 0, 0
    else:
        # In C++ layout, forcing_index = idy + nyc * (idx_full + nx * idz)
        idy = f_idx % grids.nyc
        idx_full = (f_idx // grids.nyc) % grids.nx
        idz = (f_idx // (grids.nyc * grids.nx)) % grids.nz
        active_kx_np = np.asarray(grids.active_kx_idx, dtype=np.int32)
        matches = np.where(active_kx_np == idx_full)[0]
        if cfg.forcing_init:
            if idy >= grids.naky or matches.size == 0:
                raise ValueError(
                    f"forcing_index={f_idx} maps to dealiased mode "
                    f"(idy={idy}, idx_full={idx_full}) "
                    f"outside active grid (naky={grids.naky}, nakx={grids.nakx})."
                )
            idx_act = int(matches[0])
            idy_act = int(idy)
            if not bool(np.asarray(grids.active_mask)[idy_act, idx_act]):
                raise ValueError(
                    f"forcing_index={f_idx} maps to masked mode (idy={idy_act}, ikx={idx_act})."
                )
        else:
            idx_act = int(matches[0]) if matches.size > 0 else 0
            idy_act = min(int(idy), grids.naky - 1)

    z_heli = (jnp.arange(grids.nz, dtype=jnp.float64) / float(max(1, grids.nz))) * (
        2.0 * math.pi * float(grids.zp)
    )
    cos_prof = jnp.cos(float(cfg.forcing_kz) * z_heli).astype(r_dtype)

    return ForcingOperator(
        cos_kz_profile=cos_prof,
        forcing_init=bool(cfg.forcing_init),
        forcing_type=ftype,
        stir_field=str(cfg.stir_field).strip().lower(),
        forcing_amp=float(cfg.forcing_amp),
        pos_forcing_amp=float(cfg.pos_forcing_amp),
        neg_forcing_amp=float(cfg.neg_forcing_amp),
        forcing_index=f_idx,
        idz_target=int(idz),
        idy_target=int(idy_act),
        idx_target=int(idx_act),
        forcing_kz=int(cfg.forcing_kz),
        forcing_k2min=int(cfg.forcing_k2min),
        forcing_k2max=int(cfg.forcing_k2max),
    )


def apply_forcing(
    G: jax.Array,
    forcing_op: ForcingOperator,
    dt: float | jax.Array,
    key: jax.Array | None = None,
    *,
    step_idx: int | jax.Array = 0,
) -> jax.Array:
    """Apply stochastic or impulse Langevin forcing ``Forcing::stir`` (``forcing.cu:L25-187``)."""
    if not forcing_op.forcing_init:
        return G
    if key is None:
        raise ValueError("A JAX PRNGKey is required when forcing_init=True.")

    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    ns = g_c.shape[0]
    nm = g_c.shape[1]
    nl = g_c.shape[2]
    dt_r = jnp.asarray(dt, dtype=r_dtype)
    k1, k2 = jax.random.split(key)

    # Draw independent random forces per species (matching forcing.cu species loop)
    u1 = jnp.clip(jax.random.uniform(k1, (ns,), dtype=r_dtype), 1.0e-12, 1.0)
    phase = jnp.asarray(math.pi, dtype=r_dtype) * (
        2.0 * jax.random.uniform(k2, (ns,), dtype=r_dtype) - 1.0
    )
    f_amp_r = jnp.asarray(forcing_op.forcing_amp, dtype=r_dtype)
    amp = _safe_sqrt(jnp.abs(f_amp_r * dt_r * jnp.log(u1)))

    amp_r = amp.astype(r_dtype)
    phase_r = phase.astype(r_dtype)
    ftype = forcing_op.forcing_type.strip().lower()
    if ftype == "kz":
        # KzForcing is deliberately real-only (forcing.cu:L31-32: rf.y = 0.)
        rf = (amp_r * jnp.cos(phase_r)).astype(c_dtype)
    elif ftype == "kzimpulse":
        rf_raw = jax.lax.complex(amp_r * jnp.cos(phase_r), amp_r * jnp.sin(phase_r)).astype(c_dtype)
        rf = jnp.where(jnp.asarray(step_idx) == 0, rf_raw, jnp.zeros((ns,), dtype=c_dtype))
    else:
        rf = jax.lax.complex(amp_r * jnp.cos(phase_r), amp_r * jnp.sin(phase_r)).astype(c_dtype)

    iz = forcing_op.idz_target
    iy = forcing_op.idy_target
    ix = forcing_op.idx_target
    sfield = forcing_op.stir_field
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


# ---------------------------------------------------------------------------
# 4. Gyrokinetic Nonlinear Operator & Poisson Brackets (Nonlinear_GK in nonlinear.cu)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonlinearOperator:
    """Precomputed gyrokinetic nonlinear operator container (``Nonlinear_GK`` in ``nonlinear.cu``).

    Precomputes ``GradPerp``, ``LaguerreTransform``, Hermite ``m +- 1`` flutter weights, and
    static ``float64`` Bessel ``J_0(alpha)`` and ``tz * 2 * muB * J_1(alpha) / alpha`` tables on
    the Gauss-Laguerre quadrature grid ``(nspecies, nj, nz, naky, nakx)``.
    """

    grad_perp: GradPerp
    laguerre: LaguerreTransform
    j0_grid: jax.Array  # (nspecies, nj, nz, naky, nakx) float64/float32
    tz_2mub_j1_oa_grid: jax.Array  # (nspecies, nj, nz, naky, nakx) float64/float32
    kperp2: jax.Array  # (nz, naky, nakx) float64/float32
    active_mask: jax.Array  # (naky, nakx) bool
    sp_rho2: jax.Array  # (nspecies,)
    sp_vt: jax.Array  # (nspecies,)
    sp_tz: jax.Array  # (nspecies,)
    sqrt_m: jax.Array  # (nhermite,)
    sqrt_mp1: jax.Array  # (nhermite,)

    nspecies: int
    nhermite: int
    nlaguerre: int
    nj: int
    nz: int
    naky: int
    nakx: int
    nx: int
    ny: int
    kxfac: float
    fphi: float
    fapar: float
    fbpar: float
    vpar_max: float
    muB_maxJ: float
    vtmax: float
    tzmax: float
    kx_max: float
    ky_max: float
    g_exb: float
    nonlinear_mode: bool
    non_twist: bool
    exb_shear: bool
    exb_shear_phase: bool
    long_wavelength_gk: bool

    def tree_flatten(self):
        children = (
            self.grad_perp,
            self.laguerre,
            self.j0_grid,
            self.tz_2mub_j1_oa_grid,
            self.kperp2,
            self.active_mask,
            self.sp_rho2,
            self.sp_vt,
            self.sp_tz,
            self.sqrt_m,
            self.sqrt_mp1,
        )
        aux_data = (
            self.nspecies,
            self.nhermite,
            self.nlaguerre,
            self.nj,
            self.nz,
            self.naky,
            self.nakx,
            self.nx,
            self.ny,
            self.kxfac,
            self.fphi,
            self.fapar,
            self.fbpar,
            self.vpar_max,
            self.muB_maxJ,
            self.vtmax,
            self.tzmax,
            self.kx_max,
            self.ky_max,
            self.g_exb,
            self.nonlinear_mode,
            self.non_twist,
            self.exb_shear,
            self.exb_shear_phase,
            self.long_wavelength_gk,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> NonlinearOperator:
        (
            nspecies,
            nhermite,
            nlaguerre,
            nj,
            nz,
            naky,
            nakx,
            nx,
            ny,
            kxfac,
            fphi,
            fapar,
            fbpar,
            vpar_max,
            muB_maxJ,
            vtmax,
            tzmax,
            kx_max,
            ky_max,
            g_exb,
            nonlinear_mode,
            non_twist,
            exb_shear,
            exb_shear_phase,
            long_wavelength_gk,
        ) = aux_data
        (
            grad_perp,
            laguerre,
            j0_grid,
            tz_2mub_j1_oa_grid,
            kperp2,
            active_mask,
            sp_rho2,
            sp_vt,
            sp_tz,
            sqrt_m,
            sqrt_mp1,
        ) = children
        return cls(
            grad_perp=grad_perp,
            laguerre=laguerre,
            j0_grid=j0_grid,
            tz_2mub_j1_oa_grid=tz_2mub_j1_oa_grid,
            kperp2=kperp2,
            active_mask=active_mask,
            sp_rho2=sp_rho2,
            sp_vt=sp_vt,
            sp_tz=sp_tz,
            sqrt_m=sqrt_m,
            sqrt_mp1=sqrt_mp1,
            nspecies=nspecies,
            nhermite=nhermite,
            nlaguerre=nlaguerre,
            nj=nj,
            nz=nz,
            naky=naky,
            nakx=nakx,
            nx=nx,
            ny=ny,
            kxfac=kxfac,
            fphi=fphi,
            fapar=fapar,
            fbpar=fbpar,
            vpar_max=vpar_max,
            muB_maxJ=muB_maxJ,
            vtmax=vtmax,
            tzmax=tzmax,
            kx_max=kx_max,
            ky_max=ky_max,
            g_exb=g_exb,
            nonlinear_mode=nonlinear_mode,
            non_twist=non_twist,
            exb_shear=exb_shear,
            exb_shear_phase=exb_shear_phase,
            long_wavelength_gk=long_wavelength_gk,
        )

    @property
    def fft_backend(self) -> str:
        """Resolved perpendicular spectral backend (``'fft'`` or ``'mxu'``)."""
        return self.grad_perp.fft_backend

    def astype(self, dtype: jnp.dtype) -> NonlinearOperator:
        """Cast floating/complex arrays on ``NonlinearOperator`` to runtime ``dtype``."""
        target = jnp.dtype(dtype)
        r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32

        def _hc(arr: jax.Array) -> jax.Array:
            if arr.dtype == r_target:
                return arr
            if isinstance(arr, jax.core.Tracer):
                return arr.astype(r_target)
            return jnp.asarray(np.asarray(arr), dtype=r_target)

        lag_cast = jax.tree_util.tree_map(
            lambda x: (
                _hc(x) if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating) else x
            ),
            self.laguerre,
        )
        return NonlinearOperator(
            grad_perp=self.grad_perp.astype(r_target),
            laguerre=lag_cast,
            j0_grid=_hc(self.j0_grid),
            tz_2mub_j1_oa_grid=_hc(self.tz_2mub_j1_oa_grid),
            kperp2=_hc(self.kperp2),
            active_mask=self.active_mask,
            sp_rho2=_hc(self.sp_rho2),
            sp_vt=_hc(self.sp_vt),
            sp_tz=_hc(self.sp_tz),
            sqrt_m=_hc(self.sqrt_m),
            sqrt_mp1=_hc(self.sqrt_mp1),
            nspecies=self.nspecies,
            nhermite=self.nhermite,
            nlaguerre=self.nlaguerre,
            nj=self.nj,
            nz=self.nz,
            naky=self.naky,
            nakx=self.nakx,
            nx=self.nx,
            ny=self.ny,
            kxfac=self.kxfac,
            fphi=self.fphi,
            fapar=self.fapar,
            fbpar=self.fbpar,
            vpar_max=self.vpar_max,
            muB_maxJ=self.muB_maxJ,
            vtmax=self.vtmax,
            tzmax=self.tzmax,
            kx_max=self.kx_max,
            ky_max=self.ky_max,
            g_exb=self.g_exb,
            nonlinear_mode=self.nonlinear_mode,
            non_twist=self.non_twist,
            exb_shear=self.exb_shear,
            exb_shear_phase=self.exb_shear_phase,
            long_wavelength_gk=self.long_wavelength_gk,
        )


def init_nonlinear_operator(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    lag: LaguerreTransform | None = None,
    grad_perp: GradPerp | None = None,
    *,
    dtype: jnp.dtype = jnp.float64,
    fft_backend: str | None = None,
) -> NonlinearOperator:
    """Precompute static nonlinear operator tables in IEEE ``float64`` before optional cast."""
    ns = grids.nspecies
    if len(cfg.species) != ns:
        raise ValueError(f"Expected {ns} kinetic species in cfg.species, got {len(cfg.species)}.")

    r_target = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32
    lag64 = init_laguerre(grids.nlaguerre, dtype=jnp.float64) if lag is None else lag
    gp64 = (
        init_grad_perp(cfg, grids, geo, dtype=jnp.float64, fft_backend=fft_backend)
        if grad_perp is None
        else grad_perp
    )

    kperp2_64 = jnp.asarray(geo.kperp2, dtype=jnp.float64)
    mu_b64 = jnp.asarray(lag64.roots, dtype=jnp.float64)[None, :, None, None, None]
    sp_rho2 = jnp.asarray([sp.rho2 for sp in cfg.species], dtype=jnp.float64)
    sp_vt = jnp.asarray([sp.vt for sp in cfg.species], dtype=jnp.float64)
    sp_tz = jnp.asarray([sp.tz for sp in cfg.species], dtype=jnp.float64)

    # Precompute J0(alpha) and tz * 2 * muB * (J1(alpha)/alpha) on (nspecies, nj, nz, naky, nakx)
    # (device_funcs.cu:L894-931). Note sp.rho2 == 0 when long_wavelength_gk=True (alpha=0, J0=1).
    alpha = _safe_sqrt(
        2.0 * mu_b64 * kperp2_64[None, None, :, :, :] * sp_rho2[:, None, None, None, None]
    )
    j0_grid64 = bessel_j0(alpha)
    j1_oa64 = j1_over_alpha(alpha)
    tz_2mub_j1_oa64 = sp_tz[:, None, None, None, None] * 2.0 * mu_b64 * j1_oa64

    mask5 = grids.active_mask[None, None, None, :, :]
    j0_grid64 = jnp.where(mask5, j0_grid64, 0.0)
    tz_2mub_j1_oa64 = jnp.where(mask5, tz_2mub_j1_oa64, 0.0)

    m_f = jnp.arange(grids.nhermite, dtype=jnp.float64)
    sqrt_m = jnp.sqrt(m_f)
    sqrt_mp1 = jnp.sqrt(m_f + 1.0)

    op64 = NonlinearOperator(
        grad_perp=gp64,
        laguerre=lag64,
        j0_grid=j0_grid64,
        tz_2mub_j1_oa_grid=tz_2mub_j1_oa64,
        kperp2=kperp2_64,
        active_mask=grids.active_mask,
        sp_rho2=sp_rho2,
        sp_vt=sp_vt,
        sp_tz=sp_tz,
        sqrt_m=sqrt_m,
        sqrt_mp1=sqrt_mp1,
        nspecies=int(grids.nspecies),
        nhermite=int(grids.nhermite),
        nlaguerre=int(grids.nlaguerre),
        nj=int(lag64.nj),
        nz=int(grids.nz),
        naky=int(grids.naky),
        nakx=int(grids.nakx),
        nx=int(grids.nx),
        ny=int(grids.ny),
        kxfac=float(geo.kxfac),
        fphi=float(cfg.fphi),
        fapar=float(cfg.fapar),
        fbpar=float(cfg.fbpar),
        vpar_max=float(grids.vpar_max),
        muB_maxJ=float(grids.muB_maxJ),
        vtmax=float(cfg.vtmax),
        tzmax=float(cfg.tzmax),
        kx_max=float(grids.kx_max),
        ky_max=float(grids.ky_max),
        g_exb=float(cfg.g_exb),
        nonlinear_mode=bool(cfg.nonlinear_mode),
        non_twist=bool(cfg.non_twist),
        exb_shear=bool(cfg.exb_shear),
        exb_shear_phase=bool(cfg.exb_shear_phase),
        long_wavelength_gk=bool(cfg.long_wavelength_gk),
    )
    return op64 if r_target == jnp.float64 else op64.astype(r_target)


def poisson_bracket_real(
    dg_dx: jax.Array,
    dg_dy: jax.Array,
    dchi_dx: jax.Array,
    dchi_dy: jax.Array,
    kxfac: float | jax.Array = 1.0,
) -> jax.Array:
    """Evaluate real-space Poisson bracket ``(dg_dx * dchi_dy - dg_dy * dchi_dx) * kxfac``.

    Ported strictly from ``bracket`` (``device_funcs.cu:L1265-1280``).
    Supports both 6D ``dg_d{x,y}`` of shape ``(nspecies, nhermite, nj, nz, ...)`` with 5D
    ``dchi_d{x,y}`` of shape ``(nspecies, nj, nz, ...)`` and matching-rank inputs.
    """
    r_dtype = dg_dx.dtype
    kxfac_r = jnp.asarray(kxfac, dtype=r_dtype)
    if dg_dx.ndim == 6 and dchi_dx.ndim == 5:
        dchi_dx = dchi_dx[:, None, :, :, :, :]
        dchi_dy = dchi_dy[:, None, :, :, :, :]
    elif dg_dx.ndim == 5 and dchi_dx.ndim == 4:
        dchi_dx = dchi_dx[None, :, :, :, :]
        dchi_dy = dchi_dy[None, :, :, :, :]
    return (dg_dx * dchi_dy - dg_dy * dchi_dx) * kxfac_r


def apply_nl_flutter(
    rhs: jax.Array,
    nl_apar: jax.Array,
    sp_vt: jax.Array,
    sqrt_m: jax.Array,
    sqrt_mp1: jax.Array,
    active_mask: jax.Array,
) -> jax.Array:
    """Add electromagnetic ``A_par`` flutter coupling ``m +- 1`` (``nl_flutter``, ``L1282-1302``).

    Evaluates ``rhs - vt_s * (sqrt(m + 1) * nl_apar[m + 1] + sqrt(m) * nl_apar[m - 1])``
    using scatter-free Hermite ghost padding along ``axis=1``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(rhs)
    nl_c = nl_apar.astype(c_dtype)
    vt6 = sp_vt.astype(r_dtype)[:, None, None, None, None, None]
    sm6 = sqrt_m.astype(r_dtype)[None, :, None, None, None, None]
    smp1_6 = sqrt_mp1.astype(r_dtype)[None, :, None, None, None, None]

    nl_pad = jnp.pad(nl_c, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
    flutter = -vt6 * (smp1_6 * nl_pad[:, 2:] + sm6 * nl_pad[:, :-2])
    out = rhs.astype(c_dtype) + flutter
    mask6 = active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, out, jnp.zeros_like(out))


def _gyroaverage_potentials_to_grid(
    fields: Fields,
    nl_op: NonlinearOperator,
    *,
    exb_state: ExBShearState | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute ``(J0_phi_bpar, J0_apar)`` on ``(nspecies, nj, nz, naky, nakx)``."""
    c_dtype, r_dtype = _complex_and_real_dtypes(fields.phi)
    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]
    fphi = jnp.asarray(nl_op.fphi, dtype=r_dtype)
    fapar = jnp.asarray(nl_op.fapar, dtype=r_dtype)
    fbpar = jnp.asarray(nl_op.fbpar, dtype=r_dtype)
    mask5 = nl_op.active_mask[None, None, None, :, :]

    if exb_state is not None and nl_op.g_exb != 0.0:
        # Vectorized runtime-dtype Bessel evaluation on time-shifted kperp2 (zero f64 ops in f32)
        kp2 = exb_state.kperp2.astype(r_dtype)[None, None, :, :, :]
        mu_b = nl_op.laguerre.roots.astype(r_dtype)[None, :, None, None, None]
        rho2 = nl_op.sp_rho2.astype(r_dtype)[:, None, None, None, None]
        tz = nl_op.sp_tz.astype(r_dtype)[:, None, None, None, None]
        two = jnp.asarray(2.0, dtype=r_dtype)

        alpha = _safe_sqrt(two * mu_b * kp2 * rho2)
        j0_g = _bessel_j0_runtime(alpha)
        if nl_op.fbpar > 0.0:
            j1_oa_g = _j1_over_alpha_runtime(alpha)
            tz_j1_g = tz * two * mu_b * j1_oa_g
            j0_phi_bpar = j0_g * (phi * fphi) + tz_j1_g * (bpar * fbpar)
        else:
            j0_phi_bpar = j0_g * (phi * fphi)
    else:
        j0_g = nl_op.j0_grid.astype(r_dtype)
        if nl_op.fbpar > 0.0:
            tz_j1_g = nl_op.tz_2mub_j1_oa_grid.astype(r_dtype)
            j0_phi_bpar = j0_g * (phi * fphi) + tz_j1_g * (bpar * fbpar)
        else:
            j0_phi_bpar = j0_g * (phi * fphi)

    j0_phi_bpar = jnp.where(mask5, j0_phi_bpar, jnp.zeros_like(j0_phi_bpar))
    if nl_op.fapar > 0.0:
        j0_apar = jnp.where(mask5, j0_g * (apar * fapar), jnp.zeros_like(j0_phi_bpar))
    else:
        j0_apar = jnp.zeros_like(j0_phi_bpar)
    return j0_phi_bpar, j0_apar


def nonlinear_rhs(
    G: jax.Array,
    fields: Fields,
    nl_op: NonlinearOperator,
    *,
    exb_state: ExBShearState | None = None,
) -> jax.Array:
    """Evaluate full gyrokinetic nonlinear tendency ``Nonlinear_GK::nlps`` (``nonlinear.cu:L206``).

    Computes the ``3/2``-dealiased Poisson brackets on the Gauss-Laguerre ``mu * B`` grid using
    internal ``(..., nz, nx, ny)`` real-space buffers so ``irfftn``/``rfftn`` and ``phase_mult``
    execute with zero ``stablehlo.transpose`` ops on full real arrays:
    1. Electrostatic ``phi`` + compressional ``bpar`` bracket:
       ``NL_m += {g_m, J_0 phi fphi + tz * 2 muB (J_1(alpha)/alpha) bpar fbpar}``
    2. Electromagnetic ``apar`` flutter bracket (when ``fapar > 0``):
       ``NL_m += -vt_s * (sqrt(m + 1) * {g_{m+1}, J_0 apar fapar}
                        + sqrt(m)     * {g_{m-1}, J_0 apar fapar})``
    3. Supports non-twisting flux-tube (``non_twist = True``) real-space ballooning phase factors
       ``phasefac_ntft`` / ``phasefacminus_ntft`` and equilibrium ``ExB`` flow shear.
    """
    expected_shape = (
        nl_op.nspecies,
        nl_op.nhermite,
        nl_op.nlaguerre,
        nl_op.nz,
        nl_op.naky,
        nl_op.nakx,
    )
    if G.shape != expected_shape:
        raise ValueError(f"Expected G shape {expected_shape}, got {G.shape}.")

    c_dtype, _ = _complex_and_real_dtypes(G)
    if not nl_op.nonlinear_mode:
        return jnp.zeros(expected_shape, dtype=c_dtype)

    gp = nl_op.grad_perp
    lag = nl_op.laguerre
    mask6 = nl_op.active_mask[None, None, None, None, :, :]
    g_c = jnp.where(mask6, G.astype(c_dtype), jnp.zeros_like(G, dtype=c_dtype))

    kxstar = exb_state.kxstar if exb_state is not None else None
    pf_exb_nxny = exb_state.phasefac_exb_nxny if exb_state is not None else None
    pfm_exb_nxny = exb_state.phasefacminus_exb_nxny if exb_state is not None else None
    use_exb_phase = exb_state is not None and exb_state.exb_shear_phase
    use_phase = nl_op.non_twist or use_exb_phase

    # 1. Compute real-space dG_m/dx and dG_m/dy on (nspecies, nhermite, nj, nz, nx, ny)
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

    # 2. Gyroaverage potentials onto Gauss-Laguerre mu*B grid (nonlinear.cu:L231-232, L266-270)
    j0_phi_bpar, j0_apar = _gyroaverage_potentials_to_grid(fields, nl_op, exb_state=exb_state)

    # 3. Electrostatic phi + compressional bpar Poisson bracket (nonlinear.cu:L275-293)
    if nl_op.fphi != 0.0 or nl_op.fbpar > 0.0:
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

        g_res_real = poisson_bracket_real(dg_dx, dg_dy, dj0phi_dx, dj0phi_dy, nl_op.kxfac)
        dG_res = laguerre_to_spectral(g_res_real, lag, axis=2)
        if use_phase:
            dG_res = gp.phase_mult_nxny(
                dG_res,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=False,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )
        rhs_out = gp.r2c_nxny(dG_res)
    else:
        rhs_out = jnp.zeros(expected_shape, dtype=c_dtype)

    # 4. Electromagnetic apar flutter bracket coupling m +- 1 (nonlinear.cu:L230-256, L295-303)
    if nl_op.fapar > 0.0:
        if nl_op.non_twist:
            dj0apar_dx = gp.ntft_dx_c2r_nxny(j0_apar, kxstar=kxstar)
        else:
            dj0apar_dx = gp.dx_c2r_nxny(j0_apar, kxstar=kxstar)
        if use_phase:
            dj0apar_dx = gp.phase_mult_nxny(
                dj0apar_dx,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=True,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )

        dj0apar_dy = gp.dy_c2r_nxny(j0_apar)
        if use_phase:
            dj0apar_dy = gp.phase_mult_nxny(
                dj0apar_dy,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=True,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )

        apar_res_real = poisson_bracket_real(dg_dx, dg_dy, dj0apar_dx, dj0apar_dy, nl_op.kxfac)
        dG_apar = laguerre_to_spectral(apar_res_real, lag, axis=2)
        if use_phase:
            dG_apar = gp.phase_mult_nxny(
                dG_apar,
                non_twist=nl_op.non_twist,
                exb_shear=use_exb_phase,
                positive_phase=False,
                phasefac_exb_nxny=pf_exb_nxny,
                phasefacminus_exb_nxny=pfm_exb_nxny,
            )
        nl_apar = gp.r2c_nxny(dG_apar)
        rhs_out = apply_nl_flutter(
            rhs_out,
            nl_apar,
            nl_op.sp_vt,
            nl_op.sqrt_m,
            nl_op.sqrt_mp1,
            nl_op.active_mask,
        )

    return jnp.where(mask6, rhs_out, jnp.zeros_like(rhs_out))


def compute_nonlinear_max_frequency(
    fields: Fields,
    nl_op: NonlinearOperator,
    *,
    omega_linear: tuple[float | jax.Array, float | jax.Array, float | jax.Array] = (0.0, 0.0, 0.0),
    exb_state: ExBShearState | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Estimate max nonlinear CFL frequencies ``Nonlinear_GK::get_max_frequency``.

    Ported strictly from ``gx/src/nonlinear.cu:L306-345``:
    - ``vpar_max = grids.vpar_max * pars.vtmax``
    - ``muB_max = grids.muB_maxJ * pars.tzmax``
    - ``vmax_y = max(|dxC2R(phi)| + (fapar>0) * vpar_max * |dxC2R(apar)|
                     + (fbpar>0) * muB_max * |dxC2R(bpar)|)``
    - ``vmax_x = max(|dyC2R(phi)| + (fapar>0) * vpar_max * |dyC2R(apar)|
                     + (fbpar>0) * muB_max * |dyC2R(bpar)|)``
    - ``omega_max[0] = max(omega_linear[0], |kxfac| * kx_max * vmax_x * 0.5)``
    - ``omega_max[1] = max(omega_linear[1], |kxfac| * ky_max * vmax_y * 0.5)``
    - ``omega_max[2] = omega_linear[2]``
    """
    _, r_dtype = _complex_and_real_dtypes(fields.phi)
    w0_lin = jnp.asarray(omega_linear[0], dtype=r_dtype)
    w1_lin = jnp.asarray(omega_linear[1], dtype=r_dtype)
    w2_lin = jnp.asarray(omega_linear[2], dtype=r_dtype)

    gp = nl_op.grad_perp
    kxstar = exb_state.kxstar if exb_state is not None else None
    vpar_max = jnp.asarray(nl_op.vpar_max * nl_op.vtmax, dtype=r_dtype)
    mub_max = jnp.asarray(nl_op.muB_maxJ * nl_op.tzmax, dtype=r_dtype)

    # y-direction velocity bound from d/dx of potentials (nonlinear.cu:L310-325)
    vy_grid = jnp.abs(gp.dx_c2r_nxny(fields.phi, kxstar=kxstar))
    if nl_op.fapar > 0.0:
        vy_grid = vy_grid + vpar_max * jnp.abs(gp.dx_c2r_nxny(fields.apar, kxstar=kxstar))
    if nl_op.fbpar > 0.0:
        vy_grid = vy_grid + mub_max * jnp.abs(gp.dx_c2r_nxny(fields.bpar, kxstar=kxstar))
    vmax_y = jnp.max(vy_grid)

    # x-direction velocity bound from d/dy of potentials (nonlinear.cu:L327-341)
    vx_grid = jnp.abs(gp.dy_c2r_nxny(fields.phi))
    if nl_op.fapar > 0.0:
        vx_grid = vx_grid + vpar_max * jnp.abs(gp.dy_c2r_nxny(fields.apar))
    if nl_op.fbpar > 0.0:
        vx_grid = vx_grid + mub_max * jnp.abs(gp.dy_c2r_nxny(fields.bpar))
    vmax_x = jnp.max(vx_grid)

    scale = jnp.asarray(0.5 * abs(nl_op.kxfac), dtype=r_dtype)
    w0_nl = scale * jnp.asarray(nl_op.kx_max, dtype=r_dtype) * vmax_x
    w1_nl = scale * jnp.asarray(nl_op.ky_max, dtype=r_dtype) * vmax_y

    w0 = jnp.maximum(w0_lin, w0_nl)
    w1 = jnp.maximum(w1_lin, w1_nl)
    return w0, w1, w2_lin


__all__ = [
    "ExBShearState",
    "ForcingOperator",
    "GradPerp",
    "NonlinearOperator",
    "apply_forcing",
    "apply_nl_flutter",
    "compute_nonlinear_max_frequency",
    "flow_shear_g_shift",
    "flow_shear_shift",
    "init_exb_shear",
    "init_forcing",
    "init_grad_perp",
    "init_nonlinear_operator",
    "nonlinear_rhs",
    "poisson_bracket_real",
]
