"""Linear gyrokinetic right-hand side, parallel derivatives, closures, and linear steppers.

Ported strictly from:
- ``gx/include/{linear.h,grad_parallel.h,closures.h,timestepper.h}``
- ``gx/src/{linear.cu,grad_parallel.cu,closures.cu,smith_par_closure.cu}``
- ``gx/src/device_funcs.cu`` (``streaming_rhs``, ``rhs_linear``,
  ``conservation_terms``, ``hypercollisions``, ``hypercollisions_kz``, ``hyperdiff``,
  ``HB_hyper``, ``get_s01``, ``get_s1``, ``dampEnds_linked``, ``beer_toroidal_closures``,
  ``smith_perp_toroidal_closures``, ``growthRates``, ``Phi2_summand``, ``i_kz``, ``mkz2``,
  ``abs_kz``, ``zfts``, ``kz_dealias``)
- ``gx/src/{ts_rk3.cu,ts_rk4.cu,ts_sspx2.cu,ts_sspx3.cu,ts_k10.cu}``

Numerical & layout notes:
- State ``G`` uses Option A layout ``(nspecies, nhermite, nlaguerre, nz, naky, nakx)``;
  fields use ``(nz, naky, nakx)``, so the field-line coordinate ``z`` is always ``axis=-3``.
- Static FLR drive arrays, Hermite/Laguerre stencil weights, Padé closure coefficients, and
  hyperdiffusive / hypercollisional profiles are precomputed in IEEE ``float64`` before
  optional ``.astype(jnp.float32)`` runtime execution in ``complex64``.
- Inner products pass ``precision=jax.lax.Precision.HIGHEST`` to prevent ``bfloat16`` MXU
  truncation on GCP TPUs.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigs as scipy_eigs

from gx_jax.config import GXConfig
from gx_jax.fields import (
    Fields,
    FieldSolver,
    _complex_and_real_dtypes,
    _safe_divide,
    compute_conservation_moments,
    field_line_average,
    init_field_solver,
    mask_g,
    solve_fields,
)
from gx_jax.geometry_nc import Geometry
from gx_jax.grids import Grids
from gx_jax.laguerre import jflr_b
from gx_jax.mxu_fft import (
    MXU1DPlan,
    init_mxu_1d_plan,
    mxu_fft,
    mxu_ifft,
    resolve_fft_backend,
)

# Beer 4+2 toroidal closure complex coefficients (closures.cu:L14-35)
_BEER42_NU_TUPLE: tuple[complex, ...] = (
    0.0 + 0.0j,
    2.019 - 1.620j,
    0.433 + 1.018j,
    -0.256 + 1.487j,
    -0.070 - 1.382j,
    -8.927 + 12.649j,
    8.094 + (12.638 - 11.0) * 1.0j,
    13.720 + (5.139 - 3.0) * 1.0j,
    3.368 - 8.110j,
    1.974 + (-1.984 - 1.0) * 1.0j,
    8.269 + (2.060 - 7.0) * 1.0j,
)

# Smith perpendicular toroidal closure lookup table (closures.cu:L87-159)
_SMITH_PERP_TABLE: dict[tuple[int, int], tuple[complex, ...]] = {
    (4, 3): (-2.10807 + 0.574549j, -1.25931 + 0.80951j, -0.181713 + 0.249684j),
    (5, 3): (-2.24233 + 0.551885j, -1.49324 + 0.836156j, -0.272805 + 0.292545j),
    (5, 4): (
        -2.8197 + 0.679165j,
        -2.63724 + 1.4362j,
        -0.896854 + 0.920214j,
        -0.0731348 + 0.167287j,
    ),
    (6, 3): (-2.33763 + 0.527272j, -1.66731 + 0.835661j, -0.346277 + 0.313484j),
    (6, 4): (
        -2.97138 + 0.66065j,
        -3.01477 + 1.48449j,
        -1.18109 + 1.04256j,
        -0.13387 + 0.221786j,
    ),
    (6, 5): (
        -3.53482 + 0.771299j,
        -4.52836 + 2.17916j,
        -2.50811 + 2.14358j,
        -0.537375 + 0.839509j,
        -0.0226971 + 0.102349j,
    ),
    (8, 3): (-2.46437 + 0.482544j, -1.90784 + 0.806717j, -0.454126 + 0.32645j),
    (8, 4): (
        -3.17353 + 0.616513j,
        -3.54865 + 1.48678j,
        -1.62127 + 1.15187j,
        -0.244346 + 0.283442j,
    ),
}


def smith_perp_get_as(nlaguerre: int, q: int) -> np.ndarray:
    """Return complex ``SmithPerp`` closure coefficients ``Aclos`` (``closures.cu:L87-163``)."""
    key = (int(nlaguerre), int(q))
    if key not in _SMITH_PERP_TABLE:
        raise ValueError(
            f"Unsupported Smith perpendicular closure (nlaguerre={nlaguerre}, smith_perp_q={q}). "
            f"Supported pairs: {sorted(_SMITH_PERP_TABLE.keys())}."
        )
    return np.asarray(_SMITH_PERP_TABLE[key], dtype=np.complex128)


def _smith_par_hermite_coeffs(n: int, scaling: float, kind: str) -> np.ndarray:
    """Normalized Hermite/conjugate polynomial coefficients (``smith_par_closure.cu:L285-315``)."""
    mat = np.zeros((n + 1, n + 1), dtype=np.complex128)
    if kind == "P":
        mat[0, 0] = 1.0
        if n >= 1:
            mat[1, 0] = 0.0
            mat[1, 1] = math.sqrt(2.0) * scaling
    elif kind == "Q":
        mat[0, 0] = 0.0
        if n >= 1:
            mat[1, 0] = math.sqrt(2.0) * scaling
            mat[1, 1] = 0.0
    else:
        raise ValueError(f"Unknown Hermite family {kind!r}.")

    for i in range(2, n + 1):
        di = float(i)
        s_prev2 = -math.sqrt((di - 1.0) / di)
        s_prev1 = math.sqrt(2.0 / di) * scaling
        mat[i, 0] = s_prev2 * mat[i - 2, 0]
        for j in range(1, i + 1):
            val = 0.0 + 0.0j
            if i - 2 >= j:
                val += s_prev2 * mat[i - 2, j]
            if i - 1 >= j - 1:
                val += s_prev1 * mat[i - 1, j - 1]
            mat[i, j] = val
    return mat


def smith_par_get_as(nhermite: int, q: int) -> np.ndarray:
    """Compute Padé approximant closure coefficients ``a_coefficients`` in FP64.

    Ported strictly from ``smith_par_getAs`` (``gx/src/smith_par_closure.cu``).
    """
    n = int(nhermite)
    q_int = int(q)
    if q_int <= 0 or q_int > n:
        raise ValueError(f"smith_par_q={q_int} must satisfy 1 <= smith_par_q <= nhermite ({n}).")

    # Taylor coefficients of R_00(w) = (-1/sqrt(2)) Z(w/sqrt(2)) (L251-268)
    power_series = np.zeros((q_int,), dtype=np.complex128)
    two_i = 2.0j
    fac = 1.0 + 0.0j
    root2 = math.sqrt(2.0)
    fac2 = root2 + 0.0j
    for j in range(q_int):
        res = fac / fac2
        power_series[j] = res * math.sqrt(math.pi) * (2.0 ** (-j)) / math.gamma(0.5 * j + 1.0)
        fac *= two_i
        fac2 *= root2

    # Lower-triangular Toeplitz matrix rMatrix (L271-280)
    r_mat = np.zeros((q_int, q_int), dtype=np.complex128)
    for j in range(q_int):
        for i in range(j, q_int):
            r_mat[i, j] = power_series[i - j]

    scaling = 1.0 / math.sqrt(2.0)
    p_coeffs = _smith_par_hermite_coeffs(n, scaling, "P")
    q_coeffs = _smith_par_hermite_coeffs(n, scaling, "Q")

    p_mat = np.zeros((q_int, q_int), dtype=np.complex128)
    q_mat = np.zeros((q_int, q_int), dtype=np.complex128)
    p_n = p_coeffs[n, :q_int].copy()
    q_n = q_coeffs[n, :q_int].copy()

    col = 0
    for d in range(n - 1, n - q_int - 1, -1):
        upper = min(q_int, d + 1)
        p_mat[:upper, col] = p_coeffs[d, :upper]
        q_mat[:upper, col] = q_coeffs[d, :upper]
        col += 1

    lhs = r_mat @ p_mat - 1.0j * q_mat
    rhs = r_mat @ p_n - 1.0j * q_n
    return np.linalg.solve(lhs, rhs).astype(np.complex128)


# ---------------------------------------------------------------------------
# Parallel Derivative & End-Damping Operator (GradParallelPeriodic / Local / nLinks=1)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelGrad:
    """Single-segment parallel FFT derivative & boundary damping operator on ``z`` (``axis=-3``).

    Covers:
    - ``GradParallelLocal`` (``local_limit = True``, ``grad_parallel.cu:L187-223``).
    - ``GradParallelPeriodic`` / ``GradParallelLinked`` with ``nLinks = 1``
      (``boundary_option_periodic = True``, ``nakx == 1``, or ``|jtwist| >= nakx``,
      ``grad_parallel.cu:L6-185`` & ``grad_parallel_linked.cu:L380-514``).
    Multi-link twist-and-shift chains (``nLinks > 1``) and ``nonTwist`` (NTFT) implement the
    identical duck-typed interface in ``gx_jax/boundary_parallel.py``.
    """

    kz: jax.Array  # (nz,) raw parallel wavenumbers
    kz_deriv: jax.Array  # (nz,) kzp when periodic+dealias_kz, else kz
    kzm: jax.Array  # (nz,) int32 2/3 dealiasing mask
    hyperz_weight: jax.Array  # (nz,) (|kz| / kz_max)^p_hyper_z
    end_damp_nu: jax.Array  # (nz,) dampEnds_linked spatial profile nu(z)
    active_mask: jax.Array  # (naky, nakx) bool

    nz: int
    naky: int
    nakx: int
    zp: int
    local_limit: bool
    boundary_option_periodic: bool
    dealias_kz: bool
    damp_ends_widthfrac: float
    damp_ends_amp: float
    kpar_local: float
    mxu_plan: MXU1DPlan | None = None
    fft_backend: str = "fft"

    @property
    def supports_fused_z_fft(self) -> bool:
        """True when 1D ``z`` FFTs are independent per ``(iky, ikx)`` (``nLinks == 1``)."""
        return not self.local_limit and self.nz > 1

    def tree_flatten(self):
        children = (
            self.kz,
            self.kz_deriv,
            self.kzm,
            self.hyperz_weight,
            self.end_damp_nu,
            self.active_mask,
            self.mxu_plan,
        )
        aux_data = (
            self.nz,
            self.naky,
            self.nakx,
            self.zp,
            self.local_limit,
            self.boundary_option_periodic,
            self.dealias_kz,
            self.damp_ends_widthfrac,
            self.damp_ends_amp,
            self.kpar_local,
            self.fft_backend,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> ParallelGrad:
        (
            nz,
            naky,
            nakx,
            zp,
            local_limit,
            boundary_option_periodic,
            dealias_kz,
            damp_ends_widthfrac,
            damp_ends_amp,
            kpar_local,
            fft_backend,
        ) = aux_data
        kz, kz_deriv, kzm, hyperz_weight, end_damp_nu, active_mask, mxu_plan = children
        return cls(
            kz=kz,
            kz_deriv=kz_deriv,
            kzm=kzm,
            hyperz_weight=hyperz_weight,
            end_damp_nu=end_damp_nu,
            active_mask=active_mask,
            nz=nz,
            naky=naky,
            nakx=nakx,
            zp=zp,
            local_limit=local_limit,
            boundary_option_periodic=boundary_option_periodic,
            dealias_kz=dealias_kz,
            damp_ends_widthfrac=damp_ends_widthfrac,
            damp_ends_amp=damp_ends_amp,
            kpar_local=kpar_local,
            mxu_plan=mxu_plan,
            fft_backend=fft_backend,
        )

    def astype(self, dtype: jnp.dtype) -> ParallelGrad:
        """Cast floating-point operator arrays (including MXU plan) to ``dtype``."""
        target = jnp.dtype(dtype)
        r_dtype = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32

        def _cast(x: jax.Array) -> jax.Array:
            if not jnp.issubdtype(x.dtype, jnp.floating):
                return x
            if isinstance(x, jax.core.Tracer):
                return x.astype(r_dtype)
            return jnp.asarray(np.asarray(x), dtype=r_dtype)

        return jax.tree_util.tree_map(_cast, self)

    def _mask_out(self, arr: jax.Array) -> jax.Array:
        return jnp.where(self.active_mask, arr, jnp.zeros_like(arr))

    def fft_z(self, arr: jax.Array, *, norm: str | None = None) -> jax.Array:
        """Forward 1D Fourier transform along ``z`` (``axis=-3``) using ``self.fft_backend``."""
        c_dtype, _ = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.fft_backend == "mxu":
            return mxu_fft(arr_c, n=self.nz, axis=-3, norm=norm, plan=self.mxu_plan).astype(c_dtype)
        return jnp.fft.fft(arr_c, axis=-3, norm=norm).astype(c_dtype)

    def ifft_z(self, arr: jax.Array, *, norm: str | None = None) -> jax.Array:
        """Inverse 1D Fourier transform along ``z`` (``axis=-3``) using ``self.fft_backend``."""
        c_dtype, _ = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.fft_backend == "mxu":
            return mxu_ifft(arr_c, n=self.nz, axis=-3, norm=norm, plan=self.mxu_plan).astype(
                c_dtype
            )
        return jnp.fft.ifft(arr_c, axis=-3, norm=norm).astype(c_dtype)

    def dz(
        self, arr: jax.Array, *, accumulate: bool = False, target: jax.Array | None = None
    ) -> jax.Array:
        """Parallel derivative ``d/dz`` along ``axis=-3`` (``grad_parallel.cu:dz``)."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit:
            kpar_r = jnp.asarray(self.kpar_local, dtype=r_dtype)
            res = jax.lax.complex(jnp.zeros_like(kpar_r), kpar_r).astype(c_dtype) * arr_c
        elif self.nz <= 1:
            res = jnp.zeros_like(arr_c)
        else:
            kz_r = self.kz_deriv.astype(r_dtype)[:, None, None]
            ikz = jax.lax.complex(jnp.zeros_like(kz_r), kz_r).astype(c_dtype)
            res = self.ifft_z(self.fft_z(arr_c) * ikz)
        res = self._mask_out(res)
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def abs_dz(
        self, arr: jax.Array, *, accumulate: bool = False, target: jax.Array | None = None
    ) -> jax.Array:
        """Parallel Landau operator ``|d/dz|`` (multiplication by ``|kz|``) along ``axis=-3``."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit:
            res = jnp.asarray(abs(self.kpar_local), dtype=r_dtype) * arr_c
        elif self.nz <= 1:
            res = jnp.zeros_like(arr_c)
        else:
            akz = jnp.abs(self.kz_deriv.astype(r_dtype))[:, None, None]
            res = self.ifft_z(self.fft_z(arr_c) * akz)
        res = self._mask_out(res)
        if accumulate and target is not None:
            return target.astype(c_dtype) + res
        return res

    def dz2(self, arr: jax.Array) -> jax.Array:
        """Second parallel derivative ``d^2/dz^2`` (``-kz^2``) along ``axis=-3``."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit:
            mk2 = jnp.asarray(-(self.kpar_local**2), dtype=r_dtype)
            res = mk2 * arr_c
        elif self.nz <= 1:
            res = jnp.zeros_like(arr_c)
        else:
            kz_r = self.kz_deriv.astype(r_dtype)
            mkz2 = (-(kz_r * kz_r))[:, None, None]
            res = self.ifft_z(self.fft_z(arr_c) * mkz2)
        return self._mask_out(res)

    def hyperz(
        self,
        arr: jax.Array,
        nu_hyper_z: float,
        *,
        accumulate: bool = False,
        target: jax.Array | None = None,
    ) -> jax.Array:
        """Parallel hyperviscosity ``-nu_hyper_z * (|kz| / kz_max)^p_hyper_z``."""
        if self.local_limit:
            raise NotImplementedError("hyperz is not supported when local_limit=True.")
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.nz <= 1:
            res = jnp.zeros_like(arr_c)
        else:
            w = (-self.hyperz_weight.astype(r_dtype))[:, None, None]
            res = self.ifft_z(self.fft_z(arr_c) * w)
        res = self._mask_out(res)
        if accumulate and target is not None:
            return target.astype(c_dtype) + jnp.asarray(nu_hyper_z, dtype=r_dtype) * res
        return jnp.asarray(nu_hyper_z, dtype=r_dtype) * res

    def zft(self, arr: jax.Array) -> jax.Array:
        """Normalized forward parallel Fourier transform (``grad_parallel.cu:zft``)."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit or self.nz <= 1:
            return self._mask_out(arr_c)
        if self.fft_backend == "mxu":
            return self._mask_out(self.fft_z(arr_c, norm="forward"))
        norm = jnp.asarray(1.0 / float(self.nz), dtype=r_dtype)
        return self._mask_out(jnp.fft.fft(arr_c, axis=-3).astype(c_dtype) * norm)

    def zft_inverse(self, arr: jax.Array) -> jax.Array:
        """Unnormalized inverse parallel Fourier transform inverting :meth:`zft`."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit or self.nz <= 1:
            return self._mask_out(arr_c)
        if self.fft_backend == "mxu":
            return self._mask_out(self.ifft_z(arr_c, norm="forward"))
        scale = jnp.asarray(float(self.nz), dtype=r_dtype)
        return self._mask_out(jnp.fft.ifft(arr_c, axis=-3).astype(c_dtype) * scale)

    def dealias(self, arr: jax.Array) -> jax.Array:
        """Apply periodic 2/3 ``kzm`` mask if ``boundary_option_periodic``, else C++ no-op.

        Matches C++ timestepper ``dealias_kz`` dispatch (``ts_sspx3.cu:L38-46``,
        ``ts_k10.cu:L31-32``): ``GradParallelPeriodic::dealias`` (``grad_parallel.cu:L83-97``)
        applies ``kzm`` when ``boundary_option_periodic=True``, whereas
        ``GradParallelLinked::dealias`` (``grad_parallel_linked.cu:L328-336``) is a no-op.
        """
        c_dtype, _ = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit or self.nz <= 1 or not self.boundary_option_periodic:
            return self._mask_out(arr_c)
        return self.dealias_chain(arr)

    def dealias_chain(self, arr: jax.Array) -> jax.Array:
        """Opt-in helper applying the 2/3 parallel ``kz`` mask ``kzm`` unconditionally."""
        c_dtype, r_dtype = _complex_and_real_dtypes(arr)
        arr_c = arr.astype(c_dtype)
        if self.local_limit or self.nz <= 1:
            return self._mask_out(arr_c)
        mask_z = self.kzm.astype(r_dtype)[:, None, None]
        return self._mask_out(self.ifft_z(self.fft_z(arr_c) * mask_z))

    def apply_bcs(
        self,
        G: jax.Array,
        rhs: jax.Array,
        fields: Fields,
        solver: FieldSolver,
        dt: float | jax.Array,
        *,
        H: jax.Array | None = None,
    ) -> jax.Array:
        """Apply linked flux-tube boundary end-damping on ``H`` (``dampEnds_linked``).

        Ported from ``device_funcs.cu:L2813-2876`` and ``linear.cu:L243``:
        applied only when ``not boundary_option_periodic and not local_limit``, exclusively on
        non-zonal ``iky > 0`` modes.
        """
        if self.boundary_option_periodic or self.local_limit or self.nz <= 1:
            return rhs
        c_dtype, r_dtype = _complex_and_real_dtypes(G)
        if H is None:
            H = build_nonadiabatic_h(G, fields, solver)
        h_c = H.astype(c_dtype)
        amp = jnp.asarray(self.damp_ends_amp, dtype=r_dtype) / jnp.asarray(dt, dtype=r_dtype)
        nu = self.end_damp_nu.astype(r_dtype)[None, None, None, :, None, None]
        ky_pos = (jnp.arange(self.naky) > 0)[None, None, None, None, :, None]
        mask6 = self.active_mask[None, None, None, None, :, :] & ky_pos
        damp_term = jnp.where(mask6, (nu * amp) * h_c, jnp.zeros_like(h_c))
        return rhs.astype(c_dtype) - damp_term


def _has_multilink_chains(cfg: GXConfig, grids: Grids) -> bool:
    """Return True if ``GradParallelLinked`` constructs chains with ``nLinks > 1``."""
    if cfg.local_limit or cfg.boundary_option_periodic or grids.nakx <= 1 or grids.naky <= 1:
        return False
    half = (grids.nakx + 1) // 2  # K + 1 where K = (nx - 1) // 3
    kmax = half - 1
    jshift = int(cfg.jtwist)
    for idy in range(1, grids.naky):
        for idx0 in range(-kmax, kmax + 1):
            idx_l = idx0 + idy * jshift
            idx_r = idx0 - idy * jshift
            if -kmax <= idx_l <= kmax and idx_l != idx0:
                return True
            if -kmax <= idx_r <= kmax and idx_r != idx0:
                return True
    return False


def init_parallel_grad(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry | None = None,
    *,
    dtype: jnp.dtype = jnp.float64,
    fft_backend: str | None = None,
) -> Any:
    """Initialize parallel derivative & boundary condition operator matching ``Linear_GK``.

    Delegates multi-link ``GradParallelLinked`` and ``GradParallelNTFT`` to
    ``gx_jax.boundary_parallel`` when available.
    """
    if not cfg.local_limit and (cfg.non_twist or _has_multilink_chains(cfg, grids)):
        try:
            from gx_jax.boundary_parallel import init_boundary_parallel  # type: ignore[import-not-found]

            return init_boundary_parallel(cfg, grids, geo=geo, dtype=dtype, fft_backend=fft_backend)
        except ImportError as exc:
            mode = "NTFT (nonTwist=True)" if cfg.non_twist else "multi-link twist-and-shift"
            raise NotImplementedError(
                f"{mode} parallel boundary operators require gx_jax.boundary_parallel."
            ) from exc

    target = jnp.dtype(dtype)
    r_dtype = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
    nz = grids.nz
    fb_req = cfg.fft_backend if fft_backend is None else fft_backend
    resolved_fb = resolve_fft_backend(
        fb_req,
        nx=int(grids.nx),
        ny=int(grids.ny),
        nz=max(1, int(nz)),
        max_chain_len=max(1, int(nz)),
    )
    mxu_plan = (
        init_mxu_1d_plan(nz, dtype=jnp.float64)
        if (resolved_fb == "mxu" and not cfg.local_limit and nz > 1)
        else None
    )
    kz = jnp.asarray(grids.kz, dtype=jnp.float64)
    # Linear_GK::Linear_GK (linear.cu:L27-35) constructs GradParallelLinked for both periodic
    # (jtwist = 2*nx, parameters.cu:L1246-1251) and linked runs, calling
    # init_kzLinked(..., dealias=false) (grad_parallel_linked.cu:L151), i.e. raw grids.kz.
    kz_deriv = kz
    kzm = jnp.asarray(grids.kzm, dtype=jnp.int32)

    kz_max = float(grids.kz_max)
    if nz > 1 and kz_max > 0.0:
        hyperz_weight = (jnp.abs(kz) / kz_max) ** int(cfg.p_hyper_z)
    else:
        hyperz_weight = jnp.zeros((nz,), dtype=jnp.float64)

    # Precompute dampEnds_linked profile (device_funcs.cu:L2850-2860) for nLinks = 1
    if not cfg.boundary_option_periodic and not cfg.local_limit and nz > 1:
        width = int(nz * cfg.damp_ends_widthfrac)
        if width > 0:
            idz = jnp.arange(nz, dtype=jnp.float64)
            x_lo = idz / float(width)
            x_hi = (float(nz) - idz) / float(width)
            nu_lo = 1.0 - 2.0 * (x_lo**2) / (1.0 + x_lo**4)
            nu_hi = 1.0 - 2.0 * (x_hi**2) / (1.0 + x_hi**4)
            end_damp_nu = jnp.where(
                idz <= width,
                nu_lo,
                jnp.where(idz >= nz - width, nu_hi, 0.0),
            )
        else:
            end_damp_nu = jnp.zeros((nz,), dtype=jnp.float64)
    else:
        end_damp_nu = jnp.zeros((nz,), dtype=jnp.float64)

    op = ParallelGrad(
        kz=kz,
        kz_deriv=kz_deriv,
        kzm=kzm,
        hyperz_weight=hyperz_weight,
        end_damp_nu=end_damp_nu,
        active_mask=grids.active_mask,
        nz=nz,
        naky=grids.naky,
        nakx=grids.nakx,
        zp=grids.zp,
        local_limit=cfg.local_limit,
        boundary_option_periodic=cfg.boundary_option_periodic,
        dealias_kz=cfg.dealias_kz,
        damp_ends_widthfrac=float(cfg.damp_ends_widthfrac),
        damp_ends_amp=float(cfg.damp_ends_amp),
        kpar_local=1.0 / float(grids.zp),
        mxu_plan=mxu_plan,
        fft_backend=resolved_fb,
    )
    return op if r_dtype == jnp.float64 else op.astype(r_dtype)


# ---------------------------------------------------------------------------
# Precomputed Linear Gyrokinetic Operator Container (Linear_GK)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearOperator:
    """Immutable precomputed linear gyrokinetic operator (``Linear_GK`` in ``linear.cu``)."""

    solver: FieldSolver
    grad_par: Any  # ParallelGrad, LinkedParallelGrad, or NTFTParallelGrad

    # Precomputed species-Laguerre FLR drive arrays on (nspecies, nlaguerre, nz, naky, nakx)
    jl: jax.Array
    jlb: jax.Array
    jla: jax.Array
    jlba: jax.Array

    # Precomputed 4D FLR arrays on (nspecies, nz, naky, nakx)
    b_s: jax.Array
    sqrt_b: jax.Array

    # Geometry & spectral coordinate arrays
    bmag: jax.Array  # (nz,)
    bgrad: jax.Array  # (nz,)
    vol_fac: jax.Array  # (nz,)
    cv_d: jax.Array  # (nz, naky, nakx)
    gb_d: jax.Array  # (nz, naky, nakx)
    omegad: jax.Array  # (nz, naky, nakx)
    kperp2: jax.Array  # (nz, naky, nakx)
    ky: jax.Array  # (naky,)
    kx: jax.Array  # (nakx,)
    active_mask: jax.Array  # (naky, nakx) bool

    # Precomputed Hermite/Laguerre recurrence vectors
    sqrt_m: jax.Array  # (nhermite,)
    sqrt_mp1: jax.Array  # (nhermite,)
    sqrt_mp1_mp2: jax.Array  # (nhermite,)
    sqrt_m_mm1: jax.Array  # (nhermite,)
    two_m_p1: jax.Array  # (nhermite,)
    l_idx: jax.Array  # (nlaguerre,)
    lp1: jax.Array  # (nlaguerre,)
    two_l_p1: jax.Array  # (nlaguerre,)

    # Per-species 1D vectors (nspecies,)
    sp_vt: jax.Array
    sp_zt: jax.Array
    sp_tz: jax.Array
    sp_nz: jax.Array
    sp_fprim: jax.Array
    sp_tprim: jax.Array
    sp_nu_ss: jax.Array
    sp_nuei: jax.Array
    sp_as_i: jax.Array
    sp_vt_i: jax.Array
    sp_nzvt_i: jax.Array
    sp_has_upar_i: jax.Array  # bool (nspecies,)

    # Precomputed dissipation & closure arrays
    hypercoll_kz_fac: jax.Array  # (nspecies, nhermite)
    hypercoll_const_rate: jax.Array  # (nspecies, nhermite, nlaguerre)
    hypercoll_const_lm: jax.Array  # (nspecies, nhermite, nlaguerre)
    hyperdiff_fac: jax.Array  # (naky, nakx)
    hb_d10: jax.Array  # (nakx,)
    hb_d01: jax.Array  # (naky, nakx)
    hb_d11: jax.Array  # (naky, nakx)
    closure_coeff: jax.Array  # complex (q,) or (11,) or (0,)

    # Static compile-time dimensions & physics flags
    nspecies: int
    nhermite: int
    nlaguerre: int
    nz: int
    naky: int
    nakx: int
    gradpar: float
    vtmax: float
    closure_model: str
    smith_par_q: int
    smith_perp_q: int
    collisions: bool
    coll_conservation: bool
    ei_colls: bool
    hyper: bool
    hb_hyper: bool
    d_hb: float
    p_hb: int
    w_osc: float
    hyperz: bool
    nu_hyper_z: float
    hypercollisions_const: bool
    hypercollisions_kz: bool
    nu_hyper_lm: float

    def tree_flatten(self):
        children = (
            self.solver,
            self.grad_par,
            self.jl,
            self.jlb,
            self.jla,
            self.jlba,
            self.b_s,
            self.sqrt_b,
            self.bmag,
            self.bgrad,
            self.vol_fac,
            self.cv_d,
            self.gb_d,
            self.omegad,
            self.kperp2,
            self.ky,
            self.kx,
            self.active_mask,
            self.sqrt_m,
            self.sqrt_mp1,
            self.sqrt_mp1_mp2,
            self.sqrt_m_mm1,
            self.two_m_p1,
            self.l_idx,
            self.lp1,
            self.two_l_p1,
            self.sp_vt,
            self.sp_zt,
            self.sp_tz,
            self.sp_nz,
            self.sp_fprim,
            self.sp_tprim,
            self.sp_nu_ss,
            self.sp_nuei,
            self.sp_as_i,
            self.sp_vt_i,
            self.sp_nzvt_i,
            self.sp_has_upar_i,
            self.hypercoll_kz_fac,
            self.hypercoll_const_rate,
            self.hypercoll_const_lm,
            self.hyperdiff_fac,
            self.hb_d10,
            self.hb_d01,
            self.hb_d11,
            self.closure_coeff,
        )
        aux_data = (
            self.nspecies,
            self.nhermite,
            self.nlaguerre,
            self.nz,
            self.naky,
            self.nakx,
            self.gradpar,
            self.vtmax,
            self.closure_model,
            self.smith_par_q,
            self.smith_perp_q,
            self.collisions,
            self.coll_conservation,
            self.ei_colls,
            self.hyper,
            self.hb_hyper,
            self.d_hb,
            self.p_hb,
            self.w_osc,
            self.hyperz,
            self.nu_hyper_z,
            self.hypercollisions_const,
            self.hypercollisions_kz,
            self.nu_hyper_lm,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> LinearOperator:
        (
            nspecies,
            nhermite,
            nlaguerre,
            nz,
            naky,
            nakx,
            gradpar,
            vtmax,
            closure_model,
            smith_par_q,
            smith_perp_q,
            collisions,
            coll_conservation,
            ei_colls,
            hyper,
            hb_hyper,
            d_hb,
            p_hb,
            w_osc,
            hyperz,
            nu_hyper_z,
            hypercollisions_const,
            hypercollisions_kz,
            nu_hyper_lm,
        ) = aux_data
        (
            solver,
            grad_par,
            jl,
            jlb,
            jla,
            jlba,
            b_s,
            sqrt_b,
            bmag,
            bgrad,
            vol_fac,
            cv_d,
            gb_d,
            omegad,
            kperp2,
            ky,
            kx,
            active_mask,
            sqrt_m,
            sqrt_mp1,
            sqrt_mp1_mp2,
            sqrt_m_mm1,
            two_m_p1,
            l_idx,
            lp1,
            two_l_p1,
            sp_vt,
            sp_zt,
            sp_tz,
            sp_nz,
            sp_fprim,
            sp_tprim,
            sp_nu_ss,
            sp_nuei,
            sp_as_i,
            sp_vt_i,
            sp_nzvt_i,
            sp_has_upar_i,
            hypercoll_kz_fac,
            hypercoll_const_rate,
            hypercoll_const_lm,
            hyperdiff_fac,
            hb_d10,
            hb_d01,
            hb_d11,
            closure_coeff,
        ) = children
        return cls(
            solver=solver,
            grad_par=grad_par,
            jl=jl,
            jlb=jlb,
            jla=jla,
            jlba=jlba,
            b_s=b_s,
            sqrt_b=sqrt_b,
            bmag=bmag,
            bgrad=bgrad,
            vol_fac=vol_fac,
            cv_d=cv_d,
            gb_d=gb_d,
            omegad=omegad,
            kperp2=kperp2,
            ky=ky,
            kx=kx,
            active_mask=active_mask,
            sqrt_m=sqrt_m,
            sqrt_mp1=sqrt_mp1,
            sqrt_mp1_mp2=sqrt_mp1_mp2,
            sqrt_m_mm1=sqrt_m_mm1,
            two_m_p1=two_m_p1,
            l_idx=l_idx,
            lp1=lp1,
            two_l_p1=two_l_p1,
            sp_vt=sp_vt,
            sp_zt=sp_zt,
            sp_tz=sp_tz,
            sp_nz=sp_nz,
            sp_fprim=sp_fprim,
            sp_tprim=sp_tprim,
            sp_nu_ss=sp_nu_ss,
            sp_nuei=sp_nuei,
            sp_as_i=sp_as_i,
            sp_vt_i=sp_vt_i,
            sp_nzvt_i=sp_nzvt_i,
            sp_has_upar_i=sp_has_upar_i,
            hypercoll_kz_fac=hypercoll_kz_fac,
            hypercoll_const_rate=hypercoll_const_rate,
            hypercoll_const_lm=hypercoll_const_lm,
            hyperdiff_fac=hyperdiff_fac,
            hb_d10=hb_d10,
            hb_d01=hb_d01,
            hb_d11=hb_d11,
            closure_coeff=closure_coeff,
            nspecies=nspecies,
            nhermite=nhermite,
            nlaguerre=nlaguerre,
            nz=nz,
            naky=naky,
            nakx=nakx,
            gradpar=gradpar,
            vtmax=vtmax,
            closure_model=closure_model,
            smith_par_q=smith_par_q,
            smith_perp_q=smith_perp_q,
            collisions=collisions,
            coll_conservation=coll_conservation,
            ei_colls=ei_colls,
            hyper=hyper,
            hb_hyper=hb_hyper,
            d_hb=d_hb,
            p_hb=p_hb,
            w_osc=w_osc,
            hyperz=hyperz,
            nu_hyper_z=nu_hyper_z,
            hypercollisions_const=hypercollisions_const,
            hypercollisions_kz=hypercollisions_kz,
            nu_hyper_lm=nu_hyper_lm,
        )

    @property
    def fft_backend(self) -> str:
        """Resolved parallel spectral backend (``'fft'`` or ``'mxu'``)."""
        return str(getattr(self.grad_par, "fft_backend", "fft"))

    def astype(self, dtype: jnp.dtype) -> LinearOperator:
        """Cast floating/complex arrays on ``LinearOperator`` to runtime ``dtype``."""
        target = jnp.dtype(dtype)
        r_dtype = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
        c_dtype = jnp.complex128 if r_dtype == jnp.float64 else jnp.complex64

        def _cast(leaf):
            if hasattr(leaf, "astype") and isinstance(leaf, (FieldSolver, ParallelGrad)):
                return leaf.astype(r_dtype)
            if isinstance(leaf, jax.Array):
                if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                    return (
                        leaf.astype(c_dtype)
                        if isinstance(leaf, jax.core.Tracer)
                        else jnp.asarray(np.asarray(leaf), dtype=c_dtype)
                    )
                if jnp.issubdtype(leaf.dtype, jnp.floating):
                    return (
                        leaf.astype(r_dtype)
                        if isinstance(leaf, jax.core.Tracer)
                        else jnp.asarray(np.asarray(leaf), dtype=r_dtype)
                    )
            return leaf

        return LinearOperator(
            solver=self.solver.astype(r_dtype),
            grad_par=self.grad_par.astype(r_dtype)
            if hasattr(self.grad_par, "astype")
            else self.grad_par,
            jl=_cast(self.jl),
            jlb=_cast(self.jlb),
            jla=_cast(self.jla),
            jlba=_cast(self.jlba),
            b_s=_cast(self.b_s),
            sqrt_b=_cast(self.sqrt_b),
            bmag=_cast(self.bmag),
            bgrad=_cast(self.bgrad),
            vol_fac=_cast(self.vol_fac),
            cv_d=_cast(self.cv_d),
            gb_d=_cast(self.gb_d),
            omegad=_cast(self.omegad),
            kperp2=_cast(self.kperp2),
            ky=_cast(self.ky),
            kx=_cast(self.kx),
            active_mask=self.active_mask,
            sqrt_m=_cast(self.sqrt_m),
            sqrt_mp1=_cast(self.sqrt_mp1),
            sqrt_mp1_mp2=_cast(self.sqrt_mp1_mp2),
            sqrt_m_mm1=_cast(self.sqrt_m_mm1),
            two_m_p1=_cast(self.two_m_p1),
            l_idx=_cast(self.l_idx),
            lp1=_cast(self.lp1),
            two_l_p1=_cast(self.two_l_p1),
            sp_vt=_cast(self.sp_vt),
            sp_zt=_cast(self.sp_zt),
            sp_tz=_cast(self.sp_tz),
            sp_nz=_cast(self.sp_nz),
            sp_fprim=_cast(self.sp_fprim),
            sp_tprim=_cast(self.sp_tprim),
            sp_nu_ss=_cast(self.sp_nu_ss),
            sp_nuei=_cast(self.sp_nuei),
            sp_as_i=_cast(self.sp_as_i),
            sp_vt_i=_cast(self.sp_vt_i),
            sp_nzvt_i=_cast(self.sp_nzvt_i),
            sp_has_upar_i=self.sp_has_upar_i,
            hypercoll_kz_fac=_cast(self.hypercoll_kz_fac),
            hypercoll_const_rate=_cast(self.hypercoll_const_rate),
            hypercoll_const_lm=_cast(self.hypercoll_const_lm),
            hyperdiff_fac=_cast(self.hyperdiff_fac),
            hb_d10=_cast(self.hb_d10),
            hb_d01=_cast(self.hb_d01),
            hb_d11=_cast(self.hb_d11),
            closure_coeff=_cast(self.closure_coeff),
            nspecies=self.nspecies,
            nhermite=self.nhermite,
            nlaguerre=self.nlaguerre,
            nz=self.nz,
            naky=self.naky,
            nakx=self.nakx,
            gradpar=self.gradpar,
            vtmax=self.vtmax,
            closure_model=self.closure_model,
            smith_par_q=self.smith_par_q,
            smith_perp_q=self.smith_perp_q,
            collisions=self.collisions,
            coll_conservation=self.coll_conservation,
            ei_colls=self.ei_colls,
            hyper=self.hyper,
            hb_hyper=self.hb_hyper,
            d_hb=self.d_hb,
            p_hb=self.p_hb,
            w_osc=self.w_osc,
            hyperz=self.hyperz,
            nu_hyper_z=self.nu_hyper_z,
            hypercollisions_const=self.hypercollisions_const,
            hypercollisions_kz=self.hypercollisions_kz,
            nu_hyper_lm=self.nu_hyper_lm,
        )


def init_linear_operator(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    solver: FieldSolver | None = None,
    grad_par: Any | None = None,
    *,
    dtype: jnp.dtype = jnp.float64,
    fft_backend: str | None = None,
) -> LinearOperator:
    """Precompute static linear gyrokinetic operator arrays in IEEE ``float64``."""
    ns = grids.nspecies
    if len(cfg.species) != ns:
        raise ValueError(f"Expected {ns} species in cfg.species, got {len(cfg.species)}.")
    nm = grids.nhermite
    nl = grids.nlaguerre
    nz = grids.nz
    naky = grids.naky
    nakx = grids.nakx

    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64) if solver is None else solver
    grad_par64 = (
        init_parallel_grad(cfg, grids, geo, dtype=jnp.float64, fft_backend=fft_backend)
        if grad_par is None
        else grad_par
    )

    kperp2 = jnp.asarray(geo.kperp2, dtype=jnp.float64)
    sp_rho2 = jnp.asarray([sp.rho2 for sp in cfg.species], dtype=jnp.float64)
    b_s = sp_rho2[:, None, None, None] * kperp2[None, :, :, :]
    sqrt_b = jnp.sqrt(jnp.maximum(b_s, 0.0))

    jl = jnp.asarray(solver64.jl, dtype=jnp.float64)
    jlb = jnp.asarray(solver64.jlb, dtype=jnp.float64)
    jla = jnp.asarray(solver64.jla, dtype=jnp.float64)
    ell_vec = jnp.arange(nl, dtype=jnp.int32)[None, :, None, None, None]
    ell_f = ell_vec.astype(jnp.float64)
    b_s5 = b_s[:, None, :, :, :]
    jlba = (
        ell_f * jflr_b(ell_vec - 1, b_s5)
        + 2.0 * ell_f * jflr_b(ell_vec, b_s5)
        + (ell_f + 1.0) * jflr_b(ell_vec + 1, b_s5)
    )

    m_f = jnp.arange(nm, dtype=jnp.float64)
    sqrt_m = jnp.sqrt(m_f)
    sqrt_mp1 = jnp.sqrt(m_f + 1.0)
    sqrt_mp1_mp2 = jnp.sqrt((m_f + 1.0) * (m_f + 2.0))
    sqrt_m_mm1 = jnp.sqrt(m_f * jnp.maximum(m_f - 1.0, 0.0))
    two_m_p1 = 2.0 * m_f + 1.0

    l_f = jnp.arange(nl, dtype=jnp.float64)
    lp1 = l_f + 1.0
    two_l_p1 = 2.0 * l_f + 1.0

    sp_vt = jnp.asarray([sp.vt for sp in cfg.species], dtype=jnp.float64)
    sp_zt = jnp.asarray([sp.zt for sp in cfg.species], dtype=jnp.float64)
    sp_tz = jnp.asarray([sp.tz for sp in cfg.species], dtype=jnp.float64)
    sp_nz = jnp.asarray([sp.nz for sp in cfg.species], dtype=jnp.float64)
    sp_fprim = jnp.asarray([sp.fprim for sp in cfg.species], dtype=jnp.float64)
    sp_tprim = jnp.asarray([sp.tprim for sp in cfg.species], dtype=jnp.float64)
    sp_nu_ss = jnp.asarray([sp.nu_ss for sp in cfg.species], dtype=jnp.float64)

    # Electron-ion collision parameters (device_funcs.cu:L3118-3127, L3198)
    sp0 = cfg.species[0]
    nuei_list: list[float] = []
    as_i_list: list[float] = []
    vt_i_list: list[float] = []
    nzvt_i_list: list[float] = []
    has_upar_i_list: list[bool] = []
    for sp in cfg.species:
        if sp.type_id == 1 and cfg.ei_colls:
            nuei = (sp0.z * sp.nu_ss) if ns > 1 else sp.nu_ss
            as_i = sp0.jparfac
            vt_i = sp0.vt
            nzvt_i = sp0.nz * sp0.vt
            has_ui = (ns > 1) and (as_i > 0.0)
            nuei_list.append(float(nuei))
            as_i_list.append(float(as_i if as_i > 0.0 else 1.0))
            vt_i_list.append(float(vt_i))
            nzvt_i_list.append(float(nzvt_i if nzvt_i != 0.0 else 1.0))
            has_upar_i_list.append(bool(has_ui))
        else:
            nuei_list.append(0.0)
            as_i_list.append(1.0)
            vt_i_list.append(1.0)
            nzvt_i_list.append(1.0)
            has_upar_i_list.append(False)

    sp_nuei = jnp.asarray(nuei_list, dtype=jnp.float64)
    sp_as_i = jnp.asarray(as_i_list, dtype=jnp.float64)
    sp_vt_i = jnp.asarray(vt_i_list, dtype=jnp.float64)
    sp_nzvt_i = jnp.asarray(nzvt_i_list, dtype=jnp.float64)
    sp_has_upar_i = jnp.asarray(has_upar_i_list, dtype=jnp.bool_)

    # |kz|-weighted Hermite hypercollisions (linear.cu:L225-233, device_funcs.cu:L3367-3384):
    # nu_hyp_m * m^p = nu_hyper_m * (p + 0.5) * 2.3 * vt * |gradpar| * (m / M)^p / sqrt(M)
    if nm > 1:
        m_max_f = float(nm - 1)
        p_m = float(cfg.p_hyper_m)
        m_prof_kz = jnp.where(m_f > 2.0, ((m_f / m_max_f) ** p_m) / math.sqrt(m_max_f), 0.0)
        hypercoll_kz_fac = (
            float(cfg.nu_hyper_m)
            * (p_m + 0.5)
            * 2.3
            * abs(float(geo.gradpar))
            * sp_vt[:, None]
            * m_prof_kz[None, :]
        )
    else:
        hypercoll_kz_fac = jnp.zeros((ns, nm), dtype=jnp.float64)

    # Constant-coefficient hypercollisions (linear.cu:L215-222, device_funcs.cu:L3329-3365)
    m_grid = m_f[:, None]  # (nm, 1)
    l_grid = l_f[None, :]  # (1, nl)
    active_hl = (m_grid > 2.0) | (l_grid > 1.0)
    term_l = float(nl) * float(cfg.nu_hyper_l) * ((l_grid / float(nl)) ** float(cfg.p_hyper_l))
    term_m = float(nm) * float(cfg.nu_hyper_m) * ((m_grid / float(nm)) ** float(cfg.p_hyper_m))
    base_hl = jnp.where(active_hl, term_l + term_m, 0.0)
    hypercoll_const_rate = sp_vt[:, None, None] * base_hl[None, :, :]

    lm_ratio = (2.0 * l_grid + m_grid) / float(2 * nl + nm)
    base_lm = jnp.where(active_hl, lm_ratio ** int(cfg.p_hyper_lm), 0.0)
    hypercoll_const_lm = (
        (sp_vt / float(cfg.vtmax))[:, None, None] * float(cfg.nu_hyper_lm) * base_lm[None, :, :]
    )

    # Perpendicular k-space hyperdiffusion (device_funcs.cu:L3304-3327)
    kx64 = jnp.asarray(grids.kx, dtype=jnp.float64)
    ky64 = jnp.asarray(grids.ky, dtype=jnp.float64)
    kpmax2 = float(grids.kx_max) ** 2 + float(grids.ky_max) ** 2
    kp2_grid = kx64[None, :] ** 2 + ky64[:, None] ** 2
    if kpmax2 > 0.0:
        hyperdiff_fac = float(cfg.d_hyper) * ((kp2_grid / kpmax2) ** float(cfg.p_hyper))
    else:
        hyperdiff_fac = jnp.zeros((naky, nakx), dtype=jnp.float64)

    # Hammett-Belli adaptive hyperdiffusivity weights (device_funcs.cu:L3445-3453)
    if float(grids.kx_max) > 0.0:
        kx_norm4 = (kx64 / float(grids.kx_max)) ** 4
    else:
        kx_norm4 = jnp.zeros((nakx,), dtype=jnp.float64)
    ky_norm1 = (
        (ky64 / float(grids.ky_max))
        if float(grids.ky_max) > 0.0
        else jnp.zeros((naky,), dtype=jnp.float64)
    )
    hb_d10 = float(cfg.d_hb) * kx_norm4
    hb_d01 = float(cfg.d_hb) * ky_norm1[:, None] * kx_norm4[None, :]
    hb_d11 = (
        float(cfg.d_hb) * ((kp2_grid / kpmax2) ** int(cfg.p_hb))
        if kpmax2 > 0.0
        else jnp.zeros((naky, nakx), dtype=jnp.float64)
    )

    # Closures (linear.cu:L38-54, closures.cu, smith_par_closure.cu)
    cmodel = cfg.closure_model.strip().lower()
    r_target = jnp.float64 if jnp.dtype(dtype) in (jnp.float64, jnp.complex128) else jnp.float32
    c_target = jnp.complex128 if r_target == jnp.float64 else jnp.complex64
    np_c_target = np.complex128 if r_target == jnp.float64 else np.complex64
    if cmodel == "none":
        closure_coeff = jnp.zeros((0,), dtype=c_target)
    elif cmodel == "beer4+2":
        if nm != 4 or nl != 2:
            raise ValueError(
                f"Beer 4+2 closure requires (nhermite=4, nlaguerre=2), got {(nm, nl)}."
            )
        closure_coeff = jnp.asarray(np.asarray(_BEER42_NU_TUPLE, dtype=np_c_target), dtype=c_target)
    elif cmodel == "smith_perp":
        closure_coeff = jnp.asarray(
            np.asarray(smith_perp_get_as(nl, cfg.smith_perp_q), dtype=np_c_target), dtype=c_target
        )
    elif cmodel == "smith_par":
        closure_coeff = jnp.asarray(
            np.asarray(smith_par_get_as(nm, cfg.smith_par_q), dtype=np_c_target), dtype=c_target
        )
    else:
        raise ValueError(
            f"Unsupported closure_model={cfg.closure_model!r}. "
            "Expected one of {'none', 'beer4+2', 'smith_perp', 'smith_par'}."
        )

    op64 = LinearOperator(
        solver=solver64,
        grad_par=grad_par64,
        jl=jl,
        jlb=jlb,
        jla=jla,
        jlba=jlba,
        b_s=b_s,
        sqrt_b=sqrt_b,
        bmag=jnp.asarray(geo.bmag, dtype=jnp.float64),
        bgrad=jnp.asarray(geo.bgrad, dtype=jnp.float64),
        vol_fac=jnp.asarray(geo.vol_fac, dtype=jnp.float64),
        cv_d=jnp.asarray(geo.cv_d, dtype=jnp.float64),
        gb_d=jnp.asarray(geo.gb_d, dtype=jnp.float64),
        omegad=jnp.asarray(geo.omegad, dtype=jnp.float64),
        kperp2=kperp2,
        ky=ky64,
        kx=kx64,
        active_mask=grids.active_mask,
        sqrt_m=sqrt_m,
        sqrt_mp1=sqrt_mp1,
        sqrt_mp1_mp2=sqrt_mp1_mp2,
        sqrt_m_mm1=sqrt_m_mm1,
        two_m_p1=two_m_p1,
        l_idx=l_f,
        lp1=lp1,
        two_l_p1=two_l_p1,
        sp_vt=sp_vt,
        sp_zt=sp_zt,
        sp_tz=sp_tz,
        sp_nz=sp_nz,
        sp_fprim=sp_fprim,
        sp_tprim=sp_tprim,
        sp_nu_ss=sp_nu_ss,
        sp_nuei=sp_nuei,
        sp_as_i=sp_as_i,
        sp_vt_i=sp_vt_i,
        sp_nzvt_i=sp_nzvt_i,
        sp_has_upar_i=sp_has_upar_i,
        hypercoll_kz_fac=hypercoll_kz_fac,
        hypercoll_const_rate=hypercoll_const_rate,
        hypercoll_const_lm=hypercoll_const_lm,
        hyperdiff_fac=hyperdiff_fac,
        hb_d10=hb_d10,
        hb_d01=hb_d01,
        hb_d11=hb_d11,
        closure_coeff=closure_coeff,
        nspecies=ns,
        nhermite=nm,
        nlaguerre=nl,
        nz=nz,
        naky=naky,
        nakx=nakx,
        gradpar=float(geo.gradpar),
        vtmax=float(cfg.vtmax),
        closure_model=cmodel,
        smith_par_q=int(cfg.smith_par_q),
        smith_perp_q=int(cfg.smith_perp_q),
        collisions=bool(cfg.collisions),
        coll_conservation=bool(cfg.coll_conservation),
        ei_colls=bool(cfg.ei_colls),
        hyper=bool(cfg.hyper),
        hb_hyper=bool(cfg.hb_hyper),
        d_hb=float(cfg.d_hb),
        p_hb=int(cfg.p_hb),
        w_osc=float(cfg.w_osc),
        hyperz=bool(cfg.hyperz),
        nu_hyper_z=float(cfg.nu_hyper_z),
        hypercollisions_const=bool(cfg.hypercollisions_const),
        hypercollisions_kz=bool(cfg.hypercollisions_kz),
        nu_hyper_lm=float(cfg.nu_hyper_lm),
    )
    target = jnp.dtype(dtype)
    r_target = jnp.float64 if target in (jnp.float64, jnp.complex128) else jnp.float32
    return op64 if r_target == jnp.float64 else op64.astype(r_target)


# ---------------------------------------------------------------------------
# Modular Gyrokinetic Kernels (streaming, local RHS, closures, dissipation)
# ---------------------------------------------------------------------------


def build_nonadiabatic_h(G: jax.Array, fields: Fields, solver: FieldSolver) -> jax.Array:
    """Construct non-adiabatic guiding-center distribution ``H`` from ``G`` and ``fields``.

    Ported from ``device_funcs.cu:L3152-3157``:
    - ``H[:, 0] = G[:, 0] + zt * J_l(b_s) * phi + J_{lB}(b_s) * bpar``
    - ``H[:, 1] = G[:, 1] - zt * vt * J_l(b_s) * apar``
    - ``H[:, m] = G[:, m]`` for ``m >= 2``
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    nm = g_c.shape[1]
    jl = solver.jl.astype(r_dtype)
    jlb = solver.jlb.astype(r_dtype)
    zt = solver.sp_zt.astype(r_dtype)[:, None, None, None, None]
    vt = solver.sp_vt.astype(r_dtype)[:, None, None, None, None]

    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]

    h = g_c.at[:, 0].add(zt * jl * phi + jlb * bpar)
    if nm >= 2:
        h = h.at[:, 1].add(-(zt * vt) * jl * apar)
    return h


def streaming_rhs(G: jax.Array, fields: Fields, op: LinearOperator) -> jax.Array:
    """Compute pre-derivative parallel streaming flux ``rhs_par`` (``streaming_rhs``).

    Ported from ``device_funcs.cu:L2999-3046``. Applying ``op.grad_par.dz(...)`` to the
    returned array yields the parallel free-streaming contribution to ``GRhs``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    nm = op.nhermite

    vt5 = op.sp_vt.astype(r_dtype)[:, None, None, None, None]
    zt5 = op.sp_zt.astype(r_dtype)[:, None, None, None, None]
    gpar = jnp.asarray(op.gradpar, dtype=r_dtype)
    jl = op.jl.astype(r_dtype)
    jlb = op.jlb.astype(r_dtype)

    sqrt_m = op.sqrt_m.astype(r_dtype)[None, :, None, None, None, None]
    sqrt_mp1 = op.sqrt_mp1.astype(r_dtype)[None, :, None, None, None, None]

    g_pad = jnp.pad(g_c, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
    rhs_par = -(vt5[:, :, None, :, :, :] * gpar) * (
        sqrt_mp1 * g_pad[:, 2:] + sqrt_m * g_pad[:, :-2]
    )

    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]

    apar_m0 = jl * apar * (zt5 * vt5 * vt5 * gpar)
    rhs_par = rhs_par.at[:, 0].add(apar_m0)
    if nm >= 2:
        m1_term = (-jl * phi * (zt5 * vt5) - jlb * bpar * vt5) * gpar
        rhs_par = rhs_par.at[:, 1].add(m1_term)
    if nm >= 3:
        rhs_par = rhs_par.at[:, 2].add(jnp.asarray(math.sqrt(2.0), dtype=r_dtype) * apar_m0)

    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, rhs_par, jnp.zeros_like(rhs_par))


def rhs_linear_local(
    G: jax.Array,
    fields: Fields,
    op: LinearOperator,
    *,
    H: jax.Array | None = None,
    upar_bar: jax.Array | None = None,
    uperp_bar: jax.Array | None = None,
    t_bar: jax.Array | None = None,
) -> jax.Array:
    """Compute local linear RHS (mirror, curvature/grad-B drifts, omega_* drive, collisions).

    Ported strictly from ``rhs_linear`` (``device_funcs.cu:L3050-3223``), including the
    ghost-padded ``(nm + 4) x (nl + 2)`` stencil, Dougherty/Lenard-Bernstein collision
    conservation terms, electron-ion parallel momentum exchange via Ampère's law, and the
    C++ stray-semicolon no-op on ``L3206-3207``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    nm = op.nhermite
    nl = op.nlaguerre

    h_c = build_nonadiabatic_h(g_c, fields, op.solver) if H is None else H.astype(c_dtype)

    # Single ghost-padded buffer matching shared memory tile (nm + 4) x (nl + 2) (L3081-3159)
    hp = jnp.pad(h_c, ((0, 0), (2, 2), (1, 1), (0, 0), (0, 0), (0, 0)))
    h_00 = hp[:, 2 : 2 + nm, 1 : 1 + nl]
    h_mp1 = hp[:, 3 : 3 + nm, 1 : 1 + nl]
    h_mm1 = hp[:, 1 : 1 + nm, 1 : 1 + nl]
    h_mp2 = hp[:, 4 : 4 + nm, 1 : 1 + nl]
    h_mm2 = hp[:, 0:nm, 1 : 1 + nl]
    h_lp1 = hp[:, 2 : 2 + nm, 2 : 2 + nl]
    h_lm1 = hp[:, 2 : 2 + nm, 0:nl]
    h_mp1_lm1 = hp[:, 3 : 3 + nm, 0:nl]
    h_mm1_lp1 = hp[:, 1 : 1 + nm, 2 : 2 + nl]

    vt6 = op.sp_vt.astype(r_dtype)[:, None, None, None, None, None]
    tz6 = op.sp_tz.astype(r_dtype)[:, None, None, None, None, None]
    bgrad6 = op.bgrad.astype(r_dtype)[None, None, None, :, None, None]
    cv_r = (tz6 * op.cv_d.astype(r_dtype)[None, None, None, :, :, :]).astype(r_dtype)
    gb_r = (tz6 * op.gb_d.astype(r_dtype)[None, None, None, :, :, :]).astype(r_dtype)
    icv_d = jax.lax.complex(jnp.zeros_like(cv_r), cv_r).astype(c_dtype)
    igb_d = jax.lax.complex(jnp.zeros_like(gb_r), gb_r).astype(c_dtype)

    sqrt_m = op.sqrt_m.astype(r_dtype)[None, :, None, None, None, None]
    sqrt_mp1 = op.sqrt_mp1.astype(r_dtype)[None, :, None, None, None, None]
    sqrt_mp1_mp2 = op.sqrt_mp1_mp2.astype(r_dtype)[None, :, None, None, None, None]
    sqrt_m_mm1 = op.sqrt_m_mm1.astype(r_dtype)[None, :, None, None, None, None]
    two_m_p1 = op.two_m_p1.astype(r_dtype)[None, :, None, None, None, None]

    l_idx = op.l_idx.astype(r_dtype)[None, None, :, None, None, None]
    lp1 = op.lp1.astype(r_dtype)[None, None, :, None, None, None]
    two_l_p1 = op.two_l_p1.astype(r_dtype)[None, None, :, None, None, None]

    # Mirror + curvature drift + grad-B drift (L3174-3177)
    out = (
        -vt6
        * bgrad6
        * (
            -sqrt_mp1 * (lp1 * h_mp1 + l_idx * h_mp1_lm1)
            + sqrt_m * (l_idx * h_mm1 + lp1 * h_mm1_lp1)
        )
        - icv_d * (sqrt_mp1_mp2 * h_mp2 + two_m_p1 * h_00 + sqrt_m_mm1 * h_mm2)
        - igb_d * (lp1 * h_lp1 + two_l_p1 * h_00 + l_idx * h_lm1)
    )

    # Collisional damping & conservation terms (L3178, L3193-3194, L3198, L3206, L3213)
    if op.collisions:
        nu_s = op.sp_nu_ss.astype(r_dtype)
        nuei_s = op.sp_nuei.astype(r_dtype)
        nu_tot = (nu_s + nuei_s)[:, None, None, None, None, None]
        b_s6 = op.b_s.astype(r_dtype)[:, None, None, :, :, :]
        m_idx6 = jnp.arange(nm, dtype=r_dtype)[None, :, None, None, None, None]
        out = out - nu_tot * (b_s6 + 2.0 * l_idx + m_idx6) * h_00

        jl_c = op.jl.astype(r_dtype)
        nu5 = nu_s[:, None, None, None, None]
        nuei5 = nuei_s[:, None, None, None, None]
        if op.coll_conservation:
            if upar_bar is None or uperp_bar is None or t_bar is None:
                upar_bar, uperp_bar, t_bar = compute_conservation_moments(g_c, fields, op.solver)
            upar_c = upar_bar.astype(c_dtype)[:, None, :, :, :]
            uperp_c = uperp_bar.astype(c_dtype)[:, None, :, :, :]
            t_c = t_bar.astype(c_dtype)[:, None, :, :, :]

            jlb_c = op.jlb.astype(r_dtype)
            jla_c = op.jla.astype(r_dtype)
            sqrt_b5 = op.sqrt_b.astype(r_dtype)[:, None, :, :, :]
            nu_tot5 = (nu_s + nuei_s)[:, None, None, None, None]

            out = out.at[:, 0].add(nu5 * sqrt_b5 * jlb_c * uperp_c + 2.0 * nu_tot5 * jla_c * t_c)
            if nm >= 3:
                out = out.at[:, 2].add(
                    nu_tot5 * jnp.asarray(math.sqrt(2.0), dtype=r_dtype) * jl_c * t_c
                )
        else:
            upar_c = jnp.zeros((op.nspecies, 1, op.nz, op.naky, op.nakx), dtype=c_dtype)

        # In C++ rhs_linear (device_funcs.cu:L3198, L3206), upar_bar_i includes Ampere's-law
        # kperp2 * bmag^2 * apar / as_i unconditionally even when coll_conservation=False
        # (where upar_bar is zero-initialized by Linear_GK constructor, linear.cu:L23-29).
        if nm >= 2 and (op.coll_conservation or op.ei_colls):
            apar5 = fields.apar.astype(c_dtype)[None, None, :, :, :]
            kp2_b2 = (op.kperp2.astype(r_dtype) * (op.bmag.astype(r_dtype)[:, None, None] ** 2))[
                None, None, :, :, :
            ]
            as_i5 = op.sp_as_i.astype(r_dtype)[:, None, None, None, None]
            nz_vt5 = (op.sp_nz.astype(r_dtype) * op.sp_vt.astype(r_dtype))[
                :, None, None, None, None
            ]
            nzvt_i5 = op.sp_nzvt_i.astype(r_dtype)[:, None, None, None, None]
            upar_i_raw = (kp2_b2 * apar5) / as_i5 - (nz_vt5 / nzvt_i5) * upar_c
            upar_i = jnp.where(
                op.sp_has_upar_i[:, None, None, None, None],
                upar_i_raw,
                jnp.zeros_like(upar_i_raw),
            )
            vti_ov_vt = (op.sp_vt_i.astype(r_dtype) / op.sp_vt.astype(r_dtype))[
                :, None, None, None, None
            ]
            out = out.at[:, 1].add(jl_c * (nu5 * upar_c + nuei5 * vti_ov_vt * upar_i))

    # Background density/temperature gradient drives (omega_*) (L3181-3219)
    jl = op.jl.astype(r_dtype)
    jlb = op.jlb.astype(r_dtype)
    jla = op.jla.astype(r_dtype)
    jlba = op.jlba.astype(r_dtype)
    vt5 = op.sp_vt.astype(r_dtype)[:, None, None, None, None]
    tz5 = op.sp_tz.astype(r_dtype)[:, None, None, None, None]  # 1 / zt
    fprim5 = op.sp_fprim.astype(r_dtype)[:, None, None, None, None]
    tprim5 = op.sp_tprim.astype(r_dtype)[:, None, None, None, None]
    kyr = op.ky.astype(r_dtype)[None, None, None, :, None]
    iky = jax.lax.complex(jnp.zeros_like(kyr), kyr).astype(c_dtype)

    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]

    out = out.at[:, 0].add(
        iky * phi * (fprim5 * jl + tprim5 * jla)
        + (iky * tz5) * bpar * (fprim5 * jlb + tprim5 * jlba)
    )
    if nm >= 2:
        out = out.at[:, 1].add(-vt5 * iky * apar * ((fprim5 + tprim5) * jl + tprim5 * jla))
    if nm >= 3:
        inv_sqrt2 = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        out = out.at[:, 2].add(
            iky * phi * jl * (tprim5 * inv_sqrt2) + (iky * tz5) * bpar * jlb * (tprim5 * inv_sqrt2)
        )
    if nm >= 4:
        sqrt_3_2 = jnp.asarray(math.sqrt(1.5), dtype=r_dtype)
        out = out.at[:, 3].add(-vt5 * iky * apar * (sqrt_3_2 * tprim5 * jl))

    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, out, jnp.zeros_like(out))


def apply_closures(
    G: jax.Array,
    rhs: jax.Array,
    op: LinearOperator,
    grad_par: Any | None = None,
) -> jax.Array:
    """Apply Beer 4+2, Smith perpendicular, or Smith parallel fluid-kinetic closures.

    Ported from ``closures.cu`` and ``device_funcs.cu:L1306-1363``.
    Note ``MomentsG::G(l, m)`` in C++ indexes ``(l, m)`` while Option A ``G`` indexes ``(m, l)``.
    """
    if op.closure_model == "none":
        return rhs

    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    out = rhs.astype(c_dtype)
    gp = op.grad_par if grad_par is None else grad_par
    mask5 = op.active_mask[None, None, None, :, :]
    mask6 = op.active_mask[None, None, None, None, :, :]

    if op.closure_model == "beer4+2":
        vt4 = op.sp_vt.astype(r_dtype)[:, None, None, None]
        tz4 = op.sp_tz.astype(r_dtype)[:, None, None, None]
        gpar = jnp.asarray(op.gradpar, dtype=r_dtype)

        d_par = 2.0 * math.sqrt(math.pi) / (3.0 * math.pi - 8.0)
        beta_par = (32.0 - 9.0 * math.pi) / (3.0 * math.pi - 8.0)
        d_perp = 0.5 * math.sqrt(math.pi)

        # Mask unevolved moments G(l=1, m=2) -> G[:, 2, 1] and G(l=1, m=3) -> G[:, 3, 1]
        z4 = jnp.zeros_like(out[:, 0, 0])
        out = out.at[:, 2, 1].set(z4)
        out = out.at[:, 3, 1].set(z4)

        # Parallel closures (closures.cu:L63-74)
        dz_g20 = gp.dz(g_c[:, 2, 0])
        abs_dz_g30 = gp.abs_dz(g_c[:, 3, 0])
        abs_dz_g11 = gp.abs_dz(g_c[:, 1, 1])

        c_bpar = jnp.asarray(-beta_par / math.sqrt(3.0), dtype=r_dtype) * gpar * vt4
        c_dpar = jnp.asarray(-math.sqrt(2.0) * d_par, dtype=r_dtype) * gpar * vt4
        c_dperp = jnp.asarray(-math.sqrt(2.0) * d_perp, dtype=r_dtype) * gpar * vt4

        out = out.at[:, 3, 0].add(c_bpar * dz_g20 + c_dpar * abs_dz_g30)
        out = out.at[:, 1, 1].add(c_dperp * abs_dz_g11)

        # Toroidal closures (device_funcs.cu:L1306-1336)
        wd = op.omegad.astype(r_dtype)[None, :, :, :]
        wd_r = (wd * tz4).astype(r_dtype)
        iwd_s = jax.lax.complex(jnp.zeros_like(wd_r), wd_r).astype(c_dtype)
        awd_s = jnp.abs(wd) * tz4  # Matches C++ abs(omegad) * tz_ verbatim (L1315-1318)

        nu = op.closure_coeff.astype(c_dtype)
        nu_r = jnp.real(nu)
        nu_i = jnp.imag(nu)
        s2 = jnp.asarray(math.sqrt(2.0), dtype=r_dtype)
        s6 = jnp.asarray(math.sqrt(6.0), dtype=r_dtype)
        inv_s6 = jnp.asarray(1.0 / math.sqrt(6.0), dtype=r_dtype)

        g02 = g_c[:, 2, 0]
        g10 = g_c[:, 0, 1]
        g01 = g_c[:, 1, 0]
        g03 = g_c[:, 3, 0]
        g11 = g_c[:, 1, 1]

        t_02 = -s2 * (
            awd_s * (nu_r[1] * s2 * g02 + nu_r[2] * g10)
            + iwd_s * (nu_i[1] * s2 * g02 + nu_i[2] * g10)
        )
        t_10 = -2.0 * (
            awd_s * (nu_r[3] * s2 * g02 + nu_r[4] * g10)
            + iwd_s * (nu_i[3] * s2 * g02 + nu_i[4] * g10)
        )
        t_03 = -inv_s6 * (
            awd_s * (nu_r[5] * g01 + nu_r[6] * s6 * g03 + nu_r[7] * g11)
            + iwd_s * (nu_i[5] * g01 + nu_i[6] * s6 * g03 + nu_i[7] * g11)
        )
        t_11 = -(
            awd_s * (nu_r[8] * g01 + nu_r[9] * s6 * g03 + nu_r[10] * g11)
            + iwd_s * (nu_i[8] * g01 + nu_i[9] * s6 * g03 + nu_i[10] * g11)
        )

        out = out.at[:, 2, 0].add(t_02)
        out = out.at[:, 0, 1].add(t_10)
        out = out.at[:, 3, 0].add(t_03)
        out = out.at[:, 1, 1].add(t_11)
        return jnp.where(mask6, out, jnp.zeros_like(out))

    if op.closure_model == "smith_perp":
        # smith_perp_toroidal_closures (device_funcs.cu:L1338-1363)
        q = op.smith_perp_q
        ell_last = op.nlaguerre - 1
        aclos = op.closure_coeff.astype(c_dtype)  # [A_0, A_1, ..., A_{q-1}] for l = L, L-1, ...
        aclos_rev = aclos[::-1]  # ordered l = nl - q .. L
        ar = jnp.real(aclos_rev)[None, None, :, None, None, None]
        ai = jnp.imag(aclos_rev)[None, None, :, None, None, None]

        wd5 = op.omegad.astype(r_dtype)[None, None, :, :, :]
        tz5 = op.sp_tz.astype(r_dtype)[:, None, None, None, None]
        wd5_r = (wd5 * tz5).astype(r_dtype)[:, :, None, :, :, :]
        iwd_s = jax.lax.complex(jnp.zeros_like(wd5_r), wd5_r).astype(c_dtype)
        awd_s = (jnp.abs(wd5) * tz5)[:, :, None, :, :, :]

        g_tail = g_c[:, :, op.nlaguerre - q :]  # (ns, nm, q, nz, naky, nakx)
        clos = jnp.sum((awd_s * ai + iwd_s * ar) * g_tail, axis=2)
        out = out.at[:, :, ell_last].add(
            -jnp.asarray(float(ell_last + 1), dtype=r_dtype)
            * jnp.where(mask5, clos, jnp.zeros_like(clos))
        )
        return jnp.where(mask6, out, jnp.zeros_like(out))

    if op.closure_model == "smith_par":
        # SmithPar::apply_closures (closures.cu:L215-240)
        q = op.smith_par_q
        m_last = op.nhermite - 1
        acoeff = op.closure_coeff.astype(c_dtype)[::-1]  # ordered m = nm - q .. M
        ar = jnp.real(acoeff)[None, :, None, None, None, None]
        ai = jnp.imag(acoeff)[None, :, None, None, None, None]

        g_tail = g_c[:, op.nhermite - q :, :]  # (ns, q, nl, nz, naky, nakx)
        dz_tail = gp.dz(g_tail)
        abs_dz_tail = gp.abs_dz(g_tail)
        clos = jnp.sum(-ai * abs_dz_tail + ar * dz_tail, axis=1)  # (ns, nl, nz, naky, nakx)

        fac = (
            -jnp.asarray(math.sqrt(float(m_last + 1)) * op.gradpar, dtype=r_dtype)
            * op.sp_vt.astype(r_dtype)[:, None, None, None, None]
        )
        out = out.at[:, m_last, :].add(fac * jnp.where(mask5, clos, jnp.zeros_like(clos)))
        return jnp.where(mask6, out, jnp.zeros_like(out))

    raise ValueError(f"Unknown closure_model={op.closure_model!r}.")


def apply_hypercollisions_const(
    G: jax.Array,
    op: LinearOperator,
    dt: float | jax.Array,
) -> jax.Array:
    """Return constant-coefficient hypercollision tendency (``device_funcs.cu:L3329-3365``)."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    inv_dt = jnp.asarray(1.0, dtype=r_dtype) / jnp.asarray(dt, dtype=r_dtype)
    rate = (
        op.hypercoll_const_rate.astype(r_dtype) + op.hypercoll_const_lm.astype(r_dtype) * inv_dt
    )[:, :, :, None, None, None]
    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, -rate * g_c, jnp.zeros_like(g_c))


def apply_hypercollisions_kz(
    G: jax.Array,
    op: LinearOperator,
    grad_par: Any | None = None,
) -> jax.Array:
    """Return ``|kz|``-weighted Hermite hypercollision tendency (``linear.cu:L225-233``)."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    gp = op.grad_par if grad_par is None else grad_par
    fac = op.hypercoll_kz_fac.astype(r_dtype)[:, :, None, None, None, None]
    tmp_g = -fac * g_c
    return gp.abs_dz(tmp_g)


def apply_hyperdiff(G: jax.Array, op: LinearOperator) -> jax.Array:
    """Return perpendicular ``(kx, ky)`` hyperdiffusion (``device_funcs.cu:L3304-3327``)."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    dfac = op.hyperdiff_fac.astype(r_dtype)[None, None, None, None, :, :]
    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, -dfac * g_c, jnp.zeros_like(g_c))


def apply_hb_hyper(
    G: jax.Array,
    fields: Fields,
    grids: Grids,
    geo: Geometry,
    op: LinearOperator,
) -> jax.Array:
    """Return Hammett-Belli adaptive shear hyperdiffusivity tendency (``linear.cu:L188-204``)."""
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g_c = G.astype(c_dtype)
    favg, df = field_line_average(fields.phi.astype(c_dtype), grids, geo)

    kx = op.kx.astype(r_dtype)
    ky = op.ky.astype(r_dtype)
    w_osc = jnp.asarray(op.w_osc, dtype=r_dtype)
    half = jnp.asarray(0.5, dtype=r_dtype)

    # get_s01 (device_funcs.cu:L3415-3422)
    favg2 = jnp.real(favg) ** 2 + jnp.imag(favg) ** 2
    s01_raw = jnp.sum((jnp.abs(kx) ** 4) * favg2)
    s01 = half * (-w_osc + jnp.sqrt(w_osc * w_osc + 2.0 * s01_raw))

    # get_s1 (device_funcs.cu:L3386-3413)
    df2 = jnp.real(df) ** 2 + jnp.imag(df) ** 2  # (nz, naky, nakx)
    nonzonal_mask = op.active_mask & (jnp.arange(op.naky)[:, None] > 0)
    df2_nz = jnp.where(nonzonal_mask[None, :, :], df2, jnp.zeros_like(df2))

    ky4 = (ky**4)[None, :, None]
    kp4 = ((kx[None, :] ** 2 + ky[:, None] ** 2) ** 2)[None, :, :]
    s10_raw = jnp.sum(2.0 * ky4 * df2_nz, axis=(-2, -1))  # (nz,)
    s11_raw = jnp.sum(2.0 * kp4 * df2_nz, axis=(-2, -1))  # (nz,)
    s10 = half * (-w_osc + jnp.sqrt(w_osc * w_osc + 2.0 * s10_raw))
    s11 = half * (-w_osc + jnp.sqrt(w_osc * w_osc + 2.0 * s11_raw))

    # HB_hyper (device_funcs.cu:L3425-3458)
    d10 = op.hb_d10.astype(r_dtype)
    d01 = op.hb_d01.astype(r_dtype)
    d11 = op.hb_d11.astype(r_dtype)
    is_ky0 = (jnp.arange(op.naky) == 0)[None, :, None]
    sfac = jnp.where(
        is_ky0,
        s10[:, None, None] * d10[None, None, :],
        s11[:, None, None] * d11[None, :, :] + s01 * d01[None, :, :],
    )
    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, -sfac[None, None, None, :, :, :] * g_c, jnp.zeros_like(g_c))


def _fused_parallel_terms_nlink1(
    g_c: jax.Array,
    fields: Fields,
    op: LinearOperator,
    gp: ParallelGrad,
) -> jax.Array:
    """Fuse all ``nLinks = 1`` parallel spectral terms into a single ``fft`` + ``ifft`` pair.

    Combines ``streaming_rhs`` + ``dz``, ``hypercollisions_kz`` + ``abs_dz``, ``hyperz``,
    and parallel closure terms (``beer4+2`` / ``smith_par``) in ``kz``-space without altering
    arithmetic results.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(g_c)
    nm = op.nhermite

    vt5 = op.sp_vt.astype(r_dtype)[:, None, None, None, None]
    zt5 = op.sp_zt.astype(r_dtype)[:, None, None, None, None]
    gpar = jnp.asarray(op.gradpar, dtype=r_dtype)
    jl = op.jl.astype(r_dtype)
    jlb = op.jlb.astype(r_dtype)

    sqrt_m = op.sqrt_m.astype(r_dtype)[None, :, None, None, None, None]
    sqrt_mp1 = op.sqrt_mp1.astype(r_dtype)[None, :, None, None, None, None]

    kz_r = gp.kz_deriv.astype(r_dtype)
    ikz6 = jax.lax.complex(jnp.zeros_like(kz_r), kz_r).astype(c_dtype)[
        None, None, None, :, None, None
    ]
    akz6 = jnp.abs(kz_r)[None, None, None, :, None, None]

    # Single forward 6D transform of G along z (axis=-3) via gp.fft_z (FFT or MXU)
    g_hat = gp.fft_z(g_c)
    g_hat_pad = jnp.pad(g_hat, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0)))

    # 1. Free-streaming G_{m+-1} part in kz-space
    par_hat = ikz6 * (
        -(vt5[:, :, None, :, :, :] * gpar)
        * (sqrt_mp1 * g_hat_pad[:, 2:] + sqrt_m * g_hat_pad[:, :-2])
    )

    # 2. Free-streaming field terms on m = 0, 1, 2 (at most 2 species-Laguerre 5D transforms!)
    phi = fields.phi.astype(c_dtype)[None, None, :, :, :]
    apar = fields.apar.astype(c_dtype)[None, None, :, :, :]
    bpar = fields.bpar.astype(c_dtype)[None, None, :, :, :]
    ikz5 = jax.lax.complex(jnp.zeros_like(kz_r), kz_r).astype(c_dtype)[None, None, :, None, None]

    if op.solver.fapar != 0.0:
        s_apar_hat = gp.fft_z(jl * apar * (zt5 * vt5 * vt5 * gpar)) * ikz5
        par_hat = par_hat.at[:, 0].add(s_apar_hat)
        if nm >= 3:
            par_hat = par_hat.at[:, 2].add(jnp.asarray(math.sqrt(2.0), dtype=r_dtype) * s_apar_hat)
    if nm >= 2 and (op.solver.fphi != 0.0 or op.solver.fbpar != 0.0 or op.solver.phi_ext != 0.0):
        s_phib = (-jl * phi * (zt5 * vt5) - jlb * bpar * vt5) * gpar
        s_phib_hat = gp.fft_z(s_phib) * ikz5
        par_hat = par_hat.at[:, 1].add(s_phib_hat)

    # Beer 4+2 zeroes unevolved moments (m=2, l=1) and (m=3, l=1) at closures step
    # (linear.cu:L209, BEFORE hypercollisions_kz L225 and hyperz L239)
    if op.closure_model == "beer4+2":
        z_hat = jnp.zeros_like(par_hat[:, 0, 0])
        par_hat = par_hat.at[:, 2, 1].set(z_hat)
        par_hat = par_hat.at[:, 3, 1].set(z_hat)

    # 3. |kz|-weighted Hermite hypercollisions
    if op.hypercollisions_kz:
        fac_kz = op.hypercoll_kz_fac.astype(r_dtype)[:, :, None, None, None, None]
        par_hat = par_hat - (fac_kz * akz6) * g_hat

    # 4. Parallel hyperviscosity hyperz
    if op.hyperz:
        w_hz = (jnp.asarray(op.nu_hyper_z, dtype=r_dtype) * gp.hyperz_weight.astype(r_dtype))[
            None, None, None, :, None, None
        ]
        par_hat = par_hat - w_hz * g_hat

    # 5. Parallel closure terms (Beer 4+2 / SmithPar)
    if op.closure_model == "beer4+2":
        vt4 = op.sp_vt.astype(r_dtype)[:, None, None, None]
        ikz4 = jax.lax.complex(jnp.zeros_like(kz_r), kz_r).astype(c_dtype)[None, :, None, None]
        akz4 = jnp.abs(kz_r)[None, :, None, None]
        d_par = 2.0 * math.sqrt(math.pi) / (3.0 * math.pi - 8.0)
        beta_par = (32.0 - 9.0 * math.pi) / (3.0 * math.pi - 8.0)
        d_perp = 0.5 * math.sqrt(math.pi)
        c_bpar = jnp.asarray(-beta_par / math.sqrt(3.0), dtype=r_dtype) * gpar * vt4
        c_dpar = jnp.asarray(-math.sqrt(2.0) * d_par, dtype=r_dtype) * gpar * vt4
        c_dperp = jnp.asarray(-math.sqrt(2.0) * d_perp, dtype=r_dtype) * gpar * vt4
        par_hat = par_hat.at[:, 3, 0].add(
            c_bpar * (ikz4 * g_hat[:, 2, 0]) + c_dpar * (akz4 * g_hat[:, 3, 0])
        )
        par_hat = par_hat.at[:, 1, 1].add(c_dperp * (akz4 * g_hat[:, 1, 1]))
    elif op.closure_model == "smith_par":
        q = op.smith_par_q
        m_last = nm - 1
        acoeff = op.closure_coeff.astype(c_dtype)[::-1]
        ar = jnp.real(acoeff)[None, :, None, None, None, None]
        ai = jnp.imag(acoeff)[None, :, None, None, None, None]
        g_hat_tail = g_hat[:, nm - q :, :]
        clos_hat = jnp.sum((-ai * akz6 + ar * ikz6) * g_hat_tail, axis=1)
        fac = -jnp.asarray(math.sqrt(float(m_last + 1)) * op.gradpar, dtype=r_dtype) * vt5
        par_hat = par_hat.at[:, m_last, :].add(fac * clos_hat)

    out = gp.ifft_z(par_hat)
    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, out, jnp.zeros_like(out))


_linear_rhs_parallel_fused = _fused_parallel_terms_nlink1


def linear_rhs(
    G: jax.Array,
    fields: Fields,
    op: LinearOperator,
    dt: float | jax.Array,
    *,
    grids: Grids | None = None,
    geo: Geometry | None = None,
    use_fused_fft: bool = True,
) -> jax.Array:
    """Evaluate complete linear gyrokinetic right-hand side ``Linear_GK::rhs``.

    Ported strictly from ``gx/src/linear.cu:L167-244``.
    """
    c_dtype, _ = _complex_and_real_dtypes(G)
    mask6 = op.active_mask[None, None, None, None, :, :]
    g_c = jnp.where(mask6, G.astype(c_dtype), jnp.zeros_like(G, dtype=c_dtype))
    gp = op.grad_par
    h_c = build_nonadiabatic_h(g_c, fields, op.solver)

    can_fuse = use_fused_fft and isinstance(gp, ParallelGrad) and gp.supports_fused_z_fft

    if can_fuse:
        par_rhs = _fused_parallel_terms_nlink1(g_c, fields, op, gp)
        loc_rhs = rhs_linear_local(g_c, fields, op, H=h_c)
        if op.hb_hyper:
            if grids is None or geo is None:
                raise ValueError("grids and geo are required when cfg.hb_hyper=True.")
            loc_rhs = loc_rhs + apply_hb_hyper(g_c, fields, grids, geo, op)
        if op.closure_model == "beer4+2":
            z_loc = jnp.zeros_like(loc_rhs[:, 0, 0])
            loc_rhs = loc_rhs.at[:, 2, 1].set(z_loc)
            loc_rhs = loc_rhs.at[:, 3, 1].set(z_loc)
        rhs = par_rhs + loc_rhs
        # Apply remaining non-parallel parts of Beer 4+2 or SmithPerp closures
        if op.closure_model == "smith_perp":
            rhs = apply_closures(g_c, rhs, op, gp)
        elif op.closure_model == "beer4+2":
            rhs = _apply_beer42_toroidal_only(g_c, rhs, op)
    else:
        if op.nz > 1:
            rhs_par = streaming_rhs(g_c, fields, op)
            rhs = gp.dz(rhs_par)
        else:
            rhs = jnp.zeros_like(g_c)

        rhs = rhs + rhs_linear_local(g_c, fields, op, H=h_c)

        if op.hb_hyper:
            if grids is None or geo is None:
                raise ValueError("grids and geo are required when cfg.hb_hyper=True.")
            rhs = rhs + apply_hb_hyper(g_c, fields, grids, geo, op)

        if op.closure_model != "none":
            rhs = apply_closures(g_c, rhs, op, gp)

        if op.hypercollisions_kz and op.nz > 1:
            rhs = rhs + apply_hypercollisions_kz(g_c, op, gp)

        if op.hyperz:
            rhs = gp.hyperz(g_c, op.nu_hyper_z, accumulate=True, target=rhs)

    if op.hypercollisions_const:
        rhs = rhs + apply_hypercollisions_const(g_c, op, dt)

    if op.hyper:
        rhs = rhs + apply_hyperdiff(g_c, op)

    rhs = gp.apply_bcs(g_c, rhs, fields, op.solver, dt, H=h_c)
    return jnp.where(mask6, rhs, jnp.zeros_like(rhs))


def _apply_beer42_toroidal_only(
    g_c: jax.Array,
    rhs: jax.Array,
    op: LinearOperator,
) -> jax.Array:
    """Apply toroidal part of Beer 4+2 when parallel + unevolved-mask parts were fused."""
    c_dtype, r_dtype = _complex_and_real_dtypes(g_c)
    out = rhs.astype(c_dtype)

    tz4 = op.sp_tz.astype(r_dtype)[:, None, None, None]
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

    g02, g10, g01, g03, g11 = (
        g_c[:, 2, 0],
        g_c[:, 0, 1],
        g_c[:, 1, 0],
        g_c[:, 3, 0],
        g_c[:, 1, 1],
    )
    out = out.at[:, 2, 0].add(
        -s2
        * (
            awd_s * (nu_r[1] * s2 * g02 + nu_r[2] * g10)
            + iwd_s * (nu_i[1] * s2 * g02 + nu_i[2] * g10)
        )
    )
    out = out.at[:, 0, 1].add(
        -2.0
        * (
            awd_s * (nu_r[3] * s2 * g02 + nu_r[4] * g10)
            + iwd_s * (nu_i[3] * s2 * g02 + nu_i[4] * g10)
        )
    )
    out = out.at[:, 3, 0].add(
        -inv_s6
        * (
            awd_s * (nu_r[5] * g01 + nu_r[6] * s6 * g03 + nu_r[7] * g11)
            + iwd_s * (nu_i[5] * g01 + nu_i[6] * s6 * g03 + nu_i[7] * g11)
        )
    )
    out = out.at[:, 1, 1].add(
        -(
            awd_s * (nu_r[8] * g01 + nu_r[9] * s6 * g03 + nu_r[10] * g11)
            + iwd_s * (nu_i[8] * g01 + nu_i[9] * s6 * g03 + nu_i[10] * g11)
        )
    )
    mask6 = op.active_mask[None, None, None, None, :, :]
    return jnp.where(mask6, out, jnp.zeros_like(out))


# ---------------------------------------------------------------------------
# CFL Frequency Estimator, Linear Time Integrators, & Growth-Rate Diagnostics
# ---------------------------------------------------------------------------

_SCHEME_CFL_FAC: dict[str, float] = {
    "rk4": 2.82,
    "rk3": 1.73,
    "sspx3": 1.73,
    "sspx2": 1.0,
    "k10": 1.0,
}


def compute_linear_max_frequency(
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
) -> tuple[float, float, float]:
    """Estimate maximum linear frequencies ``(omega_max_x, omega_max_y, omega_max_z)``.

    Ported strictly from ``Linear_GK::get_max_frequency`` (``linear.cu:L246-272``) and
    ``Geometry::initializeOperatorArrays`` (``geometry.cu:L964-988``).
    """
    cv_max = float(jnp.max(jnp.abs(geo.cvdrift)))
    gb_max = float(jnp.max(jnp.abs(geo.gbdrift)))
    cv0_max = float(jnp.max(jnp.abs(geo.cvdrift0)))
    gb0_max = float(jnp.max(jnp.abs(geo.gbdrift0)))
    bmag_max = float(jnp.max(jnp.abs(geo.bmag)))

    vpar2 = float(grids.vpar_max) ** 2
    mub_max = float(grids.muB_max)

    if geo.shat == 0.0:
        w0 = float(cfg.tzmax) * float(grids.kx_max) * (vpar2 * cv0_max + mub_max * gb0_max)
    elif cfg.non_twist and geo.m0.size > 0:
        # Maximize m0(nz, ky_max) * (vpar_max^2 |cvdrift0| + muB_max |gbdrift0|) (geometry.cu:L980)
        m0_top = np.abs(np.asarray(geo.m0[:, -1], dtype=np.float64))
        cv0_arr = np.abs(np.asarray(geo.cvdrift0, dtype=np.float64))
        gb0_arr = np.abs(np.asarray(geo.gbdrift0, dtype=np.float64))
        scores = m0_top * (vpar2 * cv0_arr + mub_max * gb0_arr)
        best = int(np.argmax(scores)) if scores.size > 0 else 0
        m0_max = float(m0_top[best]) if scores.size > 0 and scores[best] > 0.0 else 0.0
        cv0_ntft = float(cv0_arr[best]) if scores.size > 0 and scores[best] > 0.0 else cv0_max
        gb0_ntft = float(gb0_arr[best]) if scores.size > 0 and scores[best] > 0.0 else gb0_max
        w0 = (
            float(cfg.tzmax)
            * (float(grids.kx_max) + m0_max / float(grids.x0))
            * (vpar2 * cv0_ntft + mub_max * gb0_ntft)
        )
    else:
        w0 = (
            float(cfg.tzmax)
            * (float(grids.kx_max) / abs(float(geo.shat)))
            * (vpar2 * cv0_max + mub_max * gb0_max)
        )

    w1 = float(cfg.tzmax) * float(grids.ky_max) * (vpar2 * cv_max + mub_max * gb_max)
    if cfg.linear and cfg.etamax < 1.0e5:
        w1 += float(grids.ky_max) * (1.0 + float(cfg.etamax) * (0.5 * vpar2 + mub_max - 1.5))

    nte = float(cfg.ne) * float(cfg.te)
    mime = (float(cfg.vtmax) / float(cfg.vtmin)) ** 2
    kperprho2 = (float(grids.kperp_min) / bmag_max) ** 2 if bmag_max > 0.0 else 0.0
    denom_alfven = 0.5 * float(cfg.beta) * nte * mime + kperprho2
    alfven_fac = (
        (1.0 / math.sqrt(denom_alfven)) if (cfg.nspecies > 1 and denom_alfven > 0.0) else 0.0
    )
    w2 = (
        float(cfg.vtmax)
        * float(grids.kz_max)
        * abs(float(geo.gradpar))
        * max(float(grids.vpar_max), alfven_fac)
    )
    return (float(w0), float(w1), float(w2))


def compute_linear_dt(cfg: GXConfig, grids: Grids, geo: Geometry) -> float:
    """Compute linear CFL timestep ``dt`` matching ``Timestepper::set_timestep``."""
    if cfg.fixed_dt:
        return float(cfg.dt)
    scheme_key = cfg.scheme.strip().lower()
    if scheme_key not in _SCHEME_CFL_FAC:
        raise ValueError(
            f"Unsupported time scheme={cfg.scheme!r}. Supported: {sorted(_SCHEME_CFL_FAC.keys())}."
        )
    w0, w1, w2 = compute_linear_max_frequency(cfg, grids, geo)
    wmax = w0 + w1 + w2
    if wmax <= 0.0:
        return float(cfg.dt_max)
    cfl_fac = _SCHEME_CFL_FAC[scheme_key]
    dt_guess = cfl_fac * float(cfg.cfl) / wmax
    if scheme_key == "k10":
        return float(min(dt_guess, float(cfg.dt_max)))
    return float(min(max(dt_guess, float(cfg.dt_min)), float(cfg.dt_max)))


def step_linear(
    G: jax.Array,
    fields: Fields,
    op: LinearOperator,
    dt: float | jax.Array,
    *,
    scheme: str = "rk4",
    grids: Grids | None = None,
    geo: Geometry | None = None,
) -> tuple[jax.Array, Fields]:
    """Advance ``(G, fields)`` by one linear explicit timestep ``dt``.

    Ported from ``ts_rk4.cu``, ``ts_rk3.cu``, ``ts_sspx2.cu``, ``ts_sspx3.cu``, and ``ts_k10.cu``.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(G)
    g0 = G.astype(c_dtype)
    f0 = fields.astype(c_dtype)
    dt_r = jnp.asarray(dt, dtype=r_dtype)
    gp = op.grad_par
    dealias_kz = bool(getattr(gp, "dealias_kz", False))

    def _rhs(g_in: jax.Array, f_in: Fields) -> jax.Array:
        return linear_rhs(g_in, f_in, op, dt_r, grids=grids, geo=geo)

    def _solve(g_in: jax.Array, *, apply_kz_dealias: bool = False) -> Fields:
        f_out = solve_fields(g_in, op.solver)
        if apply_kz_dealias and dealias_kz:
            phi_d = gp.dealias(f_out.phi)
            f_out = Fields(phi=phi_d, apar=f_out.apar, bpar=f_out.bpar, phi_avg=f_out.phi_avg)
        return f_out

    sk = scheme.strip().lower()

    if sk == "rk4":
        half = jnp.asarray(0.5, dtype=r_dtype)
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        two = jnp.asarray(2.0, dtype=r_dtype)

        k1 = _rhs(g0, f0)
        g1 = g0 + (half * dt_r) * k1
        f1 = _solve(g1)

        k2 = _rhs(g1, f1)
        g2 = g0 + (half * dt_r) * k2
        f2 = _solve(g2)

        k3 = _rhs(g2, f2)
        g3 = g0 + dt_r * k3
        f3 = _solve(g3)

        k4 = _rhs(g3, f3)
        g_next = g0 + (sixth * dt_r) * (k1 + two * k2 + two * k3 + k4)
        return g_next, _solve(g_next)

    if sk == "rk3":
        # Heun's 3-stage RK3 (ts_rk3.cu:L97-118)
        c13 = jnp.asarray(1.0 / 3.0, dtype=r_dtype)
        c23 = jnp.asarray(2.0 / 3.0, dtype=r_dtype)
        c34 = jnp.asarray(0.75, dtype=r_dtype)
        c14 = jnp.asarray(0.25, dtype=r_dtype)

        k1 = _rhs(g0, f0)
        g_q1 = g0 + (c13 * dt_r) * k1
        f_q1 = _solve(g_q1)

        k2 = _rhs(g_q1, f_q1)
        g_q2 = g0 + (c23 * dt_r) * k2
        f_q2 = _solve(g_q2)

        k3 = _rhs(g_q2, f_q2)
        g_next = g0 + (c34 * dt_r) * k3 + (c14 * dt_r) * k1
        return g_next, _solve(g_next)

    if sk == "sspx2":
        # 2-stage SSP RK2 (ts_sspx2.cu:L79-90), written in increment form g0 + delta to avoid
        # O(1) g0/g1 subtraction loss in complex64 when omega * dt << 1:
        # (2 - sqrt2)*g0 + (sqrt2 - 2)*g1 + (g1 + adt*dt*r1) == g0 + (sqrt2 - 1)*d0 + adt*dt*r1
        adt = jnp.asarray(1.0 / math.sqrt(2.0), dtype=r_dtype)
        c_d0 = jnp.asarray(math.sqrt(2.0) - 1.0, dtype=r_dtype)

        d0 = (adt * dt_r) * _rhs(g0, f0)
        g1 = g0 + d0
        f1 = _solve(g1)
        g_next = g0 + (c_d0 * d0 + (adt * dt_r) * _rhs(g1, f1))
        return g_next, _solve(g_next)

    if sk == "sspx3":
        # 3-stage SSP RK3 (timestepper.h:L149-153, ts_sspx3.cu:L113-130) in increment form
        # g0 + delta (avoids O(1) g0/g1/g2 cancellation in complex64 over thousands of steps):
        #   d0 = adt * dt * r0,                g1 = g0 + d0
        #   d2 = w1 * d0 + adt * dt * r1,      g2 = g0 + d2
        #   g_next = g0 + (w3 * d0 + w2 * d2 + adt * dt * r2)
        adt_f = (1.0 / 6.0) ** (1.0 / 3.0)
        wgtfac = math.sqrt(9.0 - 2.0 * (6.0 ** (2.0 / 3.0)))
        w1_f = 0.5 * (wgtfac - 1.0)
        w2_f = 0.5 * (6.0 ** (2.0 / 3.0) - 1.0 - wgtfac)
        w3_f = 1.0 / adt_f - 1.0 - w2_f * (w1_f + 1.0)

        adt = jnp.asarray(adt_f, dtype=r_dtype)
        w1 = jnp.asarray(w1_f, dtype=r_dtype)
        w2 = jnp.asarray(w2_f, dtype=r_dtype)
        w3 = jnp.asarray(w3_f, dtype=r_dtype)

        r0 = _rhs(g0, f0)
        if dealias_kz:
            r0 = gp.dealias(r0)
        d0 = (adt * dt_r) * r0
        g1 = g0 + d0
        f1 = _solve(g1, apply_kz_dealias=True)

        r1 = _rhs(g1, f1)
        if dealias_kz:
            r1 = gp.dealias(r1)
        d2 = w1 * d0 + (adt * dt_r) * r1
        g2 = g0 + d2
        f2 = _solve(g2, apply_kz_dealias=True)

        r2 = _rhs(g2, f2)
        if dealias_kz:
            r2 = gp.dealias(r2)
        g_next = g0 + (w3 * d0 + w2 * d2 + (adt * dt_r) * r2)
        mask6 = op.active_mask[None, None, None, None, :, :]
        g_next = jnp.where(mask6, g_next, jnp.zeros_like(g_next))
        return g_next, _solve(g_next, apply_kz_dealias=True)

    if sk == "k10":
        # Ketcheson 10-stage, 4th-order low-storage RK (ts_k10.cu:L49-136) in increment form:
        #   d5 = sum_{j=0..4} (dt/6) r_j
        #   15 * (0.04 g0 + 0.36 (g0 + d5)) - 5 * (g0 + d5) == g0 + 0.4 * d5
        #   d9 = 0.4 * d5 + sum_{j=5..8} (dt/6) r_j
        #   g_next == g0 + (0.36 * d5 + 0.6 * d9 + 0.1 * dt * r9)
        sixth = jnp.asarray(1.0 / 6.0, dtype=r_dtype)
        tenth = jnp.asarray(0.1, dtype=r_dtype)

        def _euler_k10_delta(
            carry: tuple[jax.Array, jax.Array, Fields],
        ) -> tuple[jax.Array, jax.Array, Fields]:
            dq, gq, fq = carry
            rq = _rhs(gq, fq)
            if dealias_kz:
                rq = gp.dealias(rq)
            dq_next = dq + (sixth * dt_r) * rq
            gq_next = g0 + dq_next
            return dq_next, gq_next, _solve(gq_next, apply_kz_dealias=True)

        d_acc = jnp.zeros_like(g0)
        g_q1, f_q1 = g0, f0
        for _ in range(5):
            d_acc, g_q1, f_q1 = _euler_k10_delta((d_acc, g_q1, f_q1))
        d5 = d_acc

        d_acc = jnp.asarray(0.4, dtype=r_dtype) * d5
        g_q1 = g0 + d_acc
        f_q1 = _solve(g_q1, apply_kz_dealias=True)

        for _ in range(4):
            d_acc, g_q1, f_q1 = _euler_k10_delta((d_acc, g_q1, f_q1))
        d9 = d_acc

        r_last = _rhs(g_q1, f_q1)
        if dealias_kz:
            r_last = gp.dealias(r_last)
        g_next = g0 + (
            jnp.asarray(0.36, dtype=r_dtype) * d5
            + jnp.asarray(0.6, dtype=r_dtype) * d9
            + (tenth * dt_r) * r_last
        )
        return g_next, _solve(g_next, apply_kz_dealias=True)

    raise ValueError(
        f"Unsupported scheme={scheme!r}. Supported: {'rk3', 'rk4', 'sspx2', 'sspx3', 'k10'}."
    )


def advance_linear_steps(
    G: jax.Array,
    fields: Fields,
    op: LinearOperator,
    dt: float,
    nsteps: int,
    *,
    scheme: str = "rk4",
    grids: Grids | None = None,
    geo: Geometry | None = None,
) -> tuple[jax.Array, Fields, jax.Array]:
    """Advance ``nsteps`` linear timesteps inside a single compiled ``lax.fori_loop``.

    Returns ``(G_final, fields_final, phi_prev)``, where ``phi_prev`` is ``fields.phi``
    at step ``nsteps - 1`` for direct input to :func:`compute_growth_rates`.
    """
    if nsteps < 1:
        raise ValueError(f"nsteps={nsteps} must be >= 1.")

    def _one(g_in: jax.Array, f_in: Fields) -> tuple[jax.Array, Fields]:
        return step_linear(g_in, f_in, op, dt, scheme=scheme, grids=grids, geo=geo)

    if nsteps == 1:
        phi_prev = fields.phi
        g_end, f_end = _one(G, fields)
        return g_end, f_end, phi_prev

    def _body(_i: int, carry: tuple[jax.Array, Fields]) -> tuple[jax.Array, Fields]:
        return _one(carry[0], carry[1])

    g_pre, f_pre = jax.lax.fori_loop(0, nsteps - 1, _body, (G, fields))
    g_end, f_end = _one(g_pre, f_pre)
    return g_end, f_end, f_pre.phi


def compute_growth_rates(
    phi: jax.Array,
    phi_old: jax.Array,
    dt: float | jax.Array,
    grids: Grids,
) -> jax.Array:
    """Compute complex frequency ``omega + 1j * gamma`` matching ``growthRates``.

    Ported strictly from ``device_funcs.cu:L863-892``:
    samples ``phi`` at ``iz = nz // 2 + 1`` (or ``0`` if ``nz == 1``) and evaluates
    ``(log(|phi / phi_old|) + 1j * atan2(Im(r), Re(r))) * (1j / dt)`` on active modes where
    both ``Re(phi)`` and ``Im(phi)`` (and ``phi_old``) are non-zero.
    """
    c_dtype, r_dtype = _complex_and_real_dtypes(phi)
    ig = (grids.nz // 2 + 1) if grids.nz > 1 else 0
    p_now = phi.astype(c_dtype)[ig]
    p_old = phi_old.astype(c_dtype)[ig]

    valid = (
        grids.active_mask
        & (jnp.abs(jnp.real(p_now)) != 0.0)
        & (jnp.abs(jnp.imag(p_now)) != 0.0)
        & ((jnp.real(p_old) != 0.0) | (jnp.imag(p_old) != 0.0))
    )
    ratio = _safe_divide(p_now, p_old, valid)
    abs_r = jnp.where(valid, jnp.abs(ratio), jnp.ones_like(jnp.real(ratio)))
    log_r = jax.lax.complex(
        jnp.log(abs_r).astype(r_dtype),
        jnp.arctan2(jnp.imag(ratio), jnp.real(ratio)).astype(r_dtype),
    ).astype(c_dtype)
    inv_dt = (jnp.ones((), dtype=r_dtype) / jnp.asarray(dt, dtype=r_dtype)).astype(r_dtype)
    i_dt = jax.lax.complex(jnp.zeros((), dtype=r_dtype), inv_dt).astype(c_dtype)
    omg = (log_r * i_dt).astype(c_dtype)
    return jnp.where(valid, omg, jnp.zeros_like(omg))


def compute_phi2_ky(phi: jax.Array, grids: Grids, geo: Geometry) -> jax.Array:
    """Compute ``Phi2_kyt`` spectrum of shape ``(naky,)`` (``device_funcs.cu:Phi2_summand``)."""
    _, r_dtype = _complex_and_real_dtypes(phi)
    fac_ky = jnp.where(jnp.arange(grids.naky) == 0, 1.0, 2.0).astype(r_dtype)[None, :, None]
    vol = geo.vol_fac.astype(r_dtype)[:, None, None]
    p2_raw = 0.5 * fac_ky * vol * (jnp.real(phi) ** 2 + jnp.imag(phi) ** 2)
    p2_masked = jnp.where(grids.active_mask[None, :, :], p2_raw, jnp.zeros_like(p2_raw))
    return jnp.sum(p2_masked, axis=(0, 2))


def extract_single_ky_config(cfg: GXConfig, iky: int) -> GXConfig:
    """Build a single-``ky`` ``GXConfig`` (``nky=2, y0=cfg.y0/iky``) for uncoupled linear runs."""
    if iky < 1:
        raise ValueError(f"iky={iky} must be >= 1.")
    return dataclasses.replace(cfg, ny=4, y0=float(cfg.y0) / float(iky), iky_single=1)


def solve_linear_eigenmode(
    op: LinearOperator,
    dt_ref: float,
    *,
    iky: int = 1,
    ikx: int = 0,
    steps_per_iter: int = 40,
    scheme: str = "rk4",
    v0: np.ndarray | jax.Array | None = None,
    tol: float = 1.0e-6,
    maxiter: int = 200,
    ncv: int = 14,
) -> tuple[complex, jax.Array]:
    """Compute leading unstable eigenvalue ``omega + 1j * gamma`` at ``(iky, ikx)`` via Krylov.

    Applies implicitly restarted Arnoldi (``scipy.sparse.linalg.eigs(..., which='LM')``) to the
    compiled ``steps_per_iter``-step exponential propagator ``G(t) -> G(t + K * dt_ref)``.
    Mapping continuous eigenvalues ``lambda = gamma - i * omega`` to ``mu = exp(lambda * K * dt)``
    converts high-frequency damped parallel streaming modes into ``|mu| <= 1`` while mapping the
    fastest-growing physical instability (maximum ``gamma``) to the dominant modulus ``|mu|``,
    achieving ~50x faster convergence than direct time-marching.
    """
    if not (0 <= iky < op.naky and 0 <= ikx < op.nakx):
        raise ValueError(f"(iky={iky}, ikx={ikx}) out of bounds for {(op.naky, op.nakx)}.")
    if not bool(op.active_mask[iky, ikx]):
        raise ValueError(f"Mode (iky={iky}, ikx={ikx}) is masked.")
    if steps_per_iter < 1:
        raise ValueError(f"steps_per_iter={steps_per_iter} must be >= 1.")

    gp = op.grad_par
    coupled_ikx: tuple[int, ...] = (
        tuple(gp.coupled_ikx_indices(iky, ikx))
        if hasattr(gp, "coupled_ikx_indices")
        else (int(ikx),)
    )
    single_kx = len(coupled_ikx) == 1

    base_shape = (op.nspecies, op.nhermite, op.nlaguerre, op.nz)
    mode_shape = base_shape if single_kx else (*base_shape, len(coupled_ikx))
    n_dof = int(np.prod(mode_shape))
    use_f64 = op.jl.dtype == jnp.float64
    c_jax_dtype = jnp.complex128 if use_f64 else jnp.complex64
    c_np_dtype = np.complex128 if use_f64 else np.complex64
    dt_f = float(dt_ref)
    kx_sub = jnp.asarray(coupled_ikx, dtype=jnp.int32)

    if single_kx:

        @jax.jit
        def _prop(v_mode: jax.Array) -> jax.Array:
            g_full = jnp.zeros(
                (op.nspecies, op.nhermite, op.nlaguerre, op.nz, op.naky, op.nakx),
                dtype=c_jax_dtype,
            )
            g_full = g_full.at[..., iky, ikx].set(v_mode)
            f_full = solve_fields(g_full, op.solver)
            g_end, _, _ = advance_linear_steps(
                g_full, f_full, op, dt_f, steps_per_iter, scheme=scheme
            )
            return g_end[..., iky, ikx]
    else:

        @jax.jit
        def _prop(v_mode: jax.Array) -> jax.Array:
            g_full = jnp.zeros(
                (op.nspecies, op.nhermite, op.nlaguerre, op.nz, op.naky, op.nakx),
                dtype=c_jax_dtype,
            )
            g_full = g_full.at[..., iky, kx_sub].set(v_mode)
            f_full = solve_fields(g_full, op.solver)
            g_end, _, _ = advance_linear_steps(
                g_full, f_full, op, dt_f, steps_per_iter, scheme=scheme
            )
            return g_end[..., iky, kx_sub]

    def _mv(vec_flat: np.ndarray) -> np.ndarray:
        v_j = jnp.asarray(vec_flat, dtype=c_jax_dtype).reshape(mode_shape)
        out_j = _prop(v_j)
        return np.asarray(out_j, dtype=c_np_dtype).ravel()

    lin_op = ScipyLinearOperator((n_dof, n_dof), matvec=_mv, dtype=c_np_dtype)
    rng_det = np.random.default_rng(0)
    if v0 is None:
        v0_arr = (rng_det.standard_normal(n_dof) + 1.0j * rng_det.standard_normal(n_dof)).astype(
            c_np_dtype
        )
    else:
        v0_np = np.asarray(v0, dtype=c_np_dtype)
        if not single_kx and v0_np.shape == (*base_shape, op.nakx):
            v0_arr = v0_np[..., list(coupled_ikx)].ravel()
        elif not single_kx and v0_np.shape == base_shape:
            v0_sub = np.zeros(mode_shape, dtype=c_np_dtype)
            v0_sub[..., coupled_ikx.index(int(ikx))] = v0_np
            v0_arr = v0_sub.ravel()
        else:
            v0_arr = v0_np.ravel()

    evals, evecs = scipy_eigs(
        lin_op,
        k=1,
        which="LM",
        v0=v0_arr,
        tol=tol,
        maxiter=maxiter,
        ncv=min(ncv, n_dof - 1),
        rng=rng_det,
    )
    eigvec_sub = jnp.asarray(evecs[:, 0].reshape(mode_shape), dtype=jnp.complex128)

    # Extract unwrapped single-step omega + i * gamma in FP64 from 1 step on the converged
    # eigenvector (avoids 2*pi / (K * dt) branch-cut wrapping when omega * K * dt > pi).
    op64 = op if use_f64 else op.astype(jnp.float64)
    g_eig = jnp.zeros(
        (op64.nspecies, op64.nhermite, op64.nlaguerre, op64.nz, op64.naky, op64.nakx),
        dtype=jnp.complex128,
    )
    if single_kx:
        eigvec = eigvec_sub
        g_eig = g_eig.at[..., iky, ikx].set(eigvec)
    else:
        eigvec = (
            jnp.zeros((*base_shape, op64.nakx), dtype=jnp.complex128)
            .at[..., kx_sub]
            .set(eigvec_sub)
        )
        g_eig = g_eig.at[..., iky, :].set(eigvec)

    f_eig0 = solve_fields(g_eig, op64.solver)
    _, f_eig1 = step_linear(g_eig, f_eig0, op64, dt_f, scheme=scheme)
    ig = (op64.nz // 2 + 1) if op64.nz > 1 else 0
    if single_kx:
        ratio = complex(f_eig1.phi[ig, iky, ikx]) / complex(f_eig0.phi[ig, iky, ikx])
    else:
        phi0_chain = np.asarray(f_eig0.phi[:, iky, list(coupled_ikx)], dtype=np.complex128)
        phi1_chain = np.asarray(f_eig1.phi[:, iky, list(coupled_ikx)], dtype=np.complex128)
        denom = complex(np.vdot(phi0_chain, phi0_chain))
        ratio = (
            complex(np.vdot(phi0_chain, phi1_chain)) / denom
            if denom.real > 0.0
            else complex(f_eig1.phi[ig, iky, ikx]) / complex(f_eig0.phi[ig, iky, ikx])
        )
    del evals
    omega_gx = (math.log(abs(ratio)) + 1.0j * math.atan2(ratio.imag, ratio.real)) * (1.0j / dt_f)
    return omega_gx, eigvec


def mask_linear_state(G: jax.Array, grids: Grids) -> jax.Array:
    """Convenience alias for :func:`gx_jax.fields.mask_g`."""
    return mask_g(G, grids)
