"""Gauss-Laguerre velocity-space quadrature and Bessel/FLR operators for GX-JAX.

Ported strictly from:
- ``gx/include/laguerre_transform.h``
- ``gx/src/laguerre_transform.cu``
- ``gx/src/device_funcs.cu`` (``Jflr``, ``JflrA``, ``JflrB``, ``Jfac``,
  ``g0``, ``g1``, ``J0fToGrid``, ``J0phiAndBparToGrid``)

Critical GX Sign Convention (``laguerre_transform.cu:L67-71``):
GX defines its Laguerre basis polynomials with an alternating ``(-1)**l`` factor relative to
the standard physicist/mathematician ``L_l(x)``:
    ``L_hat_l(x) = (-1)**l * L_l(x) = (-1)**l * sum_{k=0}^l comb(l, k) (-x)**k / k!``
so ``L_hat_0(x) = 1``, ``L_hat_1(x) = x - 1``, ``L_hat_2(x) = (x**2 - 4x + 2) / 2``, etc.
Under this convention, the Gauss-Laguerre projection of ``J_0(sqrt(2 * x * b))`` is
``jflr(l, b) = (-b / 2)**l / l! * exp(-b / 2)`` (``device_funcs.cu:L127-139``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Precomputed 1 / m! and log(m!) tables for m = 0..30 matching device_funcs.cu:inv_factorial
_INV_FACTORIALS_TUPLE: tuple[float, ...] = tuple(1.0 / math.factorial(m) for m in range(31))
_LOG_FACTORIALS_TUPLE: tuple[float, ...] = tuple(math.lgamma(m + 1) for m in range(31))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LaguerreTransform:
    """Gauss-Laguerre quadrature nodes, weights, and spectral/grid transform matrices."""

    nl: int
    nj: int
    vmax: float
    vmax_J: float
    roots: jax.Array  # (nj,)mu*B quadrature nodes x_j > 0
    weights: jax.Array  # (nj,) Gauss-Laguerre weights w_j, sum(w_j) == 1
    to_grid: jax.Array  # (nj, nl) L_hat_l(x_j)
    to_spectral: jax.Array  # (nl, nj) w_j * L_hat_l(x_j)

    def tree_flatten(self):
        children = (self.roots, self.weights, self.to_grid, self.to_spectral)
        aux_data = (self.nl, self.nj, self.vmax, self.vmax_J)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> LaguerreTransform:
        nl, nj, vmax, vmax_J = aux_data
        roots, weights, to_grid, to_spectral = children
        return cls(
            nl=nl,
            nj=nj,
            vmax=vmax,
            vmax_J=vmax_J,
            roots=roots,
            weights=weights,
            to_grid=to_grid,
            to_spectral=to_spectral,
        )


def _eval_gx_laguerre_polynomials(x: jax.Array, nl: int) -> jax.Array:
    """Evaluate ``L_hat_l(x) = (-1)**l L_l(x)`` for ``l = 0..nl-1`` via stable 3-term recurrence.

    Recurrence in float64 (avoids catastrophic cancellation of monomial expansion at high ``nl``):
        L_hat_0(x) = 1
        L_hat_1(x) = x - 1
        L_hat_{l+1}(x) = ((x - (2*l + 1)) * L_hat_l(x) - l * L_hat_{l-1}(x)) / (l + 1)
    Returns array of shape ``(x.shape[0], nl)`` in ``float64``.
    """
    x64 = jnp.asarray(x, dtype=jnp.float64)
    n_pts = x64.shape[0]
    if nl <= 0:
        raise ValueError(f"nl must be >= 1, got {nl}.")
    l0 = jnp.ones((n_pts,), dtype=jnp.float64)
    if nl == 1:
        return l0[:, None]
    l1 = x64 - 1.0
    cols = [l0, l1]
    for ell in range(1, nl - 1):
        l_next = ((x64 - (2.0 * ell + 1.0)) * cols[-1] - float(ell) * cols[-2]) / float(ell + 1)
        cols.append(l_next)
    return jnp.stack(cols, axis=1)


def init_laguerre(nl: int, dtype: jnp.dtype = jnp.float64) -> LaguerreTransform:
    """Construct Gauss-Laguerre quadrature & transform matrices in FP64 before casting to ``dtype``.

    Matches ``LaguerreTransform::initTransforms`` (``laguerre_transform.cu:L39-109``) via the
    Golub-Welsch symmetric tridiagonal Jacobi eigensystem of order ``nj = max(1, (3*nl)//2 - 1)``.
    """
    if nl < 1:
        raise ValueError(f"nl must be >= 1, got {nl}.")
    nj = max(1, (3 * nl) // 2 - 1)

    if nj == 1:
        roots64 = jnp.ones((1,), dtype=jnp.float64)
        weights64 = jnp.ones((1,), dtype=jnp.float64)
    else:
        idx = jnp.arange(nj, dtype=jnp.float64)
        diag = 2.0 * idx + 1.0
        off = jnp.arange(1, nj, dtype=jnp.float64)
        jacobi = jnp.diag(diag) + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
        evals, evecs = jnp.linalg.eigh(jacobi)
        roots64 = evals
        weights64 = evecs[0, :] ** 2

    vmax = float(roots64[min(nl - 1, nj - 1)])
    vmax_J = float(roots64[nj - 1])

    # to_grid[j, l] = L_hat_l(roots[j])  (matches C++ toGrid_h[l + L * j])
    to_grid64 = _eval_gx_laguerre_polynomials(roots64, nl)
    # to_spectral[l, j] = w_j * L_hat_l(roots[j])  (matches C++ toSpectral_h[j + J * l])
    to_spectral64 = to_grid64.T * weights64[None, :]

    out_dtype = jnp.dtype(dtype)
    return LaguerreTransform(
        nl=nl,
        nj=nj,
        vmax=vmax,
        vmax_J=vmax_J,
        roots=roots64.astype(out_dtype),
        weights=weights64.astype(out_dtype),
        to_grid=to_grid64.astype(out_dtype),
        to_spectral=to_spectral64.astype(out_dtype),
    )


def _contract_laguerre_axis(arr: jax.Array, mat: jax.Array, ax: int) -> jax.Array:
    """Contract ``mat (out_len, in_len)`` along axis ``ax`` of ``arr`` with zero XLA transposes."""
    batch_shape = arr.shape[:ax]
    batch_axes = tuple(range(ax))
    mat_bcast = jnp.broadcast_to(
        jnp.reshape(mat, (*((1,) * ax), *mat.shape)),
        (*batch_shape, *mat.shape),
    )
    return jax.lax.dot_general(
        mat_bcast,
        arr,
        dimension_numbers=(((ax + 1,), (ax,)), (batch_axes, batch_axes)),
        precision=jax.lax.Precision.HIGHEST,
    )


def laguerre_to_grid(
    g_spec: jax.Array,
    lag: LaguerreTransform,
    *,
    axis: int = 2,
) -> jax.Array:
    """Transform Laguerre spectral moments ``(..., nl, ...)`` -> ``mu*B`` grid ``(..., nj, ...)``.

    Canonical 6D state ``G[nspecies, nhermite, nlaguerre, nz, ky, kx]`` has ``axis=2``.
    Uses ``jax.lax.Precision.HIGHEST`` to prevent TPU MXU ``bfloat16`` truncation.
    """
    ndim = g_spec.ndim
    ax = axis % ndim
    if g_spec.shape[ax] != lag.nl:
        raise ValueError(
            f"Expected axis {axis} to have length nl={lag.nl}, got shape {g_spec.shape}."
        )
    mat = lag.to_grid.astype(jnp.real(jnp.empty((), dtype=g_spec.dtype)).dtype)
    return _contract_laguerre_axis(g_spec, mat, ax)


def laguerre_to_spectral(
    g_grid: jax.Array,
    lag: LaguerreTransform,
    *,
    axis: int = 2,
) -> jax.Array:
    """Transform ``mu*B`` grid values ``(..., nj, ...)`` -> Laguerre moments ``(..., nl, ...)``.

    Uses ``jax.lax.Precision.HIGHEST`` to prevent TPU MXU ``bfloat16`` truncation.
    """
    ndim = g_grid.ndim
    ax = axis % ndim
    if g_grid.shape[ax] != lag.nj:
        raise ValueError(
            f"Expected axis {axis} to have length nj={lag.nj}, got shape {g_grid.shape}."
        )
    mat = lag.to_spectral.astype(jnp.real(jnp.empty((), dtype=g_grid.dtype)).dtype)
    return _contract_laguerre_axis(g_grid, mat, ax)


def _hankel_asymptotic(nu: int, ax: jax.Array, n_pairs: int = 10) -> jax.Array:
    """Hankel's asymptotic expansion (DLMF 10.17.3) for ``J_nu(ax)`` with ``ax >= 24``."""
    w_dtype = ax.dtype
    mu = jnp.asarray(4.0 * float(nu * nu), dtype=w_dtype)
    inv_8x = jnp.asarray(1.0, dtype=w_dtype) / (jnp.asarray(8.0, dtype=w_dtype) * ax)
    t = jnp.ones_like(ax, dtype=w_dtype)
    p_sum = jnp.ones_like(ax, dtype=w_dtype)
    q_sum = jnp.zeros_like(ax, dtype=w_dtype)
    for m in range(n_pairs):
        k1 = 2 * m + 1
        c1 = jnp.asarray(
            (4.0 * float(nu * nu) - float((2 * k1 - 1) ** 2)) / float(k1), dtype=w_dtype
        )
        t = t * c1 * inv_8x
        sign = jnp.asarray(-1.0 if (m % 2 == 1) else 1.0, dtype=w_dtype)
        q_sum = q_sum + sign * t

        k2 = 2 * m + 2
        c2 = jnp.asarray(
            (4.0 * float(nu * nu) - float((2 * k2 - 1) ** 2)) / float(k2), dtype=w_dtype
        )
        t = t * c2 * inv_8x
        sign_p = jnp.asarray(-1.0 if ((m + 1) % 2 == 1) else 1.0, dtype=w_dtype)
        p_sum = p_sum + sign_p * t

    del mu
    chi = ax - jnp.asarray((0.5 * float(nu) + 0.25) * math.pi, dtype=w_dtype)
    amp = jnp.sqrt(jnp.asarray(2.0 / math.pi, dtype=w_dtype) / ax)
    return amp * (p_sum * jnp.cos(chi) - q_sum * jnp.sin(chi))


def bessel_j0(x: jax.Array) -> jax.Array:
    """Pure-JAX Bessel function ``J_0(x)`` in ``x``'s floating dtype (FP64 on host, FP32 on TPU).

    Combines:
    - Convergent power series for ``|x| <= 4`` (condition number < 3.5).
    - 32-point Jacobi-Anger spectral midpoint rule on ``[0, pi]`` for ``4 < |x| <= 25``
      (aliasing error ``2 |J_64(x)| < 3e-19``, zero cancellation).
    - Hankel asymptotic expansion for ``|x| > 25``.
    """
    orig_dtype = jnp.result_type(x, jnp.float32)
    xw = jnp.abs(jnp.asarray(x, dtype=orig_dtype))
    n_terms = 19 if orig_dtype == jnp.float64 else 12
    n_pairs = 10 if orig_dtype == jnp.float64 else 6

    # 1. Small-|x| power series: sum_{k=0}^{n_terms-1} (-x^2 / 4)^k / (k!)^2
    z = jnp.asarray(-0.25, dtype=orig_dtype) * (xw * xw)
    term = jnp.ones_like(xw)
    s_small = term
    for k in range(1, n_terms):
        term = term * z * jnp.asarray(1.0 / float(k * k), dtype=orig_dtype)
        s_small = s_small + term

    # 2. Moderate-|x| 32-point spectral midpoint quadrature of
    #    (1/pi) int_0^pi cos(x sin theta) dtheta
    theta = (jnp.arange(32, dtype=orig_dtype) + jnp.asarray(0.5, dtype=orig_dtype)) * jnp.asarray(
        math.pi / 32.0, dtype=orig_dtype
    )
    sin_th = jnp.sin(theta)
    s_mid = jnp.mean(jnp.cos(xw[..., None] * sin_th), axis=-1)

    # 3. Large-|x| Hankel asymptotic expansion (guard x >= 25 to avoid division by 0)
    c25 = jnp.asarray(25.0, dtype=orig_dtype)
    c4 = jnp.asarray(4.0, dtype=orig_dtype)
    x_safe = jnp.where(xw > c25, xw, c25)
    s_large = _hankel_asymptotic(0, x_safe, n_pairs=n_pairs)

    out = jnp.where(xw <= c4, s_small, jnp.where(xw <= c25, s_mid, s_large))
    return out.astype(orig_dtype)


def bessel_j1(x: jax.Array) -> jax.Array:
    """Pure-JAX Bessel function ``J_1(x)`` in ``x``'s floating dtype (FP64 on host, FP32 on TPU).

    Combines:
    - Convergent odd power series ``(x/2) sum_{k=0}^{n_terms-1} (-x^2/4)^k / (k! (k+1)!)``
      for ``|x| <= 4``.
    - 32-point Jacobi-Anger spectral midpoint rule ``mean(sin(theta) * sin(x sin(theta)))``
      for ``4 < |x| <= 25`` (aliasing error ``|J_63(x)| < 6e-19``).
    - Hankel asymptotic expansion for ``|x| > 25``.
    """
    orig_dtype = jnp.result_type(x, jnp.float32)
    x_in = jnp.asarray(x, dtype=orig_dtype)
    ax = jnp.abs(x_in)
    sgn = jnp.where(
        x_in < jnp.asarray(0.0, dtype=orig_dtype),
        jnp.asarray(-1.0, dtype=orig_dtype),
        jnp.asarray(1.0, dtype=orig_dtype),
    )
    n_terms = 19 if orig_dtype == jnp.float64 else 12
    n_pairs = 10 if orig_dtype == jnp.float64 else 6

    # 1. Small-|x| power series
    z = jnp.asarray(-0.25, dtype=orig_dtype) * (ax * ax)
    term = jnp.full_like(ax, jnp.asarray(0.5, dtype=orig_dtype))
    s_over_x = term
    for k in range(1, n_terms):
        term = term * z * jnp.asarray(1.0 / float(k * (k + 1)), dtype=orig_dtype)
        s_over_x = s_over_x + term
    s_small = ax * s_over_x

    # 2. Moderate-|x| 32-point spectral midpoint quadrature
    theta = (jnp.arange(32, dtype=orig_dtype) + jnp.asarray(0.5, dtype=orig_dtype)) * jnp.asarray(
        math.pi / 32.0, dtype=orig_dtype
    )
    sin_th = jnp.sin(theta)
    s_mid = jnp.mean(sin_th * jnp.sin(ax[..., None] * sin_th), axis=-1)

    # 3. Large-|x| Hankel asymptotic expansion
    c25 = jnp.asarray(25.0, dtype=orig_dtype)
    c4 = jnp.asarray(4.0, dtype=orig_dtype)
    x_safe = jnp.where(ax > c25, ax, c25)
    s_large = _hankel_asymptotic(1, x_safe, n_pairs=n_pairs)

    out = sgn * jnp.where(ax <= c4, s_small, jnp.where(ax <= c25, s_mid, s_large))
    return out.astype(orig_dtype)


def j1_over_alpha(alpha: jax.Array) -> jax.Array:
    """Evaluate ``J_1(alpha) / alpha`` smoothly across ``alpha = 0`` (limit ``0.5``).

    Matches ``device_funcs.cu:L928`` while retaining full Taylor accuracy near ``alpha = 0``.
    """
    orig_dtype = jnp.result_type(alpha, jnp.float32)
    aw = jnp.abs(jnp.asarray(alpha, dtype=orig_dtype))
    n_terms = 19 if orig_dtype == jnp.float64 else 12
    z = jnp.asarray(-0.25, dtype=orig_dtype) * (aw * aw)
    term = jnp.full_like(aw, jnp.asarray(0.5, dtype=orig_dtype))
    series = term
    for k in range(1, n_terms):
        term = term * z * jnp.asarray(1.0 / float(k * (k + 1)), dtype=orig_dtype)
        series = series + term
    c4 = jnp.asarray(4.0, dtype=orig_dtype)
    safe_a = jnp.where(aw > c4, aw, jnp.asarray(1.0, dtype=orig_dtype))
    ratio = bessel_j1(safe_a) / safe_a
    return jnp.where(aw <= c4, series, ratio).astype(orig_dtype)


def g0(b: jax.Array) -> jax.Array:
    """Scaled modified Bessel ``Gamma_0(b) = exp(-b) I_0(b)`` (``device_funcs.cu:L156-189``)."""
    orig_dtype = jnp.result_type(b, jnp.float32)
    bw = jnp.maximum(jnp.asarray(b, dtype=orig_dtype), jnp.asarray(0.0, dtype=orig_dtype))
    return jax.scipy.special.i0e(bw).astype(orig_dtype)


def g1(b: jax.Array) -> jax.Array:
    """Scaled modified Bessel ``Gamma_1(b) = exp(-b) I_1(b)`` (``device_funcs.cu:L192-220``)."""
    orig_dtype = jnp.result_type(b, jnp.float32)
    bw = jnp.maximum(jnp.asarray(b, dtype=orig_dtype), jnp.asarray(0.0, dtype=orig_dtype))
    return jax.scipy.special.i1e(bw).astype(orig_dtype)


def jflr(
    ell: int | jax.Array,
    b: jax.Array,
    *,
    nl: int | None = None,
    enforce_jl_0: bool = False,
) -> jax.Array:
    """Laguerre-Bessel ``J_l(b) = (-b/2)^l / l! * exp(-b/2)`` (``device_funcs.cu:L127-139``).

    Evaluated in ``b``'s floating dtype (IEEE FP64 during host precomputation, pure FP32 on TPU
    device) with log-space evaluation for ``b > 8`` to avoid FP32 intermediate power overflow.
    """
    orig_dtype = jnp.result_type(b, jnp.float32)
    bw = jnp.asarray(b, dtype=orig_dtype)
    ell_arr = jnp.asarray(ell, dtype=jnp.int32)
    inv_fact_table = jnp.asarray(_INV_FACTORIALS_TUPLE, dtype=orig_dtype)
    log_fact_table = jnp.asarray(_LOG_FACTORIALS_TUPLE, dtype=orig_dtype)

    valid = (ell_arr >= 0) & (ell_arr <= 30)
    if enforce_jl_0 and nl is not None:
        valid = valid & (ell_arr < nl)

    ell_clamped = jnp.clip(ell_arr, 0, 30)
    ell_f = ell_clamped.astype(orig_dtype)
    inv_f = inv_fact_table[ell_clamped]
    log_f = log_fact_table[ell_clamped]

    c8 = jnp.asarray(8.0, dtype=orig_dtype)
    c_half = jnp.asarray(0.5, dtype=orig_dtype)
    b_small = jnp.clip(bw, jnp.asarray(-8.0, dtype=orig_dtype), c8)
    val_small = ((-c_half * b_small) ** ell_f) * inv_f * jnp.exp(-c_half * b_small)

    half_b_large = c_half * jnp.maximum(bw, c8)
    sign_l = jnp.where(
        (ell_clamped & jnp.int32(1)) == jnp.int32(1),
        jnp.asarray(-1.0, dtype=orig_dtype),
        jnp.asarray(1.0, dtype=orig_dtype),
    )
    val_large = sign_l * jnp.exp(ell_f * jnp.log(half_b_large) - log_f - half_b_large)
    val = jnp.where(bw <= c8, val_small, val_large)
    out = jnp.where(valid, val, jnp.asarray(0.0, dtype=orig_dtype))
    return out.astype(orig_dtype)


def jflr_a(ell: int | jax.Array, b: jax.Array) -> jax.Array:
    """Magnetic-drift FLR combination ``JflrA(l, b)`` (``device_funcs.cu:L141-144``)."""
    orig_dtype = jnp.result_type(b, jnp.float32)
    ell_arr = jnp.asarray(ell, dtype=jnp.int32)
    ell_f = ell_arr.astype(orig_dtype)
    two = jnp.asarray(2.0, dtype=orig_dtype)
    one = jnp.asarray(1.0, dtype=orig_dtype)
    return (
        ell_f * jflr(ell_arr - 1, b)
        + two * ell_f * jflr(ell_arr, b)
        + (ell_f + one) * jflr(ell_arr + 1, b)
    )


def jflr_b(
    ell: int | jax.Array,
    b: jax.Array,
    *,
    nl: int | None = None,
    enforce_jl_0: bool = False,
) -> jax.Array:
    """Compressional ``B_par`` FLR kernel ``JflrB(l, b) = J_l(b) + J_{l-1}(b)`` (``L146-149``)."""
    ell_arr = jnp.asarray(ell, dtype=jnp.int32)
    val = jflr(ell_arr, b, nl=nl, enforce_jl_0=enforce_jl_0) + jflr(
        ell_arr - 1, b, nl=nl, enforce_jl_0=enforce_jl_0
    )
    if enforce_jl_0 and nl is not None:
        val = jnp.where(ell_arr >= nl, jnp.zeros_like(val), val)
    return val


def jfac(ell: int | jax.Array, b: jax.Array) -> jax.Array:
    """Curvature/grad-B FLR factor ``Jfac(l, b) = 1.5 * Jflr(l, b) + JflrA(l, b)`` (``L151``)."""
    orig_dtype = jnp.result_type(b, jnp.float32)
    return jnp.asarray(1.5, dtype=orig_dtype) * jflr(ell, b) + jflr_a(ell, b)


def j0_to_grid(
    field: jax.Array,
    kperp2: jax.Array,
    lag: LaguerreTransform,
    rho2_s: float | jax.Array,
    fac: float | jax.Array = 1.0,
    active_mask: jax.Array | None = None,
) -> jax.Array:
    """Evaluate ``J_0(sqrt(2 * muB_j * kperp2 * rho2_s)) * field * fac`` on ``(nj, nz, ky, kx)``.

    Ported from ``J0fToGrid`` (``device_funcs.cu:L894-909``).
    """
    real_dtype = jnp.real(jnp.empty((), dtype=field.dtype)).dtype
    mu_b = jnp.asarray(lag.roots, dtype=real_dtype)[:, None, None, None]
    kp2 = jnp.asarray(kperp2, dtype=real_dtype)[None, ...]
    rho2 = jnp.asarray(rho2_s, dtype=real_dtype)
    alpha = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(2.0, dtype=real_dtype) * mu_b * kp2 * rho2,
            jnp.asarray(0.0, dtype=real_dtype),
        )
    )
    j0_val = bessel_j0(alpha).astype(real_dtype)
    fac_val = jnp.asarray(fac, dtype=real_dtype)
    out = j0_val * field[None, ...] * fac_val
    if active_mask is not None:
        out = jnp.where(active_mask[None, None, :, :], out, jnp.zeros_like(out))
    return out


def j0_phi_and_bpar_to_grid(
    phi: jax.Array,
    bpar: jax.Array,
    kperp2: jax.Array,
    lag: LaguerreTransform,
    rho2_s: float | jax.Array,
    tz: float | jax.Array,
    fphi: float | jax.Array = 1.0,
    fbpar: float | jax.Array = 0.0,
    active_mask: jax.Array | None = None,
) -> jax.Array:
    """Evaluate ``<chi> = J_0(alpha) phi fphi + tz * 2 muB (J_1(alpha)/alpha) bpar fbpar``.

    Ported from ``J0phiAndBparToGrid`` (``device_funcs.cu:L911-931``).
    Returns array of shape ``(nj, nz, ky, kx)``.
    """
    real_dtype = jnp.real(jnp.empty((), dtype=phi.dtype)).dtype
    mu_b = jnp.asarray(lag.roots, dtype=real_dtype)[:, None, None, None]
    kp2 = jnp.asarray(kperp2, dtype=real_dtype)[None, ...]
    rho2 = jnp.asarray(rho2_s, dtype=real_dtype)
    two = jnp.asarray(2.0, dtype=real_dtype)
    alpha = jnp.sqrt(jnp.maximum(two * mu_b * kp2 * rho2, jnp.asarray(0.0, dtype=real_dtype)))
    j0_val = bessel_j0(alpha).astype(real_dtype)
    j1_oa = j1_over_alpha(alpha).astype(real_dtype)
    fphi_val = jnp.asarray(fphi, dtype=real_dtype)
    tz_val = jnp.asarray(tz, dtype=real_dtype)
    fbpar_val = jnp.asarray(fbpar, dtype=real_dtype)
    out = (
        j0_val * phi[None, ...] * fphi_val
        + tz_val * two * mu_b * j1_oa * bpar[None, ...] * fbpar_val
    )
    if active_mask is not None:
        out = jnp.where(active_mask[None, None, :, :], out, jnp.zeros_like(out))
    return out
