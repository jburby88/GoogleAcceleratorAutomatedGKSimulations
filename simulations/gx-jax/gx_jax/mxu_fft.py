"""TPU ``v6e`` MXU split-real/imaginary matrix-multiply spectral transforms for GX-JAX.

Implements exact IEEE ``float64``-precomputed Discrete Fourier Transform (DFT / iDFT) and
real-valued Discrete Fourier Transform (rDFT / irDFT) plans executed strictly via split
real/imaginary :func:`jax.lax.dot_general` calls with ``precision=jax.lax.Precision.HIGHEST``.

Key XLA / TPU ``v6e`` invariants:
1. All trigonometric weight tables (``2 * pi * ((k * n) % N) / N`` with exact quarter/half-turn
   cardinal values ``0.0`` and ``+-1.0``) are precomputed on host in IEEE ``numpy.float64`` and
   cast once via ``.astype(r_dtype)`` to runtime ``float32`` (for ``complex64``) or ``float64``
   (for ``complex128``), emitting **zero ``f64`` array ops** in runtime ``complex64`` StableHLO.
2. Every spectral transform lowers directly to real ``stablehlo.dot_general`` instructions with
   ``precision = [HIGHEST, HIGHEST]`` (3-pass IEEE FP32 accumulation on the 256x256 TPU MXU) and
   **zero ``stablehlo.fft`` ops** (completely bypassing VPU Cooley-Tukey / Bluestein FFTs on
   composite ``3/2``-dealiased lengths such as ``96, 108, 144, 192``).
3. Arbitrary ND contractions reshape around the transform axis ``ax`` into
   ``(B_lead, N_in, B_trail)`` (or 2D when ``ax == 0`` or ``ax == ndim - 1``) and use a batched
   ``dot_general`` along ``B_lead``, emitting **zero 6D (and zero 3D) ``stablehlo.transpose``
   ops** on ``G[nspecies, nhermite, nlaguerre, nz, naky, nakx]`` and real-space brackets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

MXU_AUTO_MAX_LEN: int = 256
VALID_FFT_BACKENDS: tuple[str, ...] = ("auto", "fft", "mxu")
_VALID_NORMS: tuple[str | None, ...] = (None, "backward", "forward", "ortho")


def _real_dtype_from_target(dtype: jnp.dtype | type) -> jnp.dtype:
    """Resolve ``float64`` vs ``float32`` storage dtype from a real or complex target dtype."""
    dt = jnp.dtype(dtype)
    if dt in (jnp.float64, jnp.complex128):
        return jnp.dtype(jnp.float64)
    return jnp.dtype(jnp.float32)


def _complex_and_real_dtypes_arr(x: jax.Array) -> tuple[jnp.dtype, jnp.dtype]:
    """Return matching ``(complex_dtype, real_dtype)`` pair for array ``x``."""
    dt = jnp.dtype(x.dtype)
    if dt in (jnp.float64, jnp.complex128):
        return jnp.dtype(jnp.complex128), jnp.dtype(jnp.float64)
    if dt in (jnp.float32, jnp.complex64):
        return jnp.dtype(jnp.complex64), jnp.dtype(jnp.float32)
    # Integer / bool / weak inputs promote according to x64 configuration
    promoted = jnp.result_type(dt, jnp.float32)
    if promoted == jnp.float64:
        return jnp.dtype(jnp.complex128), jnp.dtype(jnp.float64)
    return jnp.dtype(jnp.complex64), jnp.dtype(jnp.float32)


def validate_fft_backend(fft_backend: str) -> str:
    """Validate and normalize ``fft_backend`` in ``{'auto', 'fft', 'mxu'}``."""
    if not isinstance(fft_backend, str):
        raise ValueError(
            f"Unsupported fft_backend={fft_backend!r}. Expected one of {VALID_FFT_BACKENDS}."
        )
    cleaned = fft_backend.strip().lower()
    if cleaned not in VALID_FFT_BACKENDS:
        raise ValueError(
            f"Unsupported fft_backend={fft_backend!r}. Expected one of {VALID_FFT_BACKENDS}."
        )
    return cleaned


def resolve_fft_backend(
    fft_backend: str = "auto",
    *,
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    max_chain_len: int = 1,
    backend: str | None = None,
) -> str:
    """Resolve concrete spectral backend (``'fft'`` or ``'mxu'``).

    - Validates ``fft_backend`` in ``{'auto', 'fft', 'mxu'}``.
    - Explicit ``'mxu'`` or ``'fft'`` is honored on any device backend (CPU, GPU, TPU) for any
      positive transform lengths.
    - ``'auto'`` selects ``'mxu'`` when ``(backend or jax.default_backend()) == 'tpu'`` and
      ``max(nx, ny, nz, max_chain_len) <= 256``, and ``'fft'`` otherwise.
    """
    fb = validate_fft_backend(fft_backend)
    dims = (int(nx), int(ny), int(nz), int(max_chain_len))
    if min(dims) < 1:
        raise ValueError(
            f"Transform lengths must be positive, got (nx={nx}, ny={ny}, nz={nz}, "
            f"max_chain_len={max_chain_len})."
        )
    if fb in ("fft", "mxu"):
        return fb

    active_backend = (jax.default_backend() if backend is None else str(backend)).strip().lower()
    if active_backend == "tpu" and max(dims) <= MXU_AUTO_MAX_LEN:
        return "mxu"
    return "fft"


def _validate_norm(norm: str | None) -> str:
    if norm not in _VALID_NORMS:
        raise ValueError(
            f"Unsupported FFT norm={norm!r}. Expected None, 'backward', 'forward', or 'ortho'."
        )
    return "backward" if norm is None else norm


def _dft_cos_sin_np(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact IEEE ``float64`` ``cos(2*pi*k*m/n)`` and ``sin(2*pi*k*m/n)`` on host.

    Reduces ``(k * m) % n`` in exact 64-bit integer arithmetic before multiplying by ``2*pi/n``
    and pins cardinal quarter/half-turn multiples (``0, pi/2, pi, 3*pi/2``) to exact ``0.0`` and
    ``+-1.0`` so DC and even-``n`` Nyquist imaginary weights are identically ``0.0``.
    """
    if n < 1:
        raise ValueError(f"Transform length n={n} must be >= 1.")
    idx = np.arange(n, dtype=np.int64)
    rem = (idx[:, None] * idx[None, :]) % int(n)
    theta = (2.0 * np.pi / float(n)) * rem.astype(np.float64)
    cos_m = np.cos(theta)
    sin_m = np.sin(theta)

    cos_m[rem == 0] = 1.0
    sin_m[rem == 0] = 0.0
    if n % 2 == 0:
        half = rem == (n // 2)
        cos_m[half] = -1.0
        sin_m[half] = 0.0
    if n % 4 == 0:
        q1 = rem == (n // 4)
        q3 = rem == (3 * (n // 4))
        cos_m[q1] = 0.0
        sin_m[q1] = 1.0
        cos_m[q3] = 0.0
        sin_m[q3] = -1.0
    return cos_m, sin_m


def _rdft_cos_sin_np(
    n: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Precompute host ``float64`` rFFT and irFFT cosine/sine weight tables for length ``n``."""
    if n < 1:
        raise ValueError(f"Real transform length n={n} must be >= 1.")
    n_freq = n // 2 + 1
    cos_full, sin_full = _dft_cos_sin_np(n)
    inv_n = 1.0 / float(n)

    # rfft: shape (n_freq, n)
    rfft_cos = np.ascontiguousarray(cos_full[:n_freq, :], dtype=np.float64)
    rfft_sin = np.ascontiguousarray(sin_full[:n_freq, :], dtype=np.float64)
    rfft_cos_inv = rfft_cos * inv_n
    rfft_sin_inv = rfft_sin * inv_n

    # irfft: shape (n, n_freq). Double interior frequencies 1 .. (n - 1) // 2 while keeping
    # k = 0 and (when n is even) Nyquist k = n // 2 at weight 1 with exact 0.0 sine column.
    irfft_cos_f = np.ascontiguousarray(cos_full[:, :n_freq], dtype=np.float64).copy()
    irfft_sin_f = np.ascontiguousarray(sin_full[:, :n_freq], dtype=np.float64).copy()
    last_interior = (n - 1) // 2
    if last_interior >= 1:
        irfft_cos_f[:, 1 : last_interior + 1] *= 2.0
        irfft_sin_f[:, 1 : last_interior + 1] *= 2.0
    irfft_sin_f[:, 0] = 0.0
    if n % 2 == 0:
        irfft_sin_f[:, n // 2] = 0.0

    irfft_cos_inv = irfft_cos_f * inv_n
    irfft_sin_inv = irfft_sin_f * inv_n

    return (
        rfft_cos,
        rfft_sin,
        rfft_cos_inv,
        rfft_sin_inv,
        irfft_cos_f,
        irfft_sin_f,
        irfft_cos_inv,
        irfft_sin_inv,
    )


def _cast_plan_leaf(x: jax.Array, r_dtype: jnp.dtype) -> jax.Array:
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return x
    if isinstance(x, jax.core.Tracer):
        return x.astype(r_dtype)
    return jnp.asarray(np.asarray(x), dtype=r_dtype)


# ---------------------------------------------------------------------------
# PyTree-Registered Precomputed MXU Spectral Plans
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MXU1DPlan:
    """Precomputed 1D complex DFT/iDFT split-real/imag weight matrices of length ``n``."""

    cos_f: jax.Array  # (n, n) unnormalized cos(2*pi*k*m/n)
    sin_f: jax.Array  # (n, n) unnormalized sin(2*pi*k*m/n)
    cos_inv: jax.Array  # (n, n) (1/n) * cos(2*pi*k*m/n)
    sin_inv: jax.Array  # (n, n) (1/n) * sin(2*pi*k*m/n)
    n: int

    def tree_flatten(self):
        return (self.cos_f, self.sin_f, self.cos_inv, self.sin_inv), (self.n,)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> MXU1DPlan:
        (n,) = aux_data
        cos_f, sin_f, cos_inv, sin_inv = children
        return cls(cos_f=cos_f, sin_f=sin_f, cos_inv=cos_inv, sin_inv=sin_inv, n=n)

    def astype(self, dtype: jnp.dtype | type) -> MXU1DPlan:
        """Cast weight matrices once to runtime ``float32`` or ``float64``."""
        r_dtype = _real_dtype_from_target(dtype)
        return jax.tree_util.tree_map(lambda leaf: _cast_plan_leaf(leaf, r_dtype), self)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MXUReal1DPlan:
    """Precomputed 1D real rDFT/irDFT split-real/imag weight matrices of length ``n``."""

    rfft_cos: jax.Array  # (n_freq, n)
    rfft_sin: jax.Array  # (n_freq, n)
    rfft_cos_inv: jax.Array  # (n_freq, n) scaled by 1/n
    rfft_sin_inv: jax.Array  # (n_freq, n) scaled by 1/n
    irfft_cos_f: jax.Array  # (n, n_freq) Hermitian-weighted unnormalized
    irfft_sin_f: jax.Array  # (n, n_freq) Hermitian-weighted unnormalized (0 at DC & even Nyquist)
    irfft_cos_inv: jax.Array  # (n, n_freq) Hermitian-weighted scaled by 1/n
    irfft_sin_inv: jax.Array  # (n, n_freq) Hermitian-weighted scaled by 1/n
    n: int
    n_freq: int

    def tree_flatten(self):
        children = (
            self.rfft_cos,
            self.rfft_sin,
            self.rfft_cos_inv,
            self.rfft_sin_inv,
            self.irfft_cos_f,
            self.irfft_sin_f,
            self.irfft_cos_inv,
            self.irfft_sin_inv,
        )
        return children, (self.n, self.n_freq)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> MXUReal1DPlan:
        n, n_freq = aux_data
        (
            rfft_cos,
            rfft_sin,
            rfft_cos_inv,
            rfft_sin_inv,
            irfft_cos_f,
            irfft_sin_f,
            irfft_cos_inv,
            irfft_sin_inv,
        ) = children
        return cls(
            rfft_cos=rfft_cos,
            rfft_sin=rfft_sin,
            rfft_cos_inv=rfft_cos_inv,
            rfft_sin_inv=rfft_sin_inv,
            irfft_cos_f=irfft_cos_f,
            irfft_sin_f=irfft_sin_f,
            irfft_cos_inv=irfft_cos_inv,
            irfft_sin_inv=irfft_sin_inv,
            n=n,
            n_freq=n_freq,
        )

    def astype(self, dtype: jnp.dtype | type) -> MXUReal1DPlan:
        """Cast weight matrices once to runtime ``float32`` or ``float64``."""
        r_dtype = _real_dtype_from_target(dtype)
        return jax.tree_util.tree_map(lambda leaf: _cast_plan_leaf(leaf, r_dtype), self)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MXU2DPlan:
    """Precomputed 2D perpendicular ``(nx, ny)`` spectral plan for :class:`GradPerp`."""

    plan_x: MXU1DPlan
    plan_y_real: MXUReal1DPlan
    nx: int
    ny: int
    nyc: int

    def tree_flatten(self):
        return (self.plan_x, self.plan_y_real), (self.nx, self.ny, self.nyc)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> MXU2DPlan:
        nx, ny, nyc = aux_data
        plan_x, plan_y_real = children
        return cls(plan_x=plan_x, plan_y_real=plan_y_real, nx=nx, ny=ny, nyc=nyc)

    def astype(self, dtype: jnp.dtype | type) -> MXU2DPlan:
        """Cast 2D MXU weight matrices once to runtime ``float32`` or ``float64``."""
        r_dtype = _real_dtype_from_target(dtype)
        return jax.tree_util.tree_map(lambda leaf: _cast_plan_leaf(leaf, r_dtype), self)


def init_mxu_1d_plan(n: int, *, dtype: jnp.dtype | type = jnp.float64) -> MXU1DPlan:
    """Precompute 1D DFT/iDFT weight matrices in IEEE ``float64`` and cast to ``dtype``."""
    n_int = int(n)
    r_dtype = _real_dtype_from_target(dtype)
    cos_f, sin_f = _dft_cos_sin_np(n_int)
    inv_n = 1.0 / float(n_int)
    cos_inv = cos_f * inv_n
    sin_inv = sin_f * inv_n
    return MXU1DPlan(
        cos_f=jnp.asarray(cos_f, dtype=r_dtype),
        sin_f=jnp.asarray(sin_f, dtype=r_dtype),
        cos_inv=jnp.asarray(cos_inv, dtype=r_dtype),
        sin_inv=jnp.asarray(sin_inv, dtype=r_dtype),
        n=n_int,
    )


def init_mxu_real_1d_plan(n: int, *, dtype: jnp.dtype | type = jnp.float64) -> MXUReal1DPlan:
    """Precompute 1D real rFFT/irFFT weight matrices in IEEE ``float64`` and cast to ``dtype``."""
    n_int = int(n)
    n_freq = n_int // 2 + 1
    r_dtype = _real_dtype_from_target(dtype)
    (
        rfft_cos,
        rfft_sin,
        rfft_cos_inv,
        rfft_sin_inv,
        irfft_cos_f,
        irfft_sin_f,
        irfft_cos_inv,
        irfft_sin_inv,
    ) = _rdft_cos_sin_np(n_int)
    return MXUReal1DPlan(
        rfft_cos=jnp.asarray(rfft_cos, dtype=r_dtype),
        rfft_sin=jnp.asarray(rfft_sin, dtype=r_dtype),
        rfft_cos_inv=jnp.asarray(rfft_cos_inv, dtype=r_dtype),
        rfft_sin_inv=jnp.asarray(rfft_sin_inv, dtype=r_dtype),
        irfft_cos_f=jnp.asarray(irfft_cos_f, dtype=r_dtype),
        irfft_sin_f=jnp.asarray(irfft_sin_f, dtype=r_dtype),
        irfft_cos_inv=jnp.asarray(irfft_cos_inv, dtype=r_dtype),
        irfft_sin_inv=jnp.asarray(irfft_sin_inv, dtype=r_dtype),
        n=n_int,
        n_freq=n_freq,
    )


def init_mxu_2d_plan(nx: int, ny: int, *, dtype: jnp.dtype | type = jnp.float64) -> MXU2DPlan:
    """Precompute 2D ``(nx, ny)`` MXU spectral plan in IEEE ``float64`` and cast to ``dtype``."""
    nx_int = int(nx)
    ny_int = int(ny)
    r_dtype = _real_dtype_from_target(dtype)
    return MXU2DPlan(
        plan_x=init_mxu_1d_plan(nx_int, dtype=r_dtype),
        plan_y_real=init_mxu_real_1d_plan(ny_int, dtype=r_dtype),
        nx=nx_int,
        ny=ny_int,
        nyc=ny_int // 2 + 1,
    )


# ---------------------------------------------------------------------------
# Transpose-Free Split-Real/Imaginary MXU Contraction Primitives
# ---------------------------------------------------------------------------


def _pad_or_truncate_axis(x: jax.Array, *, target_len: int, axis: int) -> jax.Array:
    """Match ``jnp.fft`` semantics when ``target_len != x.shape[axis]``."""
    ax = axis % x.ndim
    cur_len = x.shape[ax]
    if cur_len == target_len:
        return x
    if target_len < cur_len:
        slc = [slice(None)] * x.ndim
        slc[ax] = slice(0, target_len)
        return x[tuple(slc)]
    pad_spec = [(0, 0)] * x.ndim
    pad_spec[ax] = (0, target_len - cur_len)
    return jnp.pad(x, pad_spec)


def _contract_real_axis(x: jax.Array, w: jax.Array, axis: int) -> jax.Array:
    """Contract real matrix ``w (M_out, N_in)`` along ``axis`` of real array ``x``.

    Reshapes ``x`` around ``axis`` into 2D (boundary axes) or 3D ``(B_lead, N_in, B_trail)``
    (interior axes, e.g. ``axis=-3`` on 6D ``G``) and executes a batched ``dot_general`` with
    ``precision=jax.lax.Precision.HIGHEST`` so XLA emits **zero ``stablehlo.transpose`` ops**.
    """
    ndim = x.ndim
    if ndim < 1:
        raise ValueError("Spectral transform requires at least a 1D input array.")
    ax = axis % ndim
    m_out, n_in = w.shape
    if x.shape[ax] != n_in:
        raise ValueError(
            f"Axis {axis} of shape {x.shape} does not match weight matrix input length {n_in}."
        )

    lead_shape = x.shape[:ax]
    trail_shape = x.shape[ax + 1 :]
    b_lead = math.prod(lead_shape) if lead_shape else 1
    b_trail = math.prod(trail_shape) if trail_shape else 1

    if b_lead == 0 or b_trail == 0 or m_out == 0:
        return jnp.zeros((*lead_shape, m_out, *trail_shape), dtype=x.dtype)

    if ndim == 1:
        return jax.lax.dot_general(
            w,
            x,
            dimension_numbers=(((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
        )

    if ax == 0:
        x2 = x.reshape(n_in, b_trail)
        y2 = jax.lax.dot_general(
            w,
            x2,
            dimension_numbers=(((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
        )
        return y2.reshape(m_out, *trail_shape)

    if ax == ndim - 1:
        x2 = x.reshape(b_lead, n_in)
        y2 = jax.lax.dot_general(
            x2,
            w,
            dimension_numbers=(((1,), (1,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
        )
        return y2.reshape(*lead_shape, m_out)

    x3 = x.reshape(b_lead, n_in, b_trail)
    w_bcast = jax.lax.broadcast_in_dim(w, (b_lead, m_out, n_in), broadcast_dimensions=(1, 2))
    y3 = jax.lax.dot_general(
        w_bcast,
        x3,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
    )
    return y3.reshape(*lead_shape, m_out, *trail_shape)


def _resolve_1d_weights(
    n_target: int,
    r_dtype: jnp.dtype,
    norm_mode: str,
    *,
    inverse: bool,
    plan: MXU1DPlan | None,
) -> tuple[jax.Array, jax.Array]:
    p = init_mxu_1d_plan(n_target, dtype=r_dtype) if (plan is None or plan.n != n_target) else plan
    if (not inverse and norm_mode == "backward") or (inverse and norm_mode == "forward"):
        return p.cos_f.astype(r_dtype), p.sin_f.astype(r_dtype)
    if (not inverse and norm_mode == "forward") or (inverse and norm_mode == "backward"):
        return p.cos_inv.astype(r_dtype), p.sin_inv.astype(r_dtype)
    scale = jnp.asarray(1.0 / math.sqrt(float(n_target)), dtype=r_dtype)
    return p.cos_f.astype(r_dtype) * scale, p.sin_f.astype(r_dtype) * scale


def mxu_fft(
    x: jax.Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    plan: MXU1DPlan | None = None,
) -> jax.Array:
    """1D forward DFT via split-real/imag MXU ``dot_general`` matching ``jnp.fft.fft``."""
    norm_mode = _validate_norm(norm)
    x_arr = jnp.asarray(x)
    ax = axis % x_arr.ndim
    n_target = int(x_arr.shape[ax] if n is None else n)
    if n_target < 1:
        raise ValueError(f"Transform length n={n_target} must be >= 1.")

    c_dtype, r_dtype = _complex_and_real_dtypes_arr(x_arr)
    cos_w, sin_w = _resolve_1d_weights(n_target, r_dtype, norm_mode, inverse=False, plan=plan)

    if jnp.issubdtype(x_arr.dtype, jnp.complexfloating):
        x_c = _pad_or_truncate_axis(x_arr.astype(c_dtype), target_len=n_target, axis=ax)
        x_re = jnp.real(x_c)
        x_im = jnp.imag(x_c)
        out_re = _contract_real_axis(x_re, cos_w, ax) + _contract_real_axis(x_im, sin_w, ax)
        out_im = _contract_real_axis(x_im, cos_w, ax) - _contract_real_axis(x_re, sin_w, ax)
        return jax.lax.complex(out_re, out_im)

    x_r = _pad_or_truncate_axis(x_arr.astype(r_dtype), target_len=n_target, axis=ax)
    out_re = _contract_real_axis(x_r, cos_w, ax)
    out_im = -_contract_real_axis(x_r, sin_w, ax)
    return jax.lax.complex(out_re, out_im)


def mxu_ifft(
    x: jax.Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    plan: MXU1DPlan | None = None,
) -> jax.Array:
    """1D inverse DFT via split-real/imag MXU ``dot_general`` matching ``jnp.fft.ifft``."""
    norm_mode = _validate_norm(norm)
    x_arr = jnp.asarray(x)
    ax = axis % x_arr.ndim
    n_target = int(x_arr.shape[ax] if n is None else n)
    if n_target < 1:
        raise ValueError(f"Transform length n={n_target} must be >= 1.")

    c_dtype, r_dtype = _complex_and_real_dtypes_arr(x_arr)
    cos_w, sin_w = _resolve_1d_weights(n_target, r_dtype, norm_mode, inverse=True, plan=plan)

    if jnp.issubdtype(x_arr.dtype, jnp.complexfloating):
        x_c = _pad_or_truncate_axis(x_arr.astype(c_dtype), target_len=n_target, axis=ax)
        x_re = jnp.real(x_c)
        x_im = jnp.imag(x_c)
        out_re = _contract_real_axis(x_re, cos_w, ax) - _contract_real_axis(x_im, sin_w, ax)
        out_im = _contract_real_axis(x_im, cos_w, ax) + _contract_real_axis(x_re, sin_w, ax)
        return jax.lax.complex(out_re, out_im)

    x_r = _pad_or_truncate_axis(x_arr.astype(r_dtype), target_len=n_target, axis=ax)
    out_re = _contract_real_axis(x_r, cos_w, ax)
    out_im = _contract_real_axis(x_r, sin_w, ax)
    return jax.lax.complex(out_re, out_im)


def mxu_rfft(
    x: jax.Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    plan: MXUReal1DPlan | None = None,
) -> jax.Array:
    """1D real-to-complex DFT via MXU ``dot_general`` matching ``jnp.fft.rfft``."""
    norm_mode = _validate_norm(norm)
    x_arr = jnp.asarray(x)
    if jnp.issubdtype(x_arr.dtype, jnp.complexfloating):
        raise TypeError(f"mxu_rfft requires a real-valued input array, got dtype {x_arr.dtype}.")

    ax = axis % x_arr.ndim
    n_target = int(x_arr.shape[ax] if n is None else n)
    if n_target < 1:
        raise ValueError(f"Real transform length n={n_target} must be >= 1.")

    _, r_dtype = _complex_and_real_dtypes_arr(x_arr)
    p = (
        init_mxu_real_1d_plan(n_target, dtype=r_dtype)
        if (plan is None or plan.n != n_target)
        else plan
    )
    if norm_mode == "backward":
        cos_w = p.rfft_cos.astype(r_dtype)
        sin_w = p.rfft_sin.astype(r_dtype)
    elif norm_mode == "forward":
        cos_w = p.rfft_cos_inv.astype(r_dtype)
        sin_w = p.rfft_sin_inv.astype(r_dtype)
    else:
        scale = jnp.asarray(1.0 / math.sqrt(float(n_target)), dtype=r_dtype)
        cos_w = p.rfft_cos.astype(r_dtype) * scale
        sin_w = p.rfft_sin.astype(r_dtype) * scale

    x_r = _pad_or_truncate_axis(x_arr.astype(r_dtype), target_len=n_target, axis=ax)
    out_re = _contract_real_axis(x_r, cos_w, ax)
    out_im = -_contract_real_axis(x_r, sin_w, ax)
    return jax.lax.complex(out_re, out_im)


def mxu_irfft(
    x: jax.Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    plan: MXUReal1DPlan | None = None,
) -> jax.Array:
    """1D Hermitian-to-real iDFT via MXU ``dot_general`` matching ``jnp.fft.irfft``.

    Doubles interior modes ``1 .. (n - 1) // 2`` while keeping ``k = 0`` and even-``n`` Nyquist
    ``k = n // 2`` at weight 1 with exact zero imaginary contribution even when ``Im(x)`` at DC
    or Nyquist is non-zero.
    """
    norm_mode = _validate_norm(norm)
    x_arr = jnp.asarray(x)
    ax = axis % x_arr.ndim
    if n is None:
        k_in = x_arr.shape[ax]
        if k_in < 2:
            raise ValueError(
                f"Invalid input frequency axis size {k_in} for mxu_irfft when n is None."
            )
        n_target = 2 * (k_in - 1)
    else:
        n_target = int(n)
    if n_target < 1:
        raise ValueError(f"Real output length n={n_target} must be >= 1.")

    n_freq = n_target // 2 + 1
    c_dtype, r_dtype = _complex_and_real_dtypes_arr(x_arr)
    p = (
        init_mxu_real_1d_plan(n_target, dtype=r_dtype)
        if (plan is None or plan.n != n_target)
        else plan
    )
    if norm_mode == "backward":
        cos_w = p.irfft_cos_inv.astype(r_dtype)
        sin_w = p.irfft_sin_inv.astype(r_dtype)
    elif norm_mode == "forward":
        cos_w = p.irfft_cos_f.astype(r_dtype)
        sin_w = p.irfft_sin_f.astype(r_dtype)
    else:
        scale = jnp.asarray(1.0 / math.sqrt(float(n_target)), dtype=r_dtype)
        cos_w = p.irfft_cos_f.astype(r_dtype) * scale
        sin_w = p.irfft_sin_f.astype(r_dtype) * scale

    if jnp.issubdtype(x_arr.dtype, jnp.complexfloating):
        x_c = _pad_or_truncate_axis(x_arr.astype(c_dtype), target_len=n_freq, axis=ax)
        x_re = jnp.real(x_c)
        x_im = jnp.imag(x_c)
        return _contract_real_axis(x_re, cos_w, ax) - _contract_real_axis(x_im, sin_w, ax)

    x_r = _pad_or_truncate_axis(x_arr.astype(r_dtype), target_len=n_freq, axis=ax)
    return _contract_real_axis(x_r, cos_w, ax)


def mxu_fft2(
    x: jax.Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    plan: tuple[MXU1DPlan, MXU1DPlan] | None = None,
) -> jax.Array:
    """2D complex forward DFT via MXU ``dot_general`` matching ``jnp.fft.fft2``."""
    if len(axes) != 2:
        raise ValueError(f"mxu_fft2 expects 2 axes, got {axes}.")
    n0 = None if s is None else int(s[0])
    n1 = None if s is None else int(s[1])
    p0 = None if plan is None else plan[0]
    p1 = None if plan is None else plan[1]
    y = mxu_fft(x, n=n0, axis=axes[0], norm=norm, plan=p0)
    return mxu_fft(y, n=n1, axis=axes[1], norm=norm, plan=p1)


def mxu_ifft2(
    x: jax.Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    plan: tuple[MXU1DPlan, MXU1DPlan] | None = None,
) -> jax.Array:
    """2D complex inverse DFT via MXU ``dot_general`` matching ``jnp.fft.ifft2``."""
    if len(axes) != 2:
        raise ValueError(f"mxu_ifft2 expects 2 axes, got {axes}.")
    n0 = None if s is None else int(s[0])
    n1 = None if s is None else int(s[1])
    p0 = None if plan is None else plan[0]
    p1 = None if plan is None else plan[1]
    y = mxu_ifft(x, n=n0, axis=axes[0], norm=norm, plan=p0)
    return mxu_ifft(y, n=n1, axis=axes[1], norm=norm, plan=p1)


def mxu_rfftn(
    x: jax.Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    plan: MXU2DPlan | None = None,
) -> jax.Array:
    """ND real-to-complex DFT via MXU ``dot_general`` matching ``jnp.fft.rfftn``."""
    x_arr = jnp.asarray(x)
    if axes is None:
        axes_tuple = tuple(range(-len(s), 0)) if s is not None else tuple(range(x_arr.ndim))
    else:
        axes_tuple = tuple(int(a) for a in axes)
    if not axes_tuple:
        return x_arr
    if s is not None and len(s) != len(axes_tuple):
        raise ValueError(f"Length of s ({len(s)}) must match length of axes ({len(axes_tuple)}).")

    sizes = [
        int(x_arr.shape[ax % x_arr.ndim] if s is None else s[i]) for i, ax in enumerate(axes_tuple)
    ]

    p_real = (
        plan.plan_y_real
        if (plan is not None and len(axes_tuple) == 2 and plan.ny == sizes[-1])
        else None
    )
    out = mxu_rfft(x_arr, n=sizes[-1], axis=axes_tuple[-1], norm=norm, plan=p_real)

    for i in range(len(axes_tuple) - 2, -1, -1):
        p_c = (
            plan.plan_x
            if (plan is not None and len(axes_tuple) == 2 and i == 0 and plan.nx == sizes[0])
            else None
        )
        out = mxu_fft(out, n=sizes[i], axis=axes_tuple[i], norm=norm, plan=p_c)
    return out


def mxu_irfftn(
    x: jax.Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    plan: MXU2DPlan | None = None,
) -> jax.Array:
    """ND Hermitian-to-real iDFT via MXU ``dot_general`` matching ``jnp.fft.irfftn``."""
    x_arr = jnp.asarray(x)
    if axes is None:
        axes_tuple = tuple(range(-len(s), 0)) if s is not None else tuple(range(x_arr.ndim))
    else:
        axes_tuple = tuple(int(a) for a in axes)
    if not axes_tuple:
        return jnp.real(x_arr)
    if s is not None and len(s) != len(axes_tuple):
        raise ValueError(f"Length of s ({len(s)}) must match length of axes ({len(axes_tuple)}).")

    if s is None:
        sizes = [int(x_arr.shape[ax % x_arr.ndim]) for ax in axes_tuple[:-1]]
        k_last = int(x_arr.shape[axes_tuple[-1] % x_arr.ndim])
        if k_last < 2:
            raise ValueError(f"Invalid final axis size {k_last} for mxu_irfftn when s is None.")
        sizes.append(2 * (k_last - 1))
    else:
        sizes = [int(v) for v in s]

    cur = x_arr
    for i in range(len(axes_tuple) - 1):
        p_c = (
            plan.plan_x
            if (plan is not None and len(axes_tuple) == 2 and i == 0 and plan.nx == sizes[0])
            else None
        )
        cur = mxu_ifft(cur, n=sizes[i], axis=axes_tuple[i], norm=norm, plan=p_c)

    p_real = (
        plan.plan_y_real
        if (plan is not None and len(axes_tuple) == 2 and plan.ny == sizes[-1])
        else None
    )
    return mxu_irfft(cur, n=sizes[-1], axis=axes_tuple[-1], norm=norm, plan=p_real)


def mxu_rfft2(
    x: jax.Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    plan: MXU2DPlan | None = None,
) -> jax.Array:
    """2D real-to-complex DFT via MXU ``dot_general`` matching ``jnp.fft.rfft2``."""
    if len(axes) != 2:
        raise ValueError(f"mxu_rfft2 expects 2 axes, got {axes}.")
    return mxu_rfftn(x, s=s, axes=axes, norm=norm, plan=plan)


def mxu_irfft2(
    x: jax.Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    plan: MXU2DPlan | None = None,
) -> jax.Array:
    """2D Hermitian-to-real iDFT via MXU ``dot_general`` matching ``jnp.fft.irfft2``."""
    if len(axes) != 2:
        raise ValueError(f"mxu_irfft2 expects 2 axes, got {axes}.")
    return mxu_irfftn(x, s=s, axes=axes, norm=norm, plan=plan)


__all__ = [
    "MXU1DPlan",
    "MXU2DPlan",
    "MXUReal1DPlan",
    "MXU_AUTO_MAX_LEN",
    "VALID_FFT_BACKENDS",
    "init_mxu_1d_plan",
    "init_mxu_2d_plan",
    "init_mxu_real_1d_plan",
    "mxu_fft",
    "mxu_fft2",
    "mxu_ifft",
    "mxu_ifft2",
    "mxu_irfft",
    "mxu_irfft2",
    "mxu_irfftn",
    "mxu_rfft",
    "mxu_rfft2",
    "mxu_rfftn",
    "resolve_fft_backend",
    "validate_fft_backend",
]
