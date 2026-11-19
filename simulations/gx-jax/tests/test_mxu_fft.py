"""verification suite for the TPU ``v6e`` MXU matrix-multiply spectral backend.

Verifies:
1. 1D and 2D ``mxu_fft`` split-real/imag ``dot_general`` primitives against ``jnp.fft`` across
   even, odd, prime, and composite ``3/2``-dealiased lengths (``96, 108, 144, 192``), all norms
   (``backward``, ``forward``, ``ortho``), arbitrary interior/boundary axes, pad/truncate ``n``,
   and non-zero Nyquist / DC imaginary components in ``irfft``, ``irfft2``, and ``irfftn``.
2. ``resolve_fft_backend`` ``"auto"`` / ``"fft"`` / ``"mxu"`` selection logic across ``tpu``,
   ``cpu``, and ``gpu`` backends and ``N <= 256`` vs ``N > 256`` thresholds, plus ``GXConfig``
   and TOML ``load_config`` parsing/validation.
3. PyTree registration, ``.astype(jnp.float32)`` single-cast weight hygiene, ``jax.grad``, and
   ``jax.vmap`` across ``MXU1DPlan``, ``MXUReal1DPlan``, ``MXU2DPlan``, ``GradPerp``,
   ``ParallelGrad``, ``LinkedParallelGrad``, and ``NTFTParallelGrad``.
4. ``< 1e-12`` (``complex128``) and ``< 1e-6`` (``complex64``) parity between ``fft_backend="mxu"``
   and ``fft_backend="fft"`` across ``GradPerp``, ``nonlinear_rhs`` (Linked + NTFT + EM
   ``fapar``/``fbpar`` + ``ExB`` shear), ``ParallelGrad`` (both unfused and fused
   ``_linear_rhs_parallel_fused`` / ``linear_rhs``), ``LinkedParallelGrad``, ``NTFTParallelGrad``,
   multi-step ``GyrokineticOperator`` trajectories, and 4-device ``shard_tpu`` execution.
5. Lowered StableHLO IR hygiene when ``fft_backend="mxu"`` in runtime ``complex64`` (``float32``):
   - ``0`` ``stablehlo.fft`` ops
   - ``0`` full-array 6D ``stablehlo.transpose`` ops
   - ``0`` ``f64`` / ``complex<f64>`` array ops
   - ``precision = [HIGHEST, HIGHEST]`` on every ``stablehlo.dot_general`` op.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gx_jax.boundary_parallel import (
    LinkedParallelGrad,
    NTFTParallelGrad,
    init_boundary_parallel,
    init_linked_parallel_grad,
    init_ntft_parallel_grad,
)
from gx_jax.config import GXConfig, SpeciesConfig, load_config
from gx_jax.fields import enforce_reality, solve_fields
from gx_jax.geometry_nc import build_vmec_eik_nc, init_geometry, load_eik_nc
from gx_jax.grids import Grids
from gx_jax.integrator import (
    compute_cfl_dt,
    gyrokinetic_rhs,
    init_gyrokinetic_operator,
    init_integrator_state,
    integrate_scan,
    step_gyrokinetic,
)
from gx_jax.linear_rhs import (
    ParallelGrad,
    _linear_rhs_parallel_fused,
    init_linear_operator,
    linear_rhs,
)
from gx_jax.mxu_fft import (
    MXU_AUTO_MAX_LEN,
    VALID_FFT_BACKENDS,
    MXU1DPlan,
    MXU2DPlan,
    MXUReal1DPlan,
    init_mxu_1d_plan,
    init_mxu_2d_plan,
    init_mxu_real_1d_plan,
    mxu_fft,
    mxu_fft2,
    mxu_ifft,
    mxu_ifft2,
    mxu_irfft,
    mxu_irfft2,
    mxu_irfftn,
    mxu_rfft,
    mxu_rfft2,
    mxu_rfftn,
    resolve_fft_backend,
    validate_fft_backend,
)
from gx_jax.nonlinear_exb import (
    compute_nonlinear_max_frequency,
    flow_shear_shift,
    init_exb_shear,
    init_grad_perp,
    init_nonlinear_operator,
    nonlinear_rhs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KBM_IN = REPO_ROOT / "benchmarks" / "linear" / "KBM" / "kbm_miller.in"
W7X_IN = REPO_ROOT / "benchmarks" / "linear" / "ITG_w7x" / "itg_w7x_adiabatic_electrons.in"
ITG_SALPHA_IN = (
    REPO_ROOT / "benchmarks" / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
)

ARRAY_F64_RE = re.compile(r"tensor<\d+x[^>\n]*(?:complex<f64>|f64)>")
TP_6D_RE = re.compile(r"stablehlo\.transpose[^\n]*tensor<(?:\d+x){5,}\d+x[^>]+>")


def _rel_max_err(actual: jax.Array | np.ndarray, expected: jax.Array | np.ndarray) -> float:
    a = np.asarray(actual)
    b = np.asarray(expected)
    diff = float(np.max(np.abs(a - b)))
    denom = max(float(np.max(np.abs(b))), 1.0)
    return diff / denom


def _random_g(
    grids: Grids,
    *,
    seed: int,
    dtype: jnp.dtype = jnp.complex128,
    scale: float = 0.02,
) -> jax.Array:
    rng = np.random.default_rng(seed)
    shape = (grids.nspecies, grids.nhermite, grids.nlaguerre, grids.nz, grids.naky, grids.nakx)
    raw = (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)) * scale
    return enforce_reality(jnp.asarray(raw, dtype=dtype), grids)


def _assert_mxu_stablehlo_clean(hlo_txt: str, label: str) -> dict[str, int]:
    """Assert 0 stablehlo.fft, 0 6D transposes, 0 array f64 ops, and all HIGHEST dot_general."""
    fft_lines = [ln.strip() for ln in hlo_txt.splitlines() if "stablehlo.fft" in ln]
    assert not fft_lines, f"{label}: unexpected stablehlo.fft ops:\n" + "\n".join(fft_lines)

    tp6_lines = [ln.strip() for ln in hlo_txt.splitlines() if TP_6D_RE.search(ln)]
    assert not tp6_lines, f"{label}: forbidden 6D stablehlo.transpose ops:\n" + "\n".join(tp6_lines)

    f64_arrays = ARRAY_F64_RE.findall(hlo_txt)
    assert not f64_arrays, f"{label}: unexpected f64 array tensors in c64 IR: {f64_arrays[:5]}"

    dot_lines = [ln.strip() for ln in hlo_txt.splitlines() if "stablehlo.dot_general" in ln]
    assert dot_lines, f"{label}: expected at least one stablehlo.dot_general op."
    for ln in dot_lines:
        has_highest = (
            "precision = [HIGHEST, HIGHEST]" in ln
            or "[#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]" in ln
        )
        assert has_highest, f"{label}: dot_general missing HIGHEST precision: {ln}"

    return {
        "dot_general": len(dot_lines),
        "fft": len(fft_lines),
        "transpose_6d": len(tp6_lines),
        "f64_arrays": len(f64_arrays),
    }


# ---------------------------------------------------------------------------
# 1. Backend Resolution & Config / TOML Validation
# ---------------------------------------------------------------------------


def test_resolve_fft_backend_and_config_parsing(tmp_path: Path) -> None:
    """Verify 'auto' / 'fft' / 'mxu' resolution rules and TOML / GXConfig validation."""
    assert MXU_AUTO_MAX_LEN == 256
    assert VALID_FFT_BACKENDS == ("auto", "fft", "mxu")
    assert validate_fft_backend("  MXU ") == "mxu"

    # TPU <= 256 selects "mxu"; TPU > 256 falls back to "fft"
    assert (
        resolve_fft_backend("auto", nx=96, ny=144, nz=192, max_chain_len=256, backend="tpu")
        == "mxu"
    )
    assert (
        resolve_fft_backend("AUTO", nx=256, ny=256, nz=256, max_chain_len=256, backend="TPU")
        == "mxu"
    )
    assert resolve_fft_backend("auto", nx=257, ny=96, nz=32, max_chain_len=32, backend="tpu") == (
        "fft"
    )
    assert resolve_fft_backend("auto", nx=96, ny=257, nz=32, max_chain_len=32, backend="tpu") == (
        "fft"
    )
    assert resolve_fft_backend("auto", nx=96, ny=96, nz=257, max_chain_len=32, backend="tpu") == (
        "fft"
    )
    assert resolve_fft_backend("auto", nx=96, ny=96, nz=32, max_chain_len=320, backend="tpu") == (
        "fft"
    )

    # CPU and GPU select "fft" under "auto"
    assert resolve_fft_backend("auto", nx=64, ny=64, nz=32, max_chain_len=64, backend="cpu") == (
        "fft"
    )
    assert resolve_fft_backend("auto", nx=64, ny=64, nz=32, max_chain_len=64, backend="gpu") == (
        "fft"
    )

    # Explicit overrides work on every backend and for any positive N (including N > 256)
    for dev in ("cpu", "gpu", "tpu"):
        assert (
            resolve_fft_backend("mxu", nx=320, ny=192, nz=64, max_chain_len=512, backend=dev)
            == "mxu"
        )
        assert (
            resolve_fft_backend("fft", nx=64, ny=64, nz=32, max_chain_len=32, backend=dev) == "fft"
        )

    with pytest.raises(ValueError, match="fft_backend"):
        resolve_fft_backend("cufft", backend="tpu")
    with pytest.raises(ValueError, match="positive"):
        resolve_fft_backend("auto", nx=0, backend="tpu")

    # GXConfig defaults & normalization
    cfg_default = GXConfig()
    assert cfg_default.fft_backend == "auto"
    assert GXConfig(fft_backend="MXU").fft_backend == "mxu"
    assert replace(cfg_default, fft_backend="FFT").fft_backend == "fft"
    with pytest.raises(ValueError, match="fft_backend"):
        GXConfig(fft_backend="invalid")

    # TOML parsing from [Expert], [Dimensions], and top-level
    base_txt = ITG_SALPHA_IN.read_text()

    p_exp = tmp_path / "exp_mxu.in"
    p_exp.write_text(base_txt + '\n[Expert]\nfft_backend = "MXU"\n')
    assert load_config(p_exp).fft_backend == "mxu"

    p_dim = tmp_path / "dim_fft.in"
    p_dim.write_text(base_txt.replace("[Dimensions]", '[Dimensions]\nfft_backend = "FFT"'))
    assert load_config(p_dim).fft_backend == "fft"

    p_top = tmp_path / "top_mxu.in"
    p_top.write_text('fft_backend = "Mxu"\n' + base_txt)
    assert load_config(p_top).fft_backend == "mxu"

    p_bad = tmp_path / "bad_fb.in"
    p_bad.write_text(base_txt + '\n[Expert]\nfft_backend = "pocketfft"\n')
    with pytest.raises(ValueError, match="fft_backend"):
        load_config(p_bad)


# ---------------------------------------------------------------------------
# 2. 1D & 2D MXU Spectral Primitives Parity (Even/Odd + Dealiased 96..192 + Nyquist)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n",
    [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 96, 108, 144, 192, 256],
)
def test_1d_mxu_primitives_even_odd_and_composite_lengths(n: int) -> None:
    """Verify 1D mxu_fft / ifft / rfft / irfft against jnp.fft in c128/f64 and c64/f32."""
    rng = np.random.default_rng(100 + n)
    xc_np = rng.standard_normal((3, n, 2)) + 1.0j * rng.standard_normal((3, n, 2))
    xr_np = rng.standard_normal((3, n, 2))

    plan_c64 = init_mxu_1d_plan(n, dtype=jnp.float64)
    plan_c32 = plan_c64.astype(jnp.float32)
    plan_r64 = init_mxu_real_1d_plan(n, dtype=jnp.float64)
    assert isinstance(plan_r64, MXUReal1DPlan)
    plan_r32 = plan_r64.astype(jnp.float32)

    xc128 = jnp.asarray(xc_np, dtype=jnp.complex128)
    xc64 = xc128.astype(jnp.complex64)
    xr64 = jnp.asarray(xr_np, dtype=jnp.float64)
    xr32 = xr64.astype(jnp.float32)

    for norm in (None, "backward", "forward", "ortho"):
        ref_f128 = jnp.fft.fft(xc128, axis=1, norm=norm)
        got_f128 = mxu_fft(xc128, axis=1, norm=norm, plan=plan_c64)
        assert _rel_max_err(got_f128, ref_f128) < 1.0e-12

        ref_i128 = jnp.fft.ifft(xc128, axis=1, norm=norm)
        got_i128 = mxu_ifft(xc128, axis=1, norm=norm, plan=plan_c64)
        assert _rel_max_err(got_i128, ref_i128) < 1.0e-12

        ref_f64 = jnp.fft.fft(xc64, axis=1, norm=norm)
        got_f64 = mxu_fft(xc64, axis=1, norm=norm, plan=plan_c32)
        assert _rel_max_err(got_f64, ref_f64) < 1.0e-6

        ref_i64 = jnp.fft.ifft(xc64, axis=1, norm=norm)
        got_i64 = mxu_ifft(xc64, axis=1, norm=norm, plan=plan_c32)
        assert _rel_max_err(got_i64, ref_i64) < 1.0e-6

        ref_rf64 = jnp.fft.rfft(xr64, axis=1, norm=norm)
        got_rf64 = mxu_rfft(xr64, axis=1, norm=norm, plan=plan_r64)
        assert _rel_max_err(got_rf64, ref_rf64) < 1.0e-12
        # DC (and even-N Nyquist) imaginary components of rfft must be identically 0.0
        assert float(jnp.max(jnp.abs(jnp.imag(got_rf64[:, 0, :])))) == 0.0
        if n % 2 == 0:
            assert float(jnp.max(jnp.abs(jnp.imag(got_rf64[:, n // 2, :])))) == 0.0

        ref_rf32 = jnp.fft.rfft(xr32, axis=1, norm=norm)
        got_rf32 = mxu_rfft(xr32, axis=1, norm=norm, plan=plan_r32)
        assert _rel_max_err(got_rf32, ref_rf32) < 1.0e-6

        # irfft on general complex spectrum WITH large non-zero imaginary parts at DC & Nyquist!
        n_freq = n // 2 + 1
        spec_np = rng.standard_normal((3, n_freq, 2)) + 1.0j * rng.standard_normal((3, n_freq, 2))
        spec_np[:, 0, :] += 5.0j
        if n % 2 == 0:
            spec_np[:, n // 2, :] -= 7.5j
        spec128 = jnp.asarray(spec_np, dtype=jnp.complex128)
        spec64 = spec128.astype(jnp.complex64)

        ref_ir64 = jnp.fft.irfft(spec128, n=n, axis=1, norm=norm)
        got_ir64 = mxu_irfft(spec128, n=n, axis=1, norm=norm, plan=plan_r64)
        assert _rel_max_err(got_ir64, ref_ir64) < 1.0e-12

        ref_ir32 = jnp.fft.irfft(spec64, n=n, axis=1, norm=norm)
        got_ir32 = mxu_irfft(spec64, n=n, axis=1, norm=norm, plan=plan_r32)
        assert _rel_max_err(got_ir32, ref_ir32) < 1.0e-6


@pytest.mark.parametrize(
    "nx,ny",
    [
        (96, 96),
        (96, 108),
        (108, 144),
        (144, 192),
        (192, 96),
        (15, 17),
        (16, 15),
        (17, 18),
    ],
)
def test_2d_mxu_dealiased_and_odd_even_transforms(nx: int, ny: int) -> None:
    """Verify 2D mxu_rfftn/irfftn, mxu_rfft2/irfft2, and mxu_fft2/ifft2 including Nyquist imag."""
    rng = np.random.default_rng(1000 + 31 * nx + ny)
    nyc = ny // 2 + 1
    plan2d_64 = init_mxu_2d_plan(nx, ny, dtype=jnp.float64)
    plan2d_32 = plan2d_64.astype(jnp.float32)

    # Complex spectrum with deliberate non-zero imaginary parts at ky=0 and even-ny Nyquist!
    spec_np = rng.standard_normal((2, nx, nyc)) + 1.0j * rng.standard_normal((2, nx, nyc))
    spec_np[:, :, 0] += 3.3j
    if ny % 2 == 0:
        spec_np[:, :, ny // 2] -= 4.7j

    spec128 = jnp.asarray(spec_np, dtype=jnp.complex128)
    spec64 = spec128.astype(jnp.complex64)

    # 1. irfftn / rfftn with axes=(-2, -1) and norm="forward" (GradPerp.*_nxny convention)
    ref_c2r_128 = jnp.fft.irfftn(spec128, s=(nx, ny), axes=(-2, -1), norm="forward")
    got_c2r_128 = mxu_irfftn(spec128, s=(nx, ny), axes=(-2, -1), norm="forward", plan=plan2d_64)
    assert _rel_max_err(got_c2r_128, ref_c2r_128) < 1.0e-12

    ref_c2r_64 = jnp.fft.irfftn(spec64, s=(nx, ny), axes=(-2, -1), norm="forward")
    got_c2r_64 = mxu_irfftn(spec64, s=(nx, ny), axes=(-2, -1), norm="forward", plan=plan2d_32)
    assert _rel_max_err(got_c2r_64, ref_c2r_64) < 1.0e-6

    real_np = rng.standard_normal((2, nx, ny))
    real64 = jnp.asarray(real_np, dtype=jnp.float64)
    real32 = real64.astype(jnp.float32)

    ref_r2c_128 = jnp.fft.rfftn(real64, s=(nx, ny), axes=(-2, -1), norm="forward")
    got_r2c_128 = mxu_rfftn(real64, s=(nx, ny), axes=(-2, -1), norm="forward", plan=plan2d_64)
    assert _rel_max_err(got_r2c_128, ref_r2c_128) < 1.0e-12

    ref_r2c_64 = jnp.fft.rfftn(real32, s=(nx, ny), axes=(-2, -1), norm="forward")
    got_r2c_64 = mxu_rfftn(real32, s=(nx, ny), axes=(-2, -1), norm="forward", plan=plan2d_32)
    assert _rel_max_err(got_r2c_64, ref_r2c_64) < 1.0e-6

    # 2. irfft2 / rfft2 with axes=(-1, -2) and norm="forward" (public GradPerp C++ convention)
    spec_yx_128 = jnp.swapaxes(spec128, -2, -1)
    spec_yx_64 = spec_yx_128.astype(jnp.complex64)
    ref_yx_128 = jnp.fft.irfft2(spec_yx_128, s=(nx, ny), axes=(-1, -2), norm="forward")
    got_yx_128 = mxu_irfft2(spec_yx_128, s=(nx, ny), axes=(-1, -2), norm="forward", plan=plan2d_64)
    assert _rel_max_err(got_yx_128, ref_yx_128) < 1.0e-12

    ref_yx_64 = jnp.fft.irfft2(spec_yx_64, s=(nx, ny), axes=(-1, -2), norm="forward")
    got_yx_64 = mxu_irfft2(spec_yx_64, s=(nx, ny), axes=(-1, -2), norm="forward", plan=plan2d_32)
    assert _rel_max_err(got_yx_64, ref_yx_64) < 1.0e-6

    real_yx64 = jnp.swapaxes(real64, -2, -1)
    real_yx32 = real_yx64.astype(jnp.float32)
    assert (
        _rel_max_err(
            mxu_rfft2(real_yx64, s=(nx, ny), axes=(-1, -2), norm="forward", plan=plan2d_64),
            jnp.fft.rfft2(real_yx64, s=(nx, ny), axes=(-1, -2), norm="forward"),
        )
        < 1.0e-12
    )
    assert (
        _rel_max_err(
            mxu_rfft2(real_yx32, s=(nx, ny), axes=(-1, -2), norm="forward", plan=plan2d_32),
            jnp.fft.rfft2(real_yx32, s=(nx, ny), axes=(-1, -2), norm="forward"),
        )
        < 1.0e-6
    )

    # 3. Complex 2D fft2 / ifft2
    c2d_128 = jnp.asarray(
        rng.standard_normal((2, nx, ny)) + 1.0j * rng.standard_normal((2, nx, ny)),
        dtype=jnp.complex128,
    )
    py_c64 = init_mxu_1d_plan(ny, dtype=jnp.float64)
    assert (
        _rel_max_err(
            mxu_fft2(c2d_128, axes=(-2, -1), plan=(plan2d_64.plan_x, py_c64)),
            jnp.fft.fft2(c2d_128, axes=(-2, -1)),
        )
        < 1.0e-12
    )
    assert (
        _rel_max_err(
            mxu_ifft2(c2d_128, axes=(-2, -1), plan=(plan2d_64.plan_x, py_c64)),
            jnp.fft.ifft2(c2d_128, axes=(-2, -1)),
        )
        < 1.0e-12
    )


def test_mxu_pad_truncate_axes_and_autodiff_vmap() -> None:
    """Verify n != shape[axis] pad/truncate, boundary axes, jax.grad, and jax.vmap."""
    rng = np.random.default_rng(77)
    x = jnp.asarray(
        rng.standard_normal((5, 12, 4)) + 1.0j * rng.standard_normal((5, 12, 4)),
        dtype=jnp.complex128,
    )
    xr = jnp.real(x)

    for n_arg in (8, 12, 19):
        for ax in (0, 1, 2, -1, -2):
            assert _rel_max_err(mxu_fft(x, n=n_arg, axis=ax), jnp.fft.fft(x, n=n_arg, axis=ax)) < (
                1.0e-12
            )
            assert (
                _rel_max_err(mxu_ifft(x, n=n_arg, axis=ax), jnp.fft.ifft(x, n=n_arg, axis=ax))
                < 1.0e-12
            )
            assert (
                _rel_max_err(mxu_rfft(xr, n=n_arg, axis=ax), jnp.fft.rfft(xr, n=n_arg, axis=ax))
                < 1.0e-12
            )
            assert (
                _rel_max_err(mxu_irfft(x, n=n_arg, axis=ax), jnp.fft.irfft(x, n=n_arg, axis=ax))
                < 1.0e-12
            )

    # Reverse-mode AD parity (jax.grad)
    plan12 = init_mxu_1d_plan(12, dtype=jnp.float64)
    g_mxu = jax.grad(lambda z: jnp.sum(jnp.abs(mxu_fft(z, axis=1, plan=plan12)) ** 2))(xr)
    g_fft = jax.grad(lambda z: jnp.sum(jnp.abs(jnp.fft.fft(z, axis=1)) ** 2))(xr)
    assert _rel_max_err(g_mxu, g_fft) < 1.0e-12

    # Batched jax.vmap parity
    xb = jnp.asarray(rng.standard_normal((4, 6, 12, 3)), dtype=jnp.float64)
    vm_mxu = jax.vmap(lambda z: mxu_irfft(mxu_rfft(z, axis=1), n=12, axis=1))(xb)
    assert _rel_max_err(vm_mxu, xb) < 1.0e-12


# ---------------------------------------------------------------------------
# 3. GradPerp (Including Composite 96/108/144/192) & Nonlinear RHS Parity
# ---------------------------------------------------------------------------


def test_grad_perp_composite_dealiased_and_ntft_exb_parity() -> None:
    """Verify GradPerp mxu vs fft across composite 3/2 lengths (96, 108, 144, 192) and NTFT/ExB."""
    base = load_config(ITG_SALPHA_IN)
    rng = np.random.default_rng(2026)

    for nx, ny in ((96, 96), (108, 144), (192, 96)):
        cfg = replace(base, nx=nx, ny=ny, nperiod=1, zp=1, ntheta_in=4, nz=4)
        cfg, grids, geo = init_geometry(cfg)
        gp_mxu64 = init_grad_perp(cfg, grids, geo, dtype=jnp.float64, fft_backend="mxu")
        gp_fft64 = init_grad_perp(cfg, grids, geo, dtype=jnp.float64, fft_backend="fft")
        gp_mxu32 = gp_mxu64.astype(jnp.float32)
        gp_fft32 = gp_fft64.astype(jnp.float32)

        assert gp_mxu64.fft_backend == "mxu"
        assert isinstance(gp_mxu64.mxu_plan_2d, MXU2DPlan)
        assert gp_mxu32.mxu_plan_2d is not None
        assert gp_mxu32.mxu_plan_2d.plan_x.cos_f.dtype == jnp.float32
        assert gp_mxu32.mxu_plan_2d.plan_y_real.rfft_cos.dtype == jnp.float32

        act_np = rng.standard_normal((2, grids.nz, grids.naky, grids.nakx)) + (
            1.0j * rng.standard_normal((2, grids.nz, grids.naky, grids.nakx))
        )
        act128 = enforce_reality(jnp.asarray(act_np, dtype=jnp.complex128), grids)
        act64 = act128.astype(jnp.complex64)

        assert _rel_max_err(gp_mxu64.c2r_nxny(act128), gp_fft64.c2r_nxny(act128)) < 1.0e-12
        assert _rel_max_err(gp_mxu32.c2r_nxny(act64), gp_fft32.c2r_nxny(act64)) < 1.0e-6
        assert _rel_max_err(gp_mxu64.dx_c2r(act128), gp_fft64.dx_c2r(act128)) < 1.0e-12
        assert _rel_max_err(gp_mxu32.dx_c2r(act64), gp_fft32.dx_c2r(act64)) < 1.0e-6
        assert _rel_max_err(gp_mxu64.dy_c2r(act128), gp_fft64.dy_c2r(act128)) < 1.0e-12
        assert _rel_max_err(gp_mxu32.dy_c2r(act64), gp_fft32.dy_c2r(act64)) < 1.0e-6
        assert _rel_max_err(gp_mxu64.qvar(act128), gp_fft64.qvar(act128)) < 1.0e-12
        assert _rel_max_err(gp_mxu32.qvar(act64), gp_fft32.qvar(act64)) < 1.0e-6

        real_nxny64 = gp_fft64.c2r_nxny(act128)
        real_nxny32 = real_nxny64.astype(jnp.float32)
        assert _rel_max_err(gp_mxu64.r2c_nxny(real_nxny64), gp_fft64.r2c_nxny(real_nxny64)) < (
            1.0e-12
        )
        assert _rel_max_err(gp_mxu32.r2c_nxny(real_nxny32), gp_fft32.r2c_nxny(real_nxny32)) < (
            1.0e-6
        )

    # Hybrid rfft/irfft phase_mult & phase_mult_nxny with BOTH non_twist=True and ExB shear
    cfg_ntft = replace(
        base,
        nx=16,
        ny=18,
        nperiod=1,
        zp=1,
        ntheta_in=8,
        nz=8,
        non_twist=True,
        exb_shear=True,
        exb_shear_phase=True,
        g_exb=0.2,
    )
    cfg_ntft, grids_ntft, geo_ntft = init_geometry(cfg_ntft)
    gp_nt_mxu64 = init_grad_perp(
        cfg_ntft, grids_ntft, geo_ntft, dtype=jnp.float64, fft_backend="mxu"
    )
    gp_nt_fft64 = init_grad_perp(
        cfg_ntft, grids_ntft, geo_ntft, dtype=jnp.float64, fft_backend="fft"
    )
    gp_nt_mxu32 = gp_nt_mxu64.astype(jnp.float32)
    gp_nt_fft32 = gp_nt_fft64.astype(jnp.float32)

    g_nt = _random_g(grids_ntft, seed=31, dtype=jnp.complex128)
    op_lin = init_linear_operator(cfg_ntft, grids_ntft, geo_ntft)
    f_nt = solve_fields(g_nt, op_lin.solver)
    exb64 = init_exb_shear(cfg_ntft, grids_ntft, geo_ntft, dtype=jnp.float64)
    exb64, _ = flow_shear_shift(exb64, f_nt, grids_ntft, geo_ntft, 0.35)
    exb32 = exb64.astype(jnp.float32)

    r_yx64 = jnp.asarray(rng.standard_normal((2, 8, 18, 16)), dtype=jnp.float64)
    r_xy64 = jnp.swapaxes(r_yx64, -2, -1)
    for pos in (True, False):
        pm_mxu64 = gp_nt_mxu64.phase_mult(
            r_yx64,
            positive_phase=pos,
            phasefac_exb=exb64.phasefac_exb,
            phasefacminus_exb=exb64.phasefacminus_exb,
        )
        pm_fft64 = gp_nt_fft64.phase_mult(
            r_yx64,
            positive_phase=pos,
            phasefac_exb=exb64.phasefac_exb,
            phasefacminus_exb=exb64.phasefacminus_exb,
        )
        assert _rel_max_err(pm_mxu64, pm_fft64) < 1.0e-12

        pm_xy_mxu32 = gp_nt_mxu32.phase_mult_nxny(
            r_xy64.astype(jnp.float32),
            positive_phase=pos,
            phasefac_exb_nxny=exb32.phasefac_exb_nxny,
            phasefacminus_exb_nxny=exb32.phasefacminus_exb_nxny,
        )
        pm_xy_fft32 = gp_nt_fft32.phase_mult_nxny(
            r_xy64.astype(jnp.float32),
            positive_phase=pos,
            phasefac_exb_nxny=exb32.phasefac_exb_nxny,
            phasefacminus_exb_nxny=exb32.phasefacminus_exb_nxny,
        )
        assert _rel_max_err(pm_xy_mxu32, pm_xy_fft32) < 1.0e-6

    assert (
        _rel_max_err(
            gp_nt_mxu64.ntft_dx_c2r(g_nt, kxstar=exb64.kxstar),
            gp_nt_fft64.ntft_dx_c2r(g_nt, kxstar=exb64.kxstar),
        )
        < 1.0e-12
    )


@pytest.mark.parametrize("non_twist", [False, True])
def test_nonlinear_rhs_linked_ntft_em_and_exb_shear_parity(non_twist: bool) -> None:
    """Verify < 1e-12 (c128) and < 1e-6 (c64) parity on EM nonlinear_rhs with ExB shear."""
    base = load_config(KBM_IN)
    cfg = replace(
        base,
        nx=10,
        ny=12,
        nperiod=1,
        zp=1,
        ntheta_in=8,
        nz=8,
        nhermite=4,
        nlaguerre=2,
        nonlinear_mode=True,
        linear=False,
        non_twist=non_twist,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        exb_shear=True,
        exb_shear_phase=True,
        g_exb=0.18,
    )
    cfg, grids, geo = init_geometry(cfg)
    nl_mxu64 = init_nonlinear_operator(cfg, grids, geo, dtype=jnp.float64, fft_backend="mxu")
    nl_fft64 = init_nonlinear_operator(cfg, grids, geo, dtype=jnp.float64, fft_backend="fft")
    nl_mxu32 = nl_mxu64.astype(jnp.float32)
    nl_fft32 = nl_fft64.astype(jnp.float32)

    lin64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64, fft_backend="mxu")
    g128 = _random_g(grids, seed=55 + int(non_twist), dtype=jnp.complex128)
    f128 = solve_fields(g128, lin64.solver)
    exb64 = init_exb_shear(cfg, grids, geo, dtype=jnp.float64)
    exb64, f128 = flow_shear_shift(exb64, f128, grids, geo, 0.25)

    out_mxu128 = nonlinear_rhs(g128, f128, nl_mxu64, exb_state=exb64)
    out_fft128 = nonlinear_rhs(g128, f128, nl_fft64, exb_state=exb64)
    assert _rel_max_err(out_mxu128, out_fft128) < 1.0e-12

    w_mxu128 = compute_nonlinear_max_frequency(f128, nl_mxu64, exb_state=exb64)
    w_fft128 = compute_nonlinear_max_frequency(f128, nl_fft64, exb_state=exb64)
    for wm, wf in zip(w_mxu128, w_fft128, strict=True):
        assert _rel_max_err(wm, wf) < 1.0e-12

    g64 = g128.astype(jnp.complex64)
    f64 = f128.astype(jnp.complex64)
    exb32 = exb64.astype(jnp.float32)
    out_mxu64 = nonlinear_rhs(g64, f64, nl_mxu32, exb_state=exb32)
    out_fft64 = nonlinear_rhs(g64, f64, nl_fft32, exb_state=exb32)
    assert _rel_max_err(out_mxu64, out_fft64) < 1.0e-6


# ---------------------------------------------------------------------------
# 4. ParallelGrad, LinkedParallelGrad, NTFTParallelGrad, & Fused/Unfused Linear RHS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("closure_model", ["none", "beer4+2", "smith_par", "smith_perp"])
def test_parallel_grad_and_fused_unfused_linear_rhs_parity(closure_model: str) -> None:
    """Verify ParallelGrad methods and both fused & unfused linear_rhs in c128 and c64."""
    base = load_config(KBM_IN)
    nl_val = 4 if closure_model == "smith_perp" else 2
    cfg = replace(
        base,
        nx=7,
        ny=7,
        nperiod=1,
        zp=1,
        ntheta_in=16,
        nz=16,
        nhermite=4,
        nlaguerre=nl_val,
        boundary="periodic",
        boundary_option_periodic=True,
        dealias_kz=True,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        closure_model=closure_model,
        smith_par_q=2,
        smith_perp_q=3,
        hyperz=True,
        nu_hyper_z=0.15,
        hypercollisions_kz=True,
        hypercollisions_const=True,
        hyper=True,
    )
    cfg, grids, geo = init_geometry(cfg)

    op_mxu64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64, fft_backend="mxu")
    op_fft64 = init_linear_operator(cfg, grids, geo, dtype=jnp.float64, fft_backend="fft")
    op_mxu32 = op_mxu64.astype(jnp.float32)
    op_fft32 = op_fft64.astype(jnp.float32)

    gp_mxu64 = op_mxu64.grad_par
    gp_fft64 = op_fft64.grad_par
    assert isinstance(gp_mxu64, ParallelGrad)
    assert gp_mxu64.fft_backend == "mxu"
    assert isinstance(gp_mxu64.mxu_plan, MXU1DPlan)
    assert op_mxu32.grad_par.mxu_plan.cos_f.dtype == jnp.float32

    g128 = _random_g(grids, seed=81, dtype=jnp.complex128)
    f128 = solve_fields(g128, op_mxu64.solver)
    g64 = g128.astype(jnp.complex64)
    f64 = f128.astype(jnp.complex64)

    for mname in ("dz", "abs_dz", "dz2", "zft", "zft_inverse", "dealias", "dealias_chain"):
        assert _rel_max_err(getattr(gp_mxu64, mname)(g128), getattr(gp_fft64, mname)(g128)) < (
            1.0e-12
        )
        assert (
            _rel_max_err(
                getattr(op_mxu32.grad_par, mname)(g64),
                getattr(op_fft32.grad_par, mname)(g64),
            )
            < 1.0e-6
        )
    assert _rel_max_err(gp_mxu64.hyperz(g128, 0.2), gp_fft64.hyperz(g128, 0.2)) < 1.0e-12
    assert (
        _rel_max_err(op_mxu32.grad_par.hyperz(g64, 0.2), op_fft32.grad_par.hyperz(g64, 0.2))
        < 1.0e-6
    )

    # Direct _linear_rhs_parallel_fused helper + full linear_rhs (fused & unfused)
    assert (
        _rel_max_err(
            _linear_rhs_parallel_fused(g128, f128, op_mxu64, gp_mxu64),
            _linear_rhs_parallel_fused(g128, f128, op_fft64, gp_fft64),
        )
        < 1.0e-12
    )
    assert (
        _rel_max_err(
            _linear_rhs_parallel_fused(g64, f64, op_mxu32, op_mxu32.grad_par),
            _linear_rhs_parallel_fused(g64, f64, op_fft32, op_fft32.grad_par),
        )
        < 1.0e-6
    )

    for fused in (True, False):
        r_mxu128 = linear_rhs(g128, f128, op_mxu64, 0.01, use_fused_fft=fused)
        r_fft128 = linear_rhs(g128, f128, op_fft64, 0.01, use_fused_fft=fused)
        assert _rel_max_err(r_mxu128, r_fft128) < 1.0e-12

        r_mxu64 = linear_rhs(g64, f64, op_mxu32, jnp.float32(0.01), use_fused_fft=fused)
        r_fft64 = linear_rhs(g64, f64, op_fft32, jnp.float32(0.01), use_fused_fft=fused)
        assert _rel_max_err(r_mxu64, r_fft64) < 1.0e-6


def test_linked_and_ntft_parallel_grad_w7x_and_tokamak_parity(tmp_path: Path) -> None:
    """Verify LinkedParallelGrad and NTFTParallelGrad (including W7-X 3D stellarator) parity."""
    # 1. Multi-link LinkedParallelGrad with chain padding
    base = load_config(KBM_IN)
    cfg_lnk = replace(
        base,
        nx=10,
        ny=10,
        nperiod=1,
        zp=1,
        ntheta_in=12,
        nz=12,
        nhermite=4,
        nlaguerre=2,
        hyperz=True,
        hypercollisions_kz=True,
        closure_model="smith_par",
        smith_par_q=2,
    )
    cfg_lnk, grids_lnk, _ = init_geometry(cfg_lnk)
    lpg_mxu64 = init_linked_parallel_grad(
        cfg_lnk, grids_lnk, dtype=jnp.float64, chain_pad_multiple=4, fft_backend="mxu"
    )
    lpg_fft64 = init_linked_parallel_grad(
        cfg_lnk, grids_lnk, dtype=jnp.float64, chain_pad_multiple=4, fft_backend="fft"
    )
    lpg_mxu32 = lpg_mxu64.astype(jnp.float32)
    lpg_fft32 = lpg_fft64.astype(jnp.float32)

    assert isinstance(lpg_mxu64, LinkedParallelGrad)
    assert lpg_mxu64.fft_backend == "mxu"
    assert max(b.n_links for b in lpg_mxu64.buckets) > 1
    assert all(
        b.mxu_plan is not None and b.mxu_plan.cos_f.dtype == jnp.float32 for b in lpg_mxu32.buckets
    )

    # Verify PyTree round-trip
    leaves, treedef = jax.tree_util.tree_flatten(lpg_mxu32)
    lpg_unflat = jax.tree_util.tree_unflatten(treedef, leaves)
    assert lpg_unflat.fft_backend == "mxu"

    g128 = _random_g(grids_lnk, seed=101, dtype=jnp.complex128)
    g64 = g128.astype(jnp.complex64)
    for mname in ("dz", "abs_dz", "dz2", "zft", "zft_inverse", "dealias_chain"):
        assert _rel_max_err(getattr(lpg_mxu64, mname)(g128), getattr(lpg_fft64, mname)(g128)) < (
            1.0e-12
        )
        assert _rel_max_err(getattr(lpg_mxu32, mname)(g64), getattr(lpg_fft32, mname)(g64)) < (
            1.0e-6
        )
    assert _rel_max_err(lpg_mxu64.hyperz(g128, 0.25), lpg_fft64.hyperz(g128, 0.25)) < 1.0e-12
    assert _rel_max_err(lpg_mxu32.hyperz(g64, 0.25), lpg_fft32.hyperz(g64, 0.25)) < 1.0e-6

    # 2. NTFTParallelGrad on 3D W7-X stellarator equilibrium
    w7x_base = load_config(W7X_IN)
    wout_p = W7X_IN.parent / "wout_w7x.nc"
    w7x_eik = build_vmec_eik_nc(
        replace(w7x_base, ntheta_in=16, nz=16, npol=1.0),
        wout_p,
        tmp_path / "w7x_1turn_nz16.eik.nc",
    )
    cfg_w7x = replace(
        w7x_base,
        geo_option="nc",
        geofilename=str(w7x_eik),
        nx=10,
        ny=10,
        ntheta_in=16,
        nz=16,
        npol=1.0,
        nhermite=4,
        nlaguerre=2,
        non_twist=True,
        hyperz=True,
        hypercollisions_kz=True,
    )
    cfg_w7x, grids_w7x, geo_w7x = load_eik_nc(w7x_eik, cfg_w7x)
    ntft_mxu64 = init_ntft_parallel_grad(
        cfg_w7x, grids_w7x, geo_w7x, dtype=jnp.float64, chain_pad_multiple=4, fft_backend="mxu"
    )
    ntft_fft64 = init_ntft_parallel_grad(
        cfg_w7x, grids_w7x, geo_w7x, dtype=jnp.float64, chain_pad_multiple=4, fft_backend="fft"
    )
    ntft_mxu32 = ntft_mxu64.astype(jnp.float32)
    ntft_fft32 = ntft_fft64.astype(jnp.float32)

    assert isinstance(ntft_mxu64, NTFTParallelGrad)
    assert ntft_mxu64.fft_backend == "mxu"
    assert all(
        b.mxu_plan is not None and b.mxu_plan.cos_f.dtype == jnp.float32 for b in ntft_mxu32.buckets
    )

    gw128 = _random_g(grids_w7x, seed=102, dtype=jnp.complex128)
    gw64 = gw128.astype(jnp.complex64)
    for mname in ("dz", "abs_dz", "dz2", "zft", "zft_inverse", "dealias_chain"):
        err128 = _rel_max_err(getattr(ntft_mxu64, mname)(gw128), getattr(ntft_fft64, mname)(gw128))
        assert err128 < 1.0e-12
        assert _rel_max_err(getattr(ntft_mxu32, mname)(gw64), getattr(ntft_fft32, mname)(gw64)) < (
            1.0e-6
        )
    assert _rel_max_err(ntft_mxu64.hyperz(gw128, 0.1), ntft_fft64.hyperz(gw128, 0.1)) < 1.0e-12
    assert _rel_max_err(ntft_mxu32.hyperz(gw64, 0.1), ntft_fft32.hyperz(gw64, 0.1)) < 1.0e-6

    # Also check init_boundary_parallel "auto" fallback when max_chain_len > 256
    cfg_long = replace(
        base,
        nx=28,
        ny=10,
        jtwist=1,
        x0=-1.0,
        nperiod=1,
        zp=1,
        ntheta_in=32,
        nz=32,
        fft_backend="auto",
    )
    cfg_long, grids_long, _ = init_geometry(cfg_long)
    lpg_long = init_boundary_parallel(cfg_long, grids_long)
    max_cl = max(b.fft_len for b in lpg_long.buckets)
    assert max_cl > 256
    assert lpg_long.fft_backend == "fft"
    assert (
        resolve_fft_backend(
            "auto",
            nx=grids_long.nx,
            ny=grids_long.ny,
            nz=grids_long.nz,
            max_chain_len=max_cl,
            backend="tpu",
        )
        == "fft"
    )


# ---------------------------------------------------------------------------
# 5. Multi-Step GyrokineticOperator & 4-Device shard_tpu Subprocess Parity
# ---------------------------------------------------------------------------


def test_multistep_gyrokinetic_trajectory_parity() -> None:
    """Verify multi-step GyrokineticOperator trajectory parity between mxu and fft."""
    base = load_config(KBM_IN)
    sp_coll = tuple(
        SpeciesConfig.from_raw(
            z=sp.z,
            mass=sp.mass,
            dens=sp.dens,
            temp=sp.temp,
            tprim=sp.tprim,
            fprim=sp.fprim,
            nu_ss=0.04 if i == 0 else 0.08,
            stype="ion" if sp.type_id == 0 else "electron",
            beta=base.beta,
            long_wavelength_gk=base.long_wavelength_gk,
        )
        for i, sp in enumerate(base.species)
    )
    cfg = replace(
        base,
        species=sp_coll,
        nx=7,
        ny=7,
        nperiod=1,
        zp=1,
        ntheta_in=8,
        nz=8,
        nhermite=4,
        nlaguerre=2,
        boundary="periodic",
        boundary_option_periodic=True,
        nonlinear_mode=True,
        linear=False,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        collisions=True,
        closure_model="beer4+2",
        exb_shear=True,
        exb_shear_phase=True,
        g_exb=0.1,
        dealias_kz=True,
        scheme="rk3",
        fixed_dt=True,
        dt=1.0e-5,
        dt_min=1.0e-8,
    )
    cfg, grids, geo = init_geometry(cfg)
    gk_mxu64 = init_gyrokinetic_operator(
        replace(cfg, fft_backend="mxu"), grids, geo, dtype=jnp.float64
    )
    gk_fft64 = init_gyrokinetic_operator(
        replace(cfg, fft_backend="fft"), grids, geo, dtype=jnp.float64
    )
    gk_mxu32 = gk_mxu64.astype(jnp.float32)
    gk_fft32 = gk_fft64.astype(jnp.float32)

    assert gk_mxu32.linear_op.fft_backend == "mxu"
    assert gk_mxu32.nonlinear_op is not None and gk_mxu32.nonlinear_op.fft_backend == "mxu"

    g128 = _random_g(grids, seed=303, dtype=jnp.complex128, scale=0.005)
    f128 = solve_fields(g128, gk_mxu64.linear_op.solver)
    dt_step = min(
        float(compute_cfl_dt(f128, replace(gk_fft64, fixed_dt=False), scheme="rk3")),
        1.0e-5,
    )

    s_mxu = jax.jit(lambda g, f: step_gyrokinetic(g, f, gk_mxu64, dt_step, scheme="rk4"))(
        g128, f128
    )
    s_fft = jax.jit(lambda g, f: step_gyrokinetic(g, f, gk_fft64, dt_step, scheme="rk4"))(
        g128, f128
    )
    assert _rel_max_err(s_mxu.G, s_fft.G) < 1.0e-12
    assert _rel_max_err(s_mxu.fields.phi, s_fft.fields.phi) < 1.0e-12

    traj_mxu128 = integrate_scan(g128, f128, gk_mxu64, nsteps=3, dt=dt_step, save_every=1)
    traj_fft128 = integrate_scan(g128, f128, gk_fft64, nsteps=3, dt=dt_step, save_every=1)
    assert jnp.all(jnp.isfinite(traj_mxu128.G))
    assert _rel_max_err(traj_mxu128.G, traj_fft128.G) < 1.0e-12
    assert _rel_max_err(traj_mxu128.phi2_history, traj_fft128.phi2_history) < 1.0e-12

    g64 = g128.astype(jnp.complex64)
    f64 = f128.astype(jnp.complex64)
    traj_mxu64 = integrate_scan(g64, f64, gk_mxu32, nsteps=3, dt=dt_step, save_every=1)
    traj_fft64 = integrate_scan(g64, f64, gk_fft32, nsteps=3, dt=dt_step, save_every=1)
    assert jnp.all(jnp.isfinite(traj_mxu64.G))
    assert _rel_max_err(traj_mxu64.G, traj_fft64.G) < 1.0e-6
    assert _rel_max_err(traj_mxu64.phi2_history, traj_fft64.phi2_history) < 1.0e-6


def test_sharded_4device_mxu_parity_and_ir_subprocess() -> None:
    """Verify 4-device ('sp', 'm') sharded execution with fft_backend='mxu' in a subprocess."""
    code = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import re
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from gx_jax.config import load_config
from gx_jax.fields import enforce_reality, solve_fields
from gx_jax.geometry_nc import init_geometry
from gx_jax.integrator import compute_cfl_dt, init_gyrokinetic_operator
from gx_jax.shard_tpu import (
    configure_tpu_precision,
    create_device_mesh,
    gather_g_state,
    shard_g_state,
    sharded_gyrokinetic_rhs,
    step_sharded_gyrokinetic,
)

assert jax.device_count() == 4
configure_tpu_precision()
base = load_config(Path("benchmarks/linear/KBM/kbm_miller.in"))
cfg = replace(
    base,
    nx=7, ny=7, nperiod=1, zp=1, ntheta_in=8, nz=8, nhermite=4, nlaguerre=2,
    boundary="periodic", boundary_option_periodic=True,
    nonlinear_mode=True, linear=False, fphi=1.0, fapar=1.0, fbpar=1.0, dt_min=1.0e-8,
)
cfg, grids, geo = init_geometry(cfg)
gk_mxu = init_gyrokinetic_operator(replace(cfg, fft_backend="mxu"), grids, geo, dtype=jnp.float32)
gk_fft = init_gyrokinetic_operator(replace(cfg, fft_backend="fft"), grids, geo, dtype=jnp.float32)
mesh = create_device_mesh((2, 2), nspecies=2, nhermite=4)

rng = np.random.default_rng(19)
shape = (2, 4, 2, 8, grids.naky, grids.nakx)
g_np = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 0.005
g32 = enforce_reality(jnp.asarray(g_np, dtype=jnp.complex64), grids)
f32 = solve_fields(g32, gk_mxu.linear_op.solver)
dt_cfl = min(float(compute_cfl_dt(f32, replace(gk_fft, fixed_dt=False), scheme="rk3")), 1.0e-5)
g_sh = shard_g_state(g32, mesh)

j_rhs = jax.jit(lambda g, f: sharded_gyrokinetic_rhs(g, f, gk_mxu, jnp.float32(dt_cfl), mesh))
hlo = str(j_rhs.lower(g_sh, f32).compiler_ir(dialect="stablehlo"))
assert "stablehlo.fft" not in hlo
assert not re.findall(r"tensor<\\d+x[^>\\n]*(?:complex<f64>|f64)>", hlo)

step_mxu = jax.jit(
    lambda g, f: step_sharded_gyrokinetic(g, f, gk_mxu, mesh, dt=dt_cfl, scheme="rk3")
)
step_fft = jax.jit(
    lambda g, f: step_sharded_gyrokinetic(g, f, gk_fft, mesh, dt=dt_cfl, scheme="rk3")
)
res_mxu = step_mxu(g_sh, f32)
res_fft = step_fft(g_sh, f32)
g_m = np.asarray(gather_g_state(res_mxu.G, mesh))
g_f = np.asarray(gather_g_state(res_fft.G, mesh))
rel = float(np.max(np.abs(g_m - g_f)) / max(float(np.max(np.abs(g_f))), 1.0))
assert np.all(np.isfinite(g_m))
assert rel < 1.0e-6, f"Sharded MXU vs FFT rel_err={rel}"
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# 6. Strict StableHLO IR Verification (0 fft, 0 6D transpose, 0 f64, HIGHEST)
# ---------------------------------------------------------------------------


def test_stablehlo_ir_zero_fft_zero_6d_transpose_zero_f64_highest() -> None:
    """Audit lowered StableHLO across GradPerp, ParallelGrad, Linked, NTFT, and full RHS."""
    base = load_config(KBM_IN)

    # 1. Periodic nLinks=1 (ParallelGrad + fused & unfused linear_rhs + EM nonlinear_rhs + ExB)
    cfg_p = replace(
        base,
        nx=16,
        ny=16,
        nperiod=1,
        zp=1,
        ntheta_in=16,
        nz=16,
        nhermite=4,
        nlaguerre=2,
        boundary="periodic",
        boundary_option_periodic=True,
        dealias_kz=True,
        nonlinear_mode=True,
        linear=False,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        closure_model="beer4+2",
        hyperz=True,
        hypercollisions_kz=True,
        exb_shear=True,
        exb_shear_phase=True,
        g_exb=0.15,
        fft_backend="mxu",
    )
    cfg_p, grids_p, geo_p = init_geometry(cfg_p)
    gk_p32 = init_gyrokinetic_operator(cfg_p, grids_p, geo_p, dtype=jnp.float32)
    g_p32 = _random_g(grids_p, seed=701, dtype=jnp.complex64)
    st_p32 = init_integrator_state(g_p32, gk_p32, time=0.0, dt=0.005)
    gp_perp32 = gk_p32.nonlinear_op.grad_perp  # type: ignore[union-attr]

    _assert_mxu_stablehlo_clean(
        str(jax.jit(gp_perp32.c2r_nxny).lower(g_p32).compiler_ir(dialect="stablehlo")),
        "GradPerp.c2r_nxny(6D)",
    )
    real6_nxny = gp_perp32.c2r_nxny(g_p32)
    _assert_mxu_stablehlo_clean(
        str(jax.jit(gp_perp32.r2c_nxny).lower(real6_nxny).compiler_ir(dialect="stablehlo")),
        "GradPerp.r2c_nxny(6D)",
    )
    _assert_mxu_stablehlo_clean(
        str(jax.jit(gp_perp32.c2r).lower(g_p32).compiler_ir(dialect="stablehlo")),
        "GradPerp.c2r(6D)",
    )
    real6_yx = gp_perp32.c2r(g_p32)
    _assert_mxu_stablehlo_clean(
        str(jax.jit(gp_perp32.r2c).lower(real6_yx).compiler_ir(dialect="stablehlo")),
        "GradPerp.r2c(6D)",
    )
    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda r, exb: gp_perp32.phase_mult_nxny(
                    r,
                    exb_shear=True,
                    phasefac_exb_nxny=exb.phasefac_exb_nxny,
                    phasefacminus_exb_nxny=exb.phasefacminus_exb_nxny,
                )
            )
            .lower(real6_nxny, st_p32.exb_state)
            .compiler_ir(dialect="stablehlo")
        ),
        "GradPerp.phase_mult_nxny(6D)",
    )

    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda g, f: linear_rhs(
                    g, f, gk_p32.linear_op, jnp.float32(0.005), use_fused_fft=True
                )
            )
            .lower(st_p32.G, st_p32.fields)
            .compiler_ir(dialect="stablehlo")
        ),
        "linear_rhs(fused, ParallelGrad)",
    )
    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda g, f: linear_rhs(
                    g, f, gk_p32.linear_op, jnp.float32(0.005), use_fused_fft=False
                )
            )
            .lower(st_p32.G, st_p32.fields)
            .compiler_ir(dialect="stablehlo")
        ),
        "linear_rhs(unfused, ParallelGrad)",
    )
    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda g, f, exb: nonlinear_rhs(
                    g,
                    f,
                    gk_p32.nonlinear_op,
                    exb_state=exb,  # type: ignore[arg-type]
                )
            )
            .lower(st_p32.G, st_p32.fields, st_p32.exb_state)
            .compiler_ir(dialect="stablehlo")
        ),
        "nonlinear_rhs(EM + ExB)",
    )
    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda g, f, exb: gyrokinetic_rhs(
                    g,
                    f,
                    gk_p32,
                    jnp.float32(0.005),
                    exb_state=exb,
                    apply_kz_dealias_linear=True,
                    apply_kz_dealias_nonlinear=True,
                )
            )
            .lower(st_p32.G, st_p32.fields, st_p32.exb_state)
            .compiler_ir(dialect="stablehlo")
        ),
        "gyrokinetic_rhs(ParallelGrad + dealias_kz)",
    )

    # 2. Multi-link LinkedParallelGrad + nonlinear_rhs
    cfg_lnk = replace(
        cfg_p,
        nx=10,
        ny=10,
        ntheta_in=8,
        nz=8,
        boundary="linked",
        boundary_option_periodic=False,
        closure_model="smith_par",
        smith_par_q=2,
    )
    cfg_lnk, grids_lnk, geo_lnk = init_geometry(cfg_lnk)
    gk_lnk32 = init_gyrokinetic_operator(cfg_lnk, grids_lnk, geo_lnk, dtype=jnp.float32)
    assert isinstance(gk_lnk32.linear_op.grad_par, LinkedParallelGrad)
    g_lnk32 = _random_g(grids_lnk, seed=702, dtype=jnp.complex64)
    st_lnk32 = init_integrator_state(g_lnk32, gk_lnk32, time=0.0, dt=0.005)
    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda g, f, exb: gyrokinetic_rhs(g, f, gk_lnk32, jnp.float32(0.005), exb_state=exb)
            )
            .lower(st_lnk32.G, st_lnk32.fields, st_lnk32.exb_state)
            .compiler_ir(dialect="stablehlo")
        ),
        "gyrokinetic_rhs(LinkedParallelGrad + EM + ExB)",
    )

    # 3. NTFTParallelGrad + NTFT nonlinear_rhs
    cfg_nt = replace(cfg_lnk, non_twist=True)
    cfg_nt, grids_nt, geo_nt = init_geometry(cfg_nt)
    gk_nt32 = init_gyrokinetic_operator(cfg_nt, grids_nt, geo_nt, dtype=jnp.float32)
    assert isinstance(gk_nt32.linear_op.grad_par, NTFTParallelGrad)
    g_nt32 = _random_g(grids_nt, seed=703, dtype=jnp.complex64)
    st_nt32 = init_integrator_state(g_nt32, gk_nt32, time=0.0, dt=0.005)
    _assert_mxu_stablehlo_clean(
        str(
            jax.jit(
                lambda g, f, exb: gyrokinetic_rhs(g, f, gk_nt32, jnp.float32(0.005), exb_state=exb)
            )
            .lower(st_nt32.G, st_nt32.fields, st_nt32.exb_state)
            .compiler_ir(dialect="stablehlo")
        ),
        "gyrokinetic_rhs(NTFTParallelGrad + NTFT nonlinear_rhs + EM + ExB)",
    )
