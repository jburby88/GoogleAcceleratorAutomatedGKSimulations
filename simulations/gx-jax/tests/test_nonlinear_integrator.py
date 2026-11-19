"""Nonlinear 3/2-Dealiased ExB Brackets & Explicit Time Integrators.

Verifies ``gx_jax/nonlinear_exb.py`` and ``gx_jax/integrator.py`` against:
- All C++ GoogleTests in ``gx/unit_tests/src/{test_grad_perp,test_nonlinear}.cu``.
- Independent imperative C++-translated NumPy reference implementations of ``GradPerp``,
  ``Nonlinear_GK::nlps``, ``Nonlinear_GK::get_max_frequency``, ``ExB_GK``, and all five explicit
  Runge-Kutta steppers (``sspx2``, ``rk3``, ``sspx3``, ``rk4``, ``k10``) to ``< 1e-12`` in
  ``complex128`` and ``< 1e-6`` in runtime ``complex64``.
- Exact ``3/2``-rule dealiasing on out-of-band quadratic triads, Poisson-bracket skew-symmetry,
  inviscid gyrokinetic free-energy conservation across electrostatic and electromagnetic
  ``(phi, apar, bpar)`` regimes, and ``iky = 0`` reality condition preservation.
- 3D **W7-X** stellarator nonlinear evaluation in both multi-link twist-and-shift and
  Non-Twisting Flux-Tube (``non_twist = True``, ``phasefac_ntft`` / ``phasefacminus_ntft``)
  modes.
- **Nonlinear ExB & Integrator Gate**: differential verification of ``3/2``-dealiased ``ExB`` +
  ``lax.while_loop`` / ``lax.scan`` 10-step ``G(10*dt)`` and full 400-step Kelvin-Helmholtz
  secondary-instability evolution against C++ golden ``kh01_restart.nc`` -> ``kh01a_restart.nc``
  and ``kh01a.nc`` to ``rtol < 1e-5``.
- Temporal convergence orders ``(2, 3, 3, 4, 4)`` across all 5 explicit schemes, adaptive CFL
  ``lax.while_loop`` stepping, single-trace ``@jax.jit`` compilation, and reverse-mode
  ``jax.grad`` under ``jax.debug_nans(True)``.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest
from scipy import special as sp_special

import gx_jax
from gx_jax.config import GXConfig, SpeciesConfig, load_config
from gx_jax.fields import (
    Fields,
    enforce_reality,
    init_field_solver,
    init_fields,
    load_restart_nc,
    mask_g,
    solve_fields,
)
from gx_jax.geometry_nc import Geometry, build_vmec_eik_nc, init_geometry, load_eik_nc
from gx_jax.grids import Grids, pad_kykx_to_full, truncate_full_to_kykx
from gx_jax.integrator import (
    IntegratorState,
    _solve_stage_fields,
    _update_linear_op_for_exb,
    compute_cfl_dt,
    get_scheme_cfl_fac,
    gyrokinetic_rhs,
    init_gyrokinetic_operator,
    init_integrator_state,
    integrate_scan,
    integrate_while_loop,
    step_gyrokinetic,
    step_integrator_state,
)
from gx_jax.laguerre import LaguerreTransform
from gx_jax.linear_rhs import compute_growth_rates, init_linear_operator
from gx_jax.nonlinear_exb import (
    _bessel_j0_runtime,
    _j1_over_alpha_runtime,
    apply_forcing,
    compute_nonlinear_max_frequency,
    flow_shear_g_shift,
    flow_shear_shift,
    init_exb_shear,
    init_forcing,
    init_grad_perp,
    init_nonlinear_operator,
    nonlinear_rhs,
    poisson_bracket_real,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
UNIT_INPUTS_DIR = REPO_ROOT / "golden_fixtures" / "unit_tests" / "inputs"


@pytest.fixture(scope="module")
def w7x_eik_nc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build shared W7-X .eik.nc geometry file once for Session 5 tests."""
    out_dir = tmp_path_factory.mktemp("w7x_s5")
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    w7x_cfg = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    return build_vmec_eik_nc(
        dataclasses.replace(w7x_cfg, ntheta_in=16, nz=16, npol=1.0),
        w7x_dir / "wout_w7x.nc",
        out_dir / "w7x_nz16.eik.nc",
    )


def _assert_close_fp64(
    actual: np.ndarray | jax.Array | float | complex,
    expected: np.ndarray | jax.Array | float | complex,
    label: str,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> None:
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    np.testing.assert_allclose(
        np.asarray(actual, dtype=dt),
        np.asarray(expected, dtype=dt),
        rtol=rtol,
        atol=atol,
        err_msg=f"FP64 mismatch in {label}",
    )


def _assert_close_fp32_1e6(
    actual: np.ndarray | jax.Array | float | complex,
    expected: np.ndarray | jax.Array | float | complex,
    label: str,
    *,
    tol: float = 1.0e-6,
) -> None:
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    act = np.asarray(actual, dtype=dt)
    exp = np.asarray(expected, dtype=dt)
    exp_norm = float(np.linalg.norm(exp.ravel()))
    if exp_norm > 1.0e-12:
        rel_l2 = float(np.linalg.norm((act - exp).ravel())) / exp_norm
        assert rel_l2 < tol, f"{label}: relative L2 error {rel_l2:.3e} >= {tol:.1e}"
    scale = max(1.0, float(np.max(np.abs(exp)))) if exp.size > 0 else 1.0
    np.testing.assert_allclose(
        act,
        exp,
        rtol=max(2.0e-6, 2.0 * tol),
        atol=max(1.0e-6, tol) * scale,
        err_msg=f"FP32 mismatch in {label}",
    )


def _random_reality_g(
    grids: Grids,
    seed: int,
    *,
    dtype: jnp.dtype = jnp.complex128,
    scale: float = 1.0,
) -> jax.Array:
    """Generate a random distribution ``G`` satisfying ``iky=0`` reality & ``(0,0)`` masking."""
    rng = np.random.default_rng(seed)
    raw = scale * (rng.standard_normal(grids.g_shape) + 1.0j * rng.standard_normal(grids.g_shape))
    g = jnp.asarray(raw, dtype=dtype)
    return enforce_reality(mask_g(g, grids), grids)


# ---------------------------------------------------------------------------
# Independent Imperative NumPy Reference Port of C++ nonlinear.cu & grad_perp.cu
# ---------------------------------------------------------------------------


def _cpp_pad_to_full_np(arr_act: np.ndarray, grids: Grids) -> np.ndarray:
    mask = np.asarray(grids.active_mask, dtype=bool)
    masked = np.where(mask, arr_act, 0.0)
    out = np.zeros((*arr_act.shape[:-2], grids.nyc, grids.nx), dtype=np.complex128)
    idx_kx = np.asarray(grids.active_kx_idx, dtype=np.int64)
    out[..., : grids.naky, idx_kx] = masked
    return out


def _cpp_truncate_to_active_np(arr_full: np.ndarray, grids: Grids) -> np.ndarray:
    mask = np.asarray(grids.active_mask, dtype=bool)
    idx_kx = np.asarray(grids.active_kx_idx, dtype=np.int64)
    sliced = arr_full[..., : grids.naky, :][..., idx_kx]
    return np.where(mask, sliced, 0.0)


def _cpp_c2r_np(arr_full: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.fft.irfft2(arr_full, s=(nx, ny), axes=(-1, -2)) * float(nx * ny)


def _cpp_r2c_np(arr_real: np.ndarray, grids: Grids) -> np.ndarray:
    spec_full = np.fft.rfft2(arr_real, s=(grids.nx, grids.ny), axes=(-1, -2)) / float(
        grids.nx * grids.ny
    )
    return _cpp_truncate_to_active_np(spec_full, grids)


def _cpp_phase_mult_np(
    arr_real: np.ndarray,
    grids: Grids,
    geo: Geometry,
    *,
    positive_phase: bool = True,
) -> np.ndarray:
    if geo.delta_kx.size == 0:
        return arr_real
    dkx = np.asarray(geo.delta_kx, dtype=np.float64)  # (nz, naky)
    x = np.asarray(grids.x, dtype=np.float64)  # (nx,)
    sgn = 1.0 if positive_phase else -1.0
    pf = np.ones((grids.nz, grids.nyc, grids.nx), dtype=np.complex128)
    pf[:, : grids.naky, :] = np.exp(1.0j * sgn * dkx[:, :, None] * x[None, None, :])
    yk = np.fft.rfft(arr_real, axis=-2)
    return np.fft.irfft(yk * pf, n=grids.ny, axis=-2)


def _cpp_reference_nonlinear_rhs(
    G_in: np.ndarray,
    fields: Fields,
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    lag: LaguerreTransform,
) -> np.ndarray:
    """Line-by-line NumPy port of ``Nonlinear_GK::nlps`` (``nonlinear.cu:L206-304``)."""
    g_np = np.asarray(G_in, dtype=np.complex128)
    if not cfg.nonlinear_mode:
        return np.zeros_like(g_np)

    ns, nm, nl, nz, naky, nakx = grids.g_shape
    nj = lag.nj
    nx, ny = grids.nx, grids.ny
    kx_full = np.asarray(grids.kx_full, dtype=np.float64)
    ky_full = np.asarray(grids.ky_full, dtype=np.float64)
    kperp2 = np.asarray(geo.kperp2, dtype=np.float64)
    roots = np.asarray(lag.roots, dtype=np.float64)
    to_grid = np.asarray(lag.to_grid, dtype=np.float64)  # (nj, nl)
    to_spec = np.asarray(lag.to_spectral, dtype=np.float64)  # (nl, nj)
    mask2d = np.asarray(grids.active_mask, dtype=bool)

    phi_np = np.where(mask2d[None, :, :], np.asarray(fields.phi, dtype=np.complex128), 0.0)
    apar_np = np.where(mask2d[None, :, :], np.asarray(fields.apar, dtype=np.complex128), 0.0)
    bpar_np = np.where(mask2d[None, :, :], np.asarray(fields.bpar, dtype=np.complex128), 0.0)

    out = np.zeros_like(g_np)
    dkx_full = np.zeros((nz, grids.nyc), dtype=np.float64)
    if cfg.non_twist and geo.delta_kx.size > 0:
        dkx_full[:, :naky] = np.asarray(geo.delta_kx, dtype=np.float64)

    for is_idx, sp in enumerate(cfg.species):
        g_s = g_np[is_idx]  # (nm, nl, nz, naky, nakx)
        g_full = _cpp_pad_to_full_np(g_s, grids)  # (nm, nl, nz, nyc, nx)

        if cfg.non_twist:
            ikx_op = 1.0j * (kx_full[None, None, :] + dkx_full[:, :, None])
            dG_dx = _cpp_c2r_np(ikx_op[None, None, ...] * g_full, nx, ny)
            dG_dx = _cpp_phase_mult_np(dG_dx, grids, geo, positive_phase=True)
        else:
            dG_dx = _cpp_c2r_np(1.0j * kx_full[None, None, None, None, :] * g_full, nx, ny)

        dG_dy = _cpp_c2r_np(1.0j * ky_full[None, None, None, :, None] * g_full, nx, ny)
        if cfg.non_twist:
            dG_dy = _cpp_phase_mult_np(dG_dy, grids, geo, positive_phase=True)

        # laguerre->transformToGrid: (nm, nl, nz, ny, nx) -> (nm, nj, nz, ny, nx)
        dg_dx = np.einsum("jl,mlzyx->mjzyx", to_grid, dG_dx)
        dg_dy = np.einsum("jl,mlzyx->mjzyx", to_grid, dG_dy)

        # J0phiAndBparToGrid / J0fToGrid on (nj, nz, naky, nakx)
        j0_phi = np.zeros((nj, nz, naky, nakx), dtype=np.complex128)
        for j in range(nj):
            alpha = np.sqrt(np.maximum(2.0 * roots[j] * kperp2 * sp.rho2, 0.0))
            j0_v = sp_special.j0(alpha)
            if cfg.fbpar > 0.0:
                a_safe = np.where(alpha < 1.0e-8, 1.0, alpha)
                j1_oa = np.where(alpha < 1.0e-8, 0.5, sp_special.j1(a_safe) / a_safe)
                j0_phi[j] = np.where(
                    mask2d[None, :, :],
                    j0_v * phi_np * cfg.fphi + sp.tz * 2.0 * roots[j] * j1_oa * bpar_np * cfg.fbpar,
                    0.0,
                )
            else:
                j0_phi[j] = np.where(mask2d[None, :, :], j0_v * phi_np * cfg.fphi, 0.0)

        j0_phi_full = _cpp_pad_to_full_np(j0_phi, grids)
        if cfg.non_twist:
            dj0phi_dx = _cpp_c2r_np(ikx_op[None, ...] * j0_phi_full, nx, ny)
            dj0phi_dx = _cpp_phase_mult_np(dj0phi_dx, grids, geo, positive_phase=True)
        else:
            dj0phi_dx = _cpp_c2r_np(1.0j * kx_full[None, None, None, :] * j0_phi_full, nx, ny)

        dj0phi_dy = _cpp_c2r_np(1.0j * ky_full[None, None, :, None] * j0_phi_full, nx, ny)
        if cfg.non_twist:
            dj0phi_dy = _cpp_phase_mult_np(dj0phi_dy, grids, geo, positive_phase=True)

        brk_phi = (dg_dx * dj0phi_dy[None, ...] - dg_dy * dj0phi_dx[None, ...]) * geo.kxfac
        dG_res = np.einsum("lj,mjzyx->mlzyx", to_spec, brk_phi)
        if cfg.non_twist:
            dG_res = _cpp_phase_mult_np(dG_res, grids, geo, positive_phase=False)
        res_s = _cpp_r2c_np(dG_res, grids)

        if cfg.fapar > 0.0:
            j0_apar = np.zeros((nj, nz, naky, nakx), dtype=np.complex128)
            for j in range(nj):
                alpha = np.sqrt(np.maximum(2.0 * roots[j] * kperp2 * sp.rho2, 0.0))
                j0_apar[j] = np.where(
                    mask2d[None, :, :], sp_special.j0(alpha) * apar_np * cfg.fapar, 0.0
                )
            j0_apar_full = _cpp_pad_to_full_np(j0_apar, grids)
            if cfg.non_twist:
                dj0a_dx = _cpp_c2r_np(ikx_op[None, ...] * j0_apar_full, nx, ny)
                dj0a_dx = _cpp_phase_mult_np(dj0a_dx, grids, geo, positive_phase=True)
            else:
                dj0a_dx = _cpp_c2r_np(1.0j * kx_full[None, None, None, :] * j0_apar_full, nx, ny)

            dj0a_dy = _cpp_c2r_np(1.0j * ky_full[None, None, :, None] * j0_apar_full, nx, ny)
            if cfg.non_twist:
                dj0a_dy = _cpp_phase_mult_np(dj0a_dy, grids, geo, positive_phase=True)

            brk_apar = (dg_dx * dj0a_dy[None, ...] - dg_dy * dj0a_dx[None, ...]) * geo.kxfac
            dG_apar = np.einsum("lj,mjzyx->mlzyx", to_spec, brk_apar)
            if cfg.non_twist:
                dG_apar = _cpp_phase_mult_np(dG_apar, grids, geo, positive_phase=False)
            nl_apar = _cpp_r2c_np(dG_apar, grids)

            for m in range(nm):
                term_mp1 = math.sqrt(m + 1) * nl_apar[m + 1] if (m + 1 < nm) else 0.0
                term_mm1 = math.sqrt(m) * nl_apar[m - 1] if (m > 0) else 0.0
                res_s[m] = res_s[m] - sp.vt * (term_mp1 + term_mm1)

        out[is_idx] = np.where(mask2d[None, None, None, :, :], res_s, 0.0)
    return out


def _cpp_reference_nonlinear_max_frequency(
    fields: Fields,
    cfg: GXConfig,
    grids: Grids,
    geo: Geometry,
    omega_lin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Line-by-line NumPy port of ``Nonlinear_GK::get_max_frequency`` (``nonlinear.cu``)."""
    nx, ny = grids.nx, grids.ny
    kx_full = np.asarray(grids.kx_full, dtype=np.float64)[None, None, :]
    ky_full = np.asarray(grids.ky_full, dtype=np.float64)[None, :, None]
    vpar_max = grids.vpar_max * cfg.vtmax
    mub_max = grids.muB_maxJ * cfg.tzmax

    phi_full = _cpp_pad_to_full_np(np.asarray(fields.phi, dtype=np.complex128), grids)
    apar_full = _cpp_pad_to_full_np(np.asarray(fields.apar, dtype=np.complex128), grids)
    bpar_full = _cpp_pad_to_full_np(np.asarray(fields.bpar, dtype=np.complex128), grids)

    dphi_dx = np.abs(_cpp_c2r_np(1.0j * kx_full * phi_full, nx, ny))
    if cfg.fapar > 0.0:
        dphi_dx = dphi_dx + vpar_max * np.abs(_cpp_c2r_np(1.0j * kx_full * apar_full, nx, ny))
    if cfg.fbpar > 0.0:
        dphi_dx = dphi_dx + mub_max * np.abs(_cpp_c2r_np(1.0j * kx_full * bpar_full, nx, ny))
    vmax_y = float(np.max(dphi_dx))

    dphi_dy = np.abs(_cpp_c2r_np(1.0j * ky_full * phi_full, nx, ny))
    if cfg.fapar > 0.0:
        dphi_dy = dphi_dy + vpar_max * np.abs(_cpp_c2r_np(1.0j * ky_full * apar_full, nx, ny))
    if cfg.fbpar > 0.0:
        dphi_dy = dphi_dy + mub_max * np.abs(_cpp_c2r_np(1.0j * ky_full * bpar_full, nx, ny))
    vmax_x = float(np.max(dphi_dy))

    scale = 0.5
    w0 = max(float(omega_lin[0]), abs(geo.kxfac) * (grids.kx_max * vmax_x) * scale)
    w1 = max(float(omega_lin[1]), abs(geo.kxfac) * (grids.ky_max * vmax_y) * scale)
    return (w0, w1, float(omega_lin[2]))


# ---------------------------------------------------------------------------
# 1. C++ GoogleTest: TestGradPerp.EvaluateDerivative (test_grad_perp.cu)
# ---------------------------------------------------------------------------


def test_cpp_unit_test_grad_perp_evaluate_derivative() -> None:
    """Port ``TEST_F(TestGradPerp, EvaluateDerivative)`` (``test_grad_perp.cu:L43-175``)."""
    base_cfg = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")
    cfg = dataclasses.replace(
        base_cfg,
        nx=8,
        ny=8,
        nz=4,
        ntheta_in=4,
        nperiod=1,
        zp=1,
        nspecies=1,
        nhermite=1,
        nlaguerre=4,
        x0=10.0,
        y0=10.0,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    gp64 = init_grad_perp(cfg, grids, geo, dtype=jnp.float64)
    gp32 = gp64.astype(jnp.float32)

    kx = 0.2
    ky = 0.1
    n_batch = grids.nz * grids.nlaguerre
    idx = np.arange(grids.nx, dtype=np.float64)
    idy = np.arange(grids.ny, dtype=np.float64)
    # test_grad_perp.cu:L65-69 uses shifted x, y centered at Nx/2, Ny/2
    x_test = cfg.x0 * 2.0 * math.pi * (idx - grids.nx // 2) / float(grids.nx)
    y_test = cfg.y0 * 2.0 * math.pi * (idy - grids.ny // 2) / float(grids.ny)
    phase = kx * x_test[None, :] + ky * y_test[:, None]  # (ny, nx)

    scales = np.arange(1, n_batch + 1, dtype=np.float64).reshape(grids.nlaguerre, grids.nz)
    init_real = scales[:, :, None, None] * np.sin(phase)[None, None, :, :]
    dx_check = scales[:, :, None, None] * kx * np.cos(phase)[None, None, :, :]
    dy_check = scales[:, :, None, None] * ky * np.cos(phase)[None, None, :, :]

    init_j64 = jnp.asarray(init_real, dtype=jnp.float64)
    init_j32 = init_j64.astype(jnp.float32)

    # 1. R2C without accumulate (full_output=True and active-mode output)
    comp_full = jax.jit(lambda a: gp64.r2c(a, accumulate=False, full_output=True))(init_j64)
    expected_full = np.zeros((grids.nlaguerre, grids.nz, grids.nyc, grids.nx), dtype=np.complex128)
    expected_full[:, :, 1, 2] = 0.5j * scales
    _assert_close_fp64(comp_full, expected_full, "TestGradPerp:R2C_no_acc")

    comp_full_32 = jax.jit(lambda a: gp32.r2c(a, accumulate=False, full_output=True))(init_j32)
    _assert_close_fp32_1e6(comp_full_32, expected_full, "TestGradPerp:R2C_no_acc_f32")

    # 2. R2C with accumulate=True
    zeros_full = jnp.zeros_like(comp_full)
    comp_acc = jax.jit(lambda a, t: gp64.r2c(a, accumulate=True, target=t, full_output=True))(
        init_j64, zeros_full
    )
    _assert_close_fp64(comp_acc, expected_full, "TestGradPerp:R2C_with_acc")

    comp_act = jax.jit(lambda a: gp64.r2c(a, accumulate=False, full_output=False))(init_j64)
    _assert_close_fp64(
        gp64.pad_to_full(comp_act),
        pad_kykx_to_full(comp_act, grids),
        "TestGradPerp:scatter_free_pad",
    )
    _assert_close_fp64(
        gp64.truncate_to_active(comp_full),
        truncate_full_to_kykx(comp_full, grids),
        "TestGradPerp:scatter_free_truncate",
    )

    # 3. dxC2R and dyC2R from both active (naky, nakx) and full (nyc, nx)
    dx_from_act = jax.jit(gp64.dx_c2r)(comp_act)
    dy_from_act = jax.jit(gp64.dy_c2r)(comp_act)
    dx_from_full = jax.jit(gp64.dx_c2r)(comp_full)
    dy_from_full = jax.jit(gp64.dy_c2r)(comp_full)

    _assert_close_fp64(dx_from_act, dx_check, "TestGradPerp:dxC2R_act")
    _assert_close_fp64(dy_from_act, dy_check, "TestGradPerp:dyC2R_act")
    _assert_close_fp64(dx_from_full, dx_check, "TestGradPerp:dxC2R_full")
    _assert_close_fp64(dy_from_full, dy_check, "TestGradPerp:dyC2R_full")

    dx32 = jax.jit(gp32.dx_c2r)(comp_act.astype(jnp.complex64))
    dy32 = jax.jit(gp32.dy_c2r)(comp_act.astype(jnp.complex64))
    _assert_close_fp32_1e6(dx32, dx_check, "TestGradPerp:dxC2R_f32")
    _assert_close_fp32_1e6(dy32, dy_check, "TestGradPerp:dyC2R_f32")


# ---------------------------------------------------------------------------
# 2. 3/2-Rule Dealiasing Exactness & Analytical Two-Mode Triad Verification
# ---------------------------------------------------------------------------


def test_3_2_dealiasing_exactness_and_analytical_triad_bracket() -> None:
    """Verify exact analytical triad coupling and zero aliasing from out-of-band sum modes."""
    base_cfg = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")
    # nx = 10, ny = 10 -> Kx = 3, Ky = 3 (nakx = 7, naky = 4, nyc = 6)
    cfg = dataclasses.replace(
        base_cfg,
        nx=10,
        ny=10,
        nz=4,
        ntheta_in=4,
        nperiod=1,
        zp=1,
        nspecies=1,
        nhermite=2,
        nlaguerre=2,
        x0=1.0,
        y0=1.0,
        kxfac=1.5,
        boundary="periodic",
        boundary_option_periodic=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    gp = init_grad_perp(cfg, grids, geo, dtype=jnp.float64)

    # Part A: In-band triad p = (kx=1, ky=1) and q = (kx=1, ky=2) -> p + q = (kx=2, ky=3)
    # and difference p - q* -> (-kx + kx = 0, ky=1)
    # Let g(x, y) = 2 Re[A exp(i (p_x x + p_y y))], chi(x, y) = 2 Re[B exp(i (q_x x + q_y y))]
    a_amp = 0.35 - 0.2j
    b_amp = -0.15 + 0.4j
    g_spec = jnp.zeros((grids.nz, grids.naky, grids.nakx), dtype=jnp.complex128)
    chi_spec = jnp.zeros((grids.nz, grids.naky, grids.nakx), dtype=jnp.complex128)
    g_spec = g_spec.at[:, 1, 1].set(a_amp)
    chi_spec = chi_spec.at[:, 2, 1].set(b_amp)

    dg_dx = gp.dx_c2r(g_spec)
    dg_dy = gp.dy_c2r(g_spec)
    dchi_dx = gp.dx_c2r(chi_spec)
    dchi_dy = gp.dy_c2r(chi_spec)
    brk_spec = gp.r2c(poisson_bracket_real(dg_dx, dg_dy, dchi_dx, dchi_dy, geo.kxfac))

    # Sum mode k = p + q = (kx=2, ky=3): bracket coefficient is -(p_x q_y - p_y q_x) * kxfac * A * B
    # because (i p_x A)(i q_y B) - (i p_y A)(i q_x B) = -(p_x q_y - p_y q_x) A B
    px, py = 1.0, 1.0
    qx, qy = 1.0, 2.0
    expected_sum = -(px * qy - py * qx) * geo.kxfac * a_amp * b_amp
    # Difference mode k = q - p = (kx=0, ky=1): from conj(A) at (-px, -py) and B at (qx, qy)
    # ((-i px) A*)(i qy B) - ((-i py) A*)(i qx B) = +(px qy - py qx) * conj(A) * B
    expected_diff = +(px * qy - py * qx) * geo.kxfac * np.conj(a_amp) * b_amp

    _assert_close_fp64(brk_spec[:, 3, 2], np.full(grids.nz, expected_sum), "triad:sum_mode")
    _assert_close_fp64(brk_spec[:, 1, 0], np.full(grids.nz, expected_diff), "triad:diff_mode")

    # Part B: Out-of-band sum mode p = (kx=2, ky=2) and q = (kx=3, ky=2) -> p + q = (5, 4) > (3, 3)
    # Without 3/2 padding on (ny=10, nx=10), kx=5 would alias onto kx = 5 - 10 = -5 (or if p=(2,2),
    # q=(2,2) -> kx=4 -> aliases onto kx = 4 - 7 = -3 on a non-padded 7-point grid!).
    # Specifically, on an unpadded nakx=7, naky=4 grid, kx=2 + kx=2 = 4 aliases to kx = -3 (active!)
    # whereas on the 3/2-padded nx=10, ny=10 grid, kx=4 is in the dealiasing band and is zeroed!
    g_high = jnp.zeros((grids.nz, grids.naky, grids.nakx), dtype=jnp.complex128)
    chi_high = jnp.zeros((grids.nz, grids.naky, grids.nakx), dtype=jnp.complex128)
    g_high = g_high.at[:, 2, 2].set(0.5 + 0.25j)  # (ky=2, kx=+2)
    chi_high = chi_high.at[:, 1, 3].set(-0.3 + 0.4j)  # (ky=1, kx=+3)
    # Sum mode has ky = 3 (in-band) and kx = +5 (out-of-band > Kx = 3).
    # Difference mode has ky = 2 - 1 = 1 and kx = 2 - 3 = -1 (active index nakx - 1 = 6).
    brk_high = gp.r2c(
        poisson_bracket_real(
            gp.dx_c2r(g_high),
            gp.dy_c2r(g_high),
            gp.dx_c2r(chi_high),
            gp.dy_c2r(chi_high),
            geo.kxfac,
        )
    )
    # Every mode on ky = 3 MUST be identically zero (no aliasing of kx = +5 onto kx = -2 or -3!)
    assert float(jnp.max(jnp.abs(brk_high[:, 3, :]))) < 1.0e-14
    # Meanwhile the physical in-band difference mode at (ky=1, kx=-1 -> idx=6) is exact:
    exp_diff_high = (2.0 * 1.0 - 2.0 * 3.0) * geo.kxfac * (0.5 + 0.25j) * np.conj(-0.3 + 0.4j)
    _assert_close_fp64(brk_high[:, 1, 6], np.full(grids.nz, exp_diff_high), "dealias:inband_diff")


# ---------------------------------------------------------------------------
# 3. C++ GoogleTests in test_nonlinear.cu & Multi-Regime Differential Checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_label", "nspecies", "all_kinetic", "beta", "fphi", "fapar", "fbpar", "lw_gk"),
    [
        ("test_nonlinear_ES_adiab", 1, False, 0.0, 1.0, 0.0, 0.0, False),
        ("test_nonlinear_EM_1sp", 1, True, 0.02, 1.0, 1.0, 1.0, False),
        ("em_2sp_phi_apar", 2, True, 0.015, 1.0, 1.0, 0.0, False),
        ("em_2sp_phi_bpar", 2, True, 0.015, 1.0, 0.0, 1.0, False),
        ("em_2sp_full_all", 2, True, 0.03, 1.0, 1.0, 1.0, False),
        ("em_2sp_fractional", 2, True, 0.025, 0.8, 0.6, 0.4, False),
        ("long_wavelength_gk", 2, True, 0.02, 1.0, 1.0, 1.0, True),
    ],
)
def test_cpp_nonlinear_rhs_and_max_frequency_differential(
    case_label: str,
    nspecies: int,
    all_kinetic: bool,
    beta: float,
    fphi: float,
    fapar: float,
    fbpar: float,
    lw_gk: bool,
) -> None:
    """Port ``test_nonlinear.cu`` (``nlps_ES``, ``nlps_EM``, ``get_max_frequency``)."""
    cyc_raw = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")
    ion = SpeciesConfig.from_raw(
        z=1.0,
        mass=1.0,
        dens=1.0,
        temp=1.0,
        tprim=2.49,
        fprim=0.8,
        nu_ss=0.000167,
        stype="ion",
        beta=beta,
        long_wavelength_gk=lw_gk,
    )
    elec = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=2.49,
        fprim=0.8,
        nu_ss=0.01,
        stype="electron",
        beta=beta,
        long_wavelength_gk=lw_gk,
    )
    species = (ion,) if nspecies == 1 else (ion, elec)
    cfg = dataclasses.replace(
        cyc_raw,
        nx=10,
        ny=10,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=6,
        nlaguerre=4,
        nspecies=nspecies,
        species=species,
        all_kinetic=all_kinetic,
        add_boltzmann_species=(not all_kinetic),
        beta=beta,
        fphi=fphi,
        fapar=fapar,
        fbpar=fbpar,
        nonlinear_mode=True,
        linear=False,
        long_wavelength_gk=lw_gk,
    )
    cfg, grids, geo = init_geometry(cfg)
    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    solver32 = solver64.astype(jnp.float32)
    nl64 = init_nonlinear_operator(cfg, grids, geo, dtype=jnp.float64)
    nl32 = nl64.astype(jnp.float32)

    g128 = _random_reality_g(grids, seed=501 + nspecies, dtype=jnp.complex128)
    f128 = solve_fields(g128, solver64)
    g64 = g128.astype(jnp.complex64)
    f64 = solve_fields(g64, solver32)

    rhs_ref = _cpp_reference_nonlinear_rhs(np.asarray(g128), f128, cfg, grids, geo, nl64.laguerre)
    rhs128 = jax.jit(nonlinear_rhs)(g128, f128, nl64)
    rhs64 = jax.jit(nonlinear_rhs)(g64, f64, nl32)

    _assert_close_fp64(rhs128, rhs_ref, f"{case_label}:nonlinear_rhs_fp64")
    _assert_close_fp32_1e6(rhs64, rhs_ref, f"{case_label}:nonlinear_rhs_c64")
    assert rhs64.dtype == jnp.complex64

    # Verify Nonlinear_GK::get_max_frequency (test_nonlinear.cu:L159-176)
    w_lin = (0.05, 0.08, 0.25)
    w_ref = _cpp_reference_nonlinear_max_frequency(f128, cfg, grids, geo, omega_lin=w_lin)
    w128 = jax.jit(lambda f: compute_nonlinear_max_frequency(f, nl64, omega_linear=w_lin))(f128)
    w64 = jax.jit(lambda f: compute_nonlinear_max_frequency(f, nl32, omega_linear=w_lin))(f64)

    for idx_w in range(3):
        _assert_close_fp64(float(w128[idx_w]), w_ref[idx_w], f"{case_label}:wmax[{idx_w}]_fp64")
        _assert_close_fp32_1e6(float(w64[idx_w]), w_ref[idx_w], f"{case_label}:wmax[{idx_w}]_c64")


# ---------------------------------------------------------------------------
# 4. Skew-Symmetry, Inviscid Free-Energy Conservation, & Reality Symmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fphi", "fapar", "fbpar"),
    [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_poisson_bracket_skew_symmetry_inviscid_free_energy_and_reality(
    fphi: float,
    fapar: float,
    fbpar: float,
) -> None:
    """Verify skew-symmetry, inviscid free-energy conservation, and iky=0 reality preservation."""
    kbm_cfg = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    cfg = dataclasses.replace(
        kbm_cfg,
        nx=10,
        ny=10,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=6,
        nlaguerre=4,
        fphi=fphi,
        fapar=fapar,
        fbpar=fbpar,
        nonlinear_mode=True,
        linear=False,
    )
    cfg, grids, geo = init_geometry(cfg)
    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    nl64 = init_nonlinear_operator(cfg, grids, geo, dtype=jnp.float64)

    g128 = _random_reality_g(grids, seed=610, dtype=jnp.complex128)
    f128 = solve_fields(g128, solver64)
    rhs_nl = jax.jit(nonlinear_rhs)(g128, f128, nl64)

    # 1. iky=0 reality condition: rhs_nl(..., 0, -kx) == conj(rhs_nl(..., 0, +kx))
    _assert_close_fp64(
        rhs_nl,
        enforce_reality(rhs_nl, grids),
        f"reality({fphi},{fapar},{fbpar})",
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    # 2. Inviscid G-variance conservation:
    #    sum_{s,m,l,z,ky,kx} w_ky vol_fac n_s T_s Re[G* NL(G)] == 0
    # Because {g_m, <chi>} conserves L2 norm on every mu_j node and Hermite m +- 1 flutter
    # telescopes via {g_m, <A_par>} g_{m+1} + {g_{m+1}, <A_par>} g_m = {g_m g_{m+1}, <A_par>},
    # the quadratic G-entropy production rate by the nonlinear bracket vanishes identically!
    fac_ky = jnp.where(jnp.arange(grids.naky) == 0, 1.0, 2.0)[None, None, None, None, :, None]
    vol = geo.vol_fac[None, None, None, :, None, None]
    nt = solver64.sp_nt[:, None, None, None, None, None]
    dwg_dt = float(jnp.sum(fac_ky * vol * nt * jnp.real(jnp.conj(g128) * rhs_nl)))
    wg_scale = float(jnp.sum(fac_ky * vol * nt * (jnp.abs(g128) ** 2)))
    assert abs(dwg_dt) / max(1.0, wg_scale) < 1.0e-13

    # 3. Full inviscid gyrokinetic free-energy conservation dW/dt|_NL == sum Re[H* NL(G)] == 0
    # In the drift-kinetic / long-wavelength limit (rho2 = 0 -> J_0 = 1, J_{l>0} = 0),
    # the potential cross-terms <phi, {G_00, phi}> and <apar, {G_00, apar}> also vanish
    # identically for arbitrary G and self-consistent (phi, apar, bpar)!
    sp_lw = tuple(
        SpeciesConfig.from_raw(
            z=sp.z,
            mass=sp.mass,
            dens=sp.dens,
            temp=sp.temp,
            tprim=sp.tprim,
            fprim=sp.fprim,
            nu_ss=sp.nu_ss,
            stype="ion" if sp.type_id == 0 else "electron",
            beta=cfg.beta,
            long_wavelength_gk=True,
        )
        for sp in cfg.species
    )
    cfg_lw = dataclasses.replace(cfg, long_wavelength_gk=True, species=sp_lw)
    solver_lw = init_field_solver(cfg_lw, grids, geo, dtype=jnp.float64)
    nl_lw = init_nonlinear_operator(cfg_lw, grids, geo, dtype=jnp.float64)
    f_lw = solve_fields(g128, solver_lw)
    rhs_lw = jax.jit(nonlinear_rhs)(g128, f_lw, nl_lw)
    h_lw = gx_jax.build_nonadiabatic_h(g128, f_lw, solver_lw)
    dw_tot_dt = float(jnp.sum(fac_ky * vol * nt * jnp.real(jnp.conj(h_lw) * rhs_lw)))
    assert abs(dw_tot_dt) / max(1.0, wg_scale) < 1.0e-13


# ---------------------------------------------------------------------------
# 5. 3D W7-X Stellarator Nonlinear Verification (Linked & NTFT phasefac_ntft)
# ---------------------------------------------------------------------------


def test_w7x_stellarator_nonlinear_linked_and_ntft_phasefac(w7x_eik_nc: Path) -> None:
    """Verify 3D W7-X stellarator nonlinear bracket & steppers in both Linked and NTFT modes."""
    w7x_nl_in = BENCH_DIR / "nonlinear" / "w7x" / "w7x_adiabatic_electrons.in"
    base_cfg = load_config(w7x_nl_in)

    # Part A: W7-X Linked twist-and-shift (adiabatic electrons)
    cfg_lnk = dataclasses.replace(
        base_cfg,
        geo_option="nc",
        geofilename=str(w7x_eik_nc),
        nx=10,
        ny=10,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=3,
        nonlinear_mode=True,
        linear=False,
    )
    cfg_lnk, grids_lnk, geo_lnk = load_eik_nc(w7x_eik_nc, cfg_lnk)
    gk_lnk64 = init_gyrokinetic_operator(cfg_lnk, grids_lnk, geo_lnk, dtype=jnp.float64)
    gk_lnk32 = gk_lnk64.astype(jnp.float32)

    g_lnk128 = _random_reality_g(grids_lnk, seed=701, dtype=jnp.complex128)
    f_lnk128 = solve_fields(g_lnk128, gk_lnk64.linear_op.solver)
    assert gk_lnk64.nonlinear_op is not None
    rhs_lnk_ref = _cpp_reference_nonlinear_rhs(
        np.asarray(g_lnk128), f_lnk128, cfg_lnk, grids_lnk, geo_lnk, gk_lnk64.nonlinear_op.laguerre
    )
    rhs_lnk128 = jax.jit(nonlinear_rhs)(g_lnk128, f_lnk128, gk_lnk64.nonlinear_op)
    rhs_lnk64 = jax.jit(nonlinear_rhs)(
        g_lnk128.astype(jnp.complex64),
        solve_fields(g_lnk128.astype(jnp.complex64), gk_lnk32.linear_op.solver),
        gk_lnk32.nonlinear_op,  # type: ignore[arg-type]
    )
    _assert_close_fp64(rhs_lnk128, rhs_lnk_ref, "w7x_linked:nonlinear_rhs_fp64")
    _assert_close_fp32_1e6(rhs_lnk64, rhs_lnk_ref, "w7x_linked:nonlinear_rhs_c64")

    # Part B: W7-X Non-Twisting Flux Tube (non_twist=True, 2-species electromagnetic)
    beta_em = 0.01
    ion = SpeciesConfig.from_raw(
        z=1.0, mass=1.0, dens=1.0, temp=1.0, tprim=3.0, fprim=1.0, stype="ion", beta=beta_em
    )
    elec = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=3.0,
        fprim=1.0,
        stype="electron",
        beta=beta_em,
    )
    cfg_ntft = dataclasses.replace(
        cfg_lnk,
        nspecies=2,
        species=(ion, elec),
        all_kinetic=True,
        add_boltzmann_species=False,
        beta=beta_em,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        non_twist=True,
    )
    cfg_ntft, grids_nt, geo_nt = load_eik_nc(w7x_eik_nc, cfg_ntft)

    # Check phasefac_ntft and phasefacminus_ntft on Geometry (geometry.cu:L938-939)
    assert geo_nt.phasefac_ntft.shape == (grids_nt.nz, grids_nt.naky, grids_nt.nx)
    assert geo_nt.phasefacminus_ntft.shape == (grids_nt.nz, grids_nt.naky, grids_nt.nx)
    _assert_close_fp64(
        geo_nt.phasefac_ntft * geo_nt.phasefacminus_ntft,
        np.ones_like(geo_nt.phasefac_ntft),
        "w7x_ntft:phasefac_unitary",
    )
    geo_nt32 = geo_nt.astype(jnp.float32)
    assert geo_nt32.phasefac_ntft.dtype == jnp.complex64

    gk_nt64 = init_gyrokinetic_operator(cfg_ntft, grids_nt, geo_nt, dtype=jnp.float64)
    gk_nt32 = gk_nt64.astype(jnp.float32)
    assert gk_nt64.nonlinear_op is not None
    gp_nt = gk_nt64.nonlinear_op.grad_perp

    # Verify phase_mult round-trip is exact identity on real space
    rng = np.random.default_rng(702)
    real_arr = jnp.asarray(
        rng.standard_normal((grids_nt.nz, grids_nt.ny, grids_nt.nx)), dtype=jnp.float64
    )
    rt_arr = gp_nt.phase_mult(gp_nt.phase_mult(real_arr, positive_phase=True), positive_phase=False)
    _assert_close_fp64(rt_arr, real_arr, "w7x_ntft:phase_mult_roundtrip")

    # Verify delta_kx == 0 limit of non_twist=True matches non_twist=False identically
    geo_zero_dkx = dataclasses.replace(
        geo_nt,
        delta_kx=jnp.zeros_like(geo_nt.delta_kx),
        phasefac_ntft=jnp.ones_like(geo_nt.phasefac_ntft),
        phasefacminus_ntft=jnp.ones_like(geo_nt.phasefacminus_ntft),
    )
    nl_zero_dkx = init_nonlinear_operator(cfg_ntft, grids_nt, geo_zero_dkx, dtype=jnp.float64)
    nl_std = init_nonlinear_operator(
        dataclasses.replace(cfg_ntft, non_twist=False), grids_nt, geo_zero_dkx, dtype=jnp.float64
    )
    g_nt128 = _random_reality_g(grids_nt, seed=703, dtype=jnp.complex128)
    f_nt128 = solve_fields(g_nt128, gk_nt64.linear_op.solver)
    _assert_close_fp64(
        nonlinear_rhs(g_nt128, f_nt128, nl_zero_dkx),
        nonlinear_rhs(g_nt128, f_nt128, nl_std),
        "w7x_ntft:zero_delta_kx_limit",
    )

    # Verify W7-X NTFT nonlinear_rhs against independent NumPy C++ reference
    rhs_nt_ref = _cpp_reference_nonlinear_rhs(
        np.asarray(g_nt128),
        f_nt128,
        cfg_ntft,
        grids_nt,
        geo_nt,
        gk_nt64.nonlinear_op.laguerre,
    )
    rhs_nt128 = jax.jit(nonlinear_rhs)(g_nt128, f_nt128, gk_nt64.nonlinear_op)
    _assert_close_fp64(rhs_nt128, rhs_nt_ref, "w7x_ntft:fp64")

    rhs_nt64 = jax.jit(nonlinear_rhs)(
        g_nt128.astype(jnp.complex64),
        solve_fields(g_nt128.astype(jnp.complex64), gk_nt32.linear_op.solver),
        gk_nt32.nonlinear_op,  # type: ignore[arg-type]
    )
    _assert_close_fp32_1e6(rhs_nt64, rhs_nt_ref, "w7x_ntft:c64")

    # Multi-step W7-X NTFT trajectory integration via integrate_scan inside CFL limit
    traj_nt = integrate_scan(
        (0.01 * g_nt128).astype(jnp.complex64), None, gk_nt32, nsteps=3, dt=1.0e-4
    )
    assert traj_nt.G.dtype == jnp.complex64
    assert bool(jnp.all(jnp.isfinite(traj_nt.G)))


# ---------------------------------------------------------------------------
# 6. Nonlinear ExB & Integrator Gate: Golden C++ kh01a .restart.nc & 10-Step Gate
# ---------------------------------------------------------------------------


def _glibc_kh01a_initial_perturbation(cfg: GXConfig, grids: Grids) -> np.ndarray:
    """Replicate ``MomentsG::initialConditions`` (``moments.cu:L161-202``) with ``srand(22)``."""
    seed = int(cfg.random_seed)
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.srand(seed)
        next_rand = lambda: int(libc.rand())  # noqa: E731
    except OSError:
        # Mirror glibc TYPE_3 linear additive-feedback PRNG on non-glibc hosts (e.g. macOS).
        state = [0] * 31
        state[0] = seed & 0xFFFFFFFF
        for k in range(1, 31):
            prev = state[k - 1]
            if prev >= 0x80000000:
                prev -= 0x100000000
            hi = int(prev / 127773)
            lo = prev - hi * 127773
            x = 16807 * lo - 2836 * hi
            if x < 0:
                x += 2147483647
            state[k] = x & 0xFFFFFFFF
        fptr, rptr = 3, 0

        def next_rand() -> int:
            nonlocal fptr, rptr
            state[fptr] = (state[fptr] + state[rptr]) & 0xFFFFFFFF
            val = (state[fptr] >> 1) & 0x7FFFFFFF
            fptr = (fptr + 1) % 31
            rptr = (rptr + 1) % 31
            return val

        for _ in range(310):
            next_rand()

    rand_max = 2147483647
    init_pert = np.zeros(grids.g_shape, dtype=np.complex128)
    samp = float(cfg.init_amp)
    for i in range(1 + (grids.nx - 1) // 3):
        for j in range(1, 1 + (grids.ny - 1) // 3):
            r1 = next_rand()
            r2 = next_rand()
            ra = np.float32(samp * (r1 - rand_max // 2) / float(rand_max))
            rb = np.float32(samp * (r2 - rand_max // 2) / float(rand_max))
            for js in range(2):
                idx_act = 0 if i == 0 else (i if js == 0 else grids.nakx - i)
                val = (ra + 1.0j * rb) if js == 0 else (rb + 1.0j * ra)
                init_pert[0, 0, 0, :, j, idx_act] = val
    return init_pert


def test_golden_cpp_kh01a_nonlinear_restart_and_10step_gate() -> None:
    """Verify 10-step & full 400-step nonlinear ExB evolution vs golden C++ kh01a_restart.nc."""
    cfg_kh, grids_kh, geo_kh = init_geometry(load_config(UNIT_INPUTS_DIR / "kh01a.in"))
    assert cfg_kh.nonlinear_mode is True
    assert cfg_kh.scheme == "sspx3"
    assert cfg_kh.restart is True
    assert cfg_kh.restart_with_perturb is True
    assert cfg_kh.restart_scale == pytest.approx(500.0)

    gk64 = init_gyrokinetic_operator(cfg_kh, grids_kh, geo_kh, dtype=jnp.float64)
    gk32 = gk64.astype(jnp.float32)

    # 1. Load kh01_restart.nc (scaled by 500) + C++ glibc srand(22) perturbation at t = 0
    g_base, t0 = load_restart_nc(
        UNIT_INPUTS_DIR / "kh01_restart.nc",
        grids_kh,
        dtype=jnp.complex128,
        scale=cfg_kh.restart_scale,
    )
    assert t0 == pytest.approx(0.0, abs=1.0e-15)
    g0_128 = g_base + jnp.asarray(_glibc_kh01a_initial_perturbation(cfg_kh, grids_kh))
    g0_64 = g0_128.astype(jnp.complex64)

    # 2. 10-step G(10*dt) c64-vs-c128 self-consistency (lax.scan vs lax.while_loop)
    g_rest_t4_128, t4 = load_restart_nc(
        UNIT_INPUTS_DIR / "kh01a_restart.nc", grids_kh, dtype=jnp.complex128
    )
    assert t4 == pytest.approx(4.0, rel=1.0e-6)
    g_rest_t4_64 = g_rest_t4_128.astype(jnp.complex64)

    dt_kh = cfg_kh.dt  # 0.01
    scan_10_128 = integrate_scan(g_rest_t4_128, None, gk64, nsteps=10, dt=dt_kh)
    scan_10_64 = integrate_scan(g_rest_t4_64, None, gk32, nsteps=10, dt=dt_kh)
    while_10_64 = integrate_while_loop(
        g_rest_t4_64,
        None,
        gk32,
        t_max=10.0 * dt_kh,
        max_steps=10,
        dt_init=dt_kh,
        clamp_to_t_max=True,
    )
    _assert_close_fp32_1e6(scan_10_64.G, scan_10_128.G, "kh01a_10step:scan_c64_vs_fp64")
    _assert_close_fp32_1e6(while_10_64.G, scan_10_128.G, "kh01a_10step:while_c64_vs_fp64")
    assert int(while_10_64.step) == 10
    assert float(while_10_64.time) == pytest.approx(0.1, rel=1.0e-12)

    with netCDF4.Dataset(UNIT_INPUTS_DIR / "kh01a.nc", "r") as ds_kh:
        omega_ref_all = np.asarray(ds_kh["Special"]["omega_v_time"][:], dtype=np.float64)
        wkxkyst_ref = np.asarray(ds_kh["Spectra"]["Wkxkyst"][:, 0], dtype=np.float64)
        wst_ref = np.asarray(ds_kh["Spectra"]["Wst"][:, 0], dtype=np.float64)

    # Confirm bit-exact initial perturbation seed at t = 0 against kh01a.nc Wkxkyst[0]
    w0_sideband_jax = float(jnp.mean(jnp.abs(g0_128[0, 0, 0, :, 1, 1]) ** 2))
    assert w0_sideband_jax == pytest.approx(float(wkxkyst_ref[0, 1, 2]), rel=1.0e-7)

    # 3. Full 400-step trajectory from kh01_restart.nc (t=0) -> kh01a_restart.nc (t=4.0)
    traj_400 = integrate_scan(g0_64, None, gk32, nsteps=400, dt=dt_kh, save_every=100)
    rel_l2_400 = float(jnp.linalg.norm(traj_400.G - g_rest_t4_64) / jnp.linalg.norm(g_rest_t4_64))
    assert rel_l2_400 < 1.0e-5, f"400-step kh01a rel_l2={rel_l2_400:.3e} >= 1e-5"
    np.testing.assert_allclose(np.asarray(traj_400.wg_history), wst_ref[1:], rtol=1.0e-5)

    # Per-mode primary streamer (ky=0.1, kx=0) and secondary KH modes (ky in {0, 0.1}, kx=+-0.05)
    # Over 400 steps (~19.6 e-foldings, 1e17 energy growth), the legacy fixture's -2.1e-5
    # growth-rate offset yields ~5.3e-4 per-mode amplitude difference at t = 4.0.
    np.testing.assert_allclose(
        np.asarray(traj_400.G[:, :, :, :, 1, 0]),
        np.asarray(g_rest_t4_64[:, :, :, :, 1, 0]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    for iky_sec in (0, 1):
        for ikx_sec in (1, 2):
            sec_rel = float(
                jnp.linalg.norm(
                    traj_400.G[0, 0, 0, :, iky_sec, ikx_sec]
                    - g_rest_t4_64[0, 0, 0, :, iky_sec, ikx_sec]
                )
                / jnp.linalg.norm(g_rest_t4_64[0, 0, 0, :, iky_sec, ikx_sec])
            )
            assert sec_rel < 7.0e-4, f"Mode ({iky_sec},{ikx_sec}) sec_rel={sec_rel:.3e}"

    f_t4 = solve_fields(g_rest_t4_128, gk64.linear_op.solver)
    _, f_t4_next, _, _ = step_gyrokinetic(g_rest_t4_128, f_t4, gk64, dt=dt_kh)
    omg_t4 = compute_growth_rates(f_t4_next.phi, f_t4.phi, dt_kh, grids_kh)
    gamma_zonal_cpp = float(omega_ref_all[-1, 0, 2, 1])  # 4.9020333
    gamma_sideband_cpp = float(omega_ref_all[-1, 1, 2, 1])  # 4.9020791
    assert float(jnp.imag(omg_t4[0, 1])) == pytest.approx(gamma_zonal_cpp, rel=6.0e-5)
    assert float(jnp.imag(omg_t4[1, 1])) == pytest.approx(gamma_sideband_cpp, rel=6.0e-5)

    # Also verify integrate_while_loop reaches t = 4.0 in 400 steps and matches kh01a_restart.nc
    while_400 = integrate_while_loop(
        g0_64, None, gk32, t_max=4.0, max_steps=400, dt_init=dt_kh, clamp_to_t_max=True
    )
    assert int(while_400.step) == 400
    assert float(while_400.time) == pytest.approx(4.0, rel=1.0e-12)
    rel_l2_while = float(
        jnp.linalg.norm(while_400.G - g_rest_t4_64) / jnp.linalg.norm(g_rest_t4_64)
    )
    assert rel_l2_while < 1.0e-5


def test_golden_cpp_z_restart_and_saved_nc_fixtures() -> None:
    """Verify golden C++ z_restart.nc (k10 + smith_par) and restart_tests/saved.nc evolution."""
    # 1. z.in + z_restart.nc (linear k10 with SmithPar q=2 closure at t = 50)
    cfg_z, grids_z, geo_z = init_geometry(load_config(UNIT_INPUTS_DIR / "z.in"))
    g_z, t_z = load_restart_nc(UNIT_INPUTS_DIR / "z_restart.nc", grids_z, dtype=jnp.complex64)
    assert t_z == pytest.approx(50.0, rel=1.0e-6)
    gk_z = init_gyrokinetic_operator(cfg_z, grids_z, geo_z, dtype=jnp.float32)
    traj_z = integrate_scan(g_z, None, gk_z, nsteps=10, dt=cfg_z.dt)
    assert traj_z.G.shape == grids_z.g_shape
    assert bool(jnp.all(jnp.isfinite(traj_z.G)))

    # 2. restart_tests/saved.nc (Nx=4, Nz=6, Nkx=3, Nky=2, Nl=10, Nm=30, Ns=1)
    cfg_saved = dataclasses.replace(
        load_config(UNIT_INPUTS_DIR / "kh01a.in"),
        nx=4,
        ny=4,
        nz=6,
        ntheta_in=6,
        nperiod=1,
        zp=1,
        nhermite=30,
        nlaguerre=10,
        nspecies=1,
        nonlinear_mode=True,
        linear=False,
    )
    cfg_saved, grids_saved, geo_saved = init_geometry(cfg_saved)
    g_saved, t_saved = load_restart_nc(
        UNIT_INPUTS_DIR / "restart_tests" / "saved.nc",
        grids_saved,
        dtype=jnp.complex128,
    )
    assert t_saved == pytest.approx(0.2, rel=1.0e-6)
    gk_saved64 = init_gyrokinetic_operator(cfg_saved, grids_saved, geo_saved, dtype=jnp.float64)
    gk_saved32 = gk_saved64.astype(jnp.float32)
    s64 = integrate_scan(g_saved.astype(jnp.complex64), None, gk_saved32, nsteps=5, dt=1.0e-4)
    s128 = integrate_scan(g_saved, None, gk_saved64, nsteps=5, dt=1.0e-4)
    _assert_close_fp32_1e6(s64.G, s128.G, "saved_nc:5step_c64_vs_fp64", tol=5.0e-6)


# ---------------------------------------------------------------------------
# 7. Temporal Convergence Orders (2, 3, 3, 4, 4) & Increment-Form Stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scheme", "expected_order", "min_order"),
    [
        ("sspx2", 2, 1.93),
        ("rk3", 3, 2.93),
        ("sspx3", 3, 2.93),
        ("rk4", 4, 3.90),
        ("k10", 4, 3.90),
    ],
)
def test_nonlinear_integrator_convergence_orders_all_five_schemes(
    scheme: str,
    expected_order: int,
    min_order: float,
) -> None:
    """Verify temporal convergence orders (2, 3, 3, 4, 4) on coupled nonlinear + linear GK ODE."""
    base_cfg = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")
    cfg = dataclasses.replace(
        base_cfg,
        nx=7,
        ny=7,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        nspecies=1,
        beta=0.01,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        nonlinear_mode=True,
        linear=False,
        boundary="periodic",
        boundary_option_periodic=True,
        damp_ends_amp=0.0,
        hypercollisions_const=False,
        hypercollisions_kz=True,
        scheme=scheme,
        fixed_dt=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float64)

    g0 = _random_reality_g(grids, seed=808, dtype=jnp.complex128, scale=0.08)
    t_total = 0.04

    # High-accuracy reference trajectory using 64 steps of RK4
    ref_traj = integrate_scan(g0, None, gk_op, nsteps=64, dt=t_total / 64.0, scheme="rk4")
    g_ref = ref_traj.G

    n_list = [4, 8, 16]
    errors: list[float] = []
    for n_steps in n_list:
        dt_cur = t_total / float(n_steps)
        traj = integrate_scan(g0, None, gk_op, nsteps=n_steps, dt=dt_cur, scheme=scheme)
        err = float(jnp.linalg.norm(traj.G - g_ref) / jnp.linalg.norm(g_ref))
        errors.append(err)

    order_1 = math.log2(errors[0] / errors[1])
    order_2 = math.log2(errors[1] / errors[2])
    # Require the finest-pair order_2 to meet min_order
    assert order_2 >= min_order, (
        f"Scheme {scheme} (expected order {expected_order}): "
        f"finest order={order_2:.2f} (coarse={order_1:.2f}), errors={errors}"
    )


def test_dealias_kz_rules_and_c64_increment_form_cancellation_free() -> None:
    """Verify per-scheme dealias_kz rules (sspx3, k10) and complex64 increment stability."""
    base_cfg = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")
    cfg = dataclasses.replace(
        base_cfg,
        nx=7,
        ny=7,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        boundary="periodic",
        boundary_option_periodic=True,
        dealias_kz=True,
        nonlinear_mode=True,
        linear=False,
        fixed_dt=True,
        dt=1.0e-4,
    )
    cfg, grids, geo = init_geometry(cfg)
    gk64 = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float64)
    gk32 = gk64.astype(jnp.float32)

    g0_128 = _random_reality_g(grids, seed=820, dtype=jnp.complex128, scale=0.5)
    g0_64 = g0_128.astype(jnp.complex64)
    f0_128 = solve_fields(g0_128, gk64.linear_op.solver)
    kzm_zero = np.asarray(grids.kzm) == 0
    assert bool(np.any(kzm_zero))

    # 1. Directly verify dealias_kz stage asymmetry (V10):
    #    - Euler stages of sspx3 and k10 (d_lin=True, d_nl=False): linear part has middle-1/3 kz
    #      zeroed while nonlinear part retains middle-1/3 kz content.
    #    - Stage 10 of k10 (d_lin=True, d_nl=True): total RHS has middle-1/3 kz zeroed.
    rhs_euler = gyrokinetic_rhs(
        g0_128,
        f0_128,
        gk64,
        1.0e-4,
        apply_kz_dealias_linear=True,
        apply_kz_dealias_nonlinear=False,
    )
    rhs_k10_s10 = gyrokinetic_rhs(
        g0_128,
        f0_128,
        gk64,
        1.0e-4,
        apply_kz_dealias_linear=True,
        apply_kz_dealias_nonlinear=True,
    )
    rhs_nl_only = nonlinear_rhs(g0_128, f0_128, gk64.nonlinear_op)  # type: ignore[arg-type]
    hat_euler = np.fft.fft(np.asarray(rhs_euler), axis=-3)
    hat_s10 = np.fft.fft(np.asarray(rhs_k10_s10), axis=-3)
    hat_nl = np.fft.fft(np.asarray(rhs_nl_only), axis=-3)
    # In Euler stages (d_lin=True, d_nl=False), middle-1/3 kz equals the undealiased NL part (> 0)
    assert float(np.max(np.abs(hat_nl[:, :, :, kzm_zero, :, :]))) > 1.0e-4
    _assert_close_fp64(
        hat_euler[:, :, :, kzm_zero, :, :],
        hat_nl[:, :, :, kzm_zero, :, :],
        "dealias_kz:euler_middle_kz_equals_nl",
    )
    # In k10 stage 10 (d_lin=True, d_nl=True), middle-1/3 kz is zero to FP64 FFT roundoff
    assert float(np.max(np.abs(hat_s10[:, :, :, kzm_zero, :, :]))) < 1.0e-12

    for sk in ("sspx2", "sspx3", "k10"):
        # Over 100 tiny steps (dt = 1e-4), naive state combinations in complex64 lose ~10 bits,
        # whereas increment form g0 + delta tracks complex128 to < 1e-6!
        res128 = integrate_scan(0.1 * g0_128, None, gk64, nsteps=100, dt=1.0e-4, scheme=sk)
        res64 = integrate_scan(0.1 * g0_64, None, gk32, nsteps=100, dt=1.0e-4, scheme=sk)
        _assert_close_fp32_1e6(res64.G, res128.G, f"increment_stability:{sk}")

        if sk in ("sspx3", "k10"):
            # Confirm dealias_kz zeroes the middle 1/3 kz modes on phi after stepping
            phi_hat_z = np.fft.fft(np.asarray(res128.fields.phi), axis=-3)
            assert float(np.max(np.abs(phi_hat_z[kzm_zero, :, :]))) < 1.0e-14


# ---------------------------------------------------------------------------
# 8. Adaptive CFL while_loop, Equilibrium ExB Flow Shear, & Forcing Hooks
# ---------------------------------------------------------------------------


def test_adaptive_cfl_while_loop_exb_flow_shear_and_forcing() -> None:
    """Verify adaptive CFL while_loop, ExB rk3/rk4 carry & stage parity, and Forcing PRNG."""
    base_cfg = load_config(UNIT_INPUTS_DIR / "cyc_nl.in")
    cfg = dataclasses.replace(
        base_cfg,
        nx=7,
        ny=7,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        nonlinear_mode=True,
        linear=False,
        fixed_dt=False,
        dt=0.05,
        dt_max=0.05,
        dt_min=1.0e-5,
        cfl=0.8,
        scheme="rk3",
        g_exb=0.25,
        exb_shear=True,
        exb_shear_phase=True,
        forcing_init=True,
        forcing_type="KzImpulse",
        forcing_amp=1.0e-4,
        forcing_index=1,
    )
    cfg, grids, geo = init_geometry(cfg)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float64)

    g0 = _random_reality_g(grids, seed=901, dtype=jnp.complex128, scale=0.5)
    f0 = solve_fields(g0, gk_op.linear_op.solver)
    exb0 = init_exb_shear(cfg, grids, geo, dtype=jnp.float64)

    # 1. Verify CFL dt scales inversely with field amplitude and honors scheme cfl_fac
    dt_small = float(compute_cfl_dt(f0, gk_op, scheme="rk3"))
    f_big = Fields(
        phi=50.0 * f0.phi,
        apar=50.0 * f0.apar,
        bpar=50.0 * f0.bpar,
        phi_avg=50.0 * f0.phi_avg,
    )
    dt_big_rk3 = float(compute_cfl_dt(f_big, gk_op, scheme="rk3"))
    dt_big_rk4 = float(compute_cfl_dt(f_big, gk_op, scheme="rk4"))
    assert dt_big_rk3 < dt_small
    assert (dt_big_rk4 / dt_big_rk3) == pytest.approx(
        get_scheme_cfl_fac("rk4") / get_scheme_cfl_fac("rk3"), rel=1.0e-6
    )

    # 1b. Verify fixed_dt is honored by rk3/rk4/sspx3 and ignored by sspx2/k10 (V2)
    cfg_fdt = dataclasses.replace(cfg, fixed_dt=True, dt=0.0123, dt_max=0.05)
    gk_fdt = init_gyrokinetic_operator(cfg_fdt, grids, geo, dtype=jnp.float64)
    for sk_fixed in ("rk3", "rk4", "sspx3"):
        assert float(compute_cfl_dt(f_big, gk_fdt, scheme=sk_fixed)) == pytest.approx(0.0123)
    for sk_adapt in ("sspx2", "k10"):
        assert float(compute_cfl_dt(f_big, gk_fdt, scheme=sk_adapt)) < 0.0123

    # 2. Verify ExB flow shear shift across discrete kx grid jump (|ky * g_exb * dt| > 0.5 * dkx)
    exb_jumped, f_jumped = flow_shear_shift(exb0, f0, grids, geo, dt=12.0)
    g_jumped = flow_shear_g_shift(g0, exb_jumped, grids)
    assert not bool(jnp.allclose(exb_jumped.kxstar, exb0.kxstar))
    assert not bool(jnp.allclose(f_jumped.phi, f0.phi))
    assert not bool(jnp.allclose(g_jumped, g0))
    # Zonal modes (iky = 0) are invariant under ExB shear shift
    _assert_close_fp64(f_jumped.phi[:, 0, :], f0.phi[:, 0, :], "exb_shear:zonal_phi_invariant")
    _assert_close_fp64(g_jumped[..., 0, :], g0[..., 0, :], "exb_shear:zonal_g_invariant")

    # 2b. Verify step_integrator_state ExB carry & manual stage parity for rk3 and rk4 (R1, R2, V5)
    cfg_no_force = dataclasses.replace(cfg, forcing_init=False)
    gk_exb = init_gyrokinetic_operator(cfg_no_force, grids, geo, dtype=jnp.float64)
    st_init = init_integrator_state(g0, gk_exb, dt=0.01)
    assert st_init.exb_state is not None  # auto-enabled via gk_exb.exb_shear (V5a)

    # V11(a): At t=0 (exb_init.kperp2 == geo.kperp2), _solve_stage_fields and gyrokinetic_rhs
    # with exb_init must match exb_state=None to <= 1e-14 in complex128
    f_t0_no_exb = _solve_stage_fields(st_init.G, gk_exb, exb_state=None)
    _assert_close_fp64(
        st_init.fields.phi, f_t0_no_exb.phi, "v11a:t0_phi", rtol=1.0e-14, atol=1.0e-14
    )
    _assert_close_fp64(
        st_init.fields.apar, f_t0_no_exb.apar, "v11a:t0_apar", rtol=1.0e-14, atol=1.0e-14
    )
    _assert_close_fp64(
        st_init.fields.bpar, f_t0_no_exb.bpar, "v11a:t0_bpar", rtol=1.0e-14, atol=1.0e-14
    )
    rhs_t0_exb = gyrokinetic_rhs(
        st_init.G, st_init.fields, gk_exb, 0.01, exb_state=st_init.exb_state
    )
    rhs_t0_none = gyrokinetic_rhs(st_init.G, f_t0_no_exb, gk_exb, 0.01, exb_state=None)
    _assert_close_fp64(
        rhs_t0_exb, rhs_t0_none, "v11a:t0_gyrokinetic_rhs", rtol=1.0e-14, atol=1.0e-14
    )

    # V11(b): For a shifted kperp2, _update_linear_op_for_exb tables must equal init_linear_operator
    # and init_field_solver built on a Geometry carrying that shifted kperp2 (including (0, 0))
    geo_shifted = dataclasses.replace(
        geo,
        kperp2=exb_jumped.kperp2,
        omegad=exb_jumped.omegad,
        cv_d=exb_jumped.cv_d,
        gb_d=exb_jumped.gb_d,
    )
    lin_ref_shifted = init_linear_operator(cfg_no_force, grids, geo_shifted, dtype=jnp.float64)
    lin_upd_shifted = _update_linear_op_for_exb(gk_exb.linear_op, exb_jumped, jnp.float64)
    for attr in ("jl", "jlb", "jla", "jlba", "b_s", "sqrt_b", "kperp2", "omegad", "cv_d", "gb_d"):
        _assert_close_fp64(
            getattr(lin_upd_shifted, attr),
            getattr(lin_ref_shifted, attr),
            f"v11b:lin_op.{attr}",
            rtol=1.0e-14,
            atol=1.0e-14,
        )
    for sattr in ("jl", "jlb", "jla"):
        _assert_close_fp64(
            getattr(lin_upd_shifted.solver, sattr),
            getattr(lin_ref_shifted.solver, sattr),
            f"v11b:solver.{sattr}",
            rtol=1.0e-14,
            atol=1.0e-14,
        )
    # Quasineutrality/Ampere denominators remain frozen at t=0 (solver.cu:L69)
    _assert_close_fp64(
        lin_upd_shifted.solver.qneut_fac_phi,
        gk_exb.linear_op.solver.qneut_fac_phi,
        "v11b:qneut_frozen",
        rtol=1.0e-15,
        atol=1.0e-15,
    )

    # R11(a): Verify _bessel_j0_runtime and _j1_over_alpha_runtime vs scipy on [0, 60]
    x_grid = np.linspace(0.0, 60.0, 100_001, dtype=np.float64)
    j0_scipy = sp_special.j0(x_grid)
    x_safe_sp = np.where(x_grid < 1.0e-12, 1.0, x_grid)
    j1_oa_scipy = np.where(x_grid < 1.0e-12, 0.5, sp_special.j1(x_safe_sp) / x_safe_sp)

    j0_f64 = np.asarray(jax.jit(_bessel_j0_runtime)(jnp.asarray(x_grid, dtype=jnp.float64)))
    j1_oa_f64 = np.asarray(jax.jit(_j1_over_alpha_runtime)(jnp.asarray(x_grid, dtype=jnp.float64)))
    j0_f32 = np.asarray(jax.jit(_bessel_j0_runtime)(jnp.asarray(x_grid, dtype=jnp.float32)))
    j1_oa_f32 = np.asarray(jax.jit(_j1_over_alpha_runtime)(jnp.asarray(x_grid, dtype=jnp.float32)))
    assert float(np.max(np.abs(j0_f64 - j0_scipy))) < 1.0e-14
    assert float(np.max(np.abs(j1_oa_f64 - j1_oa_scipy))) < 1.0e-14
    assert float(np.max(np.abs(j0_f32 - j0_scipy))) < 1.0e-6
    assert float(np.max(np.abs(j1_oa_f32 - j1_oa_scipy))) < 1.0e-6

    # R11(b): At t=0 with exb_init (g_exb != 0) on a high-alpha config (nl=8, ky_max >= 2),
    # complex64 nonlinear_rhs(exb_state=exb_init) must match nonlinear_rhs(exb_state=None) <= 1e-6
    cfg_hi = dataclasses.replace(
        cfg_no_force,
        nx=10,
        ny=10,
        nlaguerre=8,
        y0=1.0,
        x0=1.0,
        beta=0.02,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        g_exb=0.25,
        exb_shear=True,
    )
    cfg_hi, grids_hi, geo_hi = init_geometry(cfg_hi)
    assert grids_hi.ky_max >= 2.0
    gk_hi32 = init_gyrokinetic_operator(cfg_hi, grids_hi, geo_hi, dtype=jnp.float32)
    g_hi64 = _random_reality_g(grids_hi, seed=915, dtype=jnp.complex64, scale=0.2)
    st_hi0 = init_integrator_state(g_hi64, gk_hi32, dt=0.01)
    assert gk_hi32.nonlinear_op is not None and st_hi0.exb_state is not None
    rhs_hi_exb = jax.jit(
        lambda g, f, exb: nonlinear_rhs(g, f, gk_hi32.nonlinear_op, exb_state=exb)  # type: ignore[arg-type]
    )(st_hi0.G, st_hi0.fields, st_hi0.exb_state)
    rhs_hi_none = jax.jit(
        lambda g, f: nonlinear_rhs(g, f, gk_hi32.nonlinear_op, exb_state=None)  # type: ignore[arg-type]
    )(st_hi0.G, st_hi0.fields)
    _assert_close_fp32_1e6(rhs_hi_exb, rhs_hi_none, "r11b:high_alpha_c64_exb_t0")

    dt_test = 0.01
    # Manual RK3 with ExB shift at top of advance()
    exb_rk3_m, f0_rk3_m = flow_shear_shift(st_init.exb_state, st_init.fields, grids, geo, dt_test)
    g0_rk3_m = flow_shear_g_shift(st_init.G, exb_rk3_m, grids)
    k1_3 = gyrokinetic_rhs(g0_rk3_m, f0_rk3_m, gk_exb, dt_test, exb_state=exb_rk3_m)
    gq1_3 = g0_rk3_m + (dt_test / 3.0) * k1_3
    fq1_3 = _solve_stage_fields(gq1_3, gk_exb, exb_state=exb_rk3_m)
    k2_3 = gyrokinetic_rhs(gq1_3, fq1_3, gk_exb, dt_test, exb_state=exb_rk3_m)
    gq2_3 = g0_rk3_m + (2.0 * dt_test / 3.0) * k2_3
    fq2_3 = _solve_stage_fields(gq2_3, gk_exb, exb_state=exb_rk3_m)
    k3_3 = gyrokinetic_rhs(gq2_3, fq2_3, gk_exb, dt_test, exb_state=exb_rk3_m)
    g_rk3_manual = mask_g(g0_rk3_m + 0.75 * dt_test * k3_3 + 0.25 * dt_test * k1_3, grids)

    st_rk3 = step_integrator_state(st_init, gk_exb, dt_override=dt_test, scheme="rk3")
    assert st_rk3.exb_state is not None
    assert not bool(jnp.allclose(st_rk3.exb_state.kxstar, st_init.exb_state.kxstar))
    _assert_close_fp64(st_rk3.G, g_rk3_manual, "exb_rk3:step_vs_manual")
    _assert_close_fp64(st_rk3.exb_state.kxstar, exb_rk3_m.kxstar, "exb_rk3:kxstar")

    # Manual RK4 with two half-step ExB shifts inside advance()
    k1_4 = gyrokinetic_rhs(st_init.G, st_init.fields, gk_exb, dt_test, exb_state=st_init.exb_state)
    g1_4 = st_init.G + 0.5 * dt_test * k1_4
    f1_4 = _solve_stage_fields(g1_4, gk_exb, exb_state=st_init.exb_state)
    exb_h1, f1_4 = flow_shear_shift(st_init.exb_state, f1_4, grids, geo, 0.5 * dt_test)
    g0_4 = flow_shear_g_shift(st_init.G, exb_h1, grids)
    k1_4 = flow_shear_g_shift(k1_4, exb_h1, grids)
    g1_4 = flow_shear_g_shift(g1_4, exb_h1, grids)
    k2_4 = gyrokinetic_rhs(g1_4, f1_4, gk_exb, dt_test, exb_state=exb_h1)
    g2_4 = g0_4 + 0.5 * dt_test * k2_4
    f2_4 = _solve_stage_fields(g2_4, gk_exb, exb_state=exb_h1)
    k3_4 = gyrokinetic_rhs(g2_4, f2_4, gk_exb, dt_test, exb_state=exb_h1)
    g3_4 = g0_4 + dt_test * k3_4
    f3_4 = _solve_stage_fields(g3_4, gk_exb, exb_state=exb_h1)
    exb_h2, f3_4 = flow_shear_shift(exb_h1, f3_4, grids, geo, 0.5 * dt_test)
    g0_4 = flow_shear_g_shift(g0_4, exb_h2, grids)
    k1_4 = flow_shear_g_shift(k1_4, exb_h2, grids)
    k2_4 = flow_shear_g_shift(k2_4, exb_h2, grids)
    k3_4 = flow_shear_g_shift(k3_4, exb_h2, grids)
    g3_4 = flow_shear_g_shift(g3_4, exb_h2, grids)
    k4_4 = gyrokinetic_rhs(g3_4, f3_4, gk_exb, dt_test, exb_state=exb_h2)
    g_rk4_manual = mask_g(g0_4 + (dt_test / 6.0) * (k1_4 + 2.0 * k2_4 + 2.0 * k3_4 + k4_4), grids)

    st_rk4 = step_integrator_state(st_init, gk_exb, dt_override=dt_test, scheme="rk4")
    assert st_rk4.exb_state is not None
    assert not bool(jnp.allclose(st_rk4.exb_state.kxstar, st_init.exb_state.kxstar))
    _assert_close_fp64(st_rk4.G, g_rk4_manual, "exb_rk4:step_vs_manual")
    _assert_close_fp64(st_rk4.exb_state.kxstar, exb_h2.kxstar, "exb_rk4:kxstar")

    # 3. Verify Forcing PRNG advancement, per-species kicks, and validation (R5, V9)
    assert gk_op.forcing_op is not None
    g_forced0 = apply_forcing(g0, gk_op.forcing_op, 0.01, jax.random.PRNGKey(42), step_idx=0)
    g_forced1 = apply_forcing(g0, gk_op.forcing_op, 0.01, jax.random.PRNGKey(42), step_idx=1)
    assert not bool(jnp.allclose(g_forced0, g0))
    _assert_close_fp64(g_forced1, g0, "forcing:kz_impulse_step1_noop")

    # Missing key when forcing_init=True must raise ValueError
    with pytest.raises(ValueError, match="PRNGKey"):
        apply_forcing(g0, gk_op.forcing_op, 0.01, None)

    # HeliInj raises NotImplementedError; masked/dealiased forcing_index raises ValueError
    with pytest.raises(NotImplementedError, match="HeliInj"):
        init_forcing(dataclasses.replace(cfg, forcing_type="HeliInj"), grids)
    with pytest.raises(ValueError, match="masked"):
        init_forcing(dataclasses.replace(cfg, forcing_index=0), grids)

    # Consecutive steps with Kz forcing in step_integrator_state must use distinct PRNG keys
    cfg_kz = dataclasses.replace(cfg, forcing_type="Kz", g_exb=0.0, exb_shear=False)
    gk_kz = init_gyrokinetic_operator(cfg_kz, grids, geo, dtype=jnp.float64)
    st_z0 = init_integrator_state(jnp.zeros_like(g0), gk_kz, dt=0.01)
    st_z1 = step_integrator_state(st_z0, gk_kz, dt_override=0.01)
    # Step a second zero state carrying st_z1.key vs st_z0.key: kicks must differ!
    st_z2_same_zero = step_integrator_state(
        dataclasses.replace(st_z0, key=st_z1.key, step=jnp.int32(1)),
        gk_kz,
        dt_override=0.01,
    )
    assert not bool(jnp.allclose(st_z1.G, st_z2_same_zero.G))

    # Multi-species forcing draws independent kicks per species
    g2_zero = jnp.zeros((2, *grids.g_shape[1:]), dtype=jnp.complex128)
    g2_forced = apply_forcing(g2_zero, gk_kz.forcing_op, 0.01, jax.random.PRNGKey(7))  # type: ignore[arg-type]
    assert not bool(jnp.allclose(g2_forced[0], g2_forced[1]))

    # 4. Verify adaptive while_loop default unclamped overshoot vs opt-in clamp_to_t_max=True (V3)
    target_t = 0.073
    unclamped_st = integrate_while_loop(
        g0, f0, gk_op, t_max=target_t, max_steps=100, exb_state=exb0
    )
    assert float(unclamped_st.time) > target_t
    clamped_st = integrate_while_loop(
        g0,
        f0,
        gk_op,
        t_max=target_t,
        max_steps=100,
        clamp_to_t_max=True,
        exb_state=exb0,
    )
    assert float(clamped_st.time) == pytest.approx(target_t, abs=1.0e-14)
    assert int(clamped_st.step) >= 2


# ---------------------------------------------------------------------------
# 9. Single-Trace @jax.jit, Carry Invariance, HLO Audit, & jax.grad Under debug_nans(True)
# ---------------------------------------------------------------------------


def test_jit_single_trace_carry_invariance_and_reverse_mode_ad_debug_nans() -> None:
    """Verify single-trace @jax.jit, StableHLO transpose/f64 hygiene, and jax.grad."""
    kbm_cfg = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    cfg = dataclasses.replace(
        kbm_cfg,
        nx=7,
        ny=9,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        fphi=1.0,
        fapar=1.0,
        fbpar=1.0,
        nonlinear_mode=True,
        linear=False,
        fixed_dt=True,
        dt=0.005,
        g_exb=0.15,
        exb_shear=True,
        exb_shear_phase=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    gk64 = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float64)
    gk32 = gk64.astype(jnp.float32)

    g1 = _random_reality_g(grids, seed=1001, dtype=jnp.complex64, scale=0.05)
    g2 = _random_reality_g(grids, seed=1002, dtype=jnp.complex64, scale=0.05)

    # 1. Verify IntegratorState carry structure & dtype invariance (time/dt in float64 per V4)
    st0 = init_integrator_state(g1, gk32, time=0.0, dt=0.005)
    assert st0.time.dtype == jnp.float64
    assert st0.dt.dtype == jnp.float64
    st1 = jax.jit(step_integrator_state)(st0, gk32)
    assert jax.tree_util.tree_structure(st0) == jax.tree_util.tree_structure(st1)
    for l0, l1 in zip(jax.tree_util.tree_leaves(st0), jax.tree_util.tree_leaves(st1), strict=True):
        assert l0.shape == l1.shape
        assert l0.dtype == l1.dtype

    # 1b. StableHLO check (R3, R4):
    #     - zero stablehlo.transpose on any full (nx, ny) / (ny, nx) = (7, 9) / (9, 7) real tensor
    #     - zero non-scalar f64 array tensors inside complex64 rk4 step with ExB shear
    hlo_nl = str(
        jax.jit(
            lambda g, f, exb: nonlinear_rhs(g, f, gk32.nonlinear_op, exb_state=exb)  # type: ignore[arg-type]
        )
        .lower(st0.G, st0.fields, st0.exb_state)
        .compiler_ir("stablehlo")
    )
    real_tp = [
        line
        for line in hlo_nl.splitlines()
        if "stablehlo.transpose" in line and ("7x9xf" in line or "9x7xf" in line)
    ]
    assert len(real_tp) == 0, f"Unexpected real-grid transposes in nonlinear_rhs: {real_tp}"

    hlo_rk4 = str(
        jax.jit(lambda s: step_integrator_state(s, gk32, dt_override=0.005, scheme="rk4"))
        .lower(st0)
        .compiler_ir("stablehlo")
    )
    import re

    f64_arrays = re.findall(r"tensor<\d+[^>]*xf64>", hlo_rk4)
    assert len(f64_arrays) == 0, f"Unexpected non-scalar f64 tensors in c64 rk4 step: {f64_arrays}"

    # 2. Verify @jax.jit single-trace compilation (trace_count == 1 across calls)
    trace_counter = {"scan": 0, "while": 0}

    @jax.jit
    def _jit_scan(g_in: jax.Array):
        trace_counter["scan"] += 1
        return integrate_scan(g_in, None, gk32, nsteps=4, dt=0.005, save_every=2)

    @jax.jit
    def _jit_while(g_in: jax.Array, t_end: jax.Array) -> IntegratorState:
        trace_counter["while"] += 1
        return integrate_while_loop(g_in, None, gk32, t_max=t_end, max_steps=10, dt_init=0.005)

    _ = _jit_scan(g1)
    _ = _jit_scan(g2)
    assert trace_counter["scan"] == 1

    _ = _jit_while(g1, jnp.float64(0.012))
    _ = _jit_while(g2, jnp.float64(0.018))
    assert trace_counter["while"] == 1

    # 3. Reverse-mode AD (jax.grad) under jax.debug_nans(True) across nonlinear_rhs,
    #    all 5 explicit schemes, multi-step integrate_scan (with remat=False & True), and zero CFL
    cfg_ad = dataclasses.replace(cfg, g_exb=0.0, exb_shear=False, exb_shear_phase=False)
    gk64_ad = init_gyrokinetic_operator(cfg_ad, grids, geo, dtype=jnp.float64)
    g128 = _random_reality_g(grids, seed=1003, dtype=jnp.complex128, scale=0.05)
    direction = _random_reality_g(grids, seed=1004, dtype=jnp.complex128, scale=1.0)

    with jax.debug_nans(True):
        for sk in ("sspx2", "rk3", "sspx3", "rk4", "k10"):
            op_cur = gk64 if sk == "rk4" else gk64_ad
            exb_cur = init_exb_shear(cfg, grids, geo, dtype=jnp.float64) if sk == "rk4" else None

            def _step_loss(
                g_in: jax.Array,
                scheme_name: str = sk,
                op_in=op_cur,
                exb_in=exb_cur,
            ) -> jax.Array:
                f_in = solve_fields(g_in, op_in.linear_op.solver)
                g_out, f_out, _, _ = step_gyrokinetic(
                    g_in, f_in, op_in, dt=0.002, scheme=scheme_name, exb_state=exb_in
                )
                return jnp.sum(jnp.abs(g_out) ** 2) + jnp.sum(jnp.abs(f_out.phi) ** 2)

            grad_g = jax.jit(jax.grad(_step_loss))(g128)
            assert bool(jnp.all(jnp.isfinite(grad_g)))
            assert float(jnp.linalg.norm(grad_g)) > 0.0
            # Masked origin (iky=0, ikx=0) must have exact zero cotangent
            np.testing.assert_array_equal(np.asarray(grad_g[..., 0, 0]), 0.0)

        # Multi-step integrate_scan gradient (remat=False and remat=True) vs central FD
        def _scan_loss(eps: jax.Array, use_remat: bool = False) -> jax.Array:
            g_pert = g128 + eps * direction
            traj = integrate_scan(
                g_pert, None, gk64_ad, nsteps=3, dt=0.002, scheme="rk4", remat=use_remat
            )
            return jnp.sum(traj.phi2_history) + jnp.sum(traj.wg_history)

        ad_deriv = float(jax.jit(jax.grad(lambda e: _scan_loss(e, False)))(jnp.float64(0.0)))
        ad_deriv_remat = float(jax.jit(jax.grad(lambda e: _scan_loss(e, True)))(jnp.float64(0.0)))
        delta_eps = 1.0e-6
        fd_deriv = float(
            (_scan_loss(jnp.float64(delta_eps)) - _scan_loss(jnp.float64(-delta_eps)))
            / (2.0 * delta_eps)
        )
        _assert_close_fp64(ad_deriv, ad_deriv_remat, "ad_remat_vs_noremat", rtol=1.0e-12)
        _assert_close_fp64(ad_deriv, fd_deriv, "ad_vs_fd:integrate_scan_3step", rtol=1.0e-6)

        # Zero-field CFL dt edge case under debug_nans(True)
        cfg_adaptive = dataclasses.replace(
            cfg,
            fixed_dt=False,
            species=tuple(dataclasses.replace(s, tprim=0.0, fprim=0.0) for s in cfg.species),
        )
        gk_adapt = init_gyrokinetic_operator(cfg_adaptive, grids, geo, dtype=jnp.float64)
        f_zero = init_fields(grids, dtype=jnp.complex128)
        dt_z = compute_cfl_dt(f_zero, gk_adapt)
        assert bool(jnp.isfinite(dt_z))
