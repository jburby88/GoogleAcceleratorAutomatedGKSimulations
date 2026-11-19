"""Quasineutrality & Ampère Field Solver and Moments.

Verifies ``gx_jax/fields.py`` against C++ GX ``src/{fields.cu,solver.cu,moments.cu}`` and
``device_funcs.cu`` formulas, all 5 benchmark configurations, 3D W7-X stellarator equilibria
(linked & NTFT), and golden NetCDF ``.restart.nc`` / ``.nc`` fixtures to ``rtol < 1e-6``.
"""

from __future__ import annotations

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
from gx_jax.config import BoltzmannOption, GXConfig, SpeciesConfig, load_config
from gx_jax.fields import (
    Fields,
    FieldSolver,
    compute_conservation_moments,
    compute_moments,
    enforce_reality,
    field_line_average,
    init_field_solver,
    init_fields,
    init_moments,
    load_restart_nc,
    mask_g,
    real_space_density,
    real_space_par_current,
    real_space_perp_current,
    save_restart_nc,
    solve_fields,
)
from gx_jax.geometry_nc import build_vmec_eik_nc, init_geometry, load_eik_nc
from gx_jax.grids import pad_kykx_to_full

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
UNIT_INPUTS_DIR = REPO_ROOT / "golden_fixtures" / "unit_tests" / "inputs"

BENCHMARK_INPUTS: list[tuple[str, Path]] = [
    (
        "itg_salpha_adiabatic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in",
    ),
    (
        "itg_miller_adiabatic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in",
    ),
    (
        "itg_miller_kinetic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_kinetic_electrons.in",
    ),
    (
        "kbm_miller",
        BENCH_DIR / "linear" / "KBM" / "kbm_miller.in",
    ),
    (
        "kaw_slab",
        BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01.in",
    ),
]


def _assert_close_1e6(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    label: str,
) -> None:
    """Strict FP64 relative-error check (< 1e-6) with near-zero floor atol=1e-12."""
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    np.testing.assert_allclose(
        np.asarray(actual, dtype=dt),
        np.asarray(expected, dtype=dt),
        rtol=1.0e-6,
        atol=1.0e-12,
        err_msg=f"Mismatch in {label}",
    )


def _assert_close_fp32_1e6(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    label: str,
) -> None:
    """Assert < 1e-6 relative error for runtime complex64 / float32 outputs vs FP64 reference."""
    is_cplx = np.iscomplexobj(actual) or np.iscomplexobj(expected)
    dt = np.complex128 if is_cplx else np.float64
    act = np.asarray(actual, dtype=dt)
    exp = np.asarray(expected, dtype=dt)
    exp_norm = float(np.linalg.norm(exp.ravel()))
    if exp_norm > 1.0e-12:
        rel_l2 = float(np.linalg.norm((act - exp).ravel())) / exp_norm
        assert rel_l2 < 1.0e-6, f"{label}: relative L2 error {rel_l2:.3e} >= 1e-6"
    scale = max(1.0, float(np.max(np.abs(exp)))) if exp.size > 0 else 1.0
    np.testing.assert_allclose(
        act,
        exp,
        rtol=1.0e-6,
        atol=5.0e-7 * scale,
        err_msg=f"Mismatch in {label}",
    )


def _random_complex_g(
    shape: tuple[int, ...],
    active_mask: np.ndarray,
    seed: int,
    *,
    dtype: jnp.dtype = jnp.complex128,
) -> jax.Array:
    """Generate deterministic complex state G with active ky=0 and ky>0 modes."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)
    raw = np.where(active_mask[None, None, None, None, :, :], raw, 0.0)
    return jnp.asarray(raw, dtype=dtype)


def _cpp_jflr_np(ell: int, b: np.ndarray) -> np.ndarray:
    """Independent NumPy mirror of C++ ``Jflr(l, b)`` (``device_funcs.cu:L127-139``)."""
    if ell < 0 or ell > 30:
        return np.zeros_like(b, dtype=np.float64)
    return ((-0.5 * b) ** ell) / math.factorial(ell) * np.exp(-0.5 * b)


def _cpp_jflr_b_np(ell: int, b: np.ndarray) -> np.ndarray:
    """Independent NumPy mirror of C++ ``JflrB(l, b)`` (``device_funcs.cu:L146-149``)."""
    return _cpp_jflr_np(ell, b) + _cpp_jflr_np(ell - 1, b)


def _cpp_jflr_a_np(ell: int, b: np.ndarray) -> np.ndarray:
    """Independent NumPy mirror of C++ ``JflrA(l, b)`` (``device_funcs.cu:L141-144``)."""
    return (
        ell * _cpp_jflr_np(ell - 1, b)
        + 2.0 * ell * _cpp_jflr_np(ell, b)
        + (ell + 1.0) * _cpp_jflr_np(ell + 1, b)
    )


def _cpp_reference_field_solve(
    G_np: np.ndarray,
    cfg: GXConfig,
    grids: gx_jax.Grids,
    geo: gx_jax.Geometry,
    *,
    no_fields: bool = False,
    phi_ext: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Direct NumPy port of ``Solver_GK`` in ``solver.cu`` & ``device_funcs.cu`` in FP64."""
    nz, naky, nakx = grids.field_shape
    nl = grids.nlaguerre
    nm = grids.nhermite
    mask2d = np.asarray(grids.active_mask, dtype=bool)
    mask3d = np.broadcast_to(mask2d[None, :, :], (nz, naky, nakx))

    if no_fields:
        z3 = np.zeros((nz, naky, nakx), dtype=np.complex128)
        z1 = np.zeros((nakx,), dtype=np.complex128)
        return z3, z3, z3, z1, {}

    kperp2 = np.asarray(geo.kperp2, dtype=np.float64)
    bmag = np.asarray(geo.bmag, dtype=np.float64)
    bmag_inv = np.asarray(geo.bmag_inv, dtype=np.float64)
    jacobian = np.asarray(geo.jacobian, dtype=np.float64)

    q_phi = np.zeros((nz, naky, nakx), dtype=np.float64)
    q_bpar = np.zeros((nz, naky, nakx), dtype=np.float64)
    a_par = kperp2 * (bmag[:, None, None] ** 2)
    a_phi = np.zeros((nz, naky, nakx), dtype=np.float64)
    a_bpar = np.ones((nz, naky, nakx), dtype=np.float64)

    nbar = np.zeros((nz, naky, nakx), dtype=np.complex128)
    jparbar = np.zeros((nz, naky, nakx), dtype=np.complex128)
    jperpbar = np.zeros((nz, naky, nakx), dtype=np.complex128)

    for s_idx, sp in enumerate(cfg.species):
        b_sol = kperp2 * (sp.rho2_long_wavelength_GK if cfg.long_wavelength_gk else sp.rho2)
        g0_s = np.zeros_like(b_sol)
        g01_s = np.zeros_like(b_sol)
        g11_s = np.zeros_like(b_sol)
        for ell in range(nl):
            jl_s = _cpp_jflr_np(ell, b_sol)
            jlb_s = _cpp_jflr_b_np(ell, b_sol)
            g0_s += jl_s * jl_s
            g01_s += jl_s * jlb_s
            g11_s += jlb_s * jlb_s
        one_minus_g0 = -np.expm1(-b_sol) - sum(
            _cpp_jflr_np(ell, b_sol) ** 2 for ell in range(1, nl)
        )
        if cfg.long_wavelength_gk:
            q_phi += sp.nz * sp.zt * b_sol
        else:
            q_phi += sp.nz * sp.zt * one_minus_g0
        q_bpar += -sp.nz * g01_s
        a_par += (sp.nz * sp.z / sp.mass) * (0.5 * cfg.beta) * g0_s
        a_phi += sp.nz * (0.5 * cfg.beta) * (bmag_inv[:, None, None] ** 2) * g01_s
        a_bpar += sp.nt * (0.5 * cfg.beta) * (bmag_inv[:, None, None] ** 2) * g11_s

        b_rt = kperp2 * sp.rho2
        for ell in range(nl):
            jl_rt = _cpp_jflr_np(ell, b_rt)
            jlb_rt = _cpp_jflr_b_np(ell, b_rt)
            nbar += sp.nz * jl_rt * G_np[s_idx, 0, ell]
            if nm >= 2:
                jparbar += sp.jparfac * jl_rt * G_np[s_idx, 1, ell]
            jperpbar += sp.jperpfac * (bmag_inv[:, None, None] ** 2) * jlb_rt * G_np[s_idx, 0, ell]

    q_phi = np.where(mask3d, q_phi, 0.0)
    q_bpar = np.where(mask3d, q_bpar, 0.0)
    a_par = np.where(mask3d, a_par, 0.0)
    a_phi = np.where(mask3d, a_phi, 0.0)
    a_bpar = np.where(mask3d, a_bpar, 0.0)
    det = np.where(mask3d, q_phi * a_bpar - q_bpar * a_phi, 0.0)

    nbar = np.where(mask3d, nbar, 0.0)
    jparbar = np.where(mask3d, jparbar, 0.0)
    jperpbar = np.where(mask3d, jperpbar, 0.0)

    phi = np.zeros((nz, naky, nakx), dtype=np.complex128)
    apar = np.zeros((nz, naky, nakx), dtype=np.complex128)
    bpar = np.zeros((nz, naky, nakx), dtype=np.complex128)
    phi_avg = np.zeros((nakx,), dtype=np.complex128)
    phiavgdenom = np.zeros((nakx,), dtype=np.float64)

    if cfg.all_kinetic:
        if cfg.fbpar > 0.0:
            phi[mask3d] = cfg.fphi * (
                (a_bpar[mask3d] * nbar[mask3d] - q_bpar[mask3d] * jperpbar[mask3d]) / det[mask3d]
            )
            bpar[mask3d] = cfg.fbpar * (
                (-a_phi[mask3d] * nbar[mask3d] + q_phi[mask3d] * jperpbar[mask3d]) / det[mask3d]
            )
        elif cfg.fphi != 0.0:
            phi[mask3d] = cfg.fphi * nbar[mask3d] / q_phi[mask3d]
        if cfg.fapar > 0.0 and nm >= 2:
            apar[mask3d] = cfg.fapar * jparbar[mask3d] / a_par[mask3d]

    elif cfg.boltzmann_opt == BoltzmannOption.BOLTZMANN_ELECTRONS:
        for ikx in range(1, nakx):
            q0 = q_phi[:, 0, ikx]
            phiavgdenom[ikx] = np.sum(jacobian * q0 / (cfg.tau_fac + q0))
            tmp_num = np.sum((nbar[:, 0, ikx] / (cfg.tau_fac + q0)) * jacobian)
            pa_raw = tmp_num / phiavgdenom[ikx]
            phi_avg[ikx] = cfg.fphi * pa_raw
            phi[:, 0, ikx] = (
                cfg.fphi * (nbar[:, 0, ikx] + cfg.tau_fac * pa_raw) / (cfg.tau_fac + q0)
            )
        if naky > 1 and cfg.fphi != 0.0:
            phi[:, 1:, :] = cfg.fphi * nbar[:, 1:, :] / (cfg.tau_fac + q_phi[:, 1:, :])

    elif cfg.boltzmann_opt == BoltzmannOption.BOLTZMANN_IONS:
        if cfg.fphi != 0.0:
            phi[mask3d] = cfg.fphi * nbar[mask3d] / (cfg.tau_fac + q_phi[mask3d])
        if cfg.fapar > 0.0 and nm >= 2:
            apar[mask3d] = cfg.fapar * jparbar[mask3d] / a_par[mask3d]

    if phi_ext != 0.0:
        phi = np.where(mask3d, phi + phi_ext, 0.0)

    mats = {
        "qneut_fac_phi": q_phi,
        "qneut_fac_bpar": q_bpar,
        "ampere_par_fac": a_par,
        "ampere_perp_fac_phi": a_phi,
        "ampere_perp_fac_bpar": a_bpar,
        "phi_bpar_det": det,
        "phiavgdenom": phiavgdenom,
        "nbar": nbar,
        "jparbar": jparbar,
        "jperpbar": jperpbar,
    }
    return phi, apar, bpar, phi_avg, mats


# ---------------------------------------------------------------------------
# 1. FP64 Precomputation & Cancellation-Free Low-(k_perp * rho) Verification
# ---------------------------------------------------------------------------


def test_fp64_precomputation_cancels_no_digits_at_low_kperp_rho() -> None:
    """Verify cancellation-free 1 - sum_l J_l^2 in FP64 -> FP32 on KAW (b_e ~ 5.4e-8) & b=1e-12."""
    kaw_cfg, kaw_grids, kaw_geo = init_geometry(
        load_config(BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01.in")
    )
    solver64 = init_field_solver(kaw_cfg, kaw_grids, kaw_geo, dtype=jnp.float64)
    solver32 = solver64.astype(jnp.float32)

    assert solver64.qneut_fac_phi.dtype == jnp.float64
    assert solver64.phi_bpar_det.dtype == jnp.float64
    assert solver32.qneut_fac_phi.dtype == jnp.float32
    assert solver32.phi_bpar_det.dtype == jnp.float32

    # Check KAW lowest active mode (ky=0.01, kx=0 => kperp2 = 1e-4, electron b_e = 2.7e-8)
    kp2_min = float(kaw_geo.kperp2[0, 1, 0])
    assert kp2_min == pytest.approx(1.0e-4, rel=1e-12)
    q_exact = 0.0
    for sp in kaw_cfg.species:
        b = kp2_min * sp.rho2
        # Exact analytical Taylor expansion: 1 - exp(-b) - (b^2/4)*exp(-b) - ...
        one_minus = -math.expm1(-b) - sum(
            ((-0.5 * b) ** ell / math.factorial(ell) * math.exp(-0.5 * b)) ** 2
            for ell in range(1, kaw_grids.nlaguerre)
        )
        q_exact += sp.nz * sp.zt * one_minus

    _assert_close_1e6(float(solver64.qneut_fac_phi[0, 1, 0]), q_exact, "kaw:qneut_fp64")
    _assert_close_fp32_1e6(float(solver32.qneut_fac_phi[0, 1, 0]), q_exact, "kaw:qneut_fp32")

    # Single-species ultra-low b = 1e-12 test (naive float32 1.0f - J0^2 == 0.0f -> div by zero!)
    sp_single = SpeciesConfig.from_raw(z=-1.0, mass=1.0e-8, dens=1.0, temp=1.0, stype="electron")
    cfg_tiny = dataclasses.replace(
        kaw_cfg,
        nspecies=1,
        species=(sp_single,),
        beta=0.0,
        fapar=0.0,
        fbpar=0.0,
    )
    grids_tiny = gx_jax.init_grids(cfg_tiny)
    solver_tiny64 = init_field_solver(cfg_tiny, grids_tiny, kaw_geo, dtype=jnp.float64)
    solver_tiny32 = solver_tiny64.astype(jnp.float32)

    b_tiny = kp2_min * sp_single.rho2  # 1e-4 * 1e-8 = 1e-12
    # For nl >= 2: 1 - exp(-b) - (b^2/4) exp(-b) = b - 0.75 * b^2 + O(b^3)
    expected_tiny = b_tiny - 0.75 * (b_tiny**2)
    np.testing.assert_allclose(
        float(solver_tiny64.qneut_fac_phi[0, 1, 0]),
        expected_tiny,
        rtol=1.0e-11,
        atol=0.0,
    )
    _assert_close_fp32_1e6(
        float(solver_tiny32.qneut_fac_phi[0, 1, 0]),
        expected_tiny,
        "ultra_low_b_fp32_cast",
    )


# ---------------------------------------------------------------------------
# 2. Differential Verification Across All 5 Benchmark Configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case_name", "in_path"), BENCHMARK_INPUTS)
def test_solve_fields_matches_cpp_across_all_benchmarks(case_name: str, in_path: Path) -> None:
    """Verify static matrices, complex128, and complex64 solve_fields across all 5 benchmarks."""
    cfg_raw = load_config(in_path)
    # Ensure nx >= 7 so ky=0 has multiple active kx != 0 zonal modes to exercise phiavgdenom
    cfg = dataclasses.replace(cfg_raw, nx=max(cfg_raw.nx, 7), ny=max(cfg_raw.ny, 7))
    cfg, grids, geo = init_geometry(cfg)

    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    solver32 = solver64.astype(jnp.float32)

    g128 = _random_complex_g(
        grids.g_shape,
        np.asarray(grids.active_mask),
        seed=42,
        dtype=jnp.complex128,
    )
    g64 = g128.astype(jnp.complex64)

    phi_ref, apar_ref, bpar_ref, phi_avg_ref, mats_ref = _cpp_reference_field_solve(
        np.asarray(g128), cfg, grids, geo
    )

    # 1. Static precomputed matrices
    _assert_close_1e6(solver64.qneut_fac_phi, mats_ref["qneut_fac_phi"], f"{case_name}:QPhi")
    _assert_close_1e6(solver64.qneut_fac_bpar, mats_ref["qneut_fac_bpar"], f"{case_name}:QB")
    _assert_close_1e6(solver64.ampere_par_fac, mats_ref["ampere_par_fac"], f"{case_name}:APar")
    _assert_close_1e6(
        solver64.ampere_perp_fac_phi, mats_ref["ampere_perp_fac_phi"], f"{case_name}:APhi"
    )
    _assert_close_1e6(
        solver64.ampere_perp_fac_bpar, mats_ref["ampere_perp_fac_bpar"], f"{case_name}:AB"
    )
    _assert_close_1e6(solver64.phi_bpar_det, mats_ref["phi_bpar_det"], f"{case_name}:det")
    _assert_close_1e6(solver64.phiavgdenom, mats_ref["phiavgdenom"], f"{case_name}:phiavgdenom")

    # 2. JIT FP64 solve vs C++ reference (< 1e-12)
    jit_solve = jax.jit(solve_fields)
    f128 = jit_solve(g128, solver64)
    np.testing.assert_allclose(np.asarray(f128.phi), phi_ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(f128.apar), apar_ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(f128.bpar), bpar_ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(f128.phi_avg), phi_avg_ref, rtol=1e-12, atol=1e-13)

    # 3. JIT runtime complex64 solve (with float32 precomputed solver) vs FP64 reference (< 1e-6)
    f64 = jit_solve(g64, solver32)
    assert f64.phi.dtype == jnp.complex64
    assert f64.apar.dtype == jnp.complex64
    assert f64.bpar.dtype == jnp.complex64
    assert f64.phi_avg.dtype == jnp.complex64

    _assert_close_fp32_1e6(f64.phi, phi_ref, f"{case_name}:phi_c64")
    _assert_close_fp32_1e6(f64.apar, apar_ref, f"{case_name}:apar_c64")
    _assert_close_fp32_1e6(f64.bpar, bpar_ref, f"{case_name}:bpar_c64")
    _assert_close_fp32_1e6(f64.phi_avg, phi_avg_ref, f"{case_name}:phi_avg_c64")

    # Masked origin (iky=0, ikx=0) must be identically zero
    np.testing.assert_array_equal(np.asarray(f64.phi[:, 0, 0]), 0.0)
    np.testing.assert_array_equal(np.asarray(f64.apar[:, 0, 0]), 0.0)
    np.testing.assert_array_equal(np.asarray(f64.bpar[:, 0, 0]), 0.0)
    assert complex(f64.phi_avg[0]) == 0.0j


# ---------------------------------------------------------------------------
# 3. Physics Invariants: Adiabatic <phi>, Boltzmann Ions, Coupled EM & Maxwell Laws
# ---------------------------------------------------------------------------


def test_adiabatic_electrons_zonal_phi_avg_strictly_ky0_and_flux_avg_identity() -> None:
    """Verify <phi> applies strictly to ky=0, kx!=0 and satisfies <<phi>> == phi_avg."""
    cfg_in = load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in")
    cfg = dataclasses.replace(cfg_in, nx=10, ny=10, tau_fac=1.35)
    cfg, grids, geo = init_geometry(cfg)
    solver = init_field_solver(cfg, grids, geo, dtype=jnp.float64)

    assert not solver.all_kinetic
    assert solver.boltzmann_opt == BoltzmannOption.BOLTZMANN_ELECTRONS
    assert float(solver.phiavgdenom[0]) == 0.0
    assert np.all(np.asarray(solver.phiavgdenom[1:]) > 0.0)

    g_full = _random_complex_g(
        grids.g_shape, np.asarray(grids.active_mask), seed=101, dtype=jnp.complex128
    )
    f_full = solve_fields(g_full, solver)

    # 1. Exact flux-surface average identity: sum_z vol_fac[z] * phi[z, 0, kx] == phi_avg[kx]
    favg_phi, dphi = field_line_average(f_full.phi, grids, geo)
    _assert_close_1e6(favg_phi, f_full.phi_avg, "adiab_e:flux_avg_identity")
    _assert_close_1e6(dphi[:, 0, :], f_full.phi[:, 0, :] - f_full.phi_avg[None, :], "adiab_e:dphi")
    _assert_close_1e6(dphi[:, 1:, :], f_full.phi[:, 1:, :], "adiab_e:dphi_ky_gt_0")

    # 2. Exact adiabatic-electron quasineutrality on ALL active (z, ky, kx):
    #    tau_fac * (phi - <phi>_ky0) == sum_s nz_s * n_bar_s - (sum_s nz_s zt_s) * phi
    moms = compute_moments(g_full, f_full, solver)
    phi_minus_avg = f_full.phi.at[:, 0, :].add(-f_full.phi_avg[None, :])
    lhs_ne = cfg.tau_fac * phi_minus_avg
    rhs_ni = (
        jnp.sum(solver.sp_nz[:, None, None, None] * moms.n_bar, axis=0)
        - jnp.sum(solver.sp_nz * solver.sp_zt) * f_full.phi
    )
    _assert_close_1e6(lhs_ne, rhs_ni, "adiab_e:quasineutrality_all_modes")

    # 3. Changing G on ky=0 MUST NOT affect ky>0, and G with ky=0 zeroed gives phi_avg == 0
    g_no_zonal = g_full.at[:, :, :, :, 0, :].set(0.0)
    f_no_zonal = solve_fields(g_no_zonal, solver)
    np.testing.assert_array_equal(np.asarray(f_no_zonal.phi_avg), 0.0)
    np.testing.assert_array_equal(np.asarray(f_no_zonal.phi[:, 0, :]), 0.0)
    np.testing.assert_allclose(
        np.asarray(f_no_zonal.phi[:, 1:, :]),
        np.asarray(f_full.phi[:, 1:, :]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("beta_val", [0.0, 0.02])
def test_boltzmann_ions_electrostatic_and_electromagnetic_apar(beta_val: float) -> None:
    """Verify BOLTZMANN_IONS has no <phi> zonal subtraction and supports finite-beta apar."""
    e_sp = SpeciesConfig.from_raw(
        z=-1.0,
        mass=2.7e-4,
        dens=1.0,
        temp=1.0,
        tprim=2.5,
        fprim=0.8,
        stype="electron",
        beta=beta_val,
    )
    cfg_base = load_config(
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    )
    cfg = dataclasses.replace(
        cfg_base,
        nx=7,
        ny=7,
        nspecies=1,
        species=(e_sp,),
        all_kinetic=False,
        add_boltzmann_species=True,
        boltzmann_opt=BoltzmannOption.BOLTZMANN_IONS,
        boltzmann_type="ions",
        tau_fac=1.2,
        beta=beta_val,
        fphi=1.0,
        fapar=1.0 if beta_val > 0.0 else 0.0,
        fbpar=0.0,
    )
    cfg, grids, geo = init_geometry(cfg)
    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    solver32 = solver64.astype(jnp.float32)

    g128 = _random_complex_g(
        grids.g_shape, np.asarray(grids.active_mask), seed=202, dtype=jnp.complex128
    )
    phi_ref, apar_ref, bpar_ref, phi_avg_ref, _ = _cpp_reference_field_solve(
        np.asarray(g128), cfg, grids, geo
    )

    f128 = solve_fields(g128, solver64)
    f64 = solve_fields(g128.astype(jnp.complex64), solver32)

    _assert_close_1e6(f128.phi, phi_ref, f"boltz_ions(beta={beta_val}):phi")
    _assert_close_1e6(f128.apar, apar_ref, f"boltz_ions(beta={beta_val}):apar")
    np.testing.assert_array_equal(np.asarray(f128.bpar), 0.0)
    np.testing.assert_array_equal(np.asarray(f128.phi_avg), 0.0)

    # Confirm ky=0 uses plain (tau_fac + QPhi) WITHOUT <phi> subtraction
    nbar = real_space_density(g128, solver64)
    expected_ky0 = nbar[:, 0, 1:] / (cfg.tau_fac + solver64.qneut_fac_phi[:, 0, 1:])
    _assert_close_1e6(f128.phi[:, 0, 1:], expected_ky0, "boltz_ions:ky0_no_phi_avg")

    _assert_close_fp32_1e6(f64.phi, phi_ref, f"boltz_ions(beta={beta_val}):phi_c64")
    _assert_close_fp32_1e6(f64.apar, apar_ref, f"boltz_ions(beta={beta_val}):apar_c64")


@pytest.mark.parametrize(
    ("fphi", "fapar", "fbpar"),
    [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.75, 0.5, 0.85),
        (0.0, 0.0, 0.0),
    ],
)
def test_electromagnetic_field_switches_and_exact_maxwell_identities(
    fphi: float,
    fapar: float,
    fbpar: float,
) -> None:
    """Verify coupled (phi, apar, bpar) switches (test_solver.cu) and exact Maxwell laws."""
    kbm_in = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    cfg = dataclasses.replace(kbm_in, nx=7, ny=7, fphi=fphi, fapar=fapar, fbpar=fbpar)
    cfg, grids, geo = init_geometry(cfg)

    solver64 = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    solver32 = solver64.astype(jnp.float32)

    g128 = _random_complex_g(
        grids.g_shape, np.asarray(grids.active_mask), seed=303, dtype=jnp.complex128
    )
    phi_ref, apar_ref, bpar_ref, phi_avg_ref, _ = _cpp_reference_field_solve(
        np.asarray(g128), cfg, grids, geo
    )

    f128 = jax.jit(solve_fields)(g128, solver64)
    f64 = jax.jit(solve_fields)(g128.astype(jnp.complex64), solver32)

    _assert_close_1e6(f128.phi, phi_ref, f"em({fphi},{fapar},{fbpar}):phi")
    _assert_close_1e6(f128.apar, apar_ref, f"em({fphi},{fapar},{fbpar}):apar")
    _assert_close_1e6(f128.bpar, bpar_ref, f"em({fphi},{fapar},{fbpar}):bpar")
    _assert_close_1e6(f128.phi_avg, phi_avg_ref, f"em({fphi},{fapar},{fbpar}):phi_avg")

    _assert_close_fp32_1e6(f64.phi, phi_ref, f"em({fphi},{fapar},{fbpar}):phi_c64")
    _assert_close_fp32_1e6(f64.apar, apar_ref, f"em({fphi},{fapar},{fbpar}):apar_c64")
    _assert_close_fp32_1e6(f64.bpar, bpar_ref, f"em({fphi},{fapar},{fbpar}):bpar_c64")

    # When fphi == fapar == fbpar == 1.0, check exact self-consistent gyrokinetic Maxwell laws
    if (fphi, fapar, fbpar) == (1.0, 1.0, 1.0):
        moms = compute_moments(g128, f128, solver64)

        # 1. Quasineutrality from H: sum_s (n_s Z_s) n_bar_s == (sum_s n_s Z_s^2 / T_s) phi
        qn_lhs = jnp.sum(solver64.sp_nz[:, None, None, None] * moms.n_bar, axis=0)
        qn_rhs = jnp.sum(solver64.sp_nz * solver64.sp_zt) * f128.phi
        _assert_close_1e6(qn_lhs, qn_rhs, "em_maxwell:quasineutrality_H")

        # 2. Parallel Ampere from H: sum_s jparfac_s upar_bar_s == (kperp2 * B^2) apar
        ap_lhs = jnp.sum(solver64.sp_jparfac[:, None, None, None] * moms.upar_bar, axis=0)
        ap_rhs = solver64.kperp2 * (solver64.bmag[:, None, None] ** 2) * f128.apar
        _assert_close_1e6(ap_lhs, ap_rhs, "em_maxwell:parallel_ampere_H")

        # 3. Coupled 2x2 matrix residuals:
        #    QPhi * phi + QB * bpar == nbar,  APhi * phi + AB * bpar == jperpbar
        res_q = solver64.qneut_fac_phi * f128.phi + solver64.qneut_fac_bpar * f128.bpar
        res_a = solver64.ampere_perp_fac_phi * f128.phi + solver64.ampere_perp_fac_bpar * f128.bpar
        _assert_close_1e6(res_q, moms.nbar, "em_maxwell:2x2_row1")
        _assert_close_1e6(res_a, moms.jperpbar, "em_maxwell:2x2_row2")


def test_long_wavelength_gk_no_fields_and_phi_ext_options() -> None:
    """Verify long_wavelength_GK, no_fields=True, and phi_ext source injection."""
    cfg_raw = load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_kinetic_electrons.in")
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
            beta=cfg_raw.beta,
            long_wavelength_gk=True,
        )
        for sp in cfg_raw.species
    )
    cfg_lw = dataclasses.replace(cfg_raw, nx=7, ny=7, long_wavelength_gk=True, species=sp_lw)
    cfg_lw, grids, geo = init_geometry(cfg_lw)

    g128 = _random_complex_g(
        grids.g_shape, np.asarray(grids.active_mask), seed=404, dtype=jnp.complex128
    )
    solver_lw = init_field_solver(cfg_lw, grids, geo, dtype=jnp.float64, phi_ext=0.125)
    phi_ref, apar_ref, bpar_ref, phi_avg_ref, mats_ref = _cpp_reference_field_solve(
        np.asarray(g128), cfg_lw, grids, geo, phi_ext=0.125
    )

    f_lw = solve_fields(g128, solver_lw)
    _assert_close_1e6(solver_lw.qneut_fac_phi, mats_ref["qneut_fac_phi"], "lw_gk:qneut_fac_phi")
    _assert_close_1e6(f_lw.phi, phi_ref, "lw_gk:phi_with_ext")
    _assert_close_1e6(f_lw.apar, apar_ref, "lw_gk:apar")
    _assert_close_1e6(f_lw.bpar, bpar_ref, "lw_gk:bpar")
    _assert_close_1e6(f_lw.phi_avg, phi_avg_ref, "lw_gk:phi_avg")

    # Verify no_fields=True returns exact zeros
    solver_nf = init_field_solver(cfg_lw, grids, geo, no_fields=True)
    f_nf = solve_fields(g128, solver_nf)
    np.testing.assert_array_equal(np.asarray(f_nf.phi), 0.0)
    np.testing.assert_array_equal(np.asarray(f_nf.apar), 0.0)
    np.testing.assert_array_equal(np.asarray(f_nf.bpar), 0.0)
    np.testing.assert_array_equal(np.asarray(f_nf.phi_avg), 0.0)


# ---------------------------------------------------------------------------
# 4. 3D Stellarator W7-X Verification (Linked & NTFT, Adiabatic & Full EM)
# ---------------------------------------------------------------------------


def test_w7x_stellarator_adiabatic_and_electromagnetic_field_solve(tmp_path: Path) -> None:
    """Verify W7-X stellarator field solve (non-uniform B(z), Jacobian(z), linked & NTFT)."""
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    cfg_w7x = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    eik_nc_path = tmp_path / "w7x_session2.eik.nc"
    build_vmec_eik_nc(cfg_w7x, w7x_dir / "wout_w7x.nc", eik_nc_path)

    # Case A: Adiabatic electrons with non-uniform 3D stellarator Jacobian(z)
    cfg_adiab = dataclasses.replace(
        cfg_w7x,
        geo_option="nc",
        geofilename=str(eik_nc_path),
        nx=10,
        ny=10,
    )
    cfg_adiab, grids_a, geo_a = load_eik_nc(eik_nc_path, cfg_adiab)
    solver_a64 = init_field_solver(cfg_adiab, grids_a, geo_a, dtype=jnp.float64)
    solver_a32 = solver_a64.astype(jnp.float32)

    g_a128 = _random_complex_g(
        grids_a.g_shape, np.asarray(grids_a.active_mask), seed=505, dtype=jnp.complex128
    )
    phi_ref, _, _, phi_avg_ref, _ = _cpp_reference_field_solve(
        np.asarray(g_a128), cfg_adiab, grids_a, geo_a
    )
    fa64 = jax.jit(solve_fields)(g_a128.astype(jnp.complex64), solver_a32)
    _assert_close_fp32_1e6(fa64.phi, phi_ref, "w7x_adiab:phi_c64")
    _assert_close_fp32_1e6(fa64.phi_avg, phi_avg_ref, "w7x_adiab:phi_avg_c64")

    favg_w7x, _ = field_line_average(fa64.phi, grids_a, geo_a.astype(jnp.float32))
    _assert_close_fp32_1e6(favg_w7x, phi_avg_ref, "w7x_adiab:flux_avg_identity")

    # Case B: 2-species kinetic electromagnetic (phi, apar, bpar) on W7-X with NTFT (non_twist=True)
    beta_em = 0.03
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
    cfg_em = dataclasses.replace(
        cfg_adiab,
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
    cfg_em, grids_em, geo_em = load_eik_nc(eik_nc_path, cfg_em)
    solver_em64 = init_field_solver(cfg_em, grids_em, geo_em, dtype=jnp.float64)
    solver_em32 = solver_em64.astype(jnp.float32)

    g_em128 = _random_complex_g(
        grids_em.g_shape, np.asarray(grids_em.active_mask), seed=506, dtype=jnp.complex128
    )
    phi_em_ref, apar_em_ref, bpar_em_ref, _, _ = _cpp_reference_field_solve(
        np.asarray(g_em128), cfg_em, grids_em, geo_em
    )
    fem64 = jax.jit(solve_fields)(g_em128.astype(jnp.complex64), solver_em32)
    _assert_close_fp32_1e6(fem64.phi, phi_em_ref, "w7x_ntft_em:phi_c64")
    _assert_close_fp32_1e6(fem64.apar, apar_em_ref, "w7x_ntft_em:apar_c64")
    _assert_close_fp32_1e6(fem64.bpar, bpar_em_ref, "w7x_ntft_em:bpar_c64")


# ---------------------------------------------------------------------------
# 5. Golden C++ NetCDF (.restart.nc -> .nc) Fixture Verification
# ---------------------------------------------------------------------------


def _kx_active_to_out_order(arr_ky_kx: np.ndarray, nakx: int) -> np.ndarray:
    """Reorder active FFT kx [0, 1..K, -K..-1] -> monotonic [-K..-1, 0, 1..K] matching .nc."""
    if nakx <= 1:
        return arr_ky_kx
    half = nakx // 2
    order = [*range(half + 1, nakx), 0, *range(1, half + 1)]
    return arr_ky_kx[..., order]


def test_golden_cpp_kh01_and_kh01a_restart_field_solve_and_roundtrip(tmp_path: Path) -> None:
    """Verify load_restart_nc + solve_fields against golden C++ kh01.nc and kh01a.nc."""
    # --- Part A: kh01 (linear initial state t=0) ---
    cfg_kh01, grids_kh01, geo_kh01 = init_geometry(load_config(UNIT_INPUTS_DIR / "kh01.in"))
    g_kh01, t_kh01 = load_restart_nc(
        UNIT_INPUTS_DIR / "kh01_restart.nc", grids_kh01, dtype=jnp.complex64
    )
    assert t_kh01 == pytest.approx(0.0, abs=1e-15)

    solver_kh01 = init_field_solver(cfg_kh01, grids_kh01, geo_kh01).astype(jnp.float32)
    fields_kh01 = jax.jit(solve_fields)(g_kh01, solver_kh01)

    with netCDF4.Dataset(UNIT_INPUTS_DIR / "kh01.nc", "r") as ds:
        wphi_ref = float(ds["Spectra"]["Pst"][0, 0])
        wg_ref = float(ds["Spectra"]["Wst"][0, 0])
        akxky_ref = np.asarray(ds["Spectra"]["Akxkyst"][0], dtype=np.float64)

    fac_ky = np.where(np.arange(grids_kh01.naky)[:, None] == 0, 1.0, 2.0)
    vol_fac = np.asarray(geo_kh01.vol_fac, dtype=np.float64)[:, None, None]
    phi_np = np.asarray(fields_kh01.phi, dtype=np.complex128)
    b_ion = np.asarray(geo_kh01.kperp2, dtype=np.float64) * cfg_kh01.species[0].rho2

    # Akxkyst in adiabatic-electron mode is 0.5 * fac * tau_fac * |phi - <phi>|^2
    dphi_kh01 = phi_np.copy()
    dphi_kh01[:, 0, :] -= np.asarray(fields_kh01.phi_avg, dtype=np.complex128)[None, :]
    akxky_calc = _kx_active_to_out_order(
        np.sum(
            0.5 * fac_ky[None, :, :] * vol_fac * cfg_kh01.tau_fac * np.abs(dphi_kh01) ** 2,
            axis=0,
        ),
        grids_kh01.nakx,
    )
    wphi_calc = float(
        np.sum(
            0.5 * fac_ky[None, :, :] * vol_fac * (1.0 - sp_special.i0e(b_ion)) * np.abs(phi_np) ** 2
        )
    )
    wg_calc = float(
        np.sum(
            0.5
            * fac_ky[None, :, :]
            * vol_fac
            * cfg_kh01.species[0].nt
            * np.abs(np.asarray(g_kh01, dtype=np.complex128)) ** 2
        )
    )

    _assert_close_fp32_1e6(akxky_calc, akxky_ref, "kh01:Akxkyst")
    # Note: C++ g0(b) uses a float32 rational approximation (~8e-6 vs exact i0e at b=0.01),
    # whereas Akxkyst tests |phi|^2 directly from Solver_GK to 2e-7!
    np.testing.assert_allclose(wphi_calc, wphi_ref, rtol=1.5e-5)
    _assert_close_fp32_1e6(wg_calc, wg_ref, "kh01:Wg")

    # --- Part B: kh01a (nonlinear secondary instability at t=4.0 with active ky=0 zonal flow) ---
    cfg_kh01a, grids_kh01a, geo_kh01a = init_geometry(load_config(UNIT_INPUTS_DIR / "kh01a.in"))
    g_kh01a, t_kh01a = load_restart_nc(
        UNIT_INPUTS_DIR / "kh01a_restart.nc", grids_kh01a, dtype=jnp.complex64
    )
    assert t_kh01a == pytest.approx(4.0, rel=1e-6)

    solver_kh01a = init_field_solver(cfg_kh01a, grids_kh01a, geo_kh01a).astype(jnp.float32)
    fields_kh01a = jax.jit(solve_fields)(g_kh01a, solver_kh01a)

    # Confirm non-trivial ky=0 zonal potential <phi>(kx != 0) is active and equals phi(z, ky=0, kx)
    assert float(jnp.max(jnp.abs(fields_kh01a.phi_avg[1:]))) > 0.1
    _assert_close_fp32_1e6(
        fields_kh01a.phi[:, 0, :],
        np.broadcast_to(
            np.asarray(fields_kh01a.phi_avg)[None, :], (grids_kh01a.nz, grids_kh01a.nakx)
        ),
        "kh01a:uniform_zonal_phi_equals_phi_avg",
    )

    with netCDF4.Dataset(UNIT_INPUTS_DIR / "kh01a.nc", "r") as ds_a:
        akxky_a_ref = np.asarray(ds_a["Spectra"]["Akxkyst"][-1], dtype=np.float64)
        pst_a_ref = float(ds_a["Spectra"]["Pst"][-1, 0])

    phi_a = np.asarray(fields_kh01a.phi, dtype=np.complex128)
    dphi_a = phi_a.copy()
    dphi_a[:, 0, :] -= np.asarray(fields_kh01a.phi_avg, dtype=np.complex128)[None, :]
    akxky_a_calc = _kx_active_to_out_order(
        np.sum(
            0.5 * fac_ky[None, :, :] * vol_fac * cfg_kh01a.tau_fac * np.abs(dphi_a) ** 2,
            axis=0,
        ),
        grids_kh01a.nakx,
    )
    _assert_close_fp32_1e6(akxky_a_calc, akxky_a_ref, "kh01a:Akxkyst")

    b_a = np.asarray(geo_kh01a.kperp2, dtype=np.float64) * cfg_kh01a.species[0].rho2
    # C++ Wphi_summand (device_funcs.cu:L1499-1501) computes float32(1.0) - float32(g0(b))
    one_minus_g0_f32 = (np.float32(1.0) - sp_special.i0e(b_a).astype(np.float32)).astype(np.float64)
    pst_a_f32_diag = float(
        np.sum(0.5 * fac_ky[None, :, :] * vol_fac * one_minus_g0_f32 * np.abs(phi_a) ** 2)
    )
    _assert_close_fp32_1e6(pst_a_f32_diag, pst_a_ref, "kh01a:Pst_cpp_f32_diag")
    pst_a_exact = float(
        np.sum(
            0.5 * fac_ky[None, :, :] * vol_fac * (1.0 - sp_special.i0e(b_a)) * np.abs(phi_a) ** 2
        )
    )
    np.testing.assert_allclose(pst_a_exact, pst_a_ref, rtol=1.5e-5)

    # --- Part C: save_restart_nc -> load_restart_nc round-trip ---
    rt_path = tmp_path / "roundtrip_restart.nc"
    save_restart_nc(rt_path, g_kh01a, t_kh01a, grids_kh01a)
    g_reloaded, t_reloaded = load_restart_nc(rt_path, grids_kh01a, dtype=jnp.complex64)
    assert t_reloaded == pytest.approx(t_kh01a, rel=1e-12)
    np.testing.assert_array_equal(np.asarray(g_reloaded), np.asarray(g_kh01a))


# ---------------------------------------------------------------------------
# 6. Moments, Reality Condition, Init Modes, JIT, AD (jax.grad), and NaN Safety
# ---------------------------------------------------------------------------


def test_moments_conservation_terms_reality_and_init_modes() -> None:
    """Verify compute_moments, compute_conservation_moments (nl=1 & nl>1), reality, and inits."""
    cfg_in = load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in")
    cfg = dataclasses.replace(cfg_in, nx=7, ny=7, nhermite=6, nlaguerre=3)
    cfg, grids, geo = init_geometry(cfg)
    solver = init_field_solver(cfg, grids, geo, dtype=jnp.float64)

    g128 = _random_complex_g(
        grids.g_shape, np.asarray(grids.active_mask), seed=606, dtype=jnp.complex128
    )
    fields = solve_fields(g128, solver)
    moms = jax.jit(compute_moments)(g128, fields, solver)

    # Named fluid moments
    _assert_close_1e6(moms.dens, g128[:, 0, 0], "moments:dens")
    _assert_close_1e6(moms.upar, g128[:, 1, 0], "moments:upar")
    _assert_close_1e6(moms.tpar, math.sqrt(2.0) * g128[:, 2, 0], "moments:tpar")
    _assert_close_1e6(moms.qpar, math.sqrt(6.0) * g128[:, 3, 0], "moments:qpar")
    _assert_close_1e6(moms.tperp, g128[:, 0, 1], "moments:tperp")
    _assert_close_1e6(moms.qperp, g128[:, 1, 1], "moments:qperp")

    # Independent NumPy check of H-moments (n_bar, upar_bar, uperp_bar, t_bar)
    kperp2 = np.asarray(geo.kperp2, dtype=np.float64)
    g_np = np.asarray(g128)
    phi_np = np.asarray(fields.phi)
    apar_np = np.asarray(fields.apar)
    bpar_np = np.asarray(fields.bpar)
    for s_idx, sp in enumerate(cfg.species):
        b_s = kperp2 * sp.rho2
        n_b = np.zeros_like(phi_np)
        up_b = np.zeros_like(phi_np)
        uperp_b = np.zeros_like(phi_np)
        t_b = np.zeros_like(phi_np)
        for ell in range(grids.nlaguerre):
            jl_v = _cpp_jflr_np(ell, b_s)
            jlb_v = _cpp_jflr_b_np(ell, b_s)
            jla_v = _cpp_jflr_a_np(ell, b_s)
            h0 = g_np[s_idx, 0, ell] + sp.zt * jl_v * phi_np + jlb_v * bpar_np
            h1 = g_np[s_idx, 1, ell] - sp.zt * sp.vt * jl_v * apar_np
            h2 = g_np[s_idx, 2, ell]
            n_b += jl_v * h0
            up_b += jl_v * h1
            uperp_b += jlb_v * h0
            t_b += (2.0 / 3.0) * (jla_v * h0 + (1.0 / math.sqrt(2.0)) * jl_v * h2)
        uperp_b *= np.sqrt(b_s)
        _assert_close_1e6(moms.n_bar[s_idx], n_b, f"moments:n_bar[{s_idx}]")
        _assert_close_1e6(moms.upar_bar[s_idx], up_b, f"moments:upar_bar[{s_idx}]")
        _assert_close_1e6(moms.uperp_bar[s_idx], uperp_b, f"moments:uperp_bar[{s_idx}]")
        _assert_close_1e6(moms.t_bar[s_idx], t_b, f"moments:t_bar[{s_idx}]")

    # zero_ky0=True option matches calc_*_bar diagnostic guard
    moms_ky_gt0 = compute_moments(g128, fields, solver, zero_ky0=True)
    np.testing.assert_array_equal(np.asarray(moms_ky_gt0.n_bar[:, :, 0, :]), 0.0)
    _assert_close_1e6(moms_ky_gt0.n_bar[:, :, 1:, :], moms.n_bar[:, :, 1:, :], "moments:zero_ky0")

    # Conservation moments for nl == 1 special branch (device_funcs.cu:L3509-3510)
    cfg_nl1 = dataclasses.replace(cfg, nlaguerre=1)
    _, grids_nl1, geo_nl1 = init_geometry(cfg_nl1)
    solver_nl1 = init_field_solver(cfg_nl1, grids_nl1, geo_nl1, dtype=jnp.float64)
    g_nl1 = _random_complex_g(
        grids_nl1.g_shape, np.asarray(grids_nl1.active_mask), seed=607, dtype=jnp.complex128
    )
    f_nl1 = solve_fields(g_nl1, solver_nl1)
    _, _, t_bar_nl1 = compute_conservation_moments(g_nl1, f_nl1, solver_nl1)
    expected_t_nl1 = math.sqrt(2.0) * solver_nl1.jl[:, 0] * g_nl1[:, 2, 0]
    _assert_close_1e6(t_bar_nl1, expected_t_nl1, "conservation_moments:nl1_t_bar")

    # Reality condition: enforce_reality guarantees Hermitian symmetry on ky=0 => real irfft2!
    g_real_sym = enforce_reality(g128, grids)
    f_real_sym = solve_fields(g_real_sym, solver)
    phi_full = pad_kykx_to_full(f_real_sym.phi, grids)
    # Reconstruct full 2D Fourier plane on ky=0: phi(0, -kx) == conj(phi(0, +kx))
    kx_cut = (grids.nx - 1) // 3
    _assert_close_1e6(
        f_real_sym.phi[:, 0, kx_cut + 1 :],
        jnp.conj(f_real_sym.phi[:, 0, 1 : kx_cut + 1][:, ::-1]),
        "reality:phi_ky0_conjugate_symmetry",
    )
    phi_xy = jnp.fft.irfft2(jnp.swapaxes(phi_full, -2, -1), s=(grids.nx, grids.ny))
    assert jnp.all(jnp.isfinite(phi_xy))

    # init_moments modes: single (ky=0 reality pair), gaussian, all-moments, electrons-only
    cfg_single_zonal = dataclasses.replace(
        cfg,
        init_single=True,
        init_electrons_only=False,
        iky_single=0,
        ikx_single=1,
        ikpar_init=0,
        init_amp=0.5,
    )
    g_sz = init_moments(cfg_single_zonal, grids, geo, dtype=jnp.complex128)
    _assert_close_1e6(g_sz[:, 0, 0, :, 0, 1], np.full((grids.nspecies, grids.nz), 0.5), "init:sz+")
    _assert_close_1e6(
        g_sz[:, 0, 0, :, 0, grids.nakx - 1],
        np.full((grids.nspecies, grids.nz), 0.5),
        "init:sz-",
    )

    cfg_gauss = dataclasses.replace(
        cfg,
        init_single=False,
        gaussian_init=True,
        init_field="all",
        init_electrons_only=True,
    )
    g_gauss = init_moments(cfg_gauss, grids, geo, dtype=jnp.complex64)
    assert g_gauss.shape == grids.g_shape
    assert g_gauss.dtype == jnp.complex64
    np.testing.assert_array_equal(np.asarray(g_gauss[0]), 0.0)  # ion zeroed
    assert float(jnp.max(jnp.abs(g_gauss[1]))) > 0.0  # electron populated


def test_differentiability_and_debug_nans_safety_across_all_branches() -> None:
    """Verify jax.grad through solve_fields & compute_moments is NaN-free under debug_nans."""
    cases = [
        # 1. All-kinetic full electromagnetic
        load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"),
        # 2. Adiabatic electrons with zonal <phi>
        load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in"),
        # 3. Adiabatic ions with finite-beta apar
        dataclasses.replace(
            load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"),
            all_kinetic=False,
            add_boltzmann_species=True,
            boltzmann_opt=BoltzmannOption.BOLTZMANN_IONS,
            boltzmann_type="ions",
            beta=0.01,
            fapar=1.0,
            species=(
                SpeciesConfig.from_raw(
                    z=-1.0, mass=2.7e-4, dens=1.0, temp=1.0, stype="electron", beta=0.01
                ),
            ),
        ),
    ]

    for idx, cfg_raw in enumerate(cases):
        cfg = dataclasses.replace(cfg_raw, nx=7, ny=7)
        cfg, grids, geo = init_geometry(cfg)
        solver = init_field_solver(cfg, grids, geo).astype(jnp.float32)
        g_in = _random_complex_g(
            grids.g_shape, np.asarray(grids.active_mask), seed=700 + idx, dtype=jnp.complex64
        )
        # Deliberately put non-zero garbage in masked (iky=0, ikx=0) entry to prove safe_divide
        # never triggers 0/0 = NaN in forward or backward pass!
        g_in = g_in.at[:, :, :, :, 0, 0].set(1.0 + 2.0j)

        def _energy_loss(g_state: jax.Array, slv: FieldSolver = solver) -> jax.Array:
            f: Fields = slv.solve(g_state)
            m = compute_moments(g_state, f, slv)
            return (
                jnp.sum(jnp.abs(f.phi) ** 2)
                + jnp.sum(jnp.abs(f.apar) ** 2)
                + jnp.sum(jnp.abs(f.bpar) ** 2)
                + jnp.sum(jnp.abs(f.phi_avg) ** 2)
                + jnp.sum(jnp.abs(m.n_bar) ** 2)
            )

        with jax.debug_nans(True):
            loss_val, grad_g = jax.jit(jax.value_and_grad(_energy_loss))(g_in)
        assert jnp.isfinite(loss_val)
        assert jnp.all(jnp.isfinite(grad_g))
        # Masked (0, 0) mode has zero contribution to fields/moments => exact zero gradient!
        np.testing.assert_array_equal(np.asarray(grad_g[:, :, :, :, 0, 0]), 0.0)


def test_fields_and_moments_pytrees_and_standalone_currents() -> None:
    """Verify init_fields, Fields.astype, Moments.astype, mask_g, and standalone currents."""
    cfg, grids, geo = init_geometry(load_config(BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"))
    f0 = init_fields(grids, dtype=jnp.complex64)
    assert f0.phi.shape == grids.field_shape
    assert f0.phi_avg.shape == (grids.nakx,)
    f0_128 = f0.astype(jnp.complex128)
    assert f0_128.phi.dtype == jnp.complex128

    solver = init_field_solver(cfg, grids, geo, dtype=jnp.float64)
    g_raw = (
        _random_complex_g(
            grids.g_shape, np.asarray(grids.active_mask), seed=808, dtype=jnp.complex128
        )
        .at[:, :, :, :, 0, 0]
        .set(5.0 - 3.0j)
    )
    g_masked = mask_g(g_raw, grids)
    np.testing.assert_array_equal(np.asarray(g_masked[:, :, :, :, 0, 0]), 0.0)

    f_sol = solve_fields(g_masked, solver)
    moms = compute_moments(g_masked, f_sol, solver)
    _assert_close_1e6(real_space_density(g_masked, solver), moms.nbar, "standalone:nbar")
    _assert_close_1e6(real_space_par_current(g_masked, solver), moms.jparbar, "standalone:jparbar")
    _assert_close_1e6(
        real_space_perp_current(g_masked, solver), moms.jperpbar, "standalone:jperpbar"
    )
    moms64 = moms.astype(jnp.complex64)
    assert moms64.dens.dtype == jnp.complex64
