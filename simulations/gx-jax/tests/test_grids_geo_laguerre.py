"""Grids, Geometry (.nc / Miller / s-alpha / VMEC), and Laguerre-Bessel.

Verifies GX-JAX core modules against C++ GX reference benchmarks in
``gx/benchmarks/`` and unit-test golden tables to ``rtol < 1e-6``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest
from scipy import special as sp_special

import gx_jax
from gx_jax.config import GXConfig, load_config
from gx_jax.geometry_nc import (
    build_vmec_eik_nc,
    calculate_bgrad,
    init_geometry,
    load_eik_nc,
    s_alpha_geometry,
)
from gx_jax.grids import init_grids, pad_kykx_to_full, truncate_full_to_kykx
from gx_jax.laguerre import (
    bessel_j0,
    bessel_j1,
    g0,
    g1,
    init_laguerre,
    j0_phi_and_bpar_to_grid,
    j0_to_grid,
    j1_over_alpha,
    jfac,
    jflr,
    jflr_a,
    jflr_b,
    laguerre_to_grid,
    laguerre_to_spectral,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
GOLDEN_BENCH_DIR = REPO_ROOT / "golden_fixtures" / "benchmarks"

# Golden C++ GX .out.nc benchmark pairs paired with .in decks in benchmarks/
BENCHMARK_CASES: list[tuple[str, Path, Path]] = [
    (
        "itg_salpha_adiabatic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in",
        GOLDEN_BENCH_DIR
        / "linear"
        / "ITG_cyclone"
        / "itg_salpha_adiabatic_electrons_correct.out.nc",
    ),
    (
        "itg_miller_adiabatic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_adiabatic_electrons.in",
        GOLDEN_BENCH_DIR
        / "linear"
        / "ITG_cyclone"
        / "itg_miller_adiabatic_electrons_correct.out.nc",
    ),
    (
        "itg_miller_kinetic",
        BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_kinetic_electrons.in",
        GOLDEN_BENCH_DIR / "linear" / "ITG_cyclone" / "itg_miller_kinetic_electrons_correct.out.nc",
    ),
    (
        "kbm_miller",
        BENCH_DIR / "linear" / "KBM" / "kbm_miller.in",
        GOLDEN_BENCH_DIR / "linear" / "KBM" / "kbm_miller_correct.out.nc",
    ),
    (
        "kaw_slab",
        BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01.in",
        GOLDEN_BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01_correct.out.nc",
    ),
]


def _assert_close_1e6(actual: np.ndarray | float, expected: np.ndarray | float, label: str) -> None:
    """Strict FP64 relative-error check (< 1e-6) with near-zero floor atol=1e-12."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1.0e-6,
        atol=1.0e-12,
        err_msg=f"Mismatch in {label}",
    )


def _assert_close_fp32_1e6(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    label: str,
) -> None:
    """Assert < 1e-6 relative error against single-precision (float32 / NC_FLOAT) references.

    Checks both relative L2 vector norm < 1e-6 and elementwise rtol=1e-6 with a scale-aware
    4-ULP float32 floor (5e-7 * max(1, ||expected||_inf)) at oscillatory zero-crossings.
    """
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
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


# ---------------------------------------------------------------------------
# 1. Precision & Architectural Invariants
# ---------------------------------------------------------------------------


def test_tpu_precision_invariants() -> None:
    assert gx_jax is not None
    assert jax.config.jax_enable_x64 is True
    assert jax.config.jax_default_matmul_precision == "highest"


def test_grids_nonlinear_option_a_layout_and_dealiasing() -> None:
    """Port C++ test_grids.cu (Nx=256, Ny=256, Nz=64) and Option A pad/truncate helpers."""
    cfg = GXConfig(
        nx=256,
        ny=256,
        nz=64,
        nspecies=2,
        nhermite=8,
        nlaguerre=4,
        x0=10.0,
        y0=20.0,
        zp=1,
    )
    grids = init_grids(cfg)

    assert grids.nyc == 129
    assert grids.naky == 86
    assert grids.nakx == 171
    assert grids.nj == 5
    assert grids.g_shape == (2, 8, 4, 64, 86, 171)
    assert grids.field_shape == (64, 86, 171)

    # Number of active unmasked modes matches full 2/3 dealiasing mask
    assert int(jnp.sum(grids.active_mask)) == grids.naky * grids.nakx - 1
    assert int(jnp.sum(grids.unmasked_full)) == int(jnp.sum(grids.active_mask))
    assert not bool(grids.active_mask[0, 0])

    # Spot-check wavenumber entries from gx/unit_tests/src/test_grids.cu
    _assert_close_1e6(float(grids.kx_full[1]), 0.1, "kx[1]")
    _assert_close_1e6(float(grids.kx_full[128]), 12.8, "kx[128]")
    _assert_close_1e6(float(grids.kx_full[129]), -12.7, "kx[129]")
    _assert_close_1e6(float(grids.ky[23]), 23.0 / 20.0, "ky[23]")
    _assert_close_1e6(float(grids.kz[32]), 32.0, "kz[32]")
    _assert_close_1e6(float(grids.kz[33]), -31.0, "kz[33]")

    # Exact pad -> truncate round-trip on (nz, naky, nakx)
    rng = np.random.default_rng(42)
    arr = jnp.asarray(
        rng.standard_normal(grids.field_shape) + 1j * rng.standard_normal(grids.field_shape),
        dtype=jnp.complex64,
    )
    arr = jnp.where(grids.active_mask[None, :, :], arr, 0.0)
    padded = jax.jit(pad_kykx_to_full)(arr, grids)
    assert padded.shape == (64, 129, 256)
    recovered = jax.jit(truncate_full_to_kykx)(padded, grids)
    np.testing.assert_array_equal(np.asarray(recovered), np.asarray(arr))


# ---------------------------------------------------------------------------
# 2. Grids & Geometry Verification Against All 5 C++ GX .out.nc Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_name", "in_path", "nc_path"),
    BENCHMARK_CASES,
    ids=[c[0] for c in BENCHMARK_CASES],
)
def test_grids_and_geometry_against_cpp_benchmark_nc(
    case_name: str,
    in_path: Path,
    nc_path: Path,
) -> None:
    cfg_in = load_config(in_path)
    cfg, grids, geo = init_geometry(cfg_in)

    with netCDF4.Dataset(nc_path, "r") as ds:
        g_grp = ds.groups["Grids"]
        geo_grp = ds.groups["Geometry"]
        dom_grp = ds.groups["Inputs"].groups["Domain"]

        # Verify Domain resolution (x0, y0, zp, jtwist) against NC_FLOAT .out.nc
        _assert_close_fp32_1e6(grids.x0, float(dom_grp.variables["x0"][:]), f"{case_name}:x0")
        _assert_close_fp32_1e6(grids.y0, float(dom_grp.variables["y0"][:]), f"{case_name}:y0")
        assert grids.zp == int(dom_grp.variables["zp"][:])
        assert grids.jtwist == int(dom_grp.variables["jtwist"][:])

        # Verify Grids group arrays
        _assert_close_fp32_1e6(grids.kx_out, g_grp.variables["kx"][:], f"{case_name}:kx")
        _assert_close_fp32_1e6(grids.ky, g_grp.variables["ky"][:], f"{case_name}:ky")
        _assert_close_fp32_1e6(grids.kz_out, g_grp.variables["kz"][:], f"{case_name}:kz")
        _assert_close_fp32_1e6(grids.x, g_grp.variables["x"][:], f"{case_name}:x")
        _assert_close_fp32_1e6(grids.y, g_grp.variables["y"][:], f"{case_name}:y")
        _assert_close_fp32_1e6(grids.z, g_grp.variables["theta"][:], f"{case_name}:theta")
        _assert_close_fp32_1e6(geo.z, g_grp.variables["theta"][:], f"{case_name}:geo.z")

        # Verify Geometry scalar outputs
        _assert_close_fp32_1e6(
            geo.gradpar, float(geo_grp.variables["gradpar"][:]), f"{case_name}:gradpar"
        )
        _assert_close_fp32_1e6(geo.shat, float(geo_grp.variables["shat"][:]), f"{case_name}:shat")
        _assert_close_fp32_1e6(geo.qsf, float(geo_grp.variables["q"][:]), f"{case_name}:q")
        _assert_close_fp32_1e6(geo.rmaj, float(geo_grp.variables["rmaj"][:]), f"{case_name}:rmaj")
        _assert_close_fp32_1e6(
            geo.kxfac, float(geo_grp.variables["kxfac"][:]), f"{case_name}:kxfac"
        )
        _assert_close_fp32_1e6(
            geo.drhodpsi, float(geo_grp.variables["drhodpsi"][:]), f"{case_name}:drhodpsi"
        )

        # Verify all 1D field-line geometry arrays
        bmag_expected = np.asarray(geo_grp.variables["bmag"][:], dtype=np.float64)
        bmag_scale = max(1.0, float(np.max(np.abs(bmag_expected))))
        for var_name in (
            "bmag",
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
            expected = np.asarray(geo_grp.variables[var_name][:], dtype=np.float64)
            actual = np.asarray(getattr(geo, var_name), dtype=np.float64)
            _assert_close_fp32_1e6(actual, expected, f"{case_name}:Geometry/{var_name}")

        # Verify bgrad: C++ computes bgrad via single-precision cufftExecR2C on float32 bmag (~1.0),
        # which injects ~2e-7 (< 2 FP32 ULPs of bmag) FFT roundoff noise into .out.nc.
        expected_bgrad = np.asarray(geo_grp.variables["bgrad"][:], dtype=np.float64)
        actual_bgrad = np.asarray(geo.bgrad, dtype=np.float64)
        rel_to_bmag = float(np.max(np.abs(actual_bgrad - expected_bgrad))) / bmag_scale
        assert rel_to_bmag < 1.0e-6, f"{case_name}:bgrad diff/bmag = {rel_to_bmag:.3e} >= 1e-6"
        np.testing.assert_allclose(actual_bgrad, expected_bgrad, rtol=1.0e-6, atol=3.0e-7)
        if cfg_in.geo_option == "miller":
            bgrad_from_nc_bmag = calculate_bgrad(
                jnp.asarray(bmag_expected, dtype=jnp.float64), grids.kz, geo.gradpar
            )
            np.testing.assert_allclose(
                np.asarray(bgrad_from_nc_bmag), expected_bgrad, rtol=1.0e-6, atol=3.0e-7
            )

    # Verify 3D operator arrays (Option A: shape (nz, naky, nakx))
    expected_3d = (grids.nz, grids.naky, grids.nakx)
    assert geo.kperp2.shape == expected_3d
    assert geo.omegad.shape == expected_3d
    assert geo.cv_d.shape == expected_3d
    assert geo.gb_d.shape == expected_3d
    np.testing.assert_allclose(geo.kperp2[:, 0, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(geo.omegad[:, 0, 0], 0.0, atol=1e-15)
    assert bool(jnp.all(geo.kperp2 >= 0.0))
    _assert_close_1e6(geo.omegad, geo.cv_d + geo.gb_d, f"{case_name}:omegad_sum")
    _assert_close_1e6(float(jnp.sum(geo.vol_fac)), 1.0, f"{case_name}:vol_fac_norm")
    _assert_close_1e6(float(jnp.sum(geo.flux_fac * geo.grho)), 1.0, f"{case_name}:flux_fac_norm")
    assert cfg.x0 > 0.0


# ---------------------------------------------------------------------------
# 3. Stellarator W7-X .eik.nc Load & NTFT Verification
# ---------------------------------------------------------------------------


def test_w7x_stellarator_eik_nc_load_and_symmetries(tmp_path: Path) -> None:
    """Build a W7-X stellarator .eik.nc from wout_w7x.nc and verify load_eik_nc & NTFT."""
    w7x_dir = BENCH_DIR / "linear" / "ITG_w7x"
    cfg_w7x = load_config(w7x_dir / "itg_w7x_adiabatic_electrons.in")
    wout_path = w7x_dir / "wout_w7x.nc"
    eik_nc_path = tmp_path / "itg_w7x_adiabatic_electrons.eik.nc"

    build_vmec_eik_nc(cfg_w7x, wout_path, eik_nc_path)
    assert eik_nc_path.is_file()

    cfg_nc = dataclasses.replace(
        cfg_w7x,
        geo_option="nc",
        geofilename=str(eik_nc_path),
        nx=10,
        ny=10,
    )
    cfg_loaded, grids, geo = load_eik_nc(eik_nc_path, cfg_nc)

    assert geo.nfp == 5
    assert geo.bmag.shape == (cfg_loaded.nz,)
    assert geo.kperp2.shape == (grids.nz, grids.naky, grids.nakx)
    assert geo.gradpar > 0.0

    # Verify factor of 1/2 on drifts relative to raw .eik.nc midpoint (z = 0 is index nz // 2)
    mid = cfg_loaded.nz // 2
    with netCDF4.Dataset(eik_nc_path, "r") as ds:
        raw_gb_mid = float(ds.variables["gbdrift"][mid])
        raw_cv_mid = float(ds.variables["cvdrift"][mid])
        raw_bmag_mid = float(ds.variables["bmag"][mid])
    _assert_close_1e6(float(geo.bmag[mid]), raw_bmag_mid, "w7x:bmag_mid")
    _assert_close_1e6(float(geo.gbdrift[mid]), 0.5 * raw_gb_mid, "w7x:gbdrift_half_factor")
    _assert_close_1e6(float(geo.cvdrift[mid]), 0.5 * raw_cv_mid, "w7x:cvdrift_half_factor")

    # Stellarator symmetry around z = 0 (alpha = 0, zeta_center = 0):
    # indices k = 1..nz-1 pair symmetrically as k <-> nz - k
    bmag_np = np.asarray(geo.bmag)
    gds2_np = np.asarray(geo.gds2)
    gds21_np = np.asarray(geo.gds21)
    gb0_np = np.asarray(geo.gbdrift0)
    bgrad_np = np.asarray(geo.bgrad)
    _assert_close_1e6(bmag_np[1:], bmag_np[1:][::-1], "w7x:bmag_even")
    _assert_close_1e6(gds2_np[1:], gds2_np[1:][::-1], "w7x:gds2_even")
    _assert_close_1e6(gds21_np[1:], -gds21_np[1:][::-1], "w7x:gds21_odd")
    _assert_close_1e6(gb0_np[1:], -gb0_np[1:][::-1], "w7x:gbdrift0_odd")
    _assert_close_1e6(bgrad_np[1:], -bgrad_np[1:][::-1], "w7x:bgrad_odd")

    # Flux-surface denominator ratio invariant:
    # sum(vol_fac * grho) == grho_avg and vol_fac / flux_fac == grho_avg
    _assert_close_1e6(float(jnp.sum(geo.vol_fac * geo.grho)), geo.grho_avg, "w7x:grho_avg")
    _assert_close_1e6(geo.vol_fac / geo.flux_fac, np.full(grids.nz, geo.grho_avg), "w7x:fac_ratio")

    # Non-twisting flux-tube (NTFT) mode check
    cfg_ntft = dataclasses.replace(cfg_nc, non_twist=True)
    _, grids_nt, geo_nt = load_eik_nc(eik_nc_path, cfg_ntft)
    assert geo_nt.ftwist.shape == (grids_nt.nz,)
    assert geo_nt.m0.shape == (grids_nt.nz, grids_nt.naky)
    assert geo_nt.delta_kx.shape == (grids_nt.nz, grids_nt.naky)
    np.testing.assert_array_equal(np.asarray(geo_nt.m0[grids_nt.nz // 2, :]), 0)

    # PyTree + JIT compatibility check
    @jax.jit
    def _geo_probe(g: gx_jax.Geometry, gr: gx_jax.Grids) -> jax.Array:
        return jnp.sum(g.kperp2 * g.vol_fac[:, None, None]) + jnp.sum(gr.ky)

    val64 = _geo_probe(geo, grids)
    val32 = _geo_probe(geo.astype(jnp.float32), grids)
    _assert_close_fp32_1e6(float(val32), float(val64), "w7x:jit_pytree_cast")

    # Equal-arc enforcement check: perturbing gradpar[0] must raise ValueError
    bad_nc = tmp_path / "non_equal_arc.eik.nc"
    bad_nc.write_bytes(eik_nc_path.read_bytes())
    with netCDF4.Dataset(bad_nc, "a") as ds_bad:
        gp = np.asarray(ds_bad.variables["gradpar"][:], dtype=np.float64)
        gp[0] *= 1.05
        ds_bad.variables["gradpar"][:] = gp
    with pytest.raises(ValueError, match="equal-arc"):
        load_eik_nc(bad_nc, cfg_nc)


# ---------------------------------------------------------------------------
# 4. Laguerre Quadrature & Bessel J0 / J1 Round-Trip Verification
# ---------------------------------------------------------------------------


def test_laguerre_cpp_golden_tables() -> None:
    """Verify roots, to_grid, and to_spectral against test_laguerre_transform.cu (nl=9, nj=12)."""
    roots_cpp = np.array(
        [
            1.1572e-01,
            6.1176e-01,
            1.5126e00,
            2.8338e00,
            4.5992e00,
            6.8445e00,
            9.6213e00,
            1.3006e01,
            1.7117e01,
            2.2151e01,
            2.8488e01,
            3.7099e01,
        ],
        dtype=np.float64,
    )
    to_grid_row0_cpp = np.array(
        [
            1.00000000e00,
            -8.84277880e-01,
            7.75251567e-01,
            -6.72662795e-01,
            5.76260686e-01,
            -4.85801816e-01,
            4.01049733e-01,
            -3.21775049e-01,
            2.47755125e-01,
        ],
        dtype=np.float64,
    )
    to_grid_row11_cpp = np.array(
        [
            1.00000000e00,
            3.60991211e01,
            6.14974121e02,
            6.55597705e03,
            4.88710586e04,
            2.69401969e05,
            1.13113325e06,
            3.66327200e06,
            9.12964500e06,
        ],
        dtype=np.float64,
    )

    lag = init_laguerre(9, dtype=jnp.float64)
    assert lag.nl == 9
    assert lag.nj == 12

    # C++ roots_correct in test_laguerre_transform.cu is printed to 5 sig figs (1e-4 tol in C++)
    np.testing.assert_allclose(np.asarray(lag.roots), roots_cpp, rtol=1.0e-4)
    _assert_close_fp32_1e6(np.asarray(lag.to_grid[0, :]), to_grid_row0_cpp, "to_grid[0]")
    _assert_close_fp32_1e6(np.asarray(lag.to_grid[11, :]), to_grid_row11_cpp, "to_grid[11]")
    _assert_close_1e6(float(jnp.sum(lag.weights)), 1.0, "weights_sum")

    # Exact discrete orthogonality: to_spectral @ to_grid == I_9
    ident = lag.to_spectral @ lag.to_grid
    np.testing.assert_allclose(np.asarray(ident), np.eye(9), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("nl", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("complex_dtype", [jnp.complex64, jnp.complex128])
def test_laguerre_6d_round_trip(nl: int, complex_dtype: jnp.dtype) -> None:
    """Verify 6D G[nspecies, nhermite, nlaguerre, nz, ky, kx] spectral->grid->spectral < 1e-6."""
    real_dtype = jnp.float32 if complex_dtype == jnp.complex64 else jnp.float64
    lag = init_laguerre(nl, dtype=real_dtype)

    rng = np.random.default_rng(100 + nl)
    shape = (2, 4, nl, 12, 5, 7)
    g_np = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    g_in = jnp.asarray(g_np, dtype=complex_dtype)

    @jax.jit
    def _round_trip(g_arr: jax.Array, lt: gx_jax.LaguerreTransform) -> tuple[jax.Array, jax.Array]:
        g_mu = laguerre_to_grid(g_arr, lt, axis=2)
        g_back = laguerre_to_spectral(g_mu, lt, axis=2)
        return g_mu, g_back

    g_mu, g_out = _round_trip(g_in, lag)
    assert g_mu.shape == (2, 4, lag.nj, 12, 5, 7)
    assert g_out.shape == shape
    if complex_dtype == jnp.complex128:
        _assert_close_1e6(np.real(g_out), np.real(g_in), f"nl={nl},{complex_dtype}:real")
        _assert_close_1e6(np.imag(g_out), np.imag(g_in), f"nl={nl},{complex_dtype}:imag")
    else:
        _assert_close_fp32_1e6(np.real(g_out), np.real(g_in), f"nl={nl},{complex_dtype}:real")
        _assert_close_fp32_1e6(np.imag(g_out), np.imag(g_in), f"nl={nl},{complex_dtype}:imag")


def test_bessel_j0_j1_g0_g1_against_scipy() -> None:
    """Verify pure-JAX Bessel functions against SciPy across [0, 100] (all 3 asymptotic regimes)."""
    x_vals = np.concatenate(
        [
            np.array([0.0, 1e-12, 1e-8, 1e-4]),
            np.linspace(0.01, 4.0, 200),
            np.linspace(4.0, 25.0, 300),
            np.linspace(25.0, 100.0, 200),
        ]
    )
    x_jax64 = jnp.asarray(x_vals, dtype=jnp.float64)
    x_jax32 = jnp.asarray(x_vals, dtype=jnp.float32)
    x_vals32 = np.asarray(x_jax32, dtype=np.float64)

    j0_ref = sp_special.j0(x_vals)
    j1_ref = sp_special.j1(x_vals)
    j1_oa_ref = np.where(x_vals == 0.0, 0.5, j1_ref / np.where(x_vals == 0.0, 1.0, x_vals))
    g0_ref = sp_special.i0e(x_vals)
    g1_ref = sp_special.i1e(x_vals)

    # FP64 machine-precision check
    np.testing.assert_allclose(
        np.asarray(jax.jit(bessel_j0)(x_jax64)), j0_ref, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(bessel_j1)(x_jax64)), j1_ref, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(j1_over_alpha)(x_jax64)), j1_oa_ref, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(np.asarray(jax.jit(g0)(x_jax64)), g0_ref, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax.jit(g1)(x_jax64)), g1_ref, rtol=1e-13, atol=1e-14)

    # FP32 runtime check (< 1e-6 relative L2 & 1.25 FP32 ULP elementwise at exact float32 x)
    j0_ref32 = sp_special.j0(x_vals32)
    j1_ref32 = sp_special.j1(x_vals32)
    g0_ref32 = sp_special.i0e(x_vals32)
    g1_ref32 = sp_special.i1e(x_vals32)
    _assert_close_fp32_1e6(np.asarray(jax.jit(bessel_j0)(x_jax32)), j0_ref32, "j0_fp32")
    _assert_close_fp32_1e6(np.asarray(jax.jit(bessel_j1)(x_jax32)), j1_ref32, "j1_fp32")
    _assert_close_fp32_1e6(np.asarray(jax.jit(g0)(x_jax32)), g0_ref32, "g0_fp32")
    _assert_close_fp32_1e6(np.asarray(jax.jit(g1)(x_jax32)), g1_ref32, "g1_fp32")


def test_laguerre_bessel_j0_and_j1_round_trip() -> None:
    """Verify Gauss-Laguerre <-> Bessel J0 (phi) & J1 (bpar) gyroaveraging round-trips (< 1e-6)."""
    _, grids, geo = s_alpha_geometry(
        load_config(BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in")
    )
    nl = 16
    lag = init_laguerre(nl, dtype=jnp.float64)

    rng = np.random.default_rng(7)
    f_shape = grids.field_shape
    phi = jnp.asarray(
        rng.standard_normal(f_shape) + 1j * rng.standard_normal(f_shape), dtype=jnp.complex128
    )
    bpar = jnp.asarray(
        rng.standard_normal(f_shape) + 1j * rng.standard_normal(f_shape), dtype=jnp.complex128
    )
    phi = jnp.where(grids.active_mask[None, :, :], phi, 0.0)
    bpar = jnp.where(grids.active_mask[None, :, :], bpar, 0.0)

    ell = jnp.arange(nl, dtype=jnp.int32)[:, None, None, None]
    tz = 1.5

    # Part A: Strong FLR (rho2_s = 0.25 -> kperp2 * rho2_s up to ~6.4 at ballooning ends):
    # Gauss-Laguerre grid -> spectral matches analytical jflr / jflr_b to < 1e-6 everywhere,
    # and Maxwellian-weighted sqrt(w_j) spectral -> grid reconstruction matches in L2(e^-x) < 1e-6.
    rho2_strong = 0.25
    b_strong = geo.kperp2 * rho2_strong
    j0_grid_s = jax.jit(j0_to_grid)(phi, geo.kperp2, lag, rho2_strong, 1.0, grids.active_mask)
    j0_spec_s = laguerre_to_spectral(j0_grid_s, lag, axis=0)
    j0_ana_s = jflr(ell, b_strong[None, ...]) * phi[None, ...]
    _assert_close_1e6(np.real(j0_spec_s), np.real(j0_ana_s), "J0_strong_grid_to_spec_real")
    _assert_close_1e6(np.imag(j0_spec_s), np.imag(j0_ana_s), "J0_strong_grid_to_spec_imag")

    sqrt_w = jnp.sqrt(lag.weights)[:, None, None, None]
    j0_rec_s = laguerre_to_grid(j0_ana_s, lag, axis=0)
    _assert_close_fp32_1e6(
        np.real(sqrt_w * j0_rec_s), np.real(sqrt_w * j0_grid_s), "J0_strong_L2_weighted"
    )

    chi_bpar_s = jax.jit(j0_phi_and_bpar_to_grid)(
        jnp.zeros_like(phi), bpar, geo.kperp2, lag, rho2_strong, tz, 0.0, 1.0, grids.active_mask
    )
    j1_spec_s = laguerre_to_spectral(chi_bpar_s, lag, axis=0)
    j1_ana_s = tz * jflr_b(ell, b_strong[None, ...]) * bpar[None, ...]
    _assert_close_1e6(np.real(j1_spec_s), np.real(j1_ana_s), "J1_strong_grid_to_spec_real")
    _assert_close_1e6(np.imag(j1_spec_s), np.imag(j1_ana_s), "J1_strong_grid_to_spec_imag")

    # Part B: Pointwise unweighted grid <-> spectral round-trip across all nj=23 roots
    # (including vmax_J = 77.74) for resolved FLR b = kperp2 * rho2_s in [0, 0.26]:
    rho2_s = 0.01
    b_arg = geo.kperp2 * rho2_s

    j0_grid = jax.jit(j0_to_grid)(phi, geo.kperp2, lag, rho2_s, 1.0, grids.active_mask)
    j0_spec_num = laguerre_to_spectral(j0_grid, lag, axis=0)
    j0_spec_ana = jflr(ell, b_arg[None, ...]) * phi[None, ...]
    _assert_close_1e6(np.real(j0_spec_num), np.real(j0_spec_ana), "J0_grid_to_spec_real")
    _assert_close_1e6(np.imag(j0_spec_num), np.imag(j0_spec_ana), "J0_grid_to_spec_imag")

    j0_grid_reconstructed = laguerre_to_grid(j0_spec_ana, lag, axis=0)
    _assert_close_1e6(np.real(j0_grid_reconstructed), np.real(j0_grid), "J0_spec_to_grid_real")
    _assert_close_1e6(np.imag(j0_grid_reconstructed), np.imag(j0_grid), "J0_spec_to_grid_imag")

    # 2. J_1 B_par gyroaverage: 2*muB*(J_1(alpha)/alpha)*bpar <-> jflr_b(l, b) * bpar
    chi_bpar_grid = jax.jit(j0_phi_and_bpar_to_grid)(
        jnp.zeros_like(phi),
        bpar,
        geo.kperp2,
        lag,
        rho2_s,
        tz,
        0.0,
        1.0,
        grids.active_mask,
    )
    j1_spec_num = laguerre_to_spectral(chi_bpar_grid, lag, axis=0)
    j1_spec_ana = tz * jflr_b(ell, b_arg[None, ...]) * bpar[None, ...]
    _assert_close_1e6(np.real(j1_spec_num), np.real(j1_spec_ana), "J1_grid_to_spec_real")
    _assert_close_1e6(np.imag(j1_spec_num), np.imag(j1_spec_ana), "J1_grid_to_spec_imag")

    chi_bpar_reconstructed = laguerre_to_grid(j1_spec_ana, lag, axis=0)
    _assert_close_1e6(
        np.real(chi_bpar_reconstructed), np.real(chi_bpar_grid), "J1_spec_to_grid_real"
    )
    _assert_close_1e6(
        np.imag(chi_bpar_reconstructed), np.imag(chi_bpar_grid), "J1_spec_to_grid_imag"
    )

    # 3. Gyroaveraging Parseval identities & Jfac relation:
    #    sum_{l=0}^infty J_l(b)^2 == g0(b)
    #    sum_{l=0}^infty J_l(b) * JflrB(l, b) == g0(b) - g1(b)
    b_1d = jnp.linspace(0.0, 2.5, 64, dtype=jnp.float64)
    ell_1d = jnp.arange(24, dtype=jnp.int32)[:, None]
    jl_mat = jflr(ell_1d, b_1d[None, :])
    jlb_mat = jflr_b(ell_1d, b_1d[None, :])
    _assert_close_1e6(jnp.sum(jl_mat**2, axis=0), g0(b_1d), "Parseval_J0_g0")
    _assert_close_1e6(
        jnp.sum(jl_mat * jlb_mat, axis=0), g0(b_1d) - g1(b_1d), "Parseval_J1_g0_minus_g1"
    )
    _assert_close_1e6(
        jfac(ell_1d, b_1d[None, :]),
        1.5 * jl_mat + jflr_a(ell_1d, b_1d[None, :]),
        "jfac_definition",
    )
