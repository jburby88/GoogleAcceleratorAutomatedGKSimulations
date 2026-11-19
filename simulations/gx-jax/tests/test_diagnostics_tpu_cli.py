"""NetCDF4 Diagnostics, TPU `shard_map`, CLI & Packaging."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

from gx_jax import (
    DiagnosticsWriter,
    GXConfig,
    SpeciesConfig,
    cli_main,
    compute_diagnostics_step,
    compute_heat_flux_spectra,
    compute_moments_diagnostic_arrays,
    compute_particle_flux_spectra,
    compute_turbulent_heating_spectra,
    compute_wg_spectra,
    compute_wphi_spectra,
    create_device_mesh,
    extract_kr2_diagnostics,
    gather_g_state,
    gyrokinetic_rhs,
    init_diagnostics_state,
    init_field_solver,
    init_geometry,
    init_gyrokinetic_operator,
    init_linear_operator,
    init_moments,
    linear_rhs,
    load_config,
    load_restart_nc,
    nonlinear_rhs,
    reorder_kx_from_out,
    reorder_kx_to_out,
    reorder_kz_to_out,
    run_cli_simulation,
    save_restart_nc,
    shard_g_state,
    sharded_compute_diagnostics,
    sharded_gyrokinetic_rhs,
    sharded_linear_rhs,
    sharded_nonlinear_rhs,
    sharded_solve_fields,
    solve_fields,
    step_gyrokinetic,
    step_sharded_gyrokinetic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
GOLDEN_DIR = REPO_ROOT / "golden_fixtures"


# =============================================================================
# 1. Coordinate Reordering (`kx_out`, `kz_out`), Small-nx Edge Case & Fixtures
# =============================================================================


def test_kx_and_kz_output_reordering_roundtrip(tmp_path: Path) -> None:
    """Verify monotonic kx_out / kz_out shifts, exact inverse, and nx in (2, 3) writer safety."""
    in_path = BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    cfg, grids, _geo = init_geometry(load_config(in_path))

    kx_out = np.asarray(reorder_kx_to_out(grids.kx, axis=0))
    assert np.all(np.diff(kx_out) > 0.0), "kx_out must be strictly increasing"
    np.testing.assert_allclose(kx_out, np.asarray(grids.kx_out), rtol=0.0, atol=1e-14)

    kx_back = np.asarray(reorder_kx_from_out(jnp.asarray(kx_out), axis=0))
    np.testing.assert_allclose(kx_back, np.asarray(grids.kx), rtol=0.0, atol=1e-14)

    kz_out = np.asarray(reorder_kz_to_out(grids.kz, axis=0))
    assert np.all(np.diff(kz_out) > 0.0), "kz_out must be strictly increasing"
    np.testing.assert_allclose(kz_out, np.asarray(grids.kz_out), rtol=0.0, atol=1e-14)

    # Edge case (#9 item 5): nx in (2, 3) has grids.nakx == 1 while grids.kx_out has length nx.
    for nx_small in (2, 3):
        cfg_s, grids_s, geo_s = init_geometry(replace(cfg, nx=nx_small, ny=4, nz=8))
        assert grids_s.nakx == 1
        out_small = tmp_path / f"nx_{nx_small}.out.nc"
        with DiagnosticsWriter(out_small, cfg_s, grids_s, geo_s):
            pass
        with nc.Dataset(out_small, "r") as ds:
            assert ds.groups["Grids"].variables["kx"].shape == (1,)


def test_golden_kh01a_restart_exact_spectra_parity() -> None:
    """Compare JAX spectral diagnostics on kh01a_restart.nc against C++ kh01a.nc[t=-1]."""
    in_path = GOLDEN_DIR / "unit_tests" / "inputs" / "kh01a.in"
    restart_path = GOLDEN_DIR / "unit_tests" / "inputs" / "kh01a_restart.nc"
    golden_path = GOLDEN_DIR / "unit_tests" / "inputs" / "kh01a.nc"

    cfg, grids, geo = init_geometry(load_config(in_path))
    solver = init_field_solver(cfg, grids, geo)

    g_state, t_loaded = load_restart_nc(restart_path, grids, scale=1.0)
    fields = solve_fields(g_state, solver)
    wg = compute_wg_spectra(g_state, grids, geo, solver)
    wphi = compute_wphi_spectra(fields.phi, grids, geo, solver)

    with nc.Dataset(golden_path, "r") as ds:
        sp = ds.groups["Spectra"].variables
        cpp_wg_st = np.asarray(sp["Wst"][-1])
        cpp_wphi_st = np.asarray(sp["Pst"][-1])
        cpp_wzst = np.asarray(sp["Wzst"][-1])
        cpp_pzst = np.asarray(sp["Pzst"][-1])
        cpp_wlst = np.asarray(sp["Wlst"][-1])
        cpp_wmst = np.asarray(sp["Wmst"][-1])
        cpp_wkxkyst = np.asarray(sp["Wkxkyst"][-1])
        cpp_pkxkyst = np.asarray(sp["Pkxkyst"][-1])

    np.testing.assert_allclose(np.asarray(wg.st), cpp_wg_st, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wphi.st), cpp_wphi_st, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wg.zst), cpp_wzst, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wphi.zst), cpp_pzst, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wg.lst), cpp_wlst, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wg.mst), cpp_wmst, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wg.kxkyst), cpp_wkxkyst, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(wphi.kxkyst), cpp_pkxkyst, rtol=5e-5, atol=1e-7)

    diag_state = init_diagnostics_state(g_state, fields, time=t_loaded, has_old=False)
    snap, _ = compute_diagnostics_step(
        g_state, fields, diag_state, grids, geo, solver, time=t_loaded, dt=cfg.dt, beta=cfg.beta
    )
    assert np.isfinite(float(snap.wg.total))
    moms = compute_moments_diagnostic_arrays(g_state, fields, geo, solver)
    assert moms["Density"].shape == (cfg.nspecies, grids.naky, grids.nakx, grids.nz, 2)


def test_golden_restart_test02_roundtrip_without_legacy_namelist(tmp_path: Path) -> None:
    """Verify legacy restart_test02.out.nc load & save round-trip without parsing Fortran .in."""
    golden_restart = (
        GOLDEN_DIR / "unit_tests" / "inputs" / "restart_tests" / "restart_test02.out.nc"
    )
    with nc.Dataset(golden_restart, "r") as ds:
        # Legacy C++ restart layout: G has dims (Ns, Nm, Nl, Nz, Nx, Nyc, ri)
        ns, nm, nl, nz, nx_full, nyc, _ri = ds.variables["G"].shape
        ny_full = 2 * (nyc - 1)

    cfg_meta = GXConfig(
        nx=int(nx_full),
        ny=int(ny_full),
        nz=int(nz),
        ntheta_in=int(nz),
        nhermite=int(nm),
        nlaguerre=int(nl),
        nspecies=int(ns),
        x0=10.0,
        y0=10.0,
        geo_option="slab",
        igeo=0,
        slab=True,
        species=(
            SpeciesConfig.from_raw(z=1.0, mass=1.0, dens=1.0, temp=1.0, tprim=0.0, fprim=0.0),
        ),
    )
    _cfg_r, grids_r, _geo_r = init_geometry(cfg_meta)
    g_loaded, t_loaded = load_restart_nc(golden_restart, grids_r, scale=1.0)
    assert g_loaded.shape == (ns, nm, nl, nz, grids_r.naky, grids_r.nakx)
    assert np.all(np.isfinite(np.asarray(g_loaded)))

    rt_file = tmp_path / "restart_test02_rt.nc"
    save_restart_nc(rt_file, g_loaded, t_loaded, grids_r)
    g_rt, t_rt = load_restart_nc(rt_file, grids_r, scale=1.0)
    assert np.isclose(t_rt, t_loaded)
    np.testing.assert_allclose(np.asarray(g_rt), np.asarray(g_loaded), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "rel_nc",
    [
        "benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons_correct.out.nc",
        "benchmarks/linear/ITG_cyclone/itg_miller_adiabatic_electrons_correct.out.nc",
        "benchmarks/linear/ITG_cyclone/itg_miller_kinetic_electrons_correct.out.nc",
        "benchmarks/linear/KAW/kaw_betahat10.0_kp0.01_correct.out.nc",
        "benchmarks/linear/KBM/kbm_miller_correct.out.nc",
    ],
)
def test_golden_linear_benchmark_out_nc_files_compatible_with_kr2_extractor(rel_nc: str) -> None:
    """Verify extract_kr2_diagnostics parses all 5 C++ golden linear .out.nc benchmarks."""
    nc_path = GOLDEN_DIR / rel_nc
    report = extract_kr2_diagnostics(nc_path)
    assert report["converged"] is True
    assert report["failure_type"] is None
    assert report["has_nan_or_inf"] is False
    assert report["kx_monotonic"] is True
    assert report["kz_monotonic"] is True
    assert report["n_time"] > 0
    assert report["sim_time_reached"] > 0.0
    assert "scalar_metrics" in report and "fluxes" in report


def test_grho_avg_flux_division_invariant_and_em_components() -> None:
    """Verify grho_avg division, 1-species pflux=0, and +Tperp sign."""
    in_path = BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"
    cfg, grids, geo = init_geometry(
        replace(
            load_config(in_path),
            nx=6,
            ny=6,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=4,
            nlaguerre=2,
        )
    )
    solver = init_field_solver(cfg, grids, geo)
    assert not np.isclose(float(geo.grho_avg), 1.0), "Miller geometry must have grho_avg != 1"

    g0 = init_moments(cfg, grids, geo)
    f0 = solve_fields(g0, solver)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo)
    step1 = step_gyrokinetic(g0, f0, gk_op, dt=1.0e-4, scheme="rk3")

    # Scale grho -> 2.5 * grho (so grho_avg scales by 2.5 and flux_fac scales by 1/2.5)
    scale = 2.5
    geo_scaled = replace(
        geo,
        grho=geo.grho * scale,
        grho_avg=float(geo.grho_avg) * scale,
        flux_fac=geo.flux_fac / scale,
    )
    q_base = compute_heat_flux_spectra(step1.G, step1.fields, grids, geo, solver)
    q_scaled = compute_heat_flux_spectra(step1.G, step1.fields, grids, geo_scaled, solver)
    p_base = compute_particle_flux_spectra(step1.G, step1.fields, grids, geo, solver)
    p_scaled = compute_particle_flux_spectra(step1.G, step1.fields, grids, geo_scaled, solver)
    th_base = compute_turbulent_heating_spectra(
        step1.G, step1.fields, g0, f0, 1.0e-3, grids, geo, solver
    )
    th_scaled = compute_turbulent_heating_spectra(
        step1.G, step1.fields, g0, f0, 1.0e-3, grids, geo_scaled, solver
    )

    np.testing.assert_allclose(
        np.asarray(q_scaled.total.st), np.asarray(q_base.total.st) / scale, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(p_scaled.total.st), np.asarray(p_base.total.st) / scale, rtol=1e-6
    )
    # Collisional TurbulentHeating uses geo.vol_fac (diagnostic_classes.cu:L522-524), NOT flux_fac
    np.testing.assert_allclose(
        np.asarray(th_scaled.total.st), np.asarray(th_base.total.st), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(q_base.total.st),
        np.asarray(q_base.es.st + q_base.apar.st + q_base.bpar.st),
        rtol=1e-12,
        atol=1e-15,
    )

    # Tperp sign (#9 item 1): +G_{m=0, l=1}
    moms = compute_moments_diagnostic_arrays(step1.G, step1.fields, geo, solver)
    g_m0_l1_kx0 = np.asarray(step1.G[:, 0, 1, :, :, :])
    assert np.isclose(
        np.sum(np.abs(moms["Tperp"][..., 0] + 1j * moms["Tperp"][..., 1])),
        np.sum(np.abs(g_m0_l1_kx0)),
    )
    # Check real part sign at a non-zero entry matches +g_m0_l1 (not -g_m0_l1)
    from gx_jax import reorder_complex_moment_to_nc

    np.testing.assert_allclose(
        np.asarray(moms["Tperp"]),
        np.asarray(reorder_complex_moment_to_nc(step1.G[:, 0, 1])),
        rtol=0.0,
        atol=1e-15,
    )

    # Single-species ParticleFlux zeroed by default (#9 item 2)
    itg_in = BENCH_DIR / "linear" / "ITG_cyclone" / "itg_salpha_adiabatic_electrons.in"
    itg_cfg, itg_grids, itg_geo = init_geometry(load_config(itg_in))
    itg_solver = init_field_solver(itg_cfg, itg_grids, itg_geo)
    itg_g = init_moments(itg_cfg, itg_grids, itg_geo)
    itg_f = solve_fields(itg_g, itg_solver)
    p_1sp = compute_particle_flux_spectra(itg_g, itg_f, itg_grids, itg_geo, itg_solver)
    assert np.all(np.asarray(p_1sp.total.st) == 0.0)


def test_diagnostics_writer_netcdf_schema_and_restart_roundtrip(tmp_path: Path) -> None:
    """Verify DiagnosticsWriter .out.nc group/variable hierarchy and .restart.nc round-trip."""
    in_path = BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"
    cfg = replace(
        load_config(in_path),
        nx=6,
        ny=6,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        write_omega=True,
        write_free_energy=True,
        write_fluxes=True,
        write_fields=True,
        write_moms=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    solver = init_field_solver(cfg, grids, geo)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo)

    g0 = init_moments(cfg, grids, geo)
    f0 = solve_fields(g0, solver)
    dt = 1.0e-4
    step1 = step_gyrokinetic(g0, f0, gk_op, dt=dt, scheme="rk3")

    out_nc = tmp_path / "kbm_test.out.nc"
    with DiagnosticsWriter(out_nc, cfg, grids, geo) as writer:
        dstate = init_diagnostics_state(g0, f0, time=0.0, has_old=False)
        _, dstate = writer.write_step(g0, f0, dstate, solver, time=0.0, dt=dt)
        _, dstate = writer.write_step(step1.G, step1.fields, dstate, solver, time=dt, dt=dt)

    with nc.Dataset(out_nc, "r") as ds:
        assert set(ds.groups.keys()) == {"Grids", "Geometry", "Inputs", "Diagnostics"}
        ggrp = ds.groups["Grids"]
        np.testing.assert_allclose(ggrp.variables["time"][:], [0.0, dt])
        assert ggrp.variables["ky"].shape == (grids.naky,)
        assert ggrp.variables["kx"].shape == (grids.nakx,)
        assert ggrp.variables["kz"].shape == (grids.nz,)
        assert ggrp.variables["theta"].shape == (grids.nz,)
        assert np.all(np.diff(ggrp.variables["kx"][:]) > 0.0)
        assert np.all(np.diff(ggrp.variables["kz"][:]) > 0.0)

        geogrp = ds.groups["Geometry"]
        for req_var in (
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
            "grho_avg",
        ):
            assert req_var in geogrp.variables

        dgrp = ds.groups["Diagnostics"]
        assert dgrp.variables["omega_kxkyt"].shape == (2, grids.naky, grids.nakx, 2)
        assert dgrp.variables["Phi2_kxkyt"].shape == (2, grids.naky, grids.nakx)
        assert dgrp.variables["Apar2_kxkyt"].shape == (2, grids.naky, grids.nakx)
        assert dgrp.variables["Bpar2_kxkyt"].shape == (2, grids.naky, grids.nakx)
        assert dgrp.variables["Wg_lmst"].shape == (2, cfg.nspecies, cfg.nhermite, cfg.nlaguerre)
        assert dgrp.variables["Wphi_kxkyt"].shape == (2, grids.naky, grids.nakx)
        assert dgrp.variables["Wapar_kxkyt"].shape == (2, grids.naky, grids.nakx)
        assert dgrp.variables["Wbpar_kxkyt"].shape == (2, grids.naky, grids.nakx)
        assert dgrp.variables["HeatFlux_st"].shape == (2, cfg.nspecies)
        assert dgrp.variables["ParticleFlux_st"].shape == (2, cfg.nspecies)
        assert dgrp.variables["TurbulentHeating_st"].shape == (2, cfg.nspecies)
        assert dgrp.variables["Phi"].shape == (2, grids.naky, grids.nakx, grids.nz, 2)
        assert dgrp.variables["Density"].shape == (
            2,
            cfg.nspecies,
            grids.naky,
            grids.nakx,
            grids.nz,
            2,
        )

    kr2 = extract_kr2_diagnostics(out_nc)
    assert kr2["n_time"] == 2
    assert kr2["converged"] is True
    assert np.isclose(kr2["grho_avg"], float(geo.grho_avg))

    restart_file = tmp_path / "kbm_test.restart.nc"
    save_restart_nc(restart_file, step1.G, dt, grids)
    g1_loaded, t_loaded = load_restart_nc(restart_file, grids, scale=1.0)
    assert np.isclose(t_loaded, dt)
    np.testing.assert_allclose(np.asarray(g1_loaded), np.asarray(step1.G), rtol=1e-13, atol=1e-15)


# =============================================================================
# 2. TPU `shard_map` Single-Device & 4-Device Subprocess Parity
# =============================================================================


def test_single_device_shard_map_matches_unsharded() -> None:
    """Verify (1, 1) mesh shard_map operators match single-device operators in-process."""
    jax.config.update("jax_enable_x64", True)
    in_path = BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"
    cfg = replace(
        load_config(in_path),
        nx=6,
        ny=6,
        nz=16,
        ntheta_in=16,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        nonlinear_mode=True,
    )
    cfg, grids, geo = init_geometry(cfg)
    lin_op = init_linear_operator(cfg, grids, geo)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo, linear_op=lin_op)
    solver = lin_op.solver

    g0 = init_moments(cfg, grids, geo, dtype=jnp.complex128)
    mesh = create_device_mesh(mesh_shape=(1, 1), nspecies=cfg.nspecies, nhermite=cfg.nhermite)
    g_sh = shard_g_state(g0, mesh)

    f_ref = solve_fields(g0, solver)
    f_sh = sharded_solve_fields(g_sh, solver, mesh)
    np.testing.assert_allclose(np.asarray(f_sh.phi), np.asarray(f_ref.phi), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(f_sh.apar), np.asarray(f_ref.apar), rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(f_sh.bpar), np.asarray(f_ref.bpar), rtol=1e-12, atol=1e-13
    )

    dt = 1.0e-4
    rhs_lin_ref = linear_rhs(g0, f_ref, lin_op, dt)
    rhs_lin_sh = sharded_linear_rhs(g_sh, f_sh, lin_op, dt, mesh)
    np.testing.assert_allclose(
        np.asarray(gather_g_state(rhs_lin_sh, mesh)),
        np.asarray(rhs_lin_ref),
        rtol=1e-12,
        atol=1e-13,
    )

    assert gk_op.nonlinear_op is not None
    rhs_nl_ref, _ = nonlinear_rhs(g0, f_ref, gk_op.nonlinear_op)
    rhs_nl_sh, _ = sharded_nonlinear_rhs(g_sh, f_sh, gk_op.nonlinear_op, mesh)
    np.testing.assert_allclose(
        np.asarray(gather_g_state(rhs_nl_sh, mesh)),
        np.asarray(rhs_nl_ref),
        rtol=1e-12,
        atol=1e-13,
    )

    rhs_tot_ref, _ = gyrokinetic_rhs(g0, f_ref, gk_op, dt)
    rhs_tot_sh, _ = sharded_gyrokinetic_rhs(g_sh, f_sh, gk_op, dt, mesh)
    np.testing.assert_allclose(
        np.asarray(gather_g_state(rhs_tot_sh, mesh)),
        np.asarray(rhs_tot_ref),
        rtol=1e-12,
        atol=1e-13,
    )

    step_ref = jax.jit(lambda g_a, f_a: step_gyrokinetic(g_a, f_a, gk_op, dt=dt, scheme="rk3"))(
        g0, f_ref
    )
    step_sh = jax.jit(
        lambda g_a, f_a: step_sharded_gyrokinetic(g_a, f_a, gk_op, mesh, dt=dt, scheme="rk3")
    )(g_sh, f_sh)
    np.testing.assert_allclose(
        np.asarray(gather_g_state(step_sh.G, mesh)),
        np.asarray(step_ref.G),
        rtol=1e-12,
        atol=1e-13,
    )

    d_ref0 = init_diagnostics_state(g0, f_ref, time=0.0, has_old=True)
    d_sh0 = init_diagnostics_state(g_sh, f_sh, time=0.0, has_old=True)
    snap_ref, _ = compute_diagnostics_step(
        step_ref.G, step_ref.fields, d_ref0, grids, geo, solver, time=dt, dt=dt, beta=cfg.beta
    )
    snap_sh, _ = sharded_compute_diagnostics(
        step_sh.G, step_sh.fields, d_sh0, grids, geo, solver, mesh, time=dt, dt=dt, beta=cfg.beta
    )
    np.testing.assert_allclose(np.asarray(snap_sh.wg.st), np.asarray(snap_ref.wg.st), rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(snap_sh.heat_flux.total.st), np.asarray(snap_ref.heat_flux.total.st), rtol=1e-13
    )


def test_multidevice_4cpu_shard_map_parity_subprocess() -> None:
    """Verify (1, 4), (2, 2), and (2, 1) multi-device shard_map parity in a 4-CPU subprocess."""
    script = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gx_jax import (
    create_device_mesh,
    gather_g_state,
    init_geometry,
    init_gyrokinetic_operator,
    init_moments,
    load_config,
    shard_g_state,
    sharded_solve_fields,
    solve_fields,
    step_gyrokinetic,
    step_sharded_gyrokinetic,
)

assert jax.device_count() == 4
repo = Path.cwd()
kbm_in = repo / "benchmarks" / "linear" / "KBM" / "kbm_miller.in"
cfg = replace(
    load_config(kbm_in),
    nx=6,
    ny=6,
    nz=16,
    ntheta_in=16,
    nperiod=1, zp=1,
    nhermite=4,
    nlaguerre=2,
    nonlinear_mode=True,
    closure_model="beer4+2",
    collisions=True,
    hypercollisions_const=True,
    hyper=True,
    exb_shear=True,
    g_exb=0.1,
    forcing_init=True,
    forcing_amp=0.05,
)
cfg, grids, geo = init_geometry(cfg)
gk_op = init_gyrokinetic_operator(cfg, grids, geo)
solver = gk_op.linear_op.solver
g0 = init_moments(cfg, grids, geo, dtype=jnp.complex128)
f0 = solve_fields(g0, solver)
key = jax.random.PRNGKey(42)
ref_rk4 = jax.jit(
    lambda g_a, f_a, k_a: step_gyrokinetic(g_a, f_a, gk_op, dt=1.0e-4, scheme="rk4", key=k_a)
)(g0, f0, key)

for mesh_shape in [(1, 4), (2, 2), (2, 1)]:
    mesh = create_device_mesh(
        mesh_shape=mesh_shape, nspecies=cfg.nspecies, nhermite=cfg.nhermite
    )
    g_sh = shard_g_state(g0, mesh)
    f_sh = sharded_solve_fields(g_sh, solver, mesh)
    sh_rk4 = jax.jit(
        lambda g_a, f_a, k_a, m=mesh: step_sharded_gyrokinetic(
            g_a, f_a, gk_op, m, dt=1.0e-4, scheme="rk4", key=k_a
        )
    )(g_sh, f_sh, key)
    np.testing.assert_allclose(
        np.asarray(gather_g_state(sh_rk4.G, mesh)),
        np.asarray(ref_rk4.G),
        rtol=1e-12,
        atol=1e-13,
    )

print("4-DEVICE SHARD_MAP PARITY PASSED")
"""
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pp}" if existing_pp else str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "4-DEVICE SHARD_MAP PARITY PASSED" in proc.stdout


# =============================================================================
# 3. CLI (`gx_jax/cli.py`), `pyproject.toml`, and `Dockerfile.gx-jax-tpu`
# =============================================================================


def test_cli_end_to_end_linear_nonlinear_restart_and_benchmark(tmp_path: Path) -> None:
    """Verify gx-jax CLI execution for linear, nonlinear, restart, metrics-json, and benchmark."""
    in_src = tmp_path / "compact_cyclone.in"
    in_src.write_text(
        """[Dimensions]
ntheta = 8
nperiod = 1
ny = 6
nx = 6
nhermite = 4
nlaguerre = 2
nspecies = 1

[Domain]
y0 = 10.0
boundary = "linked"

[Physics]
nonlinear_mode = false

[Time]
dt = 0.05
nstep = 2
scheme = "rk3"
fixed_dt = true

[Initialization]
init_field = "density"
init_amp = 1.0e-5
ikpar_init = 0

[Geometry]
geo_option = "s-alpha"
eps = 0.167
Rmaj = 2.77778
qinp = 1.4
shat = 0.8

[species]
z = [1.0]
mass = [1.0]
dens = [1.0]
temp = [1.0]
tprim = [2.49]
fprim = [0.8]
vnewk = [0.0]
type = ["ion"]

[Boltzmann]
add_Boltzmann_species = true
Boltzmann_type = "electrons"
tau_fac = 1.0

[Diagnostics]
nwrite = 1
fluxes = true
free_energy = true
omega = true
""",
        encoding="utf-8",
    )
    out_nc = tmp_path / "cyclone_cli.out.nc"
    restart_nc = tmp_path / "cyclone_cli.restart.nc"
    metrics_json = tmp_path / "cyclone_metrics.json"

    dt_cli = 0.05
    rc = cli_main(
        [
            str(in_src),
            "-o",
            str(out_nc),
            "--dt",
            str(dt_cli),
            "--nstep",
            "2",
            "--nwrite",
            "1",
            "--scheme",
            "rk3",
            "--mesh",
            "1x1",
            "--save-restart",
            str(restart_nc),
            "--metrics-json",
            str(metrics_json),
            "--benchmark",
            "--benchmark-steps",
            "2",
        ]
    )
    assert rc == 0
    assert out_nc.exists()
    assert restart_nc.exists()
    assert metrics_json.exists()

    payload = json.loads(metrics_json.read_text())
    assert payload["nstep"] == 2
    assert payload["n_time"] == 2
    assert payload["converged"] is True
    assert "benchmark" in payload
    assert payload["benchmark"]["steps_per_sec"] > 0.0

    # Continuation from restart_nc in nonlinear mode
    out_nc2 = tmp_path / "cyclone_cont.out.nc"
    res2 = run_cli_simulation(
        in_src,
        output_path=out_nc2,
        dt=dt_cli,
        nstep=2,
        nwrite=1,
        scheme="sspx3",
        force_nonlinear=True,
        restart_file=restart_nc,
    )
    assert out_nc2.exists()
    assert np.isclose(res2["t_final"], 4.0 * dt_cli)


def test_pyproject_and_dockerfile_packaging() -> None:
    """Verify pyproject.toml [project.scripts] and Dockerfile.gx-jax-tpu specification."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    assert data["project"]["scripts"]["gx-jax"] == "gx_jax.cli:main"
    assert data["project"]["scripts"]["gx"] == "gx_jax.cli:main"
    assert data["tool"]["ruff"]["line-length"] == 100
    assert data["tool"]["ruff"]["lint"].get("ignore", []) == []

    dockerfile_path = REPO_ROOT.parent / "gcp" / "Dockerfile.gx-jax-tpu"
    assert dockerfile_path.exists()
    df_text = dockerfile_path.read_text()
    assert "JAX_ENABLE_X64=1" in df_text
    assert "JAX_DEFAULT_MATMUL_PRECISION=highest" in df_text
    assert "HDF5_USE_FILE_LOCKING=FALSE" in df_text
    assert "THIRD_PARTY_NOTICES.md" in df_text
    assert "jax[tpu]" in df_text
    assert 'ENTRYPOINT ["gcp-gx-babysitter"]' in df_text
