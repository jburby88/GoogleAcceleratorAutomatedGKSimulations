"""Batched `jax.vmap` KR2 Ensemble Campaign Driver."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

from gx_jax.cli import run_cli_simulation
from gx_jax.config import BoltzmannOption, GXConfig, SpeciesConfig, load_config
from gx_jax.diagnostics_nc import init_diagnostics_state
from gx_jax.ensemble import (
    BatchedEnsembleOperators,
    EnsembleMemberSpec,
    advance_ensemble_chunk,
    batched_compute_diagnostics_step,
    batched_gyrokinetic_rhs,
    batched_solve_fields,
    batched_step_integrator_state,
    build_ensemble_arg_parser,
    create_ensemble_mesh,
    ensemble_main,
    gather_ensemble_tree,
    init_batched_ensemble,
    init_batched_integrator_state,
    main,
    make_ensemble_chunk_stepper,
    run_ensemble,
    shard_ensemble_tree,
    validate_ensemble_compatibility,
)
from gx_jax.fields import init_moments, load_restart_nc
from gx_jax.geometry_nc import build_vmec_eik_nc, init_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent
W7X_WOUT = REPO_ROOT / "benchmarks" / "linear" / "ITG_w7x" / "wout_w7x.nc"


def _reject_non_strict_json(token: str) -> Any:
    raise AssertionError(f"Non-RFC-8259 JSON token encountered: {token!r}")


@pytest.fixture(scope="module")
def w7x_eik_nc_8(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a reusable 8-theta W7-X stellarator `.eik.nc` geometry file once per module."""
    out_dir = tmp_path_factory.mktemp("w7x_geo")
    eik_path = out_dir / "w7x_nz8.eik.nc"
    cfg_w7x = GXConfig(
        run_name="w7x",
        nx=8,
        ny=6,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        nspecies=2,
        y0=10.0,
        geo_option="vmec",
        vmec_file=str(W7X_WOUT),
        torflux=0.64,
        npol=1.0,
        alpha=0.0,
    )
    build_vmec_eik_nc(cfg_w7x, W7X_WOUT, eik_path)
    return eik_path


def _write_heterogeneous_b4_decks(
    deck_dir: Path,
    w7x_eik_path: Path,
    *,
    nonlinear: bool = True,
    scheme: str = "rk3",
    fixed_dt: bool = False,
    t_max_per_member: tuple[float, float, float, float] = (0.04, 0.04, 0.04, 0.04),
    nstep_per_member: tuple[int, int, int, int] = (5, 5, 5, 5),
    dt_per_member: tuple[float, float, float, float] = (1.5e-4, 1.5e-4, 1.5e-4, 1.5e-4),
    cfl_per_member: tuple[float, float, float, float] = (0.85, 0.75, 0.80, 0.70),
    shat_tokamak: tuple[float, float, float] = (0.55, 0.70, 0.65),
    nwrite: int = 2,
    write_fields: bool = True,
    write_moms: bool = True,
) -> list[Path]:
    """Write 4 heterogeneous 2-species decks: ITG + TEM + EM KBM + W7-X stellarator.

    Uses `nx=8, ny=6, ntheta=8, nhermite=4, nlaguerre=2, nspecies=2`:
    - Default `shat_tokamak=(0.55, 0.70, 0.65)` gives `jtwist=(3, 4, 4)` (`n_links=(1, 2)`)
      while Deck 3 (`W7-X`, `shat ~ -0.105`) has `jtwist = -1` (`n_links=(1, 5)`), exercising
      union-bucket `LinkedParallelGrad` chain unification `(1, 2, 5)` across heterogeneous
      magnetic shear! Passing `shat_tokamak=(0.85, 1.25, 0.85)` also exercises mixed
      `ParallelGrad` + `LinkedParallelGrad` promotion.
    - Decks 0 & 1 are electrostatic (`beta = 0.0`), while Deck 2 (KBM, `beta = 0.018`) and
      Deck 3 (W7-X, `beta = 0.006`) test mixed ES + EM (`apar` + `bpar`) solving!
    - Varies `tprim`, `fprim`, `dens`, `temp`, `vnewk` (`nu`), `beta`, `qinp`, `shat`,
      `dt`, `t_max`, `cfl`, `nstep`, and 1D `Geometry` (`s-alpha`, `miller`, and `W7-X`).
    """
    deck_dir.mkdir(parents=True, exist_ok=True)
    nl_str = str(nonlinear).lower()
    fdt_str = str(fixed_dt).lower()
    wf_str = str(write_fields).lower()
    wm_str = str(write_moms).lower()

    common_dims = f"""
[Dimensions]
 ntheta = 8
 nperiod = 1
 ny = 6
 nx = 8
 nhermite = 4
 nlaguerre = 2
 nspecies = 2

[Domain]
 y0 = 10.0
 boundary = "linked"

[Initialization]
 init_field = "density"
 init_amp = 1.0e-5
 ikpar_init = 0
 random_seed = 22

[Boltzmann]
 add_Boltzmann_species = false

[Dissipation]
 closure_model = "none"
 hypercollisions = true
 nu_hyper_m = 0.1
 p_hyper_m = 4
 hyper = true
 D_hyper = 0.05

[Diagnostics]
 nwrite = {nwrite}
 omega = true
 free_energy = true
 fluxes = true
 fields = {wf_str}
 moments = {wm_str}
"""

    # 0. Collisionless electrostatic ITG (s-alpha geometry, beta = 0)
    itg_path = deck_dir / "itg_es.in"
    itg_path.write_text(
        f"""{common_dims}
[Physics]
 beta = 0.0
 nonlinear_mode = {nl_str}
 fphi = 1.0
 fapar = 0.0
 fbpar = 0.0

[Time]
 dt = {dt_per_member[0]}
 dt_max = 0.01
 nstep = {nstep_per_member[0]}
 t_max = {t_max_per_member[0]}
 cfl = {cfl_per_member[0]}
 fixed_dt = {fdt_str}
 scheme = "{scheme}"

[Geometry]
 geo_option = "s-alpha"
 eps = 0.18
 Rmaj = 2.77778
 qinp = 1.4
 shat = {shat_tokamak[0]}

[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.02 ]
 dens = [ 1.0, 1.0 ]
 temp = [ 1.0, 1.0 ]
 tprim = [ 3.0, 0.8 ]
 fprim = [ 0.9, 0.9 ]
 vnewk = [ 0.0, 0.0 ]
 type = [ "ion", "electron" ]
""".strip()
        + "\n"
    )

    # 1. Collisional electrostatic TEM (Miller geometry, distinct q/shat/dens/temp/nu, beta = 0)
    tem_path = deck_dir / "tem_coll.in"
    tem_path.write_text(
        f"""{common_dims}
[Physics]
 beta = 0.0
 nonlinear_mode = {nl_str}
 fphi = 1.0
 fapar = 0.0
 fbpar = 0.0

[Time]
 dt = {dt_per_member[1]}
 dt_max = 0.01
 nstep = {nstep_per_member[1]}
 t_max = {t_max_per_member[1]}
 cfl = {cfl_per_member[1]}
 fixed_dt = {fdt_str}
 scheme = "{scheme}"

[Geometry]
 geo_option = "miller"
 rhoc = 0.55
 Rmaj = 2.9
 R_geo = 2.9
 qinp = 1.85
 shat = {shat_tokamak[1]}
 shift = -0.05
 akappa = 1.25
 akappri = 0.1
 tri = 0.15
 tripri = 0.05
 betaprim = 0.0

[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.025 ]
 dens = [ 1.1, 1.1 ]
 temp = [ 0.85, 1.35 ]
 tprim = [ 0.7, 3.6 ]
 fprim = [ 2.2, 2.2 ]
 vnewk = [ 0.005, 0.04 ]
 type = [ "ion", "electron" ]
""".strip()
        + "\n"
    )

    # 2. Electromagnetic KBM (Miller geometry, beta = 0.018 > 0 with full apar + bpar)
    kbm_path = deck_dir / "kbm_em.in"
    kbm_path.write_text(
        f"""{common_dims}
[Physics]
 beta = 0.018
 nonlinear_mode = {nl_str}
 fphi = 1.0
 fapar = 1.0
 fbpar = 1.0

[Time]
 dt = {dt_per_member[2]}
 dt_max = 0.01
 nstep = {nstep_per_member[2]}
 t_max = {t_max_per_member[2]}
 cfl = {cfl_per_member[2]}
 fixed_dt = {fdt_str}
 scheme = "{scheme}"

[Geometry]
 geo_option = "miller"
 rhoc = 0.5
 Rmaj = 2.77778
 R_geo = 2.77778
 qinp = 1.4
 shat = {shat_tokamak[2]}
 shift = -0.15
 akappa = 1.35
 akappri = 0.15
 tri = 0.2
 tripri = 0.08
 betaprim = -0.04

[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.02 ]
 dens = [ 1.0, 1.0 ]
 temp = [ 1.0, 1.05 ]
 tprim = [ 2.6, 2.5 ]
 fprim = [ 1.4, 1.4 ]
 vnewk = [ 0.002, 0.01 ]
 type = [ "ion", "electron" ]
""".strip()
        + "\n"
    )

    # 3. W7-X 3D Stellarator (authentic .eik.nc from wout_w7x.nc, nfp=5, shat < 0)
    w7x_path = deck_dir / "w7x_stel.in"
    w7x_path.write_text(
        f"""{common_dims}
[Physics]
 beta = 0.006
 nonlinear_mode = {nl_str}
 fphi = 1.0
 fapar = 1.0
 fbpar = 1.0

[Time]
 dt = {dt_per_member[3]}
 dt_max = 0.01
 nstep = {nstep_per_member[3]}
 t_max = {t_max_per_member[3]}
 cfl = {cfl_per_member[3]}
 fixed_dt = {fdt_str}
 scheme = "{scheme}"

[Geometry]
 geo_option = "nc"
 geo_file = "{w7x_eik_path}"

[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.02 ]
 dens = [ 0.95, 0.95 ]
 temp = [ 1.15, 0.9 ]
 tprim = [ 2.9, 1.5 ]
 fprim = [ 1.1, 1.1 ]
 vnewk = [ 0.003, 0.015 ]
 type = [ "ion", "electron" ]
""".strip()
        + "\n"
    )

    return [itg_path, tem_path, kbm_path, w7x_path]


def _assert_close_tol(actual: Any, expected: Any, tol: float, label: str) -> None:
    act = np.asarray(actual)
    exp = np.asarray(expected)
    assert np.all(np.isfinite(act)), f"{label}: actual contains NaN/Inf"
    assert np.all(np.isfinite(exp)), f"{label}: expected contains NaN/Inf"
    scale = max(1.0, float(np.max(np.abs(exp)))) if exp.size > 0 else 1.0
    np.testing.assert_allclose(act, exp, rtol=tol, atol=tol * scale, err_msg=label)


# =============================================================================
# 1. Heterogeneous B=4 Ensemble Parity (ITG + TEM + EM KBM + W7-X Stellarator)
# =============================================================================


@pytest.mark.parametrize(
    ("precision", "tol"),
    [
        ("fp64", 1.0e-12),
        ("fp32", 1.0e-5),
    ],
)
def test_heterogeneous_b4_itg_tem_kbm_w7x_parity_vs_standalone(
    tmp_path: Path,
    w7x_eik_nc_8: Path,
    precision: str,
    tol: float,
) -> None:
    """Verify < 1e-12 (c128) & < 1e-6 (c64) parity on B=4 ITG + TEM + EM KBM + W7-X."""
    deck_dir = tmp_path / f"decks_{precision}"
    ens_out = tmp_path / f"ens_out_{precision}"
    ref_out = tmp_path / f"ref_out_{precision}"
    ref_out.mkdir(parents=True, exist_ok=True)

    decks = _write_heterogeneous_b4_decks(
        deck_dir,
        w7x_eik_nc_8,
        nonlinear=True,
        scheme="rk3",
        fixed_dt=False,
        nstep_per_member=(4, 4, 4, 4),
        t_max_per_member=(10.0, 10.0, 10.0, 10.0),
        cfl_per_member=(0.85, 0.75, 0.80, 0.70),
        nwrite=2,
        write_fields=True,
        write_moms=True,
    )

    ens_res = run_ensemble(
        decks,
        output_dir=ens_out,
        precision=precision,
        save_restart=True,
    )
    assert ens_res["batch_size"] == 4
    assert ens_res["summary_json"].is_file()
    summary_payload = json.loads(
        ens_res["summary_json"].read_text(), parse_constant=_reject_non_strict_json
    )
    assert summary_payload["batch_size"] == 4
    assert summary_payload["all_converged"] is True
    assert summary_payload["any_nan_or_inf"] is False

    for b, deck_path in enumerate(decks):
        stem = deck_path.stem
        ref_nc = ref_out / f"{stem}.out.nc"
        ref_rest = ref_out / f"{stem}.restart.nc"
        ref_json = ref_out / f"{stem}.diagnostics_metrics.json"
        ref_res = run_cli_simulation(
            deck_path,
            output_path=ref_nc,
            precision=precision,
            save_restart_path=ref_rest,
            metrics_json_path=ref_json,
        )

        # 1. Final state G, fields (phi, apar, bpar, phi_avg), nstep, and time
        assert int(ens_res["nstep"][b]) == int(ref_res["nstep"])
        _assert_close_tol(ens_res["t_final"][b], ref_res["t_final"], tol, f"{stem}:t_final")
        _assert_close_tol(ens_res["g_final"][b], ref_res["g_final"], tol, f"{stem}:G")
        _assert_close_tol(
            ens_res["fields_final"].phi[b], ref_res["fields_final"].phi, tol, f"{stem}:phi"
        )
        _assert_close_tol(
            ens_res["fields_final"].apar[b], ref_res["fields_final"].apar, tol, f"{stem}:apar"
        )
        _assert_close_tol(
            ens_res["fields_final"].bpar[b], ref_res["fields_final"].bpar, tol, f"{stem}:bpar"
        )

        # Explicit mixed ES/EM check: ES members (0, 1) have exact zero apar/bpar, EM (2, 3) > 0
        if b in (0, 1):
            assert float(np.max(np.abs(np.asarray(ens_res["fields_final"].apar[b])))) == 0.0
            assert float(np.max(np.abs(np.asarray(ens_res["fields_final"].bpar[b])))) == 0.0
        else:
            assert float(np.max(np.abs(np.asarray(ens_res["fields_final"].apar[b])))) > 0.0
            assert float(np.max(np.abs(np.asarray(ens_res["fields_final"].bpar[b])))) > 0.0

        # 2. .out.nc NetCDF variable-by-variable parity (note .out.nc stores float32 diagnostics)
        nc_tol = max(tol, 1.0e-6)
        ens_nc = ens_res["members"][b]["output_nc"]
        with nc.Dataset(ens_nc, "r") as ds_e, nc.Dataset(ref_nc, "r") as ds_r:
            _assert_close_tol(
                ds_e.groups["Grids"].variables["time"][:],
                ds_r.groups["Grids"].variables["time"][:],
                tol,
                f"{stem}:nc/time",
            )
            for gname in ("bmag", "gradpar", "shat", "q", "grho_avg"):
                _assert_close_tol(
                    ds_e.groups["Geometry"].variables[gname][:],
                    ds_r.groups["Geometry"].variables[gname][:],
                    nc_tol,
                    f"{stem}:Geometry/{gname}",
                )
            d_e = ds_e.groups["Diagnostics"].variables
            d_r = ds_r.groups["Diagnostics"].variables
            for dname in (
                "Phi2_t",
                "Phi2_kxkyt",
                "Phi2_zonal_t",
                "Apar2_t",
                "Bpar2_t",
                "Wg_t",
                "Wg_st",
                "Wg_lmst",
                "Wphi_t",
                "Wapar_t",
                "Wbpar_t",
                "HeatFlux_st",
                "HeatFluxES_st",
                "HeatFluxApar_st",
                "HeatFluxBpar_st",
                "ParticleFlux_st",
                "TurbulentHeating_st",
                "omega_kxkyt",
                "Phi",
                "Apar",
                "Bpar",
                "Density",
                "Upar",
                "Tpar",
                "Tperp",
            ):
                v_tol = (
                    max(nc_tol, 5.0e-3)
                    if precision == "fp32" and dname in ("omega_kxkyt", "TurbulentHeating_st")
                    else nc_tol
                )
                _assert_close_tol(d_e[dname][:], d_r[dname][:], v_tol, f"{stem}:Diag/{dname}")

            if b in (0, 1):
                assert np.all(np.asarray(d_e["Wapar_t"][:]) == 0.0)
                assert np.all(np.asarray(d_e["Wbpar_t"][:]) == 0.0)
            else:
                assert np.all(np.asarray(d_e["Wapar_t"][:]) > 0.0)
                assert np.all(np.asarray(d_e["Wbpar_t"][:]) > 0.0)

        # 3. .restart.nc checkpoint parity
        ens_rest = ens_res["members"][b]["restart_nc"]
        assert ens_rest is not None and ens_rest.is_file()
        cfg_b, grids_b, _ = init_geometry(load_config(deck_path))
        del cfg_b
        g_e_rest, t_e_rest = load_restart_nc(ens_rest, grids_b, dtype=jnp.complex128)
        g_r_rest, t_r_rest = load_restart_nc(ref_rest, grids_b, dtype=jnp.complex128)
        _assert_close_tol(t_e_rest, t_r_rest, tol, f"{stem}:restart_time")
        _assert_close_tol(g_e_rest, g_r_rest, nc_tol, f"{stem}:restart_G")

        # 4. Strict RFC-8259 KR2 diagnostics_metrics.json parity
        ens_json = ens_res["members"][b]["metrics_json"]
        m_e = json.loads(ens_json.read_text(), parse_constant=_reject_non_strict_json)
        m_r = json.loads(ref_json.read_text(), parse_constant=_reject_non_strict_json)
        assert m_e["converged"] is True
        assert m_e["has_nan_or_inf"] is False
        assert m_e["n_time"] == m_r["n_time"]
        assert m_e["nstep"] == m_r["nstep"]
        for k_metric in (
            "t_end",
            "phi2_end",
            "Q_total",
            "Wg_total",
            "Wphi_total",
            "TurbulentHeating_total",
            "sim_time_reached",
        ):
            _assert_close_tol(m_e[k_metric], m_r[k_metric], nc_tol, f"{stem}:json/{k_metric}")


# =============================================================================
# 2. Heterogeneous Per-Member `t_max` Mid-Chunk Freezing & Adaptive Variable-dt
# =============================================================================


@pytest.mark.parametrize(
    ("fixed_dt", "scheme", "precision", "tol"),
    [
        (True, "rk3", "fp64", 1.0e-12),
        (False, "sspx2", "fp64", 1.0e-12),
        (False, "rk3", "fp32", 1.0e-5),
    ],
)
def test_heterogeneous_tmax_mid_chunk_freezing_matches_standalone(
    tmp_path: Path,
    w7x_eik_nc_8: Path,
    fixed_dt: bool,
    scheme: str,
    precision: str,
    tol: float,
) -> None:
    """Verify per-member `t_max` mid-chunk stopping & penultimate `(G_old, fields_old)` freeze."""
    deck_dir = tmp_path / f"tmax_{scheme}_{fixed_dt}_{precision}"
    ens_out = tmp_path / f"tmax_ens_{scheme}_{fixed_dt}_{precision}"
    ref_out = tmp_path / f"tmax_ref_{scheme}_{fixed_dt}_{precision}"
    ref_out.mkdir(parents=True, exist_ok=True)

    # Calibrate per-member t_max to (1.5*dt0, 2.5*dt0, 5.5*dt0, 10.0) using each member's
    # true initial CFL/fixed dt0 so that with nwrite=4:
    # - Member 0 stops mid-chunk at step 2
    # - Member 1 stops mid-chunk at step 3
    # - Member 2 crosses two nwrite boundaries & stops mid-chunk at step 6
    # - Member 3 runs to nstep=7 (stopped by nstep while t_max is large)
    probe_decks = _write_heterogeneous_b4_decks(
        deck_dir / "probe",
        w7x_eik_nc_8,
        nonlinear=True,
        scheme=scheme,
        fixed_dt=fixed_dt,
        dt_per_member=(1.8e-4, 2.0e-4, 1.6e-4, 1.5e-4),
        t_max_per_member=(10.0, 10.0, 10.0, 10.0),
        nstep_per_member=(50, 50, 50, 7),
        cfl_per_member=(0.45, 0.65, 0.55, 0.70),
        nwrite=4,
        write_fields=False,
        write_moms=False,
    )
    probe_specs, probe_ops = init_batched_ensemble(probe_decks, dtype=jnp.float64)
    g_probe = jnp.stack(
        [init_moments(s.cfg, s.grids, s.geo, dtype=jnp.complex128) for s in probe_specs], axis=0
    )
    dt0_np = np.asarray(init_batched_integrator_state(g_probe, probe_ops).dt, dtype=np.float64)
    calibrated_t_max = (
        float(1.5 * dt0_np[0]),
        float(2.5 * dt0_np[1]),
        float(5.5 * dt0_np[2]),
        10.0,
    )

    decks = _write_heterogeneous_b4_decks(
        deck_dir,
        w7x_eik_nc_8,
        nonlinear=True,
        scheme=scheme,
        fixed_dt=fixed_dt,
        dt_per_member=(1.8e-4, 2.0e-4, 1.6e-4, 1.5e-4),
        t_max_per_member=calibrated_t_max,
        nstep_per_member=(50, 50, 50, 7),
        cfl_per_member=(0.45, 0.65, 0.55, 0.70),
        nwrite=4,
        write_fields=False,
        write_moms=False,
    )

    ens_res = run_ensemble(decks, output_dir=ens_out, precision=precision, save_restart=True)

    # Confirm members executed genuinely different step counts in the same batched run!
    steps_taken = [int(s) for s in ens_res["nstep"]]
    assert len(set(steps_taken)) >= 3, f"Expected heterogeneous step counts, got {steps_taken}"

    nc_tol = max(tol, 1.0e-6)
    for b, deck_path in enumerate(decks):
        stem = deck_path.stem
        ref_nc = ref_out / f"{stem}.out.nc"
        ref_res = run_cli_simulation(deck_path, output_path=ref_nc, precision=precision)

        assert int(ens_res["nstep"][b]) == int(ref_res["nstep"]), (
            f"{stem}: ensemble nstep={ens_res['nstep'][b]} != standalone {ref_res['nstep']}"
        )
        _assert_close_tol(ens_res["t_final"][b], ref_res["t_final"], tol, f"{stem}:t_final")
        _assert_close_tol(ens_res["g_final"][b], ref_res["g_final"], tol, f"{stem}:G_frozen")
        _assert_close_tol(
            ens_res["fields_final"].phi[b], ref_res["fields_final"].phi, tol, f"{stem}:phi_frozen"
        )

        ens_nc = ens_res["members"][b]["output_nc"]
        with nc.Dataset(ens_nc, "r") as ds_e, nc.Dataset(ref_nc, "r") as ds_r:
            t_e = np.asarray(ds_e.groups["Grids"].variables["time"][:], dtype=np.float64)
            t_r = np.asarray(ds_r.groups["Grids"].variables["time"][:], dtype=np.float64)
            assert t_e.shape == t_r.shape, f"{stem}: frame count {t_e.shape} != {t_r.shape}"
            _assert_close_tol(t_e, t_r, tol, f"{stem}:Grids/time")

            for vname in (
                "Phi2_t",
                "HeatFlux_st",
                "TurbulentHeating_st",
                "omega_kxkyt",
                "Wapar_t",
                "Wbpar_t",
            ):
                act_v = ds_e.groups["Diagnostics"].variables[vname][:]
                exp_v = ds_r.groups["Diagnostics"].variables[vname][:]
                if precision == "fp32" and vname == "omega_kxkyt":
                    # Zonal ky=0 and noise-seeded harmonics near fp32 floor amplify atan2/log / dt.
                    act_v = act_v[:, 1:, ...]
                    exp_v = exp_v[:, 1:, ...]
                    v_tol = max(nc_tol, 0.1)
                elif precision == "fp32" and vname == "TurbulentHeating_st":
                    v_tol = max(nc_tol, 5.0e-3)
                else:
                    v_tol = nc_tol
                _assert_close_tol(act_v, exp_v, v_tol, f"{stem}:Diag/{vname}")


# =============================================================================
# 3. Single-Trace Compilation, StableHLO IR Audit, PyTree, & Additional Regimes
# =============================================================================


def test_pytree_roundtrip_single_trace_and_stablehlo_ir(
    tmp_path: Path,
    w7x_eik_nc_8: Path,
) -> None:
    """Verify BatchedEnsembleOperators PyTree, trace_count == 1, and vectorized f32 StableHLO."""
    decks = _write_heterogeneous_b4_decks(
        tmp_path / "ir_decks",
        w7x_eik_nc_8,
        nonlinear=True,
        scheme="sspx3",
        fixed_dt=True,
        nstep_per_member=(6, 6, 6, 6),
        nwrite=2,
        write_fields=False,
        write_moms=False,
    )
    specs, ops64 = init_batched_ensemble(decks, dtype=jnp.float64)
    assert isinstance(ops64, BatchedEnsembleOperators)
    assert len(specs) == 4 and ops64.batch_size == 4

    # 1. PyTree flatten / unflatten & .astype(float32) round-trip
    leaves64, treedef64 = jax.tree_util.tree_flatten(ops64)
    assert all(isinstance(lf, jax.Array) and lf.shape[0] == 4 for lf in leaves64)
    ops64_rt = jax.tree_util.tree_unflatten(treedef64, leaves64)
    assert ops64_rt.batch_size == 4 and ops64_rt.scheme == "sspx3"

    ops32 = ops64.astype(jnp.float32)
    assert ops32.solver.jl.dtype == jnp.float32
    assert ops32.beta_b.dtype == jnp.float32
    assert ops32.t_max_b.dtype == jnp.float64  # Time bounds remain IEEE float64

    # 2. Verify trace_count == 1 across multiple dynamic chunk lengths (1, 3, 2)
    g_init = jnp.stack(
        [init_moments(sp.cfg, sp.grids, sp.geo, dtype=jnp.complex64) for sp in specs], axis=0
    )
    st0 = init_batched_integrator_state(g_init, ops32)
    dg0 = jax.vmap(lambda g_m, f_m, t_m: init_diagnostics_state(g_m, f_m, time=t_m, has_old=False))(
        st0.G, st0.fields, st0.time
    )

    trace_counter = {"n": 0}

    @jax.jit
    def _counted_chunk(st: Any, dg: Any, n_max: jax.Array) -> Any:
        trace_counter["n"] += 1
        return advance_ensemble_chunk(st, dg, ops32, n_max)

    c1 = _counted_chunk(st0, dg0, jnp.asarray(1, dtype=jnp.int32))
    c2 = _counted_chunk(c1.state, c1.diag_state, jnp.asarray(3, dtype=jnp.int32))
    _ = _counted_chunk(c2.state, c2.diag_state, jnp.asarray(2, dtype=jnp.int32))
    assert trace_counter["n"] == 1, f"Expected single JIT trace, got {trace_counter['n']}"

    # Also exercise batched_solve_fields, batched_step_integrator_state, & mesh helpers in-process
    mesh1 = create_ensemble_mesh(4, num_devices=1)
    st0_sh = shard_ensemble_tree(st0, mesh1, batch_size=4)
    st0_gat = gather_ensemble_tree(st0_sh, mesh1)
    assert st0_gat.G.shape == st0.G.shape
    stepper_fn = make_ensemble_chunk_stepper(ops32, mesh=mesh1)
    c_m1 = stepper_fn(st0_sh, shard_ensemble_tree(dg0, mesh1, batch_size=4), jnp.int32(1))
    f_direct = jax.jit(batched_solve_fields)(st0.G, ops32)
    _assert_close_tol(f_direct.phi, st0.fields.phi, 1.0e-6, "batched_solve_fields:phi")
    st1_direct = jax.jit(batched_step_integrator_state)(st0, ops32)
    _assert_close_tol(st1_direct.G, c1.state.G, 1.0e-6, "batched_step_integrator_state:G")
    _assert_close_tol(c_m1.state.G, c1.state.G, 1.0e-6, "make_ensemble_chunk_stepper:G")

    # 3. Inspect StableHLO of batched_gyrokinetic_rhs in complex64:
    #    confirm batch-vectorized tensor<4x...> IR and zero float64 ops in StableHLO body!
    dt32 = jnp.full((4,), 1.0e-4, dtype=jnp.float32)
    rhs_ir = str(
        jax.jit(batched_gyrokinetic_rhs)
        .lower(st0.G, st0.fields, ops32, dt32)
        .compiler_ir(dialect="stablehlo")
    )
    assert "tensor<4x" in rhs_ir
    rhs_ops = [ln for ln in rhs_ir.splitlines() if "stablehlo." in ln]
    assert len(rhs_ops) > 0
    assert all("f64" not in ln for ln in rhs_ops), (
        "complex64 batched_gyrokinetic_rhs leaked float64 ops in StableHLO"
    )

    diag_ir = str(
        jax.jit(
            lambda g_a, f_a, dg_a, t_a, dt_a: batched_compute_diagnostics_step(
                g_a, f_a, dg_a, ops32, time=t_a, dt=dt_a
            )
        )
        .lower(st0.G, st0.fields, dg0, st0.time, st0.dt)
        .compiler_ir(dialect="stablehlo")
    )
    assert "tensor<4x" in diag_ir

    # 4. Verify mixed ParallelGrad (|jtwist| >= nakx) + LinkedParallelGrad (W7-X) promotion
    mixed_decks = _write_heterogeneous_b4_decks(
        tmp_path / "mixed_pg_decks",
        w7x_eik_nc_8,
        nonlinear=True,
        scheme="rk3",
        fixed_dt=False,
        shat_tokamak=(0.85, 1.25, 0.85),
        nstep_per_member=(3, 3, 3, 3),
        nwrite=3,
        write_fields=False,
        write_moms=False,
    )
    mixed_ens = run_ensemble(
        [mixed_decks[0], mixed_decks[3]],
        output_dir=tmp_path / "mixed_pg_out",
        precision="fp64",
        save_restart=False,
    )
    ref0 = run_cli_simulation(
        mixed_decks[0], output_path=tmp_path / "mixed_ref0.out.nc", precision="fp64"
    )
    _assert_close_tol(mixed_ens["g_final"][0], ref0["g_final"], 1.0e-12, "mixed_pg:G0")


def test_boltzmann_tau_sweep_and_exb_forcing_rk4_k10_parity(tmp_path: Path) -> None:
    """Verify in-memory GXConfig specs with Boltzmann tau_fac sweep, ExB shear, forcing, & rk4."""
    specs_in: list[GXConfig] = []
    for idx, (tau_val, g_exb_val, f_amp) in enumerate(
        [(0.8, 0.0, 0.0), (1.2, 0.12, 0.02), (1.6, -0.08, 0.0)]
    ):
        specs_in.append(
            GXConfig(
                run_name=f"boltz_{idx}",
                nx=6,
                ny=6,
                nz=8,
                ntheta_in=8,
                nperiod=1,
                zp=1,
                nhermite=4,
                nlaguerre=2,
                nspecies=1,
                y0=10.0,
                nonlinear_mode=True,
                exb_shear=(g_exb_val != 0.0),
                exb_shear_phase=(g_exb_val != 0.0),
                g_exb=g_exb_val,
                forcing_init=(f_amp > 0.0),
                forcing_type="Kz",
                stir_field="density",
                forcing_amp=f_amp,
                forcing_index=1,
                all_kinetic=False,
                add_boltzmann_species=True,
                boltzmann_opt=BoltzmannOption.BOLTZMANN_ELECTRONS,
                boltzmann_type="electrons",
                tau_fac=tau_val,
                geo_option="s-alpha",
                eps=0.167,
                rmaj=2.77,
                qsf=1.4 + 0.1 * idx,
                shat=0.8,
                dt=0.002,
                fixed_dt=True,
                nstep=3,
                nwrite=1,
                scheme="rk4",
                init_amp=1.0e-3,
                random_seed=30 + idx,
                species=(
                    SpeciesConfig.from_raw(
                        z=1.0, mass=1.0, dens=1.0, temp=1.0, tprim=2.5 + 0.3 * idx, fprim=0.8
                    ),
                ),
            )
        )

    out_dir = tmp_path / "boltz_ens"
    ens_res = run_ensemble(specs_in, output_dir=out_dir, precision="fp64", save_restart=False)
    assert ens_res["batch_size"] == 3
    assert ens_res["ensemble_summary"]["all_converged"] is True
    for m in ens_res["members"]:
        assert m["restart_nc"] is None
        assert m["output_nc"].is_file()


# =============================================================================
# 4. Multi-Device `"batch"` Mesh Sharding Subprocess Test (4 Virtual Devices)
# =============================================================================


def test_multidevice_4_device_batch_mesh_sharding_subprocess(
    tmp_path: Path,
    w7x_eik_nc_8: Path,
) -> None:
    """Verify B=4 and B=8 `'batch'` Mesh sharding across 4 devices matches 1-device `vmap`."""
    deck_dir = tmp_path / "mp_decks"
    decks = _write_heterogeneous_b4_decks(
        deck_dir,
        w7x_eik_nc_8,
        nonlinear=True,
        scheme="rk3",
        fixed_dt=False,
        dt_per_member=(0.002, 0.002, 0.002, 0.002),
        t_max_per_member=(0.0035, 0.007, 10.0, 10.0),
        nstep_per_member=(20, 20, 4, 5),
        cfl_per_member=(0.6, 0.7, 0.75, 0.8),
        nwrite=2,
        write_fields=True,
        write_moms=True,
    )

    # Reference 1-device run in-process (both fp64 and fp32)
    ref64_dir = tmp_path / "ref_1dev_fp64"
    ref32_dir = tmp_path / "ref_1dev_fp32"
    ref64 = run_ensemble(decks, output_dir=ref64_dir, precision="fp64", batch_devices=1)
    ref32 = run_ensemble(decks, output_dir=ref32_dir, precision="fp32", batch_devices=1)

    script = f"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from gx_jax.ensemble import (
    create_ensemble_mesh,
    gather_ensemble_tree,
    init_batched_ensemble,
    init_batched_integrator_state,
    run_ensemble,
    shard_ensemble_tree,
)
from gx_jax.fields import init_moments

assert jax.device_count() == 4
decks = [Path(p) for p in {[str(p) for p in decks]!r}]
out64 = Path({str(tmp_path / "sh_4dev_fp64")!r})
out32 = Path({str(tmp_path / "sh_4dev_fp32")!r})
out8 = Path({str(tmp_path / "sh_4dev_b8")!r})

# Check explicit 4-device ('batch',) Mesh & NamedSharding placement
mesh4 = create_ensemble_mesh(4, num_devices=4)
assert mesh4.axis_names == ("batch",)
assert int(mesh4.shape["batch"]) == 4

specs, ops = init_batched_ensemble(decks, dtype=jnp.float64)
ops_sh = shard_ensemble_tree(ops, mesh4, batch_size=4)
g0 = jnp.stack([init_moments(s.cfg, s.grids, s.geo, dtype=jnp.complex128) for s in specs], axis=0)
st_sh = init_batched_integrator_state(
    shard_ensemble_tree(g0, mesh4, batch_size=4), ops_sh, mesh=mesh4
)
assert isinstance(st_sh.G.sharding, NamedSharding)
assert st_sh.G.sharding.spec == PartitionSpec("batch", None, None, None, None, None, None)
assert len(st_sh.G.addressable_shards) == 4
assert st_sh.G.addressable_shards[0].data.shape[0] == 1
g_gathered = gather_ensemble_tree(st_sh.G, mesh4)
assert g_gathered.shape == st_sh.G.shape

res64 = run_ensemble(decks, output_dir=out64, precision="fp64", batch_devices=4)
res32 = run_ensemble(decks, output_dir=out32, precision="fp32", batch_devices=4)
assert res64["num_devices"] == 4
assert res32["num_devices"] == 4

# Also test B=8 (2 members per device across 4 devices, with stem disambiguation!)
res8 = run_ensemble(decks + decks, output_dir=out8, precision="fp32", batch_devices=4, nstep=2)
assert res8["batch_size"] == 8 and res8["num_devices"] == 4
assert len({{m["output_nc"] for m in res8["members"]}}) == 8

np.save(out64 / "g_final.npy", np.asarray(res64["g_final"]))
np.save(out64 / "phi_final.npy", np.asarray(res64["fields_final"].phi))
np.save(out64 / "t_final.npy", np.asarray(res64["t_final"]))
np.save(out32 / "g_final.npy", np.asarray(res32["g_final"]))
print("4-DEVICE BATCH MESH ENSEMBLE PASSED")
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
        f"4-device subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "4-DEVICE BATCH MESH ENSEMBLE PASSED" in proc.stdout

    g64_4dev = np.load(tmp_path / "sh_4dev_fp64" / "g_final.npy")
    phi64_4dev = np.load(tmp_path / "sh_4dev_fp64" / "phi_final.npy")
    t64_4dev = np.load(tmp_path / "sh_4dev_fp64" / "t_final.npy")
    g32_4dev = np.load(tmp_path / "sh_4dev_fp32" / "g_final.npy")

    _assert_close_tol(t64_4dev, ref64["t_final"], 1.0e-12, "4dev_vs_1dev:t_final_fp64")
    _assert_close_tol(g64_4dev, ref64["g_final"], 1.0e-12, "4dev_vs_1dev:G_fp64")
    _assert_close_tol(phi64_4dev, ref64["fields_final"].phi, 1.0e-12, "4dev_vs_1dev:phi_fp64")
    _assert_close_tol(g32_4dev, ref32["g_final"], 1.0e-6, "4dev_vs_1dev:G_fp32")

    for b, deck_path in enumerate(decks):
        stem = deck_path.stem
        with (
            nc.Dataset(tmp_path / "sh_4dev_fp64" / f"{stem}.out.nc", "r") as ds4,
            nc.Dataset(ref64["members"][b]["output_nc"], "r") as ds1,
        ):
            _assert_close_tol(
                ds4.groups["Diagnostics"].variables["Phi2_t"][:],
                ds1.groups["Diagnostics"].variables["Phi2_t"][:],
                1.0e-6,
                f"4dev_nc:{stem}:Phi2_t",
            )


# =============================================================================
# 5. `gx-jax-ensemble` CLI End-to-End & Structural Compatibility Validation
# =============================================================================


def test_cli_end_to_end_and_structural_validation_errors(
    tmp_path: Path,
    w7x_eik_nc_8: Path,
) -> None:
    """Verify `gx_jax.ensemble:main` CLI end-to-end and structural mismatch `ValueError`s."""
    assert main is ensemble_main
    parser = build_ensemble_arg_parser()
    defaults = parser.parse_args(["a.in", "b.in"])
    assert defaults.precision == "fp32"

    decks = _write_heterogeneous_b4_decks(
        tmp_path / "cli_decks",
        w7x_eik_nc_8,
        nonlinear=False,
        scheme="rk3",
        fixed_dt=True,
        nstep_per_member=(3, 3, 3, 3),
        nwrite=1,
        write_fields=False,
        write_moms=False,
    )
    cli_out = tmp_path / "cli_out"
    custom_summary = tmp_path / "custom_ensemble_summary.json"

    rc = main(
        [
            *[str(p) for p in decks],
            "-o",
            str(cli_out),
            "--summary-json",
            str(custom_summary),
            "--nstep",
            "3",
            "--dt",
            "0.00015",
            "--nwrite",
            "1",
            "--scheme",
            "sspx3",
            "--linear",
            "--precision",
            "fp32",
            "--fft-backend",
            "fft",
            "--save-restart",
        ]
    )
    assert rc == 0
    assert custom_summary.is_file()
    summary = json.loads(custom_summary.read_text(), parse_constant=_reject_non_strict_json)
    assert summary["batch_size"] == 4
    assert summary["scheme"] == "sspx3"
    assert summary["nonlinear"] is False
    assert summary["precision"] == "fp32"
    assert summary["all_converged"] is True
    for p in decks:
        assert (cli_out / f"{p.stem}.out.nc").is_file()
        assert (cli_out / f"{p.stem}.restart.nc").is_file()
        assert (cli_out / f"{p.stem}.diagnostics_metrics.json").is_file()

    # Structural incompatibility ValueErrors
    with pytest.raises(ValueError, match="at least 1 member"):
        validate_ensemble_compatibility([])

    specs_ok, _ = init_batched_ensemble(decks[:2], dtype=jnp.float32)
    base_sp = specs_ok[0]

    def _mut_spec(cfg_mod: GXConfig) -> EnsembleMemberSpec:
        c_f, g_f, geo_f = init_geometry(cfg_mod)
        return EnsembleMemberSpec(cfg=c_f, grids=g_f, geo=geo_f, name="bad")

    bad_cases: list[tuple[EnsembleMemberSpec, str]] = [
        (_mut_spec(replace(base_sp.cfg, nhermite=6)), "nhermite"),
        (_mut_spec(replace(base_sp.cfg, nlaguerre=3)), "nlaguerre"),
        (_mut_spec(replace(base_sp.cfg, nx=10)), "nakx"),
        (_mut_spec(replace(base_sp.cfg, ny=10)), "naky"),
        (_mut_spec(replace(base_sp.cfg, nz=16, ntheta_in=16)), "nz"),
        (_mut_spec(replace(base_sp.cfg, scheme="rk4")), "scheme"),
        (_mut_spec(replace(base_sp.cfg, nonlinear_mode=True, linear=False)), "nonlinear_mode"),
        (_mut_spec(replace(base_sp.cfg, closure_model="beer4+2")), "closure_model"),
        (_mut_spec(replace(base_sp.cfg, non_twist=True)), "non_twist"),
        (
            _mut_spec(
                replace(
                    base_sp.cfg,
                    nspecies=1,
                    species=(base_sp.cfg.species[0],),
                    all_kinetic=False,
                    add_boltzmann_species=True,
                )
            ),
            "nspecies",
        ),
    ]
    for bad_sp, expected_substr in bad_cases:
        with pytest.raises(ValueError, match=expected_substr):
            validate_ensemble_compatibility([base_sp, bad_sp])

    # CLI exit code 2 on missing file, structural mismatch, or non-divisible --batch-devices
    with pytest.raises(SystemExit) as exc_miss:
        main([str(tmp_path / "missing.in")])
    assert exc_miss.value.code == 2

    bad_deck = tmp_path / "bad_nhermite.in"
    bad_deck.write_text(decks[0].read_text().replace("nhermite = 4", "nhermite = 6"))
    with pytest.raises(SystemExit) as exc_incompat:
        main([str(decks[0]), str(bad_deck), "-o", str(tmp_path / "fail_out")])
    assert exc_incompat.value.code == 2

    with pytest.raises(SystemExit) as exc_dev:
        main([str(decks[0]), "-o", str(tmp_path / "fail_dev"), "--batch-devices", "4"])
    assert exc_dev.value.code == 2


def test_periodic_nsave_non_dividing_nwrite_matches_standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify periodic `.restart.nc` checkpoints when `nwrite` does not divide `nsave`.

    Specifically tests `(nwrite=10, nsave=4, nstep=13)` on Member 0 (saves at steps
    `[1, 5, 9, 13]`) and `(nwrite=3, nsave=5, nstep=13)` on Member 1 (periodic saves at
    `[1, 6, 11]` + final save at `13`) in the same `jax.vmap` ensemble, and verifies exact
    `.restart.nc` & `.out.nc` parity against standalone `run_cli_simulation`.
    """
    import gx_jax.ensemble as ens_mod

    deck_dir = tmp_path / "nsave_decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    dt_val = 0.001

    def _make_nsave_deck(name: str, nwrite_v: int, nsave_v: int, tprim_v: float) -> Path:
        p = deck_dir / f"{name}.in"
        p.write_text(
            f"""
[Dimensions]
 ntheta = 4
 nperiod = 1
 ny = 4
 nx = 4
 nhermite = 2
 nlaguerre = 1
 nspecies = 1

[Domain]
 y0 = 10.0
 boundary = "linked"

[Physics]
 beta = 0.0
 nonlinear_mode = false

[Time]
 dt = {dt_val}
 nstep = 13
 fixed_dt = true
 scheme = "rk3"

[Initialization]
 init_field = "density"
 init_amp = 1.0e-5
 ikpar_init = 0

[Geometry]
 geo_option = "s-alpha"
 eps = 0.18
 Rmaj = 2.77
 qinp = 1.4
 shat = 0.8

[Boltzmann]
 add_Boltzmann_species = true
 Boltzmann_type = "electrons"
 tau_fac = 1.0

[Diagnostics]
 nwrite = {nwrite_v}
 fluxes = true
 free_energy = true

[Restart]
 save_for_restart = true
 nsave = {nsave_v}

[species]
 z = [ 1.0 ]
 mass = [ 1.0 ]
 dens = [ 1.0 ]
 temp = [ 1.0 ]
 tprim = [ {tprim_v} ]
 fprim = [ 0.8 ]
 vnewk = [ 0.0 ]
 type = [ "ion" ]
""".strip()
            + "\n"
        )
        return p

    d0 = _make_nsave_deck("nw10_ns4", nwrite_v=10, nsave_v=4, tprim_v=2.8)
    d1 = _make_nsave_deck("nw3_ns5", nwrite_v=3, nsave_v=5, tprim_v=3.4)

    saved_steps_by_member: dict[int, list[int]] = {0: [], 1: []}
    midrun_snapshots: dict[tuple[int, int], Path] = {}
    orig_save = ens_mod._atomic_save_member_restart_nc

    def _spy_save(
        path: Path, g_state: Any, time_val: float, grids: Any, member_idx: int = 0
    ) -> None:
        step_k = int(round(time_val / dt_val))
        saved_steps_by_member[member_idx].append(step_k)
        orig_save(path, g_state, time_val, grids, member_idx=member_idx)
        if (member_idx, step_k) in ((0, 9), (1, 11)):
            snap_p = tmp_path / f"mid_m{member_idx}_s{step_k}.restart.nc"
            snap_p.write_bytes(path.read_bytes())
            midrun_snapshots[(member_idx, step_k)] = snap_p

    monkeypatch.setattr(ens_mod, "_atomic_save_member_restart_nc", _spy_save)

    ens_out = tmp_path / "nsave_ens_out"
    ens_res = run_ensemble([d0, d1], output_dir=ens_out, precision="fp64", save_restart=True)
    assert ens_res["batch_size"] == 2

    # Member 0 (nwrite=10, nsave=4, nstep=13): periodic saves at (c-1)%4==0 -> [1, 5, 9, 13]
    assert saved_steps_by_member[0] == [1, 5, 9, 13]
    # Member 1 (nwrite=3, nsave=5, nstep=13): periodic saves at [1, 6, 11] + final save at 13
    assert saved_steps_by_member[1] == [1, 6, 11, 13]

    ref_out = tmp_path / "nsave_ref_out"
    ref_out.mkdir(parents=True, exist_ok=True)
    for b, (d_path, mid_step) in enumerate([(d0, 9), (d1, 11)]):
        stem = d_path.stem
        _, grids_b, _ = init_geometry(load_config(d_path))

        # Verify mid-run periodic .restart.nc (at step 9 / step 11) matches standalone CLI
        ref_mid_rest = ref_out / f"{stem}_s{mid_step}.restart.nc"
        run_cli_simulation(
            d_path,
            output_path=ref_out / f"{stem}_s{mid_step}.out.nc",
            nstep=mid_step,
            precision="fp64",
            save_restart_path=ref_mid_rest,
        )
        g_em, t_em = load_restart_nc(midrun_snapshots[(b, mid_step)], grids_b, dtype=jnp.complex128)
        g_rm, t_rm = load_restart_nc(ref_mid_rest, grids_b, dtype=jnp.complex128)
        _assert_close_tol(t_em, t_rm, 1.0e-12, f"{stem}:mid_restart_t")
        _assert_close_tol(g_em, g_rm, 1.0e-6, f"{stem}:mid_restart_G")

        ref_nc = ref_out / f"{stem}.out.nc"
        ref_rest = ref_out / f"{stem}.restart.nc"
        run_cli_simulation(
            d_path,
            output_path=ref_nc,
            precision="fp64",
            save_restart_path=ref_rest,
        )
        g_e, t_e = load_restart_nc(
            ens_res["members"][b]["restart_nc"], grids_b, dtype=jnp.complex128
        )
        g_r, t_r = load_restart_nc(ref_rest, grids_b, dtype=jnp.complex128)
        _assert_close_tol(t_e, t_r, 1.0e-12, f"{stem}:restart_t")
        _assert_close_tol(g_e, g_r, 1.0e-6, f"{stem}:restart_G")
        with (
            nc.Dataset(ens_res["members"][b]["output_nc"], "r") as ds_e,
            nc.Dataset(ref_nc, "r") as ds_r,
        ):
            _assert_close_tol(
                ds_e.groups["Grids"].variables["time"][:],
                ds_r.groups["Grids"].variables["time"][:],
                1.0e-12,
                f"{stem}:nc_time",
            )
