"""Zero-Sync Fused On-Device Stepping & Async Host NetCDF4 I/O."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

from gx_jax.cli import AsyncDiskIO, build_arg_parser, run_cli_simulation
from gx_jax.cli import main as cli_main
from gx_jax.config import GXConfig, load_config
from gx_jax.diagnostics_nc import (
    DiagnosticsSnapshot,
    DiagnosticsState,
    DiagnosticsWriter,
    init_diagnostics_state,
)
from gx_jax.fields import init_moments, load_restart_nc
from gx_jax.geometry_nc import init_geometry
from gx_jax.integrator import (
    ChunkResult,
    GyrokineticOperator,
    IntegratorState,
    advance_integrator_chunk,
    init_gyrokinetic_operator,
    init_integrator_state,
    make_chunk_stepper,
    step_integrator_state,
)
from gx_jax.shard_tpu import (
    advance_sharded_integrator_chunk,
    create_device_mesh,
    gather_g_state,
    init_sharded_integrator_state,
    make_sharded_chunk_stepper,
    run_sharded_simulation,
    sharded_compute_diagnostics,
    step_sharded_integrator_state,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def _enable_x64() -> None:
    jax.config.update("jax_enable_x64", True)


def _write_test_deck(
    path: Path,
    *,
    em: bool = False,
    nonlinear: bool = False,
    scheme: str = "rk3",
    fixed_dt: bool = True,
    dt: float = 0.05,
    dt_max: float | None = None,
    nstep: int | None = None,
    t_max: float | None = None,
    nwrite: int = 1,
    nsave: int = 10_000,
    save_for_restart: bool = False,
    init_amp: float = 1.0e-5,
    mass_e: float = 0.02,
    geo_option: str = "miller",
    write_fields: bool = False,
    write_moms: bool = False,
    exb_shear: bool = False,
    forcing_init: bool = False,
) -> None:
    """Write a compact Miller or s-alpha gyrokinetic deck for fast fused-loop tests."""
    nspecies = 2 if em else 1
    beta = 0.015 if em else 0.0
    fapar = 1.0 if em else 0.0
    fbpar = 1.0 if em else 0.0
    if nspecies == 2:
        species_block = f"""
[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, {mass_e} ]
 dens = [ 1.0, 1.0 ]
 temp = [ 1.0, 1.0 ]
 tprim = [ 2.49, 2.49 ]
 fprim = [ 0.8, 0.8 ]
 vnewk = [ 0.01, 0.01 ]
 type = [ "ion", "electron" ]

[Boltzmann]
 add_Boltzmann_species = false
"""
    else:
        species_block = """
[species]
 z = [ 1.0 ]
 mass = [ 1.0 ]
 dens = [ 1.0 ]
 temp = [ 1.0 ]
 tprim = [ 2.49 ]
 fprim = [ 0.8 ]
 vnewk = [ 0.01 ]
 type = [ "ion" ]

[Boltzmann]
 add_Boltzmann_species = true
 Boltzmann_type = "electrons"
 tau_fac = 1.0
"""

    dt_max_val = float(dt_max) if dt_max is not None else max(dt, 0.05)
    time_lines = [
        f" dt = {dt}",
        f" dt_max = {dt_max_val}",
        " dt_min = 1.0e-7",
        " cfl = 0.9",
        f' scheme = "{scheme}"',
        f" fixed_dt = {str(fixed_dt).lower()}",
    ]
    if nstep is not None:
        time_lines.append(f" nstep = {int(nstep)}")
    if t_max is not None:
        time_lines.append(f" t_max = {float(t_max)}")

    extra_phys = ""
    if exb_shear:
        extra_phys += "\n exb_shear = true\n g_exb = 0.12"
    extra_forcing = ""
    if forcing_init:
        extra_forcing = (
            '\n[Forcing]\n forcing_init = true\n forcing_type = "Kz"\n'
            " forcing_amp = 0.02\n forcing_index = 1"
        )

    geo_block = (
        """[Geometry]
 geo_option = "s-alpha"
 eps = 0.167
 qinp = 1.4
 shat = 0.8"""
        if geo_option == "s-alpha"
        else """[Geometry]
 geo_option = "miller"
 rhoc = 0.5
 Rmaj = 2.77778
 R_geo = 2.77778
 qinp = 1.4
 shat = 0.8
 shift = 0.0
 akappa = 1.0
 akappri = 0.0
 tri = 0.0
 tripri = 0.0
 betaprim = 0.0"""
    )

    deck = f"""
[Dimensions]
 ntheta = 8
 nperiod = 1
 ny = 6
 nx = 6
 nhermite = 4
 nlaguerre = 2
 nspecies = {nspecies}

[Domain]
 y0 = 10.0
 boundary = "linked"

[Physics]
 beta = {beta}
 nonlinear_mode = {str(nonlinear).lower()}
 fphi = 1.0
 fapar = {fapar}
 fbpar = {fbpar}{extra_phys}

[Time]
{chr(10).join(time_lines)}

[Initialization]
 ikpar_init = 0
 init_field = "density"
 init_amp = {init_amp}

{geo_block}
{species_block}
[Dissipation]
 closure_model = "none"
 hypercollisions = true
 nu_hyper_m = 0.1
 p_hyper_m = 6
 nu_hyper_l = 0.1
 p_hyper_l = 6
 hyper = true
 D_hyper = 0.05

[Diagnostics]
 nwrite = {nwrite}
 omega = true
 free_energy = true
 fluxes = true
 fields = {str(write_fields).lower()}
 moments = {str(write_moms).lower()}

[Restart]
 save_for_restart = {str(save_for_restart).lower()}
 nsave = {nsave}{extra_forcing}
"""
    path.write_text(deck.strip() + "\n", encoding="utf-8")


def _reference_session6_cli_loop(
    input_file: Path,
    out_nc: Path,
    *,
    dt: float | None = None,
    nstep: int | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    scheme: str | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Step-by-step Session 6 CLI reference loop matching C++ Diagnostics_GK::loop verbatim."""
    use_fp64 = precision.lower() != "fp32"
    r_dtype = jnp.float64 if use_fp64 else jnp.float32
    c_dtype = jnp.complex128 if use_fp64 else jnp.complex64

    cfg = load_config(input_file)
    ov: dict[str, Any] = {}
    if nstep is not None:
        ov["nstep"] = int(nstep)
    if dt is not None:
        ov["dt"] = float(dt)
    if t_max is not None:
        ov["t_max"] = float(t_max)
    if nwrite is not None:
        ov["nwrite"] = int(nwrite)
    if scheme is not None:
        ov["scheme"] = str(scheme).lower()
    if ov:
        cfg = replace(cfg, **ov)

    dt_val = float(cfg.dt) if cfg.dt > 0.0 else 0.05
    nstep_max = max(0, int(cfg.nstep))
    t_max_val = float(cfg.t_max)
    nwrite_val = max(1, int(cfg.nwrite))
    eps_t = 1.0e-11 * max(abs(t_max_val), 1.0e-30)

    cfg, grids, geo = init_geometry(cfg)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=r_dtype)
    solver = gk_op.linear_op.solver
    g_init = init_moments(cfg, grids, geo, dtype=c_dtype)
    dt_init = float(dt) if dt is not None else (dt_val if cfg.fixed_dt else None)

    state = init_integrator_state(g_init, gk_op, time=0.0, dt=dt_init)
    diag_state = init_diagnostics_state(state.G, state.fields, time=state.time, has_old=False)
    step_fn = jax.jit(
        lambda st_a: step_integrator_state(st_a, gk_op, dt_override=dt, scheme=cfg.scheme)
    )

    def _should_write(c: int, cur_t: float) -> bool:
        return (c >= 1 and (c - 1) % nwrite_val == 0) or (cur_t > t_max_val + eps_t)

    def _should_save_old(c: int, cur_t: float, cur_dt: float) -> bool:
        return (c % nwrite_val == 0) or (cur_t + cur_dt > t_max_val - eps_t)

    counter = 0
    with DiagnosticsWriter(out_nc, cfg, grids, geo) as writer:
        while counter < nstep_max and float(state.time) < t_max_val - eps_t:
            cur_t = float(state.time)
            cur_dt = float(state.dt)
            if _should_write(counter, cur_t):
                _, diag_state = writer.write_step(
                    state.G, state.fields, diag_state, solver, time=state.time, dt=state.dt
                )
            if _should_save_old(counter, cur_t, cur_dt):
                diag_state = init_diagnostics_state(
                    state.G, state.fields, time=state.time, has_old=True
                )
            state = step_fn(state)
            counter += 1
            end_t = float(state.time)
            if counter == nstep_max or end_t >= t_max_val - eps_t:
                if _should_write(counter, end_t):
                    _, diag_state = writer.write_step(
                        state.G, state.fields, diag_state, solver, time=state.time, dt=state.dt
                    )

    return {"g_final": state.G, "t_final": float(state.time), "nstep": counter}


def _assert_out_nc_equal(
    cand_nc: Path,
    ref_nc: Path,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Assert exact schema and variable equality between two `.out.nc` files."""
    with nc.Dataset(ref_nc, "r") as rds, nc.Dataset(cand_nc, "r") as cds:
        np.testing.assert_allclose(
            np.asarray(cds.groups["Grids"].variables["time"][:], dtype=np.float64),
            np.asarray(rds.groups["Grids"].variables["time"][:], dtype=np.float64),
            rtol=rtol,
            atol=atol,
        )
        r_diag = rds.groups["Diagnostics"].variables
        c_diag = cds.groups["Diagnostics"].variables
        assert set(c_diag.keys()) == set(r_diag.keys())
        for name in r_diag:
            cand_arr = np.asarray(c_diag[name][:], dtype=np.float64)
            ref_arr = np.asarray(r_diag[name][:], dtype=np.float64)
            assert np.all(np.isfinite(cand_arr)), f"Non-finite Diagnostics/{name}"
            assert np.all(np.isfinite(ref_arr)), f"Non-finite ref Diagnostics/{name}"
            np.testing.assert_allclose(
                cand_arr,
                ref_arr,
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch in Diagnostics/{name}",
            )


def _build_kbm_operator(
    *,
    scheme: str = "rk3",
    fixed_dt: bool = True,
    dt: float = 1.0e-3,
    t_max: float = 1.0e20,
    nonlinear: bool = True,
    exb_shear: bool = False,
    forcing_init: bool = False,
    precision: str = "fp64",
) -> tuple[GXConfig, Any, Any, GyrokineticOperator, jax.Array]:
    kbm_in = REPO_ROOT / "benchmarks" / "linear" / "KBM" / "kbm_miller.in"
    c_dtype = jnp.complex128 if precision == "fp64" else jnp.complex64
    r_dtype = jnp.float64 if precision == "fp64" else jnp.float32
    cfg = replace(
        load_config(kbm_in),
        nx=4,
        ny=4,
        nz=8,
        ntheta_in=8,
        nperiod=1,
        zp=1,
        nhermite=4,
        nlaguerre=2,
        nonlinear_mode=nonlinear,
        linear=not nonlinear,
        scheme=scheme,
        fixed_dt=fixed_dt,
        dt=dt,
        dt_max=max(dt, 0.05),
        t_max=t_max,
        exb_shear=exb_shear,
        g_exb=0.12 if exb_shear else 0.0,
        forcing_init=forcing_init,
        forcing_amp=0.02 if forcing_init else 0.0,
        forcing_index=1,
    )
    cfg, grids, geo = init_geometry(cfg)
    gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=r_dtype)
    g0 = init_moments(cfg, grids, geo, dtype=c_dtype)
    return cfg, grids, geo, gk_op, g0


# =============================================================================
# 1. ChunkResult & On-Device Chunk Steppers (while_loop + scan + trace_count)
# =============================================================================


def test_chunk_stepper_pytree_while_and_scan_parity_and_trace_count() -> None:
    """Verify ChunkResult PyTree, while/scan parity, early t_max, penultimate state, and trace=1."""
    # 1a. Fixed-dt scan fast path vs step-by-step reference (with ExB + Forcing + EM)
    _cfg, _grids, _geo, gk_op, g0 = _build_kbm_operator(
        scheme="rk3",
        fixed_dt=True,
        dt=5.0e-4,
        nonlinear=True,
        exb_shear=True,
        forcing_init=True,
        precision="fp64",
    )
    st0 = init_integrator_state(g0, gk_op, time=0.0, dt=5.0e-4)

    # Step-by-step reference for 10 steps (using compiled step_integrator_state)
    ref_step_fn = jax.jit(
        lambda s: step_integrator_state(s, gk_op, dt_override=5.0e-4, scheme="rk3")
    )
    ref_states: list[IntegratorState] = [st0]
    st_ref = st0
    for _ in range(10):
        st_ref = ref_step_fn(st_ref)
        ref_states.append(st_ref)

    stepper = make_chunk_stepper(gk_op, dt_override=5.0e-4, scheme="rk3", fixed_chunk_len=4)
    # Irregular initial (1), steady scan chunks (4, 4), irregular tail (1)
    c1 = stepper(st0, 1)
    c2 = stepper(c1.state, 4)
    c3 = stepper(c2.state, 4)
    c4 = stepper(c3.state, 1)

    assert stepper.trace_counts["scan"] == 1
    assert stepper.trace_counts["while"] == 1
    assert int(c1.steps_done) == 1
    assert int(c2.steps_done) == 4
    assert int(c3.steps_done) == 4
    assert int(c4.steps_done) == 1

    # Verify ChunkResult PyTree round-trip & exact penultimate state after c3 (step 9 -> prev=8)
    leaves, treedef = jax.tree_util.tree_flatten(c3)
    c3_rt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(c3_rt, ChunkResult)
    np.testing.assert_allclose(np.asarray(c3_rt.state.G), np.asarray(ref_states[9].G), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(c3_rt.G_prev), np.asarray(ref_states[8].G), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(c3_rt.fields_prev.phi), np.asarray(ref_states[8].fields.phi), rtol=1e-12
    )
    assert float(c3_rt.time_prev) == pytest.approx(float(ref_states[8].time), rel=0.0, abs=1e-15)
    np.testing.assert_allclose(np.asarray(c4.state.G), np.asarray(ref_states[10].G), rtol=1e-12)

    # 1b. Verify pure scan StableHLO lowers with zero stablehlo.while when unbounded / safe_for_scan
    stepper_unbounded = make_chunk_stepper(
        gk_op, dt_override=5.0e-4, scheme="rk3", t_max=math.inf, fixed_chunk_len=4
    )
    res_scan = stepper_unbounded(st0, 4)
    np.testing.assert_allclose(
        np.asarray(res_scan.state.G), np.asarray(ref_states[4].G), rtol=1e-12
    )
    assert stepper_unbounded.trace_counts == {"while": 0, "scan": 1}

    # Check direct advance_integrator_chunk (while_loop + use_scan=True)
    res_adv_while = advance_integrator_chunk(st0, gk_op, 4, dt_override=5.0e-4, scheme="rk3")
    res_adv_scan = advance_integrator_chunk(
        st0, gk_op, 4, dt_override=5.0e-4, scheme="rk3", use_scan=True
    )
    np.testing.assert_allclose(
        np.asarray(res_adv_while.state.G), np.asarray(ref_states[4].G), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(res_adv_scan.state.G), np.asarray(ref_states[4].G), rtol=1e-12, atol=1e-14
    )

    # Verify JAXPR of pure scan path uses `scan` primitive and zero `while` primitives
    jaxpr_scan = jax.make_jaxpr(
        lambda s: advance_integrator_chunk(
            s, gk_op, 4, dt_override=5.0e-4, scheme="rk3", t_max=math.inf, use_scan=True
        )
    )(st0)
    prim_names = [eqn.primitive.name for eqn in jaxpr_scan.jaxpr.eqns]
    assert "scan" in prim_names
    assert "while" not in prim_names

    # 1c. Adaptive variable-dt (sspx2 & rk3 fixed_dt=False) with mid-chunk t_max & trace=1
    for sk in ("sspx2", "rk3"):
        _cfg_a, _gr_a, _geo_a, gk_ad, g0_ad = _build_kbm_operator(
            scheme=sk, fixed_dt=False, dt=0.05, nonlinear=True, precision="fp64"
        )
        # Boost amplitude so nonlinear CFL varies dt per step
        st_ad0 = init_integrator_state(100.0 * g0_ad, gk_ad, time=0.0, dt=None)
        ref_ad_fn = jax.jit(
            lambda s, op_in=gk_ad, sk_in=sk: step_integrator_state(s, op_in, scheme=sk_in)
        )
        st_ref_ad = st_ad0
        ad_hist: list[IntegratorState] = [st_ad0]
        for _ in range(6):
            st_ref_ad = ref_ad_fn(st_ref_ad)
            ad_hist.append(st_ref_ad)
        # Choose t_max strictly between step 3 and step 4 so chunk of max_steps=6 stops at 4
        t_cut = 0.5 * (float(ad_hist[3].time) + float(ad_hist[4].time))
        step_ad = make_chunk_stepper(gk_ad, scheme=sk, t_max=t_cut, fixed_chunk_len=4)
        assert step_ad.is_fixed_dt is False

        # Varying max_steps must keep trace_count == 1
        r_a = step_ad(st_ad0, 2)
        r_b = step_ad(r_a.state, 5)
        assert step_ad.trace_counts["while"] == 1
        assert step_ad.trace_counts["scan"] == 0
        assert int(r_a.steps_done) == 2
        assert int(r_b.steps_done) == 2  # stopped at total step 4!
        np.testing.assert_allclose(np.asarray(r_b.state.G), np.asarray(ad_hist[4].G), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(r_b.G_prev), np.asarray(ad_hist[3].G), rtol=1e-12)
        assert float(r_b.time_prev) == pytest.approx(float(ad_hist[3].time), abs=1e-15)

        # Edge cases: max_steps == 0 and already past t_max
        r_zero = step_ad(st_ad0, 0)
        assert int(r_zero.steps_done) == 0
        np.testing.assert_allclose(np.asarray(r_zero.state.G), np.asarray(st_ad0.G), atol=0.0)
        r_past = step_ad(r_b.state, 3)
        assert int(r_past.steps_done) == 0


# =============================================================================
# 2. Exact Session 6 CLI & C++ Cadence Parity (async_io True/False, Adaptive CFL, nsave)
# =============================================================================


@pytest.mark.parametrize(("nwrite", "precision"), [(1, "fp64"), (3, "fp32"), (4, "fp64")])
def test_cli_fixed_dt_scan_vs_session6_reference_parity(
    tmp_path: Path, nwrite: int, precision: str
) -> None:
    """Verify exact bit/FP parity of fused scan/while CLI (async True/False) vs Session 6."""
    deck = tmp_path / f"fixed_nw{nwrite}_{precision}.in"
    _write_test_deck(
        deck,
        em=True,
        nonlinear=True,
        scheme="rk3",
        fixed_dt=True,
        dt=5.0e-4,
        nstep=8,
        nwrite=nwrite,
        save_for_restart=True,
        nsave=2,
        init_amp=1.0e-4,
        mass_e=1.0,
        geo_option="s-alpha",
        write_fields=True,
        write_moms=True,
        exb_shear=True,
        forcing_init=True,
    )

    ref_nc = tmp_path / "ref.out.nc"
    ref_res = _reference_session6_cli_loop(deck, ref_nc, precision=precision)
    tol_g = 1e-12 if precision == "fp64" else 1e-6
    tol_nc = 1e-6 if precision == "fp64" else 2e-5

    for async_io in (False, True):
        cand_nc = tmp_path / f"cand_async_{async_io}.out.nc"
        restart_nc = tmp_path / f"cand_async_{async_io}.restart.nc"
        res = run_cli_simulation(
            deck,
            output_path=cand_nc,
            precision=precision,
            save_restart_path=restart_nc,
            async_io=async_io,
        )
        assert res["nstep"] == ref_res["nstep"] == 8
        assert res["t_final"] == pytest.approx(ref_res["t_final"], rel=0.0, abs=1e-15)
        # Default use_scan=False while_loop is bit-identical to Session 6 step-by-step in fp64.
        g_rtol = 0.0 if precision == "fp64" else tol_g
        g_atol = 0.0 if precision == "fp64" else 1e-12
        # In fp32, time-derivative diagnostics (1/dt ~ 2e3) amplify ULP differences to ~2e-4.
        nc_rtol = 0.0 if precision == "fp64" else max(tol_nc, 3e-4)
        nc_atol = 0.0 if precision == "fp64" else 1e-12
        np.testing.assert_allclose(
            np.asarray(res["g_final"]), np.asarray(ref_res["g_final"]), rtol=g_rtol, atol=g_atol
        )
        _assert_out_nc_equal(cand_nc, ref_nc, rtol=nc_rtol, atol=nc_atol)
        assert restart_nc.is_file()
        assert list(tmp_path.glob("*.tmp")) == []

    # Also verify opt-in use_scan=True (--use-scan) is FP-identical on steady blocks
    if nwrite in (3, 4):
        scan_nc = tmp_path / "cand_use_scan.out.nc"
        scan_res = run_cli_simulation(
            deck,
            output_path=scan_nc,
            precision=precision,
            async_io=True,
            use_scan=True,
        )
        assert scan_res["nstep"] == ref_res["nstep"] == 8
        assert scan_res["t_final"] == pytest.approx(ref_res["t_final"], rel=0.0, abs=1e-15)
        np.testing.assert_allclose(
            np.asarray(scan_res["g_final"]),
            np.asarray(ref_res["g_final"]),
            rtol=tol_g,
            atol=1e-13 if precision == "fp64" else 1e-7,
        )
        scan_nc_rtol = tol_nc if precision == "fp64" else max(tol_nc, 3e-4)
        _assert_out_nc_equal(
            scan_nc, ref_nc, rtol=scan_nc_rtol, atol=1e-13 if precision == "fp64" else 1e-8
        )


@pytest.mark.parametrize(("scheme", "use_em"), [("rk3", False), ("sspx2", False), ("k10", True)])
def test_cli_adaptive_cfl_mid_chunk_and_exact_t_max_parity(
    tmp_path: Path, scheme: str, use_em: bool
) -> None:
    """Verify adaptive variable-dt mid-chunk t_max cutoff & exact landing match Session 6."""
    deck = tmp_path / f"adapt_{scheme}.in"
    # Choose large bounded amplitude so nonlinear CFL dt varies smoothly per step without blowup
    _write_test_deck(
        deck,
        em=use_em,
        nonlinear=True,
        scheme=scheme,
        fixed_dt=False,
        dt=0.5,
        dt_max=0.5,
        nstep=20,
        nwrite=3,
        init_amp=80.0 if use_em else 150.0,
        mass_e=1.0,
        write_fields=True,
        write_moms=True,
    )
    # Probe exact step times with nstep=5 to pick a mid-chunk t_max (between step 4 and 5)
    probe_nc = tmp_path / "probe.out.nc"
    _reference_session6_cli_loop(deck, probe_nc, nstep=5, nwrite=1, precision="fp64")
    with nc.Dataset(probe_nc, "r") as ds:
        step_times = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
    assert len(step_times) == 5
    # Verify dt actually varied over steps!
    step_dts = np.diff(np.concatenate([[0.0], step_times]))
    assert np.max(step_dts) > np.min(step_dts) * 1.0001

    t_mid_cut = 0.5 * (float(step_times[3]) + float(step_times[4]))
    t_exact_land = float(step_times[4])

    for label, t_target in [("mid_cut", t_mid_cut), ("exact_land", t_exact_land)]:
        ref_nc = tmp_path / f"ref_{scheme}_{label}.out.nc"
        cand_nc = tmp_path / f"cand_{scheme}_{label}.out.nc"
        ref_res = _reference_session6_cli_loop(
            deck, ref_nc, nstep=20, t_max=t_target, nwrite=3, precision="fp64"
        )
        cand_res = run_cli_simulation(
            deck,
            output_path=cand_nc,
            nstep=20,
            t_max=t_target,
            nwrite=3,
            precision="fp64",
            async_io=True,
        )
        assert cand_res["nstep"] == ref_res["nstep"]
        assert cand_res["t_final"] == pytest.approx(ref_res["t_final"], rel=0.0, abs=1e-15)
        np.testing.assert_allclose(
            np.asarray(cand_res["g_final"]), np.asarray(ref_res["g_final"]), rtol=1e-12, atol=1e-13
        )
        # F32 NetCDF storage + XLA FMA cancellation residual (~4e-11 vs |phi|^2/dt ~ 2e4)
        _assert_out_nc_equal(cand_nc, ref_nc, rtol=1e-6, atol=1e-9)


def test_cli_zero_per_step_sync_and_single_trace_across_nwrite_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify cli.py invokes compiled chunk stepper once per event block (zero per-step syncs)."""
    import gx_jax.cli as cli_mod

    deck = tmp_path / "sync_check.in"
    _write_test_deck(
        deck,
        em=False,
        nonlinear=False,
        scheme="rk3",
        fixed_dt=True,
        dt=0.01,
        nstep=16,
        nwrite=5,
        save_for_restart=False,
    )

    created_steppers: list[Any] = []
    orig_make_stepper = cli_mod.make_chunk_stepper

    def _spy_make_stepper(*args: Any, **kwargs: Any) -> Any:
        st = orig_make_stepper(*args, **kwargs)
        calls: list[int] = []

        def _wrapped(state: Any, max_steps: int, **kw: Any) -> Any:
            calls.append(int(max_steps))
            return st(state, max_steps, **kw)

        _wrapped.trace_counts = st.trace_counts  # type: ignore[attr-defined]
        _wrapped.calls = calls  # type: ignore[attr-defined]
        created_steppers.append(_wrapped)
        return _wrapped

    monkeypatch.setattr(cli_mod, "make_chunk_stepper", _spy_make_stepper)

    out_nc = tmp_path / "sync_check.out.nc"
    res = run_cli_simulation(deck, output_path=out_nc, precision="fp32", async_io=True)
    assert res["nstep"] == 16
    assert len(created_steppers) == 1
    sp = created_steppers[0]
    # Default use_scan=False: nstep=16, nwrite=5 -> chunks: [1, 5, 5, 5], single while trace
    assert sp.calls == [1, 5, 5, 5]
    assert sp.trace_counts["while"] == 1
    assert sp.trace_counts["scan"] == 0

    # Opt-in use_scan=True: initial irregular chunk [1] on while + steady chunks [5, 5, 5] on scan
    out_scan_nc = tmp_path / "sync_check_scan.out.nc"
    res_scan = run_cli_simulation(
        deck, output_path=out_scan_nc, precision="fp32", async_io=True, use_scan=True
    )
    assert res_scan["nstep"] == 16
    assert len(created_steppers) == 2
    sp_scan = created_steppers[1]
    assert sp_scan.calls == [1, 5, 5, 5]
    assert sp_scan.trace_counts["while"] == 1
    assert sp_scan.trace_counts["scan"] == 1


def test_cli_periodic_nsave_and_warm_append_restart_parity(tmp_path: Path) -> None:
    """Verify periodic nsave + warm append_on_restart continuation matches single continuous run."""
    deck = tmp_path / "warm.in"
    _write_test_deck(
        deck,
        em=False,
        nonlinear=False,
        scheme="rk3",
        fixed_dt=True,
        dt=0.05,
        t_max=7 * 0.05,
        nwrite=3,
        save_for_restart=True,
        nsave=2,
        write_fields=True,
        write_moms=True,
    )
    out_nc = tmp_path / "warm.out.nc"
    restart_nc = tmp_path / "warm.restart.nc"
    res1 = run_cli_simulation(
        deck, output_path=out_nc, save_restart_path=restart_nc, precision="fp64", async_io=True
    )
    assert res1["nstep"] == 7
    with nc.Dataset(out_nc, "r") as ds:
        np.testing.assert_allclose(
            ds.groups["Grids"].variables["time"][:],
            np.array([1.0, 4.0, 7.0]) * 0.05,
            rtol=1e-12,
        )
    _, t_r = load_restart_nc(restart_nc, init_geometry(load_config(deck))[1], dtype=jnp.complex128)
    assert t_r == pytest.approx(7.0 * 0.05, rel=1e-6)

    # Continue with append_on_restart=True
    deck.write_text(
        deck.read_text().replace(f"t_max = {7 * 0.05}", f"t_max = {11 * 0.05}")
        + "\n restart = true\n append_on_restart = true\n"
    )
    res2 = run_cli_simulation(deck, output_path=out_nc, precision="fp64", async_io=True)
    assert res2["nstep"] == 4
    with nc.Dataset(out_nc, "r") as ds2:
        np.testing.assert_allclose(
            ds2.groups["Grids"].variables["time"][:],
            np.array([1.0, 4.0, 7.0, 8.0, 11.0]) * 0.05,
            rtol=1e-6,
        )


# =============================================================================
# 3. AsyncDiskIO Thread Safety, Overlap, Flush/Join & Exception Propagation
# =============================================================================


def test_async_disk_io_fifo_single_worker_overlap_and_exception_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify AsyncDiskIO single-worker serialization, clean flush, and error propagation."""
    # 3a. Direct AsyncDiskIO unit check: FIFO, max concurrent == 1, runs off main thread
    main_tid = threading.get_ident()
    active_workers = 0
    max_concurrent = 0
    worker_tids: set[int] = set()
    order: list[int] = []
    lock = threading.Lock()

    def _task(idx: int) -> None:
        nonlocal active_workers, max_concurrent
        with lock:
            active_workers += 1
            max_concurrent = max(max_concurrent, active_workers)
            worker_tids.add(threading.get_ident())
        time.sleep(0.01)
        with lock:
            order.append(idx)
            active_workers -= 1

    with AsyncDiskIO(enabled=True, max_pending=2) as io:
        for i in range(5):
            io.submit(_task, i)
    assert order == [0, 1, 2, 3, 4]
    assert max_concurrent == 1
    assert len(worker_tids) == 1
    assert main_tid not in worker_tids

    # 3b. Exception propagation + remaining task drain before DiagnosticsWriter.close()
    events: list[str] = []

    class _BoomError(RuntimeError):
        pass

    with pytest.raises(_BoomError, match="disk full"):
        with AsyncDiskIO(enabled=True) as io_err:
            io_err.submit(lambda: events.append("ok1"))

            def _fail() -> None:
                events.append("fail")
                raise _BoomError("disk full")

            io_err.submit(_fail)
            io_err.submit(lambda: (time.sleep(0.02), events.append("drained_after_fail")))
    assert events == ["ok1", "fail", "drained_after_fail"]

    # 3c. End-to-end run_cli_simulation write error propagation & writer.close() ordering
    deck = tmp_path / "io_err.in"
    _write_test_deck(deck, nstep=4, nwrite=1, fixed_dt=True, dt=0.01)
    call_log: list[str] = []
    orig_write = DiagnosticsWriter.write_snapshot
    orig_close = DiagnosticsWriter.close
    write_count = 0

    def _failing_write(self: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal write_count
        write_count += 1
        call_log.append(f"write_{write_count}")
        if write_count == 2:
            raise OSError("simulated netcdf4 write failure")
        return orig_write(self, *args, **kwargs)

    def _tracked_close(self: Any) -> None:
        call_log.append("close")
        orig_close(self)

    monkeypatch.setattr(DiagnosticsWriter, "write_snapshot", _failing_write)
    monkeypatch.setattr(DiagnosticsWriter, "close", _tracked_close)

    with pytest.raises(OSError, match="simulated netcdf4 write failure"):
        run_cli_simulation(deck, output_path=tmp_path / "io_err.out.nc", async_io=True)

    # AsyncDiskIO must flush & join the failing worker task BEFORE DiagnosticsWriter.close() runs!
    assert "write_2" in call_log
    assert call_log[-1] == "close"
    assert call_log.index("write_2") < call_log.index("close")


# =============================================================================
# 4. Sharded Chunk Steppers, run_sharded_simulation, & 4-Device Subprocess Parity
# =============================================================================


def _reference_session6_run_sharded(
    g_init: jax.Array,
    gk_op: GyrokineticOperator,
    mesh: Any,
    *,
    nsteps: int,
    dt: float | None = None,
    scheme: str | None = None,
    nwrite: int = 0,
) -> tuple[IntegratorState, list[DiagnosticsSnapshot]]:
    """Step-by-step Session 6 run_sharded_simulation reference."""
    state = init_sharded_integrator_state(g_init, gk_op, mesh, dt=dt)
    diag_state = init_diagnostics_state(state.G, state.fields, time=0.0, has_old=False)
    snapshots: list[DiagnosticsSnapshot] = []
    step_fn = jax.jit(
        lambda st: step_sharded_integrator_state(st, gk_op, mesh, dt_override=dt, scheme=scheme)
    )
    for counter in range(1, nsteps + 1):
        prev_G = state.G
        prev_fields = state.fields
        state = step_fn(state)
        if nwrite > 0 and (counter % nwrite == 1 or counter == nsteps or nwrite == 1):
            if counter > 1:
                diag_state = DiagnosticsState(
                    G_old=prev_G,
                    fields_old=prev_fields,
                    time_old=state.time - state.dt,
                    has_old=jnp.asarray(True, dtype=jnp.bool_),
                )
            snap, diag_state = sharded_compute_diagnostics(
                state.G,
                state.fields,
                diag_state,
                gk_op.grids,
                gk_op.geo,
                gk_op.linear_op.solver,
                mesh,
                time=state.time,
                dt=state.dt,
                beta=gk_op.linear_op.solver.beta,
            )
            snapshots.append(snap)
    return state, snapshots


def test_run_sharded_simulation_single_device_parity() -> None:
    """Verify run_sharded_simulation matches Session 6 step-by-step across nwrite/nsteps."""
    cfg, _gr, _geo, gk_op, g0 = _build_kbm_operator(
        scheme="rk3", fixed_dt=True, dt=5.0e-4, nonlinear=True, precision="fp64"
    )
    mesh = create_device_mesh((1, 1), nspecies=cfg.nspecies, nhermite=cfg.nhermite)
    for nwrite, nsteps in [(0, 4), (1, 4), (3, 5)]:
        st_ref, snaps_ref = _reference_session6_run_sharded(
            g0, gk_op, mesh, nsteps=nsteps, dt=5.0e-4, scheme="rk3", nwrite=nwrite
        )
        # Default use_scan=False is bit-identical to Session 6 step-by-step
        st_new, snaps_new = run_sharded_simulation(
            g0, gk_op, mesh, nsteps=nsteps, dt=5.0e-4, scheme="rk3", nwrite=nwrite
        )
        np.testing.assert_allclose(
            np.asarray(gather_g_state(st_new.G, mesh)),
            np.asarray(gather_g_state(st_ref.G, mesh)),
            rtol=0.0,
            atol=0.0,
        )
        assert len(snaps_new) == len(snaps_ref)
        for s_new, s_ref in zip(snaps_new, snaps_ref, strict=True):
            for l_new, l_ref in zip(
                jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(s_ref), strict=True
            ):
                np.testing.assert_allclose(np.asarray(l_new), np.asarray(l_ref), rtol=0.0, atol=0.0)
        if nwrite == 3:
            # Opt-in use_scan=True is FP-identical (< 1e-11)
            st_scan, snaps_scan = run_sharded_simulation(
                g0,
                gk_op,
                mesh,
                nsteps=nsteps,
                dt=5.0e-4,
                scheme="rk3",
                nwrite=nwrite,
                use_scan=True,
            )
            np.testing.assert_allclose(
                np.asarray(gather_g_state(st_scan.G, mesh)),
                np.asarray(gather_g_state(st_ref.G, mesh)),
                rtol=1e-12,
                atol=1e-14,
            )
            assert len(snaps_scan) == len(snaps_ref)
            for s_sc, s_ref in zip(snaps_scan, snaps_ref, strict=True):
                for l_sc, l_ref in zip(
                    jax.tree_util.tree_leaves(s_sc), jax.tree_util.tree_leaves(s_ref), strict=True
                ):
                    np.testing.assert_allclose(
                        np.asarray(l_sc), np.asarray(l_ref), rtol=1e-11, atol=1e-13
                    )

    # Also check advance_sharded_integrator_chunk and make_sharded_chunk_stepper trace_count
    sh_stepper = make_sharded_chunk_stepper(
        gk_op, mesh, dt_override=5.0e-4, scheme="rk3", fixed_chunk_len=3, t_max=math.inf
    )
    sh0 = init_sharded_integrator_state(g0, gk_op, mesh, dt=5.0e-4)
    sh_c1 = sh_stepper(sh0, 3)
    sh_c2 = sh_stepper(sh_c1.state, 3)
    sh_adv = advance_sharded_integrator_chunk(
        sh0, gk_op, mesh, 3, dt_override=5.0e-4, scheme="rk3", t_max=math.inf, use_scan=True
    )
    assert sh_stepper.trace_counts == {"while": 0, "scan": 1}
    np.testing.assert_allclose(np.asarray(sh_c1.state.G), np.asarray(sh_adv.state.G), atol=1e-15)
    assert float(sh_c2.state.time) == pytest.approx(3.0e-3, rel=1e-14)


def test_multidevice_4cpu_sharded_chunk_and_cli_parity_subprocess() -> None:
    """Verify (2, 2) and (1, 4) 4-device sharded fused chunks, trace=1, zero all_gather, & CLI."""
    script = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import tempfile
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

jax.config.update("jax_enable_x64", True)

from gx_jax.cli import run_cli_simulation
from gx_jax.config import load_config
from gx_jax.diagnostics_nc import compute_diagnostics_step, init_diagnostics_state
from gx_jax.fields import init_moments
from gx_jax.geometry_nc import init_geometry
from gx_jax.integrator import (
    init_gyrokinetic_operator,
    init_integrator_state,
    step_integrator_state,
)
from gx_jax.shard_tpu import (
    advance_sharded_integrator_chunk,
    create_device_mesh,
    gather_g_state,
    init_sharded_integrator_state,
    make_sharded_chunk_stepper,
    run_sharded_simulation,
)

assert jax.device_count() == 4
repo = Path.cwd()
kbm_in = repo / "benchmarks" / "linear" / "KBM" / "kbm_miller.in"
cfg = replace(
    load_config(kbm_in),
    nx=4, ny=4, nz=8, ntheta_in=8, nperiod=1, zp=1, nhermite=4, nlaguerre=2,
    nonlinear_mode=True, linear=False, scheme="rk3", fixed_dt=True, dt=1.0e-3,
    exb_shear=True, g_exb=0.1, forcing_init=True, forcing_amp=0.02, forcing_index=1,
)
cfg, grids, geo = init_geometry(cfg)
gk_op = init_gyrokinetic_operator(cfg, grids, geo, dtype=jnp.float64)
g0 = init_moments(cfg, grids, geo, dtype=jnp.complex128)

# Reference unsharded 5 steps
st_ref = init_integrator_state(g0, gk_op, dt=1.0e-3)
d_ref = init_diagnostics_state(st_ref.G, st_ref.fields, time=0.0, has_old=False)
snaps_ref = []
for c in range(1, 6):
    prev_g, prev_f = st_ref.G, st_ref.fields
    st_ref = step_integrator_state(st_ref, gk_op, dt_override=1.0e-3, scheme="rk3")
    if c % 3 == 1 or c == 5:
        if c > 1:
            d_ref = init_diagnostics_state(
                prev_g, prev_f, time=st_ref.time - st_ref.dt, has_old=True
            )
        sn, d_ref = compute_diagnostics_step(
            st_ref.G, st_ref.fields, d_ref, grids, geo, gk_op.linear_op.solver,
            time=st_ref.time, dt=st_ref.dt, beta=float(cfg.beta),
        )
        snaps_ref.append(sn)

for mesh_shape in [(2, 2), (1, 4)]:
    mesh = create_device_mesh(mesh_shape, nspecies=cfg.nspecies, nhermite=cfg.nhermite)
    st_sh, snaps_sh = run_sharded_simulation(
        g0, gk_op, mesh, nsteps=5, dt=1.0e-3, scheme="rk3", nwrite=3
    )
    assert not isinstance(st_sh.time.sharding, jax.sharding.SingleDeviceSharding)
    assert not isinstance(st_sh.G.sharding, jax.sharding.SingleDeviceSharding)
    np.testing.assert_allclose(
        np.asarray(gather_g_state(st_sh.G, mesh)), np.asarray(st_ref.G), rtol=1e-12, atol=1e-13
    )
    assert len(snaps_sh) == len(snaps_ref) == 3
    for s_sh, s_r in zip(snaps_sh, snaps_ref, strict=True):
        np.testing.assert_allclose(np.asarray(s_sh.wg.st), np.asarray(s_r.wg.st), rtol=1e-12)
        # Cross-device psum summation order vs 1-device sum can perturb phi by ~1e-12,
        # which / dt (1e-3) yields ~3e-9 relative diff in omega_kxkyt.
        np.testing.assert_allclose(
            np.asarray(s_sh.omega_kxkyt), np.asarray(s_r.omega_kxkyt), rtol=1e-8, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(s_sh.turbulent_heating.total.st),
            np.asarray(s_r.turbulent_heating.total.st),
            rtol=1e-10,
            atol=1e-12,
        )

    # Also verify exact same-mesh parity vs Session 6 step-by-step sharded stepping (rtol=1e-12)
    if mesh_shape == (2, 2):
        from gx_jax.diagnostics_nc import DiagnosticsState
        from gx_jax.shard_tpu import (
            make_replicated_sharding,
            sharded_compute_diagnostics,
            step_sharded_integrator_state,
        )

        rep_s6 = make_replicated_sharding(mesh)
        st_s6 = init_sharded_integrator_state(g0, gk_op, mesh, dt=1.0e-3)
        d_s6 = init_diagnostics_state(st_s6.G, st_s6.fields, time=0.0, has_old=False)
        d_s6 = DiagnosticsState(
            G_old=d_s6.G_old,
            fields_old=d_s6.fields_old,
            time_old=jax.device_put(d_s6.time_old, rep_s6),
            has_old=jax.device_put(d_s6.has_old, rep_s6),
        )
        true_s6 = jax.device_put(jnp.asarray(True, dtype=jnp.bool_), rep_s6)
        step_s6_fn = jax.jit(
            lambda st: step_sharded_integrator_state(
                st, gk_op, mesh, dt_override=1.0e-3, scheme="rk3"
            )
        )
        for c_s6 in range(1, 6):
            p_g, p_f = st_s6.G, st_s6.fields
            st_s6 = step_s6_fn(st_s6)
            if c_s6 % 3 == 1 or c_s6 == 5:
                if c_s6 > 1:
                    d_s6 = DiagnosticsState(
                        G_old=p_g,
                        fields_old=p_f,
                        time_old=st_s6.time - st_s6.dt,
                        has_old=true_s6,
                    )
                sn_s6, d_s6 = sharded_compute_diagnostics(
                    st_s6.G, st_s6.fields, d_s6, grids, geo, gk_op.linear_op.solver,
                    mesh, time=st_s6.time, dt=st_s6.dt, beta=float(cfg.beta),
                )
                idx = [1, 4, 5].index(c_s6)
                np.testing.assert_allclose(
                    np.asarray(snaps_sh[idx].omega_kxkyt),
                    np.asarray(sn_s6.omega_kxkyt),
                    rtol=1e-12,
                    atol=1e-13,
                )

    # Check single-trace sharded chunk stepper & zero SingleDeviceSharding on ChunkResult
    stepper = make_sharded_chunk_stepper(
        gk_op, mesh, dt_override=1.0e-3, scheme="rk3", fixed_chunk_len=2, t_max=1e20
    )
    sh0 = init_sharded_integrator_state(g0, gk_op, mesh, dt=1.0e-3)
    hlo_chunk = str(
        jax.jit(
            lambda s: advance_sharded_integrator_chunk(
                s, gk_op, mesh, 2, dt_override=1.0e-3, scheme="rk3", t_max=1e20, use_scan=True
            )
        )
        .lower(sh0)
        .compiler_ir("stablehlo")
    )
    assert "all_gather" not in hlo_chunk, "Unexpected all_gather in sharded chunk HLO"

    ch1 = stepper(sh0, 2, safe_for_scan=True)
    ch2 = stepper(ch1.state, 2, safe_for_scan=True)
    ch3 = stepper(ch2.state, 1)
    assert stepper.trace_counts == {"while": 1, "scan": 1}
    for leaf in jax.tree_util.tree_leaves(ch3):
        assert not isinstance(leaf.sharding, jax.sharding.SingleDeviceSharding), (
            f"Unexpected SingleDeviceSharding in ChunkResult leaf: {leaf.sharding}"
        )
    np.testing.assert_allclose(
        np.asarray(gather_g_state(ch3.state.G, mesh)), np.asarray(st_ref.G), rtol=1e-12, atol=1e-13
    )

# Check sharded CLI (2x2) with write_moms=True, write_fields=True, async_io=True vs unsharded (1x1)
with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    deck = tmp / "sharded_cli.in"
    deck.write_text(
        f'''[Dimensions]
ntheta = 8
nperiod = 1
ny = 4
nx = 4
nhermite = 4
nlaguerre = 2
nspecies = 2

[Domain]
y0 = 10.0
boundary = "linked"

[Physics]
beta = 0.015
nonlinear_mode = true
fphi = 1.0
fapar = 1.0
fbpar = 1.0

[Time]
dt = 1.0e-3
nstep = 7
scheme = "rk3"
fixed_dt = true
cfl = 0.9

[Initialization]
ikpar_init = 0
init_field = "density"
init_amp = 1.0e-4

[Geometry]
geo_option = "s-alpha"
eps = 0.167
qinp = 1.4
shat = 0.8

[species]
z = [1.0, -1.0]
mass = [1.0, 1.0]
dens = [1.0, 1.0]
temp = [1.0, 2.0]
tprim = [2.49, 2.49]
fprim = [0.8, 0.8]
vnewk = [0.01, 0.01]
type = ["ion", "electron"]

[Boltzmann]
add_Boltzmann_species = false

[Diagnostics]
nwrite = 3
omega = true
free_energy = true
fluxes = true
fields = true
moments = true

[Restart]
save_for_restart = true
nsave = 2
'''
    )
    out_11 = tmp / "out_11.out.nc"
    out_22 = tmp / "out_22.out.nc"
    rst_11 = tmp / "rst_11.restart.nc"
    rst_22 = tmp / "rst_22.restart.nc"
    r11 = run_cli_simulation(
        deck, output_path=out_11, save_restart_path=rst_11,
        mesh_shape=(1, 1), precision="fp64", async_io=False,
    )
    r22 = run_cli_simulation(
        deck, output_path=out_22, save_restart_path=rst_22,
        mesh_shape=(2, 2), precision="fp64", async_io=True,
    )
    assert r11["nstep"] == r22["nstep"] == 7
    np.testing.assert_allclose(
        np.asarray(r22["g_final"]), np.asarray(r11["g_final"]), rtol=1e-12, atol=1e-13
    )
    with nc.Dataset(out_11, "r") as d1, nc.Dataset(out_22, "r") as d2:
        np.testing.assert_allclose(
            d2.groups["Grids"].variables["time"][:],
            d1.groups["Grids"].variables["time"][:],
            rtol=1e-12,
        )
        om0_max = float(
            np.max(np.abs(np.asarray(d1.groups["Diagnostics"].variables["omega_kxkyt"][0])))
        )
        assert om0_max > 1.0
        for vname in (
            "Phi2_t", "Wapar_st", "HeatFlux_st", "TurbulentHeating_st", "omega_kxkyt", "Density"
        ):
            np.testing.assert_allclose(
                np.asarray(d2.groups["Diagnostics"].variables[vname][:]),
                np.asarray(d1.groups["Diagnostics"].variables[vname][:]),
                rtol=1e-6,
                atol=1e-12,
            )
print("4-DEVICE FUSED CHUNK AND CLI PARITY PASSED")
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
    assert "4-DEVICE FUSED CHUNK AND CLI PARITY PASSED" in proc.stdout


# =============================================================================
# 5. CLI Flags (`--async-io`, `--no-async-io`, `--fft-backend`)
# =============================================================================


def test_cli_async_io_and_fft_backend_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify --async-io / --no-async-io and --fft-backend CLI parser and run_cli_simulation."""
    parser = build_arg_parser()
    assert parser.parse_args(["dummy.in"]).async_io is True
    assert parser.parse_args(["dummy.in", "--async-io"]).async_io is True
    assert parser.parse_args(["dummy.in", "--no-async-io"]).async_io is False
    assert parser.parse_args(["dummy.in"]).use_scan is False
    assert parser.parse_args(["dummy.in", "--use-scan"]).use_scan is True
    assert parser.parse_args(["dummy.in", "--no-use-scan"]).use_scan is False
    assert parser.parse_args(["dummy.in"]).fft_backend is None
    for backend in ("auto", "fft", "mxu"):
        assert parser.parse_args(["dummy.in", "--fft-backend", backend]).fft_backend == backend

    with pytest.raises(SystemExit) as exc_bad:
        parser.parse_args(["dummy.in", "--fft-backend", "invalid_backend"])
    assert exc_bad.value.code == 2

    deck = tmp_path / "flags_check.in"
    _write_test_deck(deck, nstep=2, nwrite=1, fixed_dt=True, dt=0.01)

    with pytest.raises(ValueError, match="fft_backend"):
        run_cli_simulation(deck, output_path=tmp_path / "bad.out.nc", fft_backend="bogus")

    # End-to-end CLI with --no-async-io, --use-scan, and --fft-backend auto
    out_nc = tmp_path / "flags_cli.out.nc"
    rc = cli_main(
        [str(deck), "-o", str(out_nc), "--no-async-io", "--use-scan", "--fft-backend", "auto"]
    )
    assert rc == 0
    assert out_nc.is_file()

    # Verify fft_backend forwarding onto the Session 7 GXConfig.fft_backend field
    import gx_jax.cli as cli_mod

    assert load_config(deck).fft_backend == "auto"
    seen_backends: list[str] = []
    orig_init_geo = cli_mod.init_geometry

    def _spy_init_geo(cfg_in: Any, **kwargs: Any) -> Any:
        seen_backends.append(str(cfg_in.fft_backend))
        return orig_init_geo(cfg_in, **kwargs)

    monkeypatch.setattr(cli_mod, "init_geometry", _spy_init_geo)
    run_cli_simulation(
        deck,
        output_path=tmp_path / "fwd_mxu.out.nc",
        fft_backend="mxu",
        async_io=False,
    )
    assert seen_backends == ["mxu"]
