"""KR2 platform, C++ post-processing, and golden .out.nc differential tests."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
import tomllib
import types
import zipfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from gx_jax.cli import build_arg_parser, run_cli_simulation
from gx_jax.cli import main as cli_main
from gx_jax.config import BoltzmannOption, GXConfig, load_config
from gx_jax.diagnostics_nc import (
    DiagnosticsWriter,
    compute_diagnostics_step,
    compute_field_squared_spectra,
    compute_heat_flux_spectra,
    compute_moments_diagnostic_arrays,
    compute_omega_kxky,
    compute_particle_flux_spectra,
    compute_turbulent_heating_spectra,
    compute_wapar_spectra,
    compute_wbpar_spectra,
    compute_wg_spectra,
    compute_wphi_spectra,
    extract_kr2_diagnostics,
    fourier_ky_weights,
    init_diagnostics_state,
    reorder_complex_moment_to_nc,
    reorder_kx_to_out,
)
from gx_jax.fields import (
    Fields,
    enforce_reality,
    init_field_solver,
    load_restart_nc,
    save_restart_nc,
    solve_fields,
)
from gx_jax.geometry_nc import init_geometry
from gx_jax.grids import Grids, init_grids
from gx_jax.shard_tpu import create_device_mesh, shard_g_state, sharded_compute_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
GOLDEN_DIR = REPO_ROOT / "golden_fixtures"
CPP_REF = GOLDEN_DIR
UT_AUSTIN_ZIP = Path("/tmp/ut-austin.zip")


@pytest.fixture(scope="module", autouse=True)
def _enable_x64() -> None:
    jax.config.update("jax_enable_x64", True)


def _install_headless_stubs_if_missing() -> None:
    """Provide lightweight matplotlib/tqdm/tomlkit/f90nml stubs if optional packages are absent."""
    if "tomlkit" not in sys.modules and importlib.util.find_spec("tomlkit") is None:
        tk = types.ModuleType("tomlkit")
        tk.parse = tomllib.loads  # type: ignore[attr-defined]
        tk.table = dict  # type: ignore[attr-defined]
        sys.modules["tomlkit"] = tk

    if "f90nml" not in sys.modules and importlib.util.find_spec("f90nml") is None:
        sys.modules["f90nml"] = types.ModuleType("f90nml")

    if "tqdm" not in sys.modules and importlib.util.find_spec("tqdm") is None:
        tqdm_mod = types.ModuleType("tqdm")

        class _DummyTqdm:
            def __init__(self, iterable: Any, **_kwargs: Any) -> None:
                self._iterable = iterable

            def __iter__(self) -> Any:
                return iter(self._iterable)

            def set_postfix_str(self, _s: str) -> None:
                pass

            @staticmethod
            def write(_msg: str) -> None:
                pass

        tqdm_mod.tqdm = _DummyTqdm  # type: ignore[attr-defined]
        sys.modules["tqdm"] = tqdm_mod

    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        mpl = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        ticker = types.ModuleType("matplotlib.ticker")

        class _DummyLocator:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

        def _noop(*_args: Any, **_kwargs: Any) -> Any:
            return None

        pyplot.figure = _noop  # type: ignore[attr-defined]
        pyplot.subplots = lambda *_a, **_k: (None, None)  # type: ignore[attr-defined]
        pyplot.plot = _noop  # type: ignore[attr-defined]
        pyplot.tight_layout = _noop  # type: ignore[attr-defined]
        pyplot.savefig = _noop  # type: ignore[attr-defined]
        pyplot.show = _noop  # type: ignore[attr-defined]
        ticker.AutoMinorLocator = _DummyLocator  # type: ignore[attr-defined]
        mpl.pyplot = pyplot  # type: ignore[attr-defined]
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = pyplot
        sys.modules["matplotlib.ticker"] = ticker


def _load_module_from_path(mod_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_minimal_miller_deck(
    path: Path,
    *,
    em: bool = False,
    nspecies: int = 1,
    write_fields: bool = False,
    write_moms: bool = False,
) -> None:
    """Write a compact Miller gyrokinetic deck for fast end-to-end CLI tests."""
    fapar = 1.0 if em else 0.0
    fbpar = 1.0 if em else 0.0
    beta = 0.015 if em else 0.0
    if nspecies == 2:
        species_block = """
[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.02 ]
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

    deck = f"""
debug = false

[Dimensions]
 ntheta = 8
 nperiod = 1
 ny = 8
 nx = 8
 nhermite = 4
 nlaguerre = 2
 nspecies = {nspecies}

[Domain]
 y0 = 10.0
 boundary = "linked"

[Physics]
 beta = {beta}
 nonlinear_mode = false
 fphi = 1.0
 fapar = {fapar}
 fbpar = {fbpar}

[Time]
 dt = 0.05
 nstep = 2
 cfl = 0.9
 scheme = "rk3"

[Initialization]
 ikpar_init = 0
 init_field = "density"
 init_amp = 1.0e-5

[Geometry]
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
 betaprim = 0.0
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
 nwrite = 1
 omega = true
 free_energy = true
 fluxes = true
 fields = {str(write_fields).lower()}
 moments = {str(write_moms).lower()}
"""
    path.write_text(deck.strip() + "\n")


# ---------------------------------------------------------------------------
# 1. KR2 Platform (`ut-austin`) Contract & End-to-End Integration
# ---------------------------------------------------------------------------


def test_kr2_dockerfile_and_pyproject_contract() -> None:
    """Verify Dockerfile.gx-jax-tpu and pyproject.toml satisfy KR2 Cloud Batch / runner contract."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    dockerfile_path = REPO_ROOT.parent / "gcp" / "Dockerfile.gx-jax-tpu"
    assert pyproject_path.is_file()
    assert dockerfile_path.is_file()

    pyproject = tomllib.loads(pyproject_path.read_text())
    scripts = pyproject.get("project", {}).get("scripts", {})
    assert scripts.get("gx-jax") == "gx_jax.cli:main"
    deps = " ".join(pyproject.get("project", {}).get("dependencies", []))
    assert "netCDF4" in deps

    docker_txt = dockerfile_path.read_text()
    assert "jax[tpu]" in docker_txt
    assert "JAX_ENABLE_X64=1" in docker_txt
    assert "JAX_DEFAULT_MATMUL_PRECISION=highest" in docker_txt
    # KR2 Dockerfile.gx-runner requires HDF5_USE_FILE_LOCKING=FALSE for GCS FUSE / Lustre mounts
    assert "HDF5_USE_FILE_LOCKING=FALSE" in docker_txt
    assert "gx-jax" in docker_txt
    # KR2 CloudBatchBackend invokes `gx gx.in`; either a `gx` console script or symlink must exist
    has_gx_alias = (scripts.get("gx") == "gx_jax.cli:main") or ("/usr/local/bin/gx" in docker_txt)
    assert has_gx_alias, "KR2 CloudBatchBackend runs `gx gx.in`; provide `gx` script or symlink"


def test_kr2_cli_cfl_run_metrics_schema_and_warm_restart(tmp_path: Path) -> None:
    """Test KR2 `gx.in` CLI lifecycle, `diagnostics_adapter.py` parsing, and Spot warm restart."""
    run_dir = tmp_path / "kr2_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    gx_in = run_dir / "gx.in"
    # Choose large nominal dt=5.0 with fixed_dt=False so CLI MUST honor CFL stepping, not force dt!
    _write_minimal_miller_deck(gx_in, em=True, nspecies=2)
    gx_in.write_text(gx_in.read_text().replace("dt = 0.05", "dt = 5.0"))

    metrics_path = run_dir / "diagnostics_metrics.json"
    rc = cli_main([str(gx_in), "--metrics-json", str(metrics_path)])
    assert rc == 0

    out_nc = run_dir / "gx.out.nc"
    restart_nc = run_dir / "gx.restart.nc"
    assert out_nc.is_file(), "CLI must emit <stem>.out.nc alongside <stem>.in"
    assert restart_nc.is_file(), "CLI must emit <stem>.restart.nc by default for KR2 preemption"
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text())
    # Detailed GX summary keys
    assert metrics["has_nan_or_inf"] is False
    assert metrics["kx_monotonic"] is True
    assert metrics["kz_monotonic"] is True
    assert metrics["n_time"] == 2
    assert 0.0 < metrics["t_end"] < 1.0, (
        f"With fixed_dt=False and nominal dt=5.0, CFL stepping must keep t_end << 10.0, "
        f"got t_end={metrics['t_end']}"
    )

    # Exact KR2 diagnostics_adapter.py / SimulationWatchdog contract keys
    assert metrics.get("converged") is True
    assert metrics.get("failure_type") is None
    assert metrics.get("sim_time_reached") == pytest.approx(metrics["t_end"], rel=1e-6)
    assert isinstance(metrics.get("scalar_metrics"), dict)
    assert float(metrics["scalar_metrics"].get("has_nan", 1.0)) == 0.0
    assert "high_k_tail_ratio" in metrics["scalar_metrics"]
    assert isinstance(metrics.get("fluxes"), dict)
    assert "heat_flux_es" in metrics["fluxes"]
    assert "heat_flux_em" in metrics["fluxes"]
    assert "particle_flux" in metrics["fluxes"]
    assert metrics["fluxes"]["heat_flux_es"] + metrics["fluxes"]["heat_flux_em"] == pytest.approx(
        metrics["Q_total"], rel=1e-5, abs=1e-10
    )

    t_first = float(metrics["t_end"])

    if UT_AUSTIN_ZIP.is_file():
        has_real_tomlkit = (
            "tomlkit" not in sys.modules and importlib.util.find_spec("tomlkit") is not None
        ) or getattr(sys.modules.get("tomlkit"), "__spec__", None) is not None
        _install_headless_stubs_if_missing()
        extract_dir = tmp_path / "ut_austin_src"
        with zipfile.ZipFile(UT_AUSTIN_ZIP) as zf:
            for member in zf.namelist():
                if "kr2-gk-database/hpc/tacc_exec/" in member:
                    zf.extract(member, extract_dir)
        tacc_exec = (
            extract_dir / "ut-austin" / "workstreams" / "kr2-gk-database" / "hpc" / "tacc_exec"
        )
        hpc_root = str(tacc_exec.parent)
        if hpc_root not in sys.path:
            sys.path.insert(0, hpc_root)

        diag_adapter = _load_module_from_path(
            "tacc_exec.diagnostics_adapter", tacc_exec / "diagnostics_adapter.py"
        )
        solver_name, detected_in = diag_adapter.detect_solver_and_input_file(run_dir)
        assert solver_name == "GX"
        assert detected_in == gx_in

        # Path A: inspect_run_directory reading pre-written diagnostics_metrics.json
        res_json = diag_adapter.inspect_run_directory(run_dir)
        assert res_json.solver == "GX"
        assert res_json.converged is True
        assert res_json.failure_type is None
        assert res_json.scalar_metrics.get("has_nan", 0.0) == 0.0
        assert res_json.sim_time_reached == pytest.approx(t_first, rel=1e-5)
        assert "heat_flux_es" in res_json.fluxes
        assert np.isfinite(res_json.fluxes["heat_flux_es"])

        # Path B: consume_stale_diagnostics_metrics archives prior strike JSON before warm restart
        diag_adapter.consume_stale_diagnostics_metrics(run_dir)
        assert not metrics_path.exists()
        assert (run_dir / "diagnostics_metrics.prev.json").is_file()

        if has_real_tomlkit:
            gx_toml = _load_module_from_path("kr2_gx_toml", tacc_exec / "patchers" / "gx_toml.py")
            gx_toml.patch_gx_toml(gx_in, restart_from_checkpoint=True)
        else:
            gx_in.write_text(gx_in.read_text() + "\n[Restart]\n restart = true\n")
    else:
        gx_in.write_text(gx_in.read_text() + "\n[Restart]\n restart = true\n")

    # Spot-preemption warm restart without passing --restart-file CLI flag
    rc_warm = cli_main([str(gx_in), "--metrics-json", str(metrics_path)])
    assert rc_warm == 0
    metrics_warm = json.loads(metrics_path.read_text())
    assert metrics_warm["has_nan_or_inf"] is False
    assert float(metrics_warm["t_end"]) > t_first


# ---------------------------------------------------------------------------
# 2. C++ `post_processing/` & `python_analysis/load_files.py` Compatibility
# ---------------------------------------------------------------------------


def test_cpp_post_processing_scripts_and_load_files_compatibility(tmp_path: Path) -> None:
    """Verify C++ post_processing scripts and python_analysis/load_files.py read .out.nc cleanly."""
    _install_headless_stubs_if_missing()
    in_path = tmp_path / "cpp_pp_em.in"
    out_nc = tmp_path / "cpp_pp_em.out.nc"
    _write_minimal_miller_deck(in_path, em=True, nspecies=2, write_fields=True, write_moms=True)

    summary = run_cli_simulation(
        in_path,
        output_path=out_nc,
        dt=1.0e-4,
        nstep=3,
        nwrite=1,
    )
    assert out_nc.is_file()
    assert summary["output_nc"] == out_nc

    post_dir = REPO_ROOT / "reference" / "gx-cpp" / "post_processing"
    if post_dir.is_dir() and importlib.util.find_spec("matplotlib") is not None:
        hf_mod = _load_module_from_path("gx_cpp_heat_flux", post_dir / "heat_flux.py")
        pf_mod = _load_module_from_path("gx_cpp_particle_flux", post_dir / "particle_flux.py")
        th_mod = _load_module_from_path(
            "gx_cpp_turbulent_heating", post_dir / "turbulent_heating.py"
        )
        gr_mod = _load_module_from_path("gx_cpp_growth_rates", post_dir / "growth_rates.py")
        hl_mod = _load_module_from_path(
            "gx_cpp_hermite_laguerre", post_dir / "hermite_laguerre_spectra.py"
        )
        lf_mod = _load_module_from_path(
            "gx_cpp_load_files", post_dir / "python_analysis" / "load_files.py"
        )
    else:
        hf_mod = pf_mod = th_mod = gr_mod = hl_mod = lf_mod = None

    with netCDF4.Dataset(out_nc, "r") as ds:
        if hf_mod is not None:
            for ispec in range(2):
                hf_mod.heat_flux(ds, ispec=ispec, plot=False)
                pf_mod.particle_flux(ds, ispec=ispec, plot=False)
                th_mod.turbulent_heating(ds, ispec=ispec, plot=False)
                hl_mod.hermite_laguerre_spectra(ds, ispec=ispec, plot=False, normalize=True)

        d_vars = ds.groups["Diagnostics"].variables
        # Total vs ES + Apar + Bpar additive decomposition
        for stem in ("HeatFlux", "ParticleFlux", "TurbulentHeating"):
            tot = np.asarray(d_vars[f"{stem}_st"][:], dtype=np.float64)
            es = np.asarray(d_vars[f"{stem}_st".replace(stem, f"{stem}ES")][:], dtype=np.float64)
            ap = np.asarray(d_vars[f"{stem}_st".replace(stem, f"{stem}Apar")][:], dtype=np.float64)
            bp = np.asarray(d_vars[f"{stem}_st".replace(stem, f"{stem}Bpar")][:], dtype=np.float64)
            np.testing.assert_allclose(tot, es + ap + bp, rtol=1e-5, atol=1e-10)

        # Verify post_processing/fields.py expected 5D layout (time, ky, kx, theta, ri=2)
        nt = len(ds.groups["Grids"].variables["time"][:])
        nky = len(ds.groups["Grids"].variables["ky"][:])
        nkx = len(ds.groups["Grids"].variables["kx"][:])
        nz = len(ds.groups["Grids"].variables["theta"][:])
        for fname in ("Phi", "Apar", "Bpar"):
            arr = np.asarray(d_vars[fname][:])
            assert arr.shape == (nt, nky, nkx, nz, 2)
            assert np.all(np.isfinite(arr))
        for mname in ("Density", "Upar", "Tpar", "Tperp"):
            marr = np.asarray(d_vars[mname][:])
            assert marr.shape == (nt, 2, nky, nkx, nz, 2)
            assert np.all(np.isfinite(marr))

    if gr_mod is not None and lf_mod is not None:
        om_avg, gam_avg = gr_mod.growth_rates(str(out_nc), ikx=0, plot=False)
        assert om_avg.shape == (2,)
        assert gam_avg.shape == (2,)
        assert np.all(np.isfinite(om_avg))
        assert np.all(np.isfinite(gam_avg))

        # load_files.py requires string `.value` attrs on `*_dum` Inputs vars.
        sims = lf_mod.load_files(
            [str(out_nc)],
            groups=["Inputs", "Geometry", "Diagnostics"],
        )
        assert len(sims) == 1, (
            "python_analysis/load_files.py dropped the .out.nc file (missing `.value` attributes "
            "on `*_dum` Inputs variables or missing Boltzmann_type_dum / stir_field_dum)"
        )
        sim_data = next(iter(sims.values()))
        assert "Inputs" in sim_data
        assert "Geometry" in sim_data
        assert "Diagnostics" in sim_data


# ---------------------------------------------------------------------------
# 3. Golden C++ `.out.nc` & `.restart.nc` Differential Verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("in_rel", "nc_rel"),
    [
        (
            "benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in",
            "benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons_correct.out.nc",
        ),
        (
            "benchmarks/linear/ITG_cyclone/itg_miller_adiabatic_electrons.in",
            "benchmarks/linear/ITG_cyclone/itg_miller_adiabatic_electrons_correct.out.nc",
        ),
        (
            "benchmarks/linear/ITG_cyclone/itg_miller_kinetic_electrons.in",
            "benchmarks/linear/ITG_cyclone/itg_miller_kinetic_electrons_correct.out.nc",
        ),
        (
            "benchmarks/linear/KBM/kbm_miller.in",
            "benchmarks/linear/KBM/kbm_miller_correct.out.nc",
        ),
        (
            "benchmarks/linear/KAW/kaw_betahat10.0_kp0.01.in",
            "benchmarks/linear/KAW/kaw_betahat10.0_kp0.01_correct.out.nc",
        ),
    ],
)
def test_golden_linear_benchmarks_grids_geometry_and_single_species_pflux(
    tmp_path: Path, in_rel: str, nc_rel: str
) -> None:
    """Compare DiagnosticsWriter Grids, Geometry, and nspecies=1 pflux against C++ goldens."""
    in_path = BENCH_DIR / in_rel.removeprefix("benchmarks/")
    golden_nc = GOLDEN_DIR / nc_rel
    cfg, grids, geo = init_geometry(load_config(in_path))
    solver = init_field_solver(cfg, grids, geo)

    # Construct non-zero Hermitian-symmetric state to test diagnostics and single-species pflux
    rng = np.random.default_rng(42)
    g_raw = rng.standard_normal(grids.g_shape) + 1j * rng.standard_normal(grids.g_shape)
    g_state = enforce_reality(jnp.asarray(g_raw, dtype=jnp.complex128) * 1e-3, grids)
    fields = solve_fields(g_state, solver)
    diag_state = init_diagnostics_state(g_state, fields, time=0.0, has_old=False)

    cand_nc = tmp_path / golden_nc.name
    with DiagnosticsWriter(cand_nc, cfg, grids, geo) as writer:
        writer.write_step(g_state, fields, diag_state, solver, time=0.0, dt=1e-3)

    with netCDF4.Dataset(golden_nc, "r") as ref_ds, netCDF4.Dataset(cand_nc, "r") as cand_ds:
        for gvar in ("kx", "ky", "kz", "theta"):
            ref_v = np.asarray(ref_ds.groups["Grids"].variables[gvar][:], dtype=np.float64)
            cand_v = np.asarray(cand_ds.groups["Grids"].variables[gvar][:], dtype=np.float64)
            np.testing.assert_allclose(cand_v, ref_v, rtol=1e-5, atol=1e-6)
            if gvar in ("kx", "kz") and len(cand_v) > 1:
                assert np.all(np.diff(cand_v) > 0.0), f"{gvar} must be strictly monotonic"

        ref_geo = ref_ds.groups["Geometry"].variables
        cand_geo = cand_ds.groups["Geometry"].variables
        for geo_arr in (
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
        ):
            if geo_arr in ref_geo:
                np.testing.assert_allclose(
                    np.asarray(cand_geo[geo_arr][:], dtype=np.float64),
                    np.asarray(ref_geo[geo_arr][:], dtype=np.float64),
                    rtol=1e-5,
                    atol=1e-6,
                )
        if "grho_avg" in ref_geo:
            assert float(cand_geo["grho_avg"][()]) == pytest.approx(
                float(ref_geo["grho_avg"][()]), rel=1e-5, abs=1e-6
            )

        # In C++ diagnostic_classes.cu:L391, ParticleFlux* sets n_s = 0 when nspecies == 1
        if cfg.nspecies == 1 and "ParticleFlux_st" in ref_ds.groups["Diagnostics"].variables:
            ref_pf = np.asarray(ref_ds.groups["Diagnostics"].variables["ParticleFlux_st"][:])
            assert float(np.max(np.abs(ref_pf))) == 0.0
            cand_pf = np.asarray(cand_ds.groups["Diagnostics"].variables["ParticleFlux_st"][:])
            assert float(np.max(np.abs(cand_pf))) == 0.0, (
                "DiagnosticsWriter must zero ParticleFlux* when nspecies == 1 to match C++ GX"
            )
            mesh = create_device_mesh(mesh_shape=(1, 1), nspecies=1, nhermite=cfg.nhermite)
            snap_sh, _ = sharded_compute_diagnostics(
                g_state, fields, diag_state, grids, geo, solver, mesh, time=0.0, dt=1e-3
            )
            assert float(jnp.max(jnp.abs(snap_sh.particle_flux.total.st))) == 0.0


def test_golden_kh01_kh01a_and_legacy_restart_parity(tmp_path: Path) -> None:
    """Verify exact float32 ULP spectral parity on kh01/kh01a restarts and legacy restart_tests."""
    u_dir = CPP_REF / "unit_tests" / "inputs"

    # 1. kh01_restart.nc -> kh01.nc[-1] (unscaled) and kh01a.nc[0] (restart_scale=500 -> 2.5e5 Wg)
    cfg_kh, grids_kh, geo_kh = init_geometry(load_config(u_dir / "kh01a.in"))
    solver_kh = init_field_solver(cfg_kh, grids_kh, geo_kh)

    g_kh01, t_kh01 = load_restart_nc(u_dir / "kh01_restart.nc", grids_kh, scale=1.0)
    assert t_kh01 == pytest.approx(0.0)
    f_kh01 = solve_fields(g_kh01, solver_kh)
    wg_kh01 = compute_wg_spectra(g_kh01, grids_kh, geo_kh, solver_kh)
    wphi_kh01 = compute_wphi_spectra(
        f_kh01.phi, grids_kh, geo_kh, solver_kh, phi_avg=f_kh01.phi_avg
    )
    hf_kh01 = compute_heat_flux_spectra(g_kh01, f_kh01, grids_kh, geo_kh, solver_kh)

    with netCDF4.Dataset(u_dir / "kh01.nc", "r") as ds_kh01:
        sp = ds_kh01.groups["Spectra"].variables
        fl = ds_kh01.groups["Fluxes"].variables
        np.testing.assert_allclose(wg_kh01.st, sp["Wst"][-1], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(wg_kh01.lmst, sp["Wlmst"][-1], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(wg_kh01.lst, sp["Wlst"][-1], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(wg_kh01.mst, sp["Wmst"][-1], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(wg_kh01.kxkyst, sp["Wkxkyst"][-1], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(wphi_kh01.kxkyst, sp["Pkxkyst"][-1], rtol=1e-5, atol=1e-12)
        if "qflux" in fl:
            np.testing.assert_allclose(hf_kh01.total.st, fl["qflux"][-1], rtol=1e-6, atol=1e-12)

    # Scaled kh01a.in initial state (scale = 500.0) and kh01a_restart.nc final state
    g_kh01_scaled, _ = load_restart_nc(
        u_dir / "kh01_restart.nc", grids_kh, scale=cfg_kh.restart_scale
    )
    wg_scaled = compute_wg_spectra(g_kh01_scaled, grids_kh, geo_kh, solver_kh)

    g_kh01a_end, t_kh01a_end = load_restart_nc(u_dir / "kh01a_restart.nc", grids_kh, scale=1.0)
    assert t_kh01a_end == pytest.approx(4.0, rel=1e-6)
    f_kh01a_end = solve_fields(g_kh01a_end, solver_kh)
    wg_end = compute_wg_spectra(g_kh01a_end, grids_kh, geo_kh, solver_kh)
    wphi_end = compute_wphi_spectra(
        f_kh01a_end.phi, grids_kh, geo_kh, solver_kh, phi_avg=f_kh01a_end.phi_avg
    )
    hf_end = compute_heat_flux_spectra(g_kh01a_end, f_kh01a_end, grids_kh, geo_kh, solver_kh)

    with netCDF4.Dataset(u_dir / "kh01a.nc", "r") as ds_kh01a:
        sp_a = ds_kh01a.groups["Spectra"].variables
        fl_a = ds_kh01a.groups["Fluxes"].variables
        np.testing.assert_allclose(wg_scaled.st, sp_a["Wst"][0], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(wg_end.st, sp_a["Wst"][-1], rtol=1e-6, atol=1e-12)
        if "Wlmst" in sp_a:
            np.testing.assert_allclose(wg_end.lmst, sp_a["Wlmst"][-1], rtol=1e-5, atol=1e-8)
        if "Pkxkyst" in sp_a:
            np.testing.assert_allclose(wphi_end.kxkyst, sp_a["Pkxkyst"][-1], rtol=5e-5, atol=1e-7)
        np.testing.assert_allclose(hf_end.total.st, fl_a["qflux"][-1], rtol=1e-5, atol=1e-10)

    # 2. Round-trip save_restart_nc <-> load_restart_nc
    rt_file = tmp_path / "roundtrip.restart.nc"
    save_restart_nc(rt_file, g_kh01a_end, float(t_kh01a_end), grids_kh)
    g_reloaded, t_reloaded = load_restart_nc(rt_file, grids_kh)
    assert t_reloaded == pytest.approx(float(t_kh01a_end), rel=1e-7)
    np.testing.assert_allclose(g_reloaded, g_kh01a_end, rtol=1e-6, atol=1e-6)

    # 3. Legacy full-grid (Ns, Nm, Nl, Nz, Nx, Nyc, 2) restart_tests fixtures
    rt_dir = u_dir / "restart_tests"
    with netCDF4.Dataset(rt_dir / "restart_test01.out.nc", "r") as ds_leg:
        ns_l, nm_l, nl_l, nz_l, nx_l, nyc_l, _ = ds_leg.variables["G"].shape
    ny_l = 2 * (nyc_l - 1)
    grids_leg: Grids = init_grids(
        GXConfig(
            nx=nx_l,
            ny=ny_l,
            nz=nz_l,
            ntheta_in=nz_l,
            nhermite=nm_l,
            nlaguerre=nl_l,
            nspecies=ns_l,
            x0=10.0,
            y0=10.0,
            zp=1,
            jtwist=1,
        )
    )
    g_leg1, t_leg1 = load_restart_nc(rt_dir / "restart_test01.out.nc", grids_leg)
    g_leg2, t_leg2 = load_restart_nc(rt_dir / "restart_test02.out.nc", grids_leg)
    assert g_leg1.shape == grids_leg.g_shape
    assert g_leg2.shape == grids_leg.g_shape
    assert t_leg1 == pytest.approx(100.0, rel=1e-5)
    assert t_leg2 == pytest.approx(200.0, rel=1e-5)
    assert np.all(np.isfinite(g_leg1))
    assert np.all(np.isfinite(g_leg2))


def test_spectral_invariants_grho_avg_ky_weights_boltzmann_wphi_em_and_tperp() -> None:
    """Verify grho_avg division, ky=0 vs ky>0 weights, Boltzmann Wphi, EM Wapar/Wbpar, and Tperp."""
    in_path = BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"
    cfg, grids, geo = init_geometry(load_config(in_path))
    solver = init_field_solver(cfg, grids, geo)

    # 1. Fourier reality weights: 0 at (0, 0), 1 on ky == 0 (kx != 0), 2 on ky > 0
    weights = np.asarray(fourier_ky_weights(grids))
    assert weights[0, 0] == 0.0
    if grids.nakx > 1:
        np.testing.assert_allclose(weights[0, 1:], 1.0)
    np.testing.assert_allclose(weights[1:, :], 2.0)

    # Single-mode ky == 0 vs ky > 0 2x energy ratio check
    if grids.nakx > 1:
        phi_ky0 = jnp.zeros(grids.field_shape, dtype=jnp.complex128).at[:, 0, 1].set(1.0 + 0.0j)
        phi_ky1 = jnp.zeros(grids.field_shape, dtype=jnp.complex128).at[:, 1, 1].set(1.0 + 0.0j)
        s_ky0 = compute_field_squared_spectra(phi_ky0, grids, geo)
        s_ky1 = compute_field_squared_spectra(phi_ky1, grids, geo)
        assert float(s_ky1.total) == pytest.approx(2.0 * float(s_ky0.total), rel=1e-12)

    # 2. Boltzmann Wphi: z-constant zonal potential has zero Boltzmann energy for electrons,
    #    nonzero for adiabatic ions
    if grids.nakx > 1:
        phi_zonal_const = (
            jnp.zeros(grids.field_shape, dtype=jnp.complex128).at[:, 0, 1].set(2.0 + 0.0j)
        )
        wphi_el = compute_wphi_spectra(
            phi_zonal_const,
            grids,
            geo,
            solver,
            add_boltzmann_species=True,
            boltzmann_opt=BoltzmannOption.BOLTZMANN_ELECTRONS,
            tau_fac=1.5,
        )
        wphi_ion = compute_wphi_spectra(
            phi_zonal_const,
            grids,
            geo,
            solver,
            add_boltzmann_species=True,
            boltzmann_opt=BoltzmannOption.BOLTZMANN_IONS,
            tau_fac=1.5,
        )
        assert float(wphi_el.boltzmann_total) == pytest.approx(0.0, abs=1e-14)
        # 0.5 * fac(ky=0 -> 1) * sum(vol_fac=1) * tau_fac(1.5) * |2.0|^2 = 3.0
        assert float(wphi_ion.boltzmann_total) == pytest.approx(3.0, rel=1e-12)

    # 3. grho_avg divides HeatFlux*, ParticleFlux*, TurbulentHeating*
    rng = np.random.default_rng(7)
    g_now = enforce_reality(
        jnp.asarray(rng.standard_normal(grids.g_shape) + 1j * rng.standard_normal(grids.g_shape)),
        grids,
    )
    g_old = enforce_reality(
        jnp.asarray(rng.standard_normal(grids.g_shape) + 1j * rng.standard_normal(grids.g_shape)),
        grids,
    )
    f_now = solve_fields(g_now, solver)
    f_old = solve_fields(g_old, solver)

    scale_grho = 2.5
    geo_scaled = dataclasses.replace(
        geo,
        grho_avg=geo.grho_avg * scale_grho,
        flux_fac=geo.flux_fac / scale_grho,
    )

    hf_1 = compute_heat_flux_spectra(g_now, f_now, grids, geo, solver)
    hf_2 = compute_heat_flux_spectra(g_now, f_now, grids, geo_scaled, solver)
    pf_1 = compute_particle_flux_spectra(g_now, f_now, grids, geo, solver)
    pf_2 = compute_particle_flux_spectra(g_now, f_now, grids, geo_scaled, solver)
    th_1 = compute_turbulent_heating_spectra(g_now, f_now, g_old, f_old, 1e-3, grids, geo, solver)
    th_2 = compute_turbulent_heating_spectra(
        g_now, f_now, g_old, f_old, 1e-3, grids, geo_scaled, solver
    )
    wg_1 = compute_wg_spectra(g_now, grids, geo, solver)
    wg_2 = compute_wg_spectra(g_now, grids, geo_scaled, solver)

    np.testing.assert_allclose(hf_2.total.st, hf_1.total.st / scale_grho, rtol=1e-12)
    np.testing.assert_allclose(pf_2.total.st, pf_1.total.st / scale_grho, rtol=1e-12)
    # Collisional TurbulentHeating uses geo.vol_fac (diagnostic_classes.cu:L522-524), NOT flux_fac
    np.testing.assert_allclose(th_2.total.st, th_1.total.st, rtol=1e-12)
    np.testing.assert_allclose(wg_2.st, wg_1.st, rtol=1e-12)

    # 4. Electromagnetic Wapar & Wbpar formulas vs golden KAW benchmark
    kaw_in = BENCH_DIR / "linear" / "KAW" / "kaw_betahat10.0_kp0.01.in"
    kaw_nc = GOLDEN_DIR / "benchmarks" / "linear" / "KAW" / "kaw_betahat10.0_kp0.01_correct.out.nc"
    cfg_kaw, grids_kaw, geo_kaw = init_geometry(load_config(kaw_in))
    apar_unit = jnp.zeros(grids_kaw.field_shape, dtype=jnp.complex128).at[:, 1, 0].set(1.0 + 0.0j)
    apar2_kaw = compute_field_squared_spectra(apar_unit, grids_kaw, geo_kaw)
    wapar_kaw = compute_wapar_spectra(apar_unit, grids_kaw, geo_kaw, cfg_kaw.nspecies)
    with netCDF4.Dataset(kaw_nc, "r") as ds_kaw:
        ref_ratio = np.asarray(
            ds_kaw.groups["Diagnostics"].variables["Wapar_st"][-1], dtype=np.float64
        ) / float(ds_kaw.groups["Diagnostics"].variables["Apar2_t"][-1])
    cand_ratio = np.asarray(wapar_kaw.st, dtype=np.float64) / float(apar2_kaw.total)
    np.testing.assert_allclose(cand_ratio, ref_ratio, rtol=1e-5)

    bpar_unit = jnp.zeros(grids.field_shape, dtype=jnp.complex128).at[:, 1, 0].set(1.0 + 0.0j)
    bpar2_spec = compute_field_squared_spectra(bpar_unit, grids, geo)
    wbpar_spec = compute_wbpar_spectra(bpar_unit, grids, geo, cfg.nspecies, beta=cfg.beta)
    assert float(wbpar_spec.total) == pytest.approx(
        float(bpar2_spec.total) / float(cfg.beta), rel=1e-12
    )

    # 5. Tperp moment sign: C++ TperpDiagnostic::calculate copies +G[:, 0, 1] (NOT -G[:, 0, 1])
    g_tperp = jnp.zeros(grids.g_shape, dtype=jnp.complex128).at[0, 0, 1, :, 1, 0].set(3.25 - 1.5j)
    f_tperp = solve_fields(g_tperp, solver)
    moms_nc = compute_moments_diagnostic_arrays(g_tperp, f_tperp, geo, solver)
    expected_tperp = reorder_complex_moment_to_nc(g_tperp[:, 0, 1])
    np.testing.assert_allclose(moms_nc["Tperp"], expected_tperp, rtol=1e-12, atol=1e-12)
    assert reorder_kx_to_out(jnp.arange(grids.nakx)).shape == (grids.nakx,)


# ---------------------------------------------------------------------------
# 4. Independent C++ CUDA Kernel Transcription & CLI Cadence / t_max / nsave
# ---------------------------------------------------------------------------


def _cpp_jflr(l_idx: int, b: np.ndarray) -> np.ndarray:
    """Line-by-line NumPy transcription of `Jflr` (`device_funcs.cu:L127-139`)."""
    if l_idx < 0 or l_idx > 30:
        return np.zeros_like(b, dtype=np.float64)
    return ((-0.5 * b) ** l_idx / float(math.factorial(l_idx))) * np.exp(-0.5 * b)


def _cpp_jflr_b(l_idx: int, b: np.ndarray) -> np.ndarray:
    """Line-by-line NumPy transcription of `JflrB` (`device_funcs.cu:L146-149`)."""
    return _cpp_jflr(l_idx, b) + _cpp_jflr(l_idx - 1, b)


def _cpp_jfac(l_idx: int, b: np.ndarray) -> np.ndarray:
    """Line-by-line NumPy transcription of `Jfac` (`device_funcs.cu:L141-154`)."""
    jflr_a = (
        l_idx * _cpp_jflr(l_idx - 1, b)
        + 2.0 * l_idx * _cpp_jflr(l_idx, b)
        + (l_idx + 1.0) * _cpp_jflr(l_idx + 1, b)
    )
    return 1.5 * _cpp_jflr(l_idx, b) + jflr_a


def _cpp_gamma0(b: np.ndarray) -> np.ndarray:
    """Exact integral `Gamma_0(b) = (1/2pi) int_0^{2pi} exp(-b (1 - cos theta)) dtheta`."""
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    return np.mean(np.exp(-b[..., None] * (1.0 - np.cos(theta))), axis=-1)


def _cpp_roll_kx_out(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    nk = arr.shape[axis] // 2
    return np.roll(arr, shift=nk, axis=axis) if nk > 0 else arr


def _cpp_reference_diagnostics(
    g_cur: np.ndarray,
    g_old: np.ndarray,
    phi: np.ndarray,
    apar: np.ndarray,
    bpar: np.ndarray,
    phi_old: np.ndarray,
    apar_old: np.ndarray,
    bpar_old: np.ndarray,
    dt: float,
    grids: Grids,
    geo: Any,
    solver: Any,
) -> dict[str, np.ndarray]:
    """Independent NumPy transcription of `device_funcs.cu` & `spectra_calc.cu` kernels."""
    ns, nm, nl, nz, naky, nakx = g_cur.shape
    active = np.asarray(grids.active_mask, dtype=bool)
    fac = np.where(np.arange(naky)[:, None] == 0, 1.0, 2.0)
    fac = np.where(active, fac, 0.0)
    vol = np.asarray(geo.vol_fac, dtype=np.float64)
    flx = np.asarray(geo.flux_fac, dtype=np.float64)
    kperp2 = np.asarray(geo.kperp2, dtype=np.float64)
    bmag = np.asarray(geo.bmag, dtype=np.float64)
    ky = np.asarray(grids.ky, dtype=np.float64)

    def _red3(s_zyx: np.ndarray) -> tuple[float, np.ndarray]:
        kxky = _cpp_roll_kx_out(np.sum(s_zyx, axis=0), axis=-1)
        return float(np.sum(kxky)), kxky

    def _red4(s_szyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kxkyst = _cpp_roll_kx_out(np.sum(s_szyx, axis=1), axis=-1)
        return np.sum(kxkyst, axis=(1, 2)), kxkyst

    phi2_tot, phi2_kxky = _red3(0.5 * fac[None, :, :] * vol[:, None, None] * np.abs(phi) ** 2)
    apar2_tot, _ = _red3(0.5 * fac[None, :, :] * vol[:, None, None] * np.abs(apar) ** 2)
    zonal_zyx = np.zeros_like(kperp2)
    zonal_zyx[:, 0, :] = np.where(active[0, :], 0.5 * vol[:, None] * np.abs(phi[:, 0, :]) ** 2, 0.0)
    phi2_zonal_tot = float(np.sum(zonal_zyx))

    wapar_st, _ = _red4(
        np.broadcast_to(
            (
                0.5
                * fac[None, :, :]
                * vol[:, None, None]
                * kperp2
                * (bmag[:, None, None] ** 2)
                * np.abs(apar) ** 2
            )[None, ...],
            (ns, nz, naky, nakx),
        )
    )

    nt_s = np.asarray(solver.sp_nt, dtype=np.float64)
    dens_s = np.asarray(solver.sp_dens, dtype=np.float64)
    rho2_s = np.asarray(solver.sp_rho2, dtype=np.float64)
    vt_s = np.asarray(solver.sp_vt, dtype=np.float64)
    tz_s = np.asarray(solver.sp_tz, dtype=np.float64)
    zt_s = np.asarray(solver.sp_zt, dtype=np.float64)
    nz_s = np.asarray(solver.sp_nz, dtype=np.float64)

    wg_lmst = np.zeros((ns, nm, nl), dtype=np.float64)
    wg_szyx = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    wphi_szyx = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    q_es = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    q_ap = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    q_bp = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    p_es = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    p_ap = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    p_bp = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    th_es = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    th_ap = np.zeros((ns, nz, naky, nakx), dtype=np.float64)
    th_bp = np.zeros((ns, nz, naky, nakx), dtype=np.float64)

    vphi_r = 1.0j * ky[None, :, None] * phi
    va_r = 1.0j * ky[None, :, None] * apar
    vb_r = 1.0j * ky[None, :, None] * bpar
    ky_gt0 = (np.arange(naky)[:, None] > 0) & active

    dphidt = (phi - phi_old) / dt
    dadt = (apar - apar_old) / dt
    dbdt = (bpar - bpar_old) / dt

    def _gm(arr: np.ndarray, s_i: int, m_i: int) -> np.ndarray:
        if m_i < nm:
            return arr[s_i, m_i]
        return np.zeros((nl, nz, naky, nakx), dtype=np.complex128)

    for s_i in range(ns):
        b_s = kperp2 * rho2_s[s_i]
        w_mode = (
            0.5
            * fac[None, None, None, :, :]
            * vol[None, None, :, None, None]
            * nt_s[s_i]
            * (np.abs(g_cur[s_i]) ** 2)
        )
        wg_lmst[s_i] = np.sum(w_mode, axis=(2, 3, 4))
        wg_szyx[s_i] = np.sum(w_mode, axis=(0, 1))
        wphi_szyx[s_i] = (
            0.5
            * fac[None, :, :]
            * vol[:, None, None]
            * (1.0 - _cpp_gamma0(b_s))
            * (np.abs(phi) ** 2)
        )

        g0 = _gm(g_cur, s_i, 0)
        g1 = _gm(g_cur, s_i, 1)
        g2 = _gm(g_cur, s_i, 2)
        g3 = _gm(g_cur, s_i, 3)
        g0_old = _gm(g_old, s_i, 0)
        g1_old = _gm(g_old, s_i, 1)

        p_bar = np.zeros((nz, naky, nakx), dtype=np.complex128)
        q_bar = np.zeros((nz, naky, nakx), dtype=np.complex128)
        qb_bar = np.zeros((nz, naky, nakx), dtype=np.complex128)
        n_g = np.zeros((nz, naky, nakx), dtype=np.complex128)
        u_g = np.zeros((nz, naky, nakx), dtype=np.complex128)
        ub_g = np.zeros((nz, naky, nakx), dtype=np.complex128)
        n_h = np.zeros((nz, naky, nakx), dtype=np.complex128)
        u_h = np.zeros((nz, naky, nakx), dtype=np.complex128)
        ub_h = np.zeros((nz, naky, nakx), dtype=np.complex128)
        n_h_old = np.zeros((nz, naky, nakx), dtype=np.complex128)
        u_h_old = np.zeros((nz, naky, nakx), dtype=np.complex128)
        ub_h_old = np.zeros((nz, naky, nakx), dtype=np.complex128)

        for l_i in range(nl):
            jl = _cpp_jflr(l_i, b_s)
            jlb = _cpp_jflr_b(l_i, b_s)
            jf = _cpp_jfac(l_i, b_s)
            jfm1 = _cpp_jfac(l_i - 1, b_s)
            p_bar += jf * g0[l_i] + (1.0 / math.sqrt(2.0)) * jl * g2[l_i]
            q_bar += jf * g1[l_i] + jl * (math.sqrt(1.5) * g3[l_i] + g1[l_i])
            qb_bar += (jf + jfm1) * g0[l_i] + (1.0 / math.sqrt(2.0)) * jlb * g2[l_i]
            n_g += jl * g0[l_i]
            u_g += jl * g1[l_i]
            ub_g += jlb * g0[l_i]

            hl0 = g0[l_i] + zt_s[s_i] * jl * phi + jlb * bpar
            hl1 = g1[l_i] - zt_s[s_i] * vt_s[s_i] * jl * apar
            hl0_old = g0_old[l_i] + zt_s[s_i] * jl * phi_old + jlb * bpar_old
            hl1_old = g1_old[l_i] - zt_s[s_i] * vt_s[s_i] * jl * apar_old
            n_h += jl * hl0
            u_h += jl * hl1
            ub_h += jlb * hl0
            n_h_old += jl * hl0_old
            u_h_old += jl * hl1_old
            ub_h_old += jlb * hl0_old

        n_eff = dens_s[s_i] if ns > 1 else 0.0
        q_es[s_i] = np.where(
            ky_gt0[None, :, :],
            2.0 * flx[:, None, None] * nt_s[s_i] * np.real(np.conj(vphi_r) * p_bar),
            0.0,
        )
        q_ap[s_i] = np.where(
            ky_gt0[None, :, :],
            2.0 * flx[:, None, None] * nt_s[s_i] * np.real(-vt_s[s_i] * np.conj(va_r) * q_bar),
            0.0,
        )
        q_bp[s_i] = np.where(
            ky_gt0[None, :, :],
            2.0 * flx[:, None, None] * nt_s[s_i] * np.real(tz_s[s_i] * np.conj(vb_r) * qb_bar),
            0.0,
        )
        p_es[s_i] = np.where(
            ky_gt0[None, :, :],
            2.0 * flx[:, None, None] * n_eff * np.real(np.conj(vphi_r) * n_g),
            0.0,
        )
        p_ap[s_i] = np.where(
            ky_gt0[None, :, :],
            2.0 * flx[:, None, None] * n_eff * np.real(-vt_s[s_i] * np.conj(va_r) * u_g),
            0.0,
        )
        p_bp[s_i] = np.where(
            ky_gt0[None, :, :],
            2.0 * flx[:, None, None] * n_eff * np.real(tz_s[s_i] * np.conj(vb_r) * ub_g),
            0.0,
        )

        # C++ TurbulentHeatingDiagnostic passes geo->vol_fac (diagnostic_classes.cu:L522-524)
        th_pre = 0.5 * fac[None, :, :] * vol[:, None, None] * nz_s[s_i]
        dn_dt = (n_h - n_h_old) / dt
        du_dt = (u_h - u_h_old) / dt
        dub_dt = (ub_h - ub_h_old) / dt
        th_es[s_i] = th_pre * np.real(np.conj(dphidt) * n_h_old - np.conj(phi_old) * dn_dt)
        th_ap[s_i] = th_pre * np.real(
            -vt_s[s_i] * (np.conj(dadt) * u_h_old - np.conj(apar_old) * du_dt)
        )
        th_bp[s_i] = th_pre * np.real(
            tz_s[s_i] * (np.conj(dbdt) * ub_h_old - np.conj(bpar_old) * dub_dt)
        )

    # C++ growthRates (device_funcs.cu:L879) checks Re != 0 AND Im != 0 on p_now only
    iz = (nz // 2 + 1) if nz > 1 else 0
    pn = phi[iz]
    pp = phi_old[iz]
    val = active & (pn.real != 0.0) & (pn.imag != 0.0) & ((pp.real != 0.0) | (pp.imag != 0.0))
    rat = np.where(val, pn / np.where(val, pp, 1.0 + 0.0j), 1.0 + 0.0j)
    om_re = np.where(val, -np.arctan2(rat.imag, rat.real) / dt, 0.0)
    om_im = np.where(val, np.log(np.abs(rat)) / dt, 0.0)
    omega_kxkyt = _cpp_roll_kx_out(np.stack([om_re, om_im], axis=-1), axis=1)

    return {
        "phi2_tot": np.asarray(phi2_tot),
        "phi2_kxky": phi2_kxky,
        "apar2_tot": np.asarray(apar2_tot),
        "phi2_zonal_tot": np.asarray(phi2_zonal_tot),
        "wapar_st": wapar_st,
        "wg_lmst": wg_lmst,
        "wg_st": _red4(wg_szyx)[0],
        "wphi_st": _red4(wphi_szyx)[0],
        "q_tot_st": _red4(q_es + q_ap + q_bp)[0],
        "q_es_st": _red4(q_es)[0],
        "q_ap_st": _red4(q_ap)[0],
        "q_bp_st": _red4(q_bp)[0],
        "p_tot_st": _red4(p_es + p_ap + p_bp)[0],
        "p_es_st": _red4(p_es)[0],
        "p_ap_st": _red4(p_ap)[0],
        "p_bp_st": _red4(p_bp)[0],
        "th_tot_st": _red4(th_es + th_ap + th_bp)[0],
        "th_es_st": _red4(th_es)[0],
        "th_ap_st": _red4(th_ap)[0],
        "th_bp_st": _red4(th_bp)[0],
        "omega_kxkyt": omega_kxkyt,
    }


def test_independent_cpp_kernel_transcription_random_g_and_random_fields() -> None:
    """Differential test vs independent C++ transcription on random G AND independent fields."""
    base_in = BENCH_DIR / "linear" / "KBM" / "kbm_miller.in"
    base_cfg = load_config(base_in)

    cases = [
        # (nhermite=3 tests ghost m=3 zero slice in heat_flux q_bar; nhermite=6 tests full m=0..3)
        dataclasses.replace(
            base_cfg,
            nx=12,
            ny=12,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=3,
            nlaguerre=3,
            fphi=1.0,
            fapar=1.0,
            fbpar=1.0,
            beta=0.02,
        ),
        dataclasses.replace(
            base_cfg,
            nx=12,
            ny=12,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=6,
            nlaguerre=3,
            fphi=1.0,
            fapar=1.0,
            fbpar=1.0,
            beta=0.02,
        ),
        dataclasses.replace(
            base_cfg,
            nx=12,
            ny=12,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=4,
            nlaguerre=3,
            nspecies=1,
            species=(base_cfg.species[0],),
            all_kinetic=False,
            add_boltzmann_species=True,
            boltzmann_opt=BoltzmannOption.BOLTZMANN_ELECTRONS,
            boltzmann_type="electrons",
            tau_fac=1.3,
            fphi=1.0,
            fapar=0.0,
            fbpar=0.0,
            beta=0.0,
        ),
        dataclasses.replace(
            base_cfg,
            nx=12,
            ny=12,
            nz=16,
            ntheta_in=16,
            nperiod=1,
            zp=1,
            nhermite=4,
            nlaguerre=3,
            nspecies=1,
            species=(base_cfg.species[1],),
            all_kinetic=False,
            add_boltzmann_species=True,
            boltzmann_opt=BoltzmannOption.BOLTZMANN_IONS,
            boltzmann_type="ions",
            tau_fac=0.8,
            fphi=1.0,
            fapar=0.0,
            fbpar=0.0,
            beta=0.0,
        ),
    ]

    for idx, raw_cfg in enumerate(cases):
        cfg, grids, geo = init_geometry(raw_cfg)
        solver = init_field_solver(cfg, grids, geo)
        rng = np.random.default_rng(500 + idx)

        def _rand_c(shape: tuple[int, ...], r: np.random.Generator = rng) -> np.ndarray:
            arr = (r.standard_normal(shape) + 1j * r.standard_normal(shape)) * 0.1
            arr[..., 0, 0] = 0.0
            return arr

        # Independent random G and independent random fields (no quasineutrality cancellation!)
        g_cur = _rand_c(grids.g_shape)
        g_old = _rand_c(grids.g_shape)
        phi = _rand_c(grids.field_shape)
        apar = _rand_c(grids.field_shape)
        bpar = _rand_c(grids.field_shape)
        phi_old = _rand_c(grids.field_shape)
        apar_old = _rand_c(grids.field_shape)
        bpar_old = _rand_c(grids.field_shape)

        # Inject pure-real p_now at iky=1 (-> 0) and pure-real p_prev at iky=2 (-> finite!)
        iz = (grids.nz // 2 + 1) if grids.nz > 1 else 0
        phi[iz, 1, 0] = 0.25 + 0.0j
        phi_old[iz, 2, 0] = 0.25 + 0.0j
        phi[iz, 2, 0] = 0.30 + 0.15j

        vol = np.asarray(geo.vol_fac, dtype=np.float64)
        phi_avg = jnp.asarray(np.sum(vol[:, None] * phi[:, 0, :], axis=0), dtype=jnp.complex128)
        phi_old_avg = jnp.asarray(
            np.sum(vol[:, None] * phi_old[:, 0, :], axis=0), dtype=jnp.complex128
        )
        f_cur = Fields(
            phi=jnp.asarray(phi), apar=jnp.asarray(apar), bpar=jnp.asarray(bpar), phi_avg=phi_avg
        )
        f_prev = Fields(
            phi=jnp.asarray(phi_old),
            apar=jnp.asarray(apar_old),
            bpar=jnp.asarray(bpar_old),
            phi_avg=phi_old_avg,
        )
        dt_val = 3.5e-3

        cpp = _cpp_reference_diagnostics(
            g_cur, g_old, phi, apar, bpar, phi_old, apar_old, bpar_old, dt_val, grids, geo, solver
        )
        dstate = init_diagnostics_state(jnp.asarray(g_old), f_prev, time=0.0, has_old=True)
        snap, _ = compute_diagnostics_step(
            jnp.asarray(g_cur),
            f_cur,
            dstate,
            grids,
            geo,
            solver,
            time=dt_val,
            dt=dt_val,
            beta=float(cfg.beta),
        )

        np.testing.assert_allclose(snap.phi2.total, cpp["phi2_tot"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.phi2.kxky, cpp["phi2_kxky"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.apar2.total, cpp["apar2_tot"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            snap.phi2_zonal.total, cpp["phi2_zonal_tot"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(snap.wapar.st, cpp["wapar_st"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.wg.lmst, cpp["wg_lmst"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.wg.st, cpp["wg_st"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.wphi.st, cpp["wphi_st"], rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(snap.heat_flux.total.st, cpp["q_tot_st"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.heat_flux.es.st, cpp["q_es_st"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.heat_flux.apar.st, cpp["q_ap_st"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(snap.heat_flux.bpar.st, cpp["q_bp_st"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            snap.particle_flux.total.st, cpp["p_tot_st"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            snap.turbulent_heating.total.st, cpp["th_tot_st"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            snap.turbulent_heating.es.st, cpp["th_es_st"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            snap.turbulent_heating.apar.st, cpp["th_ap_st"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            snap.turbulent_heating.bpar.st, cpp["th_bp_st"], rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(snap.omega_kxkyt, cpp["omega_kxkyt"], rtol=1e-12, atol=1e-13)
        # Explicitly check pure-real p_now -> 0.0 while pure-real p_prev -> finite non-zero omega
        om_direct = np.asarray(
            compute_omega_kxky(jnp.asarray(phi), jnp.asarray(phi_old), dt_val, grids)
        )
        np.testing.assert_allclose(om_direct[1, grids.nakx // 2], [0.0, 0.0], atol=0.0)
        assert np.all(np.abs(om_direct[2, grids.nakx // 2]) > 0.0)

        # Also check sharded_compute_diagnostics matches on (1, 1) mesh (including TH vol_fac!)
        mesh = create_device_mesh(mesh_shape=(1, 1), nspecies=cfg.nspecies, nhermite=cfg.nhermite)
        dstate_sh = init_diagnostics_state(
            shard_g_state(jnp.asarray(g_old), mesh), f_prev, time=0.0, has_old=True
        )
        snap_sh, _ = sharded_compute_diagnostics(
            shard_g_state(jnp.asarray(g_cur), mesh),
            f_cur,
            dstate_sh,
            grids,
            geo,
            solver,
            mesh,
            time=dt_val,
            dt=dt_val,
            beta=float(cfg.beta),
        )
        np.testing.assert_allclose(
            snap_sh.turbulent_heating.total.st, cpp["th_tot_st"], rtol=1e-12, atol=1e-13
        )


def test_cli_cpp_nwrite_cadence_t_max_periodic_nsave_and_warm_append(tmp_path: Path) -> None:
    """Verify CLI honors t_max without nstep, writes [1,4,7]*dt (no t=0), and saves restart."""
    run_dir = tmp_path / "cadence_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    deck_path = run_dir / "cadence.in"
    _write_minimal_miller_deck(deck_path, em=False, nspecies=1)

    dt_val = 0.05
    # Replace `nstep = 2` with `t_max = 7 * dt` (no nstep in deck -> cfg.nstep = 2_000_000_000!),
    # fixed_dt = true, nwrite = 3, and nsave = 2.
    txt = deck_path.read_text()
    txt = txt.replace(
        "dt = 0.05\n nstep = 2",
        f"dt = {dt_val}\n t_max = {7.0 * dt_val}\n fixed_dt = true",
    )
    txt = txt.replace("nwrite = 1", "nwrite = 3")
    txt += "\n[Restart]\n save_for_restart = true\n nsave = 2\n"
    deck_path.write_text(txt)

    metrics_path = run_dir / "metrics.json"
    rc = cli_main([str(deck_path), "--metrics-json", str(metrics_path)])
    assert rc == 0

    out_nc = run_dir / "cadence.out.nc"
    restart_nc = run_dir / "cadence.restart.nc"
    assert out_nc.is_file()
    assert restart_nc.is_file()
    # Ensure no leftover atomic temp files
    assert list(run_dir.glob("*.tmp")) == []

    with netCDF4.Dataset(out_nc, "r") as ds:
        times = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
    np.testing.assert_allclose(times, np.array([1.0, 4.0, 7.0]) * dt_val, rtol=1e-6, atol=1e-12)

    with netCDF4.Dataset(restart_nc, "r") as rds:
        assert float(rds.variables["time"][()]) == pytest.approx(7.0 * dt_val, rel=1e-6)

    payload = json.loads(metrics_path.read_text())
    assert payload["nstep"] == 7

    # Warm append continuation to 11 * dt: writes [8, 11] * dt with ZERO duplicate 7 * dt frame
    txt_warm = txt.replace(f"t_max = {7.0 * dt_val}", f"t_max = {11.0 * dt_val}").replace(
        "save_for_restart = true",
        "restart = true\n append_on_restart = true\n save_for_restart = true",
    )
    deck_path.write_text(txt_warm)
    rc2 = cli_main([str(deck_path), "--metrics-json", str(metrics_path)])
    assert rc2 == 0

    with netCDF4.Dataset(out_nc, "r") as ds2:
        times2 = np.asarray(ds2.groups["Grids"].variables["time"][:], dtype=np.float64)
    np.testing.assert_allclose(
        times2, np.array([1.0, 4.0, 7.0, 8.0, 11.0]) * dt_val, rtol=1e-6, atol=1e-12
    )

    # Batch resume via `restart = false, restart_if_exists = true` + `restart_with_perturb = true`
    txt_if_exists = txt.replace(f"t_max = {7.0 * dt_val}", f"t_max = {12.0 * dt_val}").replace(
        "save_for_restart = true",
        "restart = false\n restart_if_exists = true\n"
        " restart_with_perturb = true\n append_on_restart = true\n save_for_restart = true",
    )
    deck_path.write_text(txt_if_exists)
    res_if_exists = run_cli_simulation(deck_path)
    assert res_if_exists["nstep"] == 1
    assert res_if_exists["t_final"] == pytest.approx(12.0 * dt_val, rel=1e-6)
    with netCDF4.Dataset(out_nc, "r") as ds3:
        times3 = np.asarray(ds3.groups["Grids"].variables["time"][:], dtype=np.float64)
    np.testing.assert_allclose(
        times3, np.array([1.0, 4.0, 7.0, 8.0, 11.0, 12.0]) * dt_val, rtol=1e-6, atol=1e-12
    )


def test_reviewer_kr2_sim_time_target_strict_json_and_cli_error_codes(tmp_path: Path) -> None:
    """Verify CLI exit code 2 errors, fp32 default, sim_time_target, and RFC-8259 strict JSON."""
    parser = build_arg_parser()
    parsed_defaults = parser.parse_args(["dummy.in"])
    assert parsed_defaults.precision == "fp32"

    # 1. Missing input file exits with status 2 (parser.error)
    with pytest.raises(SystemExit) as exc_missing:
        cli_main([str(tmp_path / "does_not_exist.in")])
    assert exc_missing.value.code == 2

    deck_path = tmp_path / "strict_check.in"
    _write_minimal_miller_deck(deck_path, em=False, nspecies=1)

    # 2. Invalid --mesh and mutually exclusive --linear --nonlinear exit with status 2
    for bad_argv in (
        [str(deck_path), "--mesh", "not_a_mesh"],
        [str(deck_path), "--mesh", "0x4"],
        [str(deck_path), "--linear", "--nonlinear"],
    ):
        with pytest.raises(SystemExit) as exc_bad:
            cli_main(bad_argv)
        assert exc_bad.value.code == 2

    # 3. CLI emits sim_time_target, precision="fp32" default, and strict JSON
    metrics_path = tmp_path / "strict_metrics.json"
    rc = cli_main(
        [str(deck_path), "--t-max", "0.10", "--dt", "0.05", "--metrics-json", str(metrics_path)]
    )
    assert rc == 0

    def _reject_non_strict_json(token: str) -> Any:
        raise AssertionError(f"Non-RFC-8259 JSON constant encountered: {token!r}")

    payload = json.loads(metrics_path.read_text(), parse_constant=_reject_non_strict_json)
    assert payload["precision"] == "fp32"
    assert payload["sim_time_target"] == pytest.approx(0.10, rel=1e-6)
    assert payload["sim_time_reached"] == pytest.approx(0.10, rel=1e-6)
    assert np.isfinite(payload["scalar_metrics"]["dt_collapse_ratio"])

    # 3b. Partial final write interval [1, 1001, 2001, 2002] * dt yields dt_collapse_ratio == 1.0
    partial_nc = tmp_path / "partial_tail.out.nc"
    dt_syn = 0.01
    t_syn = np.array([1.0, 1001.0, 2001.0, 2002.0], dtype=np.float64) * dt_syn
    with netCDF4.Dataset(partial_nc, "w") as ds_syn:
        ds_syn.createDimension("time", None)
        ds_syn.createDimension("kx", 1)
        ds_syn.createDimension("ky", 1)
        ds_syn.createDimension("kz", 1)
        ds_syn.createDimension("theta", 2)
        g_grp = ds_syn.createGroup("Grids")
        g_grp.createVariable("time", "f8", ("time",))[:] = t_syn
        g_grp.createVariable("kx", "f8", ("kx",))[:] = [0.0]
        g_grp.createVariable("ky", "f8", ("ky",))[:] = [0.1]
        g_grp.createVariable("kz", "f8", ("kz",))[:] = [0.0]
        g_grp.createVariable("theta", "f8", ("theta",))[:] = [-np.pi, 0.0]
        geo_grp = ds_syn.createGroup("Geometry")
        geo_grp.createVariable("jacobian", "f8", ("theta",))[:] = [1.0, 1.0]
        geo_grp.createVariable("grho", "f8", ("theta",))[:] = [1.0, 1.0]
        diag_grp = ds_syn.createGroup("Diagnostics")
        diag_grp.createVariable("Phi2_t", "f8", ("time",))[:] = [1.0, 2.0, 3.0, 3.001]
    partial_rep = extract_kr2_diagnostics(partial_nc)
    assert partial_rep["scalar_metrics"]["dt_collapse_ratio"] == pytest.approx(1.0, abs=1e-12)

    # 4. Inject NaN and Inf into .out.nc and verify strict JSON (NaN/Inf -> null, allow_nan=False)
    out_nc = tmp_path / "strict_check.out.nc"
    with netCDF4.Dataset(out_nc, "a") as ds_mut:
        ds_mut.groups["Diagnostics"].variables["Phi2_t"][-1] = np.nan
        ds_mut.groups["Diagnostics"].variables["HeatFlux_st"][-1, 0] = np.inf
        ds_mut.groups["Diagnostics"].variables["HeatFluxES_st"][-1, 0] = np.nan

    nan_json = tmp_path / "nan_metrics.json"
    rep = extract_kr2_diagnostics(out_nc, metrics_out_path=nan_json, sim_time_target=12.5)
    assert rep["has_nan_or_inf"] is True
    assert rep["converged"] is False
    assert rep["failure_type"] == "DIVERGENCE"
    assert rep["sim_time_target"] == pytest.approx(12.5)
    assert rep["phi2_end"] is None
    assert rep["fluxes"]["heat_flux_es"] is None

    parsed_nan = json.loads(nan_json.read_text(), parse_constant=_reject_non_strict_json)
    assert parsed_nan["converged"] is False
    assert parsed_nan["failure_type"] == "DIVERGENCE"
    assert parsed_nan["sim_time_target"] == pytest.approx(12.5)
    assert parsed_nan["fluxes"]["heat_flux_es"] is None
    assert parsed_nan["scalar_metrics"]["heat_flux_es"] is None
