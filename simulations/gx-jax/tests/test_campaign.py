"""Cloud Batch & GKE Spot TPU Campaign Orchestrator."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

from gx_jax.campaign import (
    SIGTERM_EXIT_CODE,
    PreemptionController,
    SpotPreemptionInterrupt,
    build_campaign_arg_parser,
    campaign_main,
    compatibility_key,
    detect_worker_slice,
    group_compatible_items,
    is_deck_completed,
    load_campaign_manifest,
    main,
    run_campaign,
    slice_campaign_items,
)
from gx_jax.cli import run_cli_simulation
from gx_jax.config import GXConfig, load_config
from gx_jax.fields import init_moments, load_restart_nc, mask_g, save_restart_nc
from gx_jax.geometry_nc import build_vmec_eik_nc, init_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent
W7X_WOUT = REPO_ROOT / "benchmarks" / "linear" / "ITG_w7x" / "wout_w7x.nc"


def _reject_non_strict_json(token: str) -> Any:
    raise AssertionError(f"Non-RFC-8259 JSON token encountered: {token!r}")


def _subprocess_env(
    extra: dict[str, str] | None = None,
    *,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pp}" if existing_pp else str(REPO_ROOT)
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    for k in (
        "BATCH_TASK_INDEX",
        "BATCH_TASK_COUNT",
        "JOB_COMPLETION_INDEX",
        "INDEX_COMPLETION_COUNT",
        "JOB_COMPLETION_TOTAL",
    ):
        env.pop(k, None)
    if extra:
        env.update(extra)
    return env


@pytest.fixture(scope="module")
def jax_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared persistent XLA compilation cache across test subprocesses."""
    return tmp_path_factory.mktemp("s8_jax_cache")


def _assert_close_tol(actual: Any, expected: Any, tol: float, label: str) -> None:
    act = np.asarray(actual)
    exp = np.asarray(expected)
    assert np.all(np.isfinite(act)), f"{label}: actual contains NaN/Inf"
    assert np.all(np.isfinite(exp)), f"{label}: expected contains NaN/Inf"
    scale = max(1.0, float(np.max(np.abs(exp)))) if exp.size > 0 else 1.0
    np.testing.assert_allclose(act, exp, rtol=tol, atol=tol * scale, err_msg=label)


@pytest.fixture(scope="module")
def w7x_eik_nc_8(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a reusable 8-theta W7-X stellarator `.eik.nc` geometry file once per module."""
    out_dir = tmp_path_factory.mktemp("s8_w7x_geo")
    eik_path = out_dir / "w7x_nz8.eik.nc"
    cfg_w7x = GXConfig(
        run_name="w7x",
        nx=6,
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


def _write_compact_deck(
    path: Path,
    *,
    ntheta: int = 4,
    ny: int = 4,
    nx: int = 4,
    nhermite: int = 2,
    nlaguerre: int = 1,
    nspecies: int = 1,
    nonlinear: bool = False,
    scheme: str = "rk3",
    fixed_dt: bool = True,
    dt: float = 0.02,
    nstep: int | None = 6,
    t_max: float | None = None,
    cfl: float = 0.85,
    nwrite: int = 2,
    nsave: int = 2,
    save_for_restart: bool = True,
    tprim: float = 2.8,
    fprim: float = 0.9,
    shat: float = 0.8,
    beta: float = 0.0,
    closure_model: str = "none",
    write_fields: bool = True,
    write_moms: bool = True,
    geo_eik_path: Path | None = None,
) -> Path:
    """Write a compact 1- or 2-species `.in` deck for fast campaign & preemption tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time_lines = [
        f" dt = {dt}",
        " dt_max = 0.05",
        f" cfl = {cfl}",
        f" fixed_dt = {str(fixed_dt).lower()}",
        f' scheme = "{scheme}"',
    ]
    if nstep is not None:
        time_lines.append(f" nstep = {int(nstep)}")
    if t_max is not None:
        time_lines.append(f" t_max = {float(t_max)}")

    if nspecies == 1:
        boltz_block = """[Boltzmann]
 add_Boltzmann_species = true
 Boltzmann_type = "electrons"
 tau_fac = 1.0"""
        species_block = f"""[species]
 z = [ 1.0 ]
 mass = [ 1.0 ]
 dens = [ 1.0 ]
 temp = [ 1.0 ]
 tprim = [ {tprim} ]
 fprim = [ {fprim} ]
 vnewk = [ 0.005 ]
 type = [ "ion" ]"""
    else:
        boltz_block = """[Boltzmann]
 add_Boltzmann_species = false"""
        species_block = f"""[species]
 z = [ 1.0, -1.0 ]
 mass = [ 1.0, 0.02 ]
 dens = [ 1.0, 1.0 ]
 temp = [ 1.0, 1.0 ]
 tprim = [ {tprim}, 1.5 ]
 fprim = [ {fprim}, {fprim} ]
 vnewk = [ 0.002, 0.01 ]
 type = [ "ion", "electron" ]"""

    if geo_eik_path is not None:
        geo_block = f"""[Geometry]
 geo_option = "nc"
 geo_file = "{geo_eik_path}" """
    else:
        geo_block = f"""[Geometry]
 geo_option = "s-alpha"
 eps = 0.18
 Rmaj = 2.77778
 qinp = 1.4
 shat = {shat}"""

    em_flag = "1.0" if beta > 0.0 else "0.0"
    time_section = "\n".join(time_lines)
    content = f"""[Dimensions]
 ntheta = {ntheta}
 nperiod = 1
 ny = {ny}
 nx = {nx}
 nhermite = {nhermite}
 nlaguerre = {nlaguerre}
 nspecies = {nspecies}

[Domain]
 y0 = 10.0
 boundary = "linked"

[Physics]
 beta = {beta}
 nonlinear_mode = {str(nonlinear).lower()}
 fphi = 1.0
 fapar = {em_flag}
 fbpar = {em_flag}
 collisions = true

[Time]
{time_section}

[Initialization]
 init_field = "density"
 init_amp = 1.0e-5
 ikpar_init = 0
 random_seed = 22

{boltz_block}

[Dissipation]
 closure_model = "{closure_model}"
 hypercollisions = true
 nu_hyper_m = 0.1
 p_hyper_m = 2
 hyper = true
 D_hyper = 0.05

{geo_block}

[Diagnostics]
 nwrite = {nwrite}
 omega = true
 free_energy = true
 fluxes = true
 fields = {str(write_fields).lower()}
 moments = {str(write_moms).lower()}

[Restart]
 save_for_restart = {str(save_for_restart).lower()}
 nsave = {nsave}

{species_block}
"""
    path.write_text(content, encoding="utf-8")
    return path


# =============================================================================
# 1. Manifest Formats, Stem Disambiguation & Deterministic Worker Slicing
# =============================================================================


def test_manifest_formats_stem_disambiguation_and_worker_slicing(tmp_path: Path) -> None:
    """Verify .json, .jsonl, .txt, and directory manifests + Cloud Batch / GKE slicing."""
    decks_root = tmp_path / "campaign_decks"
    d_a = _write_compact_deck(decks_root / "sub_a" / "scan.in", tprim=2.2)
    d_b = _write_compact_deck(decks_root / "sub_b" / "scan.in", tprim=2.6)
    d_c = _write_compact_deck(decks_root / "sub_b" / "unique_tem.toml", tprim=3.0)
    d_d = _write_compact_deck(decks_root / "itg_extra.in", tprim=3.4)
    d_e = _write_compact_deck(decks_root / "kbm_extra.in", tprim=3.8)

    # 1a. Directory manifest: deterministic sort & collision-free stems for sub_a/scan & sub_b/scan
    dir_items = load_campaign_manifest(decks_root)
    assert [it.input_path for it in dir_items] == [
        d_d.resolve(),
        d_e.resolve(),
        d_a.resolve(),
        d_b.resolve(),
        d_c.resolve(),
    ]
    dir_stems = [it.stem for it in dir_items]
    assert len(set(dir_stems)) == 5, f"Expected 5 collision-free stems, got {dir_stems}"
    assert "itg_extra" in dir_stems and "unique_tem" in dir_stems

    # 1b. JSON manifest: list, {"decks": ...}, {"items": ...}, {"runs": ...}
    json_list_path = tmp_path / "manifest_list.json"
    json_list_path.write_text(
        json.dumps(
            [
                str(d_a.relative_to(tmp_path)),
                {"input": str(d_b.relative_to(tmp_path)), "name": "custom_b", "tag": "b"},
                {"deck": str(d_c.relative_to(tmp_path))},
            ]
        ),
        encoding="utf-8",
    )
    j_items = load_campaign_manifest(json_list_path)
    assert [it.stem for it in j_items] == ["scan", "custom_b", "unique_tem"]
    assert j_items[1].metadata == {"tag": "b"}

    for wrapper_key in ("decks", "items", "runs"):
        wrapped_p = tmp_path / f"manifest_{wrapper_key}.json"
        wrapped_p.write_text(
            json.dumps({wrapper_key: [str(p.relative_to(tmp_path)) for p in (d_a, d_b, d_c)]}),
            encoding="utf-8",
        )
        w_items = load_campaign_manifest(wrapped_p)
        assert len(w_items) == 3
        assert len({it.stem for it in w_items}) == 3

    # 1c. JSONL manifest with comments and blank lines
    jsonl_path = tmp_path / "manifest.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                "# Stellarator campaign JSONL",
                "",
                json.dumps(str(d_a.relative_to(tmp_path))),
                json.dumps({"input": str(d_b.relative_to(tmp_path)), "name": "scan_b"}),
                "   # inline comment line",
                json.dumps({"path": str(d_d.relative_to(tmp_path))}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    jl_items = load_campaign_manifest(jsonl_path)
    assert [it.stem for it in jl_items] == ["scan", "scan_b", "itg_extra"]

    # 1d. Plain-text .txt manifest with comments and blank lines
    txt_path = tmp_path / "manifest.txt"
    txt_path.write_text(
        f"""
        # Plain text deck list
        {d_a.relative_to(tmp_path)}

        {d_b.relative_to(tmp_path)}
        {d_c.relative_to(tmp_path)}
        {d_d.relative_to(tmp_path)}
        {d_e.relative_to(tmp_path)}
        """,
        encoding="utf-8",
    )
    txt_items = load_campaign_manifest(txt_path)
    assert len(txt_items) == 5
    assert len({it.stem for it in txt_items}) == 5

    # 1e. Worker slice resolution: default, Cloud Batch, GKE Indexed Job, CLI overrides
    assert detect_worker_slice(env={}) == detect_worker_slice(
        task_index=None, task_count=None, env={}
    )
    def_w = detect_worker_slice(env={})
    assert (def_w.task_index, def_w.task_count, def_w.source) == (0, 1, "default")

    cb_w = detect_worker_slice(env={"BATCH_TASK_INDEX": "2", "BATCH_TASK_COUNT": "4"})
    assert (cb_w.task_index, cb_w.task_count, cb_w.source) == (2, 4, "cloud_batch")

    gke_w = detect_worker_slice(env={"JOB_COMPLETION_INDEX": "3", "INDEX_COMPLETION_COUNT": "8"})
    assert (gke_w.task_index, gke_w.task_count, gke_w.source) == (3, 8, "gke_indexed_job")

    # GKE sets JOB_COMPLETION_INDEX automatically while --task-count is supplied on CLI
    gke_cli_w = detect_worker_slice(
        task_count=6,
        env={"JOB_COMPLETION_INDEX": "5"},
    )
    assert (gke_cli_w.task_index, gke_cli_w.task_count) == (5, 6)

    # CLI overrides Cloud Batch & GKE env vars
    cli_override_w = detect_worker_slice(
        task_index=1,
        task_count=3,
        env={
            "BATCH_TASK_INDEX": "7",
            "BATCH_TASK_COUNT": "10",
            "JOB_COMPLETION_INDEX": "4",
            "INDEX_COMPLETION_COUNT": "8",
        },
    )
    assert (cli_override_w.task_index, cli_override_w.task_count) == (1, 3)

    # Invalid bounds raise ValueError
    for bad_idx, bad_cnt in [(-1, 4), (4, 4), (5, 4), (0, 0), (0, -2)]:
        with pytest.raises(ValueError):
            detect_worker_slice(task_index=bad_idx, task_count=bad_cnt, env={})

    # Exact disjoint union across workers for task_count = 1, 2, 3, 5, 7 (including > len(items))
    for k_workers in (1, 2, 3, 5, 7):
        seen_indices: list[int] = []
        seen_stems: set[str] = set()
        for r in range(k_workers):
            sl = slice_campaign_items(txt_items, task_index=r, task_count=k_workers)
            for item in sl:
                assert item.index not in seen_indices
                assert item.stem not in seen_stems
                seen_indices.append(item.index)
                seen_stems.add(item.stem)
        assert sorted(seen_indices) == list(range(len(txt_items)))
        assert len(seen_stems) == len(txt_items)


# =============================================================================
# 2. Structural Auto-Grouping, Scratch Staging & `--skip-completed` Idempotency
# =============================================================================


def test_structural_autogrouping_scratch_staging_and_skip_completed(
    tmp_path: Path,
    w7x_eik_nc_8: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ensemble+singleton auto-grouping, local scratch promotion, and --skip-completed."""
    import gx_jax.campaign as camp_mod

    deck_dir = tmp_path / "mixed_decks"
    # 3 compatible 2-species decks (2 Tokamak ES/EM + 1 W7-X 3D stellarator)
    c0 = _write_compact_deck(
        deck_dir / "tok_es.in",
        ntheta=8,
        ny=6,
        nx=6,
        nhermite=4,
        nlaguerre=2,
        nspecies=2,
        nonlinear=True,
        beta=0.0,
        tprim=2.6,
        nstep=4,
        nwrite=2,
        nsave=2,
    )
    c1 = _write_compact_deck(
        deck_dir / "tok_em_kbm.in",
        ntheta=8,
        ny=6,
        nx=6,
        nhermite=4,
        nlaguerre=2,
        nspecies=2,
        nonlinear=True,
        beta=0.015,
        shat=0.65,
        tprim=3.1,
        nstep=4,
        nwrite=2,
        nsave=2,
    )
    c2 = _write_compact_deck(
        deck_dir / "w7x_em.in",
        ntheta=8,
        ny=6,
        nx=6,
        nhermite=4,
        nlaguerre=2,
        nspecies=2,
        nonlinear=True,
        beta=0.008,
        nstep=4,
        nwrite=2,
        nsave=2,
        geo_eik_path=w7x_eik_nc_8,
    )
    # Structurally distinct singleton deck (1 species adiabatic electrons, linear, nhermite=2)
    s0 = _write_compact_deck(
        deck_dir / "lin_singleton.in",
        ntheta=4,
        ny=4,
        nx=4,
        nhermite=2,
        nlaguerre=1,
        nspecies=1,
        nonlinear=False,
        nstep=4,
        nwrite=2,
        nsave=2,
    )

    manifest_p = tmp_path / "mixed_manifest.json"
    manifest_p.write_text(
        json.dumps([str(p.relative_to(tmp_path)) for p in (c0, s0, c1, c2)]),
        encoding="utf-8",
    )
    items = load_campaign_manifest(manifest_p)

    # With batch_size=4: [c0, c1, c2] form a B=3 ensemble batch, [s0] forms a singleton batch!
    batches_b4 = group_compatible_items(items, batch_size=4)
    assert len(batches_b4) == 2
    assert batches_b4[0].mode == "ensemble"
    assert [it.stem for it in batches_b4[0].items] == ["tok_es", "tok_em_kbm", "w7x_em"]
    assert batches_b4[1].mode == "singleton"
    assert [it.stem for it in batches_b4[1].items] == ["lin_singleton"]
    assert compatibility_key(batches_b4[0].specs[0]) != compatibility_key(batches_b4[1].specs[0])

    # With batch_size=2: [c0, c1] -> ensemble(2), [s0] -> singleton(1), [c2] -> singleton(1)
    batches_b2 = group_compatible_items(items, batch_size=2)
    assert [(b.mode, [it.stem for it in b.items]) for b in batches_b2] == [
        ("ensemble", ["tok_es", "tok_em_kbm"]),
        ("singleton", ["lin_singleton"]),
        ("singleton", ["w7x_em"]),
    ]

    # Spy on run_ensemble & run_cli_simulation to prove scratch staging + dispatch routing
    ens_calls: list[Path] = []
    cli_calls: list[Path] = []
    orig_ens = camp_mod.run_ensemble
    orig_cli = camp_mod.run_cli_simulation

    def _spy_ens(*args: Any, **kwargs: Any) -> dict[str, Any]:
        ens_calls.append(Path(kwargs["output_dir"]))
        return orig_ens(*args, **kwargs)

    def _spy_cli(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cli_calls.append(Path(kwargs["output_path"]))
        return orig_cli(*args, **kwargs)

    monkeypatch.setattr(camp_mod, "run_ensemble", _spy_ens)
    monkeypatch.setattr(camp_mod, "run_cli_simulation", _spy_cli)

    scratch_dir = tmp_path / "local_ssd_scratch"
    output_dir = tmp_path / "gcs_fuse_outputs"

    res1 = run_campaign(
        manifest_p,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        batch_size=4,
        precision="fp64",
        skip_completed=True,
    )
    assert res1["status"] == "completed"
    assert res1["exit_code"] == 0
    assert len(ens_calls) == 1
    assert len(cli_calls) == 1
    # Verify stepping wrote to scratch_dir, NEVER directly to output_dir!
    assert scratch_dir in ens_calls[0].parents
    assert scratch_dir in cli_calls[0].parents
    # Worker stage subdirectories must be cleaned up after promotion
    assert list(scratch_dir.glob("worker_*")) == []
    assert list(output_dir.glob("*.tmp")) == []
    assert list(output_dir.glob(".*.tmp")) == []

    ledger1 = json.loads(
        (output_dir / "campaign_task_0000.json").read_text(),
        parse_constant=_reject_non_strict_json,
    )
    assert ledger1["status"] == "completed"
    assert ledger1["completed_count"] == 4
    assert ledger1["skipped_count"] == 0

    # Compare every promoted deck against standalone run_cli_simulation in fp64 (< 1e-12)
    ref_dir = tmp_path / "standalone_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    mtimes_before: dict[str, int] = {}
    for deck_p in (c0, c1, c2, s0):
        stem = deck_p.stem
        out_nc = output_dir / f"{stem}.out.nc"
        rst_nc = output_dir / f"{stem}.restart.nc"
        met_json = output_dir / f"{stem}.diagnostics_metrics.json"
        assert out_nc.is_file() and rst_nc.is_file() and met_json.is_file()
        mtimes_before[stem] = out_nc.stat().st_mtime_ns

        met = json.loads(met_json.read_text(), parse_constant=_reject_non_strict_json)
        assert met["status"] == "completed"
        assert met["converged"] is True
        assert met["has_nan_or_inf"] is False

        ref_nc = ref_dir / f"{stem}.out.nc"
        ref_rst = ref_dir / f"{stem}.restart.nc"
        ref_res = run_cli_simulation(
            deck_p,
            output_path=ref_nc,
            save_restart_path=ref_rst,
            precision="fp64",
        )
        _, grids_d, _ = init_geometry(load_config(deck_p))
        g_camp, t_camp = load_restart_nc(rst_nc, grids_d, dtype=jnp.complex128)
        _assert_close_tol(t_camp, ref_res["t_final"], 1.0e-12, f"{stem}:t_final")
        _assert_close_tol(g_camp, ref_res["g_final"], 1.0e-12, f"{stem}:g_final")
        with nc.Dataset(out_nc, "r") as ds_c, nc.Dataset(ref_nc, "r") as ds_r:
            _assert_close_tol(
                ds_c.groups["Grids"].variables["time"][:],
                ds_r.groups["Grids"].variables["time"][:],
                1.0e-12,
                f"{stem}:Grids/time",
            )
            _assert_close_tol(
                ds_c.groups["Diagnostics"].variables["Phi2_t"][:],
                ds_r.groups["Diagnostics"].variables["Phi2_t"][:],
                1.0e-6,
                f"{stem}:Phi2_t",
            )

    # --skip-completed idempotency: second run skips all 4 decks without invoking solvers!
    ens_calls.clear()
    cli_calls.clear()
    res2 = run_campaign(
        manifest_p,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        batch_size=4,
        precision="fp64",
        skip_completed=True,
    )
    assert res2["status"] == "completed"
    assert len(ens_calls) == 0 and len(cli_calls) == 0
    assert res2["ledger"]["skipped_count"] == 4
    assert res2["ledger"]["completed_count"] == 0
    for stem, prev_mtime in mtimes_before.items():
        assert (output_dir / f"{stem}.out.nc").stat().st_mtime_ns == prev_mtime

    # If one deck's metrics show incomplete t_final < sim_time_target, ONLY that deck re-runs!
    bad_met_p = output_dir / "lin_singleton.diagnostics_metrics.json"
    bad_met = json.loads(bad_met_p.read_text())
    bad_met["t_final"] = 0.001
    bad_met_p.write_text(json.dumps(bad_met), encoding="utf-8")
    (output_dir / "lin_singleton.restart.nc").unlink(missing_ok=True)
    assert is_deck_completed(output_dir, "lin_singleton")[0] is False

    res3 = run_campaign(
        manifest_p,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        batch_size=4,
        precision="fp64",
        skip_completed=True,
        auto_resume=False,
    )
    assert res3["ledger"]["skipped_count"] == 3
    assert res3["ledger"]["completed_decks"] == ["lin_singleton"]
    assert len(ens_calls) == 0 and len(cli_calls) == 1


def test_fields_save_restart_nc_fp64_lossless_and_fp32_dtype_preservation(
    tmp_path: Path,
) -> None:
    """Verify save_restart_nc writes f8 iff complex128 and keeps f4 for complex64."""
    deck = _write_compact_deck(tmp_path / "dtype_check.in")
    cfg, grids, geo = init_geometry(load_config(deck))

    g32 = init_moments(cfg, grids, geo, dtype=jnp.complex64)
    g64_base = init_moments(cfg, grids, geo, dtype=jnp.complex128)
    g64 = mask_g(g64_base + (1.0e-10 - 2.5e-10j) * jnp.ones_like(g64_base), grids)

    p32 = tmp_path / "r32.restart.nc"
    p64 = tmp_path / "r64.restart.nc"
    save_restart_nc(p32, g32, 0.125, grids)
    save_restart_nc(p64, g64, 0.125, grids)

    with nc.Dataset(p32, "r") as ds32:
        assert ds32.variables["G"].datatype == np.dtype("float32")
    with nc.Dataset(p64, "r") as ds64:
        assert ds64.variables["G"].datatype == np.dtype("float64")

    g32_rt, t32_rt = load_restart_nc(p32, grids, dtype=jnp.complex64)
    g64_rt, t64_rt = load_restart_nc(p64, grids, dtype=jnp.complex128)
    assert t32_rt == 0.125 and t64_rt == 0.125
    assert np.array_equal(np.asarray(g32_rt), np.asarray(g32))
    assert np.array_equal(np.asarray(g64_rt), np.asarray(g64))


def test_preemption_controller_restores_signal_handlers_in_finally() -> None:
    """Verify PreemptionController RLock reentrancy and handler restoration in finally."""
    assert main is campaign_main
    parser = build_campaign_arg_parser()
    parsed = parser.parse_args(["manifest.json", "--skip-completed", "--auto-resume"])
    assert parsed.skip_completed is True
    assert parsed.auto_resume is True
    assert parsed.distributed is False

    orig_term = signal.getsignal(signal.SIGTERM)
    orig_int = signal.getsignal(signal.SIGINT)

    custom_called: list[int] = []

    def _custom_handler(signum: int, _frame: Any) -> None:
        custom_called.append(signum)

    signal.signal(signal.SIGTERM, _custom_handler)
    try:
        with pytest.raises(SpotPreemptionInterrupt) as exc_info:
            with PreemptionController(trap_signals=True) as ctrl:
                # C2: Verify reentrant RLock does not self-deadlock if signal arrives
                # on main thread while ctrl._lock is already held inside shield()/check()!
                inside_completed = False
                with ctrl.shield():
                    with ctrl._lock:
                        os.kill(os.getpid(), signal.SIGTERM)
                    assert ctrl.pending_signum == int(signal.SIGTERM)
                    inside_completed = True
                assert inside_completed is True  # pragma: no cover
        assert inside_completed is True
        assert exc_info.value.exit_code == SIGTERM_EXIT_CODE == 143
        assert signal.getsignal(signal.SIGTERM) is _custom_handler
        assert signal.getsignal(signal.SIGINT) == orig_int
    finally:
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT, orig_int)


# =============================================================================
# 3. Multi-Device Subprocess Slicing & Batch Sharding (`XLA_FLAGS` Isolated)
# =============================================================================


def test_multidevice_4cpu_worker_slicing_and_batch_sharding_subprocess(
    tmp_path: Path,
    jax_cache_dir: Path,
) -> None:
    """Verify 2 Cloud Batch workers x 4 virtual CPU devices cover 5 decks seamlessly."""
    decks_dir = tmp_path / "md_decks"
    for i in range(4):
        _write_compact_deck(
            decks_dir / f"compat_{i}.in",
            nspecies=2,
            nhermite=4,
            nlaguerre=2,
            ntheta=4,
            tprim=2.4 + 0.3 * i,
            nstep=3,
            nwrite=1,
            nsave=2,
        )
    _write_compact_deck(
        decks_dir / "singleton_odd.in",
        nspecies=1,
        nhermite=4,
        nlaguerre=1,
        ntheta=4,
        nstep=3,
        nwrite=1,
        nsave=2,
    )

    out_dir = tmp_path / "md_outputs"
    scratch_dir = tmp_path / "md_scratch"

    for worker_idx in (0, 1):
        env = _subprocess_env(
            {
                "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
                "BATCH_TASK_INDEX": str(worker_idx),
                "BATCH_TASK_COUNT": "2",
            },
            cache_dir=jax_cache_dir,
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "gx_jax.campaign",
                str(decks_dir),
                "--scratch-dir",
                str(scratch_dir),
                "--output-dir",
                str(out_dir),
                "--batch-size",
                "4",
                "--batch-devices",
                "4",
                "--mesh",
                "2x2",
                "--precision",
                "fp64",
                "--skip-completed",
                "--auto-resume",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert proc.returncode == 0, (
            f"Worker {worker_idx} failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    l0 = json.loads((out_dir / "campaign_task_0000.json").read_text())
    l1 = json.loads((out_dir / "campaign_task_0001.json").read_text())
    assert l0["worker_source"] == "cloud_batch"
    assert l1["worker_source"] == "cloud_batch"
    assert l0["assigned_deck_count"] + l1["assigned_deck_count"] == 5
    all_completed = set(l0["completed_decks"]) | set(l1["completed_decks"])
    assert all_completed == {
        "compat_0",
        "compat_1",
        "compat_2",
        "compat_3",
        "singleton_odd",
    }


# =============================================================================
# 4. Real Subprocess `SIGTERM` Spot Preemption -> `143` -> `--auto-resume` Parity
# =============================================================================


@pytest.mark.parametrize(
    ("precision", "tol"),
    [
        ("fp64", 1.0e-12),
        ("fp32", 1.0e-6),
    ],
)
def test_real_subprocess_sigterm_preemption_143_and_auto_resume_equivalence(
    tmp_path: Path,
    jax_cache_dir: Path,
    precision: str,
    tol: float,
) -> None:
    """Verify real SIGTERM -> exit 143 -> --auto-resume matches uninterrupted 0 -> t_max.

    Exercises in a real OS subprocess:
      1. Deck 0 (`early_done`, singleton) completes BEFORE `SIGTERM` and is skipped on resume.
      2. Batch 1 (`B=2` ensemble) runs TWO decks simultaneously with `nwrite=3, nsave=2`:
         - `tmax_bounded`: bounded by `t_max = 1.60` (`dt = 0.02`, 80 steps, `nstep = 5000`)
         - `nstep_bounded`: bounded ONLY by `nstep = 80` (`dt = 0.02`, `t_max = 1e20`)
      3. Once periodic `.restart.nc` checkpoints appear in `--output-dir`, sends `SIGTERM`.
      4. Asserts exit code `143`, zero `.tmp` leftovers, valid promoted `.restart.nc` at
         `0 < t0 < 1.60`, and partial ledger `campaign_task_0000.json` (`status="preempted"`).
      5. Relaunches `--auto-resume` to `t_max` and asserts `< 1e-12` (`fp64`) / `< 1e-6` (`fp32`)
         equivalence against an uninterrupted `0 -> t_max` reference run!
    """
    case_dir = tmp_path / f"preempt_{precision}"
    decks_dir = case_dir / "decks"

    # Singleton deck that finishes quickly in Batch 0 before Batch 1 hits steady stepping
    d_early = _write_compact_deck(
        decks_dir / "early_done.in",
        nspecies=1,
        nhermite=2,
        nlaguerre=1,
        ntheta=4,
        nonlinear=False,
        fixed_dt=True,
        dt=0.02,
        nstep=4,
        t_max=None,
        nwrite=2,
        nsave=2,
    )
    # Compatible 2-species nonlinear decks for Batch 1: one t_max-bounded, one nstep-bounded!
    total_steps = 90
    dt_val = 0.02
    target_t = total_steps * dt_val  # 1.80

    d_tmax = _write_compact_deck(
        decks_dir / "tmax_bounded.in",
        nspecies=2,
        nhermite=2,
        nlaguerre=1,
        ntheta=4,
        nonlinear=True,
        fixed_dt=True,
        dt=dt_val,
        nstep=10000,
        t_max=target_t,
        nwrite=3,
        nsave=2,
        tprim=2.7,
    )
    d_nstep = _write_compact_deck(
        decks_dir / "nstep_bounded.in",
        nspecies=2,
        nhermite=2,
        nlaguerre=1,
        ntheta=4,
        nonlinear=True,
        fixed_dt=True,
        dt=dt_val,
        nstep=total_steps,
        t_max=None,
        nwrite=3,
        nsave=2,
        tprim=3.2,
    )

    manifest_p = case_dir / "campaign.json"
    manifest_p.write_text(
        json.dumps([str(d_early), str(d_tmax), str(d_nstep)]),
        encoding="utf-8",
    )

    out_preempt = case_dir / "out_preempt_resume"
    scratch_preempt = case_dir / "scratch_preempt"
    env = _subprocess_env(cache_dir=jax_cache_dir)

    cmd = [
        sys.executable,
        "-m",
        "gx_jax.campaign",
        str(manifest_p),
        "--scratch-dir",
        str(scratch_preempt),
        "--output-dir",
        str(out_preempt),
        "--batch-size",
        "4",
        "--precision",
        precision,
        "--nsave",
        "2",
        "--skip-completed",
        "--auto-resume",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rst_tmax = out_preempt / "tmax_bounded.restart.nc"
    rst_nstep = out_preempt / "nstep_bounded.restart.nc"
    sent_sigterm = False
    deadline = time.monotonic() + 60.0

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if rst_tmax.is_file() and rst_nstep.is_file():
            # Wait briefly for at least a couple checkpoints (> step 1) if possible, then SIGTERM
            time.sleep(0.03)
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                sent_sigterm = True
            break
        time.sleep(0.005)

    stdout1, stderr1 = proc.communicate(timeout=45)
    assert sent_sigterm, (
        f"Campaign finished before SIGTERM could be sent (rc={proc.returncode}):\n"
        f"STDOUT:\n{stdout1}\nSTDERR:\n{stderr1}"
    )
    assert proc.returncode == 143, (
        f"Expected Spot preemption exit code 143, got {proc.returncode}:\n"
        f"STDOUT:\n{stdout1}\nSTDERR:\n{stderr1}"
    )

    # Zero .tmp partial files in output_dir or scratch_dir!
    assert list(out_preempt.glob("*.tmp")) == []
    assert list(out_preempt.glob(".*.tmp")) == []
    if scratch_preempt.exists():
        assert list(scratch_preempt.rglob("*.tmp")) == []
        assert list(scratch_preempt.glob("worker_*")) == []

    # Verify partial worker ledger flushed on SIGTERM
    ledger_p = out_preempt / "campaign_task_0000.json"
    assert ledger_p.is_file()
    preempt_ledger = json.loads(ledger_p.read_text(), parse_constant=_reject_non_strict_json)
    assert preempt_ledger["status"] == "preempted"
    assert preempt_ledger["exit_code"] == 143
    assert preempt_ledger["signal"] == "SIGTERM"
    assert preempt_ledger["completed_decks"] == ["early_done"]
    ckpt_stems = {c["stem"] for c in preempt_ledger["checkpointed_decks"]}
    assert ckpt_stems == {"tmax_bounded", "nstep_bounded"}

    for deck_p in (d_tmax, d_nstep):
        stem = deck_p.stem
        _, grids_d, _ = init_geometry(load_config(deck_p))
        g_mid, t_mid = load_restart_nc(
            out_preempt / f"{stem}.restart.nc",
            grids_d,
            dtype=jnp.complex128 if precision == "fp64" else jnp.complex64,
        )
        assert np.all(np.isfinite(np.asarray(g_mid)))
        assert 0.0 < t_mid < target_t, f"Expected mid-run 0 < t_mid < {target_t}, got {t_mid}"

    early_mtime = (out_preempt / "early_done.out.nc").stat().st_mtime_ns

    # Relaunch replacement Spot worker with --auto-resume to finish t0 -> t_max!
    proc2 = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc2.returncode == 0, (
        f"Auto-resume subprocess failed (rc={proc2.returncode}):\n"
        f"STDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"
    )
    assert list(out_preempt.glob(".*.tmp")) == []

    final_ledger = json.loads(ledger_p.read_text(), parse_constant=_reject_non_strict_json)
    assert final_ledger["status"] == "completed"
    assert final_ledger["exit_code"] == 0
    assert final_ledger["skipped_decks"] == ["early_done"]
    assert set(final_ledger["resumed_decks"]) == {"tmax_bounded", "nstep_bounded"}
    assert set(final_ledger["completed_decks"]) == {"tmax_bounded", "nstep_bounded"}
    assert (out_preempt / "early_done.out.nc").stat().st_mtime_ns == early_mtime

    # Reference uninterrupted 0 -> t_max campaign run
    out_ref = case_dir / "out_uninterrupted_ref"
    ref_res = run_campaign(
        manifest_p,
        output_dir=out_ref,
        scratch_dir=case_dir / "scratch_ref",
        batch_size=4,
        precision=precision,
        nsave=50,
        auto_resume=False,
        skip_completed=False,
    )
    assert ref_res["status"] == "completed"

    c_dtype = jnp.complex128 if precision == "fp64" else jnp.complex64
    nc_f4_tol = 1.0e-6  # .out.nc diagnostic spectra are stored as NetCDF4 float32 ('f4')

    for deck_p in (d_early, d_tmax, d_nstep):
        stem = deck_p.stem
        _, grids_d, _ = init_geometry(load_config(deck_p))
        g_res, t_res = load_restart_nc(out_preempt / f"{stem}.restart.nc", grids_d, dtype=c_dtype)
        g_ref, t_ref = load_restart_nc(out_ref / f"{stem}.restart.nc", grids_d, dtype=c_dtype)
        _assert_close_tol(t_res, t_ref, tol, f"{precision}:{stem}:t_final")
        _assert_close_tol(g_res, g_ref, tol, f"{precision}:{stem}:G_final")

        m_res = json.loads(
            (out_preempt / f"{stem}.diagnostics_metrics.json").read_text(),
            parse_constant=_reject_non_strict_json,
        )
        m_ref = json.loads(
            (out_ref / f"{stem}.diagnostics_metrics.json").read_text(),
            parse_constant=_reject_non_strict_json,
        )
        assert m_res["nstep"] == m_ref["nstep"] == (4 if stem == "early_done" else total_steps)
        assert m_res["n_time"] == m_ref["n_time"]
        _assert_close_tol(m_res["t_final"], m_ref["t_final"], tol, f"{precision}:{stem}:m_t")
        _assert_close_tol(
            m_res["phi2_end"], m_ref["phi2_end"], nc_f4_tol, f"{precision}:{stem}:phi2_end"
        )
        _assert_close_tol(
            m_res["Q_total"], m_ref["Q_total"], nc_f4_tol, f"{precision}:{stem}:Q_total"
        )

        with (
            nc.Dataset(out_preempt / f"{stem}.out.nc", "r") as ds_res,
            nc.Dataset(out_ref / f"{stem}.out.nc", "r") as ds_ref,
        ):
            # Grids/time is stored as f8 in .out.nc -> must match to full fp64/fp32 tol!
            _assert_close_tol(
                ds_res.groups["Grids"].variables["time"][:],
                ds_ref.groups["Grids"].variables["time"][:],
                tol,
                f"{precision}:{stem}:Grids/time",
            )
            for diag_var in (
                "Phi2_t",
                "Wg_st",
                "HeatFlux_st",
                "ParticleFlux_st",
                "TurbulentHeating_st",
                "omega_kxkyt",
                "Phi",
                "Density",
            ):
                # Finite-difference log(phi / phi_old) / dt amplifies 1-ULP fp32 roundoff by 1/dt
                diag_tol = (
                    1.0e-3 if (precision == "fp32" and diag_var == "omega_kxkyt") else nc_f4_tol
                )
                _assert_close_tol(
                    ds_res.groups["Diagnostics"].variables[diag_var][:],
                    ds_ref.groups["Diagnostics"].variables[diag_var][:],
                    diag_tol,
                    f"{precision}:{stem}:Diag/{diag_var}",
                )


def test_singleton_adaptive_cfl_sigterm_and_auto_resume_subprocess(
    tmp_path: Path,
    jax_cache_dir: Path,
) -> None:
    """Verify singleton (`--batch-size 1`) adaptive-CFL SIGTERM -> 143 -> --auto-resume in fp64."""
    deck = _write_compact_deck(
        tmp_path / "adaptive_single.in",
        nspecies=1,
        nhermite=2,
        nlaguerre=1,
        ntheta=4,
        nonlinear=True,
        fixed_dt=False,
        dt=0.005,
        nstep=5000,
        t_max=0.35,
        cfl=0.8,
        nwrite=2,
        nsave=2,
    )
    out_dir = tmp_path / "single_out"
    scratch_dir = tmp_path / "single_scratch"
    env = _subprocess_env(cache_dir=jax_cache_dir)
    cmd = [
        sys.executable,
        "-m",
        "gx_jax.campaign",
        str(deck),
        "--scratch-dir",
        str(scratch_dir),
        "--output-dir",
        str(out_dir),
        "--batch-size",
        "1",
        "--precision",
        "fp64",
        "--nsave",
        "2",
        "--skip-completed",
        "--auto-resume",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rst_p = out_dir / "adaptive_single.restart.nc"
    sent = False
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if rst_p.is_file():
            time.sleep(0.02)
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                sent = True
            break
        time.sleep(0.005)
    out1, err1 = proc.communicate(timeout=45)
    assert sent, f"Adaptive singleton finished too quickly:\n{out1}\n{err1}"
    assert proc.returncode == 143

    proc2 = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=90, check=False
    )
    assert proc2.returncode == 0, f"Resume failed:\n{proc2.stdout}\n{proc2.stderr}"

    ref_nc = tmp_path / "ref_adaptive.out.nc"
    ref_rst = tmp_path / "ref_adaptive.restart.nc"
    ref_res = run_cli_simulation(
        deck, output_path=ref_nc, save_restart_path=ref_rst, precision="fp64"
    )
    _, grids_d, _ = init_geometry(load_config(deck))
    g_res, t_res = load_restart_nc(rst_p, grids_d, dtype=jnp.complex128)
    _assert_close_tol(t_res, ref_res["t_final"], 1.0e-12, "adaptive_single:t_final")
    _assert_close_tol(g_res, ref_res["g_final"], 1.0e-12, "adaptive_single:G_final")


def test_c1_c5_c6_c7_adaptive_double_preemption_sigkill_and_cli_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify C1 (non-multiple nsave + 2 preemptions on adaptive nstep deck), C5, C6, & C7."""
    import gx_jax.campaign as camp_mod

    # --- Part A (C1): Adaptive-dt, nstep-bounded, nwrite=4, nsave=3 (not a multiple!)
    # across TWO sequential preemptions PLUS mid-prealignment SIGTERM resilience! ---
    d_adapt1 = _write_compact_deck(
        tmp_path / "adapt_pair" / "a1.in",
        nspecies=1,
        nhermite=2,
        nlaguerre=1,
        ntheta=4,
        nonlinear=True,
        fixed_dt=False,
        dt=0.005,
        nstep=13,
        t_max=None,
        cfl=0.8,
        nwrite=4,
        nsave=3,
        tprim=2.8,
    )
    d_adapt2 = _write_compact_deck(
        tmp_path / "adapt_pair" / "a2.in",
        nspecies=1,
        nhermite=2,
        nlaguerre=1,
        ntheta=4,
        nonlinear=True,
        fixed_dt=False,
        dt=0.005,
        nstep=13,
        t_max=None,
        cfl=0.72,
        nwrite=4,
        nsave=3,
        tprim=3.3,
    )
    pair_dir = tmp_path / "adapt_pair"
    out_dp = tmp_path / "out_double_preempt"
    sc_dp = tmp_path / "scratch_double_preempt"

    # C7: Seed stale SIGKILL .tmp litter for task 0 (must be swept) and task 1 (must survive)
    out_dp.mkdir(parents=True, exist_ok=True)
    stale_t0 = out_dp / ".a1.out.nc.t0.p99999.tmp"
    live_t1 = out_dp / ".a2.out.nc.t1.p88888.tmp"
    stale_t0.write_bytes(b"corrupt_sigkill_leftover")
    live_t1.write_bytes(b"concurrent_worker_1_inflight")

    orig_run_ens = camp_mod.run_ensemble

    # Preemption #1: let run_ensemble write periodic checkpoint at k=5 (since nsave=3 ->
    # aligned nsave_eff=4 saves at k=1, 5, 9, 13!), then raise SpotPreemptionInterrupt(SIGTERM)!
    def _preempt_after_k5(*args: Any, **kwargs: Any) -> dict[str, Any]:
        specs_arg = args[0]
        mod_specs = [
            camp_mod.EnsembleMemberSpec(
                cfg=camp_mod.replace(s.cfg, nstep=6),
                grids=s.grids,
                geo=s.geo,
                input_path=s.input_path,
                name=s.name,
            )
            for s in specs_arg
        ]
        orig_run_ens(mod_specs, *args[1:], **kwargs)
        # Keep only the periodic checkpoint at k=5 by running 5 steps into canonical .restart.nc
        mod_specs_5 = [
            camp_mod.EnsembleMemberSpec(
                cfg=camp_mod.replace(s.cfg, nstep=5),
                grids=s.grids,
                geo=s.geo,
                input_path=s.input_path,
                name=s.name,
            )
            for s in specs_arg
        ]
        orig_run_ens(mod_specs_5, *args[1:], **kwargs)
        raise SpotPreemptionInterrupt(signal.SIGTERM)

    monkeypatch.setattr(camp_mod, "run_ensemble", _preempt_after_k5)
    r_p1 = run_campaign(
        pair_dir,
        output_dir=out_dp,
        scratch_dir=sc_dp,
        batch_size=4,
        precision="fp64",
        auto_resume=True,
        trap_signals=False,
    )
    assert r_p1["status"] == "preempted" and r_p1["exit_code"] == 143
    assert not stale_t0.exists(), "C7: task 0 stale .tmp file must be swept on startup"
    assert live_t1.is_file(), "C7: concurrent task 1 .tmp file must not be touched"
    live_t1.unlink()

    _, grids_a1, _ = init_geometry(load_config(d_adapt1))
    _, t_ckpt1 = load_restart_nc(out_dp / "a1.restart.nc", grids_a1, dtype=jnp.complex128)

    # Sub-check C1(a): SIGTERM arriving RIGHT AFTER pre-alignment (k=5 -> k_aligned=8) BEFORE
    # any new periodic save in run_ensemble MUST NOT overwrite out_dp/a1.restart.nc with k=8!
    def _preempt_immediately(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise SpotPreemptionInterrupt(signal.SIGTERM)

    monkeypatch.setattr(camp_mod, "run_ensemble", _preempt_immediately)
    r_p_mid = run_campaign(
        pair_dir,
        output_dir=out_dp,
        scratch_dir=sc_dp,
        batch_size=4,
        precision="fp64",
        auto_resume=True,
        trap_signals=False,
    )
    assert r_p_mid["status"] == "preempted" and r_p_mid["exit_code"] == 143
    _, t_ckpt_after_mid = load_restart_nc(out_dp / "a1.restart.nc", grids_a1, dtype=jnp.complex128)
    assert t_ckpt_after_mid == t_ckpt1, "Pre-aligned source must never be salvaged!"

    # Preemption #2: Resume from k=5 -> pre-aligns 3 steps to k_aligned=8 -> runs 1 step to k=9
    # (periodic checkpoint at c_local=1 => global step 9!) -> preempts again!
    def _preempt_after_k9(*args: Any, **kwargs: Any) -> dict[str, Any]:
        specs_arg = args[0]
        # Remaining nstep after k_aligned=8 should be 13 - 8 = 5!
        for s in specs_arg:
            assert int(s.cfg.nstep) == 5, (
                f"Expected rem_nstep=5 after k_aligned=8, got {s.cfg.nstep}"
            )
        mod_specs_1 = [
            camp_mod.EnsembleMemberSpec(
                cfg=camp_mod.replace(s.cfg, nstep=1),
                grids=s.grids,
                geo=s.geo,
                input_path=s.input_path,
                name=s.name,
            )
            for s in specs_arg
        ]
        orig_run_ens(mod_specs_1, *args[1:], **kwargs)
        raise SpotPreemptionInterrupt(signal.SIGTERM)

    monkeypatch.setattr(camp_mod, "run_ensemble", _preempt_after_k9)
    r_p2 = run_campaign(
        pair_dir,
        output_dir=out_dp,
        scratch_dir=sc_dp,
        batch_size=4,
        precision="fp64",
        auto_resume=True,
        trap_signals=False,
    )
    assert r_p2["status"] == "preempted" and r_p2["exit_code"] == 143
    _, t_ckpt2 = load_restart_nc(out_dp / "a1.restart.nc", grids_a1, dtype=jnp.complex128)
    assert t_ckpt2 > t_ckpt1

    # Final Resume #2: k=9 -> pre-aligns 3 steps to k_aligned=12 -> runs final 1 step to k=13!
    monkeypatch.setattr(camp_mod, "run_ensemble", orig_run_ens)
    r_fin = run_campaign(
        pair_dir,
        output_dir=out_dp,
        scratch_dir=sc_dp,
        batch_size=4,
        precision="fp64",
        auto_resume=True,
        trap_signals=False,
    )
    assert r_fin["status"] == "completed" and r_fin["exit_code"] == 0
    assert r_fin["ledger"]["estimated_resume_decks"] == []

    # Compare against uninterrupted 13-step adaptive-dt reference run (< 1e-12 in fp64!)
    out_ref_dp = tmp_path / "out_ref_dp"
    run_campaign(
        pair_dir,
        output_dir=out_ref_dp,
        scratch_dir=tmp_path / "sc_ref_dp",
        batch_size=4,
        precision="fp64",
        auto_resume=False,
        skip_completed=False,
        trap_signals=False,
    )
    for stem, deck_p in [("a1", d_adapt1), ("a2", d_adapt2)]:
        _, grids_s, _ = init_geometry(load_config(deck_p))
        g_dp, t_dp = load_restart_nc(out_dp / f"{stem}.restart.nc", grids_s, dtype=jnp.complex128)
        g_rf, t_rf = load_restart_nc(
            out_ref_dp / f"{stem}.restart.nc", grids_s, dtype=jnp.complex128
        )
        _assert_close_tol(t_dp, t_rf, 1.0e-12, f"double_preempt:{stem}:t_final")
        _assert_close_tol(g_dp, g_rf, 1.0e-12, f"double_preempt:{stem}:G_final")
        met_dp = json.loads((out_dp / f"{stem}.diagnostics_metrics.json").read_text())
        assert met_dp["nstep"] == 13
        assert met_dp["resume_step_source"] == "frame_match"
        with (
            nc.Dataset(out_dp / f"{stem}.out.nc", "r") as ds_dp,
            nc.Dataset(out_ref_dp / f"{stem}.out.nc", "r") as ds_rf,
        ):
            _assert_close_tol(
                ds_dp.groups["Grids"].variables["time"][:],
                ds_rf.groups["Grids"].variables["time"][:],
                1.0e-12,
                f"double_preempt:{stem}:Grids/time",
            )

    # --- Part B (C1d): Hard SIGKILL simulation (mirrored .restart.nc with missing .out.nc)
    # records resume_step_source="estimated" explicitly in ledger & metrics! ---
    out_sigkill = tmp_path / "out_sigkill"
    out_sigkill.mkdir(parents=True, exist_ok=True)
    # Copy valid k=9 checkpoint without .out.nc to simulate SIGKILL before handler ran
    (out_sigkill / "a1.restart.nc").write_bytes((out_ref_dp / "a1.restart.nc").read_bytes())
    # Give it t_ckpt < final so it runs remaining step(s): use t_ckpt2 from k=9!
    save_restart_nc(
        out_sigkill / "a1.restart.nc",
        load_restart_nc(out_dp / "a1.restart.nc", grids_a1, dtype=jnp.complex128)[0],
        float(t_ckpt2),
        grids_a1,
    )
    r_est = run_campaign(
        d_adapt1,
        output_dir=out_sigkill,
        scratch_dir=tmp_path / "sc_sigkill",
        batch_size=1,
        precision="fp64",
        auto_resume=True,
        trap_signals=False,
    )
    assert r_est["ledger"]["estimated_resume_decks"] == ["a1"]
    met_est = json.loads((out_sigkill / "a1.diagnostics_metrics.json").read_text())
    assert met_est["resume_step_source"] == "estimated"

    # --- Part C (C5 & C6): --distributed requires explicit --task-index/--task-count,
    # and RuntimeError is NOT swallowed into parser.error(SystemExit(2))! ---
    with pytest.raises(SystemExit) as exc_dist:
        campaign_main([str(d_adapt1), "--distributed", "-o", str(tmp_path / "dist_err")])
    assert exc_dist.value.code == 2

    def _boom_runtime(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError("Simulated XLA OOM failure")

    monkeypatch.setattr(camp_mod, "run_campaign", _boom_runtime)
    with pytest.raises(RuntimeError, match="Simulated XLA OOM failure"):
        campaign_main([str(d_adapt1), "-o", str(tmp_path / "oom_out")])


# =============================================================================
# 5. Turnkey Cloud Deployment Templates (`deploy/*`) Validation (Stdlib Only)
# =============================================================================


def test_cloud_batch_gke_kueue_yaml_and_readme_templates() -> None:
    """Validate integrated simulations/gcp TPU runtime config, ComputeClasses, & gx-jax README."""
    gcp_dir = REPO_ROOT.parent / "gcp"
    batch_cfg_path = gcp_dir / "config.batch.example.json"
    compute_classes_path = gcp_dir / "k8s" / "compute-classes.yaml"
    dockerfile_path = gcp_dir / "Dockerfile.gx-jax-tpu"
    readme_path = REPO_ROOT / "README.md"

    assert batch_cfg_path.is_file()
    assert compute_classes_path.is_file()
    assert dockerfile_path.is_file()
    assert readme_path.is_file()

    batch_cfg = json.loads(
        batch_cfg_path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_strict_json,
    )
    assert batch_cfg.get("accelerator_family") in ("gpu", "tpu")
    assert "gx-jax-tpu" in batch_cfg.get("gx_tpu_container_image", "")

    cc_text = compute_classes_path.read_text(encoding="utf-8")
    assert "\t" not in cc_text
    for expected in ("gx-tpu-v6e", "gx-tpu-v5p", "tpu-v6e-slice", "tpu-v5p-slice"):
        assert expected in cc_text

    docker_text = dockerfile_path.read_text(encoding="utf-8")
    assert "gx_jax._verify_fp32_hlo" in docker_text
    assert "jax[tpu]" in docker_text

    readme_text = readme_path.read_text(encoding="utf-8")
    for required_snippet in ("gx-jax", "gx-jax-campaign", "THIRD_PARTY_NOTICES.md", "v6e"):
        assert required_snippet in readme_text, f"Missing {required_snippet!r} in README.md"
