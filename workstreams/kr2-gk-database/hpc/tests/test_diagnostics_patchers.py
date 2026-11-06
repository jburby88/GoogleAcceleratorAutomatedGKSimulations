"""Tests for diagnostics_adapter.py (PHYS-01/02, DOM-02/03/04, strictness, segments, checkpoints)
and the GX/GENE patchers (PHYS-04, atomic writes, strict values, key allowlist)."""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

import f90nml
import pytest
import tomlkit

from tacc_exec import diagnostics_adapter as da
from tacc_exec.diagnostics_adapter import (
    DiagnosticsSettings,
    archived_log_segments,
    consume_stale_diagnostics_metrics,
    find_checkpoint_files,
    inspect_run_directory,
)
from tacc_exec.patchers import (
    patch_gene_parameters,
    patch_gx_toml,
    read_gene_parameters,
    read_gx_parameters,
    read_gx_run_controls,
)
from tacc_exec.timeseries import dt_collapse_stats, flux_window_stats

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_GX_IN = (
    REPO_ROOT
    / "workstreams"
    / "kr2-gk-database"
    / "gcp"
    / "benchmarks"
    / "cyclone_itg_converged"
    / "gx.in"
)
MINIMAL_GX_IN = "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n[Time]\ndt = 0.02\ncfl = 0.8\n"


def _gx_line(step: int, t: float, q: float, dt: float = 1.0e-2, gam: float = 0.1) -> str:
    return (
        f"run_name: Step {step:6d}: Time = {t:10.5f}  dt = {dt:.6e}   Phi**2 = 1.0e-03   "
        f"Q_i = {q:.9e}   Q_e = 0.000e+00   Gam_i = {gam:.6e}"
    )


def _gx_run(tmp_path: Path, gx_in: str = MINIMAL_GX_IN, name: str = "run") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "gx.in").write_text(gx_in, encoding="utf-8")
    return run_dir


def _write_log(path: Path, samples: list[tuple[float, float]], dt: float = 1.0e-2) -> None:
    lines = [_gx_line(i, t, q, dt) for i, (t, q) in enumerate(samples, start=1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plateau(t: float) -> float:
    return 10.0 * t / 50.0 if t < 50.0 else 10.0 + 0.5 * math.sin(t)


# --------------------------------------------------------------------------------------------
# PHYS-02: post-transient time average + drift gate
# --------------------------------------------------------------------------------------------
def test_transient_plus_plateau_is_time_averaged_and_converged(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    samples = [(float(t), _plateau(float(t))) for t in range(0, 201, 2)]
    _write_log(run_dir / "gx.log", samples)

    report = inspect_run_directory(run_dir)

    window = [q for t, q in samples if t >= 100.0]
    assert report.convergence_method == "time_average_stationarity"
    assert report.converged is True
    assert report.failure_type is None
    assert report.sim_time_reached == pytest.approx(200.0)
    assert report.scalar_metrics["heat_flux_es"] == pytest.approx(
        statistics.fmean(window), rel=1e-6
    )
    assert report.scalar_metrics["heat_flux_es_std"] == pytest.approx(
        statistics.stdev(window), rel=1e-6
    )
    assert report.scalar_metrics["heat_flux_es_last"] == pytest.approx(window[-1], rel=1e-6)
    assert report.scalar_metrics["flux_window_drift"] < 0.5
    assert report.scalar_metrics["flux_window_samples"] == float(len(window))
    assert report.fluxes["heat_flux_es"] == report.scalar_metrics["heat_flux_es"]
    assert report.diagnostics_error is None


def test_growing_trajectory_is_not_converged(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    _write_log(run_dir / "gx.log", [(float(t), math.exp(t / 40.0)) for t in range(0, 201, 2)])

    report = inspect_run_directory(run_dir)

    assert report.convergence_method == "time_average_stationarity"
    assert report.scalar_metrics["flux_window_drift"] > 0.5
    assert report.converged is False
    # The drift gate is configurable.
    lenient = inspect_run_directory(run_dir, settings=DiagnosticsSettings(max_flux_drift=100.0))
    assert lenient.converged is True


def test_plateau_short_of_target_time_is_not_converged(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path, MINIMAL_GX_IN + "t_max = 400.0\n")
    _write_log(run_dir / "gx.log", [(float(t), _plateau(float(t))) for t in range(0, 201, 2)])

    report = inspect_run_directory(run_dir)

    assert report.sim_time_target == 400.0
    assert report.sim_time_target_source == "gx.in [Time] t_max"
    assert report.converged is False


def test_short_series_keeps_latest_sample_behaviour(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    _write_log(run_dir / "gx.log", [(185.0, 2.0), (190.0, 3.5)])

    report = inspect_run_directory(run_dir)

    assert report.convergence_method == "short_series_last_sample"
    assert report.scalar_metrics["heat_flux_es"] == 3.5
    assert "heat_flux_es_std" not in report.scalar_metrics
    assert report.converged is True  # legacy heuristic; the manifest marks it ineligible


def test_time_without_flux_never_fakes_a_zero_label(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    (run_dir / "gx.log").write_text(
        "\n".join(f"Step {i}: Time = {t:.1f}  dt = 1.0e-02" for i, t in enumerate([190.0, 195.0]))
        + "\n",
        encoding="utf-8",
    )

    report = inspect_run_directory(run_dir)

    assert "heat_flux_es" not in report.scalar_metrics
    assert report.fluxes == {}
    assert report.convergence_method == "unverified"
    assert report.converged is False


def test_absent_secondary_fluxes_are_omitted_not_zero(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    (run_dir / "diagnostics_metrics.json").write_text(
        json.dumps(
            {"converged": True, "sim_time_reached": 200.0, "scalar_metrics": {"heat_flux_es": 3.0}}
        ),
        encoding="utf-8",
    )

    report = inspect_run_directory(run_dir)

    assert report.fluxes == {"heat_flux_es": 3.0}
    assert "heat_flux_em" not in report.fluxes and "particle_flux" not in report.fluxes


# --------------------------------------------------------------------------------------------
# PHYS-01: robust timestep-collapse ratio
# --------------------------------------------------------------------------------------------
def test_single_dt_spike_does_not_trigger_collapse(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    dts = [0.01] * 30
    dts[20] = 1.0e-5  # one transient eddy spike
    lines = [_gx_line(i, 0.01 * i, 1.0, dt) for i, dt in enumerate(dts, start=1)]
    (run_dir / "gx.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = inspect_run_directory(run_dir)

    assert report.scalar_metrics["dt_collapse_ratio"] == pytest.approx(1.0)
    assert report.failure_type is None


def test_sustained_dt_collapse_triggers_and_threshold_is_configurable(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    dts = [0.01] * 10 + [5.0e-5] * 60
    lines = [_gx_line(i, 0.01 * i, 1.0, dt) for i, dt in enumerate(dts, start=1)]
    (run_dir / "gx.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = inspect_run_directory(run_dir)
    assert report.failure_type == "TIMESTEP_COLLAPSE"
    assert report.scalar_metrics["dt_initial"] == pytest.approx(0.01)
    assert report.scalar_metrics["dt_recent"] == pytest.approx(5.0e-5)

    relaxed = inspect_run_directory(
        run_dir, settings=DiagnosticsSettings(dt_collapse_threshold=0.001)
    )
    assert relaxed.failure_type is None
    floored = inspect_run_directory(
        run_dir, settings=DiagnosticsSettings(dt_collapse_threshold=0.001, dt_min_floor=1.0e-4)
    )
    assert floored.failure_type == "TIMESTEP_COLLAPSE"


def test_timeseries_helpers_match_domain_spec() -> None:
    stats = dt_collapse_stats([0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001])
    assert stats is not None and stats.method == "median_window"
    assert stats.initial_dt == pytest.approx(0.01)
    short = dt_collapse_stats([0.01, 0.002])
    assert (
        short is not None and short.method == "min_over_first" and short.ratio == pytest.approx(0.2)
    )
    assert dt_collapse_stats([float("nan"), 0.0, -1.0]) is None

    assert flux_window_stats([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) is None
    window = flux_window_stats([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 2, 2, 4, 4])
    assert window is not None
    assert window.n_samples == 4 and window.t_start == 4.0  # t >= 0.5 * 7
    assert window.mean == pytest.approx(3.0)
    assert window.drift == pytest.approx(abs(4.0 - 2.0) / 3.0)
    # Fewer than 4 samples in the time window: fall back to the last max(4, n // 2) samples.
    fallback = flux_window_stats([0, 1, 2, 3, 4, 100], [1, 1, 1, 1, 1, 9])
    assert fallback is not None and fallback.n_samples == 4 and fallback.t_start == 2.0


# --------------------------------------------------------------------------------------------
# DOM-02 sim_time_target chain, DOM-03 GENE fallback, DOM-04 species
# --------------------------------------------------------------------------------------------
def test_sim_time_target_chain(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path, MINIMAL_GX_IN + "t_max = 400.0\n")
    assert inspect_run_directory(run_dir).sim_time_target == 400.0

    (run_dir / "run_meta.json").write_text(json.dumps({"sim_time_target": 300.0}), encoding="utf-8")
    report = inspect_run_directory(run_dir)
    assert (report.sim_time_target, report.sim_time_target_source) == (300.0, "run_meta.json")

    (run_dir / "diagnostics_metrics.json").write_text(
        json.dumps({"sim_time_target": 250.0}), encoding="utf-8"
    )
    report = inspect_run_directory(run_dir)
    assert (report.sim_time_target, report.sim_time_target_source) == (
        250.0,
        "diagnostics_metrics.json",
    )

    plain = _gx_run(tmp_path, MINIMAL_GX_IN, name="plain")
    report = inspect_run_directory(plain, sim_time_target_default=123.0)
    assert (report.sim_time_target, report.sim_time_target_source) == (123.0, "default")


def test_gene_simtimelim_fallback_and_full_log_scan(tmp_path: Path) -> None:
    run_dir = tmp_path / "gene_run"
    run_dir.mkdir()
    (run_dir / "parameters").write_text(
        "&box\n  nx0 = 64\n/\n&general\n  dt_max = 0.02\n  simtimelim = 350.0\n  timelim = 86000\n/\n",
        encoding="utf-8",
    )
    lines = [f" time = {0.5 * i:.3f}  dt = 1.0e-02  q_es = 1.5" for i in range(1, 200)]
    (run_dir / "gene.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = inspect_run_directory(run_dir)
    assert report.sim_time_target == 350.0
    assert report.sim_time_target_source == "parameters &general simtimelim"
    assert report.failure_type is None
    assert report.scalar_metrics["heat_flux_es"] == pytest.approx(1.5)
    assert report.scalar_metrics["dt_collapse_ratio"] == pytest.approx(1.0)

    # A Fortran "Infinity" early in the log (far beyond the 20-line tail) is DIVERGENCE.
    lines.insert(3, " time = 1.700  dt = 1.0e-02  q_es = Infinity")
    (run_dir / "gene.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert inspect_run_directory(run_dir).failure_type == "DIVERGENCE"

    (run_dir / "gene.log").write_text(" time = 1.0 q_es = -inf\n", encoding="utf-8")
    assert inspect_run_directory(run_dir).failure_type == "DIVERGENCE"


def test_crash_markers_in_batch_stdout_are_detected(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    _write_log(run_dir / "gx.log", [(1.0, 1.0)])
    (run_dir / "slurm.out").write_text(
        "slurmstepd: error: Detected 1 oom-kill event(s) in StepId=1.batch.\n", encoding="utf-8"
    )
    assert inspect_run_directory(run_dir).failure_type == "OUT_OF_MEMORY"
    (run_dir / "slurm.out").write_text("line 12: 4242 Floating point exception\n", encoding="utf-8")
    assert inspect_run_directory(run_dir).failure_type == "DIVERGENCE"


def test_lowercase_species_table_sets_nspecies(tmp_path: Path) -> None:
    gx_in = tmp_path / "gx.in"
    gx_in.write_text("[Dimensions]\nnx = 64\n[species]\nz = [1.0, -1.0]\n", encoding="utf-8")
    assert read_gx_parameters(gx_in)["nspecies"] == 2


# --------------------------------------------------------------------------------------------
# Strict external diagnostics
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        ('{"converged": "yes", "sim_time_reached": 200}', "converged must be a JSON boolean"),
        ('{"converged": true, "sim_time_reached": "abc"}', "sim_time_reached"),
        ('{"converged": true, "sim_time_reached": 200, "sim_time_target": 0}', "sim_time_target"),
        ('{"converged": true, "sim_time_reached": NaN}', "sim_time_reached"),
        ('{"converged": true, "fluxes": {"heat_flux_es": "big"}}', "fluxes.heat_flux_es"),
        ('{"converged": true, "simulated": "no"}', "simulated must be a JSON boolean"),
        ('{"converged": true, "convergence_method": "vibes"}', "convergence_method"),
    ],
)
def test_ill_formed_diagnostics_fail_closed(tmp_path: Path, content: str, fragment: str) -> None:
    run_dir = _gx_run(tmp_path)
    # A healthy log must NOT rescue unusable external evidence.
    _write_log(run_dir / "gx.log", [(float(t), _plateau(float(t))) for t in range(0, 201, 2)])
    (run_dir / "diagnostics_metrics.json").write_text(content, encoding="utf-8")

    report = inspect_run_directory(run_dir)

    assert report.converged is False
    assert report.diagnostics_error is not None and fragment in report.diagnostics_error
    assert report.sim_time_target > 0


def test_unknown_external_failure_type_is_unknown_failure(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    (run_dir / "diagnostics_metrics.json").write_text(
        '{"failure_type": "NUMERICAL_BLOWUP"}', encoding="utf-8"
    )
    report = inspect_run_directory(run_dir)
    assert report.failure_type == "UNKNOWN"
    assert "failure_type" in (report.diagnostics_error or "")


def test_external_verdict_metadata_passthrough(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    payload = {
        "converged": True,
        "convergence_method": "short_series_last_sample",
        "sim_time_reached": 200.0,
        "scalar_metrics": {"heat_flux_es": 2.0, "provenance_ok": 1.0},
        "fluxes": {"heat_flux_es": 2.0, "heat_flux_em": None},
        "flux_units": "gyroBohm",
        "provenance": {"simulated": False, "gx_commit": "a" * 40},
    }
    (run_dir / "diagnostics_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    report = inspect_run_directory(run_dir)

    assert report.converged is True
    assert report.convergence_method == "short_series_last_sample"
    assert report.flux_units == "gyroBohm"
    assert report.code_version == "a" * 40
    assert report.simulated is False
    assert report.fluxes == {"heat_flux_es": 2.0, "heat_flux_em": None}

    payload.pop("convergence_method")
    payload["scalar_metrics"] = {"heat_flux_es": 2.0, "simulated": 1.0}
    (run_dir / "diagnostics_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    report = inspect_run_directory(run_dir)
    assert report.convergence_method == "external_diagnostics"
    assert report.simulated is True

    (run_dir / "diagnostics_metrics.json").write_text(
        '{"sim_time_reached": 200.0}', encoding="utf-8"
    )
    report = inspect_run_directory(run_dir)
    assert (report.converged, report.convergence_method) == (False, "unverified")


def test_diagnostics_cmd_output_is_bounded_and_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _gx_run(tmp_path)
    ok_cmd = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'converged': True, 'sim_time_reached': 200.0, "
        "'scalar_metrics': {'heat_flux_es': 1.25, 'x': float('nan')}}))",
    ]
    report = inspect_run_directory(run_dir, diagnostics_cmd=ok_cmd)
    stored = json.loads((run_dir / "diagnostics_metrics.json").read_text(encoding="utf-8"))
    assert stored["scalar_metrics"]["x"] == "nan"  # strict JSON on disk
    assert report.failure_type == "DIVERGENCE"  # non-finite metric = has_nan (legacy)
    assert not [p for p in run_dir.iterdir() if p.name.startswith(".")]

    (run_dir / "diagnostics_metrics.json").unlink()
    flood = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"]
    report = inspect_run_directory(run_dir, diagnostics_cmd=flood)
    assert "exceeded" in (report.diagnostics_error or "")
    assert not (run_dir / "diagnostics_metrics.json").exists()

    monkeypatch.setattr(da, "DIAGNOSTICS_CMD_TIMEOUT_S", 0.5)
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    report = inspect_run_directory(run_dir, diagnostics_cmd=sleeper)
    assert "timed out" in (report.diagnostics_error or "")
    assert report.converged is False

    report = inspect_run_directory(run_dir, diagnostics_cmd="python3 diag.py")  # type: ignore[arg-type]
    assert "list of strings" in (report.diagnostics_error or "")


def test_symlinked_log_is_not_followed(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-TOKEN\n", encoding="utf-8")
    run_dir = _gx_run(tmp_path)
    os.symlink(secret, run_dir / "gx.log")

    report = inspect_run_directory(run_dir)

    assert all("TOP-SECRET" not in line for line in report.log_tail)


# --------------------------------------------------------------------------------------------
# Log segments: rotation + restart-rule merge
# --------------------------------------------------------------------------------------------
def test_rotation_keeps_every_generation(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    (run_dir / "gx.log.prev").write_text("legacy\n", encoding="utf-8")
    (run_dir / "gx.log").write_text("first\n", encoding="utf-8")
    (run_dir / "diagnostics_metrics.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in consume_stale_diagnostics_metrics(run_dir)] == [
        "diagnostics_metrics.1.json",
        "gx.log.1",
    ]
    (run_dir / "gx.log").write_text("second\n", encoding="utf-8")
    (run_dir / "slurm.out").write_text("batch\n", encoding="utf-8")
    assert [p.name for p in consume_stale_diagnostics_metrics(run_dir)] == [
        "gx.log.2",
        "slurm.out.2",
    ]
    assert consume_stale_diagnostics_metrics(run_dir) == []
    assert [p.name for p in archived_log_segments(run_dir, "gx.log")] == [
        "gx.log.prev",
        "gx.log.1",
        "gx.log.2",
    ]
    assert (run_dir / "gx.log.1").read_text(encoding="utf-8") == "first\n"


def test_warm_restart_segments_merge_and_stale_markers_do_not_retrigger(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path, MINIMAL_GX_IN + "[Restart]\nrestart = true\n")
    first = [(float(t), _plateau(float(t))) for t in range(0, 121, 2)]
    _write_log(run_dir / "gx.log", first)
    with (run_dir / "gx.log").open("a", encoding="utf-8") as fh:
        fh.write(_gx_line(999, 121.0, float("nan")) + "\n")  # the failure that caused the strike
    consume_stale_diagnostics_metrics(run_dir)
    # Warm restart from the t=100 checkpoint: overlapping 100..120 samples are replaced.
    second = [(float(t), _plateau(float(t)) + 0.01) for t in range(100, 201, 2)]
    _write_log(run_dir / "gx.log", second)

    report = inspect_run_directory(run_dir)

    assert report.failure_type is None  # the NaN lives in gx.log.1 only
    assert report.log_segments_used == 2
    assert report.sim_time_reached == pytest.approx(200.0)
    expected = [q for t, q in second]  # window t >= 100 comes entirely from the restart
    assert report.scalar_metrics["heat_flux_es"] == pytest.approx(
        statistics.fmean(expected), rel=1e-6
    )
    assert report.converged is True


def test_restart_rule_and_cold_start_use_only_valid_history(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path, MINIMAL_GX_IN + "[Restart]\nrestart = false\n")
    _write_log(run_dir / "gx.log", [(float(t), 50.0) for t in range(0, 201, 2)])
    consume_stale_diagnostics_metrics(run_dir)
    _write_log(run_dir / "gx.log", [(float(t), 5.0) for t in range(0, 41, 2)])

    cold = inspect_run_directory(run_dir)
    assert cold.log_segments_used == 1
    assert cold.sim_time_reached == pytest.approx(40.0)
    assert cold.scalar_metrics["heat_flux_es"] == pytest.approx(5.0)

    # Even if the input claims a warm restart, a time reset drops the stale trajectory.
    (run_dir / "gx.in").write_text(MINIMAL_GX_IN + "[Restart]\nrestart = true\n", encoding="utf-8")
    reset = inspect_run_directory(run_dir)
    assert reset.sim_time_reached == pytest.approx(40.0)
    assert reset.scalar_metrics["heat_flux_es"] == pytest.approx(5.0)


def test_empty_current_log_never_certifies_from_history(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path, MINIMAL_GX_IN + "[Restart]\nrestart = true\n")
    _write_log(run_dir / "gx.log", [(float(t), _plateau(float(t))) for t in range(0, 201, 2)])
    consume_stale_diagnostics_metrics(run_dir)

    report = inspect_run_directory(run_dir)

    assert report.sim_time_reached == pytest.approx(200.0)  # progress still visible
    assert report.converged is False
    assert report.convergence_method == "unverified"


# --------------------------------------------------------------------------------------------
# Checkpoint discovery
# --------------------------------------------------------------------------------------------
def test_find_checkpoint_files(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path, MINIMAL_GX_IN + '[Restart]\nrestart_to_file = "ckpt_custom.nc"\n')
    assert find_checkpoint_files(run_dir, "GX") == []
    (run_dir / "gx.restart.nc").write_bytes(b"CDF\x01")
    (run_dir / "gx.prev.restart.nc").write_bytes(b"CDF\x01")
    (run_dir / "ckpt_custom.nc").write_bytes(b"CDF\x01")
    (run_dir / ".gx.restart.nc.tmp").write_bytes(b"partial")
    (run_dir / "empty.restart.nc").write_bytes(b"")
    os.symlink(run_dir / "gx.restart.nc", run_dir / "link.restart.nc")
    assert [p.name for p in find_checkpoint_files(run_dir, "GX")] == [
        "ckpt_custom.nc",
        "gx.prev.restart.nc",
        "gx.restart.nc",
    ]

    unsafe = _gx_run(
        tmp_path, MINIMAL_GX_IN + '[Restart]\nrestart_to_file = "../../etc/x"\n', "unsafe"
    )
    assert read_gx_run_controls(unsafe / "gx.in")["restart_to_file"] is None
    assert find_checkpoint_files(unsafe, "GX") == []

    gene = tmp_path / "gene"
    gene.mkdir()
    (gene / "checkpoint").write_bytes(b"x")
    (gene / "s_checkpoint_0001").write_bytes(b"x")
    (gene / "checkpoint_empty").write_bytes(b"")
    assert [p.name for p in find_checkpoint_files(gene, "gene")] == [
        "checkpoint",
        "s_checkpoint_0001",
    ]
    with pytest.raises(ValueError):
        find_checkpoint_files(gene, "SFINCS")


# --------------------------------------------------------------------------------------------
# Patchers (PHYS-04, atomic writes, strict values)
# --------------------------------------------------------------------------------------------
def test_gx_d_hyper_patch_on_benchmark_deck(tmp_path: Path) -> None:
    gx_in = tmp_path / "gx.in"
    shutil.copy2(BENCHMARK_GX_IN, gx_in)
    original = gx_in.read_text(encoding="utf-8")
    assert read_gx_parameters(gx_in)["hyperdiffusion_coeff"] == pytest.approx(0.05)

    after = patch_gx_toml(gx_in, {"hyperdiffusion_coeff": 0.1}, restart_from_checkpoint=False)

    text = gx_in.read_text(encoding="utf-8")
    doc = tomlkit.parse(text)
    assert after["hyperdiffusion_coeff"] == pytest.approx(0.1)
    assert doc["Dissipation"]["D_hyper"] == pytest.approx(0.1)
    assert doc["Dissipation"]["hyper"] is True
    assert "hyperviscosity" not in doc["Dissipation"]
    assert "# exponent of hyperdiffusion is 2*p_hyper = 4" in text
    assert doc["Time"]["t_max"] == 400.0 and doc["species"]["tprim"][0] == 2.49
    assert (tmp_path / "gx.in.orig").read_text(encoding="utf-8") == original
    assert (tmp_path / "gx.in.bak").read_text(encoding="utf-8") == original

    patch_gx_toml(gx_in, {"dt_max": 0.01}, restart_from_checkpoint=True)
    assert (tmp_path / "gx.in.orig").read_text(encoding="utf-8") == original  # written once
    assert (tmp_path / "gx.in.bak").read_text(encoding="utf-8") == text
    assert tomlkit.parse(gx_in.read_text(encoding="utf-8"))["Restart"]["restart"] is True


def test_gx_legacy_hyperviscosity_fallback_and_sync(tmp_path: Path) -> None:
    gx_in = tmp_path / "gx.in"
    gx_in.write_text("[Dissipation]\nhyperviscosity = 0.1\n", encoding="utf-8")
    assert read_gx_parameters(gx_in)["hyperdiffusion_coeff"] == pytest.approx(0.1)
    patch_gx_toml(gx_in, {"hyperdiffusion_coeff": 0.0}, restart_from_checkpoint=False)
    doc = tomlkit.parse(gx_in.read_text(encoding="utf-8"))
    assert doc["Dissipation"]["D_hyper"] == 0.0 and doc["Dissipation"]["hyperviscosity"] == 0.0
    assert "hyper" not in doc["Dissipation"]  # only enabled for a positive coefficient


@pytest.mark.parametrize(
    "mods",
    [
        {"nx": True},
        {"nx": 64.5},
        {"nx": 0},
        {"dt_max": float("nan")},
        {"dt_max": float("inf")},
        {"cfl_fraction": "0.5"},
        {"hyperdiffusion_coeff": -0.1},
        {"beta": 0.01},
    ],
)
def test_patchers_reject_invalid_values_without_touching_files(tmp_path: Path, mods: dict) -> None:
    gx_in = tmp_path / "gx.in"
    gx_in.write_text(MINIMAL_GX_IN, encoding="utf-8")
    params = tmp_path / "parameters"
    params.write_text("&box\n  nx0 = 64\n/\n&general\n  dt_max = 0.02\n/\n", encoding="utf-8")
    with pytest.raises(ValueError):
        patch_gx_toml(gx_in, mods, restart_from_checkpoint=False)
    with pytest.raises(ValueError):
        patch_gene_parameters(params, mods, restart_from_checkpoint=False)
    assert gx_in.read_text(encoding="utf-8") == MINIMAL_GX_IN
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gx.in", "parameters"]


def test_gene_patch_changes_only_allowlisted_keys(tmp_path: Path) -> None:
    params = tmp_path / "parameters"
    params.write_text(
        "&box\n  nx0 = 64 ! radial\n  nky0 = 32\n  kymin = 0.05\n/\n"
        "&general\n  dt_max = 0.02\n  beta = 0.035\n/\n"
        "&in_out\n  diagdir = './'\n  read_checkpoint = .false.\n/\n"
        "&species\n  name = 'ions'\n  omt = 6.9\n/\n&species\n  name = 'electrons'\n  omt = 6.9\n/\n",
        encoding="utf-8",
    )
    before = f90nml.read(str(params)).todict()

    after = patch_gene_parameters(
        params, {"nz": 48, "cfl_fraction": 0.5}, restart_from_checkpoint=True
    )

    patched = f90nml.read(str(params)).todict()
    assert after["nz"] == 48 and after["cfl_fraction"] == pytest.approx(0.5)
    assert patched["in_out"]["read_checkpoint"] is True
    assert patched["box"]["kymin"] == before["box"]["kymin"]
    assert patched["general"]["beta"] == before["general"]["beta"]
    assert patched["_grp_species_1"] == before["_grp_species_1"]
    assert "! radial" in params.read_text(encoding="utf-8")
    assert (tmp_path / "parameters.orig").is_file() and (tmp_path / "parameters.bak").is_file()

    # Duplicated target groups are ambiguous: refuse.
    params.write_text("&box\n nx0 = 64\n/\n&box\n nx0 = 32\n/\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        patch_gene_parameters(params, {"nx": 96}, restart_from_checkpoint=False)


def test_gx_patch_refuses_ambiguous_restart_tables(tmp_path: Path) -> None:
    gx_in = tmp_path / "gx.in"
    gx_in.write_text(MINIMAL_GX_IN + "[restart]\nrestart = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Ambiguous"):
        patch_gx_toml(gx_in, {}, restart_from_checkpoint=False)


def test_readers_reject_ill_typed_allowlisted_values(tmp_path: Path) -> None:
    gx_in = tmp_path / "gx.in"
    gx_in.write_text("[Dimensions]\nnx = true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_gx_parameters(gx_in)
    gx_in.write_text("[Dimensions]\nnx = 64.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_gx_parameters(gx_in)
    params = tmp_path / "parameters"
    params.write_text("&box\n  nx0 = .true.\n/\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_gene_parameters(params)


def test_gx_dt_max_syncs_explicit_time_dt_max_without_adding_when_absent(tmp_path: Path) -> None:
    absent = tmp_path / "absent.in"
    absent.write_text(MINIMAL_GX_IN, encoding="utf-8")
    patch_gx_toml(absent, {"dt_max": 0.01}, restart_from_checkpoint=True)
    absent_doc = tomlkit.parse(absent.read_text(encoding="utf-8"))
    assert absent_doc["Time"]["dt"] == pytest.approx(0.01)
    assert "dt_max" not in absent_doc["Time"]

    explicit = tmp_path / "explicit.in"
    explicit.write_text(
        "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n"
        "[Time]\ndt = 0.02 # initial dt\ndt_max = 0.02 # adaptive clamp\ncfl = 0.8\n",
        encoding="utf-8",
    )
    after = patch_gx_toml(explicit, {"dt_max": 0.005}, restart_from_checkpoint=True)
    rendered = explicit.read_text(encoding="utf-8")
    explicit_doc = tomlkit.parse(rendered)
    assert after["dt_max"] == pytest.approx(0.005)
    assert explicit_doc["Time"]["dt"] == pytest.approx(0.005)
    assert explicit_doc["Time"]["dt_max"] == pytest.approx(0.005)
    assert "# adaptive clamp" in rendered


def test_read_gx_run_controls_exposes_diagnostics_fluxes_flag(tmp_path: Path) -> None:
    missing = tmp_path / "missing.in"
    missing.write_text(MINIMAL_GX_IN, encoding="utf-8")
    assert read_gx_run_controls(missing)["fluxes"] is None

    enabled = tmp_path / "enabled.in"
    enabled.write_text(MINIMAL_GX_IN + "[Diagnostics]\nfluxes = true\n", encoding="utf-8")
    assert read_gx_run_controls(enabled)["fluxes"] is True

    disabled = tmp_path / "disabled.in"
    disabled.write_text(MINIMAL_GX_IN + "[Diagnostics]\nfluxes = false\n", encoding="utf-8")
    assert read_gx_run_controls(disabled)["fluxes"] is False


def test_gx_log_parser_handles_gam_e_and_multi_ion_species(tmp_path: Path) -> None:
    run_dir = _gx_run(tmp_path)
    lines = [
        (
            f"gx: Step {step:6d}: Time = {float(t):10.5f}  dt = 1.000e-02   Phi**2 = 2.000e+00   "
            "Q_i = 3.100e+00   Q_e = 9.000e-01   Gam_i = 2.000e-01   Gam_i = 5.000e-02   "
            "Gam_e = 2.500e-01"
        )
        for step, t in enumerate(range(0, 201, 2), start=1)
    ]
    (run_dir / "gx.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = inspect_run_directory(run_dir)
    assert report.converged is True
    assert report.fluxes["heat_flux_es"] == pytest.approx(4.0)
    assert report.fluxes["particle_flux"] == pytest.approx(0.25)

    etg_dir = _gx_run(tmp_path, name="etg")
    etg_lines = [
        (
            f"gx: Step {step:6d}: Time = {float(t):10.5f}  dt = 1.000e-02   Phi**2 = 1.500e+00   "
            "Q_e = 1.750e+00   Gam_e = 1.200e-01"
        )
        for step, t in enumerate(range(0, 201, 2), start=1)
    ]
    (etg_dir / "gx.log").write_text("\n".join(etg_lines) + "\n", encoding="utf-8")
    etg_report = inspect_run_directory(etg_dir)
    assert etg_report.fluxes["heat_flux_es"] == pytest.approx(1.75)
    assert etg_report.fluxes["particle_flux"] == pytest.approx(0.12)
