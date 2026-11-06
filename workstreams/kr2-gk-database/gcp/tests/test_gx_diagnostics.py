"""Tests for `gx-diagnose` (gcp_exec/gx_diagnostics.py).

Covers the DOMAIN.md verdicts (PHYS-01 robust dt ratio, PHYS-02 post-transient window and
drift gate, DOM-01 species-summed ky/kx tail ratios, DOM-02 sim_time_target fallback) and
the hardening rules (strict JSON, absent flux never 0.0, NetCDF file choice, truncated or
stale NetCDF, masked/fill values, restart overlap, provenance, output size). netCDF4 is not
in the dev venv, so NetCDF content comes from an injected fake dataset.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gcp_exec import gx_diagnostics as gxd
from gcp_exec.gx_diagnostics import DiagnoseSettings, diagnose, main

COMMIT = "0123456789abcdef0123456789abcdef01234567"


# ------------------------------------------------------------------------------ helpers


def p3(x: float) -> float:
    """The value as GX prints it (%.3e)."""
    return float(f"{x:.3e}")


def gx_line(step: int, t: float, dt: float, q_i: float, *, q_e: float | None = None) -> str:
    line = (
        f"cbc: Step {step:7d}: Time = {t:10.5f}  dt = {dt:.3e}   Phi**2 = 1.000e-03"
        f"   Q_i = {q_i:.3e}"
    )
    if q_e is not None:
        line += f"   Q_e = {q_e:.3e}"
    return line + "   Gam_i = 2.000e-01\n"


def make_run(
    tmp_path: Path,
    rows: list[tuple[float, float, float]] | None = None,
    *,
    t_max: float | None = None,
    meta: dict[str, Any] | None = None,
    log_name: str = "gx.log",
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    deck = "[Time]\n" + (f" t_max = {t_max}\n" if t_max is not None else "") + " cfl = 0.8\n"
    (run_dir / "gx.in").write_text(deck, encoding="utf-8")
    if meta is not None:
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if rows is not None:
        text = "".join(gx_line(i, t, dt, q) for i, (t, dt, q) in enumerate(rows))
        (run_dir / log_name).write_text(text, encoding="utf-8")
    return run_dir


def plateau_rows(n: int = 200) -> list[tuple[float, float, float]]:
    """Linear transient up to t=60, then saturated bursts around Q=5 (t = 1..n)."""
    rows = []
    for t in range(1, n + 1):
        q = 0.05 * t if t <= 60 else 5.0 + 0.5 * math.sin(t)
        rows.append((float(t), 0.05, p3(q)))
    return rows


def growing_rows(n: int = 200) -> list[tuple[float, float, float]]:
    return [(float(t), 0.05, p3(0.01 * math.exp(t / 25.0))) for t in range(1, n + 1)]


def window_stats(values: list[float]) -> tuple[float, float, float]:
    half = len(values) // 2
    mean = statistics.fmean(values)
    drift = abs(statistics.fmean(values[half:]) - statistics.fmean(values[:half])) / abs(mean)
    return mean, statistics.stdev(values), drift


def reject_constant(token: str) -> None:
    raise ValueError(f"non-strict JSON token {token}")


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[dict[str, Any], str]:
    assert main(list(args)) == 0
    out = capsys.readouterr().out
    return json.loads(out, parse_constant=reject_constant), out


class FakeVar:
    """netCDF4.Variable stand-in: `[:]`, `.dimensions`, `ncattrs()` / `getncattr()`."""

    def __init__(self, data: Any, dims: tuple[str, ...] = (), *, fill: Any = None, error=None):
        self.data = data
        self.dimensions = dims
        self.attrs = {} if fill is None else {"_FillValue": fill}
        self.error = error

    def __getitem__(self, key: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.data[key]

    def ncattrs(self) -> list[str]:
        return list(self.attrs)

    def getncattr(self, name: str) -> Any:
        return self.attrs[name]


class FakeGroup:
    def __init__(self, variables: dict[str, Any] | None = None, groups: dict | None = None):
        self.variables = dict(variables or {})
        self.groups = dict(groups or {})


class FakeDataset(FakeGroup):
    def __init__(
        self,
        grids: dict[str, Any],
        diagnostics: dict[str, Any],
        geometry: dict[str, Any] | None = None,
    ):
        groups = {"Grids": FakeGroup(grids), "Diagnostics": FakeGroup(diagnostics)}
        if geometry is not None:
            groups["Geometry"] = FakeGroup(geometry)
        super().__init__(groups=groups)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def gx_nc(
    times: list[float],
    heat: Any = None,
    *,
    geometry: dict[str, Any] | None = None,
    **diagnostics: Any,
) -> tuple[FakeDataset, dict[str, Any]]:
    """GX layout: Grids/time (+ky/kx), Diagnostics/HeatFlux_st (time, s) + extra variables."""
    grids: dict[str, Any] = {"time": FakeVar(np.asarray(times, dtype=np.float64), ("time",))}
    diags: dict[str, Any] = {}
    if heat is not None:
        heat_arr = heat if isinstance(heat, FakeVar) else np.asarray(heat, dtype=np.float32)
        diags["HeatFlux_st"] = (
            heat_arr if isinstance(heat_arr, FakeVar) else FakeVar(heat_arr, ("time", "s"))
        )
    for name, value in diagnostics.items():
        (grids if name in ("ky", "kx") else diags)[name] = value
    return FakeDataset(grids, diags, geometry=geometry), diags


def attach_nc(run_dir: Path, dataset: FakeDataset):
    (run_dir / "gx.out.nc").write_bytes(b"CDF\x02 placeholder")
    return lambda _path: dataset


def spectrum_var(last: list[list[float]], kname: str, n_t: int = 3) -> FakeVar:
    """(time, s, k) spectrum whose last time slice is `last` (earlier slices are flat)."""
    arr = np.ones((n_t, len(last), len(last[0])), dtype=np.float32)
    arr[-1] = np.asarray(last, dtype=np.float32)
    return FakeVar(arr, ("time", "s", kname))


# --------------------------------------------------------------------- defaults / CLI


def test_threshold_defaults_are_the_adjudicated_values() -> None:
    s = DiagnoseSettings()
    assert (s.dt_collapse_threshold, s.dt_min_floor, s.transient_discard_fraction) == (
        0.01,
        None,
        0.5,
    )
    assert (s.max_flux_drift, s.high_k_tail_threshold) == (0.5, 0.25)
    assert (s.convergence_time_fraction, s.sim_time_target_default) == (0.9, 200.0)
    args = gxd.build_arg_parser().parse_args(["run"])
    assert {k: v for k, v in vars(args).items() if k != "run_dir"} == {
        "dt_collapse_threshold": 0.01,
        "transient_discard_fraction": 0.5,
        "max_flux_drift": 0.5,
        "high_k_tail_threshold": 0.25,
        "convergence_time_fraction": 0.9,
        "sim_time_target_default": 200.0,
        "dt_min_floor": None,
    }


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-flux-drift", "nan"],
        ["--max-flux-drift", "-1"],
        ["--dt-collapse-threshold", "1.5"],
        ["--transient-discard-fraction", "1.0"],
        ["--dt-min-floor", "0"],
        ["--high-k-tail-threshold", "inf"],
    ],
)
def test_cli_rejects_invalid_thresholds(tmp_path: Path, flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*flags, str(tmp_path)])
    assert exc.value.code == 2


def test_cli_flag_overrides_drift_gate(tmp_path: Path, capsys) -> None:
    run_dir = make_run(tmp_path, growing_rows())
    strict, _ = run_cli(capsys, str(run_dir))
    loose, _ = run_cli(capsys, "--max-flux-drift", "1000", str(run_dir))
    assert strict["converged"] is False
    assert loose["converged"] is True


# ---------------------------------------------------------------------------- PHYS-02


def test_transient_plateau_uses_post_transient_window_mean_std_and_drift(tmp_path: Path) -> None:
    rows = plateau_rows()
    result = diagnose(make_run(tmp_path, rows))
    sm = result["scalar_metrics"]
    window = [q for t, _, q in rows if t >= 100.0]
    mean, std, drift = window_stats(window)
    assert sm["heat_flux_es"] == pytest.approx(mean, rel=1e-12)
    assert sm["heat_flux_es_std"] == pytest.approx(std, rel=1e-12)
    assert sm["heat_flux_es_last"] == rows[-1][2]
    assert sm["flux_window_drift"] == pytest.approx(round(drift, 6), abs=1e-9)
    assert sm["heat_flux_es"] != sm["heat_flux_es_last"]  # no longer Q(t_end)
    assert sm["particle_flux"] == pytest.approx(0.2) and sm["phi2_total"] == pytest.approx(1e-3)
    assert result["fluxes"] == {
        "heat_flux_es": sm["heat_flux_es"],
        "particle_flux": sm["particle_flux"],
    }
    assert result["converged"] is True and result["failure_type"] is None
    assert result["convergence_method"] == "time_average_stationarity"
    assert result["flux_source"] == "log" and result["sim_time_reached"] == 200.0


def test_growing_trajectory_is_not_converged(tmp_path: Path) -> None:
    result = diagnose(make_run(tmp_path, growing_rows()))
    assert result["scalar_metrics"]["flux_window_drift"] > 0.5
    assert result["converged"] is False and result["failure_type"] is None
    assert result["convergence_method"] == "time_average_stationarity"


def test_legacy_two_line_fixture_keeps_latest_sample(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "gx.log").write_text(
        "run_name: Step     10: Time =    0.10000  dt = 1.000e-03   Phi**2 = 1.234e-05   Q_i = 1.234e-03   Q_e = 1.234e-03   Gam_i = 0.000e+00\n"
        "run_name: Step     20: Time =    0.20000  dt = 5.000e-04   Phi**2 = 1.500e-05   Q_i = 2.000e-03   Q_e = 1.500e-03   Gam_i = 5.000e-04\n",
        encoding="utf-8",
    )
    times, dts, fluxes, scalar, has_nan = gxd.parse_gx_log(run_dir / "gx.log")
    assert (times, dts, fluxes["Q_i"], scalar["phi2_total"], has_nan) == (
        [0.1, 0.2],
        [0.001, 0.0005],
        0.002,
        1.5e-05,
        0.0,
    )
    result = diagnose(run_dir)
    sm = result["scalar_metrics"]
    assert sm["heat_flux_es"] == pytest.approx(0.0035)
    assert (sm["particle_flux"], sm["phi2_total"]) == (0.0005, 1.5e-05)
    assert sm["dt_collapse_ratio"] == 0.5 and "heat_flux_es_std" not in sm
    assert result["convergence_method"] == "short_series_last_sample"
    assert result["converged"] is False and result["sim_time_target"] == 200.0


def test_nan_and_inf_lines_are_excluded_and_flag_divergence(tmp_path: Path, capsys) -> None:
    rows = plateau_rows()
    rows[150] = (rows[150][0], 0.05, float("nan"))
    rows[170] = (rows[170][0], 0.05, float("inf"))
    run_dir = make_run(tmp_path, rows)
    with open(run_dir / "gx.log", "a", encoding="utf-8") as handle:
        handle.write("Floating point exception (core dumped)\n")
    result, out = run_cli(capsys, str(run_dir))
    sm = result["scalar_metrics"]
    finite_window = [q for t, _, q in rows if t >= 100.0 and math.isfinite(q)]
    assert sm["heat_flux_es"] == pytest.approx(statistics.fmean(finite_window), rel=1e-12)
    assert sm["has_nan"] == 1.0
    assert result["failure_type"] == "DIVERGENCE" and result["converged"] is False
    assert "NaN" not in out and "Infinity" not in out


def test_huge_finite_fluxes_never_crash_or_emit_infinity(tmp_path: Path, capsys) -> None:
    run_dir = make_run(tmp_path, [(float(t), 0.05, 1e308) for t in range(1, 11)])
    result, out = run_cli(capsys, str(run_dir))
    assert result["failure_type"] == "DIVERGENCE"
    assert "heat_flux_es" not in result["scalar_metrics"]
    assert "heat_flux_es" not in result["fluxes"]
    assert "Infinity" not in out


@pytest.mark.parametrize("content", [None, ""])
def test_missing_or_empty_log_omits_flux(tmp_path: Path, content: str | None) -> None:
    run_dir = make_run(tmp_path)
    if content is not None:
        (run_dir / "gx.log").write_text(content, encoding="utf-8")
    result = diagnose(run_dir)
    sm = result["scalar_metrics"]
    assert result["fluxes"] == {} and "heat_flux_es" not in sm  # never a fake 0.0 label
    assert "dt_collapse_ratio" not in sm and "high_k_tail_ratio" not in sm
    assert sm["has_nan"] == 0.0 and result["converged"] is False
    assert result["convergence_method"] == "unverified" and result["flux_source"] is None


def test_incomplete_last_line_is_ignored(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, plateau_rows(10))
    with open(run_dir / "gx.log", "a", encoding="utf-8") as handle:
        handle.write("cbc: Step   99: Time =   11.00000  dt = 5.000e-02   Q_i = 1.2")
    series = gxd.parse_gx_log_series(run_dir / "gx.log")
    assert len(series.rows) == 10 and series.rows[-1].time == 10.0


def test_gke_stdout_name_gx_out_is_read(tmp_path: Path) -> None:
    result = diagnose(make_run(tmp_path, plateau_rows(), log_name="gx.out"))
    assert result["flux_source"] == "log" and result["converged"] is True


# ---------------------------------------------------------------------------- PHYS-01


def dt_run(tmp_path: Path, dts: list[float]) -> Path:
    return make_run(tmp_path, [(float(i + 1), dt, 5.0) for i, dt in enumerate(dts)])


def test_sustained_dt_drop_is_timestep_collapse(tmp_path: Path) -> None:
    result = diagnose(dt_run(tmp_path, [0.01] * 10 + [5e-5] * 60))
    assert result["scalar_metrics"]["dt_collapse_ratio"] == pytest.approx(0.005)
    assert result["failure_type"] == "TIMESTEP_COLLAPSE" and result["converged"] is False


def test_single_dt_spike_is_not_a_collapse(tmp_path: Path) -> None:
    dts = [0.01] * 70
    dts[40] = 1e-6  # legacy min(dts[-50:]) / dts[0] would be 1e-4
    result = diagnose(dt_run(tmp_path, dts))
    assert result["scalar_metrics"]["dt_collapse_ratio"] == 1.0
    assert result["failure_type"] is None


def test_optional_dt_min_floor(tmp_path: Path) -> None:
    run_dir = dt_run(tmp_path, [0.01] * 70)
    assert diagnose(run_dir)["failure_type"] is None
    floored = diagnose(run_dir, DiagnoseSettings(dt_min_floor=0.02))
    assert floored["failure_type"] == "TIMESTEP_COLLAPSE"


# ---------------------------------------------------------------------------- DOM-01


def test_species_ky_and_kx_spectra_are_reduced_over_species(tmp_path: Path) -> None:
    kx = [0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1]  # FFT order: the tail is chosen by |kx|
    dataset, _ = gx_nc(
        [10.0, 20.0, 30.0],
        Wg_kyst=spectrum_var([[10, 5, 2, 1, 1, 1], [10, 5, 2, 1, 1, 1]], "ky"),
        Wg_kxst=spectrum_var([[10, 5, 0, 2, 2, 0, 1], [0, 0, 0, 0, 0, 0, 0]], "kx"),
        ky=FakeVar(np.linspace(0.0, 0.5, 6), ("ky",)),
        kx=FakeVar(np.asarray(kx), ("kx",)),
    )
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    sm = result["scalar_metrics"]
    assert sm["high_ky_tail_ratio"] == pytest.approx(6 / 40)  # [20, 10, 4, 2, 2, 2]
    assert sm["high_kx_tail_ratio"] == pytest.approx(4 / 20)  # |kx| = 0.3, 0.3, 0.2
    assert sm["high_k_tail_ratio"] == pytest.approx(0.2)
    assert result["failure_type"] is None and sm["nc_readable"] == 1.0
    assert dataset.closed


def test_spectral_pileup_from_last_complete_slice(tmp_path: Path) -> None:
    arr = np.ma.masked_array(np.ones((3, 1, 6)), mask=np.zeros((3, 1, 6), dtype=bool))
    arr[1, 0] = [1, 1, 1, 3, 3, 3]
    arr.mask[2, 0, 4:] = True  # last record half-written: use the previous one
    dataset, _ = gx_nc(
        [10.0, 20.0, 30.0],
        Wg_kyst=FakeVar(None, ("time", "s", "ky"), error=RuntimeError("HDF error")),
        Phi2_kyt=FakeVar(arr[:, 0, :], ("time", "ky")),
    )
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    assert result["scalar_metrics"]["high_k_tail_ratio"] == pytest.approx(0.75)
    assert result["failure_type"] == "SPECTRAL_PILEUP" and result["converged"] is False


def test_spectrum_tail_ratio_helper() -> None:
    assert gxd.spectrum_tail_ratio([4, 3, 2, 1]) == pytest.approx(0.6)
    assert gxd.spectrum_tail_ratio([1, 2, 9, 3], [0.3, -0.3, 0.0, 0.1]) == pytest.approx(6 / 15)
    assert gxd.spectrum_tail_ratio([1, 1, 1]) is None
    assert gxd.spectrum_tail_ratio([0, 0, 0, 0]) is None
    assert gxd.spectrum_tail_ratio([1, float("nan"), 1, 1]) is None


# ----------------------------------------------------------------- NetCDF time series


def test_nc_flux_series_is_preferred_and_summed_over_species(tmp_path: Path) -> None:
    times = [2.0 * k for k in range(1, 101)]
    dataset, _ = gx_nc(times, [[6.0, 4.0]] * 100)
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    assert result["flux_source"] == "nc"
    assert result["scalar_metrics"]["heat_flux_es"] == pytest.approx(10.0)
    assert result["converged"] is True


def test_nc_lagging_far_behind_the_log_falls_back_to_log(tmp_path: Path) -> None:
    dataset, _ = gx_nc([2.0 * k for k in range(1, 51)], [[6.0, 4.0]] * 50)
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    assert result["flux_source"] == "log"
    assert result["scalar_metrics"]["heat_flux_es"] == pytest.approx(5.0, abs=0.2)


def test_unopenable_netcdf_falls_back_to_log(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, plateau_rows())
    (run_dir / "gx.out.nc").write_bytes(b"\x89HDF\r\n truncated")

    def opener(_path: Path) -> Any:
        raise OSError("[Errno -101] NetCDF: HDF error")

    result = diagnose(run_dir, nc_opener=opener)
    assert result["scalar_metrics"]["nc_readable"] == 0.0
    assert result["flux_source"] == "log" and result["converged"] is True


def test_default_reader_never_crashes_on_a_corrupt_file(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "FALSE")  # the opener's setdefault stays scoped
    run_dir = make_run(tmp_path, plateau_rows())
    (run_dir / "gx.out.nc").write_bytes(b"\x00" * 64)  # netCDF4 absent or unreadable
    result, _ = run_cli(capsys, str(run_dir))
    assert result["scalar_metrics"]["nc_readable"] == 0.0 and result["flux_source"] == "log"


def test_per_variable_read_error_keeps_the_rest(tmp_path: Path) -> None:
    heat = FakeVar(None, ("time", "s"), error=RuntimeError("NetCDF: HDF error"))
    dataset, _ = gx_nc(
        [10.0, 20.0, 30.0],
        heat,
        Wg_kyst=spectrum_var([[4, 3, 2, 1]], "ky"),
    )
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    sm = result["scalar_metrics"]
    assert sm["nc_readable"] == 1.0 and result["flux_source"] == "log"
    assert sm["high_ky_tail_ratio"] == pytest.approx(0.6)


def test_masked_fill_and_length_mismatch_are_missing_not_nan(tmp_path: Path) -> None:
    times = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # 8 records
    heat = np.ma.masked_array(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [-999.0]],  # 7 records (shorter)
        mask=[[False], [False], [False], [False], [False], [True], [False]],
    )
    dataset, _ = gx_nc(times, FakeVar(heat, ("time", "s"), fill=-999.0))
    run_dir = make_run(tmp_path)
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    sm = result["scalar_metrics"]
    assert sm["has_nan"] == 0.0 and result["flux_source"] == "nc"
    assert sm["heat_flux_es"] == pytest.approx(statistics.fmean([2.0, 3.0, 4.0, 5.0]))
    assert result["sim_time_reached"] == 80.0


def test_nan_in_nc_series_flags_divergence(tmp_path: Path) -> None:
    dataset, _ = gx_nc([10.0, 20.0, 30.0, 40.0, 50.0], [[1.0], [1.0], [np.nan], [1.0], [1.0]])
    run_dir = make_run(tmp_path)
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    assert result["failure_type"] == "DIVERGENCE" and result["scalar_metrics"]["has_nan"] == 1.0


def test_restart_overlap_later_samples_win(tmp_path: Path) -> None:
    assert gxd.increasing_time_indices([0, 1, 2, 3, 2.5, 3, 4]) == [0, 1, 2, 4, 5, 6]
    assert gxd.increasing_time_indices([1, 2, 2]) == [0, 2]
    times = [10.0, 20.0, 30.0, 40.0, 25.0, 35.0, 45.0, 55.0]  # restarted from t=25
    dataset, _ = gx_nc(times, [[1.0]] * 4 + [[5.0]] * 4)
    run_dir = make_run(tmp_path)
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    assert result["scalar_metrics"]["heat_flux_es"] == pytest.approx(5.0)
    assert result["sim_time_reached"] == 55.0


def test_stale_outputs_from_a_previous_attempt_are_ignored(tmp_path: Path) -> None:
    dataset, _ = gx_nc(
        [10.0, 20.0, 30.0], [[9.0]] * 3, Wg_kyst=spectrum_var([[1, 1, 1, 3, 3, 3]], "ky")
    )
    run_dir = make_run(tmp_path, plateau_rows())
    opener = attach_nc(run_dir, dataset)
    old = (run_dir / "gx.in").stat().st_mtime - 3600
    os.utime(run_dir / "gx.out.nc", (old, old))
    result = diagnose(run_dir, nc_opener=opener)
    assert result["scalar_metrics"]["nc_readable"] == 0.0
    assert result["failure_type"] is None and result["flux_source"] == "log"
    os.utime(run_dir / "gx.log", (old, old))  # stale log too: nothing current to judge
    assert diagnose(run_dir, nc_opener=opener)["convergence_method"] == "unverified"


def test_attempt_fence_marks_previous_attempt_outputs_stale(tmp_path: Path) -> None:
    dataset, _ = gx_nc([10.0, 20.0, 30.0], [[float("nan")]] * 3)  # attempt N-1 diverged
    run_dir = make_run(tmp_path, plateau_rows())
    opener = attach_nc(run_dir, dataset)
    doc = {"schema_version": 1, "simulated": False, "gx_build": {"gx_commit": COMMIT}}
    (run_dir / "task_result.json").write_text(json.dumps(doc), encoding="utf-8")
    now = (run_dir / "gx.log").stat().st_mtime
    for name in ("gx.in", "gx.out.nc", "task_result.json"):
        os.utime(run_dir / name, (now - 3600, now - 3600))  # deck unchanged on resubmit
    (run_dir / gxd.FENCE_FILENAME).write_text("job-attempt-2\n", encoding="utf-8")
    os.utime(run_dir / gxd.FENCE_FILENAME, (now - 30, now - 30))  # attempt N submitted
    result = diagnose(run_dir, nc_opener=opener)
    assert result["failure_type"] is None and result["scalar_metrics"]["nc_readable"] == 0.0
    assert result["provenance"]["task_result_found"] is False


def test_output_file_selection_is_deterministic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert gxd.select_output_nc(run_dir) is None
    for name, age in (
        (".gx.out.nc", 0),
        ("a.tmp.out.nc", 0),
        ("old.out.nc", 500),
        ("b.out.nc", 100),
    ):
        path = run_dir / name
        path.write_bytes(b"x")
        stamp = 1_700_000_000 - age
        os.utime(path, (stamp, stamp))
    assert gxd.select_output_nc(run_dir).name == "b.out.nc"  # newest regular candidate
    (run_dir / "gx.out.nc").write_bytes(b"x")
    os.utime(run_dir / "gx.out.nc", (1, 1))
    assert gxd.select_output_nc(run_dir).name == "gx.out.nc"  # <input stem>.out.nc wins


# ---------------------------------------------------------------------------- DOM-02


def test_sim_time_target_precedence(tmp_path: Path, capsys) -> None:
    run_dir = make_run(tmp_path, plateau_rows(), t_max=400.0, meta={"run_id": "r1"})
    result = diagnose(run_dir)
    assert (result["sim_time_target"], result["sim_time_target_source"]) == (400.0, "gx_in")
    assert result["converged"] is False  # t=200 < 0.9 * 400: no premature convergence
    (run_dir / "run_meta.json").write_text(json.dumps({"sim_time_target": 300}), "utf-8")
    assert diagnose(run_dir)["sim_time_target_source"] == "run_meta"
    (run_dir / "run_meta.json").unlink()
    (run_dir / "gx.in").write_text("[Time\n t_max = 400.0\n", encoding="utf-8")  # broken TOML
    cli, _ = run_cli(capsys, "--sim-time-target-default", "250", str(run_dir))
    assert (cli["sim_time_target"], cli["sim_time_target_source"]) == (250.0, "default")


# ------------------------------------------------------------------------- provenance


@pytest.mark.parametrize(
    ("task_result", "found", "simulated", "commit", "ok", "converged"),
    [
        (None, False, None, None, 0.0, True),
        ({"simulated": False, "gx_build": {"gx_commit": COMMIT}}, True, False, COMMIT, 1.0, True),
        (
            {"simulated": False, "gx_build": {"gx_commit": COMMIT.upper()}},
            True,
            False,
            COMMIT,
            1.0,
            True,
        ),
        ({"simulated": True, "gx_build": {"gx_commit": COMMIT}}, True, True, COMMIT, 0.0, False),
        ({"simulated": False, "gx_build": {"gx_commit": "main"}}, True, False, None, 0.0, True),
        ({"simulated": False, "gx_build": None}, True, False, None, 0.0, True),
        ({"simulated": "no", "gx_build": {"gx_commit": COMMIT}}, True, None, COMMIT, 0.0, True),
        ({"schema_version": 2, "simulated": False}, False, None, None, 0.0, True),
        ("{not json", False, None, None, 0.0, True),
    ],
)
def test_provenance_cases(
    tmp_path: Path, task_result, found, simulated, commit, ok, converged
) -> None:
    run_dir = make_run(tmp_path, plateau_rows())
    if isinstance(task_result, dict):
        doc = {"schema_version": 1, "gx_exit_code": 0, "outcome": "completed", **task_result}
        (run_dir / "task_result.json").write_text(json.dumps(doc), encoding="utf-8")
    elif isinstance(task_result, str):
        (run_dir / "task_result.json").write_text(task_result, encoding="utf-8")
    result = diagnose(run_dir)
    assert result["provenance"] == {
        "task_result_found": found,
        "simulated": simulated,
        "gx_commit": commit,
    }
    assert result["scalar_metrics"]["provenance_ok"] == ok
    assert result["scalar_metrics"]["simulated"] == (1.0 if simulated is True else 0.0)
    assert result["simulated"] is (simulated is True)
    assert result["converged"] is converged


# ------------------------------------------------------------------- output contract


def _walk(value: Any) -> Any:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    else:
        yield value


def test_full_output_is_strict_small_and_scalar_only(tmp_path: Path) -> None:
    times = [2.0 * k for k in range(1, 101)]
    dataset, _ = gx_nc(
        times,
        [[6.123456789, 4.98765432]] * 100,
        ParticleFlux_st=FakeVar(np.full((100, 2), 0.123456789), ("time", "s")),
        Phi2_t=FakeVar(np.full(100, 1.23456789e-5), ("time",)),
        Wg_kyst=spectrum_var([[10, 5, 2, 1, 1, 1.123456789]], "ky"),
        Wg_kxst=spectrum_var([[10, 5, 2, 1, 1, 1, 1.987654321]], "kx"),
    )
    run_dir = make_run(tmp_path, plateau_rows(), t_max=200.0)
    doc = {"schema_version": 1, "simulated": False, "gx_build": {"gx_commit": COMMIT}}
    (run_dir / "task_result.json").write_text(json.dumps(doc), encoding="utf-8")
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    text = json.dumps(result, indent=2, allow_nan=False)
    assert len(text.encode("utf-8")) < 2048, len(text)  # triage payload must stay < 2 KB
    assert all(not isinstance(v, (list, tuple)) for v in _walk(result))
    assert all(type(v) is float for v in result["scalar_metrics"].values())
    assert set(result) >= {
        "converged",
        "failure_type",
        "sim_time_reached",
        "sim_time_target",
        "scalar_metrics",
        "fluxes",
        "provenance",
        "convergence_method",
        "simulated",
    }
    assert result["converged"] is True and result["scalar_metrics"]["provenance_ok"] == 1.0


def test_internal_error_fails_closed(tmp_path: Path, capsys, monkeypatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("bug")

    monkeypatch.setattr(gxd, "diagnose", boom)
    result, _ = run_cli(capsys, str(tmp_path))
    assert result["converged"] is False and result["fluxes"] == {}
    assert result["diagnostics_error"] == "internal_error:RuntimeError"


def test_heat_flux_es_prefers_heat_flux_es_st_and_computes_heat_flux_em(tmp_path: Path) -> None:
    times = [2.0 * k for k in range(1, 101)]
    dataset, _ = gx_nc(
        times,
        [[10.0, 5.0]] * 100,  # total HeatFlux_st = 15.0
        HeatFluxES_st=FakeVar(np.full((100, 2), [4.0, 2.0], dtype=np.float32), ("time", "s")),
        HeatFluxApar_st=FakeVar(np.full((100, 2), [3.5, 1.5], dtype=np.float32), ("time", "s")),
        HeatFluxBpar_st=FakeVar(np.full((100, 2), [2.5, 1.5], dtype=np.float32), ("time", "s")),
    )
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    sm = result["scalar_metrics"]
    assert sm["heat_flux_es"] == pytest.approx(6.0)
    assert sm["heat_flux_em"] == pytest.approx(9.0)
    assert result["fluxes"]["heat_flux_es"] == pytest.approx(6.0)
    assert result["fluxes"]["heat_flux_em"] == pytest.approx(9.0)


def test_fsa_grad_x_extracted_from_geometry_jacobian_and_grho(tmp_path: Path) -> None:
    times = [2.0 * k for k in range(1, 101)]
    geometry = {
        "jacobian": FakeVar(np.asarray([1.0, 3.0], dtype=np.float64), ("z",)),
        "grho": FakeVar(np.asarray([1.2, 2.0], dtype=np.float64), ("z",)),
    }
    dataset, _ = gx_nc(times, [[3.0]] * 100, geometry=geometry)
    run_dir = make_run(tmp_path, plateau_rows())
    result = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset))
    # (1.0 * 1.2 + 3.0 * 2.0) / (1.0 + 3.0) = 7.2 / 4.0 = 1.8
    assert result["scalar_metrics"]["fsa_grad_x"] == pytest.approx(1.8)

    bad_geo = {
        "jacobian": FakeVar(np.asarray([0.0, 0.0], dtype=np.float64), ("z",)),
        "grho": FakeVar(np.asarray([1.2, 2.0], dtype=np.float64), ("z",)),
    }
    dataset_bad, _ = gx_nc(times, [[3.0]] * 100, geometry=bad_geo)
    result_bad = diagnose(run_dir, nc_opener=attach_nc(run_dir, dataset_bad))
    assert "fsa_grad_x" not in result_bad["scalar_metrics"]


def test_multi_ion_stdout_log_sums_duplicate_flux_keys(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "gx.log").write_text(
        "gx: Step 10: Time = 100.0 dt = 0.05 Phi**2 = 1.0e-3 Q_i = 1.5 Q_i = 0.5 Q_e = 1.0 Gam_i = 0.2 Gam_i = 0.3 Gam_e = 0.4\n"
        "gx: Step 20: Time = 200.0 dt = 0.05 Phi**2 = 1.0e-3 Q_i = 2.0 Q_i = 0.5 Q_e = 1.5 Gam_i = 0.1 Gam_i = 0.4 Gam_e = 0.5\n",
        encoding="utf-8",
    )
    _, _, fluxes, _, _ = gxd.parse_gx_log(run_dir / "gx.log")
    assert fluxes["Q_i"] == pytest.approx(2.5)
    assert fluxes["Q_e"] == pytest.approx(1.5)
    assert fluxes["Gam_i"] == pytest.approx(0.5)
    assert fluxes["Gam_e"] == pytest.approx(0.5)
    res = diagnose(run_dir)
    assert res["scalar_metrics"]["heat_flux_es"] == pytest.approx(4.0)
    assert res["scalar_metrics"]["particle_flux"] == pytest.approx(0.5)
