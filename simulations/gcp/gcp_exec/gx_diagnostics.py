"""`gx-diagnose <run_dir>`: compact, strict-JSON diagnostics for one GX run directory.

The GCP babysitter runs this through `tacc_exec.diagnostics_adapter.inspect_run_directory`
(config `diagnostics_cmd`, e.g. ``["gx-diagnose", "--max-flux-drift", "0.5"]``; the run
directory is appended last). Its stdout becomes `diagnostics_metrics.json` and drives the
watchdog: `converged`, `failure_type` and `fluxes` decide what enters the ML dataset.

Output: one JSON object with scalars only (no arrays, well under 4 KB) and never a
NaN/Infinity token (`allow_nan=False`):
  converged, failure_type, sim_time_reached, sim_time_target, sim_time_target_source,
  scalar_metrics, fluxes, simulated (bool), provenance {task_result_found, simulated,
  gx_commit}, flux_source ("nc" | "log" | null), convergence_method
  ("time_average_stationarity": the >= 4-sample PHYS-02 window decided;
  "short_series_last_sample": < 4 samples, legacy heuristic; "unverified": no flux series).
A metric that could not be computed is omitted, never defaulted: an absent heat flux must
not look like a valid zero-flux label.

Domain rules, exactly as adjudicated in docs/review/DOMAIN.md (math shared with the TACC
adapter through `tacc_exec.timeseries`):
- PHYS-02: heat_flux_es / particle_flux / phi2_total are means over the post-transient
  window t >= transient_discard_fraction * t_reached, plus heat_flux_es_std,
  heat_flux_es_last and flux_window_drift; drift > max_flux_drift -> not converged. Fewer
  than 4 samples keep the legacy latest-sample value.
- PHYS-01: dt_collapse_ratio = median(dts[-50:]) / median(dts[1:6]) (min(dts) / dts[0]
  below 6 steps); TIMESTEP_COLLAPSE when it is < dt_collapse_threshold, or when the recent
  dt is < dt_min_floor (optional absolute floor, off by default).
- DOM-01: the last fully written spectrum is summed over every non-k axis (species, ...);
  high_ky_tail_ratio / high_kx_tail_ratio = share of the 3 highest-|k| modes,
  high_k_tail_ratio = max -> SPECTRAL_PILEUP when > high_k_tail_threshold.
- DOM-02: sim_time_target from run_meta.json, else gx.in [Time] t_max, else the default.
Failure priority (same as the TACC adapter): DIVERGENCE > SPECTRAL_PILEUP >
TIMESTEP_COLLAPSE. Any failure, or a run declared simulated, is never converged.

Data sources and robustness rules:
- GX stdout: gx.log (Cloud Batch) or gx.out (GKE). A sample is a line such as
  ``<run>: Step 10: Time = 0.1  dt = 1.0e-03  Phi**2 = 1.2e-05  Q_i = 1.2e-03  Gam_i = 0``
  (`=` or `:` separators accepted). The incomplete last line of a log still being
  written is ignored. Any nan/inf token or "floating point exception" sets has_nan=1.
- NetCDF: <input stem>.out.nc (gx.out.nc), else the newest regular *.out.nc that is not
  hidden and not a *.tmp* staging copy. GX layout: Grids/time, Grids/ky, Grids/kx and
  Diagnostics/{HeatFlux_st, ParticleFlux_st, Phi2_t, Wg_kyst, Phi2_kyt, Wg_kxst, Phi2_kxt}
  (GX docs https://gx.readthedocs.io/en/latest/ and repo https://bitbucket.org/gyrokinetics/gx;
  the GX-parity port https://github.com/uwplasma/GKX/blob/main/docs/outputs.rst lists
  the same names). Dimension order is not assumed: time / k axes are located by
  dimension name, falling back to time-first / k-last. Heat flux = HeatFlux_st summed over
  species (= Q_i + Q_e of the log); particle flux = ParticleFlux_st of species 0 (the
  first [species] entry, ions in every repo deck; the log's Gam_i). The .out.nc series is
  preferred when readable, unless the log reaches more than 1.5 NetCDF write intervals
  further in time (truncated or partially written file).
- Missing data: masked values, _FillValue / missing_value and the netCDF default fill
  9.969209968386869e36 (NUG https://docs.unidata.ucar.edu/nug/current/attribute_conventions.html)
  are missing samples; time vs. variable length mismatch is truncated to the common
  length. Non-finite samples are excluded from every statistic and set has_nan=1.
- A truncated, locked or partially written NetCDF never crashes the run: errors are caught
  per file and per variable, scalar_metrics.nc_readable reports 1/0 (only when an .out.nc
  exists) and gx.log is used instead.
- Restart overlap (non-monotonic time): the later sample wins. Each sample removes the
  earlier samples whose time is >= its own, leaving a strictly increasing axis; the
  restarted segment supersedes the overlapping part of the previous one.
- Stale outputs: a file older than the start of the current attempt (the newer of gx.in
  and the babysitter's `.gcp_active_attempt` fence, written before each job is created)
  by more than STALE_SLACK_S belongs to a previous attempt and is ignored: logs, the
  .out.nc (also when older than the current log) and task_result.json.
- Provenance: <run_dir>/task_result.json (schema_version 1) written by the GCP task
  runner. provenance_ok = 1 only if it is present, simulated is false and gx_commit is a
  40-hex SHA; simulated runs are never converged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tacc_exec.timeseries import FluxWindowStats, dt_collapse_stats, flux_window_stats

from gcp_exec.submission import FENCE_FILENAME

INPUT_FILE = "gx.in"
LOG_FILES = ("gx.log", "gx.out")  # Batch: `gx gx.in > gx.log`; GKE: `gx gx.in > gx.out`
RUN_META_FILE = "run_meta.json"
TASK_RESULT_FILE = "task_result.json"
STALE_SLACK_S = 60.0
MAX_SMALL_FILE_BYTES = 1_048_576  # run_meta.json / task_result.json / gx.in read cap
NC_DEFAULT_FILL = 9.969209968386869e36  # NC_FILL_FLOAT == NC_FILL_DOUBLE (netcdf.h)
TAIL_POINTS = 3  # high-k tail = the 3 highest-|k| modes (legacy definition)
GX_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

NC_TIME = ("Grids/time", "time")
NC_HEAT_FLUX = ("Diagnostics/HeatFluxES_st", "Diagnostics/HeatFlux_st")
NC_HEAT_FLUX_APAR = ("Diagnostics/HeatFluxApar_st",)
NC_HEAT_FLUX_BPAR = ("Diagnostics/HeatFluxBpar_st",)
NC_PARTICLE_FLUX = ("Diagnostics/ParticleFlux_st",)
NC_PHI2 = ("Diagnostics/Phi2_t",)
NC_KY_SPECTRA = ("Diagnostics/Wg_kyst", "Diagnostics/Phi2_kyt")  # preference order
NC_KX_SPECTRA = ("Diagnostics/Wg_kxst", "Diagnostics/Phi2_kxt")
NC_KY = ("Grids/ky",)
NC_KX = ("Grids/kx",)
NC_JACOBIAN = ("Geometry/jacobian",)
NC_GRHO = ("Geometry/grho",)

NcOpener = Callable[[Path], Any]
Pairs = list[tuple[float, float]]


@dataclass(frozen=True)
class DiagnoseSettings:
    """Detection and convergence thresholds (CLI flags of the same name, with dashes).

    Same names and ranges as tacc_exec.diagnostics_adapter.DiagnosticsSettings.
    """

    dt_collapse_threshold: float = 0.01  # PHYS-01, NEEDS-PHYSICS-TEAM: TODO(david)
    dt_min_floor: float | None = None  # PHYS-01: optional absolute floor on the recent dt
    transient_discard_fraction: float = 0.5  # PHYS-02
    max_flux_drift: float = 0.5  # PHYS-02
    high_k_tail_threshold: float = 0.25  # DOM-01; TODO(david): confirm 0.25
    convergence_time_fraction: float = 0.9  # TODO(david): confirm 0.9
    sim_time_target_default: float = 200.0  # DOM-02 last resort; TODO(david): confirm 200

    def __post_init__(self) -> None:
        for name in (
            "dt_collapse_threshold",
            "dt_min_floor",
            "transient_discard_fraction",
            "max_flux_drift",
            "high_k_tail_threshold",
            "convergence_time_fraction",
            "sim_time_target_default",
        ):
            value = getattr(self, name)
            if value is None and name == "dt_min_floor":
                continue
            if not _is_number(value) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not 0.0 < self.dt_collapse_threshold < 1.0:
            raise ValueError("dt_collapse_threshold must be in (0, 1)")
        if self.dt_min_floor is not None and self.dt_min_floor <= 0.0:
            raise ValueError("dt_min_floor must be > 0 when set")
        if not 0.0 <= self.transient_discard_fraction < 1.0:
            raise ValueError("transient_discard_fraction must be in [0, 1)")
        if self.max_flux_drift <= 0.0:
            raise ValueError("max_flux_drift must be > 0")
        if not 0.0 < self.high_k_tail_threshold < 1.0:
            raise ValueError("high_k_tail_threshold must be in (0, 1)")
        if not 0.0 < self.convergence_time_fraction <= 1.0:
            raise ValueError("convergence_time_fraction must be in (0, 1]")
        if self.sim_time_target_default <= 0.0:
            raise ValueError("sim_time_target_default must be > 0")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    return _is_number(value) and math.isfinite(value)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- gx.log


_NUM = r"[-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[nN][aA][nN]|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?)"
_LOG_FIELD_RE = re.compile(
    rf"(?<![\w*])(?P<key>Time|dt|Phi\*\*2|Q_i|Q_e|Gam_i|Gam_e)\s*[=:]\s*(?P<val>{_NUM})(?![\w.])"
)
_NONFINITE_RE = re.compile(r"(?<![a-z0-9_])[+-]?(?:nan|inf(?:inity)?)(?![a-z0-9_])")
_SUMMED_LOG_KEYS = frozenset({"Q_i", "Q_e", "Gam_i", "Gam_e"})


def _parse_log_line(line: str) -> dict[str, float]:
    """Parse key=value pairs from one GX stdout line, summing multi-species flux tokens."""
    values: dict[str, float] = {}
    for match in _LOG_FIELD_RE.finditer(line):
        key = match["key"]
        val = float(match["val"])
        if key in _SUMMED_LOG_KEYS and key in values:
            values[key] += val
        else:
            values[key] = val
    return values


@dataclass(frozen=True)
class LogRow:
    """One GX stdout diagnostics line (values may be non-finite)."""

    time: float
    dt: float | None = None
    heat_flux: float | None = None  # Q_i + Q_e of the species printed on the line
    particle_flux: float | None = None  # Gam_i
    phi2: float | None = None


@dataclass
class LogSeries:
    """Everything parsed from a GX stdout log, in file order."""

    rows: list[LogRow] = field(default_factory=list)
    has_nan: bool = False
    latest: dict[str, float] = field(default_factory=dict)  # last value per key
    time_dt: list[tuple[float, float]] = field(default_factory=list)  # lines with Time and dt


def parse_gx_log_series(log_path: Path) -> LogSeries:
    """Parse a GX stdout log. Missing or unreadable file -> empty series."""
    series = LogSeries()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                lower = line.lower()
                if "floating point exception" in lower or _NONFINITE_RE.search(lower):
                    series.has_nan = True
                if not raw.endswith("\n"):
                    continue  # incomplete last line of a log that is still being written
                values = _parse_log_line(line)
                if not values:
                    continue
                series.latest.update(values)
                if "Time" not in values:
                    continue
                if "dt" in values:
                    series.time_dt.append((values["Time"], values["dt"]))
                q_parts = [values[k] for k in ("Q_i", "Q_e") if k in values]
                series.rows.append(
                    LogRow(
                        time=values["Time"],
                        dt=values.get("dt"),
                        heat_flux=sum(q_parts) if q_parts else None,
                        particle_flux=values.get("Gam_i"),
                        phi2=values.get("Phi**2"),
                    )
                )
    except OSError:
        return LogSeries(has_nan=series.has_nan)
    return series


def parse_gx_log(log_path: Path) -> tuple[list[float], list[float], dict, dict, float]:
    """Legacy view: (times, dts, latest Q_i/Q_e/Gam_i/Gam_e, {"phi2_total"}, has_nan 0/1)."""
    series = parse_gx_log_series(Path(log_path))
    fluxes = {k: series.latest[k] for k in ("Q_i", "Q_e", "Gam_i", "Gam_e") if k in series.latest}
    scalar_metrics = {"phi2_total": series.latest["Phi**2"]} if "Phi**2" in series.latest else {}
    return (
        [t for t, _ in series.time_dt],
        [dt for _, dt in series.time_dt],
        fluxes,
        scalar_metrics,
        1.0 if series.has_nan else 0.0,
    )


# ------------------------------------------------------------------- pure series helpers


def increasing_time_indices(times: Sequence[float]) -> list[int]:
    """Restart-overlap rule: indices of a strictly increasing time axis, later samples win.

    Each sample removes the previously kept samples whose time is >= its own, so a
    restarted segment supersedes the overlapping tail of the previous segment.
    `times` must be finite.
    """
    keep: list[int] = []
    for i, t in enumerate(times):
        while keep and times[keep[-1]] >= t:
            keep.pop()
        keep.append(i)
    return keep


def spectrum_tail_ratio(
    spectrum: Sequence[float],
    k_values: Sequence[float] | None = None,
    n_tail: int = TAIL_POINTS,
) -> float | None:
    """Share of a 1-D k spectrum held by its `n_tail` highest-|k| modes (DOM-01).

    The tail is chosen by |k| when the grid coordinate is given (any storage order),
    else by position (GX's ky grid is ascending). Needs more than `n_tail` finite points
    and a positive total.
    """
    vals = [float(v) for v in spectrum]
    if len(vals) <= n_tail or not all(math.isfinite(v) for v in vals):
        return None
    total = math.fsum(vals)
    if not total > 0.0:
        return None
    if k_values is not None and len(k_values) == len(vals) and all(map(_finite, k_values)):
        order = sorted(range(len(vals)), key=lambda i: (abs(k_values[i]), i))
        tail = math.fsum(vals[i] for i in order[-n_tail:])
    else:
        tail = math.fsum(vals[-n_tail:])
    return tail / total


def _last_finite(pairs: Pairs) -> float | None:
    for _, value in reversed(pairs):
        if math.isfinite(value):
            return value
    return None


def _last_finite_time(pairs: Pairs) -> float | None:
    times = [t for t, v in pairs if math.isfinite(v)]
    return max(times) if times else None


def _median_spacing(times: Sequence[float]) -> float:
    tail = list(times[-51:])
    gaps = [b - a for a, b in zip(tail, tail[1:], strict=False) if b > a]
    return statistics.median(gaps) if gaps else 0.0


# ------------------------------------------------------------------------------ NetCDF


@dataclass
class NcData:
    """Usable content of a GX .out.nc (time series already restart-deduplicated)."""

    readable: bool = False
    error: str | None = None
    times: list[float] = field(default_factory=list)
    heat_flux: Pairs = field(default_factory=list)
    heat_flux_em: Pairs = field(default_factory=list)
    particle_flux: Pairs = field(default_factory=list)
    phi2: Pairs = field(default_factory=list)
    has_nan: bool = False
    ky_ratio: float | None = None
    kx_ratio: float | None = None
    fsa_grad_x: float | None = None


def select_output_nc(run_dir: Path, input_stem: str = "gx") -> Path | None:
    """<input stem>.out.nc, else the newest regular, non-hidden, non-staging *.out.nc."""
    preferred = run_dir / f"{input_stem}.out.nc"
    if _regular_file(preferred):
        return preferred
    try:
        entries = sorted(run_dir.glob("*.out.nc"))
    except OSError:
        return None
    candidates = []
    for path in entries:
        if path.name.startswith(".") or ".tmp" in path.name or not _regular_file(path):
            continue
        mtime = _mtime(path)
        if mtime is not None:
            candidates.append((mtime, path.name, path))
    return max(candidates)[2] if candidates else None


def _default_nc_opener() -> NcOpener | None:
    # A read-only diagnostic must never block on (or fail because of) a writer's HDF5 lock.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    try:
        import netCDF4  # optional: absent in the dev venv and on TACC
    except ImportError:
        return None
    return lambda path: netCDF4.Dataset(str(path), "r")


def extract_nc_metrics(nc_path: Path, *, opener: NcOpener | None = None) -> NcData:
    """Read time series and spectra from a GX .out.nc; never raises.

    `opener(path)` returns a netCDF4-like dataset (`.variables`, `.groups`, `.close()`;
    variables support `[:]`, `.dimensions`, `.ncattrs()`/`.getncattr()`); tests inject a fake.
    """
    data = NcData()
    opener = opener or _default_nc_opener()
    if opener is None:
        data.error = "netCDF4 not installed"
        return data
    try:
        dataset = opener(nc_path)
    except Exception as exc:  # truncated, locked, half-uploaded or not NetCDF at all
        data.error = f"open failed: {type(exc).__name__}"
        return data
    try:
        _read_nc(dataset, data)
    except Exception as exc:  # defensive: a reader bug must degrade to gx.log, not crash
        data = NcData(error=f"read failed: {type(exc).__name__}")
    finally:
        try:
            dataset.close()
        except Exception:
            pass
    return data


def _read_nc(dataset: Any, data: NcData) -> None:
    raw_times = _read_flat(_find_var(dataset, NC_TIME))
    if raw_times is None:
        data.error = "time axis unreadable"
        return
    times = [t if t is not None and math.isfinite(t) else None for t in raw_times]
    valid = [k for k, t in enumerate(times) if t is not None]
    keep = {valid[i] for i in increasing_time_indices([times[k] for k in valid])}
    data.times = [times[k] for k in sorted(keep)]  # type: ignore[misc]
    if not data.times:
        data.error = "empty time axis"
        return
    data.readable = True
    flags = []
    for attr, paths, first_only in (
        ("heat_flux", NC_HEAT_FLUX, False),
        ("particle_flux", NC_PARTICLE_FLUX, True),
        ("phi2", NC_PHI2, False),
    ):
        pairs, saw_nan = _nc_series(dataset, paths, times, keep, first_only=first_only)
        setattr(data, attr, pairs)
        flags.append(saw_nan)
    apar_pairs, apar_nan = _nc_series(dataset, NC_HEAT_FLUX_APAR, times, keep, first_only=False)
    bpar_pairs, bpar_nan = _nc_series(dataset, NC_HEAT_FLUX_BPAR, times, keep, first_only=False)
    flags.extend((apar_nan, bpar_nan))
    if apar_pairs and bpar_pairs:
        bpar_by_t = dict(bpar_pairs)
        em_pairs: Pairs = []
        for t, a in apar_pairs:
            if t in bpar_by_t:
                val = a + bpar_by_t[t]
                flags.append(not math.isfinite(val))
                em_pairs.append((t, val))
        data.heat_flux_em = em_pairs
    else:
        data.heat_flux_em = apar_pairs or bpar_pairs
    data.ky_ratio, ky_nan = _nc_tail_ratio(dataset, NC_KY_SPECTRA, NC_KY, "ky")
    data.kx_ratio, kx_nan = _nc_tail_ratio(dataset, NC_KX_SPECTRA, NC_KX, "kx")
    data.fsa_grad_x = _nc_fsa_grad_x(dataset)
    data.has_nan = any(flags) or ky_nan or kx_nan


def _iter_groups(node: Any, depth: int) -> Iterator[Any]:
    yield node
    if depth <= 0:
        return
    try:
        children = sorted(dict(getattr(node, "groups", None) or {}).items())
    except Exception:
        return
    for _, child in children:
        yield from _iter_groups(child, depth - 1)


def _find_var(dataset: Any, paths: Sequence[str]) -> Any | None:
    """Explicit group paths first, then the bare name anywhere 2 group levels deep (old GX)."""
    for path in paths:
        node = dataset
        *groups, name = path.split("/")
        try:
            for group in groups:
                node = node.groups[group]
            var = node.variables.get(name)
        except Exception:
            var = None
        if var is not None:
            return var
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        for node in _iter_groups(dataset, 2):
            try:
                var = node.variables.get(name)
            except Exception:
                var = None
            if var is not None:
                return var
    return None


def _dims(var: Any) -> tuple[str, ...]:
    try:
        return tuple(str(d) for d in (getattr(var, "dimensions", None) or ()))
    except Exception:
        return ()


def _axis(dims: tuple[str, ...], ndim: int, key: str, default: int | None) -> int | None:
    if len(dims) == ndim:
        for i in reversed(range(ndim)):
            if key in dims[i].lower():
                return i
    return default


def _fill_values(var: Any) -> list[float]:
    import numpy as np

    fills: list[float] = []
    try:
        names = list(var.ncattrs())
    except Exception:
        return fills
    for name in ("_FillValue", "missing_value"):
        if name not in names:
            continue
        try:
            fills.extend(float(v) for v in np.ravel(var.getncattr(name)))
        except (TypeError, ValueError, AttributeError, KeyError):
            continue
    return fills


def _masked(var: Any) -> tuple[Any, Any]:
    """(float64 values, missing mask): masked entries and declared/default fills are missing."""
    import numpy as np  # netCDF4 always brings numpy; only imported when a file is read

    raw = var[:]
    values = np.array(np.ma.getdata(raw), dtype=np.float64)
    missing = np.array(np.ma.getmaskarray(raw), dtype=bool)
    if missing.shape != values.shape:
        missing = np.broadcast_to(missing, values.shape).copy()
    for fill in [*_fill_values(var), NC_DEFAULT_FILL]:
        missing |= np.isnan(values) if math.isnan(fill) else values == fill
    return values, missing


def _read_flat(var: Any) -> list[float | None] | None:
    if var is None:
        return None
    try:
        values, missing = _masked(var)
        flat, miss = values.reshape(-1).tolist(), missing.reshape(-1).tolist()
    except Exception:
        return None
    return [None if m else v for v, m in zip(flat, miss, strict=True)]


def _nc_series(
    dataset: Any,
    paths: Sequence[str],
    times: list[float | None],
    keep: set[int],
    *,
    first_only: bool,
) -> tuple[Pairs, bool]:
    """(time, value) pairs of a (time, species...) variable; per-variable errors -> empty."""
    import numpy as np

    saw_any_nan = False
    for path in paths:
        var = _find_var(dataset, (path,))
        if var is None:
            continue
        try:
            values, missing = _masked(var)
            t_axis = _axis(_dims(var), values.ndim, "time", 0)
            values = np.moveaxis(values, t_axis, 0).reshape(values.shape[t_axis], -1)
            missing = np.moveaxis(missing, t_axis, 0).reshape(values.shape)
        except Exception:
            continue
        if values.shape[1] == 0:
            continue
        pairs: Pairs = []
        saw_nan = False
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(min(len(times), values.shape[0])):
                t = times[k]
                if t is None or k not in keep or missing[k].any():
                    continue
                value = float(values[k, 0] if first_only else np.sum(values[k]))
                saw_nan = saw_nan or not math.isfinite(value)
                pairs.append((t, value))
        saw_any_nan = saw_any_nan or saw_nan
        if pairs:
            return pairs, saw_any_nan
    return [], saw_any_nan


def _nc_fsa_grad_x(dataset: Any) -> float | None:
    """Flux-surface-averaged ``<|grad x|> = sum(jacobian * grho) / sum(jacobian)``."""
    jac = _read_flat(_find_var(dataset, NC_JACOBIAN))
    grho = _read_flat(_find_var(dataset, NC_GRHO))
    if not jac or not grho or len(jac) != len(grho):
        return None
    if not all(v is not None and math.isfinite(v) for v in (*jac, *grho)):
        return None
    jac_f = [float(v) for v in jac if v is not None]
    grho_f = [float(v) for v in grho if v is not None]
    denom = math.fsum(jac_f)
    if not (math.isfinite(denom) and denom > 0.0):
        return None
    num = math.fsum(j * g for j, g in zip(jac_f, grho_f, strict=True))
    if not math.isfinite(num):
        return None
    ratio = num / denom
    return ratio if math.isfinite(ratio) and ratio > 0.0 else None


def _nc_tail_ratio(
    dataset: Any, spectra: Sequence[str], coord: Sequence[str], k_key: str
) -> tuple[float | None, bool]:
    k_values = _read_flat(_find_var(dataset, coord))
    saw_nan = False
    for path in spectra:
        var = _find_var(dataset, (path,))
        if var is None:
            continue
        try:
            spectrum, nan_here = _last_full_spectrum(var, k_key)
        except Exception:
            continue
        saw_nan = saw_nan or nan_here
        if spectrum is None:
            continue
        ks = k_values if k_values and len(k_values) == len(spectrum) else None
        try:
            ratio = spectrum_tail_ratio(spectrum, ks)  # type: ignore[arg-type]
        except OverflowError:  # finite but astronomically large energies: a blow-up
            return None, True
        if ratio is not None:
            return ratio, saw_nan
    return None, saw_nan


def _last_full_spectrum(var: Any, k_key: str) -> tuple[list[float] | None, bool]:
    """Latest time slice with no missing entry, summed over every non-k axis."""
    import numpy as np

    values, missing = _masked(var)
    dims = _dims(var)
    named = len(dims) == values.ndim
    t_axis = _axis(dims, values.ndim, "time", None if named else 0)
    if values.ndim < (1 if named else 2):
        return None, False
    if t_axis is None:
        slices: Any = [(values, missing)]
        rest = dims
    else:
        n_t = values.shape[t_axis]
        slices = (
            (np.take(values, i, axis=t_axis), np.take(missing, i, axis=t_axis))
            for i in reversed(range(n_t))
        )
        rest = dims[:t_axis] + dims[t_axis + 1 :] if named else ()
    for slab, slab_missing in slices:
        if slab.ndim == 0 or slab_missing.any():
            continue
        if not np.isfinite(slab).all():
            return None, True
        k_axis = _axis(rest, slab.ndim, k_key, slab.ndim - 1)
        other = tuple(i for i in range(slab.ndim) if i != k_axis)
        with np.errstate(over="ignore", invalid="ignore"):
            spectrum = slab.sum(axis=other) if other else slab
        return [float(x) for x in spectrum.tolist()], False
    return None, False


# --------------------------------------------------------------- run-dir side files


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _regular_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _is_stale(mtime: float | None, reference: float | None) -> bool:
    return mtime is not None and reference is not None and mtime < reference - STALE_SLACK_S


def _read_small_text(path: Path) -> str | None:
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_SMALL_FILE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > MAX_SMALL_FILE_BYTES:
        return None
    return raw.decode("utf-8", errors="replace")


def _read_small_json(path: Path) -> Any:
    text = _read_small_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _positive_finite(value: object) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not _finite(value) or value <= 0.0:  # type: ignore[operator]
        return None
    return float(value)  # type: ignore[arg-type]


def read_sim_time_target(run_dir: Path, default: float) -> tuple[float, str]:
    """DOM-02: run_meta.json sim_time_target, else gx.in [Time] t_max, else `default`."""
    meta = _read_small_json(run_dir / RUN_META_FILE)
    if isinstance(meta, dict):
        value = _positive_finite(meta.get("sim_time_target"))
        if value is not None:
            return value, "run_meta"
    text = _read_small_text(run_dir / INPUT_FILE)
    if text is not None:
        try:
            deck = tomllib.loads(text)
        except (tomllib.TOMLDecodeError, ValueError, RecursionError):
            deck = {}
        time_section = deck.get("Time", deck.get("time"))
        if isinstance(time_section, dict):
            value = _positive_finite(time_section.get("t_max"))
            if value is not None:
                return value, "gx_in"
    return default, "default"


@dataclass(frozen=True)
class Provenance:
    """What <run_dir>/task_result.json (GCP task runner, schema_version 1) says."""

    task_result_found: bool = False
    simulated: bool | None = None
    gx_commit: str | None = None

    @property
    def ok(self) -> bool:
        return self.task_result_found and self.simulated is False and self.gx_commit is not None

    def as_json(self) -> dict[str, Any]:
        return {
            "task_result_found": self.task_result_found,
            "simulated": self.simulated,
            "gx_commit": self.gx_commit,
        }


def read_provenance(run_dir: Path, *, reference_mtime: float | None = None) -> Provenance:
    """Parse task_result.json; missing, stale, malformed or another schema -> not found."""
    path = run_dir / TASK_RESULT_FILE
    doc = _read_small_json(path)
    if not isinstance(doc, dict) or _is_stale(_mtime(path), reference_mtime):
        return Provenance()
    version = doc.get("schema_version")
    if not _is_number(version) or version != 1:
        return Provenance()
    simulated = doc.get("simulated")
    build = doc.get("gx_build")
    commit = build.get("gx_commit") if isinstance(build, dict) else None
    commit = commit.lower() if isinstance(commit, str) else None
    return Provenance(
        task_result_found=True,
        simulated=simulated if isinstance(simulated, bool) else None,
        gx_commit=commit if commit and GX_COMMIT_RE.fullmatch(commit) else None,
    )


# ---------------------------------------------------------------------------- analysis


def _log_pairs(rows: list[LogRow], attr: str) -> Pairs:
    return [(r.time, getattr(r, attr)) for r in rows if getattr(r, attr) is not None]


def _window(pairs: Pairs, fraction: float) -> tuple[FluxWindowStats | None, bool]:
    """PHYS-02 window stats, or (None, True) when finite-but-huge samples overflow."""
    try:
        return (
            flux_window_stats(
                [t for t, _ in pairs], [v for _, v in pairs], transient_discard_fraction=fraction
            ),
            False,
        )
    except (OverflowError, statistics.StatisticsError, ZeroDivisionError):
        return None, True


def diagnose(
    run_dir: Path,
    settings: DiagnoseSettings | None = None,
    *,
    nc_opener: NcOpener | None = None,
) -> dict[str, Any]:
    """Build the gx-diagnose result for `run_dir` (never raises on bad run-dir content)."""
    settings = settings or DiagnoseSettings()
    run_dir = Path(run_dir)
    input_path = run_dir / INPUT_FILE
    # Start of the current attempt: the newer of the input deck and the babysitter's fence
    # (written before each job is created). Older outputs belong to a previous attempt.
    attempt_mtime = max(
        (m for m in (_mtime(input_path), _mtime(run_dir / FENCE_FILENAME)) if m is not None),
        default=None,
    )

    log = LogSeries()
    log_mtime = None
    log_path = next((run_dir / n for n in LOG_FILES if _regular_file(run_dir / n)), None)
    if log_path is not None and not _is_stale(_mtime(log_path), attempt_mtime):
        log_mtime = _mtime(log_path)
        log = parse_gx_log_series(log_path)
    finite_rows = [r for r in log.rows if math.isfinite(r.time)]
    rows = [finite_rows[i] for i in increasing_time_indices([r.time for r in finite_rows])]

    nc = NcData()
    nc_path = select_output_nc(run_dir, input_path.stem)
    if nc_path is not None:
        reference = max((m for m in (attempt_mtime, log_mtime) if m is not None), default=None)
        if _is_stale(_mtime(nc_path), reference):
            nc.error = "older than the current attempt / log (previous attempt)"
        else:
            nc = extract_nc_metrics(nc_path, opener=nc_opener)

    log_series: dict[str, Pairs] = {
        "heat_flux": _log_pairs(rows, "heat_flux"),
        "heat_flux_em": [],
        "particle_flux": _log_pairs(rows, "particle_flux"),
        "phi2": _log_pairs(rows, "phi2"),
    }
    nc_series = {
        "heat_flux": nc.heat_flux,
        "heat_flux_em": nc.heat_flux_em,
        "particle_flux": nc.particle_flux,
        "phi2": nc.phi2,
    }

    nc_last = _last_finite_time(nc.heat_flux)
    log_last = _last_finite_time(log_series["heat_flux"])
    flux_source: str | None = None
    if nc_last is not None and (
        log_last is None or nc_last >= log_last - 1.5 * _median_spacing(nc.times)
    ):
        flux_source = "nc"
    elif log_last is not None:
        flux_source = "log"

    def pick(name: str) -> Pairs:
        order = ("log", "nc") if flux_source == "log" else ("nc", "log")
        for src in order:
            pairs = (nc_series if src == "nc" else log_series)[name]
            if _last_finite(pairs) is not None:
                return pairs
        return []

    metrics: dict[str, float] = {}
    has_nan = log.has_nan or nc.has_nan
    window = None
    for name, key in (
        ("heat_flux", "heat_flux_es"),
        ("heat_flux_em", "heat_flux_em"),
        ("particle_flux", "particle_flux"),
        ("phi2", "phi2_total"),
    ):
        pairs = pick(name)
        stats, overflow = _window(pairs, settings.transient_discard_fraction)
        has_nan = has_nan or overflow
        if stats is not None:
            metrics[key] = stats.mean
            if name == "heat_flux":
                window = stats
                metrics["heat_flux_es_std"] = stats.std
                metrics["heat_flux_es_last"] = stats.last
                metrics["flux_window_drift"] = round(stats.drift, 6)
        elif not overflow and (latest := _last_finite(pairs)) is not None:
            metrics[key] = latest  # < 4 samples: legacy latest-sample behaviour

    if nc.fsa_grad_x is not None:
        metrics["fsa_grad_x"] = nc.fsa_grad_x

    dt_stats = dt_collapse_stats([r.dt for r in rows if r.dt is not None])
    if dt_stats is not None:
        metrics["dt_collapse_ratio"] = dt_stats.ratio
    ratios = {"high_ky_tail_ratio": nc.ky_ratio, "high_kx_tail_ratio": nc.kx_ratio}
    for key, ratio in ratios.items():
        if ratio is not None:
            metrics[key] = ratio
    if nc.ky_ratio is not None or nc.kx_ratio is not None:
        metrics["high_k_tail_ratio"] = max(r for r in ratios.values() if r is not None)
    if nc_path is not None:
        metrics["nc_readable"] = 1.0 if nc.readable else 0.0

    provenance = read_provenance(run_dir, reference_mtime=attempt_mtime)
    metrics["simulated"] = 1.0 if provenance.simulated is True else 0.0
    metrics["provenance_ok"] = 1.0 if provenance.ok else 0.0

    scalar_metrics = {k: float(v) for k, v in metrics.items() if math.isfinite(v)}
    has_nan = has_nan or len(scalar_metrics) != len(metrics)
    scalar_metrics["has_nan"] = 1.0 if has_nan else 0.0

    sim_time_reached = max([0.0, *(r.time for r in rows), *nc.times])
    sim_time_target, target_source = read_sim_time_target(run_dir, settings.sim_time_target_default)

    heat = scalar_metrics.get("heat_flux_es")
    converged = (
        heat is not None
        and sim_time_reached >= settings.convergence_time_fraction * sim_time_target
    )
    if window is not None and window.drift > settings.max_flux_drift:
        converged = False  # PHYS-02: second half of the window still drifting / growing

    failure_type: str | None = None
    tail = scalar_metrics.get("high_k_tail_ratio")
    if has_nan:
        failure_type = "DIVERGENCE"
    elif tail is not None and tail > settings.high_k_tail_threshold:
        failure_type = "SPECTRAL_PILEUP"
    elif dt_stats is not None and (
        dt_stats.ratio < settings.dt_collapse_threshold
        or (settings.dt_min_floor is not None and dt_stats.recent_dt < settings.dt_min_floor)
    ):
        failure_type = "TIMESTEP_COLLAPSE"
    if failure_type is not None or provenance.simulated is True:
        converged = False
    if window is not None:
        method = "time_average_stationarity"  # PHYS-02 window + drift gate decided
    elif heat is not None:
        method = "short_series_last_sample"  # < 4 samples: legacy heuristic
    else:
        method = "unverified"  # no usable flux series

    fluxes = {
        k: scalar_metrics[k]
        for k in ("heat_flux_es", "heat_flux_em", "particle_flux")
        if k in scalar_metrics
    }
    return {
        "converged": converged,
        "failure_type": failure_type,
        "sim_time_reached": sim_time_reached,
        "sim_time_target": sim_time_target,
        "sim_time_target_source": target_source,
        "scalar_metrics": scalar_metrics,
        "fluxes": fluxes,
        "simulated": provenance.simulated is True,
        "provenance": provenance.as_json(),
        "flux_source": flux_source,
        "convergence_method": method,
    }


# --------------------------------------------------------------------------------- CLI


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be finite: {text!r}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = DiagnoseSettings()
    parser = argparse.ArgumentParser(
        prog="gx-diagnose",
        description="Print strict-JSON convergence/failure diagnostics for a GX run directory.",
    )
    parser.add_argument("run_dir", help="GX run directory (gx.in, gx.log, gx.out.nc, ...)")
    for name, help_text in (
        ("dt_collapse_threshold", "TIMESTEP_COLLAPSE when recent/initial dt is below this"),
        ("transient_discard_fraction", "average over t >= fraction * t_reached"),
        ("max_flux_drift", "not converged when the window half-to-half drift exceeds this"),
        ("high_k_tail_threshold", "SPECTRAL_PILEUP when high_k_tail_ratio exceeds this"),
        ("convergence_time_fraction", "converged needs t_reached >= fraction * target"),
        ("sim_time_target_default", "target when neither run_meta.json nor gx.in sets one"),
    ):
        default = getattr(defaults, name)
        parser.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=_finite_float,
            default=default,
            help=f"{help_text} (default: {default})",
        )
    parser.add_argument(
        "--dt-min-floor",
        dest="dt_min_floor",
        type=_finite_float,
        default=None,
        help="also TIMESTEP_COLLAPSE when the recent dt is below this (default: off)",
    )
    return parser


def _failsafe_result(settings: DiagnoseSettings, exc: BaseException) -> dict[str, Any]:
    return {
        "converged": False,
        "failure_type": None,
        "sim_time_reached": 0.0,
        "sim_time_target": settings.sim_time_target_default,
        "sim_time_target_source": "default",
        "scalar_metrics": {},
        "fluxes": {},
        "simulated": False,
        "provenance": Provenance().as_json(),
        "flux_source": None,
        "convergence_method": "unverified",
        "diagnostics_error": f"internal_error:{type(exc).__name__}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `gx-diagnose`. Always prints one strict-JSON object."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        settings = DiagnoseSettings(
            **{k: v for k, v in vars(args).items() if k != "run_dir"},
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = diagnose(Path(args.run_dir), settings)
    except Exception as exc:  # fail closed: never converged, and say why on stderr
        print(f"gx-diagnose: internal error: {exc!r}", file=sys.stderr)
        result = _failsafe_result(settings, exc)
    sys.stdout.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
