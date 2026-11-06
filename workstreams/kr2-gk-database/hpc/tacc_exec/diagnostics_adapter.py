"""Adapter for David Hatch's diagnostic scripts & `<2 KB` scalar/log extractor.

`inspect_run_directory` turns a run directory into a `DiagnosticReport`. The watchdog uses it for
failure detection; the dataset writer uses `converged`, `convergence_method` and the flux labels.

Evidence sources, in order of authority:

1. External diagnostics: `diagnostics_metrics.json`, written from the JSON stdout of the optional
   `diagnostics_cmd` (David Hatch's scripts) or by the job itself (GCP `gx-diagnose`). When the
   file exists it is the only source of convergence and flux evidence: an explicit JSON boolean
   `converged` and the `fluxes` object are respected (docs/review/DOMAIN.md, PHYS-02), and a
   `convergence_method` it declares is passed through. Without an explicit verdict the run is not
   converged (`convergence_method="unverified"`). Ill-formed evidence sets `diagnostics_error`
   and never converges.
2. Built-in log fallback (no `diagnostics_metrics.json`): GX log lines
   (`Time = .. dt = .. Phi**2 = .. Q_i = .. Q_e = .. Gam_i = ..`), or the generic
   `time|t|sim_time`, `dt` and flux `name = value` regexes for GENE logs. With >= 4 samples the
   heat-flux label is the post-transient time average with a drift gate (PHYS-02); with fewer,
   the latest sample is reported (legacy, `convergence_method="short_series_last_sample"`).

Failure markers (NaN/Inf tokens, floating point exceptions, OOM messages) and the timestep series
used for TIMESTEP_COLLAPSE (PHYS-01) always come from the *current* log files only, so a failure
that was already handled by a previous strike can never re-trigger.

Log segments: before each resubmission the watchdog calls `consume_stale_diagnostics_metrics`,
which archives the current files as generation `n` (`gx.log` -> `gx.log.<n>`,
`diagnostics_metrics.json` -> `diagnostics_metrics.<n>.json`; `n` only increases and archived files
are never overwritten; a legacy `*.prev` file counts as generation 0). The flux time series is
rebuilt from all generations of the solver log in order and merged with the restart rule: a sample
at time `t_new` drops every earlier sample with `t >= t_new` (warm-restart overlap, cold reset).
When the current input declares a cold start (GX `[Restart] restart` / GENE
`&in_out read_checkpoint` not true), only the current log is used.

Files in the run directory are opened without following symlinks and only if they are regular
files, so a planted symlink or FIFO can neither leak another file into the triage payload nor
block the driver.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpc_core.fsutil import atomic_write_json, fsync_dir

from tacc_exec.patchers import (
    read_gene_parameters,
    read_gene_run_controls,
    read_gx_parameters,
    read_gx_run_controls,
)
from tacc_exec.timeseries import dt_collapse_stats, flux_window_stats

logger = logging.getLogger(__name__)

METRICS_FILENAME = "diagnostics_metrics.json"
RUN_META_FILENAME = "run_meta.json"
LOG_FILENAMES: tuple[str, ...] = ("gx.log", "gene.log", "slurm.out", "batch.log", "run.log")
# Solver log used for the time series, in order of preference.
_PRIMARY_LOG_ORDER: dict[str, tuple[str, ...]] = {
    "GX": ("gx.log", "slurm.out", "batch.log", "run.log"),
    "GENE": ("gene.log", "slurm.out", "batch.log", "run.log"),
}
KNOWN_FAILURE_TYPES: frozenset[str] = frozenset(
    {
        "STALL",
        "TIMESTEP_COLLAPSE",
        "SPECTRAL_PILEUP",
        "DIVERGENCE",
        "CRASH",
        "UNKNOWN",
        "OUT_OF_MEMORY",
    }
)
KNOWN_CONVERGENCE_METHODS: frozenset[str] = frozenset(
    {
        "external_diagnostics",
        "time_average_stationarity",
        "short_series_last_sample",
        "unverified",
    }
)
# Scalar flux fields with a numeric contract (manifest FLUX_FIELDS).
KNOWN_FLUX_KEYS: tuple[str, ...] = (
    "heat_flux_es",
    "heat_flux_em",
    "particle_flux",
    "momentum_flux",
    "maxwell_reynolds_stress",
    "growth_rate_gamma",
    "omega_freq",
)

MAX_DIAGNOSTIC_STDOUT_BYTES = 65_536
MAX_DIAGNOSTIC_STDERR_BYTES = 4_096
DIAGNOSTICS_CMD_TIMEOUT_S = 30.0
MAX_JSON_FILE_BYTES = 1 << 20
MAX_LOG_SCAN_BYTES = 64 << 20  # per log file; only the newest bytes are scanned beyond this
MAX_LOG_LINE_BYTES = 8_192
TAIL_READ_BYTES = 64 << 10
MAX_ERROR_CHARS = 300
_METRIC_SIG_DIGITS = 8

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# glibc/CUDA print nan/inf; gfortran/ifort print NaN/Infinity (DOM-03).
_NONFINITE_TOKEN_RE = re.compile(r"\b[+-]?(?:nan|inf|infinity)\b")
_FPE_MARKER = "floating point exception"
_OOM_MARKERS = ("out of memory", "cudaerrormemoryallocation", "oom-kill", "out-of-memory")
_NONFINITE_STRINGS = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)

_LEGACY_NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_GEN_TIME_RE = re.compile(r"\b(?:t|time|sim_time)\s*[=:]\s*(" + _LEGACY_NUM + r")")
_GEN_DT_RE = re.compile(r"\bdt\s*[=:]\s*(" + _LEGACY_NUM + r")")
_GEN_FLUX_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("q_es", re.compile(r"\b(?:heat_flux_es|q_es|heatflux)\s*[=:]\s*(" + _LEGACY_NUM + r")")),
    ("q_em", re.compile(r"\b(?:heat_flux_em|q_em)\s*[=:]\s*(" + _LEGACY_NUM + r")")),
    ("gam", re.compile(r"\b(?:particle_flux|gamma_es)\s*[=:]\s*(" + _LEGACY_NUM + r")")),
)
_GX_PAIR_RE = re.compile(r"(?<!\S)(Time|dt|Phi\*\*2|Q_i|Q_e|Gam_i|Gam_e)\s*=\s*(\S+)")
# (series attribute, report key)
_SERIES_KEYS: tuple[tuple[str, str], ...] = (
    ("q_es", "heat_flux_es"),
    ("q_em", "heat_flux_em"),
    ("gam", "particle_flux"),
    ("phi2", "phi2_total"),
)


@dataclass(frozen=True)
class DiagnosticsSettings:
    """Configurable failure-detection / convergence thresholds (config keys of the same name).

    Defaults follow docs/review/DOMAIN.md; values marked TODO(david) await physics sign-off.
    """

    dt_collapse_threshold: float = 0.01  # TODO(david): PHYS-01, NEEDS-PHYSICS-TEAM
    dt_min_floor: float | None = None
    transient_discard_fraction: float = 0.5  # PHYS-02
    max_flux_drift: float = 0.5  # PHYS-02
    high_k_tail_threshold: float = 0.25  # TODO(david): confirm high_k_tail_ratio threshold
    convergence_time_fraction: float = 0.9  # TODO(david): confirm 0.9 converged threshold

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> DiagnosticsSettings:
        """Build settings from executor config, rejecting non-finite / out-of-range values."""
        cfg = config or {}
        kwargs: dict[str, Any] = {}
        for name in (
            "dt_collapse_threshold",
            "transient_discard_fraction",
            "max_flux_drift",
            "high_k_tail_threshold",
            "convergence_time_fraction",
        ):
            if name in cfg and cfg[name] is not None:
                kwargs[name] = _finite_float_setting(name, cfg[name])
        if cfg.get("dt_min_floor") is not None:
            kwargs["dt_min_floor"] = _finite_float_setting("dt_min_floor", cfg["dt_min_floor"])
        settings = cls(**kwargs)
        if not 0.0 < settings.dt_collapse_threshold < 1.0:
            raise ValueError("dt_collapse_threshold must be in (0, 1)")
        if not 0.0 <= settings.transient_discard_fraction < 1.0:
            raise ValueError("transient_discard_fraction must be in [0, 1)")
        if settings.max_flux_drift <= 0.0:
            raise ValueError("max_flux_drift must be > 0")
        if not 0.0 < settings.high_k_tail_threshold < 1.0:
            raise ValueError("high_k_tail_threshold must be in (0, 1)")
        if not 0.0 < settings.convergence_time_fraction <= 1.0:
            raise ValueError("convergence_time_fraction must be in (0, 1]")
        if settings.dt_min_floor is not None and settings.dt_min_floor <= 0.0:
            raise ValueError("dt_min_floor must be > 0 when set")
        return settings


def _finite_float_setting(name: str, raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(f"Config setting {name!r} must be a finite number, got {raw!r}")
    return float(raw)


@dataclass
class DiagnosticReport:
    """Compact scalar & log diagnostic summary extracted locally from a simulation directory."""

    solver: str  # "GENE" or "GX"
    converged: bool
    failure_type: str | None  # None if healthy/converged
    sim_time_reached: float
    sim_time_target: float
    current_parameters: dict[str, int | float] = field(default_factory=dict)
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    log_tail: list[str] = field(default_factory=list)
    fluxes: dict[str, Any] = field(default_factory=dict)
    # How `converged` was established: "external_diagnostics" (diagnostics_metrics.json said
    # so), "time_average_stationarity" (post-transient window passed the drift gate),
    # "short_series_last_sample" (< 4 samples; legacy heuristic) or "unverified".
    convergence_method: str = "unverified"
    # Non-None when diagnostics evidence exists but is unusable (unparseable or ill-typed
    # diagnostics_metrics.json). The watchdog must never harvest such a run.
    diagnostics_error: str | None = None
    # True if the diagnostics declare synthetic output (`"simulated": true`, GX_SIMULATE_RUN).
    simulated: bool = False
    code_version: str | None = None
    flux_units: str | None = None
    # Where sim_time_target came from: "diagnostics_metrics.json", "run_meta.json",
    # "gx.in [Time] t_max", "parameters &general simtimelim" or "default".
    sim_time_target_source: str = "default"
    # Number of solver-log files (archived generations + current) merged into the series.
    log_segments_used: int = 0


@dataclass
class _Sample:
    t: float
    q_es: float | None = None
    q_em: float | None = None
    gam: float | None = None
    phi2: float | None = None


@dataclass
class _LogScan:
    samples: list[_Sample] = field(default_factory=list)
    dts: list[float] = field(default_factory=list)
    latest: dict[str, float] = field(default_factory=dict)  # latest finite value per quantity
    nonfinite: bool = False
    fpe: bool = False
    oom: bool = False


def detect_solver_and_input_file(run_dir: Path) -> tuple[str, Path]:
    """Detect whether `run_dir` is a GX (`gx.in`) or GENE (`parameters`) simulation."""
    gx_in = run_dir / "gx.in"
    gene_params = run_dir / "parameters"
    if gx_in.is_file():
        return "GX", gx_in
    if gene_params.is_file():
        return "GENE", gene_params
    raise FileNotFoundError(f"Neither 'gx.in' (GX) nor 'parameters' (GENE) found in {run_dir}")


# --------------------------------------------------------------------------------------------
# Safe file access
# --------------------------------------------------------------------------------------------
def _open_regular(path: Path) -> int | None:
    """Open read-only without following symlinks or blocking on FIFOs; None if unusable."""
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        flags |= getattr(os, name, 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Ignoring unreadable file %s: %s", path, exc.strerror or exc)
        return None
    try:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            return fd
        logger.warning("Ignoring non-regular file %s", path)
    except OSError:
        pass
    os.close(fd)
    return None


def _json_float(text: str) -> float | str:
    value = float(text)
    if math.isfinite(value):
        return value
    return "inf" if value > 0 else "-inf"  # e.g. 1e999: keep the signal, stay strict JSON


def _json_constant(token: str) -> str:
    return {"NaN": "nan", "Infinity": "inf", "-Infinity": "-inf"}.get(token, token.lower())


def _loads_strict(text: str) -> Any:
    """Parse JSON keeping NaN/Infinity as the strings "nan"/"inf" (never float NaN)."""
    return json.loads(text, parse_float=_json_float, parse_constant=_json_constant)


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return (object, error). (None, None) means the file does not exist."""
    if not os.path.lexists(path):
        return None, None
    fd = _open_regular(path)
    if fd is None:
        return None, f"{label} is not a readable regular file"
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(MAX_JSON_FILE_BYTES + 1)
    if len(data) > MAX_JSON_FILE_BYTES:
        return None, f"{label} exceeds {MAX_JSON_FILE_BYTES} bytes"
    try:
        obj = _loads_strict(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        return None, f"{label} is not valid JSON ({exc.__class__.__name__})"
    if not isinstance(obj, dict):
        return None, f"{label} is not a JSON object"
    return obj, None


def _iter_log_lines(path: Path, max_bytes: int = MAX_LOG_SCAN_BYTES) -> Iterator[str]:
    """Stream lines of a regular file: newest `max_bytes` only, each line capped in length."""
    fd = _open_regular(path)
    if fd is None:
        return
    with os.fdopen(fd, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        skip_partial = False
        if size > max_bytes:
            fh.seek(size - max_bytes)
            skip_partial = True
            logger.warning("Log %s is %d bytes; scanning only the last %d", path, size, max_bytes)
        budget = min(size, max_bytes) + MAX_LOG_LINE_BYTES  # the writer may still be appending
        while budget > 0:
            raw = fh.readline(MAX_LOG_LINE_BYTES)
            if not raw:
                break
            budget -= len(raw)
            if not raw.endswith(b"\n") and len(raw) >= MAX_LOG_LINE_BYTES:
                while True:  # discard the rest of an over-long line
                    rest = fh.readline(MAX_LOG_LINE_BYTES)
                    budget -= len(rest)
                    if not rest or rest.endswith(b"\n") or budget <= 0:
                        break
            if skip_partial:
                skip_partial = False
                continue
            yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def _read_tail_text(path: Path, max_bytes: int = TAIL_READ_BYTES) -> list[str]:
    fd = _open_regular(path)
    if fd is None:
        return []
    with os.fdopen(fd, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        start = max(0, size - max_bytes)
        fh.seek(start)
        lines = fh.read(max_bytes).decode("utf-8", errors="replace").splitlines()
    return lines[1:] if start > 0 else lines


def _read_tail_lines(run_dir: Path, max_lines: int = 25) -> list[str]:
    """Last non-empty lines (<= 180 chars, control characters removed) of the first log found."""
    for name in LOG_FILENAMES:
        path = run_dir / name
        if not os.path.lexists(path):
            continue
        lines = [ln for ln in _read_tail_text(path) if ln.strip()]
        return [_CONTROL_CHARS_RE.sub("?", ln)[:180] for ln in lines[-max_lines:]]
    return []


# --------------------------------------------------------------------------------------------
# Log segments (archived generations)
# --------------------------------------------------------------------------------------------
def _segment_name(base: str, generation: int) -> str:
    if base == METRICS_FILENAME:
        return f"diagnostics_metrics.{generation}.json"
    return f"{base}.{generation}"


def _segment_index(filename: str, base: str) -> int | None:
    if base == METRICS_FILENAME:
        if filename == "diagnostics_metrics.prev.json":
            return 0
        match = re.fullmatch(r"diagnostics_metrics\.(\d{1,9})\.json", filename)
    else:
        if filename == f"{base}.prev":
            return 0
        match = re.fullmatch(re.escape(base) + r"\.(\d{1,9})", filename)
    return int(match.group(1)) if match else None


def _list_dir(run_dir: Path) -> list[str]:
    try:
        return [entry.name for entry in os.scandir(run_dir)]
    except OSError:
        return []


def archived_log_segments(run_dir: Path, base: str) -> list[Path]:
    """Archived generations of `base` in chronological order (legacy `.prev` first)."""
    found: list[tuple[int, str]] = []
    for name in _list_dir(run_dir):
        index = _segment_index(name, base)
        if index is not None:
            found.append((index, name))
    return [run_dir / name for _, name in sorted(found)]


def _is_rotatable(path: Path) -> bool:
    try:
        return not stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def consume_stale_diagnostics_metrics(run_dir: Path) -> list[Path]:
    """Archive `diagnostics_metrics.json` and the solver logs as a new generation before resubmit.

    All files rotated together share one generation number `n` (one more than any existing
    archive): `gx.log` -> `gx.log.<n>`, `diagnostics_metrics.json` ->
    `diagnostics_metrics.<n>.json`. Archived files are never overwritten (FileExistsError).
    Returns the archived paths. Prevents stale failure metrics or NaN/OOM log strings from an
    earlier strike from re-triggering, while keeping the history for the time series.
    """
    run_dir = Path(run_dir)
    bases = (METRICS_FILENAME, *LOG_FILENAMES)
    present = [base for base in bases if _is_rotatable(run_dir / base)]
    if not present:
        return []
    names = _list_dir(run_dir)
    generation = 1 + max(
        (
            idx
            for base in bases
            for name in names
            if (idx := _segment_index(name, base)) is not None
        ),
        default=0,
    )
    archived: list[Path] = []
    for base in present:
        target = run_dir / _segment_name(base, generation)
        if os.path.lexists(target):
            raise FileExistsError(f"Refusing to overwrite archived diagnostics file {target}")
        os.rename(run_dir / base, target)
        archived.append(target)
    fsync_dir(run_dir)
    return archived


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------
def find_checkpoint_files(run_dir: Path, solver: str) -> list[Path]:
    """Return the non-empty checkpoint files a warm restart could resume from (sorted by name).

    GX: `*.restart.nc` plus the `[Restart] restart_to_file` / `restart_from_file` names when they
    are plain file names. GENE: `checkpoint*` and `s_checkpoint*`. Names starting with "." (temp
    files), symlinks, non-regular and empty files are ignored. This only says that a file exists;
    the solver (or the GCP task runner) still validates its content.
    """
    run_dir = Path(run_dir)
    solver_u = str(solver).upper()
    names = _list_dir(run_dir)
    selected: set[str] = set()
    if solver_u == "GX":
        selected.update(n for n in names if n.endswith(".restart.nc"))
        gx_in = run_dir / "gx.in"
        if gx_in.is_file():
            try:
                controls = read_gx_run_controls(gx_in)
            except Exception as exc:  # noqa: BLE001 - unreadable input: fall back to the glob
                logger.warning("Cannot read [Restart] names from %s: %s", gx_in, exc)
            else:
                for key in ("restart_to_file", "restart_from_file"):
                    if controls.get(key):
                        selected.add(str(controls[key]))
    elif solver_u == "GENE":
        selected.update(n for n in names if n.startswith(("checkpoint", "s_checkpoint")))
    else:
        raise ValueError(f"Unsupported solver {solver!r}")

    found: list[Path] = []
    for name in sorted(selected):
        if name.startswith(".") or "/" in name:
            continue
        path = run_dir / name
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and st.st_size > 0:
            found.append(path)
    return found


# --------------------------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------------------------
def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_gx_line(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    q_parts: list[float] = []
    gam_i_parts: list[float] = []
    gam_e_parts: list[float] = []
    for name, raw in _GX_PAIR_RE.findall(line):
        val = _to_float(raw)
        if val is None:
            continue
        if name == "Time":
            values.setdefault("t", val)
        elif name == "dt":
            values.setdefault("dt", val)
        elif name == "Phi**2":
            values["phi2"] = val
        elif name == "Gam_i":
            gam_i_parts.append(val)
        elif name == "Gam_e":
            gam_e_parts.append(val)
        elif name in ("Q_i", "Q_e"):  # total heat flux across species (GX stdout convention)
            q_parts.append(val)
    if q_parts:
        values["q_es"] = math.fsum(q_parts)
    if gam_i_parts:
        # Prefer ion particle flux when present (ambipolar Z_i*Gam_i + Z_e*Gam_e = 0);
        # sum across multiple ion species when printed on the same line (GXSRC-07).
        values["gam"] = math.fsum(gam_i_parts)
        if any(not math.isfinite(v) for v in gam_e_parts):
            values["gam"] = math.nan
    elif gam_e_parts:
        values["gam"] = math.fsum(gam_e_parts)
    return values


def _parse_generic_line(lower: str) -> dict[str, float]:
    values: dict[str, float] = {}
    times = [v for v in (_to_float(m) for m in _GEN_TIME_RE.findall(lower)) if v is not None]
    if times:
        values["t"] = max(times)
    dt_match = _GEN_DT_RE.search(lower)
    if dt_match and (dt_val := _to_float(dt_match.group(1))) is not None:
        values["dt"] = dt_val
    for key, pattern in _GEN_FLUX_RES:
        matches = pattern.findall(lower)
        if matches and (flux_val := _to_float(matches[-1])) is not None:
            values[key] = flux_val
    return values


def _scan_log(path: Path, solver: str, *, markers: bool) -> _LogScan:
    scan = _LogScan()
    for line in _iter_log_lines(path):
        lower = line.lower()
        if markers:
            if _NONFINITE_TOKEN_RE.search(lower):
                scan.nonfinite = True
            if _FPE_MARKER in lower:
                scan.fpe = True
            if any(marker in lower for marker in _OOM_MARKERS):
                scan.oom = True
        values = _parse_gx_line(line) if solver == "GX" else _parse_generic_line(lower)
        if not values:
            continue
        for attr, _report_key in _SERIES_KEYS:
            if attr in values and math.isfinite(values[attr]):
                scan.latest[attr] = values[attr]
        if "dt" in values:
            scan.dts.append(values["dt"])
        t_val = values.get("t")
        if t_val is not None and math.isfinite(t_val) and t_val >= 0.0:
            scan.samples.append(
                _Sample(
                    t=t_val,
                    q_es=values.get("q_es"),
                    q_em=values.get("q_em"),
                    gam=values.get("gam"),
                    phi2=values.get("phi2"),
                )
            )
    return scan


def _scan_tail_markers(path: Path, scan: _LogScan) -> None:
    """Crash markers printed outside the solver log (bash `Floating point exception`, Slurm
    cgroup `oom-kill` messages) usually land in the batch stdout/stderr file."""
    for line in _read_tail_text(path):
        lower = line.lower()
        if _FPE_MARKER in lower:
            scan.fpe = True
        if any(marker in lower for marker in _OOM_MARKERS):
            scan.oom = True


def merge_restart_segments(segments: Sequence[Sequence[_Sample]]) -> list[_Sample]:
    """Concatenate per-segment samples applying the restart rule (drop earlier t >= t_new)."""
    merged: list[_Sample] = []
    for samples in segments:
        for sample in samples:
            while merged and merged[-1].t >= sample.t:
                merged.pop()
            merged.append(sample)
    return merged


def _select_primary_log(run_dir: Path, solver: str) -> str | None:
    names = set(_list_dir(run_dir))
    for base in _PRIMARY_LOG_ORDER.get(solver, LOG_FILENAMES):
        if base in names or archived_log_segments(run_dir, base):
            return base
    return None


# --------------------------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------------------------
def _compact(value: float) -> float:
    """Round to 8 significant digits (keeps tiny values like dt=1e-7, unlike round(x, 6))."""
    return float(f"{value:.{_METRIC_SIG_DIGITS}g}")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def _is_nonfinite_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _NONFINITE_STRINGS


def _sanitize_finite_metrics(raw: dict[str, Any]) -> tuple[dict[str, float], bool]:
    """Filter scalar metrics to finite floats (`allow_nan=False`), flagging `has_nan=1.0` if NaN/Inf seen."""
    clean: dict[str, float] = {}
    saw_non_finite = False
    for k, v in raw.items():
        if isinstance(v, bool):
            clean[str(k)[:32]] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                clean[str(k)[:32]] = _compact(fv)
            else:
                saw_non_finite = True
        elif _is_nonfinite_token(v):
            saw_non_finite = True
    if saw_non_finite:
        clean["has_nan"] = 1.0
    return clean, saw_non_finite


def _validate_external_fluxes(raw: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Known flux fields must be finite numbers or null; anything else is a diagnostics error.

    A non-finite flux is reported as a diagnostics error (never converged, never harvested)
    rather than DIVERGENCE: a tool writing NaN for "not computed" must not kill healthy runs.
    """
    fluxes: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if name not in KNOWN_FLUX_KEYS:
            fluxes[name] = value
            continue
        if value is None:
            fluxes[name] = None
        elif _finite_number(value) is not None:
            fluxes[name] = value
        else:
            errors.append(f"diagnostics_metrics.json fluxes.{name} is not a finite number")
    return fluxes


def _positive_finite(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number > 0.0 else None


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------------------------
# External diagnostics command
# --------------------------------------------------------------------------------------------
def _run_bounded(
    argv: list[str], *, timeout_s: float, max_stdout: int, max_stderr: int
) -> tuple[int | None, bytes, bytes, str]:
    """Run `argv` (no shell, stdin closed) with bounded output capture and wall time.

    Returns (returncode, stdout, stderr, status), status in {"ok", "timeout", "overflow"}.
    The child's whole process group is killed on timeout or stdout overflow.
    """
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    out = bytearray()
    err = bytearray()
    status = "ok"
    deadline = time.monotonic() + timeout_s
    selector = selectors.DefaultSelector()
    try:
        assert proc.stdout is not None and proc.stderr is not None
        selector.register(proc.stdout, selectors.EVENT_READ, "out")
        selector.register(proc.stderr, selectors.EVENT_READ, "err")
        while selector.get_map() and status == "ok":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timeout"
                break
            for key, _events in selector.select(timeout=remaining):
                chunk = os.read(key.fd, 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "out":
                    if len(out) + len(chunk) > max_stdout:
                        status = "overflow"
                        break
                    out += chunk
                elif len(err) < max_stderr:
                    err += chunk[: max_stderr - len(err)]
        if status == "ok":
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        if status != "ok" or proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5.0)
        selector.close()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
    return proc.returncode, bytes(out), bytes(err), status


def _refresh_external_diagnostics(run_dir: Path, diagnostics_cmd: Any) -> str | None:
    """Run the configured diagnostics command; store its JSON object atomically.

    Returns an error string (diagnostics evidence unusable) or None on success.
    """
    if (
        not isinstance(diagnostics_cmd, (list, tuple))
        or not diagnostics_cmd
        or not all(isinstance(arg, str) and arg for arg in diagnostics_cmd)
    ):
        return "diagnostics_cmd must be a non-empty list of strings"
    argv = [*diagnostics_cmd, str(run_dir)]
    try:
        returncode, stdout, stderr, status = _run_bounded(
            argv,
            timeout_s=DIAGNOSTICS_CMD_TIMEOUT_S,
            max_stdout=MAX_DIAGNOSTIC_STDOUT_BYTES,
            max_stderr=MAX_DIAGNOSTIC_STDERR_BYTES,
        )
    except OSError as exc:
        return f"diagnostics_cmd could not be started ({exc.__class__.__name__})"
    if status == "timeout":
        return f"diagnostics_cmd timed out after {DIAGNOSTICS_CMD_TIMEOUT_S:.0f} s"
    if status == "overflow":
        return f"diagnostics_cmd stdout exceeded {MAX_DIAGNOSTIC_STDOUT_BYTES} bytes"
    if returncode != 0:
        logger.warning(
            "diagnostics_cmd exited with %s for %s: %s",
            returncode,
            run_dir,
            stderr.decode("utf-8", errors="replace").strip()[-500:],
        )
        return f"diagnostics_cmd exited with code {returncode}"
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return "diagnostics_cmd produced no output"
    try:
        data = _loads_strict(text)
    except (ValueError, RecursionError):
        return "diagnostics_cmd output is not valid JSON"
    if not isinstance(data, dict):
        return "diagnostics_cmd output is not a JSON object"
    atomic_write_json(run_dir / METRICS_FILENAME, data)
    return None


# --------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------
def _read_run_controls(solver: str, input_path: Path) -> tuple[Any, bool | None, str]:
    """Return (raw time limit, restart requested?, time-limit label) from the solver input."""
    if solver == "GX":
        gx_controls = read_gx_run_controls(input_path)
        time_limit, restart = gx_controls.get("t_max"), gx_controls.get("restart")
        label = "gx.in [Time] t_max"
    else:
        gene_controls = read_gene_run_controls(input_path)
        time_limit, restart = gene_controls.get("simtimelim"), gene_controls.get("read_checkpoint")
        label = "parameters &general simtimelim"
    return time_limit, (restart if isinstance(restart, bool) else None), label


def _resolve_sim_time_target(
    raw_diag: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    time_limit: Any,
    time_limit_label: str,
    default: float,
    errors: list[str],
) -> tuple[float, str]:
    """DOM-02: diagnostics_metrics.json, then run_meta.json, then the input deck, then default."""
    for label, source in ((METRICS_FILENAME, raw_diag), (RUN_META_FILENAME, meta)):
        if source is None or source.get("sim_time_target") is None:
            continue
        value = _positive_finite(source["sim_time_target"])
        if value is None:
            errors.append(f"{label} sim_time_target must be a finite number > 0")
            continue
        return value, label
    if time_limit is not None:
        value = _positive_finite(time_limit)
        if value is not None:
            return value, time_limit_label
        errors.append(f"{time_limit_label} must be a finite number > 0")
    fallback = _positive_finite(default)
    if fallback is None:
        errors.append("sim_time_target_default must be a finite number > 0")
        fallback = 200.0
    return fallback, "default"


def inspect_run_directory(
    run_dir: Path,
    diagnostics_cmd: list[str] | None = None,
    sim_time_target_default: float = 200.0,  # TODO(david): confirm default 200.0 sim time target
    *,
    settings: DiagnosticsSettings | None = None,
) -> DiagnosticReport:
    """Run optional external diagnostic command (David's script) and parse scalar metrics + logs.

    Never marks a run converged on unusable evidence; see the module docstring for the rules.
    """
    settings = settings or DiagnosticsSettings()
    run_dir = Path(run_dir)
    solver, input_path = detect_solver_and_input_file(run_dir)
    current_params = (
        read_gx_parameters(input_path) if solver == "GX" else read_gene_parameters(input_path)
    )
    time_limit, restart_requested, time_limit_label = _read_run_controls(solver, input_path)
    errors: list[str] = []

    if diagnostics_cmd:
        cmd_error = _refresh_external_diagnostics(run_dir, diagnostics_cmd)
        if cmd_error:
            errors.append(cmd_error)
            logger.warning("External diagnostics unusable for %s: %s", run_dir, cmd_error)

    raw_diag, raw_error = _read_json_object(run_dir / METRICS_FILENAME, METRICS_FILENAME)
    if raw_error:
        errors.append(raw_error)
    meta, meta_error = _read_json_object(run_dir / RUN_META_FILENAME, RUN_META_FILENAME)
    if meta_error:
        errors.append(meta_error)
    sim_time_target, target_source = _resolve_sim_time_target(
        raw_diag, meta, time_limit, time_limit_label, sim_time_target_default, errors
    )

    # Current solver log: failure markers + the timestep series (current generation only).
    primary = _select_primary_log(run_dir, solver)
    current = _scan_log(run_dir / primary, solver, markers=True) if primary else _LogScan()
    for name in LOG_FILENAMES:
        if name != primary and os.path.lexists(run_dir / name):
            _scan_tail_markers(run_dir / name, current)

    scalar_metrics: dict[str, float] = {}
    fluxes: dict[str, Any] = {}
    saw_nan = current.nonfinite or current.fpe
    converged = False
    method = "unverified"
    explicit_ft: str | None = None
    stalled = False
    simulated = False
    code_version: str | None = None
    flux_units: str | None = None
    sim_time_reached = 0.0
    segments_used = 0

    if raw_diag is not None or raw_error:
        # External diagnostics exist: they are the only convergence / flux evidence.
        raw = raw_diag or {}
        reached = raw.get("sim_time_reached")
        if reached is not None:
            number = _finite_number(reached)
            if number is None or number < 0.0:
                errors.append(f"{METRICS_FILENAME} sim_time_reached must be a finite number >= 0")
            else:
                sim_time_reached = number

        raw_metrics = raw.get("scalar_metrics")
        if raw_metrics is not None and not isinstance(raw_metrics, dict):
            errors.append(f"{METRICS_FILENAME} scalar_metrics must be an object")
            raw_metrics = None
        scalar_metrics, metrics_nonfinite = _sanitize_finite_metrics(raw_metrics or {})
        saw_nan = saw_nan or metrics_nonfinite

        raw_fluxes = raw.get("fluxes")
        if raw_fluxes is not None and not isinstance(raw_fluxes, dict):
            errors.append(f"{METRICS_FILENAME} fluxes must be an object")
            raw_fluxes = None
        fluxes = _validate_external_fluxes(raw_fluxes or {}, errors)
        if not fluxes:
            fluxes = {
                key: scalar_metrics[key]
                for key in ("heat_flux_es", "heat_flux_em", "particle_flux")
                if key in scalar_metrics
            }

        verdict = raw.get("converged")
        if verdict is not None and not isinstance(verdict, bool):
            errors.append(f"{METRICS_FILENAME} converged must be a JSON boolean")
        elif isinstance(verdict, bool):
            converged = verdict
            method = "external_diagnostics"
            declared = raw.get("convergence_method")
            if declared is not None:
                if isinstance(declared, str) and declared in KNOWN_CONVERGENCE_METHODS:
                    method = declared
                else:
                    errors.append(
                        f"{METRICS_FILENAME} convergence_method {_short(declared)} is unknown"
                    )
                    method = "unverified"

        ft_raw = raw.get("failure_type")
        if isinstance(ft_raw, str) and ft_raw.strip().upper() in KNOWN_FAILURE_TYPES:
            explicit_ft = ft_raw.strip().upper()
        elif ft_raw is not None and not (isinstance(ft_raw, str) and not ft_raw.strip()):
            errors.append(f"{METRICS_FILENAME} failure_type {_short(ft_raw)} is not a known type")
            explicit_ft = "UNKNOWN"

        stalled_raw = raw.get("stalled")
        if isinstance(stalled_raw, bool):
            stalled = stalled_raw
        elif stalled_raw is not None:
            errors.append(f"{METRICS_FILENAME} stalled must be a JSON boolean")

        simulated_raw = raw.get("simulated")
        if isinstance(simulated_raw, bool):
            simulated = simulated_raw
        elif simulated_raw is not None:
            errors.append(f"{METRICS_FILENAME} simulated must be a JSON boolean")
            simulated = True  # fail closed: unknown provenance is treated as synthetic
        provenance = raw.get("provenance")
        if isinstance(provenance, dict):
            if provenance.get("simulated") is True:
                simulated = True
            commit = provenance.get("gx_commit")
            if isinstance(commit, str) and _COMMIT_RE.match(commit):
                code_version = commit
        if isinstance(raw.get("code_version"), str) and raw["code_version"].strip():
            code_version = raw["code_version"].strip()[:128]
        if isinstance(raw.get("flux_units"), str) and raw["flux_units"].strip():
            flux_units = raw["flux_units"].strip()[:64]
    else:
        # Built-in log fallback: merge every generation of the solver log (restart rule).
        segments: list[list[_Sample]] = []
        if primary and restart_requested is True:
            for segment in archived_log_segments(run_dir, primary):
                segments.append(_scan_log(segment, solver, markers=False).samples)
        if primary and os.path.lexists(run_dir / primary):
            segments.append(current.samples)
        segments_used = sum(1 for seg in segments if seg)
        merged = merge_restart_segments(segments)
        sim_time_reached = merged[-1].t if merged else 0.0

        window = None
        for attr, key in _SERIES_KEYS:
            pairs = [
                (s.t, float(v)) for s in merged if _finite_number(v := getattr(s, attr)) is not None
            ]
            times = [t for t, _ in pairs]
            values = [v for _, v in pairs]
            try:
                stats = flux_window_stats(
                    times, values, transient_discard_fraction=settings.transient_discard_fraction
                )
            except OverflowError:
                # Finite samples near the float limit: the run is blowing up. Never label
                # it; flag it as a divergence.
                saw_nan = True
                continue
            if stats is not None:
                scalar_metrics[key] = _compact(stats.mean)
                if attr == "q_es":
                    window = stats
            elif values:
                scalar_metrics[key] = _compact(values[-1])
            elif attr in current.latest:
                scalar_metrics[key] = _compact(current.latest[attr])
        if window is not None:
            scalar_metrics["heat_flux_es_std"] = _compact(window.std)
            scalar_metrics["heat_flux_es_last"] = _compact(window.last)
            scalar_metrics["flux_window_drift"] = _compact(window.drift)
            scalar_metrics["flux_window_samples"] = float(window.n_samples)
            scalar_metrics["flux_window_t_start"] = _compact(window.t_start)

        time_ok = sim_time_reached >= settings.convergence_time_fraction * sim_time_target
        if not current.samples:
            method = "unverified"  # the latest job produced no output: never certify
        elif window is not None:
            method = "time_average_stationarity"
            converged = time_ok and window.drift <= settings.max_flux_drift
        elif "heat_flux_es" in scalar_metrics:
            method = "short_series_last_sample"
            converged = time_ok
        fluxes = {
            key: scalar_metrics[key]
            for key in ("heat_flux_es", "heat_flux_em", "particle_flux")
            if key in scalar_metrics
        }

        dt_stats = dt_collapse_stats(current.dts)
        if dt_stats is not None:
            scalar_metrics["dt_collapse_ratio"] = _compact(dt_stats.ratio)
            scalar_metrics["dt_initial"] = _compact(dt_stats.initial_dt)
            scalar_metrics["dt_recent"] = _compact(dt_stats.recent_dt)

    if scalar_metrics.get("simulated") == 1.0:
        simulated = True
    if saw_nan:
        scalar_metrics["has_nan"] = 1.0

    failure_type: str | None = None
    dt_ratio = scalar_metrics.get("dt_collapse_ratio", 1.0)
    dt_recent = scalar_metrics.get("dt_recent")
    if explicit_ft is not None:
        failure_type = explicit_ft
    elif current.oom:
        failure_type = "OUT_OF_MEMORY"
    elif saw_nan or scalar_metrics.get("has_nan", 0.0) >= 1.0:
        failure_type = "DIVERGENCE"
    elif scalar_metrics.get("high_k_tail_ratio", 0.0) > settings.high_k_tail_threshold:
        failure_type = "SPECTRAL_PILEUP"
    elif 0.0 < dt_ratio < settings.dt_collapse_threshold or (
        settings.dt_min_floor is not None
        and dt_recent is not None
        and 0.0 < dt_recent < settings.dt_min_floor
    ):
        failure_type = "TIMESTEP_COLLAPSE"
    elif stalled:
        failure_type = "STALL"

    diagnostics_error = "; ".join(errors)[:MAX_ERROR_CHARS] if errors else None
    return DiagnosticReport(
        solver=solver,
        converged=bool(converged and failure_type is None and diagnostics_error is None),
        failure_type=failure_type,
        sim_time_reached=sim_time_reached,
        sim_time_target=sim_time_target,
        current_parameters=current_params,
        scalar_metrics=scalar_metrics,
        log_tail=_read_tail_lines(run_dir, max_lines=20),
        fluxes=fluxes,
        convergence_method=method,
        diagnostics_error=diagnostics_error,
        simulated=simulated,
        code_version=code_version,
        flux_units=flux_units,
        sim_time_target_source=target_source,
        log_segments_used=segments_used,
    )
