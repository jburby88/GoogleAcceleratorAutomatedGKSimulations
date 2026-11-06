"""Campaign dataset harvester: converged GK runs -> immutable file-per-run dataset entries.

Layout produced under ``<dataset_root>/campaigns/<campaign_id>/``::

    equilibria/<equilibrium_id>/
        equilibrium_meta.json   written once, by the first run that references the equilibrium
        <geometry files>        (*eik*, wout_*.nc, geometry*, equilibrium*) copied once; a later
                                run whose copy differs never overwrites it (that run is marked
                                ineligible instead)
    runs/<run_id>/
        <gx.in | parameters>    final solver input file
        fluxes.json             reduced transport scalars + ``flux_units``
        checksums.sha256        ``sha256sum -c`` file for the two files above
        run_manifest.json       provenance + dataset-eligibility verdict (the commit marker)

Relation to Zenodo record 14867777 (DOM-06, GXSRC-08, ZEN-12): that record ("Data for the paper
'How does ion temperature gradient turbulence depend on magnetic geometry? Insights from data and
machine learning'", Landreman et al., 2025) ships GX stellarator results as an HDF5 archive plus
a tarball. In GX (`src/geometry.cu:955-960`), raw flux outputs (`Diagnostics/HeatFlux*`,
`ParticleFlux*`) divide gyroBohm-normalized fluxes $Q / Q_{\\text{gB}}$ by the flux-surface
average $\\langle |\\nabla x| \\rangle$ (``fsa_grad_x``), matching Zenodo's
``Q_avgs_divided_by_FSA_grad_x`` dataset array, whereas Zenodo's primary ``Q_avgs`` array
multiplies by ``FSA_grad_x`` to undo that division. ``fluxes.json`` keeps ``flux_units =
"gyroBohm"`` for schema-2.0 compatibility while recording ``flux_normalization``
(``"gyroBohm_divided_by_fsa_grad_x"`` for GX, ``"gyroBohm_gene"`` for GENE) and optional
``fsa_grad_x`` so downstream loaders never mix conventions silently. This file-per-run campaign
layout is not the same HDF5 packaging as the Zenodo archive.

Commit protocol (Lustre ``$WORK``/``$SCRATCH`` and GCS FUSE alike; no directory renames):
every file is written with temp file + fsync + rename, and ``run_manifest.json`` is written
last. A run directory without ``run_manifest.json`` is an uncommitted partial entry: readers
must ignore it and the next harvest rewrites it. A committed entry is immutable (ML-08):
re-harvesting the same submission (same final job id and submit nonce) returns the existing
manifest; anything else raises DatasetEntryExistsError.

Dataset eligibility (``eligibility_policy_version`` "1"): ``dataset_eligible`` is true iff
``eligibility_reasons`` is empty. Anything that cannot be verified counts against eligibility
(fail closed); the codes are listed in ELIGIBILITY_REASONS. ``eligibility_warnings`` never
block. Strict JSON only: non-finite numbers are written as null, never as NaN/Infinity.

Zero AI/Cloud SDK dependencies (stdlib only).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from hpc_core.fsutil import SAFE_ID_RE, atomic_write_bytes, fsync_dir, is_safe_id, sha256_file
from hpc_core.state import RunRecord

MANIFEST_SCHEMA_VERSION = "2.0"
FLUXES_SCHEMA_VERSION = "2.0"
EQUILIBRIUM_META_SCHEMA_VERSION = "2.0"
ELIGIBILITY_POLICY_VERSION = "1"
# Both GX and GENE express transport in gyroBohm units, but GX divides raw flux outputs by
# `<|grad x|>` (`src/geometry.cu:955-960`, matching Zenodo 14867777 `Q_avgs_divided_by_FSA_grad_x`
# rather than `Q_avgs`; see GXSRC-08 / ZEN-12). `flux_units` stays `"gyroBohm"` for schema-2.0
# backward compatibility while `flux_normalization` disambiguates the geometric convention.
DEFAULT_FLUX_UNITS = "gyroBohm"
FLUX_NORMALIZATION_BY_SOLVER: dict[str, str] = {
    "GX": "gyroBohm_divided_by_fsa_grad_x",
    "GENE": "gyroBohm_gene",
}
ZENODO_REFERENCE = "https://zenodo.org/records/14867777"
HARVESTER_NAME = "ut-hpc-core"

MANIFEST_NAME = "run_manifest.json"
FLUXES_NAME = "fluxes.json"
CHECKSUMS_NAME = "checksums.sha256"
EQUILIBRIUM_META_NAME = "equilibrium_meta.json"

PRIMARY_FLUX_FIELD = "heat_flux_es"
FLUX_FIELDS = (
    "heat_flux_es",
    "heat_flux_em",
    "particle_flux",
    "maxwell_reynolds_stress",
    "momentum_flux",
    "growth_rate_gamma",
    "omega_freq",
)
GEOMETRY_PATTERNS = ("*eik*", "wout_*.nc", "geometry*", "equilibrium*")
GCP_BACKENDS = frozenset({"gcp_batch", "gcp_k8s"})
ELIGIBLE_ENVIRONMENT = "prod"
# `converged=True` established by these DiagnosticReport.convergence_method values is too weak
# for the dataset: no post-transient stationarity check was possible (PHYS-02).
WEAK_CONVERGENCE_METHODS = frozenset({"short_series_last_sample", "unverified"})

# Eligibility blockers in evaluation order. Any one of them makes dataset_eligible false.
ELIGIBILITY_REASONS = (
    "status_not_completed",  # record.status != "COMPLETED"
    "convergence_unknown",  # `converged` not passed (None) or not a bool
    "not_converged",  # converged=False
    "convergence_unverified",  # converged=True via a WEAK_CONVERGENCE_METHODS method
    "simulated",  # simulated=True, or final_metrics["simulated"] raised (GX_SIMULATE_RUN)
    "non_prod_environment",  # environment != "prod" (not passed counts as not prod)
    "escalated",  # any escalation / human resolution in the run's history
    "missing_equilibrium_id",  # empty or unsafe equilibrium_id: no equilibria/ entry written
    "invalid_heat_flux",  # heat_flux_es missing, bool, non-numeric or non-finite
    "invalid_flux_value",  # another FLUX_FIELDS value is present but not a finite number
    "missing_input_file",  # no unambiguous solver input file in run_dir
    "input_modified_after_submission",  # input sha256 != record.input_sha256_submitted
    "gcp_provenance_unverified",  # GCP backend and final_metrics["provenance_ok"] != 1.0
    "equilibrium_geometry_conflict",  # a geometry file differs from the published copy
    "equilibrium_metadata_conflict",  # equilibrium_meta.json unreadable, or origin differs
)
ELIGIBILITY_WARNINGS = ("equilibrium_origin_unknown", "code_version_unknown")

_COPY_CHUNK = 1 << 20
_RESERVED_GEOMETRY_NAMES = frozenset({EQUILIBRIUM_META_NAME})
_MAX_LABEL_LEN = 256


class DatasetEntryExistsError(RuntimeError):
    """A committed ``runs/<run_id>/run_manifest.json`` exists for a different submission."""


# --------------------------------------------------------------------------------------------
# Value helpers
# --------------------------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Return a strict-JSON copy of `value`: non-finite numbers become None.

    Unknown types raise TypeError: an unexpected object in a dataset record is a bug, and it
    must fail the harvest loudly rather than be stringified into the dataset.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not allowed in dataset JSON")


def _render_json(obj: Any) -> bytes:
    rendered = json.dumps(_json_safe(obj), indent=2, sort_keys=True, allow_nan=False)
    return (rendered + "\n").encode("utf-8")


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON token {token!r}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)


def _is_finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


def _extract_fsa_grad_x(metrics: Mapping[str, Any]) -> float | None:
    """Return finite positive `<|grad x|>` (`fsa_grad_x`) if present in `metrics`, else None."""
    candidates: list[Any] = [metrics.get("fsa_grad_x")]
    for nested_key in ("scalar_metrics", "fluxes"):
        nested = metrics.get(nested_key)
        if isinstance(nested, Mapping):
            candidates.append(nested.get("fsa_grad_x"))
    for raw in candidates:
        if raw is None:
            continue
        if _is_finite_real(raw) and float(raw) > 0.0:
            return float(raw)
        return None
    return None


def _flag_raised(value: Any) -> bool:
    """Fail-closed reading of a 'bad news' flag: only absent/None/False/0 mean 'not raised'."""
    if value is None or value is False:
        return False
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return not float(value) == 0.0  # NaN counts as raised
    return True


def _flag_confirmed(value: Any) -> bool:
    """Strict reading of a 'good news' flag: only True or exactly 1.0 count."""
    if value is True:
        return True
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and value == 1.0


def _rounded_or_none(value: Any) -> float | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return round(as_float, 2) if math.isfinite(as_float) else None


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()[:_MAX_LABEL_LEN]
    return None


@lru_cache(maxsize=1)
def _harvester_version() -> str:
    try:
        return importlib.metadata.version(HARVESTER_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# --------------------------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------------------------
def _default_file_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _mkdirs_durable(path: Path) -> None:
    """`mkdir -p`, then fsync the parent of every directory this call created."""
    created: list[Path] = []
    probe = path
    while not probe.is_dir() and probe.parent != probe:
        created.append(probe)
        probe = probe.parent
    path.mkdir(parents=True, exist_ok=True)
    for new_dir in reversed(created):
        fsync_dir(new_dir.parent)


def _safe_makedirs(root: Path, *parts: str) -> Path:
    """Create `root/parts...` one level at a time, refusing anything that leaves `root`.

    `parts` are validated path components; this also rejects pre-existing symlinks (or
    non-directories) at any level that would redirect writes outside the dataset root.
    """
    current = root
    for part in parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            try:
                current.mkdir()
            except FileExistsError:
                pass
            else:
                fsync_dir(current.parent)
        resolved = current.resolve()
        if not resolved.is_dir() or not resolved.is_relative_to(root):
            raise ValueError(
                f"Refusing to write into {current}: it resolves to {resolved}, which is not a "
                f"directory inside the dataset root {root}"
            )
    return current


def _atomic_copy(src: Path, dest: Path) -> tuple[str, int]:
    """Stream-copy `src` to `dest` (temp + fsync + rename); return (sha256 hex, size)."""
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
            while block := inp.read(_COPY_CHUNK):
                digest.update(block)
                size += len(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        with contextlib.suppress(OSError):  # some FUSE mounts ignore or refuse chmod
            os.chmod(tmp_name, _default_file_mode())
        os.replace(tmp_name, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    fsync_dir(dest.parent)
    return digest.hexdigest(), size


def _write_bytes(path: Path, data: bytes) -> dict[str, Any]:
    atomic_write_bytes(path, data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


# --------------------------------------------------------------------------------------------
# Run-directory inspection
# --------------------------------------------------------------------------------------------
def _select_input_file(solver: str, run_dir: Path) -> Path | None:
    """GENE: `parameters`. GX: `gx.in`, else a single unambiguous `*.in`. Otherwise None."""
    if not run_dir.is_dir():
        return None
    if solver == "GENE":
        candidate = run_dir / "parameters"
        return candidate if candidate.is_file() else None
    if solver == "GX":
        candidate = run_dir / "gx.in"
        if candidate.is_file():
            return candidate
        others = [p for p in sorted(run_dir.glob("*.in")) if is_safe_id(p.name) and p.is_file()]
        return others[0] if len(others) == 1 else None
    return None


def _geometry_sources(run_dir: Path) -> list[Path]:
    """Geometry/equilibrium files in `run_dir`, de-duplicated and sorted by name."""
    if not run_dir.is_dir():
        return []
    found: dict[str, Path] = {}
    for pattern in GEOMETRY_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            name = path.name
            if (
                name.startswith(".")
                or not name.isprintable()
                or name in _RESERVED_GEOMETRY_NAMES
                or not path.is_file()
            ):
                continue
            found.setdefault(name, path)
    return [found[name] for name in sorted(found)]


def _is_escalated(record: RunRecord, run_dir: Path) -> bool:
    if int(record.escalation_count or 0) > 0 or int(record.human_resolutions or 0) > 0:
        return True
    if any(
        isinstance(d, Mapping) and d.get("action") == "escalate_to_human"
        for d in record.directives_history
    ):
        return True
    return run_dir.is_dir() and any(p.is_file() for p in run_dir.glob("ISSUE*.md"))


def _normalize_patch(entry: Any) -> dict[str, Any]:
    """Add `changes: [{parameter, old, new}]` to a parameter_patches entry.

    Supports the watchdog format (`before`/`after` parameter snapshots) and the legacy
    single-parameter format (`parameter`/`old`/`new`).
    """
    if not isinstance(entry, Mapping):
        return {"value": entry, "changes": []}
    normalized = dict(entry)
    before, after = entry.get("before"), entry.get("after")
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes = [
            {"parameter": str(key), "old": before.get(key), "new": after.get(key)}
            for key in sorted({*before, *after}, key=str)
            if before.get(key) != after.get(key)
        ]
    elif "parameter" in entry:
        changes = [
            {"parameter": str(entry["parameter"]), "old": entry.get("old"), "new": entry.get("new")}
        ]
    else:
        changes = []
    normalized["changes"] = changes
    return normalized


def _job_ids(record: RunRecord) -> list[str]:
    """Scheduler job ids of every submission (restart_history[*].job_id), oldest first."""
    ids: list[str] = []
    for entry in record.restart_history:
        job_id = entry.get("job_id") if isinstance(entry, Mapping) else None
        if isinstance(job_id, (str, int)) and not isinstance(job_id, bool) and str(job_id):
            if str(job_id) not in ids:
                ids.append(str(job_id))
    if record.slurm_job_id and record.slurm_job_id not in ids:
        ids.append(record.slurm_job_id)
    return ids


# --------------------------------------------------------------------------------------------
# Equilibrium entry (shared by every run of the equilibrium; write-once)
# --------------------------------------------------------------------------------------------
def _publish_geometry(run_dir: Path, eq_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Copy geometry files that are not published yet; report files that conflict."""
    published: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for src in _geometry_sources(run_dir):
        dest = eq_dir / src.name
        if dest.exists() or dest.is_symlink():
            src_sha = sha256_file(src)
            try:
                same = dest.is_file() and not dest.is_symlink() and sha256_file(dest) == src_sha
            except OSError:
                same = False
            if not same:
                conflicts.append(src.name)
                continue
            published[src.name] = {"sha256": src_sha, "bytes": src.stat().st_size}
        else:
            sha, size = _atomic_copy(src, dest)
            published[src.name] = {"sha256": sha, "bytes": size}
    return published, conflicts


def _publish_equilibrium_meta(
    meta_path: Path,
    *,
    eq_id: str,
    origin: str | None,
    geometry: dict[str, dict[str, Any]],
    run_id: str,
) -> bool:
    """Write equilibrium_meta.json once. Return True if an existing one conflicts."""
    if meta_path.exists() or meta_path.is_symlink():
        try:
            existing = _read_json(meta_path)
        except (OSError, UnicodeDecodeError, ValueError):
            return True
        if not isinstance(existing, dict) or existing.get("equilibrium_id", eq_id) != eq_id:
            return True
        existing_origin = existing.get("origin")
        return bool(origin and existing_origin and existing_origin != origin)
    meta = {
        "schema_version": EQUILIBRIUM_META_SCHEMA_VERSION,
        "zenodo_reference": ZENODO_REFERENCE,
        "equilibrium_id": eq_id,
        "origin": origin,
        "geometry_files": sorted(geometry),
        "geometry_sha256": {name: entry["sha256"] for name, entry in geometry.items()},
        "created_by_run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_bytes(meta_path, _render_json(meta))
    return False


# --------------------------------------------------------------------------------------------
# Immutability (ML-08)
# --------------------------------------------------------------------------------------------
def _committed_entry(manifest_path: Path, record: RunRecord) -> Path | None:
    """Return the manifest path if this exact submission is already committed, else None.

    Raises DatasetEntryExistsError for a committed entry that belongs to another submission,
    is a legacy (v1) entry, or cannot be read: a committed entry is never overwritten.
    """
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    try:
        existing = _read_json(manifest_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DatasetEntryExistsError(
            f"{manifest_path} already exists but cannot be read ({type(exc).__name__}); "
            "a committed dataset entry is never overwritten"
        ) from exc
    provenance = existing.get("provenance") if isinstance(existing, dict) else None
    if (
        isinstance(provenance, dict)
        and record.slurm_job_id
        and provenance.get("final_job_id") == record.slurm_job_id
        and provenance.get("submit_nonce") == record.submit_nonce
    ):
        return manifest_path
    committed_job = provenance.get("final_job_id") if isinstance(provenance, dict) else None
    raise DatasetEntryExistsError(
        f"Dataset entry {manifest_path.parent} is already committed (final job "
        f"{committed_job!r}); refusing to overwrite it with run {record.run_id!r} "
        f"job {record.slurm_job_id!r}"
    )


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
def harvest_completed_run_to_dataset(
    record: RunRecord,
    run_dir: Path,
    dataset_root: Path,
    *,
    final_parameters: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    walltime_elapsed_s: float = 0.0,
    gcs_uri_prefix: str | None = None,
    converged: bool | None = None,
    convergence_method: str | None = None,
    simulated: bool | None = None,
    environment: str | None = None,
    flux_units: str | None = None,
    code_version: str | None = None,
) -> Path:
    """Publish a finished run as an immutable dataset entry; return its `run_manifest.json`.

    Every keyword added in schema 2.0 defaults to the fail-closed value: a caller that does
    not pass `converged=True` and `environment="prod"` gets an entry with
    `dataset_eligible=false`. `simulated=None` falls back to `final_metrics["simulated"]`, and
    an explicit `simulated=False` cannot override a raised `final_metrics["simulated"]`.

    Raises:
        ValueError: `record.campaign_id` / `record.run_id` is not a safe path component, or a
            dataset directory resolves outside `dataset_root`. Nothing is written.
        DatasetEntryExistsError: the entry is already committed for another submission.
    """
    campaign_id, run_id = record.campaign_id, record.run_id
    if not is_safe_id(campaign_id):
        raise ValueError(f"Unsafe campaign_id {campaign_id!r}: must match {SAFE_ID_RE.pattern}")
    if not is_safe_id(run_id):
        raise ValueError(f"Unsafe run_id {run_id!r}: must match {SAFE_ID_RE.pattern}")

    run_dir = Path(run_dir)
    metrics = dict(final_metrics)
    root = Path(dataset_root).resolve()
    _mkdirs_durable(root)
    run_out = _safe_makedirs(root, "campaigns", campaign_id, "runs", run_id)
    manifest_path = run_out / MANIFEST_NAME
    committed = _committed_entry(manifest_path, record)
    if committed is not None:
        return committed

    solver = str(record.solver).upper()
    eq_label = str(record.equilibrium_id or record.geometry_id or "")[:_MAX_LABEL_LEN]
    eq_id = eq_label if is_safe_id(eq_label) else None
    origin = _text_or_none(record.equilibrium_origin)
    version = _text_or_none(code_version) or _text_or_none(record.code_version)
    method = _text_or_none(convergence_method)
    env_text = _text_or_none(environment)
    env = env_text.lower() if env_text else None
    is_simulated = simulated is True or _flag_raised(metrics.get("simulated"))
    escalated = _is_escalated(record, run_dir)

    # 1. Shared equilibrium entry: written once, never overwritten.
    equilibrium: dict[str, Any] = {
        "equilibrium_id": eq_label or None,
        "origin": origin,
        "meta_path": None,
        "geometry_files": {},
        "geometry_conflicts": [],
    }
    meta_conflict = False
    if eq_id is not None:
        eq_dir = _safe_makedirs(root, "campaigns", campaign_id, "equilibria", eq_id)
        geometry, conflicts = _publish_geometry(run_dir, eq_dir)
        meta_conflict = _publish_equilibrium_meta(
            eq_dir / EQUILIBRIUM_META_NAME,
            eq_id=eq_id,
            origin=origin,
            geometry=geometry,
            run_id=run_id,
        )
        equilibrium.update(
            meta_path=f"campaigns/{campaign_id}/equilibria/{eq_id}/{EQUILIBRIUM_META_NAME}",
            geometry_files=geometry,
            geometry_conflicts=conflicts,
        )

    # 2. Run payload: uncommitted until run_manifest.json exists.
    files: dict[str, dict[str, Any]] = {}
    input_src = _select_input_file(solver, run_dir)
    input_sha: str | None = None
    if input_src is not None:
        input_sha, input_size = _atomic_copy(input_src, run_out / input_src.name)
        files[input_src.name] = {"sha256": input_sha, "bytes": input_size}
    flux_normalization = FLUX_NORMALIZATION_BY_SOLVER.get(solver)
    fsa_grad_x = _extract_fsa_grad_x(metrics)
    fluxes_payload: dict[str, Any] = {
        "schema_version": FLUXES_SCHEMA_VERSION,
        "run_id": run_id,
        "equilibrium_id": eq_label,
        "solver": solver,
        "fidelity": record.fidelity,
        "flux_units": _text_or_none(flux_units) or DEFAULT_FLUX_UNITS,
        **({"flux_normalization": flux_normalization} if flux_normalization is not None else {}),
        **({"fsa_grad_x": fsa_grad_x} if fsa_grad_x is not None else {}),
        **{key: metrics.get(key) for key in FLUX_FIELDS},
        "metrics": metrics,
    }
    files[FLUXES_NAME] = _write_bytes(run_out / FLUXES_NAME, _render_json(fluxes_payload))
    checksum_lines = "".join(f"{files[name]['sha256']}  {name}\n" for name in sorted(files))
    files[CHECKSUMS_NAME] = _write_bytes(run_out / CHECKSUMS_NAME, checksum_lines.encode("utf-8"))

    # 3. Eligibility verdict (fail closed: unverifiable means ineligible).
    submitted_sha = _text_or_none(record.input_sha256_submitted)
    failed = {
        "status_not_completed": record.status != "COMPLETED",
        "convergence_unknown": not isinstance(converged, bool),
        "not_converged": converged is False,
        "convergence_unverified": converged is True and method in WEAK_CONVERGENCE_METHODS,
        "simulated": is_simulated,
        "non_prod_environment": env != ELIGIBLE_ENVIRONMENT,
        "escalated": escalated,
        "missing_equilibrium_id": eq_id is None,
        "invalid_heat_flux": not _is_finite_real(metrics.get(PRIMARY_FLUX_FIELD)),
        "invalid_flux_value": any(
            metrics.get(key) is not None and not _is_finite_real(metrics.get(key))
            for key in FLUX_FIELDS
            if key != PRIMARY_FLUX_FIELD
        ),
        "missing_input_file": input_src is None,
        "input_modified_after_submission": input_src is not None
        and submitted_sha is not None
        and input_sha != submitted_sha.lower(),
        "gcp_provenance_unverified": record.backend in GCP_BACKENDS
        and not _flag_confirmed(metrics.get("provenance_ok")),
        "equilibrium_geometry_conflict": bool(equilibrium["geometry_conflicts"]),
        "equilibrium_metadata_conflict": meta_conflict,
    }
    reasons = [reason for reason in ELIGIBILITY_REASONS if failed[reason]]
    warnings = [
        warning
        for warning, raised in (
            ("equilibrium_origin_unknown", origin is None),
            ("code_version_unknown", version is None),
        )
        if raised
    ]

    rel_run_path = f"campaigns/{campaign_id}/runs/{run_id}"
    manifest: dict[str, Any] = {
        # --- schema 1.0 keys (unchanged meaning) ---
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "zenodo_layout_reference": ZENODO_REFERENCE,
        "workstream": record.workstream,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "equilibrium_id": eq_label,
        "geometry_id": record.geometry_id,
        "solver": solver,
        "fidelity": record.fidelity,
        "backend": record.backend,
        "gpu_tier": record.gpu_tier,
        "hands_free": record.status == "COMPLETED" and not escalated,
        "retry_count": record.retry_count,
        "walltime_resubmits": record.walltime_resubmits,
        "walltime_elapsed_s": _rounded_or_none(walltime_elapsed_s),
        "input_file": input_src.name if input_src is not None else None,
        "final_numerical_parameters": dict(final_parameters),
        "parameter_patches": list(record.parameter_patches),
        "gcs_uri": f"{gcs_uri_prefix.rstrip('/')}/{rel_run_path}" if gcs_uri_prefix else None,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        # --- schema 2.0 ---
        "converged": converged if isinstance(converged, bool) else None,
        "convergence_method": method,
        "escalated": escalated,
        "simulated": is_simulated,
        "environment": env,
        "patched": bool(record.parameter_patches),
        "dataset_eligible": not reasons,
        "eligibility_reasons": reasons,
        "eligibility_warnings": warnings,
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        "files": files,
        "code": {"name": solver, "version": version},
        "equilibrium": equilibrium,
        "provenance": {
            "submit_nonce": record.submit_nonce,
            "final_job_id": record.slurm_job_id,
            "job_ids": _job_ids(record),
            "restart_history": list(record.restart_history),
            "parameter_patches": [_normalize_patch(p) for p in record.parameter_patches],
            "retry_count": record.retry_count,
            "walltime_resubmits": record.walltime_resubmits,
            "escalation_count": record.escalation_count,
            "human_resolutions": record.human_resolutions,
            "harvester": {"name": HARVESTER_NAME, "version": _harvester_version()},
        },
    }
    # 4. Commit: the manifest is the last write.
    atomic_write_bytes(manifest_path, _render_json(manifest))
    return manifest_path
