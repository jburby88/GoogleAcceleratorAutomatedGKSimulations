"""Fail-closed KR1 loader for the KR2 campaign dataset, with leak-free `equilibrium_id` splits.

Layout consumed (written by KR2's ``platform/hpc-core/hpc_core/manifest.py``, which owns the
field contract)::

    <dataset_root>/campaigns/<campaign_id>/
        equilibria/<equilibrium_id>/equilibrium_meta.json
        runs/<run_id>/{run_manifest.json, fluxes.json, <solver input file>}

This expanded file-per-run campaign layout extends the provenance of the Zenodo record
14867777 GX stellarator dataset (https://zenodo.org/records/14867777, distributed there as an
HDF5 archive); it is not a byte-for-byte copy of that record's format.

Admission is fail-closed. A run becomes a :class:`DatasetSample` only if every check passes:
readable strict JSON (no NaN/Infinity tokens), a supported schema version, consistent
identifiers, ``converged is True``, ``escalated is False``, ``simulated is False``,
``environment == "prod"``, ``dataset_eligible is True`` with no ``eligibility_reasons`` (the
writer's verdict), recorded parameter patches, matching ``files`` sha256/size entries, a
recognised flux-units label and finite numeric fluxes. A missing field excludes the run; it is
never defaulted. Every exclusion is reported with a stable reason code
(:attr:`CampaignLoadReport.exclusion_counts`); writer-side blockers surface as
``marked_ineligible:<first eligibility reason>``.

Patched runs are admitted by default: KR2 only patches physics-preserving numerical parameters
and records every patch (docs/review/DOMAIN.md, ML-03). ``unpatched_only`` / ``hands_free_only``
restrict the sample set for ablations.

Fluxes are passed through unmodified: GX and GENE already write dimensionless gyroBohm-normalized
fluxes (Q / Q_gB), so applying another normalization factor would corrupt them
(docs/review/DOMAIN.md, ML-07). The units label is recorded on every sample and checked
against :data:`SUPPORTED_FLUX_UNITS`.

GX additionally divides its volume-integrated fluxes by the flux-surface average
``<|grad x|>`` (GX ``src/geometry.cu`` ``fluxDenom``; docs/review/GXSRC.md GXSRC-08). A raw GX
``heat_flux_es`` therefore corresponds to ``Q_avgs_divided_by_FSA_grad_x`` in Landreman et al.
(2025), Zenodo record 14867777, whose primary ``Q_avgs`` is that value multiplied by
``FSA_grad_x`` (ZEN-12). When the writer records them, ``flux_normalization`` (one of
:data:`SUPPORTED_FLUX_NORMALIZATIONS`) and ``fsa_grad_x`` are exposed on every sample so a
consumer can convert explicitly; the loader itself never multiplies by ``fsa_grad_x``.

Converged runs may have ``Q <= 0`` (damped / sub-critical cases; Zenodo 14867777 reports 1,773
of them). They are admitted by default because they carry stability information. A surrogate
that regresses ``log(Q)`` must drop them: pass ``min_primary_flux`` (ZEN-13), which excludes
runs whose primary flux is ``<= min_primary_flux`` with reason ``below_min_primary_flux``. Combining GX and GENE fluxes in one model additionally
requires a common thermal-velocity / reference-length convention, which is an open question
for the physics team; this loader never converts between conventions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REASON_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")

# --- KR2 dataset contract v2 (owner: platform/hpc-core/hpc_core/manifest.py, schema "2.0",
# posted by fix-hpc). Update together with the writer. ---
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({"2.0"})
SUPPORTED_FLUXES_SCHEMA_VERSIONS = frozenset({"2.0"})
SUPPORTED_FLUX_UNITS = frozenset({"gyroBohm"})
# Optional fluxes.json label written by hpc_core.manifest (GXSRC-08 / ZEN-12).
SUPPORTED_FLUX_NORMALIZATIONS = frozenset({"gyroBohm_divided_by_fsa_grad_x", "gyroBohm_gene"})
ELIGIBLE_ENVIRONMENT = "prod"
VALID_SOLVERS = frozenset({"GX", "GENE"})
VALID_FIDELITIES = frozenset({"low", "high"})
VALID_BACKENDS = frozenset({"tacc_slurm", "gcp_batch", "gcp_k8s"})
MANIFEST_NAME = "run_manifest.json"
FLUXES_NAME = "fluxes.json"
FLUX_FIELDS = (
    "heat_flux_es",
    "heat_flux_em",
    "particle_flux",
    "maxwell_reynolds_stress",
    "momentum_flux",
    "growth_rate_gamma",
    "omega_freq",
)
PRIMARY_FLUX_FIELD = "heat_flux_es"


@dataclass(frozen=True)
class DatasetSample:
    """One admitted gyrokinetic run, linked to its upstream `equilibrium_id`."""

    campaign_id: str
    run_id: str
    equilibrium_id: str
    solver: str
    fidelity: str
    backend: str
    numerical_parameters: dict[str, int | float]
    fluxes: dict[str, Any]
    equilibrium_meta: dict[str, Any]
    manifest_path: str
    flux_units: str = ""
    parameter_patches: tuple[dict[str, Any], ...] = ()
    flux_normalization: str = ""
    fsa_grad_x: float | None = None


@dataclass(frozen=True)
class ExcludedRun:
    """A run that was not admitted, with a stable machine-readable reason code."""

    run_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class CampaignLoadReport:
    """Admitted samples plus every exclusion, both in deterministic `run_id` order."""

    campaign_id: str
    samples: tuple[DatasetSample, ...]
    excluded: tuple[ExcludedRun, ...]

    @property
    def exclusion_counts(self) -> dict[str, int]:
        """Number of excluded runs per reason code (sorted by reason)."""
        return dict(sorted(Counter(item.reason for item in self.excluded).items()))


class _Excluded(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON token {token!r}")


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    """Parse strict JSON (no NaN/Infinity) into a dict; return it with the sha256/size of the bytes read."""
    if not path.is_file():
        raise _Excluded(f"missing_{label}", path.name)
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _Excluded(f"unreadable_{label}", str(exc)[:200]) from None
    if not isinstance(data, dict):
        raise _Excluded(f"unreadable_{label}", f"top-level JSON is {type(data).__name__}")
    return data, hashlib.sha256(raw).hexdigest(), len(raw)


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise _Excluded(f"missing_field:{key}")
    return data[key]


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = _require(data, key)
    if not isinstance(value, bool):
        raise _Excluded(f"invalid_field:{key}", repr(value)[:80])
    return value


def _require_choice(data: Mapping[str, Any], key: str, allowed: frozenset[str]) -> str:
    value = _require(data, key)
    if value not in allowed:
        raise _Excluded(f"invalid_field:{key}", repr(value)[:80])
    return value


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_manifest(
    manifest: dict[str, Any],
    *,
    run_id: str,
    campaign_id: str,
    hands_free_only: bool,
    unpatched_only: bool,
) -> tuple[str, list[dict[str, Any]]]:
    version = _require(manifest, "schema_version")
    if version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise _Excluded("unsupported_manifest_schema", repr(version)[:40])
    if _require(manifest, "run_id") != run_id:
        raise _Excluded("identity_mismatch", "run_manifest.run_id != run folder name")
    if _require(manifest, "campaign_id") != campaign_id:
        raise _Excluded("identity_mismatch", "run_manifest.campaign_id != campaign")
    eq_id = _require(manifest, "equilibrium_id")
    if not isinstance(eq_id, str) or not SAFE_ID_RE.match(eq_id):
        raise _Excluded("invalid_field:equilibrium_id", repr(eq_id)[:80])

    converged = _require(manifest, "converged")
    if converged is None:  # v2 writes null when the caller could not establish convergence
        raise _Excluded("convergence_unknown")
    if not isinstance(converged, bool):
        raise _Excluded("invalid_field:converged", repr(converged)[:80])
    if not converged:
        raise _Excluded("not_converged")
    if _require_bool(manifest, "escalated"):
        raise _Excluded("escalated")
    if _require_bool(manifest, "simulated"):
        raise _Excluded("simulated")
    environment = _require(manifest, "environment")
    if environment != ELIGIBLE_ENVIRONMENT:
        raise _Excluded("non_prod_environment", repr(environment)[:40])

    patches = _require(manifest, "parameter_patches")
    if not isinstance(patches, list) or not all(isinstance(p, dict) for p in patches):
        raise _Excluded("invalid_field:parameter_patches")
    retry_count = _require(manifest, "retry_count")
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise _Excluded("invalid_field:retry_count", repr(retry_count)[:40])
    if len(patches) < retry_count:
        # Every steered resubmission must leave a patch record (even an empty one).
        raise _Excluded("unrecorded_patches", f"retry_count={retry_count}, patches={len(patches)}")
    patched = _require_bool(manifest, "patched")
    if patched != bool(patches):
        raise _Excluded("invalid_field:patched", "patched flag disagrees with parameter_patches")

    eligible = _require_bool(manifest, "dataset_eligible")
    reasons = _require(manifest, "eligibility_reasons")
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        raise _Excluded("invalid_field:eligibility_reasons")
    if not eligible:
        first = reasons[0] if reasons and REASON_CODE_RE.match(reasons[0]) else ""
        raise _Excluded(
            f"marked_ineligible:{first}" if first else "marked_ineligible",
            ",".join(reasons)[:200],
        )
    if reasons:
        raise _Excluded("invalid_field:eligibility_reasons", "eligible entry lists blockers")
    if hands_free_only and not _require_bool(manifest, "hands_free"):
        raise _Excluded("not_hands_free")
    if unpatched_only and patched:
        raise _Excluded("patched")
    return eq_id, patches


def _check_files(
    manifest: Mapping[str, Any], run_folder: Path, fluxes_sha256: str, fluxes_size: int
) -> None:
    """Verify `files: {name: {sha256, bytes}}` for every file the writer published."""
    files = _require(manifest, "files")
    if not isinstance(files, dict) or FLUXES_NAME not in files:
        raise _Excluded("invalid_field:files", f"{FLUXES_NAME} entry required")
    input_file = manifest.get("input_file")
    if isinstance(input_file, str) and input_file not in files:
        raise _Excluded("invalid_field:files", f"input file {input_file!r} not covered")
    for name, entry in sorted(files.items()):
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name.startswith(".")
            or name == MANIFEST_NAME
        ):
            raise _Excluded("invalid_field:files", repr(name)[:80])
        expected_sha = entry.get("sha256") if isinstance(entry, dict) else None
        expected_size = entry.get("bytes") if isinstance(entry, dict) else None
        if (
            not isinstance(expected_sha, str)
            or not SHA256_RE.match(expected_sha)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise _Excluded("invalid_field:files", f"bad sha256/bytes for {name}")
        if name == FLUXES_NAME:
            actual_sha, actual_size = fluxes_sha256, fluxes_size  # the exact bytes parsed
        else:
            path = run_folder / name
            if path.is_symlink() or not path.is_file():
                raise _Excluded("checksum_mismatch", f"{name} missing")
            actual_sha, actual_size = _sha256(path), path.stat().st_size
        if actual_size != expected_size or actual_sha != expected_sha:
            raise _Excluded("checksum_mismatch", name)


def _check_fluxes(
    fluxes: dict[str, Any],
    manifest: Mapping[str, Any],
    *,
    require_valid_fluxes: bool,
    min_primary_flux: float | None,
) -> tuple[str, str, float | None]:
    version = _require(fluxes, "schema_version")
    if version not in SUPPORTED_FLUXES_SCHEMA_VERSIONS:
        raise _Excluded("unsupported_fluxes_schema", repr(version)[:40])
    for key in ("run_id", "equilibrium_id", "solver", "fidelity"):
        if _require(fluxes, key) != manifest.get(key):
            raise _Excluded("identity_mismatch", f"fluxes.{key} != run_manifest.{key}")
    units = _require(fluxes, "flux_units")
    if units not in SUPPORTED_FLUX_UNITS:
        raise _Excluded("unsupported_flux_units", repr(units)[:80])
    for key in FLUX_FIELDS:
        value = _require(fluxes, key)
        if value is not None and not _is_finite_number(value):
            raise _Excluded("invalid_flux_value", f"{key}={value!r}"[:120])
    primary = fluxes[PRIMARY_FLUX_FIELD]
    if (require_valid_fluxes or min_primary_flux is not None) and primary is None:
        raise _Excluded("missing_primary_flux", PRIMARY_FLUX_FIELD)
    if not isinstance(_require(fluxes, "metrics"), dict):
        raise _Excluded("invalid_field:metrics")
    # Optional fields (absent in older writers): validated fail-closed when present.
    normalization = fluxes.get("flux_normalization")
    if normalization is None:
        normalization = ""
    elif normalization not in SUPPORTED_FLUX_NORMALIZATIONS:
        raise _Excluded("unsupported_flux_normalization", repr(normalization)[:80])
    fsa_grad_x = fluxes.get("fsa_grad_x")
    if fsa_grad_x is not None:
        if not _is_finite_number(fsa_grad_x) or fsa_grad_x <= 0:
            raise _Excluded("invalid_field:fsa_grad_x", repr(fsa_grad_x)[:80])
        fsa_grad_x = float(fsa_grad_x)
    if min_primary_flux is not None and primary <= min_primary_flux:
        raise _Excluded("below_min_primary_flux", f"{PRIMARY_FLUX_FIELD}={primary!r}"[:120])
    return units, normalization, fsa_grad_x


def _check_numerical_parameters(manifest: Mapping[str, Any]) -> dict[str, int | float]:
    params = _require(manifest, "final_numerical_parameters")
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and _is_finite_number(v) for k, v in params.items()
    ):
        raise _Excluded("invalid_field:final_numerical_parameters")
    return dict(params)


def _load_equilibrium_meta(eq_root: Path, eq_id: str) -> dict[str, Any]:
    eq_dir = eq_root / eq_id
    if eq_dir.is_symlink() or not eq_dir.resolve().is_relative_to(eq_root.resolve()):
        raise _Excluded("unsafe_path", f"equilibria/{eq_id}")
    meta, _, _ = _read_json_object(eq_dir / "equilibrium_meta.json", "equilibrium_meta")
    if meta.get("equilibrium_id") != eq_id:
        raise _Excluded("identity_mismatch", "equilibrium_meta.equilibrium_id != equilibrium_id")
    return meta


def _admit_run(
    run_folder: Path,
    *,
    campaign_id: str,
    eq_root: Path,
    require_valid_fluxes: bool,
    hands_free_only: bool,
    unpatched_only: bool,
    min_primary_flux: float | None,
) -> DatasetSample:
    if run_folder.is_symlink():
        raise _Excluded("unsafe_path", run_folder.name)
    manifest_file = run_folder / MANIFEST_NAME
    # The manifest is the writer's commit marker: without it the run is partial/uncommitted.
    manifest, _, _ = _read_json_object(manifest_file, "run_manifest")
    eq_id, patches = _check_manifest(
        manifest,
        run_id=run_folder.name,
        campaign_id=campaign_id,
        hands_free_only=hands_free_only,
        unpatched_only=unpatched_only,
    )
    solver = _require_choice(manifest, "solver", VALID_SOLVERS)
    fidelity = _require_choice(manifest, "fidelity", VALID_FIDELITIES)
    backend = _require_choice(manifest, "backend", VALID_BACKENDS)
    numerical_parameters = _check_numerical_parameters(manifest)
    fluxes, fluxes_sha256, fluxes_size = _read_json_object(run_folder / FLUXES_NAME, "fluxes")
    _check_files(manifest, run_folder, fluxes_sha256, fluxes_size)
    units, normalization, fsa_grad_x = _check_fluxes(
        fluxes,
        manifest,
        require_valid_fluxes=require_valid_fluxes,
        min_primary_flux=min_primary_flux,
    )
    eq_meta = _load_equilibrium_meta(eq_root, eq_id)
    return DatasetSample(
        campaign_id=campaign_id,
        run_id=run_folder.name,
        equilibrium_id=eq_id,
        solver=solver,
        fidelity=fidelity,
        backend=backend,
        numerical_parameters=numerical_parameters,
        fluxes=fluxes,
        equilibrium_meta=eq_meta,
        manifest_path=str(manifest_file),
        flux_units=units,
        parameter_patches=tuple(patches),
        flux_normalization=normalization,
        fsa_grad_x=fsa_grad_x,
    )


def load_campaign_report(
    dataset_root: Path,
    campaign_id: str,
    *,
    require_valid_fluxes: bool = True,
    hands_free_only: bool = False,
    unpatched_only: bool = False,
    min_primary_flux: float | None = None,
) -> CampaignLoadReport:
    """Load ``campaigns/<campaign_id>/runs/*`` fail-closed; report every exclusion by reason.

    Entries of ``runs/`` that are not directories, or whose names start with ``.`` (staging
    areas), are not runs and are skipped silently. Every other entry is either admitted or
    listed in :attr:`CampaignLoadReport.excluded`.

    ``min_primary_flux`` (default ``None``: no threshold) excludes runs whose
    :data:`PRIMARY_FLUX_FIELD` is ``<= min_primary_flux`` (reason ``below_min_primary_flux``),
    e.g. ``0.0`` before fitting ``log(Q)`` (ZEN-13).
    """
    if not isinstance(campaign_id, str) or not SAFE_ID_RE.match(campaign_id):
        raise ValueError(f"Unsafe or invalid campaign_id: {campaign_id!r}")
    if min_primary_flux is not None:
        if not _is_finite_number(min_primary_flux):
            raise ValueError(f"min_primary_flux must be a finite number: {min_primary_flux!r}")
        min_primary_flux = float(min_primary_flux)
    campaign_dir = Path(dataset_root).resolve() / "campaigns" / campaign_id
    runs_dir = campaign_dir / "runs"
    eq_root = campaign_dir / "equilibria"
    if not runs_dir.is_dir():
        logger.warning("Campaign %s has no runs directory at %s", campaign_id, runs_dir)
        return CampaignLoadReport(campaign_id=campaign_id, samples=(), excluded=())

    samples: list[DatasetSample] = []
    excluded: list[ExcludedRun] = []
    for run_folder in sorted(runs_dir.iterdir()):
        if run_folder.name.startswith(".") or not run_folder.is_dir():
            continue
        try:
            samples.append(
                _admit_run(
                    run_folder,
                    campaign_id=campaign_id,
                    eq_root=eq_root,
                    require_valid_fluxes=require_valid_fluxes,
                    hands_free_only=hands_free_only,
                    unpatched_only=unpatched_only,
                    min_primary_flux=min_primary_flux,
                )
            )
        except _Excluded as exc:
            excluded.append(ExcludedRun(run_folder.name, exc.reason, exc.detail))
    return CampaignLoadReport(
        campaign_id=campaign_id, samples=tuple(samples), excluded=tuple(excluded)
    )


def load_campaign_samples(
    dataset_root: Path,
    campaign_id: str,
    *,
    require_valid_fluxes: bool = True,
    hands_free_only: bool = False,
    unpatched_only: bool = False,
    min_primary_flux: float | None = None,
) -> list[DatasetSample]:
    """Return the admitted samples of :func:`load_campaign_report`; log exclusion counts."""
    report = load_campaign_report(
        dataset_root,
        campaign_id,
        require_valid_fluxes=require_valid_fluxes,
        hands_free_only=hands_free_only,
        unpatched_only=unpatched_only,
        min_primary_flux=min_primary_flux,
    )
    if report.excluded:
        logger.warning(
            "Campaign %s: admitted %d run(s), excluded %d: %s",
            campaign_id,
            len(report.samples),
            len(report.excluded),
            report.exclusion_counts,
        )
    return list(report.samples)


def _hash_unit(seed: str, equilibrium_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{equilibrium_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _allocate_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    """Largest-remainder apportionment of ``n`` groups (ties: train, then val, then test)."""
    quotas = (n * train_frac, n * val_frac, n * (1.0 - train_frac - val_frac))
    counts = [math.floor(q) for q in quotas]
    by_remainder = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    return counts[0], counts[1], counts[2]


def split_by_equilibrium(
    samples: list[DatasetSample],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: str = "ut-austin-kr1-v1",
    stratify_by: Callable[[Sequence[DatasetSample]], str] | None = None,
) -> dict[str, list[DatasetSample]]:
    """Deterministically partition samples into `train`, `val`, `test` grouped strictly by `equilibrium_id`.

    Guarantees zero data leakage when both low-fidelity GX (Cloud Batch) and high-fidelity GENE
    (TACC) simulations exist for the same `ToBlick/mrx` equilibrium: all samples of an
    equilibrium land in exactly one split. Within each split, input order is preserved.

    Default (``stratify_by=None``): each equilibrium is placed by thresholding
    ``sha256(f"{seed}:{equilibrium_id}")``. Its split depends only on ``(seed, equilibrium_id)``,
    so it never changes when the dataset grows (no leakage across dataset versions), but split
    sizes are random and small campaigns can leave `val`/`test` empty or skewed.

    Stratified (``stratify_by`` given): ``stratify_by(group)`` maps the samples of one
    equilibrium (sorted by `run_id`) to a stratum label, e.g. ``"multi-fidelity"`` vs
    ``"gx-only"``. Within each stratum, equilibria are ordered by the same seeded hash and the
    first ``round(n * train_frac)`` / ``round(n * val_frac)`` (largest-remainder rounding) go to
    `train` / `val`, the rest to `test`. Proportions are exact per stratum, but adding equilibria
    can move existing ones across splits: freeze the dataset version before using this mode.
    """
    if (
        not (math.isfinite(train_frac) and math.isfinite(val_frac))
        or train_frac <= 0.0
        or val_frac < 0.0
        or (train_frac + val_frac) >= 1.0
    ):
        raise ValueError(
            "Require finite train_frac > 0, val_frac >= 0, and train_frac + val_frac < 1.0"
        )

    buckets: dict[str, list[DatasetSample]] = {"train": [], "val": [], "test": []}
    if stratify_by is None:
        for sample in samples:
            u = _hash_unit(seed, sample.equilibrium_id)
            if u < train_frac:
                buckets["train"].append(sample)
            elif u < (train_frac + val_frac):
                buckets["val"].append(sample)
            else:
                buckets["test"].append(sample)
        return buckets

    groups: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        groups.setdefault(sample.equilibrium_id, []).append(sample)
    strata: dict[str, list[str]] = {}
    for eq_id, members in groups.items():
        label = stratify_by(tuple(sorted(members, key=lambda s: s.run_id)))
        if not isinstance(label, str):
            raise ValueError(f"stratify_by must return a str label, got {label!r} for {eq_id!r}")
        strata.setdefault(label, []).append(eq_id)

    split_of: dict[str, str] = {}
    for label in sorted(strata):
        ordered = sorted(strata[label], key=lambda e: (_hash_unit(seed, e), e))
        n_train, n_val, _ = _allocate_counts(len(ordered), train_frac, val_frac)
        for idx, eq_id in enumerate(ordered):
            split_of[eq_id] = (
                "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            )
    for sample in samples:
        buckets[split_of[sample.equilibrium_id]].append(sample)
    return buckets
