"""Tests for the schema-2.0 dataset manifest writer (ML-05, ML-08, DOM-06): hpc_core/manifest.py."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hpc_core import manifest as manifest_mod
from hpc_core.manifest import (
    ELIGIBILITY_REASONS,
    DatasetEntryExistsError,
    harvest_completed_run_to_dataset,
)
from hpc_core.state import RunRecord

GX_INPUT = "[Dimensions]\nnx = 64\nny = 32\nntheta = 48\n\n[Time]\ndt = 0.01\ncfl = 0.8\n"
ORIGIN = "ToBlick/mrx@0123456789abcdef0123456789abcdef01234567"
CODE_VERSION = "gx@89abcdef0123456789abcdef0123456789abcdef"
V1_KEYS = (
    "schema_version",
    "zenodo_layout_reference",
    "workstream",
    "campaign_id",
    "run_id",
    "equilibrium_id",
    "geometry_id",
    "solver",
    "fidelity",
    "backend",
    "gpu_tier",
    "hands_free",
    "retry_count",
    "walltime_resubmits",
    "walltime_elapsed_s",
    "input_file",
    "final_numerical_parameters",
    "parameter_patches",
    "gcs_uri",
    "completed_at_utc",
)
_DROP = object()

Mutator = Callable[[Path, RunRecord, dict[str, Any]], None]


def _strict_load(path: Path) -> Any:
    def _reject(token: str) -> Any:
        raise AssertionError(f"non-finite JSON token {token!r} in {path.name}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run_dir(
    tmp_path: Path,
    run_id: str = "gx_run_001",
    *,
    inputs: dict[str, str] | None = None,
    geometry: dict[str, bytes] | None = None,
) -> Path:
    run_dir = tmp_path / "queue" / "completed" / run_id
    run_dir.mkdir(parents=True)
    for name, text in ({"gx.in": GX_INPUT} if inputs is None else inputs).items():
        (run_dir / name).write_text(text, encoding="utf-8")
    for name, data in (geometry or {}).items():
        (run_dir / name).write_bytes(data)
    return run_dir


def _record(run_dir: Path, **overrides: Any) -> RunRecord:
    """A RunRecord that is dataset-eligible when harvested with `_eligible_kwargs()`."""
    gx_in = run_dir / "gx.in"
    fields: dict[str, Any] = {
        "run_id": run_dir.name,
        "solver": "GX",
        "run_dir": str(run_dir),
        "status": "COMPLETED",
        "slurm_job_id": "4242",
        "submit_nonce": "a1b2c3d4",
        "retry_count": 1,
        "campaign_id": "pilot_50",
        "backend": "tacc_slurm",
        "equilibrium_id": "mrx_eq_0001",
        "geometry_id": "mrx_eq_0001",
        "equilibrium_origin": ORIGIN,
        "code_version": CODE_VERSION,
        "input_sha256_submitted": _sha256(gx_in) if gx_in.is_file() else None,
        "restart_history": [
            {"job_id": "4241", "reason": "initial"},
            {"job_id": "4242", "reason": "warm_restart"},
        ],
        "parameter_patches": [
            {
                "strike": 1,
                "action": "warm_restart",
                "before": {"dt_max": 0.02, "nx": 64},
                "after": {"dt_max": 0.01, "nx": 64},
                "restart_from_checkpoint": True,
            }
        ],
    }
    fields.update(overrides)
    return RunRecord(**fields)


def _eligible_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "final_parameters": {"nx": 64, "dt_max": 0.01},
        "final_metrics": {"heat_flux_es": 3.72, "heat_flux_em": None, "particle_flux": 0.18},
        "walltime_elapsed_s": 1450.0,
        "converged": True,
        "convergence_method": "time_average_stationarity",
        "simulated": False,
        "environment": "prod",
    }
    kwargs.update(overrides)
    return kwargs


def _harvest(record: RunRecord, run_dir: Path, root: Path, **kwargs: Any) -> dict[str, Any]:
    return _strict_load(harvest_completed_run_to_dataset(record, run_dir, root, **kwargs))


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def test_eligible_run_writes_v2_manifest_checksums_and_provenance(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, geometry={"mrx_eq_0001.eik.nc": b"eik-geometry"})
    root = tmp_path / "dataset"
    manifest_path = harvest_completed_run_to_dataset(
        _record(run_dir), run_dir, root, **_eligible_kwargs(gcs_uri_prefix="gs://bucket/")
    )
    manifest = _strict_load(manifest_path)
    run_out = manifest_path.parent

    assert manifest_path == root.resolve() / "campaigns/pilot_50/runs/gx_run_001/run_manifest.json"
    assert all(key in manifest for key in V1_KEYS)
    assert manifest["schema_version"] == "2.0"
    assert manifest["dataset_eligible"] is True
    assert manifest["eligibility_reasons"] == []
    assert manifest["eligibility_warnings"] == []
    assert manifest["eligibility_policy_version"] == "1"
    assert manifest["converged"] is True
    assert manifest["convergence_method"] == "time_average_stationarity"
    assert (manifest["escalated"], manifest["simulated"], manifest["patched"]) == (
        False,
        False,
        True,
    )
    assert manifest["environment"] == "prod"
    assert manifest["hands_free"] is True
    assert manifest["input_file"] == "gx.in"
    assert manifest["gcs_uri"] == "gs://bucket/campaigns/pilot_50/runs/gx_run_001"
    assert manifest["code"] == {"name": "GX", "version": CODE_VERSION}

    # files: every payload file with its digest and size; checksums.sha256 verifies.
    assert set(manifest["files"]) == {"gx.in", "fluxes.json", "checksums.sha256"}
    for name, entry in manifest["files"].items():
        data = (run_out / name).read_bytes()
        assert entry == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert (run_out / "gx.in").read_text(encoding="utf-8") == GX_INPUT
    expected_lines = [
        f"{manifest['files'][name]['sha256']}  {name}" for name in ("fluxes.json", "gx.in")
    ]
    assert (run_out / "checksums.sha256").read_text(encoding="utf-8").splitlines() == expected_lines
    if shutil.which("sha256sum"):
        subprocess.run(
            ["sha256sum", "--check", "--strict", "checksums.sha256"],
            cwd=run_out,
            check=True,
            capture_output=True,
        )

    fluxes = _strict_load(run_out / "fluxes.json")
    assert fluxes["schema_version"] == "2.0"
    assert fluxes["flux_units"] == "gyroBohm"
    assert fluxes["flux_normalization"] == "gyroBohm_divided_by_fsa_grad_x"
    assert "fsa_grad_x" not in fluxes
    assert fluxes["heat_flux_es"] == pytest.approx(3.72)
    assert fluxes["heat_flux_em"] is None
    for key in ("run_id", "equilibrium_id", "solver", "fidelity"):
        assert fluxes[key] == manifest[key]

    provenance = manifest["provenance"]
    assert provenance["final_job_id"] == "4242"
    assert provenance["submit_nonce"] == "a1b2c3d4"
    assert provenance["job_ids"] == ["4241", "4242"]
    assert provenance["restart_history"][0] == {"job_id": "4241", "reason": "initial"}
    assert provenance["parameter_patches"][0]["changes"] == [
        {"parameter": "dt_max", "old": 0.02, "new": 0.01}
    ]
    assert (provenance["retry_count"], provenance["walltime_resubmits"]) == (1, 0)
    assert (provenance["escalation_count"], provenance["human_resolutions"]) == (0, 0)
    assert provenance["harvester"]["name"] == "ut-hpc-core"
    assert provenance["harvester"]["version"]

    equilibrium = manifest["equilibrium"]
    assert equilibrium["equilibrium_id"] == "mrx_eq_0001"
    assert equilibrium["origin"] == ORIGIN
    assert equilibrium["meta_path"] == (
        "campaigns/pilot_50/equilibria/mrx_eq_0001/equilibrium_meta.json"
    )
    geometry_sha = hashlib.sha256(b"eik-geometry").hexdigest()
    assert equilibrium["geometry_files"] == {
        "mrx_eq_0001.eik.nc": {"sha256": geometry_sha, "bytes": len(b"eik-geometry")}
    }
    assert equilibrium["geometry_conflicts"] == []
    meta = _strict_load(root / equilibrium["meta_path"])
    assert meta["equilibrium_id"] == "mrx_eq_0001"
    assert meta["origin"] == ORIGIN
    assert meta["geometry_files"] == ["mrx_eq_0001.eik.nc"]
    assert meta["geometry_sha256"] == {"mrx_eq_0001.eik.nc": geometry_sha}


def _set(**fields: Any) -> Mutator:
    def mutate(run_dir: Path, record: RunRecord, kwargs: dict[str, Any]) -> None:
        for name, value in fields.items():
            setattr(record, name, value)

    return mutate


def _kw(**updates: Any) -> Mutator:
    def mutate(run_dir: Path, record: RunRecord, kwargs: dict[str, Any]) -> None:
        for name, value in updates.items():
            if value is _DROP:
                kwargs.pop(name, None)
            else:
                kwargs[name] = value

    return mutate


def _metric(**updates: Any) -> Mutator:
    def mutate(run_dir: Path, record: RunRecord, kwargs: dict[str, Any]) -> None:
        metrics = dict(kwargs["final_metrics"])
        for name, value in updates.items():
            if value is _DROP:
                metrics.pop(name, None)
            else:
                metrics[name] = value
        kwargs["final_metrics"] = metrics

    return mutate


def _touch(name: str) -> Mutator:
    def mutate(run_dir: Path, record: RunRecord, kwargs: dict[str, Any]) -> None:
        (run_dir / name).write_text("# escalation note\n", encoding="utf-8")

    return mutate


def _unlink(name: str) -> Mutator:
    def mutate(run_dir: Path, record: RunRecord, kwargs: dict[str, Any]) -> None:
        (run_dir / name).unlink()

    return mutate


def _combo(*mutators: Mutator) -> Mutator:
    def mutate(run_dir: Path, record: RunRecord, kwargs: dict[str, Any]) -> None:
        for mutator in mutators:
            mutator(run_dir, record, kwargs)

    return mutate


_BLOCKER_CASES: list[tuple[str, str, Mutator]] = [
    ("status_not_completed", "running", _set(status="RUNNING")),
    ("convergence_unknown", "not-passed", _kw(converged=_DROP)),
    ("not_converged", "false", _kw(converged=False)),
    ("convergence_unverified", "short-series", _kw(convergence_method="short_series_last_sample")),
    ("convergence_unverified", "unverified", _kw(convergence_method="unverified")),
    ("simulated", "kwarg", _kw(simulated=True)),
    ("simulated", "metric-only", _combo(_kw(simulated=_DROP), _metric(simulated=1.0))),
    ("simulated", "metric-beats-explicit-false", _metric(simulated=1.0)),
    ("non_prod_environment", "sim", _kw(environment="sim")),
    ("non_prod_environment", "not-passed", _kw(environment=_DROP)),
    ("escalated", "escalation-count", _set(escalation_count=1)),
    ("escalated", "human-resolution", _set(human_resolutions=1)),
    (
        "escalated",
        "directive",
        _set(directives_history=[{"action": "escalate_to_human", "reason": "review"}]),
    ),
    ("escalated", "issue-md", _touch("ISSUE.md")),
    ("escalated", "issue-resolved-md", _touch("ISSUE.resolved.md")),
    ("missing_equilibrium_id", "empty", _set(equilibrium_id="", geometry_id="")),
    ("missing_equilibrium_id", "traversal", _set(equilibrium_id="../escape", geometry_id="")),
    ("invalid_heat_flux", "missing", _metric(heat_flux_es=_DROP)),
    ("invalid_heat_flux", "nan", _metric(heat_flux_es=float("nan"))),
    ("invalid_heat_flux", "inf", _metric(heat_flux_es=float("inf"))),
    ("invalid_heat_flux", "bool", _metric(heat_flux_es=True)),
    ("invalid_heat_flux", "string", _metric(heat_flux_es="3.72")),
    ("invalid_flux_value", "nan-particle-flux", _metric(particle_flux=float("nan"))),
    ("invalid_flux_value", "list-growth-rate", _metric(growth_rate_gamma=[0.1, 0.2])),
    ("missing_input_file", "no-gx-in", _unlink("gx.in")),
    ("input_modified_after_submission", "sha-mismatch", _set(input_sha256_submitted="0" * 64)),
    ("gcp_provenance_unverified", "batch-no-flag", _set(backend="gcp_batch")),
    (
        "gcp_provenance_unverified",
        "k8s-flag-zero",
        _combo(_set(backend="gcp_k8s"), _metric(provenance_ok=0.0)),
    ),
]


@pytest.mark.parametrize(
    ("reason", "mutate"),
    [
        pytest.param(reason, mutate, id=f"{reason}-{case}")
        for reason, case, mutate in _BLOCKER_CASES
    ],
)
def test_each_blocker_alone_makes_the_run_ineligible(
    tmp_path: Path, reason: str, mutate: Mutator
) -> None:
    run_dir = _make_run_dir(tmp_path)
    record = _record(run_dir)
    kwargs = _eligible_kwargs()
    mutate(run_dir, record, kwargs)

    manifest = _harvest(record, run_dir, tmp_path / "dataset", **kwargs)

    assert manifest["eligibility_reasons"] == [reason]
    assert manifest["dataset_eligible"] is False


def test_every_eligibility_reason_has_a_test() -> None:
    covered = {reason for reason, _case, _mutate in _BLOCKER_CASES}
    covered |= {"equilibrium_geometry_conflict", "equilibrium_metadata_conflict"}  # tests below
    assert covered == set(ELIGIBILITY_REASONS)


def test_gcp_run_with_verified_provenance_is_eligible(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    metrics = {"heat_flux_es": 3.72, "provenance_ok": 1.0, "simulated": 0.0}
    manifest = _harvest(
        _record(run_dir, backend="gcp_batch"),
        run_dir,
        tmp_path / "dataset",
        **_eligible_kwargs(final_metrics=metrics, simulated=None),
    )
    assert manifest["dataset_eligible"] is True
    assert manifest["simulated"] is False


def test_unsafe_equilibrium_id_writes_no_equilibria_entry(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, geometry={"wout_mrx.nc": b"vmec"})
    root = tmp_path / "dataset"
    manifest = _harvest(
        _record(run_dir, equilibrium_id="../../escape", geometry_id=""),
        run_dir,
        root,
        **_eligible_kwargs(),
    )
    assert manifest["eligibility_reasons"] == ["missing_equilibrium_id"]
    assert manifest["equilibrium"]["meta_path"] is None
    assert manifest["equilibrium"]["geometry_files"] == {}
    assert not (root / "campaigns" / "pilot_50" / "equilibria").exists()
    assert not (tmp_path / "escape").exists()


def test_geometry_conflict_never_overwrites_the_published_copy(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    first_dir = _make_run_dir(tmp_path, "gx_run_001", geometry={"wout_mrx.nc": b"version-A"})
    assert _harvest(_record(first_dir), first_dir, root, **_eligible_kwargs())["dataset_eligible"]

    second_dir = _make_run_dir(tmp_path, "gx_run_002", geometry={"wout_mrx.nc": b"version-B"})
    second = _harvest(_record(second_dir), second_dir, root, **_eligible_kwargs())
    assert second["eligibility_reasons"] == ["equilibrium_geometry_conflict"]
    assert second["equilibrium"]["geometry_conflicts"] == ["wout_mrx.nc"]
    published = root / "campaigns" / "pilot_50" / "equilibria" / "mrx_eq_0001" / "wout_mrx.nc"
    assert published.read_bytes() == b"version-A"

    third_dir = _make_run_dir(tmp_path, "gx_run_003", geometry={"wout_mrx.nc": b"version-A"})
    third = _harvest(_record(third_dir), third_dir, root, **_eligible_kwargs())
    assert third["dataset_eligible"] is True
    assert third["equilibrium"]["geometry_files"]["wout_mrx.nc"]["sha256"] == (
        hashlib.sha256(b"version-A").hexdigest()
    )


def test_equilibrium_origin_conflict_blocks_the_later_run(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    first_dir = _make_run_dir(tmp_path, "gx_run_001")
    assert _harvest(_record(first_dir), first_dir, root, **_eligible_kwargs())["dataset_eligible"]
    meta_path = (
        root / "campaigns" / "pilot_50" / "equilibria" / "mrx_eq_0001" / "equilibrium_meta.json"
    )
    meta_before = meta_path.read_bytes()

    other_dir = _make_run_dir(tmp_path, "gx_run_002")
    other = _harvest(
        _record(other_dir, equilibrium_origin="ToBlick/mrx@" + "f" * 40),
        other_dir,
        root,
        **_eligible_kwargs(),
    )
    assert other["eligibility_reasons"] == ["equilibrium_metadata_conflict"]
    assert meta_path.read_bytes() == meta_before

    meta_path.write_text("{corrupt", encoding="utf-8")
    third_dir = _make_run_dir(tmp_path, "gx_run_003")
    third = _harvest(_record(third_dir), third_dir, root, **_eligible_kwargs())
    assert third["eligibility_reasons"] == ["equilibrium_metadata_conflict"]
    assert meta_path.read_text(encoding="utf-8") == "{corrupt"


def test_unknown_origin_and_code_version_are_warnings_only(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    run_dir = _make_run_dir(tmp_path, "gx_run_001")
    manifest = _harvest(
        _record(run_dir, equilibrium_origin=None, code_version=None),
        run_dir,
        root,
        **_eligible_kwargs(),
    )
    assert manifest["dataset_eligible"] is True
    assert manifest["eligibility_warnings"] == [
        "equilibrium_origin_unknown",
        "code_version_unknown",
    ]
    assert manifest["code"]["version"] is None

    kw_dir = _make_run_dir(tmp_path, "gx_run_002")
    by_kwarg = _harvest(
        _record(kw_dir, code_version=None),
        kw_dir,
        root,
        **_eligible_kwargs(code_version="gx@kwarg", flux_units="gyroBohm_GX"),
    )
    assert by_kwarg["code"]["version"] == "gx@kwarg"
    assert by_kwarg["eligibility_warnings"] == []
    fluxes = _strict_load(root / "campaigns" / "pilot_50" / "runs" / "gx_run_002" / "fluxes.json")
    assert fluxes["flux_units"] == "gyroBohm_GX"


def test_reharvest_of_the_same_submission_is_idempotent(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    root = tmp_path / "dataset"
    first_path = harvest_completed_run_to_dataset(
        _record(run_dir), run_dir, root, **_eligible_kwargs()
    )
    before = _snapshot(first_path.parent)

    # Crash recovery: the watchdog harvests the same submission again (even with other metrics).
    second_path = harvest_completed_run_to_dataset(
        _record(run_dir),
        run_dir,
        root,
        **_eligible_kwargs(final_metrics={"heat_flux_es": 99.0}),
    )
    assert second_path == first_path
    assert _snapshot(first_path.parent) == before


@pytest.mark.parametrize(
    "change",
    [{"slurm_job_id": "9999"}, {"submit_nonce": "ffffffff"}, {"slurm_job_id": None}],
    ids=["other-job", "other-nonce", "no-job-id"],
)
def test_reharvest_of_another_submission_raises_and_keeps_the_entry(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    run_dir = _make_run_dir(tmp_path)
    root = tmp_path / "dataset"
    path = harvest_completed_run_to_dataset(_record(run_dir), run_dir, root, **_eligible_kwargs())
    before = _snapshot(path.parent)

    with pytest.raises(DatasetEntryExistsError):
        harvest_completed_run_to_dataset(
            _record(run_dir, **change),
            run_dir,
            root,
            **_eligible_kwargs(final_metrics={"heat_flux_es": 99.0}),
        )
    assert _snapshot(path.parent) == before


@pytest.mark.parametrize(
    "content",
    ['{"schema_version": "1.0", "run_id": "gx_run_001"}\n', "{not json"],
    ids=["legacy-v1", "corrupt"],
)
def test_legacy_or_corrupt_committed_manifest_is_never_overwritten(
    tmp_path: Path, content: str
) -> None:
    run_dir = _make_run_dir(tmp_path)
    run_out = tmp_path / "dataset" / "campaigns" / "pilot_50" / "runs" / "gx_run_001"
    run_out.mkdir(parents=True)
    (run_out / "run_manifest.json").write_text(content, encoding="utf-8")

    with pytest.raises(DatasetEntryExistsError):
        harvest_completed_run_to_dataset(
            _record(run_dir), run_dir, tmp_path / "dataset", **_eligible_kwargs()
        )
    assert (run_out / "run_manifest.json").read_text(encoding="utf-8") == content
    assert not (run_out / "fluxes.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "../escape"),
        ("run_id", "a/b"),
        ("run_id", ""),
        ("run_id", ".hidden"),
        ("campaign_id", "../../etc"),
        ("campaign_id", "-rf"),
        ("campaign_id", "x" * 200),
    ],
)
def test_unsafe_run_or_campaign_id_is_rejected_before_any_write(
    tmp_path: Path, field: str, value: str
) -> None:
    run_dir = _make_run_dir(tmp_path)
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Unsafe"):
        harvest_completed_run_to_dataset(
            _record(run_dir, **{field: value}), run_dir, root, **_eligible_kwargs()
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["queue"]


def test_symlinked_dataset_dir_leaving_the_root_is_rejected(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    root = tmp_path / "dataset"
    (root / "campaigns").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "campaigns" / "pilot_50").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="inside the dataset root"):
        harvest_completed_run_to_dataset(_record(run_dir), run_dir, root, **_eligible_kwargs())
    assert list(outside.iterdir()) == []


def test_non_finite_numbers_are_written_as_null(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    metrics = {
        "heat_flux_es": 3.72,
        "phi2_total": float("nan"),
        "nested": {"x": float("-inf")},
        "series": [1.0, float("inf")],
    }
    path = harvest_completed_run_to_dataset(
        _record(run_dir),
        run_dir,
        tmp_path / "dataset",
        **_eligible_kwargs(
            final_metrics=metrics,
            final_parameters={"nx": 64, "dt_max": float("inf")},
            walltime_elapsed_s=float("nan"),
        ),
    )
    manifest = _strict_load(path)  # raises on any NaN/Infinity token
    fluxes = _strict_load(path.parent / "fluxes.json")
    assert fluxes["metrics"]["phi2_total"] is None
    assert fluxes["metrics"]["nested"] == {"x": None}
    assert fluxes["metrics"]["series"] == [1.0, None]
    assert manifest["final_numerical_parameters"] == {"nx": 64, "dt_max": None}
    assert manifest["walltime_elapsed_s"] is None


def test_legacy_call_without_new_keywords_still_writes_a_manifest(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    record = RunRecord(
        run_id="gx_run_001",
        solver="GX",
        run_dir=str(run_dir),
        status="COMPLETED",
        equilibrium_id="w7x_standard",
    )
    path = harvest_completed_run_to_dataset(
        record,
        run_dir,
        tmp_path / "dataset",
        final_parameters={"nx": 64},
        final_metrics={"heat_flux_es": 2.1},
    )
    manifest = _strict_load(path)
    assert all(key in manifest for key in V1_KEYS)
    assert manifest["hands_free"] is True
    assert manifest["patched"] is False
    assert manifest["converged"] is None
    assert manifest["environment"] is None
    assert manifest["simulated"] is False
    assert manifest["dataset_eligible"] is False
    assert manifest["eligibility_reasons"] == ["convergence_unknown", "non_prod_environment"]
    assert manifest["eligibility_warnings"] == [
        "equilibrium_origin_unknown",
        "code_version_unknown",
    ]
    assert manifest["provenance"]["job_ids"] == []


def test_manifest_is_the_last_write_and_a_crash_leaves_the_entry_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run_dir(tmp_path)
    root = tmp_path / "dataset"
    real_write = manifest_mod.atomic_write_bytes
    written: list[str] = []

    def crash_on_commit(path: Path, data: bytes, **kwargs: Any) -> None:
        written.append(Path(path).name)
        if Path(path).name == "run_manifest.json":
            raise OSError("simulated crash before commit")
        real_write(path, data, **kwargs)

    monkeypatch.setattr(manifest_mod, "atomic_write_bytes", crash_on_commit)
    with pytest.raises(OSError, match="simulated crash"):
        harvest_completed_run_to_dataset(_record(run_dir), run_dir, root, **_eligible_kwargs())
    assert written[-3:] == ["fluxes.json", "checksums.sha256", "run_manifest.json"]
    run_out = root.resolve() / "campaigns" / "pilot_50" / "runs" / "gx_run_001"
    assert (run_out / "fluxes.json").is_file()
    assert not (run_out / "run_manifest.json").exists()
    assert not [p.name for p in run_out.iterdir() if p.name.endswith(".tmp")]

    # The uncommitted partial entry is simply rewritten by the next harvest.
    monkeypatch.setattr(manifest_mod, "atomic_write_bytes", real_write)
    assert _harvest(_record(run_dir), run_dir, root, **_eligible_kwargs())["dataset_eligible"]


def test_input_selection_and_geometry_discovery(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    run_dir = _make_run_dir(
        tmp_path,
        inputs={"gx.in": GX_INPUT, "other.in": "x = 1\n"},
        geometry={
            "equilibrium_mrx.eik.nc": b"eik",  # matches two patterns: published once
            ".geometry.partial": b"hidden",  # hidden: skipped
            "equilibrium_meta.json": b"{}",  # reserved: never copied over the real meta
        },
    )
    manifest = _harvest(_record(run_dir), run_dir, root, **_eligible_kwargs())
    assert manifest["input_file"] == "gx.in"
    assert sorted(manifest["equilibrium"]["geometry_files"]) == ["equilibrium_mrx.eik.nc"]
    meta = _strict_load(root / manifest["equilibrium"]["meta_path"])
    assert meta["equilibrium_id"] == "mrx_eq_0001"

    solo_dir = _make_run_dir(tmp_path, "gx_run_002", inputs={"cyclone.in": GX_INPUT})
    solo = _harvest(_record(solo_dir), solo_dir, root, **_eligible_kwargs())
    assert solo["input_file"] == "cyclone.in"
    assert solo["dataset_eligible"] is True

    two_dir = _make_run_dir(tmp_path, "gx_run_003", inputs={"a.in": GX_INPUT, "b.in": GX_INPUT})
    ambiguous = _harvest(_record(two_dir), two_dir, root, **_eligible_kwargs())
    assert ambiguous["eligibility_reasons"] == ["missing_input_file"]

    gene_dir = _make_run_dir(tmp_path, "gene_run_004", inputs={"parameters": "&box\n nx0 = 8\n/\n"})
    gene = _harvest(_record(gene_dir, solver="GENE"), gene_dir, root, **_eligible_kwargs())
    assert gene["input_file"] == "parameters"
    assert gene["code"]["name"] == "GENE"
    assert gene["dataset_eligible"] is True


def test_parameter_patches_are_normalized_for_both_record_formats(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    patches = [
        {
            "strike": 1,
            "action": "cold_restart",
            "before": {"nx": 64, "hyperdiffusion_coeff": 0.1},
            "after": {"nx": 96, "hyperdiffusion_coeff": 0.1, "D_new": 1},
            "restart_from_checkpoint": False,
        },
        {"retry": 0, "parameter": "dt_max", "old": 0.02, "new": 0.01},
    ]
    manifest = _harvest(
        _record(run_dir, parameter_patches=patches, retry_count=2),
        run_dir,
        tmp_path / "dataset",
        **_eligible_kwargs(),
    )
    normalized = manifest["provenance"]["parameter_patches"]
    assert normalized[0]["changes"] == [
        {"parameter": "D_new", "old": None, "new": 1},
        {"parameter": "nx", "old": 64, "new": 96},
    ]
    assert normalized[0]["action"] == "cold_restart"
    assert normalized[1]["changes"] == [{"parameter": "dt_max", "old": 0.02, "new": 0.01}]
    assert manifest["parameter_patches"] == patches  # schema-1.0 key keeps the raw entries
    assert manifest["patched"] is True


def test_flux_normalization_and_fsa_grad_x_in_fluxes_json(tmp_path: Path) -> None:
    root = tmp_path / "dataset"

    gx_dir = _make_run_dir(tmp_path, "gx_fsa")
    gx_manifest = harvest_completed_run_to_dataset(
        _record(gx_dir),
        gx_dir,
        root,
        **_eligible_kwargs(final_metrics={"heat_flux_es": 2.5, "fsa_grad_x": 1.37}),
    )
    gx_fluxes = _strict_load(gx_manifest.parent / "fluxes.json")
    assert gx_fluxes["flux_units"] == "gyroBohm"
    assert gx_fluxes["flux_normalization"] == "gyroBohm_divided_by_fsa_grad_x"
    assert gx_fluxes["fsa_grad_x"] == pytest.approx(1.37)

    nested_dir = _make_run_dir(tmp_path, "gx_nested_fsa")
    nested_manifest = harvest_completed_run_to_dataset(
        _record(nested_dir),
        nested_dir,
        root,
        **_eligible_kwargs(
            final_metrics={"heat_flux_es": 2.5, "scalar_metrics": {"fsa_grad_x": 1.12}}
        ),
    )
    nested_fluxes = _strict_load(nested_manifest.parent / "fluxes.json")
    assert nested_fluxes["fsa_grad_x"] == pytest.approx(1.12)

    gene_dir = _make_run_dir(tmp_path, "gene_norm", inputs={"parameters": "&box\n nx0 = 8\n/\n"})
    gene_manifest = harvest_completed_run_to_dataset(
        _record(gene_dir, solver="GENE"),
        gene_dir,
        root,
        **_eligible_kwargs(),
    )
    gene_fluxes = _strict_load(gene_manifest.parent / "fluxes.json")
    assert gene_fluxes["flux_units"] == "gyroBohm"
    assert gene_fluxes["flux_normalization"] == "gyroBohm_gene"
    assert "fsa_grad_x" not in gene_fluxes

    for idx, bad_fsa in enumerate((0.0, -0.5, True, float("nan"))):
        bad_dir = _make_run_dir(tmp_path, f"gx_bad_fsa_{idx}")
        bad_manifest = harvest_completed_run_to_dataset(
            _record(bad_dir),
            bad_dir,
            root,
            **_eligible_kwargs(final_metrics={"heat_flux_es": 2.5, "fsa_grad_x": bad_fsa}),
        )
        bad_fluxes = _strict_load(bad_manifest.parent / "fluxes.json")
        assert "fsa_grad_x" not in bad_fluxes
