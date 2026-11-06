"""Fail-closed dataset admission (ML-03), strict schema/units checks (ML-07), splits (ML-04)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import pytest

from kr1_surrogate import (
    DatasetSample,
    load_campaign_report,
    load_campaign_samples,
    split_by_equilibrium,
)

CAMPAIGN = "camp_ml"
PATCH = {
    "strike": 1,
    "action": "cold_restart",
    "before": {"nx": 64},
    "after": {"nx": 96},
    "restart_from_checkpoint": False,
}


def test_eligible_runs_are_admitted_including_recorded_patches(tmp_path: Path, write_run) -> None:
    write_run(tmp_path, CAMPAIGN, "gx_0001", "mrx_eq_0001")
    write_run(
        tmp_path,
        CAMPAIGN,
        "gene_0001",
        "mrx_eq_0001",
        solver="GENE",
        fidelity="high",
        backend="tacc_slurm",
        manifest_overrides={"retry_count": 1, "parameter_patches": [PATCH], "patched": True},
    )

    report = load_campaign_report(tmp_path, CAMPAIGN)
    assert report.excluded == ()
    assert [s.run_id for s in report.samples] == ["gene_0001", "gx_0001"]  # deterministic order
    gene = report.samples[0]
    assert (gene.solver, gene.fidelity, gene.backend) == ("GENE", "high", "tacc_slurm")
    assert gene.flux_units == "gyroBohm"
    assert gene.parameter_patches == (PATCH,)
    assert gene.fluxes["heat_flux_es"] == 2.5  # passed through, never re-normalized (ML-07)
    assert gene.equilibrium_meta == {"equilibrium_id": "mrx_eq_0001"}
    assert gene.numerical_parameters == {"nx": 64, "dt_max": 0.02}

    unpatched = load_campaign_report(tmp_path, CAMPAIGN, unpatched_only=True)
    assert [s.run_id for s in unpatched.samples] == ["gx_0001"]
    assert unpatched.exclusion_counts == {"patched": 1}


def _tamper_fluxes(run_dir: Path) -> None:
    path = run_dir / "fluxes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["heat_flux_es"] = 99.0  # silent corruption after the manifest was written
    path.write_text(json.dumps(data), encoding="utf-8")


EXCLUSION_CASES: list[tuple[str, dict[str, Any], str]] = [
    ("not-converged", {"manifest_overrides": {"converged": False}}, "not_converged"),
    ("converged-null", {"manifest_overrides": {"converged": None}}, "convergence_unknown"),
    ("escalated", {"manifest_overrides": {"escalated": True}}, "escalated"),
    ("simulated", {"manifest_overrides": {"simulated": True}}, "simulated"),
    ("non-prod", {"manifest_overrides": {"environment": "sim"}}, "non_prod_environment"),
    ("env-null", {"manifest_overrides": {"environment": None}}, "non_prod_environment"),
    (
        "writer-verdict",
        {
            "manifest_overrides": {
                "dataset_eligible": False,
                "eligibility_reasons": ["gcp_provenance_unverified", "convergence_unverified"],
            }
        },
        "marked_ineligible:gcp_provenance_unverified",
    ),
    (
        "eligible-with-blockers",
        {"manifest_overrides": {"eligibility_reasons": ["invalid_flux_value"]}},
        "invalid_field:eligibility_reasons",
    ),
    ("no-converged", {"drop_manifest": ("converged",)}, "missing_field:converged"),
    ("no-escalated", {"drop_manifest": ("escalated",)}, "missing_field:escalated"),
    ("no-simulated", {"drop_manifest": ("simulated",)}, "missing_field:simulated"),
    ("no-environment", {"drop_manifest": ("environment",)}, "missing_field:environment"),
    (
        "no-eligibility",
        {"drop_manifest": ("dataset_eligible",)},
        "missing_field:dataset_eligible",
    ),
    (
        "no-reasons",
        {"drop_manifest": ("eligibility_reasons",)},
        "missing_field:eligibility_reasons",
    ),
    ("no-patches", {"drop_manifest": ("parameter_patches",)}, "missing_field:parameter_patches"),
    ("no-patched", {"drop_manifest": ("patched",)}, "missing_field:patched"),
    (
        "patched-flag-lies",
        {"manifest_overrides": {"patched": True}},
        "invalid_field:patched",
    ),
    ("no-eq-id", {"drop_manifest": ("equilibrium_id",)}, "missing_field:equilibrium_id"),
    (
        "empty-eq-id",
        {"manifest_overrides": {"equilibrium_id": ""}, "write_equilibrium_meta": False},
        "invalid_field:equilibrium_id",
    ),
    ("no-solver", {"drop_manifest": ("solver",)}, "missing_field:solver"),
    ("str-bool", {"manifest_overrides": {"escalated": "false"}}, "invalid_field:escalated"),
    (
        "unrecorded-patch",
        {"manifest_overrides": {"retry_count": 2, "parameter_patches": [PATCH], "patched": True}},
        "unrecorded_patches",
    ),
    (
        "legacy-schema",
        {"manifest_overrides": {"schema_version": "1.0"}},
        "unsupported_manifest_schema",
    ),
    ("run-id-mismatch", {"manifest_overrides": {"run_id": "other_run"}}, "identity_mismatch"),
    (
        "campaign-mismatch",
        {"manifest_overrides": {"campaign_id": "other_campaign"}},
        "identity_mismatch",
    ),
    (
        "unsafe-eq-id",
        {"manifest_overrides": {"equilibrium_id": "../escape"}, "write_equilibrium_meta": False},
        "invalid_field:equilibrium_id",
    ),
    ("bad-solver", {"manifest_overrides": {"solver": "gx"}}, "invalid_field:solver"),
    ("bad-backend", {"manifest_overrides": {"backend": "laptop"}}, "invalid_field:backend"),
    (
        "bool-param",
        {"manifest_overrides": {"final_numerical_parameters": {"nx": True}}},
        "invalid_field:final_numerical_parameters",
    ),
    ("no-files", {"drop_manifest": ("files",)}, "missing_field:files"),
    (
        "input-file-uncovered",
        {"manifest_overrides": {"input_file": "gx.in"}},
        "invalid_field:files",
    ),
    ("flux-units", {"fluxes_overrides": {"flux_units": "MW/m^2"}}, "unsupported_flux_units"),
    ("no-flux-units", {"drop_fluxes": ("flux_units",)}, "missing_field:flux_units"),
    ("legacy-fluxes", {"fluxes_overrides": {"schema_version": "1.0"}}, "unsupported_fluxes_schema"),
    ("fluxes-solver", {"fluxes_overrides": {"solver": "GENE"}}, "identity_mismatch"),
    ("str-flux", {"fluxes_overrides": {"particle_flux": "0.1"}}, "invalid_flux_value"),
    ("bool-flux", {"fluxes_overrides": {"heat_flux_em": True}}, "invalid_flux_value"),
    ("nan-flux", {"fluxes_overrides": {"heat_flux_em": float("nan")}}, "unreadable_fluxes"),
    ("null-primary", {"heat_flux_es": None}, "missing_primary_flux"),
    ("no-eq-meta", {"write_equilibrium_meta": False}, "missing_equilibrium_meta"),
    # GXSRC-08 / ZEN-12: optional normalization fields are validated fail-closed when present.
    (
        "unknown_flux_normalization",
        {"fluxes_overrides": {"flux_normalization": "Q_avgs"}},
        "unsupported_flux_normalization",
    ),
    ("fsa_grad_x_zero", {"fluxes_overrides": {"fsa_grad_x": 0.0}}, "invalid_field:fsa_grad_x"),
    ("fsa_grad_x_negative", {"fluxes_overrides": {"fsa_grad_x": -1.2}}, "invalid_field:fsa_grad_x"),
    ("fsa_grad_x_string", {"fluxes_overrides": {"fsa_grad_x": "1.2"}}, "invalid_field:fsa_grad_x"),
    ("fsa_grad_x_bool", {"fluxes_overrides": {"fsa_grad_x": True}}, "invalid_field:fsa_grad_x"),
    ("fsa_grad_x_nan", {"fluxes_overrides": {"fsa_grad_x": float("nan")}}, "unreadable_fluxes"),
]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [case[1:] for case in EXCLUSION_CASES],
    ids=[case[0] for case in EXCLUSION_CASES],
)
def test_ineligible_or_malformed_run_is_excluded_with_reason(
    tmp_path: Path, write_run, kwargs: dict[str, Any], reason: str
) -> None:
    write_run(tmp_path, CAMPAIGN, "gx_0001", "mrx_eq_0001", **kwargs)
    report = load_campaign_report(tmp_path, CAMPAIGN)
    assert report.samples == ()
    assert [(e.run_id, e.reason) for e in report.excluded] == [("gx_0001", reason)]


def test_checksum_mismatch_and_missing_files_are_excluded(tmp_path: Path, write_run) -> None:
    _tamper_fluxes(write_run(tmp_path, CAMPAIGN, "tampered", "eq_a"))
    (write_run(tmp_path, CAMPAIGN, "no_manifest", "eq_b") / "run_manifest.json").unlink()
    (write_run(tmp_path, CAMPAIGN, "no_fluxes", "eq_c") / "fluxes.json").unlink()
    (write_run(tmp_path, CAMPAIGN, "bad_json", "eq_d") / "run_manifest.json").write_text("[1, 2]")
    report = load_campaign_report(tmp_path, CAMPAIGN)
    assert report.samples == ()
    assert {e.run_id: e.reason for e in report.excluded} == {
        "bad_json": "unreadable_run_manifest",
        "no_fluxes": "missing_fluxes",
        "no_manifest": "missing_run_manifest",
        "tampered": "checksum_mismatch",
    }


def test_exclusion_counts_by_reason_are_reported_and_logged(
    tmp_path: Path, write_run, caplog: pytest.LogCaptureFixture
) -> None:
    write_run(tmp_path, CAMPAIGN, "ok_1", "eq_1")
    write_run(tmp_path, CAMPAIGN, "ok_2", "eq_2")
    write_run(tmp_path, CAMPAIGN, "esc_1", "eq_3", manifest_overrides={"escalated": True})
    write_run(tmp_path, CAMPAIGN, "esc_2", "eq_4", manifest_overrides={"escalated": True})
    write_run(tmp_path, CAMPAIGN, "sim_1", "eq_5", manifest_overrides={"simulated": True})
    write_run(tmp_path, CAMPAIGN, "legacy", "eq_6", drop_manifest=("converged",))
    (tmp_path / "campaigns" / CAMPAIGN / "runs" / ".staging").mkdir()  # not a run: skipped
    (tmp_path / "campaigns" / CAMPAIGN / "runs" / "README.txt").write_text("not a run")

    report = load_campaign_report(tmp_path, CAMPAIGN)
    assert [s.run_id for s in report.samples] == ["ok_1", "ok_2"]
    assert report.exclusion_counts == {
        "escalated": 2,
        "missing_field:converged": 1,
        "simulated": 1,
    }

    with caplog.at_level(logging.WARNING, logger="kr1_surrogate.dataset_loader"):
        samples = load_campaign_samples(tmp_path, CAMPAIGN)
    assert [s.run_id for s in samples] == ["ok_1", "ok_2"]
    assert "excluded 4" in caplog.text and "'escalated': 2" in caplog.text


def test_optional_filters_never_relax_eligibility(tmp_path: Path, write_run) -> None:
    write_run(tmp_path, CAMPAIGN, "null_q", "eq_1", heat_flux_es=None)
    write_run(
        tmp_path, CAMPAIGN, "esc_null_q", "eq_2", heat_flux_es=None,
        manifest_overrides={"escalated": True},
    )  # fmt: skip
    write_run(tmp_path, CAMPAIGN, "steered", "eq_3", manifest_overrides={"hands_free": False})

    relaxed = load_campaign_report(tmp_path, CAMPAIGN, require_valid_fluxes=False)
    assert [s.run_id for s in relaxed.samples] == ["null_q", "steered"]
    assert relaxed.exclusion_counts == {"escalated": 1}

    hands_free = load_campaign_report(tmp_path, CAMPAIGN, hands_free_only=True)
    assert hands_free.samples == ()
    assert hands_free.exclusion_counts == {
        "escalated": 1,
        "missing_primary_flux": 1,
        "not_hands_free": 1,
    }


def test_symlinked_run_folder_is_rejected(tmp_path: Path, write_run) -> None:
    outside = write_run(tmp_path / "elsewhere", "other", "gx_0001", "eq_1")
    runs_dir = tmp_path / "campaigns" / CAMPAIGN / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "gx_0001").symlink_to(outside, target_is_directory=True)
    report = load_campaign_report(tmp_path, CAMPAIGN)
    assert [(e.run_id, e.reason) for e in report.excluded] == [("gx_0001", "unsafe_path")]


def test_campaign_id_is_validated_and_missing_campaign_is_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="campaign_id"):
        load_campaign_report(tmp_path, "../outside")
    report = load_campaign_report(tmp_path, "no_such_campaign")
    assert report.samples == () and report.excluded == ()


def _sample(run_id: str, eq_id: str, fidelity: str = "low") -> DatasetSample:
    return DatasetSample(
        campaign_id=CAMPAIGN,
        run_id=run_id,
        equilibrium_id=eq_id,
        solver="GX" if fidelity == "low" else "GENE",
        fidelity=fidelity,
        backend="gcp_batch" if fidelity == "low" else "tacc_slurm",
        numerical_parameters={},
        fluxes={},
        equilibrium_meta={},
        manifest_path="",
        flux_units="gyroBohm",
    )


def _assignment(splits: dict[str, list[DatasetSample]]) -> dict[str, str]:
    return {s.equilibrium_id: name for name, members in splits.items() for s in members}


def test_default_split_is_stable_when_the_dataset_grows() -> None:
    v1 = [_sample(f"r{i}", f"eq_{i:03d}") for i in range(40)]
    v2 = v1 + [_sample(f"r{i}", f"eq_{i:03d}") for i in range(40, 120)]
    before, after = _assignment(split_by_equilibrium(v1)), _assignment(split_by_equilibrium(v2))
    assert all(after[eq] == split for eq, split in before.items())


def _multi_fidelity_stratum(group: tuple[DatasetSample, ...]) -> str:
    return "multi" if len({s.fidelity for s in group}) > 1 else group[0].fidelity


def test_stratified_split_is_exact_per_stratum_leak_free_and_order_independent() -> None:
    samples = [_sample(f"gx_{i}", f"eq_low_{i}") for i in range(10)]
    for i in range(10):  # 10 equilibria with both a GX and a GENE run
        samples += [
            _sample(f"gx_m{i}", f"eq_multi_{i}"),
            _sample(f"gene_m{i}", f"eq_multi_{i}", "high"),
        ]
    samples.append(_sample("gx_lonely", "eq_single", "high"))  # singleton stratum

    splits = split_by_equilibrium(
        samples, train_frac=0.6, val_frac=0.2, stratify_by=_multi_fidelity_stratum
    )
    assignment = _assignment(splits)
    eq_sets = {name: {s.equilibrium_id for s in members} for name, members in splits.items()}
    assert eq_sets["train"].isdisjoint(eq_sets["val"])
    assert eq_sets["train"].isdisjoint(eq_sets["test"])
    assert eq_sets["val"].isdisjoint(eq_sets["test"])
    assert sum(len(m) for m in splits.values()) == len(samples)
    for prefix in ("eq_low_", "eq_multi_"):
        counts = [
            sum(1 for eq, split in assignment.items() if eq.startswith(prefix) and split == name)
            for name in ("train", "val", "test")
        ]
        assert counts == [6, 2, 2]
    assert assignment["eq_single"] == "train"  # largest remainder sends a singleton to train

    shuffled = samples[:]
    random.Random(7).shuffle(shuffled)
    again = split_by_equilibrium(
        shuffled, train_frac=0.6, val_frac=0.2, stratify_by=_multi_fidelity_stratum
    )
    assert _assignment(again) == assignment
    assert [s.run_id for s in again["train"]] == [
        s.run_id for s in shuffled if assignment[s.equilibrium_id] == "train"
    ]


def test_stratify_by_must_return_a_string_label() -> None:
    with pytest.raises(ValueError, match="str label"):
        split_by_equilibrium([_sample("r", "eq")], stratify_by=lambda group: len(group))  # type: ignore[arg-type,return-value]


def test_flux_normalization_and_fsa_grad_x_are_exposed_without_rescaling(
    tmp_path: Path, write_run
) -> None:
    write_run(tmp_path, CAMPAIGN, "legacy", "eq_1")  # older writer: fields absent
    write_run(
        tmp_path,
        CAMPAIGN,
        "gx_new",
        "eq_2",
        heat_flux_es=2.5,
        fluxes_overrides={
            "flux_normalization": "gyroBohm_divided_by_fsa_grad_x",
            "fsa_grad_x": 1.25,
        },
    )
    write_run(tmp_path, CAMPAIGN, "null_fsa", "eq_3", fluxes_overrides={"fsa_grad_x": None})
    report = load_campaign_report(tmp_path, CAMPAIGN)
    assert report.excluded == ()
    by_id = {s.run_id: s for s in report.samples}
    assert (by_id["legacy"].flux_normalization, by_id["legacy"].fsa_grad_x) == ("", None)
    assert by_id["null_fsa"].fsa_grad_x is None
    new = by_id["gx_new"]
    assert (new.flux_normalization, new.fsa_grad_x) == ("gyroBohm_divided_by_fsa_grad_x", 1.25)
    assert new.fluxes["heat_flux_es"] == 2.5  # never multiplied by fsa_grad_x


def test_min_primary_flux_drops_damped_runs_for_log_targets(tmp_path: Path, write_run) -> None:
    write_run(tmp_path, CAMPAIGN, "q_pos", "eq_1", heat_flux_es=2.5)
    write_run(tmp_path, CAMPAIGN, "q_zero", "eq_2", heat_flux_es=0.0)
    write_run(tmp_path, CAMPAIGN, "q_neg", "eq_3", heat_flux_es=-0.4)
    write_run(tmp_path, CAMPAIGN, "q_small", "eq_4", heat_flux_es=0.05)

    # Default: damped / negative runs are admitted (stability information).
    assert len(load_campaign_report(tmp_path, CAMPAIGN).samples) == 4

    report = load_campaign_report(tmp_path, CAMPAIGN, min_primary_flux=0.0)
    assert [s.run_id for s in report.samples] == ["q_pos", "q_small"]
    assert report.exclusion_counts == {"below_min_primary_flux": 2}

    samples = load_campaign_samples(tmp_path, CAMPAIGN, min_primary_flux=0.1)
    assert [s.run_id for s in samples] == ["q_pos"]


def test_min_primary_flux_excludes_missing_primary_even_when_not_required(
    tmp_path: Path, write_run
) -> None:
    write_run(tmp_path, CAMPAIGN, "null_q", "eq_1", heat_flux_es=None)
    report = load_campaign_report(
        tmp_path, CAMPAIGN, require_valid_fluxes=False, min_primary_flux=0.0
    )
    assert report.exclusion_counts == {"missing_primary_flux": 1}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.0"])
def test_min_primary_flux_must_be_finite_number(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(ValueError, match="min_primary_flux"):
        load_campaign_report(tmp_path, CAMPAIGN, min_primary_flux=bad)
