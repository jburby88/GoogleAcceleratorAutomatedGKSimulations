"""Unit tests for KR1 leak-free `equilibrium_id` dataset splitting & Augmented Lagrangian state."""

from __future__ import annotations

from pathlib import Path

import pytest
from hpc_core import RunRecord, harvest_completed_run_to_dataset

from kr1_surrogate import (
    AugmentedLagrangianState,
    load_campaign_report,
    load_campaign_samples,
    split_by_equilibrium,
)


def test_leak_free_split_keeps_multifidelity_same_equilibrium_together(
    tmp_path: Path, write_run
) -> None:
    # Simulate 20 distinct ToBlick/mrx equilibria, each with BOTH GX (gcp_batch) and GENE (tacc_slurm) runs
    for eq_idx in range(1, 21):
        eq_id = f"mrx_eq_{eq_idx:04d}"
        for solver, backend in (("GX", "gcp_batch"), ("GENE", "tacc_slurm")):
            write_run(
                tmp_path / "dataset",
                "camp_mrx",
                f"{solver.lower()}_{eq_idx:04d}",
                eq_id,
                solver=solver,
                fidelity="low" if solver == "GX" else "high",
                backend=backend,
                heat_flux_es=2.5 + 0.1 * eq_idx,
            )

    samples = load_campaign_samples(tmp_path / "dataset", "camp_mrx")
    assert len(samples) == 40

    splits = split_by_equilibrium(samples, train_frac=0.7, val_frac=0.15)
    train_eqs = {s.equilibrium_id for s in splits["train"]}
    val_eqs = {s.equilibrium_id for s in splits["val"]}
    test_eqs = {s.equilibrium_id for s in splits["test"]}

    # Zero equilibrium_id overlap across train / val / test!
    assert train_eqs.isdisjoint(val_eqs)
    assert train_eqs.isdisjoint(test_eqs)
    assert val_eqs.isdisjoint(test_eqs)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 40


def test_augmented_lagrangian_outer_loop_updates() -> None:
    al = AugmentedLagrangianState(
        constraint_names=["mrx_force_balance", "bootstrap_current", "gk_heat_flux"],
        penalty_mu=10.0,
    )
    residuals = {"mrx_force_balance": 0.02, "bootstrap_current": -0.05, "gk_heat_flux": 0.12}
    obj = al.evaluate_augmented_objective(primary_objective=1.5, equality_residuals=residuals)
    assert obj > 1.5

    mults = al.update_outer_loop(residuals)
    assert mults["gk_heat_flux"] == pytest.approx(1.2)
    # Stalled residual (> constraint_tol=0.05) triggers penalty_mu growth
    al.update_outer_loop(residuals)
    assert al.penalty_mu == pytest.approx(20.0)


def test_dataset_loader_require_valid_fluxes_and_nan_split_guards(
    tmp_path: Path, write_run
) -> None:
    dataset_root = tmp_path / "dataset"
    for idx, q_es in enumerate((3.14, None)):
        write_run(dataset_root, "camp_filter", f"gx_{idx:02d}", f"eq_{idx:02d}", heat_flux_es=q_es)

    all_samples = load_campaign_samples(dataset_root, "camp_filter", require_valid_fluxes=False)
    assert len(all_samples) == 2

    valid_samples = load_campaign_samples(dataset_root, "camp_filter")
    assert len(valid_samples) == 1
    assert valid_samples[0].run_id == "gx_00"

    for bad_train, bad_val in (
        (float("nan"), 0.1),
        (0.8, float("nan")),
        (float("inf"), 0.1),
        (0.8, float("-inf")),
        (0.0, 0.1),
        (0.8, -0.1),
        (0.85, 0.15),
    ):
        with pytest.raises(ValueError):
            split_by_equilibrium(valid_samples, train_frac=bad_train, val_frac=bad_val)


def test_escalated_run_from_real_kr2_writer_is_never_admitted(tmp_path: Path) -> None:
    # The pre-fix loader admitted any run with a finite heat_flux_es, including escalated ones.
    # Whatever manifest version hpc_core writes, an escalated run must be excluded (ML-03).
    run_dir = tmp_path / "work" / "gx_escalated"
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text("[Dimensions]\nnx = 64\n", encoding="utf-8")
    rec = RunRecord(
        run_id="gx_escalated",
        solver="GX",
        run_dir=str(run_dir),
        status="COMPLETED",
        campaign_id="camp_writer",
        equilibrium_id="eq_esc",
        directives_history=[{"action": "escalate_to_human", "reason": "test"}],
    )
    harvest_completed_run_to_dataset(
        record=rec,
        run_dir=run_dir,
        dataset_root=tmp_path / "dataset",
        final_parameters={"nx": 64},
        final_metrics={"heat_flux_es": 3.0},
    )
    report = load_campaign_report(tmp_path / "dataset", "camp_writer")
    assert report.samples == ()
    assert [e.run_id for e in report.excluded] == ["gx_escalated"]


def _harvest_v2(dataset_root: Path, run_id: str, *, backend: str = "tacc_slurm", **kwargs) -> None:
    run_dir = dataset_root.parent / "work" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "gx.in").write_text("[Dimensions]\nnx = 64\n", encoding="utf-8")
    rec = RunRecord(
        run_id=run_id,
        solver="GX",
        run_dir=str(run_dir),
        status="COMPLETED",
        campaign_id="camp_v2",
        backend=backend,
        equilibrium_id="eq_v2",
        geometry_id="eq_v2",
        parameter_patches=list(kwargs.pop("parameter_patches", [])),
        retry_count=kwargs.pop("retry_count", 0),
    )
    harvest_completed_run_to_dataset(
        record=rec,
        run_dir=run_dir,
        dataset_root=dataset_root,
        final_parameters={"nx": 64},
        final_metrics=kwargs.pop("final_metrics", {"heat_flux_es": 3.1}),
        **kwargs,
    )


def test_kr2_writer_v2_contract_round_trip(tmp_path: Path) -> None:
    # End-to-end contract test against the real hpc_core writer (manifest schema 2.0).
    dataset_root = tmp_path / "dataset"
    good = {
        "converged": True,
        "convergence_method": "time_average_stationarity",
        "simulated": False,
        "environment": "prod",
        "flux_units": "gyroBohm",
        "code_version": "gx-test",
    }
    patch = {"strike": 1, "action": "cold_restart", "before": {"nx": 64}, "after": {"nx": 96}}
    _harvest_v2(dataset_root, "ok", **good)
    _harvest_v2(dataset_root, "patched", parameter_patches=[patch], retry_count=1, **good)
    _harvest_v2(dataset_root, "sim", **{**good, "simulated": True})
    _harvest_v2(dataset_root, "noenv", **{k: v for k, v in good.items() if k != "environment"})
    _harvest_v2(dataset_root, "noconv", **{k: v for k, v in good.items() if k != "converged"})
    _harvest_v2(dataset_root, "weak", **{**good, "convergence_method": "short_series_last_sample"})
    _harvest_v2(dataset_root, "gcp", backend="gcp_batch", **good)
    _harvest_v2(
        dataset_root,
        "gcp_ok",
        backend="gcp_batch",
        final_metrics={"heat_flux_es": 3.1, "provenance_ok": 1.0},
        **good,
    )

    report = load_campaign_report(dataset_root, "camp_v2")
    assert [s.run_id for s in report.samples] == ["gcp_ok", "ok", "patched"]
    assert {e.run_id: e.reason for e in report.excluded} == {
        "gcp": "marked_ineligible:gcp_provenance_unverified",
        "noconv": "convergence_unknown",
        "noenv": "non_prod_environment",
        "sim": "simulated",
        "weak": "marked_ineligible:convergence_unverified",
    }
    patched = next(s for s in report.samples if s.run_id == "patched")
    assert patched.parameter_patches[0]["after"] == {"nx": 96}
    assert patched.flux_units == "gyroBohm"
    assert patched.fluxes["heat_flux_es"] == 3.1
    unpatched = load_campaign_report(dataset_root, "camp_v2", unpatched_only=True)
    assert [s.run_id for s in unpatched.samples] == ["gcp_ok", "ok"]


def test_augmented_lagrangian_validation_tolerance_and_clamping() -> None:
    al = AugmentedLagrangianState(
        constraint_names=["mrx_force_balance", "gk_heat_flux"],
        penalty_mu=10.0,
        constraint_tol=0.05,
        multiplier_max=100.0,
    )

    # Missing key raises ValueError
    with pytest.raises(ValueError, match="Missing equality_residuals"):
        al.evaluate_augmented_objective(1.0, {"mrx_force_balance": 0.01})
    with pytest.raises(ValueError, match="Missing equality_residuals"):
        al.update_outer_loop({"mrx_force_balance": 0.01})

    # Non-finite primary_objective or residual raises ValueError
    with pytest.raises(ValueError, match="primary_objective"):
        al.evaluate_augmented_objective(
            float("nan"), {"mrx_force_balance": 0.01, "gk_heat_flux": 0.02}
        )
    with pytest.raises(ValueError, match="must be finite"):
        al.evaluate_augmented_objective(
            1.0, {"mrx_force_balance": float("inf"), "gk_heat_flux": 0.02}
        )
    with pytest.raises(ValueError, match="must be finite"):
        al.update_outer_loop({"mrx_force_balance": 0.01, "gk_heat_flux": float("-inf")})

    # Stalled residual BELOW constraint_tol (0.03 <= 0.05) does NOT grow penalty_mu
    sub_tol_res = {"mrx_force_balance": 0.01, "gk_heat_flux": 0.03}
    al.update_outer_loop(sub_tol_res)
    al.update_outer_loop(sub_tol_res)
    assert al.penalty_mu == pytest.approx(10.0)

    # Multiplier clamping to [-multiplier_max, multiplier_max]
    al.penalty_mu = 1000.0
    mults = al.update_outer_loop({"mrx_force_balance": -0.5, "gk_heat_flux": 0.5})
    assert mults["mrx_force_balance"] == pytest.approx(-100.0)
    assert mults["gk_heat_flux"] == pytest.approx(100.0)
