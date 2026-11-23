"""Command-line interface for inspecting the registry and running experiments."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from itg_surrogate.registry import (
    LEADING_EXPERIMENT,
    REFERENCE_OOD_BASELINES,
    get_experiment,
    get_sweep,
    list_experiments,
    list_sweeps,
)
from itg_surrogate.registry.specs import ExperimentConfig


def _cmd_list() -> None:
    lead = LEADING_EXPERIMENT
    bm = lead.reference_benchmark
    print("=== Leading Baseline to Beat (Unseen-Stellarator Split) ===")
    print(f"  Name        : {lead.name}")
    print(f"  Description : {lead.description}")
    if bm is not None:
        print(
            f"  OOD Metrics : 8-Seed Test R^2 = {bm.ens8_test_r2:.5f} | "
            f"Indiv Z_96 Test R^2 = {bm.indiv_z96_test_r2:.5f} | "
            f"Params = {bm.n_params:,}"
        )

    print("\n=== Reference Baselines on Canonical Unseen-Stellarator Split ===")
    for name, metrics in REFERENCE_OOD_BASELINES.items():
        metrics_str = ", ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        print(f"  {name:38s}: {metrics_str}")

    print("\n=== Registered Ablation Sweeps ===")
    for sweep_name in list_sweeps():
        exps = get_sweep(sweep_name)
        exp_names = ", ".join(e.name for e in exps)
        print(f"  {sweep_name:26s} ({len(exps)} runs): {exp_names}")

    print("\n=== Registered Individual Experiments ===")
    for exp_name in list_experiments():
        exp = get_experiment(exp_name)
        ref_str = (
            f" [ref ens8 R^2 = {exp.reference_benchmark.ens8_test_r2:.5f}]"
            if exp.reference_benchmark is not None
            else ""
        )
        print(f"  {exp_name:34s}{ref_str}")


def _cmd_show(name: str) -> None:
    exp = get_experiment(name)
    print(json.dumps(exp.to_dict(), indent=2))


def _apply_overrides(
    exp: ExperimentConfig,
    epochs: int | None,
    num_seeds: int | None,
) -> ExperimentConfig:
    updated = exp
    if epochs is not None:
        updated = dataclasses.replace(
            updated,
            optimizer=dataclasses.replace(updated.optimizer, num_epochs=epochs),
        )
    if num_seeds is not None:
        updated = dataclasses.replace(updated, num_seeds=num_seeds)
    return updated


def _cmd_train(args: argparse.Namespace) -> None:
    from itg_surrogate.data import load_stellarator_h5  # pylint: disable=g-import-not-at-top
    from itg_surrogate.models.tabular import run_xgboost_baselines  # pylint: disable=g-import-not-at-top
    from itg_surrogate.training import (
        run_sweep,  # pylint: disable=g-import-not-at-top
        train_experiment,  # pylint: disable=g-import-not-at-top
    )

    if args.sweep:
        exps = [_apply_overrides(e, args.epochs, args.num_seeds) for e in get_sweep(args.sweep)]
    else:
        exps = [_apply_overrides(get_experiment(args.experiment), args.epochs, args.num_seeds)]

    dataset = load_stellarator_h5(args.h5_path, config=exps[0].data_split)
    print(
        f"Loaded unseen-stellarator split from {args.h5_path}: "
        f"Train={len(dataset.split.train_idx):,} ({dataset.split.n_train_groups:,} groups) | "
        f"Val={len(dataset.split.val_idx):,} ({dataset.split.n_val_groups:,} groups) | "
        f"Test={len(dataset.split.test_idx):,} ({dataset.split.n_test_groups:,} groups)"
    )

    xgb_payload: list[dict[str, Any]] = []
    if args.with_xgboost:
        for res in run_xgboost_baselines(dataset):
            print(
                f"  [{res.name:28s}] Val R^2={res.val_r2:.5f} | "
                f"Test R^2={res.test_r2:.5f} | "
                f"Unstable Test R^2={res.unstable_test_r2:.5f}"
            )
            xgb_payload.append(
                {
                    "name": res.name,
                    "val_r2": res.val_r2,
                    "test_r2": res.test_r2,
                    "unstable_test_r2": res.unstable_test_r2,
                    "per_class_test_r2": dict(res.per_class_test_r2),
                }
            )

    payload: dict[str, Any] = {
        "mode": "sweep" if args.sweep else "experiment",
        "target_name": args.sweep if args.sweep else args.experiment,
        "h5_path": str(args.h5_path),
        "split_summary": {
            "n_train": int(len(dataset.split.train_idx)),
            "n_val": int(len(dataset.split.val_idx)),
            "n_test": int(len(dataset.split.test_idx)),
            "n_train_groups": int(dataset.split.n_train_groups),
            "n_val_groups": int(dataset.split.n_val_groups),
            "n_test_groups": int(dataset.split.n_test_groups),
        },
    }
    if xgb_payload:
        payload["xgboost_baselines"] = xgb_payload

    if len(exps) == 1 and not args.sweep:
        result = train_experiment(exps[0], dataset)
        print(
            f"\nCompleted {result.experiment.name}: "
            f"Indiv Z_96 R^2 = {result.indiv_test_r2_mean:.5f} +- {result.indiv_test_r2_std:.5f} | "
            f"{result.experiment.num_seeds}-Seed Test R^2 = {result.ens_test_r2:.5f} | "
            f"Unstable Test R^2 = {result.ens_unstable_test_r2:.5f} | "
            f"RMSE = {result.ens_test_rmse:.4f}"
        )
        payload["result"] = result.to_summary_dict()
    else:
        sweep_summary = run_sweep(exps, dataset)
        payload["sweep_summary"] = sweep_summary.to_summary_dict()

    if args.output_json:
        out_path = pathlib.Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved JSON metrics summary to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itg-surrogate",
        description=(
            "JAX/Flax periodic 1D CNN surrogates for nonlinear stellarator ITG "
            "turbulence on unseen stellarator equilibria."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list",
        help="List the leading baseline-to-beat, sweeps, and experiments.",
    )

    show_p = sub.add_parser(
        "show",
        help="Print full JSON configuration for a registered experiment.",
    )
    show_p.add_argument(
        "name",
        default="leading",
        nargs="?",
        help="Experiment name (default: 'leading').",
    )

    train_p = sub.add_parser(
        "train",
        help="Train and evaluate a registered experiment or ablation sweep.",
    )
    train_p.add_argument(
        "--h5-path",
        required=True,
        help="Path to 20250102-01_GX_stellarator_dataset.h5.",
    )
    group = train_p.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment",
        default="leading",
        help="Registered experiment name (default: 'leading').",
    )
    group.add_argument(
        "--sweep",
        default=None,
        help="Registered ablation sweep name (e.g. 'ood_regularization').",
    )
    train_p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    train_p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Optional override for number of sharded ensemble seeds.",
    )
    train_p.add_argument(
        "--with-xgboost",
        action="store_true",
        help="Also evaluate 3-feature and 12-feature XGBoost baselines.",
    )
    train_p.add_argument(
        "--output-json",
        default=None,
        help="Optional path (e.g. on /gcs/...) to save structured JSON metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        _cmd_list()
    elif args.command == "show":
        _cmd_show(args.name)
    elif args.command == "train":
        _cmd_train(args)
    else:
        parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
