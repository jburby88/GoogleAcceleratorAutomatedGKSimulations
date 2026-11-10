# `surrogates` — Gyrokinetic Surrogates & Dataset Ingestion

This package contains tools for loading gyrokinetic simulation outputs, training surrogates, and enqueuing active-learning validation runs.

## Modules

### `gk_surrogates` / `kr1_surrogate.dataset_loader`
Loads simulation campaign runs produced by `platform/hpc-core` and validates dataset eligibility.

- **Strict dataset validation** (`load_campaign_report`, `load_campaign_samples`): a run is admitted only if its `run_manifest.json` and `fluxes.json` are strict JSON with schema `2.0`, identifiers agree, `converged` is `true`, `escalated` and `simulated` are `false`, `environment` is `prod`, the writer's verdict is `dataset_eligible: true`, and checksums match.
- **Leak-free splits** (`split_by_equilibrium`): ensures multi-fidelity GX + GENE runs of one equilibrium never leak across train/val/test splits.

### `kr1_surrogate.augmented_lagrangian`
Outer-loop state of a Powell–Hestenes–Rockafellar augmented Lagrangian for equality and inequality constraints, holding multipliers and penalty parameters.

### `kr3_design.active_learning_queue`
Helper (`enqueue_validation_run`) that places high-uncertainty candidate equilibria into the simulation queue (`queue/pending/`) for GX or GENE validation.
