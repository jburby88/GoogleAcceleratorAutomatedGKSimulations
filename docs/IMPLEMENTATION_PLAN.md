# Implementation & Rollout Plan (`docs/IMPLEMENTATION_PLAN.md`)

## Phase 1 (Weeks 1–3): Platform Foundation & 50-Run Pilot
- Deploy `platform/infra/terraform` with `make gcp-deploy` (`aurora-steer`, private IAP `dashboard`, `sim_babysitter_telemetry` BigQuery dataset, `<project>-gk-dataset` GCS bucket, `gcp-gx-babysitter` Cloud Run job, remote Terraform state); seed the `pilot_50` row in the `campaigns` table (runbook Stage 1.3).
- Sign off `TODO(david)` and `TODO(tan)` items in [`OPEN_QUESTIONS.md`](../simulations/docs/OPEN_QUESTIONS.md).
- Run `pilot_50` across TACC Vista (`tacc-driver`) and GCP Cloud Batch Spot L4 (`gcp-gx-babysitter`), verifying `<2 KB` payloads, `<= 1.0` LLM calls/failure, and that harvested runs carry manifest v2 provenance with a `dataset_eligible` verdict.

## Phase 2 (Months 2–4): Production Campaign & KR1 Surrogates
- Ingest projected 3D equilibria from `ToBlick/mrx` into `campaigns/campaign_2000/equilibria/`.
- Scale GENE and GX simulation campaigns across TACC (Vista / LoneStar6) and GCP Cloud Batch GPUs (Spot L4 by default; Spot A100 for high-resolution runs once `allow_a100_fallback` is enabled), harvesting completed runs into the dataset (`equilibria/` + `runs/`).
- Train KR1 multi-fidelity operator/GP surrogates using `kr1_surrogate.split_by_equilibrium` (zero equilibrium data leakage).

## Phase 3 (Months 5–6): KR3 Coupled High-$\beta$ Optimization & Open-Source Release
- Couple KR1 surrogates inside the Augmented Lagrangian outer-loop optimizer (`kr1_surrogate.AugmentedLagrangianState`).
- Build the KR3 active-learning loop (deferred from Phase 1) on top of `kr3_design.enqueue_validation_run` to automatically validate high-uncertainty candidate stellarator designs (`\beta > 3%`) in GX/GENE (`<5%` relative $L_2$ error) and publish the final dataset to Zenodo.
