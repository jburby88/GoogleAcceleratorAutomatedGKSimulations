# `platform/infra/terraform` — Shared GCP Infrastructure

Provisions the GCP side of the UT Austin × Google.org platform. There is no persistent GKE cluster. Apply it with `make gcp-deploy`, or `make gcp-plan` / `make gcp-apply` for later changes: [`scripts/deploy_gcp_demo.sh`](../../../scripts/deploy_gcp_demo.sh) builds every `-var` from `.env`.

| Resource | What it is | Security / integrity |
|---|---|---|
| `aurora-steer` (Cloud Run v2) | FastAPI + Vertex AI Gemini triage and telemetry service. Env: `MODEL_ID` (default `gemini-4-argon`; `gemini-3.8-flash` is a lower-cost option), `GOOGLE_CLOUD_LOCATION=global`, `VERTEX_TIMEOUT_S`, `BQ_ENABLED=1`, `BQ_DATASET`, and `AURORA_STEER_BEARER_TOKEN` from Secret Manager | Invoker IAM check disabled, because TACC cannot mint Google ID tokens; the app requires the bearer token on every endpoint except `/healthz`. One instance (per-instance idempotency cache). Startup probe on `/healthz`. Runs as `aurora-steer-sa`: `aiplatform.user`, insert access on the `events` / `directives` / `escalations` tables only, secret accessor. |
| `aurora-steer-dashboard` (Cloud Run v2) | Streamlit dashboard over the BigQuery views (`DATA_MODE=bigquery`, `BQ_MAX_BYTES_BILLED`) | Never public: direct IAP (`enable_dashboard_iap`, default true) with `dashboard_authorized_members` as IAP accessors; with IAP off, those members are the only invokers (`gcloud run services proxy`). Runs as `aurora-dashboard-sa`: `bigquery.jobUser` and dataset `dataViewer`. |
| `<project>-gk-dataset` (GCS) | Campaign equilibria/runs, queue, state, Cloud Batch checkpoints, babysitter config | Uniform access, public access prevention enforced, soft delete (`dataset_soft_delete_retention_days`, default 30), `prevent_destroy`. Hierarchical namespace for atomic GCS FUSE renames (HNS buckets do not support Object Versioning or Bucket Lock). Optional create-only, prefix-scoped access for UW's ToBlick/mrx pipeline. |
| `sim_babysitter_telemetry` (BigQuery) | Canonical tables + 8 views from [`platform/telemetry/02_views.sql`](../../telemetry/02_views.sql) | Base tables have deletion protection; `events` / `directives` / `escalations` are partitioned by `ts_utc` and require a partition filter. The 4 aggregate KPI views count only `environment='prod' AND is_synthetic=FALSE`. |
| `sim_babysitter_audit_logs` (BigQuery) | Cloud Logging sink copy of aurora-steer's structured records | Separate dataset with partition expiration (`audit_log_retention_days`); the sink cannot touch the canonical tables. |
| Artifact Registry `aurora-platform` | Images for aurora-steer, the dashboard and gx-runner | Immutable tags. Images are deployed by `@sha256` digest. |
| Cloud Batch identities | `gx-batch-worker-sa`: objectUser on the bucket, AR reader, `logging.logWriter`, `batch.agentReporter`. `gcp-gx-babysitter-sa`: `batch.jobsEditor`, may act only as the worker SA, bucket objects, secret accessor. `gcp-gx-scheduler-sa`: may only start the babysitter job | The worker SA must always be passed explicitly (`worker_service_account_email`); the code rejects the default compute SA. |
| `gcp-gx-babysitter` (Cloud Run job + Cloud Scheduler, every 2 min) | Runs `gcp-gx-babysitter --config /mnt/gk_bucket/config/batch_babysitter.json --once` from the gx-runner image (`gx_runner_image`, owned by Terraform) | 900 s timeout; overlapping runs are excluded by the driver's lease; GCS volume mounted as uid/gid 10001. |
| GitHub WIF (`enable_github_wif`) | Pool, provider, `github-deployer-sa` | Accepts tokens only from `github_repository`, ref `refs/heads/<github_deploy_branch>`, environment `<github_deploy_environment>`. The deployer may push images and roll the two services' images only. |

## Ownership of service configuration

Terraform owns every Cloud Run setting: identity, env, secrets, scaling, probes, IAM. The two **services** ignore image drift (`lifecycle.ignore_changes`): new images are rolled by digest with `scripts/deploy_gcp_demo.sh release ...`, from the operator workstation or the manual `deploy-cloudrun` workflow. So CI needs no Terraform permissions, and a later `terraform apply` never reverts a release. The babysitter **job** image is set by Terraform (`gx_runner_image`).

## State, providers, validation

- Remote state: `backend "gcs" {}` with partial config. `make gcp-init` creates `<project>-tfstate` (versioned, uniform access, public access prevention) and runs `terraform init -backend-config=bucket=... -backend-config=prefix=platform/infra`.
- Providers are pinned to `~> 6.30` and Terraform to `>= 1.6, < 2`. Commit the generated `.terraform.lock.hcl`.
- Every input is validated: project/region format; images must be pinned by digest or tag, never `:latest`; no public principals; retention ranges; `vertex_timeout_s` < 100 s.
- CI (`make lint-infra`) runs `terraform fmt -check`, `terraform init -backend=false` and `terraform validate`.

## Secret hygiene

Terraform creates only the empty `aurora-steer-bearer-token` secret, so the token never enters tfstate or outputs. `make gcp-deploy` adds the first version straight from `openssl rand` into Secret Manager. It never prints the token and never writes it to disk. Read it with `gcloud secrets versions access latest --secret=aurora-steer-bearer-token`. Rotation: see the operations section of [`docs/RUNBOOK_DEMO_AND_TACC_SMOKE_TEST.md`](../../../docs/RUNBOOK_DEMO_AND_TACC_SMOKE_TEST.md).
