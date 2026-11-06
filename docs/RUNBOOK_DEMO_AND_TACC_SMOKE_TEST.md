# Operator Runbook: GCP Deployment, TACC Login-Node Smoke Test & GPU Obtainability

This runbook covers:
1. **Stage 1**: deploy and verify the GCP stack: aurora-steer (Cloud Run + Vertex AI Gemini), BigQuery telemetry, GCS dataset bucket, containerized GX, private dashboard.
2. **Stage 2**: verify outbound HTTPS, authentication, telemetry ingestion and a live `/triage` round trip **from a TACC Vista or Lonestar6 login node**.
3. **Stage 3**: GPU obtainability on GCP (Cloud Batch region rotation or GKE ComputeClasses).
4. **Stage 4**: activate the zero-AI TACC Slurm babysitter (`tacc-driver`).
5. Operations: token rotation, CI deploys, rollback.

Prerequisites on the operator workstation: `gcloud` (authenticated as a project owner for the first deploy), `terraform >= 1.6`, `docker`, `python3`, `openssl`.

---

## Stage 1: Deploy & verify the GCP stack

### 1. Configure `.env`

```bash
make env-init        # copies .env.example to .env with mode 600
```

Set at least:

```dotenv
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GCP_REGION=us-central1
DASHBOARD_AUTHORIZED_MEMBERS=group:<team>@<domain>     # comma-separated IAM principals
GX_COMMIT=<40-hex GX commit, see workstreams/kr2-gk-database/gcp/README.md>
# Projects without an OAuth consent screen: set ENABLE_DASHBOARD_IAP=false. The dashboard stays
# private; the members above open it with `gcloud run services proxy`.
ENABLE_DASHBOARD_IAP=true
# Projects with 0 GPU quota: GCP_BATCH_CPU_ONLY=1 runs the babysitter as environment=sim
# (CPU-only GX output never enters KPIs or the dataset).
GCP_BATCH_CPU_ONLY=0
```

`MODEL_ID` defaults to `gemini-4-argon`, a GA Vertex AI model; set `MODEL_ID=gemini-3.8-flash` for a lower-cost option. aurora-steer validates the ID at startup, and its `/healthz` startup probe keeps a broken revision from taking traffic.

### 2. Deploy (`~10–40 min`; the first gx-runner build compiles GX)

```bash
make install
make gcp-deploy
```

`scripts/deploy_gcp_demo.sh` is idempotent: re-running it reuses existing image tags, never replaces the token, and leaves services already on the target digest untouched. At the end it prints:
- the **aurora-steer URL** (`https://aurora-steer-<hash>-uc.a.run.app`), also written to `AURORA_STEER_URL` in `.env`;
- the **dashboard URL**: open it in a browser with IAP, or run `gcloud run services proxy aurora-steer-dashboard --region "$GCP_REGION"` without IAP;
- the three image digests.

The bearer token is generated directly into Secret Manager on the first deploy. It is not written to `.env`. Fetch it when needed:

```bash
gcloud secrets versions access latest --secret=aurora-steer-bearer-token --project="$GOOGLE_CLOUD_PROJECT"
```

Infrastructure-only changes later: `make gcp-plan` / `make gcp-apply` run Terraform with the currently deployed images.

### 3. Seed the campaign target (needed for completion % and projections)

`v_kpis_campaign` reads the target from the `campaigns` table, newest row per campaign. Without a row, the target falls back to the number of runs seen.

```bash
bq query --use_legacy_sql=false --project_id="$GOOGLE_CLOUD_PROJECT" "
INSERT INTO sim_babysitter_telemetry.campaigns
  (campaign_id, workstream, target_runs, start_date, deadline_date, notes, environment, is_synthetic, updated_at_utc)
VALUES ('pilot_50', 'kr2-gk-database', 50, CURRENT_DATE(), DATE_ADD(CURRENT_DATE(), INTERVAL 21 DAY),
        'Phase 1 pilot', 'prod', FALSE, CURRENT_TIMESTAMP())"
```

### 4. The GCP babysitter

Cloud Scheduler starts the `gcp-gx-babysitter` Cloud Run job every 2 minutes. The job reads `gs://<project>-gk-dataset/config/batch_babysitter.json`, which the deploy script generates from Terraform outputs: worker service account, bucket, aurora-steer URL, GX image digest. It processes runs queued under `gs://<project>-gk-dataset/queue/pending/<run_id>/`. Overlapping executions are excluded by the driver's lease in the state directory.

---

## Stage 2: TACC login node → Cloud Run smoke test

Run this before installing the Slurm babysitter. `scripts/tacc_smoke_test.py` uses only the Python standard library and runs on the system `python3` (3.8 or newer): no `pip install` and no virtual environment on TACC.

```bash
# On the TACC login node:
export AURORA_STEER_URL="https://aurora-steer-xxxx-uc.a.run.app"
export AURORA_STEER_BEARER_TOKEN="<token from Secret Manager>"
python3 scripts/tacc_smoke_test.py
```

From the workstation, `make tacc-smoke-test` does the same. It reads the token from Secret Manager at run time when `AURORA_STEER_BEARER_TOKEN` is not set, and never puts the token on a command line.

The test checks, in order:
1. `GET /health` (note: Cloud Run's Google Front End reserves `/healthz` on external `*.run.app` URLs, so external probes use `/health` or `/v1/healthz` while the internal Cloud Run startup probe uses `/healthz`);
2. an authenticated `POST /v1/events`;
3. a `POST` triage call that must come back from the live LLM path. The response header `X-Aurora-Decision` must be `llm`; an escalation from a dead Vertex AI path fails the test.

Smoke-test events use `environment = preflight`. They are stored flagged synthetic and are **excluded from every KPI view and from the dashboard**, which shows production rows only. To confirm ingestion, query the raw table:

```bash
bq query --use_legacy_sql=false --project_id="$GOOGLE_CLOUD_PROJECT" "
SELECT run_id, event_type, environment, ts_utc FROM sim_babysitter_telemetry.events
WHERE ts_utc > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR) AND environment = 'preflight'
ORDER BY ts_utc DESC LIMIT 20"
```

Plain `curl` check before copying the repository (the token stays in an environment variable):

```bash
curl -sS -f "${AURORA_STEER_URL}/health" && echo " <- /health OK"
curl -sS -f -X POST "${AURORA_STEER_URL}/v1/events" \
  -H "Authorization: Bearer ${AURORA_STEER_BEARER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"schema_version\": \"1.0\", \"events\": [{
        \"event_id\": \"evt-tacc-curl-$(date +%s)\", \"schema_version\": \"1.0\",
        \"workstream\": \"kr2-gk-database\", \"backend\": \"tacc_slurm\",
        \"run_id\": \"tacc_curl_$(hostname -s)\", \"campaign_id\": \"pilot_50\",
        \"solver\": \"GX\", \"fidelity\": \"low\", \"event_type\": \"running\",
        \"ts_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"retry_count\": 0,
        \"environment\": \"preflight\", \"payload\": {\"smoke_test\": true}}]}" \
  && echo " <- /v1/events OK"
```

---

## Stage 3: GPU obtainability on GCP

The babysitter config (`config/batch_babysitter.json`) controls spend. By default it uses **Spot** capacity only. On-demand VMs (`allow_on_demand`) and the A100 fallback (`allow_a100_fallback`, about 5× the L4 Spot price) are off unless an operator sets them, and both escalation paths are capped (see `workstreams/kr2-gk-database/gcp/README.md`).

### Mode A: multi-region Cloud Batch rotation (`CloudBatchBackend`)

1. In `.env`, set `GCP_BATCH_CANDIDATE_REGIONS=us-central1,us-east4,...`. The dataset bucket and Artifact Registry are regional, so jobs in other regions pay cross-region egress for checkpoints and image pulls.
2. Re-run `make gcp-deploy` to regenerate the babysitter config. On a stockout (`ZONE_RESOURCE_POOL_EXHAUSTED`) or a Spot preemption, the backend rotates regions and GPU tiers within the configured caps, without calling the LLM.

### Mode B: GKE ComputeClasses (`GKEComputeClassBackend`)

Terraform does not create a GKE cluster. With an existing GKE Autopilot or node-auto-provisioning cluster:

1. `kubectl apply -f workstreams/kr2-gk-database/gcp/k8s/compute-classes.yaml`.
2. Set `"backend": "gcp_k8s"` in the babysitter config. Jobs bind to `cloud.google.com/compute-class` and set `ttlSecondsAfterFinished`. On-demand classes are used only with `allow_on_demand`.

---

## Stage 4: Activate TACC Slurm babysitting (after Stage 2 passes)

```bash
cd "$WORK"
git clone https://github.com/jburby88/GoogleAcceleratorAutomatedGKSimulations.git
cd GoogleAcceleratorAutomatedGKSimulations
make install-tacc                        # .venv-tacc + zero-AI allowlist gate
mkdir -p "$WORK/gk-campaigns"
cp workstreams/kr2-gk-database/hpc/config.example.json "$WORK/gk-campaigns/config.json"
# Edit config.json: aurora_steer_url, queue/state/dataset dirs on $WORK ($SCRATCH is purged)
export AURORA_STEER_BEARER_TOKEN="<token>"
.venv-tacc/bin/tacc-preflight --config "$WORK/gk-campaigns/config.json"
.venv-tacc/bin/tacc-driver --config "$WORK/gk-campaigns/config.json" --once
```

For unattended runs from cron, use `hpc/scripts/ensure_running_cron.sh`. It pins the job to one login node (`TACC_EXPECTED_LOGIN_HOST`), takes a `flock`, and the driver adds a shared-filesystem lease. Keep the token out of the crontab line itself. See `workstreams/kr2-gk-database/hpc/README.md`.

---

## Operations

### Rotating the bearer token (zero downtime)

The server accepts a comma-separated list in `AURORA_STEER_BEARER_TOKEN`.
1. `gcloud secrets versions add aurora-steer-bearer-token --data-file=-`, entering `OLD,NEW` on stdin. Then roll a new revision so instances pick it up: `gcloud run services update aurora-steer --region "$GCP_REGION" --update-labels=token-rotated=$(date +%s)`.
2. Update the TACC token file and anything else using `OLD`. The babysitter job reads the secret on every execution.
3. Add a version containing only `NEW`, roll again, then disable the older versions.

### Deploying images from GitHub Actions

One-time setup:
1. Set `ENABLE_GITHUB_WIF=true` and `GITHUB_WIF_REPOSITORY=<owner>/<repo>` in `.env`, then run `make gcp-apply`.
2. Create a GitHub environment `production` with required reviewers.
3. Add the repository variables `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER` and `GCP_DEPLOYER_SA`, taking the last two from `terraform -chdir=platform/infra/terraform output`.

Then run **Deploy Cloud Run images (manual)** from `main`. WIF accepts tokens only from that repository, the `main` branch and the `production` environment. The deployer can only push images and roll the two services.

### Rollback

Roll a service back to a previous digest (Artifact Registry keeps every immutable tag):

```bash
gcloud run services update aurora-steer --region "$GCP_REGION" \
  --image="$GCP_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/aurora-platform/aurora-steer@sha256:<digest>"
```

### Data protection

- The dataset bucket has soft delete (30 days by default): restore deleted or overwritten objects with `gcloud storage restore`. Terraform refuses to destroy the bucket (`prevent_destroy`).
- BigQuery base tables have Terraform deletion protection. `events`, `directives` and `escalations` require a `ts_utc` filter in queries.
- Raw structured logs are copied to the `sim_babysitter_audit_logs` dataset, with partition expiration (365 days by default) for recovery.
