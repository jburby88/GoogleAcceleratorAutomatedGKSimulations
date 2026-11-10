# Operational Runbooks (`RUNBOOKS.md`)

## 1. Running on TACC Vista / Lonestar6 (Backend 1 — Slurm)

```bash
# On ONE specific TACC login node (see "Crontab on TACC" below), e.g.:
ssh login1.vista.tacc.utexas.edu
make install-tacc
cp workstreams/kr2-gk-database/hpc/config.example.json workstreams/kr2-gk-database/hpc/config.json
export AURORA_STEER_BEARER_TOKEN="<secret-token>"   # never commit it or echo it into logs

# Verify config + outbound HTTPS connectivity (<2 KB) to Cloud Run:
.venv-tacc/bin/tacc-preflight --config workstreams/kr2-gk-database/hpc/config.json

# Run one flock-guarded babysitter tick by hand (or install the crontab keepalive below):
.venv-tacc/bin/tacc-driver --config workstreams/kr2-gk-database/hpc/config.json --once
```

### Crontab on TACC (pin one login node)

TACC login hostnames (`vista.tacc.utexas.edu`, `ls6.tacc.utexas.edu`) are round-robin DNS over several physical login nodes, each login node keeps its **own** crontab spool, and Slurm `scrontab` is not offered to users ([Vista user guide](https://docs.tacc.utexas.edu/hpc/vista/), [Lonestar6 user guide](https://docs.tacc.utexas.edu/hpc/lonestar6/)). Therefore:

1. Always SSH to a named node (`ssh login1.vista.tacc.utexas.edu`) to install or inspect the crontab with `crontab -e` / `crontab -l`. A crontab edited after a plain `ssh vista.tacc.utexas.edu` may land on a different node than the one you inspect later.
2. Install the keepalive with the node pinned:
   ```
   */5 * * * * TACC_EXPECTED_LOGIN_HOST=login1 /abs/path/to/workstreams/kr2-gk-database/hpc/scripts/ensure_running_cron.sh
   ```
   `ensure_running_cron.sh` exits quietly on any other node, wraps the tick in `flock -n` on `hpc/state/cron.lock` (an overlapping tick exits immediately), and `tacc-driver` holds its own lock, so a crontab accidentally copied to a second node cannot start a second driver.
3. If you move the crontab to another node, remove it from the old node first (`ssh login1... crontab -r`).

## 2. Running Containerized GX on GCP Cloud Batch (Backend 2 — Spot L4 / A100)

Babysitting works the same way for GX runs on GCP Cloud Batch:
- Drop GX run folders (`gx.in` + `run_meta.json` referencing `equilibrium_id` in `gs://<bucket>/campaigns/<id>/equilibria/`) into `gs://<bucket>/queue/pending/<run_id>/`.
- Either let the automated Cloud Scheduler -> `gcp-gx-babysitter` Cloud Run v2 Job tick every 2 minutes, or run manually:
  ```bash
  gcp-gx-babysitter --config workstreams/kr2-gk-database/gcp/config.batch.json --once
  ```
- **What happens automatically on GCP Cloud Batch:**
  1. Standard GX grids (`<= 20 GB` VRAM) submit to **Spot NVIDIA L4** (`g2-standard-8`) with GCS FUSE mounted at `/mnt/gk_bucket`.
  2. If Spot L4 is preempted (`PREEMPTED`), `SimulationWatchdog` sets `Restart.restart = true` in `gx.in` and resubmits locally from `.restart.nc` with **0 LLM calls**.
  3. If an L4 job hits `OUT_OF_MEMORY` (or if `aurora-steer` refines `nx`/`ny`/`nz` past 20 GB on `SPECTRAL_PILEUP`), the run automatically upgrades to **Spot A100** (`a2-highgpu-1g`, 40 GB).
  4. On `COMPLETED`, `harvest_completed_run_to_dataset` writes `fluxes.json` + `run_manifest.json` (the manifest is written last and carries the dataset-eligibility verdict) into the campaign layout on GCS for KR1/KR3 surrogate training.

## 3. Resolving Escalated Runs (`queue/needs_human/<run_id>/ISSUE.md`)

```bash
# 1. Read ISSUE.md and inspect solver logs / parameters in queue/needs_human/<run_id>/
# 2. Edit parameters (GENE) or gx.in (GX), then re-queue:
tacc-resolve --config config.json --run-id <run_id> --minutes 20 --note "Lowered initial dt for high-beta KBM"
```

Why a run was escalated is in `ISSUE.md` ("Escalation Reason") and in the aurora-steer logs:
- `3-strike circuit breaker` / `Per-run LLM budget exhausted`: automated retries are used up.
- `Guardrail rejected LLM directive: ...`: the model proposed something the runbook or the allowlist forbids (wrong action for the failure type, a non-allowlisted or out-of-bounds parameter, a grid or dissipation change on a warm restart, or a patch that changes nothing). The proposed directive is in the aurora-steer Cloud Logging entry `Guardrail rejected LLM directive`.
- `LLM steering error (...)` / `LLM steering timed out`: Vertex AI was unavailable or returned malformed output. Check the aurora-steer `ERROR` logs (same `request_id`); a burst of these usually means a quota, IAM or `MODEL_ID` problem, not a physics problem.

## 4. aurora-steer operations (Cloud Run)

- `GET /healthz` returns HTTP 503 (`"status": "degraded"`) when the OKF skill or the model configuration is invalid; the service also refuses to start in that case, so a bad deploy never takes traffic.
- Every `/triage` response carries `X-Aurora-Decision` (`llm`, `cached`, `circuit_breaker`, `budget_exhausted`, `guardrail_rejected`, `llm_error`, `llm_timeout`) and `X-Request-Id`; search Cloud Logging for the request id to see the full decision trail.
- `GET /v1/events/summary` reads only the serving instance's local DuckDB store (debug aid). Campaign KPIs come from the BigQuery views / dashboard.
