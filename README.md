# UT Austin × Google.org AI for Science: Transport-Consistent Fusion Design

Open-source platform (Apache-2.0) developed under the UT Austin / Google.org AI for Science partnership to accelerate transport-consistent magnetic fusion plasma optimization via automated gyrokinetic simulations and machine-learning surrogates.

---

## Architecture Overview

```text
                   ┌──────────────────────────────────────────┐
                   │   aurora-steer (Cloud Run + Vertex AI)   │
                   │  OKF runbooks + server-side guardrails   │
                   │  BigQuery telemetry  →  private dashboard │
                   └────────────▲────────────────▲────────────┘
     Outbound HTTPS + Bearer    │                │  HTTPS + Bearer
           (<2 KB JSON)         │                │  (<2 KB JSON)
    ┌───────────────────────────┴──┐          ┌──┴───────────────────────────┐
    │ Backend 1: TACC (tacc-driver)│          │ Backend 2: GCP (Cloud Batch) │
    │ • GENE (Fortran) + TACC GX   │          │ • Containerized GX on GPUs   │
    │ • Slurm (sbatch/squeue/sacct)│          │ • Spot L4 by default         │
    │ • Dataset on $WORK           │          │ • GCS FUSE checkpoints       │
    └──────────────┬───────────────┘          └──────────────┬───────────────┘
                   │  Dataset output                         │  Dataset output
                   └────────────────────┬────────────────────┘
                                        ▼
    ┌────────────────────────┐   ┌──────────────────────────────────────┐
    │ Upstream Equilibria:   │   │  Dataset (TACC $WORK / GCS bucket)   │
    │ ToBlick/mrx (3D MHD)   ├──►│  campaigns/<id>/equilibria/<eq_id>/  │
    │ (projected geometries) │   │  campaigns/<id>/runs/<run_id>/       │
    └────────────────────────┘   └──────────────────┬───────────────────┘
                                                    ▼
                                   Surrogate training & stellarator design
                                                (KR1 / KR3)
```

See [`docs/architecture.md`](docs/architecture.md) for the sequence diagrams.

---

## Repository Layout

```text
GoogleAcceleratorAutomatedGKSimulations/
├── platform/
│   ├── aurora-steer/      # Cloud Run FastAPI + Google ADK agent (Vertex AI Gemini); loads OKF skills by name
│   ├── dashboard/         # Streamlit KR2 dashboard: BigQuery views, prod rows only (local DuckDB mode for inspection)
│   ├── hpc-core/          # Dependency-free HPC client: Slurm backend, state, HTTPS client, telemetry outbox, dataset manifest
│   ├── telemetry/         # BigQuery DDL (01_tables.sql) + 8 KPI/operational views (02_views.sql) + contract tests
│   └── infra/terraform/   # Cloud Run, IAP, BigQuery, GCS, Artifact Registry, Secret Manager, Cloud Batch IAM, WIF
├── workstreams/
│   ├── kr1-optimizer-orchestration/   # KR1: leak-free dataset loader, augmented-Lagrangian outer loop
│   ├── kr2-gk-database/               # KR2: automated gyrokinetic database ("sim-babysitter")
│   │   ├── skill/sim-babysitter/okf/  #   OKF manifest + physics runbooks + parameter allowlist (edited by the physics team)
│   │   ├── hpc/                       #   Backend 1 (TACC Slurm): watchdog, GENE/GX patchers, cron driver
│   │   ├── gcp/                       #   Backend 2 (GCP Cloud Batch / GKE): containerized GX
│   │   └── docs/                      #   OPEN_QUESTIONS.md, METRICS.md, COST_ESTIMATE.md, RUNBOOKS.md
│   └── kr3-transport-consistent-design/  # KR3: validation-run enqueue helper (active learning is a Phase 3 item)
├── docs/                  # Architecture, OKR map, implementation plan, operator runbook
├── scripts/               # Zero-AI/boundary gate, GCP deploy/release tool, TACC smoke test (+ their tests)
└── tests/e2e/             # Dual-backend end-to-end test
```

### OKF Physics Runbooks & Parameter Guardrails (for physics-team review)

All troubleshooting rules and allowlisted solver parameters live in Markdown/YAML under [`workstreams/kr2-gk-database/skill/sim-babysitter/`](workstreams/kr2-gk-database/skill/sim-babysitter/SKILL.md), so physicists can change agent behavior through pull requests without touching Python:

- **[`okf/manifest.yaml`](workstreams/kr2-gk-database/skill/sim-babysitter/okf/manifest.yaml)**: maps each failure type (`STALL`, `TIMESTEP_COLLAPSE`, `SPECTRAL_PILEUP`, `DIVERGENCE`, `CRASH`) to its runbook and to the actions the server accepts for it.
- **[`okf/concepts/allowed-parameters.md`](workstreams/kr2-gk-database/skill/sim-babysitter/okf/concepts/allowed-parameters.md)**: the parameter allowlist (names, GENE/GX keys, `[min, max]` bounds, `warm_restart_allowed`). The TACC and GCP example configs mirror it; CI fails on any drift (`scripts/check_boundaries_and_deps.py`).
- **Runbooks** (`runbook-stalled.md`, `runbook-spectra.md`, `runbook-divergence.md`, `escalation-policy.md`): recovery procedures; open physics questions for Prof. David Hatch's team are tracked in [`OPEN_QUESTIONS.md`](workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md).

---

## Design Principles & Guardrails

1. **Dependency-free HPC client.** `platform/hpc-core/` and `workstreams/kr2-gk-database/hpc/` use only the Python standard library plus `f90nml`, `tomlkit` and `requests`: no AI, LLM or cloud SDK runs on TACC login nodes. CI enforces it with an AST import scan (including dynamic-import tricks), a `pyproject.toml` audit and an isolated `.venv-tacc` allowlist (`make install-tacc`). `scripts/tacc_smoke_test.py` is held to the standard library only.
2. **Compact telemetry.** Executors send `DiagnosticSummary` payloads under 2048 bytes (≤30 log lines plus scalar metrics). Oversized payloads are refused, never truncated. Raw HDF5/NetCDF output stays in storage.
3. **Numerical-only, reject-only guardrails.** Only the allowlisted numerical parameters (`nx`, `ny`, `nz`, `dt_max`, `cfl_fraction`, `hyperdiffusion_coeff`, which maps to GX `[Dissipation] D_hyper`) may change, never geometry or physics profiles. Out-of-bounds, non-finite or non-allowlisted values are rejected, never clamped, on both the server and the client. Warm restarts (from checkpoint) are allowed only for timestep parameters (`dt_max`, `cfl_fraction`). Every other change forces a cold restart.
4. **Bounded automation.** Duplicate deliveries hit an `Idempotency-Key` cache (no extra LLM call). `retry_count >= 2` escalates to a human without calling the LLM. A per-run LLM budget and rate limits cap spend. Slurm `TIMEOUT` and GCP Spot preemptions resume locally from checkpoint, with no LLM call. GCP never falls back to on-demand VMs or A100s unless the operator opts in (`allow_on_demand`, `allow_a100_fallback`).
5. **Fail closed.** If aurora-steer is unreachable or returns an invalid directive, the run is escalated, never resubmitted blindly. Cancellations by users or admins, jobs that can never start, and unknown states all escalate.
6. **Dataset provenance.** Each harvested run writes `runs/<run_id>/{input, fluxes.json, checksums.sha256, run_manifest.json}` (manifest schema 2.0). The manifest is written last, as the commit marker, and committed entries are never overwritten. It records checksums, code version, equilibrium origin, parameter patches, restart history, and a `dataset_eligible` verdict with reasons: converged, not escalated, not simulated, prod, provenance verified. The KR1 loader admits only eligible runs and splits train/val/test by `equilibrium_id`, so no geometry leaks across splits. The campaign layout extends the provenance of the Landreman et al. (2025) Zenodo GX stellarator dataset [10.5281/zenodo.14867777](https://zenodo.org/records/14867777), which itself is distributed as an HDF5 archive. Raw GX heat fluxes harvested in `fluxes.json` (`flux_normalization = "gyroBohm_divided_by_fsa_grad_x"`) match Zenodo's `Q_avgs_divided_by_FSA_grad_x` ($Q_{\text{gB}} / \langle |\nabla x|\rangle$), whereas Zenodo's `Q_avgs` multiplies by `<|grad x|>` (`fsa_grad_x`).

---

## Security Model

- **aurora-steer** is reachable over HTTPS from TACC, which cannot mint Google identity tokens. The Cloud Run invoker IAM check is therefore disabled, and every endpoint except `/healthz` requires a constant-time-checked bearer token. The token lives only in Secret Manager (comma-separated values allow rotation). The service runs as a dedicated service account with Vertex AI access and insert-only grants on three telemetry tables.
- **The dashboard is never public.** It sits behind direct Cloud Run IAP for `DASHBOARD_AUTHORIZED_MEMBERS`; with IAP off, only those members may invoke it, through `gcloud run services proxy`. It reads BigQuery views only, and only production rows.
- **Least-privilege identities**: a Batch worker SA (bucket objects, image pulls, logs, Batch agent reporting), a babysitter SA (Batch jobs, may act only as the worker SA), a scheduler SA (may only start the babysitter job), and a GitHub deployer SA (push images, roll the two services) trusted only from `main` in the `production` environment through Workload Identity Federation.
- **Dataset bucket**: uniform access, public access prevention enforced, soft delete (30 days by default), `prevent_destroy`. Terraform state lives in a versioned, private GCS bucket.
- **Supply chain**: GitHub Actions pinned to commit SHAs (Dependabot keeps them current), images built under immutable tags and deployed by `@sha256` digest, non-root containers.

---

## Getting Started

The full operator procedure, including token handling, CI deploy setup and GPU obtainability, is in **[`docs/RUNBOOK_DEMO_AND_TACC_SMOKE_TEST.md`](docs/RUNBOOK_DEMO_AND_TACC_SMOKE_TEST.md)**.

### 1. Deploy the GCP stack

```bash
make env-init            # creates .env (mode 600) from .env.example
# edit .env: GOOGLE_CLOUD_PROJECT, DASHBOARD_AUTHORIZED_MEMBERS, GX_COMMIT (and GCP_REGION if needed)
make install
make gcp-deploy          # idempotent; safe to re-run
```

`make gcp-deploy` runs [`scripts/deploy_gcp_demo.sh`](scripts/deploy_gcp_demo.sh), which:
1. creates the versioned Terraform state bucket and runs `terraform init`;
2. bootstraps APIs, Artifact Registry, the Secret Manager shell, GCS and BigQuery;
3. generates the bearer token directly into Secret Manager if none exists (never printed, never written to `.env`);
4. builds and pushes aurora-steer, the dashboard and gx-runner under an immutable git-SHA tag (`GX_COMMIT` pins the GX source);
5. applies the full Terraform stack with the images pinned by digest;
6. rolls the Cloud Run services to those digests;
7. uploads the GCP babysitter config.

Retrieve the token when a TACC operator needs it:

```bash
gcloud secrets versions access latest --secret=aurora-steer-bearer-token --project="$GOOGLE_CLOUD_PROJECT"
```

### 2. Verify outbound HTTPS from a TACC login node (Vista / Lonestar6)

```bash
export AURORA_STEER_URL="https://aurora-steer-xxxx-uc.a.run.app"
export AURORA_STEER_BEARER_TOKEN="<token from Secret Manager>"
python3 scripts/tacc_smoke_test.py      # Python >= 3.8 standard library only, no venv needed
```

### 3. Activate the TACC Slurm babysitter (`tacc-driver`) for GENE & GX

```bash
cd "$WORK" && git clone https://github.com/jburby88/GoogleAcceleratorAutomatedGKSimulations.git
cd GoogleAcceleratorAutomatedGKSimulations
make install-tacc                         # .venv-tacc: zero AI/cloud SDKs, allowlist verified
cp workstreams/kr2-gk-database/hpc/config.example.json "$WORK/gk-campaigns/config.json"   # keep dataset/state on $WORK, not $SCRATCH
export AURORA_STEER_BEARER_TOKEN="<token>"
.venv-tacc/bin/tacc-preflight --config "$WORK/gk-campaigns/config.json"
.venv-tacc/bin/tacc-driver --config "$WORK/gk-campaigns/config.json" --once
```

For unattended runs from cron, pinned to one login node with a lock, see [`workstreams/kr2-gk-database/hpc/README.md`](workstreams/kr2-gk-database/hpc/README.md).

### 4. Containerized GX on GCP Cloud Batch

After `make gcp-deploy`, Cloud Scheduler starts the `gcp-gx-babysitter` Cloud Run job every 2 minutes. The job processes runs queued under `gs://<project>-gk-dataset/queue/pending/`. GPU jobs need GPU quota. `GCP_BATCH_CPU_ONLY=1` is a CPU-only smoke-test mode that runs as `environment=sim`, so it never enters KPIs or the dataset.

---

## Development & CI

```bash
make install        # .venv with every package and its dev extras
make install-tacc   # .venv-tacc + zero-AI allowlist gate
make lint           # gate + ruff check + ruff format --check
make lint-infra     # terraform fmt/validate, shellcheck, hadolint, actionlint (skips tools you lack)
make test           # all unit, contract and E2E tests, in a hermetic environment
make coverage       # make test with coverage
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs exactly these targets on Python 3.11, 3.12 and 3.13. Images ship on Python 3.12. `make test` never sees your `.env` or the deployed endpoints: the test environment strips URLs, tokens and BigQuery settings.

Deploying from GitHub: [`.github/workflows/deploy-cloudrun.yml`](.github/workflows/deploy-cloudrun.yml) is manual, runs from `main` only in the protected `production` environment, authenticates through Workload Identity Federation, and only rolls new image digests. Terraform owns all service configuration.

---

## Licensing

- This repository is licensed under the [Apache License 2.0](LICENSE).
- **GENE is not open source.** It is distributed by the GENE development team under the GENE User Agreement (<https://genecode.org/>). This repository contains no GENE source code or binaries, and no container image built from it includes GENE. GENE runs only on TACC, from the operator's own licensed installation. The GCP backend runs GX only.
- **GX** is open source. `Dockerfile.gx-runner` builds it from its upstream repository (<https://bitbucket.org/gyrokinetics/gx>) at the pinned `GX_COMMIT`. GX is distributed under its own license.

See also [`docs/architecture.md`](docs/architecture.md), [`docs/OKR_MAP.md`](docs/OKR_MAP.md), [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) and [`workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md`](workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md).
