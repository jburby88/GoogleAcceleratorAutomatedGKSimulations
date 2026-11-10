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
│   ├── dashboard/         # Streamlit dashboard: BigQuery views, prod rows only (local DuckDB mode for inspection)
│   ├── hpc-core/          # HPC client: Slurm backend, state, HTTPS client, telemetry outbox, dataset manifest
│   ├── telemetry/         # BigQuery DDL (01_tables.sql) + 8 KPI/operational views (02_views.sql) + contract tests
│   └── infra/terraform/   # Cloud Run, IAP, BigQuery, GCS, Artifact Registry, Secret Manager, Cloud Batch IAM, WIF
├── simulations/
│   ├── skill/sim-babysitter/okf/  # OKF manifest + physics runbooks + parameter allowlist (edited by physics team)
│   ├── hpc/                       # Backend 1 (TACC Slurm): watchdog, GENE/GX patchers, cron driver
│   ├── gcp/                       # Backend 2 (GCP Cloud Batch / GKE): containerized GX
│   └── docs/                      # OPEN_QUESTIONS.md, METRICS.md, COST_ESTIMATE.md, RUNBOOKS.md
├── surrogates/            # Leak-free dataset loader, augmented Lagrangian optimization, active learning hooks
├── docs/                  # Architecture, OKR map, implementation plan, operator runbook
├── scripts/               # Boundary checks, GCP deploy tool, TACC smoke test (+ tests)
└── tests/e2e/             # Dual-backend end-to-end test
```

### Physics Runbooks & Parameter Tuning

All recovery strategies and allowed numerical knobs live in Markdown/YAML under [`simulations/skill/sim-babysitter/`](simulations/skill/sim-babysitter/SKILL.md). This lets the physics team tweak heuristics and parameter bounds via git without touching the Python driver:

- **[`okf/manifest.yaml`](simulations/skill/sim-babysitter/okf/manifest.yaml)**: Maps each failure mode (`STALL`, `TIMESTEP_COLLAPSE`, `SPECTRAL_PILEUP`, `DIVERGENCE`, `CRASH`) to its triage runbook and valid recovery actions.
- **[`okf/concepts/allowed-parameters.md`](simulations/skill/sim-babysitter/okf/concepts/allowed-parameters.md)**: Allowlist for GENE/GX parameters (safe bounds, grid limits, and whether a checkpoint restart is permitted).
- **Runbooks** (`runbook-stalled.md`, `runbook-spectra.md`, `runbook-divergence.md`, `escalation-policy.md`): Step-by-step diagnostic logic. Open physics questions for Prof. David Hatch's team are tracked in [`OPEN_QUESTIONS.md`](simulations/docs/OPEN_QUESTIONS.md).

---

## Key Design & Safety Principles

1. **Lightweight HPC Footprint**: The TACC client (`platform/hpc-core/` and `simulations/hpc/`) has zero heavy cloud or AI dependencies. It runs on standard Python with `requests`, `f90nml`, and `tomlkit`, keeping login node installations clean and trouble-free.
2. **Compact Telemetry**: Instead of streaming heavy simulation files, executors send small diagnostic summaries (<2 KB of scalar metrics and recent tail logs) over HTTPS. Large NetCDF/HDF5 simulation files stay in local storage (`$WORK` or GCS).
3. **Strict Parameter Boundaries**: Automated triage can only modify allowlisted numerical knobs (`nx`, `ny`, `nz`, `dt_max`, `cfl_fraction`, `hyperdiffusion_coeff`). It never alters physics profiles, equilibrium geometry, or driving gradients. Any proposed modification outside safe bounds is safely rejected.
4. **Pragmatic Automation**: Routine operational interruptions (like Slurm timeouts or Spot preemptions) resume automatically from checkpoints without calling the LLM. If a simulation fails repeatedly on the same issue, it escalates to a human after two tries.
5. **Reproducible Dataset Provenance**: Every finished run writes an immutable manifest recording git hashes, equilibrium sources, parameter modifications, and convergence criteria, guaranteeing clean provenance for downstream ML surrogates.

---

## Security & Deployment

- **TACC to Cloud Run Authentication**: TACC communicates with `aurora-steer` over HTTPS using a bearer token managed in Secret Manager, avoiding the need for cloud credentials on HPC nodes.
- **Private Analytics Dashboard**: The Streamlit dashboard sits behind Cloud Run IAP for authorized team members, querying BigQuery production telemetry directly.
- **Least-Privilege Service Accounts**: Distinct identities for Batch workers, job schedulers, and CI deployment via Workload Identity Federation.
- **Protected Storage**: Simulation datasets are stored in versioned buckets with soft-delete enabled to prevent accidental data loss.

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
make install-tacc                         # .venv-tacc: lightweight TACC runtime
cp simulations/hpc/config.example.json "$WORK/gk-campaigns/config.json"   # keep dataset/state on $WORK, not $SCRATCH
export AURORA_STEER_BEARER_TOKEN="<token>"
.venv-tacc/bin/tacc-preflight --config "$WORK/gk-campaigns/config.json"
.venv-tacc/bin/tacc-driver --config "$WORK/gk-campaigns/config.json" --once
```

For unattended runs from cron, pinned to one login node with a lock, see [`simulations/hpc/README.md`](simulations/hpc/README.md).

### 4. Containerized GX on GCP Cloud Batch

After `make gcp-deploy`, Cloud Scheduler starts the `gcp-gx-babysitter` Cloud Run job every 2 minutes. The job processes runs queued under `gs://<project>-gk-dataset/queue/pending/`. GPU jobs need GPU quota. `GCP_BATCH_CPU_ONLY=1` is a CPU-only smoke-test mode that runs as `environment=sim`, so it never enters KPIs or the dataset.

---

## Development & CI

```bash
make install        # .venv with every package and its dev extras
make install-tacc   # .venv-tacc + dependency check
make lint           # gate + ruff check + ruff format --check
make lint-infra     # terraform fmt/validate, shellcheck, hadolint, actionlint (skips tools you lack)
make test           # all unit, contract and E2E tests, in a hermetic environment
make coverage       # make test with coverage
```

CI runs linting and hermetic tests across Python 3.11–3.13 ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Deployments can be triggered via GitHub Actions ([`.github/workflows/deploy-cloudrun.yml`](.github/workflows/deploy-cloudrun.yml)) or locally via `make gcp-deploy`.

See also [`docs/architecture.md`](docs/architecture.md), [`docs/OKR_MAP.md`](docs/OKR_MAP.md), [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md), and [`simulations/docs/OPEN_QUESTIONS.md`](simulations/docs/OPEN_QUESTIONS.md).
