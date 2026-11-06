# System Architecture & Updated Sequence Diagrams (`docs/architecture.md`)

## 1. End-to-End Platform Architecture

```mermaid
flowchart TD
    subgraph EQ["Upstream 3D MHD Equilibria"]
        MRX["ToBlick/mrx (3D Broken-Surface Equilibria projected to nested surfaces)"]
    end

    subgraph GCP_CTRL["GCP Control Plane (Cloud Run + BigQuery)"]
        STEER["aurora-steer (Cloud Run + Vertex AI Gemini + OKF Skills)"]
        BQ["BigQuery sim_babysitter_telemetry (8 SQL Views)"]
        DASH["Private Dashboard (Cloud Run + IAP, prod rows only)"]
        STEER -->|"insertAll (+ audit log sink)"| BQ
        BQ --> DASH
    end

    subgraph TACC["Backend 1: TACC Vista / LoneStar6 (Zero AI SDKs)"]
        TD["tacc-driver + SimulationWatchdog (Login Node)"]
        SLURM["Slurm Compute Nodes (GENE Fortran + TACC GX)"]
        TD <-->|"sbatch / squeue / scancel"| SLURM
    end

    subgraph BATCH["Backend 2: GCP Cloud Batch (Serverless GPUs)"]
        GB["gcp-gx-babysitter (Cloud Run v2 Job / 2m Scheduler)"]
        CB["Cloud Batch Containerized GX (Spot L4 default; A100 / on-demand only by opt-in)"]
        GB <-->|"submit / query / cancel"| CB
    end

    subgraph DATA["Gyrokinetic Training Dataset (TACC $WORK & GCS Bucket)"]
        GCS_EQ["campaigns/<id>/equilibria/<equilibrium_id>/"]
        GCS_RUNS["campaigns/<id>/runs/<run_id>/{input, fluxes.json, checksums.sha256, run_manifest.json}"]
    end

    subgraph ML_OPT["KR1 & KR3: Surrogate Training & Coupled Optimization"]
        KR1["KR1 Leak-Free Loader + Multi-Fidelity Surrogates"]
        KR3["KR3 High-Beta (>3%) Stellarator Design + Active-Learning Queue"]
        KR1 --> KR3
    end

    MRX -->|"Projected equilibria"| GCS_EQ
    GCS_EQ -->|"FUSE Mount"| CB
    GCS_EQ -->|"Equilibrium inputs"| SLURM
    TD -->|"Outbound HTTPS (<2 KB)"| STEER
    GB -->|"HTTPS + Bearer (<2 KB)"| STEER
    TD -->|"Harvest completed runs"| GCS_RUNS
    CB <-->|"GCS FUSE .restart.nc"| GCS_RUNS
    GB -->|"Direct FUSE Harvest"| GCS_RUNS
    GCS_RUNS -->|"Grouped by equilibrium_id"| KR1
    KR3 -->|"High-uncertainty validation runs"| GB
    KR3 -->|"High-fidelity GENE validation"| TD
```

---

## 2. Dual-Backend Babysitting Sequence Diagram (TACC Slurm **and** GCP Cloud Batch)

Yes — **automated babysitting works identically on both TACC Slurm and GCP Cloud Batch** via the shared `SchedulerBackend` protocol (`hpc_core/protocol.py`) and `SimulationWatchdog` (`tacc_exec/watchdog.py`):

```mermaid
sequenceDiagram
    autonumber
    participant GCS as "Dataset & Queue ($WORK / GCS Bucket)"
    participant WD as "Watchdog (tacc-driver OR gcp-gx-babysitter)"
    participant EXEC as "Scheduler (TACC Slurm OR GCP Cloud Batch Spot L4/A100)"
    participant BRAIN as "aurora-steer (Cloud Run + Gemini)"
    participant KR1 as "KR1 Surrogate & Augmented Lagrangian"

    WD->>EXEC: Submit run (GENE/GX on TACC Slurm OR GX on GCP Cloud Batch)
    Note over EXEC,GCS: On Cloud Batch, GCS FUSE mounts /mnt/gk_bucket for .restart.nc

    alt Slurm TIMEOUT or Cloud Batch Spot PREEMPTED (0 LLM Calls)
        EXEC-->>WD: State = TIMEOUT or PREEMPTED
        WD->>WD: Set Restart.restart=true / read_checkpoint=.true.
        WD->>EXEC: Resubmit from checkpoint (0 LLM calls, 0 strikes)
    else Cloud Batch Spot L4 OUT_OF_MEMORY (0 LLM Calls)
        EXEC-->>WD: State = OUT_OF_MEMORY (exitCode 137 on 24 GB L4)
        WD->>EXEC: Upgrade L4 -> A100 only if allow_a100_fallback is set (capped), else escalate
    else Numerical Instability (STALL / TIMESTEP_COLLAPSE / SPECTRAL_PILEUP)
        EXEC-->>WD: Mid-run or terminal numerical failure detected
        WD->>EXEC: Early cancel if still RUNNING (saves GPU hours)
        WD->>BRAIN: POST /v1/skills/sim-babysitter/triage (<2 KB JSON + Idempotency-Key)
        BRAIN->>BRAIN: Load OKF runbooks + 3-strike check + Reject-Only Guardrails
        BRAIN-->>WD: ActionDirective (warm_restart or cold_restart)
        WD->>WD: Client-side guardrails (reject-only) + patch parameters / gx.in
        WD->>EXEC: Resubmit patched simulation (retry_count + 1)
    else Divergence (NaN) or Strike 3 (retry_count >= 2)
        WD->>GCS: Move to queue/needs_human/<run_id>/ + write ISSUE.md + Slack alert
    end

    EXEC-->>WD: State = COMPLETED (converged fluxes)
    WD->>GCS: Harvest fluxes.json + checksums + run_manifest.json v2 (dataset_eligible verdict, write-once)
    WD->>BRAIN: POST /v1/events (<1 KB lifecycle telemetry -> BigQuery)
    GCS->>KR1: Eligible runs only, train/val/test split grouped by equilibrium_id
```

---

## 3. Execution Environments Across TACC & GCP

| Layer | Where It Runs | Why |
| :--- | :--- | :--- |
| **GENE & TACC GX Simulation Campaigns** | **TACC Vista / LoneStar6 (`SlurmBackend`)** | Primary HPC allocation running multi-node GENE (Fortran) and GX simulations with a zero-AI login-node watchdog (`tacc-driver`). |
| **Cloud GX Simulation Campaigns** | **GCP Cloud Batch (`CloudBatchBackend`)** | Serverless GPU execution (**Spot L4** by default; Spot A100 and on-demand capacity only when the operator enables `allow_a100_fallback` / `allow_on_demand`), mounting the GCS dataset bucket via FUSE. |
| **`aurora-steer` + Telemetry Dashboard + `gcp-gx-babysitter`** | **GCP Cloud Run (Services & Job)** | Scale-to-zero HTTPS triage endpoint (single instance; `MODEL_ID`, default `gemini-4-argon`), BigQuery telemetry, private IAP dashboard, and 2-minute Cloud Scheduler watchdog ticks. |
