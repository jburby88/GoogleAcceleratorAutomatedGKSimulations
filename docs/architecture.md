# System Architecture & Sequence Diagrams (`docs/architecture.md`)

## 1. End-to-End Platform Architecture

![UT Austin × Google.org AI for Science Platform Architecture](assets/architecture.svg)

<details>
<summary>Mermaid flowchart view</summary>

```mermaid
flowchart TD
    subgraph EQ["Upstream 3D MHD Equilibria"]
        MRX["ToBlick/mrx (JAX 3D Broken-Surface Equilibria projected to nested surfaces)"]
    end

    subgraph GCP_CTRL["GCP Control Plane (Cloud Run + BigQuery)"]
        STEER["aurora-steer (Cloud Run + Gemini Enterprise Agent Platform / gemini-4-argon + OKF Skills)"]
        BQ["BigQuery sim_babysitter_telemetry (8 SQL Views)"]
        DASH["Private Dashboard (Cloud Run + IAP, prod rows only)"]
        STEER -->|"insertAll (+ audit log sink)"| BQ
        BQ --> DASH
    end

    subgraph TACC["Backend 1: TACC Vista / LoneStar6 (Slurm)"]
        TD["tacc-driver + SimulationWatchdog (Login Node)"]
        SLURM["Slurm Compute Nodes (GENE Fortran + TACC GX)"]
        TD <-->|"sbatch / squeue / scancel"| SLURM
    end

    subgraph BATCH["Backend 2: GCP Cloud Batch & GKE (Serverless GPUs & TPUs)"]
        GB["gcp-gx-babysitter (Cloud Run v2 Job / 2m Scheduler)"]
        CB_GPU["C++ GX GPU Container (Spot L4 default; opt-in A100 / on-demand)"]
        CB_TPU["GX-JAX TPU Container (Spot TPU v6e / v5p; opt-in on-demand)"]
        GB <-->|"accelerator_family=gpu"| CB_GPU
        GB <-->|"accelerator_family=tpu"| CB_TPU
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
    GCS_EQ -->|"FUSE Mount"| CB_GPU
    GCS_EQ -->|"FUSE Mount"| CB_TPU
    GCS_EQ -->|"Equilibrium inputs"| SLURM
    TD -->|"Outbound HTTPS (<2 KB)"| STEER
    GB -->|"HTTPS + Bearer (<2 KB)"| STEER
    TD -->|"Harvest completed runs"| GCS_RUNS
    CB_GPU <-->|"GCS FUSE .restart.nc"| GCS_RUNS
    CB_TPU <-->|"GCS FUSE .restart.nc"| GCS_RUNS
    GB -->|"Direct FUSE Harvest"| GCS_RUNS
    GCS_RUNS -->|"Grouped by equilibrium_id"| KR1
    KR3 -->|"High-uncertainty validation runs"| GB
    KR3 -->|"High-fidelity GENE validation"| TD
```

</details>

---

## 2. Dual-Backend Babysitting Sequence Diagram (TACC Slurm **and** GCP GPU/TPU)

Automated babysitting works identically across **TACC Slurm** and **GCP Cloud Batch / GKE (GPU or TPU)** via the shared `SchedulerBackend` protocol (`hpc_core/protocol.py`), `SimulationWatchdog` (`tacc_exec/watchdog.py`), and the `gx-task` container wrapper (which invokes either C++ `gx` or the `gx-jax` CLI):

```mermaid
sequenceDiagram
    autonumber
    participant GCS as "Dataset & Queue ($WORK / GCS Bucket)"
    participant WD as "Watchdog (tacc-driver OR gcp-gx-babysitter)"
    participant EXEC as "Scheduler (TACC Slurm OR GCP Batch/GKE GPU & TPU)"
    participant BRAIN as "aurora-steer (Cloud Run + gemini-4-argon)"
    participant KR1 as "KR1 Surrogate & Augmented Lagrangian"

    WD->>EXEC: Submit run (GENE/GX on TACC OR C++ GX GPU / GX-JAX TPU on GCP)
    Note over EXEC,GCS: On GCP, gx-task mounts GCS FUSE (/mnt/gk_bucket) for .out.nc & .restart.nc

    alt Slurm TIMEOUT or GCP Spot PREEMPTED (0 LLM Calls)
        EXEC-->>WD: State = TIMEOUT or PREEMPTED
        WD->>WD: Set Restart.restart=true / read_checkpoint=.true.
        WD->>EXEC: Resubmit from checkpoint (0 LLM calls, 0 strikes)
    else GCP OUT_OF_MEMORY (0 LLM Calls)
        EXEC-->>WD: State = OUT_OF_MEMORY (exitCode 137)
        WD->>EXEC: Step up memory tier (GPU L4 -> opt-in A100, or TPU v6e -> v5p), else escalate
    else Numerical Instability (STALL / TIMESTEP_COLLAPSE / SPECTRAL_PILEUP)
        EXEC-->>WD: Mid-run or terminal numerical failure detected
        WD->>EXEC: Early cancel if still RUNNING (saves GPU/TPU hours)
        WD->>BRAIN: POST /v1/skills/sim-babysitter/triage (<2 KB JSON + Idempotency-Key)
        BRAIN->>BRAIN: Load OKF runbooks + 3-strike check + allowlist validation
        BRAIN-->>WD: ActionDirective (warm_restart or cold_restart)
        WD->>WD: Client-side allowlist check + patch parameters / gx.in
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
| **GENE & TACC GX Simulation Campaigns** | **TACC Vista / LoneStar6 (`SlurmBackend`)** | Primary HPC allocation running multi-node GENE (Fortran) and C++ GX simulations with a lightweight login-node watchdog (`tacc-driver`). |
| **Cloud GX GPU Campaigns** | **GCP Cloud Batch / GKE (`accelerator_family="gpu"`)** | Serverless C++ GX execution (**Spot L4** by default; Spot A100 and on-demand capacity only when enabled via `allow_a100_fallback` / `allow_on_demand`), mounting the GCS dataset bucket via FUSE. |
| **Cloud GX-JAX TPU Campaigns** | **GCP Cloud Batch / GKE (`accelerator_family="tpu"`)** | JAX/XLA port of the GX stellarator flux-tube core (`simulations/gx-jax`) running on **Spot TPU `v6e` / `v5p`** (Cloud Batch or GKE Compute Classes) through the same `gx-task` wrapper, `.in` decks, and `.out.nc` / `.restart.nc` schema. |
| **`aurora-steer` + Telemetry Dashboard + `gcp-gx-babysitter`** | **GCP Cloud Run (Services & Job)** | Scale-to-zero HTTPS triage endpoint (single instance; `MODEL_ID`, default `gemini-4-argon`), BigQuery telemetry, private IAP dashboard, and 2-minute Cloud Scheduler watchdog ticks. |
