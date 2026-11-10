# KR2 Physics Tuning Guide & Decisions Tracker (`OPEN_QUESTIONS.md`)

This guide outlines the solver thresholds, grid bounds, and environment settings set up for automated gyrokinetic runs (GENE and GX). 

All settings are pre-configured with conservative, safe defaults so the system works out of the box. During our tuning session with Prof. David Hatch, Josh Burby, and Tan, we can calibrate these values to match your specific research conventions.

---

## 1. Physics & Solver Thresholds — `TODO(david)`

### 1. Parameter Allowlist & Safe Bounds
*File:* [`simulations/skill/sim-babysitter/okf/concepts/allowed-parameters.md`](../skill/sim-babysitter/okf/concepts/allowed-parameters.md)
- `TODO(david): confirm exact GENE & GX numerical parameter bounds for Vista / Cloud Batch campaigns`
- **Current Defaults:** `nx: [16, 512]`, `ny: [8, 256]`, `nz: [16, 256]`, `dt_max: [1.0e-6, 0.1]`, `cfl_fraction: [0.05, 0.95]`, `hyperdiffusion_coeff: [0.0, 2.0]`. Templates are also in place if you'd like to expose more parameters (`nv`/`nhermite`, `nw`/`nlaguerre`, `hyp_x`, `hyp_y`, `hyp_v`, `lx`).
- **Warm Restart Policy:** Currently, only `dt_max` and `cfl_fraction` can change on a warm restart from checkpoint (`warm_restart_allowed: true`). Any grid resolution or dissipation change forces a clean cold restart. Let us know if you want other knobs allowed for warm restarts.
- **Grid Semantics:** In GENE, `nx0`/`nky0` count Fourier modes, whereas in GX, `nx`/`ny` count real-space grid points before dealiasing. Should we maintain separate GENE and GX step rules rather than a shared range?
- **Hyperdiffusion Mapping:** GENE `&general hyp_z` is parallel ($z$) hyperdiffusion, while GX `[Dissipation] D_hyper` is perpendicular ($k_x, k_y$) dissipation (`src/linear.cu`). GX's parallel hyperdiffusion is `nu_hyper_z` (`hyperz = true`). Here are three straightforward options:

  | Option | Physics Meaning | GENE Mapping | GX Mapping (`gx.in`) | Note |
  | :--- | :--- | :--- | :--- | :--- |
  | **Option A** *(Current default)* | 1 stabilization knob mapped to each solver's standard dissipation term. | `general.hyp_z` | `Dissipation.D_hyper` + `hyper = true` | Ready now (no changes needed). |
  | **Option B** *(Recommended)* | Split into separate perpendicular and parallel knobs (`hyperdiffusion_perp` and `hyperdiffusion_par`). | `perp`: `hyp_x`<br>`par`: `hyp_z` | `perp`: `D_hyper` (`hyper = true`)<br>`par`: `nu_hyper_z` (`hyperz = true`) | Cleanest physical separation; quick update across configs. |
  | **Option C** | Keep 1 knob (`hyperdiffusion_coeff`), but map both solvers strictly to parallel-$z$. | `general.hyp_z` | `Dissipation.nu_hyper_z` + `hyperz = true` | Consistent parallel dissipation across both. |

- **Velocity-Space Hypercollisions (`hyp_v`):** GX provides Hermite/Laguerre hypercollision coefficients (`nu_hyper_m`, `nu_hyper_l`). If you want to steer `hyp_v`, we can map it to Hermite/Laguerre damping or keep it fixed in the base input decks for Phase 1.

### 2. Fidelity Tiers & Equilibrium Naming
*File:* [`platform/hpc-core/hpc_core/telemetry.py`](../../platform/hpc-core/hpc_core/telemetry.py)
- `TODO(david): confirm exact low vs high fidelity grid criteria and equilibrium/geometry naming`
- Confirm how external `ToBlick/mrx` equilibrium IDs (`mrx_eq_*`) should map into low-fidelity GX screening vs. high-fidelity GENE validation runs.

### 3. GPU VRAM Sizing for GX
*File:* [`simulations/gcp/gcp_exec/gpu_selector.py`](../gcp/gcp_exec/gpu_selector.py)
- `TODO(david): calibrate empirical cuFFT buffer multiplier against high-beta kinetic-electron GX runs`
- By default, runs estimated $\le 20.0$ GiB route to Spot NVIDIA L4 (24 GB), and larger runs route to Spot NVIDIA A100 (40 GB).

### 4. Baseline Metrics & Benchmarks
*Files:* [`platform/telemetry/01_tables.sql`](../../platform/telemetry/01_tables.sql), [`platform/telemetry/02_views.sql`](../../platform/telemetry/02_views.sql), [`platform/aurora-steer/app/events.py`](../../platform/aurora-steer/app/events.py), [`platform/dashboard/streamlit_app.py`](../../platform/dashboard/streamlit_app.py), [`simulations/docs/METRICS.md`](METRICS.md)
- `TODO(david): confirm baseline manual troubleshooting duration` (default assumption: `45.0` min/fix)
- `TODO(david): confirm manual walltime continuation overhead` (default: `10.0` min)
- `TODO(david): confirm Vista queue walltime defaults` (`4.0h` low / `16.0h` high for GENE)
- `TODO(david): confirm GX walltime defaults` (`2.0h` low / `8.0h` high for GX)
- `TODO(david): confirm GENE GPU count per fidelity` (`4` low / `16` high)
- `TODO(david): confirm GX GPU count per fidelity` (`1` low / `4` high)

### 5. Active Learning & Physics Defaults
*File:* [`surrogates/kr3_design/active_learning_queue.py`](../../surrogates/kr3_design/active_learning_queue.py)
- `TODO(david): confirm resolution nx/ny/ntheta mappings for high/low fidelity`
- `TODO(david): confirm nhermite=16, nlaguerre=8` and `nspecies = 2` (kinetic electrons) when `beta > 0`, else `1`
- `TODO(david): confirm stellarator [Domain] y0/x0/boundary` (e.g. `y0 = 28.2`, `boundary = "linked"` matching Landreman et al. 2025 Zenodo `14867777`)
- `TODO(david): confirm ITG gradient & collisionality defaults` (`tprim`, `fprim`, `vnewk`)
- `TODO(david): confirm nv0=16, nw0=8` for GENE
- **Simulation Time & Transient Discard:** GX defaults to `t_max = 200.0`. Zenodo `14867777` discards $t < 150$. Confirm whether we should increase `t_max` (e.g. to 350–400) or pair `t_max = 200` with a 50% transient discard window.
- `TODO(david): confirm dt=0.015 and cfl=0.75`
- `TODO(david): confirm D_hyper=0.05`
- `TODO(david): confirm general gene parameters`
- `TODO(david): confirm partitions`
- `TODO(david): confirm walltime 02:00:00`

### 6. Surrogate Optimization Parameters
*File:* [`surrogates/kr1_surrogate/augmented_lagrangian.py`](../../surrogates/kr1_surrogate/augmented_lagrangian.py)
- `TODO(david): confirm parameter values`

### 7. Diagnostics & Convergence Criteria
*Files:* [`simulations/hpc/tacc_exec/diagnostics_adapter.py`](../hpc/tacc_exec/diagnostics_adapter.py), [`simulations/gcp/gcp_exec/gx_diagnostics.py`](../gcp/gcp_exec/gx_diagnostics.py), [`platform/hpc-core/hpc_core/manifest.py`](../../platform/hpc-core/hpc_core/manifest.py), [`surrogates/kr1_surrogate/dataset_loader.py`](../../surrogates/kr1_surrogate/dataset_loader.py)
- **Timestep Collapse Threshold:** `TODO(david): confirm dt_collapse_ratio < 0.01 threshold` (ratio of recent $\Delta t$ vs early post-initialization $\Delta t$).
- **Stationary Averaging Window:** Current default averages fluxes over $t \ge 0.5 \cdot t_{\text{reached}}$ and flags runs where half-window drift exceeds `0.5` (`max_flux_drift`). Let us know if you prefer a fixed cutoff (e.g. discard $t < 150$) like the Landreman et al. Zenodo dataset.
- `TODO(david): confirm 0.9 converged threshold` (target fraction of simulation time reached)
- `TODO(david): confirm high_k_tail_ratio > 0.25 threshold` (for detecting high-$k$ pileup)
- `TODO(david): confirm default 200.0 sim time target`
- **Flux Normalization Conventions:** GX outputs heat flux divided by the flux-surface average $\langle |\nabla x| \rangle$, matching Zenodo's `Q_avgs_divided_by_FSA_grad_x`. For surrogate training, we can train directly on this native quantity or multiply by `fsa_grad_x` to match Zenodo's `Q_avgs`.

### 8. HPC Watchdog Resource Limits
*File:* [`simulations/hpc/tacc_exec/watchdog.py`](../hpc/tacc_exec/watchdog.py)
- `TODO(david): confirm A100 is the highest available tier`

### 9. Automated Triage Runbooks
*Files:* [`runbook-spectra.md`](../skill/sim-babysitter/okf/concepts/runbook-spectra.md), [`runbook-stalled.md`](../skill/sim-babysitter/okf/concepts/runbook-stalled.md), [`runbook-divergence.md`](../skill/sim-babysitter/okf/concepts/runbook-divergence.md)
- `TODO(david): confirm runbook procedures`
- Today, `STALL` triggers a warm restart; `SPECTRAL_PILEUP` triggers a cold restart with higher resolution or dissipation; and numerical divergence escalates to a human. Let us know if you'd prefer auto-retries on early divergence (e.g. lowering initial $\Delta t$).

### 10. Upstream GX Git Pin
*File:* [`simulations/gcp/Dockerfile.gx-runner`](../gcp/Dockerfile.gx-runner)
- Current build tracks validated GX HEAD (including parallel NetCDF-4 and electrostatic diagnostics). We can also pin a specific commit hash matching previous published runs.

---

## 2. Infrastructure & Environment — `TODO(tan)`

### 1. GCP Project & Regional Quotas
*Files:* [`platform/telemetry/01_tables.sql`](../../../platform/telemetry/01_tables.sql), [`platform/infra/terraform/variables.tf`](../../../platform/infra/terraform/variables.tf)
- `TODO(tan): confirm UT Austin / Google.org project ID` and target region (`us-central1` default; confirm Spot L4 and A100 quota).

### 2. TACC Slurm Queues & Modules
*File:* [`platform/hpc-core/hpc_core/slurm.py`](../../../platform/hpc-core/hpc_core/slurm.py)
- `TODO(tan): confirm any required TACC cluster flags or environment modules` for Vista (`gh`/`gg`) vs Frontera (`rtx`/`normal`).

### 3. Telemetry Privacy
*File:* [`platform/hpc-core/hpc_core/telemetry.py`](../../../platform/hpc-core/hpc_core/telemetry.py)
- `TODO(tan): confirm UT Austin data policy on operator usernames in telemetry` (handles are anonymized with SHA-256 hashes by default).

### 4. External Data Access
*File:* [`platform/infra/terraform/variables.tf`](../../../platform/infra/terraform/variables.tf)
- Confirm external UW service account emails (`var.uw_mrx_writer_members`) so projected equilibria from `mrx` can write directly to the campaign storage bucket.
