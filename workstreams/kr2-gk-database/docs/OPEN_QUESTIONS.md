# KR2 Open Questions & Sign-Off Tracker (`OPEN_QUESTIONS.md`)

Every unconfirmed physics threshold, Slurm/GCP assumption, or infrastructure policy in the codebase is tagged with `TODO(david)` (Prof. David Hatch / physics team) or `TODO(tan)` (Tan / UT Austin & GCP infra) and tracked below. CI (`scripts/check_boundaries_and_deps.py`) verifies parity between inline `TODO(david)`/`TODO(tan)` markers and this file.

Until a question is answered, the code fails safe: the current value is a conservative, configurable default, and anything outside the documented envelope escalates to a human (`escalate_to_human`) or is excluded from the dataset. Items marked **NEEDS-PHYSICS-TEAM** come from the independent domain review (`docs/review/DOMAIN.md`, internal).

---

## 1. Physics & Solver Thresholds — `TODO(david)`

1. **`allowed-parameters.md` (`workstreams/kr2-gk-database/skill/sim-babysitter/okf/concepts/allowed-parameters.md`)**
   - `TODO(david): confirm exact GENE & GX numerical parameter bounds for Vista / Cloud Batch campaigns`
   - Current defaults: `nx: [16, 512]`, `ny: [8, 256]`, `nz: [16, 256]`, `dt_max: [1.0e-6, 0.1]`, `cfl_fraction: [0.05, 0.95]`, `hyperdiffusion_coeff: [0.0, 2.0]`. Includes commented YAML templates for expanding to the full ~12 numerical parameters (`nv`/`nhermite`, `nw`/`nlaguerre`, `hyp_x`, `hyp_y`, `hyp_v`, `lx`).
   - **Warm-restart rule:** only `dt_max` and `cfl_fraction` may change on a checkpoint (warm) restart (`warm_restart_allowed: true`); grid, box-size and dissipation changes force a cold restart. Are there other parameters you consider safe to change from a checkpoint, or should `cfl_fraction`/`dt_max` changes also be flagged in provenance?
   - **Solver-specific grid semantics:** GENE `nx0`/`nky0` count Fourier modes, while GX `nx`/`ny` count real-space grid points before dealiasing. Should `nx`/`ny` have separate GENE and GX bounds (and ~1.5x step rules) instead of one shared range?
   - **NEEDS-PHYSICS-TEAM (`PHYS-04`, reopened after GX source audit) — `hyperdiffusion_coeff` operator mismatch:** GENE `&general hyp_z` is parallel ($z$) hyperdiffusion, whereas GX `[Dissipation] D_hyper` (enabled by `hyper = true`) is perpendicular ($k_x, k_y$) hyperviscosity (`gx` `src/linear.cu:236-239`); GX's parallel-$z$ hyperdiffusion is `[Dissipation] nu_hyper_z` with `hyperz = true` (`src/parameters.cu:178,193`). Please pick one of the three options below (also confirm whether GX's exponent `p_hyper` stays fixed in the base deck):

     | Option | Physics Meaning | GENE Mapping | GX Mapping (`gx.in`) | Code / Config Update Needed |
     | :--- | :--- | :--- | :--- | :--- |
     | **Option A** *(current default)* | Keep 1 stabilization knob mapped to each solver's standard primary dissipation term. | `general.hyp_z` | `Dissipation.D_hyper` + `hyper = true` | **0 code changes** (delete this open item). |
     | **Option B** *(recommended if steering perp & parallel separately)* | Split `hyperdiffusion_coeff` into two directional parameters: `hyperdiffusion_perp` and `hyperdiffusion_par` (`warm_restart_allowed: false` on both). | `perp`: `general.hyp_x` (plus sync `hyp_y` if isotropic)<br>`par`: `general.hyp_z` | `perp`: `Dissipation.D_hyper` + `hyper = true`<br>`par`: `Dissipation.nu_hyper_z` + `hyperz = true` | ~20 lines across `allowed-parameters.md`, `hpc/config.example.json`, `gcp/config.batch.example.json`, `gx_toml.py`, `gene_namelist.py`, and `runbook-spectra.md` (**0** Cloud Run / Terraform / BigQuery schema changes). |
     | **Option C** | Keep 1 knob (`hyperdiffusion_coeff`), but map both solvers strictly to parallel-$z$ hyperdiffusion. | `general.hyp_z` | `Dissipation.nu_hyper_z` + `hyperz = true` | ~6 lines in `allowed-parameters.md`, the 2 example JSON configs, and `gx_toml.py`. |

   - **Candidate GX velocity-space hypercollision mappings (`GXSRC-05`):** GENE `hyp_x`/`hyp_y` have no separate GX keys (GX has one isotropic `D_hyper`), and GX has no `Dissipation.nu_hyper` key (`src/parameters.cu:178-181` provides Hermite/Laguerre hypercollision coefficients `nu_hyper_m`, `nu_hyper_l`, `nu_hyper_lm` alongside `hypercollisions = true`). Before enabling `hyp_v`, please choose:
     - **Option V1:** Map `hyp_v` (`GENE general.hyp_v`) $\leftrightarrow$ `GX Dissipation.nu_hyper_m` (or `nu_hyper_lm`) and auto-set `hypercollisions = true`.
     - **Option V2:** Split velocity hypercollisions into `hyp_hermite` (`nu_hyper_m`) and `hyp_laguerre` (`nu_hyper_l`) for GX (confirm GENE counterpart).
     - **Option V3:** Keep velocity-space hypercollisions fixed in the base input decks and do **not** expose `hyp_v` to automated LLM steering in Phase 1.
2. **Fidelity & Equilibrium Naming (`platform/hpc-core/hpc_core/telemetry.py`)**
   - `TODO(david): confirm exact low vs high fidelity grid criteria and equilibrium/geometry naming`
   - Confirm mapping from external UW `ToBlick/mrx` projected equilibrium IDs (`mrx_eq_*`) into low-fidelity GX vs high-fidelity GENE validation tiers.
3. **GX CUDA / cuFFT VRAM Multiplier (`workstreams/kr2-gk-database/gcp/gcp_exec/gpu_selector.py`)**
   - `TODO(david): calibrate empirical cuFFT buffer multiplier against high-beta kinetic-electron GX runs`
   - Current safe ceiling routes GX runs with estimated footprint `<= 20.0 GiB` to **Spot NVIDIA L4** (`g2-standard-8`, 24 GB) and `> 20.0 GiB` to **Spot NVIDIA A100** (`a2-highgpu-1g`, 40 GB).
4. **Compute & Human Time Savings Assumptions (`platform/telemetry/01_tables.sql`, `platform/telemetry/02_views.sql`, `platform/aurora-steer/app/events.py`, `platform/dashboard/streamlit_app.py`, `workstreams/kr2-gk-database/docs/METRICS.md`)**
   - `TODO(david): confirm baseline manual troubleshooting duration` (default: `45.0` min/fix)
   - `TODO(david): confirm manual walltime continuation overhead` (default: `10.0` min/continuation)
   - `TODO(david): confirm Vista queue walltime defaults` (`4.0h` low / `16.0h` high for GENE)
   - `TODO(david): confirm GX walltime defaults` (`2.0h` low / `8.0h` high for GX)
   - `TODO(david): confirm GENE GPU count per fidelity` (`4` low / `16` high)
   - `TODO(david): confirm GX GPU count per fidelity` (`1` low / `4` high)
5. **KR3 Active Learning Queue (`workstreams/kr3-transport-consistent-design/kr3_design/active_learning_queue.py`)**
   - `TODO(david): confirm resolution nx/ny/ntheta mappings for high/low fidelity`
   - `TODO(david): confirm nhermite=16, nlaguerre=8` and `nspecies = 2` (kinetic electrons) when `beta > 0`, else `1` (DOM-05)
   - `TODO(david): confirm stellarator [Domain] y0/x0/boundary` (`ZEN-02`: e.g. `y0 = 28.2`, `boundary = "linked"` vs periodic / twist-and-shift for uniform-length `24*pi` flux tubes in Landreman et al. 2025, Zenodo `14867777`)
   - `TODO(david): confirm ITG gradient & collisionality defaults` (`ZEN-05`: `tprim`, `fprim`, `vnewk`, and adiabatic-electron `[Boltzmann]` vs kinetic-electron `[species]` tables)
   - `TODO(david): confirm nv0=16, nw0=8` for GENE (DOM-05); `kymin` is deliberately omitted (per-equilibrium physics choice)
   - **End time vs. post-transient cutoff (`ZEN-03`):** Confirm `[Time] t_max = 200.0` (GX) and `&general simtimelim = 200.0` (GENE; equal time units when `m_ref = m_i`, `T_ref = T_i`). Note that Landreman et al. (2025) Zenodo `14867777` (`dataset.README:63`) omits the transient period `t < 150` (`t > 150` fixed cutoff), which leaves only `50 a/v_th` of averaging window if `t_max = 200.0`; confirm whether `t_max` should be increased (e.g. `>= 300–400`) or paired with `transient_discard_fraction = 0.5`.
   - `TODO(david): confirm dt=0.015 and cfl=0.75`
   - `TODO(david): confirm D_hyper=0.05` (with `hyper = true`; replaces the invalid key `hyperviscosity = 0.15`)
   - `TODO(david): confirm general gene parameters`
   - `TODO(david): confirm partitions`
   - `TODO(david): confirm walltime 02:00:00`
6. **Augmented Lagrangian Optimizer (`workstreams/kr1-optimizer-orchestration/kr1_surrogate/augmented_lagrangian.py`)**
   - `TODO(david): confirm parameter values`
7. **Diagnostics: TACC adapter and GX diagnostics (`workstreams/kr2-gk-database/hpc/tacc_exec/diagnostics_adapter.py`, `workstreams/kr2-gk-database/gcp/gcp_exec/gx_diagnostics.py`, `platform/hpc-core/hpc_core/manifest.py`, `workstreams/kr1-optimizer-orchestration/kr1_surrogate/dataset_loader.py`)**
   - **NEEDS-PHYSICS-TEAM (PHYS-01) — timestep collapse:** *"In GX (when `[Time] dt` is omitted and `cfl` controls adaptive RK3/RK4 stepping) and GENE (`calc_dt = .t.`), what is the expected ratio $\langle \Delta t_{\text{sat}} \rangle / \Delta t_{\text{lin}}$ across Cyclone ITG and high-$\beta$ `ToBlick/mrx` stellarator runs, and what `dt_collapse_threshold` (or absolute `dt_min` floor in $L_{\text{ref}}/v_{\text{th}}$ units) should trigger `TIMESTEP_COLLAPSE`?"* Today: `dt_collapse_ratio` = median of recent `dt` / median of early post-initialization `dt`, `TODO(david): confirm dt_collapse_ratio < 0.01 threshold` (configurable `dt_collapse_threshold`, optional `dt_min_floor` off by default).
   - **Saturation / convergence window (`PHYS-02`, `ZEN-03`, `ZEN-09`):** fluxes are the time average over the post-transient window `t >= transient_discard_fraction * t_reached` (default `0.5`), and a run is marked unconverged when the relative half-window drift exceeds `max_flux_drift` (default `0.5`). Please choose:

     | Option | Window & End-Time Rule | Trade-Off | Code / Config Update Needed |
     | :--- | :--- | :--- | :--- |
     | **Option W1** *(current default)* | `t_max = 200.0`, fractional cutoff `t >= 0.5 * t_reached` (`transient_discard_fraction = 0.5`), `max_flux_drift = 0.5`. | Lower GPU hours/run; works across differing `t_max`. | **0 code changes** (all thresholds already config-driven). |
     | **Option W2** *(Zenodo `14867777` style)* | Increase `t_max` to `350–400`, discard fixed `t < 150` (`dataset.README:63`), keep `max_flux_drift = 0.5` (plus optional standard-error gate). | Longer stationary averaging window for slow-saturating stellarator runs. | Update `t_max` in campaign base decks / `active_learning_queue.py` + optional fixed `transient_min_time` in `timeseries.py` (~10 lines). |
   - `TODO(david): confirm 0.9 converged threshold` (fraction of `sim_time_target` that must be reached)
   - `TODO(david): confirm high_k_tail_ratio > 0.25 threshold` (`SPECTRAL_PILEUP`, both $k_x$ and $k_y$ tails)
   - `TODO(david): confirm default 200.0 sim time target` (last resort after `run_meta.json` and the solver input's `t_max` / `simtimelim`)
   - **Cross-code & Zenodo `<|grad x|>` flux normalization (`ML-07`, `GXSRC-08`, `ZEN-12`):** GX divides raw `HeatFlux*` and `ParticleFlux*` outputs by the flux-surface average $\langle |\nabla x| \rangle$ (`gx` `src/geometry.cu:955-960`), matching Zenodo `14867777`'s `Q_avgs_divided_by_FSA_grad_x`, whereas Zenodo's primary `Q_avgs` array multiplies by `FSA_grad_x` (`dataset.README:61-68`). `fluxes.json` records `flux_normalization` (`"gyroBohm_divided_by_fsa_grad_x"` for GX, `"gyroBohm_gene"` for GENE) and optional `fsa_grad_x` from `Geometry/{jacobian,grho}`. Please choose the KR1 surrogate target convention:

     | Option | Surrogate Training Target | When to Use | Code / Config Update Needed |
     | :--- | :--- | :--- | :--- |
     | **Option N1** *(current default for single-solver GX)* | Train directly on GX's native `Q / <\|grad x\|>` (`Q_avgs_divided_by_FSA_grad_x`); loader rejects mixed `flux_normalization` unless explicitly harmonized. | Phase 1 GX-only surrogate campaigns. | **0 code changes.** |
     | **Option N2** *(recommended for Zenodo `Q_avgs` & multi-fidelity `GENE`+`GX`)* | Multiply GX fluxes by `fsa_grad_x` (`Q = Q_gx * fsa_grad_x`, matching Zenodo `Q_avgs`) and apply confirmed $v_{\text{th}}$ ($1$ vs $2^{3/2}$) and $L_{\text{ref}}$ conversion factor when combining `GENE` and `GX` runs. | Multi-fidelity `GX` (low-fi) + `GENE` (high-fi) surrogate training. | ~15 lines in `kr1_surrogate/dataset_loader.py` once `v_th` / `L_ref` ratio is confirmed. |
8. **Watchdog (`workstreams/kr2-gk-database/hpc/tacc_exec/watchdog.py`)**
   - `TODO(david): confirm A100 is the highest available tier`
9. **Runbooks (`workstreams/kr2-gk-database/skill/sim-babysitter/okf/concepts/runbook-spectra.md`, `runbook-stalled.md`, `runbook-divergence.md`)**
   - `TODO(david): confirm runbook procedures`
   - The per-failure action policy is enforced server-side from `okf/manifest.yaml`:

     | Option | Policy on `DIVERGENCE` & Repeated `TIMESTEP_COLLAPSE` | Code / Config Update Needed |
     | :--- | :--- | :--- |
     | **Option R1** *(current fail-safe default)* | `STALL` / `TIMESTEP_COLLAPSE` $\rightarrow$ `warm_restart` only; `SPECTRAL_PILEUP` $\rightarrow$ `cold_restart` only; `DIVERGENCE` / `CRASH` / `UNKNOWN` $\rightarrow$ `escalate_to_human` only. | **0 code changes.** |
     | **Option R2** *(relaxed retry policy)* | Allow `cold_restart` (with reduced `dt_max`) on early linear-phase `DIVERGENCE`, and allow `cold_restart` (with increased hyperdiffusion) on `TIMESTEP_COLLAPSE` if a `warm_restart` already failed. | Update `allowed_actions_by_failure_mode` in `okf/manifest.yaml` + prompt text in `runbook-divergence.md` / `runbook-stalled.md` (**0** Python changes). |

10. **Upstream GX Commit Pin (`ZEN-14`, `workstreams/kr2-gk-database/gcp/Dockerfile.gx-runner`)**
   - Pick which 40-hex `GX_COMMIT` SHA from `https://bitbucket.org/gyrokinetics/gx.git` to pin in CI / Cloud Batch:
     - **Option G1 *(recommended)*:** Pin current validated GX HEAD (`git ls-remote https://bitbucket.org/gyrokinetics/gx.git HEAD | awk '{print $1}'`), which includes parallel MPI NetCDF-4 (`nc_create_par`) and electrostatic `HeatFluxES_st` diagnostics supported by `Dockerfile.gx-runner` and `gx_diagnostics.py`.
     - **Option G2:** Pin an older historical commit matching Landreman et al. (2025) Zenodo `14867777` (note: `gx_diagnostics.py` already handles older schemas by falling back to `HeatFlux_st`, though `Dockerfile.gx-runner` `Makefile.ubuntu` flags should be smoke-checked against that commit).

---

## 2. GCP, Security & UT Policy — `TODO(tan)`

1. **GCP Project ID & Region (`platform/telemetry/01_tables.sql`, `platform/infra/terraform/variables.tf`)**
   - `TODO(tan): confirm UT Austin / Google.org project ID` and primary region (`us-central1` default; confirm Spot `g2-standard-8` L4 & `a2-highgpu-1g` A100 quota in `us-central1`).
2. **TACC Slurm Cluster Flags & Environment Modules (`platform/hpc-core/hpc_core/slurm.py`)**
   - `TODO(tan): confirm any required TACC cluster flags or environment modules` for Vista (`gh` / `gg` queues) vs Frontera (`rtx` / `normal` queues).
3. **Operator Handle Hashing Policy (`platform/hpc-core/hpc_core/telemetry.py`)**
   - `TODO(tan): confirm UT Austin data policy on operator usernames in telemetry`
   - Default behavior hashes operator handles with salted SHA-256 (`op_<16hex>`) unless `TELEMETRY_STORE_USERNAMES=true`.
4. **Cross-Project GCS IAM for External UW `ToBlick/mrx` Pipeline (`platform/infra/terraform/variables.tf`)**
   - Confirm exact UW GCP service account email(s) for `var.uw_mrx_writer_members` so Alan Kaptanoglu / Tobias Blickhan's GKE TPU `mrx` runs can write projected equilibria directly to `gs://<bucket>/campaigns/*/equilibria/`.
