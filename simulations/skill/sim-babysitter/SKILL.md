---
name: sim-babysitter
description: OKF 2.0 physics-informed numerical stability runbooks and parameter guardrails for automated GENE and GX gyrokinetic fusion simulations on TACC Slurm and GCP Cloud Batch.
---

# `sim-babysitter` OKF 2.0 Skill (KR2: Automated Gyrokinetic Simulation Database)

This Open Knowledge Framework (OKF 2.0) skill is loaded dynamically by `aurora-steer` (`platform/aurora-steer/app/okf_loader.py`) to diagnose numerical instabilities in **GENE** (Fortran CPU/GPU on TACC Vista & LoneStar6) and **GX** (CUDA pseudo-spectral GPU on TACC and GCP Cloud Batch).

---

## How to Collaborate on This Skill (Guide for Prof. David Hatch & UT Physics Team)

Instead of hardcoding troubleshooting heuristics inside Python or training a custom classifier, `aurora-steer` reads your physics troubleshooting intuition directly from the Markdown files in [`okf/concepts/`](okf/concepts/):

1. **[`okf/manifest.yaml`](okf/manifest.yaml)** — Maps each failure category (`STALL`, `TIMESTEP_COLLAPSE`, `SPECTRAL_PILEUP`, `DIVERGENCE`, `CRASH`, `UNKNOWN`) to the specific Markdown runbooks loaded into the Google ADK 2.0 agent prompt, and lists each runbook's `allowed_actions` (enforced server-side: any other action is escalated).
2. **[`okf/concepts/allowed-parameters.md`](okf/concepts/allowed-parameters.md)** — **Single Source of Truth for Guardrails.** The YAML frontmatter (`parameters:`) defines every numerical solver parameter the agent is allowed to adjust, its `[min, max]` bounds, its exact namelist/TOML path in GENE (`parameters`) and GX (`gx.in`), and whether it may change on a checkpoint restart (`warm_restart_allowed`). Editing this YAML frontmatter automatically updates the server-side reject-only guardrails (`app/guardrails.py`) — **zero Python code changes required.** Parsing is strict: a malformed file stops the service from starting rather than weakening a guardrail.
3. **[`okf/concepts/runbook-stalled.md`](okf/concepts/runbook-stalled.md)** — Troubleshooting recipe for stalled runs and adaptive timestep collapse (`dt` drop).
4. **[`okf/concepts/runbook-spectra.md`](okf/concepts/runbook-spectra.md)** — Troubleshooting recipe for high-$k$ spectral pile-up and unresolved sub-grid turbulence cascade.
5. **[`okf/concepts/runbook-divergence.md`](okf/concepts/runbook-divergence.md)** — Rules for `NaN`/`Inf` blow-ups, singular metric tensors, and solver crashes.
6. **[`okf/concepts/escalation-policy.md`](okf/concepts/escalation-policy.md)** — 3-strike circuit breaker (`retry_count >= 2` -> `0` LLM calls), zero-LLM checkpoint continuations on Slurm `TIMEOUT` / Spot `PREEMPTED`, and `ISSUE.md` human handoff.

---

## Core Scientific Invariants

1. **Numerical Solver Parameters ONLY**: The agent may ONLY modify allowlisted numerical discretization, hyper-diffusivity, and time-stepping parameters defined in `okf/concepts/allowed-parameters.md`. It is strictly forbidden from modifying magnetic equilibrium geometry (`equilibrium_id`, `vmec_file`), plasma $\beta$, temperature/density gradients (`omt`, `omn`), collisionality, or species definitions.
2. **Only Timestep Controls Change on a Warm Restart**: A checkpoint (warm) restart may change only parameters marked `warm_restart_allowed` (`dt_max`, `cfl_fraction`). Spatial or velocity-space grid changes (`nx`, `ny`, `nz`) make binary checkpoint files (`checkpoint_*` / `.restart.nc`) incompatible, and dissipation changes (`hyperdiffusion_coeff`) would alter the physics mid-trajectory: such directives MUST set `action = "cold_restart"` and `restart_from_checkpoint = false`.
3. **Reject-Only Guardrails**: Both `aurora-steer` and the TACC/GCP executors enforce exact `[min, max]` bounds in reject-only mode (no silent clamping). Any out-of-bounds or non-allowlisted parameter, any action not allowed by the failure type's runbook, and any no-op patch forces `escalate_to_human`.
4. **3-Strike Circuit Breaker & Zero-LLM Continuations**: Slurm `TIMEOUT` and GCP Spot `PREEMPTED` events recover locally from checkpoint with **0 LLM calls**. Any run reaching `retry_count >= 2` escalates immediately to `queue/needs_human/<run_id>/ISSUE.md`.
