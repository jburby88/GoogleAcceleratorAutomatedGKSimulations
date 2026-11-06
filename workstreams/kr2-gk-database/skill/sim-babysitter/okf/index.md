# `sim-babysitter` OKF 2.0 Knowledge Graph Index

This directory contains the Open Knowledge Framework (OKF 2.0) concept graph for `aurora-steer`. When a simulation failure arrives at `POST /triage`, `okf_loader.py` selects all concepts with `triggers: ["ALL"]` plus the concept matching `diagnostic.failure_type`. The `allowed_actions` of the matching runbook (in `manifest.yaml`) are enforced server-side.

| Concept ID | Path | Triggers | Allowed actions | Purpose & Physics Owner Sign-Off |
| :--- | :--- | :--- | :--- | :--- |
| `allowed-parameters` | [`concepts/allowed-parameters.md`](concepts/allowed-parameters.md) | `ALL` | (none granted) | Authoritative numerical parameter allowlist, types, `[min, max]` bounds, GENE/GX file mappings (GX TOML `gx.in`) and `warm_restart_allowed` flags (Prof. David Hatch: confirm bounds & expand from 6 starter parameters to full ~12 parameters) |
| `runbook-stalled` | [`concepts/runbook-stalled.md`](concepts/runbook-stalled.md) | `STALL`, `TIMESTEP_COLLAPSE` | `warm_restart`, `escalate_to_human` | Warm-restart recovery via `cfl_fraction` and `dt_max` reduction (Prof. David Hatch: confirm step ratios & diagnostic metric cutoffs) |
| `runbook-spectra` | [`concepts/runbook-spectra.md`](concepts/runbook-spectra.md) | `SPECTRAL_PILEUP` | `cold_restart`, `escalate_to_human` | Cold-restart recovery via perpendicular/parallel grid refinement (`nx`, `ny`, `nz`) and `hyperdiffusion_coeff` (Prof. David Hatch: confirm spectral tail metric & scaling rules) |
| `runbook-divergence` | [`concepts/runbook-divergence.md`](concepts/runbook-divergence.md) | `DIVERGENCE`, `CRASH`, `UNKNOWN` | `escalate_to_human` | Escalation vs. recovery rules for `NaN`/`Inf` blow-ups, high-$\beta$ electromagnetic instabilities, and solver crashes (Prof. David Hatch: confirm if any divergence subtypes are recoverable) |
| `escalation-policy` | [`concepts/escalation-policy.md`](concepts/escalation-policy.md) | `ALL` | (none granted) | 3-strike circuit breaker, per-run LLM budget, checkpoint (NetCDF `.restart.nc`) vs. cold restart rules, and `ISSUE.md` handoff requirements |
