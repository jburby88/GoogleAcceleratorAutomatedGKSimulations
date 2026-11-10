---
id: runbook-stalled
triggers: ["STALL", "TIMESTEP_COLLAPSE"]
---

# Runbook: Stalled Progress & Timestep Collapse (`STALL`, `TIMESTEP_COLLAPSE`)

<!-- TODO(david): confirm runbook procedures -->

**Allowed actions (enforced by aurora-steer, `manifest.yaml`):** `warm_restart` or `escalate_to_human`.

## Symptoms
- Sustained adaptive-timestep collapse: `dt_collapse_ratio` (median of the recent `dt` values over the median of the early post-initialization `dt` values, read from `gx.log` / NetCDF `.out.nc` time arrays) falls below the configurable `dt_collapse_threshold` (default `0.01`, i.e. a sustained >100x drop; threshold pending physics-team sign-off, see `docs/OPEN_QUESTIONS.md`), or simulation wall-clock advances without advancing physical time $t \, (c_s / a)$.
- A run that completed its walltime without reaching its convergence criteria is also reported as `STALL`.
- Energy and fluxes remain finite (no `NaN` / `Inf`).

## Recovery Action (`warm_restart`)
1. Set `action = "warm_restart"` and `restart_from_checkpoint = true` (spatial grid is unchanged, so NetCDF restart files (`.restart.nc`, managed by `[Restart] save_for_restart = true`) remain valid).
2. Reduce `cfl_fraction` by 20–30% (keeping `>= 0.05`).
3. Reduce `dt_max` by 50% (keeping `>= 1e-6`).
4. Change ONLY `dt_max` and/or `cfl_fraction`: they are the only parameters marked `warm_restart_allowed`. Do **not** change `nx`, `ny`, `nz` (checkpoint shape) or `hyperdiffusion_coeff` (dissipation operator) during a `warm_restart`; such a directive is rejected and escalated.
5. If the same failure recurs after a previous timestep reduction (see `previous_directives`), or no further reduction is possible within bounds, escalate instead.

---

## Questions for Prof. David Hatch to Confirm in This Runbook

1. **Diagnostic Script Metric Cutoffs (`scalar_metrics`):**
   - In your existing Python diagnostic script (which outputs `converged` + 10–20 monitoring numbers), what is the exact metric name and threshold that indicates a timestep collapse or nonlinear stall? (Current placeholder: `dt_collapse_ratio < dt_collapse_threshold = 0.01`, with an optional absolute `dt_min_floor`, or file mtime stall `> 900s`).
2. **Step Size Reduction Factors (GENE vs. GX):**
   - When `dt` collapses in high-$\beta$ electromagnetic runs, what reduction factor do you prefer for `general.courant` / `[Time] cfl` (e.g., $\times 0.75$) and `general.dt_max` / `[Time] dt` (e.g., $\times 0.5$)?
   - Should the agent also slightly increase hyper-diffusivity (`general.hyp_z` / `[Dissipation] D_hyper`) on the second attempt (`retry_count == 1`) if reducing `dt_max` alone did not resolve the timestep collapse? (That would require a `cold_restart`, which this runbook does not currently allow.)
