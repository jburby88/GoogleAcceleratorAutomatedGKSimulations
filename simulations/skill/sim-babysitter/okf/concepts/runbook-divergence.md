---
id: runbook-divergence
triggers: ["DIVERGENCE", "CRASH", "UNKNOWN"]
---

# Runbook: Numerical Divergence, NaN/Inf Blow-Up, or Solver Crash (`DIVERGENCE`, `CRASH`, `UNKNOWN`)

<!-- TODO(david): confirm runbook procedures -->

**Allowed actions (enforced by aurora-steer, `manifest.yaml`):** `escalate_to_human` only. Any other directive for these failure types is rejected and escalated.

## Symptoms
- `NaN` or `Inf` detected in electrostatic heat flux $Q_{\text{es}}$, electromagnetic flux $Q_{\text{em}}$, or field amplitude (`has_nan = 1.0` (from `NaN` detected in NetCDF `.out.nc` flux arrays, enabling `[Diagnostics] fluxes = true`)).
- Unhandled MPI/CUDA segmentation fault, singular metric tensor from broken-surface projection, or unclassified non-zero exit code.

## Mandatory Action (`escalate_to_human`)
1. Set `action = "escalate_to_human"`, `parameter_modifications = {}`, and `restart_from_checkpoint = false`.
2. Do **not** guess blind grid or timestep tweaks on `NaN` divergence or unknown solver crashes — high-$\beta$ electromagnetic instabilities (KBM / microtearing) or pathological projected equilibria require physicist review (`ISSUE.md`).

---

## Questions for Prof. David Hatch to Confirm in This Runbook

1. **Are Any `DIVERGENCE` Subtypes Automatically Recoverable on Strike 1 (`retry_count == 0`)?**
   - Currently, `DIVERGENCE` (`NaN`/`Inf` in fluxes) immediately escalates to a human physicist (`escalate_to_human`) for safety.
   - If an initial explicit timestep (`dt_max`) is simply too large during the linear-to-nonlinear overshoot and causes an early `NaN` overflow, should Strike 1 (`retry_count == 0`) attempt a `cold_restart` with `dt_max` reduced by 50% and `cfl_fraction` reduced by 25%, or always escalate immediately?
2. **Projected Equilibrium Pathology Signatures (`ToBlick/mrx` -> GENE/GX):**
   - What log strings or metric tensor warnings indicate that the upstream projected 3D equilibrium itself has a bad Jacobian / metric spike (so the run should immediately escalate without wasting GPU hours)?
