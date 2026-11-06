---
id: allowed-parameters
triggers: ["ALL"]
# TODO(david): confirm exact GENE & GX numerical parameter bounds for Vista / Cloud Batch campaigns
# NOTE FOR PROF. DAVID HATCH:
#   1. `okf_loader.py` parses the `parameters:` YAML block below at runtime to enforce
#      strict server-side `[min, max]` guardrails (`app/guardrails.py`). Parsing fails
#      closed (the service refuses to start) on: non-numeric or non-finite bounds
#      (write `1.0e-6`, not `1e-6`, which YAML reads as a string), fractional bounds on
#      `int` parameters, duplicate keys, unknown keys, a missing `gene_path`/`gx_path`
#      (`section.key`), or two parameters mapped to the same solver key.
#   2. `warm_restart_allowed: true` marks the ONLY parameters that may change on a
#      checkpoint (warm) restart: the pure timestep controls `dt_max` and `cfl_fraction`
#      (docs/review/DOMAIN.md PHYS-03). Grid, box-size and dissipation changes force
#      `action = "cold_restart"`. Omitting the key means `false`.
#   3. Currently 6 core numerical parameters are active below (`nx`, `ny`, `nz`, `dt_max`,
#      `cfl_fraction`, `hyperdiffusion_coeff`).
#   4. Beneath the active 6 parameters are commented-out templates for the remaining ~6
#      numerical parameters discussed on Nov 4 (velocity-space grids, directional
#      hyper-diffusivity, box size). Once you confirm their exact names/bounds,
#      uncomment them here and mirror them in `config.example.json` / `config.batch.example.json`.
#   5. Do not put comments on the same line as a value (CI parses this block line by line).
parameters:
  nx:
    type: int
    min: 16
    max: 512
    gene_path: "box.nx0"
    gx_path: "Dimensions.nx"
    warm_restart_allowed: false
  ny:
    type: int
    min: 8
    max: 256
    gene_path: "box.nky0"
    gx_path: "Dimensions.ny"
    warm_restart_allowed: false
  nz:
    type: int
    min: 16
    max: 256
    gene_path: "box.nz0"
    gx_path: "Dimensions.ntheta"
    warm_restart_allowed: false
  dt_max:
    type: float
    min: 1.0e-6
    max: 0.1
    gene_path: "general.dt_max"
    gx_path: "Time.dt"
    warm_restart_allowed: true
  cfl_fraction:
    type: float
    min: 0.05
    max: 0.95
    gene_path: "general.courant"
    gx_path: "Time.cfl"
    warm_restart_allowed: true
  hyperdiffusion_coeff:
    type: float
    min: 0.0
    max: 2.0
    gene_path: "general.hyp_z"
    gx_path: "Dissipation.D_hyper"
    warm_restart_allowed: false
  # ---------------------------------------------------------------------------
  # CANDIDATE ADDITIONAL NUMERICAL PARAMETERS (~12 TOTAL — FOR DAVID TO CONFIRM)
  # Velocity grids and box size change the checkpoint array shape and dissipation
  # changes the physics: all of them stay `warm_restart_allowed: false`.
  # ---------------------------------------------------------------------------
  # nv:
  #   type: int
  #   min: 16
  #   max: 128
  #   gene_path: "box.nv0"
  #   gx_path: "Dimensions.nhermite"
  #   warm_restart_allowed: false
  # nw:
  #   type: int
  #   min: 8
  #   max: 64
  #   gene_path: "box.nw0"
  #   gx_path: "Dimensions.nlaguerre"
  #   warm_restart_allowed: false
  # PHYS-04 Option B template (if Prof. Hatch chooses to split `hyperdiffusion_coeff` into
  # perpendicular + parallel operators; see `workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md`):
  # hyperdiffusion_perp:
  #   type: float
  #   min: 0.0
  #   max: 2.0
  #   gene_path: "general.hyp_x"
  #   gx_path: "Dissipation.D_hyper"
  #   warm_restart_allowed: false
  # hyperdiffusion_par:
  #   type: float
  #   min: 0.0
  #   max: 2.0
  #   gene_path: "general.hyp_z"
  #   gx_path: "Dissipation.nu_hyper_z"
  #   warm_restart_allowed: false
  # Note on GENE `hyp_x` / `hyp_y`: GENE has separate directional perpendicular keys (`general.hyp_x`,
  #   `general.hyp_y`), whereas GX has one isotropic perpendicular coefficient (`Dissipation.D_hyper`).
  #   Two OKF parameters may not share one solver key; if Option B is chosen and GENE should keep
  #   isotropic perp damping, `gene_namelist.py` can sync `hyp_y = hyp_x` alongside `general.hyp_x`.
  # hyp_v: in GX `Dissipation.hypercollisions = true` is only the boolean toggle; GX has
  #   separate Hermite/Laguerre hypercollision coefficients (`Dissipation.nu_hyper_m`,
  #   `Dissipation.nu_hyper_l`, `Dissipation.nu_hyper_lm`; parallel-z hyperdiffusion is
  #   `Dissipation.nu_hyper_z` with `hyperz = true`, and there is no `Dissipation.nu_hyper`).
  # hyp_v:
  #   type: float
  #   min: 0.0
  #   max: 2.0
  #   gene_path: "general.hyp_v"
  #   gx_path: "<to be confirmed: nu_hyper_m / nu_hyper_l>"
  #   warm_restart_allowed: false
  # lx:
  #   type: float
  #   min: 20.0
  #   max: 400.0
  #   gene_path: "box.lx"
  #   gx_path: "Domain.x0"
  #   warm_restart_allowed: false
---

# Allowlisted Numerical Solver Parameters

Only the allowlisted numerical parameters below may appear in `parameter_modifications`, always as absolute new values:

| Parameter | Type, bounds | GENE (`parameters`) | GX (`gx.in`) | Warm restart? |
| :--- | :--- | :--- | :--- | :--- |
| `nx` | `int`, `[16, 512]` | `&box nx0` | `[Dimensions] nx` | no: cold restart |
| `ny` | `int`, `[8, 256]` | `&box nky0` | `[Dimensions] ny` | no: cold restart |
| `nz` | `int`, `[16, 256]` | `&box nz0` | `[Dimensions] ntheta` | no: cold restart |
| `dt_max` | `float`, `[1.0e-6, 0.1]` | `&general dt_max` | `[Time] dt` | yes |
| `cfl_fraction` | `float`, `[0.05, 0.95]` | `&general courant` | `[Time] cfl` | yes |
| `hyperdiffusion_coeff` | `float`, `[0.0, 2.0]` | `&general hyp_z` | `[Dissipation] D_hyper` (the GX patcher also sets `hyper = true` when the value is > 0) | no: cold restart |

- **Warm restart** (`action = "warm_restart"`, `restart_from_checkpoint = true`): ONLY `dt_max` and `cfl_fraction` may change. Grid changes (`nx`, `ny`, `nz`) make binary checkpoints (`.restart.nc` / `checkpoint_*`) incompatible, and a dissipation change (`hyperdiffusion_coeff`) would alter the damping operator in the middle of one flux time trace. Any such change requires `action = "cold_restart"` with `restart_from_checkpoint = false`.
- Note: GENE `hyp_z` is parallel ($z$) hyperdiffusion, whereas GX `D_hyper` is perpendicular ($k_\perp$) hyperviscosity (`gx` `src/linear.cu:236-239`; GX parallel-$z$ hyperdiffusion is `[Dissipation] nu_hyper_z` with `hyperz = true`). The shared name `hyperdiffusion_coeff` therefore controls different operators in the two codes (open question for the physics team, `PHYS-04`).

**Forbidden**: Never modify equilibrium files (`equilibrium_id`, `vmec_file`), plasma $\beta$, temperature/density gradients (`omt`, `omn`), safety factor / rotational transform $\iota$, or species charges/masses.

---

## Action Required from Prof. David Hatch (Sign-Off Checklist)

To expand from the 6 starter parameters above to the full **~12 tunable numerical parameters** discussed on Nov 4, please confirm:

1. **Exact `[min, max]` bounds** for the 6 active parameters above on TACC Vista/LoneStar6 (GENE) and GCP Cloud Batch (GX), and whether `nx`/`ny` need solver-specific bounds (GENE `nx0`/`nky0` count Fourier modes; GX `nx`/`ny` count real-space grid points before dealiasing).
2. **Hyperdiffusion mapping (`PHYS-04` — choose Option A, B, or C):**

   | Option | Behaviour | GENE (`parameters`) | GX (`gx.in`) | Update Needed |
   | :--- | :--- | :--- | :--- | :--- |
   | **Option A** *(current)* | Keep 1 knob (`hyperdiffusion_coeff`) mapped to each solver's standard stabilization term. | `&general hyp_z` | `[Dissipation] D_hyper` + `hyper = true` | None (0 code changes). |
   | **Option B** *(split perp vs. par)* | Replace `hyperdiffusion_coeff` with `hyperdiffusion_perp` and `hyperdiffusion_par` (template commented above). | `perp`: `&general hyp_x` (optional `hyp_y` sync)<br>`par`: `&general hyp_z` | `perp`: `[Dissipation] D_hyper` + `hyper = true`<br>`par`: `[Dissipation] nu_hyper_z` + `hyperz = true` | Uncomment template above + mirror in `config.example.json`, `config.batch.example.json`, `gx_toml.py`, `gene_namelist.py`, and `runbook-spectra.md`. |
   | **Option C** *(parallel-$z$ only)* | Keep 1 knob (`hyperdiffusion_coeff`), map both codes to parallel-$z$ hyperdiffusion. | `&general hyp_z` | `[Dissipation] nu_hyper_z` + `hyperz = true` | Update `gx_path` here, in the 2 JSON configs, and in `gx_toml.py`. |

3. **Which additional numerical parameters (from the commented YAML template above) should be enabled:**
   - Velocity-space resolution: `box.nv0` / `Dimensions.nhermite` and `box.nw0` / `Dimensions.nlaguerre`?
   - Velocity-space hypercollisions (`GXSRC-05`): map `general.hyp_v` to GX `Dissipation.nu_hyper_m` / `nu_hyper_l` / `nu_hyper_lm` (with `hypercollisions = true`), or keep velocity hypercollisions fixed in the base deck?
   - Flux-tube box size / minimum $k_y$: `box.lx` / `box.kymin` (`Domain.y0` in GX), or is `kymin` considered fixed physics for the campaign?
4. **Cold-restart vs. warm-restart rule:** the current rule allows only `dt_max` / `cfl_fraction` on a warm restart (velocity-grid and box-size changes are treated like spatial-grid changes). Confirm, or name other parameters that are safe to change from a checkpoint.
