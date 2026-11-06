# `workstreams/kr1-optimizer-orchestration` — Multi-Fidelity Surrogates & Outer-Loop Orchestration

**Owners:** Joshua Burby (UT Austin), David Hatch (UT Austin), Alan Kaptanoglu (NYU), Tobias Blickhan, Nova Fallen

**Status:** KR1 surrogate training (Phase 2) and the coupled outer loop (Phase 3) are not implemented yet (`docs/IMPLEMENTATION_PLAN.md`). This package ships the two building blocks they will use: a fail-closed dataset loader and the augmented Lagrangian multiplier/penalty state. Both are pure Python (stdlib only) and unit-tested.

## Upstream Equilibrium Inputs & Scientific References
1. **3D Broken-Surface MHD Equilibria (`ToBlick/mrx`, T. Blickhan & A. Kaptanoglu)**: https://github.com/ToBlick/mrx/
   - Generates 3D broken-surface MHD equilibria projected onto nested flux surfaces (`equilibrium_id`), stored under `campaigns/<id>/equilibria/<equilibrium_id>/` and consumed by KR2 (`tacc-driver` on TACC + `gcp-gx-babysitter` on GCP Cloud Batch) to produce ground-truth GX & GENE transport fluxes.
2. **AI-Driven Stellarator Parameter Orchestration**: https://arxiv.org/abs/2603.15240
3. **Augmented Lagrangian method for stellarator design**: P. F. Gil, W. Li, J. Stratton, A. A. Kaptanoglu & E. V. Stenson, *Stellarator coils for future fusion reactors via an augmented Lagrangian approach*, Phys. Rev. Lett. 137, 065101 (2026), https://doi.org/10.1103/n7gk-922h
4. **Method references**: J. Nocedal & S. J. Wright, *Numerical Optimization*, 2nd ed. (2006), Ch. 17 (§17.3–17.4); R. T. Rockafellar, J. Optim. Theory Appl. 12, 555 (1973); R. Andreani, E. G. Birgin, J. M. Martínez & M. L. Schuverdt, SIAM J. Optim. 18, 1286 (2008).

## Modules in This Package

### `kr1_surrogate/dataset_loader.py`
Loads the KR2 campaign layout written by `platform/hpc-core/hpc_core/manifest.py` (`campaigns/<id>/{equilibria,runs}/`; an expanded file-per-run layout that extends the provenance of the Zenodo record [14867777](https://zenodo.org/records/14867777), which itself ships as an HDF5 archive).

- **Fail-closed admission** (`load_campaign_report`, `load_campaign_samples`): a run is admitted only if its `run_manifest.json` (the writer's commit marker) and `fluxes.json` are strict JSON (no `NaN`/`Infinity`) with schema `2.0`, identifiers agree (folder, manifest, fluxes, equilibrium), `converged` is `true`, `escalated` and `simulated` are `false`, `environment` is `prod`, the writer's verdict is `dataset_eligible: true` with no `eligibility_reasons`, every steered resubmission has a recorded parameter patch, every `files` entry matches its sha256 and size, the flux-units label is `gyroBohm` and the flux values are finite numbers. A missing field excludes the run; nothing is defaulted.
- **Exclusion report**: `load_campaign_report(...).exclusion_counts` gives the number of excluded runs per stable reason code (`not_converged`, `escalated`, `simulated`, `non_prod_environment`, `missing_field:<name>`, `checksum_mismatch`, `marked_ineligible:<writer reason>`, ...). `load_campaign_samples` logs these counts as a warning.
- **Patched runs are admitted by default.** KR2 only patches physics-preserving numerical parameters and records every patch. `unpatched_only=True` / `hands_free_only=True` restrict the set for ablations.
- **No flux re-normalization.** GX and GENE already write dimensionless gyroBohm fluxes (Q/Q_gB); the loader passes them through and records `flux_units` on each sample. Mixing GX and GENE fluxes in one model also needs a common thermal-velocity/reference-length convention: that is an open physics question, and the loader never converts between conventions.
- **Leak-free splits** (`split_by_equilibrium`): all samples of an `equilibrium_id` land in exactly one of `train`/`val`/`test`, so multi-fidelity GX + GENE runs of one equilibrium never leak across splits. The default hash split is stable when the dataset grows (an equilibrium's split depends only on `seed` and its id). The optional `stratify_by=` callback gives exact per-stratum proportions (largest-remainder allocation), but assignments can shift when equilibria are added, so freeze the dataset version first.

The manifest contract is owned by KR2 (`hpc_core/manifest.py`); the supported schema versions and field names are the constants at the top of `dataset_loader.py`.

### `kr1_surrogate/augmented_lagrangian.py`
Outer-loop state of a Powell–Hestenes–Rockafellar augmented Lagrangian for equality constraints `c_i(x) = 0` and inequality constraints `g_j(x) <= 0` (positive residual = violation). It holds multipliers and the penalty `mu` and provides `evaluate_augmented_objective`, `gradient_weights` (for gradient-based inner solvers), `update_outer_loop` (safeguarded multiplier updates, with `mu` growing on insufficient progress in combined feasibility and complementarity), and the termination checks `kkt_violation` / `is_feasible` / `converged`. The inner minimization over the design variables is the caller's. Tests drive it to the known KKT points of analytic problems. Default hyperparameters await physics-team confirmation (marked in the code and tracked in `workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md`).
