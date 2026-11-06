# `workstreams/kr3-transport-consistent-design` — High-$\beta$ Transport-Consistent Stellarator Design

**Owners:** Whole Team (UT Austin, UW, Courant/NYU, Google)

**Goal (Phase 3, `docs/IMPLEMENTATION_PLAN.md`):** close the active-learning optimization loop across all three Key Results:
1. Couple the KR1 multi-fidelity neural/GP surrogates (`ToBlick/mrx` broken-surface equilibrium + neoclassical bootstrap + GX/GENE gyrokinetic transport) inside the augmented Lagrangian outer loop (`>3%` plasma $\beta$, `10x` faster design iteration).
2. When the optimizer enters an out-of-distribution geometry where the surrogate's epistemic uncertainty exceeds a threshold, write a GX or GENE validation run into KR2's `queue/pending/`, so that `sim-babysitter` runs it and harvests ground-truth transport fluxes (`<5%` relative $L_2$ error).

## What is implemented today

Only the hand-off into KR2's queue is implemented: `kr3_design.enqueue_validation_run`.

- It refuses a `run_id` that already exists in any KR2 queue state (`pending`, `running`, `completed`, `needs_human`) with `RunAlreadyQueuedError`. Otherwise a re-enqueued live run would be resubmitted and its running directory replaced.
- It writes the run folder in a hidden staging directory, flushes it to disk, then publishes it with one atomic rename, so the cron driver never picks up a half-written folder.
- It validates its inputs: safe identifiers, `solver` in `GX`/`GENE`, `fidelity` in `low`/`high`, finite `surrogate_epistemic_std >= 0`, `0 <= beta_target < 1`.

The generated GX/GENE input decks are **templates pending physics sign-off**. Every value awaiting the physics team is marked in the code and tracked in `workstreams/kr2-gk-database/docs/OPEN_QUESTIONS.md` (section 5). The GX deck carries placeholder species tables (`[species]`/`[Boltzmann]`, so GX starts with well-defined species); the GENE deck still has no `&species`. Real gradients, temperatures and collisionality must come from the equilibrium/profile data. Do not point the function at a production queue before the decks are confirmed.

## Not implemented yet (deferred to Phase 3)

The active-learning loop itself: acquisition function (e.g. UCB/EI on surrogate uncertainty), batch selection with diversity and de-duplication across design points, budget enforcement and reproducible seeding. These depend on the KR1 surrogates and on a candidate design-space representation that do not exist yet. Until then, callers that want one validation per design point should derive `run_id` deterministically from that design point.
