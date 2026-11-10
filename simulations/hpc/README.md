# `simulations/hpc` — HPC Simulation Driver (`tacc-exec`)

Deterministic simulation driver and watchdog for GENE (Fortran namelists via `f90nml`) and GX (TOML via `tomlkit`).

- **Lightweight TACC login-node footprint:** imports only the Python stdlib + `ut-hpc-core`, `f90nml`, `tomlkit` and `requests`. Runs on TACC Vista / Lonestar6 login nodes over outbound HTTPS (`<2 KB` payloads).
- **Dual-backend shared engine:** `SimulationWatchdog` and `QueueDriver` depend only on `hpc_core.SchedulerBackend`, so the same state machine, guardrails and patchers steer **TACC Slurm** (`SlurmBackend`) and **GCP Cloud Batch / GKE** (`../gcp/`).
- **GENE licensing:** GENE is never bundled or redistributed by this repository. On TACC it runs from the operator's own authorized installation under the [GENE User Agreement](https://genecode.org/).

## Run lifecycle

```
pending/ --(validate)--> SUBMITTING --(scheduler accepted)--> RUNNING --(converged + harvested)--> completed/
                 \                         \--(TIMEOUT / PREEMPTED / NODE_FAIL: checkpoint continuation, 0 LLM calls)
                  \                         \--(physics failure: aurora-steer directive, guarded patch, resubmit)
                   \--(invalid / duplicate)---> needs_human/quarantined_<name>/ISSUE.md
any active state --(guardrail, 3-strike breaker, unknown outcome, external cancel, ...)--> needs_human/<run_id>/ISSUE.md
```

Crash-safety guarantees (see `tacc_exec/watchdog.py` module docstring):

- Every scheduler submission is preceded by a durable `SUBMITTING` record with a unique `submission_id` (`bsit-<run_id>-<nonce>-<seq>`, also used as the Slurm job name). After a crash, the next tick adopts the job (`squeue/sacct --name`), resubmits idempotently, or escalates. It never resubmits blindly.
- No run directory is ever deleted. Name collisions (duplicate `run_id`, an existing `completed/<run_id>`) are quarantined or escalated.
- A run directory is never patched or resubmitted while its job may still be active. Mid-run failures `scancel` and wait for a terminal state; escalations cancel a possibly-active job before moving its folder; `tacc-resolve` refuses to re-queue while the previous job is still queued or running.
- All writes (`state.json`, manifests, patched inputs, `ISSUE.md`) are temp file + `fsync` + atomic rename. A corrupted `state.json` stops the driver with exit code 2 instead of being replaced; never delete it while jobs are queued.

## Deploying on a TACC login node

1. `make install-tacc` (creates `.venv-tacc` with verified dependencies) and copy `config.example.json` to `config.json`.
2. Edit `config.json`:
   - Use **absolute paths** on `$WORK` for `queue_dir`, `state_dir` and `dataset_output_dir`. Relative paths resolve against the directory of `config.json`, never the cron working directory. `$VARS` that are unset under cron are rejected.
   - Do **not** put `dataset_output_dir` or `state_dir` on `$SCRATCH`: TACC purges files that have not been accessed for about 10 days. `tacc-preflight` warns about this.
   - `allowed_parameters` must mirror `skill/sim-babysitter/okf/concepts/allowed-parameters.md` (names, bounds, `gene_path`, `gx_path`, `warm_restart_allowed`). The driver refuses to start if a path disagrees with the patchers.
3. Export the bearer token (never commit it): `export AURORA_STEER_BEARER_TOKEN=...`. For cron, keep it in a `chmod 600` env file sourced by the crontab line.
4. `tacc-preflight --config config.json` (prints warnings), then `python3 scripts/tacc_smoke_test.py`.
5. **Pin cron to one login node.** `vista.tacc.utexas.edu` / `ls6.tacc.utexas.edu` are round-robin DNS over several login nodes, each with its own crontab spool, and Slurm `scrontab` is not offered to users. SSH to one node explicitly and install the crontab there:

   ```bash
   ssh login1.vista.tacc.utexas.edu
   crontab -e
   */5 * * * * . $HOME/.bsit_env && TACC_EXPECTED_LOGIN_HOST=login1 /abs/path/to/simulations/hpc/scripts/ensure_running_cron.sh
   ```

   `ensure_running_cron.sh` exits on any other host, uses `flock -n` against overlapping ticks, and logs to `state/cron.log` (rotated at 50 MB). The driver's own lock (a `flock` plus a lease file with heartbeat on the shared filesystem) also refuses a second driver started from another login node. A lease left by a dead host is taken over after `lock_stale_after_seconds`.
6. Slurm batch scripts must load their own modules: jobs inherit cron's minimal environment.

## Config keys (besides the ones in `config.example.json` defaults)

| Key | Default | Meaning |
|---|---|---|
| `max_retries` | 2 | Aurora-steer patch attempts before the local 3-strike breaker escalates |
| `max_walltime_resubmits` | 5 | Zero-LLM checkpoint continuations (TIMEOUT / PREEMPTED / NODE_FAIL / BOOT_FAIL) |
| `max_unknown_polls` | 10 | Consecutive `UNKNOWN` scheduler answers before escalation (job cancelled first) |
| `max_poll_errors` | 3 | Consecutive poll exceptions before escalation (persisted across cron runs) |
| `max_triage_deferrals` | 3 | Ticks to retry an unreachable aurora-steer (same idempotency key) before escalating |
| `max_pending_hours` | 168 | Escalate a job that has been pending longer than this (job cancelled first) |
| `cancel_wait_seconds` / `cancel_timeout_seconds` | 30 / 1800 | Wait for a cancelled job to terminate in the same tick / before escalating |
| `lock_stale_after_seconds` | 1800 | Lease age after which another login node may take the driver lock over |
| `require_equilibrium_id` | `true` in prod | Quarantine pending runs whose `run_meta.json` lacks `equilibrium_id` |
| `dt_collapse_threshold`, `dt_min_floor`, `transient_discard_fraction`, `max_flux_drift`, `high_k_tail_threshold`, `convergence_time_fraction` | see `DiagnosticsSettings` | Detection and convergence thresholds (docs/review/DOMAIN.md; physics sign-off pending in `../docs/OPEN_QUESTIONS.md`) |

## `run_meta.json` (per run folder)

`equilibrium_id` (required in prod), `campaign_id` (≤ 64 chars), `fidelity` (`low`/`high`), optional `sim_time_target`, `code_version` and `equilibrium_origin` (e.g. `ToBlick/mrx@<commit>`). These values feed the dataset manifest's provenance and eligibility. Identifiers must match `[A-Za-z0-9][A-Za-z0-9_.-]*`.
