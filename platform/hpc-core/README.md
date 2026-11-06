# `platform/hpc-core` (`ut-hpc-core`)

Shared TACC-safe infrastructure primitives used across HPC/Cloud executors.

## Lightweight Zero-AI TACC Footprint
- **Allowed third-party dependency:** `requests` ONLY (plus Python 3.11+ standard library).
- **Zero** AI, LLM, GCP, or web framework packages (`google-genai`, `google-cloud-*`, `fastapi`, `pydantic`, etc.).

## Modules
- `hpc_core.protocol`: `SchedulerBackend` protocol (`submit`, `query_job`, `cancel`), `JobStatus` (incl. `reason`, `blocked`, `cancelled_externally`) and the optional idempotent-submission contract (`BackendCapabilities.idempotent_submit`, `submit(..., submission_id=...)`, `find_submission`).
- `hpc_core.slurm`: `SlurmBackend` (`sbatch`, `squeue`, `sacct`, `scancel`) with state normalization and `--dry-run`.
- `hpc_core.state`: crash-safe `StateStore` (schema-versioned, fails closed on corruption), `RunRecord`, fsync'd `audit.jsonl`, and `acquire_single_instance_lock` (`flock` + cross-login-node lease with heartbeat and stale takeover).
- `hpc_core.fsutil`: `atomic_write_*` (temp file + `fsync` + rename + directory `fsync`), `sha256_file`, `is_safe_id`.
- `hpc_core.client`: `AuroraSteerClient` (`<2 KB` payload cap, HTTPS enforcement, no retry on `ReadTimeout`).
- `hpc_core.telemetry`: Durable `outbox.jsonl` emitter (`<1 KB` per event, `workstream` + `environment`, HTTP 422 poison-event quarantine).
- `hpc_core.notify`: Optional outbound webhook notifier for `needs_human` escalations.
- `hpc_core.manifest`: dataset harvester writing `equilibria/<equilibrium_id>/` + `runs/<run_id>/{input, fluxes.json, run_manifest.json, checksums.sha256}` with provenance and a dataset-eligibility verdict (layout extends the provenance of Zenodo record 14867777).
