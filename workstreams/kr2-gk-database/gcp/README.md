# `workstreams/kr2-gk-database/gcp`: GX on GCP Cloud Batch and GKE Compute Classes

This package runs containerized **GX** gyrokinetic simulations on **Google Cloud Batch** (`CloudBatchBackend`) or on **GKE with Custom Compute Classes** (`GKEComputeClassBackend`). Runs are babysat by the same `SimulationWatchdog`, `QueueDriver` and `aurora-steer` guardrails that are used on TACC.

## Components
- **`gcp-gx-babysitter`** (`gcp_exec/runner.py`) runs one tick per Cloud Run job execution (Cloud Scheduler).
  - It validates the config and fails closed on:
    - a missing worker service account;
    - a `:latest` or untagged image in prod;
    - `cpu_only` or `GX_SIMULATE_RUN` in prod;
    - allowlist drift.
  - It holds the executor's heartbeated lease lock, so overlapping executions skip the tick.
- **Idempotent submission** (`gcp_exec/submission.py`):
  - The job id is a pure function of the watchdog's submission id.
  - Batch creates are sent with `requestId`. A 409 adopts the existing job, after a label check.
  - A write-ahead ledger (`.gcp_submission.json`) and a fencing token (`.gcp_active_attempt`) in the run directory guarantee at most one active job per run.
- **`gx-task`** (`gcp_exec/task_runner.py`) is the container entrypoint for GX jobs.
  - It stages the run directory to local disk and runs GX as a child process.
  - It publishes `gx.log`, outputs and checkpoints to the GCS FUSE mount, but only:
    - while the fence names its own attempt;
    - for checkpoints, once they are stable and pass validation (`gcp_exec/checkpoint.py`).
  - Publishing goes through a temp file and an atomic replace, keeping `gx.prev.restart.nc`.
  - It validates the checkpoint before a restart and fails closed (exit 65) if it is invalid.
  - On SIGTERM (Spot preemption) it publishes what is safe, then stops GX.
  - It writes `task_result.json` (outcome, GX build commit, restart decision).
  - It always refuses `GX_SIMULATE_RUN`, and refuses an unverified GX build in prod.
- **`gx-diagnose`** (`gcp_exec/gx_diagnostics.py`) turns `gx.log` / `*.out.nc` into the diagnostics JSON consumed by the watchdog. Thresholds follow docs/review/DOMAIN.md.

## Spend controls (bounded by design)
- **Cloud Batch.** The default waterfall is Spot L4 in each candidate region, then Spot T4 when the run fits in 13.5 GB of VRAM. Opt-ins:
  - `allow_a100_fallback=true` adds Spot A100 on stockouts.
  - `allow_on_demand=true` adds on-demand STANDARD capacity.
  - Past the last step, submission is refused and the run escalates to a human.
- **When the waterfall advances.** The region/SKU step advances only when a job never obtained capacity: a stockout, or a job stuck in QUEUED/SCHEDULED for longer than `max_pending_s` (default 1800 s). Preemption continuations stay on the same step.
- **GKE.** The default Compute Classes in `k8s/compute-classes.yaml` are Spot-only. The `<class>-ondemand` variants (DWS flex-start and on-demand) are selected only with `allow_on_demand=true`. Finished Jobs are garbage-collected after `ttlSecondsAfterFinished` (7 days).
- **Caps.** Continuations are capped by `max_walltime_resubmits`, LLM-steered retries by `max_retries`, and each job by `max_run_duration_s`.

## Configuration
See `config.batch.example.json`. The `allowed_parameters` block mirrors `skill/sim-babysitter/okf/concepts/allowed-parameters.md` and `hpc/config.example.json`, and must stay identical to them.
- **Custom VPC & private Batch networking (`network`, `subnetwork`, `no_external_ip`):** GCP projects enforcing `constraints/compute.skipDefaultNetworkCreation` and `constraints/compute.vmExternalIpAccess = DENY` have no `default` VPC and forbid public VM IPs. Set `network` (`projects/<project>/global/networks/<vpc>`), `subnetwork` (`projects/<project>/regions/<region>/subnetworks/<subnet>`, or `projects/<project>/regions/{region}/subnetworks/<subnet>` when subnets exist in every `candidate_regions` entry), and `no_external_ip: true` (default). `CloudBatchBackend` attaches `allocationPolicy.network.networkInterfaces` with `noExternalIpAddress: true` and fails closed if a regional `subnetwork` conflicts with `region` or `candidate_regions`. The subnet must enable Private Google Access and Cloud NAT (provisioned by `platform/infra/terraform`).
- **GKE:** Use `"backend": "gcp_k8s"` for GKE. In-cluster use needs no extra settings; from Cloud Run, set `k8s_api_url`, `k8s_ca_cert_path` and `"k8s_auth": "google"`.

## Building the gx-runner image
```bash
docker build -f workstreams/kr2-gk-database/gcp/Dockerfile.gx-runner \
  --build-arg GX_COMMIT=<full 40-hex GX commit SHA> -t <registry>/gx-runner:gx<sha12> .
```
- **`GX_COMMIT` is required and has no default.** GX publishes no release tags; it is developed on `main` (https://gx.readthedocs.io). Use the commit that the physics team (Prof. Hatch's group) validated for the campaign. The build fails unless HEAD equals that commit, and it records the commit in `/opt/gx/gx_build.json`. `gx-task` copies that file into each run's `task_result.json`.
- **Base image:** CUDA 12.6.3 on Ubuntu 24.04 (Python 3.12). GX ships SASS for T4 (sm_75), A100 (sm_80) and L4 (sm_89), so R535+ drivers work through CUDA minor-version compatibility; R560+ is native. GKE Compute Classes request `driverVersion: latest`.
- **Runtime user:** UID/GID 10001. Batch mounts the bucket with `--uid=10001 --gid=10001 --file-mode=664 --dir-mode=775 --o=allow_other` (`DEFAULT_GCS_MOUNT_OPTIONS`). GKE uses the CSI `mountOptions` `uid=10001,gid=10001,...`. The Cloud Run babysitter job mounts the bucket with `uid=10001,gid=10001`.
- **Not yet built end to end:** the image has never been built against upstream GX. The build prints diagnostics and fails early if `Makefiles/Makefile.$GK_SYSTEM` or a dependency (MPI, cuTENSOR, NCCL, netCDF) is missing.

## Benchmarks
`benchmarks/` holds the Cyclone Base Case GX decks. Their `run_meta.json` declares `sim_time_target` = `t_max` = 400.
