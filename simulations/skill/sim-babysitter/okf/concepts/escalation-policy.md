---
id: escalation-policy
triggers: ["ALL"]
---

# Escalation & Checkpoint Policy

1. **3-Strike Circuit Breaker**: Each run allows at most 2 automated numerical recovery patches (`retry_count` 0 and 1). When `retry_count >= 2`, `aurora-steer` and the local watchdog immediately return `escalate_to_human` with **0 LLM calls**. `aurora-steer` additionally caps LLM triage calls per run (default 6 per 24 h) so a client that fails to count retries still escalates.
2. **Runbook Action Policy**: For each failure type, only the actions listed in that runbook's `allowed_actions` (`manifest.yaml`) are accepted; everything else, and every guardrail violation, is replaced by `escalate_to_human`. `escalate_to_human` is always allowed and must carry no parameter changes.
3. **Walltime & Spot Preemption Continuations**: Slurm `TIMEOUT` on TACC and Spot VM `PREEMPTED` events on GCP Cloud Batch are handled locally via `resubmit_walltime` from NetCDF restart files (`.restart.nc` configured by `[Restart] restart_to_file`) / GENE `checkpoint_*` (`0` LLM calls, `0` physics strikes, bounded by `max_walltime_resubmits`). `/triage` never returns `resubmit_walltime`.
4. **Warm vs. Cold Restart**: Only parameters marked `warm_restart_allowed` in `allowed-parameters` (`dt_max`, `cfl_fraction`) may change on a checkpoint restart. Grid, box-size and dissipation changes require `cold_restart` with `restart_from_checkpoint = false`.
5. **Human Handoff (`ISSUE.md`)**: Escalated runs move to `queue/needs_human/<run_id>/` with a complete `ISSUE.md` summarizing failure metrics, log tail, and prior directives so Prof. David Hatch's group can inspect and optionally run `tacc-resolve`.
