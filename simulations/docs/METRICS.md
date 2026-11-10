# KR2 Telemetry & KPI Definitions (`METRICS.md`)

All KPIs are computed deterministically in SQL inside [`02_views.sql`](../../../platform/telemetry/02_views.sql) and shared identically between BigQuery (`DATA_MODE=bigquery`, authoritative) and local DuckDB (`DATA_MODE=duckdb`). `GET /v1/events/summary` on aurora-steer evaluates the same views over the serving instance's own in-memory store only (a debug aid; on Cloud Run it is not the campaign-wide picture).

## 1. Observed KPIs (Directly Measured)

| Metric | SQL Column (`v_kpis_campaign`) | Exact Formula |
| :--- | :--- | :--- |
| **Completed Runs** | `completed_runs` | `COUNT(DISTINCT run_id)` where `current_status = 'completed'` |
| **Hands-Free Rate** | `hands_free_rate` | `hands_free_completed_runs / finished_runs` (0 escalations, 0 human resolutions) |
| **Agent Recovery Rate** | `agent_recovery_rate` | `agent_recovered_runs / recoverable_failed_runs` (excludes unrecoverable `DIVERGENCE` escalations) |
| **Escalation Rate** | `escalation_rate` | `escalated_runs / failed_runs` |
| **LLM Calls / Failure** | `llm_calls_per_failed_run` | `llm_calls_count / failed_events_count` (enforced `<= 1.0` via `Idempotency-Key` + 0-LLM continuations) |

## 2. Estimated Savings KPIs (`assumptions` Table — Tagged `TODO(david)`)

- **Estimated GPU Hours Saved (`estimated_gpu_hours_saved`)**: Sum over early-cancelled (`scancel` / Cloud Batch `cancel`) failed runs of `max(0, allocated_walltime_hours - elapsed_hours) * gpus_per_run`.
- **Estimated Human Hours Avoided (`estimated_human_hours_avoided`)**: `(agent_recovered_runs * minutes_per_manual_fix + walltime_resubmit_events * minutes_per_walltime_resubmit) / 60`.
