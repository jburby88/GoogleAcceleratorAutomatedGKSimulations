-- =============================================================================
-- BigQuery Views DDL: `sim_babysitter_telemetry` Views (`platform/telemetry/02_views.sql`)
-- Every view deduplicates on `event_id`/`directive_id`, exposes `workstream` and
-- `environment`, and excludes `preflight`.
--
-- Production-only KPI rule (ML-06): the four aggregate KPI views (`v_compute_savings`,
-- `v_kpis_campaign`, `v_failure_taxonomy`, `v_runbook_effectiveness`) count ONLY rows with
-- `environment = 'prod' AND is_synthetic = FALSE`. A NULL environment never counts as prod.
-- The row-level operational views (`v_run_status`, `v_directive_outcomes`, `v_param_coverage`,
-- `v_timeline`) keep `sim` and `synthetic` rows for debugging, scoped by their
-- `environment` and `is_synthetic` columns, so their consumers must filter on both.
--
-- `${project_id}` is a placeholder: Terraform rewrites it to the target project, and
-- the DuckDB translation in aurora-steer (app/events.py) strips the project and dataset qualifiers.
-- Do not put semicolons inside comments or view bodies (Terraform splits this file on them).
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. `v_run_status`: Latest state and lifecycle summary per `(workstream, campaign_id, run_id, environment, is_synthetic)`
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_run_status` AS
WITH dedup_events AS (
  SELECT
    event_id,
    schema_version,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    COALESCE(backend, 'tacc_slurm') AS backend,
    run_id,
    campaign_id,
    solver,
    fidelity,
    event_type,
    ts_utc,
    retry_count,
    COALESCE(walltime_resubmits, 0) AS walltime_resubmits,
    COALESCE(environment, 'unknown') AS environment,
    is_synthetic,
    payload,
    CASE event_type
      WHEN 'submitted' THEN 1
      WHEN 'running' THEN 2
      WHEN 'progress' THEN 3
      WHEN 'walltime_resubmitted' THEN 4
      WHEN 'resubmit_walltime' THEN 4
      WHEN 'failed' THEN 5
      WHEN 'triage_requested' THEN 6
      WHEN 'triage' THEN 6
      WHEN 'directive_received' THEN 7
      WHEN 'patched' THEN 8
      WHEN 'resubmitted' THEN 9
      WHEN 'escalated' THEN 10
      WHEN 'human_resolved' THEN 11
      WHEN 'completed' THEN 12
      ELSE 0
    END AS event_rank
  FROM `${project_id}.sim_babysitter_telemetry.events`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND COALESCE(environment, 'unknown') != 'preflight'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ts_utc ASC, ingested_at_utc ASC) = 1
),
dedup_directives AS (
  SELECT
    directive_id,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    run_id,
    campaign_id,
    COALESCE(environment, 'unknown') AS environment,
    is_synthetic,
    action,
    llm_called
  FROM `${project_id}.sim_babysitter_telemetry.directives`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND COALESCE(environment, 'unknown') != 'preflight'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY directive_id ORDER BY ts_utc ASC) = 1
),
dir_counts AS (
  SELECT
    workstream,
    campaign_id,
    run_id,
    environment,
    is_synthetic,
    COUNT(*) AS directive_table_count,
    SUM(CASE WHEN llm_called THEN 1 ELSE 0 END) AS llm_calls_for_run
  FROM dedup_directives
  GROUP BY workstream, campaign_id, run_id, environment, is_synthetic
),
latest_state_event AS (
  SELECT
    workstream,
    campaign_id,
    run_id,
    environment,
    is_synthetic,
    event_type AS latest_state_event_type,
    ts_utc AS latest_state_ts_utc
  FROM dedup_events
  WHERE event_type IN ('submitted', 'running', 'resubmitted', 'walltime_resubmitted', 'resubmit_walltime', 'failed', 'escalated', 'human_resolved', 'completed')
  QUALIFY ROW_NUMBER() OVER (PARTITION BY workstream, campaign_id, run_id, environment, is_synthetic ORDER BY ts_utc DESC, event_rank DESC) = 1
),
latest_param_event AS (
  SELECT
    workstream,
    campaign_id,
    run_id,
    environment,
    is_synthetic,
    JSON_QUERY(payload, '$.current_parameters') AS latest_parameters_json,
    COALESCE(JSON_VALUE(payload, '$.equilibrium_id'), JSON_VALUE(payload, '$.geometry_id')) AS geometry_id
  FROM dedup_events
  WHERE JSON_QUERY(payload, '$.current_parameters') IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY workstream, campaign_id, run_id, environment, is_synthetic ORDER BY ts_utc DESC, event_rank DESC) = 1
),
latest_metric_event AS (
  SELECT
    workstream,
    campaign_id,
    run_id,
    environment,
    is_synthetic,
    JSON_QUERY(payload, '$.metrics') AS latest_metrics_json
  FROM dedup_events
  WHERE JSON_QUERY(payload, '$.metrics') IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY workstream, campaign_id, run_id, environment, is_synthetic ORDER BY ts_utc DESC, event_rank DESC) = 1
),
latest_escalation_event AS (
  SELECT
    workstream,
    campaign_id,
    run_id,
    environment,
    is_synthetic,
    JSON_VALUE(payload, '$.issue_md_text') AS issue_md_text,
    JSON_QUERY(payload, '$.directives_tried') AS directives_tried_json,
    COALESCE(JSON_VALUE(payload, '$.tacc_folder_path'), CONCAT('queue/needs_human/', run_id)) AS tacc_folder_path,
    JSON_VALUE(payload, '$.reason') AS escalation_reason
  FROM dedup_events
  WHERE event_type = 'escalated'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY workstream, campaign_id, run_id, environment, is_synthetic ORDER BY ts_utc DESC, event_rank DESC) = 1
),
run_agg AS (
  SELECT
    e.workstream,
    e.campaign_id,
    e.run_id,
    e.environment,
    e.is_synthetic,
    ANY_VALUE(e.backend) AS backend,
    ANY_VALUE(e.solver) AS solver,
    ANY_VALUE(e.fidelity) AS fidelity,
    COALESCE(MAX(COALESCE(JSON_VALUE(e.payload, '$.equilibrium_id'), JSON_VALUE(e.payload, '$.geometry_id'))), 'w7x_standard') AS fallback_geometry_id,
    MIN(e.ts_utc) AS first_seen_ts_utc,
    MAX(e.ts_utc) AS last_updated_ts_utc,
    MAX(CASE WHEN e.event_type = 'completed' THEN e.ts_utc END) AS completed_ts_utc,
    MAX(e.retry_count) AS max_retry_count,
    SUM(CASE WHEN e.event_type = 'failed' THEN 1 ELSE 0 END) AS failed_count,
    SUM(CASE WHEN e.event_type = 'patched' THEN 1 ELSE 0 END) AS patched_count,
    SUM(CASE WHEN e.event_type IN ('directive_received', 'triage') THEN 1 ELSE 0 END) AS directive_event_count,
    GREATEST(
      SUM(CASE WHEN e.event_type IN ('walltime_resubmitted', 'resubmit_walltime') THEN 1 ELSE 0 END),
      MAX(e.walltime_resubmits)
    ) AS walltime_resubmits,
    SUM(CASE WHEN e.event_type = 'escalated' THEN 1 ELSE 0 END) AS escalated_count,
    SUM(CASE WHEN e.event_type = 'human_resolved' THEN 1 ELSE 0 END) AS human_resolved_count,
    SUM(CASE WHEN e.event_type = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    SUM(CASE WHEN e.event_type = 'human_resolved' THEN COALESCE(CAST(JSON_VALUE(e.payload, '$.minutes') AS FLOAT64), 0.0) ELSE 0.0 END) AS human_minutes_logged,
    MAX(CASE WHEN e.event_type = 'failed' THEN JSON_VALUE(e.payload, '$.failure_type') END) AS last_failure_type,
    MAX(CASE WHEN e.event_type = 'failed' AND JSON_VALUE(e.payload, '$.failure_type') = 'DIVERGENCE' THEN 1 ELSE 0 END) AS has_divergence_failure
  FROM dedup_events e
  GROUP BY e.workstream, e.campaign_id, e.run_id, e.environment, e.is_synthetic
)
SELECT
  r.workstream,
  r.campaign_id,
  r.run_id,
  r.is_synthetic,
  r.environment,
  r.backend,
  r.solver,
  r.fidelity,
  COALESCE(lp.geometry_id, r.fallback_geometry_id) AS geometry_id,
  CASE
    WHEN r.completed_count >= 1 AND (ls.latest_state_event_type = 'completed' OR r.completed_ts_utc >= ls.latest_state_ts_utc) THEN 'completed'
    WHEN ls.latest_state_event_type = 'escalated' THEN 'escalated'
    WHEN ls.latest_state_event_type = 'human_resolved' THEN 'human_resolved'
    WHEN ls.latest_state_event_type = 'failed' THEN 'failed'
    WHEN ls.latest_state_event_type IN ('running', 'resubmitted', 'walltime_resubmitted', 'resubmit_walltime') THEN 'running'
    ELSE 'submitted'
  END AS current_status,
  r.first_seen_ts_utc,
  r.last_updated_ts_utc,
  r.completed_ts_utc,
  r.max_retry_count AS retry_count,
  r.failed_count,
  r.patched_count,
  GREATEST(r.directive_event_count, COALESCE(dc.directive_table_count, 0)) AS directive_count,
  COALESCE(dc.llm_calls_for_run, 0) AS llm_calls_count,
  r.walltime_resubmits,
  r.escalated_count,
  r.human_resolved_count,
  r.human_minutes_logged,
  r.last_failure_type,
  (r.has_divergence_failure = 1) AS is_divergence_escalated,
  (
    r.completed_count >= 1
    AND r.failed_count >= 1
    AND (r.patched_count >= 1 OR r.directive_event_count >= 1 OR COALESCE(dc.directive_table_count, 0) >= 1)
    AND r.human_resolved_count = 0
  ) AS is_agent_recovered,
  (
    r.completed_count >= 1
    AND r.human_resolved_count = 0
    AND r.escalated_count = 0
  ) AS is_hands_free_completed,
  lm.latest_metrics_json,
  lp.latest_parameters_json,
  le.issue_md_text,
  le.directives_tried_json,
  le.tacc_folder_path,
  le.escalation_reason
FROM run_agg r
LEFT JOIN latest_state_event ls
  ON r.workstream = ls.workstream AND r.campaign_id = ls.campaign_id AND r.run_id = ls.run_id AND r.environment = ls.environment AND r.is_synthetic = ls.is_synthetic
LEFT JOIN latest_param_event lp
  ON r.workstream = lp.workstream AND r.campaign_id = lp.campaign_id AND r.run_id = lp.run_id AND r.environment = lp.environment AND r.is_synthetic = lp.is_synthetic
LEFT JOIN latest_metric_event lm
  ON r.workstream = lm.workstream AND r.campaign_id = lm.campaign_id AND r.run_id = lm.run_id AND r.environment = lm.environment AND r.is_synthetic = lm.is_synthetic
LEFT JOIN latest_escalation_event le
  ON r.workstream = le.workstream AND r.campaign_id = le.campaign_id AND r.run_id = le.run_id AND r.environment = le.environment AND r.is_synthetic = le.is_synthetic
LEFT JOIN dir_counts dc
  ON r.workstream = dc.workstream AND r.campaign_id = dc.campaign_id AND r.run_id = dc.run_id AND r.environment = dc.environment AND r.is_synthetic = dc.is_synthetic;


-- -----------------------------------------------------------------------------
-- 2. `v_compute_savings`: Estimated GPU hours saved & human hours avoided (prod only)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_compute_savings` AS
WITH dedup_events AS (
  SELECT
    event_id,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    run_id,
    campaign_id,
    solver,
    fidelity,
    event_type,
    ts_utc,
    environment,
    is_synthetic,
    payload
  FROM `${project_id}.sim_babysitter_telemetry.events`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND environment = 'prod'
    AND is_synthetic = FALSE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ts_utc ASC, ingested_at_utc ASC) = 1
),
assumptions_pivoted AS (
  SELECT
    MAX(CASE WHEN assumption_key = 'minutes_per_manual_fix' THEN numeric_value END) AS minutes_per_manual_fix,
    MAX(CASE WHEN assumption_key = 'minutes_per_manual_fix' THEN source END) AS minutes_per_manual_fix_source,
    MAX(CASE WHEN assumption_key = 'minutes_per_walltime_resubmit' THEN numeric_value END) AS minutes_per_walltime_resubmit,
    MAX(CASE WHEN assumption_key = 'minutes_per_walltime_resubmit' THEN source END) AS minutes_per_walltime_resubmit_source,
    MAX(CASE WHEN assumption_key = 'allocated_walltime_hours_gene_low' THEN numeric_value END) AS walltime_h_gene_low,
    MAX(CASE WHEN assumption_key = 'allocated_walltime_hours_gene_high' THEN numeric_value END) AS walltime_h_gene_high,
    MAX(CASE WHEN assumption_key = 'allocated_walltime_hours_gx_low' THEN numeric_value END) AS walltime_h_gx_low,
    MAX(CASE WHEN assumption_key = 'allocated_walltime_hours_gx_high' THEN numeric_value END) AS walltime_h_gx_high,
    MAX(CASE WHEN assumption_key = 'gpus_per_run_gene_low' THEN numeric_value END) AS gpus_gene_low,
    MAX(CASE WHEN assumption_key = 'gpus_per_run_gene_high' THEN numeric_value END) AS gpus_gene_high,
    MAX(CASE WHEN assumption_key = 'gpus_per_run_gx_low' THEN numeric_value END) AS gpus_gx_low,
    MAX(CASE WHEN assumption_key = 'gpus_per_run_gx_high' THEN numeric_value END) AS gpus_gx_high
  FROM `${project_id}.sim_babysitter_telemetry.assumptions`
),
early_cancel_events AS (
  SELECT
    e.workstream,
    e.campaign_id,
    e.environment,
    e.is_synthetic,
    e.solver,
    e.fidelity,
    e.run_id,
    COALESCE(CAST(JSON_VALUE(e.payload, '$.walltime_elapsed_s') AS FLOAT64), 0.0) / 3600.0 AS elapsed_hours,
    CASE
      WHEN e.solver = 'GENE' AND e.fidelity = 'high' THEN COALESCE(a.walltime_h_gene_high, 16.0)
      WHEN e.solver = 'GENE' THEN COALESCE(a.walltime_h_gene_low, 4.0)
      WHEN e.solver = 'GX' AND e.fidelity = 'high' THEN COALESCE(a.walltime_h_gx_high, 8.0)
      ELSE COALESCE(a.walltime_h_gx_low, 2.0)
    END AS allocated_walltime_hours,
    CASE
      WHEN e.solver = 'GENE' AND e.fidelity = 'high' THEN COALESCE(a.gpus_gene_high, 16.0)
      WHEN e.solver = 'GENE' THEN COALESCE(a.gpus_gene_low, 4.0)
      WHEN e.solver = 'GX' AND e.fidelity = 'high' THEN COALESCE(a.gpus_gx_high, 4.0)
      ELSE COALESCE(a.gpus_gx_low, 1.0)
    END AS gpus_per_run
  FROM dedup_events e
  CROSS JOIN assumptions_pivoted a
  WHERE e.event_type = 'failed'
    AND (
      JSON_VALUE(e.payload, '$.early_cancelled') = 'true'
      OR JSON_VALUE(e.payload, '$.slurm_state') = 'RUNNING'
    )
),
gpu_savings_by_campaign AS (
  SELECT
    workstream,
    campaign_id,
    environment,
    is_synthetic,
    COUNT(*) AS early_cancelled_failures,
    SUM(GREATEST(0.0, allocated_walltime_hours - elapsed_hours) * gpus_per_run) AS estimated_gpu_hours_saved,
    SUM(CASE WHEN solver = 'GENE' THEN GREATEST(0.0, allocated_walltime_hours - elapsed_hours) * gpus_per_run ELSE 0.0 END) AS gene_gpu_hours_saved,
    SUM(CASE WHEN solver = 'GX' THEN GREATEST(0.0, allocated_walltime_hours - elapsed_hours) * gpus_per_run ELSE 0.0 END) AS gx_gpu_hours_saved
  FROM early_cancel_events
  GROUP BY workstream, campaign_id, environment, is_synthetic
),
run_summary AS (
  SELECT
    workstream,
    campaign_id,
    environment,
    is_synthetic,
    SUM(CASE WHEN is_agent_recovered THEN 1 ELSE 0 END) AS agent_recovered_runs,
    SUM(walltime_resubmits) AS walltime_resubmit_events,
    SUM(human_minutes_logged) AS human_minutes_logged
  FROM `${project_id}.sim_babysitter_telemetry.v_run_status`
  WHERE environment = 'prod'
    AND is_synthetic = FALSE
  GROUP BY workstream, campaign_id, environment, is_synthetic
)
SELECT
  rs.workstream,
  rs.campaign_id,
  rs.environment,
  rs.is_synthetic,
  COALESCE(gs.early_cancelled_failures, 0) AS early_cancelled_failures,
  ROUND(COALESCE(gs.estimated_gpu_hours_saved, 0.0), 2) AS estimated_gpu_hours_saved,
  ROUND(COALESCE(gs.gene_gpu_hours_saved, 0.0), 2) AS gene_gpu_hours_saved,
  ROUND(COALESCE(gs.gx_gpu_hours_saved, 0.0), 2) AS gx_gpu_hours_saved,
  rs.agent_recovered_runs,
  rs.walltime_resubmit_events,
  ROUND(
    (
      (rs.agent_recovered_runs * COALESCE(a.minutes_per_manual_fix, 45.0))
      + (rs.walltime_resubmit_events * COALESCE(a.minutes_per_walltime_resubmit, 10.0))
    ) / 60.0,
    2
  ) AS estimated_human_hours_avoided,
  ROUND(rs.human_minutes_logged / 60.0, 2) AS actual_human_hours_logged,
  COALESCE(a.minutes_per_manual_fix, 45.0) AS minutes_per_manual_fix,
  COALESCE(a.minutes_per_manual_fix_source, 'TODO(david): confirm baseline manual troubleshooting duration') AS minutes_per_manual_fix_source,
  COALESCE(a.minutes_per_walltime_resubmit, 10.0) AS minutes_per_walltime_resubmit,
  COALESCE(a.minutes_per_walltime_resubmit_source, 'TODO(david): confirm manual walltime continuation overhead') AS minutes_per_walltime_resubmit_source
FROM run_summary rs
CROSS JOIN assumptions_pivoted a
LEFT JOIN gpu_savings_by_campaign gs
  ON rs.workstream = gs.workstream AND rs.campaign_id = gs.campaign_id AND rs.environment = gs.environment AND rs.is_synthetic = gs.is_synthetic;


-- -----------------------------------------------------------------------------
-- 3. `v_kpis_campaign`: Single-row-per-campaign KPI summary (prod only)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_kpis_campaign` AS
WITH run_stats AS (
  SELECT
    r.workstream,
    r.campaign_id,
    r.environment,
    r.is_synthetic,
    COUNT(DISTINCT r.run_id) AS total_runs,
    COUNT(DISTINCT CASE WHEN r.current_status = 'completed' THEN r.run_id END) AS completed_runs,
    COUNT(DISTINCT CASE WHEN r.current_status = 'running' THEN r.run_id END) AS running_runs,
    COUNT(DISTINCT CASE WHEN r.current_status = 'escalated' THEN r.run_id END) AS currently_escalated_runs,
    COUNT(DISTINCT CASE WHEN r.current_status IN ('completed', 'escalated') THEN r.run_id END) AS finished_runs,
    COUNT(DISTINCT CASE WHEN r.current_status = 'completed' AND r.human_resolved_count = 0 AND r.escalated_count = 0 THEN r.run_id END) AS hands_free_completed_runs,
    COUNT(DISTINCT CASE WHEN r.is_agent_recovered THEN r.run_id END) AS agent_recovered_runs,
    COUNT(DISTINCT CASE WHEN r.failed_count >= 1 AND NOT r.is_divergence_escalated THEN r.run_id END) AS recoverable_failed_runs,
    COUNT(DISTINCT CASE WHEN r.failed_count >= 1 THEN r.run_id END) AS failed_runs,
    COUNT(DISTINCT CASE WHEN r.escalated_count >= 1 AND r.failed_count >= 1 THEN r.run_id END) AS escalated_runs,
    SUM(r.failed_count) AS failed_events_count,
    SUM(r.walltime_resubmits) AS walltime_resubmit_events,
    COUNT(DISTINCT CASE WHEN r.walltime_resubmits >= 1 THEN r.run_id END) AS walltime_resubmit_runs,
    COUNT(DISTINCT CASE WHEN r.solver = 'GENE' AND r.current_status = 'completed' THEN r.run_id END) AS completed_gene_runs,
    COUNT(DISTINCT CASE WHEN r.solver = 'GX' AND r.current_status = 'completed' THEN r.run_id END) AS completed_gx_runs,
    COUNT(DISTINCT CASE WHEN r.fidelity = 'low' AND r.current_status = 'completed' THEN r.run_id END) AS completed_low_fidelity_runs,
    COUNT(DISTINCT CASE WHEN r.fidelity = 'high' AND r.current_status = 'completed' THEN r.run_id END) AS completed_high_fidelity_runs,
    MAX(r.last_updated_ts_utc) AS ref_ts_utc
  FROM `${project_id}.sim_babysitter_telemetry.v_run_status` r
  WHERE r.environment = 'prod'
    AND r.is_synthetic = FALSE
  GROUP BY r.workstream, r.campaign_id, r.environment, r.is_synthetic
),
trailing_throughput AS (
  SELECT
    r.workstream,
    r.campaign_id,
    r.environment,
    r.is_synthetic,
    COUNT(DISTINCT CASE
      WHEN r.current_status = 'completed'
       AND r.completed_ts_utc >= TIMESTAMP_SUB(s.ref_ts_utc, INTERVAL 14 DAY)
      THEN r.run_id
    END) / 14.0 AS trailing_14d_runs_per_day
  FROM `${project_id}.sim_babysitter_telemetry.v_run_status` r
  INNER JOIN run_stats s
    ON r.workstream = s.workstream AND r.campaign_id = s.campaign_id AND r.environment = s.environment AND r.is_synthetic = s.is_synthetic
  GROUP BY r.workstream, r.campaign_id, r.environment, r.is_synthetic
),
dedup_directives AS (
  SELECT
    directive_id,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    campaign_id,
    environment,
    is_synthetic,
    llm_called
  FROM `${project_id}.sim_babysitter_telemetry.directives`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND environment = 'prod'
    AND is_synthetic = FALSE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY directive_id ORDER BY ts_utc ASC) = 1
),
llm_stats AS (
  SELECT
    workstream,
    campaign_id,
    environment,
    is_synthetic,
    SUM(CASE WHEN llm_called THEN 1 ELSE 0 END) AS llm_calls_count,
    COUNT(*) AS total_directives_count
  FROM dedup_directives
  GROUP BY workstream, campaign_id, environment, is_synthetic
),
latest_campaigns AS (
  -- Campaign metadata is append-only: the newest row per campaign wins, so a re-inserted
  -- target never duplicates the KPI row.
  SELECT
    workstream,
    campaign_id,
    environment,
    is_synthetic,
    target_runs,
    start_date,
    deadline_date
  FROM `${project_id}.sim_babysitter_telemetry.campaigns`
  WHERE environment = 'prod'
    AND is_synthetic = FALSE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY workstream, campaign_id, environment, is_synthetic ORDER BY updated_at_utc DESC) = 1
)
SELECT
  s.workstream,
  s.campaign_id,
  s.is_synthetic,
  s.environment,
  s.total_runs,
  s.completed_runs,
  s.finished_runs,
  COALESCE(c.target_runs, s.total_runs) AS target_runs,
  ROUND(SAFE_DIVIDE(s.completed_runs, COALESCE(c.target_runs, s.total_runs)) * 100.0, 2) AS completion_pct,
  s.running_runs,
  s.currently_escalated_runs,
  s.hands_free_completed_runs,
  ROUND(SAFE_DIVIDE(s.hands_free_completed_runs, s.finished_runs), 4) AS hands_free_rate,
  ROUND(SAFE_DIVIDE(s.hands_free_completed_runs, s.completed_runs), 4) AS hands_free_completed_rate,
  ROUND(SAFE_DIVIDE(s.hands_free_completed_runs, s.finished_runs), 4) AS hands_free_finished_rate,
  s.agent_recovered_runs,
  s.recoverable_failed_runs,
  ROUND(SAFE_DIVIDE(s.agent_recovered_runs, s.recoverable_failed_runs), 4) AS agent_recovery_rate,
  s.escalated_runs,
  s.failed_runs,
  ROUND(SAFE_DIVIDE(s.escalated_runs, s.failed_runs), 4) AS escalation_rate,
  s.walltime_resubmit_events,
  s.walltime_resubmit_runs,
  COALESCE(cs.estimated_gpu_hours_saved, 0.0) AS estimated_gpu_hours_saved,
  COALESCE(cs.estimated_human_hours_avoided, 0.0) AS estimated_human_hours_avoided,
  COALESCE(cs.actual_human_hours_logged, 0.0) AS actual_human_hours_logged,
  COALESCE(cs.minutes_per_manual_fix, 45.0) AS minutes_per_manual_fix,
  COALESCE(cs.minutes_per_manual_fix_source, 'TODO(david): confirm') AS minutes_per_manual_fix_source,
  ROUND(COALESCE(tt.trailing_14d_runs_per_day, 0.0), 3) AS trailing_14d_runs_per_day,
  c.start_date,
  c.deadline_date,
  CASE
    WHEN s.completed_runs >= COALESCE(c.target_runs, s.total_runs) THEN DATE(s.ref_ts_utc)
    WHEN COALESCE(tt.trailing_14d_runs_per_day, 0.0) > 0.0 THEN
      DATE_ADD(
        DATE(s.ref_ts_utc),
        INTERVAL CAST(CEIL((COALESCE(c.target_runs, s.total_runs) - s.completed_runs) / tt.trailing_14d_runs_per_day) AS INT64) DAY
      )
    ELSE NULL
  END AS projected_completion_date,
  CASE
    WHEN c.deadline_date IS NOT NULL AND s.completed_runs >= COALESCE(c.target_runs, s.total_runs) THEN
      DATE_DIFF(c.deadline_date, DATE(s.ref_ts_utc), DAY)
    WHEN c.deadline_date IS NOT NULL AND COALESCE(tt.trailing_14d_runs_per_day, 0.0) > 0.0 THEN
      DATE_DIFF(
        c.deadline_date,
        DATE_ADD(
          DATE(s.ref_ts_utc),
          INTERVAL CAST(CEIL((COALESCE(c.target_runs, s.total_runs) - s.completed_runs) / tt.trailing_14d_runs_per_day) AS INT64) DAY
        ),
        DAY
      )
    ELSE NULL
  END AS days_ahead_of_deadline,
  COALESCE(ls.llm_calls_count, 0) AS llm_calls_count,
  s.failed_events_count,
  ROUND(SAFE_DIVIDE(COALESCE(ls.llm_calls_count, 0), s.failed_events_count), 3) AS llm_calls_per_failed_run,
  s.completed_gene_runs,
  s.completed_gx_runs,
  s.completed_low_fidelity_runs,
  s.completed_high_fidelity_runs
FROM run_stats s
LEFT JOIN latest_campaigns c
  ON s.workstream = c.workstream AND s.campaign_id = c.campaign_id AND s.environment = c.environment AND s.is_synthetic = c.is_synthetic
LEFT JOIN trailing_throughput tt
  ON s.workstream = tt.workstream AND s.campaign_id = tt.campaign_id AND s.environment = tt.environment AND s.is_synthetic = tt.is_synthetic
LEFT JOIN `${project_id}.sim_babysitter_telemetry.v_compute_savings` cs
  ON s.workstream = cs.workstream AND s.campaign_id = cs.campaign_id AND s.environment = cs.environment AND s.is_synthetic = cs.is_synthetic
LEFT JOIN llm_stats ls
  ON s.workstream = ls.workstream AND s.campaign_id = ls.campaign_id AND s.environment = ls.environment AND s.is_synthetic = ls.is_synthetic;


-- -----------------------------------------------------------------------------
-- 4. `v_failure_taxonomy` (prod only)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_failure_taxonomy` AS
WITH dedup_failed_events AS (
  SELECT
    event_id,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    campaign_id,
    run_id,
    solver,
    fidelity,
    environment,
    is_synthetic,
    COALESCE(JSON_VALUE(payload, '$.failure_type'), 'UNKNOWN') AS failure_type
  FROM `${project_id}.sim_babysitter_telemetry.events`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND event_type = 'failed'
    AND environment = 'prod'
    AND is_synthetic = FALSE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ts_utc ASC, ingested_at_utc ASC) = 1
)
SELECT
  f.workstream,
  f.campaign_id,
  f.environment,
  f.is_synthetic,
  f.solver,
  f.fidelity,
  f.failure_type,
  COUNT(*) AS failure_events_count,
  COUNT(DISTINCT f.run_id) AS affected_runs_count,
  COUNT(DISTINCT CASE WHEN r.is_agent_recovered THEN f.run_id END) AS recovered_runs_count,
  COUNT(DISTINCT CASE WHEN r.escalated_count >= 1 THEN f.run_id END) AS escalated_runs_count,
  ROUND(
    SAFE_DIVIDE(
      COUNT(DISTINCT CASE WHEN r.is_agent_recovered THEN f.run_id END),
      COUNT(DISTINCT f.run_id)
    ),
    4
  ) AS recovery_rate_for_type
FROM dedup_failed_events f
LEFT JOIN `${project_id}.sim_babysitter_telemetry.v_run_status` r
  ON f.workstream = r.workstream AND f.campaign_id = r.campaign_id AND f.run_id = r.run_id AND f.environment = r.environment AND f.is_synthetic = r.is_synthetic
GROUP BY f.workstream, f.campaign_id, f.environment, f.is_synthetic, f.solver, f.fidelity, f.failure_type;


-- -----------------------------------------------------------------------------
-- 5. `v_directive_outcomes` (row-level, all non-preflight environments)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_directive_outcomes` AS
WITH dedup_directives AS (
  SELECT
    directive_id,
    schema_version,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    run_id,
    campaign_id,
    solver,
    fidelity,
    ts_utc,
    retry_count,
    COALESCE(environment, 'unknown') AS environment,
    failure_type,
    action,
    reason,
    parameter_modifications_before,
    parameter_modifications_after,
    restart_from_checkpoint,
    confidence,
    okf_concepts_used,
    guardrail_outcome,
    forced_escalation,
    llm_called,
    latency_ms,
    is_synthetic
  FROM `${project_id}.sim_babysitter_telemetry.directives`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND COALESCE(environment, 'unknown') != 'preflight'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY directive_id ORDER BY ts_utc ASC) = 1
)
SELECT
  d.directive_id,
  d.workstream,
  d.campaign_id,
  d.run_id,
  d.environment,
  d.is_synthetic,
  d.solver,
  d.fidelity,
  d.ts_utc,
  d.retry_count,
  d.failure_type,
  d.action,
  d.reason,
  d.parameter_modifications_before,
  d.parameter_modifications_after,
  d.restart_from_checkpoint,
  d.confidence,
  d.okf_concepts_used,
  d.guardrail_outcome,
  d.forced_escalation,
  d.llm_called,
  d.latency_ms,
  r.current_status AS run_final_status,
  r.is_agent_recovered,
  CASE
    WHEN d.action = 'escalate_to_human' OR d.forced_escalation THEN 'escalated'
    WHEN r.is_agent_recovered THEN 'recovered'
    WHEN r.current_status = 'completed' THEN 'completed_with_human'
    ELSE 'unrecovered'
  END AS directive_outcome
FROM dedup_directives d
LEFT JOIN `${project_id}.sim_babysitter_telemetry.v_run_status` r
  ON d.workstream = r.workstream AND d.campaign_id = r.campaign_id AND d.run_id = r.run_id AND d.environment = r.environment AND d.is_synthetic = r.is_synthetic;


-- -----------------------------------------------------------------------------
-- 6. `v_runbook_effectiveness` (prod only)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_runbook_effectiveness` AS
WITH concept_rows AS (
  SELECT
    d.directive_id,
    d.workstream,
    d.campaign_id,
    d.environment,
    d.run_id,
    d.is_synthetic,
    d.solver,
    d.failure_type,
    d.action,
    d.guardrail_outcome,
    d.forced_escalation,
    d.is_agent_recovered,
    d.directive_outcome,
    CASE
      WHEN TO_JSON_STRING(d.okf_concepts_used) LIKE '%runbook-stalled%' THEN 'runbook-stalled'
      WHEN TO_JSON_STRING(d.okf_concepts_used) LIKE '%runbook-spectra%' THEN 'runbook-spectra'
      WHEN TO_JSON_STRING(d.okf_concepts_used) LIKE '%runbook-divergence%' THEN 'runbook-divergence'
      WHEN TO_JSON_STRING(d.okf_concepts_used) LIKE '%escalation-policy%' THEN 'escalation-policy'
      ELSE 'allowed-parameters'
    END AS primary_okf_concept
  FROM `${project_id}.sim_babysitter_telemetry.v_directive_outcomes` d
  WHERE d.environment = 'prod'
    AND d.is_synthetic = FALSE
)
SELECT
  workstream,
  campaign_id,
  environment,
  is_synthetic,
  primary_okf_concept AS okf_concept_id,
  solver,
  COUNT(*) AS directives_count,
  COUNT(DISTINCT run_id) AS distinct_runs_count,
  COUNT(DISTINCT CASE WHEN is_agent_recovered THEN run_id END) AS recovered_runs_count,
  SUM(CASE WHEN guardrail_outcome = 'rejected' THEN 1 ELSE 0 END) AS rejected_count,
  ROUND(
    SAFE_DIVIDE(
      COUNT(DISTINCT CASE WHEN is_agent_recovered THEN run_id END),
      COUNT(DISTINCT CASE WHEN action != 'escalate_to_human' THEN run_id END)
    ),
    4
  ) AS concept_recovery_rate
FROM concept_rows
GROUP BY workstream, campaign_id, environment, is_synthetic, primary_okf_concept, solver;


-- -----------------------------------------------------------------------------
-- 7. `v_param_coverage` (row-level, all non-preflight environments)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_param_coverage` AS
SELECT
  r.workstream,
  r.campaign_id,
  r.run_id,
  r.environment,
  r.is_synthetic,
  r.solver,
  r.fidelity,
  r.geometry_id,
  CASE
    WHEN r.current_status = 'completed' THEN 'converged'
    WHEN r.current_status IN ('escalated', 'human_resolved') OR r.escalated_count >= 1 THEN 'escalated'
    ELSE 'failed'
  END AS coverage_status,
  r.retry_count,
  r.is_agent_recovered,
  CAST(JSON_VALUE(r.latest_parameters_json, '$.nx') AS FLOAT64) AS nx,
  CAST(JSON_VALUE(r.latest_parameters_json, '$.ny') AS FLOAT64) AS ny,
  CAST(JSON_VALUE(r.latest_parameters_json, '$.nz') AS FLOAT64) AS nz,
  CAST(JSON_VALUE(r.latest_parameters_json, '$.dt_max') AS FLOAT64) AS dt_max,
  CAST(JSON_VALUE(r.latest_parameters_json, '$.cfl_fraction') AS FLOAT64) AS cfl_fraction,
  CAST(JSON_VALUE(r.latest_parameters_json, '$.hyperdiffusion_coeff') AS FLOAT64) AS hyperdiffusion_coeff,
  CAST(JSON_VALUE(r.latest_metrics_json, '$.heat_flux_es') AS FLOAT64) AS heat_flux_es,
  CAST(JSON_VALUE(r.latest_metrics_json, '$.growth_rate_gamma') AS FLOAT64) AS growth_rate_gamma
FROM `${project_id}.sim_babysitter_telemetry.v_run_status` r;


-- -----------------------------------------------------------------------------
-- 8. `v_timeline` (row-level, all non-preflight environments)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${project_id}.sim_babysitter_telemetry.v_timeline` AS
WITH dedup_events AS (
  SELECT
    event_id,
    COALESCE(workstream, 'kr2-gk-database') AS workstream,
    campaign_id,
    run_id,
    solver,
    fidelity,
    event_type,
    ts_utc,
    retry_count,
    COALESCE(environment, 'unknown') AS environment,
    is_synthetic,
    payload
  FROM `${project_id}.sim_babysitter_telemetry.events`
  WHERE ts_utc >= TIMESTAMP('2025-01-01T00:00:00Z')
    AND schema_version LIKE '1.%'
    AND COALESCE(environment, 'unknown') != 'preflight'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ts_utc ASC, ingested_at_utc ASC) = 1
)
SELECT
  e.event_id,
  e.workstream,
  e.campaign_id,
  e.environment,
  e.is_synthetic,
  e.run_id,
  e.solver,
  e.fidelity,
  e.event_type,
  e.ts_utc,
  e.retry_count,
  JSON_VALUE(e.payload, '$.failure_type') AS failure_type,
  JSON_VALUE(e.payload, '$.action') AS directive_action,
  JSON_VALUE(e.payload, '$.reason') AS reason,
  COALESCE(JSON_VALUE(e.payload, '$.equilibrium_id'), JSON_VALUE(e.payload, '$.geometry_id')) AS geometry_id,
  JSON_QUERY(e.payload, '$.metrics') AS metrics_json,
  JSON_QUERY(e.payload, '$.current_parameters') AS parameters_json,
  JSON_VALUE(e.payload, '$.log_tail') AS log_tail,
  SUM(CASE WHEN e.event_type = 'submitted' THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS submitted_cum,
  SUM(CASE WHEN e.event_type = 'completed' THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS completed_cum,
  SUM(CASE WHEN e.event_type = 'failed' THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS failed_events_cum,
  SUM(CASE WHEN e.event_type = 'patched' THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS patches_cum,
  SUM(CASE WHEN e.event_type IN ('walltime_resubmitted', 'resubmit_walltime') THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS walltime_resubmits_cum,
  SUM(CASE WHEN e.event_type = 'escalated' THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS escalated_cum,
  SUM(CASE WHEN e.event_type = 'human_resolved' THEN 1 ELSE 0 END)
    OVER (PARTITION BY e.workstream, e.campaign_id, e.environment, e.is_synthetic ORDER BY e.ts_utc ASC, e.event_id ASC) AS human_resolved_cum
FROM dedup_events e;
