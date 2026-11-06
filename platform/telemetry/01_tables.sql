-- =============================================================================
-- Shared Platform Telemetry Schema (`platform/telemetry/01_tables.sql`)
-- Dataset: `sim_babysitter_telemetry` | Region: us-central1 (TODO(tan): confirm)
-- Supports multi-workstream separation via `workstream` and `environment`.
--
-- Terraform (platform/infra/terraform/main.tf) creates these tables. This file is the
-- reviewable DDL and must stay column-for-column identical to the Terraform schemas
-- (platform/telemetry/tests/test_telemetry_contract.py enforces it). `${project_id}` is a
-- placeholder for the target GCP project.
-- In BigQuery, `events`, `directives` and `escalations` require a `ts_utc` partition filter.
-- =============================================================================

-- 1. Raw Lifecycle Events Table (Partitioned by DATE(ts_utc), Clustered by workstream, campaign_id, run_id)
CREATE TABLE IF NOT EXISTS `${project_id}.sim_babysitter_telemetry.events` (
  event_id STRING NOT NULL OPTIONS(description="UUIDv4 idempotency key generated on executor"),
  schema_version STRING NOT NULL OPTIONS(description="Contract version, e.g. 1.0"),
  workstream STRING NOT NULL OPTIONS(description="OKR workstream, e.g. kr2-gk-database, kr1-optimizer-orchestration, kr3-transport-consistent-design"),
  backend STRING OPTIONS(description="Execution backend: tacc_slurm | gcp_batch"),
  run_id STRING NOT NULL OPTIONS(description="Simulation run identifier"),
  campaign_id STRING NOT NULL OPTIONS(description="Campaign identifier, e.g. pilot_50 or campaign_2000"),
  solver STRING NOT NULL OPTIONS(description="Solver: GENE, GX, MRX, or SFINCS"),
  fidelity STRING NOT NULL OPTIONS(description="Grid fidelity tier: low or high"),
  event_type STRING NOT NULL OPTIONS(description="submitted|running|progress|completed|failed|triage_requested|triage|directive_received|patched|resubmitted|walltime_resubmitted|resubmit_walltime|escalated|human_resolved"),
  ts_utc TIMESTAMP NOT NULL OPTIONS(description="Event creation timestamp in UTC"),
  retry_count INT64 NOT NULL OPTIONS(description="Number of physics recovery retries at event time"),
  walltime_resubmits INT64 OPTIONS(description="Number of local walltime/Spot continuations at event time"),
  environment STRING NOT NULL OPTIONS(description="Execution environment: prod|sim|synthetic|preflight"),
  operator STRING OPTIONS(description="Optional SHA-256 salted operator handle"),
  is_synthetic BOOL NOT NULL OPTIONS(description="TRUE for synthetic/preflight rows; FALSE for live telemetry"),
  payload JSON NOT NULL OPTIONS(description="Typed event payload (<1 KB total event size)"),
  ingested_at_utc TIMESTAMP NOT NULL OPTIONS(description="Cloud Run ingestion timestamp in UTC")
)
PARTITION BY DATE(ts_utc)
CLUSTER BY workstream, campaign_id, run_id;

-- 2. Triage Directives Table
CREATE TABLE IF NOT EXISTS `${project_id}.sim_babysitter_telemetry.directives` (
  directive_id STRING NOT NULL OPTIONS(description="UUIDv4 identifier for the directive record"),
  schema_version STRING NOT NULL OPTIONS(description="Contract version, e.g. 1.0"),
  workstream STRING NOT NULL OPTIONS(description="OKR workstream identifier"),
  backend STRING OPTIONS(description="Execution backend: tacc_slurm | gcp_batch"),
  run_id STRING NOT NULL OPTIONS(description="Simulation run identifier"),
  campaign_id STRING NOT NULL OPTIONS(description="Campaign identifier"),
  solver STRING NOT NULL OPTIONS(description="GENE, GX, MRX, or SFINCS"),
  fidelity STRING NOT NULL OPTIONS(description="low or high"),
  ts_utc TIMESTAMP NOT NULL OPTIONS(description="Directive emission timestamp in UTC"),
  retry_count INT64 NOT NULL OPTIONS(description="Retry strike count (0, 1, or >=2)"),
  walltime_resubmits INT64 OPTIONS(description="Number of local walltime/Spot continuations"),
  environment STRING NOT NULL OPTIONS(description="Execution environment: prod|sim|synthetic|preflight"),
  failure_type STRING NOT NULL OPTIONS(description="STALL|TIMESTEP_COLLAPSE|SPECTRAL_PILEUP|DIVERGENCE|CRASH|UNKNOWN"),
  action STRING NOT NULL OPTIONS(description="warm_restart|cold_restart|resubmit_walltime|escalate_to_human"),
  reason STRING NOT NULL OPTIONS(description="Concise technical reason (<=300 chars)"),
  parameter_modifications_before JSON NOT NULL OPTIONS(description="Allowlisted parameter values prior to patch"),
  parameter_modifications_after JSON NOT NULL OPTIONS(description="Allowlisted parameter values recommended/applied"),
  restart_from_checkpoint BOOL NOT NULL OPTIONS(description="Whether restart reads checkpoint"),
  confidence STRING NOT NULL OPTIONS(description="low|medium|high"),
  okf_concepts_used JSON NOT NULL OPTIONS(description="JSON array of OKF concept IDs consulted"),
  guardrail_outcome STRING NOT NULL OPTIONS(description="accepted|rejected"),
  forced_escalation BOOL NOT NULL OPTIONS(description="TRUE if circuit-breaker or guardrail forced escalation"),
  llm_called BOOL NOT NULL OPTIONS(description="TRUE if Vertex AI / mock engine was invoked"),
  latency_ms FLOAT64 NOT NULL OPTIONS(description="End-to-end triage handler latency in ms"),
  is_synthetic BOOL NOT NULL OPTIONS(description="TRUE for synthetic rows; FALSE for live telemetry")
)
PARTITION BY DATE(ts_utc)
CLUSTER BY workstream, campaign_id, run_id;

-- 3. Human Escalations Table
CREATE TABLE IF NOT EXISTS `${project_id}.sim_babysitter_telemetry.escalations` (
  escalation_id STRING NOT NULL,
  event_id STRING NOT NULL,
  schema_version STRING NOT NULL,
  workstream STRING NOT NULL,
  run_id STRING NOT NULL,
  campaign_id STRING NOT NULL,
  solver STRING NOT NULL,
  fidelity STRING NOT NULL,
  ts_utc TIMESTAMP NOT NULL,
  retry_count INT64 NOT NULL,
  failure_type STRING NOT NULL,
  issue_md_text STRING NOT NULL,
  directives_tried JSON NOT NULL,
  tacc_folder_path STRING NOT NULL,
  environment STRING NOT NULL,
  is_synthetic BOOL NOT NULL
)
PARTITION BY DATE(ts_utc)
CLUSTER BY workstream, campaign_id, run_id;

-- 4. Campaigns Metadata Table
CREATE TABLE IF NOT EXISTS `${project_id}.sim_babysitter_telemetry.campaigns` (
  campaign_id STRING NOT NULL,
  workstream STRING NOT NULL,
  target_runs INT64 NOT NULL,
  start_date DATE NOT NULL,
  deadline_date DATE NOT NULL,
  notes STRING,
  environment STRING NOT NULL,
  is_synthetic BOOL NOT NULL,
  updated_at_utc TIMESTAMP NOT NULL
)
CLUSTER BY workstream, campaign_id;

-- 5. Assumptions Table (Seeded with TODO(david) placeholders for Estimated KPIs)
CREATE TABLE IF NOT EXISTS `${project_id}.sim_babysitter_telemetry.assumptions` (
  assumption_key STRING NOT NULL,
  numeric_value FLOAT64 NOT NULL,
  unit STRING NOT NULL,
  description STRING NOT NULL,
  source STRING NOT NULL,
  updated_date DATE NOT NULL,
  todo_owner STRING NOT NULL
)
CLUSTER BY assumption_key;

-- =============================================================================
-- 6. Cloud Logging sink -> cold-storage audit copy (separate dataset)
-- aurora-steer writes every telemetry record to stdout as a structured JSON envelope
-- (`emit_structured_cloud_log()`, `jsonPayload.log_stream = "sim_babysitter_telemetry"`).
-- Terraform routes those entries into the separate `sim_babysitter_audit_logs` dataset
-- (table `run_googleapis_com_stdout`, partitioned, with partition expiration). The sink writer
-- can write only to that dataset (never to the canonical tables), and the dashboard service
-- account cannot read it.
--
-- The audit copy is for recovery only. KPIs and the dashboard read the native `events`,
-- `directives` and `escalations` tables, which aurora-steer fills with BigQuery `insertAll`
-- (insertId = event_id / directive_id / escalation_id for best-effort dedup). To recover rows,
-- parse `jsonPayload.record` from the audit table.
-- =============================================================================
