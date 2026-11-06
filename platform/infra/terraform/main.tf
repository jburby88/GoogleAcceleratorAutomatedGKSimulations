provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

data "google_project" "current" {}

locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "batch.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "iap.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ]

  # The KPI view DDL lives in platform/telemetry/02_views.sql (aurora-steer's local DuckDB store
  # runs the same file). The `${project_id}` placeholder is rewritten to this project/dataset.
  # Only chunks that contain a CREATE OR REPLACE VIEW header are kept, so comments after the last
  # statement cannot break the parse. View bodies must not contain semicolons
  # (platform/telemetry/tests/test_telemetry_contract.py enforces this).
  raw_views_sql = replace(
    file("${path.module}/../../telemetry/02_views.sql"),
    "$${project_id}.sim_babysitter_telemetry",
    "${var.project_id}.${google_bigquery_dataset.telemetry.dataset_id}"
  )

  view_statements = [
    for s in split(";", local.raw_views_sql) : trimspace(s)
    if can(regex("CREATE OR REPLACE VIEW `[^`]+\\.[a-z0-9_]+` AS", s))
  ]

  view_queries = {
    for stmt in local.view_statements :
    regex("CREATE OR REPLACE VIEW `[^`]+\\.([a-z0-9_]+)` AS", stmt)[0] => trimspace(
      replace(stmt, "/(?s)^.*CREATE OR REPLACE VIEW `[^`]+` AS\\s*/", "")
    )
  }

  views_tier1 = toset([
    "v_run_status",
  ])
  views_tier2 = toset([
    "v_failure_taxonomy",
    "v_directive_outcomes",
    "v_compute_savings",
    "v_param_coverage",
    "v_timeline",
  ])
  views_tier3 = toset([
    "v_runbook_effectiveness",
    "v_kpis_campaign",
  ])

  # Tables aurora-steer appends to (BigQuery insertAll). It gets write access to these only.
  aurora_steer_insert_tables = toset(["events", "directives", "escalations"])

  dashboard_iap_members    = var.enable_dashboard_iap ? toset(var.dashboard_authorized_members) : toset([])
  dashboard_direct_members = var.enable_dashboard_iap ? toset([]) : toset(var.dashboard_authorized_members)

  babysitter_enabled = var.enable_gcp_batch_babysitter_cron && var.gx_runner_image != ""
  net_prefix         = var.vpc_name_prefix != "" ? var.vpc_name_prefix : var.project_id
}

resource "google_project_service" "apis" {
  for_each           = toset(local.required_apis)
  service            = each.value
  disable_on_destroy = false
}

# 1. Artifact Registry. Tags are immutable: every build gets a unique tag and is deployed by
#    digest, so a tag can never be silently re-pointed at different code.
resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = "aurora-platform"
  format        = "DOCKER"
  description   = "Containers for aurora-steer, the Streamlit dashboard, and the GCP Cloud Batch GX runner"

  docker_config {
    immutable_tags = true
  }

  depends_on = [google_project_service.apis]
}

# Dedicated Cloud Build service account for remote image builds (`gcloud builds submit`) when the
# Compute Engine default service account has zero automatic IAM grants under Org Policy.
resource "google_service_account" "cloudbuild" {
  account_id   = "aurora-cloudbuild-sa"
  display_name = "Cloud Build Image Builder Service Account"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "cloudbuild_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_artifact_registry_repository_iam_member" "cloudbuild_ar_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.containers.location
  repository = google_artifact_registry_repository.containers.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_project_iam_member" "cloudbuild_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# 2. Secret Manager shell for the aurora-steer bearer token. Terraform never creates a version,
#    so the token never enters tfstate; scripts/deploy_gcp_demo.sh adds the first version.
resource "google_secret_manager_secret" "bearer_token" {
  secret_id = "aurora-steer-bearer-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# 3. GCS dataset & queue bucket (campaigns/<id>/{equilibria,runs}/, queue/, state/, config/).
#    Hierarchical namespace gives GCS FUSE atomic renames for checkpoints and queue moves. HNS
#    buckets do not support Object Versioning, Bucket Lock or object holds, so soft delete is
#    the recovery control for deleted or overwritten dataset objects.
resource "google_storage_bucket" "gk_dataset" {
  name                        = "${var.project_id}-gk-dataset"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  hierarchical_namespace {
    enabled = true
  }

  soft_delete_policy {
    retention_duration_seconds = var.dataset_soft_delete_retention_days * 86400
  }

  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Optional cross-project create-only access for UW's ToBlick/mrx equilibrium generator
# (equilibria/ prefixes only; objectCreator cannot overwrite or delete).
resource "google_storage_bucket_iam_member" "uw_mrx_equilibria_publisher" {
  for_each = toset(var.uw_mrx_writer_members)
  bucket   = google_storage_bucket.gk_dataset.name
  role     = "roles/storage.objectCreator"
  member   = each.value

  condition {
    title       = "uw_mrx_equilibria_prefix_only"
    description = "Restrict external UW MRX pipeline writes strictly to top-level campaign equilibria prefixes"
    expression  = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.gk_dataset.name}/objects/campaigns/\") && resource.name.contains(\"/equilibria/\") && !resource.name.contains(\"/runs/\")"
  }
}

# 4. BigQuery telemetry dataset (canonical tables + KPI views)
resource "google_bigquery_dataset" "telemetry" {
  dataset_id  = "sim_babysitter_telemetry"
  location    = var.region
  description = "Canonical multi-workstream telemetry dataset for UT Austin x Google.org"
  depends_on  = [google_project_service.apis]
}

# Cold-storage audit copy of aurora-steer's structured stdout records, kept in its own dataset
# so the sink writer cannot touch the canonical tables and the dashboard cannot read raw logs.
resource "google_bigquery_dataset" "audit_logs" {
  dataset_id                      = "sim_babysitter_audit_logs"
  location                        = var.region
  description                     = "Cloud Logging sink copy of aurora-steer telemetry records (audit/recovery only; not read by the dashboard)"
  default_partition_expiration_ms = var.audit_log_retention_days * 86400000
  depends_on                      = [google_project_service.apis]
}

resource "google_logging_project_sink" "telemetry_sink" {
  name                   = "aurora-steer-telemetry-bq-sink"
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit_logs.dataset_id}"
  filter                 = "resource.type=\"cloud_run_revision\" AND jsonPayload.log_stream=\"sim_babysitter_telemetry\""
  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }

  depends_on = [google_project_service.apis]
}

resource "google_bigquery_dataset_iam_member" "sink_writer" {
  dataset_id = google_bigquery_dataset.audit_logs.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.telemetry_sink.writer_identity
}

resource "google_bigquery_table" "campaigns" {
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = "campaigns"
  deletion_protection = true
  clustering          = ["workstream", "campaign_id"]

  schema = jsonencode([
    { name = "campaign_id", type = "STRING", mode = "REQUIRED" },
    { name = "workstream", type = "STRING", mode = "REQUIRED" },
    { name = "target_runs", type = "INT64", mode = "REQUIRED" },
    { name = "start_date", type = "DATE", mode = "REQUIRED" },
    { name = "deadline_date", type = "DATE", mode = "REQUIRED" },
    { name = "notes", type = "STRING", mode = "NULLABLE" },
    { name = "environment", type = "STRING", mode = "REQUIRED" },
    { name = "is_synthetic", type = "BOOL", mode = "REQUIRED" },
    { name = "updated_at_utc", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "assumptions" {
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = "assumptions"
  deletion_protection = true
  clustering          = ["assumption_key"]

  schema = jsonencode([
    { name = "assumption_key", type = "STRING", mode = "REQUIRED" },
    { name = "numeric_value", type = "FLOAT64", mode = "REQUIRED" },
    { name = "unit", type = "STRING", mode = "REQUIRED" },
    { name = "description", type = "STRING", mode = "REQUIRED" },
    { name = "source", type = "STRING", mode = "REQUIRED" },
    { name = "updated_date", type = "DATE", mode = "REQUIRED" },
    { name = "todo_owner", type = "STRING", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "events" {
  dataset_id               = google_bigquery_dataset.telemetry.dataset_id
  table_id                 = "events"
  deletion_protection      = true
  require_partition_filter = true
  clustering               = ["workstream", "campaign_id", "run_id"]

  time_partitioning {
    type  = "DAY"
    field = "ts_utc"
  }

  schema = jsonencode([
    { name = "event_id", type = "STRING", mode = "REQUIRED" },
    { name = "schema_version", type = "STRING", mode = "REQUIRED" },
    { name = "workstream", type = "STRING", mode = "REQUIRED" },
    { name = "backend", type = "STRING", mode = "NULLABLE" },
    { name = "run_id", type = "STRING", mode = "REQUIRED" },
    { name = "campaign_id", type = "STRING", mode = "REQUIRED" },
    { name = "solver", type = "STRING", mode = "REQUIRED" },
    { name = "fidelity", type = "STRING", mode = "REQUIRED" },
    { name = "event_type", type = "STRING", mode = "REQUIRED" },
    { name = "ts_utc", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "retry_count", type = "INT64", mode = "REQUIRED" },
    { name = "walltime_resubmits", type = "INT64", mode = "NULLABLE" },
    { name = "environment", type = "STRING", mode = "REQUIRED" },
    { name = "operator", type = "STRING", mode = "NULLABLE" },
    { name = "is_synthetic", type = "BOOL", mode = "REQUIRED" },
    { name = "payload", type = "JSON", mode = "REQUIRED" },
    { name = "ingested_at_utc", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "directives" {
  dataset_id               = google_bigquery_dataset.telemetry.dataset_id
  table_id                 = "directives"
  deletion_protection      = true
  require_partition_filter = true
  clustering               = ["workstream", "campaign_id", "run_id"]

  time_partitioning {
    type  = "DAY"
    field = "ts_utc"
  }

  schema = jsonencode([
    { name = "directive_id", type = "STRING", mode = "REQUIRED" },
    { name = "schema_version", type = "STRING", mode = "REQUIRED" },
    { name = "workstream", type = "STRING", mode = "REQUIRED" },
    { name = "backend", type = "STRING", mode = "NULLABLE" },
    { name = "run_id", type = "STRING", mode = "REQUIRED" },
    { name = "campaign_id", type = "STRING", mode = "REQUIRED" },
    { name = "solver", type = "STRING", mode = "REQUIRED" },
    { name = "fidelity", type = "STRING", mode = "REQUIRED" },
    { name = "ts_utc", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "retry_count", type = "INT64", mode = "REQUIRED" },
    { name = "walltime_resubmits", type = "INT64", mode = "NULLABLE" },
    { name = "environment", type = "STRING", mode = "REQUIRED" },
    { name = "failure_type", type = "STRING", mode = "REQUIRED" },
    { name = "action", type = "STRING", mode = "REQUIRED" },
    { name = "reason", type = "STRING", mode = "REQUIRED" },
    { name = "parameter_modifications_before", type = "JSON", mode = "REQUIRED" },
    { name = "parameter_modifications_after", type = "JSON", mode = "REQUIRED" },
    { name = "restart_from_checkpoint", type = "BOOL", mode = "REQUIRED" },
    { name = "confidence", type = "STRING", mode = "REQUIRED" },
    { name = "okf_concepts_used", type = "JSON", mode = "REQUIRED" },
    { name = "guardrail_outcome", type = "STRING", mode = "REQUIRED" },
    { name = "forced_escalation", type = "BOOL", mode = "REQUIRED" },
    { name = "llm_called", type = "BOOL", mode = "REQUIRED" },
    { name = "latency_ms", type = "FLOAT64", mode = "REQUIRED" },
    { name = "is_synthetic", type = "BOOL", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "escalations" {
  dataset_id               = google_bigquery_dataset.telemetry.dataset_id
  table_id                 = "escalations"
  deletion_protection      = true
  require_partition_filter = true
  clustering               = ["workstream", "campaign_id", "run_id"]

  time_partitioning {
    type  = "DAY"
    field = "ts_utc"
  }

  schema = jsonencode([
    { name = "escalation_id", type = "STRING", mode = "REQUIRED" },
    { name = "event_id", type = "STRING", mode = "REQUIRED" },
    { name = "schema_version", type = "STRING", mode = "REQUIRED" },
    { name = "workstream", type = "STRING", mode = "REQUIRED" },
    { name = "run_id", type = "STRING", mode = "REQUIRED" },
    { name = "campaign_id", type = "STRING", mode = "REQUIRED" },
    { name = "solver", type = "STRING", mode = "REQUIRED" },
    { name = "fidelity", type = "STRING", mode = "REQUIRED" },
    { name = "ts_utc", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "retry_count", type = "INT64", mode = "REQUIRED" },
    { name = "failure_type", type = "STRING", mode = "REQUIRED" },
    { name = "issue_md_text", type = "STRING", mode = "REQUIRED" },
    { name = "directives_tried", type = "JSON", mode = "REQUIRED" },
    { name = "tacc_folder_path", type = "STRING", mode = "REQUIRED" },
    { name = "environment", type = "STRING", mode = "REQUIRED" },
    { name = "is_synthetic", type = "BOOL", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "telemetry_views_tier1" {
  for_each            = local.views_tier1
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = each.value
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = local.view_queries[each.value]
  }

  depends_on = [
    google_bigquery_table.events,
    google_bigquery_table.directives,
    google_bigquery_table.escalations,
    google_bigquery_table.campaigns,
    google_bigquery_table.assumptions,
  ]
}

resource "google_bigquery_table" "telemetry_views_tier2" {
  for_each            = local.views_tier2
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = each.value
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = local.view_queries[each.value]
  }

  depends_on = [google_bigquery_table.telemetry_views_tier1]
}

resource "google_bigquery_table" "telemetry_views_tier3" {
  for_each            = local.views_tier3
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = each.value
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = local.view_queries[each.value]
  }

  depends_on = [google_bigquery_table.telemetry_views_tier2]
}

# 5. aurora-steer Cloud Run v2 service
resource "google_service_account" "aurora_steer" {
  account_id   = "aurora-steer-sa"
  display_name = "aurora-steer Cloud Run Service Account"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "aurora_steer_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.aurora_steer.email}"
}

# Append-only telemetry writer: table-scoped grants on the three tables it inserts into, so it
# cannot create, alter or drop other tables or the KPI views.
resource "google_bigquery_table_iam_member" "aurora_steer_bq_writer" {
  for_each   = local.aurora_steer_insert_tables
  project    = var.project_id
  dataset_id = google_bigquery_dataset.telemetry.dataset_id
  table_id   = each.value
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.aurora_steer.email}"

  depends_on = [
    google_bigquery_table.events,
    google_bigquery_table.directives,
    google_bigquery_table.escalations,
  ]
}

resource "google_secret_manager_secret_iam_member" "aurora_steer_secret_access" {
  secret_id = google_secret_manager_secret.bearer_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.aurora_steer.email}"
}

# Terraform owns the service configuration (env, secrets, identity, scaling, probes). The
# running image is rolled by digest by the release step (scripts/deploy_gcp_demo.sh or the
# deploy-cloudrun workflow, both via `gcloud run services update --image`), so image drift and
# the gcloud client fields are ignored here.
resource "google_cloud_run_v2_service" "aurora_steer" {
  name     = "aurora-steer"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  # TACC login nodes sit outside GCP and cannot mint Google identity tokens, so the Cloud Run
  # invoker IAM check is disabled and every endpoint except /healthz is authenticated by the
  # app's constant-time bearer-token check (token in Secret Manager). Disabling the check, rather
  # than granting allUsers, also works under the Domain Restricted Sharing org policy.
  invoker_iam_disabled = true
  deletion_protection  = true

  template {
    service_account = google_service_account.aurora_steer.email

    # Idempotency/dedup state is per instance (STEER-C06): keep a single instance.
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = var.aurora_steer_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      # /healthz returns 503 when the OKF skill or the model config is broken, so a bad
      # revision never receives traffic.
      startup_probe {
        initial_delay_seconds = 0
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 12
        http_get {
          path = "/healthz"
          port = 8080
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.model_location
      }
      env {
        name  = "MODEL_ID"
        value = var.model_id
      }
      env {
        name  = "VERTEX_TIMEOUT_S"
        value = tostring(var.vertex_timeout_s)
      }
      env {
        name  = "BQ_ENABLED"
        value = "1"
      }
      env {
        name  = "BQ_DATASET"
        value = google_bigquery_dataset.telemetry.dataset_id
      }
      env {
        name = "AURORA_STEER_BEARER_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.bearer_token.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      client,
      client_version,
      template[0].containers[0].image,
    ]
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_iam_member.aurora_steer_secret_access,
  ]
}

# 6. KR2 Streamlit dashboard: private. Direct Cloud Run IAP when enable_dashboard_iap is true;
#    otherwise only dashboard_authorized_members can invoke it (open it with
#    `gcloud run services proxy aurora-steer-dashboard --region <region>`). Never public.
resource "google_project_service_identity" "iap_sa" {
  count      = var.enable_dashboard_iap ? 1 : 0
  provider   = google-beta
  project    = var.project_id
  service    = "iap.googleapis.com"
  depends_on = [google_project_service.apis["iap.googleapis.com"]]
}

resource "google_service_account" "dashboard" {
  account_id   = "aurora-dashboard-sa"
  display_name = "KR2 Streamlit Dashboard Read-Only Service Account"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "dashboard_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.dashboard.email}"
}

resource "google_bigquery_dataset_iam_member" "dashboard_bq_viewer" {
  dataset_id = google_bigquery_dataset.telemetry.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.dashboard.email}"
}

resource "google_cloud_run_v2_service" "dashboard" {
  provider            = google-beta
  name                = "aurora-steer-dashboard"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  launch_stage        = "BETA"
  iap_enabled         = var.enable_dashboard_iap
  deletion_protection = true

  template {
    service_account = google_service_account.dashboard.email
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = var.dashboard_image

      ports {
        container_port = 8080
      }

      startup_probe {
        initial_delay_seconds = 0
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 24
        http_get {
          path = "/_stcore/health"
          port = 8080
        }
      }

      env {
        name  = "DATA_MODE"
        value = "bigquery"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "BQ_DATASET"
        value = google_bigquery_dataset.telemetry.dataset_id
      }
      env {
        name  = "BQ_MAX_BYTES_BILLED"
        value = tostring(var.dashboard_max_bytes_billed)
      }
    }
  }

  lifecycle {
    ignore_changes = [
      client,
      client_version,
      template[0].containers[0].image,
    ]
  }

  depends_on = [google_project_service.apis]
}

# IAP mode: only the IAP service agent may invoke the service; humans get IAP access.
resource "google_cloud_run_v2_service_iam_member" "dashboard_iap_invoker" {
  count    = var.enable_dashboard_iap ? 1 : 0
  name     = google_cloud_run_v2_service.dashboard.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_project_service_identity.iap_sa[0].email}"
}

resource "google_iap_web_cloud_run_service_iam_member" "dashboard_users" {
  for_each               = local.dashboard_iap_members
  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.dashboard.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value
}

# Non-IAP mode: authenticated Cloud Run invokers only (no allUsers).
resource "google_cloud_run_v2_service_iam_member" "dashboard_direct_invokers" {
  for_each = local.dashboard_direct_members
  name     = google_cloud_run_v2_service.dashboard.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}

check "dashboard_has_viewers" {
  assert {
    condition     = length(var.dashboard_authorized_members) > 0
    error_message = "dashboard_authorized_members is empty, so nobody can open the dashboard. Add e.g. group:<team>@<domain>."
  }
}

# 7. Custom VPC + Cloud NAT + Cloud Batch GX GPU worker + serverless GCP babysitter
#    Org policies on ut-austin-bsitter enforce skipDefaultNetworkCreation and deny VM external
#    IPs. Private Google Access serves Artifact Registry, GCS, Cloud Logging and Batch APIs
#    internally; Cloud NAT with dynamic port allocation serves external NVIDIA driver fetches
#    across 32+ concurrent private GPU VMs without SNAT port exhaustion. No ingress rules are
#    created, so inbound internet traffic stays default-denied.
resource "google_compute_network" "batch_vpc" {
  name                    = "${local.net_prefix}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "batch_subnet" {
  name                     = "${local.net_prefix}-subnet"
  ip_cidr_range            = var.vpc_subnet_cidr
  region                   = var.region
  network                  = google_compute_network.batch_vpc.id
  private_ip_google_access = true
}

# Optional Cloud Router + Cloud NAT (disabled by default to avoid ~$32/mo standing idle NAT cost).
# Enable via `enable_cloud_nat = true` (`ENABLE_CLOUD_NAT=true` in .env) if your GCP environment
# Org Policy (`constraints/compute.vmExternalIpAccess`) does not allow VMs to have public IPs.
resource "google_compute_router" "batch_router" {
  count   = var.enable_cloud_nat ? 1 : 0
  name    = "${local.net_prefix}-router"
  region  = var.region
  network = google_compute_network.batch_vpc.id
}

resource "google_compute_router_nat" "batch_nat" {
  count                               = var.enable_cloud_nat ? 1 : 0
  name                                = "${local.net_prefix}-nat"
  router                              = google_compute_router.batch_router[0].name
  region                              = google_compute_router.batch_router[0].region
  nat_ip_allocate_option              = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat  = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  enable_endpoint_independent_mapping = false
  enable_dynamic_port_allocation      = true
  min_ports_per_vm                    = 64
  max_ports_per_vm                    = 2048

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Least-privilege identity for Batch VMs (GCP-05). gcp_exec must always pass this account
# explicitly so jobs never fall back to the Compute Engine default service account.
resource "google_service_account" "gcp_batch_gx_worker" {
  account_id   = "gx-batch-worker-sa"
  display_name = "Cloud Batch GX GPU Simulation Worker (Spot L4 / A100)"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "gx_worker_dataset_rw" {
  bucket = google_storage_bucket.gk_dataset.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.gcp_batch_gx_worker.email}"
}

resource "google_artifact_registry_repository_iam_member" "gx_worker_ar_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.containers.location
  repository = google_artifact_registry_repository.containers.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.gcp_batch_gx_worker.email}"
}

# Batch task logs (logsPolicy CLOUD_LOGGING) are written with the VM's service account.
resource "google_project_iam_member" "gx_worker_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gcp_batch_gx_worker.email}"
}

resource "google_project_iam_member" "gx_worker_batch_reporter" {
  project = var.project_id
  role    = "roles/batch.agentReporter"
  member  = "serviceAccount:${google_service_account.gcp_batch_gx_worker.email}"
}

resource "google_service_account" "gcp_gx_babysitter" {
  account_id   = "gcp-gx-babysitter-sa"
  display_name = "Serverless Watchdog for GCP Cloud Batch GX Runs"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "babysitter_batch_editor" {
  project = var.project_id
  role    = "roles/batch.jobsEditor"
  member  = "serviceAccount:${google_service_account.gcp_gx_babysitter.email}"
}

# The babysitter may launch Batch jobs as the worker identity, and only as that identity.
resource "google_service_account_iam_member" "babysitter_act_as_gx_worker" {
  service_account_id = google_service_account.gcp_batch_gx_worker.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.gcp_gx_babysitter.email}"
}

resource "google_storage_bucket_iam_member" "babysitter_dataset_rw" {
  bucket = google_storage_bucket.gk_dataset.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.gcp_gx_babysitter.email}"
}

resource "google_secret_manager_secret_iam_member" "babysitter_secret_access" {
  secret_id = google_secret_manager_secret.bearer_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.gcp_gx_babysitter.email}"
}

resource "google_cloud_run_v2_job" "gcp_gx_babysitter" {
  count    = local.babysitter_enabled ? 1 : 0
  name     = "gcp-gx-babysitter"
  location = var.region

  # Stateless and recreatable; keeps enable_gcp_batch_babysitter_cron toggleable.
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.gcp_gx_babysitter.email
      max_retries     = 0
      # Must exceed one triage call (client timeout 100 s) plus cancel waits. Overlapping
      # executions are excluded by the driver's heartbeated lease in the state dir.
      timeout = "900s"

      volumes {
        name = "gk-bucket"
        gcs {
          bucket    = google_storage_bucket.gk_dataset.name
          read_only = false
        }
      }

      containers {
        image   = var.gx_runner_image
        command = ["gcp-gx-babysitter", "--config", "/mnt/gk_bucket/config/batch_babysitter.json", "--once"]
        volume_mounts {
          name       = "gk-bucket"
          mount_path = "/mnt/gk_bucket"
        }
        env {
          name  = "AURORA_STEER_URL"
          value = google_cloud_run_v2_service.aurora_steer.uri
        }
        env {
          name = "AURORA_STEER_BEARER_TOKEN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.bearer_token.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_iam_member.babysitter_secret_access,
  ]
}

# Dedicated scheduler identity: it can only start the babysitter job.
resource "google_service_account" "gcp_gx_scheduler" {
  account_id   = "gcp-gx-scheduler-sa"
  display_name = "Cloud Scheduler invoker for the gcp-gx-babysitter job"
  depends_on   = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_invoke_babysitter" {
  count    = local.babysitter_enabled ? 1 : 0
  name     = google_cloud_run_v2_job.gcp_gx_babysitter[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.gcp_gx_scheduler.email}"
}

resource "google_cloud_scheduler_job" "gcp_gx_babysitter_tick" {
  count       = local.babysitter_enabled ? 1 : 0
  name        = "gcp-gx-babysitter-every-2m"
  description = "Polls GCP Cloud Batch GX simulations every 2 minutes via shared Watchdog + aurora-steer"
  schedule    = "*/2 * * * *"
  time_zone   = "Etc/UTC"
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${data.google_project.current.number}/jobs/${google_cloud_run_v2_job.gcp_gx_babysitter[0].name}:run"
    oauth_token {
      service_account_email = google_service_account.gcp_gx_scheduler.email
    }
  }

  depends_on = [
    google_project_service.apis,
    google_cloud_run_v2_job_iam_member.scheduler_invoke_babysitter,
  ]
}

# 8. Optional GitHub Actions Workload Identity Federation for the deploy-cloudrun workflow.
#    Tokens are accepted only from var.github_repository, on var.github_deploy_branch, in jobs
#    bound to the var.github_deploy_environment GitHub environment (put required reviewers on
#    it). The deployer can only push images and roll the two Cloud Run services' images.
resource "google_iam_workload_identity_pool" "github" {
  count                     = var.enable_github_wif ? 1 : 0
  workload_identity_pool_id = "github-actions-pool"
  display_name              = "GitHub Actions WIF Pool"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  count                              = var.enable_github_wif ? 1 : 0
  workload_identity_pool_id          = google_iam_workload_identity_pool.github[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC Provider"
  attribute_condition = join(" && ", [
    "assertion.repository == '${var.github_repository}'",
    "assertion.ref == 'refs/heads/${var.github_deploy_branch}'",
    "assertion.environment == '${var.github_deploy_environment}'",
  ])
  attribute_mapping = {
    "google.subject"         = "assertion.sub"
    "attribute.repository"   = "assertion.repository"
    "attribute.ref"          = "assertion.ref"
    "attribute.environment"  = "assertion.environment"
    "attribute.workflow_ref" = "assertion.workflow_ref"
  }
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  lifecycle {
    precondition {
      condition     = var.github_repository != ""
      error_message = "enable_github_wif = true requires github_repository (owner/repo)."
    }
  }
}

resource "google_service_account" "github_deployer" {
  count        = var.enable_github_wif ? 1 : 0
  account_id   = "github-deployer-sa"
  display_name = "GitHub Actions deployer (push images, roll Cloud Run images)"
  depends_on   = [google_project_service.apis]
}

resource "google_service_account_iam_member" "github_deployer_wif" {
  count              = var.enable_github_wif ? 1 : 0
  service_account_id = google_service_account.github_deployer[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principal://iam.googleapis.com/${google_iam_workload_identity_pool.github[0].name}/subject/repo:${var.github_repository}:environment:${var.github_deploy_environment}"
}

resource "google_artifact_registry_repository_iam_member" "github_deployer_ar_writer" {
  count      = var.enable_github_wif ? 1 : 0
  project    = var.project_id
  location   = google_artifact_registry_repository.containers.location
  repository = google_artifact_registry_repository.containers.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_deployer[0].email}"
}

resource "google_cloud_run_v2_service_iam_member" "github_deployer_aurora_steer" {
  count    = var.enable_github_wif ? 1 : 0
  name     = google_cloud_run_v2_service.aurora_steer.name
  location = var.region
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.github_deployer[0].email}"
}

resource "google_cloud_run_v2_service_iam_member" "github_deployer_dashboard" {
  count    = var.enable_github_wif ? 1 : 0
  name     = google_cloud_run_v2_service.dashboard.name
  location = var.region
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.github_deployer[0].email}"
}

# Rolling a revision requires actAs on the service's runtime identity (and on nothing else).
resource "google_service_account_iam_member" "github_deployer_act_as_aurora_steer" {
  count              = var.enable_github_wif ? 1 : 0
  service_account_id = google_service_account.aurora_steer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.github_deployer[0].email}"
}

resource "google_service_account_iam_member" "github_deployer_act_as_dashboard" {
  count              = var.enable_github_wif ? 1 : 0
  service_account_id = google_service_account.dashboard.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.github_deployer[0].email}"
}
