output "aurora_steer_url" {
  description = "HTTPS endpoint of aurora-steer (/healthz, /triage, /v1/skills/<skill>/triage, /v1/events, /v1/events/summary)"
  value       = google_cloud_run_v2_service.aurora_steer.uri
}

output "dashboard_url" {
  description = "Cloud Run URL of the KR2 dashboard. With IAP on, open it in a browser; with IAP off, use `gcloud run services proxy aurora-steer-dashboard --region <region>`."
  value       = google_cloud_run_v2_service.dashboard.uri
}

output "gk_dataset_bucket" {
  description = "gs:// URL of the bucket holding equilibria, runs, queue, state and Cloud Batch checkpoints"
  value       = google_storage_bucket.gk_dataset.url
}

output "gk_dataset_bucket_name" {
  description = "Name of the GK dataset bucket"
  value       = google_storage_bucket.gk_dataset.name
}

output "bigquery_telemetry_dataset" {
  description = "BigQuery dataset ID with the canonical telemetry tables and KPI views"
  value       = google_bigquery_dataset.telemetry.dataset_id
}

output "artifact_registry_repository" {
  description = "Docker repository prefix for all images (tags are immutable)"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "gcp_batch_worker_service_account" {
  description = "Least-privilege service account that Cloud Batch GX VMs must run as (config key worker_service_account_email)"
  value       = google_service_account.gcp_batch_gx_worker.email
}

output "vpc_network" {
  description = "Custom VPC network URI for private Cloud Batch GX VMs (config key network)"
  value       = "projects/${var.project_id}/global/networks/${google_compute_network.batch_vpc.name}"
}

output "vpc_subnetwork" {
  description = "Regional private-Google-access subnetwork URI for Cloud Batch GX VMs (config key subnetwork)"
  value       = "projects/${var.project_id}/regions/${var.region}/subnetworks/${google_compute_subnetwork.batch_subnet.name}"
}

output "gcp_babysitter_service_account" {
  description = "Service account of the gcp-gx-babysitter Cloud Run job"
  value       = google_service_account.gcp_gx_babysitter.email
}

output "github_wif_provider" {
  description = "Full WIF provider resource name for the GitHub repository variable GCP_WIF_PROVIDER (null when WIF is disabled)"
  value       = var.enable_github_wif ? google_iam_workload_identity_pool_provider.github[0].name : null
}

output "github_deployer_service_account" {
  description = "Deployer service account for the GitHub repository variable GCP_DEPLOYER_SA (null when WIF is disabled)"
  value       = var.enable_github_wif ? google_service_account.github_deployer[0].email : null
}
