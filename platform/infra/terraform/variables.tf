variable "project_id" {
  description = "Target GCP project ID (TODO(tan): confirm the UT Austin / Google.org project ID)."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a GCP project ID: 6-30 chars, lowercase letters, digits and hyphens, starting with a letter."
  }
}

variable "region" {
  description = "Primary GCP region for Cloud Run, Cloud Batch, Artifact Registry, GCS and BigQuery (TODO(tan): confirm)."
  type        = string
  default     = "us-central1"

  validation {
    condition     = can(regex("^[a-z]+(-[a-z]+)+[0-9]+$", var.region))
    error_message = "region must be a GCP region name such as us-central1."
  }
}

variable "vpc_name_prefix" {
  description = "Optional name prefix for the custom Cloud Batch VPC, subnet, router and Cloud NAT (defaults to project_id when empty)."
  type        = string
  default     = ""

  validation {
    condition     = var.vpc_name_prefix == "" || can(regex("^[a-z][a-z0-9-]{0,53}[a-z0-9]$", var.vpc_name_prefix))
    error_message = "vpc_name_prefix must be empty or 2-55 lowercase letters, digits and hyphens, starting with a letter and ending with a letter or digit."
  }
}

variable "vpc_subnet_cidr" {
  description = "IPv4 CIDR range for the regional Cloud Batch subnet (override to avoid RFC1918 overlap with UT Austin / TACC VPN, Interconnect or Shared VPC)."
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrnetmask(var.vpc_subnet_cidr)) && try(cidrhost(var.vpc_subnet_cidr, 0) == split("/", var.vpc_subnet_cidr)[0], false) && try(tonumber(split("/", var.vpc_subnet_cidr)[1]), 33) >= 8 && try(tonumber(split("/", var.vpc_subnet_cidr)[1]), 33) <= 24
    error_message = "vpc_subnet_cidr must be a valid IPv4 network CIDR (zero host bits) with prefix length between /8 and /24 (e.g. 10.0.0.0/16)."
  }
}

variable "enable_cloud_nat" {
  description = "Provision Cloud Router + Cloud NAT on the VPC subnet. Default is false ($0 idle cost; Private Google Access is always enabled for free). Set to true if your GCP environment Org Policy (constraints/compute.vmExternalIpAccess) does not allow VMs to have public IPs and GPU VMs need outbound internet access for driver installation."
  type        = bool
  default     = false
}

variable "model_location" {
  description = "Vertex AI endpoint location used by aurora-steer (GOOGLE_CLOUD_LOCATION)."
  type        = string
  default     = "global"

  validation {
    condition     = var.model_location == "global" || can(regex("^[a-z]+(-[a-z]+)+[0-9]+$", var.model_location))
    error_message = "model_location must be \"global\" or a GCP region name."
  }
}

variable "model_id" {
  description = "Vertex AI Gemini model ID used by aurora-steer (MODEL_ID). Must be a generally available Vertex AI model; gemini-3.8-flash is a lower-cost alternative."
  type        = string
  default     = "gemini-4-argon"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,62}[a-z0-9]$", var.model_id))
    error_message = "model_id must be a plain Vertex AI model ID such as gemini-4-argon."
  }
}

variable "vertex_timeout_s" {
  description = "Server-side deadline for one Vertex AI call (VERTEX_TIMEOUT_S). Must stay below the HPC client's 100 s request timeout."
  type        = number
  default     = 60

  validation {
    condition     = var.vertex_timeout_s >= 5 && var.vertex_timeout_s <= 80
    error_message = "vertex_timeout_s must be between 5 and 80 seconds (aurora-steer's MAX_VERTEX_TIMEOUT_S)."
  }
}

variable "aurora_steer_image" {
  description = "Initial aurora-steer image (Artifact Registry URI pinned by digest or a unique tag, never :latest). Terraform sets it at creation only; later images are rolled by the release step (scripts/deploy_gcp_demo.sh or .github/workflows/deploy-cloudrun.yml)."
  type        = string

  validation {
    condition     = !endswith(var.aurora_steer_image, ":latest") && can(regex("^[a-z0-9.-]+(:[0-9]+)?/[^@:]+(@sha256:[a-f0-9]{64}|:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$", var.aurora_steer_image))
    error_message = "aurora_steer_image must be pinned by @sha256 digest or by a unique tag (not :latest)."
  }
}

variable "dashboard_image" {
  description = "Initial dashboard image (digest or unique tag, never :latest). Rolled by the release step after creation, like aurora_steer_image."
  type        = string

  validation {
    condition     = !endswith(var.dashboard_image, ":latest") && can(regex("^[a-z0-9.-]+(:[0-9]+)?/[^@:]+(@sha256:[a-f0-9]{64}|:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$", var.dashboard_image))
    error_message = "dashboard_image must be pinned by @sha256 digest or by a unique tag (not :latest)."
  }
}

variable "gx_runner_image" {
  description = "gx-runner image for the serverless GCP babysitter Cloud Run job (digest or unique tag, never :latest). Empty disables the job. Terraform owns this image."
  type        = string
  default     = ""

  validation {
    condition     = var.gx_runner_image == "" || (!endswith(var.gx_runner_image, ":latest") && can(regex("^[a-z0-9.-]+(:[0-9]+)?/[^@:]+(@sha256:[a-f0-9]{64}|:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$", var.gx_runner_image)))
    error_message = "gx_runner_image must be empty or pinned by @sha256 digest or a unique tag (not :latest)."
  }
}

variable "enable_dashboard_iap" {
  description = "Protect the dashboard with direct Cloud Run IAP (needs an OAuth consent screen). When false, the dashboard is still private: only dashboard_authorized_members may invoke it, via `gcloud run services proxy`."
  type        = bool
  default     = true
}

variable "dashboard_authorized_members" {
  description = "IAM principals (e.g. group:ut-fusion-team@utexas.edu) allowed to open the dashboard: IAP accessors when IAP is on, Cloud Run invokers when it is off. Public principals are rejected."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for m in var.dashboard_authorized_members :
      can(regex("^(user|group|serviceAccount|domain|principal|principalSet):.+$", m))
    ])
    error_message = "dashboard_authorized_members entries must be user:, group:, serviceAccount:, domain:, principal: or principalSet: principals (allUsers/allAuthenticatedUsers are not allowed)."
  }
}

variable "dashboard_max_bytes_billed" {
  description = "Per-query BigQuery maximum_bytes_billed cap for the dashboard (BQ_MAX_BYTES_BILLED). Queries that would scan more fail instead of billing."
  type        = number
  default     = 104857600

  validation {
    condition     = var.dashboard_max_bytes_billed >= 10485760
    error_message = "dashboard_max_bytes_billed must be at least 10 MiB (10485760)."
  }
}

variable "dataset_soft_delete_retention_days" {
  description = "Soft-delete retention for the GK dataset bucket: deleted or overwritten objects stay recoverable this long. Hierarchical-namespace buckets do not support Object Versioning or Bucket Lock, so this is the recovery control."
  type        = number
  default     = 30

  validation {
    condition     = var.dataset_soft_delete_retention_days >= 7 && var.dataset_soft_delete_retention_days <= 90
    error_message = "dataset_soft_delete_retention_days must be between 7 and 90."
  }
}

variable "audit_log_retention_days" {
  description = "Partition expiration for the Cloud Logging audit copy of telemetry (sim_babysitter_audit_logs dataset)."
  type        = number
  default     = 365

  validation {
    condition     = var.audit_log_retention_days >= 1 && var.audit_log_retention_days <= 3650
    error_message = "audit_log_retention_days must be between 1 and 3650."
  }
}

variable "uw_mrx_writer_members" {
  description = "Optional cross-project principals from UW's GCP account allowed to create (never overwrite or delete) ToBlick/mrx equilibria under gs://<bucket>/campaigns/*/equilibria/."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for m in var.uw_mrx_writer_members :
      can(regex("^(user|group|serviceAccount|principal|principalSet):.+$", m))
    ])
    error_message = "uw_mrx_writer_members entries must be user:, group:, serviceAccount:, principal: or principalSet: principals."
  }
}

variable "enable_gcp_batch_babysitter_cron" {
  description = "Provision a Cloud Run v2 job + Cloud Scheduler 2-minute tick that babysits GCP Cloud Batch GX runs (requires gx_runner_image)."
  type        = bool
  default     = true
}

variable "enable_github_wif" {
  description = "Provision GitHub Actions Workload Identity Federation (pool, provider, least-privilege deployer service account)."
  type        = bool
  default     = false
}

variable "github_repository" {
  description = "GitHub owner/repo allowed to deploy via WIF (e.g. jburby88/GoogleAcceleratorAutomatedGKSimulations). Required when enable_github_wif is true."
  type        = string
  default     = ""

  validation {
    condition     = var.github_repository == "" || can(regex("^[A-Za-z0-9-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "github_repository must be empty or of the form owner/repo."
  }
}

variable "github_deploy_branch" {
  description = "Only workflow runs on this branch can obtain GCP credentials through WIF."
  type        = string
  default     = "main"

  validation {
    condition     = can(regex("^[A-Za-z0-9._/-]+$", var.github_deploy_branch))
    error_message = "github_deploy_branch must be a plain branch name."
  }
}

variable "github_deploy_environment" {
  description = "Only jobs running in this GitHub environment (configure required reviewers on it) can obtain GCP credentials through WIF."
  type        = string
  default     = "production"

  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+$", var.github_deploy_environment))
    error_message = "github_deploy_environment must be a plain environment name."
  }
}
