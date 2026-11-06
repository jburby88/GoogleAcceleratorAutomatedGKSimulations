terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  # Remote state with native locking. The state bucket is created out of band (versioned,
  # uniform access, public access prevention) by `make gcp-init` / scripts/deploy_gcp_demo.sh,
  # and its name is passed at init time:
  #   terraform init -backend-config="bucket=<project>-tfstate" -backend-config="prefix=platform/infra"
  backend "gcs" {}

  # One provider major version, bumped deliberately (read the upgrade guide, run a plan).
  # Commit the generated .terraform.lock.hcl after the first `terraform init`.
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.30"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.30"
    }
  }
}
