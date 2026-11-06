#!/usr/bin/env bash
# ==============================================================================
# GCP bootstrap, deploy and release tool (`make gcp-init|gcp-plan|gcp-apply|gcp-deploy`,
# and the release step of .github/workflows/deploy-cloudrun.yml).
#
#   scripts/deploy_gcp_demo.sh [deploy]          full idempotent deploy (default):
#       1. ensure the versioned Terraform state bucket and run `terraform init`
#       2. bootstrap APIs, Artifact Registry, the Secret Manager shell, GCS and BigQuery
#       3. make sure the bearer-token secret has a version (generated straight into Secret
#          Manager; the token is never printed and never written to disk or .env)
#       4. build + push images under an immutable tag (git SHA); existing tags are reused
#       5. apply the full Terraform stack with the images pinned by digest
#       6. roll the Cloud Run service images by digest (Terraform ignores image drift)
#       7. upload config/batch_babysitter.json (from Terraform outputs) to the dataset bucket
#   scripts/deploy_gcp_demo.sh init              steps 1 only
#   scripts/deploy_gcp_demo.sh plan|apply        Terraform plan/apply with the images that are
#                                                currently deployed (no builds)
#   scripts/deploy_gcp_demo.sh release SERVICE.. build/push (if the tag is new) and roll
#                                                aurora-steer and/or dashboard (used by CI;
#                                                needs no Terraform permissions)
#
# Configuration comes from the environment, or from .env when present (see .env.example):
#   GOOGLE_CLOUD_PROJECT (required), GCP_REGION, VPC_NAME_PREFIX, VPC_SUBNET_CIDR,
#   MODEL_ID, GOOGLE_CLOUD_LOCATION, ENABLE_DASHBOARD_IAP, DASHBOARD_AUTHORIZED_MEMBERS,
#   UW_MRX_WRITER_MEMBERS, ENABLE_GITHUB_WIF, GITHUB_WIF_REPOSITORY, TF_STATE_BUCKET,
#   CAMPAIGN_ID, GCP_BATCH_CPU_ONLY, GCP_BATCH_CANDIDATE_REGIONS, GX_COMMIT (40-hex GX git SHA),
#   IMAGE_TAG
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TF_DIR="${REPO_ROOT}/platform/infra/terraform"
SECRET_ID="aurora-steer-bearer-token"
STEER_SERVICE="aurora-steer"
DASHBOARD_SERVICE="aurora-steer-dashboard"
BABYSITTER_JOB="gcp-gx-babysitter"

log() { printf '=== %s\n' "$*"; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

strip_quotes() {
  local v="$1"
  v="${v%\"}"
  v="${v#\"}"
  v="${v%\'}"
  v="${v#\'}"
  printf '%s' "${v}"
}

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

require_tools() {
  local tool
  for tool in "$@"; do
    command -v "${tool}" >/dev/null 2>&1 || die "required tool '${tool}' is not installed"
  done
}

use_cloud_build() {
  if [ -n "${USE_CLOUD_BUILD:-}" ]; then
    is_true "${USE_CLOUD_BUILD}"
    return $?
  fi
  return 0
}

load_config() {
  if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi

  PROJECT_ID="$(strip_quotes "${GOOGLE_CLOUD_PROJECT:-}")"
  if [ -z "${GCP_REGION:-}" ] && [ -n "${GCP_BATCH_REGION:-}" ]; then
    echo "NOTE: GCP_BATCH_REGION is deprecated; set GCP_REGION instead." >&2
    GCP_REGION="${GCP_BATCH_REGION}"
  fi
  REGION="$(strip_quotes "${GCP_REGION:-us-central1}")"
  VPC_PREFIX="$(strip_quotes "${VPC_NAME_PREFIX:-}")"
  SUBNET_CIDR="$(strip_quotes "${VPC_SUBNET_CIDR:-10.0.0.0/16}")"
  ENABLE_NAT="$(strip_quotes "${ENABLE_CLOUD_NAT:-false}")"
  NO_EXT_IP="$(strip_quotes "${GCP_BATCH_NO_EXTERNAL_IP:-true}")"
  MODEL_LOCATION="$(strip_quotes "${GOOGLE_CLOUD_LOCATION:-global}")"
  MODEL="$(strip_quotes "${MODEL_ID:-gemini-4-argon}")"
  ENABLE_IAP="$(strip_quotes "${ENABLE_DASHBOARD_IAP:-true}")"
  DASHBOARD_MEMBERS="$(strip_quotes "${DASHBOARD_AUTHORIZED_MEMBERS:-}")"
  UW_MEMBERS="$(strip_quotes "${UW_MRX_WRITER_MEMBERS:-}")"
  ENABLE_WIF="$(strip_quotes "${ENABLE_GITHUB_WIF:-false}")"
  WIF_REPOSITORY="$(strip_quotes "${GITHUB_WIF_REPOSITORY:-}")"
  CAMPAIGN="$(strip_quotes "${CAMPAIGN_ID:-pilot_50}")"
  CPU_ONLY="$(strip_quotes "${GCP_BATCH_CPU_ONLY:-0}")"
  CANDIDATE_REGIONS="$(strip_quotes "${GCP_BATCH_CANDIDATE_REGIONS:-}")"

  [ -n "${PROJECT_ID}" ] || die "GOOGLE_CLOUD_PROJECT is not set (copy .env.example to .env and set it)"
  [[ "${PROJECT_ID}" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "GOOGLE_CLOUD_PROJECT='${PROJECT_ID}' is not a valid project ID"
  [ "${PROJECT_ID}" != "your-gcp-project-id" ] || die "GOOGLE_CLOUD_PROJECT is still the .env.example placeholder"
  [[ "${REGION}" =~ ^[a-z]+(-[a-z]+)+[0-9]+$ ]] || die "GCP_REGION='${REGION}' is not a valid region"
  if [ -n "${VPC_PREFIX}" ]; then
    [[ "${VPC_PREFIX}" =~ ^[a-z][a-z0-9-]{0,53}[a-z0-9]$ ]] ||
      die "VPC_NAME_PREFIX='${VPC_PREFIX}' must be 2-55 lowercase letters, digits and hyphens, starting with a letter and ending with a letter or digit"
  fi
  python3 -c 'import ipaddress, sys; s = sys.argv[1]; n = ipaddress.IPv4Network(s, strict=True) if "/" in s else None; raise SystemExit(0 if n is not None and 8 <= n.prefixlen <= 24 else 1)' "${SUBNET_CIDR}" 2>/dev/null ||
    die "VPC_SUBNET_CIDR='${SUBNET_CIDR}' must be a valid IPv4 network CIDR with prefix length /8 to /24 (e.g. 10.0.0.0/16)"
  if [ -n "${CANDIDATE_REGIONS}" ]; then
    python3 -c 'import sys; reg, raw = sys.argv[1], sys.argv[2]; rs = [r.strip() for r in raw.split(",") if r.strip()]; raise SystemExit(0 if rs and all(r == reg for r in rs) else 1)' "${REGION}" "${CANDIDATE_REGIONS}" 2>/dev/null ||
      die "GCP_BATCH_CANDIDATE_REGIONS='${CANDIDATE_REGIONS}' must match GCP_REGION='${REGION}' (Terraform provisions a single regional subnet in ${REGION})"
  fi
  case "${DASHBOARD_MEMBERS}" in
    *your-team@example.org*) die "DASHBOARD_AUTHORIZED_MEMBERS is still the .env.example placeholder" ;;
  esac
  case "$(printf '%s' "${ENABLE_IAP}" | tr '[:upper:]' '[:lower:]')" in
    true | false) ENABLE_IAP="$(printf '%s' "${ENABLE_IAP}" | tr '[:upper:]' '[:lower:]')" ;;
    *) die "ENABLE_DASHBOARD_IAP must be true or false" ;;
  esac
  case "$(printf '%s' "${ENABLE_WIF}" | tr '[:upper:]' '[:lower:]')" in
    true | false) ENABLE_WIF="$(printf '%s' "${ENABLE_WIF}" | tr '[:upper:]' '[:lower:]')" ;;
    *) die "ENABLE_GITHUB_WIF must be true or false" ;;
  esac
  case "$(printf '%s' "${ENABLE_NAT}" | tr '[:upper:]' '[:lower:]')" in
    true | false) ENABLE_NAT="$(printf '%s' "${ENABLE_NAT}" | tr '[:upper:]' '[:lower:]')" ;;
    *) die "ENABLE_CLOUD_NAT must be true or false" ;;
  esac
  case "$(printf '%s' "${NO_EXT_IP}" | tr '[:upper:]' '[:lower:]')" in
    true | false) NO_EXT_IP="$(printf '%s' "${NO_EXT_IP}" | tr '[:upper:]' '[:lower:]')" ;;
    *) die "GCP_BATCH_NO_EXTERNAL_IP must be true or false" ;;
  esac

  STATE_BUCKET="$(strip_quotes "${TF_STATE_BUCKET:-${PROJECT_ID}-tfstate}")"
  AR_HOST="${REGION}-docker.pkg.dev"
  AR_REPO="${AR_HOST}/${PROJECT_ID}/aurora-platform"
}

# Comma-separated principals -> HCL/JSON list literal for -var.
to_hcl_list() {
  python3 -c 'import json, sys; print(json.dumps([m.strip() for m in sys.argv[1].split(",") if m.strip()]))' "$1"
}

image_tag() {
  if [ -n "${IMAGE_TAG:-}" ]; then
    printf '%s' "${IMAGE_TAG}"
  elif [ -n "${GIT_SHA:-}" ]; then
    printf '%s' "${GIT_SHA}"
  elif git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    local sha
    sha="$(git rev-parse HEAD)"
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
      # Uncommitted changes: unique, clearly marked tag (never reused).
      printf '%s-dirty-%s' "${sha:0:12}" "$(date -u +%Y%m%dT%H%M%SZ)"
    else
      printf '%s' "${sha}"
    fi
  else
    printf 'build-%s' "$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

dockerfile_for() {
  case "$1" in
    aurora-steer) printf '%s' "platform/aurora-steer/Dockerfile" ;;
    dashboard) printf '%s' "platform/dashboard/Dockerfile" ;;
    gx-runner) printf '%s' "workstreams/kr2-gk-database/gcp/Dockerfile.gx-runner" ;;
    *) die "unknown image '$1'" ;;
  esac
}

# Prints "<repo>/<image>@sha256:<digest>". Builds and pushes only when the tag is new
# (Artifact Registry tags are immutable). gx-runner builds need GX_COMMIT (40-hex GX git SHA);
# its tag carries that commit so a GX bump always produces a new image.
build_image() {
  local name="$1" tag="$2" ref digest dockerfile
  local build_args=()
  dockerfile="$(dockerfile_for "${name}")"
  if [ "${name}" = "gx-runner" ]; then
    [[ "${GX_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die "GX_COMMIT must be the 40-hex GX git commit to build (see .env.example)"
    build_args=(--build-arg "GX_COMMIT=${GX_COMMIT}")
    tag="${tag}-gx${GX_COMMIT:0:12}"
  fi
  ref="${AR_REPO}/${name}:${tag}"
  if digest="$(gcloud artifacts docker images describe "${ref}" --project="${PROJECT_ID}" \
    --format='value(image_summary.digest)' 2>/dev/null)" && [ -n "${digest}" ]; then
    echo "Reusing existing ${ref}" >&2
  elif use_cloud_build; then
    echo "Building ${ref} via Cloud Build" >&2
    local tmp_cb
    tmp_cb="$(mktemp)"
    python3 - "${tmp_cb}" "${dockerfile}" "${ref}" "${name}" "${GX_COMMIT:-}" <<'PY'
import json
import pathlib
import sys

out, dockerfile, ref, name, gx_commit = sys.argv[1:6]
args = ["build", "--platform", "linux/amd64"]
if name == "gx-runner":
    args.extend(["--build-arg", f"GX_COMMIT={gx_commit}"])
args.extend(["-f", dockerfile, "-t", ref, "."])
cfg = {
    "steps": [{"name": "gcr.io/cloud-builders/docker", "args": args}],
    "images": [ref],
    "timeout": "3600s",
    "options": {
        "logging": "CLOUD_LOGGING_ONLY",
        **({"machineType": "E2_HIGHCPU_32"} if name == "gx-runner" else {}),
    },
}
pathlib.Path(out).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY
    local attempt
    for attempt in 1 2 3; do
      if gcloud builds submit . \
        --project="${PROJECT_ID}" \
        --region="${REGION}" \
        --ignore-file="${REPO_ROOT}/.dockerignore" \
        --config="${tmp_cb}" \
        --service-account="projects/${PROJECT_ID}/serviceAccounts/aurora-cloudbuild-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --default-buckets-behavior=regional-user-owned-bucket \
        --quiet >&2; then
        break
      fi
      [ "${attempt}" -lt 3 ] || { rm -f "${tmp_cb}"; die "Cloud Build failed for ${ref} after 3 attempts"; }
      echo "Retrying Cloud Build for ${ref} (attempt $((attempt + 1))/3)..." >&2
      sleep 3
    done
    rm -f "${tmp_cb}"
    digest="$(gcloud artifacts docker images describe "${ref}" --project="${PROJECT_ID}" \
      --format='value(image_summary.digest)')"
  else
    echo "Building ${ref}" >&2
    docker build --platform linux/amd64 ${build_args[@]+"${build_args[@]}"} -f "${dockerfile}" -t "${ref}" . >&2
    docker push "${ref}" >&2
    digest="$(gcloud artifacts docker images describe "${ref}" --project="${PROJECT_ID}" \
      --format='value(image_summary.digest)')"
  fi
  [[ "${digest}" =~ ^sha256:[a-f0-9]{64}$ ]] || die "could not resolve the digest of ${ref}"
  printf '%s/%s@%s' "${AR_REPO}" "${name}" "${digest}"
}

current_service_image() {
  gcloud run services describe "$1" --project="${PROJECT_ID}" --region="${REGION}" \
    --format='value(spec.template.spec.containers[0].image)' 2>/dev/null || true
}

current_job_image() {
  gcloud run jobs describe "${BABYSITTER_JOB}" --project="${PROJECT_ID}" --region="${REGION}" \
    --format='value(spec.template.spec.template.spec.containers[0].image)' 2>/dev/null || true
}

# Roll a Cloud Run service to an image digest; configuration stays owned by Terraform.
roll_service() {
  local service="$1" image="$2"
  if [ "$(current_service_image "${service}")" = "${image}" ]; then
    echo "${service} already runs ${image}"
    return 0
  fi
  gcloud run services update "${service}" --project="${PROJECT_ID}" --region="${REGION}" \
    --image="${image}" --quiet
}

service_for_image() {
  case "$1" in
    aurora-steer) printf '%s' "${STEER_SERVICE}" ;;
    dashboard) printf '%s' "${DASHBOARD_SERVICE}" ;;
    *) die "release supports aurora-steer and dashboard (got '$1')" ;;
  esac
}

ensure_state_bucket() {
  if gcloud storage buckets describe "gs://${STATE_BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    return 0
  fi
  log "Creating Terraform state bucket gs://${STATE_BUCKET}"
  gcloud storage buckets create "gs://${STATE_BUCKET}" --project="${PROJECT_ID}" \
    --location="${REGION}" --uniform-bucket-level-access --public-access-prevention
  gcloud storage buckets update "gs://${STATE_BUCKET}" --project="${PROJECT_ID}" --versioning
}

run_terraform() {
  if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    local tok
    tok="$(gcloud auth print-access-token 2>/dev/null || true)"
    if [ -n "${tok}" ]; then
      GOOGLE_OAUTH_ACCESS_TOKEN="${tok}" terraform -chdir="${TF_DIR}" "$@"
      return $?
    fi
  fi
  terraform -chdir="${TF_DIR}" "$@"
}

terraform_init() {
  ensure_state_bucket
  run_terraform init -input=false \
    -backend-config="bucket=${STATE_BUCKET}" -backend-config="prefix=platform/infra"
}

tf_args() {
  # $1 aurora-steer image, $2 dashboard image, $3 gx-runner image (may be empty)
  TF_ARGS=(
    -var="project_id=${PROJECT_ID}"
    -var="region=${REGION}"
    -var="vpc_name_prefix=${VPC_PREFIX}"
    -var="vpc_subnet_cidr=${SUBNET_CIDR}"
    -var="enable_cloud_nat=${ENABLE_NAT}"
    -var="model_location=${MODEL_LOCATION}"
    -var="model_id=${MODEL}"
    -var="aurora_steer_image=$1"
    -var="dashboard_image=$2"
    -var="gx_runner_image=$3"
    -var="enable_dashboard_iap=${ENABLE_IAP}"
    -var="dashboard_authorized_members=$(to_hcl_list "${DASHBOARD_MEMBERS}")"
    -var="uw_mrx_writer_members=$(to_hcl_list "${UW_MEMBERS}")"
    -var="enable_github_wif=${ENABLE_WIF}"
    -var="github_repository=${WIF_REPOSITORY}"
  )
}

ensure_secret_version() {
  local existing
  existing="$(gcloud secrets versions list "${SECRET_ID}" --project="${PROJECT_ID}" \
    --filter="state=ENABLED" --format="value(name)" --limit=1)"
  if [ -n "${existing}" ]; then
    echo "Secret Manager '${SECRET_ID}' already has an enabled version (left unchanged)."
    return 0
  fi
  # Generated straight into Secret Manager: never stored in a variable, a file or .env.
  openssl rand -hex 32 | tr -d '\n' |
    gcloud secrets versions add "${SECRET_ID}" --project="${PROJECT_ID}" --data-file=- >/dev/null
  echo "Added the first version of '${SECRET_ID}'."
}

write_babysitter_config() {
  # $1 output path, $2 gx-runner image, $3 aurora-steer URL, $4 bucket, $5 worker SA,
  # $6 vpc_network URI, $7 vpc_subnetwork URI
  local environment="prod"
  if is_true "${CPU_ONLY}"; then
    environment="sim"
    echo "NOTE: GCP_BATCH_CPU_ONLY is set, so the babysitter runs with environment=sim (CPU-only GX is refused in prod; sim runs never count in KPIs)." >&2
  fi
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "${PROJECT_ID}" "${REGION}" "${CAMPAIGN}" \
    "${environment}" "${CANDIDATE_REGIONS}" "${NO_EXT_IP}" <<'PY'
import json
import pathlib
import sys

(
    out,
    image,
    steer_url,
    bucket,
    worker_sa,
    vpc_net,
    vpc_subnet,
    project,
    region,
    campaign,
    environment,
    regions,
    no_ext_ip_raw,
) = sys.argv[1:14]
no_ext_ip = no_ext_ip_raw.strip().lower() == "true"
cfg = json.loads(
    pathlib.Path("workstreams/kr2-gk-database/gcp/config.batch.example.json").read_text(encoding="utf-8")
)
cfg.update(
    {
        "workstream": "kr2-gk-database",
        "backend": "gcp_batch",
        "campaign_id": campaign,
        "environment": environment,
        "aurora_steer_url": steer_url,
        "gcp_project_id": project,
        "gcp_region": region,
        "candidate_regions": [r.strip() for r in regions.split(",") if r.strip()] if regions.strip() else [region],
        "network": vpc_net,
        "subnetwork": vpc_subnet,
        "no_external_ip": no_ext_ip,
        "gcs_bucket": bucket,
        "gcs_mount_path": "/mnt/disks/gk_bucket",
        "gx_container_image": image,
        "worker_service_account_email": worker_sa,
        "cpu_only": environment != "prod",
        "queue_dir": "/mnt/gk_bucket/queue",
        "state_dir": "/mnt/gk_bucket/state",
        "dataset_output_dir": "/mnt/gk_bucket",
        "dataset_gcs_uri_prefix": f"gs://{bucket}",
    }
)
cfg.pop("project_id", None)
cfg.pop("region", None)
pathlib.Path(out).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY
}

update_env_file() {
  # Non-secret convenience values only.
  [ -f .env ] || return 0
  python3 - "$1" "$2" <<'PY'
import os
import pathlib
import re
import sys

steer_url, bucket = sys.argv[1], sys.argv[2]
path = pathlib.Path(".env")
text = path.read_text(encoding="utf-8")
for key, value in (("AURORA_STEER_URL", steer_url), ("GK_DATASET_BUCKET", bucket)):
    if re.search(rf"^{key}=", text, flags=re.M):
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
    else:
        text += f"\n{key}={value}\n"
tmp = path.with_name(".env.tmp")
tmp.write_text(text, encoding="utf-8")
os.chmod(tmp, path.stat().st_mode & 0o777)
os.replace(tmp, path)
PY
}

cmd_init() {
  require_tools gcloud terraform python3
  load_config
  log "Terraform init (state: gs://${STATE_BUCKET}/platform/infra)"
  terraform_init
}

cmd_plan_or_apply() {
  local action="$1" steer_img dash_img gx_img
  require_tools gcloud terraform python3
  load_config
  terraform_init
  steer_img="${AURORA_STEER_IMAGE:-$(current_service_image "${STEER_SERVICE}")}"
  dash_img="${DASHBOARD_IMAGE:-$(current_service_image "${DASHBOARD_SERVICE}")}"
  gx_img="${GX_CONTAINER_IMAGE:-$(current_job_image)}"
  [ -n "${steer_img}" ] && [ -n "${dash_img}" ] ||
    die "Cloud Run services not found; run 'make gcp-deploy' once to build images and create them"
  tf_args "${steer_img}" "${dash_img}" "${gx_img}"
  if [ "${action}" = "plan" ]; then
    run_terraform plan -input=false "${TF_ARGS[@]}"
  else
    run_terraform apply -input=false "${TF_ARGS[@]}"
  fi
}

cmd_release() {
  [ "$#" -gt 0 ] || die "usage: $0 release aurora-steer|dashboard [...]"
  : "${USE_CLOUD_BUILD:=false}"
  require_tools gcloud python3
  load_config
  if ! use_cloud_build; then
    require_tools docker
    gcloud auth configure-docker "${AR_HOST}" --quiet >/dev/null
  fi
  local tag name image
  tag="$(image_tag)"
  for name in "$@"; do
    service_for_image "${name}" >/dev/null
    image="$(build_image "${name}" "${tag}")"
    log "Rolling $(service_for_image "${name}") to ${image}"
    roll_service "$(service_for_image "${name}")" "${image}"
  done
}

cmd_deploy() {
  require_tools gcloud terraform python3 openssl
  load_config
  if ! use_cloud_build; then
    require_tools docker
  fi
  local tag steer_img dash_img gx_img steer_url dash_url bucket worker_sa vpc_net vpc_subnet
  if [ -n "${AURORA_STEER_BEARER_TOKEN:-}" ]; then
    echo "NOTE: AURORA_STEER_BEARER_TOKEN is set locally; this script never uploads local tokens (see the token rotation section of docs/RUNBOOK_DEMO_AND_TACC_SMOKE_TEST.md)." >&2
  fi

  [[ "${GX_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die "GX_COMMIT must be set to the 40-hex GX git commit to build (see .env.example)"

  log "[1/7] Terraform state bucket + init"
  terraform_init

  log "[2/7] Bootstrapping APIs, Artifact Registry, Cloud Build SA, Secret shell, GCS and BigQuery"
  tag="$(image_tag)"
  tf_args "${AR_REPO}/aurora-steer:${tag}" "${AR_REPO}/dashboard:${tag}" ""
  run_terraform apply -input=false -auto-approve "${TF_ARGS[@]}" \
    -target=google_project_service.apis \
    -target=google_artifact_registry_repository.containers \
    -target=google_service_account.cloudbuild \
    -target=google_project_iam_member.cloudbuild_logs \
    -target=google_artifact_registry_repository_iam_member.cloudbuild_ar_writer \
    -target=google_project_iam_member.cloudbuild_storage \
    -target=google_secret_manager_secret.bearer_token \
    -target=google_storage_bucket.gk_dataset \
    -target=google_bigquery_dataset.telemetry \
    -target=google_bigquery_table.campaigns \
    -target=google_bigquery_table.assumptions \
    -target=google_bigquery_table.events \
    -target=google_bigquery_table.directives \
    -target=google_bigquery_table.escalations \
    -target=google_bigquery_table.telemetry_views_tier1 \
    -target=google_bigquery_table.telemetry_views_tier2 \
    -target=google_bigquery_table.telemetry_views_tier3

  log "[3/7] Ensuring Secret Manager '${SECRET_ID}' has an enabled version"
  ensure_secret_version

  log "[4/7] Building and pushing images (tag ${tag}) to ${AR_REPO}"
  if ! use_cloud_build; then
    gcloud auth configure-docker "${AR_HOST}" --quiet >/dev/null
  fi
  steer_img="$(build_image aurora-steer "${tag}")"
  dash_img="$(build_image dashboard "${tag}")"
  gx_img="$(build_image gx-runner "${tag}")"

  log "[5/7] Applying the full Terraform stack"
  tf_args "${steer_img}" "${dash_img}" "${gx_img}"
  run_terraform apply -input=false -auto-approve "${TF_ARGS[@]}"

  log "[6/7] Rolling Cloud Run service images by digest"
  roll_service "${STEER_SERVICE}" "${steer_img}"
  roll_service "${DASHBOARD_SERVICE}" "${dash_img}"

  steer_url="$(run_terraform output -raw aurora_steer_url)"
  dash_url="$(run_terraform output -raw dashboard_url)"
  bucket="$(run_terraform output -raw gk_dataset_bucket_name)"
  worker_sa="$(run_terraform output -raw gcp_batch_worker_service_account)"
  vpc_net="$(run_terraform output -raw vpc_network)"
  vpc_subnet="$(run_terraform output -raw vpc_subnetwork)"
  [ -n "${vpc_net}" ] && [ -n "${vpc_subnet}" ] || die "Terraform outputs vpc_network / vpc_subnetwork are empty"
  update_env_file "${steer_url}" "${bucket}"

  log "[7/7] Uploading babysitter config to gs://${bucket}/config/batch_babysitter.json"
  TMP_CFG="$(mktemp)"
  trap 'rm -f "${TMP_CFG:-}"' EXIT
  write_babysitter_config "${TMP_CFG}" "${gx_img}" "${steer_url}" "${bucket}" "${worker_sa}" \
    "${vpc_net}" "${vpc_subnet}"
  gcloud storage cp "${TMP_CFG}" "gs://${bucket}/config/batch_babysitter.json" --project="${PROJECT_ID}"

  cat <<EOF

==============================================================================
GCP deployment complete.
  aurora-steer URL : ${steer_url}
  Dashboard URL    : ${dash_url}
  GCS dataset      : gs://${bucket}
  Batch subnet     : ${vpc_subnet} (no external IP)
  Images           : ${steer_img}
                     ${dash_img}
                     ${gx_img}

The bearer token lives only in Secret Manager. To hand it to a TACC operator:
  gcloud secrets versions access latest --secret=${SECRET_ID} --project=${PROJECT_ID}
Then run 'python3 scripts/tacc_smoke_test.py' on a TACC login node
(AURORA_STEER_URL=${steer_url}, AURORA_STEER_BEARER_TOKEN=<token>).
==============================================================================
EOF
}

main() {
  local cmd="${1:-deploy}"
  [ "$#" -gt 0 ] && shift
  case "${cmd}" in
    deploy | all) cmd_deploy ;;
    init) cmd_init ;;
    plan) cmd_plan_or_apply plan ;;
    apply) cmd_plan_or_apply apply ;;
    release) cmd_release "$@" ;;
    -h | --help | help) sed -n '2,30p' "${BASH_SOURCE[0]}" ;;
    *) die "unknown command '${cmd}' (deploy|init|plan|apply|release)" ;;
  esac
}

main "$@"
