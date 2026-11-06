"""Behavioural tests for scripts/deploy_gcp_demo.sh with fake gcloud/terraform/docker/openssl.

The fakes log every invocation, so the tests can check the deploy contract without a cloud:
remote state, a secret version generated straight into Secret Manager (the token is never
printed and never written to .env), images built under an immutable tag and rolled by digest,
and a babysitter config built from Terraform outputs.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from gcp_exec.runner import build_backend, validate_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "deploy_gcp_demo.sh"
FAKE_TOKEN = "f" * 64
PROJECT = "demo-project-123"
GX_COMMIT = "0123456789abcdef0123456789abcdef01234567"

FAKE_GCLOUD = r"""#!/usr/bin/env bash
set -u
echo "gcloud $*" >> "$FAKE_LOG"
args="$*"
case "$args" in
  "storage buckets describe"*) [ -f "$FAKE_STATE/state_bucket" ] && exit 0 || exit 1 ;;
  "storage buckets create"*) touch "$FAKE_STATE/state_bucket" ;;
  "storage buckets update"*) : ;;
  "secrets versions list"*) if [ -f "$FAKE_STATE/secret" ]; then echo "projects/p/secrets/s/versions/1"; fi ;;
  "secrets versions add"*) cat > "$FAKE_STATE/secret" ;;
  "auth configure-docker"*) : ;;
  "auth print-access-token"*) echo "fake-oauth-access-token" ;;
  "artifacts docker images describe"*)
    ref="$5"
    grep -qxF "$ref" "$FAKE_STATE/registry" 2>/dev/null || exit 1
    printf 'sha256:%s\n' "$(printf '%s' "$ref" | (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-64)" ;;
  "builds submit"*)
    for a in "$@"; do
      case "$a" in
        --config=*)
          cfg_file="${a#--config=}"
          cat "$cfg_file" >> "$FAKE_LOG"
          python3 -c 'import json, sys; c = json.load(open(sys.argv[1])); print("\n".join(c["images"]))' "$cfg_file" >> "$FAKE_STATE/registry"
          ;;
      esac
    done ;;
  "run services describe"*)
    svc="$4"; [ -f "$FAKE_STATE/svc_$svc" ] && cat "$FAKE_STATE/svc_$svc" || exit 1 ;;
  "run services update"*)
    svc="$4"
    for a in "$@"; do case "$a" in --image=*) printf '%s\n' "${a#--image=}" > "$FAKE_STATE/svc_$svc" ;; esac; done ;;
  "run jobs describe"*) exit 1 ;;
  "storage cp"*) cp "$3" "$FAKE_STATE/uploaded_config.json" ;;
  *) echo "unexpected gcloud call: $args" >&2; exit 3 ;;
esac
"""

FAKE_DOCKER = r"""#!/usr/bin/env bash
echo "docker $*" >> "$FAKE_LOG"
if [ "$1" = "push" ]; then echo "$2" >> "$FAKE_STATE/registry"; fi
"""

FAKE_TERRAFORM = r"""#!/usr/bin/env bash
echo "terraform $*" >> "$FAKE_LOG"
case "$2" in
  apply)
    for a in "$@"; do
      case "$a" in
        -var=project_id=*) printf '%s' "${a#-var=project_id=}" > "$FAKE_STATE/tf_project" ;;
        -var=region=*) printf '%s' "${a#-var=region=}" > "$FAKE_STATE/tf_region" ;;
        -var=vpc_name_prefix=*) printf '%s' "${a#-var=vpc_name_prefix=}" > "$FAKE_STATE/tf_vpc_prefix" ;;
      esac
    done ;;
  output)
    proj="demo-project-123"
    reg="us-central1"
    pfx=""
    if [ -f "$FAKE_STATE/tf_project" ]; then proj="$(cat "$FAKE_STATE/tf_project")"; fi
    if [ -f "$FAKE_STATE/tf_region" ]; then reg="$(cat "$FAKE_STATE/tf_region")"; fi
    if [ -f "$FAKE_STATE/tf_vpc_prefix" ]; then pfx="$(cat "$FAKE_STATE/tf_vpc_prefix")"; fi
    net_pfx="${pfx:-$proj}"
    case "$4" in
      aurora_steer_url) printf 'https://aurora-steer-abc-uc.a.run.app' ;;
      dashboard_url) printf 'https://aurora-steer-dashboard-abc-uc.a.run.app' ;;
      gk_dataset_bucket_name) printf '%s-gk-dataset' "$proj" ;;
      gcp_batch_worker_service_account) printf 'gx-batch-worker-sa@%s.iam.gserviceaccount.com' "$proj" ;;
      vpc_network) printf 'projects/%s/global/networks/%s-vpc' "$proj" "$net_pfx" ;;
      vpc_subnetwork) printf 'projects/%s/regions/%s/subnetworks/%s-subnet' "$proj" "$reg" "$net_pfx" ;;
    esac ;;
esac
"""

FAKE_OPENSSL = f"""#!/usr/bin/env bash
echo "openssl $*" >> "$FAKE_LOG"
echo {FAKE_TOKEN}
"""


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture()
def sandbox(tmp_path: Path) -> dict[str, Path]:
    if shutil.which("bash") is None or shutil.which("python3") is None:
        pytest.skip("bash and python3 are required")
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    (root / "platform" / "infra" / "terraform").mkdir(parents=True)
    cfg_dir = root / "workstreams" / "kr2-gk-database" / "gcp"
    cfg_dir.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "workstreams/kr2-gk-database/gcp/config.batch.example.json", cfg_dir)
    (root / ".env").write_text(
        f"GOOGLE_CLOUD_PROJECT={PROJECT}\nAURORA_STEER_URL=http://127.0.0.1:8080\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_exe(bin_dir / "gcloud", FAKE_GCLOUD)
    _write_exe(bin_dir / "docker", FAKE_DOCKER)
    _write_exe(bin_dir / "terraform", FAKE_TERRAFORM)
    _write_exe(bin_dir / "openssl", FAKE_OPENSSL)
    state = tmp_path / "state"
    state.mkdir()
    return {"root": root, "bin": bin_dir, "state": state, "log": tmp_path / "calls.log"}


def _run(sb: dict[str, Path], *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {
        "PATH": f"{sb['bin']}:/usr/bin:/bin",
        "HOME": str(sb["root"]),
        "FAKE_LOG": str(sb["log"]),
        "FAKE_STATE": str(sb["state"]),
        "IMAGE_TAG": "abc123",
        "GX_COMMIT": GX_COMMIT,
        "USE_CLOUD_BUILD": "false",
        **env,
    }
    return subprocess.run(
        ["bash", str(sb["root"] / "scripts" / SCRIPT.name), *args],
        cwd=sb["root"],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_script_is_executable_and_parses() -> None:
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert ":latest" not in text


def test_full_deploy_contract(sandbox: dict[str, Path]) -> None:
    proc = _run(sandbox)
    assert proc.returncode == 0, proc.stderr
    calls = sandbox["log"].read_text()

    # Remote state: bucket created (versioned, private) before init with backend config.
    assert "gcloud storage buckets create gs://demo-project-123-tfstate" in calls
    assert "--public-access-prevention" in calls and "--versioning" in calls
    assert "-backend-config=bucket=demo-project-123-tfstate" in calls

    # Token generated straight into Secret Manager, never echoed, never written to .env.
    assert (sandbox["state"] / "secret").read_text() == FAKE_TOKEN
    assert FAKE_TOKEN not in proc.stdout + proc.stderr + calls
    env_text = (sandbox["root"] / ".env").read_text()
    assert FAKE_TOKEN not in env_text
    assert "AURORA_STEER_URL=https://aurora-steer-abc-uc.a.run.app" in env_text

    # Immutable tag, digest-pinned Terraform vars, image rolls by digest.
    registry = (sandbox["state"] / "registry").read_text().split()
    repo = f"us-central1-docker.pkg.dev/{PROJECT}/aurora-platform"
    assert sorted(registry) == sorted(
        [
            f"{repo}/aurora-steer:abc123",
            f"{repo}/dashboard:abc123",
            f"{repo}/gx-runner:abc123-gx{GX_COMMIT[:12]}",
        ]
    )
    assert f"--build-arg GX_COMMIT={GX_COMMIT}" in calls
    full_apply = [
        line for line in calls.splitlines() if " apply " in line and "-target" not in line
    ]
    assert len(full_apply) == 1
    assert (
        "aurora_steer_image=us-central1-docker.pkg.dev/demo-project-123/aurora-platform/aurora-steer@sha256:"
        in full_apply[0]
    )
    assert "enable_dashboard_iap=true" in full_apply[0]
    assert "vpc_name_prefix=" in full_apply[0]
    assert "vpc_subnet_cidr=10.0.0.0/16" in full_apply[0]
    assert "enable_cloud_nat=false" in full_apply[0]
    steer = (sandbox["state"] / "svc_aurora-steer").read_text().strip()
    assert "@sha256:" in steer

    # Babysitter config comes from Terraform outputs and pins the GX image by digest.
    cfg = json.loads((sandbox["state"] / "uploaded_config.json").read_text())
    assert cfg["environment"] == "prod" and cfg["cpu_only"] is False
    assert (
        cfg["worker_service_account_email"]
        == "gx-batch-worker-sa@demo-project-123.iam.gserviceaccount.com"
    )
    assert "@sha256:" in cfg["gx_container_image"]
    assert cfg["gcs_bucket"] == "demo-project-123-gk-dataset"
    assert cfg["aurora_steer_url"] == "https://aurora-steer-abc-uc.a.run.app"
    assert cfg["gcp_region"] == "us-central1"
    assert cfg["candidate_regions"] == ["us-central1"]
    assert cfg["network"] == "projects/demo-project-123/global/networks/demo-project-123-vpc"
    assert (
        cfg["subnetwork"]
        == "projects/demo-project-123/regions/us-central1/subnetworks/demo-project-123-subnet"
    )
    assert cfg["no_external_ip"] is True
    validate_config(cfg)
    backend, mode = build_backend(cfg, sandbox["state"])
    assert mode == "gcp_batch"
    assert backend.no_external_ip is True


def test_redeploy_is_idempotent(sandbox: dict[str, Path]) -> None:
    assert _run(sandbox).returncode == 0
    first_registry = (sandbox["state"] / "registry").read_text()
    sandbox["log"].write_text("")
    proc = _run(sandbox)
    assert proc.returncode == 0, proc.stderr
    calls = sandbox["log"].read_text()
    assert "docker build" not in calls, "existing immutable tags must be reused, not rebuilt"
    assert "secrets versions add" not in calls, "an existing secret version must never be replaced"
    assert "run services update" not in calls, (
        "services already on the digest must not be re-rolled"
    )
    assert (sandbox["state"] / "registry").read_text() == first_registry


def test_cpu_only_never_runs_as_prod(sandbox: dict[str, Path]) -> None:
    proc = _run(sandbox, GCP_BATCH_CPU_ONLY="1")
    assert proc.returncode == 0, proc.stderr
    cfg = json.loads((sandbox["state"] / "uploaded_config.json").read_text())
    assert cfg["environment"] == "sim" and cfg["cpu_only"] is True


def test_custom_vpc_and_region_wiring(sandbox: dict[str, Path]) -> None:
    proc = _run(
        sandbox,
        GCP_REGION="us-east4",
        VPC_NAME_PREFIX="ut-gk-prod",
        VPC_SUBNET_CIDR="172.24.0.0/18",
        ENABLE_CLOUD_NAT="true",
    )
    assert proc.returncode == 0, proc.stderr
    calls = sandbox["log"].read_text()
    full_apply = [
        line for line in calls.splitlines() if " apply " in line and "-target" not in line
    ]
    assert len(full_apply) == 1
    assert "-var=region=us-east4" in full_apply[0]
    assert "-var=vpc_name_prefix=ut-gk-prod" in full_apply[0]
    assert "-var=vpc_subnet_cidr=172.24.0.0/18" in full_apply[0]
    assert "-var=enable_cloud_nat=true" in full_apply[0]

    cfg = json.loads((sandbox["state"] / "uploaded_config.json").read_text())
    assert cfg["gcp_region"] == "us-east4"
    assert cfg["candidate_regions"] == ["us-east4"]
    assert cfg["network"] == "projects/demo-project-123/global/networks/ut-gk-prod-vpc"
    assert (
        cfg["subnetwork"]
        == "projects/demo-project-123/regions/us-east4/subnetworks/ut-gk-prod-subnet"
    )
    assert cfg["no_external_ip"] is True
    validate_config(cfg)
    backend, mode = build_backend(cfg, sandbox["state"])
    assert mode == "gcp_batch"
    assert backend.region == "us-east4"


def test_explicit_co_located_candidate_region_wiring(sandbox: dict[str, Path]) -> None:
    proc = _run(sandbox, GCP_REGION="us-east4", GCP_BATCH_CANDIDATE_REGIONS=" us-east4 ")
    assert proc.returncode == 0, proc.stderr
    cfg = json.loads((sandbox["state"] / "uploaded_config.json").read_text())
    assert cfg["candidate_regions"] == ["us-east4"]
    validate_config(cfg)


def test_release_rolls_only_the_requested_service(sandbox: dict[str, Path]) -> None:
    proc = _run(sandbox, "release", "aurora-steer", GIT_SHA="d" * 40, IMAGE_TAG="")
    assert proc.returncode == 0, proc.stderr
    calls = sandbox["log"].read_text()
    assert "terraform" not in calls, "release must not need Terraform permissions"
    assert f"aurora-steer:{'d' * 40}" in calls
    assert "svc_aurora-steer-dashboard" not in str(list(sandbox["state"].iterdir()))
    assert "@sha256:" in (sandbox["state"] / "svc_aurora-steer").read_text()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"GOOGLE_CLOUD_PROJECT": "your-gcp-project-id"}, "placeholder"),
        ({"GOOGLE_CLOUD_PROJECT": "Bad_Project"}, "not a valid project ID"),
        ({"ENABLE_DASHBOARD_IAP": "maybe"}, "ENABLE_DASHBOARD_IAP"),
        ({"GX_COMMIT": "main"}, "GX_COMMIT"),
        ({"VPC_NAME_PREFIX": "Bad_Prefix"}, "VPC_NAME_PREFIX"),
        ({"VPC_NAME_PREFIX": "a"}, "VPC_NAME_PREFIX"),
        ({"VPC_NAME_PREFIX": "a" * 56}, "VPC_NAME_PREFIX"),
        ({"VPC_NAME_PREFIX": "bad-trailing-"}, "VPC_NAME_PREFIX"),
        ({"VPC_SUBNET_CIDR": "10.0.0.0/28"}, "VPC_SUBNET_CIDR"),
        ({"VPC_SUBNET_CIDR": "10.0.0.0/7"}, "VPC_SUBNET_CIDR"),
        ({"VPC_SUBNET_CIDR": "10.0.0.1/16"}, "VPC_SUBNET_CIDR"),
        ({"VPC_SUBNET_CIDR": "10.0.0.0"}, "VPC_SUBNET_CIDR"),
        ({"VPC_SUBNET_CIDR": "2001:db8::/32"}, "VPC_SUBNET_CIDR"),
        ({"VPC_SUBNET_CIDR": "not-a-cidr"}, "VPC_SUBNET_CIDR"),
        ({"ENABLE_CLOUD_NAT": "maybe"}, "ENABLE_CLOUD_NAT"),
        ({"GCP_BATCH_NO_EXTERNAL_IP": "maybe"}, "GCP_BATCH_NO_EXTERNAL_IP"),
        ({"GCP_BATCH_CANDIDATE_REGIONS": "us-central1,us-east4"}, "GCP_BATCH_CANDIDATE_REGIONS"),
    ],
)
def test_invalid_configuration_fails_before_any_cloud_call(
    sandbox: dict[str, Path], env: dict[str, str], message: str
) -> None:
    (sandbox["root"] / ".env").unlink()
    env = {"GOOGLE_CLOUD_PROJECT": PROJECT, **env}
    proc = _run(sandbox, **env)
    assert proc.returncode != 0
    assert message in proc.stderr
    assert not sandbox["log"].exists() or "terraform" not in sandbox["log"].read_text()


def test_release_rejects_unknown_service(sandbox: dict[str, Path]) -> None:
    proc = _run(sandbox, "release", "gx-runner")
    assert proc.returncode != 0
    assert "release supports aurora-steer and dashboard" in proc.stderr


def test_deploy_uses_cloud_build_when_docker_absent(sandbox: dict[str, Path]) -> None:
    (sandbox["bin"] / "docker").unlink()
    proc = _run(sandbox, USE_CLOUD_BUILD="")
    assert proc.returncode == 0, proc.stderr
    calls = sandbox["log"].read_text()
    assert "gcloud builds submit ." in calls
    assert (
        "--service-account=projects/demo-project-123/serviceAccounts/"
        "aurora-cloudbuild-sa@demo-project-123.iam.gserviceaccount.com"
    ) in calls
    assert f"GX_COMMIT={GX_COMMIT}" in calls
    registry = (sandbox["state"] / "registry").read_text().split()
    repo = f"us-central1-docker.pkg.dev/{PROJECT}/aurora-platform"
    assert sorted(registry) == sorted(
        [
            f"{repo}/aurora-steer:abc123",
            f"{repo}/dashboard:abc123",
            f"{repo}/gx-runner:abc123-gx{GX_COMMIT[:12]}",
        ]
    )
