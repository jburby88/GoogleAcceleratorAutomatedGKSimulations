# Operator settings come from .env (gitignored; see .env.example). They are read as make
# variables only and deliberately NOT exported: production endpoints, tokens and BigQuery
# settings must never leak into lint/test subprocesses. Targets that need a value get it
# explicitly (target-specific exports or the deploy script, which reads .env itself).
-include .env

.PHONY: help env-init install install-tacc check-tacc-deps lint lint-infra lint-terraform \
	lint-shell lint-docker lint-workflows format test coverage dashboard docker-build \
	gcp-init gcp-plan gcp-apply gcp-deploy tacc-smoke-test clean

PYTHON ?= python3
VENV ?= .venv
VENV_TACC ?= .venv-tacc
# Recursive (=) so `make install lint` re-evaluates it after the venv exists.
VENV_PY = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(PYTHON))
PYTEST_ARGS := --import-mode=importlib
LINT_PATHS := platform simulations surrogates scripts tests

TEST_DIRS := \
	platform/hpc-core/tests \
	platform/aurora-steer/tests \
	platform/dashboard/tests \
	platform/telemetry/tests \
	simulations/hpc/tests \
	simulations/gcp/tests \
	surrogates/tests \
	scripts/tests \
	tests/e2e

# Hermetic tests: whatever is in .env or the developer's shell, the test suite never sees the
# deployed aurora-steer URL, bearer tokens, BigQuery streaming, webhooks or runtime overrides.
TEST_UNSET := AURORA_STEER_URL AURORA_STEER_BEARER_TOKEN AURORA_STEER_TOKEN AURORA_API_TOKEN \
	AURORA_STEER_API_TOKEN BQ_ENABLED BQ_DATASET GOOGLE_CLOUD_PROJECT GCP_PROJECT_ID \
	GOOGLE_CLOUD_LOCATION MODEL_ID VERTEX_TIMEOUT_S ENVIRONMENT DATA_MODE \
	LOCAL_EVENTS_DUCKDB_PATH TACC_NOTIFY_WEBHOOK_URL GX_SIMULATE_RUN GCP_BATCH_CPU_ONLY \
	GCP_BATCH_CANDIDATE_REGIONS VPC_NAME_PREFIX VPC_SUBNET_CIDR ENABLE_CLOUD_NAT \
	GCP_BATCH_NO_EXTERNAL_IP USE_CLOUD_BUILD
TEST_ENV := env $(foreach v,$(TEST_UNSET),-u $(v)) NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
COV_PACKAGES := hpc_core tacc_exec gcp_exec app kr1_surrogate kr3_design gk_surrogates db streamlit_app theme

TF_DIR := platform/infra/terraform
SHELL_SCRIPTS := scripts/deploy_gcp_demo.sh simulations/hpc/scripts/ensure_running_cron.sh
DOCKERFILES := platform/aurora-steer/Dockerfile platform/dashboard/Dockerfile \
	simulations/gcp/Dockerfile.gx-runner
TERRAFORM ?= terraform
SHELLCHECK ?= shellcheck
HADOLINT ?= hadolint
ACTIONLINT ?= actionlint
# DL3008/DL3013: apt/pip version pins are not used (base images and pyproject bounds pin instead).
HADOLINT_FLAGS := --failure-threshold warning --ignore DL3008 --ignore DL3013
# STRICT_TOOLS=1 (CI) turns a missing linter into a failure instead of a skip.
STRICT_TOOLS ?= 0

GCP_REGION ?= us-central1
BQ_DATASET ?= sim_babysitter_telemetry
IMAGE_TAG ?= dev
# 40-hex GX git commit compiled into the gx-runner image (required by Dockerfile.gx-runner).
GX_COMMIT ?=

# $(call tool_guard,<command line>): continue if the tool exists, else skip (or fail if strict).
define tool_guard
command -v $(firstword $(1)) >/dev/null 2>&1 || { if [ "$(STRICT_TOOLS)" = "1" ]; then echo "ERROR: $(firstword $(1)) is not installed" >&2; exit 1; fi; echo "SKIP: $(firstword $(1)) is not installed"; exit 0; }
endef

help:
	@echo "UT Austin x Google.org AI for Science — Automated GK Fusion Platform"
	@echo ""
	@echo "Development & verification (CI runs exactly these targets):"
	@echo "  make env-init         Create a local .env from .env.example if missing"
	@echo "  make install          Create .venv with every workspace package and its dev extras"
	@echo "  make install-tacc     Create .venv-tacc (lightweight TACC runtime) and verify dependencies"
	@echo "  make check-tacc-deps  Check dependency boundaries, tracking parity, and repo safety"
	@echo "  make lint             check-tacc-deps + ruff check + ruff format --check"
	@echo "  make lint-infra       terraform fmt/validate, shellcheck, hadolint, actionlint (skips missing tools)"
	@echo "  make test             All unit, contract and E2E tests in a hermetic environment"
	@echo "  make coverage         make test with a pytest-cov report"
	@echo "  make dashboard        Run the Streamlit dashboard locally against BigQuery"
	@echo ""
	@echo "GCP (reads .env; see docs/RUNBOOK_DEMO_AND_TACC_SMOKE_TEST.md):"
	@echo "  make gcp-deploy       Full idempotent deploy: state bucket, Terraform, secret, images by digest, babysitter config"
	@echo "  make gcp-init         Create the Terraform state bucket if needed and run terraform init"
	@echo "  make gcp-plan         terraform plan with the currently deployed images"
	@echo "  make gcp-apply        terraform apply with the currently deployed images"
	@echo "  make docker-build     Build the three images locally (tag IMAGE_TAG=$(IMAGE_TAG), not pushed)"
	@echo "  make tacc-smoke-test  Check HTTPS, /v1/events and /triage against aurora-steer"
	@echo "  make clean            Remove local Python caches (never touches .env or Terraform files)"

env-init:
	@if [ ! -f .env ]; then cp .env.example .env && chmod 600 .env && echo "Created .env from .env.example"; else echo ".env already exists"; fi

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install \
		-e "./platform/hpc-core" \
		-e "./platform/aurora-steer[dev]" \
		-e "./simulations/hpc[dev]" \
		-e "./simulations/gcp[dev]" \
		-e "./platform/dashboard[dev]" \
		-e "./surrogates[dev]" \
		"pytest-cov>=5,<8"

install-tacc:
	$(PYTHON) -m venv $(VENV_TACC)
	$(VENV_TACC)/bin/python -m pip install --upgrade pip
	$(VENV_TACC)/bin/python -m pip install "./platform/hpc-core" "./simulations/hpc"
	$(PYTHON) scripts/check_boundaries_and_deps.py --venv-python $(VENV_TACC)/bin/python

check-tacc-deps:
	@if [ -x "$(VENV_TACC)/bin/python" ]; then \
		$(VENV_PY) scripts/check_boundaries_and_deps.py --venv-python "$(VENV_TACC)/bin/python"; \
	else \
		$(VENV_PY) scripts/check_boundaries_and_deps.py; \
	fi

lint: check-tacc-deps
	$(VENV_PY) -m ruff check $(LINT_PATHS)
	$(VENV_PY) -m ruff format --check $(LINT_PATHS)

lint-infra: lint-terraform lint-shell lint-docker lint-workflows

lint-terraform:
	@$(call tool_guard,$(TERRAFORM)); set -e; \
	$(TERRAFORM) -chdir=$(TF_DIR) fmt -check -recursive -diff; \
	tmp_tf="$$(mktemp -d)"; trap 'rm -rf "$$tmp_tf"' EXIT; \
	if [ -d "$(TF_DIR)/.terraform/providers" ]; then ln -s "$$(cd "$(TF_DIR)/.terraform" && pwd)/providers" "$$tmp_tf/providers"; fi; \
	TF_DATA_DIR="$$tmp_tf" $(TERRAFORM) -chdir=$(TF_DIR) init -backend=false -input=false >/dev/null; \
	TF_DATA_DIR="$$tmp_tf" $(TERRAFORM) -chdir=$(TF_DIR) validate

lint-shell:
	@$(call tool_guard,$(SHELLCHECK)); $(SHELLCHECK) $(SHELL_SCRIPTS)

lint-docker:
	@$(call tool_guard,$(HADOLINT)); set -e; \
	for f in $(DOCKERFILES); do echo "hadolint $$f"; $(HADOLINT) $(HADOLINT_FLAGS) - < "$$f"; done

lint-workflows:
	@$(call tool_guard,$(ACTIONLINT)); $(ACTIONLINT)

format:
	$(VENV_PY) -m ruff format $(LINT_PATHS)
	$(VENV_PY) -m ruff check --fix $(LINT_PATHS)

test: check-tacc-deps
	$(TEST_ENV) $(VENV_PY) -m pytest $(PYTEST_ARGS) $(TEST_DIRS)

coverage:
	$(TEST_ENV) $(VENV_PY) -m pytest $(PYTEST_ARGS) $(foreach p,$(COV_PACKAGES),--cov=$(p)) \
		--cov-report=term-missing $(TEST_DIRS)

dashboard: export DATA_MODE := bigquery
dashboard: export GOOGLE_CLOUD_PROJECT := $(GOOGLE_CLOUD_PROJECT)
dashboard: export BQ_DATASET := $(BQ_DATASET)
dashboard:
	$(VENV_PY) -m streamlit run platform/dashboard/streamlit_app.py

docker-build:
	docker build -f platform/aurora-steer/Dockerfile -t "aurora-steer:$(IMAGE_TAG)" .
	docker build -f platform/dashboard/Dockerfile -t "aurora-dashboard:$(IMAGE_TAG)" .
	docker build --build-arg GX_COMMIT="$(GX_COMMIT)" -f simulations/gcp/Dockerfile.gx-runner -t "gx-runner:$(IMAGE_TAG)" .

gcp-init:
	./scripts/deploy_gcp_demo.sh init

gcp-plan:
	./scripts/deploy_gcp_demo.sh plan

gcp-apply:
	./scripts/deploy_gcp_demo.sh apply

gcp-deploy:
	./scripts/deploy_gcp_demo.sh deploy

# The token is exported (never placed on a command line): from .env/shell when set, otherwise
# read from Secret Manager at run time.
tacc-smoke-test: export AURORA_STEER_URL := $(AURORA_STEER_URL)
tacc-smoke-test: export AURORA_STEER_BEARER_TOKEN := $(AURORA_STEER_BEARER_TOKEN)
tacc-smoke-test:
	@if [ -z "$${AURORA_STEER_BEARER_TOKEN:-}" ]; then \
		AURORA_STEER_BEARER_TOKEN="$$(gcloud secrets versions access latest --secret=aurora-steer-bearer-token --project='$(GOOGLE_CLOUD_PROJECT)')" || exit 1; \
		export AURORA_STEER_BEARER_TOKEN; \
	fi; \
	$(PYTHON) scripts/tacc_smoke_test.py

clean:
	find platform simulations surrogates scripts tests -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name "build" -o -name "*.egg-info" \) -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist
