# Orchestrates GKE Cloud TPU setup, Cloud Build container builds, and sweep jobs.
# Usage from repository root:
#   cp surrogates/itg_surrogate/gke/gcp.env.example surrogates/itg_surrogate/gke/gcp.env
#   make -C surrogates/itg_surrogate/gke setup
#   make -C surrogates/itg_surrogate/gke upload-dataset H5_LOCAL=/path/to/20250102-01_GX_stellarator_dataset.h5
#   make -C surrogates/itg_surrogate/gke build
#   make -C surrogates/itg_surrogate/gke deploy SWEEP=ood_regularization
#   make -C surrogates/itg_surrogate/gke deploy-all-sweeps

SHELL := /bin/bash
GKE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(GKE_DIR)/../../..)

ENV_FILE ?= $(GKE_DIR)/gcp.env
ifneq ("$(wildcard $(ENV_FILE))","")
  include $(ENV_FILE)
  export
endif

GCLOUD ?= gcloud
KUBECTL ?= kubectl

# Defaults (overridden by surrogates/itg_surrogate/gke/gcp.env or CLI arguments)
REGION ?= europe-west4
ZONE ?= europe-west4-a
CLUSTER_NAME ?= gk-tpu-cluster
TPU_POOL_NAME ?= tpu-v6e-8-pool
K8S_NAMESPACE ?= default
KSA_NAME ?= itg-surrogate-ksa
AR_REPO ?= gk-simulations
IMAGE_NAME ?= itg-surrogate-tpu
IMAGE_TAG ?= latest
JAX_VERSION ?= 0.11.2
GCS_DATASET_OBJECT ?= datasets/20250102-01_GX_stellarator_dataset.h5
GCS_RESULTS_PREFIX ?= itg_surrogate/sweeps

TPU_MACHINE_TYPE ?= ct6e-standard-8t
TPU_ACCELERATOR ?= tpu-v6e-slice
TPU_TOPOLOGY ?= 2x4
TPU_CHIPS ?= 8
TPU_NODE_LOCATIONS ?= $(ZONE)
MAX_TPU_NODES ?= 4
USE_SPOT ?= true
PARALLEL_SWEEPS ?= 4

# Sweep / experiment runtime controls
SWEEP ?= ood_regularization
EXPERIMENT ?=
EPOCHS ?=
NUM_SEEDS ?=
WITH_XGBOOST ?= 0
RUN_UNIT_TESTS ?= 1
RUN_ID ?= $(shell date -u +"%Y%m%d-%H%M%S")

IMAGE_URI = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(AR_REPO)/$(IMAGE_NAME)

ifneq ($(strip $(EXPERIMENT)),)
  SWEEP_NAME :=
  EXPERIMENT_NAME := $(EXPERIMENT)
  SLUG := $(subst _,-,$(EXPERIMENT))
else
  SWEEP_NAME := $(SWEEP)
  EXPERIMENT_NAME :=
  SLUG := $(subst _,-,$(SWEEP))
endif

JOB_NAME ?= itg-$(SLUG)-$(RUN_ID)
MATRIX_JOB_NAME ?= itg-all-sweeps-$(RUN_ID)

ENVSUBST_VARS := '$$JOB_NAME $$K8S_NAMESPACE $$RUN_ID $$KSA_NAME $$TPU_ACCELERATOR $$TPU_TOPOLOGY $$TPU_CHIPS $$IMAGE_URI $$IMAGE_TAG $$SWEEP_NAME $$EXPERIMENT_NAME $$EPOCHS $$NUM_SEEDS $$WITH_XGBOOST $$RUN_UNIT_TESTS $$GCS_DATASET_OBJECT $$GCS_RESULTS_PREFIX $$GCS_BUCKET $$PARALLEL_SWEEPS'

.PHONY: help check-env setup setup-tpu-pool credentials upload-dataset build render render-all-sweeps deploy deploy-all-sweeps status logs clean-jobs teardown

help:
	@echo "GKE Cloud TPU Automation for itg_surrogate Sweeps"
	@echo ""
	@echo "Targets:"
	@echo "  make -C surrogates/itg_surrogate/gke setup                         Provision GKE cluster, GCS Fuse CSI, Workload Identity, and TPU pool"
	@echo "  make -C surrogates/itg_surrogate/gke setup-tpu-pool                Create only the autoscaling Cloud TPU node pool"
	@echo "  make -C surrogates/itg_surrogate/gke upload-dataset H5_LOCAL=...   Upload local GX HDF5 dataset to gs://\$$(GCS_BUCKET)/\$$(GCS_DATASET_OBJECT)"
	@echo "  make -C surrogates/itg_surrogate/gke build                         Build and push Dockerfile.tpu via Google Cloud Build"
	@echo "  make -C surrogates/itg_surrogate/gke deploy [SWEEP=...]            Submit a single sweep (or EXPERIMENT=leading) Job on TPU"
	@echo "  make -C surrogates/itg_surrogate/gke deploy-all-sweeps             Run all 4 registered sweeps in parallel via an Indexed Job"
	@echo "  make -C surrogates/itg_surrogate/gke render [SWEEP=...]            Print rendered Kubernetes YAML without submitting"
	@echo "  make -C surrogates/itg_surrogate/gke status                        Show active GKE TPU jobs and pods"
	@echo "  make -C surrogates/itg_surrogate/gke logs [JOB=...]                Stream logs from the latest (or named) itg-surrogate job"
	@echo "  make -C surrogates/itg_surrogate/gke clean-jobs                    Delete completed/active itg-surrogate batch jobs"
	@echo "  make -C surrogates/itg_surrogate/gke teardown                      Delete GKE cluster (including TPU pool) when finished"

check-env:
	@if [[ -z "$(PROJECT_ID)" || "$(PROJECT_ID)" == "your-gcp-project-id" ]]; then \
		echo "ERROR: Copy surrogates/itg_surrogate/gke/gcp.env.example to surrogates/itg_surrogate/gke/gcp.env and set PROJECT_ID and GCS_BUCKET." >&2; \
		exit 1; \
	fi
	@if [[ -z "$(GCS_BUCKET)" || "$(GCS_BUCKET)" == "your-gk-simulations-bucket" ]]; then \
		echo "ERROR: Set GCS_BUCKET in surrogates/itg_surrogate/gke/gcp.env." >&2; \
		exit 1; \
	fi

setup: check-env
	$(GCLOUD) services enable \
		compute.googleapis.com \
		container.googleapis.com \
		artifactregistry.googleapis.com \
		cloudbuild.googleapis.com \
		tpu.googleapis.com \
		storage.googleapis.com \
		--project="$(PROJECT_ID)"
	@if ! $(GCLOUD) artifacts repositories describe "$(AR_REPO)" --location="$(REGION)" --project="$(PROJECT_ID)" >/dev/null 2>&1; then \
		$(GCLOUD) artifacts repositories create "$(AR_REPO)" \
			--repository-format=docker \
			--location="$(REGION)" \
			--project="$(PROJECT_ID)"; \
	fi
	@if ! $(GCLOUD) storage buckets describe "gs://$(GCS_BUCKET)" --project="$(PROJECT_ID)" >/dev/null 2>&1; then \
		$(GCLOUD) storage buckets create "gs://$(GCS_BUCKET)" \
			--location="$(REGION)" \
			--uniform-bucket-level-access \
			--project="$(PROJECT_ID)"; \
	fi
	@if ! $(GCLOUD) container clusters describe "$(CLUSTER_NAME)" --zone="$(ZONE)" --project="$(PROJECT_ID)" >/dev/null 2>&1; then \
		$(GCLOUD) container clusters create "$(CLUSTER_NAME)" \
			--zone="$(ZONE)" \
			--project="$(PROJECT_ID)" \
			--release-channel=regular \
			--machine-type=e2-standard-4 \
			--num-nodes=1 \
			--workload-pool="$(PROJECT_ID).svc.id.goog" \
			--addons=GcsFuseCsiDriver; \
	fi
	$(MAKE) credentials
	@if ! $(KUBECTL) get namespace "$(K8S_NAMESPACE)" >/dev/null 2>&1; then \
		$(KUBECTL) create namespace "$(K8S_NAMESPACE)"; \
	fi
	@if ! $(KUBECTL) get serviceaccount "$(KSA_NAME)" -n "$(K8S_NAMESPACE)" >/dev/null 2>&1; then \
		$(KUBECTL) create serviceaccount "$(KSA_NAME)" -n "$(K8S_NAMESPACE)"; \
	fi
	@PROJ_NUM="$(PROJECT_NUMBER)"; \
	if [[ -z "$$PROJ_NUM" ]]; then \
		PROJ_NUM=$$($(GCLOUD) projects describe "$(PROJECT_ID)" --format="value(projectNumber)"); \
	fi; \
	WIF_PRINCIPAL="principal://iam.googleapis.com/projects/$${PROJ_NUM}/locations/global/workloadIdentityPools/$(PROJECT_ID).svc.id.goog/subject/ns/$(K8S_NAMESPACE)/sa/$(KSA_NAME)"; \
	$(GCLOUD) storage buckets add-iam-policy-binding "gs://$(GCS_BUCKET)" \
		--member="$$WIF_PRINCIPAL" \
		--role="roles/storage.objectUser" \
		--project="$(PROJECT_ID)" >/dev/null
	$(MAKE) setup-tpu-pool

setup-tpu-pool: check-env
	@SPOT_FLAG=""; \
	if [[ "$(USE_SPOT)" == "true" ]]; then SPOT_FLAG="--spot"; fi; \
	if ! $(GCLOUD) container node-pools describe "$(TPU_POOL_NAME)" --cluster="$(CLUSTER_NAME)" --zone="$(ZONE)" --project="$(PROJECT_ID)" >/dev/null 2>&1; then \
		$(GCLOUD) container node-pools create "$(TPU_POOL_NAME)" \
			--cluster="$(CLUSTER_NAME)" \
			--zone="$(ZONE)" \
			--node-locations="$(TPU_NODE_LOCATIONS)" \
			--project="$(PROJECT_ID)" \
			--machine-type="$(TPU_MACHINE_TYPE)" \
			--num-nodes=0 \
			--enable-autoscaling \
			--location-policy=ANY \
			--total-min-nodes=0 \
			--total-max-nodes="$(MAX_TPU_NODES)" \
			$$SPOT_FLAG; \
	else \
		echo "TPU node pool $(TPU_POOL_NAME) already exists."; \
	fi

credentials: check-env
	$(GCLOUD) container clusters get-credentials "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" \
		--project="$(PROJECT_ID)"

upload-dataset: check-env
	@if [[ -z "$(H5_LOCAL)" || ! -f "$(H5_LOCAL)" ]]; then \
		echo "ERROR: Pass H5_LOCAL=/path/to/20250102-01_GX_stellarator_dataset.h5" >&2; \
		exit 1; \
	fi
	$(GCLOUD) storage cp "$(H5_LOCAL)" "gs://$(GCS_BUCKET)/$(GCS_DATASET_OBJECT)" --project="$(PROJECT_ID)"

build: check-env
	$(GCLOUD) builds submit "$(REPO_ROOT)" \
		--project="$(PROJECT_ID)" \
		--config="$(GKE_DIR)/cloudbuild.yaml" \
		--substitutions="_IMAGE_URI=$(IMAGE_URI),_IMAGE_TAG=$(IMAGE_TAG),_JAX_VERSION=$(JAX_VERSION)"

render:
	@JOB_NAME="$(JOB_NAME)" \
	K8S_NAMESPACE="$(K8S_NAMESPACE)" \
	RUN_ID="$(RUN_ID)" \
	KSA_NAME="$(KSA_NAME)" \
	TPU_ACCELERATOR="$(TPU_ACCELERATOR)" \
	TPU_TOPOLOGY="$(TPU_TOPOLOGY)" \
	TPU_CHIPS="$(TPU_CHIPS)" \
	IMAGE_URI="$(IMAGE_URI)" \
	IMAGE_TAG="$(IMAGE_TAG)" \
	SWEEP_NAME="$(SWEEP_NAME)" \
	EXPERIMENT_NAME="$(EXPERIMENT_NAME)" \
	EPOCHS="$(EPOCHS)" \
	NUM_SEEDS="$(NUM_SEEDS)" \
	WITH_XGBOOST="$(WITH_XGBOOST)" \
	RUN_UNIT_TESTS="$(RUN_UNIT_TESTS)" \
	GCS_DATASET_OBJECT="$(GCS_DATASET_OBJECT)" \
	GCS_RESULTS_PREFIX="$(GCS_RESULTS_PREFIX)" \
	GCS_BUCKET="$(GCS_BUCKET)" \
	envsubst $(ENVSUBST_VARS) < "$(GKE_DIR)/job-tpu-sweep.yaml"

render-all-sweeps:
	@JOB_NAME="$(MATRIX_JOB_NAME)" \
	K8S_NAMESPACE="$(K8S_NAMESPACE)" \
	RUN_ID="$(RUN_ID)" \
	KSA_NAME="$(KSA_NAME)" \
	TPU_ACCELERATOR="$(TPU_ACCELERATOR)" \
	TPU_TOPOLOGY="$(TPU_TOPOLOGY)" \
	TPU_CHIPS="$(TPU_CHIPS)" \
	IMAGE_URI="$(IMAGE_URI)" \
	IMAGE_TAG="$(IMAGE_TAG)" \
	EPOCHS="$(EPOCHS)" \
	NUM_SEEDS="$(NUM_SEEDS)" \
	WITH_XGBOOST="$(WITH_XGBOOST)" \
	RUN_UNIT_TESTS="$(RUN_UNIT_TESTS)" \
	GCS_DATASET_OBJECT="$(GCS_DATASET_OBJECT)" \
	GCS_RESULTS_PREFIX="$(GCS_RESULTS_PREFIX)" \
	GCS_BUCKET="$(GCS_BUCKET)" \
	PARALLEL_SWEEPS="$(PARALLEL_SWEEPS)" \
	envsubst $(ENVSUBST_VARS) < "$(GKE_DIR)/job-tpu-all-sweeps.yaml"

deploy: check-env
	@$(MAKE) --no-print-directory render JOB_NAME="$(JOB_NAME)" RUN_ID="$(RUN_ID)" | $(KUBECTL) apply -f -
	@echo "Submitted TPU job $(JOB_NAME) -> results will write to gs://$(GCS_BUCKET)/$(GCS_RESULTS_PREFIX)/$(RUN_ID)/"

deploy-all-sweeps: check-env
	@$(MAKE) --no-print-directory render-all-sweeps MATRIX_JOB_NAME="$(MATRIX_JOB_NAME)" RUN_ID="$(RUN_ID)" | $(KUBECTL) apply -f -
	@echo "Submitted 4-way Indexed TPU job $(MATRIX_JOB_NAME) -> results will write to gs://$(GCS_BUCKET)/$(GCS_RESULTS_PREFIX)/$(RUN_ID)/"

status:
	$(KUBECTL) get jobs,pods -n "$(K8S_NAMESPACE)" -l app=itg-surrogate -o wide

logs:
	@TARGET_JOB="$(JOB)"; \
	if [[ -z "$$TARGET_JOB" ]]; then \
		TARGET_JOB=$$($(KUBECTL) get jobs -n "$(K8S_NAMESPACE)" -l app=itg-surrogate --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}'); \
	fi; \
	echo "Streaming logs for job/$$TARGET_JOB ..."; \
	$(KUBECTL) logs -n "$(K8S_NAMESPACE)" "job/$$TARGET_JOB" --all-containers=true -f

clean-jobs:
	$(KUBECTL) delete jobs -n "$(K8S_NAMESPACE)" -l app=itg-surrogate --ignore-not-found=true

teardown: check-env
	$(GCLOUD) container clusters delete "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" \
		--project="$(PROJECT_ID)" \
		--quiet
