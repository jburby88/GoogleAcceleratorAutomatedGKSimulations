# Running `itg_surrogate` Sweeps on GKE with Cloud TPUs

This directory provides reproducible Google Kubernetes Engine (GKE) + Cloud TPU configurations (modeled after [`ToBlick/mrx/gke`](https://github.com/ToBlick/mrx/tree/main/gke)) to run the unseen-stellarator ablation sweeps in [`itg_surrogate.registry.sweeps`](../src/itg_surrogate/registry/sweeps.py).

---

## New GCP Project Setup Checklist

When onboarding a **brand-new GCP project** to run `itg_surrogate` on GKE Cloud TPUs, complete or verify the following five items once per project:

### 1. Operator Workstation Prerequisites

Ensure your local machine / Cloudtop has:
- **Up-to-date Google Cloud CLI** (`gcloud >= 500.0.0`, authenticated via `gcloud auth login`).
  - *gLinux / Cloudtop note:* Use `/usr/bin/gcloud` (`GCLOUD=/usr/bin/gcloud` in `gcp.env`) and ensure no legacy `~/google-cloud-sdk` installation precedes `/usr/bin` on `$PATH` (older `v434` SDKs fail Enterprise Certificate Proxy checks with `KeyError('tls_offload')`).
- **`kubectl` + GKE Auth Plugin** and **`envsubst` / `make`**:
  ```bash
  # Debian / Ubuntu / gLinux:
  sudo apt-get update && sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin gettext-base make
  ```

### 2. Required GCP APIs

`make -C surrogates/itg_surrogate/gke setup` enables these automatically, or an admin can enable them upfront:

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"
```

### 3. Required IAM Roles (New-Project Default Service Account Policy)

1. **Human operator roles** (to run `make setup`, `make build`, and `make deploy`):
   - `roles/serviceusage.serviceUsageAdmin` (if enabling APIs)
   - `roles/container.admin` and `roles/compute.admin` (cluster + TPU node pool creation)
   - `roles/artifactregistry.admin` and `roles/storage.admin` (creating `${AR_REPO}` and `gs://${GCS_BUCKET}` + binding bucket IAM)
   - `roles/cloudbuild.builds.editor` and `roles/iam.serviceAccountUser`
2. **Default Compute Engine Service Account (`${PROJECT_NUMBER}-compute@developer.gserviceaccount.com`):**
   - Newer GCP organizations enforce `constraints/iam.automaticIamGrantsForDefaultServiceAccounts`, so `${PROJECT_NUMBER}-compute@developer.gserviceaccount.com` (used by default by **Cloud Build** and **GKE nodes**) starts with **no** permissions instead of `roles/editor`. On a brand-new project, grant it least-privilege build/node roles:
     ```bash
     PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
     COMPUTE_SA="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

     # Allow GKE nodes & Cloud Build to write logs/metrics:
     gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
       --member="${COMPUTE_SA}" --role="roles/container.defaultNodeServiceAccount"
     gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
       --member="${COMPUTE_SA}" --role="roles/logging.logWriter"

     # Allow Cloud Build to push images and GKE TPU nodes to pull them (after creating ${AR_REPO}):
     gcloud artifacts repositories add-iam-policy-binding "${AR_REPO}" \
       --location="${REGION}" --project="${PROJECT_ID}" \
       --member="${COMPUTE_SA}" --role="roles/artifactregistry.writer"
     ```
3. **Pod-to-GCS Access via Direct Workload Identity Federation (Automatic in `make setup`):**
   - No custom GCP Service Account (`GSA`) or key file is needed. `make setup` enables `--workload-pool=${PROJECT_ID}.svc.id.goog` and `--addons=GcsFuseCsiDriver`, then grants `roles/storage.objectUser` on `gs://${GCS_BUCKET}` directly to the federated Kubernetes ServiceAccount principal:
     ```text
     principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${K8S_NAMESPACE}/sa/${KSA_NAME}
     ```

### 4. Verify `compute.googleapis.com` TPU Quotas & Physical Zones Before Choosing `REGION`

- **Important:** GKE TPU node pools consume **Compute Engine (`compute.googleapis.com`)** TPU quotas — **not** `tpu.googleapis.com` quotas (which apply only to standalone GCE TPU VMs).
- Check which regions and TPU families have non-zero Spot/Preemptible or on-demand quota in your project before editing `gcp.env`:
  ```bash
  gcloud alpha services quota list \
    --service=compute.googleapis.com \
    --consumer="projects/${PROJECT_ID}" \
    --filter="metric~tpu" \
    --format="json"
  ```
  Look for positive `effectiveLimit` values on:
  - **TPU v6e:** `compute.googleapis.com/preemptible_tpu_v6e` (Spot) or `compute.googleapis.com/tpus_per_tpu_family` (`tpu_family=CT6E`)
  - **TPU v5p:** `compute.googleapis.com/preemptible_tpu_v5p` (Spot) or `compute.googleapis.com/tpu_v5p`
  - **TPU v5e:** `compute.googleapis.com/preemptible_tpu_lite_podslice_v5` (Spot) or `compute.googleapis.com/tpu_lite_podslice_v5`
- **Sub-zone physical TPU cell placement (`TPU_NODE_LOCATIONS`):** Within a region, physical TPU pods only exist in specific zones (for example, in `us-east5`, `v5p` is hosted in `us-east5-b` and `us-east5-c`, whereas requesting `us-east5-a` or `v5e` fails node scale-up with `CONDITION_NOT_MET: Unsupported TPU configuration`). Set `TPU_NODE_LOCATIONS` in `gcp.env` to all candidate TPU zones in that region (e.g. `TPU_NODE_LOCATIONS=us-east5-b,us-east5-c`); `make setup-tpu-pool` passes `--location-policy=ANY` so GKE Cluster Autoscaler automatically provisions Spot TPU nodes in whichever listed zone has physical hardware and Spot capacity.

### 5. Supported Single-Host Cloud TPU Presets

Configure these five variables in `surrogates/itg_surrogate/gke/gcp.env` (see [`gcp.env.example`](gcp.env.example)):

| Hardware Generation | `TPU_MACHINE_TYPE` | `TPU_ACCELERATOR` | `TPU_TOPOLOGY` | `TPU_CHIPS` | Example `TPU_NODE_LOCATIONS` |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **TPU v6e (8 chips, default)** | `ct6e-standard-8t` | `tpu-v6e-slice` | `2x4` | `8` | `europe-west4-a`, `us-south1-a` |
| **TPU v6e (4 chips)** | `ct6e-standard-4t` | `tpu-v6e-slice` | `2x2` | `4` | `europe-west4-a`, `us-south1-a` |
| **TPU v5p (4 chips)** | `ct5p-hightpu-4t` | `tpu-v5p-slice` | `2x2x1` | `4` | `us-east5-b,us-east5-c`, `us-central1-a` |
| **TPU v5e (8 chips)** | `ct5lp-hightpu-8t` | `tpu-v5-lite-podslice` | `2x4` | `8` | `us-west4-a`, `us-central1-a` |
| **TPU v5e (4 chips)** | `ct5lp-hightpu-4t` | `tpu-v5-lite-podslice` | `2x2` | `4` | `us-west4-a`, `us-central1-a` |

- On an **8-chip** single-host node (`ct6e-standard-8t` or `ct5lp-hightpu-8t`), all `num_seeds=8` ensemble members of each sweep run train **simultaneously in 1 GSPMD `Mesh(devices, ('ensemble',))` wave**.
- On a **4-chip** single-host node (`ct5p-hightpu-4t`, `ct6e-standard-4t`, or `ct5lp-hightpu-4t`), `num_seeds=8` executes automatically in two 4-chip waves (or `NUM_SEEDS=4` in a single wave).

---

## End-to-End Quickstart

### 1. Configure Environment

```bash
cp surrogates/itg_surrogate/gke/gcp.env.example surrogates/itg_surrogate/gke/gcp.env
# Edit surrogates/itg_surrogate/gke/gcp.env to set PROJECT_ID, REGION, ZONE, TPU_NODE_LOCATIONS, and GCS_BUCKET
```

### 2. Provision Cluster, GCS Bucket, Workload Identity, & Zero-Scale TPU Pool

```bash
make -C surrogates/itg_surrogate/gke setup
```

When no sweep jobs are queued, `--total-min-nodes=0` keeps the TPU node pool at **0 TPU VMs**.

### 3. Upload GX Stellarator HDF5 Dataset to GCS

```bash
make -C surrogates/itg_surrogate/gke upload-dataset H5_LOCAL=/path/to/20250102-01_GX_stellarator_dataset.h5
```

### 4. Build & Push TPU Container Image via Cloud Build

```bash
make -C surrogates/itg_surrogate/gke build
```

Builds [`Dockerfile.tpu`](Dockerfile.tpu) (`python:3.12-slim-bookworm` + `jax[tpu]==0.11.2` + `itg_surrogate[baselines,dev]`), records `/workspace/pip-freeze.txt`, and runs `JAX_PLATFORMS=cpu` unit tests during the Cloud Build step.

### 5. Launch Validation Run or Full Ablation Sweeps

Conservative 15-epoch validation smoke run on 4 TPU chips (verifies GCS Fuse mount -> `/dev/shm` RAM staging -> on-TPU `pytest` -> XGBoost baselines -> GSPMD `ResFiLM-MultiPool-80` training -> GCS JSON export in ~30s of TPU compute):

```bash
make -C surrogates/itg_surrogate/gke deploy EXPERIMENT=leading EPOCHS=15 NUM_SEEDS=4 WITH_XGBOOST=1
```

Full 110-epoch single ablation sweep (`architecture_ladder`, `ood_regularization`, `weight_decay_scan`, or `normalization_diagnostic`):

```bash
make -C surrogates/itg_surrogate/gke deploy SWEEP=ood_regularization
```

Or fan out **all 4 ablation sweeps in parallel** via the Kubernetes Indexed Job ([`job-tpu-all-sweeps.yaml`](job-tpu-all-sweeps.yaml)):

```bash
make -C surrogates/itg_surrogate/gke deploy-all-sweeps
```

### 6. Monitor Progress & Retrieve Results

```bash
make -C surrogates/itg_surrogate/gke status
make -C surrogates/itg_surrogate/gke logs

source surrogates/itg_surrogate/gke/gcp.env
gcloud storage ls --recursive "gs://${GCS_BUCKET}/${GCS_RESULTS_PREFIX}/"
```

### 7. Cleanup Jobs & Cluster Teardown

Finished jobs automatically expire after 24 hours (`ttlSecondsAfterFinished: 86400`), and GKE Cluster Autoscaler scales the TPU pool back down to `0` nodes when idle. To delete job records or tear down the entire GKE cluster when a campaign is complete:

```bash
make -C surrogates/itg_surrogate/gke clean-jobs
make -C surrogates/itg_surrogate/gke teardown
```
