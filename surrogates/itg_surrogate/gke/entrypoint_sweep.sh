#!/usr/bin/env bash
# Entrypoint for `itg_surrogate` GKE Cloud TPU sweep & experiment jobs.
set -euo pipefail

echo "=== ITG Surrogate GKE Cloud TPU Runner ==="
echo "Host             : $(hostname)"
echo "Timestamp (UTC)  : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python - <<'PY'
import jax
print(f"JAX version      : {jax.__version__}")
print(f"Backend          : {jax.default_backend()}")
print(f"Device count     : {jax.device_count()}")
print(f"Devices          : {jax.devices()}")
PY

# Optional on-TPU smoke verification of symmetry, split, and sharding tests before sweep.
if [[ "${RUN_UNIT_TESTS:-1}" == "1" ]]; then
  echo "=== Running pre-flight unit tests on TPU node ==="
  pytest -o cache_dir=/tmp/pytest_cache /workspace/surrogates/itg_surrogate/tests -q
fi

# Resolve target sweep or experiment.
# When launched as a Kubernetes Indexed Job (`SWEEP_INDEXED=1`), `JOB_COMPLETION_INDEX`
# maps 0..3 onto the four registered ablation sweeps in `itg_surrogate.registry.sweeps`.
ALL_SWEEPS=(
  "architecture_ladder"
  "ood_regularization"
  "weight_decay_scan"
  "normalization_diagnostic"
)

TARGET_FLAG=""
TARGET_NAME=""
if [[ "${SWEEP_INDEXED:-0}" == "1" ]]; then
  IDX="${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX must be set when SWEEP_INDEXED=1}"
  if (( IDX < 0 || IDX >= ${#ALL_SWEEPS[@]} )); then
    echo "ERROR: JOB_COMPLETION_INDEX=${IDX} out of range [0, $(( ${#ALL_SWEEPS[@]} - 1 ))]" >&2
    exit 1
  fi
  TARGET_NAME="${ALL_SWEEPS[$IDX]}"
  TARGET_FLAG="--sweep"
elif [[ -n "${SWEEP_NAME:-}" ]]; then
  TARGET_NAME="${SWEEP_NAME}"
  TARGET_FLAG="--sweep"
else
  TARGET_NAME="${EXPERIMENT_NAME:-leading}"
  TARGET_FLAG="--experiment"
fi

GCS_MOUNT="${GCS_MOUNT_PATH:-/gcs}"
DATASET_SRC="${H5_PATH:-${GCS_MOUNT}/${GCS_DATASET_OBJECT:-datasets/20250102-01_GX_stellarator_dataset.h5}}"
if [[ ! -f "${DATASET_SRC}" ]]; then
  echo "ERROR: HDF5 dataset not found at ${DATASET_SRC}." >&2
  echo "Upload it first via: make -C surrogates/itg_surrogate/gke upload-dataset H5_LOCAL=/path/to/20250102-01_GX_stellarator_dataset.h5" >&2
  exit 1
fi

# Stage HDF5 from GCS FUSE into RAM-backed /dev/shm (fallback /tmp) for fast random/sequential reads.
STAGE_DIR="/dev/shm"
if [[ ! -d "${STAGE_DIR}" || ! -w "${STAGE_DIR}" ]]; then
  STAGE_DIR="/tmp"
fi
LOCAL_H5="${STAGE_DIR}/gx_stellarator_dataset.h5"
if [[ "${DATASET_SRC}" != "${LOCAL_H5}" ]]; then
  echo "Staging ${DATASET_SRC} -> ${LOCAL_H5} ..."
  cp -f "${DATASET_SRC}" "${LOCAL_H5}"
fi

RUN_ID="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}"
RESULTS_DIR="${RESULTS_DIR:-${GCS_MOUNT}/${GCS_RESULTS_PREFIX:-itg_surrogate/sweeps}/${RUN_ID}}"
mkdir -p "${RESULTS_DIR}"
OUT_JSON="${RESULTS_DIR}/${TARGET_NAME}.json"

CMD=(
  itg-surrogate train
  --h5-path "${LOCAL_H5}"
  "${TARGET_FLAG}" "${TARGET_NAME}"
  --output-json "${OUT_JSON}"
)
if [[ -n "${EPOCHS:-}" ]]; then
  CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${NUM_SEEDS:-}" ]]; then
  CMD+=(--num-seeds "${NUM_SEEDS}")
fi
if [[ "${WITH_XGBOOST:-0}" == "1" ]]; then
  CMD+=(--with-xgboost)
fi

cp -f /workspace/pip-freeze.txt "${RESULTS_DIR}/pip-freeze.txt" 2>/dev/null || true

echo "=== Launching: ${CMD[*]} ==="
"${CMD[@]}"
echo "=== Completed ${TARGET_NAME}; JSON summary written to ${OUT_JSON} ==="
