#!/usr/bin/env bash
# Crontab keepalive wrapper for ONE pinned TACC login node.
#
# TACC login hostnames (vista.tacc.utexas.edu, ls6.tacc.utexas.edu) are round-robin DNS over
# several login nodes, and every login node keeps its own crontab spool (Slurm scrontab is not
# offered to users). Install and inspect the crontab on one specific node only:
#
#   ssh login1.vista.tacc.utexas.edu
#   crontab -e
#   */5 * * * * TACC_EXPECTED_LOGIN_HOST=login1 /abs/path/to/workstreams/kr2-gk-database/hpc/scripts/ensure_running_cron.sh
#
# Safety layers:
#   - TACC_EXPECTED_LOGIN_HOST (optional): exit quietly on any other node, so a crontab copied
#     to a second login node cannot start a second driver.
#   - flock -n on state/cron.lock: an overlapping tick on this node exits immediately (code 0).
#   - tacc-driver's own lock (flock + shared-filesystem lease) also refuses a driver started on
#     another login node, whichever crontab started it.
# Paths in config.json that are relative resolve against the config file's directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${HPC_DIR}/../../.." && pwd)"
VENV_BIN="${REPO_ROOT}/.venv-tacc/bin"
CONFIG_PATH="${TACC_EXEC_CONFIG:-${HPC_DIR}/config.json}"
STATE_DIR="${HPC_DIR}/state"
LOG_FILE="${STATE_DIR}/cron.log"
MAX_LOG_BYTES=$((50 * 1024 * 1024))

# cron starts with a minimal PATH; Slurm commands live in the system bin directories.
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

mkdir -p "${STATE_DIR}"

log() {
  printf '%s ensure_running_cron: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${LOG_FILE}"
}

if [[ -n "${TACC_EXPECTED_LOGIN_HOST:-}" ]]; then
  this_host="$(hostname -s)"
  expected_host="${TACC_EXPECTED_LOGIN_HOST%%.*}"
  if [[ "${this_host}" != "${expected_host}" ]]; then
    log "host ${this_host} is not the pinned login node ${expected_host}; not starting the driver"
    exit 0
  fi
fi

# Keep the cron log bounded (one rotated generation).
if [[ -f "${LOG_FILE}" ]] && (( $(stat -c %s "${LOG_FILE}") > MAX_LOG_BYTES )); then
  mv -f "${LOG_FILE}" "${LOG_FILE}.1"
fi

cd "${HPC_DIR}"
exec flock -n -E 0 "${STATE_DIR}/cron.lock" \
  "${VENV_BIN}/tacc-driver" --config "${CONFIG_PATH}" --once >> "${LOG_FILE}" 2>&1
