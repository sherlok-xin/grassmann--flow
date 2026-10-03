#!/usr/bin/env bash
set -euo pipefail

PHASE="research/experiments/phase3a_safe_lambda_diagnostic"
RUNNER="${PHASE}/scripts/run_diagnostics.py"
LOG_DIR="${PHASE}/logs"
RAW_DIR="${PHASE}/raw/diagnostics"
mkdir -p "${LOG_DIR}" "${RAW_DIR}"

launch() {
  local gpu="$1"
  local label="$2"
  local ids="$3"
  python "${RUNNER}" \
    --gpu-id "${gpu}" \
    --condition-ids "${ids}" \
    --output-dir "${RAW_DIR}" \
    --skip-existing \
    >"${LOG_DIR}/${label}.log" 2>&1 &
  echo "$! ${label} gpu=${gpu}"
}

launch 0 gpu0_ptb_j "PTB_F_S42,PTB_F_S123,PTB_F_S456,WT2_J_S42,WT2_J_S123,WT2_J_S456"
launch 1 gpu1_a_t "WT2_A_S42,WT2_A_S123,WT2_A_S456,WT2_T_S42,WT2_T_S123,WT2_T_S456"
launch 2 gpu2_g_tt "WT2_G_S42,WT2_G_S123,WT2_G_S456,WT2_TT_S42,WT2_TT_S123,WT2_TT_S456"
launch 3 gpu3_ts "TS_F_S42,TS_F_S123,TS_F_S456"
wait
