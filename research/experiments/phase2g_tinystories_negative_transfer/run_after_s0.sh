#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <s0-controller-pid>" >&2
  exit 2
fi

CONTROLLER_PID="$1"
ROOT="research/experiments/phase2g_tinystories_negative_transfer"

while ps -p "${CONTROLLER_PID}" -o args= | grep -q 'run_s0.sh'; do
  sleep 60
done

S0_123="$(find outputs/hybrid_experiments -maxdepth 1 -type d -name '*_phase2g_ts_s0_seed123_e20' | sort | tail -n 1)"
S0_456="$(find outputs/hybrid_experiments -maxdepth 1 -type d -name '*_phase2g_ts_s0_seed456_e20' | sort | tail -n 1)"
test -f "${S0_123}/summary.json"
test -f "${S0_456}/summary.json"
test -f "${S0_123}/checkpoints/hybrid_best.pt"
test -f "${S0_456}/checkpoints/hybrid_best.pt"

HASH_42="e4d0959603bb39f0ad7b7a98025024975e2b87d117d36037e9f9aa9da966f9c2"
HASH_123="$(sha256sum "${S0_123}/checkpoints/hybrid_best.pt" | awk '{print $1}')"
HASH_456="$(sha256sum "${S0_456}/checkpoints/hybrid_best.pt" | awk '{print $1}')"
test "${HASH_123}" != "${HASH_456}"
test "${HASH_123}" != "${HASH_42}"
test "${HASH_456}" != "${HASH_42}"

bash "${ROOT}/run_kd.sh" "${S0_123}" "${S0_456}"

CE_123="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_ce_seed123' | sort | tail -n 1)"
KD_123="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_kd_seed123' | sort | tail -n 1)"
CE_456="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_ce_seed456' | sort | tail -n 1)"
KD_456="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_kd_seed456' | sort | tail -n 1)"

python "${ROOT}/collect_results.py" \
  --s0-123 "${S0_123}" --ce-123 "${CE_123}" --kd-123 "${KD_123}" \
  --s0-456 "${S0_456}" --ce-456 "${CE_456}" --kd-456 "${KD_456}" \
  2>&1 | tee "${ROOT}/logs/collector.log"

date --iso-8601=seconds > "${ROOT}/logs/PIPELINE_COMPLETE"
