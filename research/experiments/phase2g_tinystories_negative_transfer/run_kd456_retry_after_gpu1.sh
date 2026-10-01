#!/usr/bin/env bash
set -euo pipefail

ROOT="research/experiments/phase2g_tinystories_negative_transfer"
WAIT_PID="469047"
S0_123="outputs/hybrid_experiments/20260930_085658_phase2g_ts_s0_seed123_e20"
S0_456="outputs/hybrid_experiments/20260930_085713_phase2g_ts_s0_seed456_e20"
TEACHER="outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10"
EXPECTED_S0_HASH="6eaa7b735ee425e98711103b594e68709cc94359cbfbddbe0eb9a87795c6fbc7"

# Wait for the healthy GPU-1 KD seed-123 arm to finish successfully.
while ps -p "${WAIT_PID}" -o args= | grep -q 'phase2g_ts_kd_seed123'; do
  sleep 60
done

KD_123="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_kd_seed123' | sort | tail -n 1)"
test -f "${KD_123}/summary.json"

# Do not start until GPU 1 has actually released its model allocation.
while true; do
  USED="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${USED}" -lt 100 ]]; then
    break
  fi
  sleep 30
done

ACTUAL_S0_HASH="$(sha256sum "${S0_456}/checkpoints/hybrid_best.pt" | awk '{print $1}')"
test "${ACTUAL_S0_HASH}" = "${EXPECTED_S0_HASH}"

python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --teacher-run-dir "${TEACHER}" --student-init-run-dir "${S0_456}" \
  --expected-student-init-sha256 "${EXPECTED_S0_HASH}" --student-type hybrid_lite \
  --tokenizer-dir ./gpt2_local --gpu-id 1 \
  --output-dir outputs/distill_experiments \
  --experiment-name phase2g_ts_kd_seed456 \
  --notes "Phase 2G TinyStories negative-transfer confirmation; frozen protocol; infrastructure retry from scratch" \
  --tags phase2g,confirmatory,tinystories,matched,infrastructure-retry \
  --seed 456 --batch-size 32 --epochs 10 --lr 1e-4 \
  --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --amp \
  --log-interval 50 --model-dim 224 --num-layers 6 --num-heads 8 \
  --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 \
  --student-late-k 1 --distill-alpha 0.0 --temperature 2.0 \
  --kd-loss-mode token_mean --kd-lambda 5 \
  --dataset-name tinystories \
  --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
  --text-field text --max-seq-len 256 --max-lines 300000 \
  --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
  --split-seed 42 --offline 2>&1 | tee "${ROOT}/logs/kd_seed456_retry_gpu1.log"

# Wait for both CE arms as well, then run the strict collector with the retry.
while true; do
  CE_123="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_ce_seed123' | sort | tail -n 1)"
  CE_456="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_ce_seed456' | sort | tail -n 1)"
  if [[ -f "${CE_123}/summary.json" && -f "${CE_456}/summary.json" ]]; then
    break
  fi
  sleep 60
done

KD_456="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2g_ts_kd_seed456' | sort | tail -n 1)"
test -f "${KD_456}/summary.json"

python "${ROOT}/collect_results.py" \
  --s0-123 "${S0_123}" --ce-123 "${CE_123}" --kd-123 "${KD_123}" \
  --s0-456 "${S0_456}" --ce-456 "${CE_456}" --kd-456 "${KD_456}" \
  2>&1 | tee "${ROOT}/logs/collector.log"

date --iso-8601=seconds > "${ROOT}/logs/PIPELINE_COMPLETE"
