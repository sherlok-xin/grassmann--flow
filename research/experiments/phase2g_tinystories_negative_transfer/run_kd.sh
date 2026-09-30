#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <seed123-s0-run-dir> <seed456-s0-run-dir>" >&2
  exit 2
fi

S0_123="$1"
S0_456="$2"
ROOT="research/experiments/phase2g_tinystories_negative_transfer"
TEACHER="outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10"
mkdir -p "${ROOT}/logs"

run_endpoint () {
  local gpu="$1"
  local seed="$2"
  local lambda="$3"
  local label="$4"
  local init_dir="$5"
  local init_hash
  init_hash="$(sha256sum "${init_dir}/checkpoints/hybrid_best.pt" | awk '{print $1}')"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --teacher-run-dir "${TEACHER}" --student-init-run-dir "${init_dir}" \
    --expected-student-init-sha256 "${init_hash}" --student-type hybrid_lite \
    --tokenizer-dir ./gpt2_local --gpu-id "${gpu}" \
    --output-dir outputs/distill_experiments \
    --experiment-name "phase2g_ts_${label}_seed${seed}" \
    --notes "Phase 2G TinyStories negative-transfer confirmation; frozen protocol" \
    --tags phase2g,confirmatory,tinystories,matched \
    --seed "${seed}" --batch-size 32 --epochs 10 --lr 1e-4 \
    --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --amp \
    --log-interval 50 --model-dim 224 --num-layers 6 --num-heads 8 \
    --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 \
    --student-late-k 1 --distill-alpha 0.0 --temperature 2.0 \
    --kd-loss-mode token_mean --kd-lambda "${lambda}" \
    --dataset-name tinystories \
    --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
    --text-field text --max-seq-len 256 --max-lines 300000 \
    --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
    --split-seed 42 --offline 2>&1 | tee "${ROOT}/logs/${label}_seed${seed}.log"
}

run_endpoint 0 123 0 ce "${S0_123}" &
PID_CE_123=$!
run_endpoint 1 123 5 kd "${S0_123}" &
PID_KD_123=$!
run_endpoint 2 456 0 ce "${S0_456}" &
PID_CE_456=$!
run_endpoint 3 456 5 kd "${S0_456}" &
PID_KD_456=$!

wait "${PID_CE_123}"
wait "${PID_KD_123}"
wait "${PID_CE_456}"
wait "${PID_KD_456}"
