#!/usr/bin/env bash
set -euo pipefail

ROOT="research/experiments/phase2g_tinystories_negative_transfer"
mkdir -p "${ROOT}/logs"

run_s0 () {
  local gpu="$1"
  local seed="$2"
  python train_hybrid_lite_latefusion_baseline_v2.py \
    --tokenizer-dir ./gpt2_local --gpu-id "${gpu}" \
    --output-dir outputs/hybrid_experiments \
    --experiment-name "phase2g_ts_s0_seed${seed}_e20" \
    --notes "Phase 2G independent TinyStories S0; frozen protocol" \
    --tags phase2g,confirmatory,tinystories,s0 \
    --seed "${seed}" --batch-size 32 --epochs 20 --lr 2e-4 \
    --weight-decay 0.01 --num-workers 4 --amp --log-interval 50 \
    --init-alpha 0.5 --late-k 1 --model-dim 224 --num-layers 6 \
    --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 \
    --dataset-name tinystories \
    --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
    --text-field text --max-seq-len 256 --max-lines 300000 \
    --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
    --split-seed 42 --offline 2>&1 | tee "${ROOT}/logs/s0_seed${seed}.log"
}

run_s0 0 123 &
PID_123=$!
run_s0 1 456 &
PID_456=$!
wait "${PID_123}"
wait "${PID_456}"
