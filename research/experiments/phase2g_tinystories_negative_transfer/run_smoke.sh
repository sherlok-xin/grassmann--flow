#!/usr/bin/env bash
set -euo pipefail

ROOT="research/experiments/phase2g_tinystories_negative_transfer"
mkdir -p "${ROOT}/logs"

python train_hybrid_lite_latefusion_baseline_v2.py \
  --tokenizer-dir ./gpt2_local --gpu-id 0 \
  --output-dir outputs/hybrid_experiments \
  --experiment-name phase2g_smoke_ts_s0_seed123_e1 \
  --notes "Phase 2G reduced-data pipeline smoke" \
  --tags phase2g,smoke,tinystories,s0 \
  --seed 123 --batch-size 32 --epochs 1 --lr 2e-4 --weight-decay 0.01 \
  --num-workers 4 --amp --log-interval 50 --init-alpha 0.5 --late-k 1 \
  --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
  --dropout 0.1 --dataset-name tinystories \
  --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
  --text-field text --max-seq-len 256 --max-lines 2000 \
  --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
  --split-seed 42 --offline 2>&1 | tee "${ROOT}/logs/s0_smoke.log"

S0_DIR="$(find outputs/hybrid_experiments -maxdepth 1 -type d -name '*_phase2g_smoke_ts_s0_seed123_e1' | sort | tail -n 1)"
test -f "${S0_DIR}/checkpoints/hybrid_best.pt"

run_endpoint () {
  local gpu="$1"
  local lambda="$2"
  local label="$3"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --teacher-run-dir outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10 \
    --student-init-run-dir "${S0_DIR}" --student-type hybrid_lite \
    --tokenizer-dir ./gpt2_local --gpu-id "${gpu}" \
    --output-dir outputs/distill_experiments \
    --experiment-name "phase2g_smoke_ts_${label}_seed123_e1" \
    --notes "Phase 2G reduced-data pipeline smoke" \
    --tags phase2g,smoke,tinystories,matched \
    --seed 123 --batch-size 32 --epochs 1 --lr 1e-4 --weight-decay 0.01 \
    --warmup-ratio 0.05 --num-workers 4 --amp --log-interval 50 \
    --model-dim 224 --num-layers 6 --num-heads 8 --reduced-dim 56 \
    --window-sizes 1,2,4 --dropout 0.1 --student-late-k 1 \
    --distill-alpha 0.0 --temperature 2.0 --kd-loss-mode token_mean \
    --kd-lambda "${lambda}" --dataset-name tinystories \
    --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
    --text-field text --max-seq-len 256 --max-lines 2000 \
    --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
    --split-seed 42 --offline 2>&1 | tee "${ROOT}/logs/${label}_smoke.log"
}

run_endpoint 0 0 ce &
PID_CE=$!
run_endpoint 1 5 kd &
PID_KD=$!
wait "${PID_CE}"
wait "${PID_KD}"
