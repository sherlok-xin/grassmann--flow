#!/usr/bin/env bash
set -u

cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs

run_retry() {
  local arm="$1"
  local strategy="$2"
  local branch_lambda="$3"
  timeout 18000 python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id 3 \
    --teacher-run-dir outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10 \
    --student-init-run-dir outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20 \
    --student-type hybrid_lite --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments \
    --experiment-name "stage_c_ts20k_${arm}_retry_gpu3" \
    --notes "Stage-C exact-config retry after GPU3 clock event; frozen protocol 2026-09-17" \
    --tags stage_c,prototype,tinystories,matched,infrastructure_retry \
    --seed 42 --batch-size 8 --epochs 2 --lr 1e-4 --weight-decay 0.01 \
    --warmup-ratio 0.05 --num-workers 2 --model-dim 224 --num-layers 6 \
    --num-heads 8 --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 \
    --student-late-k 1 --temperature 2.0 --distill-alpha 0.0 \
    --kd-loss-mode token_mean --kd-lambda 5 --kd-chunk-tokens 64 \
    --distill-strategy "${strategy}" --branch-kd-lambda "${branch_lambda}" \
    --routing-tau 0.25 --routing-seed 1729 --dataset-name tinystories \
    --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
    --text-field text --max-seq-len 256 --max-lines 20000 \
    --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
    --split-seed 42 --amp --offline \
    >"logs/stage_c_ts20k_${arm}_retry_gpu3.log" 2>&1
}

run_retry c3_disagreement disagreement_suppressed 0
run_retry c7_swapped crbd_swapped 2.5
