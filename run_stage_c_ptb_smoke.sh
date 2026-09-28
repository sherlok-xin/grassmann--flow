#!/usr/bin/env bash
set -u

cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs

run_arm() {
  local gpu="$1"
  local arm="$2"
  local strategy="$3"
  local log="logs/stage_c_ptb_smoke_${arm}.log"
  timeout 3600 python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "${gpu}" \
    --teacher-run-dir outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint \
    --student-init-run-dir outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20 \
    --student-type hybrid_lite \
    --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments \
    --experiment-name "stage_c_ptb_smoke_${arm}" \
    --notes "Stage-C PTB max_lines=2000 smoke; frozen protocol 2026-09-17" \
    --tags stage_c,smoke,ptb,matched \
    --seed 42 --batch-size 8 --epochs 1 --lr 1e-4 --weight-decay 0.01 \
    --warmup-ratio 0.05 --num-workers 2 --model-dim 224 --num-layers 6 \
    --num-heads 8 --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 \
    --student-late-k 1 --temperature 2.0 --distill-alpha 0.0 \
    --kd-loss-mode token_mean --kd-lambda 5 --kd-chunk-tokens 64 \
    --distill-strategy "${strategy}" --branch-kd-lambda 2.5 --routing-tau 0.25 \
    --routing-seed 1729 --dataset-name ptb \
    --dataset-path /workspace/grassmannflows/datasets/ptb_text_only_saved \
    --text-field sentence --max-seq-len 256 --max-lines 2000 \
    --encode-chars-per-batch 200000 --split-seed 42 --amp --offline \
    >"${log}" 2>&1
}

run_arm 0 c1_fixed fixed_fused & p0=$!
run_arm 1 c5_crbd crbd & p1=$!
run_arm 2 c6_shuffled crbd_shuffled & p2=$!
run_arm 3 c7_swapped crbd_swapped & p3=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
