#!/usr/bin/env bash
set -u

cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs

run_arm() {
  local gpu="$1"
  local arm="$2"
  local strategy="$3"
  local fused_lambda="$4"
  local branch_lambda="$5"
  local log="logs/stage_c_ts20k_${arm}.log"
  timeout 14400 python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "${gpu}" \
    --teacher-run-dir outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10 \
    --student-init-run-dir outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20 \
    --student-type hybrid_lite --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments \
    --experiment-name "stage_c_ts20k_${arm}" \
    --notes "Stage-C TinyStories max_lines=20000 prototype; frozen protocol 2026-09-17" \
    --tags stage_c,prototype,tinystories,matched \
    --seed 42 --batch-size 8 --epochs 2 --lr 1e-4 --weight-decay 0.01 \
    --warmup-ratio 0.05 --num-workers 2 --model-dim 224 --num-layers 6 \
    --num-heads 8 --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 \
    --student-late-k 1 --temperature 2.0 --distill-alpha 0.0 \
    --kd-loss-mode token_mean --kd-lambda "${fused_lambda}" --kd-chunk-tokens 64 \
    --distill-strategy "${strategy}" --branch-kd-lambda "${branch_lambda}" \
    --routing-tau 0.25 --routing-seed 1729 --dataset-name tinystories \
    --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
    --text-field text --max-seq-len 256 --max-lines 20000 \
    --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 \
    --split-seed 42 --amp --offline >"${log}" 2>&1
}

(run_arm 0 c0_ce fixed_fused 0 0; run_arm 0 c5_crbd crbd 5 2.5) & p0=$!
(run_arm 1 c1_fixed fixed_fused 5 0; run_arm 1 c6_shuffled crbd_shuffled 5 2.5) & p1=$!
(run_arm 2 c2_entropy entropy_gated 5 0; run_arm 2 c4_branch branch_only 0 2.5) & p2=$!
(run_arm 3 c3_disagreement disagreement_suppressed 5 0; run_arm 3 c7_swapped crbd_swapped 5 2.5) & p3=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
