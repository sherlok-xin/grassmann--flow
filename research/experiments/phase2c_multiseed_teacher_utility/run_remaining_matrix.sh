#!/usr/bin/env bash
set -euo pipefail

# Execute only after both formal S0 jobs have completed. This script is a frozen
# launcher for the preregistered Phase 2C matrix; it performs no selection or
# hyperparameter tuning.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

EXP="research/experiments/phase2c_multiseed_teacher_utility"
TEACHER="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
S0_123="outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123"
S0_456="outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456"
DATASET="/workspace/grassmannflows/datasets/wikitext2_v1_saved"

deadline=$((SECONDS + 7200))
until [[ -s "$S0_123/summary.json" && -s "$S0_456/summary.json" ]]; do
  if (( SECONDS >= deadline )); then
    printf 'Timed out waiting for both formal S0 summaries.\n' >&2
    exit 1
  fi
  sleep 30
done

for run_dir in "$S0_123" "$S0_456"; do
  test -s "$run_dir/config.json"
  test -s "$run_dir/summary.json"
  test -s "$run_dir/checkpoints/hybrid_best.pt"
done

HASH_123="$(sha256sum "$S0_123/checkpoints/hybrid_best.pt" | cut -d' ' -f1)"
HASH_456="$(sha256sum "$S0_456/checkpoints/hybrid_best.pt" | cut -d' ' -f1)"

run_utility() {
  local seed="$1"
  local state="$2"
  local student_dir="$3"
  local output_dir="$4"
  local log_name="$5"
  python "$EXP/run_teacher_utility.py" \
    --student-seed "$seed" \
    --student-state "$state" \
    --student-run-dir "$student_dir" \
    --teacher-run-dir "$TEACHER" \
    --dataset-name wikitext2 \
    --dataset-path "$DATASET" \
    --text-field text \
    --tokenizer-dir ./gpt2_local \
    --selection-seed 20260920 \
    --max-chunks 512 \
    --batch-size 1 \
    --num-workers 2 \
    --temperature 2 \
    --token-block 32 \
    --alphas 0.5,0.3,0.0 \
    --model-dim 224 \
    --num-layers 6 \
    --num-heads 8 \
    --reduced-dim 56 \
    --window-sizes 1,2,4 \
    --dropout 0.1 \
    --student-late-k 1 \
    --split-seed 42 \
    --gpu-id 3 \
    --output-dir "$output_dir" \
    --offline >"$EXP/logs/$log_name" 2>&1
}

run_arm() {
  local gpu="$1"
  local seed="$2"
  local arm="$3"
  local alpha="$4"
  local lambda="$5"
  local student_dir="$6"
  local student_hash="$7"
  local experiment_name="phase2c_wt2_${arm}_seed${seed}"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --teacher-run-dir "$TEACHER" \
    --student-type hybrid_lite \
    --tokenizer-dir ./gpt2_local \
    --student-init-run-dir "$student_dir" \
    --expected-student-init-sha256 "$student_hash" \
    --output-dir outputs/distill_experiments \
    --seed "$seed" \
    --batch-size 32 \
    --epochs 10 \
    --lr 0.0001 \
    --weight-decay 0.01 \
    --warmup-ratio 0.05 \
    --num-workers 4 \
    --amp \
    --log-interval 50 \
    --model-dim 224 \
    --num-layers 6 \
    --num-heads 8 \
    --reduced-dim 56 \
    --window-sizes 1,2,4 \
    --dropout 0.1 \
    --student-late-k 1 \
    --temperature 2 \
    --kd-loss-mode token_mean \
    --kd-chunk-tokens 1024 \
    --distill-strategy fixed_fused \
    --dataset-name wikitext2 \
    --dataset-path "$DATASET" \
    --text-field text \
    --max-seq-len 256 \
    --max-lines 0 \
    --encode-chars-per-batch 200000 \
    --tinystories-val-frac 0.02 \
    --split-seed 42 \
    --offline \
    --gpu-id "$gpu" \
    --experiment-name "$experiment_name" \
    --teacher-alpha-override "$alpha" \
    --kd-lambda "$lambda" \
    --notes "phase2c_preregistered_multiseed_replication" \
    --tags "phase2c,formal,seed${seed},${arm}" \
    >"$EXP/logs/${experiment_name}.log" 2>&1
}

wait_all() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  if [[ "$status" -ne 0 ]]; then
    printf 'At least one preregistered arm failed; stopping before the next wave.\n' >&2
    return 1
  fi
}

# Utility is evaluation-only, so GPU 3 is reserved for it while formal training
# uses the three previously verified training devices (0, 1, and 2).
run_utility 123 S0 "$S0_123" "$EXP/raw/utility_s0_seed123" utility_s0_seed123.log &
utility_pid=$!

run_arm 0 123 c0 0.5 0 "$S0_123" "$HASH_123" &
p0=$!
run_arm 1 123 c1_a05 0.5 5 "$S0_123" "$HASH_123" &
p1=$!
run_arm 2 123 c3_a00 0.0 5 "$S0_123" "$HASH_123" &
p2=$!
wait_all "$p0" "$p1" "$p2"
wait "$utility_pid"

C0_123="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2c_wt2_c0_seed123' -print | sort | tail -n 1)"
test -s "$C0_123/summary.json"
run_utility 123 C0 "$C0_123" "$EXP/raw/utility_c0_seed123" utility_c0_seed123.log &
utility_pid=$!

run_arm 0 456 c0 0.5 0 "$S0_456" "$HASH_456" &
p0=$!
run_arm 1 456 c1_a05 0.5 5 "$S0_456" "$HASH_456" &
p1=$!
run_arm 2 456 c3_a00 0.0 5 "$S0_456" "$HASH_456" &
p2=$!
wait_all "$p0" "$p1" "$p2"
wait "$utility_pid"

C0_456="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2c_wt2_c0_seed456' -print | sort | tail -n 1)"
test -s "$C0_456/summary.json"
run_utility 456 S0 "$S0_456" "$EXP/raw/utility_s0_seed456" utility_s0_seed456.log
run_utility 456 C0 "$C0_456" "$EXP/raw/utility_c0_seed456" utility_c0_seed456.log

python "$EXP/aggregate_utility.py"
python "$EXP/collect_results.py"
python "$EXP/plot_results.py"
