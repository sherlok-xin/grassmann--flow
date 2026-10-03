#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
phase="research/experiments/phase4a_modern_generalization_pilot"
export PYTHONPATH="outputs/phase4a_modern/python_deps"
export TOKENIZERS_PARALLELISM=false
out="outputs/phase4a_modern/runs/$dataset"
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 python "$phase/scripts/train.py" "$dataset" teacher > "$out/teacher.log" 2>&1 &
teacher_pid=$!
CUDA_VISIBLE_DEVICES=1 python "$phase/scripts/train.py" "$dataset" s0 > "$out/s0.log" 2>&1 &
student_pid=$!
printf '%s\n' "$teacher_pid" "$student_pid" > "$out/preparation.pids"
if ! wait "$teacher_pid"; then
    wait "$student_pid" || true
    exit 1
fi
wait "$student_pid"
CUDA_VISIBLE_DEVICES=0 python "$phase/scripts/init_audit.py" "$dataset" > "$out/initialization_audit.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python "$phase/scripts/train.py" "$dataset" ce > "$out/ce.log" 2>&1 &
ce_pid=$!
CUDA_VISIBLE_DEVICES=1 python "$phase/scripts/train.py" "$dataset" kd1 > "$out/kd1.log" 2>&1 &
kd1_pid=$!
CUDA_VISIBLE_DEVICES=2 python "$phase/scripts/train.py" "$dataset" kd5 > "$out/kd5.log" 2>&1 &
kd5_pid=$!
printf '%s\n' "$ce_pid" "$kd1_pid" "$kd5_pid" > "$out/continuation.pids"
failed=0
wait "$ce_pid" || failed=1
wait "$kd1_pid" || failed=1
wait "$kd5_pid" || failed=1
[[ "$failed" == 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=0 python "$phase/scripts/finish_dataset.py" "$dataset" > "$out/final_evaluation.log" 2>&1
