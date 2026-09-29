#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
EXP="research/experiments/phase2d_fixed_composition_teacher_quality"
RAW="$EXP/raw"
LOGS="$EXP/logs"
mkdir -p "$RAW/gradient" "$LOGS" "$EXP/figures"

TEACHER_J="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
TEACHER_A="outputs/hybrid_experiments/20260328_084319_wt2_v1_hybrid_alpha_only"
S0_42="outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20"
S0_123="outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123"
S0_456="outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456"
HASH_42="9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef"
HASH_123="a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13"
HASH_456="3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757"

python "$EXP/audit_teacher_integrity.py" >"$LOGS/teacher_integrity.log" 2>&1

python "$EXP/run_teacher_preflight.py" \
  --teacher-label J --teacher-run-dir "$TEACHER_J" --gpu-id 0 \
  --output "$RAW/teacher_J_preflight.json" >"$LOGS/teacher_J_preflight.log" 2>&1 &
p0=$!
python "$EXP/run_teacher_preflight.py" \
  --teacher-label A --teacher-run-dir "$TEACHER_A" --gpu-id 1 \
  --output "$RAW/teacher_A_preflight.json" >"$LOGS/teacher_A_preflight.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"

run_gradient() {
  local gpu="$1" seed="$2" label="$3" teacher="$4" student="$5" hash="$6"
  python "$EXP/run_gradient_scale_preflight.py" \
    --teacher-label "$label" --teacher-run-dir "$teacher" \
    --student-seed "$seed" --student-run-dir "$student" \
    --expected-student-sha256 "$hash" --gpu-id "$gpu" \
    --output "$RAW/gradient/seed${seed}_${label}.json" \
    >"$LOGS/gradient_seed${seed}_${label}.log" 2>&1
}

run_gradient 0 42 J "$TEACHER_J" "$S0_42" "$HASH_42" & p0=$!
run_gradient 1 42 A "$TEACHER_A" "$S0_42" "$HASH_42" & p1=$!
run_gradient 2 123 J "$TEACHER_J" "$S0_123" "$HASH_123" & p2=$!
run_gradient 3 123 A "$TEACHER_A" "$S0_123" "$HASH_123" & p3=$!
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"

run_gradient 0 456 J "$TEACHER_J" "$S0_456" "$HASH_456" & p0=$!
run_gradient 1 456 A "$TEACHER_A" "$S0_456" "$HASH_456" & p1=$!
wait "$p0"; wait "$p1"

python "$EXP/collect_preflight.py" | tee "$LOGS/collect_preflight.log"
