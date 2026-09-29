#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
EXP="research/experiments/phase2e_teacher_branch_ablation"
RAW="$EXP/raw"
LOGS="$EXP/logs"
mkdir -p "$RAW/gradient" "$LOGS" "$EXP/figures"

TEACHER="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
S0_42="outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20"
S0_123="outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123"
S0_456="outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456"
HASH_42="9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef"
HASH_123="a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13"
HASH_456="3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757"

python "$EXP/audit_start.py" --output "$RAW/start_manifest.json" >"$LOGS/audit_start.log" 2>&1

run_utility() {
  local gpu="$1" seed="$2" student="$3" hash="$4"
  python "$EXP/run_branch_utility.py" \
    --student-seed "$seed" --student-run-dir "$student" \
    --expected-student-sha256 "$hash" --teacher-run-dir "$TEACHER" \
    --gpu-id "$gpu" --output "$RAW/utility_seed${seed}.json" \
    >"$LOGS/utility_seed${seed}.log" 2>&1
}

run_utility 0 42 "$S0_42" "$HASH_42" & p0=$!
run_utility 1 123 "$S0_123" "$HASH_123" & p1=$!
run_utility 2 456 "$S0_456" "$HASH_456" & p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  printf 'At least one Phase 2E branch-utility job failed.\n' >&2
  exit 1
fi

run_gradient() {
  local gpu="$1" seed="$2" condition="$3" alpha="$4" student="$5" hash="$6"
  python "$EXP/run_gradient_preflight.py" \
    --condition "$condition" --teacher-alpha "$alpha" --teacher-run-dir "$TEACHER" \
    --student-seed "$seed" --student-run-dir "$student" \
    --expected-student-sha256 "$hash" --gpu-id "$gpu" \
    --output "$RAW/gradient/seed${seed}_${condition}.json" \
    >"$LOGS/gradient_seed${seed}_${condition}.log" 2>&1
}

run_gradient_wave() {
  local condition="$1" alpha="$2"
  run_gradient 0 42 "$condition" "$alpha" "$S0_42" "$HASH_42" & p0=$!
  run_gradient 1 123 "$condition" "$alpha" "$S0_123" "$HASH_123" & p1=$!
  run_gradient 2 456 "$condition" "$alpha" "$S0_456" "$HASH_456" & p2=$!
  local status=0
  local pid
  for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
  if [[ "$status" -ne 0 ]]; then
    printf 'Phase 2E gradient wave %s failed.\n' "$condition" >&2
    return 1
  fi
}

run_gradient_wave F 0.5
run_gradient_wave T 1.0
run_gradient_wave G 0.0
python "$EXP/collect_preflight.py" | tee "$LOGS/collect_preflight.log"
