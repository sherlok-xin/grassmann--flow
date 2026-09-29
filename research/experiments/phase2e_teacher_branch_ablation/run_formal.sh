#!/usr/bin/env bash
set -euo pipefail

# This launcher creates exactly the three preregistered T endpoints. F, G, and
# C0 are reused from Phase 2C and must not be rerun.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
EXP="research/experiments/phase2e_teacher_branch_ablation"
TEACHER="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
DATASET="/workspace/grassmannflows/datasets/wikitext2_v1_saved"
REDUCED_GATE="$EXP/raw/smoke/reduced_smoke_summary.json"
MATCHED_GATE="$EXP/raw/smoke/matched_full_epoch_smoke_summary.json"

python - "$REDUCED_GATE" "$MATCHED_GATE" <<'PY'
import json
import pathlib
import sys

authorized = False
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("formal_training_authorized") is True:
        authorized = True
if not authorized:
    raise SystemExit("Neither the reduced nor matched full-epoch stability gate authorized formal training")
PY

run_arm() {
  local gpu="$1" seed="$2" student_dir="$3" student_hash="$4"
  local experiment_name="phase2e_wt2_t_seed${seed}"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --teacher-run-dir "$TEACHER" --student-type hybrid_lite --tokenizer-dir ./gpt2_local \
    --student-init-run-dir "$student_dir" --expected-student-init-sha256 "$student_hash" \
    --output-dir outputs/distill_experiments --seed "$seed" --batch-size 32 --epochs 10 \
    --lr 0.0001 --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --amp \
    --log-interval 50 --model-dim 224 --num-layers 6 --num-heads 8 \
    --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 --student-late-k 1 \
    --temperature 2 --kd-loss-mode token_mean --kd-chunk-tokens 1024 \
    --distill-strategy fixed_fused --dataset-name wikitext2 --dataset-path "$DATASET" \
    --text-field text --max-seq-len 256 --max-lines 0 --encode-chars-per-batch 200000 \
    --tinystories-val-frac 0.02 --split-seed 42 --offline --gpu-id "$gpu" \
    --experiment-name "$experiment_name" --teacher-alpha-override 1.0 --kd-lambda 5 \
    --notes phase2e_preregistered_teacher_branch_ablation \
    --tags "phase2e,formal,seed${seed},T" \
    >"$EXP/logs/${experiment_name}.log" 2>&1
}

run_arm 0 42 \
  outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20 \
  9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef & pid_42=$!
run_arm 1 123 \
  outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123 \
  a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13 & pid_123=$!
run_arm 2 456 \
  outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456 \
  3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757 & pid_456=$!

status=0
for pid in "$pid_42" "$pid_123" "$pid_456"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  printf 'At least one Phase 2E formal T arm failed.\n' >&2
  exit 1
fi
