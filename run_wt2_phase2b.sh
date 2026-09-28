#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
TEACHER_DIR="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
DATASET_PATH="/workspace/grassmannflows/datasets/wikitext2_v1_saved"
DISTILL_OUT="outputs/distill_experiments"

cd "$PROJECT_ROOT"
mkdir -p logs "$DISTILL_OUT"

BASELINE_DIR=$(python3 -c "
import pathlib
matches = sorted(pathlib.Path('outputs/hybrid_experiments').glob('*wt2_hybrid_lite_baseline_224x56_l6_e20'))
print(matches[-1] if matches else '')
")

echo "Baseline dir: $BASELINE_DIR"

run_distill() {
  local gpu="$1"
  local tag="$2"
  local distill_alpha="$3"
  local student_init="$4"

  echo "[gpu=$gpu] Starting: $tag (alpha=$distill_alpha)"

  local extra_args=""
  if [[ -n "$student_init" ]]; then
    extra_args="--student-init-run-dir $student_init"
  fi

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "$gpu" \
    --teacher-run-dir "$TEACHER_DIR" \
    $extra_args \
    --student-type hybrid_lite \
    --dataset-path "$DATASET_PATH" \
    --tokenizer-dir ./gpt2_local \
    --output-dir "$DISTILL_OUT" \
    --experiment-name "$tag" \
    --model-dim 224 \
    --num-layers 6 \
    --reduced-dim 56 \
    --window-sizes 1,2,4 \
    --student-late-k 1 \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --temperature 2.0 \
    --distill-alpha "$distill_alpha" \
    --amp --offline 2>&1 | tee "logs/${tag}.log"

  echo "[gpu=$gpu] Done: $tag"
}

# Only use GPUs 0,1,2 — avoid GPU 3
run_distill 0 "wt2_warmstart_kd020"  0.2  "$BASELINE_DIR" &
run_distill 1 "wt2_warmstart_kd002"  0.02 "$BASELINE_DIR" &

wait

echo "=== Phase 2b done ==="

python3 <<'PY'
import json, pathlib

print(f"\n{'='*90}")
print(f"{'Experiment':<45} {'Type':<12} {'Test PPL':<10} {'Best Val PPL':<13} {'Params':<10}")
print(f"{'-'*90}")

for tag, etype in [
    ("wt2_warmstart_kd020", "warm+KD020"),
    ("wt2_warmstart_kd002", "warm+KD002"),
]:
    matches = sorted(pathlib.Path("outputs/distill_experiments").glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"  {'(missing) ' + tag:<55} {'':<12} {'-':<10}")
        continue
    p = matches[-1]
    d = json.loads(p.read_text())
    block = d.get("student", {})
    test_ppl = block.get("test_ppl", "-")
    val_ppl = block.get("best_val_ppl", "-")
    params = block.get("num_params", "-")
    print(f"  {tag:<45} {etype:<12} {str(test_ppl):<10} {str(val_ppl):<13} {str(params):<10}")

print(f"{'-'*90}")
PY
