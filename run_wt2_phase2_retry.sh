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

if [[ -z "$BASELINE_DIR" ]]; then
  echo "[FATAL] baseline directory not found"
  exit 1
fi

echo "Baseline dir: $BASELINE_DIR"
echo "Teacher dir:  $TEACHER_DIR"

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

# Run 6 experiments on 4 GPUs
run_distill 0 "wt2_random_init_kd005"      0.05 ""               &
run_distill 1 "wt2_warmstart_ce_only"      0.0  "$BASELINE_DIR" &
run_distill 2 "wt2_warmstart_kd005"        0.05 "$BASELINE_DIR" &
run_distill 3 "wt2_warmstart_kd010"        0.1  "$BASELINE_DIR" &

wait

echo "=== Phase 2a done, starting 2b ==="

run_distill 0 "wt2_warmstart_kd020"        0.2  "$BASELINE_DIR" &
run_distill 1 "wt2_warmstart_kd002"        0.02 "$BASELINE_DIR" &

wait

echo "=== All distill experiments done ==="

python3 <<'PY'
import json, pathlib

print(f"\n{'='*90}")
print(f"{'Experiment':<45} {'Type':<12} {'Test PPL':<10} {'Best Val PPL':<13} {'Params':<10}")
print(f"{'-'*90}")

targets = [
    ("wt2_random_init_kd005",      "random+KD",   "outputs/distill_experiments"),
    ("wt2_warmstart_ce_only",      "warm+CEonly", "outputs/distill_experiments"),
    ("wt2_warmstart_kd005",        "warm+KD005",  "outputs/distill_experiments"),
    ("wt2_warmstart_kd010",        "warm+KD010",  "outputs/distill_experiments"),
    ("wt2_warmstart_kd020",        "warm+KD020",  "outputs/distill_experiments"),
    ("wt2_warmstart_kd002",        "warm+KD002",  "outputs/distill_experiments"),
]

for tag, etype, root in targets:
    matches = sorted(pathlib.Path(root).glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"  {'(missing) ' + tag:<55} {'':<12} {'-':<10}")
        continue
    p = matches[-1]
    d = json.loads(p.read_text())
    block = d.get("student", {})
    test_ppl = block.get("test_ppl", "-")
    val_ppl = block.get("best_val_ppl", "-")
    params = block.get("num_params", "-")
    beats_g = block.get("beats_grassmann_baseline", None)
    beats_t = block.get("beats_transformer_baseline", None)
    print(f"  {tag:<45} {etype:<12} {str(test_ppl):<10} {str(val_ppl):<13} {str(params):<10}")

print(f"{'-'*90}")
print("\n=== Key Comparison ===")
print(f"Baseline (CE-only from scratch): 70.16")
for tag, etype, root in targets:
    matches = sorted(pathlib.Path(root).glob(f"*{tag}/summary.json"))
    if not matches:
        continue
    p = matches[-1]
    d = json.loads(p.read_text())
    block = d.get("student", {})
    print(f"  {tag}: test_ppl={block.get('test_ppl', '?')}, beats_grassmann={block.get('beats_grassmann_baseline')}, beats_transformer={block.get('beats_transformer_baseline')}")
PY
