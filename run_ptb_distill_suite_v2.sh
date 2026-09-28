#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
TEACHER_RUN="${TEACHER_RUN:-outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/distill_experiments}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_DIR"

echo "=== PTB distillation suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TEACHER_RUN=$TEACHER_RUN"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python train_distill_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_distill_grassmann_balanced \
  --student-type grassmann \
  --num-layers 6 \
  --model-dim 224 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.7 \
  --amp --offline 2>&1 | tee logs/ptb_distill_grassmann_balanced_v3.log

python train_distill_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_distill_grassmann_light \
  --student-type grassmann \
  --num-layers 4 \
  --model-dim 192 \
  --reduced-dim 48 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.7 \
  --amp --offline 2>&1 | tee logs/ptb_distill_grassmann_light_v3.log

python train_distill_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_distill_transformer_control \
  --student-type transformer \
  --num-layers 6 \
  --model-dim 224 \
  --num-heads 8 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.7 \
  --amp --offline 2>&1 | tee logs/ptb_distill_transformer_control_v3.log

python - <<'PY'
import json, pathlib
print("=== PTB distillation suite summary ===")
for tag in [
    "ptb_distill_grassmann_balanced",
    "ptb_distill_grassmann_light",
    "ptb_distill_transformer_control",
]:
    matches = sorted(pathlib.Path("outputs/distill_experiments").glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"[missing] {tag}")
        continue
    p = matches[-1]
    data = json.loads(p.read_text())["student"]
    print("=" * 80)
    print(p)
    print("best_val_ppl:", data.get("best_val_ppl"))
    print("test_ppl:", data.get("test_ppl"))
    print("beats_grassmann_baseline:", data.get("beats_grassmann_baseline"))
    print("beats_transformer_baseline:", data.get("beats_transformer_baseline"))
    print("beats_both_baselines:", data.get("beats_both_baselines"))
PY