#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
TEACHER_RUN="${TEACHER_RUN:-outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/distill_experiments}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_DIR"

echo "=== PTB hybrid-lite distillation suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TEACHER_RUN=$TEACHER_RUN"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# 1) Mild compression hybrid-lite; most likely to work
python train_distill_hybrid_lite_from_latefusion_teacher_v1.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_distill_hybrid_lite_mild \
  --student-type hybrid_lite \
  --student-late-k 1 \
  --num-layers 6 \
  --model-dim 224 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.3 \
  --amp --offline 2>&1 | tee logs/ptb_distill_hybrid_lite_mild.log

# 2) Balanced smaller hybrid-lite
python train_distill_hybrid_lite_from_latefusion_teacher_v1.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_distill_hybrid_lite_balanced \
  --student-type hybrid_lite \
  --student-late-k 1 \
  --num-layers 6 \
  --model-dim 192 \
  --reduced-dim 48 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.3 \
  --amp --offline 2>&1 | tee logs/ptb_distill_hybrid_lite_balanced.log

# 3) Mild compression with slightly stronger KD
python train_distill_hybrid_lite_from_latefusion_teacher_v1.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_distill_hybrid_lite_mild_hot \
  --student-type hybrid_lite \
  --student-late-k 1 \
  --num-layers 6 \
  --model-dim 224 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 4.0 \
  --distill-alpha 0.5 \
  --amp --offline 2>&1 | tee logs/ptb_distill_hybrid_lite_mild_hot.log

python - <<'PY'
import json, pathlib
print("=== PTB hybrid-lite distillation suite summary ===")
for tag in [
    "ptb_distill_hybrid_lite_mild",
    "ptb_distill_hybrid_lite_balanced",
    "ptb_distill_hybrid_lite_mild_hot",
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
