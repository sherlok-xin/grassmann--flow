#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hybrid_experiments}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_DIR"

echo "=== PTB hybrid-lite baseline suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id "$GPU_ID" \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_hybrid_lite_baseline_mild \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_baseline_mild.log

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id "$GPU_ID" \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_hybrid_lite_baseline_balanced \
  --model-dim 192 \
  --num-layers 6 \
  --reduced-dim 48 \
  --window-sizes 1,2,4 \
  --late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_baseline_balanced.log

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id "$GPU_ID" \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_hybrid_lite_baseline_mild_e20 \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --late-k 1 \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_baseline_mild_e20.log

python - <<'PY'
import json, pathlib
print("=== PTB hybrid-lite baseline suite summary ===")
for tag in [
    "ptb_hybrid_lite_baseline_mild",
    "ptb_hybrid_lite_baseline_balanced",
    "ptb_hybrid_lite_baseline_mild_e20",
]:
    matches = sorted(pathlib.Path("outputs/hybrid_experiments").glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"[missing] {tag}")
        continue
    p = matches[-1]
    data = json.loads(p.read_text())["hybrid"]
    print("=" * 80)
    print(p)
    print("best_val_ppl:", data.get("best_val_ppl"))
    print("test_ppl:", data.get("test_ppl"))
    print("best_alpha:", data.get("best_alpha"))
    print("best_alpha_vector:", data.get("best_alpha_vector"))
PY
