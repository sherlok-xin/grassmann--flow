#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
TEACHER_RUN="${TEACHER_RUN:-outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
BASE_OUT="${BASE_OUT:-outputs/experiments}"
DISTILL_OUT="${DISTILL_OUT:-outputs/distill_experiments}"

cd "$PROJECT_ROOT"
mkdir -p logs "$BASE_OUT" "$DISTILL_OUT"

echo "=== PTB student sanity suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TEACHER_RUN=$TEACHER_RUN"
echo "DATASET_PATH=$DATASET_PATH"

# 1) Grassmann baseline: weak compression sanity
python train_exp4_ddp_v5.py \
  --gpu-id "$GPU_ID" \
  --model grassmann \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$BASE_OUT" \
  --experiment-name ptb_baseline_grassmann_sanity_240x64 \
  --num-layers 6 \
  --model-dim 240 \
  --reduced-dim 64 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --amp \
  --offline 2>&1 | tee logs/ptb_baseline_grassmann_sanity_240x64.log

# 2) Transformer baseline: control
python train_exp4_ddp_v5.py \
  --gpu-id "$GPU_ID" \
  --model transformer \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$BASE_OUT" \
  --experiment-name ptb_baseline_transformer_control_224x6 \
  --num-layers 6 \
  --model-dim 224 \
  --num-heads 8 \
  --epochs 10 \
  --batch-size 32 \
  --amp \
  --offline 2>&1 | tee logs/ptb_baseline_transformer_control_224x6.log

# 3) Distill sanity A: same-capacity-ish Grassmann, more CE weight
python train_distill_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_distill_grassmann_sanity_240x64_a \
  --student-type grassmann \
  --num-layers 6 \
  --model-dim 240 \
  --reduced-dim 64 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.3 \
  --amp \
  --offline 2>&1 | tee logs/ptb_distill_grassmann_sanity_240x64_a.log

# 4) Distill sanity B: same-capacity-ish Grassmann, hotter KD
python train_distill_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_distill_grassmann_sanity_240x64_b \
  --student-type grassmann \
  --num-layers 6 \
  --model-dim 240 \
  --reduced-dim 64 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 4.0 \
  --distill-alpha 0.5 \
  --amp \
  --offline 2>&1 | tee logs/ptb_distill_grassmann_sanity_240x64_b.log

# 5) Distill sanity C: mild compression Grassmann
python train_distill_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_distill_grassmann_mild_224x56 \
  --student-type grassmann \
  --num-layers 6 \
  --model-dim 224 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 2e-4 \
  --temperature 2.0 \
  --distill-alpha 0.3 \
  --amp \
  --offline 2>&1 | tee logs/ptb_distill_grassmann_mild_224x56.log

python - <<'PY'
import json
import pathlib

print("=== PTB student sanity suite summary ===")

targets = [
    ("baseline", "ptb_baseline_grassmann_sanity_240x64", "outputs/experiments"),
    ("baseline", "ptb_baseline_transformer_control_224x6", "outputs/experiments"),
    ("distill", "ptb_distill_grassmann_sanity_240x64_a", "outputs/distill_experiments"),
    ("distill", "ptb_distill_grassmann_sanity_240x64_b", "outputs/distill_experiments"),
    ("distill", "ptb_distill_grassmann_mild_224x56", "outputs/distill_experiments"),
]

for kind, tag, root in targets:
    matches = sorted(pathlib.Path(root).glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"[missing] {tag}")
        continue
    p = matches[-1]
    data = json.loads(p.read_text())
    print("=" * 80)
    print(p)
    if kind == "baseline":
        block = data.get("grassmann") or data.get("transformer") or data
        print("best_val_ppl:", block.get("best_val_ppl"))
        print("test_ppl:", block.get("test_ppl"))
        print("checkpoint_path:", block.get("checkpoint_path"))
    else:
        block = data["student"]
        print("best_val_ppl:", block.get("best_val_ppl"))
        print("test_ppl:", block.get("test_ppl"))
        print("beats_grassmann_baseline:", block.get("beats_grassmann_baseline"))
        print("beats_transformer_baseline:", block.get("beats_transformer_baseline"))
        print("beats_both_baselines:", block.get("beats_both_baselines"))
PY
