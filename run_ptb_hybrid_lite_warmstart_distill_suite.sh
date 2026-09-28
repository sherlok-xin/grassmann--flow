#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
TEACHER_RUN="${TEACHER_RUN:-outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint}"
STUDENT_INIT_RUN="${STUDENT_INIT_RUN:-outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/distill_experiments}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_DIR"

echo "=== PTB hybrid-lite warm-start distillation suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TEACHER_RUN=$TEACHER_RUN"
echo "STUDENT_INIT_RUN=$STUDENT_INIT_RUN"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# CE-only warm-start control
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$STUDENT_INIT_RUN" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_hybrid_lite_warmstart_ce_only \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 5e-5 \
  --temperature 2.0 \
  --distill-alpha 0.0 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_ce_only.log

# Light KD
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$STUDENT_INIT_RUN" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_hybrid_lite_warmstart_kd01 \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 5e-5 \
  --temperature 2.0 \
  --distill-alpha 0.1 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_kd01.log

# Medium KD
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$STUDENT_INIT_RUN" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name ptb_hybrid_lite_warmstart_kd02 \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 5e-5 \
  --temperature 2.0 \
  --distill-alpha 0.2 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_kd02.log

python - <<'PY'
import json, pathlib
print("=== PTB hybrid-lite warm-start distillation suite summary ===")
for tag in [
    "ptb_hybrid_lite_warmstart_ce_only",
    "ptb_hybrid_lite_warmstart_kd01",
    "ptb_hybrid_lite_warmstart_kd02",
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
