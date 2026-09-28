#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
TEACHER_RUN="${TEACHER_RUN:-outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint}"
BEST_STUDENT_INIT="${BEST_STUDENT_INIT:-outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
HYBRID_OUT="${HYBRID_OUT:-outputs/hybrid_experiments}"
DISTILL_OUT="${DISTILL_OUT:-outputs/distill_experiments}"

cd "$PROJECT_ROOT"
mkdir -p logs "$HYBRID_OUT" "$DISTILL_OUT"

echo "=== PTB next-stage warm-start suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TEACHER_RUN=$TEACHER_RUN"
echo "BEST_STUDENT_INIT=$BEST_STUDENT_INIT"
echo "DATASET_PATH=$DATASET_PATH"

# ------------------------------------------------------------------
# Part A: small KD scan around the current best route
# ------------------------------------------------------------------

python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$BEST_STUDENT_INIT" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_hybrid_lite_warmstart_kd005 \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --temperature 2.0 \
  --distill-alpha 0.05 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_kd005.log

python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$BEST_STUDENT_INIT" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_hybrid_lite_warmstart_kd015 \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --temperature 2.0 \
  --distill-alpha 0.15 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_kd015.log

# ------------------------------------------------------------------
# Part B: smaller student candidate 1 = fewer layers (224/56, 4 layers)
# ------------------------------------------------------------------

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id "$GPU_ID" \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$HYBRID_OUT" \
  --experiment-name ptb_hybrid_lite_baseline_224x56_l4_e20 \
  --model-dim 224 \
  --num-layers 4 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --late-k 1 \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_baseline_224x56_l4_e20.log

L4_INIT=$(python - <<'PY'
import pathlib
matches = sorted(pathlib.Path("outputs/hybrid_experiments").glob("*ptb_hybrid_lite_baseline_224x56_l4_e20"))
print(matches[-1] if matches else "")
PY
)

python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$L4_INIT" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_hybrid_lite_warmstart_224x56_l4_kd01 \
  --model-dim 224 \
  --num-layers 4 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --temperature 2.0 \
  --distill-alpha 0.1 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_224x56_l4_kd01.log

# ------------------------------------------------------------------
# Part C: smaller student candidate 2 = narrower (192/48, 6 layers)
# ------------------------------------------------------------------

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id "$GPU_ID" \
  --dataset-name ptb \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$HYBRID_OUT" \
  --experiment-name ptb_hybrid_lite_baseline_192x48_e20 \
  --model-dim 192 \
  --num-layers 6 \
  --reduced-dim 48 \
  --window-sizes 1,2,4 \
  --late-k 1 \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_baseline_192x48_e20.log

NARROW_INIT=$(python - <<'PY'
import pathlib
matches = sorted(pathlib.Path("outputs/hybrid_experiments").glob("*ptb_hybrid_lite_baseline_192x48_e20"))
print(matches[-1] if matches else "")
PY
)

python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id "$GPU_ID" \
  --teacher-run-dir "$TEACHER_RUN" \
  --student-init-run-dir "$NARROW_INIT" \
  --student-type hybrid_lite \
  --dataset-path "$DATASET_PATH" \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$DISTILL_OUT" \
  --experiment-name ptb_hybrid_lite_warmstart_192x48_kd01 \
  --model-dim 192 \
  --num-layers 6 \
  --reduced-dim 48 \
  --window-sizes 1,2,4 \
  --student-late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --temperature 2.0 \
  --distill-alpha 0.1 \
  --amp --offline 2>&1 | tee logs/ptb_hybrid_lite_warmstart_192x48_kd01.log

python - <<'PY'
import json, pathlib

print("=== PTB next-stage warm-start suite summary ===")
targets = [
    ("distill", "ptb_hybrid_lite_warmstart_kd005", "outputs/distill_experiments"),
    ("distill", "ptb_hybrid_lite_warmstart_kd01", "outputs/distill_experiments"),
    ("distill", "ptb_hybrid_lite_warmstart_kd015", "outputs/distill_experiments"),
    ("baseline", "ptb_hybrid_lite_baseline_224x56_l4_e20", "outputs/hybrid_experiments"),
    ("distill", "ptb_hybrid_lite_warmstart_224x56_l4_kd01", "outputs/distill_experiments"),
    ("baseline", "ptb_hybrid_lite_baseline_192x48_e20", "outputs/hybrid_experiments"),
    ("distill", "ptb_hybrid_lite_warmstart_192x48_kd01", "outputs/distill_experiments"),
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
        block = data["hybrid"]
        print("best_val_ppl:", block.get("best_val_ppl"))
        print("test_ppl:", block.get("test_ppl"))
        print("best_alpha:", block.get("best_alpha"))
    else:
        block = data["student"]
        print("best_val_ppl:", block.get("best_val_ppl"))
        print("test_ppl:", block.get("test_ppl"))
        print("beats_grassmann_baseline:", block.get("beats_grassmann_baseline"))
        print("beats_transformer_baseline:", block.get("beats_transformer_baseline"))
        print("beats_both_baselines:", block.get("beats_both_baselines"))
PY
