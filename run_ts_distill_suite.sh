#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# TinyStories Distillation Control Suite
# Full pipeline: Teacher -> Baseline -> Distillation Contrasts
#
# Replicates PTB findings on TinyStories:
#   "random-init KD fails, warm-start KD works"
#
# Usage:
#   ./agent-tools/run197_submit.sh --name ts_suite -- bash run_ts_distill_suite.sh
# ============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
DATASET_PATH="/workspace/grassmannflows/datasets/tinystories_saved"
HYBRID_OUT="outputs/hybrid_experiments"
DISTILL_OUT="outputs/distill_experiments"

# Source checkpoints for teacher (e20 versions, best performing)
GRASSMANN_RUN="outputs/experiments/20260322_094948_tinystories_ws124_rd64_md240_300k_e20_split3"
TRANSFORMER_RUN="outputs/experiments/20260322_095111_tinystories_transformer_300k_e20_split3"

# TS dataset params
MAX_LINES=300000
ENCODE_CHARS=200000
VAL_FRAC=0.02
SPLIT_SEED=42

cd "$PROJECT_ROOT"
mkdir -p logs "$HYBRID_OUT" "$DISTILL_OUT"

echo "============================================================"
echo " TinyStories Distillation Control Suite"
echo " Project root: $PROJECT_ROOT"
echo " Dataset:      $DATASET_PATH"
echo " Grassmann:    $GRASSMANN_RUN"
echo " Transformer:  $TRANSFORMER_RUN"
echo "============================================================"

# ============================================================================
# Phase 1: Train Hybrid Late-Fusion Teacher
# ============================================================================

echo ""
echo "========== Phase 1: Hybrid Late-Fusion Teacher =========="

TEACHER_TAG="ts_latefusion_last1_joint_e10"

python train_hybrid_latefusion_alpha_ddp_v1.py \
  --gpu-id 0 \
  --grassmann-run-dir "$GRASSMANN_RUN" \
  --transformer-run-dir "$TRANSFORMER_RUN" \
  --dataset-name tinystories \
  --dataset-path "$DATASET_PATH" \
  --text-field text \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$HYBRID_OUT" \
  --experiment-name "$TEACHER_TAG" \
  --train-mode joint \
  --init-alpha 0.5 \
  --late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-5 \
  --alpha-lr 1e-2 \
  --max-lines "$MAX_LINES" \
  --encode-chars-per-batch "$ENCODE_CHARS" \
  --tinystories-val-frac "$VAL_FRAC" \
  --split-seed "$SPLIT_SEED" \
  --amp --offline 2>&1 | tee "logs/${TEACHER_TAG}.log"

TEACHER_DIR=$(python3 -c "
import pathlib
matches = sorted(pathlib.Path('$HYBRID_OUT').glob('*${TEACHER_TAG}'))
print(matches[-1] if matches else '')
")

if [[ -z "$TEACHER_DIR" ]]; then
  echo "[FATAL] Phase 1 failed: could not find teacher output directory"
  exit 1
fi

echo "[Phase 1 done] teacher dir: $TEACHER_DIR"
python3 -c "import json; d=json.load(open('${TEACHER_DIR}/summary.json')); h=d['hybrid']; print(f'  best_val_ppl: {h[\"best_val_ppl\"]:.4f}, test_ppl: {h[\"test_ppl\"]:.4f}, params: {h[\"num_params\"]}')"

# ============================================================================
# Phase 2: Train Hybrid-Lite Baseline from scratch
# ============================================================================

echo ""
echo "========== Phase 2: Hybrid-Lite Baseline (from scratch, 20 epochs) =========="

BASELINE_TAG="ts_hybrid_lite_baseline_224x56_l6_e20"

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id 0 \
  --dataset-name tinystories \
  --dataset-path "$DATASET_PATH" \
  --text-field text \
  --tokenizer-dir ./gpt2_local \
  --output-dir "$HYBRID_OUT" \
  --experiment-name "$BASELINE_TAG" \
  --model-dim 224 \
  --num-layers 6 \
  --reduced-dim 56 \
  --window-sizes 1,2,4 \
  --late-k 1 \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4 \
  --max-lines "$MAX_LINES" \
  --encode-chars-per-batch "$ENCODE_CHARS" \
  --tinystories-val-frac "$VAL_FRAC" \
  --split-seed "$SPLIT_SEED" \
  --amp --offline 2>&1 | tee "logs/${BASELINE_TAG}.log"

BASELINE_DIR=$(python3 -c "
import pathlib
matches = sorted(pathlib.Path('$HYBRID_OUT').glob('*${BASELINE_TAG}'))
print(matches[-1] if matches else '')
")

if [[ -z "$BASELINE_DIR" ]]; then
  echo "[FATAL] Phase 2 failed: could not find baseline output directory"
  exit 1
fi

echo "[Phase 2 done] baseline dir: $BASELINE_DIR"
python3 -c "import json; d=json.load(open('${BASELINE_DIR}/summary.json')); h=d['hybrid']; print(f'  best_val_ppl: {h[\"best_val_ppl\"]:.4f}, test_ppl: {h[\"test_ppl\"]:.4f}, params: {h[\"num_params\"]}')"

# ============================================================================
# Phase 3: Distillation Contrasts (4 GPUs parallel)
# ============================================================================

echo ""
echo "========== Phase 3: Distillation Contrasts =========="

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
    --dataset-name tinystories \
    --dataset-path "$DATASET_PATH" \
    --text-field text \
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
    --max-lines "$MAX_LINES" \
    --encode-chars-per-batch "$ENCODE_CHARS" \
    --tinystories-val-frac "$VAL_FRAC" \
    --split-seed "$SPLIT_SEED" \
    --amp --offline 2>&1 | tee "logs/${tag}.log"

  echo "[gpu=$gpu] Finished: $tag"
}

# 4 experiments in parallel on 4 GPUs
run_distill 0 "ts_random_init_kd005"      0.05 ""               &
run_distill 1 "ts_warmstart_ce_only"      0.0  "$BASELINE_DIR" &
run_distill 2 "ts_warmstart_kd005"        0.05 "$BASELINE_DIR" &
run_distill 3 "ts_warmstart_kd010"        0.1  "$BASELINE_DIR" &

wait

echo ""
echo "========== Phase 3b: Additional KD points =========="

run_distill 0 "ts_warmstart_kd020"        0.2  "$BASELINE_DIR" &
run_distill 1 "ts_warmstart_kd002"        0.02 "$BASELINE_DIR" &

wait

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo " TinyStories Distillation Suite Summary"
echo "============================================================"

python3 <<'PY'
import json, pathlib, os

hybrid_out = os.environ.get("HYBRID_OUT", "outputs/hybrid_experiments")
distill_out = os.environ.get("DISTILL_OUT", "outputs/distill_experiments")

print(f"\n{'='*90}")
print(f"{'Experiment':<45} {'Type':<12} {'Test PPL':<10} {'Best Val PPL':<13} {'Params':<10}")
print(f"{'-'*90}")

targets = [
    ("ts_latefusion_last1_joint_e10",        "teacher",      hybrid_out),
    ("ts_hybrid_lite_baseline_224x56_l6_e20", "baseline",     hybrid_out),
    ("ts_random_init_kd005",                  "random+KD",    distill_out),
    ("ts_warmstart_ce_only",                  "warm+CEonly",  distill_out),
    ("ts_warmstart_kd005",                    "warm+KD005",   distill_out),
    ("ts_warmstart_kd010",                    "warm+KD010",   distill_out),
    ("ts_warmstart_kd020",                    "warm+KD020",   distill_out),
    ("ts_warmstart_kd002",                    "warm+KD002",   distill_out),
]

for tag, etype, root in targets:
    matches = sorted(pathlib.Path(root).glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"  {'(missing) ' + tag:<55} {'':<12} {'-':<10}")
        continue
    p = matches[-1]
    d = json.loads(p.read_text())
    if "hybrid" in d:
        block = d["hybrid"]
    elif "student" in d:
        block = d["student"]
    else:
        continue
    test_ppl = block.get("test_ppl", "-")
    val_ppl = block.get("best_val_ppl", "-")
    params = block.get("num_params", "-")
    print(f"  {tag:<45} {etype:<12} {str(test_ppl):<10} {str(val_ppl):<13} {str(params):<10}")

print(f"{'-'*90}")
PY

echo ""
echo "[done] TinyStories suite complete"
