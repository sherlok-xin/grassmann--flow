#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# WT2 Distillation Control Suite
# Replicates PTB-style contrast on Wikitext-2:
#   "random-init KD fails, warm-start KD works"
#
# Usage (run on 197 via agent-tools):
#   ./agent-tools/run197_submit.sh --name wt2_suite -- bash run_wt2_distill_suite.sh
# ============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
TEACHER_DIR="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
DATASET_PATH="/workspace/grassmannflows/datasets/wikitext2_v1_saved"
HYBRID_OUT="outputs/hybrid_experiments"
DISTILL_OUT="outputs/distill_experiments"

cd "$PROJECT_ROOT"
mkdir -p logs "$HYBRID_OUT" "$DISTILL_OUT"

echo "============================================================"
echo " WT2 Distillation Control Suite"
echo " Project root: $PROJECT_ROOT"
echo " Teacher:      $TEACHER_DIR"
echo " Dataset:      $DATASET_PATH"
echo "============================================================"

# ============================================================================
# Phase 1: Train hybrid-lite baseline from scratch (CE-only reference)
#           This is BOTH the CE-only baseline AND the warm-start init.
# ============================================================================

echo ""
echo "========== Phase 1: Hybrid-Lite Baseline (from scratch, 20 epochs) =========="

BASELINE_TAG="wt2_hybrid_lite_baseline_224x56_l6_e20"

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id 0 \
  --dataset-name wikitext2 \
  --dataset-path "$DATASET_PATH" \
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
  --amp --offline 2>&1 | tee "logs/${BASELINE_TAG}.log"

# Locate the baseline output directory
BASELINE_DIR=$(python3 -c "
import pathlib
matches = sorted(pathlib.Path('$HYBRID_OUT').glob('*${BASELINE_TAG}'))
print(matches[-1] if matches else '')
")

if [[ -z "$BASELINE_DIR" ]]; then
  echo "[FATAL] Phase 1 failed: could not find baseline output directory"
  exit 1
fi

BASELINE_CKPT="$BASELINE_DIR/checkpoints/hybrid_best.pt"
if [[ ! -f "$BASELINE_CKPT" ]]; then
  echo "[FATAL] Phase 1 failed: no checkpoint at $BASELINE_CKPT"
  exit 1
fi

BASELINE_SUMMARY="$BASELINE_DIR/summary.json"
echo "[Phase 1 done] baseline dir: $BASELINE_DIR"
python3 -c "import json; d=json.load(open('$BASELINE_SUMMARY')); h=d['hybrid']; print(f'  best_val_ppl: {h[\"best_val_ppl\"]:.4f}, test_ppl: {h[\"test_ppl\"]:.4f}, params: {h[\"num_params\"]}')"

# ============================================================================
# Phase 2: Distillation contrast experiments (parallel across GPUs)
# ============================================================================

echo ""
echo "========== Phase 2: Distillation Contrasts (4 GPUs in parallel) =========="

run_distill() {
  local gpu="$1"
  local tag="$2"
  local distill_alpha="$3"
  local student_init="$4"   # "" for random init, or path to baseline

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

  echo "[gpu=$gpu] Finished: $tag"
}

# Launch all 4 in parallel
run_distill 0 "wt2_random_init_kd005"      0.05 ""                 &
run_distill 1 "wt2_warmstart_ce_only"      0.0  "$BASELINE_DIR"   &
run_distill 2 "wt2_warmstart_kd005"        0.05 "$BASELINE_DIR"   &
run_distill 3 "wt2_warmstart_kd010"        0.1  "$BASELINE_DIR"   &

wait

echo ""
echo "========== Phase 2b: Additional KD alpha points =========="

run_distill 0 "wt2_warmstart_kd020"        0.2  "$BASELINE_DIR"   &
run_distill 1 "wt2_warmstart_kd002"        0.02 "$BASELINE_DIR"   &

wait

# ============================================================================
# Phase 3: Summary
# ============================================================================

echo ""
echo "============================================================"
echo " WT2 Distillation Suite Summary"
echo "============================================================"

python3 <<'PY'
import json, pathlib, os

hybrid_out = os.environ.get("HYBRID_OUT", "outputs/hybrid_experiments")
distill_out = os.environ.get("DISTILL_OUT", "outputs/distill_experiments")

print(f"\n{'='*90}")
print(f"{'Experiment':<45} {'Type':<12} {'Test PPL':<10} {'Best Val PPL':<13} {'Params':<10}")
print(f"{'-'*90}")

targets = [
    # Phase 1
    ("wt2_hybrid_lite_baseline_224x56_l6_e20", "baseline", hybrid_out),
    # Phase 2
    ("wt2_random_init_kd005",      "random+KD",   distill_out),
    ("wt2_warmstart_ce_only",      "warm+CEonly", distill_out),
    ("wt2_warmstart_kd005",        "warm+KD005",  distill_out),
    ("wt2_warmstart_kd010",        "warm+KD010",  distill_out),
    ("wt2_warmstart_kd020",        "warm+KD020",  distill_out),
    ("wt2_warmstart_kd002",        "warm+KD002",  distill_out),
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
        print(f"  {tag:<45} {'unknown':<12} {'-':<10}")
        continue
    test_ppl = block.get("test_ppl", "-")
    val_ppl = block.get("best_val_ppl", "-")
    params = block.get("num_params", "-")
    print(f"  {tag:<45} {etype:<12} {str(test_ppl):<10} {str(val_ppl):<13} {str(params):<10}")

print(f"{'-'*90}")

# Also print key comparison
print("\n=== Key Comparisons ===")
for tag, etype, root in targets:
    matches = sorted(pathlib.Path(root).glob(f"*{tag}/summary.json"))
    if not matches:
        continue
    p = matches[-1]
    d = json.loads(p.read_text())
    block = d.get("student", d.get("hybrid", {}))
    test_ppl = block.get("test_ppl", float("inf"))
    val_ppl = block.get("best_val_ppl", float("inf"))
    beats_g = block.get("beats_grassmann_baseline", None)
    beats_t = block.get("beats_transformer_baseline", None)
    print(f"  {tag}: test_ppl={test_ppl:.2f}, beats_grassmann={beats_g}, beats_transformer={beats_t}")
PY
