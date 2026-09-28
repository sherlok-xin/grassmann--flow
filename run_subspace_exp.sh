#!/usr/bin/env bash
# ============================================================================
# Subspace Rank Collapse Experiment
#
# Runs 3 training conditions on PTB with spectral monitoring:
#   ce_scratch:   Random init + CE (no teacher)
#   kd_scratch:   Random init + KD (teacher signal from scratch)
#   kd_warmstart: Warm-start + KD (baseline init + teacher)
#
# Hypothesis: Random init KD collapses Plucker subspace rank;
#             warmstart preserves healthy geometric diversity.
# ============================================================================
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/subspace_analysis

TEACHER="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
BASELINE="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET="/workspace/grassmannflows/datasets/ptb_text_only_saved"

echo "===== Subspace Analysis: 3 conditions on PTB ====="
echo "Teacher: $TEACHER"
echo "Baseline: $BASELINE"

run_condition() {
  local gpu=$1 cond=$2 tag=$3
  echo "[GPU$gpu] Starting: $tag"
  python run_subspace_analysis.py \
    --gpu-id "$gpu" --condition "$cond" \
    --dataset-path "$DATASET" \
    --teacher-dir "$TEACHER" --baseline-dir "$BASELINE" \
    --output-dir outputs/subspace_analysis \
    --epochs 10 --batch-size 16 --lr 1e-4 \
    --distill-alpha 0.02 --temperature 2.0 --seed 42 \
    2>&1 | tee "logs/subspace_${tag}.log"
  echo "[GPU$gpu] DONE: $tag"
}

# Run all 3 in parallel
run_condition 0 ce_scratch   subspace_ce_scratch &
run_condition 1 kd_scratch   subspace_kd_scratch &
run_condition 2 kd_warmstart subspace_kd_warmstart &

wait

echo "===== All subspace experiments done ====="
echo ""
echo "=== Spectral History Summary ==="
for cond in ce_scratch kd_scratch kd_warmstart; do
  D=$(ls -dt outputs/subspace_analysis/*_subspace_${cond} 2>/dev/null | head -1)
  if [[ -f "$D/spectral_history.json" ]]; then
    echo ""
    echo "--- $cond ---"
    python3 -c "
import json
h = json.load(open('$D/spectral_history.json'))
print(f\"{'Ep':>4s} {'val_ppl':>8s} {'eff_rank':>9s} {'spect_ent':>9s} {'top5':>7s}\")
for r in h:
    print(f\"{r['epoch']:4d} {r['val_ppl']:8.2f} {r['effective_rank']:9.1f} {r['spectral_entropy']:9.4f} {r['top5_ratio']:7.4f}\")
"
  fi
done
