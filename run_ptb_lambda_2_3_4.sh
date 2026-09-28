#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
BASELINE="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET="/workspace/grassmannflows/datasets/ptb_text_only_saved"

for lambda in 0.02 0.03 0.04; do
  lam3=$(python3 -c "print(int($lambda * 1000))")
  TAG="ptb_warmstart_lambda_${lam3}"
  echo "=== [$lambda] $TAG ==="

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id 1 --teacher-run-dir "$TEACHER" \
    --student-init-run-dir "$BASELINE" --student-type hybrid_lite \
    --dataset-path "$DATASET" --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments --experiment-name "$TAG" \
    --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
    --student-late-k 1 --epochs 10 --batch-size 32 --lr 1e-4 \
    --temperature 2.0 --distill-alpha "$lambda" --amp --offline

  echo "=== $TAG completed ==="
  D=$(ls -dt outputs/distill_experiments/*_${TAG} 2>/dev/null | head -1)
  if [[ -f "$D/summary.json" ]]; then
    python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'$TAG: test_ppl={s[\"test_ppl\"]:.4f}')"
  fi
done

echo "===== ALL DONE ====="
for lam in 20 30 40; do
  D=$(ls -dt outputs/distill_experiments/*ptb_warmstart_lambda_${lam}* 2>/dev/null | head -1)
  if [[ -f "$D/summary.json" ]]; then
    python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'kd${lam}: test_ppl={s[\"test_ppl\"]:.4f}')"
  fi
done
