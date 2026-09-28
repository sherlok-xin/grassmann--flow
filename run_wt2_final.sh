#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
DATASET="/workspace/grassmannflows/datasets/wikitext2_v1_saved"
BASELINE=$(python3 -c "import pathlib; m=sorted(pathlib.Path('outputs/hybrid_experiments').glob('*wt2_hybrid_lite_baseline_224x56_l6_e20')); print(m[-1] if m else '')")

echo "=== WT2 Final Distill: kd020 + kd002 ==="
echo "Baseline: $BASELINE"

run_one() {
  local gpu=$1 tag=$2 alpha=$3
  echo "[gpu=$gpu] $tag alpha=$alpha"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "$gpu" --teacher-run-dir "$TEACHER" \
    --student-init-run-dir "$BASELINE" \
    --student-type hybrid_lite --dataset-path "$DATASET" \
    --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments \
    --experiment-name "$tag" --model-dim 224 --num-layers 6 --reduced-dim 56 \
    --window-sizes 1,2,4 --student-late-k 1 --epochs 10 --batch-size 32 \
    --lr 1e-4 --temperature 2.0 --distill-alpha "$alpha" --amp --offline
  echo "[gpu=$gpu] DONE: $tag"
}

run_one 0 wt2_warmstart_kd020 0.2 &
run_one 1 wt2_warmstart_kd002 0.02 &
wait

echo "=== ALL DONE ==="
for tag in wt2_warmstart_kd020 wt2_warmstart_kd002; do
  d=$(ls -dt outputs/distill_experiments/*_${tag} 2>/dev/null | head -1)
  if [[ -f "$d/summary.json" ]]; then
    python3 -c "import json; s=json.load(open('$d/summary.json'))['student']; print(f'$tag: test_ppl={s[\"test_ppl\"]:.4f}, best_val_ppl={s[\"best_val_ppl\"]:.4f}')"
  fi
done
