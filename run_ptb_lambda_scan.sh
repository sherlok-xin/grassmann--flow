#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
BASELINE="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET="/workspace/grassmannflows/datasets/ptb_text_only_saved"

echo "===== PTB Lambda Fine Scan: 0.01, 0.02, 0.03, 0.04 ====="
echo "Teacher: $TEACHER"
echo "Baseline: $BASELINE"

for lambda in 0.01 0.02 0.03 0.04; do
  lam3=$(python3 -c "print(int($lambda * 1000))")
  TAG="ptb_warmstart_kd${lam3}"
  echo ""
  echo "=== [$lambda] $TAG ==="

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id 1 \
    --teacher-run-dir "$TEACHER" \
    --student-init-run-dir "$BASELINE" \
    --student-type hybrid_lite \
    --dataset-path "$DATASET" \
    --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments \
    --experiment-name "$TAG" \
    --model-dim 224 --num-layers 6 --reduced-dim 56 \
    --window-sizes 1,2,4 --student-late-k 1 \
    --epochs 10 --batch-size 32 --lr 1e-4 \
    --temperature 2.0 --distill-alpha "$lambda" \
    --amp --offline 2>&1 | tee "logs/${TAG}.log"

  echo "=== $TAG done ==="
  D=$(ls -dt outputs/distill_experiments/*_${TAG} 2>/dev/null | head -1)
  if [[ -f "$D/summary.json" ]]; then
    python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'$TAG: test_ppl={s[\"test_ppl\"]:.4f}, best_val_ppl={s[\"best_val_ppl\"]:.4f}')"
  fi
done

echo ""
echo "===== PTB Lambda Scan Complete ====="
python3 <<'PY'
import json, pathlib
print(f"\n{'Tag':<35} {'lambda':<8} {'Test PPL':<10} {'Best Val PPL':<13}")
print("-" * 68)
# Map lambda to expected experiment tag substring
for lam, tag_sub in [("0.01","kd10"), ("0.02","kd20"), ("0.03","kd30"), ("0.04","kd40"), ("0.05","kd005")]:
    matches = sorted(pathlib.Path("outputs/distill_experiments").glob(f"*ptb_warmstart_{tag_sub}*/summary.json"))
    if not matches and lam == "0.05":
        matches = sorted(pathlib.Path("outputs/distill_experiments").glob("*ptb_hybrid_lite_warmstart_kd005*/summary.json"))
    if matches:
        s = json.loads(matches[-1].read_text())["student"]
        print(f"  {matches[-1].parent.name:<35} λ={lam:<6} {s['test_ppl']:<10.4f} {s['best_val_ppl']:<13.4f}")
    else:
        print(f"  {'(missing)':<35} λ={lam:<6} {'-':<10} {'-':<13}")
PY
