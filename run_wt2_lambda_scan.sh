#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
BASELINE="outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20"
DATASET="/workspace/grassmannflows/datasets/wikitext2_v1_saved"

for lambda in 0.01 0.03 0.04; do
  lam3=$(python3 -c "print(int($lambda * 1000))")
  TAG="wt2_warmstart_lambda_${lam3}"
  echo "=== [$lambda] $TAG ==="

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id 2 --teacher-run-dir "$TEACHER" \
    --student-init-run-dir "$BASELINE" --student-type hybrid_lite \
    --dataset-path "$DATASET" --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments --experiment-name "$TAG" \
    --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
    --student-late-k 1 --epochs 10 --batch-size 32 --lr 1e-4 \
    --temperature 2.0 --distill-alpha "$lambda" --amp --offline

  D=$(ls -dt outputs/distill_experiments/*_${TAG} 2>/dev/null | head -1)
  [[ -f "$D/summary.json" ]] && python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'$TAG: test_ppl={s[\"test_ppl\"]:.4f}')"
  echo "=== $TAG done ==="
done

echo "===== ALL DONE ====="
python3 << 'PY'
import json, pathlib, re
print("\nWT2 Lambda Scan Full Results:")
# Collect all WT2 warmstart results
for p in sorted(pathlib.Path("outputs/distill_experiments").glob("*wt2_warmstart_*")):
    s = p / "summary.json"
    if not s.exists(): continue
    d = json.loads(s.read_text())
    if "student" not in d: continue
    # Try to get alpha from config
    cfg_path = p / "config.json"
    alpha = "?"
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        alpha = c.get("config", c).get("distill_alpha", "?")
    print(f"  {p.name}: test_ppl={d['student']['test_ppl']:.4f}  alpha={alpha}")
PY
