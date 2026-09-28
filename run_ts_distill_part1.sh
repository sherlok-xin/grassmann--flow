#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER=$(python3 -c "import pathlib; m=sorted(pathlib.Path('outputs/hybrid_experiments').glob('*ts_latefusion_last1_joint_e10')); print(m[-1])")
echo "TS Teacher: $TEACHER"

# Random init KD (no warm-start needed)
echo "=== TS random_init_kd005 ==="
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id 1 --teacher-run-dir "$TEACHER" --student-type hybrid_lite \
  --dataset-name tinystories --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
  --text-field text --tokenizer-dir ./gpt2_local \
  --output-dir outputs/distill_experiments --experiment-name ts_random_init_kd005 \
  --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
  --student-late-k 1 --epochs 10 --batch-size 32 --lr 1e-4 \
  --temperature 2.0 --distill-alpha 0.05 \
  --max-lines 300000 --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 --split-seed 42 \
  --amp --offline

echo "=== random_init_kd005 done ==="
D=$(ls -dt outputs/distill_experiments/*ts_random_init_kd005 2>/dev/null | head -1)
[[ -f "$D/summary.json" ]] && python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'test_ppl={s[\"test_ppl\"]:.4f}')"
