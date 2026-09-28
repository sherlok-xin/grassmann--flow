#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows

mkdir -p logs outputs/hybrid_experiments

echo "===== TinyStories Hybrid Teacher Training ====="

python train_hybrid_latefusion_alpha_ddp_v1.py \
  --gpu-id 2 \
  --grassmann-run-dir outputs/experiments/20260322_094948_tinystories_ws124_rd64_md240_300k_e20_split3 \
  --transformer-run-dir outputs/experiments/20260322_095111_tinystories_transformer_300k_e20_split3 \
  --dataset-name tinystories \
  --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
  --text-field text \
  --tokenizer-dir ./gpt2_local \
  --output-dir outputs/hybrid_experiments \
  --experiment-name ts_latefusion_last1_joint_e10 \
  --train-mode joint \
  --init-alpha 0.5 \
  --late-k 1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-5 \
  --alpha-lr 1e-2 \
  --max-lines 300000 \
  --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 \
  --split-seed 42 \
  --amp --offline 2>&1 | tee logs/ts_latefusion_last1_joint_e10.log

echo "===== Teacher training done ====="

TEACHER_DIR=$(python3 -c "
import pathlib
matches = sorted(pathlib.Path('outputs/hybrid_experiments').glob('*ts_latefusion_last1_joint_e10'))
print(matches[-1] if matches else '')
")

if [[ -n "$TEACHER_DIR" ]]; then
  python3 -c "import json; d=json.load(open('${TEACHER_DIR}/summary.json')); h=d['hybrid']; print(f'Teacher test_ppl: {h[\"test_ppl\"]:.4f}, params: {h[\"num_params\"]}')"
fi
