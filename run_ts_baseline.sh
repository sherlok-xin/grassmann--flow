#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/hybrid_experiments

echo "===== TS Hybrid-Lite Baseline (224x56 l6, 20ep) ====="

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id 0 \
  --dataset-name tinystories \
  --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
  --text-field text \
  --tokenizer-dir ./gpt2_local \
  --output-dir outputs/hybrid_experiments \
  --experiment-name ts_hybrid_lite_baseline_224x56_l6_e20 \
  --model-dim 224 --num-layers 6 --reduced-dim 56 \
  --window-sizes 1,2,4 --late-k 1 \
  --epochs 20 --batch-size 32 --lr 2e-4 \
  --max-lines 300000 --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 --split-seed 42 \
  --amp --offline 2>&1 | tee logs/ts_baseline.log

echo "===== Baseline done ====="

D=$(python3 -c "import pathlib; m=sorted(pathlib.Path('outputs/hybrid_experiments').glob('*ts_hybrid_lite_baseline_224x56_l6_e20')); print(m[-1] if m else '')")
[[ -n "$D" ]] && python3 -c "import json; h=json.load(open('${D}/summary.json'))['hybrid']; print(f'Baseline test_ppl: {h[\"test_ppl\"]:.4f}, params: {h[\"num_params\"]}')"
