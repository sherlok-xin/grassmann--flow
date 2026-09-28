#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "===== TinyStories 100k Transformer smoke test ====="
python train_exp4.py \
  --offline \
  --dataset-name tinystories \
  --dataset-path /root/songxin/datasets/tinystories_saved \
  --text-field text \
  --model transformer \
  --epochs 5 \
  --max-lines 100000 \
  --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 \
  --split-seed 42 \
  --experiment-name tinystories_transformer_100k_split3 \
  --tags baseline,tinystories,smoketest 2>&1 | tee logs/tinystories_transformer_100k_split3.log

echo "===== TinyStories 100k Grassmann smoke test ====="
python train_exp4.py \
  --offline \
  --dataset-name tinystories \
  --dataset-path /root/songxin/datasets/tinystories_saved \
  --text-field text \
  --model grassmann \
  --model-dim 240 \
  --window-sizes 1,2,4 \
  --reduced-dim 64 \
  --epochs 5 \
  --max-lines 100000 \
  --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 \
  --split-seed 42 \
  --experiment-name tinystories_ws124_rd64_md240_100k_split3 \
  --tags ablation,tinystories,smoketest 2>&1 | tee logs/tinystories_ws124_rd64_md240_100k_split3.log

echo "===== All smoke tests finished ====="