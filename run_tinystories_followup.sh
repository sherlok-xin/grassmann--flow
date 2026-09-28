#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

run_exp () {
  local name="$1"
  shift
  echo "===== Running: ${name} ====="
  "$@" 2>&1 | tee "logs/${name}.log"
  echo "===== Finished: ${name} ====="
}

# 1) TinyStories fusion: transformer + grassmann, both 300k / 10 epoch / split3
run_exp tinystories_fusion_300k_e10_split3 \
  python eval_fusion3.py \
    --grassmann-run-dir outputs/experiments/20260319_020425_tinystories_ws124_rd64_md240_300k_e10_split3 \
    --transformer-run-dir outputs/experiments/20260319_010013_tinystories_transformer_300k_e10_split3 \
    --alpha-step 0.05 \
    --output-json outputs/experiments/tinystories_fusion_300k_e10_split3.json

# 2) TinyStories distill: a=0.1, T=1.0
#run_exp tinystories_distill_t2g_ws124_rd64_md240_a01_t1 \
 # python train_distill3.py \
  #  --offline \
   # --teacher-run-dir outputs/experiments/20260319_010013_tinystories_transformer_300k_e10_split3 \
    #--dataset-name tinystories \
    #--dataset-path /root/songxin/datasets/tinystories_saved \
    #--text-field text \
    #--experiment-name tinystories_distill_t2g_ws124_rd64_md240_a01_t1 \
    #--window-sizes 1,2,4 \
    #--reduced-dim 64 \
    #--model-dim 240 \
    #--distill-alpha 0.1 \
    #--temperature 1.0 \
    #--max-lines 300000

# 3) TinyStories distill: a=0.05, T=1.0
run_exp tinystories_distill_t2g_ws124_rd64_md240_a005_t1 \
  python train_distill3.py \
    --offline \
    --teacher-run-dir outputs/experiments/20260319_010013_tinystories_transformer_300k_e10_split3 \
    --dataset-name tinystories \
    --dataset-path /songxin/datasets/tinystories_saved \
    --text-field text \
    --experiment-name tinystories_distill_t2g_ws124_rd64_md240_a005_t1 \
    --window-sizes 1,2,4 \
    --reduced-dim 64 \
    --model-dim 240 \
    --distill-alpha 0.05 \
    --temperature 1.0 \
    --max-lines 300000

### 4) TinyStories short-window ablation: [1,2] vs current [1,2,4]
#run_exp tinystories_ws12_rd64_md240_300k_e10_split3 \
# python train_exp4.py \
#    --offline \
#    --dataset-name tinystories \
#    --dataset-path /songxin/datasets/tinystories_saved \
#    --text-field text \
#    --model grassmann \
#    --model-dim 240 \
#    --window-sizes 1,2 \
#    --reduced-dim 64 \
#    --epochs 10 \
#    --max-lines 300000 \
#    --encode-chars-per-batch 200000 \
#    --tinystories-val-frac 0.02 \
#    --split-seed 42 \
#    --experiment-name tinystories_ws12_rd64_md240_300k_e10_split3 \
#    --tags ablation,tinystories
#
#echo "===== All TinyStories follow-up experiments finished ====="
###