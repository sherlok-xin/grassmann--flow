#!/usr/bin/env bash
#利用多gpu跑实验
set -euo pipefail

GPU_LIST="${1:-2,3}"
PER_GPU_BATCH="${2:-4}"
NUM_WORKERS="${3:-2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"

NPROC=$(python3 - <<PY
print(len("${GPU_LIST}".split(",")))
PY
)

cd "$PROJECT_ROOT"
source /workspace/.venvs/grassmannflows/bin/activate

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

echo "[info] GPUs=$GPU_LIST nproc=$NPROC per_gpu_batch=$PER_GPU_BATCH workers=$NUM_WORKERS"

#torchrun --standalone --nproc_per_node="$NPROC" train_distill3_ddp.py \
##  --offline \
#  --teacher-run-dir outputs/experiments/20260319_010013_tinystories_transformer_300k_e10_split3 \
#  --dataset-name tinystories \
#  --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
##  --text-field text \
#  --experiment-name tinystories_distill_t2g_ws124_rd64_md240_a005_t1 \
#  --window-sizes 1,2,4 \
##  --reduced-dim 64 \
#  --model-dim 240 \
#  --distill-alpha 0.05 \
##  --temperature 1.0 \
#  --max-lines 300000 \
#  --batch-size "$PER_GPU_BATCH" \
#  --num-workers "$NUM_WORKERS"

torchrun --standalone --nproc_per_node="$NPROC" train_exp4_ddp.py \
  --offline \
  --dataset-name tinystories \
  --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
  --text-field text \
  --model grassmann \
  --model-dim 240 \
  --window-sizes 1,2 \
  --reduced-dim 64 \
  --epochs 10 \
  --max-lines 300000 \
  --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 \
  --split-seed 42 \
  --experiment-name tinystories_ws12_rd64_md240_300k_e10_split3 \
  --tags ablation,tinystories \
  --batch-size "$PER_GPU_BATCH" \
  --num-workers "$NUM_WORKERS"
