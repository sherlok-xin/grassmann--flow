#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/hybrid_experiments
DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"
GRASS="outputs/experiments/20260318_102137_ptb_ws124_rd64_md240"
TRANS="outputs/experiments/20260318_094720_ptb_baseline_both"
echo "===== B: Code Teacher ====="
python train_hybrid_latefusion_alpha_ddp_v1.py \
  --gpu-id 1 --grassmann-run-dir "$GRASS" --transformer-run-dir "$TRANS" \
  --dataset-path "$DATASET" --text-field text \
  --tokenizer-dir ./gpt2_local --output-dir outputs/hybrid_experiments \
  --experiment-name code_latefusion_teacher_joint \
  --train-mode joint --init-alpha 0.5 --late-k 1 \
  --epochs 10 --batch-size 16 --lr 1e-5 --alpha-lr 1e-2 \
  --max-seq-len 256 --max-lines 5000 --amp --offline
echo "===== DONE ====="
D=$(ls -dt outputs/hybrid_experiments/*code_latefusion_teacher_joint 2>/dev/null | head -1)
[[ -f "$D/summary.json" ]] && python3 -c "import json; h=json.load(open('$D/summary.json'))['hybrid']; print(f'Code Teacher: test_ppl={h[\"test_ppl\"]:.4f}')"
