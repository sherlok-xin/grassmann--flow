#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/hybrid_experiments
DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"
echo "===== A: Code CE from scratch ====="
python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id 0 --dataset-path "$DATASET" --text-field text \
  --tokenizer-dir ./gpt2_local --output-dir outputs/hybrid_experiments \
  --experiment-name code_hybrid_lite_baseline_ce_scratch \
  --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
  --late-k 1 --epochs 10 --batch-size 16 --lr 2e-4 --max-seq-len 256 \
  --max-lines 5000 --amp --offline
echo "===== DONE ====="
D=$(ls -dt outputs/hybrid_experiments/*code_hybrid_lite_baseline_ce_scratch 2>/dev/null | head -1)
[[ -f "$D/summary.json" ]] && python3 -c "import json; h=json.load(open('$D/summary.json'))['hybrid']; print(f'Code CE scratch: test_ppl={h[\"test_ppl\"]:.4f}')"
