#!/usr/bin/env bash
# ============================================================================
# Code Cross-Task Verification
#
# A: Hybrid-lite baseline on code (CE from scratch)
#    → establishes code-domain CE baseline
# B: Late-fusion Teacher on code (PTB branches + joint fine-tune)
#    → tests whether Grassmann+Transformer fusion works cross-domain
#
# Key comparisons:
#   A vs existing code_warmstart_ce_only (4.17): does PTB pretrain help?
#   B vs A: does fusion improve on code?
#   B vs PTB Teacher (50.11): how well does fusion transfer cross-domain?
# ============================================================================
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/hybrid_experiments

DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"

echo "============================================================"
echo " Code Cross-Task Verification"
echo " Dataset: $DATASET"
echo "============================================================"

# ============================================================================
# A: Code CE from scratch (GPU 0, background)
# ============================================================================
(
echo ""
echo "===== A: Code CE from scratch ====="

python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id 0 \
  --dataset-path "$DATASET" --text-field text \
  --tokenizer-dir ./gpt2_local --output-dir outputs/hybrid_experiments \
  --experiment-name code_hybrid_lite_baseline_ce_scratch \
  --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
  --late-k 1 --epochs 10 --batch-size 16 --lr 2e-4 \
  --max-seq-len 256 \
  --amp --offline 2>&1 | tee logs/code_ce_scratch.log

echo "===== A done ====="
D_A=$(ls -dt outputs/hybrid_experiments/*code_hybrid_lite_baseline_ce_scratch 2>/dev/null | head -1)
[[ -f "$D_A/summary.json" ]] && python3 -c "import json; h=json.load(open('$D_A/summary.json'))['hybrid']; print(f'Code CE scratch: test_ppl={h[\"test_ppl\"]:.4f}')"
) &

# ============================================================================
# B: Code Teacher (GPU 1, background)
# ============================================================================
(
echo ""
echo "===== B: Code Teacher ====="

GRASS="outputs/experiments/20260318_102137_ptb_ws124_rd64_md240"
TRANS="outputs/experiments/20260318_094720_ptb_baseline_both"

python train_hybrid_latefusion_alpha_ddp_v1.py \
  --gpu-id 1 \
  --grassmann-run-dir "$GRASS" --transformer-run-dir "$TRANS" \
  --dataset-path "$DATASET" --text-field text \
  --tokenizer-dir ./gpt2_local --output-dir outputs/hybrid_experiments \
  --experiment-name code_latefusion_teacher_joint \
  --train-mode joint --init-alpha 0.5 --late-k 1 \
  --epochs 10 --batch-size 16 --lr 1e-5 --alpha-lr 1e-2 \
  --max-seq-len 256 \
  --amp --offline 2>&1 | tee logs/code_teacher.log

echo "===== B done ====="
D_B=$(ls -dt outputs/hybrid_experiments/*code_latefusion_teacher_joint 2>/dev/null | head -1)
[[ -f "$D_B/summary.json" ]] && python3 -c "import json; h=json.load(open('$D_B/summary.json'))['hybrid']; print(f'Code Teacher: test_ppl={h[\"test_ppl\"]:.4f} params={h[\"num_params\"]}')"
) &

wait

echo ""
echo "===== ALL DONE ====="
