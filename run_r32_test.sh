#!/usr/bin/env bash
# Test: r=32 vs r=56 on PTB — does smaller r preserve KD benefit?
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows

TEACHER="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
BASELINE="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET="/workspace/grassmannflows/datasets/ptb_text_only_saved"

# Train r=32 baseline first (no student-init, from scratch)
echo "===== r=32 CE scratch baseline ====="
python train_hybrid_lite_latefusion_baseline_v2.py \
  --gpu-id 0 --dataset-path "$DATASET" --dataset-name ptb \
  --tokenizer-dir ./gpt2_local --output-dir outputs/hybrid_experiments \
  --experiment-name ptb_r32_baseline --model-dim 224 --num-layers 6 \
  --reduced-dim 32 --window-sizes 1,2,4 --late-k 1 \
  --epochs 10 --batch-size 32 --lr 2e-4 --max-seq-len 256 --amp --offline

R32_BL=$(ls -dt outputs/hybrid_experiments/*ptb_r32_baseline 2>/dev/null | head -1)
echo "r=32 baseline: $R32_BL"

# r=32 warmstart KD
echo ""
echo "===== r=32 warmstart KD λ=0.02 ====="
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id 1 --teacher-run-dir "$TEACHER" --student-init-run-dir "$R32_BL" \
  --student-type hybrid_lite --dataset-path "$DATASET" --dataset-name ptb \
  --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments \
  --experiment-name ptb_r32_warmstart_kd002 --model-dim 224 --num-layers 6 \
  --reduced-dim 32 --window-sizes 1,2,4 --student-late-k 1 \
  --epochs 10 --batch-size 32 --lr 1e-4 --temperature 2.0 --distill-alpha 0.02 \
  --max-seq-len 256 --amp --offline

echo ""
echo "===== DONE ====="

python3 << 'PY'
import json, pathlib

# r=32 results
for tag in ["ptb_r32_baseline", "ptb_r32_warmstart_kd002"]:
    matches = sorted(pathlib.Path(".").glob(f"outputs/*_experiments/*{tag}*/summary.json"))
    if not matches:
        matches = sorted(pathlib.Path(".").glob(f"outputs/*_experiments/*{tag}/summary.json"))
    if matches:
        d = json.loads(matches[-1].read_text())
        key = "hybrid" if "hybrid" in d else "student"
        r = d[key]
        print(f"{tag}: test_ppl={r['test_ppl']:.4f}, params={r['num_params']:,}")

# Compare with r=56
print()
print("=== r=56 (reference) ===")
print("r=56 baseline: test_ppl=58.53, params=31,434,257")
print("r=56 KD λ=0.02: test_ppl=51.27, params=31,434,257")
PY
