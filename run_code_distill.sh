#!/usr/bin/env bash
# ============================================================================
# Cross-Domain Distillation: PTB models → Code (CodeParrot subset)
#
# Uses PTB-pretrained Hybrid Teacher and PTB-pretrained Hybrid-Lite Baseline
# to test whether warm-start KD transfers to Python code.
#
# 3 conditions (same as PTB/WT2/TS):
#   kd_scratch:   Random init + KD (tests: does KD work from scratch on code?)
#   ce_warmstart: Warmstart + CE-only (tests: does baseline transfer to code?)
#   kd_warmstart: Warmstart + KD λ=0.02 (tests: does KD help cross-domain?)
# ============================================================================
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

# PTB-pretrained models
TEACHER="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
BASELINE="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"

echo "============================================================"
echo " Cross-Domain Distillation: PTB → Code"
echo " Teacher:  $TEACHER (PTB-pretrained)"
echo " Baseline: $BASELINE (PTB-pretrained)"
echo " Dataset:  $DATASET"
echo "============================================================"

run_one() {
  local gpu=$1 tag=$2 lam=$3 warm=$4
  echo "[GPU$gpu] $tag λ=$lam warm=$warm"

  local init_arg=""
  if [[ "$warm" == "yes" ]]; then
    init_arg="--student-init-run-dir $BASELINE"
  fi

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "$gpu" --teacher-run-dir "$TEACHER" \
    $init_arg --student-type hybrid_lite \
    --dataset-path "$DATASET" --text-field text \
    --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments \
    --experiment-name "$tag" \
    --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
    --student-late-k 1 --epochs 10 --batch-size 16 --lr 1e-4 \
    --temperature 2.0 --distill-alpha "$lam" \
    --max-seq-len 256 \
    --amp --offline 2>&1 | tee "logs/${tag}.log"

  echo "[GPU$gpu] DONE: $tag"
}

# 3 experiments in parallel on GPU 0,1,2
run_one 0 code_random_init_kd005  0.05 no  &
run_one 1 code_warmstart_ce_only  0.0  yes &
run_one 2 code_warmstart_kd002    0.02 yes &

wait

echo "===== ALL DONE ====="

python3 << 'PY'
import json, pathlib
print("\n=== Code Dataset Distillation Results ===")
for tag in ["code_random_init_kd005", "code_warmstart_ce_only", "code_warmstart_kd002"]:
    matches = sorted(pathlib.Path("outputs/distill_experiments").glob(f"*{tag}/summary.json"))
    if matches:
        s = json.loads(matches[-1].read_text())["student"]
        print(f"  {tag}: test_ppl={s['test_ppl']:.4f}")
    else:
        print(f"  {tag}: (not done yet)")
PY
