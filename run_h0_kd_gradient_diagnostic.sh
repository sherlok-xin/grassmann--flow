#!/usr/bin/env bash
set -euo pipefail

cd /workspace/grassmannflows/grassmann-flows

TEACHER_RUN="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
STUDENT_INIT="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET_PATH="/workspace/grassmannflows/datasets/ptb_text_only_saved"

run_one() {
  local gpu_id="$1"
  local tag="$2"
  local loss_mode="$3"
  local alpha="$4"
  local kd_lambda="$5"

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "${gpu_id}" \
    --teacher-run-dir "${TEACHER_RUN}" \
    --student-init-run-dir "${STUDENT_INIT}" \
    --student-type hybrid_lite \
    --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments \
    --experiment-name "${tag}" \
    --notes "H0 gradient diagnostic after finite/nonfinite AMP metric separation" \
    --tags "h0,smoke,ptb,kd-normalization,gradient-diagnostic" \
    --seed 42 \
    --batch-size 32 \
    --epochs 1 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.05 \
    --num-workers 4 \
    --model-dim 224 \
    --num-layers 6 \
    --num-heads 8 \
    --reduced-dim 56 \
    --window-sizes 1,2,4 \
    --dropout 0.1 \
    --student-late-k 1 \
    --temperature 2.0 \
    --distill-alpha "${alpha}" \
    --kd-loss-mode "${loss_mode}" \
    --kd-lambda "${kd_lambda}" \
    --dataset-name ptb \
    --dataset-path "${DATASET_PATH}" \
    --text-field sentence \
    --max-seq-len 256 \
    --max-lines 0 \
    --encode-chars-per-batch 200000 \
    --split-seed 42 \
    --amp \
    --offline \
    > "logs/${tag}.log" 2>&1
}

run_one 1 h0_ptb_diag_legacy_a002_seed42 legacy_batchmean 0.02 1.0 &
pid_legacy=$!
run_one 2 h0_ptb_diag_token_l5_seed42 token_mean 0.0 5.0 &
pid_token=$!

wait "${pid_legacy}"
wait "${pid_token}"

python3 - <<'PY'
import json
from pathlib import Path

for tag in ("h0_ptb_diag_legacy_a002_seed42", "h0_ptb_diag_token_l5_seed42"):
    run_dir = sorted(Path("outputs/distill_experiments").glob(f"*_{tag}"))[-1]
    metric = json.loads((run_dir / "distill_metrics.jsonl").read_text().splitlines()[-1])
    print(
        f"{tag}: val_ppl={metric['val_ppl']:.4f} "
        f"grad_finite_mean={metric['train_grad_norm_mean']:.4f} "
        f"clip_frac={metric['train_grad_clip_fraction']:.3f} "
        f"nonfinite_frac={metric['train_grad_nonfinite_fraction']:.3f} "
        f"amp_overflow_frac={metric['train_amp_overflow_fraction']:.3f}"
    )
PY
