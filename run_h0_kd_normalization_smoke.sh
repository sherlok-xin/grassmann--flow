#!/usr/bin/env bash
set -euo pipefail

cd /workspace/grassmannflows/grassmann-flows

TEACHER_RUN="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint"
STUDENT_INIT="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20"
DATASET_PATH="/workspace/grassmannflows/datasets/ptb_text_only_saved"
OUTPUT_ROOT="outputs/distill_experiments"

mkdir -p logs "${OUTPUT_ROOT}"

for required in \
  "${TEACHER_RUN}/summary.json" \
  "${STUDENT_INIT}/summary.json" \
  "${DATASET_PATH}" \
  "gpt2_local"; do
  if [[ ! -e "${required}" ]]; then
    echo "[error] missing required path: ${required}" >&2
    exit 1
  fi
done

run_one() {
  local gpu_id="$1"
  local tag="$2"
  local loss_mode="$3"
  local alpha="$4"
  local kd_lambda="$5"

  if find "${OUTPUT_ROOT}" -maxdepth 2 -path "*_${tag}/summary.json" -print -quit | grep -q .; then
    echo "[skip] completed result already exists for tag=${tag}"
    return 0
  fi

  echo "[start] gpu=${gpu_id} tag=${tag} mode=${loss_mode} alpha=${alpha} lambda=${kd_lambda}"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "${gpu_id}" \
    --teacher-run-dir "${TEACHER_RUN}" \
    --student-init-run-dir "${STUDENT_INIT}" \
    --student-type hybrid_lite \
    --tokenizer-dir ./gpt2_local \
    --output-dir "${OUTPUT_ROOT}" \
    --experiment-name "${tag}" \
    --notes "H0 one-epoch KD normalization smoke; not a final performance result" \
    --tags "h0,smoke,ptb,kd-normalization" \
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
  echo "[done] gpu=${gpu_id} tag=${tag}"
}

run_one 1 h0_ptb_smoke_legacy_a002_seed42 legacy_batchmean 0.02 1.0 &
pid_legacy=$!
run_one 2 h0_ptb_smoke_token_l0_seed42 token_mean 0.0 0.0 &
pid_l0=$!
run_one 3 h0_ptb_smoke_token_l2p5_seed42 token_mean 0.0 2.5 &
pid_l2p5=$!

wait "${pid_legacy}"
wait "${pid_l0}"
wait "${pid_l2p5}"

run_one 1 h0_ptb_smoke_token_l5_seed42 token_mean 0.0 5.0 &
pid_l5=$!
run_one 2 h0_ptb_smoke_token_l10_seed42 token_mean 0.0 10.0 &
pid_l10=$!

wait "${pid_l5}"
wait "${pid_l10}"

python3 - <<'PY'
import json
from pathlib import Path

tags = [
    "h0_ptb_smoke_legacy_a002_seed42",
    "h0_ptb_smoke_token_l0_seed42",
    "h0_ptb_smoke_token_l2p5_seed42",
    "h0_ptb_smoke_token_l5_seed42",
    "h0_ptb_smoke_token_l10_seed42",
]
print("H0 KD normalization smoke summary")
for tag in tags:
    matches = sorted(Path("outputs/distill_experiments").glob(f"*_{tag}"))
    completed = [path for path in matches if (path / "summary.json").is_file()]
    if not completed:
        print(f"{tag}: MISSING")
        continue
    run_dir = completed[-1]
    summary = json.loads((run_dir / "summary.json").read_text())["student"]
    metric = json.loads((run_dir / "distill_metrics.jsonl").read_text().splitlines()[-1])
    print(
        f"{tag}: val_ppl={summary['best_val_ppl']:.4f} "
        f"test_ppl={summary['test_ppl']:.4f} "
        f"loss={metric['train_loss']:.4f} ce={metric['train_ce']:.4f} "
        f"kl_batch={metric['train_kl_batchmean']:.4f} "
        f"kl_token={metric['train_kl_token_mean']:.6f} "
        f"grad={metric['train_grad_norm_mean']:.4f} "
        f"clip_frac={metric['train_grad_clip_fraction']:.3f}"
    )
PY
