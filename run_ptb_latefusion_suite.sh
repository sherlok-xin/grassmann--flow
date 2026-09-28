#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
ROOT_DIR="${ROOT_DIR:-/workspace/grassmannflows/grassmann-flows}"
TRAINER="${TRAINER:-train_hybrid_latefusion_alpha_ddp_v1.py}"

cd "${ROOT_DIR}"
mkdir -p logs outputs/hybrid_experiments

GRASSMANN_RUN="outputs/experiments/20260318_102137_ptb_ws124_rd64_md240"
TRANSFORMER_RUN="outputs/experiments/20260318_094720_ptb_baseline_both"
DATASET_PATH="/workspace/grassmannflows/datasets/ptb_text_only_saved"
TOKENIZER_DIR="./gpt2_local"
OUTPUT_DIR="outputs/hybrid_experiments"

echo "[1/3] PTB late-fusion last1 alpha_only"
python "${TRAINER}" \
  --gpu-id "${GPU_ID}" \
  --grassmann-run-dir "${GRASSMANN_RUN}" \
  --transformer-run-dir "${TRANSFORMER_RUN}" \
  --dataset-path "${DATASET_PATH}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name ptb_latefusion_last1_alpha_only \
  --train-mode alpha_only \
  --late-k 1 \
  --init-alpha 0.5 \
  --epochs 5 \
  --batch-size 32 \
  --alpha-lr 1e-2 \
  --amp \
  --offline 2>&1 | tee "logs/ptb_latefusion_last1_alpha_only.log"

echo "[2/3] PTB late-fusion last1 joint"
python "${TRAINER}" \
  --gpu-id "${GPU_ID}" \
  --grassmann-run-dir "${GRASSMANN_RUN}" \
  --transformer-run-dir "${TRANSFORMER_RUN}" \
  --dataset-path "${DATASET_PATH}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name ptb_latefusion_last1_joint \
  --train-mode joint \
  --freeze-branch-epochs 1 \
  --late-k 1 \
  --init-alpha 0.5 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-5 \
  --alpha-lr 5e-3 \
  --amp \
  --offline 2>&1 | tee "logs/ptb_latefusion_last1_joint.log"

echo "[3/3] PTB late-fusion last2 joint"
python "${TRAINER}" \
  --gpu-id "${GPU_ID}" \
  --grassmann-run-dir "${GRASSMANN_RUN}" \
  --transformer-run-dir "${TRANSFORMER_RUN}" \
  --dataset-path "${DATASET_PATH}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name ptb_latefusion_last2_joint \
  --train-mode joint \
  --freeze-branch-epochs 1 \
  --late-k 2 \
  --init-alpha 0.5 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-5 \
  --alpha-lr 5e-3 \
  --amp \
  --offline 2>&1 | tee "logs/ptb_latefusion_last2_joint.log"

echo
echo "=== PTB late-fusion suite summary ==="
python - <<'PY'
import json
from pathlib import Path

tags = [
    "ptb_latefusion_last1_alpha_only",
    "ptb_latefusion_last1_joint",
    "ptb_latefusion_last2_joint",
]

root = Path("outputs/hybrid_experiments")
for tag in tags:
    matches = sorted(root.glob(f"*{tag}/summary.json"))
    if not matches:
        print(f"{tag}: summary.json not found")
        continue
    p = matches[-1]
    data = json.loads(p.read_text())["hybrid"]
    print("=" * 80)
    print(p)
    print("best_val_ppl:", data.get("best_val_ppl"))
    print("test_ppl:", data.get("test_ppl"))
    print("best_alpha(mean):", data.get("best_alpha"))
    print("best_alpha_vector:", data.get("best_alpha_vector"))
PY
