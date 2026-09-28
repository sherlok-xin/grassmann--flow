#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"

cd "$PROJECT_ROOT"
mkdir -p logs

echo "=== Enable Grassmann CUDA extension ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("cuda version:", torch.version.cuda)
PY

echo
echo "=== Build extension from src/cuda/setup.py ==="
cd src/cuda
python setup.py install 2>&1 | tee ../../logs/build_grassmann_cuda.log
cd ../..

echo
echo "=== Verify import and CUDA path ==="
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src/cuda")
import grassmann_fused
print("grassmann_fused.CUDA_AVAILABLE =", getattr(grassmann_fused, "CUDA_AVAILABLE", None))
print("grassmann_cuda module =", getattr(grassmann_fused, "grassmann_cuda", None))
PY

echo
echo "=== Run repo kernel tests ==="
python test_cuda_kernels.py 2>&1 | tee logs/test_cuda_kernels.log || true

echo
echo "=== Run repo CUDA micro-benchmark ==="
python benchmark_cuda.py 2>&1 | tee logs/benchmark_cuda.log || true

echo
echo "=== Important note ==="
echo "The current PTB training/evaluation models only use the CUDA kernels if the model code imports grassmann_fused/FusedGrassmannMixing."
echo "If your grassmann_v4.py is still pure PyTorch, building the extension alone will NOT accelerate the end-to-end PTB model benchmark."
