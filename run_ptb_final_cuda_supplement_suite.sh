#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
TOKENIZER_DIR="${TOKENIZER_DIR:-./gpt2_local}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
BENCH_OUT="${BENCH_OUT:-outputs/benchmark_reports}"
LOG_DIR="${LOG_DIR:-logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$BENCH_OUT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

MASTER_LOG="$LOG_DIR/final_cuda_suite_${STAMP}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "======================================================================"
echo "PTB FINAL CUDA SUPPLEMENT SUITE"
echo "======================================================================"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "BENCH_OUT=$BENCH_OUT"
echo "STAMP=$STAMP"
echo

echo "=== Step 0: environment snapshot ==="
python - <<'PY'
import os, torch, platform, sys
print("python:", sys.version)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
echo

echo "=== Step 1: build/install repo CUDA extension ==="
pushd src/cuda >/dev/null
python setup.py install
popd >/dev/null
echo

echo "=== Step 2: verify CUDA extension import ==="
python - <<'PY'
import sys
sys.path.insert(0, "src/cuda")
import grassmann_fused
print("grassmann_fused.CUDA_AVAILABLE =", grassmann_fused.CUDA_AVAILABLE)
try:
    import grassmann_cuda
    print("grassmann_cuda import: OK")
except Exception as e:
    print("grassmann_cuda import failed:", repr(e))
PY
echo

echo "=== Step 3: run repo kernel tests ==="
python test_cuda_kernels.py || true
echo

echo "=== Step 4: run repo CUDA micro-benchmark ==="
python benchmark_cuda.py || true
echo

echo "=== Step 5: benchmark final PTB candidates on current end-to-end path ==="
COMMON_ARGS=(--dataset-path "$DATASET_PATH" --tokenizer-dir "$TOKENIZER_DIR" --gpu-id "$GPU_ID" --batch-size 32 --warmup-batches 10 --measure-batches 50 --amp --offline)

python benchmark_ptb_efficiency_v4.py \
  --model-kind hybrid_teacher \
  --run-dir outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint \
  "${COMMON_ARGS[@]}" \
  --output-json "$BENCH_OUT/ptb_hybrid_teacher_final.json"

python benchmark_ptb_efficiency_v4.py \
  --model-kind hybrid_lite \
  --run-dir outputs/distill_experiments/20260402_112236_ptb_hybrid_lite_warmstart_kd005 \
  "${COMMON_ARGS[@]}" \
  --output-json "$BENCH_OUT/ptb_student_best_224x56_l6_kd005_final.json"

python benchmark_ptb_efficiency_v4.py \
  --model-kind hybrid_lite \
  --run-dir outputs/distill_experiments/20260402_131852_ptb_hybrid_lite_warmstart_192x48_l4_kd005 \
  "${COMMON_ARGS[@]}" \
  --output-json "$BENCH_OUT/ptb_student_tradeoff_192x48_l4_kd005_final.json"

echo
echo "=== Step 6: final summary table ==="
python - <<'PY'
import json, pathlib

files = [
    ("Hybrid-teacher", "outputs/benchmark_reports/ptb_hybrid_teacher_final.json"),
    ("Student-best-224x56-l6-kd005", "outputs/benchmark_reports/ptb_student_best_224x56_l6_kd005_final.json"),
    ("Student-tradeoff-192x48-l4-kd005", "outputs/benchmark_reports/ptb_student_tradeoff_192x48_l4_kd005_final.json"),
]

rows = []
for name, fp in files:
    p = pathlib.Path(fp)
    if not p.exists():
        rows.append((name, None, None, None, None, None))
        continue
    data = json.loads(p.read_text())
    rows.append((
        name,
        data.get("test_ppl"),
        data.get("params_m"),
        data.get("avg_batch_latency_ms"),
        data.get("tokens_per_sec"),
        data.get("peak_memory_gb"),
    ))

print("=" * 96)
print(f"{'model':36s} {'test_ppl':>10s} {'params_m':>10s} {'lat_ms':>10s} {'tok/s':>12s} {'mem_gb':>10s}")
print("=" * 96)
for r in rows:
    name, ppl, pm, lat, tok, mem = r
    def fmt(x, nd=2):
        return "NA" if x is None else f"{x:.{nd}f}"
    print(f"{name:36s} {fmt(ppl,2):>10s} {fmt(pm,3):>10s} {fmt(lat,3):>10s} {fmt(tok,1):>12s} {fmt(mem,3):>10s}")
print("=" * 96)
PY

echo
echo "=== DONE ==="
echo "Master log saved to: $MASTER_LOG"
echo "Benchmark jsons saved to: $BENCH_OUT"
echo "You can leave this running overnight; all outputs are already tee'd to disk."
