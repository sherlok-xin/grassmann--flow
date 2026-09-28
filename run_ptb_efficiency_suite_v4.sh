#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/grassmannflows/grassmann-flows}"
GPU_ID="${GPU_ID:-1}"
DATASET_PATH="${DATASET_PATH:-/workspace/grassmannflows/datasets/ptb_text_only_saved}"
OUT_DIR="${OUT_DIR:-outputs/benchmark_reports}"
TOKENIZER_DIR="${TOKENIZER_DIR:-./gpt2_local}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR" logs

echo "=== PTB efficiency benchmark suite ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUT_DIR=$OUT_DIR"

COMMON_ARGS=(--dataset-path "$DATASET_PATH" --tokenizer-dir "$TOKENIZER_DIR" --gpu-id "$GPU_ID" --batch-size 32 --warmup-batches 10 --measure-batches 50 --amp --offline)

python benchmark_ptb_efficiency_v4.py --model-kind grassmann --run-dir outputs/experiments/20260318_102137_ptb_ws124_rd64_md240 "${COMMON_ARGS[@]}" --output-json "$OUT_DIR/ptb_grassmann_big.json" 2>&1 | tee logs/bench_ptb_grassmann_big.log

python benchmark_ptb_efficiency_v4.py --model-kind transformer --run-dir outputs/experiments/20260318_094720_ptb_baseline_both "${COMMON_ARGS[@]}" --output-json "$OUT_DIR/ptb_transformer_big.json" 2>&1 | tee logs/bench_ptb_transformer_big.log

python benchmark_ptb_efficiency_v4.py --model-kind hybrid_teacher --run-dir outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint "${COMMON_ARGS[@]}" --output-json "$OUT_DIR/ptb_hybrid_teacher.json" 2>&1 | tee logs/bench_ptb_hybrid_teacher.log

python benchmark_ptb_efficiency_v4.py --model-kind hybrid_lite --run-dir outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20 "${COMMON_ARGS[@]}" --output-json "$OUT_DIR/ptb_hybrid_lite_baseline.json" 2>&1 | tee logs/bench_ptb_hybrid_lite_baseline.log

python benchmark_ptb_efficiency_v4.py --model-kind hybrid_lite --run-dir outputs/distill_experiments/20260329_112627_ptb_hybrid_lite_warmstart_kd01 "${COMMON_ARGS[@]}" --output-json "$OUT_DIR/ptb_hybrid_lite_warmstart_kd01.json" 2>&1 | tee logs/bench_ptb_hybrid_lite_warmstart_kd01.log

python - <<'PY'
import json, pathlib
files = [
    ("Grassmann-big", "outputs/benchmark_reports/ptb_grassmann_big.json"),
    ("Transformer-big", "outputs/benchmark_reports/ptb_transformer_big.json"),
    ("Hybrid-teacher", "outputs/benchmark_reports/ptb_hybrid_teacher.json"),
    ("Hybrid-lite-baseline", "outputs/benchmark_reports/ptb_hybrid_lite_baseline.json"),
    ("Hybrid-lite-warmstart-kd01", "outputs/benchmark_reports/ptb_hybrid_lite_warmstart_kd01.json"),
]
print("=== PTB efficiency summary ===")
for name, fp in files:
    p = pathlib.Path(fp)
    if not p.exists():
        print(f"[missing] {name}: {fp}")
        continue
    data = json.loads(p.read_text())
    print("=" * 80)
    print(name)
    print("test_ppl:", data.get("test_ppl"))
    print("params_m:", round(data.get("params_m", 0), 3))
    print("avg_batch_latency_ms:", round(data.get("avg_batch_latency_ms", 0), 3))
    print("tokens_per_sec:", round(data.get("tokens_per_sec", 0), 1))
    print("peak_memory_gb:", round(data.get("peak_memory_gb", 0), 3))
PY
