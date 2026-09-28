#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------
# PTB final batch-size latency suite
#
# Benchmarks three final candidates under multiple batch sizes:
#   1) Hybrid-teacher
#   2) Student-best-224x56-l6-kd005
#   3) Student-tradeoff-192x48-l4-kd005
#
# Batch sizes:
#   - 1   : single-request latency
#   - 8   : small-batch serving
#   - 32  : throughput-oriented comparison
#
# Outputs:
#   - individual json files in outputs/benchmark_reports
#   - one consolidated text summary at the end
# --------------------------------------------------------------------

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

MASTER_LOG="$LOG_DIR/final_batch_latency_suite_${STAMP}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "======================================================================"
echo "PTB FINAL BATCH-SIZE LATENCY SUITE"
echo "======================================================================"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GPU_ID=$GPU_ID"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "BENCH_OUT=$BENCH_OUT"
echo "STAMP=$STAMP"
echo

COMMON_BASE_ARGS=(--dataset-path "$DATASET_PATH" --tokenizer-dir "$TOKENIZER_DIR" --gpu-id "$GPU_ID" --amp --offline)

run_one () {
  local model_name="$1"
  local model_kind="$2"
  local run_dir="$3"
  local batch_size="$4"
  local out_json="$5"

  local warmup=20
  local measure=100

  if [ "$batch_size" -ge 32 ]; then
    warmup=10
    measure=50
  elif [ "$batch_size" -ge 8 ]; then
    warmup=15
    measure=80
  fi

  echo "------------------------------------------------------------------"
  echo "Benchmarking: model=$model_name kind=$model_kind batch_size=$batch_size"
  echo "run_dir=$run_dir"
  echo "out_json=$out_json"
  echo "warmup=$warmup measure=$measure"
  echo "------------------------------------------------------------------"

  python benchmark_ptb_efficiency_v4.py \
    --model-kind "$model_kind" \
    --run-dir "$run_dir" \
    "${COMMON_BASE_ARGS[@]}" \
    --batch-size "$batch_size" \
    --warmup-batches "$warmup" \
    --measure-batches "$measure" \
    --output-json "$out_json"
}

# --------------------------------------------------------------------
# Three final candidates x three batch sizes
# --------------------------------------------------------------------

BATCH_SIZES=(1 8 32)

for BS in "${BATCH_SIZES[@]}"; do
  run_one \
    "Hybrid-teacher" \
    "hybrid_teacher" \
    "outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint" \
    "$BS" \
    "$BENCH_OUT/ptb_final_teacher_bs${BS}.json"

  run_one \
    "Student-best-224x56-l6-kd005" \
    "hybrid_lite" \
    "outputs/distill_experiments/20260402_112236_ptb_hybrid_lite_warmstart_kd005" \
    "$BS" \
    "$BENCH_OUT/ptb_final_student_best_bs${BS}.json"

  run_one \
    "Student-tradeoff-192x48-l4-kd005" \
    "hybrid_lite" \
    "outputs/distill_experiments/20260402_131852_ptb_hybrid_lite_warmstart_192x48_l4_kd005" \
    "$BS" \
    "$BENCH_OUT/ptb_final_student_tradeoff_bs${BS}.json"
done

echo
echo "=== Final batch-size summary ==="
python - <<'PY'
import json
import pathlib

rows = []
specs = [
    ("Hybrid-teacher", "teacher"),
    ("Student-best-224x56-l6-kd005", "student_best"),
    ("Student-tradeoff-192x48-l4-kd005", "student_tradeoff"),
]
name_map = {
    "teacher": "ptb_final_teacher_bs{bs}.json",
    "student_best": "ptb_final_student_best_bs{bs}.json",
    "student_tradeoff": "ptb_final_student_tradeoff_bs{bs}.json",
}

for label, key in specs:
    for bs in [1, 8, 32]:
        p = pathlib.Path("outputs/benchmark_reports") / name_map[key].format(bs=bs)
        if not p.exists():
            rows.append((label, bs, None, None, None, None, None))
            continue
        data = json.loads(p.read_text())
        rows.append((
            label,
            bs,
            data.get("test_ppl"),
            data.get("params_m"),
            data.get("avg_batch_latency_ms"),
            data.get("tokens_per_sec"),
            data.get("peak_memory_gb"),
        ))

print("=" * 118)
print(f"{'model':36s} {'bs':>4s} {'test_ppl':>10s} {'params_m':>10s} {'lat_ms':>10s} {'tok/s':>14s} {'mem_gb':>10s}")
print("=" * 118)
for r in rows:
    name, bs, ppl, pm, lat, tok, mem = r
    def fmt(x, nd=2):
        return "NA" if x is None else f"{x:.{nd}f}"
    print(f"{name:36s} {str(bs):>4s} {fmt(ppl,2):>10s} {fmt(pm,3):>10s} {fmt(lat,3):>10s} {fmt(tok,1):>14s} {fmt(mem,3):>10s}")
print("=" * 118)

print()
print("Relative latency ratios vs teacher:")
teacher = {}
for _, bs, ppl, pm, lat, tok, mem in rows:
    pass
# reload structured
table = {}
for label, key in specs:
    for bs in [1, 8, 32]:
        p = pathlib.Path("outputs/benchmark_reports") / name_map[key].format(bs=bs)
        if p.exists():
            data = json.loads(p.read_text())
            table[(label, bs)] = data
for bs in [1, 8, 32]:
    t = table.get(("Hybrid-teacher", bs))
    b = table.get(("Student-best-224x56-l6-kd005", bs))
    s = table.get(("Student-tradeoff-192x48-l4-kd005", bs))
    print(f"batch_size={bs}")
    if t and b:
        print(f"  best_student latency / teacher latency = {b['avg_batch_latency_ms']/t['avg_batch_latency_ms']:.3f}")
    if t and s:
        print(f"  tradeoff_student latency / teacher latency = {s['avg_batch_latency_ms']/t['avg_batch_latency_ms']:.3f}")
PY

echo
echo "=== DONE ==="
echo "Master log saved to: $MASTER_LOG"
echo "Benchmark jsons saved to: $BENCH_OUT"
