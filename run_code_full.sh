#!/usr/bin/env bash
# ============================================================================
# Code Full Pipeline — replicate PTB/WT2/TS workflow on code
#
# Phase 1: Train single-branch baselines (GPU0=g, GPU1=t, parallel)
# Phase 2: Code Teacher = late-fusion of code branches (GPU0)
# Phase 3: Code baseline already done (A=7.67)
# Phase 4: Code warmstart KD with code teacher (GPU0)
#           + Code random init KD with code teacher (GPU1)
# ============================================================================
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/experiments outputs/hybrid_experiments outputs/distill_experiments

DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"

echo "============================================================"
echo " CODE FULL PIPELINE"
echo "============================================================"

# ============================================================================
# Phase 1: Single-branch baselines (GPU 0 & 1, parallel)
# ============================================================================
echo ""
echo "===== Phase 1: Code Grassmann + Transformer baselines ====="

# Code Grassmann (GPU 0, background)
(
CUDA_VISIBLE_DEVICES=0 python train_exp4.py \
  --offline \
  --dataset-path "$DATASET" --text-field text \
  --model grassmann --model-dim 224 --window-sizes 1,2,4 --reduced-dim 56 \
  --epochs 10 --batch-size 16 --lr 3e-4 \
  --max-lines 5000 --max-seq-len 256 \
  --experiment-name code_grassmann_224x56 \
  --tags code,baseline 2>&1 | tee logs/code_grassmann.log
echo "Phase 1a DONE: Code Grassmann"
) &

# Code Transformer (GPU 1, background)
(
CUDA_VISIBLE_DEVICES=1 python train_exp4.py \
  --offline \
  --dataset-path "$DATASET" --text-field text \
  --model transformer --model-dim 224 \
  --epochs 10 --batch-size 16 --lr 3e-4 \
  --max-lines 5000 --max-seq-len 256 \
  --experiment-name code_transformer_224x8 \
  --tags code,baseline 2>&1 | tee logs/code_transformer.log
echo "Phase 1b DONE: Code Transformer"
) &

wait

# Locate checkpoint dirs
G_DIR=$(ls -dt outputs/experiments/*code_grassmann_224x56 2>/dev/null | head -1)
T_DIR=$(ls -dt outputs/experiments/*code_transformer_224x8 2>/dev/null | head -1)

echo "Grassmann dir: $G_DIR"
echo "Transformer dir: $T_DIR"

python3 -c "
import json
g=json.load(open('$G_DIR/summary.json'))['grassmann']
t=json.load(open('$T_DIR/summary.json'))['transformer']
print(f'Code Grassmann:  test_ppl={g[\"test_ppl\"]:.4f}')
print(f'Code Transformer: test_ppl={t[\"test_ppl\"]:.4f}')
"

# ============================================================================
# Phase 2: Code Teacher (GPU 0, from code branches)
# ============================================================================
echo ""
echo "===== Phase 2: Code Teacher (late-fusion, code branches) ====="

python train_hybrid_latefusion_alpha_ddp_v1.py \
  --gpu-id 0 \
  --grassmann-run-dir "$G_DIR" --transformer-run-dir "$T_DIR" \
  --dataset-path "$DATASET" --text-field text --dataset-name ptb \
  --tokenizer-dir ./gpt2_local --output-dir outputs/hybrid_experiments \
  --experiment-name code_teacher_last1_joint \
  --train-mode joint --init-alpha 0.5 --late-k 1 \
  --epochs 10 --batch-size 16 --lr 1e-5 --alpha-lr 1e-2 \
  --max-seq-len 256 --max-lines 5000 --amp --offline 2>&1 | tee logs/code_teacher_full.log

TEACHER_DIR=$(ls -dt outputs/hybrid_experiments/*code_teacher_last1_joint 2>/dev/null | head -1)
echo "Teacher dir: $TEACHER_DIR"
python3 -c "import json; h=json.load(open('${TEACHER_DIR}/summary.json'))['hybrid']; print(f'Code Teacher: test_ppl={h[\"test_ppl\"]:.4f}')"

# ============================================================================
# Phase 3: Code baseline (already done: A=7.67)
# ============================================================================
BASELINE_DIR=$(ls -dt outputs/hybrid_experiments/*code_hybrid_lite_baseline_ce_scratch 2>/dev/null | head -1)
echo ""
echo "===== Phase 3: Code Baseline (already done) ====="
echo "Baseline dir: $BASELINE_DIR"
python3 -c "import json; h=json.load(open('${BASELINE_DIR}/summary.json'))['hybrid']; print(f'Code Baseline: test_ppl={h[\"test_ppl\"]:.4f}')"

# ============================================================================
# Phase 4: Code distillation (GPU 0 & 1, parallel)
# ============================================================================
echo ""
echo "===== Phase 4: Code Distillation (code teacher → code student) ====="

# random init KD (GPU 0)
(
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id 0 --teacher-run-dir "$TEACHER_DIR" --student-type hybrid_lite \
  --dataset-path "$DATASET" --text-field text --tokenizer-dir ./gpt2_local \
  --output-dir outputs/distill_experiments \
  --experiment-name code_random_init_kd005_code_teacher \
  --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
  --student-late-k 1 --epochs 10 --batch-size 16 --lr 1e-4 \
  --temperature 2.0 --distill-alpha 0.05 \
  --max-lines 5000 --max-seq-len 256 --amp --offline 2>&1 | tee logs/code_random_kd_codeteacher.log
echo "Phase 4a DONE"
) &

# warmstart KD with code teacher (GPU 1)
(
python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --gpu-id 1 --teacher-run-dir "$TEACHER_DIR" \
  --student-init-run-dir "$BASELINE_DIR" --student-type hybrid_lite \
  --dataset-path "$DATASET" --text-field text --tokenizer-dir ./gpt2_local \
  --output-dir outputs/distill_experiments \
  --experiment-name code_warmstart_kd002_code_teacher \
  --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
  --student-late-k 1 --epochs 10 --batch-size 16 --lr 1e-4 \
  --temperature 2.0 --distill-alpha 0.02 \
  --max-lines 5000 --max-seq-len 256 --amp --offline 2>&1 | tee logs/code_warmstart_kd_codeteacher.log
echo "Phase 4b DONE"
) &

wait

echo ""
echo "============================================================"
echo " CODE FULL PIPELINE — RESULTS"
echo "============================================================"

python3 << 'PY'
import json, pathlib
base = pathlib.Path(".")

def get_ppl(path, key):
    try:
        return json.loads((base / path / "summary.json").read_text())[key]["test_ppl"]
    except: return None

# Find all code results
print("\nCode Full Pipeline Results:")
print(f"{'Model':<40} {'Test PPL':>10}")
print("-" * 52)

# Single branch
g_dir = sorted(base.glob("outputs/experiments/*code_grassmann_224x56"))[-1]
t_dir = sorted(base.glob("outputs/experiments/*code_transformer_224x8"))[-1]
print(f"  Code Grassmann (from scratch)          {get_ppl(g_dir, 'grassmann'):>10.2f}")
print(f"  Code Transformer (from scratch)        {get_ppl(t_dir, 'transformer'):>10.2f}")

# Teacher
teach_dir = sorted(base.glob("outputs/hybrid_experiments/*code_teacher_last1_joint"))[-1]
print(f"  Code Teacher (late-fusion)             {get_ppl(teach_dir, 'hybrid'):>10.2f}")

# Baseline
bl_dir = sorted(base.glob("outputs/hybrid_experiments/*code_hybrid_lite_baseline_ce_scratch"))[-1]
print(f"  Code Baseline (CE scratch)             {get_ppl(bl_dir, 'hybrid'):>10.2f}")

# Distillation
for tag, lam in [("code_random_init_kd005_code_teacher", "0.05"),
                  ("code_warmstart_kd002_code_teacher", "0.02")]:
    matches = sorted(base.glob(f"outputs/distill_experiments/*{tag}*/summary.json"))
    if matches:
        s = json.loads(matches[-1].read_text())["student"]
        print(f"  {tag:<40} {s['test_ppl']:>10.2f}")
    else:
        print(f"  {tag:<40} {'missing':>10}")

# Cross reference with old PTB-based results
print(f"\n  -- PTB-based (cross-domain, for reference) --")
print(f"  code_warmstart_ce_only (PTB baseline)      {4.17:>10.2f}")
print(f"  code_warmstart_kd002 (PTB baseline+teacher) {9.10:>10.2f}")
PY
