#!/usr/bin/env bash
# ============================================================================
# Code Full Distillation + Subspace Analysis
#
# Phase D: Lambda scan (0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1)
#           on 3 GPUs: warmstart KD + random init KD + warmstart CE
# Phase S: Subspace analysis (Plucker SVD spectrum)
#           on code CE scratch vs warmstart KD
# ============================================================================
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments outputs/subspace_analysis

TEACHER=$(python3 -c "
import pathlib
candidates = sorted(pathlib.Path('outputs/hybrid_experiments').glob('*code_teacher_last1_joint*'))
for d in reversed(candidates):
    if (d / 'summary.json').exists():
        print(d)
        break
")
BASELINE=$(python3 -c "
import pathlib
candidates = sorted(pathlib.Path('outputs/hybrid_experiments').glob('*code_hybrid_lite_baseline_ce_scratch*'))
# find the one with a summary
for d in reversed(candidates):
    if (d / 'summary.json').exists():
        print(d)
        break
")
DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"

echo "============================================================"
echo " CODE FULL DISTILLATION + SUBSPACE"
echo " Teacher:  $TEACHER"
echo " Baseline: $BASELINE"
echo "============================================================"

TEACHER_PPL=$(python3 -c "import json; print(json.load(open('${TEACHER}/summary.json'))['hybrid']['test_ppl'])")
BL_PPL=$(python3 -c "import json; print(json.load(open('${BASELINE}/summary.json'))['hybrid']['test_ppl'])")
echo "  Teacher test_ppl:  $TEACHER_PPL"
echo "  Baseline test_ppl: $BL_PPL"

# ============================================================================
# Phase D1: Parallel batch — 3 GPUs
#   GPU0: warmstart CE-only (λ=0.0)
#   GPU1: random init KD (λ=0.05)
#   GPU2: warmstart KD λ=0.02
# ============================================================================
echo ""
echo "===== Phase D1: 3 experiments in parallel ====="

run_distill() {
  local gpu=$1 tag=$2 lam=$3 warm=$4
  echo "[GPU$gpu] $tag λ=$lam"

  local init=""
  [[ "$warm" == "yes" ]] && init="--student-init-run-dir $BASELINE"

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "$gpu" --teacher-run-dir "$TEACHER" $init --student-type hybrid_lite \
    --dataset-path "$DATASET" --text-field text --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments --experiment-name "$tag" \
    --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
    --student-late-k 1 --epochs 10 --batch-size 16 --lr 1e-4 \
    --temperature 2.0 --distill-alpha "$lam" \
    --max-lines 5000 --max-seq-len 256 --amp --offline 2>&1 | tee "logs/${tag}.log"

  D=$(ls -dt outputs/distill_experiments/*_${tag} 2>/dev/null | head -1)
  [[ -f "$D/summary.json" ]] && python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'$tag: test_ppl={s[\"test_ppl\"]:.4f}')"
  echo "[GPU$gpu] DONE: $tag"
}

run_distill 0 code_ws_ce_only  0.0  yes &
run_distill 1 code_ri_kd005    0.05 no  &
run_distill 2 code_ws_kd002    0.02 yes &

wait

echo ""
echo "===== Phase D2: Remaining lambdas (GPU 0 & 1, sequential) ====="

for lam in 0.01 0.03 0.04 0.05 0.1; do
  lam3=$(python3 -c "print(int($lam*1000))")
  TAG="code_ws_kd${lam3}"
  echo "--- λ=$lam $TAG ---"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id 0 --teacher-run-dir "$TEACHER" --student-init-run-dir "$BASELINE" \
    --student-type hybrid_lite --dataset-path "$DATASET" --text-field text \
    --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments \
    --experiment-name "$TAG" --model-dim 224 --num-layers 6 --reduced-dim 56 \
    --window-sizes 1,2,4 --student-late-k 1 --epochs 10 --batch-size 16 \
    --lr 1e-4 --temperature 2.0 --distill-alpha "$lam" \
    --max-lines 5000 --max-seq-len 256 --amp --offline 2>&1 | tee "logs/${TAG}.log"

  D=$(ls -dt outputs/distill_experiments/*_${TAG} 2>/dev/null | head -1)
  [[ -f "$D/summary.json" ]] && python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'$TAG: test_ppl={s[\"test_ppl\"]:.4f}')"
done

echo ""
echo "===== ALL DISTILL DONE ====="

# ============================================================================
# Summary
# ============================================================================
python3 << 'PY'
import json, pathlib

print("\n" + "=" * 75)
print("  CODE FULL RESULTS")
print("=" * 75)

# Single branches
for label, glob_pat, key in [
    ("Code Grassmann", "outputs/experiments/*code_grassmann_224x56*", "grassmann"),
    ("Code Transformer", "outputs/experiments/*code_transformer_224x8*", "transformer"),
    ("Code Teacher", "outputs/hybrid_experiments/*code_teacher_last1_joint*", "hybrid"),
    ("Code Baseline", "outputs/hybrid_experiments/*code_hybrid_lite_baseline_ce_scratch*", "hybrid"),
]:
    matches = sorted(pathlib.Path(glob_pat).parent.glob(pathlib.Path(glob_pat).name))
    if matches:
        s = json.loads((matches[-1] / "summary.json").read_text())[key]
        print(f"  {label:<30} test_ppl={s['test_ppl']:>7.4f}")

print()
print("  -- Distillation --")
for p in sorted(pathlib.Path("outputs/distill_experiments").glob("*code_*/summary.json")):
    d = json.loads(p.read_text())
    if "student" not in d: continue
    s = d["student"]
    cfg = json.loads((p.parent / "config.json").read_text())
    lam = cfg.get("config", cfg).get("distill_alpha", "?")
    print(f"  {p.parent.name:<42} test_ppl={s['test_ppl']:>7.4f}  λ={lam}")

print()
print("  -- Cross-dataset comparison --")
print("  PTB:   λ=0.02 → 51.27 (best)")
print("  WT2:   λ=0.02 → 60.83 (best)")
print("  TS:    CE-only → 4.86 (best, KD harmful)")
print("  Code:  ?")
PY
