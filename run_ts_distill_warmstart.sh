#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER=$(python3 -c "import pathlib; m=sorted(pathlib.Path('outputs/hybrid_experiments').glob('*ts_latefusion_last1_joint_e10')); print(m[-1])")
BASELINE=$(python3 -c "import pathlib; m=sorted(pathlib.Path('outputs/hybrid_experiments').glob('*ts_hybrid_lite_baseline_224x56_l6_e20')); print(m[-1])")
echo "Teacher: $TEACHER"
echo "Baseline: $BASELINE"

run_one() {
  local gpu=$1 tag=$2 lam=$3
  echo "[GPU$gpu] lambda=$lam $tag"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id $gpu --teacher-run-dir "$TEACHER" \
    --student-init-run-dir "$BASELINE" --student-type hybrid_lite \
    --dataset-name tinystories --dataset-path /workspace/grassmannflows/datasets/tinystories_saved \
    --text-field text --tokenizer-dir ./gpt2_local \
    --output-dir outputs/distill_experiments --experiment-name "$tag" \
    --model-dim 224 --num-layers 6 --reduced-dim 56 --window-sizes 1,2,4 \
    --student-late-k 1 --epochs 10 --batch-size 32 --lr 1e-4 \
    --temperature 2.0 --distill-alpha "$lam" \
    --max-lines 300000 --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 --split-seed 42 \
    --amp --offline
  echo "[GPU$gpu] DONE: $tag"
}

# Phase 1: 3 experiments in parallel on GPU 0,1,2
run_one 0 ts_warmstart_ce_only    0.0  &
run_one 1 ts_warmstart_lambda_20  0.02 &
run_one 2 ts_warmstart_lambda_50  0.05 &
wait

# Phase 2: remaining lambdas
for lam in 0.01 0.03 0.04; do
  lam3=$(python3 -c "print(int($lam*1000))")
  run_one 0 "ts_warmstart_lambda_${lam3}" "$lam"
done

echo "===== ALL DONE ====="
python3 << 'PY'
import json, pathlib
base = pathlib.Path("outputs/distill_experiments")
print("\nTS Distillation Full Results:")
print(f"{'Experiment':<35} {'λ':>6} {'Test PPL':>10}")
print("-" * 54)
for p in sorted(base.glob("*ts_*/summary.json")):
    s = json.loads(p.read_text())
    if "student" not in s: continue
    r = s["student"]
    cfg = json.loads((p.parent / "config.json").read_text())
    lam = cfg.get("config", cfg).get("distill_alpha", "?")
    print(f"  {p.parent.name:<35} {lam:>6} {r['test_ppl']:>10.4f}")
PY
