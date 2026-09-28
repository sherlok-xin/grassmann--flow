#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grassmannflows/grassmann-flows
mkdir -p logs outputs/distill_experiments

TEACHER="outputs/hybrid_experiments/20260529_120718_code_teacher_last1_joint"
BASELINE="outputs/hybrid_experiments/20260527_074518_code_hybrid_lite_baseline_ce_scratch"
DATASET="/workspace/grassmannflows/datasets/codeparrot_saved"

echo "===== Code Remain: λ=0.03, 0.04, 0.1 ====="

run_one() {
  local gpu=$1 lam=$2
  local lam3=$(python3 -c "print(int($lam*1000))")
  local tag="code_ws_kd${lam3}"
  echo "[GPU$gpu] λ=$lam $tag"

  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --gpu-id "$gpu" --teacher-run-dir "$TEACHER" --student-init-run-dir "$BASELINE" \
    --student-type hybrid_lite --dataset-path "$DATASET" --text-field text \
    --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments \
    --experiment-name "$tag" --model-dim 224 --num-layers 6 --reduced-dim 56 \
    --window-sizes 1,2,4 --student-late-k 1 --epochs 10 --batch-size 16 \
    --lr 1e-4 --temperature 2.0 --distill-alpha "$lam" \
    --max-lines 5000 --max-seq-len 256 --amp --offline 2>&1 | tee "logs/${tag}.log"

  D=$(ls -dt outputs/distill_experiments/*_${tag} 2>/dev/null | head -1)
  [[ -f "$D/summary.json" ]] && python3 -c "import json; s=json.load(open('$D/summary.json'))['student']; print(f'$tag: test_ppl={s[\"test_ppl\"]:.4f}')"
  echo "[GPU$gpu] DONE: $tag"
}

run_one 0 0.03 &
run_one 1 0.04 &
run_one 2 0.1  &

wait

echo "===== ALL DONE ====="
python3 << 'PY'
import json, pathlib
print("\nCode Distill Full Results:")
for s in sorted(pathlib.Path("outputs/distill_experiments").glob("*code_ws_kd*/summary.json")):
    d = json.loads(s.read_text())
    if "student" not in d: continue
    r = d["student"]
    cfg = json.loads((s.parent / "config.json").read_text())
    lam = cfg.get("config", cfg).get("distill_alpha", "?")
    print(f"  λ={lam:<5} test_ppl={r['test_ppl']:.4f}")
PY
