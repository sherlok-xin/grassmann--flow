#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
EXP="research/experiments/phase2e_teacher_branch_ablation"
mkdir -p "$EXP/logs" "$EXP/raw/smoke"

python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --teacher-run-dir outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint \
  --student-type hybrid_lite --tokenizer-dir ./gpt2_local \
  --student-init-run-dir outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20 \
  --expected-student-init-sha256 9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef \
  --output-dir outputs/distill_experiments --seed 42 --batch-size 32 --epochs 1 \
  --lr 0.0001 --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --amp \
  --log-interval 10 --model-dim 224 --num-layers 6 --num-heads 8 \
  --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 --student-late-k 1 \
  --temperature 2 --kd-loss-mode token_mean --kd-chunk-tokens 1024 \
  --distill-strategy fixed_fused --dataset-name wikitext2 \
  --dataset-path /workspace/grassmannflows/datasets/wikitext2_v1_saved --text-field text \
  --max-seq-len 256 --max-lines 2000 --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 --split-seed 42 --offline --gpu-id 0 \
  --experiment-name phase2e_smoke_wt2_t_seed42 --teacher-alpha-override 1.0 \
  --kd-lambda 5 --notes phase2e_preregistered_reduced_smoke \
  --tags phase2e,smoke,seed42,T \
  >"$EXP/logs/smoke_T_seed42.log" 2>&1

run_dir="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2e_smoke_wt2_t_seed42' -print | sort | tail -n 1)"
python "$EXP/evaluate_smoke_gate.py" --run-dir "$run_dir"
