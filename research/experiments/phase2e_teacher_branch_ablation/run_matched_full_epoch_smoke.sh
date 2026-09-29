#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
EXP="research/experiments/phase2e_teacher_branch_ablation"
DATASET="/workspace/grassmannflows/datasets/wikitext2_v1_saved"
TEACHER="outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
S0="outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20"
S0_HASH="9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef"

run_arm() {
  local gpu="$1" label="$2" alpha="$3"
  local experiment_name="phase2e_full_epoch_smoke_${label}_seed42"
  python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
    --teacher-run-dir "$TEACHER" --student-type hybrid_lite --tokenizer-dir ./gpt2_local \
    --student-init-run-dir "$S0" --expected-student-init-sha256 "$S0_HASH" \
    --output-dir outputs/distill_experiments --seed 42 --batch-size 32 --epochs 1 \
    --lr 0.0001 --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --amp \
    --log-interval 50 --model-dim 224 --num-layers 6 --num-heads 8 \
    --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 --student-late-k 1 \
    --temperature 2 --kd-loss-mode token_mean --kd-chunk-tokens 1024 \
    --distill-strategy fixed_fused --dataset-name wikitext2 --dataset-path "$DATASET" \
    --text-field text --max-seq-len 256 --max-lines 0 --encode-chars-per-batch 200000 \
    --tinystories-val-frac 0.02 --split-seed 42 --offline --gpu-id "$gpu" \
    --experiment-name "$experiment_name" --teacher-alpha-override "$alpha" --kd-lambda 5 \
    --notes phase2e_authorized_matched_full_epoch_stability_gate \
    --tags "phase2e,smoke,full_epoch,seed42,${label}" \
    >"$EXP/logs/${experiment_name}.log" 2>&1
}

run_arm 0 F 0.5 & pid_f=$!
run_arm 1 T 1.0 & pid_t=$!
status=0
wait "$pid_f" || status=1
wait "$pid_t" || status=1
if [[ "$status" -ne 0 ]]; then
  printf 'At least one Phase 2E matched full-epoch smoke arm failed.\n' >&2
  exit 1
fi

run_f="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2e_full_epoch_smoke_F_seed42' -print | sort | tail -n 1)"
run_t="$(find outputs/distill_experiments -maxdepth 1 -type d -name '*_phase2e_full_epoch_smoke_T_seed42' -print | sort | tail -n 1)"
python "$EXP/evaluate_matched_smoke_gate.py" \
  --run-f "$run_f" --run-t "$run_t" --authorize-clip-saturation
