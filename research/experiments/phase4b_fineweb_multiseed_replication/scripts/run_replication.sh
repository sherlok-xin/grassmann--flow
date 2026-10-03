#!/usr/bin/env bash
set -euo pipefail
phase="research/experiments/phase4b_fineweb_multiseed_replication"
out="outputs/phase4b_fineweb"
export PYTHONPATH="outputs/phase4a_modern/python_deps"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$out/runs/123" "$out/runs/456"
run_arm() {
    local seed="$1" arm="$2" gpu="$3"
    CUDA_VISIBLE_DEVICES="$gpu" python "$phase/scripts/train.py" fineweb "$arm" --seed "$seed" > "$out/runs/$seed/$arm.log" 2>&1 &
    local task_pid=$!
    printf '%s %s %s %s\n' "$task_pid" "$seed" "$arm" "$gpu" >> "$out/process_registry.txt"
    wait "$task_pid"
}
run_arm 123 s0 0 &
prep123=$!
run_arm 456 s0 1 &
prep456=$!
failed=0
wait "$prep123" || failed=1
wait "$prep456" || failed=1
[[ "$failed" == 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=0 python "$phase/scripts/init_audit.py" fineweb --seed 123 > "$out/runs/123/initialization_audit.log" 2>&1
CUDA_VISIBLE_DEVICES=1 python "$phase/scripts/init_audit.py" fineweb --seed 456 > "$out/runs/456/initialization_audit.log" 2>&1
# Each worker holds one GPU, and its second arm starts only after the first exits.
(run_arm 123 ce 0 && run_arm 456 kd1 0) &
worker0=$!
(run_arm 123 kd1 1 && run_arm 456 kd5 1) &
worker1=$!
run_arm 123 kd5 2 &
worker2=$!
run_arm 456 ce 3 &
worker3=$!
failed=0
wait "$worker0" || failed=1
wait "$worker1" || failed=1
wait "$worker2" || failed=1
wait "$worker3" || failed=1
[[ "$failed" == 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=0 python "$phase/scripts/finish.py" > "$out/final_evaluation.log" 2>&1
printf 'PHASE4B_TRAINING_AND_TEST_COMPLETE\n'
