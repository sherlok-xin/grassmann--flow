# Phase 2F Provenance

Status: complete; Phase 2G not launched.

## Start state

- User-provided and verified repository HEAD: `411b7a19e7e0cab393286bedbee4626347bd2587`
- Execution host: `10.42.0.197`, accessed through `~/fuwuqi/agent-tools/exec_grassmann.sh`
- Project path in the execution container: `/workspace/grassmannflows/grassmann-flows`
- Heavy training is remote-only.

## Data recovery

The exact historical Hugging Face Arrow cache for `wikitext-2-raw-v1`, revision directory `b08601e04326c79dfdd32d625aee71d232d685c3`, was recovered from a March 2026 local attachment cache and copied to the shared Hugging Face cache under `/workspace/grassmannflows/hub`. The protected project `datasets/` directory was not modified.

Arrow SHA256 values:

- train: `57947bc7b58df4b19662c0609cc30651bc84328dab5fd588860b752072911789`
- validation: `8e136ab7130c5fca602eb406ce2557967a974d901b53c4f2992f87b873e0fe5b`
- test: `2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea`

Using the original `train_exp.py` preprocessing and repository `gpt2_local` tokenizer reproduces the T1 config statistics exactly: token counts 2,391,884/247,289/283,287 and chunk counts 9,343/965/1,106 for train/validation/test.

## Compatibility implementation

`train_hybrid_latefusion_alpha_ddp_v1.py` and `train_distill_hybrid_lite_from_latefusion_teacher_v2.py` accept explicit `teacher_type=tg|tt`. The default remains `tg`; the historical TG class and state-dict names remain unchanged. TT checkpoints use explicit `transformer1` and `transformer2` branches.

Remote tests passed before smoke: three Phase 2F topology/checkpoint tests plus four existing Phase 2B teacher-control tests. The historical TG checkpoint loaded successfully with alpha 0.506482 and `late_k=1` after the compatibility change.

## Known preregistered mismatch

The user-authorized TT protocol fixes alpha learning rate at `1e-2`; the actual historical TG Teacher J used `5e-3`. This difference is frozen before endpoints and will be reported as a possible teacher-quality/training confound.

## T2 training

- Smoke: `outputs/experiments/20260930_033109_phase2f_t2_seed123_smoke_e1`
- Formal: `outputs/experiments/20260930_033237_phase2f_t2_seed123_e20`
- Parameters: 17,670,400
- Validation-selected epoch: 12
- Validation/test NLL: 5.230674231731830 / 5.288460377974277
- Test PPL: 198.038269
- Checkpoint SHA256: `d56b566023118a1089ee908cde6822f5e9b0bcd961aa1d6d3da8a2f776aad442`

For reference, T1 has 17,670,400 parameters, selected epoch 12, validation NLL 5.238534310197583, and test NLL 5.286473298374610.

## TT teacher training and evaluation

- Reduced smoke: `outputs/hybrid_experiments/20260930_034443_phase2f_tt_teacher_smoke`
- Formal: `outputs/hybrid_experiments/20260930_034520_phase2f_tt_teacher_joint_e10`
- Formal parameters: 35,340,801
- Validation-selected epoch: 10
- Training summary validation/test NLL: 4.293405053112750 / 4.194386502440167
- Learned checkpoint alpha: 0.5117690563201904
- Checkpoint SHA256: `870d2667f0c5169fb42498a7195551fdc92f5bdf3fd41869d2167998c964e6f9`

The final teacher comparison uses full validation inference, float evaluation, and forced alpha 0.5 for both teachers. The exact output is `raw/teacher_diagnostics.json`. TT/TG validation NLL is 4.293589157793147/4.303697201842470. TT has 2,408,832 fewer parameters and is better by 0.010108044049323 NLL. The teachers are not parameter-matched.

## KD stability gates and formal runs

- Reduced TT KD smoke: `outputs/distill_experiments/20260930_035317_phase2f_tt_kd_smoke_seed42`
- Matched full-data one-epoch gate: `outputs/distill_experiments/20260930_035418_phase2f_tt_kd_full_gate_seed42`
- Gate clipping fraction: 0.5408163265306123
- Gate AMP overflow/non-finite fractions: 0.013605442176870748 / 0.013605442176870748
- Formal seed 42: `outputs/distill_experiments/20260930_035736_phase2f_wt2_tt_seed42`
- Formal seed 123: `outputs/distill_experiments/20260930_035736_phase2f_wt2_tt_seed123`
- Formal seed 456: `outputs/distill_experiments/20260930_035736_phase2f_wt2_tt_seed456`

Formal validation-selected TT checkpoint SHA256 values are:

- seed 42: `5299fab6563588d32fe39f669c3d635c314a8b3488510f5cc529dc3a8f874fa7`
- seed 123: `ef9e5394d276485decab9aa9ba80d707d1a7610e5186657e057bcc01612637f9`
- seed 456: `b37392ee399bdd2944ee4aeac97cc6d6c363216ff1993a81dfae36491b494590`

All formal arms selected epoch 10. Maximum AMP overflow and non-finite fractions are 0.013605 for every seed. Epoch-1 clipping fractions are 0.928571/0.843537/0.874150 and decline to 0.003401. There is no NaN, training failure, or clip saturation.

## Endpoint reconstruction

The strict collector validates the three distinct S0 hashes, frozen TT configurations, teacher checkpoint identity, validation selection, and formal completion before writing `results_multiseed.csv` and `raw/results_summary.json`. C0 and TG endpoints are reused from Phase 2C/2E and are not rerun.

The primary `H` values for seeds 42/123/456 are -0.013952082274556/-0.011329102915994/-0.012099307296766. Mean H is -0.012460164162439 with sample SD 0.001348209344645 and 3/3 negative signs.

## Repository chronology

- Starting HEAD: `411b7a19e7e0cab393286bedbee4626347bd2587`
- Compatibility/test commit: `6217783`
- Evaluation/collection tooling commit: `3180cfc`
- Final report/artifact commit: recorded in the Git history after completion

No file under project `datasets/` or `checkpoints/` was modified. No historical output or checkpoint was overwritten.
