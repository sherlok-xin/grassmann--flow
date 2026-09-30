# Phase 2F Provenance

Status: protocol frozen; formal endpoints not yet generated.

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
