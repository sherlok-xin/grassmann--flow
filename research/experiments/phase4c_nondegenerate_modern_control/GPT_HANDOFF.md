# Phase4C — preregistered CE gate, pending

Read experiment_plan.md and input_audit.json. Exact fixed preparation step256
exists and passes archived state/order/metric checks. continuation_manifest.json
excludes8192 consumed chunks and fixes31250 new chunks, exactly8M targets.
No KD or test evaluation is authorized until the validation-only CE gate passes.
Only three LR candidates5e-6/1e-5/2e-5; selection includes fixed S0 step0.
Gate improvement>=.01 plus trained selected checkpoint. If gate fails STOP.
If gate passes commit the selected protocol BEFORE KD025/KD1. No lambda5,
clipping sweep, new method, paper/final_evidence edit or new seed replication.

Large weights/data/logs are outputs-only; train and numerical helpers reuse
unchanged Phase4A source. Frozen manifests contain indices/hashes, not text.
