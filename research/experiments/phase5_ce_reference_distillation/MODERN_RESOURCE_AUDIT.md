# Qwen resource audit and bounded-budget amendment

2026-11-26 Asia/Shanghai. Both official pinned model weight SHA-256 values match the official revision/LFS metadata in `modern_assets_verified.json`. All four real-Qwen eight-update smokes completed on idle GPU 2. No adapted teacher, prepared S0, selection-dev or test endpoint was available for this resource decision. Smoke loss checks use four training probes and do not establish task headroom or method superiority.

| Arm | Response targets/s | Peak GiB | Clip fraction | Nonfinite |
|---|---:|---:|---:|---:|
| qwen_resource_ce8 | 70.431 | 11.273 | 1.000 | 0.000 |
| qwen_resource_kd8 | 70.040 | 14.405 | 1.000 | 0.000 |
| qwen_resource_A8 | 66.293 | 14.405 | 1.000 | 0.000 |
| qwen_resource_teacher8 | 131.061 | 4.208 | 1.000 | 0.000 |

Student smokes use the full FP32 trainable 0.6B model with BF16 autocast; the 1.7B teacher uses a BF16 frozen base and trainable LoRA. No student LoRA or smaller-model fallback is required. Every smoke retains `CLIP_SATURATION_WARNING`. Finite clipped gradients are not numerical failure; these eight-update observations do not establish formal-run clipping prevalence or its causal role.

The slowest student rate is 66.293474 supervised response targets/s. The original 2M minimum would project 10.475306 GPUh including the fixed 25% overhead, exceeding the original 8 GPUh per-run planning threshold. The original supervisor stopped at that gate after completing all four smokes. The resource-only amendment adds a uniform 1M option to the existing 8M/4M/2M choices. The selected 1M projects 5.237653 GPUh. It preserves full-parameter Qwen, strong baselines and the planned seed confirmations under the shared 230 GPUh ceiling. No smoke NLL was used to select this budget.

All continuation arms and seeds use the same selected target budget. The preparation, data order/splits, teacher construction, search grid, optimizer, temperature, clipping and headroom criterion remain unchanged. The resource lock and amendment must be committed on the local workstation before teacher adaptation. The resumed supervisor checks a hash-and-commit receipt before launching formal work.

For planning, the fixed seed-42 CE/KD/A/literature screens total 17 x 250k = 4.25M continuation targets; four seed-42 full arms add 4M. Two confirmation seeds require six additional full CE/KD/literature arms (8 if A passes its promotion gate), plus their own S0 preparation. The second teacher is used for only two key full comparisons. Counting the optional A confirmations and one headroom fallback plus its 250k CE screen gives at most 19.50M planned continuation targets, about 102.13 GPUh at the slowest measured rate and overhead. This is a planning estimate, not spent compute. Student preparations, both teacher adaptations, evaluation, failed attempts and all legacy work are additional and enforced by the shared allocation ledger. Subsequent resource pressure reduces secondary work; no automatic budget extension is authorized.

This audit changes resource scheduling only. Modern CE headroom, KD utility, generation metrics and independent-seed confirmation remain unestablished.
