# Phase 2E Teacher Branch Ablation

Status: formal Transformer-only training authorized under a documented protocol amendment. No formal endpoint had been launched or observed when the amendment was recorded.

## Scope and intervention

Phase 2E asks whether KD from the fixed jointly trained Transformer--Grassmann teacher transfers more effectively than KD from either branch alone. The frozen checkpoint SHA256 is `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`. The only planned new formal condition is Transformer-only supervision (`alpha=1.0`); C0, fused (`alpha=0.5`), and Grassmann-only (`alpha=0.0`) endpoints are reused from Phase 2C.

## Completed preflight

The same 512 validation chunks were used for all conditions, with selected-index SHA256 `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

| Condition | Alpha | Teacher validation NLL | PPL |
|---|---:|---:|---:|
| Fused | 0.5 | 4.300646 | 73.747 |
| Transformer-only | 1.0 | 4.445647 | 85.255 |
| Grassmann-only | 0.0 | 4.652064 | 104.801 |

The mean residual teacher advantages over the three seed-specific S0 students were `+0.074515` for fused, `-0.070486` for Transformer-only, and `-0.276903` for Grassmann-only. Thus only the fused teacher was globally better than every evaluated S0 on this subset.

The Transformer-only to fused full-parameter KD-gradient ratios were `1.3164`, `1.2788`, and `1.2324` for seeds 42, 123, and 456. All values lie within the preregistered `[0.5, 2.0]` scale range. The gradient preflight therefore did not produce `KD_SCALE_MISMATCH`.

## Observational branch complementarity

The two branches assigned higher gold-token probability on substantial and complementary token subsets: Transformer exceeded Grassmann on 51.56% of tokens, while Grassmann exceeded Transformer on 48.44%. Transformer was top-1 correct while Grassmann was wrong on 5.89% of tokens; the reverse occurred on 5.20%.

On the hardest student-loss quintile, mean gold-token utility was `0.6356` for Transformer-only, `0.2133` for Grassmann-only, and `0.7396` for fused. The fused teacher improved gold log-probability over Transformer by `0.1040` NLL-equivalent units and over Grassmann by `0.5263` on this quintile. These are observational validation-token summaries and do not establish a causal geometric mechanism.

## Smoke and stability gates

The preregistered reduced-data Transformer-only smoke reproduced the known short-window artifact. Its clipping fraction was `1.00`, and AMP overflow and non-finite fractions were both `0.16`; all identity, hash, alpha, objective, finite-loss, serialization, and reload checks passed. The reduced gate therefore failed.

The preauthorized fallback used matched full-data one-epoch fused and Transformer-only arms from the same seed-42 S0. Fused passed with clipping `0.2347`, overflow `0.0136`, and non-finite fraction `0.0136`. Transformer-only had finite losses, a reloadable checkpoint, the exact frozen teacher and S0, overflow `0.0136`, and non-finite fraction `0.0136`, but its clipping fraction was `1.00`. It therefore failed the frozen requirement that clipping remain below `0.95`.

## Authorized amendment before formal endpoints

On 2026-09-29, the research lead explicitly authorized formal Transformer-only training despite gate-stage `clip_fraction=1.0`. The authorization was recorded before any formal endpoint was launched. It is conditional on finite parameter gradients, T/F parameter-gradient ratios within `[0.5, 2.0]`, and overflow/non-finite fractions remaining approximately `0.0136`. No alpha, lambda, temperature, AMP setting, gradient clip value, training budget, data, student S0, or evaluation rule may change.

`CLIP_SATURATION_WARNING`: Transformer-only gradients continuously triggered clipping during the full-data gate. Formal endpoints may therefore reflect an optimization constraint in addition to teacher-source differences. The warning remains permanent even if formal training completes normally.
