# Phase 2E Teacher Branch Ablation

Status: blocked at the preregistered full-data stability gate. No Transformer-only formal endpoint has been launched or observed.

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

## Blocking decision

The execution specification permits formal runs only if the reduced or matched full-data stability gate passes and explicitly prohibits inventing a new rule after observing the gate. Consequently, the three formal Transformer-only runs were not launched. `results_multiseed.csv`, endpoint figures, and the seven final Phase 2E decisions cannot be produced without either terminating Phase 2E as gate-blocked or explicitly authorizing a documented prospective protocol amendment. Existing Phase 2C endpoint evidence remains unchanged.
