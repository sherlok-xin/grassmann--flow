# Phase 2D GPT Handoff

Phase 2D is complete. It compares a jointly trained WikiText-2 Teacher J with an alpha-only Teacher A while fixing architecture, source branch identities, fusion semantics, and effective alpha at 0.5. The objective is `CE + 5 * KL_token_mean`, temperature 2, using three independent S0 checkpoints and validation-selected endpoints. Phase 2C C0 and Teacher-J C1 results are reused; only the Teacher-A D2 arms are new.

Teacher J checkpoint SHA256 is `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`; Teacher A is `43d288db1dba826d8ad1bb3fcbd09b9720bdde24c198e858e00ce4c4658df012`. S0 hashes for seeds 42/123/456 are `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`, `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13`, and `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757`. Integrity checks pass and 179/191 teacher state tensors differ.

Seed-level endpoint results are:

| Seed | C0 NLL | J NLL | A NLL | Gain J | Gain A | Q=A-J |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4.270742 | 4.108241 | 4.293136 | +0.162501 | -0.022394 | +0.184895 |
| 123 | 4.277520 | 4.130183 | 4.312882 | +0.147337 | -0.035362 | +0.182699 |
| 456 | 4.280830 | 4.123673 | 4.306036 | +0.157157 | -0.025206 | +0.182364 |

Mean/sample SD is +0.155665/0.007692 for Gain J, -0.027654/0.006822 for Gain A, and +0.183319/0.001375 for Q. Every J gain is positive, every A gain is negative, and every Q is positive.

Optimization integrity does not explain the endpoint contrast. Full-parameter KD-gradient ratios `G_A/G_J` are 1.102792, 1.148522, and 1.093056. The original 2,000-line Teacher-A smoke failed, but the explicitly authorized matched full-data one-epoch J/A gate passed: clipping was 0.234694/0.275510 and overflow/non-finite was 0.013605 for both. Teacher-A formal epoch-1 clipping was 0.666667/0.503401/0.513605, with overflow/non-finite 0.013605 for all seeds. No scale-control arm was triggered.

On the same 512 validation chunks, J/A fused NLL is 4.300644/6.376220, Transformer NLL is 4.445648/6.540640, and Grassmann NLL is 4.652064/6.696787. Joint training improves both branches. J has slightly higher JSD and lower top-1 agreement, but its fusion gain is smaller, so the dominant change is branch quality rather than a larger fusion-gain margin. Teacher A is globally negative-utility but has positive hardest-quintile utility for all seeds; this localized signal does not produce positive endpoint transfer here.

Final decisions are: Decision 1 YES; Decision 2 NO; Decision 3 YES; Decision 4 NO; Decision 5 broad improvement of both branches with mixed complementarity indicators and no fusion-gain increase; Decision 6 NO because Phase 2C still shows positive transfer from another globally weak teacher; Decision 7 option A, Transformer-only vs Grassmann-only vs fused-teacher KD. Option A is selected but not launched.

The Phase 2D artifact commit is `dc9033583df861c00b71135426d8b8fd52c46d05`. Full evidence is in `REPORT.md`, `results_multiseed.csv`, `raw/results_summary.json`, and `provenance.md`.
