# Phase 2B Preflight

This preflight uses validation data only. Teacher NLL values come from the frozen Phase 2A alpha grid. Each WS+CE NLL was recomputed from the exact S1 checkpoint on the identical validation chunks used for its teacher landscape.

Classification thresholds are: `T_good` when Delta_teacher >= +0.05, `T_near` when |Delta_teacher| <= 0.025, and `T_bad` when Delta_teacher <= -0.05. Remaining points are marked `transition_unclassified`.

| Dataset | alpha | Teacher val NLL | WS+CE val NLL | Delta_teacher | classification |
|---|---:|---:|---:|---:|---|
| ptb | 0.0 | 4.200023 | 4.237279 | +0.037256 | transition_unclassified |
| ptb | 0.1 | 4.146415 | 4.237279 | +0.090864 | T_good |
| ptb | 0.2 | 4.104831 | 4.237279 | +0.132448 | T_good |
| ptb | 0.3 | 4.075359 | 4.237279 | +0.161920 | T_good |
| ptb | 0.4 | 4.058225 | 4.237279 | +0.179054 | T_good |
| ptb | 0.5 | 4.053784 | 4.237279 | +0.183495 | T_good |
| ptb | 0.6 | 4.062498 | 4.237279 | +0.174781 | T_good |
| ptb | 0.7 | 4.084927 | 4.237279 | +0.152352 | T_good |
| ptb | 0.8 | 4.121706 | 4.237279 | +0.115573 | T_good |
| ptb | 0.9 | 4.173515 | 4.237279 | +0.063764 | T_good |
| ptb | 1.0 | 4.241051 | 4.237279 | -0.003772 | T_near |
| wikitext2 | 0.0 | 4.652055 | 4.385545 | -0.266510 | T_bad |
| wikitext2 | 0.1 | 4.543900 | 4.385545 | -0.158355 | T_bad |
| wikitext2 | 0.2 | 4.455310 | 4.385545 | -0.069766 | T_bad |
| wikitext2 | 0.3 | 4.385215 | 4.385545 | +0.000330 | T_near |
| wikitext2 | 0.4 | 4.333548 | 4.385545 | +0.051997 | T_good |
| wikitext2 | 0.5 | 4.300645 | 4.385545 | +0.084900 | T_good |
| wikitext2 | 0.6 | 4.287094 | 4.385545 | +0.098450 | T_good |
| wikitext2 | 0.7 | 4.293700 | 4.385545 | +0.091844 | T_good |
| wikitext2 | 0.8 | 4.321477 | 4.385545 | +0.064067 | T_good |
| wikitext2 | 0.9 | 4.371653 | 4.385545 | +0.013891 | T_near |
| wikitext2 | 1.0 | 4.445656 | 4.385545 | -0.060111 | T_bad |
| tinystories | 0.0 | 2.112059 | 1.580178 | -0.531881 | T_bad |
| tinystories | 0.1 | 1.960566 | 1.580178 | -0.380389 | T_bad |
| tinystories | 0.2 | 1.835315 | 1.580178 | -0.255137 | T_bad |
| tinystories | 0.3 | 1.736168 | 1.580178 | -0.155990 | T_bad |
| tinystories | 0.4 | 1.663756 | 1.580178 | -0.083578 | T_bad |
| tinystories | 0.5 | 1.618701 | 1.580178 | -0.038523 | transition_unclassified |
| tinystories | 0.6 | 1.600853 | 1.580178 | -0.020675 | T_near |
| tinystories | 0.7 | 1.609456 | 1.580178 | -0.029279 | transition_unclassified |
| tinystories | 0.8 | 1.643613 | 1.580178 | -0.063435 | T_bad |
| tinystories | 0.9 | 1.702408 | 1.580178 | -0.122230 | T_bad |
| tinystories | 1.0 | 1.785048 | 1.580178 | -0.204870 | T_bad |
| codeparrot_common5k | 0.0 | 2.680898 | 2.057750 | -0.623149 | T_bad |
| codeparrot_common5k | 0.1 | 2.410857 | 2.057750 | -0.353107 | T_bad |
| codeparrot_common5k | 0.2 | 2.189363 | 2.057750 | -0.131614 | T_bad |
| codeparrot_common5k | 0.3 | 2.015658 | 2.057750 | +0.042092 | transition_unclassified |
| codeparrot_common5k | 0.4 | 1.889613 | 2.057750 | +0.168137 | T_good |
| codeparrot_common5k | 0.5 | 1.811533 | 2.057750 | +0.246216 | T_good |
| codeparrot_common5k | 0.6 | 1.780086 | 2.057750 | +0.277664 | T_good |
| codeparrot_common5k | 0.7 | 1.791483 | 2.057750 | +0.266267 | T_good |
| codeparrot_common5k | 0.8 | 1.843727 | 2.057750 | +0.214023 | T_good |
| codeparrot_common5k | 0.9 | 1.941639 | 2.057750 | +0.116111 | T_good |
| codeparrot_common5k | 1.0 | 2.089248 | 2.057750 | -0.031498 | transition_unclassified |

## Triplet decision

WikiText-2 provides a complete triplet on the existing frozen alpha grid:

- `T_good`: alpha 0.5, Delta_teacher = +0.084900 nat/token.
- `T_near`: alpha 0.3, Delta_teacher = +0.000329 nat/token.
- `T_bad`: alpha 0.0, Delta_teacher = -0.266510 nat/token.

PTB lacks T_bad, TinyStories lacks T_good, and the existing CodeParrot grid lacks T_near. No cross-dataset triplet and no fabricated condition is used.

## Route selection

Route A is selected on WikiText-2. Its separation is clean, its frozen S0 and S1 checkpoints are available, its dataset manifest is reproducible, and its 9,379 training chunks make the bounded four-arm pilot materially cheaper than CodeParrot common-5k. No additional alpha evaluation is required for route selection.

All four arms must load the same S0 checkpoint with SHA256 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`.
