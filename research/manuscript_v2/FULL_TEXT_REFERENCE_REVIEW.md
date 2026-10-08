# Focused full-text reference review

Date: 2026-10-08. Input: `56ab6cee930da08637c3fa7c6b36d5cb22ccf09f`.

This review supports the conservative protocol distinctions in Section 2 and the foundational citations in Sections 3–8/appendix. Relevant methods/background sections were read; this is not an exhaustive systematic review. No adaptive KD method, GRACE implementation, or new experiment was evaluated. Bibliographic metadata were imported from primary exports, not invented. The earlier metadata-only ledger is retained as history; this ledger resolves its method-reading TODOs.

| Key | Reviewed scope | Allowed use / boundary |
|---|---|---|
| `hinton2015distilling` | Section 2, equations 1-2 | Temperature-softened soft targets, hard-label objective and temperature-squared gradient scaling. |
| `cho2019efficacy` | Sections 5.1-5.3 | Capacity mismatch; early stopping and harmful late-training teaching in the studied image-classification setting. |
| `park2021friendly` | Section 3.1 | Student branches participate in teacher training; not a frozen-logit warm-start intervention. |
| `kaplun2022bad` | Sections 2.2 and 3 | Definition and theoretical assumptions for useful teachers that are poor classifiers; not a universal autoregressive likelihood rule. |
| `menon2021statistical` | Sections 4.1-4.2 | Probability-estimation quality and bias-variance account; does not make our fixed-state intervention a scalar causal-quality experiment. |
| `yuan2021selection` | RL-based teacher-selection method | Instance-dependent selection uses student feedback, rather than one fixed fused teacher. |
| `zhou2022metadistil` | Algorithm 1 and meta-update method | Quiz-set performance after simulated student learning updates the teacher. |
| `zhong2024atkd` | Sections 3-4 | Easy/hard-token analysis and target-related adaptive teaching; prior harmful-LM-supervision observation. |
| `gu2024minillm` | Sections 2.1-2.2 | Reverse KL, policy-gradient optimization and student sampling differ from corpus-based token-mean forward KL. |
| `agarwal2024gkd` | Algorithm 1 and objective | Student-generated sequences, on-policy mixtures and flexible divergences; not our fixed continuation recipe. |
| `du2020agree` | Sections 3.1-3.2 | Teacher-loss gradient guidance for adaptive ensemble weighting. |
| `stanton2021really` | Section 4.4 and ensemble experiments | Fidelity and generalization need not coincide; not an architectural heterogeneity advantage. |
| `wang2021selective` | Section 5, selective distillation | Harmful supervision motivates selective NMT KD; our work is not the first negative-transfer finding. |
| `xie2026adakd` | Adaptive method: Hellinger difficulty, LATF and IDTS | Token difficulty and EMA loss feedback adapt focus and temperature. |
| `jin2026entropy` | Algorithm 1 and equation 9 | On-policy reverse KL supplemented by forward KL at high teacher entropy. |
| `busbridge2025scaling` | Section 4.2 and scaling-law formulation | Distilled student likelihood depends on teacher quality, size and compute allocation; no claim to supersede the scaling law. |
| `panigrahi2026grace` | Sections 2.1-2.2 | Student gradients on teacher-generated autoregressive responses; cross-validated spectral teacher ranking without teacher logits. Our validation-gradient/HVP diagnostic is different and does not refute GRACE. |
| `marcus1993ptb` | Title page and corpus introduction | Penn Treebank corpus attribution only, not modern preprocessing or endpoint evidence. |
| `merity2016wikitext` | Title page and WikiText dataset section | Dataset attribution; verified arXiv 2016 version, without inventing conference metadata. |
| `eldan2023tinystories` | Title page and dataset construction | TinyStories attribution; our split/token budget is separately sourced to archived manifests. |
| `vaswani2017attention` | Section 3, model architecture | Transformer background only; no claim that our compact test bed reproduces the original recipe. |
| `benallal2025smollm2` | Section 6, 135M/360M; model cards | Small pretrained-model background. Official 360M model card explicitly lists FineWeb-Edu; document-level overlap with our subset remains unknown. |
| `penedo2024fineweb` | FineWeb and FineWeb-Edu construction sections | Data attribution; pinned subset revision and splits come from archived manifests. |
| `edelman1998geometry` | Section 2, Grassmann/Stiefel geometry | Basis-equivalence/subspace background only; no evidence for a geometric KD mechanism. |

Primary PDF/export URLs and SHA256 hashes are recorded in `full_text_reference_verification.json`. Full PDFs, extracted text and raw export scratch files remain under `outputs/manuscript_v2_reference_review/` and are not committed.

Retrieval corrections were explicit: AdaKD uses the official attachment 44662; SmolLM2 falls back from a forbidden OpenReview download to its matching arXiv version; Edelman uses direct Crossref export. Cho reuses the previously verified publisher metadata export. Ben Allal name normalization was checked against primary author text. An unverified Dorst book record was not added; the author-maintained Plücker companion page is cited as a footnote for exterior-product background only. The official 360M model card lists FineWeb-Edu, but does not resolve exact document-level overlap with this study.

No claim is made to first discover harmful KD, weak-teacher usefulness, teacher compatibility, gradient-based teacher selection, or superiority over adaptive methods. The contribution remains matched warm-start CE-relative endpoints, the bounded state intervention, homogeneous-ensemble falsification and negative local-diagnostic evidence.
