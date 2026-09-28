# Late-fusion Teacher Distillation Report

- Experiment: `phase2b_wt2_c0_ws_ce_a05`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.7`
- KD lambda (token mode): `0.0`
- Teacher effective alpha: `[0.5]`
- Student init SHA256: `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`
- Teacher best test ppl: `66.8663`
- Grassmann baseline test ppl: `216.45419311523438`
- Transformer baseline test ppl: `197.64515686035156`

## Final student metrics

- Best epoch: `1`
- Best val ppl: `80.0902`
- Test ppl: `71.5748`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `True`
- Beats both baselines: `True`