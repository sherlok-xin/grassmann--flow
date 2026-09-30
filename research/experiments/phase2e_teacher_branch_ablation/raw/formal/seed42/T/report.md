# Late-fusion Teacher Distillation Report

- Experiment: `phase2e_wt2_t_seed42`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.7`
- KD lambda (token mode): `5.0`
- Teacher effective alpha: `[1.0]`
- Student init SHA256: `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`
- Teacher best test ppl: `66.8663`
- Grassmann baseline test ppl: `216.45419311523438`
- Transformer baseline test ppl: `197.64515686035156`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `69.1348`
- Test ppl: `62.0510`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `True`
- Beats both baselines: `True`