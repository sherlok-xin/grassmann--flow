# Late-fusion Teacher Distillation Report

- Experiment: `phase2e_wt2_t_seed456`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.7`
- KD lambda (token mode): `5.0`
- Teacher effective alpha: `[1.0]`
- Student init SHA256: `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757`
- Teacher best test ppl: `66.8663`
- Grassmann baseline test ppl: `216.45419311523438`
- Transformer baseline test ppl: `197.64515686035156`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `70.5236`
- Test ppl: `63.0695`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `True`
- Beats both baselines: `True`