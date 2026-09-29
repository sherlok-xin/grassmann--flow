# Late-fusion Teacher Distillation Report

- Experiment: `phase2d_wt2_d2_a_seed456`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260328_084319_wt2_v1_hybrid_alpha_only`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.7`
- KD lambda (token mode): `5.0`
- Teacher effective alpha: `[0.5]`
- Student init SHA256: `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757`
- Teacher best test ppl: `618.7059`
- Grassmann baseline test ppl: `216.45419311523438`
- Transformer baseline test ppl: `197.64515686035156`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `82.2118`
- Test ppl: `74.1460`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `True`
- Beats both baselines: `True`