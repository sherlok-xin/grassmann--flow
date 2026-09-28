# Late-fusion Teacher Distillation Report

- Experiment: `phase2c_wt2_c0_seed123`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.7`
- KD lambda (token mode): `0.0`
- Teacher effective alpha: `[0.5]`
- Student init SHA256: `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13`
- Teacher best test ppl: `66.8663`
- Grassmann baseline test ppl: `216.45419311523438`
- Transformer baseline test ppl: `197.64515686035156`

## Final student metrics

- Best epoch: `1`
- Best val ppl: `81.3579`
- Test ppl: `72.0615`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `True`
- Beats both baselines: `True`