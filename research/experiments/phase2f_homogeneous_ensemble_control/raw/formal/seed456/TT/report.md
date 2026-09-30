# Late-fusion Teacher Distillation Report

- Experiment: `phase2f_wt2_tt_seed456`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260930_034520_phase2f_tt_teacher_joint_e10`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.7`
- KD lambda (token mode): `5.0`
- Teacher effective alpha: `[0.5]`
- Student init SHA256: `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757`
- Teacher best test ppl: `66.3130`
- Grassmann baseline test ppl: `None`
- Transformer baseline test ppl: `197.64515686035156`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `68.1260`
- Test ppl: `61.0427`
- Params: `31434257`
- Beats grassmann baseline: `False`
- Beats transformer baseline: `True`
- Beats both baselines: `False`