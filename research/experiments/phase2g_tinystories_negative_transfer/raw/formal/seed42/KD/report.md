# Late-fusion Teacher Distillation Report

- Experiment: `h0_ts_confirm_token_l5_seed42`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.0`
- KD lambda (token mode): `5.0`
- Teacher best test ppl: `4.9691`
- Grassmann baseline test ppl: `7.514091491699219`
- Transformer baseline test ppl: `5.148688793182373`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `5.3957`
- Test ppl: `5.4127`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `False`
- Beats both baselines: `False`