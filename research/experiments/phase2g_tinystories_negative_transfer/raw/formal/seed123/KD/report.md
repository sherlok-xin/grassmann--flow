# Late-fusion Teacher Distillation Report

- Experiment: `phase2g_ts_kd_seed123`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.0`
- KD lambda (token mode): `5.0`
- Teacher effective alpha: `[0.5994059443473816]`
- Student init SHA256: `b30d662bcd51db22e841455bd29a8b83617447d583a71c0b0cc97eb362ede8b3`
- Teacher best test ppl: `4.9691`
- Grassmann baseline test ppl: `7.514091491699219`
- Transformer baseline test ppl: `5.148688793182373`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `5.3972`
- Test ppl: `5.4154`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `False`
- Beats both baselines: `False`