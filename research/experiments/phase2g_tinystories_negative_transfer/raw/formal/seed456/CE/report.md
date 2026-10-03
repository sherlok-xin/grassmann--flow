# Late-fusion Teacher Distillation Report

- Experiment: `phase2g_ts_ce_seed456`
- Student type: `hybrid_lite`
- Student config: dim=224, layers=6, reduced_dim=56, heads=8
- Teacher run: `/workspace/grassmannflows/grassmann-flows/outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10`
- KD loss mode: `token_mean`
- Distill alpha (legacy mode): `0.0`
- KD lambda (token mode): `0.0`
- Teacher effective alpha: `[0.5994059443473816]`
- Student init SHA256: `6eaa7b735ee425e98711103b594e68709cc94359cbfbddbe0eb9a87795c6fbc7`
- Teacher best test ppl: `4.9691`
- Grassmann baseline test ppl: `7.514091491699219`
- Transformer baseline test ppl: `5.148688793182373`

## Final student metrics

- Best epoch: `10`
- Best val ppl: `4.8511`
- Test ppl: `4.8721`
- Params: `31434257`
- Beats grassmann baseline: `True`
- Beats transformer baseline: `True`
- Beats both baselines: `True`