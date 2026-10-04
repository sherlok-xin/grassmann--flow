# V2 paper claim set

Only four central claims are allowed. `C1`–`C4` are writing identifiers, not
new preregistered hypotheses. Exact support and scope are in
`EVIDENCE_INTEGRATION_V2.md`; numbers below are test NLL unless labeled.

## C1 — Warm-start KD has reproducible benefits and harms under matched protocols

Allowed wording: Under the tested warm-start protocols, KD improves matched
CE continuation on PTB and WT2 but harms it on TinyStories, with the same
direction across three independent student initializations in each domain.

Support: E1/PTB gain +0.144937 ± 0.003938; E2/Phase 2C WT2
+0.155665 ± 0.007692; E3/Phase 2G TinyStories −0.107219 ± 0.000415.
Each is 3/3 sign-consistent, token_mean lambda5 T2, exact within-seed S0 and
validation-selected endpoints. Status CONFIRMED within these protocols.

Limit: one teacher per domain, three student seeds, differing corpus/teacher
states and data scales. Do not write “dataset causes transfer sign,” “KD always
helps/hurts,” “warm-start is necessary,” or imply random-init superiority was
established. The modern TinyStories single-seed pilot is not another
replication of this architecture/protocol.

## C2 — Teacher training state and predictive quality strongly modulate transfer at fixed composition

Allowed wording: A fixed-composition WT2 teacher-state intervention reverses
the sign of KD transfer across the three matched students. Joint-trained J
provides positive gain, whereas alpha-only A harms the same students.

Support: E4/Phase 2D forced alpha0.5, J gain +0.155665 ± 0.007692,
A gain −0.027654 ± 0.006822; Q=NLL_A−NLL_J +0.183319 ± 0.001375,
3/3 positive. Teacher validation-subset NLL J4.300644 versus A6.376220.
Status CONFIRMED state-level contrast; PARTIAL scalar-quality explanation.

Limit: teacher state/representations and likelihood change together; J is
reused, not independently retrained for this contrast. Preserve the original
failed short smoke and explicit full-data gate amendment, including formal
AMP event rates. Do not call it a causal intervention on likelihood alone,
claim teacher quality is irrelevant, or imply all parameters were fixed.

## C3 — Global teacher likelihood alone is insufficient to determine distillability

Allowed wording: Better global held-out teacher likelihood is not a necessary
condition for positive KD transfer in the tested WT2 configuration; a
Grassmann-only teacher worse than matched CE on the same validation subset
still yields positive transfer for all three students. Together with C2,
this shows why a scalar teacher-quality ranking is insufficient for the
observed endpoint behavior.

Support: E5/Phase 2C residuals vs C0 −0.266520/−0.253492/−0.251006,
test gains +0.046771/+0.033598/+0.042604, mean +0.040991 ± 0.006733.
Status CONFIRMED counterexample to the universal weaker-teacher⇒harm rule.
E4 supplies the complementary controlled state/quality modulation.

Limit: “weaker” is defined against validation-selected C0 CE on the identical
subset, not automatically against S0. This does not show weak teachers are
generally better. Existing literature already establishes related teacher
quality/compatibility limitations; no first-discovery or theory claim.
E11 modern better-teacher harm is secondary, with E12/E13 degeneracy explicitly
attached; it must not be the clean modern proof of this central claim.

## C4 — Ensemble benefits do not imply heterogeneous Grassmann-specific superiority

Allowed wording: Fused WT2 supervision improves over its two component
sources in this protocol, but a homogeneous TT teacher outperforms TG on all
three students. These controls do not support attributing the gains to
Grassmann-specific supervision or architectural heterogeneity.

Support: E6 fused−T +0.020797 ± 0.001214 (PARTIAL, T clipping warning),
fused−G +0.114674 ± 0.001001; E7 H=Gain_TG−Gain_TT
−0.012460 ± 0.001348, 3/3 TT better. Teacher NLL on common full validation
is TG4.303697/TT4.293589; parameters37,749,633/35,340,801.

Limit: TT teacher quality, parameter count, alpha LR and optimization differ;
the contrast contradicts the tested superiority claim, not an isolated causal
mechanism or universal superiority of homogeneous ensembles. Larger TG JSD
with smaller gain and E8's small incremental JSD signal do not show diversity
is useless. No Pluecker-causality claim is allowed.

## Secondary observations, not extra central contributions

E10/Phase 3A: local first/second-order diagnostic signs fail to reliably predict
long-horizon outcomes; seed-condition AUROC0.644444, sign accuracy0.571429,
stable calibration signs11/21; curvature adds no sign separation. Report as
negative diagnostic in Section7, details in appendix. Do not propose
Safe-lambda as a successful method or a useful tuning rule.

E11–E13/Phase 4: CE-relative modern FineWeb harm replicates at lambda1/5
(means −0.013550460935/−0.043924082202), but all9/9 step0 choices prefer S0.
Best disjoint-continuation CE gate improvement0.004849292809 fails0.01;
no Phase4C KD/test. Report as bounded external-validity stress and unresolved
limitation in Section8, not an Abstract headline. Preserve clipping and
gradient-event warnings. No clean modern negative/beneficial KD conclusion.

Synthesis: The evidence is consistent with distillability depending jointly
on teacher state, student state and optimization configuration, rather than
being captured by a single quality/diversity/local-compatibility measure.
This is a conservative interpretation, not a theorem or a fully factorial
causal decomposition. The current study does not separate every factor.

## Forbidden claim checklist

Delete Grassmann-specific superiority, Pluecker geometry causes KD gain,
heterogeneity beats homogeneous ensembles, strong JSD prediction, successful
CRBD or Safe-lambda, reliable local endpoint prediction, irrelevant teacher
quality, weaker-teachers-usually-better, universal weaker-teacher⇒harm,
dataset-caused transfer sign, universal modern/FineWeb KD failure, and clean
large-model confirmation. Do not turn failed Phase4C gate into “zero CE
headroom” or import old test values as new endpoints. No new method, experiment,
manuscript-wide prose or edits to frozen evidence are authorized in this stage.
