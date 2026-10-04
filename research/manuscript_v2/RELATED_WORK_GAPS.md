# Related-work overlap audit (V2)

Checked on 2026-10-04. This is a focused overlap audit, not a systematic review,
and not evidence of being the first work on any concept. Searches covered
official CVF, NeurIPS, ACL Anthology, AAAI, ICLR/OpenReview, PMLR and arXiv.
Queries included teacher quality/student-friendly teaching; teacher selection
and compatibility; ATKD autoregressive adaptive distillation; ensemble KD;
gradient-space/validation-driven KD; harmful supervision and weak teachers;
and recent token/entropy-adaptive LM distillation. Abstracts and publisher
metadata were checked; complete method/protocol comparisons remain writing
TODOs. No adaptive method listed below was evaluated in our experiments.

Publisher BibTeX was fetched, title/year/authors/venue presence checked and
reduced mechanically to compact metadata. One exported author-case/given-name
spacing issue was corrected against the official PDF title page and recorded.
`citation_verification.json`
records the export URL, identifier, metadata scope and source hash. A failed
fetch is `VERIFY_EXTERNAL` and is excluded from `references.bib`; it is not
quietly replaced by a hand-invented citation. Abstract text is not reproduced.

## Positioning by area

Teacher-quality and student-oriented training: Cho/Hariharan, Park et al.,
Menon et al. and Kaplun et al. already undermine simple teacher-accuracy rules.
Teacher selection/compatibility: Yuan et al. and MetaDistil already adapt
supervision to the student. Autoregressive adaptive KD: ATKD is the closest
direct antecedent for harmful LM KD; MiniLLM/GKD/AdaKD show that the objective
and student distribution/state are established concerns. Ensemble/multi-teacher:
Stanton et al. and Du et al. make diversity, fidelity and optimization relevant.
Gradient/validation-driven: Du et al. and MetaDistil prevent any novelty claim
for gradient or held-out feedback alone. Harmful supervision: Wang et al. and
ATKD directly precede our negative-transfer observations. Recent AdaKD and
entropy-aware on-policy work must not be omitted from a final 2026 literature
pass. Our failed offline diagnostic does not invalidate those adaptive methods.

## Verified records and bounded distinctions

### cho2019efficacy
Title: On the Efficacy of Knowledge Distillation.

Authors: Cho, Jang Hyun and Hariharan, Bharath.

Year / venue: 2019 / ICCV. Identifiers: arXiv:1910.01348.

Status: VERIFIED_METADATA. Area: Teacher quality / compatibility. [Primary metadata/export](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html); [publisher paper/page](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html).

Exact overlap: More accurate teachers need not yield better distilled students; teacher capacity and training duration matter.

Exact remaining distinction: We pair CE and KD from identical warm-start LM states and triangulate teacher-state, source and ensemble controls. Accuracy insufficiency itself is not new.

### park2021friendly
Title: Learning Student-Friendly Teacher Networks for Knowledge Distillation.

Authors: Park, Dae Young and Cha, Moon-Hyun and Jeong, Changwook and Kim, Dae Sin and Han, Bohyung.

Year / venue: 2021 / NeurIPS. Identifiers: arXiv:2102.07650.

Status: VERIFIED_METADATA. Area: Student-oriented teacher training. [Primary metadata/export](https://proceedings.neurips.cc/paper_files/paper/12641-/bibtex); [publisher paper/page](https://proceedings.neurips.cc/paper_files/paper/2021/file/6e7d2da6d3953058db75714ac400b584-Paper.pdf).

Exact overlap: Student-friendly teacher representations are explicitly trained with student branches.

Exact remaining distinction: Our teacher-state intervention changes frozen supervision while holding composition fixed; no new student-friendly teacher-training algorithm is proposed.

### yuan2021selection
Title: Reinforced Multi-Teacher Selection for Knowledge Distillation.

Authors: Yuan, Fei and Shou, Linjun and Pei, Jian and Lin, Wutao and Gong, Ming and Fu, Yan and Jiang, Daxin.

Year / venue: 2021 / AAAI. Identifiers: DOI:10.1609/aaai.v35i16.17680; arXiv:2012.06048; DOI:10.1609/aaai.v35i16.17680.

Status: VERIFIED_METADATA. Area: Teacher selection / multi-teacher compatibility. [Primary metadata/export](https://doi.org/10.1609/aaai.v35i16.17680); [publisher paper/page](http://dx.doi.org/10.1609/aaai.v35i16.17680).

Exact overlap: Instance-dependent teacher selection accounts for differing teacher suitability and student capacity.

Exact remaining distinction: Our source and ensemble contrasts keep supervision rules fixed rather than optimizing a teacher-selection policy. We do not benchmark that policy.

### zhong2024atkd
Title: Revisiting Knowledge Distillation for Autoregressive Language Models.

Authors: Zhong, Qihuang and Ding, Liang and Shen, Li and Liu, Juhua and Du, Bo and Tao, Dacheng.

Year / venue: 2024 / ACL. Identifiers: ACL:2024.acl-long.587; arXiv:2402.11890; DOI:10.18653/v1/2024.acl-long.587.

Status: VERIFIED_METADATA. Area: Autoregressive adaptive KD; closest direct prior. [Primary metadata/export](https://aclanthology.org/2024.acl-long.587.bib); [publisher paper/page](https://aclanthology.org/2024.acl-long.587/).

Exact overlap: Larger autoregressive teachers can yield poorer students; token teaching modes motivate ATKD.

Exact remaining distinction: The contribution is a matched warm-start CE-relative evidence chain, including independent student seeds, fixed-composition state changes and a homogeneous ensemble falsification. No ATKD baseline was run; do not claim superiority or first discovery of harmful LM KD.

### stanton2021really
Title: Does Knowledge Distillation Really Work?.

Authors: Stanton, Samuel and Izmailov, Pavel and Kirichenko, Polina and Alemi, Alexander A and Wilson, Andrew G.

Year / venue: 2021 / NeurIPS. Identifiers: arXiv:2106.05945; OpenReview:7J-fKoXiReA.

Status: VERIFIED_METADATA. Area: Ensemble KD / optimization. [Primary metadata/export](https://proceedings.neurips.cc/paper_files/paper/12152-/bibtex); [publisher paper/page](https://proceedings.neurips.cc/paper_files/paper/2021/file/376c6b9ff3bedbbea56751a84fffc10c-Paper.pdf).

Exact overlap: Student imitation fidelity and generalization differ; optimization can impede teacher matching.

Exact remaining distinction: Our endpoint is improvement over matched CE rather than fidelity. The bounded branch/ensemble and continuation-regime controls are empirical evidence, not a new account of optimization.

### du2020agree
Title: Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space.

Authors: Du, Shangchen and You, Shan and Li, Xiaojie and Wu, Jianlong and Wang, Fei and Qian, Chen and Zhang, Changshui.

Year / venue: 2020 / NeurIPS. Identifiers: NeurIPS:91c77393975889bd08f301c9e13a44b7.

Status: VERIFIED_METADATA. Area: Gradient-aware ensemble distillation. [Primary metadata/export](https://proceedings.neurips.cc/paper_files/paper/10759-/bibtex); [publisher paper/page](https://proceedings.neurips.cc/paper_files/paper/2020/file/91c77393975889bd08f301c9e13a44b7-Paper.pdf).

Exact overlap: Gradient-space teacher agreement/diversity informs adaptive ensemble supervision.

Exact remaining distinction: We test fixed teacher composition and offline local-gradient endpoint diagnostics. Their failure does not refute online adaptive weighting or the full method of this work.

### zhou2022metadistil
Title: {BERT} Learns to Teach: Knowledge Distillation with Meta Learning.

Authors: Zhou, Wangchunshu and Xu, Canwen and McAuley, Julian.

Year / venue: 2022 / ACL. Identifiers: ACL:2022.acl-long.485; arXiv:2106.04570; DOI:10.18653/v1/2022.acl-long.485.

Status: VERIFIED_METADATA. Area: Validation-driven / student-oriented distillation. [Primary metadata/export](https://aclanthology.org/2022.acl-long.485.bib); [publisher paper/page](https://aclanthology.org/2022.acl-long.485/).

Exact overlap: Student feedback and meta learning improve the suitability of teacher supervision.

Exact remaining distinction: Our fixed teacher-state intervention and failed offline prediction are not meta-optimization. We do not test or invalidate MetaDistil.

### kaplun2022bad
Title: Knowledge Distillation: Bad Models Can Be Good Role Models.

Authors: Kaplun, Gal and Malach, Eran and Nakkiran, Preetum and Shalev-Shwartz, Shai.

Year / venue: 2022 / NeurIPS. Identifiers: arXiv:2203.14649; NeurIPS:b88edf805e96654a4f9e7b783e854ae3; DOI:10.52202/068431-2079.

Status: VERIFIED_METADATA. Area: Weak teacher / limits of teacher quality. [Primary metadata/export](https://proceedings.neurips.cc/paper_files/paper/19353-/bibtex); [publisher paper/page](https://proceedings.neurips.cc/paper_files/paper/2022/file/b88edf805e96654a4f9e7b783e854ae3-Paper-Conference.pdf).

Exact overlap: A poor classifier can nevertheless provide useful distillation targets under the studied theoretical setting.

Exact remaining distinction: We give a held-out-NLL counterexample in warm-start autoregressive modeling, not a new weak-teacher principle or extension of the classifier theorem.

### menon2021statistical
Title: A statistical perspective on distillation.

Authors: Menon, Aditya K and Rawat, Ankit Singh and Reddi, Sashank and Kim, Seungyeon and Kumar, Sanjiv.

Year / venue: 2021 / ICML. Identifiers: PMLR:v139-menon21a; arXiv:2005.10419 (different preprint title).

Status: VERIFIED_METADATA. Area: Teacher quality / statistical explanation. [Primary metadata/export](https://proceedings.mlr.press/v139/menon21a.html); [publisher paper/page](https://proceedings.mlr.press/v139/menon21a.html).

Exact overlap: Probability estimation and bias/variance can explain distillation benefits beyond classifier accuracy.

Exact remaining distinction: Our scalar-likelihood counterexample does not imply teacher quality is irrelevant; we supply controlled LM endpoints rather than a new statistical theory.

### wang2021selective
Title: Selective Knowledge Distillation for Neural Machine Translation.

Authors: Wang, Fusheng and Yan, Jianhao and Meng, Fandong and Zhou, Jie.

Year / venue: 2021 / ACL-IJCNLP. Identifiers: ACL:2021.acl-long.504; arXiv:2105.12967; DOI:10.18653/v1/2021.acl-long.504.

Status: VERIFIED_METADATA. Area: Harmful supervision / negative transfer. [Primary metadata/export](https://aclanthology.org/2021.acl-long.504.bib); [publisher paper/page](https://aclanthology.org/2021.acl-long.504/).

Exact overlap: Teacher supervision on some training samples can hurt NMT; selective distillation addresses this.

Exact remaining distinction: We use frozen token-mean forward KL and paired continuation outcomes, without sample selection. Harmful supervision and selection are established ideas, not novel claims here.

### gu2024minillm
Title: MiniLLM: Knowledge Distillation of Large Language Models.

Authors: Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie.

Year / venue: 2024 / ICLR. Identifiers: arXiv:2306.08543; ICLR:8ac015d409635f196f9e3e9dcfb9a94e.

Status: VERIFIED_METADATA. Area: Modern autoregressive KD / loss choice. [Primary metadata/export](https://proceedings.iclr.cc/paper_files/paper/4166-/bibtex); [publisher paper/page](https://proceedings.iclr.cc/paper_files/paper/2024/file/8ac015d409635f196f9e3e9dcfb9a94e-Paper-Conference.pdf).

Exact overlap: Reverse-KL distillation addresses distributional properties of large autoregressive teachers and students.

Exact remaining distinction: Our forward-KL NLL protocol is not an evaluation of MiniLLM or instruction-following generation. Phase 4 cannot establish failure of modern KD in general.

### agarwal2024gkd
Title: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.

Authors: Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and Stanczyk, Piotr and Ramos Garea, Sabela and Geist, Matthieu and Bachem, Olivier.

Year / venue: 2024 / ICLR. Identifiers: OpenReview:3zKtaqxLhW; arXiv:2306.13649.

Status: VERIFIED_METADATA. Area: Modern autoregressive KD / student distribution. [Primary metadata/export](https://proceedings.iclr.cc/paper_files/paper/4414-/bibtex); [publisher paper/page](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf).

Exact overlap: Student-generated sequences and flexible divergence address train/inference distribution mismatch.

Exact remaining distinction: We evaluate fixed-corpus next-token likelihood with a fixed objective, not on-policy generation or RL fine-tuning; no comparison against GKD was executed.

### xie2026adakd
Title: LLM-Oriented Token-Adaptive Knowledge Distillation.

Authors: Xie, Xurong and Xue, Zhucun and Wu, Jiafu and Li, Jian and Wang, Yabiao and Hu, Xiaobin and Liu, Yong and Zhang, Jiangning.

Year / venue: 2026 / AAAI. Identifiers: DOI:10.1609/aaai.v40i40.40701; DOI:10.1609/aaai.v40i40.40701.

Status: VERIFIED_METADATA. Area: Recent token-adaptive / student-state KD. [Primary metadata/export](https://doi.org/10.1609/aaai.v40i40.40701); [publisher paper/page](http://dx.doi.org/10.1609/aaai.v40i40.40701).

Exact overlap: Token difficulty and dynamic student learning state guide adaptive loss focusing and temperature.

Exact remaining distinction: The present synthesis does not newly introduce student-state dependence and contains no adaptive token method. Exact protocol overlap still requires full-text checking before a novelty claim.

### jin2026entropy
Title: Entropy-Aware On-Policy Distillation of Language Models.

Authors: Jin, Woogyeol and Min, Taywon and Yang, Yongjin and Wei, Dennis and Zhou, Yi and Kadhe, Swanand Ravindra and Baracaldo, Nathalie and Lee, Kimin.

Year / venue: 2026 / ICML. Identifiers: PMLR:v306-jin26e; arXiv:2603.07079.

Status: VERIFIED_METADATA. Area: Recent uncertainty-aware on-policy KD. [Primary metadata/export](https://proceedings.mlr.press/v306/jin26e.html); [publisher paper/page](https://proceedings.mlr.press/v306/jin26e.html).

Exact overlap: Teacher entropy informs a mixture of forward and reverse KL to maintain diversity and learning signals.

Exact remaining distinction: We study fixed forward KL and corpus likelihood, not on-policy reasoning or adaptive divergence. Diversity matters in prior algorithms; our bounded JSD result is not a general rejection of it.

## Remaining gaps before prose is finalized

Read the closest full papers, especially ATKD, MetaDistil, MiniLLM/GKD and
recent AdaKD, before asserting differences in initialization, CE comparators,
checkpoint selection or training budgets. The current abstract-level audit
does not prove that any prior work lacks a warm-start control. Do not claim
state-of-the-art KD, superiority over adaptive KD, or a new general theory.
The defensible position is the specific paired, multi-seed intervention and
falsification evidence chain assembled here. It is a bounded empirical study.

Dataset/model/architecture citations still need primary-source verification
before full manuscript writing (PTB, WT2, TinyStories, FineWeb-Edu, SmolLM2,
Transformer, Grassmann flow/Pluecker background and foundational KD).
These are `VERIFY_EXTERNAL` writing TODOs, not entries copied from the old
manuscript. A venue is not specified: `criteria_binding_unavailable`.
Do a focused literature refresh at the actual writing/submission date,
without launching experiments or implying a complete literature census.
