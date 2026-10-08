#!/usr/bin/env python3
"""Print proposed LaTeX tables as JSON; only formatting archived CSV records.

No model imports, endpoint recomputation, source mutations, or file writes.
Apply the returned text with apply_patch after review.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'research/manuscript_v2/tables'


def rows(name):
    with (PACKAGE / f'table_{name}.csv').open() as f:
        return list(csv.DictReader(f))


def tex(name, headers, values, caption, long=False):
    env = 'longtable' if long else 'tabular'
    lines = [r'\begingroup\small', r'\setlength{\tabcolsep}{5pt}']
    if not long:
        lines += [r'\begin{table}[htbp]\centering']
    lines += [rf'\begin{{{env}}}{{l'+ 'r'*(len(headers)-1)+'}']
    if long:
        lines += [rf'\caption{{{caption}}}\label{{tab:{name}}}\\']
    lines += [r'\toprule', ' & '.join(headers)+r' \\', r'\midrule']
    if long:
        lines += [r'\endfirsthead',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead',r'\bottomrule\endfoot']
    lines += [' & '.join(v)+r' \\' for v in values]
    if not long:
        lines += [r'\bottomrule']
    lines += [rf'\end{{{env}}}']
    if not long:
        lines += [rf'\caption{{{caption}}}\label{{tab:{name}}}',r'\end{table}']
    lines += [r'\endgroup']
    return '\n'.join(lines)+'\n'


def number(s):
    return f'{float(s):.6f}' if s else '--'


def main():
    files = {}
    labels={'PTB_fused':'PTB fused','TinyStories_fused':'TinyStories fused','WT2_J_fused_TG':'WT2 J/TG','WT2_A_fixed_composition':'WT2 A','WT2_Grassmann_only':'WT2 G','WT2_Transformer_only':'WT2 T','WT2_TT_homogeneous':'WT2 TT'}
    rr=rows('endpoint_inventory')
    values=[]
    for family,label in labels.items():
        group={r['student_seed']:r for r in rr if r['family']==family}
        values.append([label]+[number(group[s]['test_gain']) for s in ['42','123','456']])
    files['per_seed_gains.tex']=tex('per_seed_gains',['Condition','Seed 42','Seed 123','Seed 456'],values,'Archived paired test-NLL gains. Positive values favor KD; every WikiText-2 arm shares the same CE endpoint within a seed.')
    values=[[r['seed'],r['arm'].upper(),r['best_validation_step'],number(r['clip_fraction']),number(r['gradient_norm_preclip_mean']),number(r['gradient_norm_preclip_max'])] for r in rows('modern_optimization')]
    files['modern_optimization.tex']=tex('modern_optimization',['Seed','Arm','Step','Clip fraction','Mean norm','Max norm'],values,'FineWeb-Edu optimization audit. Step is selected by validation among trained checkpoints. Norms are measured before clipping; all recorded nonfinite and overflow fractions are zero.')
    values=[[r['predictor'].replace('&',r'\&'),number(r['auroc']),number(r['sign_accuracy']),number(r['spearman_rho'])] for r in rows('local_diagnostics') if r['level']=='seed']
    files['full_predictors.tex']=tex('full_predictors',['Predictor','AUROC','Sign accuracy',r'$\rho$'],values,'Full seed-level retrospective predictor comparison over 21 dependent conditions. A dash denotes a metric not defined for that predictor; it does not indicate zero.')
    values=[[r['Seed'],number(r['Teacher residual advantage']),number(r['KD mean KL']),number(r['CE mean grad norm']),number(r['KD mean grad norm']),number(r['KD mean clip fraction'])] for r in csv.DictReader((ROOT/'research/experiments/phase2g_tinystories_negative_transfer/results_multiseed.csv').open())]
    files['tinystories_optimization.tex']=tex('tinystories_optimization',['Seed','Residual','KD KL','CE norm','KD norm','KD clip'],values,'Primary TinyStories telemetry. Residual is teacher advantage over the CE test endpoint. KL includes temperature-squared scaling; norms and clip fractions are epoch means across the completed run.')
    # Read stored NLL fields, never derive them from rounded PPL or evaluate models.
    ptb=[('42','20260916_150712_h0_ptb_confirm_token_l0_seed42_retry50m_v2','20260916_150712_h0_ptb_confirm_token_l5_seed42_retry50m'),('123','20260916_152242_h0_ptb_confirm_token_l0_seed123','20260916_155134_h0_ptb_confirm_token_l5_seed123'),('456','20260916_153710_h0_ptb_confirm_token_l0_seed456','20260916_155348_h0_ptb_confirm_token_l5_seed456')]
    values=[]; sources={}
    for seed,ce,kd in ptb:
        records=[]
        for run in [ce,kd]:
            source=ROOT/'outputs/distill_experiments'/run/'summary.json'
            sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
            records.append(json.loads(source.read_text())['student'])
        c,k=records
        inventory=next(r for r in rr if r['family']=='PTB_fused' and r['student_seed']==seed)
        assert abs((c['test_loss']-k['test_loss'])-float(inventory['test_gain']))<5e-7
        values.append([seed,number(c['test_loss']),number(k['test_loss']),number(c['best_val_loss']),number(k['best_val_loss']),str(c['best_epoch']),str(k['best_epoch'])])
    files['ptb_endpoints.tex']=tex('ptb_endpoints',['Seed','CE test','KD test','CE val.','KD val.','CE epoch','KD epoch'],values,'PTB test and validation NLL from the archived selected-checkpoint summaries. The paired differences match the confirmatory inventory; epochs are independently validation-selected.')
    e=list(csv.DictReader((ROOT/'research/experiments/phase2e_teacher_branch_ablation/results_multiseed.csv').open()))
    d=list(csv.DictReader((ROOT/'research/experiments/phase2d_fixed_composition_teacher_quality/results_multiseed.csv').open()))
    f=list(csv.DictReader((ROOT/'research/experiments/phase2f_homogeneous_ensemble_control/results_multiseed.csv').open()))
    # In the original D CSV, "Teacher ... test NLL" means the KD student endpoint.
    values=[]
    for x,y,z in zip(e,d,f):
        assert x['Seed']==y['Seed']==z['Seed']
        values.append([x['Seed']]+[number(x[k]) for k in ['C0 test NLL','F test NLL','T test NLL','G test NLL']]+[number(y['Teacher A test NLL']),number(z['TT test NLL'])])
    files['wt2_endpoints.tex']=tex('wt2_endpoints',['Seed','CE','KD(J/TG)','KD(T)','KD(G)','KD(A)','KD(TT)'],values,'WikiText-2 selected student test NLL. J/TG is the same reused fused endpoint, not two experiments. All KD columns report student endpoints, not teacher likelihood.')
    values=[]
    for r in rows('modern_per_seed'):
        for scope in ['validation','test']:
            values.append([r['seed'],scope]+[number(r[f'{arm}_{scope}_nll']) for arm in ['s0','ce','kd1','kd5']])
    files['modern_endpoints.tex']=tex('modern_endpoints',['Seed','Split',r'$S_0$','CE','KD1','KD5'],values,'Frozen FineWeb-Edu primary endpoints. Validation selection excludes step 0 for CE/KD. Test is evaluated only after that selection; the separate best-including-S0 diagnostic selects S0 in all nine arms.')
    for folder,run in [('hybrid_experiments','20260328_080104_wt2_hybrid_alpha_joint'),('hybrid_experiments','20260328_084319_wt2_v1_hybrid_alpha_only'),('hybrid_experiments','20260930_034520_phase2f_tt_teacher_joint_e10'),('experiments','20260317_114514_wt2_baseline_both'),('experiments','20260930_033237_phase2f_t2_seed123_e20')]:
        source=ROOT/'outputs'/folder/run/'config.json'
        sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    files['endpoint_summary_sources.json']=json.dumps({'activity':'Read stored NLL summaries and original teacher/source configs only; no endpoint computation or model access.','sources':sources},indent=2)+'\n'
    print(json.dumps(files))


if __name__ == '__main__':
    main()
