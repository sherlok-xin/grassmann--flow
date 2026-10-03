"""Read-only integrity audit: budgets, finite telemetry, selection and provenance."""
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from selection import including_s0, decisions

ART=Path(__file__).resolve().parents[1]
ROOT=ART.parents[2]
OLD=ROOT/'outputs/phase4a_modern/runs/fineweb'
NEW=ROOT/'outputs/phase4b_fineweb/runs'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    initial=json.loads((ART/'preflight_audit.json').read_text())
    data=ROOT/'outputs/phase4a_modern/data/fineweb'
    for split,h in initial['data_sha256'].items():
        assert sha(data/(split+'.bin'))==h
    assert sha(data/'document_order.jsonl.gz')==initial['document_order_sha256']
    assert sha(OLD/'teacher/selected.pt')==initial['teacher_sha256']
    frozen=sorted(p for p in (ROOT/'research/final_evidence').iterdir() if p.is_file())
    listing=''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in frozen)
    assert hashlib.sha256(listing.encode()).hexdigest()==initial['final_evidence_aggregate_sha256']
    assert sha(ROOT/'论文投稿/cac/conference_101719.tex')==initial['manuscript_sha256']
    results=list(csv.DictReader((ART/'results_multiseed.csv').open()))
    diagnostic=list(csv.DictReader((ART/'best_including_s0.csv').open()))
    choices=json.loads((ART/'selection_with_s0.json').read_text())['choices']
    s0_hashes,preparation_orders,continuation_orders=[],[],[]
    counts={'reused_runs':0,'new_runs':0,'new_target_tokens':0}
    for seed in [42,123,456]:
        root=OLD if seed==42 else NEW/str(seed)
        final=json.loads((root/'final_result.json').read_text())
        assert final['test_evaluation_completed'] and final['numerically_valid']
        ss={a:json.loads((root/a/'summary.json').read_text()) for a in ['s0','ce','kd1','kd5']}
        s0_hashes.append(ss['s0']['selected_sha256'])
        preparation_orders.append(ss['s0']['order_sha256'])
        r=next(r for r in results if int(r['seed'])==seed)
        for arm,s in ss.items():
            assert s['seed']==seed and s['steps']==1221 and s['consumed_target_tokens']==10_000_000
            assert s['gradient_clip']==1 and s['temperature']==2
            assert s['order_seed']==(seed if arm=='s0' else 4242)
            assert sha(root/arm/'selected.pt')==s['selected_sha256']==final['arms'][arm]['selected_sha256']
            rows=[json.loads(x) for x in (root/arm/'metrics.jsonl').read_text().splitlines()]
            assert len(rows)==1222 and not rows[0]['eligible_for_selection']
            assert [x['step'] for x in rows]==list(range(1222))
            assert rows[-1]['tokens']==10_000_000
            assert [x['step'] for x in rows[1:] if 'validation_nll' in x]==[256,512,768,1024,1221]
            best=min((x for x in rows[1:] if 'validation_nll' in x),key=lambda x:(x['validation_nll'],x['step']))
            assert best['step']==s['best_validation_step'] and best['validation_nll']==s['best_validation_nll']
            assert all(all(math.isfinite(x[k]) for k in ['ce','scaled_kl','gradient_norm_preclip','lr']) for x in rows[1:])
            assert s['clip_fraction']==sum(x['clipped'] for x in rows[1:])/1221
            assert s['nonfinite_fraction']==s['overflow_fraction']==0
            assert rows[1]['lr']>0 and rows[-1]['lr']==0
            if arm!='s0':
                assert s['s0_sha256']==ss['s0']['selected_sha256']
                continuation_orders.append(s['order_sha256'])
            if arm in ['kd1','kd5']:
                assert s['teacher_sha256']==initial['teacher_sha256']
                assert float(r['Gain_'+arm[-1]])==final['arms']['ce']['test_nll']-final['arms'][arm]['test_nll']
            with gzip.open(ART/'raw'/f'seed{seed}_{arm}_metrics.jsonl.gz','rt') as f:
                assert [json.loads(x) for x in f]==rows
            assert json.loads((ART/'raw'/f'seed{seed}_{arm}_summary.json').read_text())==s
            if seed==42:
                counts['reused_runs']+=1
            else:
                counts['new_runs']+=1
                counts['new_target_tokens']+=s['consumed_target_tokens']
        for arm in ['ce','kd1','kd5']:
            choice=next(c for c in choices if c['seed']==seed and c['arm']==arm)
            assert choice['selected_step']==including_s0(ss['s0']['best_validation_nll'],ss[arm]['best_validation_nll'],ss[arm]['best_validation_step'])
            d=next(d for d in diagnostic if int(d['seed'])==seed and d['arm']==arm)
            assert d['selected_source']==choice['selected_source']
            assert float(d['test_nll'])==final['arms'][choice['selected_source']]['test_nll']
            assert d['primary_endpoint_changed']=='False'
    assert len(set(s0_hashes))==len(set(preparation_orders))==3
    assert len(set(continuation_orders))==1
    assert counts=={'reused_runs':4,'new_runs':8,'new_target_tokens':80_000_000}
    stats=list(csv.DictReader((ART/'summary_statistics.csv').open()))
    for row in stats:
        values=[float(r[row['metric']]) for r in results]
        assert float(row['mean'])==statistics.mean(values)
        assert float(row['sample_sd'])==statistics.stdev(values)
        assert int(row['negative_signs'])==sum(v<0 for v in values)
        assert int(row['positive_signs'])==sum(v>0 for v in values)
    flags=decisions([float(r['Gain_1']) for r in results],
                    [float(r['Gain_5']) for r in results],
                    [r['gain5_validation_test_agrees']=='True' for r in results])
    decision=json.loads((ART/'decision.json').read_text())
    assert decision['flags']==flags and decision['decision']==flags[-1]
    assert decision['best_including_s0_count']==sum(int(r['selected_step'])==0 for r in diagnostic)
    event=json.loads((ART/'raw/ce456_gradient_event.json').read_text())
    ce456=[json.loads(x) for x in (NEW/'456/ce/metrics.jsonl').read_text().splitlines()][1:]
    peak=max(ce456,key=lambda r:r['gradient_norm_preclip'])
    assert peak['step']==event['event_step'] and peak['gradient_norm_preclip']==event['event_gradient_norm_preclip']
    assert sum(r['gradient_norm_preclip']>2 for r in ce456)==event['norms_above_2_count']
    result={'status':'PASS',**counts,'independent_adapted_s0_states':3,
            'matched_continuation_orders':True,'teacher_sha256':initial['teacher_sha256'],
            'data_unchanged':True,'frozen_evidence_unchanged':True,'manuscript_unchanged':True,
            'validation_only_diagnostic_selection':True,'paired_statistics_and_decision_verified':True,
            'gradient_event_telemetry_verified':True,'test_model_forwards':0}
    (ART/'completed_run_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
