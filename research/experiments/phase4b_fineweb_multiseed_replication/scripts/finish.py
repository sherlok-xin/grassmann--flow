"""Freeze validation-only diagnostic choices, then evaluate selected test states."""
import gc
import json
import math
import torch
from train import ART, OUT, RUNS, load, sha, stream, evaluate
from selection import including_s0


def run_root(seed):
    return OUT / 'runs/fineweb' if seed == 42 else RUNS / str(seed)


def main():
    torch.set_num_threads(4)
    summaries, choices = {}, []
    teacher_path = OUT / 'runs/fineweb/teacher/selected.pt'
    expected_teacher = 'c7411708b61e6524ef05bd24e9e618f19fcc119e9a3c0664c5f4f169dbf16928'
    assert sha(teacher_path) == expected_teacher
    for seed in [42,123,456]:
        root = run_root(seed)
        summaries[seed] = {arm: json.loads((root/arm/'summary.json').read_text())
                           for arm in ['s0','ce','kd1','kd5']}
        ss = summaries[seed]
        assert all(s['status']=='TRAIN_COMPLETE_VALIDATION_SELECTED' and
                   s['consumed_target_tokens']==10_000_000 for s in ss.values())
        assert len({ss[a]['s0_sha256'] for a in ['ce','kd1','kd5']}) == 1
        assert all(ss[a]['s0_sha256']==ss['s0']['selected_sha256'] for a in ['ce','kd1','kd5'])
        assert len({ss[a]['order_sha256'] for a in ['ce','kd1','kd5']}) == 1
        assert all(ss[a]['teacher_sha256']==expected_teacher for a in ['kd1','kd5'])
        for arm,s in ss.items():
            assert sha(root/arm/'selected.pt') == s['selected_sha256']
            rows = [json.loads(r) for r in (root/arm/'metrics.jsonl').read_text().splitlines()]
            trained = [r for r in rows[1:] if 'validation_nll' in r]
            selected = min(trained,key=lambda r:(r['validation_nll'],r['step']))
            assert selected['step']==s['best_validation_step']
            assert selected['validation_nll']==s['best_validation_nll']
            assert s['nonfinite_fraction']==0
        for arm in ['ce','kd1','kd5']:
            s = ss[arm]
            assert abs(s['initial_validation_nll']-ss['s0']['best_validation_nll'])<1e-5
            step = including_s0(ss['s0']['best_validation_nll'],s['best_validation_nll'],s['best_validation_step'])
            chosen = ss['s0'] if step==0 else s
            choices.append({'seed':seed,'arm':arm,'selected_step':step,
                            'selected_source':'s0' if step==0 else arm,
                            'selected_sha256':chosen['selected_sha256'],
                            'selection_validation_nll':chosen['best_validation_nll'],
                            'primary_selected_step':s['best_validation_step'],
                            's0_preparation_selected_step':ss['s0']['best_validation_step']})
    # This artifact contains no test metric and is saved before any new test forward.
    selection = {'selection':'best_including_S0','diagnostic_only':True,
                 'primary_endpoint_changed':False,'uses_test_for_selection':False,
                 'choices':choices}
    (ART/'selection_with_s0.json').write_text(json.dumps(selection,indent=2)+'\n')
    test_file = OUT/'data/fineweb/test.bin'
    manifest = json.loads((OUT/'data/fineweb/manifest.json').read_text())
    assert sha(test_file)==manifest['sha256']['test']
    reused = json.loads((OUT/'runs/fineweb/final_result.json').read_text())
    assert reused['test_evaluation_completed']
    for seed in [123,456]:
        root = run_root(seed)
        if (root/'final_result.json').exists():
            raise RuntimeError('Refuse to overwrite completed test evaluation')
        result = {'dataset':'fineweb','seed':seed,'arms':{},'teacher_sha256':expected_teacher,
                  'teacher_endpoint_reused_from_phase4a':reused['arms']['teacher'],
                  'test_evaluation_completed':False,'diagnostic_selection_artifact':'selection_with_s0.json'}
        for arm in ['s0','ce','kd1','kd5']:
            s = summaries[seed][arm]
            model = load('SmolLM2-135M',root/arm/'selected.pt')
            val = evaluate(model,stream('fineweb','validation'))
            assert abs(val-s['best_validation_nll'])<1e-5
            test = evaluate(model,stream('fineweb','test'))
            result['arms'][arm]={'validation_nll':val,'test_nll':test,'test_ppl':math.exp(test),
                                 'selected_sha256':s['selected_sha256'],'best_step':s['best_validation_step']}
            print(seed,arm,result['arms'][arm],flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for arm in ['kd1','kd5']:
            gain = result['arms']['ce']['test_nll']-result['arms'][arm]['test_nll']
            vg = result['arms']['ce']['validation_nll']-result['arms'][arm]['validation_nll']
            result['arms'][arm].update({'gain_test_nll':gain,'gain_validation_nll':vg,
                                        'validation_test_direction_agrees':gain*vg>0})
        result['numerically_valid']=True
        result['test_evaluation_completed']=True
        (root/'final_result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert sha(teacher_path)==expected_teacher
    print('BOTH_NEW_SEEDS_FINAL_TEST_COMPLETE',flush=True)


if __name__=='__main__':
    main()
