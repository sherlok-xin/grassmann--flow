"""Read-only bounded-run monitoring; no training, selection or process changes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/phase4b_fineweb'
progress=[]
completed=0
registry={}
if (OUT/'process_registry.txt').exists():
    for line in (OUT/'process_registry.txt').read_text().splitlines():
        pid,seed,arm,gpu=line.split()
        command=Path('/proc')/pid/'cmdline'
        try:
            active=b'phase4b_fineweb_multiseed_replication/scripts/train.py' in command.read_bytes()
        except FileNotFoundError:
            active=False
        registry[(int(seed),arm)]={'pid':int(pid),'gpu':int(gpu),'active':active}
for seed in [123,456]:
    for arm in ['s0','ce','kd1','kd5']:
        run=OUT/'runs'/str(seed)/arm
        if (run/'summary.json').exists():
            s=json.loads((run/'summary.json').read_text())
            completed+=1
            progress.append({'seed':seed,'arm':arm,'status':'COMPLETE',
                             'tokens':s['consumed_target_tokens'],'selected_step':s['best_validation_step'],
                             'clip_fraction':s['clip_fraction'],'nonfinite_fraction':s['nonfinite_fraction']})
        elif (run/'metrics.jsonl').exists():
            text=(run/'metrics.jsonl').read_text()
            lines=text.splitlines()
            if text and not text.endswith('\n'):
                lines=lines[:-1]
            if lines:
                r=json.loads(lines[-1])
                progress.append({'seed':seed,'arm':arm,'status':'RUNNING' if registry.get((seed,arm),{}).get('active',False) else 'EXITED_WITHOUT_SUMMARY',
                                 'step':r['step'],'tokens':r['tokens'],
                                 'gradient_norm_preclip':r.get('gradient_norm_preclip')})
        elif (seed,arm) in registry:
            progress.append({'seed':seed,'arm':arm,
                             'status':'LOADING' if registry[(seed,arm)]['active'] else 'EXITED_WITHOUT_SUMMARY'})
log=(OUT/'launcher.log').read_text() if (OUT/'launcher.log').exists() else ''
done='PHASE4B_TRAINING_AND_TEST_COMPLETE' in log
finals=[(OUT/'runs'/str(seed)/'final_result.json').exists() for seed in [123,456]]
metadata=json.loads((Path(__file__).resolve().parents[1]/'launch_metadata.json').read_text())
try:
    launcher_active=b'run_replication.sh' in (Path('/proc')/str(metadata['launcher_pid'])/'cmdline').read_bytes()
except FileNotFoundError:
    launcher_active=False
print(json.dumps({'completed_new_runs':completed,'progress':progress,
                  'new_seed_final_tests_complete':all(finals),'launcher_complete':done,
                  'launcher_active':launcher_active}))
