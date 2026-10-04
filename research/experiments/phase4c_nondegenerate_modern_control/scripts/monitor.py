"""Read-only exact-PID and telemetry monitor; never matches process text."""
import json
from pathlib import Path
from protocol import OUT, STEPS

registry_path = OUT/'ce_launcher_registry.json'
if not registry_path.exists():
    print(json.dumps({'status':'LAUNCHER_STARTING'}),flush=True)
else:
    registry = json.loads(registry_path.read_text())
    result = []
    for entry in registry['children']:
        path = OUT/'runs'/entry['arm']
        rows = []
        if (path/'metrics.jsonl').exists():
            for line in (path/'metrics.jsonl').read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # In-progress final line only; formal collector is strict.
        last = rows[-1] if rows else {}
        proc = Path('/proc')/str(entry['pid'])/'cmdline'
        command = proc.read_bytes().replace(b'\x00',b' ').decode() if proc.exists() else ''
        live = 'phase4c_nondegenerate_modern_control/scripts/train.py' in command and entry['arm'] in command
        validations = [{'step':r['step'],'nll':r['validation_nll']} for r in rows if 'validation_nll' in r]
        trained = [r for r in rows if 'gradient_norm_preclip' in r]
        result.append({'arm':entry['arm'],'gpu':entry['gpu'],'pid':entry['pid'],'owned_process_live':live,
                       'step':last.get('step'),'total_steps':STEPS,'targets':last.get('tokens'),
                       'validation':validations,'summary_exists':(path/'summary.json').exists(),
                       'failed':(OUT/('FAILURE_'+entry['arm']+'.json')).exists(),
                       'clip_fraction_so_far':sum(r['clipped'] for r in trained)/len(trained) if trained else None,
                       'nonfinite_observed':any(r.get('nonfinite',False) for r in trained)})
    completion = OUT/'ce_launcher_complete.json'
    print(json.dumps({'runs':result,'completion':json.loads(completion.read_text()) if completion.exists() else None}),flush=True)
