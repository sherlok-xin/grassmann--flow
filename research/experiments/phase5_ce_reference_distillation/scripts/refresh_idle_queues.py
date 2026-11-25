"""Replace only our five verified CPU waiters with durable detached waiters.

This maintenance command refuses to operate if a GPU is available, a Phase 5
allocation is active, or any expected supervisor is absent/duplicated. It never
terminates a training process or another project's process.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def main():
    from resources import POLL_SECONDS
    specifications = {'wt2_queue.py': 0, 'ts_queue.py': 1, 'modern_queue.py': 2,
                      'wt2_followups.py': 0, 'legacy_confirmations.py': 0}
    ledger = json.loads((ROOT / 'outputs/phase5/accounting.json').read_text())
    if any(row['status'] == 'RUNNING' for row in ledger['runs'].values()):
        raise RuntimeError('active allocation: do not refresh')
    for gpu in {0, 1, 2}:
        memory = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if memory <= 256:
            raise RuntimeError('GPU may launch a child; refresh deferred')
    found = {name: [] for name in specifications}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = [v.decode() for v in (process / 'cmdline').read_bytes().split(b'\0') if v]
            if len(argv) < 3 or not Path(argv[0]).name.startswith('python'):
                continue
            if (process / 'cwd').resolve() != ROOT:
                continue
            for arg in argv[1:]:
                candidate = Path(arg)
                if candidate.name in found and (ROOT / candidate).resolve() == HERE / candidate.name:
                    found[candidate.name].append((int(process.name), argv))
                if candidate.name in {'train_legacy.py', 'train_modern.py', 'fork_legacy.py'}:
                    raise RuntimeError('model child found: refresh forbidden')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    if any(len(v) != 1 for v in found.values()):
        raise RuntimeError(f'expected exactly one of each own waiter: {found}')
    # The checks above all precede the first mutation.
    for matches in found.values():
        os.kill(matches[0][0], signal.SIGTERM)
    time.sleep(1)
    logs = ROOT / 'outputs/phase5/launcher_logs'
    launched = []
    for name, gpu in specifications.items():
        env = dict(os.environ, OMP_NUM_THREADS='4')
        if name == 'modern_queue.py':
            env['PYTHONPATH'] = 'outputs/phase5/python_deps'
        command = [sys.executable, '-u', str(HERE / name), '--gpu', str(gpu)]
        with (logs / ('supervisor_' + name.removesuffix('.py') + '.log')).open('a') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        launched.append(dict(script=name, gpu=gpu, pid=child.pid, command=command))
    record = dict(time_unix=time.time(), host='197 grassmann_lab',
                  reason='user requested hourly monitoring of 197; refresh only verified CPU waiters',
                  resource_poll_seconds=POLL_SECONDS,
                  old_waiter_pids={name: values[0][0] for name, values in found.items()},
                  new_waiters=launched, training_hyperparameters_changed=False,
                  no_other_processes_modified=True)
    (ROOT / 'outputs/phase5/queue_processes.json').write_text(json.dumps(record, indent=2) + '\n')
    with (ROOT / 'outputs/phase5/queue_refresh_history.jsonl').open('a') as history:
        history.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
