"""Launch only the three preregistered CE arms; never launch KD."""
import json
import os
import subprocess
import sys
import time
from protocol import *

smoke = json.loads((ART / 'ce_smoke_audit.json').read_text())
assert smoke['status'] == 'SMOKE_PASS' and smoke['test_evaluated'] is False
assert smoke['lambda'] == 0 and smoke['consumed_target_tokens'] == 16384
verify_inputs()
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / 'ce_launcher_registry.json').exists(), 'No duplicate launcher'
memory = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(v.strip()) < 500 for v in memory.splitlines()[:3]), 'GPU0/1/2 not idle; do not displace other jobs'
children, registry, logs = [], [], []
for gpu, (arm, lr) in enumerate(zip(CE_ARMS, LRS)):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    logfile = (OUT / (arm + '.log')).open('x')
    logs.append(logfile)
    command = [sys.executable,'-u',str(ART/'scripts/train.py'),arm,'--lr',str(lr)]
    proc = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT)
    children.append(proc)
    registry.append({'pid':proc.pid,'gpu':gpu,'arm':arm,'lr':lr,'command':command})
(OUT / 'ce_launcher_registry.json').write_text(json.dumps({'launcher_pid':os.getpid(),'children':registry},indent=2)+'\n')
print(json.dumps(registry),flush=True)
while any(proc.poll() is None for proc in children):
    if any(proc.poll() not in (None,0) for proc in children):
        # Stop only sibling PIDs owned by this launcher, never other users.
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        break
    time.sleep(5)
codes = [proc.wait() for proc in children]
for logfile in logs:
    logfile.close()
(OUT / 'ce_launcher_complete.json').write_text(json.dumps({'exit_codes':codes,'status':'PASS' if codes==[0,0,0] else 'FAIL'},indent=2)+'\n')
print('PHASE4C_CE_GATE_RUNS_COMPLETE', codes, flush=True)
sys.exit(0 if codes == [0,0,0] else 1)
