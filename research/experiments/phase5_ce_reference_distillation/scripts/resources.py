"""Cooperative idle wait; never alters GPU clocks or terminates another process."""
import json
import subprocess
import time


def wait_idle(gpu, hours=24):
    started=time.monotonic();last=-1
    while True:
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<=256:return
        elapsed=time.monotonic()-started
        if elapsed>hours*3600:raise TimeoutError('shared GPU did not become idle; no GPU allocated')
        if int(elapsed//300)!=last:
            print(json.dumps({'status':'WAITING_FOR_IDLE_GPU','gpu':gpu,'used_MiB':used,'wait_seconds':elapsed}),flush=True)
            last=int(elapsed//300)
        time.sleep(30)
