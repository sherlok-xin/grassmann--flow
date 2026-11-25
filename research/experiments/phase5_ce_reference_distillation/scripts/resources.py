"""Cooperative idle wait; never alters GPU clocks or terminates another process."""
import json
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

POLL_SECONDS = 3600

def wait_idle(gpu, hours=None):
    started=time.monotonic()
    while True:
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<=256:return
        elapsed=time.monotonic()-started
        if hours is not None and elapsed>hours*3600:
            raise TimeoutError('shared GPU did not become idle; no GPU allocated')
        print(json.dumps({'status':'WAITING_FOR_IDLE_GPU','gpu':gpu,'used_MiB':used,
            'wait_seconds':elapsed,'poll_seconds':POLL_SECONDS,
            'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}),flush=True)
        time.sleep(POLL_SECONDS)
