"""Shared Phase 5 GPU-wall-time ledger, including failed analyses and probes."""
import fcntl
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


class Budget:
    def __init__(self, name, gpu, maximum_hours):
        self.name, self.gpu, self.maximum = name, gpu, maximum_hours
        self.root = ROOT / 'outputs/phase5'

    def edit(self, fn):
        with (self.root / 'accounting.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'accounting.json'
            data = json.loads(path.read_text())
            fn(data)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2) + '\n')
            tmp.replace(path)

    def __enter__(self):
        self.start = time.monotonic()
        def reserve(d):
            if self.name in d['runs']:
                raise ValueError('duplicate budget/run identifier')
            used = d['smoke_reserve_gpu_hours'] + sum(x.get('gpu_hours', x['reservation']) for x in d['runs'].values())
            if used + self.maximum > d['ceiling']:
                raise RuntimeError('Phase 5 GPU-hour ceiling')
            d['runs'][self.name] = dict(status='RUNNING', reservation=self.maximum,
                gpu=self.gpu, pid=os.getpid(), started_unix=time.time())
        self.edit(reserve)
        return self

    def check(self):
        if (time.monotonic() - self.start) / 3600 > self.maximum:
            raise TimeoutError('reserved run wall-time exhausted')

    def __exit__(self, typ, value, traceback):
        def finish(d):
            d['runs'][self.name].update(status='COMPLETE' if typ is None else 'FAILED',
                gpu_hours=(time.monotonic()-self.start)/3600, finished_unix=time.time())
        self.edit(finish)

