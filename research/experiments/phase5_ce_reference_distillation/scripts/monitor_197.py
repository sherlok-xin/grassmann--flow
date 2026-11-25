"""Hourly, read-only GPU/queue monitor. No training, repair or notifications.

Run inside the existing 197 research container. Logs belong only to Phase 5.
"""
import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'outputs/phase5/monitoring'
PHASE = ROOT / 'research/experiments/phase5_ce_reference_distillation'
INTERVAL = 3600
TZ = ZoneInfo('Asia/Shanghai')


def query(kind, fields):
    result = subprocess.run(['nvidia-smi', f'--query-{kind}=' + ','.join(fields),
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=30)
    return [dict(zip(fields, (v.strip() for v in row)))
            for row in csv.reader(io.StringIO(result.stdout)) if row]


def snapshot():
    now = datetime.now(TZ)
    record = dict(checked_at=now.isoformat(timespec='seconds'), timezone='Asia/Shanghai',
                  server='10.42.0.197', interval_seconds=INTERVAL,
                  next_check_at=(now + timedelta(seconds=INTERVAL)).isoformat(timespec='seconds'),
                  monitor_pid=os.getpid(), mode='read_only_no_GPU_allocation')
    errors = []
    for name, kind, fields in [
        ('gpus', 'gpu', ['index', 'uuid', 'name', 'memory.total', 'memory.used',
                        'utilization.gpu', 'temperature.gpu']),
        ('compute_processes', 'compute-apps', ['gpu_uuid', 'pid', 'used_memory']),
    ]:
        try:
            record[name] = query(kind, fields)
        except (subprocess.SubprocessError, OSError) as exc:
            errors.append(dict(component=name, error=str(exc)))
    try:
        queues = json.loads((ROOT / 'outputs/phase5/queue_processes.json').read_text())
        record['queues'] = []
        for entry in queues['new_waiters']:
            process = Path('/proc') / str(entry['pid'])
            try:
                argv = [v.decode() for v in (process / 'cmdline').read_bytes().split(b'\0') if v]
                matches = argv == entry['command']
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                matches = False
            record['queues'].append(dict(script=entry['script'], gpu=entry['gpu'],
                                         pid=entry['pid'], matching_process_alive=matches))
        ledger = json.loads((ROOT / 'outputs/phase5/accounting.json').read_text())
        record['active_runs'] = {name: {'gpu': row['gpu'], 'pid': row.get('pid'),
                                       'started_unix': row['started_unix']}
                                 for name, row in ledger['runs'].items() if row['status'] == 'RUNNING'}
        record['recorded_gpu_hours'] = ledger['smoke_reserve_gpu_hours'] + sum(
            row.get('gpu_hours', 0.) for row in ledger['runs'].values())
        record['gpu_hour_ceiling'] = ledger['ceiling']
        record['assets_download_complete'] = json.loads((PHASE / 'modern_assets.json').read_text())['download_complete']
        record['assets_verified'] = (PHASE / 'modern_assets_verified.json').exists()
    except (OSError, KeyError, ValueError) as exc:
        errors.append(dict(component='phase5_status', error=str(exc)))
    record['errors'] = errors
    record['status'] = 'CHECK_COMPLETE' if not errors else 'CHECK_WITH_ERRORS'
    with (OUTPUT / 'monitor_197.jsonl').open('a') as log:
        log.write(json.dumps(record) + '\n')
    tmp = OUTPUT / 'monitor_197_latest.tmp'
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(OUTPUT / 'monitor_197_latest.json')
    print(json.dumps({'checked_at': record['checked_at'], 'status': record['status'],
                      'next_check_at': record['next_check_at']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'monitor_197.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUTPUT / 'monitor_197_process.json').write_text(json.dumps(dict(
            pid=os.getpid(), interval_seconds=INTERVAL, command=__file__,
            started_at=datetime.now(TZ).isoformat(timespec='seconds')), indent=2) + '\n')
        while True:
            started = time.monotonic()
            snapshot()
            if args.once:
                return
            time.sleep(max(1., INTERVAL - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
