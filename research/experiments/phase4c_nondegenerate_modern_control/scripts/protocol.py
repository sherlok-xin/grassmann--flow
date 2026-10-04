"""Frozen Phase4C bookkeeping; no model or test access."""
import hashlib
import json
import math
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
BASE = ROOT / 'research/experiments/phase4a_modern_generalization_pilot'
OUT = ROOT / 'outputs/phase4c_nondegenerate'
OLD = ROOT / 'outputs/phase4a_modern'
S0 = OLD / 'runs/fineweb/s0/selected.pt'
TEACHER = OLD / 'runs/fineweb/teacher/selected.pt'
S0_SHA = 'f497cf2310050b163ecb167f75ae4f89c2780d4165aa8f25ff77025715f4bd28'
TEACHER_SHA = 'c7411708b61e6524ef05bd24e9e618f19fcc119e9a3c0664c5f4f169dbf16928'
BUDGET = 8_000_000
STEPS = math.ceil(BUDGET / (32 * 256))
CHECK_STEPS = [0, 256, 512, 768, STEPS]
LRS = [5e-6, 1e-5, 2e-5]
CE_ARMS = ['ce_lr5e-6', 'ce_lr1e-5', 'ce_lr2e-5']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select(rows):
    return min(rows, key=lambda r: (r['validation_nll'], r['step']))


def gate(summaries):
    best = min(summaries, key=lambda s: (s['best_validation_nll'], s['optimizer']['lr']))
    gain = best['initial_validation_nll'] - best['best_validation_nll']
    passed = best['best_validation_step'] > 0 and gain >= 0.01
    return best, gain, passed


def manifest():
    path = ART / 'continuation_manifest.json'
    audit = json.loads((ART / 'input_audit.json').read_text())
    assert sha(path) == audit['continuation_manifest_sha256']
    return json.loads(path.read_text()), audit


def verify_inputs():
    m, audit = manifest()
    assert sha(S0) == S0_SHA and sha(TEACHER) == TEACHER_SHA
    for split, expected in m['split_sha256'].items():
        assert sha(OLD / 'data/fineweb' / (split + '.bin')) == expected
    assert sha(BASE / 'scripts/train.py') == 'd7d8ad350f45164b657daf1895b38b3f49e2e2e9466996961641a653a0c1c565'
    assert sha(BASE / 'scripts/core.py') == '427c90edc3ba353d929b2d75a942d67779429bf2707525ddb55c271e8ce8c3cf'
    assert sha(ROOT / '论文投稿/cac/conference_101719.tex') == '8b8e3fcc3f04b596058cd4b63ddfe38b6eab91f338c7d4f12db4b602c9982ec8'
    aggregate = ''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in sorted((ROOT / 'research/final_evidence').iterdir()))
    assert hashlib.sha256(aggregate.encode()).hexdigest() == '3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96'
    return m, audit
