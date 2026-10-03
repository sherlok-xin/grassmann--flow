"""Verify immutable Phase 4A inputs before any Phase 4B endpoint training."""
import hashlib
import json
from pathlib import Path
import sys

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
P4A = ROOT / 'research/experiments/phase4a_modern_generalization_pilot'
OUT = ROOT / 'outputs/phase4a_modern'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((P4A / 'data_manifest_fineweb.json').read_text())
    actual = json.loads((OUT / 'data/fineweb/manifest.json').read_text())
    assert manifest == actual
    checked = {s: sha(OUT / 'data/fineweb' / (s + '.bin')) for s in ['train','validation','test']}
    assert checked == manifest['sha256']
    teacher = OUT / 'runs/fineweb/teacher/selected.pt'
    assert sha(teacher) == 'c7411708b61e6524ef05bd24e9e618f19fcc119e9a3c0664c5f4f169dbf16928'
    frozen = sorted(p for p in (ROOT / 'research/final_evidence').iterdir() if p.is_file())
    listing = ''.join(sha(p) + '  ' + str(p.relative_to(ROOT)) + '\n' for p in frozen)
    digest = hashlib.sha256(listing.encode()).hexdigest()
    assert digest == '3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96'
    result = {'status': 'PASS', 'teacher_sha256': sha(teacher),
              'data_sha256': checked, 'document_order_sha256': manifest['document_order_sha256'],
              'final_evidence_aggregate_sha256': digest,
              'manuscript_sha256': sha(ROOT / '论文投稿/cac/conference_101719.tex'),
              'phase4a_train_sha256': sha(P4A / 'scripts/train.py'),
              'phase4a_core_sha256': sha(P4A / 'scripts/core.py'),
              'no_model_forward': True, 'new_training_seeds': [123,456]}
    (ART / 'preflight_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
