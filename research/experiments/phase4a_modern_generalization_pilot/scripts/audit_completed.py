"""Read-only endpoint/telemetry/data/state integrity checks; no model forward."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
OUT = ROOT / 'outputs/phase4a_modern'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    result = {'checks': {}, 'total_formal_target_tokens': 0, 'runs': 0, 'test_model_forwards': 0}
    for dataset in ['tinystories', 'fineweb']:
        data = OUT / 'data' / dataset
        manifest = json.loads((data / 'manifest.json').read_text())
        for split, n in manifest['token_counts'].items():
            assert (data / (split + '.bin')).stat().st_size == n * 4
            assert sha(data / (split + '.bin')) == manifest['sha256'][split]
        assert sha(data / 'document_order.jsonl.gz') == manifest['document_order_sha256']
        with gzip.open(data / 'document_order.jsonl.gz', 'rt') as f:
            docs = [json.loads(s) for s in f]
        ids = {split: {r['id'] for r in docs if r['split'] == split} for split in manifest['token_counts']}
        assert not (ids['train'] & ids['validation'] or ids['train'] & ids['test'] or ids['validation'] & ids['test'])
        assert len({r['text_sha256'] for r in docs}) == len(docs)
        assert {split: sum(r['tokens_used'] for r in docs if r['split'] == split) for split in ids} == manifest['token_counts']
        root = OUT / 'runs' / dataset
        endpoint = json.loads((root / 'final_result.json').read_text())
        assert endpoint['test_evaluation_completed'] and endpoint['numerically_valid']
        hashes, order_hashes = [], []
        for arm in ['teacher', 's0', 'ce', 'kd1', 'kd5']:
            summary = json.loads((root / arm / 'summary.json').read_text())
            assert summary['consumed_target_tokens'] == 10_000_000 and summary['steps'] == 1221
            assert sha(root / arm / 'selected.pt') == summary['selected_sha256'] == endpoint['arms'][arm]['selected_sha256']
            records = [json.loads(s) for s in (root / arm / 'metrics.jsonl').read_text().splitlines()]
            assert len(records) == 1222 and not records[0]['eligible_for_selection']
            assert records[-1]['tokens'] == 10_000_000
            assert [r['step'] for r in records] == list(range(1222))
            assert [r['step'] for r in records[1:] if 'validation_nll' in r] == [256, 512, 768, 1024, 1221]
            selected = min((r for r in records[1:] if 'validation_nll' in r), key=lambda r: (r['validation_nll'], r['step']))
            assert selected['step'] == summary['best_validation_step']
            assert selected['validation_nll'] == summary['best_validation_nll']
            assert sum(r['clipped'] for r in records[1:]) / 1221 == summary['clip_fraction']
            assert all(all(math.isfinite(r[k]) for k in ['ce', 'scaled_kl', 'gradient_norm_preclip', 'lr']) for r in records[1:])
            assert summary['nonfinite_fraction'] == 0
            assert records[1]['lr'] > 0 and records[-1]['lr'] == 0
            if arm in ['ce', 'kd1', 'kd5']:
                hashes.append(summary['s0_sha256'])
                order_hashes.append(summary['order_sha256'])
            result['total_formal_target_tokens'] += summary['consumed_target_tokens']
            result['runs'] += 1
        assert len(set(hashes)) == len(set(order_hashes)) == 1
        assert hashes[0] == sha(root / 's0/selected.pt')
        for arm in ['kd1', 'kd5']:
            assert json.loads((root / arm / 'summary.json').read_text())['teacher_sha256'] == sha(root / 'teacher/selected.pt')
            r = endpoint['arms'][arm]
            gain = endpoint['arms']['ce']['test_nll'] - r['test_nll']
            vg = endpoint['arms']['ce']['validation_nll'] - r['validation_nll']
            assert gain == r['gain_test_nll'] and vg == r['gain_validation_nll']
            assert r['clear_effect'] == (abs(gain) >= 0.02 and gain * vg > 0)
        result['checks'][dataset] = 'PASS_ALL_DATA_SELECTION_BUDGET_FINITE_HASH_GAIN_CHECKS'
    assert result['runs'] == 10 and result['total_formal_target_tokens'] == 100_000_000
    frozen = sorted(p for p in (ROOT / 'research/final_evidence').iterdir() if p.is_file())
    listing = ''.join(sha(p) + '  ' + str(p.relative_to(ROOT)) + '\n' for p in frozen)
    digest = hashlib.sha256(listing.encode()).hexdigest()
    assert digest == '3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96'
    result['final_evidence_aggregate_sha256'] = digest
    result['status'] = 'PASS'
    (ART / 'completed_run_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
