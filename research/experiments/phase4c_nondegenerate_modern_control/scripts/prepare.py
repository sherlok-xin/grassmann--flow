"""Freeze disjoint training order and audit exact preparation step256."""
import json
import hashlib
import numpy as np
from protocol import *

assert not (ART / 'continuation_manifest.json').exists(), 'No manifest rewrite'
summary = json.loads((S0.parent / 'summary.json').read_text())
metrics = [json.loads(line) for line in (S0.parent / 'metrics.jsonl').read_text().splitlines()]
chosen = next(r for r in metrics if r['step'] == 256)
assert summary['seed'] == 42 and summary['best_validation_step'] == 256
assert summary['best_validation_tokens'] == chosen['tokens'] == 2_097_152
assert chosen['validation_nll'] == summary['best_validation_nll']
assert sha(S0) == summary['selected_sha256'] == S0_SHA
assert sha(TEACHER) == TEACHER_SHA
data = json.loads((OLD / 'data/fineweb/manifest.json').read_text())
for split, expected in data['sha256'].items():
    assert sha(OLD / 'data/fineweb' / (split + '.bin')) == expected
chunks = (20_000_000 - 1) // 256
prep_order = np.random.default_rng(42).permutation(chunks).astype('<i8')
assert hashlib.sha256(prep_order.tobytes()).hexdigest() == summary['order_sha256']
seen = prep_order[:256 * 32]
candidate = np.random.default_rng(4242).permutation(chunks)
order = candidate[~np.isin(candidate, seen)][:BUDGET // 256].astype('<i8')
assert len(order) == 31250 and len(set(order.tolist())) == len(order)
assert not set(order.tolist()).intersection(seen.tolist())
m = {
    'schema_version': 1, 'dataset': 'FineWeb-Edu sample-10BT',
    'revision': data.get('revision', '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'),
    'split_sha256': data['sha256'], 's0_path': str(S0.relative_to(ROOT)),
    's0_sha256': S0_SHA, 'fixed_s0_preparation_step': 256,
    'fixed_s0_targets': 2097152, 's0_order_seed': 42,
    's0_full_order_sha256': summary['order_sha256'],
    'excluded_chunk_ids': seen.tolist(), 'available_full_train_chunks': chunks,
    'continuation_order_seed': 4242,
    'algorithm': 'Filter full seed4242 train-chunk permutation against first8192 seed42 preparation chunks; take first31250 remaining chunks.',
    'ordered_chunk_ids': order.tolist(),
    'order_sha256_int64_le': hashlib.sha256(order.tobytes()).hexdigest(),
    'target_budget': BUDGET, 'seq_len': 256, 'batch': 32, 'microbatch': 4,
    'optimizer_steps': STEPS, 'checkpoint_schedule': CHECK_STEPS,
    'final_step_rows': 18, 'final_step_targets': 4608,
    'target_overlap_with_s0': 0, 'repeated_continuation_target_chunks': 0,
    'context_overlap_note': 'Adjacent chunks share one context token; their supervised target positions do not overlap.',
    'heldout_policy': 'Only existing train.bin chunk indices; validation/test are never used as training inputs.',
    'test_access': 'File checksum only; no test forwards or NLL selection.'
}
path = ART / 'continuation_manifest.json'
path.write_text(json.dumps(m, separators=(',', ':')) + '\n')
audit = {'status': 'PASS', 'parent_commit': '6506c79e560efd2bbdacce7b87e7164b21d56180',
         'fixed_s0_step': 256, 'fixed_s0_targets': 2097152, 's0_sha256': S0_SHA,
         's0_validation_nll_recorded': chosen['validation_nll'],
         'historical_selection_note': 'Historical selected.pt happens to be exact step256; Phase4C chooses this fixed time point, not a new validation optimum.',
         'teacher_sha256': TEACHER_SHA,
         'continuation_manifest_sha256': sha(path),
         'continuation_order_sha256': m['order_sha256_int64_le'],
         'excluded_chunks': len(seen), 'additional_targets': BUDGET,
         'preparation_summary_sha256': sha(S0.parent / 'summary.json'),
         'preparation_metrics_sha256': sha(S0.parent / 'metrics.jsonl'),
         'test_forward': False}
(ART / 'input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
verify_inputs()
print(json.dumps(audit, indent=2), flush=True)
