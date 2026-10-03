"""Only evaluate test once all five validation-selected runs are complete."""
import argparse
import gc
import json
import math
import torch
from train import OUT, load, sha, stream, evaluate


def main(dataset):
    torch.set_num_threads(4)
    root = OUT / 'runs' / dataset
    names = ['teacher', 's0', 'ce', 'kd1', 'kd5']
    summaries = {a: json.loads((root / a / 'summary.json').read_text()) for a in names}
    assert all(s['status'] == 'TRAIN_COMPLETE_VALIDATION_SELECTED' and s['consumed_target_tokens'] == 10_000_000 for s in summaries.values())
    assert len({summaries[a]['s0_sha256'] for a in ['ce', 'kd1', 'kd5']}) == 1
    assert len({summaries[a]['order_sha256'] for a in ['ce', 'kd1', 'kd5']}) == 1
    result = {'dataset': dataset, 'seed': 42, 'arms': {}}
    for arm in names:
        state = root / arm / 'selected.pt'
        assert sha(state) == summaries[arm]['selected_sha256']
        model = load('SmolLM2-360M' if arm == 'teacher' else 'SmolLM2-135M', state)
        val = evaluate(model, stream(dataset, 'validation'))
        assert abs(val - summaries[arm]['best_validation_nll']) < 1e-5
        test_path = OUT / 'data' / dataset / 'test.bin'
        manifest = json.loads((test_path.parent / 'manifest.json').read_text())
        assert sha(test_path) == manifest['sha256']['test']
        nll = evaluate(model, stream(dataset, 'test'))
        result['arms'][arm] = {'test_nll': nll, 'test_ppl': math.exp(nll), 'validation_nll': val,
                                'selected_sha256': sha(state), 'best_step': summaries[arm]['best_validation_step']}
        print(arm, result['arms'][arm], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for arm in ['kd1', 'kd5']:
        gain = result['arms']['ce']['test_nll'] - result['arms'][arm]['test_nll']
        vg = result['arms']['ce']['validation_nll'] - result['arms'][arm]['validation_nll']
        result['arms'][arm].update({'gain_test_nll': gain, 'gain_validation_nll': vg,
                                   'clear_effect': abs(gain) >= 0.02 and gain * vg > 0})
    result['numerically_valid'] = all(s['nonfinite_fraction'] == 0 for s in summaries.values())
    result['decision'] = 'MODERN_REPLICATION_WORTHWHILE' if result['numerically_valid'] and any(result['arms'][a]['clear_effect'] for a in ['kd1', 'kd5']) else 'STOP_MODERN_REPLICATION'
    (root / 'final_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['tinystories', 'fineweb'])
    main(p.parse_args().dataset)
