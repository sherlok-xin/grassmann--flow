import hashlib
import json
from pathlib import Path
import sys
import unittest

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
P4A = ROOT / 'research/experiments/phase4a_modern_generalization_pilot'
sys.path.insert(0, str(ART / 'scripts'))
from selection import including_s0, decisions


class Protocol(unittest.TestCase):
    def test_exact_frozen_loop_adapter(self):
        source = (P4A / 'scripts/train.py').read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         'd7d8ad350f45164b657daf1895b38b3f49e2e2e9466996961641a653a0c1c565')
        for old, new in json.loads((ART / 'training_adapter_changes.json').read_text()):
            self.assertIn(old, source)
            source = source.replace(old, new)
        self.assertEqual(source, (ART / 'scripts/train.py').read_text())
        self.assertEqual(hashlib.sha256((P4A / 'scripts/core.py').read_bytes()).hexdigest(),
                         '427c90edc3ba353d929b2d75a942d67779429bf2707525ddb55c271e8ce8c3cf')

    def test_step0_validation_only_and_tie(self):
        self.assertEqual(including_s0(2.9, 3.0, 1221), 0)
        self.assertEqual(including_s0(3.0, 2.9, 256), 256)
        self.assertEqual(including_s0(3.0, 3.0, 1221), 0)
        with self.assertRaises(ValueError):
            including_s0(float('nan'), 3, 1221)

    def test_decision_cases(self):
        self.assertIn('FINEWEB_L1_L5_NEGATIVE_REPLICATED',
                      decisions([-.01]*3, [-.04]*3, [True]*3))
        self.assertIn('FINEWEB_STRENGTH_DEPENDENT',
                      decisions([-.01, .01, -.01], [-.04]*3, [True]*3))
        self.assertEqual(decisions([-.01]*3, [-.04, .04, -.04], [True]*3),
                         ['MODERN_NEGATIVE_TRANSFER_NOT_ROBUST'])
        self.assertEqual(decisions([-.01]*3, [-.04]*3, [True, False, True]),
                         ['MODERN_NEGATIVE_TRANSFER_NOT_ROBUST'])


if __name__ == '__main__':
    unittest.main()
