import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from protocol import *

class ProtocolTests(unittest.TestCase):
    def test_budget_and_schedule(self):
        self.assertEqual(STEPS, 977)
        self.assertEqual(BUDGET-976*8192, 4608)
        self.assertEqual(CHECK_STEPS, [0,256,512,768,977])

    def test_step_zero_and_tie(self):
        rows = [{'step':0,'validation_nll':3.}, {'step':256,'validation_nll':3.}]
        self.assertEqual(select(rows)['step'],0)

    def test_gate_threshold_and_lr_tie(self):
        def s(lr, nll, step):
            return {'optimizer':{'lr':lr},'best_validation_nll':nll,'initial_validation_nll':3.,'best_validation_step':step}
        self.assertFalse(gate([s(5e-6,2.995,256)])[2])
        self.assertTrue(gate([s(5e-6,2.98,256)])[2])
        self.assertFalse(gate([s(5e-6,3.,0)])[2])
        self.assertEqual(gate([s(2e-5,2.98,256),s(5e-6,2.98,256)])[0]['optimizer']['lr'],5e-6)

    def test_manifest(self):
        m, _ = manifest()
        a, b = set(m['excluded_chunk_ids']), set(m['ordered_chunk_ids'])
        self.assertEqual(len(a),8192)
        self.assertEqual(len(b),31250)
        self.assertFalse(a & b)
        self.assertEqual(m['target_budget'],8_000_000)

    def test_no_test_in_train(self):
        source = (ART/'scripts/train.py').read_text()
        self.assertNotIn("stream('fineweb', 'test')",source)
        self.assertIn("assert frozen['ce_gate_passed']",source)

if __name__ == '__main__':
    unittest.main()
