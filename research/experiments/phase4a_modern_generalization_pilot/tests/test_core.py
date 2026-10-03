import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1] / 'scripts/core.py'
spec = importlib.util.spec_from_file_location('core', path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class TestNumerics(unittest.TestCase):
    def test_chunk_mask_temperature_and_gradient(self):
        torch.manual_seed(42)
        s = torch.randn(3, 7, 13, requires_grad=True)
        t = torch.randn_like(s)
        y = torch.randint(13, (3, 7))
        y[1, 3:] = -100
        ce, kl, n = core.chunk_losses(s, y, t, chunk=4)
        mask = y.reshape(-1) != -100
        sf, tf = s.reshape(-1, 13)[mask], t.reshape(-1, 13)[mask]
        ref_ce = F.cross_entropy(sf, y.reshape(-1)[mask])
        ref_kl = F.kl_div(F.log_softmax(sf / 2, -1), F.softmax(tf / 2, -1), reduction='sum') * 4 / n
        torch.testing.assert_close(ce, ref_ce)
        torch.testing.assert_close(kl, ref_kl)
        a = torch.autograd.grad(ce + 5 * kl, s, retain_graph=True)[0]
        b = torch.autograd.grad(ref_ce + 5 * ref_kl, s)[0]
        torch.testing.assert_close(a, b)

    def test_identical_logits_zero_kl_and_direction(self):
        s = torch.tensor([[[4., 0., -1.]]], requires_grad=True)
        t = torch.tensor([[[0., 1., 2.]]])
        y = torch.tensor([[0]])
        _, k, _ = core.chunk_losses(s, y, s.detach())
        self.assertEqual(k.item(), 0)
        _, k, _ = core.chunk_losses(s, y, t)
        _, reverse, _ = core.chunk_losses(t, y, s.detach())
        self.assertGreater(abs(k.item() - reverse.item()), 0.01)

    def test_packing_and_exact_final_budget(self):
        tokens = np.arange(19, dtype=np.uint32)
        x, y = core.packed_batch(tokens, [0, 1, 2, 3, 4], 4)
        torch.testing.assert_close(y[y != -100], torch.arange(1, 19))
        x, y = core.packed_batch(tokens, [0, 1, 2], 4, valid_limit=9)
        self.assertEqual(int((y != -100).sum()), 9)
        self.assertEqual(x[2, 0], 8)
        self.assertEqual(y[2, 0], 9)

    def test_schedule(self):
        self.assertGreater(core.schedule(0, 100), 0)
        self.assertAlmostEqual(core.schedule(4, 100), 1)
        self.assertAlmostEqual(core.schedule(99, 100), 0)


if __name__ == '__main__':
    unittest.main()
