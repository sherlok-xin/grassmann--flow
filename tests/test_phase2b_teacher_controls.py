import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_distill_hybrid_lite_from_latefusion_teacher_v2 import (  # noqa: E402
    apply_teacher_alpha_override,
    normalize_legacy_alpha_state_dict,
    sha256_file,
)


class DummyTeacher(torch.nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        logit = torch.logit(torch.tensor(float(alpha), dtype=torch.float32))
        self.logit_alpha = torch.nn.Parameter(logit.view(1), requires_grad=False)

    def alpha(self):
        return torch.sigmoid(self.logit_alpha)


class Phase2BTeacherControlsTest(unittest.TestCase):
    def test_alpha_override_handles_fusion_and_endpoints(self):
        for requested in [0.0, 0.3, 0.5, 1.0]:
            teacher = DummyTeacher()
            checkpoint, effective = apply_teacher_alpha_override(teacher, requested)
            self.assertAlmostEqual(checkpoint[0], 0.6, places=6)
            self.assertAlmostEqual(effective[0], requested, places=7)

    def test_negative_alpha_preserves_checkpoint_value(self):
        teacher = DummyTeacher(alpha=0.4)
        checkpoint, effective = apply_teacher_alpha_override(teacher, -1.0)
        self.assertEqual(checkpoint, effective)
        self.assertAlmostEqual(effective[0], 0.4, places=6)

    def test_sha256_file(self):
        payload = b"phase2b-controlled-teacher"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_legacy_scalar_alpha_is_reshaped_without_value_change(self):
        state = {"logit_alpha": torch.tensor(0.25), "weight": torch.tensor([1.0])}
        target = {"logit_alpha": torch.empty(1), "weight": torch.empty(1)}
        normalized = normalize_legacy_alpha_state_dict(state, target)
        self.assertEqual(tuple(normalized["logit_alpha"].shape), (1,))
        self.assertEqual(normalized["logit_alpha"].item(), 0.25)
        self.assertIs(normalized["weight"], state["weight"])


if __name__ == "__main__":
    unittest.main()
