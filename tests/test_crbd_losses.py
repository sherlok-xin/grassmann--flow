import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crbd_losses import causal_lm_crbd_loss


class CRBDLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.student_fused = torch.randn(2, 5, 13, dtype=torch.float64, requires_grad=True)
        self.teacher_fused = torch.randn(2, 5, 13, dtype=torch.float64)
        self.student_branches = {
            "transformer": torch.randn(2, 5, 13, dtype=torch.float64, requires_grad=True),
            "grassmann": torch.randn(2, 5, 13, dtype=torch.float64, requires_grad=True),
        }
        self.teacher_branches = {
            "transformer": torch.randn(2, 5, 13, dtype=torch.float64),
            "grassmann": torch.randn(2, 5, 13, dtype=torch.float64),
        }
        self.labels = torch.randint(0, 13, (2, 5))

    def loss(self, strategy, **kwargs):
        return causal_lm_crbd_loss(
            self.student_fused,
            self.teacher_fused,
            self.student_branches,
            self.teacher_branches,
            self.labels,
            strategy=strategy,
            chunk_tokens=3,
            **kwargs,
        )

    def test_identical_teacher_branches_have_zero_disagreement(self):
        branches = {
            "transformer": self.teacher_fused,
            "grassmann": self.teacher_fused,
        }
        result = causal_lm_crbd_loss(
            self.student_fused,
            self.teacher_fused,
            self.student_branches,
            branches,
            self.labels,
            strategy="crbd",
            chunk_tokens=2,
        )
        torch.testing.assert_close(result.teacher_jsd_mean, torch.zeros_like(result.teacher_jsd_mean), atol=1e-12, rtol=0)
        torch.testing.assert_close(result.branch_kl_token_mean, torch.zeros_like(result.branch_kl_token_mean), atol=1e-12, rtol=0)

    def test_identical_student_teacher_has_only_ce(self):
        branches = {
            "transformer": self.teacher_branches["transformer"],
            "grassmann": self.teacher_branches["grassmann"],
        }
        result = causal_lm_crbd_loss(
            self.teacher_fused,
            self.teacher_fused,
            branches,
            self.teacher_branches,
            self.labels,
            strategy="crbd",
            chunk_tokens=2,
        )
        torch.testing.assert_close(result.total, result.ce, atol=2e-6, rtol=0)

    def test_branch_only_has_no_fused_term(self):
        result = self.loss("branch_only", branch_lambda=2.5)
        torch.testing.assert_close(result.fused_kl_token_mean, torch.zeros_like(result.fused_kl_token_mean))
        torch.testing.assert_close(result.total, result.ce + 2.5 * result.branch_kl_token_mean)

    def test_shuffled_routing_is_deterministic_and_changes_pairing(self):
        first = self.loss("crbd_shuffled", shuffle_seed=19)
        second = self.loss("crbd_shuffled", shuffle_seed=19)
        unshuffled = self.loss("crbd")
        torch.testing.assert_close(first.total, second.total)
        self.assertFalse(torch.isclose(first.total, unshuffled.total))

    def test_swapped_pairing_changes_branch_loss(self):
        aligned = self.loss("crbd")
        swapped = self.loss("crbd_swapped")
        self.assertFalse(torch.isclose(aligned.branch_kl_token_mean, swapped.branch_kl_token_mean))

    def test_ignore_mask_changes_valid_token_count(self):
        labels = self.labels.clone()
        labels[0, 2] = -100
        result = causal_lm_crbd_loss(
            self.student_fused,
            self.teacher_fused,
            self.student_branches,
            self.teacher_branches,
            labels,
            strategy="disagreement_suppressed",
            chunk_tokens=2,
        )
        self.assertEqual(result.valid_tokens.item(), 7)


if __name__ == "__main__":
    unittest.main()
