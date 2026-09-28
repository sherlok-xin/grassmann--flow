import unittest
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kd_losses import causal_lm_kd_loss


class CausalLMKDLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.student = torch.randn(2, 5, 11, dtype=torch.float64, requires_grad=True)
        self.teacher = torch.randn(2, 5, 11, dtype=torch.float64)
        self.labels = torch.randint(0, 11, (2, 5))

    def test_legacy_mode_matches_original_formula(self):
        temperature = 2.0
        alpha = 0.02
        result = causal_lm_kd_loss(
            self.student,
            self.teacher,
            self.labels,
            temperature=temperature,
            mode="legacy_batchmean",
            alpha=alpha,
        )
        shifted_student = self.student[:, :-1, :]
        shifted_teacher = self.teacher[:, :-1, :]
        shifted_labels = self.labels[:, 1:]
        ce = F.cross_entropy(shifted_student.reshape(-1, 11), shifted_labels.reshape(-1))
        kl = F.kl_div(
            F.log_softmax(shifted_student / temperature, dim=-1),
            F.softmax(shifted_teacher / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
        expected = (1.0 - alpha) * ce + alpha * kl
        torch.testing.assert_close(result.ce, ce)
        torch.testing.assert_close(result.kl_batchmean, kl)
        torch.testing.assert_close(result.total, expected)

    def test_token_mean_has_expected_sequence_scaling(self):
        result = causal_lm_kd_loss(
            self.student,
            self.teacher,
            self.labels,
            mode="token_mean",
            kd_lambda=5.0,
        )
        prediction_positions = self.student.size(1) - 1
        torch.testing.assert_close(
            result.kl_batchmean,
            result.kl_token_mean * prediction_positions,
        )
        torch.testing.assert_close(result.total, result.ce + 5.0 * result.kl_token_mean)
        self.assertEqual(result.valid_tokens.item(), self.labels.size(0) * prediction_positions)

    def test_ignored_token_does_not_affect_token_mean(self):
        labels = self.labels.clone()
        labels[0, 3] = -100
        changed_teacher = self.teacher.clone()
        changed_teacher[0, 2, :] = torch.linspace(-100.0, 100.0, 11)
        first = causal_lm_kd_loss(
            self.student,
            self.teacher,
            labels,
            mode="token_mean",
            kd_lambda=1.0,
            chunk_tokens=2,
        )
        second = causal_lm_kd_loss(
            self.student,
            changed_teacher,
            labels,
            mode="token_mean",
            kd_lambda=1.0,
            chunk_tokens=2,
        )
        torch.testing.assert_close(first.kl_token_mean, second.kl_token_mean)
        self.assertEqual(first.valid_tokens.item(), 7)

    def test_identical_logits_have_zero_kl(self):
        result = causal_lm_kd_loss(
            self.student,
            self.student.detach(),
            self.labels,
            mode="token_mean",
            kd_lambda=10.0,
        )
        torch.testing.assert_close(result.kl_token_mean, torch.zeros_like(result.kl_token_mean), atol=1e-12, rtol=0)
        torch.testing.assert_close(result.total, result.ce, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
