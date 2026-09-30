import json
import tempfile
import unittest
from pathlib import Path

import torch

import train_distill_hybrid_lite_from_latefusion_teacher_v2 as kd
import train_hybrid_latefusion_alpha_ddp_v1 as trainer


def _transformer(vocab_size=32, max_seq_len=8):
    return kd.create_transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_dim=8,
        num_layers=1,
        num_heads=8,
        dropout=0.0,
    )


class Phase2FTeacherTypeTests(unittest.TestCase):
    def test_tt_checkpoint_layout_round_trips_between_trainer_and_kd_loader_model(self):
        torch.manual_seed(7)
        source = trainer.TransformerEnsembleLateFusionAlphaModel(
            _transformer(), _transformer(), init_alpha=0.5, late_k=1,
        ).eval()
        target = kd.TransformerEnsembleLateFusionAlphaModel(
            _transformer(), _transformer(), init_alpha=0.2, late_k=1,
        ).eval()
        target.load_state_dict(source.state_dict(), strict=True)
        tokens = torch.randint(0, 32, (2, 8))
        source_logits, _, _ = source(tokens)
        target_logits, _, branches = target(tokens, return_branches=True)
        torch.testing.assert_close(source_logits, target_logits)
        torch.testing.assert_close(target_logits, 0.5 * branches["transformer1"] + 0.5 * branches["transformer2"])

    def test_explicit_tt_loader_resolves_two_transformer_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dirs = []
            source_models = []
            config = {
                "model_dim": 8,
                "num_layers": 1,
                "num_heads": 8,
                "max_seq_len": 8,
                "dropout": 0.0,
            }
            for index in (1, 2):
                run_dir = root / f"t{index}"
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                model = _transformer()
                checkpoint = checkpoint_dir / "transformer_best.pt"
                torch.save(model.state_dict(), checkpoint)
                (run_dir / "config.json").write_text(json.dumps({"config": config}))
                (run_dir / "summary.json").write_text(json.dumps({
                    "transformer": {"checkpoint_path": str(checkpoint), "test_ppl": 10.0 + index}
                }))
                source_dirs.append(run_dir)
                source_models.append(model)

            teacher_dir = root / "teacher"
            checkpoint_dir = teacher_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            teacher = trainer.TransformerEnsembleLateFusionAlphaModel(
                source_models[0], source_models[1], init_alpha=0.5, late_k=1,
            )
            teacher_checkpoint = checkpoint_dir / "hybrid_best.pt"
            torch.save(teacher.state_dict(), teacher_checkpoint)
            (teacher_dir / "config.json").write_text(json.dumps({
                "config": {"teacher_type": "tt", "max_seq_len": 8, "init_alpha": 0.5, "late_k": 1},
                "source_runs": {
                    "teacher_type": "tt",
                    "transformer1_run_dir": str(source_dirs[0]),
                    "transformer2_run_dir": str(source_dirs[1]),
                    "transformer1_checkpoint": str(source_dirs[0] / "checkpoints" / "transformer_best.pt"),
                    "transformer2_checkpoint": str(source_dirs[1] / "checkpoints" / "transformer_best.pt"),
                },
            }))
            (teacher_dir / "summary.json").write_text(json.dumps({
                "hybrid": {
                    "checkpoint_path": str(teacher_checkpoint),
                    "late_k": 1,
                    "final_alpha": 0.5,
                }
            }))

            loaded, context = kd.load_teacher_and_context(
                teacher_dir, 32, search_roots=[], remaps=[], device=torch.device("cpu"),
                teacher_type="tt",
            )
            self.assertEqual(context["teacher_type"], "tt")
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            self.assertEqual(set(loaded(torch.randint(0, 32, (1, 8)), return_branches=True)[2]),
                             {"transformer1", "transformer2"})

    def test_tg_state_dict_names_remain_legacy_compatible(self):
        model = kd.HybridLateFusionAlphaModel(
            kd.create_grassmann(32, 8, 8, 1, 4, 0.0, "1"),
            _transformer(), init_alpha=0.5, late_k=1,
        )
        keys = set(model.state_dict())
        self.assertIn("logit_alpha", keys)
        self.assertTrue(any(key.startswith("grassmann.") for key in keys))
        self.assertTrue(any(key.startswith("transformer.") for key in keys))
        self.assertFalse(any(key.startswith("transformer1.") for key in keys))


if __name__ == "__main__":
    unittest.main()
