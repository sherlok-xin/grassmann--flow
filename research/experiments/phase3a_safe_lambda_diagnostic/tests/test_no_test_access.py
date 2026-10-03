from pathlib import Path


def test_diagnostic_runner_does_not_construct_test_split():
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "run_diagnostics.py").read_text(encoding="utf-8")
    assert 'TextDataset("test"' not in runner
    assert "TextDataset('test'" not in runner
    assert "test_loader" not in runner
    assert "endpoint_source" not in runner
