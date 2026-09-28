import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "research" / "experiments" / "phase2c_multiseed_teacher_utility"
    / "run_teacher_utility.py"
)
SPEC = importlib.util.spec_from_file_location("phase2c_teacher_utility", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_utility_sign_and_mass():
    result = MODULE.summarize_utility(np.array([-2.0, -1.0, 0.0, 1.0, 3.0]))
    assert result["count"] == 5
    assert result["mean"] == 0.2
    assert result["median"] == 0.0
    assert result["fraction_positive"] == 0.4
    assert result["positive_mean"] == 2.0
    assert result["negative_mean"] == -1.5
    assert result["positive_mass"] == 4.0
    assert result["negative_mass_signed"] == -3.0
    assert result["negative_mass_abs"] == 3.0


def test_group_record_rescue_and_harm_conditioning():
    utility = np.array([1.0, -1.0, 0.5, -0.5])
    student_correct = np.array([False, False, True, True])
    teacher_correct = np.array([True, False, True, False])
    result = MODULE.group_record(
        42, 0.0, "all", "all", utility, teacher_correct, student_correct
    )
    assert result["rescue_rate"] == 0.5
    assert result["harm_rate"] == 0.5
    assert result["teacher_accuracy"] == 0.5
    assert result["student_accuracy"] == 0.5
