#!/usr/bin/env python3
"""Validate and collect reused C0/F/G controls and new Phase 2E T endpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DISTILL = ROOT / "outputs/distill_experiments"
PHASE2C = ROOT / "research/experiments/phase2c_multiseed_teacher_utility/results_multiseed.csv"
TEACHER = "outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"
TEACHER_HASH = "a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694"
SEEDS = [42, 123, 456]
S0_HASHES = {
    42: "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef",
    123: "a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13",
    456: "3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757",
}
INVARIANTS = {
    "batch_size": 32, "epochs": 10, "lr": 0.0001, "weight_decay": 0.01,
    "warmup_ratio": 0.05, "amp": True, "model_dim": 224, "num_layers": 6,
    "num_heads": 8, "reduced_dim": 56, "window_sizes": "1,2,4", "dropout": 0.1,
    "student_late_k": 1, "temperature": 2.0, "kd_loss_mode": "token_mean",
    "kd_lambda": 5.0, "kd_chunk_tokens": 1024, "distill_strategy": "fixed_fused",
    "dataset_name": "wikitext2", "text_field": "text", "max_seq_len": 256,
    "max_lines": 0, "encode_chars_per_batch": 200000,
    "tinystories_val_frac": 0.02, "split_seed": 42, "offline": True,
}
WARNING = "CLIP_SATURATION_WARNING"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(label: str, actual, expected) -> None:
    if isinstance(expected, float):
        matched = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        matched = actual == expected
    if not matched:
        raise RuntimeError(f"{label}: expected {expected!r}, found {actual!r}")


def summarize(values: list[float]) -> dict:
    mean = statistics.mean(values)
    return {
        "values": values, "mean": mean, "sample_sd": statistics.stdev(values),
        "min": min(values), "max": max(values),
        "positive_signs": sum(value > 0 for value in values),
        "negative_signs": sum(value < 0 for value in values),
        "consistent_sign": all(value > 0 for value in values) or all(value < 0 for value in values),
        "practically_nontrivial": abs(mean) >= 0.02,
    }


def find_complete(seed: int) -> Path:
    name = f"phase2e_wt2_t_seed{seed}"
    complete = [
        path for path in sorted(DISTILL.glob(f"*_{name}"))
        if all((path / item).is_file() for item in [
            "config.json", "summary.json", "distill_metrics.jsonl", "report.md",
            "checkpoints/student_best.pt",
        ])
    ]
    if len(complete) != 1:
        raise RuntimeError(f"Expected exactly one complete {name} run, found {complete}")
    return complete[0]


def main() -> None:
    with PHASE2C.open(encoding="utf-8") as handle:
        controls = {(int(row["Seed"]), row["Arm"]): row for row in csv.DictReader(handle)}
    teacher_checkpoint = ROOT / TEACHER / "checkpoints/hybrid_best.pt"
    require_equal("teacher checkpoint SHA256", sha256_file(teacher_checkpoint), TEACHER_HASH)

    t_runs = {}
    for seed in SEEDS:
        run_dir = find_complete(seed)
        config_root = load_json(run_dir / "config.json")
        config = config_root["config"]
        summary = load_json(run_dir / "summary.json")["student"]
        metric_lines = [line for line in (run_dir / "distill_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if len(metric_lines) != 10:
            raise RuntimeError(f"seed {seed}: expected 10 epochs, found {len(metric_lines)}")
        metrics = [json.loads(line) for line in metric_lines]
        require_equal(f"seed {seed}", int(config["seed"]), seed)
        for key, expected in INVARIANTS.items():
            require_equal(f"seed {seed} config {key}", config.get(key), expected)
        if not str(config["teacher_run_dir"]).endswith(TEACHER):
            raise RuntimeError(f"seed {seed}: teacher identity mismatch")
        require_equal(f"seed {seed} config S0", config_root["student_init"]["student_init_checkpoint_sha256"], S0_HASHES[seed])
        require_equal(f"seed {seed} summary S0", summary["student_init_checkpoint_sha256"], S0_HASHES[seed])
        require_equal(f"seed {seed} effective alpha config", config_root["source_teacher"]["effective_alpha"], [1.0])
        require_equal(f"seed {seed} effective alpha summary", summary["teacher_effective_alpha"], [1.0])
        require_equal(f"seed {seed} experiment name", config_root["experiment_name"], f"phase2e_wt2_t_seed{seed}")
        require_equal(f"seed {seed} notes", config_root["notes"], "phase2e_preregistered_teacher_branch_ablation")
        require_equal(f"seed {seed} tags", config_root["tags"], ["phase2e", "formal", f"seed{seed}", "T"])
        if int(config_root["dataset_stats"]["train"]["max_lines"]) != 0:
            raise RuntimeError(f"seed {seed}: formal run did not use full training data")
        finite = []
        for epoch in metrics:
            finite.extend([epoch["train_loss"], epoch["train_ce"], epoch["train_kl_token_mean"], epoch["train_grad_norm_mean"], epoch["val_loss"]])
        if not all(math.isfinite(float(value)) for value in finite):
            raise RuntimeError(f"seed {seed}: non-finite endpoint metric")
        if not math.isfinite(float(summary["test_loss"])):
            raise RuntimeError(f"seed {seed}: non-finite test endpoint")
        expected_best_epoch = min(range(len(metrics)), key=lambda index: float(metrics[index]["val_loss"])) + 1
        require_equal(f"seed {seed} validation-best epoch", int(summary["best_epoch"]), expected_best_epoch)
        require_equal(
            f"seed {seed} validation-best NLL",
            float(summary["best_val_loss"]),
            float(metrics[expected_best_epoch - 1]["val_loss"]),
        )
        max_overflow = max(float(epoch["train_amp_overflow_fraction"]) for epoch in metrics)
        max_nonfinite = max(float(epoch["train_grad_nonfinite_fraction"]) for epoch in metrics)
        if max_overflow > 0.02 or max_nonfinite > 0.02:
            raise RuntimeError(
                f"seed {seed}: optimization-confounded overflow={max_overflow} nonfinite={max_nonfinite}"
            )
        destination = HERE / f"raw/formal/seed{seed}/T"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ["config.json", "summary.json", "distill_metrics.jsonl", "report.md"]:
            shutil.copy2(run_dir / filename, destination / filename)
        t_runs[seed] = {
            "run_dir": run_dir, "summary": summary, "metrics": metrics,
            "checkpoint_sha256": sha256_file(run_dir / "checkpoints/student_best.pt"),
            "max_overflow": max_overflow, "max_nonfinite": max_nonfinite,
        }

    rows = []
    for seed in SEEDS:
        c0 = controls[(seed, "C0")]
        f = controls[(seed, "C1")]
        g = controls[(seed, "C3")]
        for label, reused in [("C0", c0), ("F", f), ("G", g)]:
            require_equal(f"seed {seed} reused {label} S0", reused["Student S0 hash"], S0_HASHES[seed])
        nll_c0 = float(c0["Test NLL"])
        nll_f = float(f["Test NLL"])
        nll_t = float(t_runs[seed]["summary"]["test_loss"])
        nll_g = float(g["Test NLL"])
        first = t_runs[seed]["metrics"][0]
        clip_values = [float(epoch["train_grad_clip_fraction"]) for epoch in t_runs[seed]["metrics"]]
        rows.append({
            "Seed": seed,
            "C0 test NLL": nll_c0, "F test NLL": nll_f,
            "T test NLL": nll_t, "G test NLL": nll_g,
            "T test PPL": float(t_runs[seed]["summary"]["test_ppl"]),
            "Gain_F": nll_c0 - nll_f, "Gain_T": nll_c0 - nll_t, "Gain_G": nll_c0 - nll_g,
            "C_FT": nll_t - nll_f, "C_FG": nll_g - nll_f, "C_TG": nll_g - nll_t,
            "T best epoch": int(t_runs[seed]["summary"]["best_epoch"]),
            "T best validation NLL": float(t_runs[seed]["summary"]["best_val_loss"]),
            "T epoch1 clip fraction": float(first["train_grad_clip_fraction"]),
            "T epoch1 AMP overflow fraction": float(first["train_amp_overflow_fraction"]),
            "T epoch1 nonfinite fraction": float(first["train_grad_nonfinite_fraction"]),
            "T min clip fraction": min(clip_values),
            "T max clip fraction": max(clip_values),
            "T mean clip fraction": statistics.mean(clip_values),
            "T max AMP overflow fraction": t_runs[seed]["max_overflow"],
            "T max nonfinite fraction": t_runs[seed]["max_nonfinite"],
            "C0 run directory": c0["Run directory"], "F run directory": f["Run directory"],
            "T run directory": str(t_runs[seed]["run_dir"].relative_to(ROOT)), "G run directory": g["Run directory"],
            "Student S0 hash": S0_HASHES[seed],
            "T checkpoint SHA256": t_runs[seed]["checkpoint_sha256"],
            "Warning": WARNING,
            "Status": f"COMPLETE|{WARNING}",
        })

    with (HERE / "results_multiseed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    statistics_payload = {
        key: summarize([float(row[key]) for row in rows])
        for key in ["Gain_F", "Gain_T", "Gain_G", "C_FT", "C_FG", "C_TG"]
    }
    payload = {
        "complete": True, "teacher_checkpoint_sha256": TEACHER_HASH,
        "formal_invariants": INVARIANTS, "practical_threshold_nll": 0.02,
        "primary_contrast": "C_FT", "warning": WARNING,
        "warning_interpretation": (
            "Transformer-only gradients continuously triggered clipping in the full-data gate; "
            "formal endpoint differences may partly reflect this optimization constraint."
        ),
        "rows": rows, "statistics": statistics_payload,
    }
    (HERE / "raw/results_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
