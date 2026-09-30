#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research/experiments/phase2f_homogeneous_ensemble_control"
PHASE2E = ROOT / "research/experiments/phase2e_teacher_branch_ablation/results_multiseed.csv"
SEEDS = (42, 123, 456)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual, expected, name, atol=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise RuntimeError(f"{name}: expected {expected}, found {actual}")


def find_run(seed: int) -> Path:
    matches = sorted((ROOT / "outputs/distill_experiments").glob(f"*_phase2f_wt2_tt_seed{seed}"))
    complete = [path for path in matches if (path / "summary.json").is_file()]
    if len(complete) != 1:
        raise RuntimeError(f"seed {seed}: expected one complete TT run, found {complete}")
    return complete[0]


def validate_run(seed: int, run: Path) -> tuple[dict, dict]:
    config = load_json(run / "config.json")
    summary = load_json(run / "summary.json")
    cfg = config["config"]
    expected = {
        "teacher_type": "tt", "student_type": "hybrid_lite", "seed": seed,
        "batch_size": 32, "epochs": 10, "lr": 1e-4, "weight_decay": 0.01,
        "warmup_ratio": 0.05, "amp": True, "model_dim": 224, "num_layers": 6,
        "num_heads": 8, "reduced_dim": 56, "window_sizes": "1,2,4",
        "temperature": 2.0, "kd_loss_mode": "token_mean", "kd_lambda": 5.0,
        "distill_strategy": "fixed_fused", "teacher_alpha_override": 0.5,
        "max_seq_len": 256, "max_lines": 0,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError(f"seed {seed} config mismatch for {key}: {cfg.get(key)!r} != {value!r}")
    if config["source_teacher"].get("teacher_type") != "tt":
        raise RuntimeError(f"seed {seed}: source teacher type is not tt")
    if config["student_init"].get("student_init_checkpoint_sha256") is None:
        raise RuntimeError(f"seed {seed}: missing S0 checkpoint hash")
    for split, expected_chunks in (("train", 9379), ("validation", 991), ("test", 1133)):
        if config["dataset_stats"][split]["num_chunks"] != expected_chunks:
            raise RuntimeError(f"seed {seed}: {split} chunk mismatch")
    student = summary["student"]
    for key in ("test_loss", "test_ppl", "best_val_loss"):
        if not math.isfinite(float(student[key])):
            raise RuntimeError(f"seed {seed}: non-finite {key}")
    for key in ("train_grad_nonfinite_fraction", "train_amp_overflow_fraction"):
        if max(float(value) for value in student[key]) > 0.02:
            raise RuntimeError(f"seed {seed}: elevated {key}")
    checkpoint = run / "checkpoints/student_best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"seed {seed}: missing student checkpoint")
    return config, summary


def main():
    with PHASE2E.open(newline="", encoding="utf-8") as handle:
        reused = {int(row["Seed"]): row for row in csv.DictReader(handle)}

    rows = []
    raw_root = EXP / "raw/formal"
    for seed in SEEDS:
        run = find_run(seed)
        config, summary = validate_run(seed, run)
        student = summary["student"]
        old = reused[seed]
        c0 = float(old["C0 test NLL"])
        tg = float(old["F test NLL"])
        tt = float(student["test_loss"])
        gain_tg = c0 - tg
        gain_tt = c0 - tt
        h_value = tt - tg
        checkpoint = run / "checkpoints/student_best.pt"
        row = {
            "Seed": seed,
            "C0 test NLL": c0,
            "TG test NLL": tg,
            "TT test NLL": tt,
            "TT test PPL": float(student["test_ppl"]),
            "Gain_TG": gain_tg,
            "Gain_TT": gain_tt,
            "H": h_value,
            "TT best epoch": int(student["best_epoch"]),
            "TT best validation NLL": float(student["best_val_loss"]),
            "TT epoch1 clip fraction": float(student["train_grad_clip_fraction"][0]),
            "TT min clip fraction": min(float(x) for x in student["train_grad_clip_fraction"]),
            "TT max clip fraction": max(float(x) for x in student["train_grad_clip_fraction"]),
            "TT mean clip fraction": statistics.fmean(float(x) for x in student["train_grad_clip_fraction"]),
            "TT max AMP overflow fraction": max(float(x) for x in student["train_amp_overflow_fraction"]),
            "TT max nonfinite fraction": max(float(x) for x in student["train_grad_nonfinite_fraction"]),
            "C0 run directory": old["C0 run directory"],
            "TG run directory": old["F run directory"],
            "TT run directory": str(run.relative_to(ROOT)),
            "Student S0 hash": config["student_init"]["student_init_checkpoint_sha256"],
            "TT checkpoint SHA256": sha256(checkpoint),
            "Status": "COMPLETE",
        }
        rows.append(row)

        destination = raw_root / f"seed{seed}/TT"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "summary.json", "distill_metrics.jsonl", "report.md"):
            shutil.copy2(run / name, destination / name)
        (destination / "checkpoint_sha256.txt").write_text(
            f"{row['TT checkpoint SHA256']}  checkpoints/student_best.pt\n", encoding="utf-8"
        )

    fieldnames = list(rows[0])
    with (EXP / "results_multiseed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for metric in ("Gain_TG", "Gain_TT", "H"):
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "values": dict(zip(map(str, SEEDS), values)),
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values),
            "positive_signs": sum(value > 0 for value in values),
            "sign_consistency": f"{sum(value > 0 for value in values)}/3 positive",
        }
    payload = {"rows": rows, "aggregate": aggregate}
    (EXP / "raw/results_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (EXP / "raw/results_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
