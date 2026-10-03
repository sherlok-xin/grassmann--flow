#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEACHER = Path("outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10")
REFERENCE = {
    42: {
        "s0": Path("outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20"),
        "ce": Path("outputs/distill_experiments/20260916_073243_h0_ts_confirm_token_l0_seed42"),
        "kd": Path("outputs/distill_experiments/20260916_073219_h0_ts_confirm_token_l5_seed42"),
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    for seed in (123, 456):
        parser.add_argument(f"--s0-{seed}", required=True, type=Path)
        parser.add_argument(f"--ce-{seed}", required=True, type=Path)
        parser.add_argument(f"--kd-{seed}", required=True, type=Path)
    return parser.parse_args()


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def mean(values):
    return statistics.mean(values)


def validate_s0(run_dir, seed, reference_config):
    config = load(run_dir / "config.json")["config"]
    summary = load(run_dir / "summary.json")["hybrid"]
    fields = [
        "batch_size", "epochs", "lr", "weight_decay", "amp", "init_alpha",
        "late_k", "model_dim", "num_layers", "reduced_dim", "window_sizes",
        "dropout", "dataset_name", "dataset_path", "text_field", "max_seq_len",
        "max_lines", "encode_chars_per_batch", "tinystories_val_frac", "split_seed",
        "offline",
    ]
    for field in fields:
        check_equal(config[field], reference_config[field], f"S0 seed {seed} {field}")
    check_equal(config["seed"], seed, f"S0 seed {seed}")
    check_equal(summary["num_params"], 31434257, f"S0 seed {seed} parameters")
    checkpoint = run_dir / "checkpoints/student_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "checkpoints/hybrid_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return config, summary, checkpoint, sha256(checkpoint)


def validate_endpoint(run_dir, seed, kd_lambda, reference_config, s0_hash, teacher_hash):
    payload = load(run_dir / "config.json")
    config = payload["config"]
    summary = load(run_dir / "summary.json")["student"]
    fields = [
        "student_type", "batch_size", "epochs", "lr", "weight_decay",
        "warmup_ratio", "amp", "model_dim", "num_layers", "num_heads",
        "reduced_dim", "window_sizes", "dropout", "student_late_k",
        "distill_alpha", "temperature", "kd_loss_mode", "dataset_name",
        "dataset_path", "text_field", "max_seq_len", "max_lines",
        "encode_chars_per_batch", "tinystories_val_frac", "split_seed", "offline",
    ]
    for field in fields:
        check_equal(config[field], reference_config[field], f"endpoint seed {seed} {field}")
    check_equal(config["seed"], seed, f"endpoint seed {seed}")
    check_equal(config["kd_lambda"], float(kd_lambda), f"endpoint seed {seed} lambda")
    check_equal(summary["kd_loss_mode"], "token_mean", f"endpoint seed {seed} loss mode")
    check_equal(summary["temperature"], 2.0, f"endpoint seed {seed} temperature")
    recorded_s0_hash = summary.get("student_init_checkpoint_sha256")
    if recorded_s0_hash is None:
        # The historical seed-42 runs predate the summary hash field. Verify the
        # resolved initialization checkpoint recorded in config.json instead.
        init_checkpoint = Path(payload["student_init"]["student_init_checkpoint"])
        recorded_s0_hash = sha256(init_checkpoint)
    check_equal(recorded_s0_hash, s0_hash, f"endpoint seed {seed} S0 hash")
    teacher_checkpoint = Path(payload["source_teacher"]["teacher_checkpoint"])
    check_equal(sha256(teacher_checkpoint), teacher_hash, f"endpoint seed {seed} teacher hash")
    checkpoint = run_dir / "checkpoints/student_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    for key in ("test_loss", "best_val_loss"):
        if not math.isfinite(float(summary[key])):
            raise ValueError(f"endpoint seed {seed} has non-finite {key}")
    return payload, summary, checkpoint, sha256(checkpoint)


def diagnostics(summary, prefix):
    return {
        f"{prefix} epoch1 KL": summary["train_kl_token_mean"][0],
        f"{prefix} mean KL": mean(summary["train_kl_token_mean"]),
        f"{prefix} epoch1 grad norm": summary["train_grad_norm_mean"][0],
        f"{prefix} mean grad norm": mean(summary["train_grad_norm_mean"]),
        f"{prefix} epoch1 clip fraction": summary["train_grad_clip_fraction"][0],
        f"{prefix} mean clip fraction": mean(summary["train_grad_clip_fraction"]),
        f"{prefix} max AMP overflow fraction": max(summary["train_amp_overflow_fraction"]),
        f"{prefix} max nonfinite fraction": max(summary["train_grad_nonfinite_fraction"]),
    }


def copy_compact(seed, arm, run_dir, checkpoint_hash):
    destination = ROOT / "raw" / "formal" / f"seed{seed}" / arm
    destination.mkdir(parents=True, exist_ok=True)
    names = ["config.json", "summary.json"]
    if arm != "S0":
        names.extend(["distill_metrics.jsonl", "report.md"])
    else:
        names.extend(["hybrid_metrics.jsonl", "report.md"])
    for name in names:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    (destination / "checkpoint_sha256.txt").write_text(
        f"{checkpoint_hash}  {run_dir}/checkpoints/"
        f"{'hybrid_best.pt' if arm == 'S0' else 'student_best.pt'}\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    runs = dict(REFERENCE)
    for seed in (123, 456):
        runs[seed] = {
            "s0": getattr(args, f"s0_{seed}"),
            "ce": getattr(args, f"ce_{seed}"),
            "kd": getattr(args, f"kd_{seed}"),
        }

    teacher_hash = sha256(TEACHER / "checkpoints/hybrid_best.pt")
    check_equal(teacher_hash, "db9ffda81d1f95ebc3f3d5967e3121b1c6dd859104c9c47cc80b0bab1fcaedfb",
                "teacher checkpoint")
    teacher_nll = load(TEACHER / "summary.json")["hybrid"]["test_loss"]
    s0_reference = load(REFERENCE[42]["s0"] / "config.json")["config"]
    endpoint_reference = load(REFERENCE[42]["ce"] / "config.json")["config"]

    rows = []
    s0_hashes = []
    for seed in (42, 123, 456):
        run = runs[seed]
        _, s0_summary, _, s0_hash = validate_s0(run["s0"], seed, s0_reference)
        _, ce, _, ce_hash = validate_endpoint(
            run["ce"], seed, 0, endpoint_reference, s0_hash, teacher_hash)
        _, kd, _, kd_hash = validate_endpoint(
            run["kd"], seed, 5, endpoint_reference, s0_hash, teacher_hash)
        row = {
            "Seed": seed,
            "S0 test NLL": s0_summary["test_loss"],
            "S0 test PPL": s0_summary["test_ppl"],
            "WS+CE best epoch": ce["best_epoch"],
            "WS+CE validation NLL": ce["best_val_loss"],
            "WS+CE test NLL": ce["test_loss"],
            "WS+CE test PPL": ce["test_ppl"],
            "WS+KD best epoch": kd["best_epoch"],
            "WS+KD validation NLL": kd["best_val_loss"],
            "WS+KD test NLL": kd["test_loss"],
            "WS+KD test PPL": kd["test_ppl"],
            "Teacher test NLL": teacher_nll,
            "Teacher residual advantage": ce["test_loss"] - teacher_nll,
            "Delta_KD": ce["test_loss"] - kd["test_loss"],
            **diagnostics(ce, "CE"),
            **diagnostics(kd, "KD"),
            "S0 run directory": str(run["s0"]),
            "CE run directory": str(run["ce"]),
            "KD run directory": str(run["kd"]),
            "S0 checkpoint SHA256": s0_hash,
            "CE checkpoint SHA256": ce_hash,
            "KD checkpoint SHA256": kd_hash,
            "Status": "COMPLETE",
        }
        rows.append(row)
        s0_hashes.append(s0_hash)
        for arm, summary, checkpoint_hash in (
            ("S0", s0_summary, s0_hash), ("CE", ce, ce_hash), ("KD", kd, kd_hash)
        ):
            del summary
            copy_compact(seed, arm, run[arm.lower()], checkpoint_hash)

    if len(set(s0_hashes)) != 3:
        raise ValueError("S0 checkpoint hashes are not distinct across seeds")

    fieldnames = list(rows[0])
    with (ROOT / "results_multiseed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    deltas = [float(row["Delta_KD"]) for row in rows]
    negative_count = sum(value < 0 for value in deltas)
    if negative_count == 3:
        decision = "CONFIRMED"
    elif negative_count == 2:
        decision = "PARTIAL"
    else:
        decision = "NEGATIVE TRANSFER NOT ROBUST"
    aggregate = {
        "Delta_KD": {
            "values": {str(row["Seed"]): row["Delta_KD"] for row in rows},
            "mean": statistics.mean(deltas),
            "sample_sd": statistics.stdev(deltas),
            "negative_signs": negative_count,
            "sign_consistency": f"{negative_count}/3 negative",
        },
        "decision": decision,
        "teacher_checkpoint_sha256": teacher_hash,
        "distinct_s0_hashes": True,
    }
    (ROOT / "raw").mkdir(exist_ok=True)
    (ROOT / "raw" / "results_summary.json").write_text(
        json.dumps({"rows": rows, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
