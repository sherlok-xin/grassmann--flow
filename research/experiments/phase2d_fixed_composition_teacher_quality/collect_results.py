#!/usr/bin/env python3
"""Validate and collect reused Phase 2C controls and Phase 2D D2 endpoints."""

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
TEACHER_A = "outputs/hybrid_experiments/20260328_084319_wt2_v1_hybrid_alpha_only"
TEACHER_A_HASH = "43d288db1dba826d8ad1bb3fcbd09b9720bdde24c198e858e00ce4c4658df012"
SEEDS = [42, 123, 456]
S0_HASHES = {
    42: "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef",
    123: "a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13",
    456: "3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757",
}
INVARIANTS = {
    "batch_size": 32,
    "epochs": 10,
    "lr": 0.0001,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "amp": True,
    "model_dim": 224,
    "num_layers": 6,
    "num_heads": 8,
    "reduced_dim": 56,
    "window_sizes": "1,2,4",
    "dropout": 0.1,
    "student_late_k": 1,
    "temperature": 2.0,
    "kd_loss_mode": "token_mean",
    "kd_lambda": 5.0,
    "kd_chunk_tokens": 1024,
    "distill_strategy": "fixed_fused",
    "dataset_name": "wikitext2",
    "max_seq_len": 256,
    "max_lines": 0,
    "split_seed": 42,
    "offline": True,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "positive_signs": sum(value > 0 for value in values),
        "negative_signs": sum(value < 0 for value in values),
        "zero_signs": sum(value == 0 for value in values),
        "total": len(values),
    }


def find_complete(seed: int) -> Path:
    name = f"phase2d_wt2_d2_a_seed{seed}"
    candidates = sorted(DISTILL.glob(f"*_{name}"))
    complete = [
        path for path in candidates
        if all((path / file_name).is_file() for file_name in [
            "config.json", "summary.json", "distill_metrics.jsonl", "report.md",
            "checkpoints/student_best.pt",
        ])
    ]
    if len(complete) != 1:
        raise RuntimeError(f"Expected exactly one complete {name} run, found {complete}")
    return complete[0]


def require_equal(label: str, actual, expected) -> None:
    if isinstance(expected, float):
        matched = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        matched = actual == expected
    if not matched:
        raise RuntimeError(f"{label}: expected {expected!r}, found {actual!r}")


def main() -> None:
    with PHASE2C.open(encoding="utf-8") as handle:
        phase2c_rows = list(csv.DictReader(handle))
    controls = {(int(row["Seed"]), row["Arm"]): row for row in phase2c_rows}

    teacher_checkpoint = ROOT / TEACHER_A / "checkpoints/hybrid_best.pt"
    require_equal("Teacher A checkpoint SHA256", sha256_file(teacher_checkpoint), TEACHER_A_HASH)

    d2 = {}
    raw_formal = HERE / "raw/formal"
    for seed in SEEDS:
        run_dir = find_complete(seed)
        config_root = load_json(run_dir / "config.json")
        config = config_root["config"]
        summary = load_json(run_dir / "summary.json")["student"]
        metric_lines = [line for line in (run_dir / "distill_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if len(metric_lines) != 10:
            raise RuntimeError(f"seed {seed}: expected 10 epoch records, found {len(metric_lines)}")
        metrics = [json.loads(line) for line in metric_lines]
        require_equal(f"seed {seed} config seed", int(config["seed"]), seed)
        for key, expected in INVARIANTS.items():
            require_equal(f"seed {seed} config {key}", config.get(key), expected)
        if not str(config["teacher_run_dir"]).endswith(TEACHER_A):
            raise RuntimeError(f"seed {seed}: Teacher A identity mismatch: {config['teacher_run_dir']}")
        require_equal(
            f"seed {seed} config S0 hash",
            config_root["student_init"]["student_init_checkpoint_sha256"],
            S0_HASHES[seed],
        )
        require_equal(
            f"seed {seed} summary S0 hash",
            summary["student_init_checkpoint_sha256"],
            S0_HASHES[seed],
        )
        require_equal(f"seed {seed} effective alpha", config_root["source_teacher"]["effective_alpha"], [0.5])
        require_equal(f"seed {seed} summary alpha", summary["teacher_effective_alpha"], [0.5])
        if int(config_root["dataset_stats"]["train"]["max_lines"]) != 0:
            raise RuntimeError(f"seed {seed}: formal run did not use the full training split")
        finite = []
        for epoch in metrics:
            finite.extend([
                epoch["train_loss"], epoch["train_ce"], epoch["train_kl_token_mean"],
                epoch["train_grad_norm_mean"], epoch["val_loss"],
            ])
        if not all(math.isfinite(float(value)) for value in finite):
            raise RuntimeError(f"seed {seed}: non-finite stored endpoint metric")
        destination = raw_formal / f"seed{seed}" / "D2"
        destination.mkdir(parents=True, exist_ok=True)
        for file_name in ["config.json", "summary.json", "distill_metrics.jsonl", "report.md"]:
            shutil.copy2(run_dir / file_name, destination / file_name)
        d2[seed] = {
            "run_dir": run_dir,
            "summary": summary,
            "metrics": metrics,
            "checkpoint_sha256": sha256_file(run_dir / "checkpoints/student_best.pt"),
        }

    rows = []
    for seed in SEEDS:
        c0 = controls[(seed, "C0")]
        c1 = controls[(seed, "C1")]
        require_equal(f"seed {seed} reused C0 S0", c0["Student S0 hash"], S0_HASHES[seed])
        require_equal(f"seed {seed} reused C1 S0", c1["Student S0 hash"], S0_HASHES[seed])
        c0_nll = float(c0["Test NLL"])
        j_nll = float(c1["Test NLL"])
        a_nll = float(d2[seed]["summary"]["test_loss"])
        gain_j = c0_nll - j_nll
        gain_a = c0_nll - a_nll
        q_value = a_nll - j_nll
        first = d2[seed]["metrics"][0]
        rows.append({
            "Seed": seed,
            "C0 test NLL": c0_nll,
            "Teacher J test NLL": j_nll,
            "Teacher A test NLL": a_nll,
            "Teacher A test PPL": float(d2[seed]["summary"]["test_ppl"]),
            "Gain_J": gain_j,
            "Gain_A": gain_a,
            "Q": q_value,
            "Teacher A best epoch": int(d2[seed]["summary"]["best_epoch"]),
            "Teacher A best validation NLL": float(d2[seed]["summary"]["best_val_loss"]),
            "Teacher A epoch1 clip fraction": float(first["train_grad_clip_fraction"]),
            "Teacher A epoch1 AMP overflow fraction": float(first["train_amp_overflow_fraction"]),
            "Teacher A epoch1 nonfinite fraction": float(first["train_grad_nonfinite_fraction"]),
            "C0 run directory": c0["Run directory"],
            "Teacher J run directory": c1["Run directory"],
            "Teacher A run directory": str(d2[seed]["run_dir"].relative_to(ROOT)),
            "Student S0 hash": S0_HASHES[seed],
            "Teacher A recorded S0 hash": d2[seed]["summary"]["student_init_checkpoint_sha256"],
            "Teacher A checkpoint SHA256": d2[seed]["checkpoint_sha256"],
            "Status": "COMPLETE",
        })

    with (HERE / "results_multiseed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "complete": True,
        "teacher_a_checkpoint_sha256": TEACHER_A_HASH,
        "formal_invariants": INVARIANTS,
        "rows": rows,
        "statistics": {
            key: summarize([float(row[key]) for row in rows])
            for key in ["Gain_J", "Gain_A", "Q"]
        },
    }
    (HERE / "raw/results_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
