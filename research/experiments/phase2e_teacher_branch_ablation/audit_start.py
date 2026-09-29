#!/usr/bin/env python3
"""Record and verify immutable inputs for the Phase 2E intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPECTED = {
    "teacher": (
        ROOT / "outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/checkpoints/hybrid_best.pt",
        "a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694",
    ),
    "s0_seed42": (
        ROOT / "outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20/checkpoints/hybrid_best.pt",
        "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef",
    ),
    "s0_seed123": (
        ROOT / "outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123/checkpoints/hybrid_best.pt",
        "a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13",
    ),
    "s0_seed456": (
        ROOT / "outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456/checkpoints/hybrid_best.pt",
        "3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757",
    ),
}
SOURCE_FILES = [
    "train_distill_hybrid_lite_from_latefusion_teacher_v2.py",
    "src/kd_losses.py",
    "research/experiments/phase2e_teacher_branch_ablation/experiment_plan.md",
    "research/experiments/phase2e_teacher_branch_ablation/preregistered_interpretation.md",
    "research/experiments/phase2e_teacher_branch_ablation/run_branch_utility.py",
    "research/experiments/phase2e_teacher_branch_ablation/run_gradient_preflight.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifacts = {}
    for label, (path, expected_hash) in EXPECTED.items():
        actual_hash = sha256_file(path)
        artifacts[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_matches": actual_hash == expected_hash,
        }
    source_hashes = {
        path: sha256_file(ROOT / path)
        for path in SOURCE_FILES
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status_short": command("git", "status", "--short"),
        "artifacts": artifacts,
        "source_sha256": source_hashes,
        "gpu_inventory": command(
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ),
        "gate_passed": all(item["hash_matches"] for item in artifacts.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["gate_passed"]:
        raise SystemExit("Phase 2E immutable-input gate failed")


if __name__ == "__main__":
    main()
