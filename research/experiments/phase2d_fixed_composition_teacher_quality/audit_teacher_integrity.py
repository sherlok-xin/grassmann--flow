#!/usr/bin/env python3
"""Verify the frozen Phase 2D teacher intervention without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

TEACHERS = {
    "J": {
        "run": ROOT / "outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint",
        "sha256": "a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694",
        "validation_nll": 4.300646,
    },
    "A": {
        "run": ROOT / "outputs/hybrid_experiments/20260328_084319_wt2_v1_hybrid_alpha_only",
        "sha256": "43d288db1dba826d8ad1bb3fcbd09b9720bdde24c198e858e00ce4c4658df012",
        "validation_nll": 6.376223,
    },
}

EXPECTED_SOURCE_HASHES = {
    "grassmann_checkpoint": "6916246499dc2e6066c21760421571c76d9f63f1006fab7de838534ef2f733f4",
    "transformer_checkpoint": "4c166d20245bd923dbe126b3f9be39dbb1947d568358fd6cc152c7b9faaafe6c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    marker = "outputs/"
    if marker in value:
        remapped = ROOT / value[value.index(marker) :]
        if remapped.exists():
            return remapped.resolve()
    raise FileNotFoundError(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(HERE / "raw/teacher_integrity.json"))
    args = parser.parse_args()

    import torch

    records = {}
    states = {}
    for label, spec in TEACHERS.items():
        run_dir = spec["run"]
        checkpoint = run_dir / "checkpoints/hybrid_best.pt"
        config = load_json(run_dir / "config.json")
        summary = load_json(run_dir / "summary.json")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_hash = sha256_file(checkpoint)
        sources = config["source_runs"]
        source_hashes = {
            key: sha256_file(resolve_recorded_path(sources[key]))
            for key in EXPECTED_SOURCE_HASHES
        }
        schema = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
            for key, value in state.items()
        }
        records[label] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_hash,
            "expected_checkpoint_sha256": spec["sha256"],
            "checkpoint_hash_matches": checkpoint_hash == spec["sha256"],
            "source_runs": sources,
            "source_checkpoint_sha256": source_hashes,
            "source_hashes_match_expected": source_hashes == EXPECTED_SOURCE_HASHES,
            "train_mode": config.get("config", {}).get("train_mode"),
            "late_k": summary.get("hybrid", {}).get("late_k", config.get("config", {}).get("late_k", 1)),
            "state_tensor_count": len(state),
            "state_numel": sum(int(value.numel()) for value in state.values()),
            "state_schema": schema,
            "recorded_validation_nll": spec["validation_nll"],
            "forced_effective_alpha": 0.5,
            "fusion_semantics": "0.5 * transformer_logits + 0.5 * grassmann_logits",
        }
        states[label] = state

    j_state = states["J"]
    a_state = states["A"]
    same_keys = list(j_state) == list(a_state)
    compatible_shapes = True
    differing = []
    shape_mismatches = []
    if set(j_state) != set(a_state):
        same_keys = False
    for key in sorted(set(j_state) & set(a_state)):
        j_value = j_state[key]
        a_value = a_state[key]
        if key == "logit_alpha" and j_value.numel() == a_value.numel():
            a_value = a_value.reshape_as(j_value)
        if j_value.shape != a_value.shape:
            compatible_shapes = False
            shape_mismatches.append({"key": key, "J": list(j_value.shape), "A": list(a_value.shape)})
            continue
        if not torch.equal(j_value, a_value):
            differing.append(key)

    j_sources = records["J"]["source_runs"]
    a_sources = records["A"]["source_runs"]
    source_identity_matches = all(
        str(j_sources.get(key)) == str(a_sources.get(key))
        for key in [
            "grassmann_run_dir",
            "transformer_run_dir",
            "grassmann_checkpoint",
            "transformer_checkpoint",
        ]
    )
    schema_matches = records["J"]["state_schema"] == records["A"]["state_schema"]
    if not schema_matches:
        # A historical scalar alpha is compatible with the current length-one alpha.
        j_schema = dict(records["J"]["state_schema"])
        a_schema = dict(records["A"]["state_schema"])
        if "logit_alpha" in j_schema and "logit_alpha" in a_schema:
            j_schema["logit_alpha"] = {**j_schema["logit_alpha"], "shape": [1]}
            a_schema["logit_alpha"] = {**a_schema["logit_alpha"], "shape": [1]}
        schema_matches = j_schema == a_schema

    checks = {
        "teacher_hashes_match": all(records[label]["checkpoint_hash_matches"] for label in records),
        "source_hashes_match": all(records[label]["source_hashes_match_expected"] for label in records),
        "source_identity_matches": source_identity_matches,
        "state_keys_match": same_keys,
        "state_shapes_compatible": compatible_shapes,
        "parameter_schema_matches": schema_matches,
        "late_k_matches": records["J"]["late_k"] == records["A"]["late_k"] == 1,
        "effective_alpha_exact": records["J"]["forced_effective_alpha"] == records["A"]["forced_effective_alpha"] == 0.5,
        "differing_tensor_count_matches_audit": len(differing) == 179,
        "total_tensor_count_matches_audit": len(j_state) == len(a_state) == 191,
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True,
        "teachers": records,
        "comparison": {
            "differing_tensor_count": len(differing),
            "total_tensor_count": len(j_state),
            "differing_tensor_names": differing,
            "shape_mismatches": shape_mismatches,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "gate_passed": payload["gate_passed"],
        "differing_tensor_count": len(differing),
        "total_tensor_count": len(j_state),
        "checks": checks,
    }, indent=2))
    if not payload["gate_passed"]:
        raise SystemExit("Teacher-integrity gate failed")


if __name__ == "__main__":
    main()
