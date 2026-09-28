#!/usr/bin/env python3
"""Create lightweight, content-addressed manifests for the four frozen datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "research" / "snapshots" / "20260920"

DATASETS = {
    "ptb": {
        "path": "/workspace/grassmannflows/datasets/ptb_text_only_saved",
        "source": "ptb_text_only/penn_treebank saved with Hugging Face Datasets",
        "text_field": "sentence",
        "split_scheme": "native train/validation/test",
        "teacher_config": ROOT / "outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint/config.json",
    },
    "wikitext2": {
        "path": "/workspace/grassmannflows/datasets/wikitext2_v1_saved",
        "source": "wikitext/wikitext-2-raw-v1 saved with Hugging Face Datasets",
        "text_field": "text",
        "split_scheme": "native train/validation/test",
        "teacher_config": ROOT / "outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/config.json",
    },
    "tinystories": {
        "path": "/workspace/grassmannflows/datasets/tinystories_saved",
        "source": "roneneldan/TinyStories saved with Hugging Face Datasets",
        "text_field": "text",
        "split_scheme": "train split is deterministically split 98/2 with seed 42; original validation is test",
        "teacher_config": ROOT / "outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10/config.json",
    },
    "codeparrot_common5k": {
        "path": "/workspace/grassmannflows/datasets/codeparrot_saved",
        "source": "locally saved CodeParrot corpus",
        "text_field": "text",
        "split_scheme": "native train/validation/test; first 5,000 non-empty records per split",
        "teacher_config": ROOT / "outputs/hybrid_experiments/20260529_120718_code_teacher_last1_joint/config.json",
    },
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, spec in DATASETS.items():
        path = Path(spec["path"])
        ds = load_from_disk(str(path))
        config = json.loads(spec["teacher_config"].read_text(encoding="utf-8"))
        metadata_hashes = {}
        for item in sorted(path.rglob("*")):
            if not item.is_file() or item.suffix == ".arrow" or item.stat().st_size > 10 * 1024 * 1024:
                continue
            metadata_hashes[str(item.relative_to(path))] = sha256(item)
        splits = {}
        for split_name, split in ds.items():
            splits[split_name] = {
                "rows": len(split),
                "fingerprint": getattr(split, "_fingerprint", None),
                "columns": split.column_names,
                "features": str(split.features),
            }
        manifest = {
            "name": name,
            "source_path": str(path),
            "source_description": spec["source"],
            "text_field": spec["text_field"],
            "split_scheme": spec["split_scheme"],
            "tokenizer": {
                "implementation": "transformers.GPT2Tokenizer",
                "resolved_path": "/workspace/grassmannflows/grassmann-flows/gpt2_local",
                "vocab_size": 50257,
                "add_special_tokens": False,
            },
            "dataset_splits": splits,
            "preprocessing": config.get("dataset_stats", {}),
            "metadata_file_sha256": metadata_hashes,
            "metadata_digest": hashlib.sha256(
                json.dumps(metadata_hashes, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "note": "Arrow payloads were not copied or hashed; Hugging Face fingerprints and metadata hashes identify the mounted snapshot.",
        }
        manifests[name] = manifest
        (SNAPSHOT / f"dataset_manifest_{name}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    combined = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": manifests,
    }
    (SNAPSHOT / "dataset_manifests.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
