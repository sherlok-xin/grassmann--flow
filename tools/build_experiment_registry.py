#!/usr/bin/env python3
"""Build a non-destructive registry of all experiment artifacts.

The project accumulated several generations of runners with slightly different
JSON schemas.  This script normalizes the metadata that is useful for auditing
comparability without moving, deleting, or rewriting any experiment output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


RUN_ROOTS = (
    "experiments",
    "hybrid_experiments",
    "distill_experiments",
    "transformer_experiments",
)

CSV_FIELDS = (
    "run_dir",
    "family",
    "timestamp",
    "experiment_name",
    "status",
    "result_block",
    "dataset_name",
    "dataset_label_raw",
    "dataset_path",
    "text_field",
    "dataset_source",
    "train_tokens",
    "val_tokens",
    "test_tokens",
    "max_lines",
    "max_seq_len",
    "split_seed",
    "tinystories_val_frac",
    "student_type",
    "init_mode",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "model_dim",
    "num_layers",
    "num_heads",
    "reduced_dim",
    "window_sizes",
    "late_k",
    "kd_loss_mode",
    "distill_alpha",
    "kd_lambda",
    "kd_chunk_tokens",
    "temperature",
    "num_params",
    "best_epoch",
    "best_val_ppl",
    "test_ppl",
    "teacher_test_ppl",
    "teacher_run_dir",
    "student_init_run_dir",
    "replicate_key",
    "control_key",
    "config_path",
    "summary_path",
)

CHECKPOINT_FIELDS = (
    "checkpoint_path",
    "size_mib",
    "run_dir",
    "family",
    "run_status",
    "referenced_by_local_summary",
    "recommendation",
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def first_nonempty(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value not in (None, "", -1):
            return value
    return default


def first_positive(*values: Any, default: Any = "") -> Any:
    for value in values:
        try:
            if float(value) > 0:
                return value
        except (TypeError, ValueError):
            continue
    return default


def nested(mapping: dict[str, Any] | None, *keys: str, default: Any = "") -> Any:
    current: Any = mapping or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def basename(value: Any) -> str:
    return Path(str(value)).name if value else ""


def canonical_dataset_name(raw_name: Any, dataset_path: Any, experiment_name: Any, tags: Any) -> str:
    """Recover the effective dataset from paths when legacy metadata is wrong.

    Several code experiments reused runners whose default dataset label remained
    ``ptb`` or ``wikitext2`` even though ``codeparrot_saved`` was loaded.  The
    path is therefore the strongest signal, followed by the experiment name and
    tags, with the recorded label used only as the final fallback.
    """
    signals = " ".join(
        [
            str(dataset_path or ""),
            str(experiment_name or ""),
            " ".join(tags) if isinstance(tags, list) else str(tags or ""),
        ]
    ).lower()
    if "codeparrot" in signals or "code_" in signals or signals.endswith(" code"):
        return "code"
    if "tinystories" in signals or "tiny_stories" in signals or "ts_" in signals:
        return "tinystories"
    if "wikitext" in signals or "wt2" in signals:
        return "wikitext2"
    if "penn_treebank" in signals or "ptb" in signals:
        return "ptb"
    normalized = str(raw_name or "").strip().lower()
    aliases = {"wt2": "wikitext2", "wikitext-2": "wikitext2"}
    return aliases.get(normalized, normalized or "unknown")


def stable_key(parts: Iterable[Any]) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def extract_timestamp(run_dir: Path) -> str:
    prefix = run_dir.name.split("_", 2)
    if len(prefix) >= 2 and len(prefix[0]) == 8 and len(prefix[1]) == 6:
        return f"{prefix[0]}_{prefix[1]}"
    return ""


def result_blocks(summary: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    if not summary:
        return [("", {})]
    blocks = []
    for name in ("student", "hybrid", "grassmann", "transformer"):
        item = summary.get(name)
        if isinstance(item, dict):
            blocks.append((name, item))
    return blocks or [("", {})]


def run_status(config_path: Path, summary_path: Path) -> str:
    if config_path.is_file() and summary_path.is_file():
        return "complete"
    if summary_path.is_file():
        return "summary_only"
    if config_path.is_file():
        return "config_only"
    return "incomplete"


def all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item)


def build_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outputs = project_root / "outputs"
    for family in RUN_ROOTS:
        family_root = outputs / family
        if not family_root.is_dir():
            continue
        for run_dir in sorted(path for path in family_root.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            summary_path = run_dir / "summary.json"
            meta = load_json(config_path) or {}
            summary = load_json(summary_path) or {}
            cfg = meta.get("config", meta) if isinstance(meta, dict) else {}
            if not isinstance(cfg, dict):
                cfg = {}

            stats = meta.get("dataset_stats", {}) if isinstance(meta, dict) else {}
            train_stats = stats.get("train", {}) if isinstance(stats, dict) else {}
            val_stats = stats.get("validation", {}) if isinstance(stats, dict) else {}
            test_stats = stats.get("test", {}) if isinstance(stats, dict) else {}

            dataset_label_raw = first_nonempty(
                cfg.get("dataset_name"), train_stats.get("dataset_name"), default="unknown"
            )
            dataset_path = first_nonempty(
                cfg.get("dataset_path"), train_stats.get("dataset_path"), default=""
            )
            experiment_name = first_nonempty(
                meta.get("experiment_name"), cfg.get("experiment_name"), run_dir.name
            )
            dataset_name = canonical_dataset_name(
                dataset_label_raw, dataset_path, experiment_name, meta.get("tags", cfg.get("tags", ""))
            )
            max_seq_len = first_positive(
                cfg.get("max_seq_len"), train_stats.get("max_seq_len"), default=""
            )
            max_lines = first_nonempty(cfg.get("max_lines"), train_stats.get("max_lines"), default="")
            split_seed = first_nonempty(
                cfg.get("split_seed"), train_stats.get("split_seed"), default=""
            )
            student_init = first_nonempty(
                cfg.get("student_init_run_dir"),
                nested(meta, "student_init", "student_init_run_dir"),
                default="",
            )
            teacher_run = first_nonempty(
                cfg.get("teacher_run_dir"),
                nested(meta, "source_teacher", "teacher_run_dir"),
                default="",
            )
            distill_alpha = cfg.get("distill_alpha", "")
            kd_loss_mode = cfg.get("kd_loss_mode", "")
            if family == "distill_experiments" and not kd_loss_mode:
                kd_loss_mode = "legacy_batchmean"
            kd_lambda = cfg.get("kd_lambda", "")
            kd_weight = kd_lambda if kd_loss_mode == "token_mean" else distill_alpha
            init_mode = ""
            if family == "distill_experiments":
                init_mode = "warm_start" if student_init else "random_init"

            base_parts = (
                dataset_name,
                basename(dataset_path),
                max_lines,
                max_seq_len,
                split_seed,
                cfg.get("tinystories_val_frac", train_stats.get("tinystories_val_frac", "")),
                cfg.get("student_type", ""),
                init_mode,
                cfg.get("model_dim", ""),
                cfg.get("num_layers", ""),
                cfg.get("num_heads", ""),
                cfg.get("reduced_dim", ""),
                cfg.get("window_sizes", ""),
                cfg.get("student_late_k", cfg.get("late_k", "")),
                cfg.get("epochs", ""),
                cfg.get("batch_size", ""),
                cfg.get("lr", ""),
                cfg.get("weight_decay", ""),
                cfg.get("warmup_ratio", ""),
                cfg.get("temperature", ""),
                kd_loss_mode,
            )
            for block_name, block in result_blocks(summary):
                replicate_key = stable_key((*base_parts, block_name, kd_weight))
                control_key = stable_key((*base_parts, block_name))
                row = {
                    "run_dir": str(run_dir.relative_to(project_root)),
                    "family": family,
                    "timestamp": extract_timestamp(run_dir),
                    "experiment_name": experiment_name,
                    "status": run_status(config_path, summary_path),
                    "result_block": block_name,
                    "dataset_name": dataset_name,
                    "dataset_label_raw": dataset_label_raw,
                    "dataset_path": dataset_path,
                    "text_field": first_nonempty(cfg.get("text_field"), train_stats.get("text_field")),
                    "dataset_source": train_stats.get("source", ""),
                    "train_tokens": train_stats.get("token_count_before_trim", ""),
                    "val_tokens": val_stats.get("token_count_before_trim", ""),
                    "test_tokens": test_stats.get("token_count_before_trim", ""),
                    "max_lines": max_lines,
                    "max_seq_len": max_seq_len,
                    "split_seed": split_seed,
                    "tinystories_val_frac": first_nonempty(
                        cfg.get("tinystories_val_frac"), train_stats.get("tinystories_val_frac")
                    ),
                    "student_type": cfg.get("student_type", ""),
                    "init_mode": init_mode,
                    "seed": cfg.get("seed", ""),
                    "epochs": cfg.get("epochs", ""),
                    "batch_size": cfg.get("batch_size", ""),
                    "learning_rate": cfg.get("lr", ""),
                    "weight_decay": cfg.get("weight_decay", ""),
                    "warmup_ratio": cfg.get("warmup_ratio", ""),
                    "model_dim": cfg.get("model_dim", ""),
                    "num_layers": cfg.get("num_layers", ""),
                    "num_heads": cfg.get("num_heads", ""),
                    "reduced_dim": cfg.get("reduced_dim", ""),
                    "window_sizes": cfg.get("window_sizes", ""),
                    "late_k": cfg.get("student_late_k", cfg.get("late_k", "")),
                    "kd_loss_mode": kd_loss_mode,
                    "distill_alpha": distill_alpha,
                    "kd_lambda": kd_lambda,
                    "kd_chunk_tokens": cfg.get("kd_chunk_tokens", ""),
                    "temperature": cfg.get("temperature", ""),
                    "num_params": block.get("num_params", ""),
                    "best_epoch": block.get("best_epoch", ""),
                    "best_val_ppl": block.get("best_val_ppl", ""),
                    "test_ppl": block.get("test_ppl", ""),
                    "teacher_test_ppl": block.get("teacher_test_ppl", ""),
                    "teacher_run_dir": teacher_run,
                    "student_init_run_dir": student_init,
                    "replicate_key": replicate_key,
                    "control_key": control_key,
                    "config_path": str(config_path.relative_to(project_root)) if config_path.exists() else "",
                    "summary_path": str(summary_path.relative_to(project_root)) if summary_path.exists() else "",
                }
                rows.append(row)
    return rows


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_checkpoint_rows(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    outputs = project_root / "outputs"
    for checkpoint in sorted(outputs.rglob("*.pt")):
        relative = checkpoint.relative_to(project_root)
        family = relative.parts[1]
        run_dir = (
            project_root.joinpath(*relative.parts[:3])
            if len(relative.parts) >= 4
            else checkpoint.parent
        )
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        summary = load_json(summary_path) or {}
        references = list(all_strings(summary))
        referenced = any(
            ref.endswith(checkpoint.name) or ref.endswith(str(checkpoint.relative_to(run_dir)))
            for ref in references
        )
        status = run_status(config_path, summary_path)
        if referenced and status == "complete":
            recommendation = "keep_referenced"
        elif status == "complete":
            recommendation = "review_unreferenced"
        else:
            recommendation = "archive_candidate_incomplete"
        rows.append(
            {
                "checkpoint_path": str(relative),
                "size_mib": f"{checkpoint.stat().st_size / 1024 / 1024:.3f}",
                "run_dir": str(run_dir.relative_to(project_root)),
                "family": family,
                "run_status": status,
                "referenced_by_local_summary": str(referenced).lower(),
                "recommendation": recommendation,
            }
        )
    return rows


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def write_inventory(
    project_root: Path,
    rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    unique_runs = {row["run_dir"] for row in rows}
    status_by_run = {}
    for row in rows:
        status_by_run[row["run_dir"]] = row["status"]
    status_counts = Counter(status_by_run.values())
    family_counts = Counter()
    for run in unique_runs:
        family_counts[run.split("/")[1]] += 1
    dataset_counts = Counter(row["dataset_name"] for row in rows if row["dataset_name"])
    replicate_counts = Counter(
        row["replicate_key"] for row in rows if row["status"] == "complete" and row["test_ppl"] != ""
    )

    output_sizes = {}
    outputs_root = project_root / "outputs"
    for root in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        output_sizes[root.name] = directory_size(root)
    logs_size = directory_size(project_root / "logs") if (project_root / "logs").exists() else 0

    incomplete = sorted(
        (run, status) for run, status in status_by_run.items() if status != "complete"
    )
    repeated_groups = sum(1 for count in replicate_counts.values() if count >= 2)
    checkpoint_groups = Counter(row["recommendation"] for row in checkpoint_rows)
    checkpoint_sizes = Counter()
    for row in checkpoint_rows:
        checkpoint_sizes[row["recommendation"]] += float(row["size_mib"])

    lines = [
        "# Experiment Artifact Inventory",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This inventory is generated without moving or deleting any artifact. The CSV registry is the source for detailed filtering and matched-run audits.",
        "",
        "## Coverage",
        "",
        f"- Run directories indexed: {len(unique_runs)}",
        f"- Complete runs: {status_counts.get('complete', 0)}",
        f"- Config-only runs: {status_counts.get('config_only', 0)}",
        f"- Summary-only runs: {status_counts.get('summary_only', 0)}",
        f"- Other incomplete runs: {status_counts.get('incomplete', 0)}",
        f"- Exact replicate groups with at least two completed rows: {repeated_groups}",
        "",
        "## Runs by family",
        "",
    ]
    for family, count in sorted(family_counts.items()):
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Rows by dataset label", ""])
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"- `{dataset}`: {count}")
    lines.extend(["", "## Storage", ""])
    for family, size in sorted(output_sizes.items()):
        lines.append(f"- `outputs/{family}`: {human_size(size)}")
    lines.append(f"- `logs`: {human_size(logs_size)}")
    lines.extend(["", "## Checkpoint manifest", ""])
    lines.append(f"- Checkpoints indexed: {len(checkpoint_rows)}")
    for recommendation in (
        "keep_referenced",
        "review_unreferenced",
        "archive_candidate_incomplete",
    ):
        lines.append(
            f"- `{recommendation}`: {checkpoint_groups.get(recommendation, 0)} files, "
            f"{checkpoint_sizes.get(recommendation, 0.0) / 1024:.2f} GiB"
        )
    lines.extend(
        [
            "",
            "Most storage is held by model checkpoints. Cleanup should therefore be based on an explicit keep/archive manifest, never on directory age alone.",
            "",
            "## Incomplete run directories",
            "",
        ]
    )
    if incomplete:
        lines.extend(f"- `{run}`: {status}" for run, status in incomplete)
    else:
        lines.append("No incomplete run directory was found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("docs/generated"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(project_root)
    csv_path = output_dir / "experiment_registry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_rows = build_checkpoint_rows(project_root)
    checkpoint_path = output_dir / "checkpoint_manifest.csv"
    with checkpoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_FIELDS)
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    inventory_path = output_dir / "experiment_inventory.md"
    write_inventory(project_root, rows, checkpoint_rows, inventory_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {len(checkpoint_rows)} rows to {checkpoint_path}")
    print(f"Wrote inventory to {inventory_path}")


if __name__ == "__main__":
    main()
