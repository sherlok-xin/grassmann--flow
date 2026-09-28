#!/usr/bin/env python3
"""Collect compact branch-disagreement and CE--KD conflict diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=str, default="")
    parser.add_argument("--teacher-run-dir", type=str, required=True)
    parser.add_argument("--student-init-run-dir", type=str, required=True)
    parser.add_argument("--ce-run-dir", type=str, required=True)
    parser.add_argument("--kd-run-dir", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--dataset-name", choices=["ptb", "tinystories"], required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--text-field", type=str, required=True)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--tinystories-val-frac", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20260916)
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--diagnostic-chunk-tokens", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reduced-dim", type=int, default=56)
    parser.add_argument("--window-sizes", type=str, default="1,2,4")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--student-late-k", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_student_checkpoint(run_dir: Path, train_module, search_roots):
    summary = load_json(run_dir / "summary.json")
    checkpoint = summary.get("student", {}).get("checkpoint_path")
    if not checkpoint:
        checkpoint = str(run_dir / "checkpoints" / "student_best.pt")
    return train_module.resolve_migrated_path(
        checkpoint,
        run_dir=run_dir,
        search_roots=search_roots,
        remaps=[],
        kind="student_checkpoint",
        must_exist=True,
    )


def resolve_run_dir(value: str, train_module, search_roots, kind: str):
    return Path(
        train_module.resolve_migrated_path(
            value,
            search_roots=search_roots,
            remaps=[],
            kind=kind,
            must_exist=True,
        )
    )


def load_endpoint_student(args, run_dir, train_module, vocab_size, max_seq_len, device, search_roots):
    model = train_module.instantiate_student_from_args(
        "hybrid_lite", vocab_size, max_seq_len, args
    ).to(device)
    checkpoint = resolve_student_checkpoint(run_dir, train_module, search_roots)
    state = train_module.torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, str(Path(checkpoint).resolve())


def main():
    args = parse_args()
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    import torch
    from torch.amp import autocast
    from torch.utils.data import DataLoader, Subset
    from transformers import GPT2Tokenizer

    import train_distill_hybrid_lite_from_latefusion_teacher_v2 as train_module
    from src.branch_diagnostics import token_branch_diagnostics

    if not torch.cuda.is_available():
        raise RuntimeError("The approved diagnostic protocol requires a CUDA device on server 197.")
    device = torch.device("cuda", 0)
    search_roots = train_module.build_search_roots([])

    teacher_run_dir = resolve_run_dir(args.teacher_run_dir, train_module, search_roots, "teacher_run_dir")
    init_run_dir = resolve_run_dir(args.student_init_run_dir, train_module, search_roots, "student_init_run_dir")
    ce_run_dir = resolve_run_dir(args.ce_run_dir, train_module, search_roots, "ce_run_dir")
    kd_run_dir = resolve_run_dir(args.kd_run_dir, train_module, search_roots, "kd_run_dir")
    tokenizer_dir = train_module.resolve_migrated_path(
        args.tokenizer_dir,
        search_roots=search_roots,
        remaps=[],
        kind="tokenizer_dir",
        must_exist=True,
    )
    dataset_path = train_module.resolve_migrated_path(
        args.dataset_path,
        search_roots=search_roots,
        remaps=[],
        kind="dataset_path",
        must_exist=True,
    )

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    vocab_size = len(tokenizer)
    teacher, context = train_module.load_teacher_and_context(
        teacher_run_dir, vocab_size, search_roots, [], device
    )
    max_seq_len = int(context["max_seq_len"])

    validation_dataset = train_module.TextDataset(
        "validation",
        tokenizer,
        max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=str(dataset_path),
        text_field=args.text_field,
        max_lines=args.max_lines,
        encode_chars_per_batch=200000,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )
    if args.max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    selection_count = min(args.max_chunks, len(validation_dataset))
    rng = random.Random(args.selection_seed)
    selected_indices = sorted(rng.sample(range(len(validation_dataset)), selection_count))
    loader = DataLoader(
        Subset(validation_dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    init_student = train_module.instantiate_student_from_args(
        "hybrid_lite", vocab_size, max_seq_len, args
    ).to(device)
    init_info = train_module.maybe_load_student_init(
        init_student, str(init_run_dir), search_roots, [], device
    )
    init_student.eval()
    ce_student, ce_checkpoint = load_endpoint_student(
        args, ce_run_dir, train_module, vocab_size, max_seq_len, device, search_roots
    )
    kd_student, kd_checkpoint = load_endpoint_student(
        args, kd_run_dir, train_module, vocab_size, max_seq_len, device, search_roots
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "token_diagnostics.csv"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    fields = [
        "dataset",
        "chunk_id",
        "position",
        "branch_jsd",
        "branch_jsd_normalized",
        "teacher_entropy",
        "teacher_nll",
        "transformer_nll",
        "grassmann_nll",
        "teacher_correct",
        "transformer_correct",
        "grassmann_correct",
        "grad_dot",
        "grad_cosine",
        "ce_endpoint_nll",
        "kd_endpoint_nll",
        "delta_nll",
    ]

    running = {key: [] for key in fields[3:]}
    selected_cursor = 0
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        with torch.inference_mode():
            for batch_number, (x, y) in enumerate(loader, start=1):
                current_indices = selected_indices[selected_cursor:selected_cursor + x.size(0)]
                selected_cursor += x.size(0)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast("cuda", enabled=True):
                    teacher_fused, _, teacher_branches = teacher(x, return_branches=True)
                    init_logits, _ = init_student(x)
                    ce_logits, _ = ce_student(x)
                    kd_logits, _ = kd_student(x)
                diagnostics = token_branch_diagnostics(
                    init_logits,
                    teacher_fused,
                    teacher_branches["transformer"],
                    teacher_branches["grassmann"],
                    y,
                    temperature=args.temperature,
                    chunk_tokens=args.diagnostic_chunk_tokens,
                    ce_endpoint_logits=ce_logits,
                    kd_endpoint_logits=kd_logits,
                )
                arrays = {key: value.detach().float().cpu().numpy() for key, value in diagnostics.items()}
                for row_idx in range(len(arrays["position"])):
                    local_batch_index = int(arrays["batch_index"][row_idx])
                    values = [
                        float(arrays["branch_jsd"][row_idx]),
                        float(arrays["branch_jsd"][row_idx] / np.log(2.0)),
                        float(arrays["teacher_entropy"][row_idx]),
                        float(arrays["teacher_nll"][row_idx]),
                        float(arrays["transformer_nll"][row_idx]),
                        float(arrays["grassmann_nll"][row_idx]),
                        float(arrays["teacher_correct"][row_idx]),
                        float(arrays["transformer_correct"][row_idx]),
                        float(arrays["grassmann_correct"][row_idx]),
                        float(arrays["grad_dot"][row_idx]),
                        float(arrays["grad_cosine"][row_idx]),
                        float(arrays["ce_endpoint_nll"][row_idx]),
                        float(arrays["kd_endpoint_nll"][row_idx]),
                        float(arrays["delta_nll"][row_idx]),
                    ]
                    writer.writerow(
                        [
                            args.dataset_name,
                            current_indices[local_batch_index],
                            int(arrays["position"][row_idx]),
                            *values,
                        ]
                    )
                    for key, value in zip(fields[3:], values):
                        running[key].append(value)
                print(
                    f"batch={batch_number}/{len(loader)} chunks={selected_cursor}/{selection_count} "
                    f"tokens={sum(len(v) for v in [running['branch_jsd']])}",
                    flush=True,
                )
                del teacher_fused, teacher_branches, init_logits, ce_logits, kd_logits, diagnostics

    selected_digest = hashlib.sha256(
        ",".join(str(index) for index in selected_indices).encode("utf-8")
    ).hexdigest()
    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset": args.dataset_name,
        "dataset_path": str(Path(dataset_path).resolve()),
        "validation_dataset_stats": validation_dataset.stats,
        "selection_seed": args.selection_seed,
        "selected_chunk_count": selection_count,
        "selected_indices_sha256": selected_digest,
        "selected_indices": selected_indices,
        "teacher_run_dir": str(teacher_run_dir.resolve()),
        "student_init_run_dir": str(init_run_dir.resolve()),
        "student_init_checkpoint": init_info["student_init_checkpoint"],
        "ce_run_dir": str(ce_run_dir.resolve()),
        "ce_checkpoint": ce_checkpoint,
        "kd_run_dir": str(kd_run_dir.resolve()),
        "kd_checkpoint": kd_checkpoint,
        "temperature": args.temperature,
        "raw_text_saved": False,
        "raw_logits_saved": False,
        "arguments": vars(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dataset": args.dataset_name,
        "chunks": selection_count,
        "valid_tokens": len(running["branch_jsd"]),
        "metrics": {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for key, values in running.items()
        },
        "negative_gradient_cosine_fraction": float(
            np.mean(np.asarray(running["grad_cosine"]) < 0)
        ),
        "positive_delta_nll_fraction": float(
            np.mean(np.asarray(running["delta_nll"]) > 0)
        ),
        "outputs": {
            "token_diagnostics": str(csv_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
