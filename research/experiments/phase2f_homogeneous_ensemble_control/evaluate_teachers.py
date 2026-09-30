#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from train_distill_hybrid_lite_from_latefusion_teacher_v2 import (  # noqa: E402
    TextDataset,
    apply_teacher_alpha_override,
    build_search_roots,
    load_teacher_and_context,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_one(run_dir: Path, teacher_type: str, loader: DataLoader, vocab_size: int,
                 device: torch.device) -> dict:
    teacher, context = load_teacher_and_context(
        run_dir, vocab_size, build_search_roots([]), [], device, teacher_type=teacher_type,
    )
    checkpoint_alpha, effective_alpha = apply_teacher_alpha_override(teacher, 0.5)
    teacher.eval()

    fused_nll_sum = 0.0
    branch_nll_sums = None
    jsd_sum = 0.0
    agreement_sum = 0
    token_count = 0
    branch_names = None

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        fused_logits, _, branches = teacher(input_ids, labels=None, return_branches=True)
        if branch_names is None:
            branch_names = list(branches)
            if len(branch_names) != 2:
                raise RuntimeError(f"Expected exactly two branches, found {branch_names}")
            branch_nll_sums = {name: 0.0 for name in branch_names}

        targets = labels[:, 1:].contiguous()
        count = int(targets.numel())
        token_count += count
        shifted_fused = fused_logits[:, :-1, :].contiguous().float()
        fused_nll_sum += float(F.cross_entropy(
            shifted_fused.view(-1, shifted_fused.size(-1)), targets.view(-1), reduction="sum",
        ).item())

        shifted = {}
        for name in branch_names:
            value = branches[name][:, :-1, :].contiguous().float()
            shifted[name] = value
            branch_nll_sums[name] += float(F.cross_entropy(
                value.view(-1, value.size(-1)), targets.view(-1), reduction="sum",
            ).item())

        logp = F.log_softmax(shifted[branch_names[0]], dim=-1)
        logq = F.log_softmax(shifted[branch_names[1]], dim=-1)
        logm = torch.logsumexp(torch.stack([logp, logq], dim=0), dim=0) - math.log(2.0)
        jsd = 0.5 * (
            (logp.exp() * (logp - logm)).sum(dim=-1)
            + (logq.exp() * (logq - logm)).sum(dim=-1)
        )
        jsd_sum += float(jsd.sum().item())
        agreement_sum += int(
            shifted[branch_names[0]].argmax(dim=-1)
            .eq(shifted[branch_names[1]].argmax(dim=-1)).sum().item()
        )

    fused_nll = fused_nll_sum / token_count
    branch_nll = {name: value / token_count for name, value in branch_nll_sums.items()}
    checkpoint = Path(context["hybrid_ckpt"])
    result = {
        "teacher_type": teacher_type,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_alpha": checkpoint_alpha,
        "effective_alpha": effective_alpha,
        "late_k": int(context["late_k"]),
        "total_params": int(sum(parameter.numel() for parameter in teacher.parameters())),
        "branch_params": {
            name: int(sum(parameter.numel() for parameter in getattr(teacher, name).parameters()))
            for name in branch_names
        },
        "validation_tokens": token_count,
        "validation_nll": fused_nll,
        "validation_ppl": math.exp(fused_nll),
        "branch_nll": branch_nll,
        "branch_ppl": {name: math.exp(value) for name, value in branch_nll.items()},
        "branch_jsd": jsd_sum / token_count,
        "branch_top1_agreement": agreement_sum / token_count,
        "fusion_gain_over_better_branch": min(branch_nll.values()) - fused_nll,
    }
    del teacher
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tg-run-dir", required=True)
    parser.add_argument("--tt-run-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tokenizer-dir", default="gpt2_local")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    dataset = TextDataset(
        "validation", tokenizer, max_seq_len=256, dataset_name="wikitext2",
        dataset_path=args.dataset_path, text_field="text", max_lines=0,
        encode_chars_per_batch=200000, tinystories_val_frac=0.02, split_seed=42,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    payload = {
        "protocol": {
            "selection_split": "validation",
            "effective_alpha": 0.5,
            "amp": False,
            "test_selection": False,
        },
        "dataset_stats": dataset.stats,
        "teachers": {},
    }
    for label, run_dir, teacher_type in (
        ("TG", Path(args.tg_run_dir), "tg"),
        ("TT", Path(args.tt_run_dir), "tt"),
    ):
        payload["teachers"][label] = evaluate_one(
            run_dir, teacher_type, loader, len(tokenizer), device,
        )
    payload["comparison"] = {
        "tt_minus_tg_params": (
            payload["teachers"]["TT"]["total_params"]
            - payload["teachers"]["TG"]["total_params"]
        ),
        "tt_minus_tg_validation_nll": (
            payload["teachers"]["TT"]["validation_nll"]
            - payload["teachers"]["TG"]["validation_nll"]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
