#!/usr/bin/env python3
"""Offline teacher-transfer landscape audit from frozen checkpoints.

This script performs no optimization.  It evaluates the two teacher branches,
offline logit fusions, and logit-level CE/KD gradient compatibility at the
warm-start (S0) and WS+CE (S1) student states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--teacher-run-dir", required=True)
    parser.add_argument("--student-init-run-dir", required=True)
    parser.add_argument("--ce-run-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--tokenizer-dir", default="./gpt2_local")
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--tinystories-val-frac", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20260920)
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--token-block", type=int, default=32)
    parser.add_argument("--alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--historical-alpha", type=float, default=0.5)
    parser.add_argument("--model-dim", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reduced-dim", type=int, default=56)
    parser.add_argument("--window-sizes", default="1,2,4")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--student-late-k", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(value, train_module, search_roots, kind):
    return Path(train_module.resolve_migrated_path(
        value, search_roots=search_roots, remaps=[], kind=kind, must_exist=True
    ))


def resolve_student_checkpoint(run_dir, train_module, search_roots):
    summary = load_json(run_dir / "summary.json")
    checkpoint = summary.get("student", {}).get("checkpoint_path")
    if not checkpoint:
        checkpoint = str(run_dir / "checkpoints" / "student_best.pt")
    return train_module.resolve_migrated_path(
        checkpoint, run_dir=run_dir, search_roots=search_roots, remaps=[],
        kind="student_checkpoint", must_exist=True,
    )


def load_ce_student(args, run_dir, train_module, vocab_size, max_seq_len, device, search_roots):
    model = train_module.instantiate_student_from_args(
        "hybrid_lite", vocab_size, max_seq_len, args
    ).to(device)
    checkpoint = resolve_student_checkpoint(run_dir, train_module, search_roots)
    state = train_module.torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, str(Path(checkpoint).resolve())


class ScalarAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, key, values):
        self.sums[key] += float(values.double().sum().item())
        self.counts[key] += int(values.numel())

    def add_sum(self, key, value, count):
        self.sums[key] += float(value)
        self.counts[key] += int(count)

    def mean(self, key):
        return self.sums[key] / max(self.counts[key], 1)


def main():
    args = parse_args()
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import torch.nn.functional as F
    from torch.amp import autocast
    from torch.utils.data import DataLoader, Subset
    from transformers import GPT2Tokenizer

    import train_distill_hybrid_lite_from_latefusion_teacher_v2 as train_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this bounded evaluation on server 197.")
    if args.temperature <= 0 or args.token_block <= 0:
        raise ValueError("temperature and token-block must be positive")

    device = torch.device("cuda", 0)
    search_roots = train_module.build_search_roots([])
    teacher_dir = resolve_run_dir(args.teacher_run_dir, train_module, search_roots, "teacher_run_dir")
    init_dir = resolve_run_dir(args.student_init_run_dir, train_module, search_roots, "student_init_run_dir")
    ce_dir = resolve_run_dir(args.ce_run_dir, train_module, search_roots, "ce_run_dir")
    tokenizer_dir = train_module.resolve_migrated_path(
        args.tokenizer_dir, search_roots=search_roots, remaps=[],
        kind="tokenizer_dir", must_exist=True,
    )
    dataset_path = train_module.resolve_migrated_path(
        args.dataset_path, search_roots=search_roots, remaps=[],
        kind="dataset_path", must_exist=True,
    )

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    vocab_size = len(tokenizer)
    teacher, context = train_module.load_teacher_and_context(
        teacher_dir, vocab_size, search_roots, [], device
    )
    max_seq_len = int(context["max_seq_len"])
    learned_alpha = float(teacher.alpha().mean().detach().cpu().item())
    alpha_vector = [float(v) for v in teacher.alpha().detach().cpu().view(-1).tolist()]
    if len(alpha_vector) != 1:
        raise RuntimeError(
            "The offline scalar fusion reconstruction is exact only for late_k=1; "
            f"found alpha vector {alpha_vector}."
        )

    dataset = train_module.TextDataset(
        "validation", tokenizer, max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=str(dataset_path),
        text_field=args.text_field,
        max_lines=args.max_lines,
        encode_chars_per_batch=200000,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )
    if args.max_chunks <= 0:
        raise ValueError("max-chunks must be positive")
    selection_count = min(args.max_chunks, len(dataset))
    if selection_count == len(dataset):
        indices = list(range(len(dataset)))
        selection_mode = "full_validation"
    else:
        rng = random.Random(args.selection_seed)
        indices = sorted(rng.sample(range(len(dataset)), selection_count))
        selection_mode = "fixed_random_subset"
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    init_student = train_module.instantiate_student_from_args(
        "hybrid_lite", vocab_size, max_seq_len, args
    ).to(device)
    init_info = train_module.maybe_load_student_init(
        init_student, str(init_dir), search_roots, [], device
    )
    init_student.eval()
    ce_student, ce_checkpoint = load_ce_student(
        args, ce_dir, train_module, vocab_size, max_seq_len, device, search_roots
    )

    grid = [float(v.strip()) for v in args.alpha_grid.split(",") if v.strip()]
    if any(v < 0 or v > 1 for v in grid + [learned_alpha, args.historical_alpha]):
        raise ValueError("all alpha values must lie in [0, 1]")
    unique_alphas = sorted(set(grid + [learned_alpha, args.historical_alpha]))
    alpha_acc = {alpha: ScalarAccumulator() for alpha in unique_alphas}
    branch_acc = ScalarAccumulator()
    complement = defaultdict(int)
    fusion_max_abs_error = 0.0
    total_valid_tokens = 0
    cursor = 0
    eps = torch.finfo(torch.float32).tiny

    with torch.inference_mode():
        for batch_number, (x, y) in enumerate(loader, start=1):
            cursor += x.size(0)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", enabled=True):
                canonical, _, branches = teacher(x, return_branches=True)
                s0_logits, _ = init_student(x)
                s1_logits, _ = ce_student(x)

            t_logits = branches["transformer"][:, :-1, :]
            g_logits = branches["grassmann"][:, :-1, :]
            canonical = canonical[:, :-1, :]
            s0_logits = s0_logits[:, :-1, :]
            s1_logits = s1_logits[:, :-1, :]
            labels = y[:, 1:]
            valid = labels.ne(-100)
            labels = labels[valid]
            t_logits = t_logits[valid]
            g_logits = g_logits[valid]
            canonical = canonical[valid]
            s0_logits = s0_logits[valid]
            s1_logits = s1_logits[valid]
            total_valid_tokens += int(labels.numel())

            reconstructed = learned_alpha * t_logits + (1.0 - learned_alpha) * g_logits
            fusion_max_abs_error = max(
                fusion_max_abs_error,
                float((reconstructed.float() - canonical.float()).abs().max().item()),
            )

            for start in range(0, labels.numel(), args.token_block):
                end = min(start + args.token_block, labels.numel())
                yy = labels[start:end]
                tt = t_logits[start:end].float()
                gg = g_logits[start:end].float()
                ss0 = s0_logits[start:end].float()
                ss1 = s1_logits[start:end].float()
                row = torch.arange(yy.numel(), device=device)

                log_t = F.log_softmax(tt, dim=-1)
                log_g = F.log_softmax(gg, dim=-1)
                p_t = log_t.exp()
                p_g = log_g.exp()
                t_pred = tt.argmax(dim=-1)
                g_pred = gg.argmax(dim=-1)
                t_correct = t_pred.eq(yy)
                g_correct = g_pred.eq(yy)
                log_s0_base = F.log_softmax(ss0, dim=-1)
                log_s1_base = F.log_softmax(ss1, dim=-1)
                branch_acc.add("transformer_nll", -log_t[row, yy])
                branch_acc.add("grassmann_nll", -log_g[row, yy])
                branch_acc.add("s0_nll", -log_s0_base[row, yy])
                branch_acc.add("s1_nll", -log_s1_base[row, yy])
                branch_acc.add_sum("s0_correct", ss0.argmax(dim=-1).eq(yy).sum().item(), yy.numel())
                branch_acc.add_sum("s1_correct", ss1.argmax(dim=-1).eq(yy).sum().item(), yy.numel())
                branch_acc.add_sum("transformer_correct", t_correct.sum().item(), yy.numel())
                branch_acc.add_sum("grassmann_correct", g_correct.sum().item(), yy.numel())
                branch_acc.add("transformer_entropy", -(p_t * log_t).sum(dim=-1))
                branch_acc.add("grassmann_entropy", -(p_g * log_g).sum(dim=-1))
                mixture = 0.5 * (p_t + p_g)
                log_mix = mixture.clamp_min(eps).log()
                jsd = 0.5 * (
                    (p_t * (log_t - log_mix)).sum(dim=-1)
                    + (p_g * (log_g - log_mix)).sum(dim=-1)
                ).clamp_min(0.0)
                branch_acc.add("branch_jsd", jsd)
                branch_acc.add_sum("branch_top1_agreement", t_pred.eq(g_pred).sum().item(), yy.numel())

                for alpha in unique_alphas:
                    acc = alpha_acc[alpha]
                    fused = alpha * tt + (1.0 - alpha) * gg
                    log_f = F.log_softmax(fused, dim=-1)
                    p_f = log_f.exp()
                    f_pred = fused.argmax(dim=-1)
                    acc.add("teacher_nll", -log_f[row, yy])
                    acc.add("teacher_entropy", -(p_f * log_f).sum(dim=-1))
                    acc.add_sum("teacher_correct", f_pred.eq(yy).sum().item(), yy.numel())
                    acc.add_sum("agreement_transformer", f_pred.eq(t_pred).sum().item(), yy.numel())
                    acc.add_sum("agreement_grassmann", f_pred.eq(g_pred).sum().item(), yy.numel())

                    p_f_temp = F.softmax(fused / args.temperature, dim=-1)
                    log_f_temp = F.log_softmax(fused / args.temperature, dim=-1)
                    for state, student in (("s0", ss0), ("s1", ss1)):
                        log_s = F.log_softmax(student, dim=-1)
                        p_s = log_s.exp()
                        kl_t1 = (p_f * (log_f - log_s)).sum(dim=-1).clamp_min(0.0)
                        acc.add(f"kl_teacher_to_{state}_t1", kl_t1)
                        acc.add_sum(
                            f"top1_agreement_{state}",
                            student.argmax(dim=-1).eq(f_pred).sum().item(), yy.numel(),
                        )

                        p_s_temp = F.softmax(student / args.temperature, dim=-1)
                        log_s_temp = F.log_softmax(student / args.temperature, dim=-1)
                        kl_temp = (
                            p_f_temp * (log_f_temp - log_s_temp)
                        ).sum(dim=-1).clamp_min(0.0)
                        acc.add(f"kl_teacher_to_{state}_t{args.temperature:g}", kl_temp)
                        g_ce = p_s.clone()
                        g_ce[row, yy] -= 1.0
                        g_kd = args.temperature * (p_s_temp - p_f_temp)
                        ce_norm = g_ce.square().sum(dim=-1).sqrt()
                        kd_norm = g_kd.square().sum(dim=-1).sqrt()
                        cosine = (g_ce * g_kd).sum(dim=-1) / (
                            ce_norm.clamp_min(eps) * kd_norm.clamp_min(eps)
                        )
                        acc.add(f"ce_grad_norm_{state}", ce_norm)
                        acc.add(f"kd_grad_norm_{state}", kd_norm)
                        acc.add(f"grad_cosine_{state}", cosine.clamp(-1.0, 1.0))
                        acc.add_sum(
                            f"negative_grad_fraction_{state}",
                            cosine.lt(0).sum().item(), yy.numel(),
                        )

                    if abs(alpha - learned_alpha) < 1e-12:
                        p_t_gold = p_t[row, yy]
                        p_g_gold = p_g[row, yy]
                        f_correct = f_pred.eq(yy)
                        complement["transformer_errors"] += int((~t_correct).sum().item())
                        complement["transformer_correct"] += int(t_correct.sum().item())
                        complement["grassmann_corrects_transformer"] += int((~t_correct & g_correct).sum().item())
                        complement["grassmann_harms_transformer"] += int((t_correct & ~g_correct).sum().item())
                        complement["fused_corrects_transformer_with_higher_g_gold"] += int(
                            (~t_correct & f_correct & p_g_gold.gt(p_t_gold)).sum().item()
                        )
                        complement["fused_harms_transformer_with_lower_g_gold"] += int(
                            (t_correct & ~f_correct & p_g_gold.lt(p_t_gold)).sum().item()
                        )

            print(
                f"domain={args.domain} batch={batch_number}/{len(loader)} "
                f"chunks={cursor}/{selection_count} valid_tokens={total_valid_tokens}",
                flush=True,
            )
            del canonical, branches, s0_logits, s1_logits, t_logits, g_logits

    def alpha_record(alpha, source):
        acc = alpha_acc[alpha]
        record = {
            "domain": args.domain,
            "alpha": alpha,
            "alpha_source": source,
            "chunks": selection_count,
            "valid_tokens": total_valid_tokens,
            "teacher_nll": acc.mean("teacher_nll"),
            "teacher_ppl": math.exp(acc.mean("teacher_nll")),
            "teacher_entropy": acc.mean("teacher_entropy"),
            "teacher_top1_accuracy": acc.mean("teacher_correct"),
            "agreement_transformer": acc.mean("agreement_transformer"),
            "agreement_grassmann": acc.mean("agreement_grassmann"),
            "branch_jsd": branch_acc.mean("branch_jsd"),
        }
        for state in ("s0", "s1"):
            record.update({
                f"kl_teacher_to_{state}_t1": acc.mean(f"kl_teacher_to_{state}_t1"),
                f"kl_teacher_to_{state}_t{args.temperature:g}": acc.mean(
                    f"kl_teacher_to_{state}_t{args.temperature:g}"
                ),
                f"top1_agreement_{state}": acc.mean(f"top1_agreement_{state}"),
                f"ce_grad_norm_{state}": acc.mean(f"ce_grad_norm_{state}"),
                f"kd_grad_norm_{state}": acc.mean(f"kd_grad_norm_{state}"),
                f"grad_cosine_{state}": acc.mean(f"grad_cosine_{state}"),
                f"negative_grad_fraction_{state}": acc.mean(f"negative_grad_fraction_{state}"),
            })
        return record

    alpha_rows = [alpha_record(alpha, "grid") for alpha in grid]
    alpha_rows.append(alpha_record(learned_alpha, "learned_canonical"))
    alpha_rows.append(alpha_record(args.historical_alpha, "historical_static"))
    canonical_row = alpha_record(learned_alpha, "learned_canonical")
    branch_metrics = {
        "domain": args.domain,
        "chunks": selection_count,
        "valid_tokens": total_valid_tokens,
        "transformer_nll": branch_acc.mean("transformer_nll"),
        "transformer_ppl": math.exp(branch_acc.mean("transformer_nll")),
        "transformer_entropy": branch_acc.mean("transformer_entropy"),
        "transformer_top1_accuracy": branch_acc.mean("transformer_correct"),
        "grassmann_nll": branch_acc.mean("grassmann_nll"),
        "grassmann_ppl": math.exp(branch_acc.mean("grassmann_nll")),
        "grassmann_entropy": branch_acc.mean("grassmann_entropy"),
        "grassmann_top1_accuracy": branch_acc.mean("grassmann_correct"),
        "branch_jsd": branch_acc.mean("branch_jsd"),
        "branch_top1_agreement": branch_acc.mean("branch_top1_agreement"),
        "s0_nll": branch_acc.mean("s0_nll"),
        "s0_ppl": math.exp(branch_acc.mean("s0_nll")),
        "s0_top1_accuracy": branch_acc.mean("s0_correct"),
        "s1_nll": branch_acc.mean("s1_nll"),
        "s1_ppl": math.exp(branch_acc.mean("s1_nll")),
        "s1_top1_accuracy": branch_acc.mean("s1_correct"),
        "canonical_alpha": learned_alpha,
        "fused_nll": canonical_row["teacher_nll"],
        "fused_ppl": canonical_row["teacher_ppl"],
        "fused_entropy": canonical_row["teacher_entropy"],
        "fused_top1_accuracy": canonical_row["teacher_top1_accuracy"],
        "fusion_improvement_over_best_branch_nll": min(
            branch_acc.mean("transformer_nll"), branch_acc.mean("grassmann_nll")
        ) - canonical_row["teacher_nll"],
        "fusion_max_abs_reconstruction_error": fusion_max_abs_error,
    }
    complementarity = dict(complement)
    complementarity.update({
        "valid_tokens": total_valid_tokens,
        "correction_fraction_of_transformer_errors": (
            complement["fused_corrects_transformer_with_higher_g_gold"]
            / max(complement["transformer_errors"], 1)
        ),
        "harm_fraction_of_transformer_correct": (
            complement["fused_harms_transformer_with_lower_g_gold"]
            / max(complement["transformer_correct"], 1)
        ),
    })
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": args.domain,
        "evaluation_only": True,
        "selection_mode": selection_mode,
        "selection_seed": args.selection_seed,
        "selected_indices": indices,
        "selected_indices_sha256": hashlib.sha256(
            ",".join(map(str, indices)).encode("utf-8")
        ).hexdigest(),
        "validation_dataset_stats": dataset.stats,
        "teacher_run_dir": str(teacher_dir.resolve()),
        "teacher_checkpoint": context["hybrid_ckpt"],
        "teacher_source_grassmann_checkpoint": context["g_ckpt"],
        "teacher_source_transformer_checkpoint": context["t_ckpt"],
        "student_init_run_dir": str(init_dir.resolve()),
        "student_init_checkpoint": init_info["student_init_checkpoint"],
        "ce_run_dir": str(ce_dir.resolve()),
        "ce_checkpoint": ce_checkpoint,
        "fusion_convention": "alpha * transformer_logits + (1-alpha) * grassmann_logits",
        "canonical_alpha_vector": alpha_vector,
        "temperature": args.temperature,
        "raw_text_saved": False,
        "raw_logits_saved": False,
        "arguments": vars(args),
    }
    result = {
        "manifest": manifest,
        "branch_metrics": branch_metrics,
        "alpha_landscape": alpha_rows,
        "complementarity": complementarity,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "domain": args.domain,
        "chunks": selection_count,
        "valid_tokens": total_valid_tokens,
        "branch_metrics": branch_metrics,
        "complementarity": complementarity,
        "output": str((output_dir / "results.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
