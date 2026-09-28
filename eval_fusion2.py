import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, "src")
from models import GrassmannGPTv4

from train_exp2 import TextDataset, SmallTransformer
from models import GrassmannGPTv4


def load_summary(run_dir: Path):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(run_dir: Path):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_grassmann(vocab_size, cfg):
    return GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        reduced_dim=cfg.get("reduced_dim", 32),
        ff_dim=4 * cfg["model_dim"],
        window_sizes=[int(x) for x in str(cfg.get("window_sizes", "1,2,4,8,12,16")).split(",") if x.strip()],
        dropout=cfg.get("dropout", 0.1),
    )


def build_transformer(vocab_size, cfg):
    return SmallTransformer(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        num_heads=8,
        ff_dim=4 * cfg["model_dim"],
        dropout=cfg.get("dropout", 0.1),
    )


@torch.no_grad()
def evaluate_fusion(model_g, model_t, dataloader, device, alpha=0.5):
    model_g.eval()
    model_t.eval()

    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits_g, _ = model_g(x, labels=None)
        logits_t, _ = model_t(x, labels=None)

        logits = alpha * logits_t + (1.0 - alpha) * logits_g

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate logits fusion of Grassmann and Transformer")
    parser.add_argument("--grassmann-run-dir", type=str, required=True)
    parser.add_argument("--transformer-run-dir", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    g_run_dir = Path(args.grassmann_run_dir)
    t_run_dir = Path(args.transformer_run_dir)

    g_summary = load_summary(g_run_dir)
    t_summary = load_summary(t_run_dir)
    g_config = load_config(g_run_dir)["config"]
    t_config = load_config(t_run_dir)["config"]

    if "grassmann" not in g_summary:
        raise ValueError(f"{g_run_dir} does not contain grassmann results")
    if "transformer" not in t_summary:
        raise ValueError(f"{t_run_dir} does not contain transformer results")

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    vocab_size = len(tokenizer)

    # use transformer config as the reference for dataset seq len, assuming same dataset regime
    max_seq_len = t_config["max_seq_len"]

    dataset_name = t_config.get("dataset_name", "wikitext2")
    dataset_path = t_config.get("dataset_path", "")
    text_field = t_config.get("text_field", "")

    val_dataset = TextDataset(
        "validation",
        tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        text_field=text_field,
    )

    test_dataset = TextDataset(
        "test",
        tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        text_field=text_field,
    )

    print(f"Fusion eval dataset_name = {dataset_name}")
    print(f"Fusion eval dataset_path = {dataset_path}")
    print(f"Fusion eval text_field = {text_field}")
    print(f"Val chunks = {len(val_dataset)}, Test chunks = {len(test_dataset)}")
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model_g = build_grassmann(vocab_size, g_config)
    model_t = build_transformer(vocab_size, t_config)

    g_ckpt = Path(g_summary["grassmann"]["checkpoint_path"])
    if not g_ckpt.exists():
        g_ckpt = g_run_dir / "checkpoints" / "grassmann_best.pt"

    t_ckpt = Path(t_summary["transformer"]["checkpoint_path"])
    if not t_ckpt.exists():
        t_ckpt = t_run_dir / "checkpoints" / "transformer_best.pt"

    print(f"Loading grassmann checkpoint: {g_ckpt}")
    print(f"Loading transformer checkpoint: {t_ckpt}")

    model_g.load_state_dict(torch.load(g_ckpt, map_location=device))
    model_t.load_state_dict(torch.load(t_ckpt, map_location=device))

    model_g.to(device)
    model_t.to(device)

    alphas = []
    alpha = 0.0
    while alpha <= 1.0 + 1e-8:
        alphas.append(round(alpha, 6))
        alpha += args.alpha_step

    best = None
    sweep = []

    for alpha in alphas:
        val_loss, val_ppl = evaluate_fusion(model_g, model_t, val_loader, device, alpha=alpha)
        record = {
            "alpha": float(alpha),
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
        }
        sweep.append(record)
        print(f"alpha={alpha:.2f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if best is None or val_ppl < best["val_ppl"]:
            best = record.copy()

    test_loss, test_ppl = evaluate_fusion(model_g, model_t, test_loader, device, alpha=best["alpha"])
    best["test_loss"] = float(test_loss)
    best["test_ppl"] = float(test_ppl)

    print("\nBest fusion result:")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    output = {
        "grassmann_run_dir": str(g_run_dir.resolve()),
        "transformer_run_dir": str(t_run_dir.resolve()),
        "best": best,
        "sweep": sweep,
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("fusion_result.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved fusion results to: {out_path}")


if __name__ == "__main__":
    main()