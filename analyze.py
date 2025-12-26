"""
Analysis and visualization of training results.

Usage:
    python analyze.py --runs outputs/gpt2_* outputs/grassmann_*
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load all metrics from a run directory."""
    metrics = {}

    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            metrics["config"] = json.load(f)

    # Load training metrics
    for name in ["train_metrics", "eval_metrics", "epoch_metrics"]:
        for model_type in ["gpt2", "grassmann"]:
            path = run_dir / f"{model_type}_{name}.json"
            if path.exists():
                with open(path) as f:
                    metrics[name] = json.load(f)
                    metrics["model_type"] = model_type
                    break

    return metrics


def plot_training_curves(runs: List[Dict], output_path: Path):
    """Plot training loss curves for all runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"gpt2": "#2196F3", "grassmann": "#4CAF50"}
    labels = {"gpt2": "GPT-2 (Attention)", "grassmann": "GrassmannGPT (No Attention)"}

    # Training loss
    ax = axes[0, 0]
    for run in runs:
        if "train_metrics" not in run:
            continue
        model = run["model_type"]
        steps = [m["step"] for m in run["train_metrics"]]
        losses = [m["loss"] for m in run["train_metrics"]]
        ax.plot(steps, losses, color=colors[model], label=labels[model], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training perplexity
    ax = axes[0, 1]
    for run in runs:
        if "train_metrics" not in run:
            continue
        model = run["model_type"]
        steps = [m["step"] for m in run["train_metrics"]]
        ppls = [m["perplexity"] for m in run["train_metrics"]]
        ax.plot(steps, ppls, color=colors[model], label=labels[model], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Training Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Validation loss
    ax = axes[1, 0]
    for run in runs:
        if "eval_metrics" not in run:
            continue
        model = run["model_type"]
        steps = [m["step"] for m in run["eval_metrics"]]
        losses = [m["val_loss"] for m in run["eval_metrics"]]
        ax.plot(steps, losses, color=colors[model], label=labels[model], marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation perplexity
    ax = axes[1, 1]
    for run in runs:
        if "eval_metrics" not in run:
            continue
        model = run["model_type"]
        steps = [m["step"] for m in run["eval_metrics"]]
        ppls = [m["val_perplexity"] for m in run["eval_metrics"]]
        ax.plot(steps, ppls, color=colors[model], label=labels[model], marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_epoch_comparison(runs: List[Dict], output_path: Path):
    """Plot epoch-level comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"gpt2": "#2196F3", "grassmann": "#4CAF50"}
    labels = {"gpt2": "GPT-2", "grassmann": "GrassmannGPT"}

    # Collect final epoch metrics
    final_metrics = {}
    for run in runs:
        if "epoch_metrics" not in run or not run["epoch_metrics"]:
            continue
        model = run["model_type"]
        final = run["epoch_metrics"][-1]
        final_metrics[model] = final

    if len(final_metrics) < 2:
        print("Not enough data for epoch comparison")
        return

    models = list(final_metrics.keys())
    x = np.arange(len(models))
    width = 0.35

    # Final loss comparison
    ax = axes[0]
    train_losses = [final_metrics[m]["train_loss"] for m in models]
    val_losses = [final_metrics[m]["val_loss"] for m in models]
    bars1 = ax.bar(x - width/2, train_losses, width, label="Train Loss", color=[colors[m] for m in models], alpha=0.7)
    bars2 = ax.bar(x + width/2, val_losses, width, label="Val Loss", color=[colors[m] for m in models], alpha=1.0)
    ax.set_ylabel("Loss")
    ax.set_title("Final Epoch Loss Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Final perplexity comparison
    ax = axes[1]
    train_ppls = [final_metrics[m]["train_perplexity"] for m in models]
    val_ppls = [final_metrics[m]["val_perplexity"] for m in models]
    bars1 = ax.bar(x - width/2, train_ppls, width, label="Train PPL", color=[colors[m] for m in models], alpha=0.7)
    bars2 = ax.bar(x + width/2, val_ppls, width, label="Val PPL", color=[colors[m] for m in models], alpha=1.0)
    ax.set_ylabel("Perplexity")
    ax.set_title("Final Epoch Perplexity Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved epoch comparison to {output_path}")


def plot_throughput(runs: List[Dict], output_path: Path):
    """Plot training throughput comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"gpt2": "#2196F3", "grassmann": "#4CAF50"}
    labels = {"gpt2": "GPT-2 (Attention)", "grassmann": "GrassmannGPT (No Attention)"}

    for run in runs:
        if "train_metrics" not in run:
            continue
        model = run["model_type"]
        steps = [m["step"] for m in run["train_metrics"]]
        tps = [m["tokens_per_sec"] for m in run["train_metrics"]]

        # Smooth with moving average
        window = 10
        if len(tps) > window:
            tps_smooth = np.convolve(tps, np.ones(window)/window, mode='valid')
            steps_smooth = steps[window-1:]
            ax.plot(steps_smooth, tps_smooth, color=colors[model], label=labels[model], alpha=0.8)
        else:
            ax.plot(steps, tps, color=colors[model], label=labels[model], alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Training Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved throughput plot to {output_path}")


def print_summary(runs: List[Dict]):
    """Print a summary of all runs."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    for run in runs:
        model = run.get("model_type", "unknown")
        config = run.get("config", {})

        print(f"\n{model.upper()}")
        print("-"*40)

        if config:
            print(f"  Model dim: {config.get('model_dim', 'N/A')}")
            print(f"  Layers: {config.get('num_layers', 'N/A')}")
            print(f"  Batch size: {config.get('batch_size', 'N/A')}")
            print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")

        if "epoch_metrics" in run and run["epoch_metrics"]:
            final = run["epoch_metrics"][-1]
            print(f"\n  Final Results (Epoch {final['epoch']}):")
            print(f"    Train Loss: {final['train_loss']:.4f}")
            print(f"    Train PPL:  {final['train_perplexity']:.2f}")
            print(f"    Val Loss:   {final['val_loss']:.4f}")
            print(f"    Val PPL:    {final['val_perplexity']:.2f}")
            print(f"    Total time: {final['wall_time']/60:.1f} min")

        if "eval_metrics" in run and run["eval_metrics"]:
            best = min(run["eval_metrics"], key=lambda x: x["val_loss"])
            print(f"\n  Best Validation:")
            print(f"    Step: {best['step']}")
            print(f"    Val Loss: {best['val_loss']:.4f}")
            print(f"    Val PPL:  {best['val_perplexity']:.2f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument(
        "--runs", nargs="+", type=str, required=True,
        help="Paths to run directories"
    )
    parser.add_argument(
        "--output", type=str, default="analysis",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = []
    for run_path in args.runs:
        path = Path(run_path)
        if path.exists():
            metrics = load_metrics(path)
            if metrics:
                runs.append(metrics)
                print(f"Loaded: {path}")

    if not runs:
        print("No valid runs found!")
        return

    # Print summary
    print_summary(runs)

    # Generate plots
    plot_training_curves(runs, output_dir / "training_curves.png")
    plot_epoch_comparison(runs, output_dir / "epoch_comparison.png")
    plot_throughput(runs, output_dir / "throughput.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
