import json
import csv
from pathlib import Path

root = Path("outputs/experiments")
rows = []

for run_dir in sorted(root.glob("*")):
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    if not config_path.exists() or not summary_path.exists():
        continue

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for model_name, item in summary.items():
        rows.append({
            "run_id": config["run_id"],
            "experiment_name": config["experiment_name"],
            "model": model_name,
            "notes": config.get("notes", ""),
            "tags": ",".join(config.get("tags", [])),
            "epochs": config["config"]["epochs"],
            "batch_size": config["config"]["batch_size"],
            "model_dim": config["config"]["model_dim"],
            "num_layers": config["config"]["num_layers"],
            "reduced_dim": config["config"].get("reduced_dim", ""),
            "window_sizes": config["config"].get("window_sizes", ""),
            "dropout": config["config"].get("dropout", ""),
            "num_params": item["num_params"],
            "best_epoch": item["best_epoch"],
            "best_val_ppl": item["best_val_ppl"],
            "test_ppl": item["test_ppl"],
        })

out_path = root / "all_results.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {out_path}")