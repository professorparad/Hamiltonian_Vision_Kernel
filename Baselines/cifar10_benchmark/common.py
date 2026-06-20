from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_DATASET_DIR = REPO_ROOT / "Baselines" / "datasets" / "cifar10_subset"


def image_paths(dataset_dir: Path, count: int) -> list[Path]:
    paths = sorted((dataset_dir / "images").glob("*.png"))[:count]
    if not paths:
        raise FileNotFoundError(f"No PNG images found under {dataset_dir / 'images'}")
    return paths


def psnr_from_mse(value: float) -> float:
    if value <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(value))


def run_command(command: list[str], done_file: Path) -> None:
    if done_file.exists():
        print(f"Skipping completed run: {done_file.parent}")
        return
    done_file.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    done_file.write_text("ok\n", encoding="utf-8")


def read_hvk_metrics(csv_path: Path, model: str, image_name: str) -> dict:
    rows = pd.read_csv(csv_path)
    final = rows.iloc[-1]
    best = rows.loc[rows["reconstruction_mse"].idxmin()]
    critical = rows.loc[rows["order_parameter_susceptibility"].idxmax()]
    final_mse = float(final["reconstruction_mse"])
    return {
        "model": model,
        "image": image_name,
        "final_mse": final_mse,
        "best_mse": float(best["reconstruction_mse"]),
        "best_epoch": int(best["epoch"]),
        "psnr": psnr_from_mse(final_mse),
        "ssim": "",
        "final_energy": float(final["mean_energy"]),
        "critical_epoch": int(critical["epoch"]),
    }


def read_gan_metrics(summary_path: Path, history_path: Path, image_name: str) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path)
    best = history.loc[history["reconstruction_mse"].idxmin()]
    return {
        "model": "GAN",
        "image": image_name,
        "final_mse": float(summary["metrics"]["mse"]),
        "best_mse": float(best["reconstruction_mse"]),
        "best_epoch": int(best["epoch"]),
        "psnr": float(summary["metrics"]["psnr"]),
        "ssim": float(summary["metrics"]["ssim"]),
        "final_energy": "",
        "critical_epoch": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    metrics = {"model": rows[0]["model"], "images": len(rows)}
    for metric in ["final_mse", "best_mse", "psnr"]:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        metrics[f"mean_{metric}"] = float(values.mean())
        metrics[f"std_{metric}"] = float(values.std(ddof=0))
    ssim_values = [float(row["ssim"]) for row in rows if row["ssim"] != ""]
    metrics["mean_ssim"] = float(np.mean(ssim_values)) if ssim_values else ""
    energy_values = [
        float(row["final_energy"]) for row in rows if row["final_energy"] != ""
    ]
    metrics["mean_final_energy"] = (
        float(np.mean(energy_values)) if energy_values else ""
    )
    critical_values = [
        float(row["critical_epoch"]) for row in rows if row["critical_epoch"] != ""
    ]
    metrics["mean_critical_epoch"] = (
        float(np.mean(critical_values)) if critical_values else ""
    )
    return metrics


def write_metric_outputs(output_dir: Path, rows: list[dict]) -> dict:
    summary = aggregate(rows)
    write_csv(output_dir / "per_image_metrics.csv", rows)
    write_csv(output_dir / "aggregate_metrics.csv", [summary])
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_summary(output_dir / "metrics_summary.png", rows, summary)
    return summary


def plot_summary(path: Path, rows: list[dict], summary: dict) -> None:
    names = [row["image"].replace("_cifar10_test.png", "") for row in rows]
    mse = [float(row["final_mse"]) for row in rows]
    psnr = [float(row["psnr"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, mse, color="#4c78a8")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{summary['model']} Final MSE")
    axes[0].set_ylabel("MSE (log scale)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(names, psnr, color="#54a24b")
    axes[1].set_title(f"{summary['model']} PSNR")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
