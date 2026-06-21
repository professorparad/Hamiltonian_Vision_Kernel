#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "Baselines" / "outputs"
BENCHMARK_DIR = REPO_ROOT / "Baselines" / "cifar10_benchmark"

def psnr_from_mse(value: float) -> float:
    if value <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(value))

def load_and_average_history(model_type: str, history_filename: str, num_images: int = 10) -> tuple[np.ndarray, np.ndarray]:
    mse_all = []
    epochs = None
    
    for i in range(num_images):
        image_name = f"{i:04d}_cifar10_test"
        csv_path = BENCHMARK_DIR / model_type / "outputs" / "runs" / image_name / history_filename
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)
        if epochs is None:
            epochs = df["epoch"].values
        
        mse_all.append(df["reconstruction_mse"].values)
        
    if not mse_all:
        raise ValueError(f"No history data found for {model_type}")
        
    mse_mean = np.mean(mse_all, axis=0)
    return epochs, mse_mean

def build_plot() -> Path:
    num_images = 10
    
    print("Loading 1D HVK history...")
    epochs_1d, mse_1d = load_and_average_history("hvk1d", "hvk_epoch_reconstruction_table.csv", num_images)
    
    print("Loading 2D HVK history...")
    epochs_2d, mse_2d = load_and_average_history("hvk2d", "hvk_epoch_reconstruction_table.csv", num_images)
    
    print("Loading GAN history...")
    epochs_gan, mse_gan = load_and_average_history("gan", "gan_training_history.csv", num_images)
    
    # Load aggregate metrics json files for the final bar chart values
    with (BENCHMARK_DIR / "hvk1d" / "outputs" / "aggregate_metrics.json").open(encoding="utf-8") as f:
        metrics_1d = json.load(f)
    with (BENCHMARK_DIR / "hvk2d" / "outputs" / "aggregate_metrics.json").open(encoding="utf-8") as f:
        metrics_2d = json.load(f)
    with (BENCHMARK_DIR / "gan" / "outputs" / "aggregate_metrics.json").open(encoding="utf-8") as f:
        metrics_gan = json.load(f)
        
    final_mse = {
        "1D HVK": metrics_1d["mean_final_mse"],
        "2D HVK": metrics_2d["mean_final_mse"],
        "GAN": metrics_gan["mean_final_mse"]
    }
    
    final_psnr = {
        "1D HVK": metrics_1d["mean_psnr"],
        "2D HVK": metrics_2d["mean_psnr"],
        "GAN": metrics_gan["mean_psnr"]
    }
    
    colors = {
        "1D HVK": "#4c78a8",
        "2D HVK": "#f58518",
        "GAN": "#54a24b",
    }
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    
    # Left subplot: MSE history descent
    axes[0].plot(
        epochs_1d,
        mse_1d,
        label="1D HVK",
        color=colors["1D HVK"],
        linewidth=2.2,
    )
    axes[0].plot(
        epochs_2d,
        mse_2d,
        label="2D HVK",
        color=colors["2D HVK"],
        linewidth=2.2,
    )
    axes[0].plot(
        epochs_gan,
        mse_gan,
        label="GAN",
        color=colors["GAN"],
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Reconstruction MSE (log scale)")
    axes[0].set_title("CIFAR-10 Error Descent (10-Image Mean)")
    axes[0].legend(frameon=True)
    
    # Right subplot: Final MSE and PSNR comparison
    names = list(final_mse.keys())
    bars = axes[1].bar(
        names,
        [final_mse[name] for name in names],
        color=[colors[name] for name in names],
        width=0.5
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Final mean MSE (log scale)")
    axes[1].set_title("Final CIFAR-10 Mean Error and PSNR")
    
    for bar, name in zip(bars, names):
        mse_value = final_mse[name]
        psnr_value = final_psnr[name]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            mse_value * 1.18,
            f"{mse_value:.2e}\n{psnr_value:.2f} dB",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        
    fig.suptitle("HVK and GAN Reconstruction Metric Benchmark (CIFAR-10 Test Subset)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    
    output_path = OUTPUT_DIR / "cifar10_metric_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main() -> None:
    output_path = build_plot()
    print(f"Saved CIFAR-10 benchmark metric plot to {output_path}")

if __name__ == "__main__":
    main()
