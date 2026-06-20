from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "Baselines" / "outputs"


def psnr_from_mse(value: float) -> float:
    if value <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(value))


def load_metric_tables():
    hvk_1d = pd.read_csv(
        REPO_ROOT / "Main" / "outputs" / "training_analysis" / "hvk_epoch_reconstruction_table.csv"
    )
    hvk_2d = pd.read_csv(
        REPO_ROOT / "Main2" / "outputs" / "training_analysis" / "hvk_epoch_reconstruction_table.csv"
    )
    gan = pd.read_csv(OUTPUT_DIR / "gan" / "gan_training_history.csv")
    with (OUTPUT_DIR / "gan" / "gan_summary.json").open(encoding="utf-8") as handle:
        gan_summary = json.load(handle)
    return hvk_1d, hvk_2d, gan, gan_summary


def build_plot() -> Path:
    hvk_1d, hvk_2d, gan, gan_summary = load_metric_tables()

    final_mse = {
        "1D HVK": float(hvk_1d["reconstruction_mse"].iloc[-1]),
        "2D HVK": float(hvk_2d["reconstruction_mse"].iloc[-1]),
        "GAN": float(gan_summary["metrics"]["mse"]),
    }
    final_psnr = {
        name: psnr_from_mse(value)
        for name, value in final_mse.items()
    }
    colors = {
        "1D HVK": "#4c78a8",
        "2D HVK": "#f58518",
        "GAN": "#54a24b",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].plot(
        hvk_1d["epoch"],
        hvk_1d["reconstruction_mse"],
        label="1D HVK",
        color=colors["1D HVK"],
        linewidth=2.2,
    )
    axes[0].plot(
        hvk_2d["epoch"],
        hvk_2d["reconstruction_mse"],
        label="2D HVK",
        color=colors["2D HVK"],
        linewidth=2.2,
    )
    axes[0].plot(
        gan["epoch"],
        gan["reconstruction_mse"],
        label="GAN",
        color=colors["GAN"],
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE (log scale)")
    axes[0].set_title("Error Descent")
    axes[0].legend(frameon=True)

    names = list(final_mse.keys())
    bars = axes[1].bar(
        names,
        [final_mse[name] for name in names],
        color=[colors[name] for name in names],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Final reconstruction MSE (log scale)")
    axes[1].set_title("Final Error and PSNR")
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

    fig.suptitle("HVK and GAN Reconstruction Metric Benchmark", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = OUTPUT_DIR / "benchmark_metric_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = build_plot()
    print(f"Saved benchmark metric plot to {output_path}")


if __name__ == "__main__":
    main()
