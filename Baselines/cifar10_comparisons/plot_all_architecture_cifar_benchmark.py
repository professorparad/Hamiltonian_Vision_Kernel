from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_PATH = OUTPUT_DIR / "cifar32_aggregate_metrics.csv"
FIGURE_PATH = OUTPUT_DIR / "cifar32_all_architecture_benchmark.png"


def load_rows() -> list[dict]:
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row: dict, key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in {"", None} else float("nan")


def main() -> None:
    rows = load_rows()
    models = [row["model"] for row in rows]
    mean_mse = np.array([value(row, "mean_mse") for row in rows], dtype=float)
    best_mse = np.array([value(row, "best_mse") for row in rows], dtype=float)
    mean_psnr = np.array([value(row, "mean_psnr") for row in rows], dtype=float)

    colors = {
        "Autoencoder": "#8c564b",
        "CNN": "#9467bd",
        "GAN": "#2ca02c",
        "HVK1D": "#4c78a8",
        "HVK2D": "#f58518",
        "MLP": "#17becf",
        "PHL": "#d62728",
        "SymmetricHVK1D": "#7f7f7f",
    }
    bar_colors = [colors.get(model, "#4c78a8") for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("CIFAR Reconstruction Metric Benchmark Across Architectures", fontsize=18, y=0.98)

    x = np.arange(len(models))
    width = 0.36
    axes[0].bar(x - width / 2, mean_mse, width, label="Mean MSE", color=bar_colors, alpha=0.82)
    axes[0].bar(x + width / 2, best_mse, width, label="Best MSE", color=bar_colors, alpha=0.45, hatch="//")
    axes[0].set_title("Error Comparison", fontsize=14)
    axes[0].set_ylabel("Reconstruction MSE (log scale)")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=35, ha="right")
    axes[0].grid(True, which="both", axis="y", alpha=0.35)
    axes[0].legend()

    axes[1].bar(x, mean_psnr, color=bar_colors, alpha=0.9)
    axes[1].set_title("Mean PSNR with Mean MSE Labels", fontsize=14)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_ylim(0, float(np.nanmax(mean_psnr)) * 1.16)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.35)
    for index, (psnr, mse) in enumerate(zip(mean_psnr, mean_mse)):
        axes[1].annotate(
            f"{mse:.2e}\n{psnr:.2f} dB",
            xy=(index, psnr),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
