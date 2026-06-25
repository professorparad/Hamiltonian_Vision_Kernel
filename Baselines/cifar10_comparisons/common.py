"""
Shared utilities for CIFAR-32 comparison benchmarks.
All images are 32×32, patches are 8×8, n_sites=6 for MPS.
"""

from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "datasets"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load image at native resolution (no resize) — CIFAR images are already 32×32."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image.astype(np.float32) / 255.0


def image_paths(dataset_dir: Path, count: int) -> list[Path]:
    images_dir = dataset_dir / "images"
    paths = sorted(images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")
    return paths[:count]


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract non-overlapping patches from a 2D grayscale image."""
    h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image dims {h}×{w} not divisible by patch_size {patch_size}")
    patches = []
    for row in range(0, h, patch_size):
        for col in range(0, w, patch_size):
            patches.append(image[row:row + patch_size, col:col + patch_size])
    return np.asarray(patches, dtype=np.float32)


def stitch_patches(patches: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    """Reconstruct image from patches."""
    grid = image_size // patch_size
    image = np.zeros((image_size, image_size), dtype=np.float32)
    idx = 0
    for row in range(grid):
        for col in range(grid):
            r0, c0 = row * patch_size, col * patch_size
            image[r0:r0 + patch_size, c0:c0 + patch_size] = patches[idx]
            idx += 1
    return image


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def psnr_from_mse(value: float) -> float:
    if value <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(value))


def simple_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    cov = float(((x - mu_x) * (y - mu_y)).mean())
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / den)


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    mse_val = mse(prediction, target)
    return {
        "mse": mse_val,
        "psnr": psnr_from_mse(mse_val),
        "ssim": simple_ssim(prediction, target),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantum_order_summary(
    observables: torch.Tensor,
    energies: torch.Tensor,
    epoch: int,
    loss: float,
    reconstruction_loss: float,
    previous_order: float | None = None,
) -> dict:
    obs = observables.detach().cpu().numpy()
    energy = energies.detach().cpu().numpy()
    z_obs = obs[:, :6]
    x_obs = obs[:, 6:12]
    patch_order = z_obs.mean(axis=1)
    mean_order = float(patch_order.mean())
    susceptibility = 0.0 if previous_order is None else abs(mean_order - previous_order)
    return {
        "epoch": epoch,
        "total_loss": float(loss),
        "reconstruction_loss": float(reconstruction_loss),
        "mean_energy": float(energy.mean()),
        "mean_order_parameter": mean_order,
        "mean_abs_order_parameter": float(np.abs(patch_order).mean()),
        "mean_transverse_order_parameter": float(x_obs.mean(axis=1).mean()),
        "order_parameter_susceptibility": float(susceptibility),
    }


def save_order_parameter_plot(rows: list[dict], output_path: Path, title: str) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.array([row["epoch"] for row in rows], dtype=np.float32)
    order = np.array([row["mean_order_parameter"] for row in rows], dtype=np.float32)
    abs_order = np.array([row["mean_abs_order_parameter"] for row in rows], dtype=np.float32)
    transverse = np.array(
        [row["mean_transverse_order_parameter"] for row in rows], dtype=np.float32
    )
    susceptibility = np.array(
        [row["order_parameter_susceptibility"] for row in rows], dtype=np.float32
    )
    energy = np.array([row["mean_energy"] for row in rows], dtype=np.float32)
    energy_scale = max(float(np.abs(energy).max()), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, order, label="Z order")
    axes[0].plot(epochs, abs_order, label="|Z order|")
    axes[0].plot(epochs, transverse, label="X order")
    axes[0].set_title("Order parameters")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, susceptibility, color="tab:orange")
    axes[1].set_title("Susceptibility")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, energy / energy_scale, label="energy (scaled)")
    axes[2].plot(epochs, order, label="Z order")
    axes[2].set_title("Energy vs order")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_reconstruction_comparison(
    original: np.ndarray,
    reconstruction: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error = np.abs(original - reconstruction)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    panels = [
        ("Original", original, "gray"),
        ("Reconstruction", reconstruction, "gray"),
        ("Absolute error", error, "magma"),
    ]
    for ax, (panel_title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metric_comparison(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = [row["model"] for row in rows]
    metric_panels = [
        ("mean_mse", "Mean MSE"),
        ("mean_psnr", "Mean PSNR"),
        ("mean_ssim", "Mean SSIM"),
        ("mean_dice", "Mean Dice"),
        ("mean_iou", "Mean IoU"),
    ]
    available = [
        (key, label)
        for key, label in metric_panels
        if any(key in row and row[key] is not None for row in rows)
    ]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.8 * len(available), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, available):
        values = [
            float(row[key]) if key in row and row[key] is not None else np.nan
            for row in rows
        ]
        ax.bar(models, values, color="tab:blue")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
