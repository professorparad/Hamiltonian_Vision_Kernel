from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_reconstruction_frame(original, reconstruction, epoch, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:04d}.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(np.clip(reconstruction, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Epoch {epoch}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_order_curve(epoch_rows, output_path: Path):
    epochs = np.array([row["epoch"] for row in epoch_rows])
    order = np.array([row["mean_order_parameter"] for row in epoch_rows])
    susceptibility = np.array(
        [row["order_parameter_susceptibility"] for row in epoch_rows]
    )
    energy = np.array([row["mean_energy"] for row in epoch_rows])
    energy_scale = np.max(np.abs(energy)) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, order, label="order parameter")
    axes[0].plot(epochs, energy / energy_scale, label="energy (scaled)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, susceptibility, color="orange", label="susceptibility")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_gif(frame_paths, output_path: Path, duration_ms=120):
    if not frame_paths:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()
    return output_path
