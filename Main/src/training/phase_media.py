from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def save_epoch_frame(
    *,
    original: np.ndarray,
    reconstruction: np.ndarray,
    epoch: int,
    total_loss: float,
    order_parameter: float,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / f"epoch_{epoch:04d}.png"

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(np.clip(reconstruction, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Epoch {epoch}")
    axes[1].axis("off")
    fig.suptitle(
        f"loss={total_loss:.5f} | order={order_parameter:.5f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)
    return frame_path


def save_order_parameter_plot(epoch_rows: list[dict], output_path: str | Path):
    if not epoch_rows:
        return None

    output_path = Path(output_path)
    epochs = np.array([row["epoch"] for row in epoch_rows], dtype=np.int32)
    order = np.array(
        [row["mean_order_parameter"] for row in epoch_rows], dtype=np.float32
    )
    susceptibility = np.array(
        [row["order_parameter_susceptibility"] for row in epoch_rows], dtype=np.float32
    )
    energy = np.array([row["mean_energy"] for row in epoch_rows], dtype=np.float32)
    energy_scale = np.max(np.abs(energy)) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, order, label="order parameter")
    axes[0].plot(epochs, energy / energy_scale, label="energy (scaled)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Order Parameter and Energy")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, susceptibility, color="orange", label="susceptibility")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Susceptibility")
    axes[1].set_title("Phase Transition Signal")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_order_parameter_gif(
    epoch_rows: list[dict],
    phase_transition: dict,
    output_path: str | Path,
    duration_ms: int = 120,
):
    if not epoch_rows:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    output_path = Path(output_path)
    epochs = np.array([row["epoch"] for row in epoch_rows], dtype=np.int32)
    order = np.array(
        [row["mean_order_parameter"] for row in epoch_rows], dtype=np.float32
    )
    susceptibility = np.array(
        [row["order_parameter_susceptibility"] for row in epoch_rows], dtype=np.float32
    )
    critical_epoch = int(phase_transition.get("critical_epoch", -1))
    has_transition = bool(phase_transition.get("detected", False))

    order_pad = max(float(np.ptp(order)) * 0.1, 1e-3)
    sus_pad = max(float(np.ptp(susceptibility)) * 0.1, 1e-3)
    order_ylim = (float(order.min() - order_pad), float(order.max() + order_pad))
    sus_ylim = (
        float(susceptibility.min() - sus_pad),
        float(susceptibility.max() + sus_pad),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for frame_index in range(1, len(epoch_rows) + 1):
            current_epochs = epochs[:frame_index]
            current_order = order[:frame_index]
            current_sus = susceptibility[:frame_index]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(current_epochs, current_order, color="tab:blue", linewidth=2)
            axes[0].scatter(
                current_epochs[-1], current_order[-1], color="tab:blue", s=35
            )
            axes[0].set_title("Order Parameter vs Epoch")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Order Parameter")
            axes[0].set_ylim(order_ylim)
            axes[0].grid(True)

            axes[1].plot(current_epochs, current_sus, color="tab:orange", linewidth=2)
            axes[1].scatter(
                current_epochs[-1], current_sus[-1], color="tab:orange", s=35
            )
            axes[1].set_title("Susceptibility Signal")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Susceptibility")
            axes[1].set_ylim(sus_ylim)
            axes[1].grid(True)

            if critical_epoch >= 0 and current_epochs[-1] >= critical_epoch:
                label = (
                    f"phase transition @ {critical_epoch}"
                    if has_transition
                    else f"peak susceptibility @ {critical_epoch}"
                )
                for ax in axes:
                    ax.axvline(
                        critical_epoch,
                        color="crimson",
                        linestyle="--",
                        linewidth=1.4,
                        label=label,
                    )
                    ax.legend(loc="best")

            fig.suptitle(f"HVK Order Parameter Evolution | Epoch {current_epochs[-1]}")
            fig.tight_layout()
            frame_path = Path(tmpdir) / f"order_epoch_{frame_index:04d}.png"
            fig.savefig(frame_path, dpi=120)
            plt.close(fig)
            frame_paths.append(frame_path)

        images = [
            Image.open(path).convert("P", palette=Image.ADAPTIVE)
            for path in frame_paths
        ]
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


def save_phase_transition_order_parameter_gif(
    epoch_rows: list[dict],
    phase_transition: dict,
    output_path: str | Path,
    duration_ms: int = 120,
):
    if not epoch_rows:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([row["epoch"] for row in epoch_rows], dtype=np.int32)
    order = np.array(
        [row["mean_order_parameter"] for row in epoch_rows], dtype=np.float32
    )
    critical_epoch = int(phase_transition.get("critical_epoch", -1))
    has_transition = bool(phase_transition.get("detected", False))

    order_pad = max(float(np.ptp(order)) * 0.1, 1e-3)
    order_ylim = (float(order.min() - order_pad), float(order.max() + order_pad))
    epoch_pad = max(int(np.ptp(epochs)) * 0.03, 1.0)
    epoch_xlim = (float(epochs.min() - epoch_pad), float(epochs.max() + epoch_pad))
    transition_label = (
        f"phase transition @ epoch {critical_epoch}"
        if has_transition
        else f"peak susceptibility @ epoch {critical_epoch}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for frame_index in range(1, len(epoch_rows) + 1):
            current_epochs = epochs[:frame_index]
            current_order = order[:frame_index]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(current_epochs, current_order, color="tab:blue", linewidth=2.4)
            ax.scatter(
                current_epochs[-1],
                current_order[-1],
                color="tab:blue",
                edgecolor="white",
                linewidth=0.8,
                s=70,
                zorder=3,
            )
            if critical_epoch >= 0 and current_epochs[-1] >= critical_epoch:
                ax.axvline(
                    critical_epoch,
                    color="crimson",
                    linestyle="--",
                    linewidth=1.6,
                    label=transition_label,
                )
                ax.legend(loc="best")

            ax.set_xlim(epoch_xlim)
            ax.set_ylim(order_ylim)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Order Parameter")
            ax.set_title("Phase Transition: Epoch vs Order Parameter")
            ax.grid(True, alpha=0.35)
            fig.suptitle(f"HVK Order Parameter Evolution | Epoch {current_epochs[-1]}")
            fig.tight_layout()

            frame_path = Path(tmpdir) / f"phase_order_epoch_{frame_index:04d}.png"
            fig.savefig(frame_path, dpi=120)
            plt.close(fig)
            frame_paths.append(frame_path)

        images = [
            Image.open(path).convert("P", palette=Image.ADAPTIVE)
            for path in frame_paths
        ]
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


def save_merged_phase_transition_gif(
    epoch_rows: list[dict],
    phase_transition: dict,
    reconstruction_frame_paths: list[Path],
    output_path: str | Path,
    duration_ms: int = 120,
):
    if not epoch_rows or not reconstruction_frame_paths:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([row["epoch"] for row in epoch_rows], dtype=np.int32)
    order = np.array(
        [row["mean_order_parameter"] for row in epoch_rows], dtype=np.float32
    )
    susceptibility = np.array(
        [row["order_parameter_susceptibility"] for row in epoch_rows], dtype=np.float32
    )
    critical_epoch = int(phase_transition.get("critical_epoch", -1))
    has_transition = bool(phase_transition.get("detected", False))
    threshold = float(phase_transition.get("susceptibility_threshold", 0.0))
    max_susceptibility = float(phase_transition.get("max_susceptibility", 0.0))
    order_jump = float(phase_transition.get("order_parameter_jump", 0.0))

    order_pad = max(float(np.ptp(order)) * 0.1, 1e-3)
    sus_pad = max(float(np.ptp(susceptibility)) * 0.1, threshold * 0.05, 1e-4)
    epoch_pad = max(int(np.ptp(epochs)) * 0.03, 1.0)
    order_ylim = (float(order.min() - order_pad), float(order.max() + order_pad))
    sus_ylim = (
        0.0,
        float(max(susceptibility.max(), threshold) + sus_pad),
    )
    epoch_xlim = (float(epochs.min() - epoch_pad), float(epochs.max() + epoch_pad))
    transition_label = (
        f"phase transition @ epoch {critical_epoch}"
        if has_transition
        else f"peak susceptibility @ epoch {critical_epoch}"
    )
    proof_text = (
        f"Proof: {transition_label}; susceptibility peak={max_susceptibility:.6f}, "
        f"threshold={threshold:.6f}, order jump={order_jump:.6f}."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        frame_count = min(len(epoch_rows), len(reconstruction_frame_paths))
        for frame_index in range(1, frame_count + 1):
            current_epochs = epochs[:frame_index]
            current_order = order[:frame_index]
            current_sus = susceptibility[:frame_index]
            reconstruction_path = Path(reconstruction_frame_paths[frame_index - 1])
            if not reconstruction_path.exists():
                continue

            with Image.open(reconstruction_path) as reconstruction_image:
                reconstruction = np.asarray(reconstruction_image.convert("RGB"))

            fig = plt.figure(figsize=(14, 7))
            grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.35)
            ax_reconstruction = fig.add_subplot(grid[:, 0])
            ax_order = fig.add_subplot(grid[0, 1])
            ax_susceptibility = fig.add_subplot(grid[1, 1])

            ax_reconstruction.imshow(reconstruction)
            ax_reconstruction.set_title("Original + Reconstruction")
            ax_reconstruction.axis("off")

            ax_order.plot(current_epochs, current_order, color="tab:blue", linewidth=2)
            ax_order.scatter(
                current_epochs[-1],
                current_order[-1],
                color="tab:blue",
                edgecolor="white",
                linewidth=0.8,
                s=55,
                zorder=3,
            )
            ax_order.set_xlim(epoch_xlim)
            ax_order.set_ylim(order_ylim)
            ax_order.set_xlabel("Epoch")
            ax_order.set_ylabel("Order Parameter")
            ax_order.set_title("Epoch vs Order Parameter")
            ax_order.grid(True, alpha=0.35)

            ax_susceptibility.plot(
                current_epochs, current_sus, color="tab:orange", linewidth=2
            )
            ax_susceptibility.scatter(
                current_epochs[-1],
                current_sus[-1],
                color="tab:orange",
                edgecolor="white",
                linewidth=0.8,
                s=55,
                zorder=3,
            )
            ax_susceptibility.axhline(
                threshold,
                color="purple",
                linestyle=":",
                linewidth=1.5,
                label="detection threshold",
            )
            ax_susceptibility.set_xlim(epoch_xlim)
            ax_susceptibility.set_ylim(sus_ylim)
            ax_susceptibility.set_xlabel("Epoch")
            ax_susceptibility.set_ylabel("Susceptibility")
            ax_susceptibility.set_title("Proof Signal")
            ax_susceptibility.grid(True, alpha=0.35)

            if critical_epoch >= 0 and current_epochs[-1] >= critical_epoch:
                for ax in (ax_order, ax_susceptibility):
                    ax.axvline(
                        critical_epoch,
                        color="crimson",
                        linestyle="--",
                        linewidth=1.5,
                        label=transition_label,
                    )
                    ax.legend(loc="best")
            else:
                ax_susceptibility.legend(loc="best")

            fig.suptitle(
                f"HVK Phase Transition + Reconstruction | Epoch {current_epochs[-1]}",
                fontsize=14,
            )
            fig.text(
                0.5,
                0.015,
                textwrap.fill(proof_text, width=130),
                ha="center",
                va="bottom",
                fontsize=10,
            )
            fig.subplots_adjust(
                left=0.04,
                right=0.98,
                bottom=0.11,
                top=0.9,
                wspace=0.22,
                hspace=0.35,
            )

            frame_path = Path(tmpdir) / f"merged_phase_epoch_{frame_index:04d}.png"
            fig.savefig(frame_path, dpi=115)
            plt.close(fig)
            frame_paths.append(frame_path)

        if not frame_paths:
            return None

        images = [
            Image.open(path).convert("P", palette=Image.ADAPTIVE)
            for path in frame_paths
        ]
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


def save_frames_as_gif(
    frame_paths: list[Path],
    output_path: str | Path,
    duration_ms: int = 120,
):
    if not frame_paths:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    images = [
        Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths
    ]
    output_path = Path(output_path)
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
