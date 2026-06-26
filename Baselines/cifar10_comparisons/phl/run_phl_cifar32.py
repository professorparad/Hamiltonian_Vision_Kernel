"""
Parameterized Hamiltonian Learning baseline on CIFAR-10 at native 32x32.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    compute_metrics,
    image_paths,
    load_grayscale_image,
    resolve_device,
    seed_everything,
    write_csv,
)

CIFAR_IMAGE_SIZE = 32
CIFAR_EPOCHS = 200


class ParameterizedHamiltonianSegmenter(nn.Module):
    """Small differentiable Ising-style segmentation model."""

    def __init__(self) -> None:
        super().__init__()
        self.intensity_scale = nn.Parameter(torch.tensor(8.0))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.horizontal_coupling = nn.Parameter(torch.tensor(1.0))
        self.vertical_coupling = nn.Parameter(torch.tensor(1.0))

    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return self.intensity_scale * (image - self.threshold) + self.bias

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(image)
        return logits, torch.sigmoid(logits)

    def pairwise_energy(self, probabilities: torch.Tensor) -> torch.Tensor:
        h_diff = torch.abs(probabilities[:, :, 1:] - probabilities[:, :, :-1]).mean()
        v_diff = torch.abs(probabilities[:, 1:, :] - probabilities[:, :-1, :]).mean()
        return F.softplus(self.horizontal_coupling) * h_diff + F.softplus(
            self.vertical_coupling
        ) * v_diff


def otsu_pseudo_mask(image: np.ndarray) -> np.ndarray:
    uint_image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(uint_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (mask.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())
    eps = 1e-8
    return {
        "dice": float((2.0 * tp) / (2.0 * tp + fp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "pixel_accuracy": float((tp + tn) / (tp + fp + fn + tn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
    }


def boundary_f_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_edges = cv2.Canny((prediction * 255).astype(np.uint8), 50, 150) > 0
    target_edges = cv2.Canny((target * 255).astype(np.uint8), 50, 150) > 0
    if pred_edges.sum() == 0 and target_edges.sum() == 0:
        return 1.0
    tp = float(np.logical_and(pred_edges, target_edges).sum())
    precision = tp / (float(pred_edges.sum()) + 1e-8)
    recall = tp / (float(target_edges.sum()) + 1e-8)
    return float((2.0 * precision * recall) / (precision + recall + 1e-8))


def train_phl(image: np.ndarray, device: torch.device, epochs: int) -> dict:
    target_np = otsu_pseudo_mask(image)
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target_np).unsqueeze(0).to(device)
    model = ParameterizedHamiltonianSegmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    history = []
    for epoch in range(epochs + 1):
        logits, probabilities = model(image_tensor)
        unary = F.binary_cross_entropy_with_logits(logits, target_tensor)
        pairwise = model.pairwise_energy(probabilities)
        loss = unary + 0.25 * pairwise

        if epoch < epochs:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            pred_np = (probabilities.detach().cpu().numpy()[0] >= 0.5).astype(np.float32)
            seg = segmentation_metrics(pred_np, target_np)
            print(
                f"  PHL Epoch {epoch:>4d}: loss={float(loss.detach().cpu()):.5f} "
                f"dice={seg['dice']:.4f} iou={seg['iou']:.4f}"
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "unary": float(unary.detach().cpu()),
                    "pairwise": float(pairwise.detach().cpu()),
                    **seg,
                }
            )

    with torch.no_grad():
        _, probabilities = model(image_tensor)
    probability_np = probabilities.detach().cpu().numpy()[0]
    prediction_np = (probability_np >= 0.5).astype(np.float32)
    seg_metrics = segmentation_metrics(prediction_np, target_np)
    seg_metrics["boundary_f_score"] = boundary_f_score(prediction_np, target_np)
    return {
        "probability": probability_np,
        "mask": prediction_np,
        "target_mask": target_np,
        "history": history,
        "segmentation": seg_metrics,
    }


def save_phl_visualization(
    image: np.ndarray,
    result: dict,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    panels = [
        ("Image", image, "gray"),
        ("Target mask", result["target_mask"], "gray"),
        ("PHL probability", result["probability"], "viridis"),
        ("PHL mask", result["mask"], "gray"),
    ]
    for ax, (name, panel, cmap) in zip(axes, panels):
        ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    rows = []
    output_dir = Path(__file__).resolve().parent / "outputs"
    history_dir = output_dir / "training_history"
    for img_path in image_paths(args.dataset_dir, args.count):
        print(f"\nPHL processing: {img_path.name}")
        image = load_grayscale_image(img_path)
        result = train_phl(image, device, args.epochs)
        reconstruction_metrics = compute_metrics(result["probability"], image)
        seg = result["segmentation"]
        rows.append(
            {
                "model": "PHL",
                "image": img_path.name,
                "image_size": CIFAR_IMAGE_SIZE,
                "patch_size": 0,
                "epochs": args.epochs,
                **reconstruction_metrics,
                **seg,
            }
        )
        safe_name = img_path.stem.replace(" ", "_")
        write_csv(history_dir / f"{safe_name}_phl_history.csv", result["history"])
        save_phl_visualization(
            image,
            result,
            output_dir / f"{safe_name}_phl_segmentation.png",
            f"PHL segmentation: {img_path.name}",
        )
        print(
            f"  MSE={reconstruction_metrics['mse']:.6f} "
            f"Dice={seg['dice']:.4f} IoU={seg['iou']:.4f}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHL on CIFAR-10 at 32x32.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=CIFAR_EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything()
    rows = run(args)
    output_dir = Path(__file__).resolve().parent / "outputs"
    write_csv(output_dir / "phl_cifar32_metrics.csv", rows)
    print(f"\nResults saved to {output_dir / 'phl_cifar32_metrics.csv'}")
