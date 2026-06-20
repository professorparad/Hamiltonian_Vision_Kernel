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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Baselines.common import (  # noqa: E402
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_ROOT,
    load_grayscale_image,
    load_mask,
    otsu_pseudo_mask,
    resolve_device,
    save_grayscale,
    save_json,
    seed_everything,
    segmentation_metrics,
)


class ParameterizedHamiltonianSegmenter(nn.Module):
    """Differentiable Ising-style Hamiltonian for binary image segmentation."""

    def __init__(self):
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
        probabilities = torch.sigmoid(logits)
        return logits, probabilities

    def pairwise_energy(self, probabilities: torch.Tensor) -> torch.Tensor:
        h_diff = torch.abs(probabilities[:, :, 1:] - probabilities[:, :, :-1]).mean()
        v_diff = torch.abs(probabilities[:, 1:, :] - probabilities[:, :-1, :]).mean()
        j_h = F.softplus(self.horizontal_coupling)
        j_v = F.softplus(self.vertical_coupling)
        return j_h * h_diff + j_v * v_diff

    def mean_energy(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, probabilities = self.forward(image)
        unary = F.binary_cross_entropy_with_logits(logits, target)
        return unary + self.pairwise_energy(probabilities)


def boundary_f_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_edges = cv2.Canny((prediction * 255).astype(np.uint8), 50, 150) > 0
    target_edges = cv2.Canny((target * 255).astype(np.uint8), 50, 150) > 0
    if pred_edges.sum() == 0 and target_edges.sum() == 0:
        return 1.0
    tp = float(np.logical_and(pred_edges, target_edges).sum())
    precision = tp / (float(pred_edges.sum()) + 1e-8)
    recall = tp / (float(target_edges.sum()) + 1e-8)
    return float((2.0 * precision * recall) / (precision + recall + 1e-8))


def train_phl(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_np = load_grayscale_image(args.image_path, args.image_size)
    if args.mask_path:
        target_np = load_mask(args.mask_path, args.image_size)
        supervision = "mask"
    else:
        target_np = otsu_pseudo_mask(image_np)
        supervision = "otsu_pseudo_mask"

    image = torch.from_numpy(image_np).unsqueeze(0).to(device)
    target = torch.from_numpy(target_np).unsqueeze(0).to(device)

    model = ParameterizedHamiltonianSegmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs + 1):
        logits, probabilities = model(image)
        unary = F.binary_cross_entropy_with_logits(logits, target)
        pairwise = model.pairwise_energy(probabilities)
        loss = unary + args.pairwise_weight * pairwise

        if epoch < args.epochs:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            pred_np = (probabilities.detach().cpu().numpy()[0] >= 0.5).astype(np.float32)
            metrics = segmentation_metrics(pred_np, target_np)
            metrics["boundary_f_score"] = boundary_f_score(pred_np, target_np)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "unary": float(unary.detach().cpu()),
                    "pairwise": float(pairwise.detach().cpu()),
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "boundary_f_score": metrics["boundary_f_score"],
                }
            )

    with torch.no_grad():
        _, probabilities = model(image)
    probability_np = probabilities.detach().cpu().numpy()[0]
    prediction_np = (probability_np >= 0.5).astype(np.float32)

    save_grayscale(output_dir / "phl_probability.png", probability_np)
    save_grayscale(output_dir / "phl_mask.png", prediction_np)
    save_grayscale(output_dir / "phl_target_mask.png", target_np)

    rows_path = output_dir / "phl_training_history.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    final_metrics = segmentation_metrics(prediction_np, target_np)
    final_metrics["boundary_f_score"] = boundary_f_score(prediction_np, target_np)
    summary = {
        "algorithm": "Parameterized Hamiltonian Learning segmentation baseline",
        "supervision": supervision,
        "image_path": str(args.image_path),
        "mask_path": str(args.mask_path) if args.mask_path else None,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "pairwise_weight": args.pairwise_weight,
        "parameters": {
            name: float(value.detach().cpu())
            for name, value in model.named_parameters()
        },
        "metrics": final_metrics,
    }
    save_json(output_dir / "phl_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Parameterized Hamiltonian Learning segmentation baseline."
    )
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--mask-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "phl")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    summary = train_phl(parse_args())
    print("PHL baseline complete.")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Dice: {summary['metrics']['dice']:.4f}")
    print(f"IoU: {summary['metrics']['iou']:.4f}")


if __name__ == "__main__":
    main()
