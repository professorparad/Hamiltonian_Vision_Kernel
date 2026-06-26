"""
CNN autoencoder baseline on CIFAR-10 at 32×32.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common import (
    DEFAULT_DATASET_DIR,
    compute_metrics,
    image_paths,
    load_grayscale_image,
    resolve_device,
    seed_everything,
    write_csv,
)


class CNNAutoencoder(nn.Module):
    """Convolutional autoencoder for 32×32 grayscale images."""
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=4),                # 4 -> 1
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_cnn(image: np.ndarray, device: torch.device, epochs: int = 200) -> np.ndarray:
    img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    model = CNNAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = nn.functional.mse_loss(output, img_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  CNN Epoch {epoch:>4d}: loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        reconstruction = model(img_tensor).cpu().squeeze().numpy()
    return reconstruction


def run(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    rows = []
    paths = image_paths(args.dataset_dir, args.count)
    for img_path in paths:
        print(f"\nCNN processing: {img_path.name}")
        image = load_grayscale_image(img_path)
        reconstruction = train_cnn(image, device, args.epochs)
        metrics = compute_metrics(reconstruction, image)
        rows.append({
            "model": "CNN",
            "image": img_path.name,
            "image_size": 32,
            "patch_size": 32,
            "epochs": args.epochs,
            **metrics,
        })
        print(f"  MSE={metrics['mse']:.6f} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN on CIFAR-10 at 32×32.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything()
    rows = run(args)
    output_dir = Path(__file__).resolve().parent / "outputs"
    write_csv(output_dir / "cnn_cifar32_metrics.csv", rows)
    print(f"\nResults saved to {output_dir / 'cnn_cifar32_metrics.csv'}")
