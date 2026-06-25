"""
Classical autoencoder baseline on CIFAR-10 at 32×32 (patch-based, like HVK).
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
    seed_everything,
    extract_patches,
    stitch_patches,
    write_csv,
)


class PatchAutoencoder(nn.Module):
    """Patch-based autoencoder (mimics HVK structure but classical)."""
    def __init__(self, patch_size: int = 8, latent_dim: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Linear(patch_size * patch_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, patch_size * patch_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)
        # Encode/decode each patch
        latent = self.encoder(patches)
        recon_patches = self.decoder(latent)
        # Stitch back
        grid = H // self.patch_size
        recon_patches = recon_patches.view(B, grid, grid, self.patch_size, self.patch_size)
        recon = recon_patches.permute(0, 1, 3, 2, 4).contiguous().view(
            B, 1, H, W
        )
        return recon


def train_autoencoder(
    image: np.ndarray, device: torch.device, epochs: int = 200
) -> np.ndarray:
    img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    model = PatchAutoencoder(patch_size=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = nn.functional.mse_loss(output, img_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  Autoencoder Epoch {epoch:>4d}: loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        reconstruction = model(img_tensor).cpu().squeeze().numpy()
    return reconstruction


def run(args: argparse.Namespace) -> list[dict]:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = []
    paths = image_paths(args.dataset_dir, args.count)
    for img_path in paths:
        print(f"\nAutoencoder processing: {img_path.name}")
        image = load_grayscale_image(img_path)
        reconstruction = train_autoencoder(image, device, args.epochs)
        metrics = compute_metrics(reconstruction, image)
        rows.append({
            "model": "Autoencoder",
            "image": img_path.name,
            "image_size": 32,
            "patch_size": 8,
            "epochs": args.epochs,
            **metrics,
        })
        print(f"  MSE={metrics['mse']:.6f} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencoder on CIFAR-10 at 32×32.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything()
    rows = run(args)
    output_dir = Path(__file__).resolve().parent / "outputs"
    write_csv(output_dir / "autoencoder_cifar32_metrics.csv", rows)
    print(f"\nResults saved to {output_dir / 'autoencoder_cifar32_metrics.csv'}")
