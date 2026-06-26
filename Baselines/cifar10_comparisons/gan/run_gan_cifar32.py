"""
GAN baseline on CIFAR-10 at native 32×32.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
    extract_patches,
    stitch_patches,
    write_csv,
)

CIFAR_IMAGE_SIZE = 32
CIFAR_PATCH_SIZE = 8
CIFAR_EPOCHS = 200


class PatchGenerator(nn.Module):
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),   # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(patches))


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.net(patches)


def train_gan(image: np.ndarray, device: torch.device, epochs: int = CIFAR_EPOCHS) -> np.ndarray:
    patches_np = extract_patches(image, CIFAR_PATCH_SIZE)
    patch_tensor = torch.from_numpy(patches_np).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(patch_tensor), batch_size=4, shuffle=True, drop_last=False
    )

    generator = PatchGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0008, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            # Train discriminator
            with torch.no_grad():
                fake_detached = generator(real)
            real_logits = discriminator(real)
            fake_logits = discriminator(fake_detached)
            d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            )
            if epoch < epochs:
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # Train generator
            fake = generator(real)
            fake_logits = discriminator(fake)
            rec_loss = F.mse_loss(fake, real)
            adv_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )
            g_loss = 50.0 * rec_loss + 1.0 * adv_loss
            if epoch < epochs:
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  GAN Epoch {epoch:>4d}: g={g_loss.item():.4f} d={d_loss.item():.4f} rec={rec_loss.item():.6f}")

    generator.eval()
    all_recon = []
    with torch.no_grad():
        for (real,) in DataLoader(TensorDataset(patch_tensor), batch_size=4):
            all_recon.append(generator(real.to(device)).cpu())
    recon_patches = torch.cat(all_recon).squeeze(1).numpy()
    return stitch_patches(recon_patches, CIFAR_IMAGE_SIZE, CIFAR_PATCH_SIZE)


def run(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    rows = []
    paths = image_paths(args.dataset_dir, args.count)
    for img_path in paths:
        print(f"\nGAN processing: {img_path.name}")
        image = load_grayscale_image(img_path)
        reconstruction = train_gan(image, device, args.epochs)
        metrics = compute_metrics(reconstruction, image)
        rows.append({
            "model": "GAN",
            "image": img_path.name,
            "image_size": CIFAR_IMAGE_SIZE,
            "patch_size": CIFAR_PATCH_SIZE,
            "epochs": args.epochs,
            **metrics,
        })
        print(f"  MSE={metrics['mse']:.6f} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAN on CIFAR-10 at 32×32.")
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
    write_csv(output_dir / "gan_cifar32_metrics.csv", rows)
    print(f"\nResults saved to {output_dir / 'gan_cifar32_metrics.csv'}")
